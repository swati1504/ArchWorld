"""Shared calibration/confirmation coordinator, one scene per GPU at a time."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from protocol import (HERE,SOURCE,make_plan,save,load,fingerprint,candidates,candidate_names,conditions,select_settings,partition,job_budget)
from prepare import input_key
sys.path.insert(0,str(HERE.parent))
from hypothesis_study import bounded_run as _bounded_run,clean_environment,BudgetExceeded
from full_dataset import sanitize

STOP=threading.Event()


def bounded_run(*args,**kwargs):
    return _bounded_run(*args,**kwargs,cancel_event=STOP)


def interrupt_study(signum,frame):
    # Child workers live in their own process groups. Let each owning lane stop
    # its active group before the coordinator exits, including notebook Stop.
    STOP.set()
    raise KeyboardInterrupt('Study interrupted')


def code_hash():
    digest=hashlib.sha256()
    files=[]
    for folder in ('eval','intervention','experiments/point3/shared54'):
        files.extend(p for p in (SOURCE/folder).rglob('*') if p.suffix in ('.py','.json') and
                     not set(p.parts)&{'__pycache__','dist','validation','snapshots'})
    files.extend(HERE.parent/n for n in ('paired_eval.py','fast_eval.py','full_dataset.py','hypothesis_study.py'))
    for p in sorted(set(files)):
        digest.update(p.relative_to(SOURCE).as_posix().encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def scene_dir(output,row):
    return output/'scenes'/hashlib.sha256(row['scene'].encode()).hexdigest()[:16]


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Another writer or interrupted run owns {path}; confirm it stopped before removing this lock') from None
    try:
        os.write(fd,json.dumps({'pid':os.getpid(),'host':os.environ.get('COMPUTERNAME',os.uname().nodename if hasattr(os,'uname') else ''),'time':time.time()}).encode())
        os.close(fd);yield
    finally:path.unlink(missing_ok=True)


def ensure_study(output,plan):
    output.mkdir(parents=True,exist_ok=True)
    identity={'plan':plan,'source_sha256':code_hash()}
    path=output/'study.json'
    if path.exists():
        if load(path)!=identity:raise ValueError('Different plan/source: choose a new output directory')
        return identity
    with exclusive(output/'.initializing.lock'):
        if path.exists() and load(path)!=identity:raise ValueError('Different plan/source: choose a new output directory')
        save(path,identity)
    return identity


def valid_saved(row,output,identity,requested):
    path=scene_dir(output,row)/'results.json'
    if not path.exists():return False
    result=load(path);job=result['identity']
    if job.get('plan_sha256')!=fingerprint(identity['plan']) or job.get('source_sha256')!=identity['source_sha256']:
        raise ValueError('Saved scene belongs to another study')
    if result.get('status')!='complete' or not set(requested)<=set(result['conditions']):return False
    count=len(load(scene_dir(output,row)/'input_manifest.json')['views'])
    minimum=identity['plan'].get('min_views',2) if identity['plan'].get('view_count_policy')=='up_to_available' else identity['plan']['num_views']
    if not minimum<=count<=identity['plan']['num_views'] or job.get('num_views')!=count:
        raise ValueError('Checkpoint view count differs from the selected input images')
    # Check every checkpoint, including its CRC and matching scene/settings, before reuse.
    import zipfile
    from full_dataset import validate_result,METRICS
    with (HERE/'inputs/ArchWorld_metadata.csv').open(encoding='utf-8-sig',newline='') as f:
        meta=next(r for r in csv.DictReader(f) if r['scene']==row['scene'])
    factor=float(meta['Scale'])*1000/float(meta['Extent'])
    for name in requested:
        with zipfile.ZipFile(scene_dir(output,row)/(name+'.zip')) as z:
            if z.testzip() is not None:raise ValueError('Corrupt scene checkpoint')
            r=lambda p:json.loads(z.read(p))
            expected={**job,'condition':name,'definition':candidates()[name],'reference':False}
            if r('job.json')!=expected:raise ValueError('Checkpoint identity differs from scene summary')
            metrics=r(METRICS)
            validate_result(metrics,r('strength_trace.json'),r('base_evaluation.json'),r('inter_evaluation.json'),name,count)
            for arm,key in [('base','CD_norm_base'),('inter','CD_norm')]:
                expected=factor*(metrics[arm]['Acc-mean']+metrics[arm]['Comp-mean'])/2
                if not math.isclose(result['conditions'][name][key],expected,rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Saved normalized score differs from its checkpoint')
    return True


def freeze(output,identity):
    plan=identity['plan'];scores={}
    for row in plan['rows']:
        if row['split']!='calibration':continue
        if not valid_saved(row,output,identity,candidate_names(plan.get('hypothesis'))):
            raise RuntimeError(f"Calibration incomplete: {row['scene']}; confirmation stays locked")
        result=load(scene_dir(output,row)/'results.json')['conditions']
        scores[row['scene']]={k:v['CD_norm'] for k,v in result.items() if k!='no_intervention'}
    chosen=select_settings(plan,scores)
    path=output/'selection.json'
    with exclusive(output/'.selection.lock'):
        if path.exists() and load(path)!=chosen:raise ValueError('Frozen choices changed')
        save(path,chosen);save(output/'calibration_scores.json',scores)
    return chosen


def run_phase(output,work,sam_python,identity,phase,devices,deadline,selection=None,shard_index=0,shards=1,pilot=False,mask_options=None):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required; no GPU speedup has yet been verified')
    if any(d<0 or d>=torch.cuda.device_count() for d in devices):raise ValueError('Requested GPU is not available')
    if len(set(devices))!=len(devices):raise ValueError('Use distinct GPUs, not concurrent workers on the same GPU')
    if len({(torch.cuda.get_device_name(d),torch.cuda.get_device_capability(d)) for d in devices})!=1:
        raise ValueError('Use matching GPU models for this study; record a different study for heterogeneous hardware')
    from worker import hardware
    profile=hardware(torch)
    profile['gpu']=torch.cuda.get_device_name(devices[0])
    profile['capability']=list(torch.cuda.get_device_capability(devices[0]))
    profile['total_memory_bytes']=torch.cuda.get_device_properties(devices[0]).total_memory
    profile_path=output/'execution_environment.json'
    if profile_path.exists():
        if load(profile_path)!=profile:raise ValueError('GPU/software changed; resume on matching hardware or start a new study')
    else:
        with exclusive(output/'.environment.lock'):
            if profile_path.exists() and load(profile_path)!=profile:raise ValueError('GPU/software changed')
            save(profile_path,profile)
    plan=identity['plan']
    required=lambda row:conditions(row,selection,plan.get('hypothesis'))
    rows=[r for r in plan['rows'] if r['split']==phase]
    if not 0<=shard_index<shards:raise ValueError('Invalid shard index')
    rows=partition(rows,shards)[shard_index]
    # Complete scenes are verified before scheduling, not merely skipped by filename.
    rows=[r for r in rows if not valid_saved(r,output,identity,required(r))]
    if pilot:
        if mask_options and mask_options.get('mask_role')=='read':
            from mask_cache import require_ready
            ready=require_ready(mask_options['mask_cache'],plan)
            counts={e['identity']['scene']:len(e['identity']['images']) for e in ready['entries']}
            rows.sort(key=lambda r:-counts[r['scene']])
        rows=rows[:1]
    if not rows:return {'attempted':0,'failed':[]}
    bins=partition(rows,len(devices));archives=load(HERE/'inputs/archive_sources.json')
    errors=[]
    def lane(device,assigned):
        failures=[]
        env=clean_environment(SOURCE,plan['seed'])
        # CUDA IDs here are logical IDs after any external visibility restriction.
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        physical=visible.split(',')[device] if visible else str(device)
        env['CUDA_VISIBLE_DEVICES']=physical
        env.pop('VGGT_HALF',None)
        gate=True
        for row in assigned:
            if STOP.is_set():break
            saved=scene_dir(output,row);saved.mkdir(parents=True,exist_ok=True)
            key=input_key(plan,row,archives[row['scene']])
            prepared=work/'prepared'/key
            request={'plan':plan,'row':row,'input_key':key,'prepared':str(prepared),'saved':str(saved),
                'source_sha256':identity['source_sha256'],'conditions':required(row),
                'reference_check':gate,'gate_candidate':selection['global_scalar'] if selection else 'scalar_eta100'}
            if mask_options:request.update(mask_options,mask_deadline=time.time()+max(0,deadline-time.monotonic()))
            request_file=saved/'request.json'
            try:
                with exclusive(saved/'.running.lock'):
                    save(request_file,request)
                    print(f"GPU {device}: {phase} / {row['scene']}",flush=True)
                    start=time.monotonic()
                    bounded_run([sys.executable,'-u',HERE/'prepare.py','prepare',request_file],
                        saved/'preparation.log',SOURCE,env,deadline)
                    mask_python=sys.executable if request.get('mask_role')=='read' else sam_python
                    bounded_run([mask_python,'-u',HERE/'prepare.py','masks',request_file],
                        saved/'masks.log',SOURCE,env,deadline,echo=bool(mask_options))
                    prep_seconds=time.monotonic()-start
                    bounded_run([sys.executable,'-u',HERE/'worker.py',request_file],
                        saved/'model.log',SOURCE,env,deadline,echo=True)
                    if not valid_saved(row,output,identity,required(row)):
                        raise RuntimeError('Worker returned without all validated results')
                    save(saved/'scene_timing.json',{'preparation_seconds':prep_seconds,'wall_seconds':time.monotonic()-start,
                         'gpu':torch.cuda.get_device_name(device),'num_views':len(load(saved/'input_manifest.json')['views'])})
                    gate=False
                    (saved/'failure.json').unlink(missing_ok=True)
                    # The persistent mask/results caches remain. Release large local inputs/GT
                    # only after the entire scene has valid checkpoints on persistent storage.
                    if prepared.resolve().parent!=(work/'prepared').resolve():raise ValueError('Unsafe owned cache path')
                    shutil.rmtree(prepared)
            except Exception as exc:
                message=sanitize(f'{type(exc).__name__}: {exc}',env.get('HF_TOKEN',''))
                save(saved/'failure.json',{'scene':row['scene'],'stage':phase,'reason':message})
                print(message,flush=True);failures.append(row['scene'])
                # A failed gate, OOM, timeout or mismatched predictions stops this lane.
                if STOP.is_set() or gate or isinstance(exc,BudgetExceeded) or 'out of memory' in message.lower():break
        return failures
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        tasks=[pool.submit(lane,d,group) for d,group in zip(devices,bins) if group]
        for task in as_completed(tasks):errors.extend(task.result())
    return {'attempted':len(rows),'failed':errors}


def run_study(output,work,sam_python,calibration_size=18,devices=(0,),hours=8,phase='all',shard_index=0,shards=1,hypothesis=None,mask_cache=None,mask_role=None,mask_search_roots=()):
    if hours<=0:raise ValueError('Positive session duration required')
    if shards>1 and phase=='all':raise ValueError('For independent shards, run calibration first, freeze once, then confirmation')
    plan=make_plan(calibration_size,hypothesis);identity=ensure_study(output,plan)
    if not (output/'job_budget.json').exists():save(output/'job_budget.json',job_budget(plan))
    if phase=='plan':return {'status':'planned',**job_budget(plan)}
    if phase=='freeze':return freeze(output,identity)
    if phase=='report':
        from report import report
        return report(output)
    mask_options=None
    if mask_cache:
        role=mask_role or 'read'
        if hypothesis and role!='read':raise ValueError('All hypothesis notebooks read masks; use prepare_masks.py to build them')
        from mask_cache import discover
        if role=='build':
            found=discover(mask_search_roots)
            save(output/'mask_discovery.json',found)
            print(f"Mask search: {len(found['sources'])} documented archives; {len(found['unverified'])} unverified locations",flush=True)
        else:
            from mask_cache import require_ready
            require_ready(mask_cache,plan)
            found={'sources':[]}
        mask_options={'mask_cache':str(mask_cache),'mask_role':role,'mask_sources':found['sources']}
        save(output/'mask_configuration.json',{'cache':str(mask_cache),'role':role})
    deadline=time.monotonic()+hours*3600
    status={'status':'running','phase':phase,'shard_index':shard_index,'shards':shards}
    state_path=output/f'session_{shards}_{shard_index}.json';save(state_path,status)
    try:
        if phase in ('all','calibration','pilot'):
            outcome=run_phase(output,work,sam_python,identity,'calibration',list(devices),deadline,
                      shard_index=shard_index,shards=shards,pilot=phase=='pilot',mask_options=mask_options)
            if outcome['failed']:raise RuntimeError('Calibration incomplete; inspect scene failure logs and resume')
            if phase=='pilot':status['status']='pilot_finished_check_reference_and_timings'
            elif phase=='all':freeze(output,identity)
            else:status['status']='calibration_shard_finished'
        if phase in ('all','confirmation'):
            chosen=load(output/'selection.json')
            if chosen['plan_sha256']!=fingerprint(plan):raise ValueError('Selection belongs to another plan')
            outcome=run_phase(output,work,sam_python,identity,'confirmation',list(devices),deadline,chosen,shard_index,shards,mask_options=mask_options)
            if outcome['failed']:raise RuntimeError('Confirmation incomplete; inspect scene failure logs and resume')
        if phase=='confirmation':status['status']='confirmation_shard_finished'
    except BaseException as exc:
        status.update(status='partial',reason=sanitize(f'{type(exc).__name__}: {exc}',os.environ.get('HF_TOKEN','')))
        raise
    finally:
        # Keep session state truthful even after a failed freeze, deadline, or interruption.
        save(state_path,status)
        try:
            from report import report
            # Separate process shards must not overwrite the shared report concurrently.
            if shards==1:
                result=report(output)
                status['report_status']=result['status']
                if phase=='all' and status['status']=='running':status['status']=result['status']
        except Exception as exc:
            status['report_error']=sanitize(f'{type(exc).__name__}: {exc}',os.environ.get('HF_TOKEN',''))
            status['status']='partial'
        save(state_path,status)
    return status


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupt_study)
    signal.signal(signal.SIGINT,interrupt_study)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,default=Path('/content/point3_shared54'))
    p.add_argument('--sam-python',type=Path,default=Path('/content/adaptiveattend_point3/envs/sam3/bin/python'))
    p.add_argument('--calibration-size',type=int,choices=[18,54],default=18)
    p.add_argument('--devices',default='0');p.add_argument('--hours',type=float,default=8)
    p.add_argument('--hypothesis',choices=['H1','H2','H3'],help='Run one hypothesis with its own controls/calibration/results')
    p.add_argument('--mask-cache',type=Path,help='Common SAM cache created by prepare_masks.py; all hypotheses read it')
    p.add_argument('--mask-role',choices=['build','read'])
    p.add_argument('--mask-search-root',type=Path,action='append',default=[],help='Existing result or mask location to inspect for reuse')
    p.add_argument('--phase',choices=['plan','pilot','calibration','freeze','confirmation','all','report'],default='all')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    result=run_study(a.output,a.work,a.sam_python,a.calibration_size,[int(x) for x in a.devices.split(',')],a.hours,a.phase,a.shard_index,a.shards,a.hypothesis,a.mask_cache,a.mask_role,a.mask_search_root)
    print(json.dumps(result,indent=2),flush=True)
    if result.get('status') in ('partial','failed'):sys.exit(2)
