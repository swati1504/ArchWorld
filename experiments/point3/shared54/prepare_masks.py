"""Prepare shared SAM masks once, without running any reconstruction experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from protocol import HERE,SOURCE,make_plan,load,save,fingerprint
from prepare import input_key
from mask_cache import (discover,identity,profile,sha,ready_path,require_ready,check_entry,publish_ready)
from run import ensure_study,scene_dir,exclusive,bounded_run,interrupt_study,STOP,BudgetExceeded
from full_dataset import sanitize


def reuse_preparation(sources,plan,row,key,saved,cache):
    """Reuse verified completed scenes across source bundles without downloading inputs again."""
    for source in sources:
        if (source['scene']!=row['scene'] or source['anchor']!=row['anchor'] or
                source['profile']!=profile(plan)):continue
        previous=Path(source['folder'])
        if previous.resolve()==saved.resolve() or not (previous/'shared_mask_entry.json').exists():continue
        manifest=load(previous/'input_manifest.json')
        if manifest['input_key']!=key:continue
        validation=load(previous/'mask_validation.json');entry=load(previous/'shared_mask_entry.json')
        if (validation.get('numeric_checks')!='passed' or validation.get('input_key')!=key or
                validation['files']!=len(manifest['views']) or identity(plan,manifest)!=entry['identity']):
            raise ValueError(f'Completed mask preparation has inconsistent provenance: {previous}')
        check_entry(cache,plan,entry,deep=True)
        # The masks remain in the verified common cache. Copy only their small audit records/preview.
        for name in ('input_manifest.json','mask_validation.json','mask_preview.jpg'):
            target=saved/name;payload=(previous/name).read_bytes()
            if target.exists() and target.read_bytes()!=payload:
                raise ValueError(f'Existing mask audit record differs: {target}')
            if not target.exists():target.write_bytes(payload)
        entry={**entry,'previews':str(saved/'mask_preview.jpg'),'reused_from_preparation':str(previous)}
        save(saved/'mask_reuse.json',{'mode':'validated_preparation','reused':len(manifest['views']),
             'generated':0,'imported':0,'source':str(previous)})
        save(saved/'shared_mask_entry.json',entry)
        return entry
    return None


def run_preparation(output,work,sam_python,cache,hours=8,search_roots=()):
    if hours<=0:raise ValueError('Positive session duration required')
    plan=make_plan();ensure_study(output,plan)
    deadline=time.monotonic()+hours*3600
    if ready_path(cache,plan).exists():
        ready=require_ready(cache,plan,deep=True)
        save(output/'summary.json',{**ready,'note':'Existing complete cohort verified; no SAM inference needed'})
        return ready
    found=discover(search_roots);save(output/'mask_discovery.json',found)
    print(f"Mask search: {len(found['sources'])} documented archives; {len(found['unverified'])} unverified locations",flush=True)
    archives=load(HERE/'inputs/archive_sources.json');entries=[];failures=[]
    environment=os.environ.copy()
    summary={'status':'partial','completed_landmarks':0,'expected_landmarks':len(plan['rows']),
             'max_num_views':plan['num_views'],'view_count_policy':plan.get('view_count_policy','exact'),
             'entries':entries,'failures':failures}
    try:
        # One preparation notebook is the designated writer. Do not run duplicate
        # preparation sessions against a Drive mount: it is not a distributed lock.
        with exclusive(Path(cache)/'.preparation.lock'):
            for index,row in enumerate(plan['rows']):
                if STOP.is_set() or time.monotonic()>=deadline:raise BudgetExceeded('Mask session budget reached; resume this notebook')
                saved=scene_dir(output,row);saved.mkdir(parents=True,exist_ok=True)
                entry_path=saved/'shared_mask_entry.json'
                try:
                    if entry_path.exists():
                        entry=load(entry_path);check_entry(cache,plan,entry,deep=True)
                        entries.append(entry)
                        print(f"[{index+1}/{len(plan['rows'])}] {row['scene']}: validated masks already saved",flush=True)
                        continue
                    key=input_key(plan,row,archives[row['scene']]);prepared=work/'prepared'/key
                    reused=reuse_preparation(found['sources'],plan,row,key,saved,cache)
                    if reused is not None:
                        entries.append(reused)
                        print(f"[{index+1}/{len(plan['rows'])}] {row['scene']}: reused validated previous preparation; no download or SAM inference",flush=True)
                        continue
                    request={'plan':plan,'row':row,'input_key':key,'prepared':str(prepared),'saved':str(saved),
                             'mask_cache':str(cache),'mask_role':'build','mask_sources':found['sources']}
                    request_file=saved/'mask_request.json';save(request_file,request)
                    start=time.monotonic()
                    print(f"[{index+1}/{len(plan['rows'])}] {row['scene']}: preparing up to {plan['num_views']} selected images and masks",flush=True)
                    for action,log in [('prepare','preparation.log'),('masks','masks.log')]:
                        bounded_run([sam_python,'-u',HERE/'prepare.py',action,request_file],
                                    saved/log,SOURCE,environment,deadline,echo=True)
                    manifest=load(saved/'input_manifest.json');validation=load(saved/'mask_validation.json')
                    count=len(manifest['views'])
                    if validation['numeric_checks']!='passed' or validation['files']!=count or manifest['input_key']!=key:
                        raise ValueError('Mask preparation did not produce complete matching validation')
                    definition=identity(plan,manifest)
                    archive=Path(cache)/'sets'/fingerprint(definition)/'masks.zip'
                    entry={'identity':definition,'archive_sha256':sha(archive),'previews':str(saved/'mask_preview.jpg')}
                    check_entry(cache,plan,entry,deep=True);save(entry_path,entry);entries.append(entry)
                    save(saved/'mask_timing.json',{'wall_seconds':time.monotonic()-start,'num_views':count})
                    (saved/'mask_failure.json').unlink(missing_ok=True)
                    if prepared.exists():
                        if prepared.resolve().parent!=(work/'prepared').resolve():raise ValueError('Unsafe owned preparation path')
                        shutil.rmtree(prepared)
                except Exception as exc:
                    message=sanitize(f'{type(exc).__name__}: {exc}',environment.get('HF_TOKEN',''))
                    failure={'scene':row['scene'],'reason':message};failures.append(failure)
                    save(saved/'mask_failure.json',failure);print(message,flush=True)
                    if isinstance(exc,BudgetExceeded) or 'out of memory' in message.lower():raise
                finally:
                    summary['completed_landmarks']=len(entries)
                    summary['view_counts']={e['identity']['scene']:len(e['identity']['images']) for e in entries}
                    save(output/'summary.json',summary)
                    with (output/'view_counts.csv').open('w',newline='',encoding='utf-8') as handle:
                        writer=csv.DictWriter(handle,fieldnames=['scene','num_views','max_num_views'])
                        writer.writeheader();writer.writerows({'scene':s,'num_views':n,'max_num_views':plan['num_views']}
                                                             for s,n in summary['view_counts'].items())
            if failures:raise RuntimeError('Some masks are incomplete; inspect summary.json and scene logs, then resume')
            ready=publish_ready(cache,plan,entries,output)
            summary.update(status='ready',images=ready['images'],ready_manifest=str(ready_path(cache,plan)))
            print(f"READY: {len(entries)} landmarks / {ready['images']} masks. H1, H2 and H3 can now run.",flush=True)
            return summary
    except BaseException as exc:
        summary.update(status='partial',reason=sanitize(f'{type(exc).__name__}: {exc}',environment.get('HF_TOKEN','')))
        raise
    finally:
        save(output/'summary.json',summary)
        environment.pop('HF_TOKEN',None)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupt_study);signal.signal(signal.SIGINT,interrupt_study)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--sam-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--mask-cache',type=Path,required=True);parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--mask-search-root',type=Path,action='append',default=[])
    a=parser.parse_args()
    print(json.dumps(run_preparation(a.output,a.work,a.sam_python,a.mask_cache,a.hours,a.mask_search_root),indent=2))
