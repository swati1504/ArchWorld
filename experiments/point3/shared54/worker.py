"""One scene per process; retain a single VGGT model across every condition."""
import argparse
import builtins
import csv
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from protocol import HERE,SOURCE,candidates,fingerprint,load,save
from prepare import validate_prepared
sys.path.insert(0,str(HERE.parent))
from full_dataset import METRICS,read_checkpoint,save_checkpoint,validate_result,clear_owned_scratch
from paired_eval import PROTOCOL,paired_worker
from fast_eval import (baseline_identity,compare_results,install,prepare_baseline_environment,attach_baseline_audit)


def seed_all(seed,torch):
    import numpy as np
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def deterministic_intervention(intervention_class,torch):
    """Make fused attention gradients repeatable without changing head kernels.

    PyTorch 2.3 sends even no-grad bilinear resizing through a large temporary
    decomposition in global deterministic mode. Keep the original inference
    path; require deterministic algorithms only inside the intervention hook.
    The full fresh/cached prediction-equivalence gate still covers both paths.
    """
    class DeterministicIntervention(intervention_class):
        def intervene_forward_pre_hook(self,*args,**kwargs):
            previous=torch.are_deterministic_algorithms_enabled()
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
            torch.use_deterministic_algorithms(True,warn_only=False)
            try:
                return super().intervene_forward_pre_hook(*args,**kwargs)
            finally:
                torch.use_deterministic_algorithms(previous,warn_only=warn_only)
    return DeterministicIntervention


def load_result(saved,name):
    with zipfile.ZipFile(saved/(name+'.zip')) as z:
        return {n:json.loads(z.read(n)) for n in (METRICS,'strength_trace.json','base_evaluation.json','inter_evaluation.json','timing.json')}


def cache_images(original,cache):
    def wrapped(paths,*args,**kwargs):
        key=(tuple(str(Path(p).resolve()) for p in paths),repr(args),repr(sorted(kwargs.items())))
        if key not in cache:cache[key]=original(paths,*args,**kwargs).detach().cpu()
        # A fresh tensor keeps any in-place model changes out of the cache.
        return cache[key].clone()
    return wrapped


def hardware(torch):
    packages={}
    for name in ('torch','numpy','scipy','open3d','safetensors','vggt'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='not-installed'
    return {'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
            'torch_version':torch.__version__,'packages':packages,
            'total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
            'intervention_hook_determinism':'strict',
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def run_scene(request):
    # Seeding alone does not make fused attention gradients repeatable. Set the
    # CUDA workspace before initialization. Both fresh and accelerated attention
    # hooks require deterministic kernels; unsupported operations fail loudly.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
    plan,row=request['plan'],request['row']
    prepared=Path(request['prepared']);manifest=validate_prepared(prepared)
    saved=Path(request['saved']);saved.mkdir(parents=True,exist_ok=True)
    if load(saved/'mask_validation.json')['input_key']!=manifest['input_key']:
        raise ValueError('SAM masks belong to different inputs')
    aliases=[v['alias'] for v in manifest['views']]
    num_views=len(aliases)
    hw=hardware(torch)
    identity={'protocol':PROTOCOL,'source_sha256':request['source_sha256'],
        'revision':plan['dataset_revision'],'num_views':num_views,'seed':plan['seed'],
        'scene':row['scene'],'anchor':row['anchor'],'views':aliases,
        'gpu':hw['gpu'],'torch_version':hw['torch_version'],'hardware':hw,
        'input_key':manifest['input_key'],'plan_sha256':fingerprint(plan)}
    parent=prepared/'model_jobs';parent.mkdir(exist_ok=True)
    stats=prepared/'stats'/row['scene'];stats.mkdir(parents=True,exist_ok=True)
    save(stats/'stats.json',{row['anchor']:{'sum':1.}})
    sys.path[:0]=[str(SOURCE/'intervention'),str(SOURCE)]
    from intervention.run import prepare_scene
    model=None;images_cache={};loads=0
    definitions=candidates()
    def execute(name,reference=False,force_replay=False):
        nonlocal model,loads
        label=('reference_' if reference else '')+name
        job={**identity,'condition':name,'definition':definitions[name],'reference':reference}
        checkpoint=saved/(label+'.zip')
        result=read_checkpoint(checkpoint,job)
        if result is not None and not force_replay:return result
        run_dir=parent/(label+'_'+str(time.time_ns()));run_dir.mkdir();(run_dir/'.point3-owned').touch()
        strength_path=run_dir/'strength.json';save(strength_path,definitions[name]['strength'])
        folder=run_dir/'scene';prepare_scene(prepared/'data',row['scene'],folder,aliases,num_views)
        for key in list(os.environ):
            if key.startswith(('INTERVENTION_','POINT3_')) or key in {
                'METRIC_EVAL_BASE','SKIP_BASELINE_PASS','SKIP_EVAL','STRUCTURED_STRENGTH_CONFIG',
                'IMAGE_FOLDER','SOURCE_IMAGE_DIR','PTS_500M_IMAGE_BASE','VGGT_HALF','SAVE_RAW_TENSORS',
                'PER_LAYER_INTERVENTION','ALIGNMENT_MAX_POINTS','ALIGNMENT_N_ITERATIONS','ICP_MAX_ITER'}:
                os.environ.pop(key,None)
        os.environ.update(DEVICE='cuda',NUM_IMAGES=str(num_views),SCENE_NAME=row['scene'],
            USE_ICP='1',ICP_THRESHOLD='0.1',USE_VALIDITY_MASKS='1',
            VALIDITY_MASKS_BASE=str(prepared/'data/manual_filtration_masks'),
            SAM_MASK_DIR=str(prepared/'masks'),ATTENTION_STATS_DIR=str(prepared/'stats'),
            INTERVENTION_LOSS_SCALE='1e5',FIRST_INTERVENTION_LAYER='0',LAST_INTERVENTION_LAYER='23',
            INTERVENTION_STEP_SIZE=str(definitions[name]['eta']),EVAL_TIMEOUT_SEC='0',TOP_K_ELEMENTS='1',
            STRUCTURED_STRENGTH_CONFIG=str(strength_path),PROBE='1',POINT3_SOURCE=str(SOURCE),
            POINT3_EVAL_SEED=str(plan['seed']))
        baseline=None
        if not reference and name!='no_intervention':
            zero=load_result(saved,'no_intervention')
            cache={'identity':baseline_identity(job),'metrics':zero[METRICS]['base'],'audit':zero['base_evaluation.json']}
            cache_path=prepared/'baseline_cache.json';save(cache_path,cache)
            os.environ.update(POINT3_BASELINE_CACHE=str(cache_path),POINT3_BASELINE_IDENTITY=json.dumps(baseline_identity(job)))
            baseline=prepare_baseline_environment()
        module_name='models.vggt.run_pass'
        module=(importlib.reload(sys.modules[module_name]) if module_name in sys.modules else importlib.import_module(module_name))
        module.VGGTAttentionIntervention=deterministic_intervention(module.VGGTAttentionIntervention,torch)
        module.print=builtins.print # reload retains attributes absent from source definitions.
        if module.EVAL_TIMEOUT_SEC!=0:raise ValueError('Paired in-process evaluation is required')
        module._run_alignment_and_eval_worker=paired_worker(module._run_alignment_and_eval_worker,plan['seed'])
        if not reference:
            install(module,prepared/'ground_truth_cache')
            if baseline:attach_baseline_audit(module,baseline)
            module.load_and_preprocess_images=cache_images(module.load_and_preprocess_images,images_cache)
        probe={};emit=module.print
        def capture(*args,**kwargs):
            nonlocal probe
            if args and isinstance(args[0],str) and args[0].startswith('PROBE_RESULT '):
                probe=json.loads(args[0].removeprefix('PROBE_RESULT '))
            emit(*args,**kwargs)
        module.print=capture
        seed_all(plan['seed'],torch)
        start=time.perf_counter()
        reused_model=model is not None
        if reference:
            if model is not None:raise ValueError('Reference check must precede the persistent model')
            fresh=module.main(scene_name=row['scene'],work_dir=folder,model=None)
            del fresh;gc.collect();torch.cuda.empty_cache()
        else:
            if model is None:loads+=1
            model=module.main(scene_name=row['scene'],work_dir=folder,model=model)
        elapsed=time.perf_counter()-start
        validate_result(load(folder/METRICS),load(folder/'strength_trace.json'),
            load(folder/'base_evaluation.json'),load(folder/'inter_evaluation.json'),name,num_views)
        save(folder/'timing.json',{'elapsed_seconds':elapsed,**probe,
            'model_reused':not reference and reused_model,
            'baseline_reused':bool(baseline),'reference':reference})
        if result is not None: # Replay the first two jobs to check persistent-model behavior after resume.
            previous=load_result(saved,label)
            compare_results(previous[METRICS],load(folder/METRICS),previous['inter_evaluation.json'],load(folder/'inter_evaluation.json'))
        result=save_checkpoint(folder,checkpoint,job,False)
        clear_owned_scratch(run_dir,parent)
        return result
    check_path=saved/'reference_check.json'
    gate_candidate=request.get('gate_candidate','scalar_eta100')
    if request.get('reference_only',False):
        execute(gate_candidate,reference=True)
        return
    gate_identity={**identity,'gate':'fresh_vs_persistent_cached_v1','candidate':gate_candidate}
    gate_ok=check_path.exists() and load(check_path).get('identity')==gate_identity and load(check_path).get('status')=='passed'
    must_check=request.get('reference_check',False) and not gate_ok
    if must_check:execute(gate_candidate,reference=True)
    requested=list(request['conditions'])
    if must_check:
        # Explicitly evaluate zero then scalar in the SAME process before accepting
        # any speedup, even when their isolated checkpoints already exist.
        execute('no_intervention',force_replay=True)
        execute(gate_candidate,force_replay=True)
        fresh,fast=load_result(saved,'reference_'+gate_candidate),load_result(saved,gate_candidate)
        compare_results(fresh[METRICS],fast[METRICS],fresh['inter_evaluation.json'],fast['inter_evaluation.json'])
        if fresh['base_evaluation.json']['prediction_sha256']!=fast['base_evaluation.json']['prediction_sha256']:
            raise ValueError('Fresh/reference baseline predictions differ')
        save(check_path,{'status':'passed','identity':gate_identity,
            'reference_seconds':fresh['timing.json']['elapsed_seconds'],'fast_seconds':fast['timing.json']['elapsed_seconds'],
            'note':'One-scene equivalence/timing check, not a dataset-wide speedup guarantee'})
    for name in requested:
        print(f"SHARED_JOB {row['scene']} / {name}",flush=True);execute(name)
    with (prepared/'data/ArchWorld_metadata.csv').open(encoding='utf-8-sig',newline='') as f:
        meta=next(r for r in csv.DictReader(f) if r['scene']==row['scene'])
    factor=float(meta['Scale'])*1000/float(meta['Extent'])
    if not factor>0:raise ValueError('Invalid normalization factor')
    results={}
    for name in requested:
        result=load_result(saved,name);m=result[METRICS]
        results[name]={'CD_norm_base':factor*(m['base']['Acc-mean']+m['base']['Comp-mean'])/2,
            'CD_norm':factor*(m['inter']['Acc-mean']+m['inter']['Comp-mean'])/2,
            'seconds':result['timing.json']['elapsed_seconds']}
    save(saved/'results.json',{'identity':identity,'conditions':results,'status':'complete',
        'model_loads_this_attempt':loads,'split':row['split'],'mask_validation':load(saved/'mask_validation.json')})
    return results


def run_request(request):
    # The fresh-reference CUDA context must exit before the persistent model
    # starts. Merely deleting its model and emptying the allocator left too
    # little headroom for the next 50-view baseline on L4.
    if request.get('reference_check',False) and not request.get('reference_only',False):
        prepared=Path(request['prepared'])
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json',prefix='reference_request_',
                                         dir=prepared,delete=False,encoding='utf-8') as handle:
            json.dump({**request,'reference_only':True},handle)
            child_request=Path(handle.name)
        try:
            print('Running fresh reference in an isolated process before persistent-model checks',flush=True)
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),str(child_request)],check=True)
        finally:
            child_request.unlink(missing_ok=True)
    return run_scene(request)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('request',type=Path)
    run_request(load(parser.parse_args().request))
