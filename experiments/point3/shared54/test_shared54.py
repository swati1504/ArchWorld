"""CPU-only regressions; fake model output is never experimental evidence."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parents[2])]
import protocol
import prepare
import worker
import run
from protocol import load,save,fingerprint


class SharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=protocol.make_plan()

    def test_obelisk_correction_preserves_frozen_allocation_and_h3_fallback(self):
        expected='ac8ba5f86415d243bf496f0f2b18c9154a81b48cfe493d8125363709d0bf5027'
        for hypothesis in (None,'H1','H2','H3'):
            plan=protocol.make_plan(hypothesis=hypothesis)
            calibration=[r['scene'] for r in plan['rows'] if r['split']=='calibration']
            self.assertEqual(fingerprint(calibration),expected)
            row=next(r for r in plan['rows'] if r['scene']=='Washington_Monument')
            self.assertEqual((row['supplied_anchor'],row['anchor'],row['split']),('altar','obelisk','confirmation'))
            self.assertIn('obelisk',plan['fallback_anchor_types'])
            self.assertNotIn('obelisk',plan['supported_anchor_types'])
        scores={r['scene']:{k:float(i+1) for i,k in enumerate(protocol.candidates()) if k!='no_intervention'}
                for r in self.plan['rows'] if r['split']=='calibration'}
        chosen=protocol.select_settings(self.plan,scores)
        self.assertEqual(chosen['H3']['obelisk'],chosen['global_scalar'])

    def test_split_retains54_and_never_reads_holdout_results(self):
        plan=self.plan
        cal=[r for r in plan['rows'] if r['split']=='calibration']
        con=[r for r in plan['rows'] if r['split']=='confirmation']
        self.assertEqual((len(cal),len(con)),(18,36))
        self.assertFalse({r['scene'] for r in plan['rows']} & set(plan['heldout_ids']))
        self.assertEqual(len(plan['supported_anchor_types']),6)
        self.assertIn('dome',plan['fallback_anchor_types'])
        self.assertEqual(next(r['anchor'] for r in plan['rows'] if r['scene']=='Forth_Bridge'),'bridge')
        self.assertEqual(protocol.job_budget(plan)['total_upper_bound'],306)
        pools=protocol.partition(plan['rows'],3)
        self.assertEqual(len({r['scene'] for pool in pools for r in pool}),54)
        self.assertEqual(sum(map(len,pools)),54)

    def test_selection_uses_only_calibration_and_deduplicates_policies(self):
        names=[r['scene'] for r in self.plan['rows'] if r['split']=='calibration']
        scores={name:{k:float(i+1) for i,k in enumerate(protocol.candidates()) if k!='no_intervention'} for name in names}
        chosen=protocol.select_settings(self.plan,scores)
        row=next(r for r in self.plan['rows'] if r['anchor']=='dome')
        self.assertEqual(chosen['H3']['dome'],chosen['global_scalar'])
        self.assertEqual(len(protocol.conditions(row,chosen)),4)
        with self.assertRaises(ValueError):protocol.select_settings(self.plan,{**scores,row['scene']:scores[names[0]]})
        missing=copy.deepcopy(scores);missing[names[0]].pop('view_eta100')
        with self.assertRaises(ValueError):protocol.select_settings(self.plan,missing)

    def test_view_identity_and_ambiguity(self):
        eligible=[{'original':f'folder/{i}.jpg','rgb_sha256':str(i)} for i in range(60)]
        self.assertEqual(prepare.choose_views(eligible,1729,'scene'),prepare.choose_views(list(reversed(eligible)),1729,'scene'))
        duplicated=[{'original':f'copy/{i}.jpg','rgb_sha256':str(i%20)} for i in range(60)]
        with self.assertRaises(ValueError):prepare.choose_views(duplicated,1729,'scene')
        smaller=prepare.choose_views(duplicated,1729,'scene',allow_fewer=True)
        self.assertEqual(len(smaller),20)
        self.assertEqual(len({v['rgb_sha256'] for v in smaller}),20)
        with self.assertRaisesRegex(ValueError,'at least two'):
            prepare.choose_views(duplicated[:1],1729,'scene',allow_fewer=True)
        self.assertIsNone(prepare.derived_name('a.jpg',{'a.jpg.npy'}, {'a.jpg':2}))
        self.assertIsNone(prepare.derived_name('x/a.jpg',{'a.jpg.npy'}, {'a.jpg':2}))
        self.assertEqual(prepare.derived_name('x/a.jpg',{'x/a.jpg.npy'}, {'a.jpg':2}),'x/a.jpg.npy')
        for path in ('../bad','/bad','a\\b','C:/bad'):
            with self.assertRaises(ValueError):prepare.safe_parts(path)

    def test_failed_calibration_persists_partial_status_and_never_freezes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(run,'make_plan',return_value=self.plan), \
                 patch.object(run,'run_phase',return_value={'attempted':1,'failed':['fixture']}), \
                 patch.object(run,'freeze') as frozen, \
                 patch('report.report',return_value={'status':'partial'}):
                with self.assertRaisesRegex(RuntimeError,'Calibration incomplete'):
                    run.run_study(root/'out',root/'work',Path('sam-python'))
                frozen.assert_not_called()
            status=load(root/'out/session_1_0.json')
            self.assertEqual(status['status'],'partial')
            self.assertEqual(status['report_status'],'partial')

    def test_cancellation_terminates_the_active_child(self):
        import hypothesis_study
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cancel=threading.Event();children=[]
            original=hypothesis_study.subprocess.Popen
            def launch(*args,**kwargs):
                child=original(*args,**kwargs);children.append(child);return child
            timer=threading.Timer(.2,cancel.set);timer.start()
            try:
                with patch.object(hypothesis_study.subprocess,'Popen',side_effect=launch):
                    with self.assertRaisesRegex(hypothesis_study.BudgetExceeded,'cancelled'):
                        hypothesis_study.bounded_run([sys.executable,'-c','import time;time.sleep(30)'],
                            root/'child.log',root,os.environ.copy(),time.monotonic()+40,cancel_event=cancel)
                self.assertEqual(len(children),1)
                self.assertIsNotNone(children[0].poll())
            finally:timer.cancel()

    def test_prepare_preserves_pose_depth_mask_and_unique_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sparse=root/'sparse';sparse.mkdir()
            spec=importlib.util.spec_from_file_location('fixture_reader',protocol.SOURCE/'eval/read_write_model.py')
            rw=importlib.util.module_from_spec(spec);spec.loader.exec_module(rw)
            cameras={1:rw.Camera(id=1,model='PINHOLE',width=3,height=2,params=np.array([2.,2.,1.,1.]))}
            images={i:rw.Image(id=i,qvec=np.array([1.,0,0,0]),tvec=np.array([i,0.,0.]),camera_id=1,
                    name=f'group{i%2}/visión_{i}.png',xys=np.zeros((0,2)),point3D_ids=np.array([],dtype=np.int64)) for i in range(1,55)}
            rw.write_cameras_binary(cameras,sparse/'cameras.bin');prepare.write_images_binary_utf8(images,sparse/'images.bin')
            rw.write_points3D_binary({},sparse/'points3D.bin')
            archive=root/'archive.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                def add(name,payload):
                    member=tarfile.TarInfo(name);member.size=len(payload);t.addfile(member,io.BytesIO(payload))
                for p in sparse.iterdir():add('archworld/scene/sparse/'+p.name,p.read_bytes())
                for i,im in images.items():
                    rgb=io.BytesIO();Image.new('RGB',(3,2),(i,20,30)).save(rgb,format='PNG')
                    add('archworld/scene/images/'+im.name,rgb.getvalue())
                    depth=io.BytesIO();np.save(depth,np.ones((2,3),dtype=np.float32)*(i if i>10 else 0))
                    valid=io.BytesIO();np.save(valid,np.full((2,3),i>20,dtype=bool))
                    add('archworld/scene/depth/'+im.name+'.npy',depth.getvalue())
                    add('manual_filtration_masks/scene/'+im.name+'.npy',valid.getvalue())
            info={'path':'data/scene.tar.gz','size':archive.stat().st_size,'lfs':{'oid':prepare.file_hash(archive)}}
            real_load=prepare.load
            def patched_load(path):return {'scene':info} if Path(path).name=='archive_sources.json' else real_load(path)
            request={'plan':self.plan,'row':{'scene':'scene','anchor':'arch'},'prepared':str(root/'prepared'),'input_key':'fixture'}
            with patch.object(prepare,'load',side_effect=patched_load), \
                 patch('huggingface_hub.hf_hub_download',return_value=str(archive)) as download, \
                 patch.dict(os.environ,{'HF_TOKEN':'test-placeholder'}), \
                 patch.object(prepare.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)):
                # A real download creates this owned directory; make it in the fixture.
                (root/'prepared/download').mkdir(parents=True)
                result=prepare.prepare(request)
                self.assertEqual(len(result['views']),50)
                self.assertEqual(result['num_views'],50)
                self.assertGreater(result['zero_depth_views'],0)
                self.assertGreater(result['zero_validity_views'],0)
                self.assertEqual(download.call_args.kwargs['token'],'test-placeholder')
                rewritten=rw.read_images_binary(root/'prepared/data/archworld/scene/sparse/images.bin')
                self.assertEqual(len(rewritten),len(images))
                chosen={v['colmap_id'] for v in result['views']}
                for key in set(images)-chosen:self.assertEqual(rewritten[key].name,images[key].name)
                for v in result['views']:
                    i=v['colmap_id'];self.assertEqual(rewritten[i].name,v['alias'])
                    np.testing.assert_array_equal(rewritten[i].tvec,images[i].tvec)
                    np.testing.assert_array_equal(np.load(root/'prepared/data/archworld/scene/depth'/(v['alias']+'.npy')),np.ones((2,3))*(i if i>10 else 0))
                self.assertEqual(prepare.validate_prepared(root/'prepared'),result)
                # Validate cached SAM files without downloading or running a model.
                result['anchor_override']={'anchor':'arch','provenance':'synthetic test'}
                save(root/'prepared/prepared.json',result)
                paths=prepare.mask_paths(root/'prepared',result)
                for path in paths:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.save(path,np.ones((1,1,2,3),dtype=np.uint8))
                request['saved']=str(root/'saved')
                prepare.masks(request)
                checked=load(root/'saved/mask_validation.json')
                self.assertEqual((checked['files'],checked['nonempty_views'],checked['preview_views']),(50,50,50))
                self.assertEqual(len(checked['coverage_by_view']),50)
                # An existing readable array is insufficient if its dimensions are wrong.
                np.save(paths[0],np.ones((1,1),dtype=np.uint8))
                with self.assertRaisesRegex(ValueError,'dimensions differ'):
                    prepare.masks({**request,'saved':str(root/'invalid')})

    def test_colmap_writer_matches_ascii_reference_and_preserves_unicode_tracks(self):
        spec=importlib.util.spec_from_file_location('writer_test',protocol.SOURCE/'eval/read_write_model.py')
        rw=importlib.util.module_from_spec(spec);spec.loader.exec_module(rw)
        im=rw.Image(id=9,qvec=np.array([1.,0.,0.,0.]),tvec=np.array([1.,2.,3.]),camera_id=2,
                    name='view.jpg',xys=np.array([[1.5,2.5],[3.5,4.5]]),point3D_ids=np.array([-1,7],dtype=np.int64))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=root/'reference.bin';actual=root/'actual.bin'
            rw.write_images_binary({9:im},reference);prepare.write_images_binary_utf8({9:im},actual)
            self.assertEqual(reference.read_bytes(),actual.read_bytes())
            im=im._replace(name='Catedral_á_日本.jpg');prepare.write_images_binary_utf8({9:im},actual)
            got=rw.read_images_binary(actual)[9];self.assertEqual(got.name,im.name)
            np.testing.assert_array_equal(got.qvec,im.qvec);np.testing.assert_array_equal(got.tvec,im.tvec)
            np.testing.assert_array_equal(got.xys,im.xys);np.testing.assert_array_equal(got.point3D_ids,im.point3D_ids)

    def test_model_reuse_reference_gate_resume_and_image_cache(self):
        deterministic=torch.are_deterministic_algorithms_enabled()
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark=torch.backends.cudnn.benchmark
        cudnn_deterministic=torch.backends.cudnn.deterministic
        self.addCleanup(torch.use_deterministic_algorithms,deterministic,warn_only=warn_only)
        self.addCleanup(setattr,torch.backends.cudnn,'benchmark',benchmark)
        self.addCleanup(setattr,torch.backends.cudnn,'deterministic',cudnn_deterministic)
        # Load PyTorch's lazy determinism support before patch.dict restores
        # sys.modules; otherwise newer local PyTorch re-registers its libraries.
        torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);saved=root/'saved';saved.mkdir();prepared=root/'prepared';prepared.mkdir()
            row=next(r for r in self.plan['rows'] if r['split']=='calibration');scene=row['scene']
            images=prepared/'data/archworld'/scene/'images';images.mkdir(parents=True)
            views=[]
            count=42
            for i in range(count):
                name=f'p3_{i}.jpg';(images/name).write_bytes(bytes([i]))
                views.append({'alias':name,'rgb_sha256':str(i),'original':name})
            metadata=prepared/'data/ArchWorld_metadata.csv';metadata.write_text(f'scene,Scale,Extent\n{scene},1,1000\n',encoding='utf8')
            save(prepared/'prepared.json',{'scene':scene,'anchor':row['anchor'],'input_key':'fixture','views':views,
                'files':{p.relative_to(prepared).as_posix():prepare.file_hash(p) for p in (prepared/'data').rglob('*') if p.is_file()}})
            save(saved/'mask_validation.json',{'input_key':'fixture'})
            calls=[]
            def fake_main(scene_name,work_dir,model=None):
                self.assertFalse(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
                self.assertEqual(int(os.environ['NUM_IMAGES']),count)
                self.assertTrue((Path(work_dir)/f'images_{count}').exists())
                calls.append(model)
                definition=load(os.environ['STRUCTURED_STRENGTH_CONFIG']);eta=float(os.environ['INTERVENTION_STEP_SIZE'])
                zero=eta==0;mode=definition['mode'];folder=Path(work_dir)
                value=2. if zero else 1.8 if mode=='fixed' else 1.6
                save(folder/worker.METRICS,{'base':{'Acc-mean':2.,'Comp-mean':2.},'inter':{'Acc-mean':value,'Comp-mean':value}})
                for arm in ('base','inter'):
                    save(folder/(arm+'_evaluation.json'),{'status':'ok','protocol':worker.PROTOCOL,
                        'prediction_sha256':{'depth_map':'base' if arm=='base' or zero else f'{eta}:{mode}'}})
                save(folder/'strength_trace.json',{'num_views':count,'layers':[{'layer':i,'accepted':not zero,
                    'anchor_tokens_by_view':[1]*count,'eta_by_view':[eta]*count,'clipped_views':0,
                    'applied_masked_relative_update':.01} for i in range(24)]})
                return object() if model is None else model
            module=SimpleNamespace()
            def reset(m):
                m.main=fake_main;m.EVAL_TIMEOUT_SEC=0;m.print=print
                m.VGGTAttentionIntervention=object
                m._run_alignment_and_eval_worker=lambda *a:('ok',{})
                m.generate_gt_by_unproject=lambda *a,**kw:None
                m.load_and_preprocess_images=lambda paths:torch.ones((count,3,2,2))
                return m
            reset(module)
            request={'plan':self.plan,'row':row,'prepared':str(prepared),'saved':str(saved),'source_sha256':'fixture',
                'conditions':list(protocol.candidates()),'reference_check':True}
            props=SimpleNamespace(total_memory=80*2**30)
            with patch.dict(os.environ,{},clear=False),patch.dict(sys.modules,{'models.vggt.run_pass':module}), \
                 patch.object(worker.importlib,'reload',side_effect=reset), \
                 patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_name',return_value='Fake GPU'), \
                 patch.object(torch.cuda,'get_device_capability',return_value=(8,0)),patch.object(torch.cuda,'get_device_properties',return_value=props), \
                 patch.object(torch.cuda,'manual_seed_all'),patch.object(torch.cuda,'empty_cache'):
                worker.run_scene(request)
                self.assertEqual(len(calls),8) # Fresh reference + seven shared conditions.
                self.assertIsNone(calls[0]);self.assertIsNone(calls[1])
                self.assertTrue(all(m is calls[2] for m in calls[2:]))
                self.assertEqual(load(saved/'results.json')['model_loads_this_attempt'],1)
                self.assertEqual(load(saved/'results.json')['identity']['num_views'],count)
                self.assertEqual(load(saved/'results.json')['identity']['hardware']['intervention_hook_determinism'],'strict')
                self.assertEqual(load(saved/'reference_check.json')['status'],'passed')
                worker.run_scene(request);self.assertEqual(len(calls),8)
            processed={};counter=[]
            wrapped=worker.cache_images(lambda paths:(counter.append(True) or torch.ones(2)),processed)
            first=wrapped([images/'p3_0.jpg']);first.zero_()
            self.assertEqual(wrapped([images/'p3_0.jpg']).sum().item(),2)
            self.assertEqual(len(counter),1)

    def test_attention_determinism_scoped_and_restored_after_failure(self):
        previous=torch.are_deterministic_algorithms_enabled()
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms,previous,warn_only=warn_only)
        test=self
        class Hook:
            def intervene_forward_pre_hook(self,value,fail=False):
                test.assertTrue(torch.are_deterministic_algorithms_enabled())
                test.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                if fail:raise RuntimeError('simulated attention failure')
                return value
        hook=worker.deterministic_intervention(Hook,torch)()
        for enabled,warn in ((False,False),(True,True)):
            torch.use_deterministic_algorithms(enabled,warn_only=warn)
            self.assertEqual(hook.intervene_forward_pre_hook(7),7)
            self.assertEqual(torch.are_deterministic_algorithms_enabled(),enabled)
            self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(),warn)
            with self.assertRaisesRegex(RuntimeError,'simulated attention failure'):
                hook.intervene_forward_pre_hook(7,fail=True)
            self.assertEqual(torch.are_deterministic_algorithms_enabled(),enabled)
            self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(),warn)

    def test_reference_process_finishes_before_persistent_model_and_blocks_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            request={'prepared':tmp,'reference_check':True,'num_views':50}
            order=[]
            def child(command,check):
                isolated=load(Path(command[-1]))
                self.assertTrue(check)
                self.assertTrue(isolated['reference_only'])
                self.assertEqual(isolated['num_views'],50)
                order.append('reference')
            with patch.object(worker.subprocess,'run',side_effect=child), \
                 patch.object(worker,'run_scene',side_effect=lambda r:order.append('persistent')):
                worker.run_request(request)
            self.assertEqual(order,['reference','persistent'])
            self.assertEqual(list(Path(tmp).glob('reference_request_*.json')),[])
            with patch.object(worker.subprocess,'run',side_effect=RuntimeError('reference failed')), \
                 patch.object(worker,'run_scene') as persistent:
                with self.assertRaisesRegex(RuntimeError,'reference failed'):
                    worker.run_request(request)
                persistent.assert_not_called()
            self.assertEqual(list(Path(tmp).glob('reference_request_*.json')),[])


if __name__=='__main__':unittest.main()
