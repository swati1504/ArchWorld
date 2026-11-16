"""Verify preparation/resume and environment routing without a GPU or network."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import mask_cache as cache
import prepare_masks as coordinator
import protocol
from prepare import mask_paths
from gpu_setup import gpu_stack,setup_for
from notebook_setup import SETUP


class MaskPreparationTests(unittest.TestCase):
    def test_preparation_resumes_completed_landmarks_and_all_hypotheses_read_ready(self):
        plan=protocol.make_plan();plan['rows']=plan['rows'][:2]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';shared=root/'common';calls=[];interrupt=[True]
            def fake_worker(command,*args,**kwargs):
                self.assertEqual(str(command[0]),'sam-python')
                self.assertEqual(Path(command[2]).name,'prepare.py')
                action=command[3];request=protocol.load(command[4]);row=request['row']
                calls.append((row['scene'],action));folder=Path(request['prepared']);saved=Path(request['saved'])
                count=50 if row==plan['rows'][0] else 42
                if action=='prepare':
                    views=[]
                    for i in range(count):
                        image=folder/'data/archworld'/row['scene']/'images'/f'{i}.png'
                        image.parent.mkdir(parents=True,exist_ok=True)
                        Image.new('RGB',(4,3),(i,30,40)).save(image)
                        views.append({'alias':image.name,'original':image.name,'rgb_sha256':cache.sha(image)})
                    protocol.save(folder/'prepared.json',{'scene':row['scene'],'anchor':row['anchor'],
                                  'views':views,'input_key':request['input_key']})
                elif action=='masks':
                    if row==plan['rows'][1] and interrupt[0]:
                        interrupt[0]=False;raise coordinator.BudgetExceeded('Simulated session end')
                    manifest=protocol.load(folder/'prepared.json');paths=mask_paths(folder,manifest)
                    def generate(views,checkpoint):
                        for path in paths:
                            path.parent.mkdir(parents=True,exist_ok=True)
                            np.save(path,np.ones((1,1,3,4),dtype=np.uint8))
                        checkpoint()
                    result=cache.ensure(request,manifest,paths,generate)
                    protocol.save(saved/'mask_reuse.json',result)
                    protocol.save(saved/'input_manifest.json',manifest)
                    protocol.save(saved/'mask_validation.json',{'numeric_checks':'passed','files':count,'input_key':request['input_key']})
                    Image.new('RGB',(4,3),'red').save(saved/'mask_preview.jpg')
                else:self.fail('Unexpected reconstruction work')
            with patch.object(coordinator,'make_plan',return_value=plan),patch.object(coordinator,'bounded_run',side_effect=fake_worker):
                with self.assertRaises(coordinator.BudgetExceeded):
                    coordinator.run_preparation(output,root/'work',Path('sam-python'),shared)
                self.assertEqual(protocol.load(output/'summary.json')['completed_landmarks'],1)
                self.assertFalse(cache.ready_path(shared,plan).exists())
                result=coordinator.run_preparation(output,root/'work',Path('sam-python'),shared)
                self.assertEqual(result['status'],'ready')
                self.assertEqual(calls.count((plan['rows'][0]['scene'],'masks')),1)
                before=list(calls)
                coordinator.run_preparation(output,root/'work',Path('sam-python'),shared)
                self.assertEqual(calls,before)
            for hypothesis in ('H1','H2','H3'):
                ready=cache.require_ready(shared,{**plan,'hypothesis':hypothesis},deep=True)
                self.assertEqual((ready['landmarks'],ready['images']),(2,92))
                self.assertEqual(ready['view_counts'][plan['rows'][1]['scene']],42)
            changed={**plan,'seed':999}
            with self.assertRaises(FileNotFoundError):cache.require_ready(shared,changed)
            row=plan['rows'][0];previous=coordinator.scene_dir(output,row)
            manifest=protocol.load(previous/'input_manifest.json')
            source={'folder':str(previous),'scene':row['scene'],'anchor':row['anchor'],'profile':cache.profile(plan)}
            copied=root/'new_output';copied.mkdir()
            reused=coordinator.reuse_preparation([source],plan,row,manifest['input_key'],copied,shared)
            self.assertEqual(reused['identity'],ready['entries'][0]['identity'])
            self.assertEqual(protocol.load(copied/'mask_reuse.json')['generated'],0)
            self.assertEqual((copied/'mask_preview.jpg').read_bytes(),(previous/'mask_preview.jpg').read_bytes())
            self.assertIsNone(coordinator.reuse_preparation([source],plan,row,'changed-inputs',root/'different',shared))
            self.assertIsNone(coordinator.reuse_preparation([source],plan,{**row,'anchor':'different'},manifest['input_key'],root/'different',shared))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                cache.publish_ready(shared,plan,ready['entries'][:1],output)
            entry=ready['entries'][0]
            archive=shared/'sets'/protocol.fingerprint(entry['identity'])/'masks.zip'
            archive.write_bytes(archive.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                cache.require_ready(shared,plan,deep=True)

    def test_gpu_setup_routes_only_requested_environment_and_supports_blackwell(self):
        import hypothesis_study
        import os
        for capability in ('8.9','12.0'):
            for mode in ('sam','evaluation'):
                with self.subTest(capability=capability,mode=mode),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);commands=[]
                    def checked_output(command,**kwargs):
                        if command[0]=='nvidia-smi':return capability if 'compute_cap' in command[1] else 'Synthetic GPU, 96 GB'
                        return ''
                    def bounded(command,*args,**kwargs):commands.append([str(x) for x in command])
                    namespace={'subprocess':types.SimpleNamespace(check_output=checked_output),
                        'WORK':root,'OUTPUT':root,'SOURCE':HERE.parents[2],
                        'EVAL_PY':root/'envs/vggt/bin/python','SAM_PY':root/'envs/sam3/bin/python',
                        'sys':sys,'os':os,'time':time,'DEADLINE':time.monotonic()+100}
                    generated=setup_for(SETUP,mode);ast.parse(generated)
                    with patch.object(hypothesis_study,'bounded_run',side_effect=bounded):exec(generated,namespace)
                    installs=[c for c in commands if 'install' in c and '--python' in c]
                    self.assertTrue(installs)
                    intended='sam3' if mode=='sam' else 'vggt'
                    self.assertTrue(all(intended in c[c.index('--python')+1] for c in installs))
                    stack=gpu_stack([capability]);cuda=stack['sam_cuda' if mode=='sam' else 'eval_cuda']
                    self.assertTrue(any('https://download.pytorch.org/whl/cu'+cuda.replace('.','') in c for c in installs))
                    if capability=='12.0':
                        self.assertTrue(any('torch==2.7.0' in c for c in installs))
                        self.assertTrue(all('cu128' in c[c.index('--python')+1] for c in installs))


if __name__=='__main__':unittest.main()
