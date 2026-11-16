"""Shared-mask reuse tests with real RGB/NPY/ZIP bytes; no SAM model required."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import zipfile

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import mask_cache as cache
from protocol import save,load,fingerprint


class MaskCacheTests(unittest.TestCase):
    plan={'sam_revision':'pinned-sam','sam_confidence':.5}

    def prepared(self,root,reverse=False):
        folder=root/'prepared';images=folder/'data/archworld/scene/images';images.mkdir(parents=True)
        views=[]
        for i in range(3):
            name=f'{"renamed" if reverse else "original"}_{i}.png';image=images/name
            Image.new('RGB',(4,3),(i*50,20,30)).save(image)
            views.append({'alias':name,'original':f'folder/{i}.png','rgb_sha256':cache.sha(image)})
        if reverse:views.reverse()
        manifest={'scene':'scene','anchor':'arch','views':views}
        paths=[folder/'masks/scene/mask'/(Path(v['alias']).stem+'_arch.npy') for v in views]
        return folder,manifest,paths

    def request(self,root,folder,role='build',sources=()):
        return {'prepared':str(folder),'mask_cache':str(root/'common_masks'),'mask_role':role,
                'plan':self.plan,'mask_sources':list(sources),'mask_deadline':time.time()+2}

    def generate(self,folder,manifest,calls,fail=False):
        def fn(views,checkpoint):
            for i,view in enumerate(views):
                calls.append(view['rgb_sha256'])
                path=folder/'masks/scene/mask'/(Path(view['alias']).stem+'_arch.npy')
                path.parent.mkdir(parents=True,exist_ok=True);np.save(path,np.ones((1,1,3,4),dtype=np.uint8))
                checkpoint()
                if fail and i==0:raise RuntimeError('Simulated interruption')
        return fn

    def source(self,root,manifest,wrong_shape=False):
        folder=root/'old_results/scenes/scene';folder.mkdir(parents=True)
        old={**manifest,'views':copy.deepcopy(manifest['views'][:2]),'input_key':'old-code-key'}
        for i,v in enumerate(old['views']):v['alias']=f'old_name_{i}.png'
        save(folder/'input_manifest.json',old)
        save(root/'old_results/study.json',{'plan':self.plan})
        with zipfile.ZipFile(folder/'masks.zip','w') as z:
            z.writestr('identity.json',json.dumps({'input_key':'old-code-key'}))
            for view in old['views']:
                data=io.BytesIO();np.save(data,np.ones((2,2) if wrong_shape else (1,1,3,4),dtype=np.uint8))
                z.writestr('scene/mask/'+Path(view['alias']).stem+'_arch.npy',data.getvalue())
        save(folder/'mask_validation.json',{'input_key':'old-code-key','numeric_checks':'passed',
                                          'mask_archive_sha256':cache.sha(folder/'masks.zip')})
        return {'folder':str(folder),'scene':'scene','anchor':'arch','profile':cache.profile(self.plan)}

    def test_imports_partial_old_set_and_reader_reuses_after_rename_reorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,manifest,paths=self.prepared(root/'H3')
            source=self.source(root,manifest);original=cache.sha(Path(source['folder'])/'masks.zip');calls=[]
            result=cache.ensure(self.request(root,folder,sources=[source]),manifest,paths,self.generate(folder,manifest,calls))
            self.assertEqual((result['imported'],result['generated'],len(calls)),(2,1,1))
            self.assertEqual(cache.sha(Path(source['folder'])/'masks.zip'),original)
            second,other,targets=self.prepared(root/'H1',reverse=True)
            result=cache.ensure(self.request(root,second,'read'),other,targets,lambda *a:self.fail('Reader generated masks'))
            self.assertEqual((result['reused'],result['generated']),(3,0))
            for path in targets:self.assertTrue(np.load(path,allow_pickle=False).all())

    def test_interrupted_generation_restores_partial_then_generates_only_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,manifest,paths=self.prepared(root/'first');calls=[]
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                cache.ensure(self.request(root,folder),manifest,paths,self.generate(folder,manifest,calls,True))
            self.assertEqual(len(calls),1)
            self.assertFalse(list((root/'common_masks').rglob('masks.zip')))
            second,other,targets=self.prepared(root/'resumed');calls=[]
            result=cache.ensure(self.request(root,second),other,targets,self.generate(second,other,calls))
            self.assertEqual((result['reused'],result['generated'],len(calls)),(1,2,2))

    def test_reader_fails_immediately_and_never_falls_back_to_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,manifest,paths=self.prepared(root/'reader')
            request=self.request(root,folder,'read');request['mask_deadline']=time.time()-1
            with self.assertRaisesRegex(FileNotFoundError,'Prepare_Masks'):
                cache.ensure(request,manifest,paths,lambda *a:self.fail('Reader must not generate'))
            self.assertFalse((root/'common_masks').exists())

    def test_wrong_settings_bad_dimensions_and_damaged_cache_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,manifest,paths=self.prepared(root/'H3')
            source=self.source(root,manifest,True);calls=[]
            wrong={**source,'profile':{**source['profile'],'confidence':.9}}
            result=cache.ensure(self.request(root,folder,sources=[wrong,source]),manifest,paths,self.generate(folder,manifest,calls))
            self.assertEqual(result['imported'],0);self.assertEqual(len(calls),3)
            self.assertEqual(len(result['rejected_sources']),2)
            archive=Path(result['cache'])
            with zipfile.ZipFile(archive) as z:entries={name:z.read(name) for name in z.namelist()}
            key=next(name for name in entries if name.endswith('.npy'));entries[key]=b'corrupt'
            with zipfile.ZipFile(archive,'w') as z:
                for name,data in entries.items():z.writestr(name,data)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                cache.ensure(self.request(root,folder,'read'),manifest,paths,lambda *a:None)

    def test_discovery_reports_unverified_loose_files_without_importing_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,manifest,paths=self.prepared(root/'H3')
            source=self.source(root,manifest)
            loose=root/'old_runtime/masks/scene/mask';loose.mkdir(parents=True)
            np.save(loose/'image_arch.npy',np.ones((3,4),dtype=np.uint8))
            result=cache.discover([root,root/'old_results'])
            self.assertEqual(len(result['sources']),1)
            self.assertEqual(result['sources'][0]['folder'],source['folder'])
            self.assertEqual(len(result['unverified']),1)


if __name__=='__main__':unittest.main()
