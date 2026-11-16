"""Synthetic execution/merge regressions; these are not GPU experiment results."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parents[2])]
import protocol
import run
import report
import compare_hypotheses as combined
from protocol import load,save,fingerprint
from full_dataset import METRICS,save_checkpoint
from paired_eval import PROTOCOL
from prepare import file_hash


class HypothesisExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=protocol.make_plan()
        with (HERE/'inputs/ArchWorld_metadata.csv').open(encoding='utf-8-sig',newline='') as f:
            cls.metadata={r['scene']:r for r in csv.DictReader(f)}

    def scoped(self,h,mini=False):
        p=copy.deepcopy(self.plan);p['hypothesis']=h
        p['candidate_definitions']={k:protocol.candidates()[k] for k in protocol.candidate_names(h)}
        if mini:
            p['rows']=[r for r in p['rows'] if r['anchor']=='arch'][:3]
            for i,row in enumerate(p['rows']):row['split']='calibration' if i<2 else 'confirmation'
            p['calibration_size']=2
        return p

    def test_only_requested_hypothesis_candidates_are_selected(self):
        for h,total,size in [('H1',198,5),('H2',198,5),('H3',162,3)]:
            p=self.scoped(h);names=protocol.candidate_names(h)
            self.assertEqual(len(names),size)
            scores={r['scene']:{k:float(i+1) for i,k in enumerate(names) if k!='no_intervention'}
                    for r in p['rows'] if r['split']=='calibration'}
            selected=protocol.select_settings(p,scores)
            self.assertEqual(set(selected)&{'H1','H2','H3'},{h})
            self.assertEqual(protocol.job_budget(p)['total_upper_bound'],total)
            for row in p['rows']:
                needed=protocol.conditions(row,selected,h)
                self.assertTrue(set(needed)<=set(names))
                self.assertEqual(len(needed),size if row['split']=='calibration' else (2 if h=='H3' else 3))
            changed=copy.deepcopy(scores);changed[next(iter(changed))]['unrelated']=1.
            with self.assertRaises(ValueError):protocol.select_settings(p,changed)

    def fixture_phase(self,output,work,sam_python,identity,phase,devices,deadline,selection=None,shard_index=0,shards=1,**kwargs):
        plan=identity['plan'];h=plan['hypothesis']
        save(output/'execution_environment.json',{'gpu':'Synthetic GPU','torch':'fixture'})
        values={'no_intervention':2.,'scalar_eta100':1.8,'scalar_eta1000':1.9,
                'layer_alpha0.01':1.7,'layer_alpha0.02':1.75,'view_eta100':1.6,'view_eta1000':1.65}
        for row in plan['rows']:
            if row['split']!=phase:continue
            needed=protocol.conditions(row,selection,h)
            if run.valid_saved(row,output,identity,needed):continue
            saved=run.scene_dir(output,row);saved.mkdir(parents=True,exist_ok=True)
            count=42 if row['split']=='confirmation' else 45
            job={'plan_sha256':fingerprint(plan),'source_sha256':identity['source_sha256'],
                 'num_views':count,'scene':row['scene'],'anchor':row['anchor'],'protocol':PROTOCOL}
            meta=self.metadata[row['scene']];factor=float(meta['Scale'])*1000/float(meta['Extent'])
            results={}
            for name in needed:
                self.calls.append((h,row['scene'],name))
                with tempfile.TemporaryDirectory() as tmp:
                    folder=Path(tmp);value=values[name];eta=protocol.candidates()[name]['eta']
                    save(folder/METRICS,{'base':{'Acc-mean':2.,'Comp-mean':2.},'inter':{'Acc-mean':value,'Comp-mean':value}})
                    save(folder/'strength_trace.json',{'num_views':count,'layers':[{'layer':i,'accepted':eta!=0,
                        'anchor_tokens_by_view':[1]*count,'eta_by_view':[eta]*count,'clipped_views':0,
                        'applied_masked_relative_update':.01 if eta else 0.} for i in range(24)]})
                    for arm in ('base','inter'):
                        save(folder/(arm+'_evaluation.json'),{'status':'ok','protocol':PROTOCOL,
                            'prediction_sha256':{'depth_map':'base' if arm=='base' or eta==0 else name}})
                    save(folder/'timing.json',{'elapsed_seconds':1.})
                    save_checkpoint(folder,saved/(name+'.zip'),{**job,'condition':name,'definition':protocol.candidates()[name],
                                    'reference':False},False)
                results[name]={'CD_norm_base':2*factor,'CD_norm':value*factor,'seconds':1.}
            save(saved/'results.json',{'identity':job,'conditions':results,'status':'complete'})
            save(saved/'input_manifest.json',{'input_key':row['scene'],'views':[str(i) for i in range(count)],'anchor':row['anchor']})
            mask_hashes={}
            with zipfile.ZipFile(saved/'masks.zip','w') as z:
                for i in range(count):
                    name=f'{i}.npy';payload=b'synthetic-mask';z.writestr(name,payload)
                    mask_hashes[name]=hashlib.sha256(payload).hexdigest()
            save(saved/'mask_validation.json',{'numeric_checks':'passed','input_key':row['scene'],
                'mask_archive_sha256':file_hash(saved/'masks.zip'),'mask_sha256_by_file':mask_hashes})
        return {'attempted':sum(r['split']==phase for r in plan['rows']),'failed':[]}

    def test_independent_execution_freeze_report_resume_and_combination(self):
        self.calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            real_report=report.report
            for h in ('H1','H2','H3'):
                plan=self.scoped(h,True)
                with patch.object(run,'make_plan',return_value=plan), \
                     patch.object(run,'run_phase',side_effect=self.fixture_phase), \
                     patch.object(report,'report',side_effect=lambda out:real_report(out,plots=False)):
                    result=run.run_study(root/h,root/'scratch'/h,Path('sam'),hypothesis=h)
                    self.assertEqual(result['status'],'complete')
                    before=len(self.calls)
                    run.run_study(root/h,root/'scratch'/h,Path('sam'),hypothesis=h)
                    self.assertEqual(before,len(self.calls))
                self.assertEqual(set(load(root/h/'summary.json')['hypotheses']),{h})
                with (root/h/'reports'/h/'paired_results.csv').open(newline='',encoding='utf-8') as f:
                    self.assertEqual(next(csv.DictReader(f))['num_views'],'42')
            self.assertEqual(len(self.calls),34) # 13 H1 + 13 H2 + 8 H3; reference GPU work is stubbed.
            result=combined.compare(root)
            self.assertEqual(result['status'],'complete_and_comparable')
            self.assertEqual(result['confirmation_landmarks'],1)
            with (root/'comparison/paired_hypotheses.csv').open(newline='',encoding='utf-8') as f:
                self.assertEqual(next(csv.DictReader(f))['num_views'],'42')
            self.assertEqual(result['hypotheses']['H3']['mean_difference'],0.)
            row=self.scoped('H2',True)['rows'][-1];saved=run.scene_dir(root/'H2',row)
            mask=load(saved/'mask_validation.json');changed=copy.deepcopy(mask)
            changed['mask_sha256_by_file']['0.npy']='different'
            save(saved/'mask_validation.json',changed)
            with self.assertRaisesRegex(ValueError,'SAM masks differ'):combined.compare(root)
            save(saved/'mask_validation.json',mask)
            # Valid individual metrics can still have different control predictions.
            checkpoint=saved/'scalar_eta100.zip'
            with zipfile.ZipFile(checkpoint) as z:entries={n:z.read(n) for n in z.namelist()}
            audit=json.loads(entries['inter_evaluation.json']);audit['prediction_sha256']['depth_map']='changed'
            entries['inter_evaluation.json']=json.dumps(audit).encode()
            with zipfile.ZipFile(checkpoint,'w') as z:
                for name,payload in entries.items():z.writestr(name,payload)
            with self.assertRaisesRegex(ValueError,'predictions differ'):combined.compare(root)


if __name__=='__main__':unittest.main()
