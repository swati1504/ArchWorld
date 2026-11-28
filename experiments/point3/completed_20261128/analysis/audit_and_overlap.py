"""Audit saved confirmation evidence and describe overlapping method successes.

No inference, outcome-based retuning or held-out evaluation is performed.
"""
import argparse,hashlib,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'sources' if (ROOT/'sources').exists() else ROOT.parent/'final_report/combined/evidence');parser.add_argument('--output',type=Path,default=ROOT/'audit')
args=parser.parse_args();SOURCE=args.source
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
def read_json(name):return json.loads((SOURCE/name).read_text(encoding='utf-8'))
manifest=read_json('evidence_sha256.json')
checks={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items()}
assert all(checks.values()),[name for name,ok in checks.items() if not ok]
data=pd.read_csv(SOURCE/'paired_hypotheses.csv')
assert len(data)==36 and data.scene.nunique()==36
methods=['H1','H2','H3'];numeric=['baseline','scalar',*methods]
assert np.isfinite(data[numeric]).all().all() and (data[numeric]>0).all().all()
plans={h:read_json(h.lower()+'_run_study.json' if h!='H3' else 'h3_study.json')['plan'] for h in methods}
assert all(len(p['rows'])==54 for p in plans.values())
common=[{x['scene']:x['split'] for x in p['rows']} for p in plans.values()]
assert common[0]==common[1]==common[2]
cal={s for s,t in common[0].items() if t=='calibration'}
confirm={s for s,t in common[0].items() if t=='confirmation'}
assert len(cal)==18 and confirm==set(data.scene) and not cal&confirm
heldout=set(plans['H1']['heldout_ids'])
assert len(heldout)==79 and not heldout&(cal|confirm)
raw_checks={}
for h in methods:
 name=h.lower()+'_report_paired_results.csv' if h!='H3' else 'h3_paired_results.csv'
 d=pd.read_csv(SOURCE/name).set_index('scene').loc[data.scene]
 for column in ['anchor','num_views']:
  assert d[column].tolist()==data[column].tolist()
 for source_col,target_col in [('baseline','baseline'),('scalar','scalar'),('structured',h)]:
  np.testing.assert_allclose(d[source_col],data[target_col],atol=1e-12,rtol=0)
 raw_checks[h]='Matched scenes, anchors, image counts, both controls and selected hypothesis scores'
meta=pd.read_csv(SOURCE/'ArchWorld_metadata.csv')
assert meta.scene.nunique()==len(meta)==396
train=read_json('training_metadata.json');assert len(train)==317
assert set(data.scene)<=set(train)
assert not (set(train)&heldout)
audit={'source_manifest_files':len(checks),'source_hash_checks':checks,'confirmation_count':36,
 'calibration_count':18,'selected_count':54,'training_count':317,'full_metadata_count':396,'final_holdout_count':79,
 'calibration_confirmation_disjoint':True,'selected_holdout_disjoint':True,'final_holdout_outcomes_read':False,
 'control_and_selected_result_checks':raw_checks,'confirmation_views':int(data.num_views.sum()),
 'analysis_unit':'one landmark; not images, layers, conditions or exported point samples',
 'source_files':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['paired_hypotheses.csv','ArchWorld_metadata.csv','h1_run_study.json','h2_run_study.json','h3_study.json']}}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
patterns=[];overlap={};agreement=[]
for ref in ['baseline','scalar','both']:
 threshold=data[['baseline','scalar']].min(axis=1).values if ref=='both' else data[ref].values
 wins=data[methods].values<threshold[:,None]-1e-10
 codes=['+'.join(h for h,w in zip(methods,row) if w) or 'none' for row in wins]
 overlap[ref]={'any':int(wins.any(axis=1).sum()),'all_three':int(wins.all(axis=1).sum()),
  'none':int((~wins.any(axis=1)).sum()),'patterns':{c:codes.count(c) for c in sorted(set(codes))}}
 for scene,anchor,code in zip(data.scene,data.anchor,codes):patterns.append({'scene':scene,'anchor':anchor,'reference':ref,'winning_methods':code})
 for i,j in itertools.combinations(range(3),2):
  both=int((wins[:,i]&wins[:,j]).sum());either=int((wins[:,i]|wins[:,j]).sum())
  agreement.append({'reference':ref,'method_a':methods[i],'method_b':methods[j],
   'both_win':both,'only_a':int((wins[:,i]&~wins[:,j]).sum()),'only_b':int((~wins[:,i]&wins[:,j]).sum()),
   'neither_win':int((~wins[:,i]&~wins[:,j]).sum()),'win_jaccard':both/either if either else None})
pd.DataFrame(patterns).to_csv(OUT/'win_patterns.csv',index=False)
pd.DataFrame(agreement).to_csv(OUT/'pairwise_win_overlap.csv',index=False)
(OUT/'overlap_summary.json').write_text(json.dumps(overlap,indent=2),encoding='utf-8')
lines=['# Evidence audit and overlap of improvements','',
 'All '+str(len(checks))+' evidence files match their archived SHA-256 checksums. The three paired-result files reproduce the combined 36-landmark table, including anchors, image counts, both controls and selected outcomes.',
 '', 'The 18 calibration landmarks and 36 confirmation landmarks are disjoint. All 54 are in the 317-landmark training partition and outside the 79-landmark final holdout. Full metadata covers 396 landmarks. No holdout outcomes were read.',
 '', 'Each statistical observation is a landmark. The 1,776 confirmation images are not 1,776 independent evaluation cases.',
 '', '| Reference | At least one H improves | All three improve | None improves |','|---|---:|---:|---:|']
for ref,x in overlap.items():lines.append(f"| {ref} | {x['any']} | {x['all_three']} | {x['none']} |")
lines += ['', '“Both” means lower error than both unmodified VGGT and the scalar control. Counts are descriptive post hoc overlaps. They do not evaluate a selector: picking the winning method after inspecting a landmark’s ground-truth error would leak the evaluation outcome.',
 '', 'H3 often reuses the scalar prediction, so agreement with H3 is not independent replication. Shared controls also make correlations among method-minus-control scores mathematically dependent.',
 '', 'Inputs: the frozen files in `analysis/sources`; exact hashes and audit details are in `audit.json`.']
(OUT/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'audit':'passed','files':len(checks),'overlap':overlap},indent=2))
