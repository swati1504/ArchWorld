"""Coverage audit and one fixed, retrospective appearance-neighbor selector.

Uses saved image-derived appearance distances, never 2D plot coordinates or
the composite distance containing physical extent. No parameter search.
"""
from pathlib import Path
import csv
import json
import statistics
import collections
import hashlib
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
METHODS = ('scalar', 'H1', 'H2', 'H3')
K = 3
TOL = 1e-10
source = ROOT/'selection/evidence/selection_representation_distances.npz'
confirmation_path = ROOT/'analysis/groups/confirmation_with_groups.csv'
calibration_path = OUT/'calibration_candidates.csv'
plan_path = ROOT/'report/evidence/h1_run_study.json'

def read_csv(path):
    with path.open(encoding='utf-8',newline='') as handle:
        return list(csv.DictReader(handle))

def write_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

protocol = {
    'status':'retrospective exploratory coverage extension; no prospective validation',
    'representation':'saved appearance_distance from image-derived VGGT descriptors',
    'neighbors':K,'neighbor_weights':'uniform',
    'selection_objective':'minimum mean calibration CD_norm among the 3 nearest calibration scenes',
    'candidate_methods':list(METHODS),'method_tie_order':list(METHODS),
    'distance_tie_order':'lexicographic calibration scene name',
    'parameter_search':False,
    'composite_distance_used':False,
    'confirmation_errors_used_to_choose_neighbors_or_method':False,
    'limitations':[
        'This analysis was motivated after inspecting confirmation outcomes.',
        'Original-VGGT calibration errors are absent from these calibration tables; no steering is an evaluation comparator but not a selectable action.',
        'H1/H2 settings and H3 lookup were already selected on the calibration landmarks.',
        'Saved appearance descriptors are inherited from dataset selection; matched reconstruction-view and compute-budget equivalence is not established.',
        'Coverage means every landmark receives a choice, not that similarity predicts intervention response.',
        'No significance claim or new model inference is introduced.'
    ],
    'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [source,confirmation_path,calibration_path,plan_path]}
}
(OUT/'similarity_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

calibration=sorted(read_csv(calibration_path),key=lambda r:r['scene'])
confirmation=read_csv(confirmation_path)
for row in calibration:
    for m in METHODS:row[m]=float(row[m])
for row in confirmation:
    for m in ('baseline',*METHODS):row[m]=float(row[m])
assert len(calibration)==18 and len(confirmation)==36
cal_names=[r['scene'] for r in calibration]
assert len(set(cal_names))==18
assert set(cal_names).isdisjoint(r['scene'] for r in confirmation)
with np.load(source,allow_pickle=False) as arrays:
    names=arrays['training_landmarks'].tolist()
    distances=arrays['appearance_distance']
assert len(names)==len(set(names))==317
assert distances.shape==(317,317) and np.isfinite(distances).all()
assert np.allclose(distances,distances.T,atol=1e-12,rtol=0)
index={name:i for i,name in enumerate(names)}
assert all(r['scene'] in index for r in calibration+confirmation)

records=[]
for row in confirmation:
    neighbor_indices=sorted(range(18),key=lambda i:(float(distances[index[row['scene']],index[cal_names[i]]]),cal_names[i]))[:K]
    neighbors=[calibration[i] for i in neighbor_indices]
    expected={m:statistics.mean(r[m] for r in neighbors) for m in METHODS}
    low=min(expected.values())
    selected=next(m for m in METHODS if abs(expected[m]-low)<=TOL)
    rec={
        'scene':row['scene'],'anchor':row['anchor'],'selected_method':selected,
        'selected_error':row[selected],**{m+'_error':row[m] for m in ('baseline',*METHODS)},
    }
    for j,neighbor in enumerate(neighbors,1):
        rec[f'neighbor_{j}']=neighbor['scene']
        rec[f'neighbor_{j}_anchor']=neighbor['anchor']
        rec[f'neighbor_{j}_distance']=float(distances[index[row['scene']],index[neighbor['scene']]])
    rec.update({f'neighbor_mean_{m}':expected[m] for m in METHODS})
    records.append(rec)

write_csv('similarity_confirmation_predictions.csv',records)
mean=statistics.mean(r['selected_error'] for r in records)
effects={}
for m in ('baseline',*METHODS):
    control=statistics.mean(r[m+'_error'] for r in records)
    deltas=[r['selected_error']-r[m+'_error'] for r in records]
    effects[m]={
        'mean_difference':statistics.mean(deltas),
        'percent_change_of_means':100*(mean/control-1),
        'wins':sum(d < -TOL for d in deltas),'ties':sum(abs(d)<=TOL for d in deltas),
        'losses':sum(d > TOL for d in deltas),
    }

plan=json.loads(plan_path.read_text(encoding='utf-8'))['plan']['rows']
cal_counts=collections.Counter(r['anchor'] for r in calibration)
all_counts=collections.Counter(r['anchor'] for r in plan)
anchor_rows=[]
for anchor in sorted(all_counts):
    rs=[r for r in records if r['anchor']==anchor]
    anchor_rows.append({
        'anchor':anchor,'n_all54':all_counts[anchor],'n_calibration':cal_counts[anchor],
        'n_confirmation':len(rs),'exact_anchor_has_two_calibration_cases':cal_counts[anchor]>=2,
        'similarity_choices':json.dumps(dict(collections.Counter(r['selected_method'] for r in rs))),
        'similarity_mean_error':statistics.mean(r['selected_error'] for r in rs) if rs else '',
        'original_mean_error':statistics.mean(r['baseline_error'] for r in rs) if rs else '',
        'mean_delta_vs_original':statistics.mean(r['selected_error']-r['baseline_error'] for r in rs) if rs else '',
    })
write_csv('coverage_all_anchor_groups.csv',anchor_rows)
old=json.loads((OUT/'selection_audit.json').read_text(encoding='utf-8'))
highlighted={'facade','column','statue','window'}
coverage={
    'previously_highlighted_groups':sorted(highlighted),
    'highlighted_calibration':sum(r['anchor'] in highlighted for r in calibration),
    'highlighted_confirmation':sum(r['anchor'] in highlighted for r in confirmation),
    'highlighted_all54':sum(r['anchor'] in highlighted for r in plan),
    'categorical_selectors':{
        field:{'direct_group_support':p['group_supported_confirmation'],
               'fallback':36-p['group_supported_confirmation'],
               'evaluated_confirmation':36}
        for field,p in old['policies'].items()
    },
    'similarity_selector':{'assigned':36,'omitted':0,'neighbors_per_scene':K}
}
summary={
    'protocol':protocol,'coverage':coverage,'mean_error':mean,
    'method_counts':dict(collections.Counter(r['selected_method'] for r in records)),
    'effects':effects
}
(OUT/'coverage_similarity_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
assert len(records)==len({r['scene'] for r in records})==36
assert sum(r['n_confirmation'] for r in anchor_rows)==36
assert sum(r['n_all54'] for r in anchor_rows)==54
for r in records:
    assert r['selected_error']==r[r['selected_method']+'_error']
    assert all(r[f'neighbor_{i}'] in cal_names for i in range(1,K+1))
print(json.dumps({k:v for k,v in summary.items() if k!='protocol'},indent=2))
