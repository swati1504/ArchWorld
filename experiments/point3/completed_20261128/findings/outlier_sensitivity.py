"""Post-hoc deletion sensitivity. Original scores and cohort remain unchanged."""
from pathlib import Path
import csv
import json
import statistics
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'analysis/groups/confirmation_with_groups.csv'
METHODS=('baseline','scalar','H1','H2','H3')
with SOURCE.open(encoding='utf-8',newline='') as handle:
    rows=list(csv.DictReader(handle))
for row in rows:
    for method in METHODS:row[method]=float(row[method])
assert len(rows)==len({r['scene'] for r in rows})==36
top3={}
for method in ('H1','H2','H3'):
    ranked=sorted(rows,key=lambda r: (-(r[method]-r['baseline']),r['scene']))
    assert all(r[method]>r['baseline'] for r in ranked[:3])
    top3[method]=[r['scene'] for r in ranked[:3]]
single=top3['H1'][0]
assert all(v[0]==single for v in top3.values())
union=sorted(set().union(*map(set,top3.values())))
assert len(union)==6
cohorts={
    'full_36':[],
    'remove_shared_largest_regression':[single],
    'remove_H1_worst_three_from_all_methods':top3['H1'],
    'remove_H2_worst_three_from_all_methods':top3['H2'],
    'remove_H3_worst_three_from_all_methods':top3['H3'],
    'remove_union_of_each_hypothesis_worst_three':union,
}
outputs=[]
for name,excluded in cohorts.items():
    kept=[r for r in rows if r['scene'] not in excluded]
    assert len(kept)==36-len(excluded)
    means={m:statistics.mean(r[m] for r in kept) for m in METHODS}
    for method in ('scalar','H1','H2','H3'):
        delta=[r[method]-r['baseline'] for r in kept]
        outputs.append({
            'scenario':name,'method':method,'n_retained':len(kept),'n_removed':len(excluded),
            'original_mean':means['baseline'],'scalar_mean':means['scalar'],
            'method_mean':means[method],
            'mean_difference_vs_original':statistics.mean(delta),
            'percent_change_vs_original':100*(means[method]/means['baseline']-1),
            'percent_change_vs_scalar':100*(means[method]/means['scalar']-1),
            'wins_vs_original':sum(d < -1e-10 for d in delta),
            'ties_vs_original':sum(abs(d)<=1e-10 for d in delta),
            'losses_vs_original':sum(d > 1e-10 for d in delta),
            'excluded_landmarks':' | '.join(excluded),
        })
with (OUT/'outlier_sensitivity.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(outputs[0]))
    writer.writeheader();writer.writerows(outputs)
summary={
    'status':'post-hoc descriptive sensitivity only; no original records changed',
    'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'definition_of_worst':'largest positive method-minus-original normalized Chamfer difference',
    'worst_three_by_hypothesis':top3,'cohorts':cohorts,'results':outputs,
    'interpretation':[
        'Every scenario excludes the same identities from every method and both controls.',
        'Comparing H1/H2/H3 each after its own worst-three removal uses different retained cohorts and is not a fair method ranking.',
        'Outcome-based deletions favor the method used to choose the deletions by construction.',
        'Large regressions have not been established as invalid data or protocol violations.',
        'The full 36-landmark evaluation remains primary; no filtered p-values or superiority claim are provided.',
        'All percentages compare means over the retained cohort, not the original full-cohort baseline.'
    ]
}
(OUT/'outlier_sensitivity.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
for name in ['full_36','remove_shared_largest_regression','remove_union_of_each_hypothesis_worst_three']:
    print(name)
    for r in outputs:
        if r['scenario']==name and r['method']!='scalar':
            print(json.dumps({k:r[k] for k in ['method','n_retained','original_mean','method_mean','percent_change_vs_original','percent_change_vs_scalar']},ensure_ascii=False))
print('Own-worst-three deletion (different cohorts):')
for method in ['H1','H2','H3']:
    r=next(r for r in outputs if r['method']==method and r['scenario']==f'remove_{method}_worst_three_from_all_methods')
    print(json.dumps({k:r[k] for k in ['method','n_retained','original_mean','method_mean','percent_change_vs_original','percent_change_vs_scalar','excluded_landmarks']},ensure_ascii=False))
print('Common union:',union)
