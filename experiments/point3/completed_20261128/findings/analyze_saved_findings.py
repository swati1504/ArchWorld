"""Descriptive findings from frozen outputs only. No new inference or tuning."""
from pathlib import Path
import csv
import json
import statistics
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'report/evidence'
CONFIRM=ROOT/'analysis/groups/confirmation_with_groups.csv'
DIAGNOSTICS=ROOT/'analysis/correlations/joined_confirmation_features.csv'
METHODS=('baseline','scalar','H1','H2','H3')
INTERVENTIONS=METHODS[1:]
TOL=1e-10

def load_csv(path):
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))
def load_json(name):return json.loads((SOURCE/name).read_text(encoding='utf-8'))
def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

rows=load_csv(CONFIRM)
for r in rows:
    for m in METHODS:r[m]=float(r[m])
assert len(rows)==len({r['scene'] for r in rows})==36
means={m:statistics.mean(r[m] for r in rows) for m in METHODS}
oracle_rows=[]
for r in rows:
    all_best=min(r[m] for m in METHODS)
    intervention_best=min(r[m] for m in INTERVENTIONS)
    safe_h1=min(r['baseline'],r['H1'])
    oracle_rows.append({
        'scene':r['scene'],'anchor':r['anchor'],**{m+'_error':r[m] for m in METHODS},
        'oracle_interventions_only':intervention_best,
        'oracle_all_including_original':all_best,
        'oracle_H1_or_original':safe_h1,
        'oracle_best_methods_including_ties':'|'.join(m for m in METHODS if abs(r[m]-all_best)<=TOL),
        'oracle_available_gain_vs_original':r['baseline']-all_best,
        'H1_available_gain_vs_original':r['baseline']-safe_h1,
        'H1_harm_vs_original':max(r['H1']-r['baseline'],0),
    })
write_csv('oracle_landmarks.csv',oracle_rows)
oracle={}
for name in ['oracle_interventions_only','oracle_all_including_original','oracle_H1_or_original']:
    value=statistics.mean(r[name] for r in oracle_rows)
    oracle[name]={'mean_error':value,'percent_reduction_vs_original':100*(1-value/means['baseline'])}
oracle['H1_abstention_fraction_of_all_oracle_gain']=(means['baseline']-oracle['oracle_H1_or_original']['mean_error'])/(means['baseline']-oracle['oracle_all_including_original']['mean_error'])
oracle['landmarks_with_any_intervention_better_than_original']=sum(min(r[m] for m in INTERVENTIONS)<r['baseline']-TOL for r in rows)
oracle['landmarks_with_original_best']=sum(r['baseline']<=min(r[m] for m in INTERVENTIONS)+TOL for r in rows)

harm=[]
for m in INTERVENTIONS:
    gains=[r['baseline']-r[m] for r in rows if r[m]<r['baseline']-TOL]
    regressions=sorted([(r[m]-r['baseline'],r['scene']) for r in rows if r[m]>r['baseline']+TOL],reverse=True)
    gross_gain=sum(gains);gross_harm=sum(d for d,_ in regressions)
    assert abs((gross_harm-gross_gain)/36-(means[m]-means['baseline']))<1e-12
    harm.append({
        'method':m,'improved':len(gains),'worsened':len(regressions),
        'gross_error_reductions_sum':gross_gain,'gross_error_increases_sum':gross_harm,
        'mean_gain_contribution':gross_gain/36,'mean_harm_contribution':gross_harm/36,
        'net_mean_difference':(gross_harm-gross_gain)/36,
        'largest_regression_share_of_gross_harm':regressions[0][0]/gross_harm,
        'top_three_regression_share_of_gross_harm':sum(d for d,_ in regressions[:3])/gross_harm,
        'top_three_landmarks':' | '.join(scene for _,scene in regressions[:3]),
    })
write_csv('benefit_harm_decomposition.csv',harm)

diagnostics=load_csv(DIAGNOSTICS)
proxy=[]
for m in ['H1','H2','H3']:
    ds=[r for r in diagnostics if r['hypothesis']==m]
    assert len(ds)==36 and {r['scene'] for r in ds}=={r['scene'] for r in rows}
    all_accepted=[r for r in ds if int(float(r['accepted_layers']))==24]
    harmful=[r for r in all_accepted if float(r['hypothesis_error'])>float(r['baseline_error'])+TOL]
    proxy.append({'method':m,'n':36,'all_24_layers_accepted':len(all_accepted),
                  'all_layers_accepted_but_error_worse_than_original':len(harmful),
                  'fraction_harmful_despite_all_layers_accepted':len(harmful)/len(all_accepted)})
write_csv('attention_acceptance_failures.csv',proxy)

plan=load_json('h3_study.json')['plan']['rows']
meta={r['scene']:r for r in plan}
c2=load_json('h2_run_calibration_scores.json')
c3=load_json('h3_calibration.json')
selection=load_json('h3_selection.json')['H3']
matched=[]
for scene,values in c2.items():
    assert values['scalar_eta100']==c3[scene]['scalar_eta100']
    assert meta[scene]['split']=='calibration'
    matched.append({'scene':scene,'anchor':meta[scene]['anchor'],'split':'calibration',
                    'scalar_eta100':values['scalar_eta100'],'view_eta100':values['view_eta100'],
                    'difference':values['view_eta100']-values['scalar_eta100']})
for r in rows:
    if selection[r['anchor']]=='scalar_eta100':
        matched.append({'scene':r['scene'],'anchor':r['anchor'],'split':'confirmation',
                        'scalar_eta100':r['H3'],'view_eta100':r['H2'],
                        'difference':r['H2']-r['H3']})
write_csv('available_H2_matched_strength.csv',matched)
matched_summary={}
for split in ['calibration','confirmation']:
    subset=[r for r in matched if r['split']==split]
    sm=statistics.mean(r['scalar_eta100'] for r in subset)
    vm=statistics.mean(r['view_eta100'] for r in subset)
    matched_summary[split]={
        'n':len(subset),'scalar_eta100_mean':sm,'view_eta100_mean':vm,
        'mean_difference':vm-sm,'percent_change_of_means':100*(vm/sm-1),
        'view_wins':sum(r['difference'] < -TOL for r in subset),
        'view_losses':sum(r['difference'] > TOL for r in subset),
    }
assert matched_summary['calibration']['n']==18
assert matched_summary['confirmation']['n']==3

results={
    'status':'descriptive audit of saved outputs; no new model runs',
    'n_confirmation_landmarks':36,'confirmation_means':means,
    'observed_candidate_oracle_diagnostics':oracle,
    'benefit_harm_decomposition':harm,'attention_acceptance':proxy,
    'available_H2_matched_strength':matched_summary,
    'limitations':[
        'Oracle scores select using ground-truth errors and are unattainable hindsight references, not achieved selector performance.',
        'The candidate oracle covers only the measured conditions, not a complete strength sweep or all possible interventions.',
        'Worst-three shares are post-hoc descriptions; the corresponding landmarks remain in every headline mean.',
        'Acceptance failure counts show a monitored-attention increase is insufficient to guarantee geometry improvement, not that attention causes the regression.',
        'H3 often duplicates scalar predictions; method-scene observations are not independent landmarks.',
        'The H2 matched-strength confirmation subset consists only of two statues and one column that received eta100 under H3; it is not representative.',
        'Calibration matched-strength results are in-sample and must not be pooled with confirmation.',
        'No additional hypothesis tests or claims of statistically significant superiority are made.'
    ],
    'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        CONFIRM,DIAGNOSTICS,SOURCE/'h2_run_calibration_scores.json',SOURCE/'h3_calibration.json',
        SOURCE/'h3_study.json',SOURCE/'h3_selection.json']}
}
(OUT/'saved_findings.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in results.items() if k not in ('sources','limitations')},indent=2,ensure_ascii=False))
