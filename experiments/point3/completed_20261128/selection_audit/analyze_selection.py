"""Exploratory categorical method selection using existing, frozen predictions.

No new inference, outcome-dependent category merges, thresholds, or p-values.
Policies are fit using calibration errors only, but this audit was initiated
after confirmation results had been inspected. It is not a prospective test.
"""
from pathlib import Path
import csv
import json
import collections
import statistics
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'report/evidence'
CONFIRM = ROOT / 'analysis/groups/confirmation_with_groups.csv'
METHODS = ('scalar', 'H1', 'H2', 'H3')
GROUPINGS = ('anchor', 'building_type', 'region')
MIN_SUPPORT = 2  # Existing H3 minimum-support convention, not optimized here.
TOL = 1e-10

def read_json(name):
    return json.loads((SOURCE / name).read_text(encoding='utf-8'))

def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def means(rows, methods=METHODS):
    return {method: statistics.mean(r[method] for r in rows) for method in methods}

def winners(scores):
    low = min(scores.values())
    return [m for m, value in scores.items() if abs(value-low) <= TOL]

def choose(rows):
    # Deterministic tie order favors the simpler scalar when identical.
    return winners(means(rows))[0]

def grouped(rows, field):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return groups

def effect(rows, method, control):
    deltas = [r[method]-r[control] for r in rows]
    return {
        'mean_error': statistics.mean(r[method] for r in rows),
        'mean_difference': statistics.mean(deltas),
        'pct_change_of_means': 100*(statistics.mean(r[method] for r in rows)
                                    / statistics.mean(r[control] for r in rows)-1),
        'wins': sum(d < -TOL for d in deltas),
        'ties': sum(abs(d) <= TOL for d in deltas),
        'losses': sum(d > TOL for d in deltas),
    }

plan = read_json('h1_run_study.json')['plan']
assert read_json('h2_run_study.json')['plan']['rows'] == plan['rows']
assert read_json('h3_study.json')['plan']['rows'] == plan['rows']
c1 = read_json('h1_run_calibration_scores.json')
c2 = read_json('h2_run_calibration_scores.json')
c3 = read_json('h3_calibration.json')
h1_selection = read_json('h1_run_selection.json')['H1']
h2_selection = read_json('h2_run_selection.json')['H2']
h3_selection = read_json('h3_selection.json')['H3']
calibration = []
for row in plan['rows']:
    if row['split'] != 'calibration':
        continue
    scene = row['scene']
    assert c1[scene]['scalar_eta1000'] == c2[scene]['scalar_eta1000'] == c3[scene]['scalar_eta1000']
    r = dict(row)
    r.update(scalar=c1[scene]['scalar_eta1000'], H1=c1[scene][h1_selection],
             H2=c2[scene][h2_selection], H3=c3[scene][h3_selection[row['anchor']]])
    calibration.append(r)
with CONFIRM.open(encoding='utf-8', newline='') as handle:
    confirmation = list(csv.DictReader(handle))
for row in confirmation:
    for method in (*METHODS, 'baseline'):
        row[method] = float(row[method])
assert len(calibration) == 18 and len(confirmation) == 36
assert set(r['scene'] for r in calibration).isdisjoint(r['scene'] for r in confirmation)
assert set(r['scene'] for r in confirmation) == {r['scene'] for r in plan['rows'] if r['split']=='confirmation'}
global_choice = choose(calibration)

group_rows, policy_rows, policy_summary = [], [], {}
for field in GROUPINGS:
    cg, vg = grouped(calibration, field), grouped(confirmation, field)
    choices = {}
    for group in sorted(set(cg) | set(vg)):
        cal, conf = cg[group], vg[group]
        supported = len(cal) >= MIN_SUPPORT
        choice = choose(cal) if supported else global_choice
        choices[group] = choice
        cm = means(cal) if cal else {}
        vm = means(conf) if conf else {}
        best_conf = winners(means(conf, ('baseline', *METHODS))) if conf else []
        loo_cal = [choose(cal[:i]+cal[i+1:]) for i in range(len(cal))] if len(cal)>1 else []
        cf_effect = effect(conf, choice, 'scalar') if conf else {}
        cb_effect = effect(conf, choice, 'baseline') if conf else {}
        cal_effect = effect(cal, choice, 'scalar') if cal else {}
        best_count = sum(choice in winners({m:r[m] for m in ('baseline', *METHODS)}) for r in conf)
        group_rows.append({
            'grouping': field, 'group': group, 'n_calibration': len(cal),
            'n_confirmation': len(conf), 'minimum_support_met': supported,
            'calibration_best_interventions': '|'.join(winners(cm)) if cm else '',
            'selected_method': choice, 'selection_source': 'group_calibration' if supported else 'global_calibration_fallback',
            'confirmation_best_interventions': '|'.join(winners(vm)) if vm else '',
            'confirmation_best_including_original': '|'.join(best_conf),
            'calibration_percent_vs_scalar': cal_effect.get('pct_change_of_means',''),
            'calibration_wins_vs_scalar': cal_effect.get('wins',''),
            'confirmation_percent_vs_scalar': cf_effect.get('pct_change_of_means',''),
            'confirmation_wins_vs_scalar': cf_effect.get('wins',''),
            'confirmation_percent_vs_original': cb_effect.get('pct_change_of_means',''),
            'confirmation_wins_vs_original': cb_effect.get('wins',''),
            'confirmation_selected_method_is_best_count_including_ties': best_count,
            'calibration_choice_retained_in_each_fixed_candidate_omission':
                all(m==choice for m in loo_cal) if loo_cal and supported else '',
            'calibration_omission_choices': '|'.join(loo_cal),
            **{f'calibration_mean_{m}': cm.get(m,'') for m in METHODS},
            **{f'confirmation_mean_{m}': vm.get(m,'') for m in METHODS},
        })
    evaluated=[]
    for row in confirmation:
        choice=choices[row[field]]
        r=dict(row, selected=row[choice])
        evaluated.append(r)
        policy_rows.append({
            'policy':field,'scene':row['scene'],'group':row[field],
            'n_group_calibration':len(cg[row[field]]),
            'selected_method':choice,'selected_error':row[choice],
            'original_error':row['baseline'],'scalar_error':row['scalar'],
            'H1_error':row['H1'],'H2_error':row['H2'],'H3_error':row['H3'],
        })
    policy_summary[field] = {
        'mean_error':statistics.mean(r['selected'] for r in evaluated),
        'method_counts':dict(collections.Counter(r['selected_method'] for r in policy_rows if r['policy']==field)),
        'group_supported_confirmation':sum(len(cg[r[field]])>=MIN_SUPPORT for r in confirmation),
        'effects':{control:effect(evaluated,'selected',control) for control in ('baseline',*METHODS)},
    }

write_csv('group_calibration_confirmation.csv',group_rows)
write_csv('policy_confirmation_predictions.csv',policy_rows)
calibration_rows=[{key:r[key] for key in ('scene','anchor','building_type','region',*METHODS)} for r in calibration]
write_csv('calibration_candidates.csv',calibration_rows)
summary = {
    'status':'exploratory retrospective audit, not prospective validation',
    'n_calibration':18,'n_confirmation':36,'candidate_methods':list(METHODS),
    'minimum_group_support':MIN_SUPPORT,'tie_tolerance':TOL,'tie_order':list(METHODS),
    'global_calibration_choice':global_choice,'global_calibration_means':means(calibration),
    'confirmation_fixed_means':means(confirmation,('baseline',*METHODS)),
    'policies':policy_summary,
    'limitations':[
        'The audit was initiated after inspecting confirmation outcomes and promising subgroups.',
        'Rule fitting uses calibration errors only; this does not erase post-hoc analysis selection.',
        'Only existing anchor, building-type, and region categories are used; no outcome-driven merges or cutoffs.',
        'Original-model calibration errors are absent from these calibration-score files, so no-steering cannot be a learned candidate in this audit.',
        'Fixed H1/H2 hyperparameters and H3 lookup were themselves fitted on these calibration landmarks.',
        'Calibration omission checks retain those frozen candidates; they are not nested cross-validation.',
        'Subgroup estimates use small, overlapping, deliberately selected cohorts.',
        'No new p-values, multiple-testing discoveries, model runs, or final-evaluation outcomes are introduced.'
    ],
    'evidence_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        SOURCE/'h1_run_study.json',SOURCE/'h1_run_calibration_scores.json',
        SOURCE/'h2_run_calibration_scores.json',SOURCE/'h3_calibration.json',
        SOURCE/'h1_run_selection.json',SOURCE/'h2_run_selection.json',SOURCE/'h3_selection.json',CONFIRM]},
}
(OUT/'selection_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('evidence_sha256','limitations')},indent=2))
print('Promising categories, plus contrasting arches:')
for row in group_rows:
    if (row['grouping'],row['group']) in [('anchor','facade'),('anchor','statue'),('anchor','window'),('anchor','arch'),('anchor','column'),('building_type','Misc'),('region','North America')]:
        print(json.dumps(row,ensure_ascii=False))
