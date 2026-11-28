"""Exploratory, landmark-level correlations using frozen confirmation settings.

Run with the bundled Python. No experiment files are modified.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'sources'
SEED = 20261126
PERMUTATIONS = 49999
BOOTSTRAPS = 4000
FILES = ['paired_hypotheses.csv', 'training_metadata.json', 'ArchWorld_metadata.csv',
         'h1_run_strength_diagnostics.csv', 'h2_run_strength_diagnostics.csv', 'h3_strength_diagnostics.csv',
         'h1_run_selection.json', 'h2_run_selection.json', 'h3_selection.json',
         'h1_run_study.json', 'h2_run_study.json', 'h3_study.json']

def read_json(name):
    return json.loads((SOURCE / name).read_text(encoding='utf-8'))

def read_csv(name):
    with (SOURCE / name).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def write_csv(name, rows):
    if not rows:
        return
    with (ROOT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_values[1:] != sorted_values[:-1])]
    ends = np.r_[starts[1:], len(values)]
    out = np.empty(len(values), dtype=float)
    out[order] = np.repeat((starts + ends - 1) / 2 + 1, ends - starts)
    return out

def standardized_ranks(values):
    result = ranks(values); result -= result.mean()
    norm = np.linalg.norm(result)
    return result / norm if norm else np.full(len(result), np.nan)

def spearman(x, y):
    return float(standardized_ranks(x) @ standardized_ranks(y))

def bh(values):
    p = np.array(values, dtype=float)
    order = np.argsort(p); m = len(p)
    sorted_q = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(1, sorted_q)
    return out.tolist()

def rank_adjusted_for_baseline(x, y, baseline):
    # Descriptive sensitivity only: baseline appears in the paired outcome.
    z = np.column_stack([np.ones(len(x)), ranks(baseline)])
    xr = ranks(x) - z @ np.linalg.lstsq(z, ranks(x), rcond=None)[0]
    yr = ranks(y) - z @ np.linalg.lstsq(z, ranks(y), rcond=None)[0]
    denominator = np.linalg.norm(xr) * np.linalg.norm(yr)
    return float(xr @ yr / denominator) if denominator > 1e-12 else None

def correlation_stats(x, y, scenes, rng, uncertainty=True):
    rho = spearman(x, y)
    xr, yr = standardized_ranks(x), standardized_ranks(y)
    count = 0
    for start in range(0, PERMUTATIONS, 2000):
        size = min(2000, PERMUTATIONS - start)
        permutations = np.argsort(rng.random((size, len(y))), axis=1)
        null = yr[permutations] @ xr
        count += int(np.count_nonzero(np.abs(null) >= abs(rho) - 1e-12))
    result = {'n': len(x), 'spearman_rho': rho,
              'permutation_p_two_sided': (count + 1) / (PERMUTATIONS + 1)}
    if uncertainty:
        boot = []
        for _ in range(BOOTSTRAPS):
            ix = rng.integers(0, len(y), len(y)); value = spearman(x[ix], y[ix])
            if math.isfinite(value): boot.append(value)
        low, high = np.quantile(boot, [.025, .975])
        loo = [spearman(np.delete(x, i), np.delete(y, i)) for i in range(len(y))]
        valid_loo = [v for v in loo if math.isfinite(v)]
        worst_index = int(np.nanargmax(np.abs(np.array(loo) - rho)))
        result.update(bootstrap_ci_low=float(low), bootstrap_ci_high=float(high),
                      bootstrap_valid_replicates=len(boot),
                      bootstrap_ci_scope=('Pointwise unadjusted percentile interval' if len(boot) == BOOTSTRAPS
                                          else 'Conditional on nonconstant bootstrap outcomes; not an unconditional 95% interval'),
                      loo_rho_min=min(valid_loo), loo_rho_max=max(valid_loo),
                      loo_sign_changes=int(sum(np.sign(v) != np.sign(rho) for v in valid_loo)),
                      most_influential_omitted_scene=scenes[worst_index],
                      rho_omitting_most_influential=loo[worst_index])
    return result

def main():
    global ROOT, SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT)
    arguments = parser.parse_args()
    ROOT, SOURCE = arguments.output.resolve(), arguments.source.resolve()
    ROOT.mkdir(exist_ok=True, parents=True)
    paired = read_csv('paired_hypotheses.csv')
    assert len(paired) == 36 and len({r['scene'] for r in paired}) == 36
    scenes = [r['scene'] for r in paired]
    training = read_json('training_metadata.json')
    metadata_rows = read_csv('ArchWorld_metadata.csv')
    assert len({r['scene'] for r in metadata_rows}) == len(metadata_rows)
    metadata = {r['scene']: r for r in metadata_rows}
    sources = {f: hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES}
    selected = {}; records = []; audits = {}; excluded = []
    numeric_columns = ['accepted_layers', 'coefficient_min', 'coefficient_max',
                       'clipped_view_steps', 'mean_applied_relative_update']
    for h in ('H1', 'H2', 'H3'):
        prefix = h.lower(); infix = '_run' if h != 'H3' else ''
        select = read_json(prefix+infix+'_selection.json')
        plan = read_json(prefix+infix+'_study.json')['plan']
        confirmation = {r['scene'] for r in plan['rows'] if r['split'] == 'confirmation'}
        assert confirmation == set(scenes)
        assert not (set(select['calibration_scenes']) & confirmation)
        rows = read_csv(prefix+infix+'_strength_diagnostics.csv')
        lookup = {(r['scene'], r['condition']): r for r in rows}
        assert len(lookup) == len(rows), 'Duplicate scene-condition diagnostics'
        selected[h] = select[h]
        per_h = []
        for p in paired:
            scene = p['scene']; condition = select[h] if h != 'H3' else select[h][p['anchor']]
            diag = lookup[scene, condition]; scalar = lookup[scene, select['global_scalar']]
            meta = training[scene]
            assert math.isclose(float(meta['extent_m']), float(metadata[scene]['Extent']), rel_tol=1e-12)
            r = {'hypothesis': h, 'scene': scene, 'anchor': p['anchor'], 'region': meta['region'],
                 'building_type': meta['building_type'], 'condition': condition,
                 'num_views': int(p['num_views']), 'extent_m': float(meta['extent_m']),
                 'baseline_error': float(p['baseline']), 'scalar_error': float(p['scalar']),
                 'hypothesis_error': float(p[h]),
                 'scalar_mean_applied_relative_update': float(scalar['mean_applied_relative_update'])}
            r.update({k: float(diag[k]) for k in numeric_columns})
            r['coefficient_max_to_min'] = r['coefficient_max'] / r['coefficient_min']
            for control in ('baseline', 'scalar'):
                r['delta_vs_'+control] = r['hypothesis_error'] - r[control+'_error']
                r['log_ratio_vs_'+control] = math.log(r['hypothesis_error'] / r[control+'_error'])
            records.append(r); per_h.append(r)
        audits[h] = {'diagnostic_rows': len(rows), 'unique_scene_condition_rows': len(lookup),
                     'confirmation_joined': len(per_h), 'missing_confirmation_rows': 0,
                     'calibration_scenes_excluded': len(select['calibration_scenes']),
                     'frozen_setting': select[h]}
    feature_sets = {
        'H1': ['extent_m', 'coefficient_min', 'coefficient_max', 'coefficient_max_to_min',
               'scalar_mean_applied_relative_update'],
        'H2': ['extent_m', 'coefficient_min', 'coefficient_max', 'coefficient_max_to_min',
               'mean_applied_relative_update', 'scalar_mean_applied_relative_update'],
        'H3': ['extent_m', 'mean_applied_relative_update', 'scalar_mean_applied_relative_update']}
    for h in selected:
        excluded.extend([{'hypothesis': h, 'feature': 'accepted_layers', 'reason': 'Constant 24 in all 36 selected runs'},
                         {'hypothesis': h, 'feature': 'clipped_view_steps', 'reason': 'Constant zero in all 36 selected runs'},
                         {'hypothesis': h, 'feature': 'num_views', 'reason': '35 scenes have 50, one has 26; singleton contrast, not a useful continuous correlation'},
                         {'hypothesis': h, 'feature': 'image_count/duplicate_fraction/median_megapixels', 'reason': 'Not present in assigned source metadata; num_views is not substituted for these features'}])
    excluded.extend([{'hypothesis': 'H1', 'feature': 'mean_applied_relative_update', 'reason': 'Fixed target 0.02; variation under 1.2e-9 is floating-point noise'},
                     {'hypothesis': 'H3', 'feature': 'coefficient_min/coefficient_max', 'reason': 'Binary policy 100 for 3 scenes vs 1000 for 33; analyzed as policy/category distinction elsewhere'},
                     {'hypothesis': 'H3', 'feature': 'coefficient_max_to_min', 'reason': 'Constant 1'},
                     {'hypothesis': 'all', 'feature': 'Model_Diagonal/Scale', 'reason': 'Arbitrary reconstruction units or linked metric scale; use physically interpretable extent_m rather than expanding a redundant/coupled test family'},
                     {'hypothesis': 'all', 'feature': 'baseline_error', 'reason': 'Mathematically coupled to H-baseline; separate descriptive diagnostic with no inferential p value'}])
    protocol = {
        'seed': SEED, 'permutations_per_test': PERMUTATIONS, 'bootstrap_replicates': BOOTSTRAPS,
        'unit': 'landmark', 'split': 'confirmation only', 'n': 36,
        'primary_outcomes': ['hypothesis error minus baseline error', 'hypothesis error minus scalar error'],
        'positive_delta': 'worse than comparator; positive rho means larger feature associated with worse paired effect',
        'feature_sets': feature_sets, 'primary_test_count': 28,
        'primary_multiplicity': 'BH across all 28 primary feature-outcome tests, pooled across H1/H2/H3 and both comparators',
        'sensitivity_outcomes': 'log(hypothesis_error / comparator_error), identical 28 feature-outcome tests',
        'sensitivity_multiplicity': 'Separate BH across 28 sensitivity tests plus joint BH across all 56 primary+sensitivity tests',
        'permutation': 'Two-sided Spearman rho; average ranks for ties, permutation of outcome ranks, (exceedances+1)/(B+1)',
        'bootstrap': 'Paired resampling of 36 landmarks; pointwise unadjusted percentile interval; ties reranked each replicate. H3-versus-scalar intervals are conditional on nonconstant resamples and must not be read as unconditional 95% intervals; undefined replicates are discarded and counted.',
        'rank_adjustment': 'Descriptive residual correlation after separately regressing feature and outcome ranks on intercept+baseline-error ranks; no p value; does not establish causal/confounding control',
        'scope_limit': 'Exploratory associations within nonrandom selected confirmation scenes; not random-sample population confidence or causal evidence',
        'excluded_features': excluded, 'source_sha256': sources, 'join_audit': audits,
        'protocol_written_before_outcome_correlation_calculation': True}
    (ROOT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    write_csv('joined_confirmation_features.csv', records)
    write_csv('excluded_features.csv', excluded)
    rng = np.random.default_rng(SEED)
    primary = []; sensitivity = []; coupled = []
    for h in ('H1','H2','H3'):
        rows = [r for r in records if r['hypothesis'] == h]
        baseline = np.array([r['baseline_error'] for r in rows])
        for comparator in ('baseline','scalar'):
            delta = np.array([r['delta_vs_'+comparator] for r in rows])
            logratio = np.array([r['log_ratio_vs_'+comparator] for r in rows])
            coupled.append({'hypothesis': h, 'comparator': comparator,
                            'rho_baseline_error_vs_delta': spearman(baseline, delta),
                            'rho_baseline_error_vs_log_ratio': spearman(baseline, logratio),
                            'warning': 'Diagnostic only. Baseline/comparator and change share terms; cannot infer hard scenes improve.'})
            for feature in feature_sets[h]:
                x = np.array([r[feature] for r in rows]); assert np.isfinite(x).all()
                assert len(np.unique(x)) > 1
                item = {'test_id': f'{h}__{comparator}__{feature}__delta',
                        'hypothesis': h, 'comparator': comparator, 'feature': feature,
                        'outcome': 'signed_error_difference'}
                item.update(correlation_stats(x, delta, scenes, rng))
                item['baseline_adjusted_rank_rho_descriptive'] = rank_adjusted_for_baseline(x, delta, baseline)
                primary.append(item)
                item = {'test_id': f'{h}__{comparator}__{feature}__log_ratio',
                        'hypothesis': h, 'comparator': comparator, 'feature': feature, 'outcome': 'log_error_ratio'}
                item.update(correlation_stats(x, logratio, scenes, rng, uncertainty=False))
                sensitivity.append(item)
    assert len(primary) == len(sensitivity) == 28
    for family in (primary, sensitivity):
        for row, q in zip(family, bh([r['permutation_p_two_sided'] for r in family])):
            row['bh_q_within_family'] = q
    for row, q in zip(primary+sensitivity, bh([r['permutation_p_two_sided'] for r in primary+sensitivity])):
        row['bh_q_all_56'] = q
    primary.sort(key=lambda x: x['permutation_p_two_sided']); sensitivity.sort(key=lambda x: x['permutation_p_two_sided'])
    write_csv('primary_correlations.csv', primary)
    write_csv('log_ratio_sensitivity.csv', sensitivity)
    write_csv('coupled_baseline_diagnostics.csv', coupled)
    summary = {'primary': primary, 'sensitivity': sensitivity, 'coupled_diagnostics': coupled,
               'primary_discoveries_q05': [r for r in primary if r['bh_q_within_family'] <= .05],
               'joint_discoveries_q05': [r for r in primary+sensitivity if r['bh_q_all_56'] <= .05],
               'protocol': protocol}
    (ROOT/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'primary_tests':len(primary),'primary_discoveries':len(summary['primary_discoveries_q05']),
                      'joint_discoveries':len(summary['joint_discoveries_q05']), 'top_primary':primary[:5],
                      'top_sensitivity':sensitivity[:3]},indent=2))

if __name__ == '__main__':
    main()
