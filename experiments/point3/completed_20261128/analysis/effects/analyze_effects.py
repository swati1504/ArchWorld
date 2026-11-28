"""Paired landmark-level effects; frozen 36-landmark confirmation set only.

Run with bundled Python + NumPy. All inputs are read-only; outputs stay beside
this script. Negative changes favor the hypothesis. No new model evaluation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path,
                    default=SCRIPT_DIRECTORY.parent / 'sources',
                    help='Read-only evidence directory containing paired_hypotheses.csv and frozen JSON files.')
parser.add_argument('--output', type=Path, default=SCRIPT_DIRECTORY,
                    help='Destination for derived CSV, JSON and Markdown files.')
args = parser.parse_args()
ROOT, EVIDENCE = args.output.resolve(), args.source.resolve()
assert ROOT != EVIDENCE, 'Derived outputs must not overwrite the raw evidence directory.'
ROOT.mkdir(parents=True, exist_ok=True)
SEED = 20261126
BOOTSTRAPS = 100_000
TIE_ATOL = 1e-12
INPUTS = ['paired_hypotheses.csv', 'verified_statistics.json', 'h1_run_selection.json',
          'h2_run_selection.json', 'h3_selection.json', 'h1_run_calibration_scores.json',
          'h2_run_calibration_scores.json', 'h3_calibration.json', 'h3_study.json']

def load_json(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))

def save_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def save_csv(name, rows):
    with (ROOT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def ci(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]

def exact_sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)

with (EVIDENCE / 'paired_hypotheses.csv').open(encoding='utf-8', newline='') as handle:
    rows = sorted(list(csv.DictReader(handle)), key=lambda r: r['scene'])
assert len(rows) == 36 and len({r['scene'] for r in rows}) == 36
study = load_json('h3_study.json')['plan']['rows']
confirmation = {r['scene'] for r in study if r['split'] == 'confirmation'}
calibration = {r['scene'] for r in study if r['split'] == 'calibration'}
assert {r['scene'] for r in rows} == confirmation
assert len(calibration) == 18 and not calibration.intersection(confirmation)
arrays = {c: np.array([float(r[c]) for r in rows]) for c in ['baseline', 'scalar', 'H1', 'H2', 'H3']}
assert all(np.isfinite(a).all() and (a > 0).all() for a in arrays.values())
prior = load_json('verified_statistics.json')
for c, a in arrays.items():
    assert abs(a.mean() - prior['conditions'][c]['mean']) < 1e-10
for h in ['H1', 'H2', 'H3']:
    recorded = np.array([float(r[h + '_minus_scalar']) for r in rows])
    np.testing.assert_allclose(arrays[h] - arrays['scalar'], recorded, atol=1e-12, rtol=1e-12)

rng = np.random.default_rng(SEED)
indices = rng.integers(0, len(rows), size=(BOOTSTRAPS, len(rows)), dtype=np.int32)
comparisons, influence, condition_rows = [], [], []
for c, a in arrays.items():
    lo, hi = ci(a[indices].mean(axis=1))
    condition_rows.append(dict(condition=c, n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                               mean_ci95_low=lo, mean_ci95_high=hi))
for h in ['H1', 'H2', 'H3']:
    for control in ['baseline', 'scalar']:
        a, b = arrays[h], arrays[control]
        d = a - b
        percent = 100 * d / b
        boot_delta = d[indices]
        mean_ci = ci(boot_delta.mean(axis=1))
        median_ci = ci(np.median(boot_delta, axis=1))
        ratio_ci = ci(100 * (a[indices].mean(axis=1) / b[indices].mean(axis=1) - 1))
        percent_ci = ci(percent[indices].mean(axis=1))
        wins, losses, ties = int((d < -TIE_ATOL).sum()), int((d > TIE_ATOL).sum()), int((np.abs(d) <= TIE_ATOL).sum())
        leave_one_out = (d.sum() - d) / (len(d) - 1)
        k = int(np.floor(.1 * len(d)))
        ordered = np.sort(d)
        best, worst = int(d.argmin()), int(d.argmax())
        r = dict(hypothesis=h, control=control, n=len(d), mean_delta=float(d.mean()),
                 mean_delta_ci95_low=mean_ci[0], mean_delta_ci95_high=mean_ci[1],
                 median_paired_delta=float(np.median(d)), median_delta_ci95_low=median_ci[0], median_delta_ci95_high=median_ci[1],
                 percent_change_of_means=float(100 * (a.mean() / b.mean() - 1)),
                 percent_change_of_means_ci95_low=ratio_ci[0], percent_change_of_means_ci95_high=ratio_ci[1],
                 mean_landmark_percent_change=float(percent.mean()),
                 mean_landmark_percent_ci95_low=percent_ci[0], mean_landmark_percent_ci95_high=percent_ci[1],
                 median_landmark_percent_change=float(np.median(percent)),
                 wins=wins, ties=ties, losses=losses, exact_two_sided_sign_p=exact_sign_p(wins, losses),
                 sign_test_n_non_ties=wins+losses,
                 trimmed_mean_delta_exploratory=float(ordered[k:len(d)-k].mean()),
                 trim_each_tail_count=k, retained_after_trimming=len(d)-2*k,
                 leave_one_out_min_mean_delta=float(leave_one_out.min()), leave_one_out_max_mean_delta=float(leave_one_out.max()),
                 leave_one_out_negative_count=int((leave_one_out < -TIE_ATOL).sum()),
                 leave_one_out_positive_count=int((leave_one_out > TIE_ATOL).sum()),
                 best_landmark=rows[best]['scene'], best_delta=float(d[best]),
                 worst_landmark=rows[worst]['scene'], worst_delta=float(d[worst]),
                 mean_without_best_landmark=float(leave_one_out[best]), mean_without_worst_landmark=float(leave_one_out[worst]),
                 gross_improvements=float(-d[d < 0].sum()), gross_regressions=float(d[d > 0].sum()),
                 net_sum_delta=float(d.sum()), bootstrap_probability_all_ties=float(np.mean(np.all(np.abs(boot_delta) <= TIE_ATOL, axis=1))))
        comparisons.append(r)
        for i, source in enumerate(rows):
            influence.append(dict(hypothesis=h, control=control, scene=source['scene'], anchor=source['anchor'],
                                  num_views=int(source['num_views']), control_error=float(b[i]), hypothesis_error=float(a[i]),
                                  delta=float(d[i]), paired_percent_change=float(percent[i]), contribution_to_mean_delta=float(d[i] / len(d)),
                                  leave_this_landmark_out_mean_delta=float(leave_one_out[i])))

# Holm's step-down adjustment across the six sign tests. These test win/loss
# balance conditional on non-ties, not the primary mean-error estimand.
order = np.argsort([r['exact_two_sided_sign_p'] for r in comparisons], kind='stable')
running = 0.0
for rank, index in enumerate(order):
    adjusted = min(1.0, (len(comparisons)-rank) * comparisons[index]['exact_two_sided_sign_p'])
    running = max(running, adjusted)
    comparisons[index]['sign_p_holm_six'] = running

calibration_rows = []
by_scene = {r['scene']: r for r in study}
for h, selection_name, calibration_name in [
    ('H1', 'h1_run_selection.json', 'h1_run_calibration_scores.json'),
    ('H2', 'h2_run_selection.json', 'h2_run_calibration_scores.json'),
    ('H3', 'h3_selection.json', 'h3_calibration.json')]:
    selection = load_json(selection_name)
    values = load_json(calibration_name)
    assert set(values) == calibration and set(selection['calibration_scenes']) == calibration
    assert selection['selection_uses_confirmation'] is False
    method_scores, scalar_scores = [], []
    for scene in sorted(calibration):
        choice = selection[h] if h != 'H3' else selection[h][by_scene[scene]['anchor']]
        method_scores.append(values[scene][choice]); scalar_scores.append(values[scene][selection['global_scalar']])
    method_scores, scalar_scores = np.array(method_scores), np.array(scalar_scores)
    calibration_rows.append(dict(hypothesis=h, n_calibration=18, scalar_mean=float(scalar_scores.mean()),
                                 selected_method_mean=float(method_scores.mean()),
                                 selected_minus_scalar=float((method_scores-scalar_scores).mean()),
                                 percent_change_of_means=float(100*(method_scores.mean()/scalar_scores.mean()-1)),
                                 warning='Descriptive only: settings were selected on these same calibration landmarks.'))

save_csv('paired_effects.csv', comparisons)
save_csv('landmark_influence.csv', influence)
save_csv('condition_summary.csv', condition_rows)
save_csv('calibration_descriptive_only.csv', calibration_rows)
save_json('results.json', {'comparisons': comparisons, 'conditions': condition_rows,
                         'calibration_descriptive_only': calibration_rows})
save_json('provenance.json', {
    'seed': SEED, 'bootstrap_replicates': BOOTSTRAPS, 'n_confirmation': len(rows), 'n_calibration': len(calibration),
    'resampling_unit': 'Landmark, paired across conditions; 36 draws with replacement per replicate.',
    'canonical_order': [r['scene'] for r in rows],
    'tie_tolerance_absolute': TIE_ATOL,
    'confidence_intervals': 'Pointwise percentile bootstrap 95%; not simultaneous and conditional on frozen choices.',
    'multiplicity': 'Holm step-down adjusted exact two-sided sign-test p-values across all six comparisons.',
    'scope': 'Existing confirmation outcomes only. No holdout outcomes, tuning, reruns or calibration pooling.',
    'input_sha256': {name: hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest() for name in INPUTS}})

lines = ['# Paired effects and robustness', '',
         'The analysis uses the same 36 confirmation landmarks for each method. A negative change means lower error and favors the hypothesis. All numbers below are whole-landmark CD_norm results.', '',
         '## Primary paired results', '',
         '| Method | Control | Mean change | 95% bootstrap interval | Change of means | Wins / ties / losses | Sign p (Holm) |',
         '|---|---|---:|---:|---:|---:|---:|']
for r in comparisons:
    lines.append(f"| {r['hypothesis']} | {r['control']} | {r['mean_delta']:+.4f} | [{r['mean_delta_ci95_low']:+.4f}, {r['mean_delta_ci95_high']:+.4f}] | {r['percent_change_of_means']:+.2f}% | {r['wins']} / {r['ties']} / {r['losses']} | {r['sign_p_holm_six']:.4f} |")
lines += ['', 'Every pointwise mean-change interval crosses zero. None of the six win/loss sign tests rejects equal win/loss probabilities after Holm correction. These results do not establish equivalence or prove that effects are absent.', '',
          '## Influence and robust summaries', '',
          '| Method / control | Median paired change | Trimmed mean change | Range after omitting any one landmark |',
          '|---|---:|---:|---:|']
for r in comparisons:
    lines.append(f"| {r['hypothesis']} / {r['control']} | {r['median_paired_delta']:+.4f} | {r['trimmed_mean_delta_exploratory']:+.4f} | [{r['leave_one_out_min_mean_delta']:+.4f}, {r['leave_one_out_max_mean_delta']:+.4f}] |")
lines += ['', 'The trimmed sensitivity removes the three smallest and three largest paired differences (8.33% from each tail), leaving 30 landmarks. It is exploratory and does not replace the prespecified untrimmed mean.', '']
for r in comparisons:
    lines.append(f"- **{r['hypothesis']} versus {r['control']}:** largest improvement: {r['best_landmark']} ({r['best_delta']:+.4f}); largest regression: {r['worst_landmark']} ({r['worst_delta']:+.4f}). Omitting the largest improvement gives mean change {r['mean_without_best_landmark']:+.4f}; omitting the largest regression gives {r['mean_without_worst_landmark']:+.4f}.")
lines += ['', '## Distinct summaries answer distinct questions', '',
          '| Method / control | Change of means (%) | Mean landmark change (%) | Median landmark change (%) |',
          '|---|---:|---:|---:|']
for r in comparisons:
    lines.append(f"| {r['hypothesis']} / {r['control']} | {r['percent_change_of_means']:+.2f} | {r['mean_landmark_percent_change']:+.2f} | {r['median_landmark_percent_change']:+.2f} |")
lines += ['', 'The prespecified absolute mean error gives each landmark equal weight in error units. The percentage change of means equals `100 × (mean(method) / mean(control) − 1)`, and implicitly weights per-landmark relative changes by control error. Mean landmark percentage change instead averages `100 × (method_i / control_i − 1)` equally. Neither may be substituted for the other. A positive mean paired change can coexist with more individual wins than losses because regressions can be larger than improvements.', '',
          '## Calibration remains separate', '',
          '| Method | Selected calibration mean | Scalar calibration mean | Selected minus scalar |',
          '|---|---:|---:|---:|']
for r in calibration_rows:
    lines.append(f"| {r['hypothesis']} | {r['selected_method_mean']:.4f} | {r['scalar_mean']:.4f} | {r['selected_minus_scalar']:+.4f} |")
lines += ['', 'The 18 calibration landmarks selected the settings. Their scores above are descriptive and optimistic for selection purposes; they are not an additional independent test, are not pooled with the 36 confirmation landmarks, and are not used for confirmation confidence intervals.', '',
          '## Method and limits', '',
          '- Input: `analysis/sources/paired_hypotheses.csv`, checked against the frozen study split and existing verified condition means. Scene keys are unique and all 36 scenes match the confirmation split.',
          '- Bootstrap: 100,000 draws of 36 landmarks with replacement, seed 20261126, using identical sampled indices for all paired conditions. Intervals are the 2.5th and 97.5th percentiles and are pointwise, not simultaneous.',
          '- Exact two-sided sign tests condition on non-ties and test equal probabilities of improvement/regression. Ties use absolute tolerance 1e-12. Holm correction covers six comparisons. This is a complementary direction-of-effect test, not a test of the mean-error estimand.',
          '- The 54-landmark study subset was deliberately selected for coverage. Resampling these 36 confirmation landmarks describes sensitivity under exchangeability of this observed set; it does not justify population-wide confidence coverage for all 396 landmarks.',
          '- H3 changes only three confirmation predictions relative to scalar. Its 33 ties reflect the learned policy, not 33 independent demonstrations of H3-specific benefit. Only three non-ties inform its sign test; leave-one-out and bootstrap estimates are correspondingly discrete and fragile.',
          '- No per-view replication, holdout outcomes, additional model runs, or post hoc retuning was used. Deleting difficult landmarks is not proposed as a new official score.', '',
          'Reproduce by running `analyze_effects.py` with Python and NumPy. The default source is the existing `analysis/sources` directory relative to the script, and default output is the script directory. Explicit `--source PATH --output PATH` arguments support relocation. `provenance.json` records source SHA-256 hashes and the resampling protocol; CSV files expose all estimates and every landmark contribution.']
(ROOT/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
assert all(r['mean_delta_ci95_low'] < 0 < r['mean_delta_ci95_high'] for r in comparisons)
assert all(r['sign_p_holm_six'] >= .05 for r in comparisons)
print(json.dumps({'output': str(ROOT), 'comparisons': comparisons}, ensure_ascii=True))
