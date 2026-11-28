"""Exploratory confirmation-only results grouped by the executed anchor label.

Run: python analyze_by_anchor.py [--plots]
Only --plots needs numpy and matplotlib. No model inference is rerun.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / 'evidence'
if not EVIDENCE.exists():
    EVIDENCE = ROOT.parent / 'combined' / 'evidence'
METHODS = ('H1', 'H2', 'H3')
REFERENCES = ('scalar', 'baseline')


def main(plots=False):
    source = EVIDENCE / 'paired_hypotheses.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    plan = json.loads((EVIDENCE / 'h3_study.json').read_text(encoding='utf-8'))['plan']
    confirmation = {r['scene']: r for r in plan['rows'] if r['split'] == 'confirmation'}
    assert len(rows) == len(confirmation) == 36
    assert {r['scene'] for r in rows} == set(confirmation)
    groups = defaultdict(list)
    for r in rows:
        assert r['anchor'] == confirmation[r['scene']]['anchor']
        for k in ('baseline', 'scalar', *METHODS):
            r[k] = float(r[k])
            assert math.isfinite(r[k]) and r[k] > 0
        groups[r['anchor']].append(r)
    anchors = sorted(groups, key=lambda a: (-len(groups[a]), a))
    recorded = json.loads((EVIDENCE / 'comparison_summary.json').read_text(encoding='utf-8'))
    assert recorded['status'] == 'complete_and_comparable'
    detail = []
    for anchor in anchors:
        members = groups[anchor]
        for method in METHODS:
            for reference in REFERENCES:
                m = [r[method] for r in members]
                b = [r[reference] for r in members]
                d = [x-y for x, y in zip(m, b)]
                detail.append(dict(
                    anchor=anchor, n=len(members), method=method, reference=reference,
                    method_mean=statistics.mean(m), reference_mean=statistics.mean(b),
                    mean_difference=statistics.mean(d), median_difference=statistics.median(d),
                    percent_change_of_means=100*(statistics.mean(m)/statistics.mean(b)-1),
                    wins=sum(x < 0 for x in d), ties=sum(x == 0 for x in d),
                    losses=sum(x > 0 for x in d),
                ))
    # Recover the whole-cohort differences by weighting each anchor by its count.
    for method in METHODS:
        for reference in REFERENCES:
            subset = [r for r in detail if r['method'] == method and r['reference'] == reference]
            weighted = sum(r['n']*r['mean_difference'] for r in subset)/36
            actual = statistics.mean(r[method]-r[reference] for r in rows)
            assert math.isclose(weighted, actual, abs_tol=1e-12)
            assert sum(r['n'] for r in subset) == 36
            if reference == 'scalar':
                saved = recorded['hypotheses'][method]
                assert math.isclose(weighted, saved['mean_difference'], abs_tol=1e-12)
                for key in ['wins', 'ties', 'losses']:
                    assert sum(r[key] for r in subset) == saved[key]
    diagnostics = {}
    for method in METHODS:
        diagnostics[method] = {}
        for reference in REFERENCES:
            delta = [r[method]-r[reference] for r in rows]
            leave_one_out = [(sum(delta)-x)/(len(rows)-1) for x in delta]
            ordered = sorted(rows, key=lambda r: r[method]-r[reference])
            def case(r):
                return dict(scene=r['scene'], anchor=r['anchor'],
                            reference_error=r[reference], method_error=r[method],
                            difference=r[method]-r[reference])
            diagnostics[method][reference] = dict(
                mean_difference=statistics.mean(delta), median_difference=statistics.median(delta),
                wins=sum(x < 0 for x in delta), ties=sum(x == 0 for x in delta),
                losses=sum(x > 0 for x in delta),
                sum_improvements=-sum(x for x in delta if x < 0),
                sum_regressions=sum(x for x in delta if x > 0),
                leave_one_out_mean_min=min(leave_one_out),
                leave_one_out_mean_max=max(leave_one_out),
                largest_improvements=[case(r) for r in ordered if r[method] < r[reference]][:3],
                largest_regressions=[case(r) for r in reversed(ordered) if r[method] > r[reference]][:3],
            )
    (ROOT / 'hypothesis_diagnostics.json').write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with (ROOT / 'anchor_comparisons.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader(); w.writerows(detail)
    scene_rows = []
    for r in sorted(rows, key=lambda r: (anchors.index(r['anchor']), r['scene'])):
        out = {k: r[k] for k in ('scene', 'anchor', 'num_views', 'baseline', 'scalar', *METHODS)}
        for method in METHODS:
            for reference in REFERENCES:
                out[f'{method}_minus_{reference}'] = r[method]-r[reference]
                out[f'{method}_percent_vs_{reference}'] = 100*(r[method]/r[reference]-1)
        scene_rows.append(out)
    with (ROOT / 'landmark_comparisons.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(scene_rows[0]))
        w.writeheader(); w.writerows(scene_rows)
    output = dict(
        analysis='Exploratory post-hoc anchor subgroup analysis; frozen confirmation results only',
        source=str(source.relative_to(ROOT.parent)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        confirmation_landmarks=len(rows), anchor_groups=len(groups),
        repeated_groups=sum(len(v) >= 2 for v in groups.values()),
        singleton_groups=sum(len(v) == 1 for v in groups.values()),
        metric='Whole-landmark normalized Chamfer error, not anchor-region geometry error',
        percent_definition='100 * (group mean method error / group mean reference error - 1)',
        weighting='Equal landmark weight within each anchor group',
        grouping='Exact executed anchor labels; pillar and column remain distinct',
        inference='Descriptive only; no hypothesis significance tests or new selector fitting',
        comparisons=detail,
    )
    (ROOT / 'anchor_analysis.json').write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = [
        'AdaptiveAttend: exploratory anchor subgroup analysis',
        '36 confirmation landmarks; 23 executed anchor labels.',
        'Six labels have at least two landmarks; seventeen labels have one.',
        'Negative percentages indicate lower mean whole-reconstruction error.',
        'Percentages are ratios of group means, not means of landmark percentages.',
        'W/T/L counts improved / tied / worsened landmarks against the named reference.',
        'All group definitions and model selections are unchanged; no confirmation tuning.',
        'Small post-hoc subgroups do not establish a reliable anchor-specific policy.', '',
    ]
    for anchor in anchors:
        lines.append(f'{anchor.upper()} (n={len(groups[anchor])})')
        for method in METHODS:
            for ref in REFERENCES:
                r = next(r for r in detail if r['anchor'] == anchor and r['method'] == method and r['reference'] == ref)
                lines.append(f"  {method} vs {ref}: {r['percent_change_of_means']:+.2f}% mean error; "
                             f"delta {r['mean_difference']:+.5f}; W/T/L {r['wins']}/{r['ties']}/{r['losses']}")
        lines.append('  Landmarks: '+ '; '.join(r['scene'] for r in groups[anchor]))
        lines.append('')
    (ROOT / 'anchor_results.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    tables(detail, anchors, groups, diagnostics)
    if plots:
        plot(detail, anchors, groups)
    print(json.dumps({k: output[k] for k in ['confirmation_landmarks', 'anchor_groups', 'repeated_groups', 'singleton_groups']}, indent=2))


def tables(detail, anchors, groups, diagnostics):
    lookup = {(r['anchor'], r['method'], r['reference']): r for r in detail}
    lines = [r'\begin{tabular}{lr rrrrrr}', r'\toprule',
             r'& & \multicolumn{2}{c}{H1} & \multicolumn{2}{c}{H2} & \multicolumn{2}{c}{H3} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'Anchor & $n$ & Scalar & No inter. & Scalar & No inter. & Scalar & No inter. \\',
             r'\midrule']
    for i, anchor in enumerate(anchors):
        if i and len(groups[anchor]) == 1 and len(groups[anchors[i-1]]) > 1:
            lines.append(r'\midrule')
        vals = []
        for method in METHODS:
            for ref in REFERENCES:
                value = lookup[(anchor, method, ref)]['percent_change_of_means']
                vals.append('0.00' if value == 0 else f'{value:+.2f}')
        lines.append(' & '.join([anchor.capitalize(), str(len(groups[anchor])), *vals])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT / 'all_anchors_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    lines = [r'\begin{tabular}{lr cccccc}', r'\toprule',
             r'& & \multicolumn{2}{c}{H1} & \multicolumn{2}{c}{H2} & \multicolumn{2}{c}{H3} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'Anchor & $n$ & Scalar & No inter. & Scalar & No inter. & Scalar & No inter. \\',
             r'\midrule']
    for anchor in anchors:
        if len(groups[anchor]) < 2:
            continue
        vals = []
        for method in METHODS:
            for ref in REFERENCES:
                r = lookup[(anchor, method, ref)]
                vals.append(f"{r['wins']}/{r['ties']}/{r['losses']}")
        lines.append(' & '.join([anchor.capitalize(), str(len(groups[anchor])), *vals])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT / 'repeated_anchor_wins_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Method & Vs. scalar & Vs. no intervention \\', r'\midrule']
    for method in METHODS:
        cells = [method]
        for ref in REFERENCES:
            r = diagnostics[method][ref]
            cells.append(f"$[{r['leave_one_out_mean_min']:+.5f}, {r['leave_one_out_mean_max']:+.5f}]$")
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT / 'sensitivity_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def plot(detail, anchors, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    cmap = LinearSegmentedColormap.from_list('error_change', ['#197b82', '#f8f8f6', '#b44d33'])
    lookup = {(r['anchor'], r['method'], r['reference']): r for r in detail}
    for name, selected, title, height in [
        ('repeated_anchor_groups', [a for a in anchors if len(groups[a]) >= 2],
         'Anchor groups with multiple confirmation landmarks', 4.2),
        ('single_landmark_anchors', [a for a in anchors if len(groups[a]) == 1],
         'Single-landmark anchor groups: individual cases only', 7.3),
    ]:
        values = [np.array([[lookup[(a, h, ref)]['percent_change_of_means'] for h in METHODS]
                           for a in selected]) for ref in REFERENCES]
        limit = max(5, np.ceil(max(np.abs(x).max() for x in values)/5)*5)
        fig, axes = plt.subplots(1, 2, figsize=(8.3, height), sharey=True)
        fig.subplots_adjust(left=.24, right=.98, bottom=.23 if height < 5 else .15,
                            top=.79 if height < 5 else .88, wspace=.13)
        for ax, val, subtitle in zip(axes, values, ['Versus calibrated scalar', 'Versus no intervention']):
            im = ax.imshow(val, aspect='auto', cmap=cmap, norm=TwoSlopeNorm(0, -limit, limit))
            ax.set_title(subtitle, fontsize=10, pad=8)
            ax.set_xticks(range(3), METHODS)
            ax.set_yticks(range(len(selected)), [f'{a.capitalize()}  (n={len(groups[a])})' for a in selected])
            ax.tick_params(axis='both', length=0, pad=7)
            for spine in ax.spines.values(): spine.set_visible(False)
            for i in range(len(selected)):
                for j in range(3):
                    v = val[i, j]
                    label = '0.0%' if abs(v) < .05 else f'{v:+.1f}%'
                    ax.text(j, i, label, ha='center', va='center', fontsize=9,
                            color='white' if abs(v) > .64*limit else '#172a35')
        fig.suptitle(title, x=.53, y=.98, fontsize=12, fontweight='bold')
        fig.text(.53, .91 if height < 5 else .94,
                 'Change in group mean normalized Chamfer error, lower is better', ha='center', fontsize=9)
        cax = fig.add_axes([.35, .105 if height < 5 else .07, .52, .025 if height < 5 else .015])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-limit, 0, limit])
        cb.ax.set_xticklabels([f'{-limit:.0f}%', '0%', f'+{limit:.0f}%'])
        cb.outline.set_visible(False)
        foot = ('Exploratory results. 19 landmarks in six groups. The other 17 groups each contain one landmark.'
                if height < 5 else 'Exploratory results. each row is one landmark, not evidence of an anchor-wide effect.')
        fig.text(.53, .015, foot, ha='center', fontsize=8, color='#46545e')
        for ext in ['pdf', 'png']:
            fig.savefig(ROOT / (name+'.'+ext), dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots', action='store_true')
    main(parser.parse_args().plots)
