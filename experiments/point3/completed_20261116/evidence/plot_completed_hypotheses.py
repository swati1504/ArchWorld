"""Verify matched H1/H2 controls and render the milestone comparison figure.

Run from any folder with Python and Matplotlib; no model inference is performed.
The panels sort landmarks independently to show each distribution of paired effects.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent


def read_hypothesis(name):
    folder = HERE / f'{name.lower()}_completed_20261116'
    provenance = json.loads((folder / 'drive_provenance.json').read_text(encoding='utf-8'))
    for item in provenance['files']:
        assert hashlib.sha256((folder / item['file']).read_bytes()).hexdigest() == item['sha256']
    summary = json.loads((folder / 'run_summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'complete' and summary['confirmation_complete'] == 36
    assert summary['incomplete_landmarks'] == 0 and summary['numeric_conditions_complete'] == 198
    with (folder / 'report_paired_results.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ['baseline', 'scalar', 'structured', 'structured_minus_scalar', 'structured_minus_baseline']:
            row[key] = float(row[key])
            assert math.isfinite(row[key])
        assert math.isclose(row['structured_minus_scalar'], row['structured'] - row['scalar'], abs_tol=1e-12)
    assert len({r['scene'] for r in rows}) == 36
    stats = summary['hypotheses'][name]
    differences = [r['structured_minus_scalar'] for r in rows]
    assert math.isclose(statistics.mean(differences), stats['mean_difference'], abs_tol=1e-12)
    assert sum(d < 0 for d in differences) == stats['wins']
    assert sum(d > 0 for d in differences) == stats['losses']
    return rows, stats


def main():
    data = {h: read_hypothesis(h) for h in ['H1', 'H2']}
    indexed = {h: {r['scene']: r for r in rows} for h, (rows, _) in data.items()}
    assert set(indexed['H1']) == set(indexed['H2'])
    for scene, first in indexed['H1'].items():
        second = indexed['H2'][scene]
        for key in ['baseline', 'scalar', 'num_views', 'anchor', 'scalar_candidate']:
            assert first[key] == second[key], (scene, key, first[key], second[key])
    result = {'confirmation_landmarks': 36, 'controls_match_exactly': True,
              'baseline': {}, 'scalar': {}, 'H1': {}, 'H2': {}}
    for name, column, rows in [('baseline', 'baseline', data['H1'][0]),
                               ('scalar', 'scalar', data['H1'][0]),
                               ('H1', 'structured', data['H1'][0]),
                               ('H2', 'structured', data['H2'][0])]:
        values = [r[column] for r in rows]
        result[name] = {'mean': statistics.mean(values), 'median': statistics.median(values)}
    (HERE / 'completed_hypotheses_comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.12), sharey=True, layout='constrained')
    for ax, h in zip(axes, ['H1', 'H2']):
        rows, stats = data[h]
        delta = sorted(r['structured_minus_scalar'] for r in rows)
        rank = list(range(1, 37))
        colors = ['#237e96' if d < 0 else '#b95324' for d in delta]
        ax.vlines(rank, 0, delta, colors=colors, linewidth=0.85, alpha=0.85)
        ax.scatter(rank, delta, c=colors, s=9, edgecolors='white', linewidths=0.2, zorder=3)
        ax.axhline(0, color='#343434', linewidth=0.6)
        ax.set_xlim(0, 37)
        ax.set_ylim(-13, 9)
        ax.set_xticks([1, 18, 36])
        ax.set_yticks([-12, -6, 0, 6])
        label = 'layer' if h == 'H1' else 'view'
        ax.set_title(f"{h}: {label}\n{stats['wins']} improved / {stats['losses']} worsened", fontsize=7, pad=4)
        ax.grid(axis='y', alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(r'$\Delta\,\mathrm{CD}_{\mathrm{norm}}$ (structured $-$ scalar)')
    fig.supxlabel('Landmark rank (sorted separately in each panel)', fontsize=7)
    fig.savefig(SOURCE / 'figures/point3_h1_h2_confirmation.pdf')
    fig.savefig(SOURCE / 'figures/point3_h1_h2_confirmation.png', dpi=240)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
