"""Verify the saved study and reproduce report tables/plots without a GPU.

Usage: python reproduce_results.py [--plots]
Only --plots needs matplotlib and numpy. Other checks use the standard library.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parent
E = ROOT/'evidence'


def load(name):
    return json.loads((E/name).read_text(encoding='utf-8'))


def rows(name):
    with (E/name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def compare(values, reference):
    delta = [a-b for a, b in zip(values, reference)]
    return dict(mean=stats.mean(delta), median=stats.median(delta),
                wins=sum(x < 0 for x in delta), ties=sum(x == 0 for x in delta),
                losses=sum(x > 0 for x in delta),
                percent_change_of_means=100*(stats.mean(values)/stats.mean(reference)-1))


def main(plots=False):
    paired = rows('paired_hypotheses.csv')
    assert len(paired) == 36 and len({r['scene'] for r in paired}) == 36
    values = {k: [float(r[k]) for r in paired] for k in ['baseline','scalar','H1','H2','H3']}
    assert all(math.isfinite(v) for vs in values.values() for v in vs)
    combined = load('comparison_summary.json')
    assert combined['status'] == 'complete_and_comparable'
    h3_study = load('h3_study.json')
    plan = h3_study['plan']
    cal = [r for r in plan['rows'] if r['split'] == 'calibration']
    confirm = [r for r in plan['rows'] if r['split'] == 'confirmation']
    assert len(cal) == 18 and len(confirm) == 36
    assert set(r['scene'] for r in confirm) == {r['scene'] for r in paired}
    assert not {r['scene'] for r in plan['rows']} & set(plan['heldout_ids'])
    assert combined['source_sha256'] == h3_study['source_sha256']
    for row in paired:
        for h in ['H1','H2','H3']:
            close(float(row[h])-float(row['scalar']), float(row[h+'_minus_scalar']))

    output = {'n_calibration': 18, 'n_confirmation': 36, 'hypotheses': {}, 'conditions': {},
              'numeric_conditions_complete': {}, 'calibration_means': {}}
    env = load('h3_environment.json')
    for key, vs in values.items():
        output['conditions'][key] = dict(mean=stats.mean(vs), median=stats.median(vs))
    for h in ['H1','H2','H3']:
        lower = h.lower()
        selection = load(f'{lower}_selection.json' if h == 'H3' else f'{lower}_run_selection.json')
        scores = load(f'{lower}_calibration.json' if h == 'H3' else f'{lower}_run_calibration_scores.json')
        study = h3_study if h == 'H3' else load(f'{lower}_run_study.json')
        summary = load(f'{lower}_summary.json' if h == 'H3' else f'{lower}_run_summary.json')
        paired_one = rows(f'{lower}_paired_results.csv' if h == 'H3' else f'{lower}_report_paired_results.csv')
        assert selection['plan_sha256'] == fingerprint(study['plan'])
        assert selection['calibration_scores_sha256'] == fingerprint(scores)
        assert selection['selection_uses_confirmation'] is False
        assert set(scores) == set(selection['calibration_scenes']) == {r['scene'] for r in cal}
        assert summary['status'] == 'complete' and summary['incomplete_landmarks'] == 0
        assert summary['confirmation_complete'] == 36
        assert study['source_sha256'] == combined['source_sha256']
        assert {r['scene'] for r in paired_one} == {r['scene'] for r in paired}
        if h != 'H3':
            assert load(f'{lower}_run_execution_environment.json') == env
        own = {r['scene']: r for r in paired_one}
        for r in paired:
            q = own[r['scene']]
            assert int(q['num_views']) == int(r['num_views'])
            for col in ['baseline','scalar']:
                assert float(q[col]) == float(r[col])
            assert float(q['structured']) == float(r[h])
        means = {candidate: stats.mean(s[candidate] for s in scores.values())
                 for candidate in next(iter(scores.values()))}
        assert min(['scalar_eta100','scalar_eta1000'], key=lambda c: means[c]) == selection['global_scalar']
        if h in ('H1','H2'):
            prefix = 'layer_' if h == 'H1' else 'view_'
            assert min([c for c in means if c.startswith(prefix)], key=lambda c: means[c]) == selection[h]
        else:
            for anchor, selected in selection['H3'].items():
                members = [r['scene'] for r in cal if r['anchor'] == anchor]
                expected = (min(['scalar_eta100','scalar_eta1000'],
                            key=lambda c: stats.mean(scores[s][c] for s in members))
                            if len(members) >= 2 else selection['global_scalar'])
                assert expected == selected
        output['calibration_means'][h] = means
        output['numeric_conditions_complete'][h] = summary['numeric_conditions_complete']
        vs_scalar = compare(values[h], values['scalar'])
        vs_base = compare(values[h], values['baseline'])
        for key in ['mean','median']:
            close(vs_scalar[key], combined['hypotheses'][h][key+'_difference'])
        for key in ['wins','ties','losses']:
            assert vs_scalar[key] == combined['hypotheses'][h][key]
        output['hypotheses'][h] = {'vs_scalar': vs_scalar, 'vs_baseline': vs_base, 'selection': selection[h]}

    assert output['numeric_conditions_complete'] == {'H1': 198, 'H2': 198, 'H3': 129}
    assert not rows('h3_failures.csv')
    assert len(rows('h3_strength_diagnostics.csv')) == 129
    active = [r for r in paired if float(r['H3_minus_scalar']) != 0]
    own = rows('h3_paired_results.csv')
    assert sum(r['same_prediction_as_scalar'] == 'True' for r in own) == 33
    assert len(active) == 3 and sum(r['H3_fallback'] == 'True' for r in paired) == 19
    output['h3_changed_landmarks'] = active
    output['h3_supported_confirmation'] = 17
    output['h3_fallback_confirmation'] = 19
    output['confirmation_views'] = sum(int(r['num_views']) for r in paired)
    output['all_views'] = sum(load('h3_summary.json')['view_counts'].values())
    assert output['confirmation_views'] == 1776 and output['all_views'] == 2622
    spec = importlib.util.spec_from_file_location('subset_reproduction', E/'reproduce_subset_table.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output['subset_audit'] = module.comparison()
    (E/'verified_statistics.json').write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    # Record hashes of packaged evidence; fetched text was normalized to UTF-8/LF.
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(E.iterdir()) if p.is_file() and p.name != 'evidence_sha256.json'}
    (E/'evidence_sha256.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    gen = ROOT/'generated'
    gen.mkdir(exist_ok=True)
    labels = {'baseline':'No intervention','scalar':r'Global scalar ($\eta=1000$)',
              'H1':r'H1 per-layer ($\alpha=0.02$)', 'H2':r'H2 per-view (base $\eta=100$)',
              'H3':'H3 per-anchor'}
    text = '\n'.join(f"{labels[k]} & {s['mean']:.5f} & {s['median']:.5f} \\\\"
                     for k,s in output['conditions'].items())
    (gen/'main_rows.tex').write_text(text+'\n', encoding='utf-8')
    text = '\n'.join(f"{h} & {v['vs_scalar']['mean']:+.5f} & {v['vs_scalar']['median']:+.5f} & "
                     f"{v['vs_scalar']['wins']}/{v['vs_scalar']['ties']}/{v['vs_scalar']['losses']} & "
                     f"{v['vs_baseline']['mean']:+.5f} \\\\"
                     for h,v in output['hypotheses'].items())
    (gen/'paired_rows.tex').write_text(text+'\n', encoding='utf-8')
    selection = load('h3_selection.json')
    scores = load('h3_calibration.json')
    h3_rows = []
    for anchor in plan['supported_anchor_types']:
        members = [r['scene'] for r in cal if r['anchor'] == anchor]
        count = sum(r['anchor'] == anchor for r in confirm)
        a, b = [stats.mean(scores[s][c] for s in members) for c in ['scalar_eta100','scalar_eta1000']]
        eta = selection['H3'][anchor].replace('scalar_eta','')
        h3_rows.append(f"{anchor.capitalize()} & {len(members)} & {count} & {a:.5f} & {b:.5f} & {eta} \\\\")
    (gen/'h3_calibration_rows.tex').write_text('\n'.join(h3_rows)+'\n', encoding='utf-8')
    for name, cols, header in [
        ('main','lrr',r'Condition & Mean normalized CD & Median \\'),
        ('paired','lrrrr',r'Method & Mean vs. scalar & Median vs. scalar & W/T/L vs. scalar & Mean vs. no intervention \\'),
        ('h3_calibration','lrrrrr',r'Anchor & Calibration $n$ & Confirmation $n$ & Mean at $\eta=100$ & Mean at $\eta=1000$ & Selected $\eta$ \\')]:
        body = (gen/(name+'_rows.tex')).read_text(encoding='utf-8')
        table = ('\\begin{tabular}{'+cols+'}\n\\toprule\n'+header+'\n\\midrule\n'+body+
                 '\\bottomrule\n\\end{tabular}\n')
        (gen/(name+'_table.tex')).write_text(table, encoding='utf-8')
    if plots:
        plot(paired, output)
    print(json.dumps({'verified': True, 'landmarks': 54, 'confirmation': 36,
                      'conditions': output['conditions'], 'h3_changes': len(active),
                      'total_condition_jobs': sum(output['numeric_conditions_complete'].values())}, indent=2))


def plot(paired, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    ordered = sorted(paired, key=lambda r: float(r['H1_minus_scalar']))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharex=True, sharey=True)
    for ax, h, title in zip(axes, ['H1','H2','H3'], ['H1 per-layer','H2 per-view','H3 per-anchor']):
        y = np.array([float(r[h+'_minus_scalar']) for r in ordered])
        x = np.arange(1,37)
        ax.axhline(0, color='#555555', lw=.8)
        ax.vlines(x, 0, y, colors=np.where(y < 0,'#197b82','#b44d33'), lw=1)
        ax.scatter(x, y, c=np.where(y < 0,'#197b82',np.where(y > 0,'#b44d33','#777777')), s=13, zorder=3)
        ax.axhline(y.mean(), color='#23395b', ls='--', lw=.9)
        v = output['hypotheses'][h]['vs_scalar']
        ax.set_title(title)
        ax.text(.04,.96,f"Mean {v['mean']:+.3f}\nW/T/L {v['wins']}/{v['ties']}/{v['losses']}",
                transform=ax.transAxes, va='top', fontsize=7.5)
        ax.set_xlabel('Landmark index (same order)')
        ax.set_xticks([1,12,24,36])
        ax.grid(axis='y', color='#dddddd', lw=.5)
    axes[0].set_ylabel('Normalized CD, method minus scalar')
    axes[0].set_ylim(-13,10)
    fig.tight_layout(pad=.8)
    for ext in ['pdf','png']:
        fig.savefig(ROOT/'figures'/('paired_effects.'+ext), dpi=220)
    plt.close(fig)

    active = output['h3_changed_landmarks']
    labels = ['Mansudae\nGrand Monument','Marine Corps\nWar Memorial','Palacio Legislativo,\nUruguay']
    fig, ax = plt.subplots(figsize=(7.1, 2.65))
    x = np.arange(3)
    for offset, key, label, color in [(-.25,'baseline','No intervention','#888888'),
                                    (0,'scalar','Scalar, eta = 1000','#d59c47'),
                                    (.25,'H3','H3, eta = 100','#197b82')]:
        ax.bar(x+offset, [float(r[key]) for r in active], width=.24, label=label, color=color)
    ax.set_xticks(x,labels)
    ax.set_ylabel('Normalized Chamfer error')
    ax.legend(frameon=False, loc='upper left', ncols=1)
    ax.set_ylim(0,42)
    ax.grid(axis='y', color='#dddddd', lw=.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=.8)
    for ext in ['pdf','png']:
        fig.savefig(ROOT/'figures'/('h3_changed_landmarks.'+ext), dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots', action='store_true')
    main(parser.parse_args().plots)
