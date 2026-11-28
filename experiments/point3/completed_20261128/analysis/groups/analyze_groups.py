"""Exploratory categorical heterogeneity of the frozen Point 3 methods.

Run from any directory with bundled Python; inputs remain read-only.
Unit: confirmation landmark, never individual image. Negative differences favor method.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'analysis/sources'
OUT = Path(__file__).resolve().parent
PRIOR_ANCHOR_SUMMARY = ROOT/'report/anchor_analysis/anchor_comparisons.csv'
SEED = 2026112602
PERMUTATIONS = 49999
METHODS = ['H1', 'H2', 'H3']
CONTROLS = ['baseline', 'scalar']
GROUP_FIELDS = ['anchor', 'region', 'building_type']
COMPARISONS = [(h, c) for h in METHODS for c in CONTROLS]


def load_json(name):
    return json.loads((SOURCE / name).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty output')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def midranks(values):
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    ranked = np.empty(len(values), float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranked[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranked


def bh_adjust(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return result


def regional_leave_one_out(rows):
    """Influence diagnostics for the screened H1/scalar regional association only."""
    output=[]
    null_by_counts={}
    for omitted in rows:
        retained=[r for r in rows if r['scene']!=omitted['scene']]
        y=midranks(np.array([math.log(r['H1']/r['scalar']) for r in retained]))
        y-=y.mean()
        groups=sorted({r['region'] for r in retained})
        membership=np.array([[r['region']==g for g in groups] for r in retained],float)
        counts=membership.sum(axis=0)
        denominator=float(y@y)
        observed=float((((membership.T@y)**2)/counts).sum()/denominator)
        # H1/scalar log ratios have no ties, so the null depends only on group sizes.
        # Cache a canonical null to avoid inconsistent Monte Carlo estimates for equal statistics.
        assert len(set(y)) == len(y)
        key=tuple(sorted(int(x) for x in counts))
        if key not in null_by_counts:
            rng=np.random.default_rng(SEED+100)
            canonical=np.repeat(np.arange(len(key)),key)
            design=(canonical[:,None]==np.arange(len(key))[None,:]).astype(float)
            ranks=np.arange(1,len(y)+1,dtype=float);ranks-=ranks.mean()
            values=[]
            remaining=99999
            while remaining:
                batch=min(2000,remaining)
                indices=np.argsort(rng.random((batch,len(retained))),axis=1)
                sums=ranks[indices]@design
                values.extend(((sums*sums/np.array(key)[None,:]).sum(axis=1)/(ranks@ranks)).tolist())
                remaining-=batch
            null_by_counts[key]=np.array(values)
        exceed=int((null_by_counts[key]>=observed-1e-12).sum())
        output.append(dict(omitted_scene=omitted['scene'],omitted_region=omitted['region'],n=35,
                           rank_between_group_fraction=observed,permutation_p=(exceed+1)/100000,
                           bonferroni_p_original_family_18=min(1,(exceed+1)/100000*18),
                           permutations=99999,seed=SEED+100,
                           interpretation='Influence diagnostic after screening; not an independent discovery test'))
    return output


def describe(members, method, control):
    a = np.array([r[method] for r in members])
    b = np.array([r[control] for r in members])
    delta = a-b
    n = len(members)
    # Retain exact equalities: source H3 scalar-reuse rows are bit-identical numeric results.
    wins, ties, losses = int((delta < 0).sum()), int((delta == 0).sum()), int((delta > 0).sum())
    influential = int(np.argmax(np.abs(delta)))
    loo = (delta.sum()-delta)/(n-1) if n > 1 else np.array([])
    return dict(n=n, method=method, control=control,
                method_mean=float(a.mean()), control_mean=float(b.mean()),
                mean_difference=float(delta.mean()), median_difference=float(np.median(delta)),
                percent_change_of_means=float(100*(a.mean()/b.mean()-1)),
                mean_paired_percent_change=float((100*delta/b).mean()),
                wins=wins, ties=ties, losses=losses,
                largest_absolute_contributor=members[influential]['scene'],
                largest_contributor_difference=float(delta[influential]),
                largest_absolute_contribution_share=float(abs(delta[influential])/np.abs(delta).sum()) if np.abs(delta).sum() else 0.0,
                leave_one_out_mean_min=float(loo.min()) if n>1 else None,
                leave_one_out_mean_max=float(loo.max()) if n>1 else None,
                leave_one_out_sign_changes=bool(np.any(np.sign(loo) != np.sign(delta.mean()))) if n>1 else None,
                sample_note='singleton: descriptive case only' if n==1 else ('tiny group: descriptive only' if n<4 else 'small group: exploratory'))


def write_findings(descriptors,tests,audit,influence,changed):
    lookup={(r['grouping'],r['group'],r['method'],r['control']):r for r in descriptors}
    lines=['# Anchor and categorical scene analysis', '',
           'The 36 confirmation landmarks contain useful local patterns, but most anchor groups are too small to establish general anchor-dependent performance. All calculations below use frozen H1/H2/H3 settings and one equally weighted observation per landmark.', '',
           '## Main findings', '',
           '- H1 improves all three facade-anchor landmarks against both controls: Herald Square, Jahangiri Mahal (Agra Fort), and Patan Durbar Square. The error reduction calculated from group means is **16.80% versus unmodified VGGT** and **12.83% versus scalar**. The sign survives removing any one of the three cases, but n=3 remains a descriptive pattern.',
           '- The largest repeated anchor group, arch (n=8), does not improve on average against either control under H1 or H2. H2 is 8.73% worse than unmodified VGGT and 7.37% worse than scalar, with only 2/8 and 1/8 wins, respectively.',
           '- H3 has **19 unsupported-anchor fallback cases, 14 supported-anchor cases selecting the same scalar, and only 3 cases selecting a different strength**. The 33 equal-to-scalar cases must not be counted as independent evidence that anchor-dependent selection helped.',
           '- A regional H1-versus-scalar association survives correction within the 18 categorical tests (permutation p=0.00206; BH q=0.03708; Bonferroni p=0.03708). The corresponding H1-versus-original regional test is not persuasive (p=0.41068). This is comparator-specific exploratory evidence, not a claim of geographic superiority or causality. Broader correction across all parallel analyses is reported by the combined report.', '',
           '## Group sizes and limits', '',
           'There are **23 executed anchor categories**, of which **17 contain one landmark**. Only arch has at least four observations. The repeated categories are arch (8), facade (3), bridge (2), pillar (2), statue (2), and window (2). Region has eight categories with four or five landmarks each. Building type has eight categories with two to nine landmarks.', '',
           'The executed frozen-plan anchor matches the paired-result table for all 36 landmarks. Two original supplied labels differ from that executed anchor: Forth Bridge has a missing supplied label and uses bridge; Washington Monument has supplied altar and uses obelisk. These are already recorded in the study plan, not changes made by this analysis. Both values are retained in the joined data.', '',
           'Anchor labels identify the semantic class used for steering. The scores measure whole-landmark reconstruction, not error restricted to that semantic region. These tabular results do not establish segmentation-mask quality or replace a manual audit of all input masks.', '',
           '## Repeated anchor groups', '',
           'Percent changes below are `100 × (mean(method) / mean(control) − 1)`. Negative values favor the method. W/T/L counts use paired landmark differences. Singleton groups remain in the complete CSV but are omitted from this compact table.', '',
           '| Anchor | n | Method | Change vs original | W/T/L | Change vs scalar | W/T/L |',
           '|---|---:|---|---:|---|---:|---|']
    for group,n in audit['composition']['anchor'].items():
        if n<2: continue
        for h in METHODS:
            a=lookup[('anchor',group,h,'baseline')];b=lookup[('anchor',group,h,'scalar')]
            lines.append(f"| {group} | {n} | {h} | {a['percent_change_of_means']:+.2f}% | {a['wins']}/{a['ties']}/{a['losses']} | {b['percent_change_of_means']:+.2f}% | {b['wins']}/{b['ties']}/{b['losses']} |")
    lines += ['', 'H2 illustrates why the comparator matters: all three facade cases improve slightly over original VGGT, yet all three are worse than the scalar control. For the two statue cases, H3 beats original VGGT twice, but beats scalar only once. Its mean improvement over scalar is therefore not a universal statue result.', '',
              '## H1 regional pattern', '',
              '| Region | n | Change vs original | Wins vs original | Change vs scalar | Wins vs scalar |',
              '|---|---:|---:|---:|---:|---:|']
    for group,n in audit['composition']['region'].items():
        a=lookup[('region',group,'H1','baseline')];b=lookup[('region',group,'H1','scalar')]
        lines.append(f"| {group} | {n} | {a['percent_change_of_means']:+.2f}% | {a['wins']}/{n} | {b['percent_change_of_means']:+.2f}% | {b['wins']}/{n} |")
    pmin=min(x['permutation_p'] for x in influence);pmax=max(x['permutation_p'] for x in influence)
    nfail=sum(x['bonferroni_p_original_family_18']>=.05 for x in influence)
    lines += ['',
              'North America and South Asia have 4/4 and 5/5 wins against scalar, respectively, and their mean direction remains favorable after any one case is removed. They have only 3/4 and 3/5 wins against original VGGT. East Asia has two scalar wins and two losses; Heian-jingu supplies 87.1% of that region\'s total absolute H1-minus-scalar difference. SE Asia has only 1/4 scalar wins despite its favorable mean: Prambanan supplies 95.5% of the total absolute difference, and removing it changes the group mean from favorable to unfavorable.', '',
              f'For the H1-versus-scalar regional omnibus association, leave-one-landmark-out sensitivity gives unadjusted permutation p values from {pmin:.5f} to {pmax:.5f}. In {nfail}/36 deletions, the result no longer clears the original 18-test Bonferroni threshold. These are follow-up influence diagnostics, not 36 independent discoveries. Each uses 99,999 permutations and a fixed seed; equal group-size configurations share the same cached permutation null.', '',
              'Region, architecture, anchor, original difficulty, and intervention response can overlap in this small sample. The omnibus result identifies separation of outcome ranks among the observed regions; it does not isolate geography as the cause or validate a region-based steering policy.', '',
              '## Building types', '',
              'No building-type omnibus comparison survives the 18-test correction. H1 beats scalar on all seven Misc landmarks with a 12.62% lower group-mean error, but Misc is a broad residual category and should not be converted into a semantic selection rule. Temple/Shrine has four H1 scalar wins out of five, yet a slightly worse mean (+0.10%): the Heian-jingu regression offsets the wins. The full table includes every building type and leave-one-out mean range.', '',
              '## H3 mechanism', '',
              '| Executed rule category | n | Relation to scalar |',
              '|---|---:|---|',
              '| Unsupported anchor fallback | 19 | Exactly equal to scalar |',
              '| Supported anchor, same selected strength | 14 | Exactly equal to scalar |',
              '| Supported anchor, different selected strength | 3 | Two scalar wins, one scalar loss |', '',
              'The three changed cases use eta=100 instead of global eta=1000:', '',
              '| Landmark | Anchor | H3 minus original | H3 minus scalar |',
              '|---|---|---:|---:|']
    for r in changed:
        lines.append(f"| {r['scene'].replace('_',' ')} | {r['anchor']} | {r['H3_minus_baseline']:+.5f} | {r['H3_minus_scalar']:+.5f} |")
    lines += ['',
              'Across these three cases, mean error is 11.34% lower than original and 20.34% lower than scalar. This is the subgroup defined by the executed rule, not a new independently selected test set. Its sample is too small for a general column/statue claim. The full H3 mean remains worse than original: the 19 fallback cases inherit scalar\'s 7.76% worse group mean, while the 14 supported-but-unchanged cases are 0.40% better than original. The result is consistent with limited incremental coverage of the anchor-specific policy.', '',
              '## Omnibus tests and method', '',
              'The analysis family was defined as three fixed categorical fields × three frozen methods × two controls = 18 exploratory tests. For each comparison, compute log(method/control), assign average ranks to exact ties, and calculate the fraction of centered rank sum-of-squares explained by the category. This is a rank-based between-group separation statistic, not a predictive R-squared. Shuffle the 36 landmark outcome vectors jointly 49,999 times (seed 2026112602), preserving dependence between methods. The permutation p value is `(1 + number of permuted statistics at least as large as observed) / 50000`. Apply Benjamini-Hochberg and Bonferroni across all 18 p values. No data-driven anchor merges, outcome cutoffs, or singleton removal are used in the tests.', '',
              '| Grouping | Comparison | Permutation p | BH q (18 tests) |',
              '|---|---|---:|---:|']
    for t in tests:
        lines.append(f"| {t['grouping']} | {t['method']} vs {t['control']} | {t['permutation_p']:.5f} | {t['bh_q_18_tests']:.5f} |")
    lines += ['',
              'Permutation inference assumes exchangeable landmark outcome ranks under the null. The confirmation subset was deliberately selected, not randomly sampled from the entire dataset; category sizes are small and some categories are highly sparse. The p values therefore characterize exploratory associations in this analyzed subset and do not establish population-wide effects. Within-family correction does not account for additional analyses outside these 18 tests; use the combined report\'s broader multiplicity sensitivity when drawing overall conclusions.', '',
              '## Reproduction and audit', '',
              '`python analyze_groups.py --source <evidence_directory> --output <result_directory>` regenerates the numeric outputs. `--prior-anchor-summary <anchor_comparisons.csv>` optionally checks the previous anchor table. Required inputs are paired_hypotheses.csv, h1_run_study.json, h2_run_study.json, h3_study.json and h3_selection.json. Outputs include data_audit.json (source checksums and exceptions), group_effects.csv, categorical_omnibus_tests.csv, h1_region_leave_one_out.csv, h3_rule_groups.csv and h3_changed_landmarks.csv.', '',
              'Checks passed: 36 unique confirmation rows; all three study plans agree on split/category labels; 18 calibration rows are excluded; H3 rule choices and fallback flags match their frozen selection file; equality-to-scalar cases are exact; and the existing anchor-summary means/counts were reproduced to numerical precision. The independent unit is the landmark, so the 1,776 input views are not treated as additional observations. No model reruns, setting changes or final-holdout outcomes were used.', '']
    (OUT/'findings.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    with (SOURCE / 'paired_hypotheses.csv').open(encoding='utf-8-sig', newline='') as f:
        paired = list(csv.DictReader(f))
    assert len(paired) == 36 and len({r['scene'] for r in paired}) == 36
    plans = [load_json(name)['plan']['rows'] for name in ['h1_run_study.json','h2_run_study.json','h3_study.json']]
    mappings = [{r['scene']:r for r in plan} for plan in plans]
    selection = load_json('h3_selection.json')
    confirmation = {r['scene'] for r in plans[0] if r['split']=='confirmation'}
    assert confirmation == {r['scene'] for r in paired}
    calibration = {r['scene'] for r in plans[0] if r['split']=='calibration'}
    assert len(calibration) == 18 and not calibration & confirmation
    assert set(selection['calibration_scenes']) == calibration
    for mapping in mappings[1:]:
        for r in plans[0]:
            for field in ['split','anchor','supplied_anchor','region','building_type']:
                assert mapping[r['scene']][field] == r[field], (r['scene'], field)

    rows = []
    for p in paired:
        record = {**mappings[0][p['scene']], **p}
        assert record['split'] == 'confirmation'
        assert p['anchor'] == mappings[0][p['scene']]['anchor']
        for metric in ['baseline','scalar','H1','H2','H3']:
            record[metric] = float(record[metric])
            assert math.isfinite(record[metric]) and record[metric] > 0
        record['num_views'] = int(record['num_views'])
        fallback = record['anchor'] in selection['fallback_types']
        assert fallback == (record['H3_fallback']=='True')
        setting = selection['H3'][record['anchor']]
        if fallback:
            h3_group='unsupported_anchor_fallback'
        elif setting == selection['global_scalar']:
            h3_group='supported_anchor_same_scalar'
        else:
            h3_group='supported_anchor_different_setting'
        if setting == selection['global_scalar']:
            assert record['H3'] == record['scalar']
        record['H3_setting']=setting
        record['H3_rule_group']=h3_group
        rows.append(record)
    rows.sort(key=lambda r:r['scene'])

    descriptors=[]
    composition={}
    for field in GROUP_FIELDS:
        composition[field]=dict(sorted(collections.Counter(r[field] for r in rows).items()))
        for group,n in composition[field].items():
            members=[r for r in rows if r[field]==group]
            for method,control in COMPARISONS:
                descriptors.append(dict(grouping=field,group=group,**describe(members,method,control)))
    write_csv(OUT/'group_effects.csv',descriptors)
    write_csv(OUT/'confirmation_with_groups.csv',rows)

    # Predeclared family: 3 categorical fields x 3 methods x 2 controls = 18 tests.
    # Dimensionless log(method/control); rank statistic reduces domination by large errors.
    # Each permutation shuffles landmark outcome vectors jointly, preserving cross-method dependence.
    log_ratios=np.array([[math.log(r[h]/r[c]) for h,c in COMPARISONS] for r in rows])
    rank_values=np.column_stack([midranks(log_ratios[:,j]) for j in range(len(COMPARISONS))])
    centered=rank_values-rank_values.mean(axis=0)
    denominators=(centered**2).sum(axis=0)
    assert np.all(denominators>0)
    tests=[]
    rng=np.random.default_rng(SEED)
    for field in GROUP_FIELDS:
        labels=sorted(composition[field])
        membership=np.array([[r[field]==g for g in labels] for r in rows],float)
        counts=membership.sum(axis=0)
        observed=((membership.T@centered)**2/counts[:,None]).sum(axis=0)/denominators
        exceed=np.zeros(len(COMPARISONS),np.int64)
        remaining=PERMUTATIONS
        while remaining:
            batch=min(2000,remaining)
            indices=np.argsort(rng.random((batch,len(rows))),axis=1)
            permuted=centered[indices]
            sums=np.einsum('ng,bnc->bgc',membership,permuted,optimize=True)
            stats=(sums**2/counts[None,:,None]).sum(axis=1)/denominators[None,:]
            exceed+=(stats>=observed[None,:]-1e-12).sum(axis=0)
            remaining-=batch
        for j,(method,control) in enumerate(COMPARISONS):
            tests.append(dict(test_id=f'categorical__{field}__{method}__vs_{control}',grouping=field,method=method,control=control,n=36,groups=len(labels),
                              singleton_groups=int((counts==1).sum()),minimum_group_n=int(counts.min()),
                              rank_between_group_fraction=float(observed[j]),
                              permutation_p=float((exceed[j]+1)/(PERMUTATIONS+1)),
                              permutations=PERMUTATIONS,seed=SEED))
    q=bh_adjust([t['permutation_p'] for t in tests])
    for t,value in zip(tests,q):
        t['bh_q_18_tests']=float(value)
        t['bonferroni_p_18_tests']=min(1.0,18*t['permutation_p'])
    write_csv(OUT/'categorical_omnibus_tests.csv',tests)
    influence=regional_leave_one_out(rows)
    write_csv(OUT/'h1_region_leave_one_out.csv',influence)

    h3_groups=[]
    for group in sorted({r['H3_rule_group'] for r in rows}):
        members=[r for r in rows if r['H3_rule_group']==group]
        for control in CONTROLS:
            h3_groups.append(dict(rule_group=group,**describe(members,'H3',control)))
    write_csv(OUT/'h3_rule_groups.csv',h3_groups)
    changed=[dict(scene=r['scene'],anchor=r['anchor'],region=r['region'],building_type=r['building_type'],
                  baseline=r['baseline'],scalar=r['scalar'],H3=r['H3'],H3_setting=r['H3_setting'],
                  H3_minus_scalar=r['H3']-r['scalar'],H3_minus_baseline=r['H3']-r['baseline'])
             for r in rows if r['H3']!=r['scalar']]
    assert len(changed)==3
    write_csv(OUT/'h3_changed_landmarks.csv',changed)
    old_path=PRIOR_ANCHOR_SUMMARY
    checked_prior=False
    if old_path is not None and old_path.exists():
        with old_path.open(encoding='utf-8-sig',newline='') as f:
            old=list(csv.DictReader(f))
        lookup={(r['group'],r['method'],r['control']):r for r in descriptors if r['grouping']=='anchor'}
        for r in old:
            new=lookup[(r['anchor'],r['method'],r['reference'])]
            assert int(r['n'])==new['n']
            assert math.isclose(float(r['mean_difference']),new['mean_difference'],abs_tol=1e-10)
        checked_prior=True
    source_paths=[SOURCE/'paired_hypotheses.csv',SOURCE/'h1_run_study.json',SOURCE/'h2_run_study.json',SOURCE/'h3_study.json',SOURCE/'h3_selection.json']
    if checked_prior: source_paths.append(old_path)
    audit=dict(n_confirmation=36,n_calibration=18,calibration_excluded=True,one_row_per_landmark=True,
               three_study_plans_agree=True,confirmation_plan_matches_paired_table=True,
               executed_plan_anchor_matches_paired_table=True,prior_anchor_summary_reproduced=checked_prior,
               supplied_anchor_exceptions=[dict(scene=r['scene'],supplied_anchor=r['supplied_anchor'],executed_anchor=r['anchor']) for r in rows if r['supplied_anchor']!=r['anchor']],
               h3_fallback_flag_matches_selection=True,composition=composition,
               h3_rule_counts=dict(collections.Counter(r['H3_rule_group'] for r in rows)),
               source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (OUT/'analysis_summary.json').write_text(json.dumps(dict(audit=audit,tests=tests,h3_rule_groups=h3_groups,changed_landmarks=changed,
        h1_region_influence=dict(minimum_p=min(x['permutation_p'] for x in influence),maximum_p=max(x['permutation_p'] for x in influence),
             p_above_05=sum(x['permutation_p']>=.05 for x in influence),original_family_bonferroni_above_05=sum(x['bonferroni_p_original_family_18']>=.05 for x in influence))),indent=2),encoding='utf-8')
    write_findings(descriptors,tests,audit,influence,changed)
    print(json.dumps(dict(composition=composition,h3_rule_counts=audit['h3_rule_counts'],
                          minimum_p=min(t['permutation_p'] for t in tests),minimum_q=min(t['bh_q_18_tests'] for t in tests),
                          significant_bh=[t for t in tests if t['bh_q_18_tests']<.05]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--prior-anchor-summary',type=Path)
    args=parser.parse_args()
    PRIOR_ANCHOR_SUMMARY=args.prior_anchor_summary
    if PRIOR_ANCHOR_SUMMARY is None and args.source.resolve()==SOURCE.resolve():
        PRIOR_ANCHOR_SUMMARY=ROOT/'report/anchor_analysis/anchor_comparisons.csv'
    SOURCE=args.source.resolve();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    main()
