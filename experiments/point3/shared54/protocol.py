"""Frozen shared experiment design. No model outcomes enter split selection."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]
REVISION = '5d4404323a21aa3de948f8bc57a053b29fa3d921'
SCHEMA = 'point3-shared54-v1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def candidates():
    return {
        'no_intervention': {'eta': 0, 'strength': {'mode': 'fixed'}},
        'scalar_eta100': {'eta': 100, 'strength': {'mode': 'fixed'}},
        'scalar_eta1000': {'eta': 1000, 'strength': {'mode': 'fixed'}},
        'layer_alpha0.01': {'eta': 1, 'strength': {'mode': 'layer_relative', 'alpha': .01, 'max_step': 100000.}},
        'layer_alpha0.02': {'eta': 1, 'strength': {'mode': 'layer_relative', 'alpha': .02, 'max_step': 100000.}},
        'view_eta100': {'eta': 100, 'strength': {'mode': 'view_fixed', 'view_exponent': .5, 'view_min': .5, 'view_max': 2.}},
        'view_eta1000': {'eta': 1000, 'strength': {'mode': 'view_fixed', 'view_exponent': .5, 'view_min': .5, 'view_max': 2.}},
    }


def candidate_names(hypothesis=None):
    if hypothesis not in (None,'H1','H2','H3'):raise ValueError('Unknown hypothesis')
    prefixes={None:('scalar_','layer_','view_'),'H1':('scalar_','layer_'),
              'H2':('scalar_','view_'),'H3':('scalar_',)}[hypothesis]
    return [name for name in candidates() if name=='no_intervention' or name.startswith(prefixes)]


def split_metadata(rows, count=18, seed=1729):
    """Two calibration examples per well-supported anchor, at least one left outside.

    Search 512 seeded metadata-only allocations, selecting the best type/region
    coverage, then marginal balance. The rule and all identities are frozen.
    No claim of a statistical optimum or population representation is made.
    """
    if count == len(rows):
        return sorted(r['scene'] for r in rows)
    groups = {}
    for row in rows:
        groups.setdefault(row['anchor'], []).append(row['scene'])
    supported = sorted(a for a, values in groups.items() if a and len(values) >= 3)
    if count < len(supported)*2 or count >= len(rows):
        raise ValueError('Calibration size cannot retain the stated per-anchor support')
    by_id = {r['scene']: r for r in rows}
    rng = random.Random(seed)
    def score(ids):
        chosen = [by_id[s] for s in ids]
        coverage = sum(len({r[k] for r in chosen}) for k in ('region', 'building_type'))
        tv = 0.
        for key in ('region', 'building_type'):
            full, part = Counter(r[key] for r in rows), Counter(r[key] for r in chosen)
            tv += sum(abs(part[x]/len(chosen)-full[x]/len(rows)) for x in full)/2
        return (-coverage, tv, tuple(sorted(ids)))
    options = []
    for _ in range(512):
        chosen = set()
        for anchor in supported:
            chosen.update(rng.sample(sorted(groups[anchor]), 2))
        while len(chosen) < count:
            available = [r['scene'] for r in rows if r['scene'] not in chosen and
                (r['anchor'] not in supported or
                 sum(s in chosen for s in groups[r['anchor']]) < len(groups[r['anchor']])-1)]
            rng.shuffle(available)
            best = min(available, key=lambda s: score(chosen | {s})[:2])
            chosen.add(best)
        options.append(sorted(chosen))
    return min(options, key=score)


def make_plan(calibration_size=18,hypothesis=None):
    names=candidate_names(hypothesis)
    selected = load(HERE/'inputs/selection54.json')['selected_landmarks']
    training = load(HERE/'inputs/training_landmarks.json')
    heldout = load(HERE/'inputs/evaluation_landmarks.json')
    # Frozen ID files are lists, not data to be inferred from their directory names.
    if isinstance(training, dict): training = training.get('landmarks', training.get('scenes'))
    if isinstance(heldout, dict): heldout = heldout.get('landmarks', heldout.get('scenes'))
    if len(selected) != 54 or len(set(selected)) != 54 or not set(selected) <= set(training):
        raise ValueError('Expected the frozen 54 training IDs')
    if set(selected) & set(heldout) or len(heldout) != 79:
        raise ValueError('Frozen 79-landmark holdout must remain separate')
    metadata = load(HERE/'inputs/training_metadata.json')
    overrides = {'Forth_Bridge': {'anchor': 'bridge', 'provenance':
        'User explicitly authorized manual bridge anchor on 2026-11-15; requires SAM mask validation'},
        'Washington_Monument': {'anchor': 'obelisk', 'preserve_split_anchor': 'altar', 'provenance':
        'User explicitly authorized manual obelisk anchor on 2026-11-15 after all 50 supplied-altar masks were empty; '
        'preserve the original calibration/confirmation allocation; review every selected-view overlay'}}
    rows = []
    for scene in sorted(selected):
        row = {'scene': scene, **metadata[scene]}
        row['supplied_anchor'] = row['anchor']
        if scene in overrides: row['anchor'] = overrides[scene]['anchor']
        if row['anchor'] == '__MISSING__': row['anchor'] = None
        rows.append(row)
    # The mask correction must not redraw the allocation frozen before inference.
    allocation_rows = [{**row, 'anchor': overrides.get(row['scene'], {}).get('preserve_split_anchor', row['anchor'])}
                       for row in rows]
    calibration = split_metadata(allocation_rows, calibration_size)
    for row in rows:
        row['split'] = 'calibration' if row['scene'] in calibration else 'confirmation'
    counts = Counter(r['anchor'] for r in rows if r['split']=='calibration')
    supported = sorted(a for a, n in counts.items() if a and n >= 2)
    plan=dict(schema=SCHEMA, dataset_revision=REVISION, num_views=50, seed=1729,
        view_rule='seeded_uniform_registered_rgb_depth_validity_up_to50_v2',
        view_count_policy='up_to_available', min_views=2,
        view_count_decision='User authorized keeping the exact 54 landmarks with all eligible distinct views up to 50 on 2026-11-15',
        calibration_size=calibration_size, rows=rows, heldout_ids=heldout,
        anchor_overrides=overrides, calibration_anchor_counts=dict(counts),
        supported_anchor_types=supported,
        fallback_anchor_types=sorted({r['anchor'] for r in rows if r['anchor']} - set(supported)),
        candidate_definitions={k:candidates()[k] for k in names}, sam_confidence=.5,
        sam_revision='11dec2936de97f2857c1f76b66d982d5a001155d',
        vggt_revision='97bbde571faddde3ace3cfa7724a20448026c4c8',
        weights_precision='float32_weights_autocast_reference',
        min_anchor_calibration=2, note='36 confirmation landmarks are internal validation; 79 final holdout IDs stay untouched')
    if hypothesis:plan['hypothesis']=hypothesis
    return plan


def select_settings(plan, scores):
    calibration = [r for r in plan['rows'] if r['split']=='calibration']
    scope=plan.get('hypothesis')
    expected = set(candidate_names(scope)) - {'no_intervention'}
    if set(scores) != {r['scene'] for r in calibration}:
        raise ValueError('Freeze only after every calibration landmark completes; confirmation/holdout scores are forbidden')
    if any(set(s) != expected for s in scores.values()):
        raise ValueError('Every calibration candidate is required')
    if any(not math.isfinite(v) or v < 0 for s in scores.values() for v in s.values()):
        raise ValueError('Invalid calibration metric')
    def best(prefix, names):
        choices = [k for k in expected if k.startswith(prefix)]
        return min(choices, key=lambda k: (statistics.mean(scores[s][k] for s in names),
            candidates()[k]['eta'], candidates()[k]['strength'].get('alpha', 0)))
    names = [r['scene'] for r in calibration]
    scalar = best('scalar_', names)
    anchor_map, fallback = {}, []
    for anchor in sorted({r['anchor'] for r in plan['rows'] if r['anchor']}):
        group = [r['scene'] for r in calibration if r['anchor']==anchor]
        if len(group) >= plan['min_anchor_calibration']:
            anchor_map[anchor] = best('scalar_', group)
        else:
            anchor_map[anchor] = scalar
            fallback.append(anchor)
    chosen=dict(plan_sha256=fingerprint(plan), calibration_scores_sha256=fingerprint(scores),
        global_scalar=scalar, fallback_types=fallback, calibration_scenes=names,
        objective='mean landmark CD_norm', selection_uses_confirmation=False)
    if scope in (None,'H1'):chosen['H1']=best('layer_',names)
    if scope in (None,'H2'):chosen['H2']=best('view_',names)
    if scope in (None,'H3'):chosen['H3']=anchor_map
    return chosen


def policies(row, selection):
    result={'no_intervention':'no_intervention', 'global_scalar':selection['global_scalar']}
    for h in ('H1','H2'):
        if h in selection:result[h]=selection[h]
    if 'H3' in selection:result['H3']=selection['H3'].get(row['anchor'], selection['global_scalar'])
    return result


def conditions(row, selection=None,hypothesis=None):
    if row['split']=='calibration': return candidate_names(hypothesis)
    if selection is None: raise ValueError('Freeze settings before confirmation')
    requested = set(policies(row,selection).values())
    return [c for c in candidate_names(hypothesis) if c in requested]


def partition(rows, count):
    """Deterministic disjoint scene assignment; all conditions of a scene stay together."""
    if count < 1: raise ValueError('At least one worker is required')
    bins, loads = [[] for _ in range(count)], [0]*count
    archives = load(HERE/'inputs/archive_sources.json')
    for row in sorted(rows,key=lambda r:(-archives.get(r['scene'],{}).get('size',1),r['scene'])):
        target = min(range(count),key=lambda i:(loads[i],i))
        bins[target].append(row)
        loads[target] += archives.get(row['scene'],{}).get('size',1)
    return bins


def job_budget(plan):
    n = sum(r['split']=='calibration' for r in plan['rows'])
    scope=plan.get('hypothesis');cal=len(candidate_names(scope))*n
    confirm=(3 if scope else 5)*(len(plan['rows'])-n)
    return {'hypothesis':scope or 'all', 'calibration_jobs':cal, 'confirmation_jobs_upper_bound':confirm,
            'total_upper_bound':cal+confirm, 'reference_checks_extra':True,
            'independent_full54_notebooks':702, 'shared_full54_sweep':378}


if __name__=='__main__':
    plan=make_plan()
    print(json.dumps({'budget':job_budget(plan), 'calibration':[r['scene'] for r in plan['rows'] if r['split']=='calibration'],
                     'anchor_counts':plan['calibration_anchor_counts'], 'fallback':plan['fallback_anchor_types']},indent=2))
