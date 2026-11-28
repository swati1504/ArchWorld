"""Keep the frozen original 40; add exactly one landmark per missing anchor label."""
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import math
import shutil
import sys

import numpy as np

if __package__:
    from . import select_landmarks as core
else:
    import select_landmarks as core

HERE = Path(__file__).resolve().parent
BASE_SHA256 = "4f8e8885f1cf3bc0d48e92f662555c76293cf4ebf683941f27c8e232b6c36d2c"
BASE_SOURCE = "https://drive.google.com/file/d/19FmHySwP1LXIQde8-l4fQPipoDHoy8Qi/view"
SIZE_STATUS = "fixed_original40_plus14_missing_anchor_categories"
LIMITATIONS = [
    "54 is the user-chosen repair size: original 40 plus one per missing category, not a coverage-derived optimum.",
    "All original 40 landmarks are locked. Building-type, region and anchor proportions are audited and penalized, not enforced as quotas.",
    "Every supplied anchor label, including the separate missing-label category, must appear. Presence does not prove visible elements or usable masks.",
    "The exhaustive minimum is only among additions to this fixed original 40 under the stated score, not among all possible 54-landmark subsets.",
    "Random comparisons are descriptive on training features used for selection; they are not independent validation or statistical significance tests.",
] + [x for x in core.LIMITATIONS if not x.startswith("The chosen size")]


@dataclass(frozen=True)
class RepairConfig:
    seed: int = 1729
    random_subsets: int = 1000
    appearance_weight: float = 1.0
    composition_weight: float = 1.0
    scale_weight: float = 1.0
    collection_weight: float = 1.0

    def __post_init__(self):
        core.SelectionConfig(**asdict(self))  # Shared validation of these fields.

    @property
    def family_weights(self):
        return [self.appearance_weight, self.composition_weight, self.scale_weight, self.collection_weight]

    @property
    def anchor_count_slack(self):
        return None  # Count quotas do not apply to this explicit repair policy.

    @property
    def anchor_policy(self):
        return "Presence of every supplied anchor label including __MISSING__; proportions audited, no proportional count bounds"


class RepairPopulation(core.Population):
    def anchor_bounds(self, size):
        return np.ones(len(self.levels['anchor']), dtype=int), self.counts['anchor'].copy()

    def valid_selection(self, selected, quota=None):
        selected = np.asarray(selected, dtype=int)
        if (not len(selected) or len(np.unique(selected)) != len(selected)
                or selected.min() < 0 or selected.max() >= self.n):
            return False
        return bool(self.region_complete(selected) and self.anchors_valid(selected)
                    and len(np.unique(self.codes['building_type'][selected])) == len(self.levels['building_type']))


def load_base(training, seed_file=HERE / 'seeds/original_40_landmarks.json'):
    if core.sha(seed_file) != BASE_SHA256:
        raise ValueError('Original 40-landmark seed checksum mismatch')
    base = core.read_json(seed_file)
    if len(base) != 40 or len(set(base)) != 40 or not set(base) <= set(training):
        raise ValueError('Seed must contain exactly the frozen 40 distinct training landmarks')
    return base


def enumerate_repairs(pop, base_names):
    base = np.asarray(sorted(pop.scenes.index(s) for s in base_names), dtype=int)
    if len(base) != 40 or len(set(base)) != 40:
        raise ValueError('Expected 40 distinct locked landmarks')
    present = set(pop.codes['anchor'][base])
    missing = [i for i in range(len(pop.levels['anchor'])) if i not in present]
    if len(missing) != 14 or not pop.region_complete(base) or len(np.unique(pop.codes['building_type'][base])) != 9:
        raise ValueError('The frozen base must cover all types/regions and miss exactly 14 anchor categories')
    groups = [np.flatnonzero(pop.codes['anchor'] == i) for i in missing]
    total = math.prod(len(g) for g in groups)
    if total > 100000:
        raise ValueError('Unexpected repair space; refusing to silently replace exhaustive search')
    candidates = []
    for additions in product(*groups):
        selected = np.sort(np.r_[base, additions])
        if len(selected) != 54 or not pop.valid_selection(selected):
            raise RuntimeError('Repair violates size, anchor, type or region presence')
        candidates.append({'indices': selected.tolist(), 'added_indices': list(map(int, additions)),
                           'metrics': pop.metrics(selected)})
    winner = min(candidates, key=lambda x: (x['metrics']['score'], x['indices']))
    return winner, candidates, base, groups, [pop.levels['anchor'][i] for i in missing]


def random_repair_reference(pop, base, groups, output):
    """Uniform independent draws from the same fixed-base repair space, with replacement."""
    rng = core.seeded(pop.config, 7, 54)
    draws = [np.sort(np.r_[base, [rng.choice(g) for g in groups]]) for _ in range(pop.config.random_subsets)]
    metrics = [pop.metrics(x) for x in draws]
    columns = [k for k, v in metrics[0].items() if isinstance(v, (int, float, bool))]
    core.save_npz(Path(output) / 'random_repairs.npz', training_landmarks=np.asarray(pop.scenes),
                  selected_indices=np.asarray(draws, dtype=np.int32), metric_columns=np.asarray(columns),
                  metrics=np.asarray([[m[k] for k in columns] for m in metrics], dtype=float))
    return {'draws': len(draws), 'distinct_memberships': len({tuple(x) for x in draws}),
            'possible_memberships': math.prod(len(g) for g in groups),
            'conditioning': 'frozen original40 plus one uniform choice per missing anchor category; repeated memberships expected'}


def common_target_comparison(pop, base, chosen):
    target = np.setdiff1d(np.arange(pop.n), chosen)
    def measure(selected):
        near = pop.d[np.ix_(target, selected)].min(axis=1)
        return {'mean': float(near.mean()), 'p95': float(np.quantile(near, .95)), 'worst': float(near.max())}
    return {'target_count': len(target), 'target_landmarks': [pop.scenes[i] for i in target],
            'original40': measure(base), 'repaired54': measure(chosen),
            'interpretation': 'Both evaluated on the SAME 263 landmarks outside the repaired set; additions cannot increase nearest distance on this fixed target'}


def repair_figures(audit, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison = audit['common_target_comparison']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(2)
    for shift, key, label in [(-.18, 'original40', 'Original 40'), (.18, 'repaired54', 'Repaired 54')]:
        ax.bar(x + shift, [comparison[key]['mean'], comparison[key]['p95']], .36, label=label)
    ax.set(xticks=x, xticklabels=['Average distance', '95th-percentile distance'],
           ylabel='Combined descriptor distance (lower is better)', title='Coverage of the SAME 263 remaining training landmarks')
    ax.legend(); fig.tight_layout(); fig.savefig(output / 'coverage_comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 15))
    for ax, field in zip(axes, core.CATEGORIES):
        rows = audit['distributions'][field]; y = np.arange(len(rows))
        ax.barh(y - .18, [r['training_share'] * 100 for r in rows], .36, label='Training pool')
        ax.barh(y + .18, [r['subset_share'] * 100 for r in rows], .36, label='Repaired 54')
        ax.set(yticks=y, yticklabels=[r['labels'][0] for r in rows], xlabel='Landmarks (%)', title=field.replace('_', ' ').title())
        ax.legend()
    fig.tight_layout(); fig.savefig(output / 'composition_balance.png', dpi=160); plt.close(fig)


def write_repair_report(pop, chosen, added, audit, output):
    lines = ['# Original 40 repaired to 54 landmarks', '',
             f'**54 training landmarks, {sum(pop.rows[i]["image_count"] for i in chosen):,} original images.** The 79 evaluation landmarks remain separate and unchanged.', '',
             'The original 40 are retained. Exactly one landmark was added for each of 13 missing named anchor labels and the separate missing-label category. This is the requested fixed-size repair; no elbow search or full-training fallback is used.', '',
             f'All **{audit["candidate_count"]} possible repairs** were scored. The winner minimizes the declared combined coverage/balance score; ties use sorted landmark indices.', '',
             '## Added landmarks', '', '| Anchor category | Added landmark |', '|---|---|']
    lines += [f'| {r["anchor"]} | {r["scene"]} |' for r in added]
    lines += ['', '## Training-only comparisons', '',
              f'The main reference contains {pop.config.random_subsets:,} random 54-landmark draws with the same building-type by anchor counts as the winner, conditional on all regions appearing. It does not lock the original 40. A separate {pop.config.random_subsets:,} draws randomly fill the missing categories while keeping the original 40. Repeated memberships are allowed and their distinct count is saved.', '',
              '| Metric (lower is better) | Selected | Matched random median | Random 5–95% |', '|---|---:|---:|---:|']
    for key in ('coverage_mean', 'coverage_p95', 'appearance_mean', 'score'):
        r = audit['random_comparison'][key]
        lines.append(f'| {key} | {r["selected"]:.4f} | {r["random_median"]:.4f} | {r["random_p05"]:.4f}–{r["random_p95"]:.4f} |')
    lines += ['', 'Every named anchor and the missing-label category is present. All nine building types and eight regions are present. **Proportional count quotas are not required in this repair mode.** Distribution gaps are measured in `audit.json` and `composition_balance.png`.', '',
              'The before/after chart uses the same 263 remaining training landmarks for both sets. Adding examples necessarily cannot worsen their nearest-selected distances; this comparison is descriptive, not evidence of better intervention outcomes.', '',
              '## Interpretation and limits', ''] + [f'- {x}' for x in LIMITATIONS]
    lines += ['', '## Saved files', '', '`selected_landmarks.json`, `fixed_base_landmarks.json`, `added_landmarks.json`, complete `selected_image_manifests/`, unchanged `evaluation_landmarks.json`, `subset_manifest.json`, all `repair_candidates.json`, random-draw files, `audit.json`, and `selection_protocol.json`.', '',
              '![Coverage comparison](coverage_comparison.png)', '', '![Composition balance](composition_balance.png)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def run_repair_selection(encodings, output_dir, config=None, inputs_dir=core.HERE / 'inputs'):
    config = config or RepairConfig()
    if not isinstance(config, RepairConfig):
        raise TypeError('Fixed54 repair requires RepairConfig, not the old size-search configuration')
    inputs = core.load_inputs(inputs_dir)
    train, evaluation, metadata, sources, checks = inputs
    base_names = load_base(train)
    encodings, root = Path(encodings).resolve(), Path(output_dir).resolve()
    if root == encodings or root.is_relative_to(encodings) or encodings.is_relative_to(root):
        raise ValueError('Selection outputs must be separate from encoder outputs')
    core.check_completion(encodings, train, evaluation)
    root.mkdir(parents=True, exist_ok=True)
    with core.output_lock(root):
        rows, means, stds, provenance = core.load_descriptors(encodings, inputs, root)
        import scipy
        protocol = {'version': core.VERSION, 'mode': 'fixed54_anchor_repair', 'config': asdict(config),
                    'repair_sha256': core.sha(__file__), 'core_sha256': core.sha(core.__file__),
                    'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__},
                    'base_sha256': BASE_SHA256, 'base_source': BASE_SOURCE, 'original_run': 'run_3181fc0dfdf53b02',
                    'provenance': provenance, 'reference': '317_training_landmarks_only', 'image_policy': 'all_original_images',
                    'population_weighting': 'one_equal_weight_per_landmark', 'distance_scaling': 'positive training-pair median per family',
                    'score': '0.35*unselected_mean_distance + 0.35*unselected_p95_distance + 0.15*mean_marginal_TV + 0.15*mean_numeric_KS',
                    'size_rule': 'exactly40 locked + exactly1 per14 missing categories =54; no fallback',
                    'count_quotas': 'none; proportions measured and included in score', 'anchor_policy': config.anchor_policy,
                    'search': 'exhaustive over additions, tie by sorted training indices', 'limitations': LIMITATIONS}
        run_id = core.signature(protocol)[:16]
        output = root / f'repair54_{run_id}'; output.mkdir(exist_ok=True)
        core.save_json(output / 'selection_protocol.json', protocol)
        pop = RepairPopulation(rows, means, stds, config)
        required = ('result.json', 'selected_landmarks.json', 'subset_manifest.json', 'audit.json', 'REPORT.md',
                    'evaluation_landmarks.json', 'fixed_base_landmarks.json', 'added_landmarks.json', 'size_decision.json',
                    'repair_candidates.json', 'anchor_counts.csv', 'random_repairs.npz', 'anchor_matched_random_subsets.npz',
                    'coverage_comparison.png', 'composition_balance.png')
        if all((output / name).is_file() for name in required):
            saved = core.read_json(output / 'selected_landmarks.json')
            if (len(saved) == len(set(saved)) == 54 and set(base_names) <= set(saved) <= set(train)
                    and pop.valid_selection([train.index(x) for x in saved])
                    and (output / 'evaluation_landmarks.json').read_bytes() == (Path(inputs_dir) / 'evaluation_landmarks.json').read_bytes()
                    and all((output / 'selected_image_manifests' / f'{x}.json').is_file()
                            and core.sha(output / 'selected_image_manifests' / f'{x}.json') == provenance['landmarks'][x]['manifest.json'] for x in saved)):
                result = core.read_json(output / 'result.json'); core.save_json(root / 'latest_result.json', result)
                core.emit(f'Reusing completed fixed54 repair: {output}')
                return result
        winner, candidates, base, groups, missing = enumerate_repairs(pop, base_names)
        chosen = np.asarray(winner['indices']); additions = np.asarray(winner['added_indices'])
        core.emit(f'Compared all {len(candidates)} repairs; retained40 and added14 =54 landmarks')
        core.save_json(output / 'repair_candidates.json', candidates)
        core.save_json(output / 'training_descriptors.json', rows)
        core.save_npz(output / 'landmark_descriptors.npz', training_landmarks=np.asarray(train), means=means, stds=stds,
                      composite_distance=pop.d, **{k + '_distance': v for k, v in pop.families.items()})
        core.save_json(output / 'distance_scales.json', pop.scales)
        audit = core.detailed_audit(pop, chosen); audit['limitations'] = LIMITATIONS
        audit['anchor_constraints']['mode'] = 'presence_only'
        audit['candidate_count'] = audit['distinct_final_candidates'] = len(candidates)
        audit['base_retained'] = set(base).issubset(chosen)
        audit['missing_categories_repaired'] = missing
        audit['common_target_comparison'] = common_target_comparison(pop, base, chosen)
        audit['random_repair_reference'] = random_repair_reference(pop, base, groups, output)
        audit['random_repair_comparison'] = core.compare_random(winner['metrics'], output, 'random_repairs.npz')
        audit['matched_random_reference'] = core.anchor_matched_reference(pop, chosen, output)
        audit['random_comparison'] = core.compare_random(winner['metrics'], output, 'anchor_matched_random_subsets.npz')
        core.save_json(output / 'audit.json', audit)
        core.write_csv(output / 'anchor_counts.csv', audit['anchor_constraints']['counts'])
        added_rows = [{'anchor': rows[i]['anchor'], 'scene': train[i], 'image_count': rows[i]['image_count']} for i in additions]
        core.save_json(output / 'added_landmarks.json', [train[i] for i in additions])
        core.write_csv(output / 'added_landmarks.csv', added_rows)
        shutil.copyfile(HERE / 'seeds/original_40_landmarks.json', output / 'fixed_base_landmarks.json')
        decision = {'size': 54, 'status': SIZE_STATUS, 'base_count': 40, 'added_count': 14,
                    'missing_categories': missing, 'candidate_count': len(candidates), 'size_search_run': False}
        core.save_json(output / 'size_decision.json', decision)
        selected_scenes = [train[i] for i in chosen]
        manifest = {'training_population': 317, 'evaluation_holdout': 79, 'image_policy': 'all_original_images',
                    'selected_landmarks': selected_scenes, 'selected_count': 54, 'size_decision': SIZE_STATUS,
                    'image_count': sum(rows[i]['image_count'] for i in chosen), 'archives': {s: sources[s] for s in selected_scenes},
                    'dataset_revision': provenance['encoder_settings']['dataset_revision'],
                    'encoder_settings_sha256': core.signature(provenance['encoder_settings']),
                    'selection_run': run_id, 'representation_reference': 'training_only', 'limitations': LIMITATIONS,
                    'anchor_constraints': audit['anchor_constraints'], 'fixed_base_sha256': BASE_SHA256}
        for scene in selected_scenes:
            dest = output / 'selected_image_manifests' / f'{scene}.json'; dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(encodings / 'landmarks' / scene / 'manifest.json', dest)
        core.save_json(output / 'subset_manifest.json', manifest)
        core.save_json(output / 'selected_landmarks.json', selected_scenes)
        core.write_csv(output / 'selected_landmarks.csv', [rows[i] for i in chosen])
        shutil.copyfile(Path(inputs_dir) / 'evaluation_landmarks.json', output / 'evaluation_landmarks.json')
        repair_figures(audit, output); write_repair_report(pop, chosen, added_rows, audit, output)
        result = {'output_dir': str(output), 'selected_landmarks': 54, 'original_landmarks_retained': 40, 'added_landmarks': 14,
                  'selected_images': manifest['image_count'], 'evaluation_landmarks': 79, 'size_status': SIZE_STATUS,
                  'report': str(output / 'REPORT.md'), 'all_anchor_constraints_passed': audit['anchor_constraints']['all_passed'],
                  'interventions_run': False, 'selection_complete': True}
        core.save_json(output / 'result.json', result); core.save_json(root / 'latest_result.json', result)
        core.emit(f'Saved exactly54 complete landmarks and audit: {output}')
        return result
