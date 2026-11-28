"""Fixed54 repair, original membership, sampling and export regressions."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import select_landmarks as core
import repair_landmarks as repair
from test_selection import small_population, encoded_fixture


def repair_population():
    train, _, metadata, *_ = core.load_inputs()
    rows = [{**r, **metadata[s], 'scene': s} for r, s in zip(small_population(317).rows, train)]
    rng = np.random.default_rng(73)
    return repair.RepairPopulation(rows, rng.normal(size=(317, 8)), np.abs(rng.normal(size=(317, 8))), repair.RepairConfig())


class RepairTests(unittest.TestCase):
    def test_frozen_base_cannot_change_or_include_evaluation(self):
        train, evaluation, *_ = core.load_inputs()
        base = repair.load_base(train)
        self.assertEqual(len(base), 40)
        self.assertFalse(set(base) & set(evaluation))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'changed.json'
            core.save_json(path, [evaluation[0]] + base[1:])
            with self.assertRaisesRegex(ValueError, 'checksum'):
                repair.load_base(train, path)

    def test_all_336_repairs_keep_40_and_fill_exactly_14_categories(self):
        pop = repair_population()
        winner, candidates, base, groups, missing = repair.enumerate_repairs(pop, repair.load_base(pop.scenes))
        self.assertEqual(len(candidates), 336)
        self.assertEqual(len({tuple(x['indices']) for x in candidates}), 336)
        self.assertEqual(len(missing), 14)
        self.assertIn('__MISSING__', missing)
        for candidate in candidates:
            chosen = candidate['indices']
            self.assertEqual(len(chosen), len(set(chosen)))
            self.assertEqual(len(chosen), 54)
            self.assertTrue(set(base) <= set(chosen))
            self.assertTrue(pop.valid_selection(chosen))
            anchors = Counter(pop.rows[i]['anchor'] for i in candidate['added_indices'])
            self.assertEqual(anchors, Counter({a: 1 for a in missing}))
        self.assertEqual(winner, min(candidates, key=lambda c: (c['metrics']['score'], c['indices'])))
        comparison = repair.common_target_comparison(pop, base, np.asarray(winner['indices']))
        self.assertEqual(comparison['target_count'], 263)
        for key in ('mean', 'p95', 'worst'):
            self.assertLessEqual(comparison['repaired54'][key], comparison['original40'][key])
        # The seed has incomplete anchors; category presence is a real enforced rule.
        self.assertFalse(pop.valid_selection(base))

    def test_random_references_have_the_declared_conditioning(self):
        pop = repair_population()
        winner, _, base, groups, missing = repair.enumerate_repairs(pop, repair.load_base(pop.scenes))
        chosen = np.asarray(winner['indices'])
        with tempfile.TemporaryDirectory() as tmp:
            reference = repair.random_repair_reference(pop, base, groups, tmp)
            self.assertEqual(reference['draws'], 1000)
            self.assertEqual(reference['possible_memberships'], 336)
            self.assertLessEqual(reference['distinct_memberships'], 336)
            with np.load(Path(tmp) / 'random_repairs.npz', allow_pickle=False) as data:
                original_draws = data['selected_indices'].copy()
                for draw in original_draws:
                    self.assertTrue(set(base) <= set(draw))
                    self.assertTrue(pop.valid_selection(draw))
                    self.assertEqual(len(draw), 54)
            repair.random_repair_reference(pop, base, groups, tmp)
            with np.load(Path(tmp) / 'random_repairs.npz', allow_pickle=False) as data:
                np.testing.assert_equal(original_draws, data['selected_indices'])
            matched = core.anchor_matched_reference(pop, chosen, tmp)
            self.assertEqual(matched['draws'], 1000)
            counts = Counter((pop.rows[i]['building_type'], pop.rows[i]['anchor']) for i in chosen)
            with np.load(Path(tmp) / 'anchor_matched_random_subsets.npz', allow_pickle=False) as data:
                for draw in data['selected_indices']:
                    self.assertEqual(counts, Counter((pop.rows[i]['building_type'], pop.rows[i]['anchor']) for i in draw))
                    self.assertTrue(pop.valid_selection(draw))

    def test_full_repair_export_resume_and_all_image_manifests(self):
        train, evaluation, _, _, checks = core.load_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); enc = root / 'enc'; enc.mkdir()
            settings = {'dataset_revision': '5d4404323a21aa3de948f8bc57a053b29fa3d921', 'input_hashes': checks}
            core.save_json(enc / 'settings.json', settings)
            core.save_json(enc / 'progress.json', {'requested': train, 'completed': train, 'failures': [],
                           'all_317_training_landmarks_complete': True, 'holdout_encoded': False})
            rng = np.random.default_rng(74)
            for scene in train:
                encoded_fixture(enc / 'landmarks' / scene, scene, settings, rng)
            (enc / 'landmarks' / evaluation[0]).mkdir()
            (enc / 'landmarks' / evaluation[0] / 'manifest.json').write_text('DO NOT READ EVALUATION')
            with patch.object(core, 'emit'):
                result = repair.run_repair_selection(enc, root / 'out')
                repeated = repair.run_repair_selection(enc, root / 'out')
            self.assertEqual(result, repeated)
            self.assertEqual(result['selected_landmarks'], 54)
            self.assertEqual(result['selected_images'], 162)
            output = Path(result['output_dir'])
            selected = core.read_json(output / 'selected_landmarks.json')
            additions = core.read_json(output / 'added_landmarks.json')
            self.assertEqual(set(selected), set(repair.load_base(train)) | set(additions))
            self.assertEqual(len(additions), 14)
            self.assertFalse(set(selected) & set(evaluation))
            self.assertEqual((output / 'evaluation_landmarks.json').read_bytes(), (core.HERE / 'inputs/evaluation_landmarks.json').read_bytes())
            self.assertEqual(core.sha(output / 'fixed_base_landmarks.json'), repair.BASE_SHA256)
            for scene in selected:
                self.assertEqual((output / 'selected_image_manifests' / f'{scene}.json').read_bytes(), (enc / 'landmarks' / scene / 'manifest.json').read_bytes())
            audit = core.read_json(output / 'audit.json')
            self.assertTrue(audit['base_retained'])
            self.assertEqual(audit['candidate_count'], 336)
            self.assertEqual(audit['omitted_levels']['anchor'], [])
            self.assertEqual(audit['anchor_constraints']['mode'], 'presence_only')
            self.assertFalse(core.read_json(output / 'size_decision.json')['size_search_run'])
            self.assertFalse((output / 'size_curve.json').exists())
            self.assertFalse((root / 'out/.selector.lock').exists())
            missing = output / 'selected_image_manifests' / f'{selected[0]}.json'; missing.unlink()
            with patch.object(core, 'emit'):
                self.assertEqual(result, repair.run_repair_selection(enc, root / 'out'))
            self.assertTrue(missing.is_file())


if __name__ == '__main__':
    unittest.main()
