"""Scientific bookkeeping and selection checks; no GPU or interventions."""
from itertools import product
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import ks_2samp

import select_landmarks as s


def small_population(n=24, config=None):
    rng = np.random.default_rng(100)
    rows = [{"scene": f"scene_{i}", "building_type": f"type_{i % 3}", "region": f"region_{i % 4}",
             "anchor": f"anchor_{i % 5}", "extent_m": 5. + i, "image_count": 2 + i,
             "duplicate_fraction": (i % 3) / 10, "median_megapixels": 1. + i / 10,
             "p10_megapixels": 1., "p90_megapixels": 4., "portrait_fraction": .3,
             "mean_log_aspect": .1 * (i % 2), "std_log_aspect": .5} for i in range(n)]
    return s.Population(rows, rng.normal(size=(n, 8)), np.abs(rng.normal(size=(n, 8))), config or s.SelectionConfig())


def encoded_fixture(folder, scene, settings, rng, dimension=1024):
    folder.mkdir(parents=True, exist_ok=True)
    # Two distinct original paths have identical content/features: both must stay.
    images = [{"name": f"nested/image_{i}.jpg", "bytes": 10,
               "sha256": f"{i if i < 2 else 0:064x}"} for i in range(3)]
    identity = {"scene": scene, "settings": settings, "images": images}
    fp = s.signature(identity)
    z = rng.normal(size=(3, dimension)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z[2] = z[0]
    s.save_json(folder / "manifest.json", {**identity, "fingerprint": fp, "image_policy": "all_original_images"})
    summary = {"scene": scene, "fingerprint": fp, "complete": True, "image_count": 3,
               "duplicate_content_count": 1, "source": "sha256_verified_pinned_archive",
               "embedding_mean": z.astype(float).mean(axis=0).tolist(), "embedding_std": z.astype(float).std(axis=0).tolist()}
    s.save_json(folder / "summary.json", summary)
    s.save_npz(folder / "encodings.npz", names=np.asarray([r["name"] for r in images]), fingerprint=np.asarray(fp),
               embeddings=z, original_wh=np.asarray([[100, 200], [300, 200], [100, 200]]))
    return images


class SelectionTests(unittest.TestCase):
    def test_frozen_population_and_missing_labels(self):
        train, evaluation, metadata, sources, _ = s.load_inputs()
        self.assertEqual(len(train), 317)
        self.assertEqual(len(evaluation), 79)
        self.assertFalse(set(train) & set(evaluation))
        self.assertEqual(set(metadata), set(train))
        self.assertEqual(set(sources), set(train))
        self.assertEqual(sum(r["anchor"] == "__MISSING__" for r in metadata.values()), 3)
        self.assertFalse(any("eligible" in key for row in metadata.values() for key in row))

    def test_incomplete_or_contaminated_progress_rejected(self):
        train, evaluation, *_ = s.load_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            p = {"requested": train, "completed": train[:-1], "failures": [], "holdout_encoded": False}
            s.save_json(path, p)
            with self.assertRaisesRegex(RuntimeError, "316/317"):
                s.check_completion(tmp, train, evaluation)
            p.update(completed=train, all_317_training_landmarks_complete=True)
            s.save_json(path, p)
            self.assertEqual(s.check_completion(tmp, train, evaluation), p)
            p["completed"] = train[:-1] + evaluation[:1]
            s.save_json(path, p)
            with self.assertRaises(RuntimeError):
                s.check_completion(tmp, train, evaluation)

    def test_all_images_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            images = encoded_fixture(folder, "scene", {"test": True}, np.random.default_rng(2))
            mean, std, collection, fp = s.validate_landmark(folder, "scene", {"test": True})
            self.assertEqual(mean.shape, (1024,))
            self.assertEqual(collection["image_count"], 3)
            self.assertAlmostEqual(collection["duplicate_fraction"], 1 / 3)
            manifest = s.read_json(folder / "manifest.json")
            self.assertEqual(manifest["images"], images)
            with np.load(folder / "encodings.npz", allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            arrays["names"] = arrays["names"][::-1]
            s.save_npz(folder / "encodings.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                s.validate_landmark(folder, "scene", {"test": True})
            arrays["names"] = arrays["names"][::-1]
            arrays["embeddings"][0, 0] = np.nan
            s.save_npz(folder / "encodings.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                s.validate_landmark(folder, "scene", {"test": True})

    def test_quotas_and_optimization_preserve_membership_rules(self):
        pop = small_population()
        for n in range(4, pop.n):
            q = pop.allocation(n)
            self.assertEqual(q.sum(), n)
            self.assertTrue((q >= 1).all())
            self.assertTrue((q <= pop.counts["building_type"]).all())
        quota = pop.allocation(9)
        start = pop.feasible_start(quota, np.random.default_rng(3))
        selected, metrics = pop.improve(start, np.random.default_rng(4), 128)
        np.testing.assert_equal(np.bincount(pop.codes["building_type"][selected]), quota)
        self.assertTrue(pop.region_complete(selected))
        self.assertTrue(pop.anchors_valid(selected))
        self.assertLessEqual(metrics["score"], pop.metrics(start)["score"])
        # Independently brute-force the rounding objective on this small type table.
        for n in range(3, 12):
            q = pop.allocation(n)
            ideal = n * pop.counts["building_type"] / pop.n
            feasible = [a for a in product(range(1, 9), repeat=3) if sum(a) == n]
            minimum = min(np.sum((np.asarray(a) - ideal) ** 2) for a in feasible)
            self.assertAlmostEqual(np.sum((q - ideal) ** 2), minimum)

    def test_coverage_excludes_selected_zero_distances_and_ks_matches(self):
        pop = small_population(6)
        positions = np.arange(6, dtype=float)
        pop.d = np.abs(positions[:, None] - positions[None, :])
        metrics = pop.metrics(np.array([0]))
        self.assertEqual(metrics["unselected_count"], 5)
        self.assertEqual(metrics["coverage_mean"], 3.)
        self.assertEqual(metrics["coverage_worst"], 5.)
        selected = np.array([0, 2, 5])
        for field in s.NUMERICS:
            expected = ks_2samp(pop.numeric[field], pop.numeric[field][selected]).statistic
            self.assertAlmostEqual(pop.metrics(selected)[field + "_ks"], expected)
        self.assertIsNone(pop.metrics(np.arange(pop.n))["coverage_mean"])

    def test_size_rule_bend_and_no_bend(self):
        curve = [{"size": n, "feasible": True, "metrics": {"coverage_mean": loss, "coverage_p95": loss}}
                 for n, loss in zip([10, 20, 30, 40, 50], [10, 4, 3, 2.8, 2.6])]
        self.assertEqual(s.choose_size(curve, 60)["size"], 20)
        for row, loss in zip(curve, [5, 4, 3, 2, 1]):
            row["metrics"] = {"coverage_mean": loss, "coverage_p95": loss}
        self.assertEqual(s.choose_size(curve, 60)["size"], 60)
        fitted = s.isotonic_decreasing([8, 4, 6, 2])
        np.testing.assert_equal(fitted, [8, 5, 5, 2])

    def test_thousand_random_subsets_and_reproducibility(self):
        config = s.SelectionConfig(curve_candidates=2, swap_attempts=12)
        pop = small_population(config=config)
        with tempfile.TemporaryDirectory() as tmp:
            a = s.study_size(pop, 9, Path(tmp) / "a")
            b = s.study_size(pop, 9, Path(tmp) / "b")
            self.assertEqual(a, b)
            with np.load(Path(tmp) / "a/random_subsets.npz", allow_pickle=False) as data:
                draws = data["selected_indices"]
                self.assertEqual(draws.shape, (1000, 9))
                for selected in draws:
                    self.assertEqual(len(np.unique(selected)), 9)
                    np.testing.assert_equal(np.bincount(pop.codes["building_type"][selected]), pop.allocation(9))
            audit = s.detailed_audit(pop, np.asarray(a["indices"]))
            self.assertEqual(len(audit["coverage_unselected_landmarks"]), pop.n - 9)
            self.assertIn("building_type × region", audit["distributions"])
            selected = np.asarray(a["indices"])
            reference = s.anchor_matched_reference(pop, selected, Path(tmp) / "a")
            self.assertEqual(reference["draws"], 1000)
            self.assertTrue(reference["all_anchor_constraints_passed"])
            with np.load(Path(tmp) / "a/anchor_matched_random_subsets.npz", allow_pickle=False) as data:
                draws = data["selected_indices"]
                for draw in draws:
                    self.assertTrue(pop.valid_selection(draw))
                    np.testing.assert_equal(np.bincount(pop.codes["anchor"][draw], minlength=5),
                                            np.bincount(pop.codes["anchor"][selected], minlength=5))

    def test_real_anchor_quotas_force_larger_size_and_preserve_singletons(self):
        train, _, metadata, *_ = s.load_inputs()
        base = small_population(317)
        rows = [{**row, **metadata[scene], "scene": scene} for row, scene in zip(base.rows, train)]
        rng = np.random.default_rng(33)
        pop = s.Population(rows, rng.normal(size=(317, 8)), np.abs(rng.normal(size=(317, 8))), s.SelectionConfig(anchor_count_slack=0))
        self.assertIsNone(pop.feasible_start(pop.allocation(40), rng))
        self.assertEqual(pop.minimum_feasible_size(), 120)
        start = pop.feasible_start(pop.allocation(120), rng)
        selected, _ = pop.improve(start, rng, 256)
        self.assertTrue(pop.valid_selection(selected))
        actual = np.bincount(pop.codes["anchor"][selected], minlength=len(pop.levels["anchor"]))
        self.assertTrue((actual > 0).all())
        self.assertTrue((np.abs(actual - 120 * pop.counts["anchor"] / 317) < 1).all())
        for label in ("dome", "minaret", "__MISSING__", "altar"):
            self.assertGreater(actual[pop.levels["anchor"].index(label)], 0)
        altar = next(i for i in selected if rows[i]["anchor"] == "altar")
        replacement = next(i for i in range(pop.n) if i not in selected and rows[i]["building_type"] == rows[altar]["building_type"])
        broken = selected.copy()
        broken[np.flatnonzero(broken == altar)[0]] = replacement
        self.assertFalse(pop.valid_selection(broken))
        with self.assertRaisesRegex(ValueError, "anchor-feasible"):
            pop.improve(broken, rng, 16)

    def test_integer_anchor_bounds_at_exact_proportions(self):
        pop = small_population(24, s.SelectionConfig(anchor_count_slack=0))
        low, high = pop.anchor_bounds(24)
        np.testing.assert_equal(low, pop.counts["anchor"])
        np.testing.assert_equal(high, pop.counts["anchor"])
        self.assertTrue(pop.anchors_valid(np.arange(24)))
        low, high = pop.anchor_bounds(1)
        self.assertTrue((low == 1).all())
        self.assertGreater(low.sum(), 1)

    def test_relaxed_real_anchor_counts_preserve_rare_labels(self):
        train, _, metadata, *_ = s.load_inputs()
        base = small_population(317)
        rows = [{**row, **metadata[scene], "scene": scene} for row, scene in zip(base.rows, train)]
        rng = np.random.default_rng(34)
        pop = s.Population(rows, rng.normal(size=(317, 8)), np.abs(rng.normal(size=(317, 8))), s.SelectionConfig())
        self.assertEqual(pop.config.anchor_count_slack, 1)
        self.assertEqual(pop.minimum_feasible_size(), 48)
        self.assertIsNone(pop.feasible_start(pop.allocation(47), rng))
        start = pop.feasible_start(pop.allocation(48), rng)
        selected, _ = pop.improve(start, rng, 256)
        self.assertTrue(pop.valid_selection(selected))
        audit = s.detailed_audit(pop, selected)
        self.assertEqual(audit['omitted_levels']['anchor'], [])
        self.assertEqual(audit['anchor_constraints']['anchor_count_slack'], 1)
        self.assertTrue(audit['anchor_constraints']['all_passed'])
        # At 48 landmarks, the arch target is 14.69: allow 13..16.
        arch = next(r for r in audit['anchor_constraints']['counts'] if r['anchor'] == 'arch')
        self.assertEqual((arch['minimum'], arch['maximum']), (13, 16))
        for label in ('dome', 'minaret', 'altar', '__MISSING__'):
            row = next(r for r in audit['anchor_constraints']['counts'] if r['anchor'] == label)
            self.assertGreaterEqual(row['selected_count'], 1)
        # Even the relaxed rule must reject replacing the sole altar landmark.
        altar = next(i for i in selected if rows[i]['anchor'] == 'altar')
        replacement = next(i for i in range(pop.n) if i not in selected and rows[i]['building_type'] == rows[altar]['building_type'])
        broken = selected.copy()
        broken[np.flatnonzero(broken == altar)[0]] = replacement
        self.assertFalse(pop.valid_selection(broken))
        for invalid in (-1, .5, True, '1'):
            with self.subTest(slack=invalid), self.assertRaisesRegex(ValueError, 'anchor_count_slack'):
                s.SelectionConfig(anchor_count_slack=invalid)

    def test_full_pipeline_and_resume_with_frozen_317_split(self):
        train, evaluation, metadata, sources, checks = s.load_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enc = root / "encodings"
            enc.mkdir()
            settings = {"dataset_revision": "5d4404323a21aa3de948f8bc57a053b29fa3d921", "input_hashes": checks}
            s.save_json(enc / "settings.json", settings)
            s.save_json(enc / "progress.json", {"requested": train, "completed": train, "failures": [],
                        "all_317_training_landmarks_complete": True, "holdout_encoded": False})
            rng = np.random.default_rng(17)
            for scene in train:
                encoded_fixture(enc / "landmarks" / scene, scene, settings, rng)
            # Deliberately invalid evaluation directory: no code should inspect it.
            (enc / "landmarks" / evaluation[0]).mkdir()
            (enc / "landmarks" / evaluation[0] / "manifest.json").write_text("INVALID HOLDOUT DATA")
            config = s.SelectionConfig(curve_points=3, refinement_points=3, curve_candidates=2,
                                       final_candidates=3, swap_attempts=8)
            with patch.object(s, "emit"):
                result = s.run_selection(enc, root / "selection", config)
                repeated = s.run_selection(enc, root / "selection", config)
            self.assertEqual(result, repeated)
            output = Path(result["output_dir"])
            selected = s.read_json(output / "selected_landmarks.json")
            self.assertTrue(set(selected) <= set(train))
            self.assertFalse(set(selected) & set(evaluation))
            self.assertEqual((output / "evaluation_landmarks.json").read_bytes(), (s.HERE / "inputs/evaluation_landmarks.json").read_bytes())
            for scene in selected:
                self.assertEqual((output / "selected_image_manifests" / f"{scene}.json").read_bytes(),
                                 (enc / "landmarks" / scene / "manifest.json").read_bytes())
            self.assertEqual(s.read_json(output / "subset_manifest.json")["image_count"], len(selected) * 3)
            self.assertTrue((output / "coverage_curve.png").stat().st_size > 1000)
            self.assertTrue((output / "REPORT.md").is_file())
            self.assertFalse(result["interventions_run"])
            self.assertTrue(result["all_anchor_constraints_passed"])
            self.assertEqual(result["minimum_feasible_size"], 48)
            audit = s.read_json(output / "audit.json")
            self.assertTrue(audit["anchor_constraints"]["all_passed"])
            self.assertEqual(audit["anchor_constraints"]["anchor_count_slack"], 1)
            self.assertEqual(s.read_json(output / "selection_protocol.json")["config"]["anchor_count_slack"], 1)
            self.assertEqual(audit["omitted_levels"]["anchor"], [])
            self.assertTrue((output / "anchor_counts.csv").is_file())
            self.assertFalse((root / "selection/.selector.lock").exists())
            # A missing exported image manifest must be repaired, not treated as complete.
            missing = output / "selected_image_manifests" / f"{selected[0]}.json"
            missing.unlink()
            with patch.object(s, "emit"):
                repaired = s.run_selection(enc, root / "selection", config)
            self.assertEqual(repaired, result)
            self.assertTrue(missing.is_file())
            # Optional retained plots are explicitly labeled as synthetic test outputs.
            import os
            if os.environ.get("POINT3_SELECTION_TEST_ARTIFACTS"):
                target = Path(os.environ["POINT3_SELECTION_TEST_ARTIFACTS"])
                target.mkdir(parents=True, exist_ok=True)
                for name in ("coverage_curve.png", "building_type_balance.png"):
                    shutil.copyfile(output / name, target / ("synthetic_" + name))


if __name__ == "__main__":
    unittest.main()
