"""CPU regressions for lossless reuse and the fast/reference equivalence gate."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from fast_eval import (baseline_identity, validate_baseline, compare_results,
                       cached_ground_truth, install, attach_baseline_audit)


class FastEvalTests(unittest.TestCase):
    def test_baseline_cache_rejects_different_images_or_seed(self):
        job = dict(protocol="p", source_sha256="s", revision="r", num_views=50, seed=1729,
                   scene="scene", anchor="arch", views=["a.jpg"], torch_version="t", gpu="g", condition="scalar")
        identity = baseline_identity(job)
        cache = {"identity": identity, "audit": {"status": "ok", "prediction_sha256": {"x": "a"}}}
        validate_baseline(cache, baseline_identity({**job, "condition": "view_relative"}))
        for change in [{"seed": 1}, {"views": ["b.jpg"]}, {"gpu": "different"}]:
            with self.assertRaises(ValueError):
                validate_baseline(cache, baseline_identity({**job, **change}))

    def test_equivalence_gate_checks_both_arms_and_predictions(self):
        metrics = {arm: {"Acc-mean": .2, "Comp-mean": .3} for arm in ("base", "inter")}
        audit = {"prediction_sha256": {"depth_map": "abc"}}
        compare_results(metrics, copy.deepcopy(metrics), audit, audit)
        for arm in ("base", "inter"):
            changed = copy.deepcopy(metrics)
            changed[arm]["Acc-mean"] += .01
            with self.assertRaises(ValueError):
                compare_results(metrics, changed, audit, audit)
        with self.assertRaises(ValueError):
            compare_results(metrics, metrics, audit, {"prediction_sha256": {"depth_map": "changed"}})

    def test_ground_truth_cache_is_lossless_and_remaps_staged_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = str(root / "000_a.jpg")
            points = np.arange(18, dtype=np.float64).reshape(6, 3) / 7
            depth = np.arange(6, dtype=np.float32).reshape(2, 3)
            masks = {image: np.ones((2, 3), dtype=bool)}
            calls = []

            def original(image_paths, image_folder, depth_folders, recon_path, output_glb_path,
                         validity_masks_by_path=None):
                calls.append(True)
                return points, [{"image_path": "old/staging/000_a.jpg", "depth_map": depth,
                                 "K": np.eye(3), "validity_mask": masks[image]}]

            wrapped = cached_ground_truth(original, root / "cache")
            args = [[image], str(root), [str(root / "depth")], str(root / "sparse"), "gt.glb"]
            first = wrapped(*args, validity_masks_by_path=masks)
            second = wrapped(*args, validity_masks_by_path=masks)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1][0]["depth_map"], second[1][0]["depth_map"])
            self.assertEqual(second[1][0]["image_path"], image)
            self.assertEqual(second[0].dtype, points.dtype)
            self.assertEqual(second[1][0]["depth_map"].dtype, depth.dtype)
            # ICP's native bindings reject read-only buffers. Mutating a loaded
            # buffer must still leave both the original arrays and disk intact.
            self.assertTrue(second[0].flags.writeable)
            self.assertTrue(second[1][0]["depth_map"].flags.writeable)
            point_bytes = (root / "cache/points.npy").read_bytes()
            depth_bytes = (root / "cache/sample_0_depth_map.npy").read_bytes()
            second[0][0, 0] = -123
            second[1][0]["depth_map"][0, 0] = -456
            second[0].flush()
            second[1][0]["depth_map"].flush()
            self.assertEqual((root / "cache/points.npy").read_bytes(), point_bytes)
            self.assertEqual((root / "cache/sample_0_depth_map.npy").read_bytes(), depth_bytes)
            np.testing.assert_array_equal(np.load(root / "cache/points.npy"), points)
            np.testing.assert_array_equal(np.load(root / "cache/sample_0_depth_map.npy"), depth)
            with self.assertRaises(ValueError):
                wrapped(*args, validity_masks_by_path={image: np.zeros((2, 3), dtype=bool)})
            # Windows holds mapped files open; workers exit before scratch cleanup in Colab.
            second[0]._mmap.close()
            for value in second[1][0].values():
                if isinstance(value, np.memmap):
                    value._mmap.close()

    def test_visual_suppression_does_not_replace_metric_functions(self):
        def metric():
            return "untouched"
        def original(image_paths, image_folder, depth_folders, recon_path, output_glb_path,
                     validity_masks_by_path=None):
            return None
        module = SimpleNamespace(generate_gt_by_unproject=original, _run_alignment_and_eval_worker=metric)
        with tempfile.TemporaryDirectory() as tmp:
            install(module, tmp)
            self.assertIs(module._run_alignment_and_eval_worker, metric)
            self.assertIsNone(module.save_prediction_glb())
            self.assertIsNone(module.pointcloud_to_glb())

    def test_reused_audit_is_labeled_and_only_written_after_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cache = {"identity": {"scene": "scene"}, "audit": {"status": "ok", "prediction_sha256": {"x": "a"}}}
            module = SimpleNamespace(main=lambda **kwargs: "model")
            attach_baseline_audit(module, cache)
            self.assertEqual(module.main(work_dir=folder), "model")
            self.assertFalse((folder / "base_evaluation.json").exists())
            (folder / "inter_evaluation.json").write_text('{}')
            module.main(work_dir=folder)
            result = json.loads((folder / "base_evaluation.json").read_text())
            self.assertEqual(result["evaluation_source"], "reused_validated_zero_control")
            self.assertEqual(result["prediction_sha256"], cache["audit"]["prediction_sha256"])


if __name__ == "__main__":
    unittest.main()
