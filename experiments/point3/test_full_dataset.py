"""CPU checks for paired evaluation, safe extraction and resumable checkpoints."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from full_dataset import (PROTOCOL, METRICS, inventory, extract_selected, validate_result,
                          save_checkpoint, read_checkpoint, clear_owned_scratch, sanitize,
                          preflight_identity, check_preflight, CONDITIONS)
from paired_eval import paired_worker
import full_dataset


class FullDatasetTests(unittest.TestCase):
    def test_inventory_never_silently_drops_scenes(self):
        rows = inventory([{"scene": s} for s in ["good", "missing", "short"]],
                         {"good": ["a", "b"], "short": ["a"]},
                         {"good": "arch", "short": "dome"}, 2)
        self.assertEqual([r["status"] for r in rows], ["ready", "blocked", "blocked"])
        self.assertEqual(len(rows), 3)

    def test_fifty_views_are_strict_and_never_padded(self):
        views = {"full": [str(i) for i in range(50)], "short": [str(i) for i in range(49)],
                 "duplicate": [str(i) for i in range(49)] + ["0"]}
        rows = inventory([{"scene": s} for s in views], views,
                         {s: "arch" for s in views}, 50)
        self.assertEqual([r["status"] for r in rows], ["ready", "blocked", "blocked"])
        self.assertEqual(len(rows[1]["views"]), 49)
        self.assertEqual(rows[1]["requested_views"], 50)

    def test_preflight_required_for_matching_gpu_and_all_conditions(self):
        manifest = dict(source_sha256="abc", revision="def", num_views=50, seed=1729, eta=100, alpha=.02)
        identity = preflight_identity(manifest, "GPU", "torch")
        record = {"identity": identity, "status": "passed", "completed_conditions": list(CONDITIONS),
                  "reference_equivalence": {"status": "passed"}}
        check_preflight(record, identity)
        for invalid in [{}, {**record, "status": "running"}, {**record, "completed_conditions": ["no_intervention"]},
                        {**record, "reference_equivalence": {"status": "failed"}}]:
            with self.assertRaises(RuntimeError):
                check_preflight(invalid, identity)
        with self.assertRaises(RuntimeError):
            check_preflight(record, {**identity, "num_views": 5})
        with self.assertRaises(RuntimeError):
            check_preflight(record, {**identity, "gpu": "different GPU"})

    def test_selected_extraction_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "data.tar.gz"
            names = ["archworld/scene/images/a.jpg", "archworld/scene/images/b.jpg",
                     "archworld/scene/depth/a.jpg.npy", "archworld/scene/depth/b.jpg.npy",
                     "archworld/scene/sparse/0/images.bin", "manual_filtration_masks/scene/a.jpg.npy"]
            with tarfile.open(archive, "w:gz") as handle:
                for name in names:
                    item = tarfile.TarInfo(name)
                    item.size = 4
                    handle.addfile(item, io.BytesIO(b"data"))
            dest = root / "extract"
            dest.mkdir()
            extract_selected(archive, dest, "scene", ["a.jpg"])
            self.assertTrue((dest / names[0]).exists())
            self.assertTrue((dest / names[2]).exists())
            self.assertTrue((dest / names[4]).exists())
            self.assertTrue((dest / names[5]).exists())
            self.assertFalse((dest / names[1]).exists())
            self.assertFalse((dest / names[3]).exists())
            with tarfile.open(archive, "w:gz") as handle:
                handle.addfile(tarfile.TarInfo("../escape"))
            with self.assertRaises(ValueError):
                extract_selected(archive, dest, "scene", ["a.jpg"])

    def fixture(self):
        metrics = {arm: {"Acc-mean": .2, "Comp-mean": .4} for arm in ["base", "inter"]}
        trace = {"num_views": 5, "layers": [{"layer": n, "accepted": False,
                 "anchor_tokens_by_view": [1] * 5} for n in range(24)]}
        audit = {"protocol": PROTOCOL, "status": "ok", "prediction_sha256": {"depth": "abc"}}
        return metrics, trace, audit

    def test_failed_controls_and_partial_results_rejected(self):
        metrics, trace, audit = self.fixture()
        validate_result(metrics, trace, audit, audit, "no_intervention", 5)
        metrics["inter"]["Acc-mean"] += .001
        with self.assertRaisesRegex(ValueError, "changed metrics"):
            validate_result(metrics, trace, audit, audit, "no_intervention", 5)
        with self.assertRaisesRegex(ValueError, "changed predictions"):
            validate_result(metrics, trace, audit, {**audit, "prediction_sha256": {}}, "no_intervention", 5)
        trace["layers"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete strength trace"):
            validate_result(metrics, trace, audit, audit, "scalar", 5)

    def test_checkpoint_verified_and_inputs_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "job"
            folder.mkdir()
            metrics, trace, audit = self.fixture()
            for name, data in [(METRICS, metrics), ("strength_trace.json", trace),
                               ("base_evaluation.json", audit), ("inter_evaluation.json", audit),
                               ("timing.json", {"elapsed_seconds": 1})]:
                (folder / name).write_text(json.dumps(data))
            (folder / "large.glb").write_bytes(b"not saved by default")
            job = {"condition": "no_intervention", "num_views": 5}
            path = root / "saved.zip"
            result = save_checkpoint(folder, path, job, False)
            self.assertAlmostEqual(result["CD_inter"], .3)
            with zipfile.ZipFile(path) as handle:
                self.assertNotIn("large.glb", handle.namelist())
            self.assertEqual(read_checkpoint(path, job), result)
            with self.assertRaisesRegex(ValueError, "inputs differ"):
                read_checkpoint(path, {**job, "num_views": 50})

    def test_paired_worker_repeats_sampling_and_restores_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction = {k: torch.ones(3) for k in ["depth_map", "depth_conf", "extrinsics", "intrinsics"]}
            original = lambda *a, **kw: ("ok", float(np.random.random()))
            worker = paired_worker(original, 42)
            np.random.seed(12)
            expected = np.random.random()
            np.random.seed(12)
            args = [None, None, prediction, [], tmp, "metrics.json", 0, 23, "base"]
            base = worker(*args)
            self.assertEqual(np.random.random(), expected)
            args[-1] = "inter"
            self.assertEqual(base, worker(*args))
            self.assertEqual(json.loads((Path(tmp) / "base_evaluation.json").read_text())["prediction_sha256"],
                             json.loads((Path(tmp) / "inter_evaluation.json").read_text())["prediction_sha256"])

    def test_cleanup_requires_owned_child_and_secrets_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "job"
            child.mkdir()
            with self.assertRaises(ValueError):
                clear_owned_scratch(child, root)
            (child / ".point3-owned").touch()
            with self.assertRaises(ValueError):
                clear_owned_scratch(root, root)
            clear_owned_scratch(child, root)
            self.assertFalse(child.exists())
        self.assertNotIn("hf_", sanitize("bad hf_exampletoken"))
        self.assertNotIn("secret", sanitize("bad secret", "secret"))

    def test_preflight_reference_check_and_resume_orchestration(self):
        # Exercise the complete runner without a GPU/network; model processes are fixtures.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "results"
            assets = source / "intervention/assets"
            assets.mkdir(parents=True)
            views = [f"{i}.jpg" for i in range(5)]
            (assets / "views.json").write_text(json.dumps({"scene": views}))
            (assets / "anchors.json").write_text('{"scene":"arch"}')
            metadata = root / "metadata.csv"
            metadata.write_text('scene,Scale,Extent\nscene,1,1\n')
            archive = root / "scene.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for name in views:
                    for member_name in [f"archworld/scene/images/{name}",
                                        f"archworld/scene/depth/{name}.npy",
                                        f"manual_filtration_masks/scene/{name}.npy"]:
                        member = tarfile.TarInfo(member_name)
                        member.size = 1
                        handle.addfile(member, io.BytesIO(b"x"))
            calls = []

            def fake_process(command, log_path, cwd, environment):
                command = [str(x) for x in command]
                if "-c" in command:
                    scene_scratch = next((root / "work/full_dataset_scratch").iterdir())
                    masks = scene_scratch / "masks/scene/mask"
                    masks.mkdir(parents=True)
                    for view in views:
                        np.save(masks / f"{Path(view).stem}_arch.npy", np.ones((2, 2), dtype=np.uint8))
                    Path(log_path).write_text('mask fixture')
                    return 1
                run_dir = Path(command[command.index("--out") + 1])
                eta = command[command.index("--eta") + 1]
                name = run_dir.name
                calls.append(name)
                if name not in ("no_intervention", "reference_scalar"):
                    cache = json.loads(Path(environment["POINT3_BASELINE_CACHE"]).read_text())
                    self.assertEqual(cache["metrics"]["Acc-mean"], .2)
                if name == "reference_scalar":
                    self.assertNotIn("POINT3_FAST_MODE", environment)
                folder = run_dir / f"output_{eta}/scene"
                folder.mkdir(parents=True)
                metrics, trace, audit = self.fixture()
                if name != "no_intervention":
                    metrics["inter"]["Acc-mean"] = .25
                for filename, data in [(METRICS, metrics), ("strength_trace.json", trace),
                                       ("base_evaluation.json", audit), ("inter_evaluation.json", audit)]:
                    (folder / filename).write_text(json.dumps(data))
                probe = {"base_gb": 2 if name in ("no_intervention", "reference_scalar") else None,
                         "inter_gb": 1}
                Path(log_path).write_text('PROBE_RESULT ' + json.dumps(probe) + '\n')
                return 20 if name == "reference_scalar" else 10

            args = ['full_dataset.py', '--source', str(source), '--work', str(root / 'work'),
                    '--output', str(output), '--sam-python', sys.executable, '--num-views', '5']
            from types import SimpleNamespace
            with patch('huggingface_hub.hf_hub_download', side_effect=lambda repo, name, **kw:
                       str(metadata if name.endswith('.csv') else archive)), \
                 patch.object(full_dataset, 'run_logged', side_effect=fake_process), \
                 patch.object(full_dataset.subprocess, 'run'), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'get_device_name', return_value='Test GPU'), \
                 patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=80*2**30)):
                with patch.object(sys, 'argv', args + ['--preflight-only']):
                    full_dataset.main()
                preflight = json.loads((output / 'gpu_preflight.json').read_text())
                self.assertEqual(preflight['status'], 'passed')
                self.assertEqual(preflight['reference_equivalence']['measured_scalar_speedup'], 2)
                self.assertEqual(calls, list(CONDITIONS) + ['reference_scalar'])
                with patch.object(sys, 'argv', args):
                    full_dataset.main()
                self.assertEqual(len(calls), 5)  # No repeated inference on resume.
                self.assertEqual(json.loads((output / 'progress.json').read_text())['completed_scenes'], 1)


if __name__ == "__main__":
    unittest.main()
