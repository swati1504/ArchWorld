"""Protocol leakage, deadline, and orchestration checks; no GPU claims."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile
import torch
import hypothesis_study
from full_dataset import METRICS
from paired_eval import PROTOCOL

from hypothesis_protocols import protocol, variants, select_settings, job_count
from hypothesis_study import (validate_config, bounded_run, BudgetExceeded, Study,
                              export_results, write_json)

ROOT = Path(__file__).resolve().parents[2]


class HypothesisTests(unittest.TestCase):
    def test_real_job_checkpoint_reference_gate_and_resume_with_fake_model(self):
        # Exercise actual runner/checkpoint logic, with only GPU work and scene prep replaced.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'results'
            output.mkdir()
            for name in ('evaluation', 'sam'):
                write_json(output / f'{name}_packages.json', [['fixture', '1']])
            calls = []
            def prepare(study, row):
                scratch = study.work / row['scene']
                scratch.mkdir(exist_ok=True)
                study.prepared[row['scene']] = scratch
                study.preparation_times.append(.001)
                write_json(study.saved(row) / 'normalization.json', {'factor': 1})
                return scratch
            def fake_run(command, log, source, env, deadline):
                command = [str(x) for x in command]
                val = lambda key: command[command.index(key)+1]
                self.assertEqual(val('--num-images'), '50')
                scene, eta = val('--scene'), val('--eta')
                definition = json.loads(Path(val('--strength-config')).read_text())
                reference = env.get('POINT3_FAST_MODE') != '1'
                calls.append((scene, eta, definition['mode'], reference))
                if eta != '0' and not reference:
                    self.assertTrue(Path(env['POINT3_BASELINE_CACHE']).exists())
                folder = Path(val('--out')) / f'output_{eta}' / scene
                folder.mkdir(parents=True)
                cost = 2 if eta == '0' else 1.5 if definition['mode']=='layer_relative' else 1.8
                metrics = {'base': {'Acc-mean': 2., 'Comp-mean': 2.},
                           'inter': {'Acc-mean': cost, 'Comp-mean': cost}}
                audit = {'status':'ok','protocol':PROTOCOL,'prediction_sha256':{'depth_map':'same'}}
                trace = {'num_views':50,'layers':[{'layer':i,'accepted':False,'anchor_tokens_by_view':[1]*50}
                                                for i in range(24)]}
                for name, data in ((METRICS,metrics),('base_evaluation.json',audit),
                                   ('inter_evaluation.json',audit),('strength_trace.json',trace)):
                    write_json(folder/name,data)
                Path(log).write_text('PROBE_RESULT '+json.dumps({'base_gb':2.,'inter_gb':1.})+'\n')
                return .01
            with patch.object(torch.cuda,'is_available',return_value=True), \
                 patch.object(torch.cuda,'get_device_name',return_value='Fake GPU'), \
                 patch.object(Study,'prepare',prepare), \
                 patch.object(hypothesis_study,'bounded_run',side_effect=fake_run):
                def new_study():
                    return Study(protocol('H1'),ROOT,root/'work',output,Path(sys.executable),time.monotonic()+100)
                study=new_study()
                study.run()
                self.assertEqual(len(calls),17)
                self.assertEqual(json.loads((output/'reference_check.json').read_text())['status'],'passed')
                self.assertEqual(len(list(output.glob('scenes/*/*.zip'))),17)
                new_study().run()
                self.assertEqual(len(calls),17)  # No repeated inference after validated resume.
                saved=next(output.glob('scenes/*/scalar_eta100.zip'))
                with zipfile.ZipFile(saved) as handle:
                    self.assertEqual(json.loads(handle.read('job.json'))['num_views'],50)

    def test_exact_50_disjoint_roles_and_counts(self):
        calibration, evaluation = set(), set()
        for h, n in (("H1", 17), ("H2", 17), ("H3", 19)):
            config = protocol(h)
            rows = validate_config(config, ROOT)
            self.assertEqual(job_count(config), n)
            for row in rows:
                self.assertEqual(len(set(row["views"])), 50)
                (calibration if row["split"] == "calibration" else evaluation).add(row["scene"])
        self.assertFalse(calibration & evaluation)
        bad = protocol("H1")
        bad["num_views"] = 5
        with self.assertRaises(ValueError):
            validate_config(bad, ROOT)

    def test_anchor_selection_and_no_test_tuning(self):
        config = protocol("H3")
        scores = {r["scene"]: {"scalar_eta100": 1. if r["anchor"] == "dome" else 3.,
                              "scalar_eta1000": 3. if r["anchor"] == "dome" else 1.}
                  for r in config["scenes"] if r["split"] == "calibration"}
        selected = select_settings(config, scores)
        self.assertEqual(selected["global_scalar"], "scalar_eta100")
        self.assertEqual(selected["anchor_map"], {"dome": "scalar_eta100", "minaret": "scalar_eta1000"})
        with self.assertRaises(ValueError):
            select_settings(config, {**scores, "test_scene": {"scalar_eta100": 0}})
        partial = copy.deepcopy(scores)
        partial.pop(next(iter(partial)))
        with self.assertRaises(ValueError):
            select_settings(config, partial)
        config["min_calibration_per_anchor"] = 3
        fallback = select_settings(config, scores)
        self.assertEqual(set(fallback["fallback_types"]), {"dome", "minaret"})

    def test_deadline_terminates_and_redacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.txt"
            env = {**os.environ, "HF_TOKEN": "hf_fakeTestSecret"}
            with self.assertRaises(BudgetExceeded):
                bounded_run([sys.executable, "-u", "-c",
                             "import os,time;print(os.environ['HF_TOKEN']);time.sleep(30)"],
                            log, ROOT, env, time.monotonic()+1.1)
            self.assertNotIn(env["HF_TOKEN"], log.read_text())
            self.assertIn("[HF_TOKEN hidden]", log.read_text())

    def test_orchestration_freezes_before_evaluation_and_labels_dedup(self):
        # A fake model tests sequencing. Its numbers are deliberately not empirical evidence.
        with tempfile.TemporaryDirectory() as tmp:
            study = Study.__new__(Study)
            study.config = protocol("H3")
            study.rows = validate_config(study.config, ROOT)
            study.output = Path(tmp)
            study.records, study.scores, study.selection = [], {}, None
            study.eval_scenes = [r["scene"] for r in study.rows if r["split"] == "evaluation"]
            jobs = set()
            def record(row, candidate, policy):
                if row["split"] == "evaluation":
                    self.assertTrue((study.output / "selection.json").exists())
                    self.assertIsNotNone(study.selection)
                jobs.add((row["scene"], candidate))
                cost = 2 if candidate == "no_intervention" else (
                    1 if (candidate == "scalar_eta100") == (row["anchor"] == "dome") else 3)
                study.records.append(dict(scene=row["scene"], split=row["split"], anchor=row["anchor"],
                    candidate=candidate, policy=policy, CD_norm_base=2, CD_norm_inter=cost, seconds=1))
                return cost
            study.record = record
            study.reference_check = lambda *args: None
            study.forecast = lambda: None
            study.run()
            self.assertEqual(len(jobs), 17)  # One duplicate evaluation policy and separate reference omitted.
            self.assertEqual(len(study.records), 18)
            self.assertTrue((study.output / "comparison.csv").exists())
            partial = study.records[:-1]
            self.assertFalse(export_results(study.output, partial, study.selection, study.eval_scenes))
            self.assertEqual(len((study.output / "comparison.csv").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
