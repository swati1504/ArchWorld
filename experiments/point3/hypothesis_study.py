"""Bounded, resumable 50-view hypothesis studies. No test-set hyperparameter tuning."""
import argparse
from collections import deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import threading
import zipfile

from full_dataset import (METRICS, extract_selected, inventory, read_json, write_json,
                          sanitize, read_checkpoint, save_checkpoint, source_hash,
                          validate_result, clear_owned_scratch)
from paired_eval import PROTOCOL
from fast_eval import baseline_identity, compare_results
from hypothesis_protocols import variants, select_settings, job_count


class BudgetExceeded(RuntimeError):
    pass


def bounded_run(command, log, cwd, environment, deadline, echo=False, cancel_event=None):
    """Bound the entire child process tree; redact credentials before saving output."""
    remaining = deadline - time.monotonic()
    if remaining <= 1:
        raise BudgetExceeded("Automatic execution budget exhausted")
    if cancel_event is not None and cancel_event.is_set():
        raise BudgetExceeded("Execution cancelled")
    start = time.monotonic()
    child = subprocess.Popen([str(x) for x in command], cwd=cwd, env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             start_new_session=os.name != "nt")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=15)
    def consume():
        with Path(log).open("a", encoding="utf-8") as handle:
            handle.write("\n--- Attempt ---\n")
            for line in child.stdout:
                line = sanitize(line, environment.get("HF_TOKEN", ""))
                tail.append(line)
                handle.write(line)
                handle.flush()
                if echo:
                    print(line, end="", flush=True)
    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    try:
        if cancel_event is None:
            child.wait(timeout=remaining)
        else:
            while child.poll() is None:
                if cancel_event.is_set():raise BudgetExceeded("Execution cancelled")
                remaining=deadline-time.monotonic()
                if remaining<=0:raise BudgetExceeded("Automatic execution budget exhausted")
                try:child.wait(timeout=min(.25,remaining))
                except subprocess.TimeoutExpired:pass
    except BaseException:
        if child.poll() is None:
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            child.wait()
        if time.monotonic() >= deadline:
            raise BudgetExceeded("Stopped at the automatic execution deadline") from None
        raise
    finally:
        reader.join(timeout=5)
        if not reader.is_alive():
            child.stdout.close()
    if child.returncode:
        raise RuntimeError(f"Process exited {child.returncode}; {log}\n" +
                           "".join(tail))
    return time.monotonic() - start


def clean_environment(source, seed):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("INTERVENTION_", "POINT3_")) or key in {
            "PER_LAYER_INTERVENTION", "TOP_K_ELEMENTS", "SKIP_BASELINE_PASS", "SKIP_EVAL",
            "METRIC_EVAL_BASE", "STRUCTURED_STRENGTH_CONFIG", "SAVE_RAW_TENSORS",
            "ALIGNMENT_MAX_POINTS", "ALIGNMENT_N_ITERATIONS", "ICP_MAX_ITER"}:
            env.pop(key, None)
    env.update(PROBE="1", EVAL_TIMEOUT_SEC="0", POINT3_SOURCE=str(source),
               POINT3_EVAL_SEED=str(seed), PYTHONHASHSEED=str(seed), PYTHONIOENCODING="utf-8",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="0")
    return env


def study_hash(source):
    h = hashlib.sha256(source_hash(source).encode())
    for name in ("hypothesis_study.py", "hypothesis_protocols.py"):
        h.update((source / "experiments/point3" / name).read_bytes())
    return h.hexdigest()


def validate_config(config, source):
    if config["num_views"] != 50:
        raise ValueError("These notebooks require exactly 50 distinct images per landmark")
    if config["hypothesis"] not in {"H1", "H2", "H3"}:
        raise ValueError("Unknown hypothesis")
    if not config["eta_candidates"] or any(type(e) is not int or e <= 0 for e in config["eta_candidates"]):
        raise ValueError("Use a nonempty grid of positive integer eta values")
    if not config["alpha_candidates"] or any(not math.isfinite(a) or a <= 0 for a in config["alpha_candidates"]):
        raise ValueError("Use positive finite alpha candidates")
    if not 1 <= config["runtime_minutes"] <= 55 or config["forecast_margin"] < 1:
        raise ValueError("Execution budget must be at most 55 minutes; forecast margin at least 1")
    rows = config["scenes"]
    if len({r["scene"] for r in rows}) != len(rows):
        raise ValueError("Calibration and evaluation must be distinct landmarks")
    if {r["split"] for r in rows} != {"calibration", "evaluation"}:
        raise ValueError("Both calibration and evaluation landmarks are required")
    anchors = read_json(source / "intervention/assets/anchors.json")
    views = read_json(source / "intervention/assets/views.json")
    checked = inventory(rows, views, anchors, 50)
    for supplied, row in zip(rows, checked):
        if row["status"] != "ready" or supplied["anchor"] != row["anchor"]:
            raise ValueError(f"Invalid supplied landmark/anchor/views: {supplied['scene']}")
    return [{**r, "views": views[r["scene"]][:50]} for r in rows]


def prepare_scene(request):
    """Isolated worker: downloads/extraction and mask validation are time bounded by parent."""
    from huggingface_hub import hf_hub_download
    import numpy as np
    row, root = request["row"], Path(request["scratch"])
    scene, anchor = row["scene"], row["anchor"]
    data, masks, saved = root / "data", root / "masks", Path(request["saved"])
    data.mkdir(parents=True, exist_ok=True)
    metadata = hf_hub_download("ArchWorld/ArchWorld", "ArchWorld_metadata.csv", repo_type="dataset",
                               revision=request["revision"], token=False)
    shutil.copy2(metadata, data / "ArchWorld_metadata.csv")
    marker = root / "extracted.json"
    identity = {"scene": scene, "views": row["views"], "revision": request["revision"]}
    if not marker.exists() or read_json(marker) != identity:
        if shutil.disk_usage(root).free < 16 * 2**30:
            raise OSError("Need 16 GiB free disk before extracting a new landmark")
        archive = hf_hub_download("ArchWorld/ArchWorld", f"data/{scene}.tar.gz", repo_type="dataset",
                                  revision=request["revision"], token=False)
        extract_selected(archive, data, scene, row["views"])
        write_json(marker, identity)
    if (saved / "masks.zip").exists():
        with zipfile.ZipFile(saved / "masks.zip") as handle:
            if handle.testzip() is not None:
                raise ValueError("Corrupt mask checkpoint")
            for member in handle.infolist():
                if not (masks / member.filename).resolve().is_relative_to(masks.resolve()):
                    raise ValueError("Unsafe mask checkpoint")
            handle.extractall(masks)
    for view in row["views"]:
        if not (data / "archworld" / scene / "images" / view).is_file():
            raise ValueError(f"Missing selected photograph: {view}")
        valid = data / "manual_filtration_masks" / scene
        if not any((valid / (name + ".npy")).is_file() for name in (Path(view).name, Path(view).stem)):
            raise ValueError(f"Missing landmark validity mask: {view}")
    # SAM is run separately in its own environment by the parent.
    mask_files = [masks / scene / "mask" / f"{Path(v).stem}_{anchor}.npy" for v in row["views"]]
    if not all(p.is_file() for p in mask_files):
        write_json(root / "mask_status.json", {"status": "needs_segmentation"})
        return
    nonempty = sum(bool((a := np.load(p, allow_pickle=False)).size and a.any()) for p in mask_files)
    if nonempty < 2:
        raise ValueError(f"Anchor visible in only {nonempty}/50 views; no valid cross-view study")
    write_json(root / "mask_status.json", {"status": "ready", "nonempty_views": nonempty, "views": 50})
    temporary = saved / "masks.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in mask_files:
            handle.write(path, path.relative_to(masks).as_posix())
    with zipfile.ZipFile(temporary) as handle:
        if handle.testzip() is not None:
            raise ValueError("Mask archive failed verification")
    temporary.replace(saved / "masks.zip")
    shutil.copy2(root / "mask_status.json", saved / "mask_status.json")


def csv_write(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(output, records, selection, expected_eval):
    fields = ["scene", "split", "anchor", "policy", "candidate", "CD_norm_base", "CD_norm_inter", "seconds"]
    csv_write(output / "results.csv", records, fields)
    csv_write(output / "calibration_scores.csv", [r for r in records if r["split"] == "calibration"], fields)
    evaluation = [r for r in records if r["split"] == "evaluation"]
    complete = all({r["policy"] for r in evaluation if r["scene"] == s} ==
                   {"no_intervention", "global_scalar", "structured"} for s in expected_eval)
    comparison = []
    if complete and selection:
        for scene in expected_eval:
            group = {r["policy"]: r for r in evaluation if r["scene"] == scene}
            baseline = group["no_intervention"]["CD_norm_inter"]
            scalar = group["global_scalar"]["CD_norm_inter"]
            structured = group["structured"]["CD_norm_inter"]
            comparison.append(dict(scene=scene, anchor=group["structured"]["anchor"],
                CD_norm_baseline=baseline, CD_norm_scalar=scalar, CD_norm_structured=structured,
                structured_minus_scalar=structured-scalar, structured_minus_baseline=structured-baseline,
                same_candidate=group["structured"]["candidate"] == group["global_scalar"]["candidate"]))
    csv_write(output / "comparison.csv", comparison, ["scene", "anchor", "CD_norm_baseline", "CD_norm_scalar",
              "CD_norm_structured", "structured_minus_scalar", "structured_minus_baseline", "same_candidate"])
    return complete


class Study:
    def __init__(self, config, source, work, output, sam_python, deadline):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime; the 50-image memory check has not yet passed")
        self.config, self.source, self.output = config, source, output
        self.rows = validate_config(config, source)
        self.sam_python, self.deadline = sam_python, deadline
        self.env = clean_environment(source, config["seed"])
        self.identity = {"protocol": PROTOCOL, "source_sha256": study_hash(source),
            "revision": config["dataset_revision"], "num_views": 50, "seed": config["seed"],
            "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
        self.identity["environment_sha256"] = hashlib.sha256(b"".join(
            (output / f"{name}_packages.json").read_bytes() for name in ("evaluation", "sam"))).hexdigest()
        manifest = {"identity": self.identity, "config": config}
        output.mkdir(parents=True, exist_ok=True)
        if (output / "manifest.json").exists() and read_json(output / "manifest.json") != manifest:
            raise ValueError("Run inputs or GPU changed. Use a new RUN_LABEL; do not mix results")
        write_json(output / "manifest.json", manifest)
        self.work = work / hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
        self.work.mkdir(parents=True, exist_ok=True)
        self.candidates = {"no_intervention": {"eta": 0, "config": {"mode": "fixed"}}, **variants(config)}
        self.records, self.scores, self.selection, self.prepared = [], {}, None, {}
        self.new_job_times, self.preparation_times = [], []
        self.eval_scenes = [r["scene"] for r in self.rows if r["split"] == "evaluation"]

    def saved(self, row):
        path = self.output / "scenes" / row["scene"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepare(self, row):
        scene = row["scene"]
        if scene in self.prepared:
            return self.prepared[scene]
        start = time.monotonic()
        scratch = self.work / hashlib.sha256(scene.encode()).hexdigest()[:12]
        scratch.mkdir(exist_ok=True)
        request = {"row": row, "scratch": str(scratch), "saved": str(self.saved(row)),
                   "revision": self.config["dataset_revision"]}
        request_file = scratch / "prepare_request.json"
        write_json(request_file, request)
        command = [sys.executable, "-u", Path(__file__), "--prepare-scene", request_file]
        bounded_run(command, self.saved(row) / "prepare.log", self.source, self.env, self.deadline)
        if read_json(scratch / "mask_status.json")["status"] != "ready":
            script = (f"import sys;sys.path.insert(0,{str(self.source)!r});"
                "from intervention import make_masks as m;"
                "from sam3.model_builder import build_sam3_image_model;"
                "from sam3.model.sam3_image_processor import Sam3Processor;"
                f"m.views_for=lambda s:{row['views']!r};"
                "p=Sam3Processor(build_sam3_image_model(device='cuda'),confidence_threshold=m.CONFIDENCE);"
                f"m.segment_scene(p,{str(scratch / 'data')!r},{scene!r},{str(scratch / 'masks')!r},False,True)")
            bounded_run([self.sam_python, "-u", "-c", script], self.saved(row) / "masks.log",
                        self.source, self.env, self.deadline)
            bounded_run(command, self.saved(row) / "prepare.log", self.source, self.env, self.deadline)
        with (scratch / "data/ArchWorld_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
            meta = next(r for r in csv.DictReader(handle) if r["scene"] == scene)
        factor = float(meta["Scale"]) * 1000 / float(meta["Extent"])
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid scene metric normalization")
        write_json(self.saved(row) / "normalization.json", {"factor": factor, "metadata": meta})
        self.prepared[scene] = scratch
        self.preparation_times.append(time.monotonic() - start)
        return scratch

    def job(self, row, candidate, reference=False):
        definition = self.candidates[candidate]
        label = "reference_" + candidate if reference else candidate
        job = {**self.identity, "scene": row["scene"], "anchor": row["anchor"], "views": row["views"],
               "condition": candidate, "definition": definition, "reference": reference}
        saved = self.saved(row)
        checkpoint = saved / f"{label}.zip"
        result = read_checkpoint(checkpoint, job)
        if result is not None:
            return result
        scratch = self.prepare(row)
        parent = scratch / "jobs"
        parent.mkdir(exist_ok=True)
        run_dir = parent / f"{label}_{time.time_ns()}"
        run_dir.mkdir()
        (run_dir / ".point3-owned").touch()
        config_path = run_dir / "strength_config.json"
        write_json(config_path, definition["config"])
        env = self.env.copy()
        if not reference:
            env.update(POINT3_FAST_MODE="1", POINT3_GT_CACHE=str(scratch / "gt_cache"))
            if candidate != "no_intervention":
                with zipfile.ZipFile(saved / "no_intervention.zip") as handle:
                    cache = {"identity": baseline_identity(job), "metrics": json.loads(handle.read(METRICS))["base"],
                             "audit": json.loads(handle.read("base_evaluation.json"))}
                cache_path = scratch / "baseline_cache.json"
                write_json(cache_path, cache)
                env.update(POINT3_BASELINE_CACHE=str(cache_path),
                           POINT3_BASELINE_IDENTITY=json.dumps(baseline_identity(job)))
        print(f"  {row['scene']} / {label} (50 images)", flush=True)
        folder = run_dir / f"output_{definition['eta']}" / row["scene"]
        log = saved / f"{label}.log"
        seconds = bounded_run([sys.executable, "-u", Path(__file__).with_name("paired_eval.py"),
            "--model", "vggt", "--root", scratch / "data", "--masks-dir", scratch / "masks",
            "--scene", row["scene"], "--eta", definition["eta"], "--out", run_dir,
            "--num-images", 50, "--seed", self.config["seed"], "--strength-config", config_path,
            "--work-dir", run_dir / "staging"], log, self.source, env, self.deadline)
        validate_result(read_json(folder / METRICS), read_json(folder / "strength_trace.json"),
            read_json(folder / "base_evaluation.json"), read_json(folder / "inter_evaluation.json"), candidate, 50)
        probe = {}
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.startswith("PROBE_RESULT "):
                probe = json.loads(line.removeprefix("PROBE_RESULT "))
        write_json(folder / "timing.json", {"elapsed_seconds": seconds, **probe,
                    "baseline_reused": candidate != "no_intervention" and not reference})
        result = save_checkpoint(folder, checkpoint, job, False)
        clear_owned_scratch(run_dir, parent)
        self.new_job_times.append(seconds)
        return result

    def record(self, row, candidate, policy):
        result = self.job(row, candidate)
        factor = read_json(self.saved(row) / "normalization.json")["factor"]
        record = dict(scene=row["scene"], split=row["split"], anchor=row["anchor"], policy=policy,
                      candidate=candidate, CD_norm_base=result["CD_base"]*factor,
                      CD_norm_inter=result["CD_inter"]*factor, seconds=result["elapsed_seconds"])
        self.records.append(record)
        export_results(self.output, self.records, self.selection, self.eval_scenes)
        return record["CD_norm_inter"]

    def reference_check(self, row, candidate):
        self.job(row, candidate, reference=True)
        saved = self.saved(row)
        def load_zip(name):
            with zipfile.ZipFile(saved / f"{name}.zip") as handle:
                return [json.loads(handle.read(p)) for p in (METRICS, "inter_evaluation.json", "base_evaluation.json")]
        reference, fast = load_zip("reference_"+candidate), load_zip(candidate)
        compare_results(reference[0], fast[0], reference[1], fast[1])
        if reference[2]["prediction_sha256"] != fast[2]["prediction_sha256"]:
            raise ValueError("Fast/reference baseline predictions differ")
        write_json(self.output / "reference_check.json", {"status": "passed", "scene": row["scene"],
            "candidate": candidate, "identity": self.identity})

    def forecast(self):
        # First-scene conservative screening, NOT a completion guarantee. Subsequent
        # subprocesses still have a strict deadline. Archive sizes and geometry vary.
        if not self.new_job_times:
            return
        remaining_jobs, missing_scenes = 0, set()
        first_calibration = next(r for r in self.rows if r["split"] == "calibration")
        for row in self.rows:
            labels = {"no_intervention"}
            if row["split"] == "calibration":
                labels.update(variants(self.config))
                if row == first_calibration:
                    labels.add("reference_" + next(iter(variants(self.config))))
            elif self.selection:
                labels.add(self.selection["global_scalar"])
                labels.add(self.selection.get("structured") or
                           self.selection["anchor_map"].get(row["anchor"], self.selection["global_scalar"]))
            else:
                # Selected policies are not known until calibration is complete.
                labels.update({"pending_global", "pending_structured"})
            missing = sum(not (self.saved(row) / f"{label}.zip").exists() for label in labels)
            remaining_jobs += missing
            if missing and row["scene"] not in self.prepared:
                missing_scenes.add(row["scene"])
        remaining_scenes = len(missing_scenes)
        prediction = self.config["forecast_margin"] * (
            remaining_jobs * max(self.new_job_times) +
            remaining_scenes * max(self.preparation_times or [0]))
        available = max(0, self.deadline-time.monotonic())
        write_json(self.output / "budget_forecast.json", {"remaining_seconds_estimate": prediction,
            "available_seconds": available, "remaining_jobs_upper_bound": remaining_jobs,
            "basis": "slowest measured job and preparation, multiplied by forecast_margin; rough estimate"})
        print(f"  Conservative remaining estimate {prediction/60:.1f} min; available {available/60:.1f} min", flush=True)
        if prediction > available:
            raise BudgetExceeded("Measured 50-image throughput does not fit this session's remaining budget. "
                                 "Use a faster GPU/prepared runtime or resume; views were not reduced")

    def run(self):
        for index, row in enumerate(r for r in self.rows if r["split"] == "calibration"):
            print(f"Calibrating {row['scene']}", flush=True)
            self.record(row, "no_intervention", "no_intervention")
            self.scores[row["scene"]] = {}
            for j, candidate in enumerate(variants(self.config)):
                self.scores[row["scene"]][candidate] = self.record(row, candidate, "calibration_candidate")
                if index == 0 and j == 0:
                    self.reference_check(row, candidate)
                    self.forecast()
            self.forecast()
        self.selection = select_settings(self.config, self.scores)
        selection_path = self.output / "selection.json"
        if selection_path.exists() and read_json(selection_path) != self.selection:
            raise ValueError("Frozen calibration selection changed")
        write_json(selection_path, self.selection)
        print("Calibration complete; choices frozen before evaluation:", self.selection, flush=True)
        for row in (r for r in self.rows if r["split"] == "evaluation"):
            self.record(row, "no_intervention", "no_intervention")
            self.record(row, self.selection["global_scalar"], "global_scalar")
            candidate = (self.selection["structured"] if self.config["hypothesis"] != "H3"
                         else self.selection["anchor_map"].get(row["anchor"], self.selection["global_scalar"]))
            self.record(row, candidate, "structured")
            self.forecast()
        if not export_results(self.output, self.records, self.selection, self.eval_scenes):
            raise RuntimeError("Missing paired evaluation outcomes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-scene", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sam-python", type=Path)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.prepare_scene:
        prepare_scene(read_json(args.prepare_scene))
        return
    if any(getattr(args, name) is None for name in ("config", "source", "work", "output", "sam_python", "seconds")):
        parser.error("All study paths and --seconds are required")
    config = read_json(args.config)
    deadline = time.monotonic() + min(args.seconds, config["runtime_minutes"]*60)
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "hypothesis": config["hypothesis"], "num_views": 50}
    initialized = False
    existing_manifest = (args.output / "manifest.json").exists()
    def interrupted(signum, frame):
        raise BudgetExceeded("Execution interrupted; completed checkpoints retained")
    if os.name != "nt":
        signal.signal(signal.SIGTERM, interrupted)
    try:
        study = Study(config, args.source.resolve(), args.work, args.output, args.sam_python, deadline)
        initialized = True
        write_json(args.output / "status.json", status)
        study.run()
        status.update(status="complete", evaluation_landmarks=len(study.eval_scenes))
    except Exception as exc:
        status.update(status="budget_insufficient" if isinstance(exc, BudgetExceeded) else "failed",
                      reason=sanitize(f"{type(exc).__name__}: {exc}", os.environ.get("HF_TOKEN", "")))
        print(status["reason"], flush=True)
    finally:
        # A rejected attempt with a different GPU/config must not replace the old
        # study's completion status; its manifest and results remain authoritative.
        if initialized or not existing_manifest:
            write_json(args.output / "status.json", status)
        print(json.dumps(status, indent=2), flush=True)
    if status["status"] != "complete":
        sys.exit(2)


if __name__ == "__main__":
    main()
