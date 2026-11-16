"""Resumable, scene-at-a-time point-3 experiment; launch from the Colab notebook.

All metadata scenes receive a status, including missing anchors/views, failed
masks and failed evaluations. Requested view counts are strict: no downsampling
or repeated images are used to fill a scene with insufficient supplied views.
Only scratch data owned by this runner is removed; verified results and masks
are checkpointed to the persistent destination before scratch is reclaimed.
"""
import argparse
from collections import Counter, deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile

from paired_eval import PROTOCOL
from fast_eval import baseline_identity, compare_results

REVISION = "5d4404323a21aa3de948f8bc57a053b29fa3d921"
CONDITIONS = ("no_intervention", "scalar", "layer_relative", "view_relative")
METRICS = "eval_metrics_0_23.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def safe_scene(scene):
    if not scene or scene in (".", "..") or "/" in scene or "\\" in scene:
        raise ValueError(f"Unsafe scene name: {scene!r}")
    return scene


def inventory(metadata, views, anchors, num_views):
    rows = []
    for item in metadata:
        scene = safe_scene(item["scene"])
        reasons = []
        selected = views.get(scene, [])[:num_views]
        if len(selected) != num_views:
            reasons.append(f"only {len(selected)} of {num_views} selected views supplied")
        if len(set(selected)) != len(selected):
            reasons.append("duplicate supplied views")
        if not anchors.get(scene):
            reasons.append("no supplied anchor")
        rows.append({"scene": scene, "anchor": anchors.get(scene), "views": selected,
                     "requested_views": num_views, "supplied_views": len(views.get(scene, [])),
                     "status": "blocked" if reasons else "ready", "reason": "; ".join(reasons)})
    return rows


def extract_selected(archive, destination, scene, views):
    """Extract selected photographs/depth/masks plus the scene's reconstruction."""
    destination = Path(destination).resolve()
    image_names = set(views)
    bases = {PurePosixPath(v).name for v in views}
    stems = {PurePosixPath(v).stem for v in views}
    derived = {n + ext for n in bases | stems for ext in (".npy", ".npz")}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if not parts or ".." in parts or PurePosixPath(member.name).is_absolute():
                raise ValueError("Unsafe archive member")
            if member.isdir():
                continue
            selected = False
            if len(parts) >= 4 and parts[:2] == ("archworld", scene):
                selected = (parts[2] == "sparse" or
                            (parts[2] == "images" and "/".join(parts[3:]) in image_names) or
                            (parts[2] == "depth" and parts[-1] in derived))
            elif len(parts) >= 3 and parts[:2] == ("manual_filtration_masks", scene):
                selected = parts[-1] in derived
            if not selected:
                continue
            if not member.isfile():
                raise ValueError("Selected archive entries must be regular files")
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive member escapes destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.size + 2 * 2**30 > shutil.disk_usage(destination).free:
                raise OSError("Insufficient free disk to extract the next file")
            with handle.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def sanitize(text, token=""):
    if token:
        text = text.replace(token, "[HF_TOKEN hidden]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[HF_TOKEN hidden]", text)


def run_logged(command, log_path, source, environment):
    start = time.monotonic()
    tail = deque(maxlen=15)
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write("\n--- New attempt ---\n")
        with subprocess.Popen([str(a) for a in command], cwd=source, env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            try:
                for line in process.stdout:
                    safe = sanitize(line, environment.get("HF_TOKEN", ""))
                    tail.append(safe)
                    log.write(safe)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if code:
        raise RuntimeError(f"Process exited {code}; inspect {log_path.name}\n" + "".join(tail))
    return time.monotonic() - start


def validate_result(metrics, trace, base_audit, inter_audit, condition, views):
    for arm in ("base", "inter"):
        for field in ("Acc-mean", "Comp-mean"):
            number = metrics[arm][field]
            if not math.isfinite(number) or number < 0:
                raise ValueError("Missing/nonfinite reconstruction metrics")
    layers = trace["layers"]
    if len(layers) != 24 or {r["layer"] for r in layers} != set(range(24)):
        raise ValueError("Incomplete strength trace")
    if trace["num_views"] != views or any(r["accepted"] is None for r in layers):
        raise ValueError("Incomplete view/acceptance trace")
    if any(sum(r["anchor_tokens_by_view"]) == 0 for r in layers):
        raise ValueError("No intervention anchor tokens")
    for audit in (base_audit, inter_audit):
        if audit["protocol"] != PROTOCOL or audit["status"] != "ok":
            raise ValueError("Evaluation audit is missing or failed")
    base = (metrics["base"]["Acc-mean"] + metrics["base"]["Comp-mean"]) / 2
    inter = (metrics["inter"]["Acc-mean"] + metrics["inter"]["Comp-mean"]) / 2
    if condition == "no_intervention":
        if base_audit["prediction_sha256"] != inter_audit["prediction_sha256"]:
            raise ValueError("Zero-step control changed predictions; stop and inspect")
        if not math.isclose(base, inter, rel_tol=1e-5, abs_tol=1e-9):
            raise ValueError("Zero-step control changed metrics; stop and inspect")
    return base, inter


def read_checkpoint(path, job):
    if not Path(path).exists():
        return None
    with zipfile.ZipFile(path) as handle:
        if handle.testzip() is not None:
            raise ValueError("Corrupt checkpoint")
        load = lambda name: json.loads(handle.read(name))
        if load("job.json") != job:
            raise ValueError("Checkpoint inputs differ; use a new run directory")
        base, inter = validate_result(load(METRICS), load("strength_trace.json"),
            load("base_evaluation.json"), load("inter_evaluation.json"),
            job["condition"], job["num_views"])
        return {"CD_base": base, "CD_inter": inter, **load("timing.json")}


def save_checkpoint(folder, checkpoint, job, keep_visuals):
    write_json(Path(folder) / "job.json", job)
    temporary = Path(checkpoint).with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as handle:
        for p in Path(folder).iterdir():
            if p.is_file() and (p.suffix in (".json", ".log") or
                               (keep_visuals and p.suffix in (".glb", ".png"))):
                handle.write(p, p.name)
    result = read_checkpoint(temporary, job)
    temporary.replace(checkpoint)
    return result


def clear_owned_scratch(path, scratch_root):
    path, scratch_root = Path(path).resolve(), Path(scratch_root).resolve()
    if path.parent != scratch_root or not (path / ".point3-owned").is_file():
        raise ValueError("Refusing to remove unowned scratch directory")
    shutil.rmtree(path)


def source_hash(source):
    digest = hashlib.sha256()
    for subdir in ("eval", "intervention"):
        for path in sorted((source / subdir).rglob("*")):
            if path.suffix in (".py", ".json"):
                digest.update(path.relative_to(source).as_posix().encode())
                digest.update(path.read_bytes())
    for name in ("full_dataset.py", "paired_eval.py", "fast_eval.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def preflight_identity(manifest, gpu, torch_version):
    return {key: manifest[key] for key in ("source_sha256", "revision", "num_views", "seed", "eta", "alpha")} | {
        "gpu": gpu, "torch_version": torch_version}


def check_preflight(record, identity):
    if (record.get("identity") != identity or record.get("status") != "passed" or
            set(record.get("completed_conditions", [])) != set(CONDITIONS) or
            record.get("reference_equivalence", {}).get("status") != "passed"):
        raise RuntimeError("Run the GPU preflight cell with these settings and this GPU before the full run")


def summarize(rows, state, output, metadata):
    records = []
    for row in rows:
        scene = row["scene"]
        for name in CONDITIONS:
            record = {"scene": scene, "anchor": row["anchor"], "condition": name,
                      "requested_views": row["requested_views"], "supplied_views": row["supplied_views"],
                      "status": row["status"] if row["status"] == "blocked" else "pending",
                      "reason": row["reason"]}
            record.update(state.get(scene, {}).get(name, {}))
            if record["status"] == "complete":
                meta = metadata[scene]
                scale, extent = float(meta["Scale"]), float(meta["Extent"])
                if not math.isfinite(scale) or not math.isfinite(extent) or scale <= 0 or extent <= 0:
                    raise ValueError(f"Invalid normalization metadata: {scene}")
                factor = scale * 1000 / extent
                record.update(CD_norm_base=record["CD_base"] * factor,
                              CD_norm_inter=record["CD_inter"] * factor)
            records.append(record)
    fields = sorted({key for record in records for key in record})
    temporary = output / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "results.csv")
    # Compare conditions on exactly the same successfully completed scenes.
    matched = {r["scene"] for r in rows if all(
        state.get(r["scene"], {}).get(c, {}).get("status") == "complete" for c in CONDITIONS)}
    comparison = []
    for anchor in ["ALL"] + sorted({r["anchor"] for r in rows if r["anchor"]}):
        for condition in CONDITIONS:
            cohort = [r for r in records if r["scene"] in matched and r["condition"] == condition
                      and (anchor == "ALL" or r["anchor"] == anchor)]
            if cohort:
                before = statistics.mean(r["CD_norm_base"] for r in cohort)
                after = statistics.mean(r["CD_norm_inter"] for r in cohort)
                comparison.append({"anchor": anchor, "condition": condition, "scenes": len(cohort),
                                   "CD_norm_base": before, "CD_norm_inter": after,
                                   "change_percent": 100 * (after / before - 1) if before else None})
    temporary = output / "comparison.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anchor", "condition", "scenes", "CD_norm_base",
                                                   "CD_norm_inter", "change_percent"])
        writer.writeheader()
        writer.writerows(comparison)
    temporary.replace(output / "comparison.csv")
    write_json(output / "progress.json", {
        "total_dataset_scenes": len(rows), "statuses": dict(Counter(r["status"] for r in records)),
        "completed_scenes": sum(all(state.get(r["scene"], {}).get(c, {}).get("status") == "complete"
                                    for c in CONDITIONS) for r in rows),
        "note": "Fixed settings; not a tuned comparison or 50-view replication unless configured accordingly.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam-python", type=Path, required=True)
    parser.add_argument("--num-views", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--eta", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=.02)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run only the first eligible scene through all four conditions, then stop")
    args = parser.parse_args()
    if not 2 <= args.num_views <= 50:
        parser.error("num-views must be between 2 and 50")
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scratch_root = (args.work / "full_dataset_scratch").resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    metadata_file = hf_hub_download("ArchWorld/ArchWorld", "ArchWorld_metadata.csv",
        repo_type="dataset", revision=REVISION, token=False)
    with open(metadata_file, encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    views = read_json(source / "intervention/assets/views.json")
    anchors = read_json(source / "intervention/assets/anchors.json")
    rows = inventory(metadata, views, anchors, args.num_views)
    manifest = {"protocol": PROTOCOL, "source_sha256": source_hash(source), "revision": REVISION,
                "num_views": args.num_views, "seed": args.seed, "eta": args.eta, "alpha": args.alpha,
                "acceleration": "cached_baseline_ground_truth_no_3d_exports_v1",
                "conditions": list(CONDITIONS), "inventory": rows}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError("Run inputs changed; choose a new persistent output directory")
    write_json(manifest_path, manifest)
    shutil.copy2(metadata_file, output / "ArchWorld_metadata.csv")
    state_path = output / "state.json"
    state = read_json(state_path) if state_path.exists() else {}
    meta = {r["scene"]: r for r in metadata}
    summarize(rows, state, output, meta)
    print(f"Dataset: {len(rows)} scenes; ready: {sum(r['status']=='ready' for r in rows)}; "
          f"blocked: {sum(r['status']=='blocked' for r in rows)}; {args.num_views} views per scene", flush=True)
    for row in rows:
        if row["status"] == "blocked":
            print("BLOCKED", row["scene"], row["reason"], flush=True)
    if args.audit_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before starting the full run")
    identity = preflight_identity(manifest, torch.cuda.get_device_name(0), torch.__version__)
    preflight_path = output / "gpu_preflight.json"
    if not args.preflight_only:
        check_preflight(read_json(preflight_path) if preflight_path.exists() else {}, identity)
    else:
        write_json(preflight_path, {"status": "running", "identity": identity})
        print(f"GPU preflight: {identity['gpu']}, {args.num_views} images, all four conditions. "
              "Stops after one scene; no automatic reduction in view count.", flush=True)
    if not args.sam_python.exists():
        raise RuntimeError("Install the SAM 3 environment first")
    # Fail once for missing environments instead of downloading hundreds of scenes.
    subprocess.run([str(args.sam_python), "-c", "import torch; "
                    "from sam3.model_builder import build_sam3_image_model; "
                    "from sam3.model.sam3_image_processor import Sam3Processor; "
                    "assert torch.cuda.is_available(), 'SAM 3 needs a GPU'"], check=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("INTERVENTION_") or key in {
            "PER_LAYER_INTERVENTION", "TOP_K_ELEMENTS", "SKIP_BASELINE_PASS", "SKIP_EVAL",
            "METRIC_EVAL_BASE", "STRUCTURED_STRENGTH_CONFIG", "SAVE_RAW_TENSORS",
            "ALIGNMENT_MAX_POINTS", "ALIGNMENT_N_ITERATIONS", "ICP_MAX_ITER"}:
            env.pop(key, None)
    env.update(PROBE="1", EVAL_TIMEOUT_SEC="0", POINT3_SOURCE=str(source),
               POINT3_EVAL_SEED=str(args.seed), PYTHONHASHSEED=str(args.seed),
               PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_IMPLICIT_TOKEN="0")
    for key in ("POINT3_FAST_MODE", "POINT3_BASELINE_CACHE", "POINT3_BASELINE_IDENTITY", "POINT3_GT_CACHE"):
        env.pop(key, None)
    ready = [r for r in rows if r["status"] == "ready"]
    ready.sort(key=lambda r: (r["scene"] != "Akdamar_Church", r["scene"]))
    if not ready:
        raise RuntimeError("No scenes have the requested number of supplied views and an anchor")
    if args.preflight_only:
        ready = ready[:1]
    visual_scenes = {"Akdamar_Church", "Al-Azhar_Mosque", "Koutoubia_Mosque",
                     "Five-storied_Pagoda,_Kofuku-ji"}
    for index, row in enumerate(ready, 1):
        scene = row["scene"]
        saved = output / "scenes" / scene
        saved.mkdir(parents=True, exist_ok=True)
        jobs = {name: {k: v for k, v in manifest.items() if k != "inventory"} |
                {"scene": scene, "anchor": row["anchor"], "views": row["views"], "condition": name,
                 "effective_eta": 0 if name == "no_intervention" else args.eta,
                 "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
                for name in CONDITIONS}
        completed = {}
        for name in CONDITIONS:
            result = read_checkpoint(saved / f"{name}.zip", jobs[name])
            if result:
                completed[name] = {"status": "complete", **result}
        state[scene] = completed
        reference_check = read_json(saved / "reference_check.json") if (saved / "reference_check.json").exists() else {}
        if len(completed) == len(CONDITIONS) and (not args.preflight_only or
                (reference_check.get("identity") == identity and reference_check.get("status") == "passed")):
            continue
        print(f"[{index}/{len(ready)}] {scene} ({row['anchor']})", flush=True)
        scratch = scratch_root / hashlib.sha256(scene.encode()).hexdigest()[:16]
        if scratch.exists():
            clear_owned_scratch(scratch, scratch_root)
        scratch.mkdir()
        (scratch / ".point3-owned").touch()
        data, masks = scratch / "data", scratch / "masks"
        data.mkdir()
        stage = "preparation"
        try:
            if shutil.disk_usage(scratch).free < 16 * 2**30:
                raise OSError("Need at least 16 GiB free scratch space before downloading a scene")
            archive = hf_hub_download("ArchWorld/ArchWorld", f"data/{scene}.tar.gz",
                repo_type="dataset", revision=REVISION, cache_dir=scratch / "download", token=False)
            extract_selected(archive, data, scene, row["views"])
            shutil.copy2(metadata_file, data / "ArchWorld_metadata.csv")
            mask_zip = saved / "masks.zip"
            if mask_zip.exists():
                with zipfile.ZipFile(mask_zip) as handle:
                    if handle.testzip() is not None:
                        raise ValueError("Corrupt saved mask archive")
                    for member in handle.infolist():
                        if not (masks / member.filename).resolve().is_relative_to(masks.resolve()):
                            raise ValueError("Unsafe mask archive")
                    handle.extractall(masks)
            script = (
                "import sys;sys.path.insert(0," + repr(str(source)) + ");"
                "from intervention import make_masks as m;"
                "from sam3.model_builder import build_sam3_image_model;"
                "from sam3.model.sam3_image_processor import Sam3Processor;"
                f"m.views_for=lambda scene:{row['views']!r};"
                "p=Sam3Processor(build_sam3_image_model(device='cuda'),confidence_threshold=m.CONFIDENCE);"
                f"m.segment_scene(p,{str(data)!r},{scene!r},{str(masks)!r},False,True)"
            )
            if not mask_zip.exists():
                run_logged([args.sam_python, "-u", "-c", script], saved / "masks.log", source, env)
            import numpy as np
            nonempty = 0
            for view in row["views"]:
                if not (data / "archworld" / scene / "images" / view).is_file():
                    raise ValueError(f"Selected image missing: {view}")
                mask = np.load(masks / scene / "mask" / f"{Path(view).stem}_{row['anchor']}.npy",
                               allow_pickle=False)
                nonempty += bool(mask.size and mask.any())
                valid_dir = data / "manual_filtration_masks" / scene
                if not any((valid_dir / (n + ".npy")).is_file() for n in (Path(view).name, Path(view).stem)):
                    raise ValueError(f"Missing landmark validity mask: {view}")
            if nonempty < 2:
                raise ValueError(f"Anchor visible in only {nonempty} selected views")
            if not mask_zip.exists():
                temporary = saved / "masks.zip.tmp"
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as handle:
                    for file in masks.rglob("*.npy"):
                        handle.write(file, file.relative_to(masks).as_posix())
                with zipfile.ZipFile(temporary) as handle:
                    if handle.testzip() is not None:
                        raise OSError("Mask checkpoint verification failed")
                temporary.replace(mask_zip)
            for name in CONDITIONS:
                if name in completed:
                    continue
                stage = name
                print(f"  Running {name}", flush=True)
                config = {"mode": "fixed" if name in ("no_intervention", "scalar") else name,
                          "alpha": args.alpha}
                config_path = scratch / f"{name}.json"
                write_json(config_path, config)
                eta = 0 if name == "no_intervention" else args.eta
                run_dir = scratch / name
                folder = run_dir / f"output_{eta}" / scene
                log = saved / f"{name}.log"
                job_env = env.copy()
                job_env.update(POINT3_FAST_MODE="1", POINT3_GT_CACHE=str(scratch / "gt_cache"))
                if name != "no_intervention":
                    with zipfile.ZipFile(saved / "no_intervention.zip") as handle:
                        baseline = {"identity": baseline_identity(jobs[name]),
                                    "metrics": json.loads(handle.read(METRICS))["base"],
                                    "audit": json.loads(handle.read("base_evaluation.json"))}
                    cache_path = scratch / "baseline_cache.json"
                    write_json(cache_path, baseline)
                    job_env.update(POINT3_BASELINE_CACHE=str(cache_path),
                                   POINT3_BASELINE_IDENTITY=json.dumps(baseline_identity(jobs[name])))
                command = [sys.executable, "-u", Path(__file__).with_name("paired_eval.py"),
                    "--model", "vggt", "--root", data, "--masks-dir", masks, "--scene", scene,
                    "--eta", eta, "--out", run_dir, "--num-images", args.num_views,
                    "--seed", args.seed, "--strength-config", config_path,
                    "--work-dir", scratch / "staging"]
                seconds = run_logged(command, log, source, job_env)
                # The reference driver can exit zero after a failed scene; inspect artifacts.
                validate_result(read_json(folder / METRICS), read_json(folder / "strength_trace.json"),
                    read_json(folder / "base_evaluation.json"), read_json(folder / "inter_evaluation.json"),
                    name, args.num_views)
                if name != "no_intervention":
                    with zipfile.ZipFile(saved / "no_intervention.zip") as handle:
                        control = json.loads(handle.read("base_evaluation.json"))
                    if control["prediction_sha256"] != read_json(folder / "base_evaluation.json")["prediction_sha256"]:
                        raise ValueError("Baseline predictions differ across conditions")
                probe = {}
                for line in log.read_text(encoding="utf-8").splitlines():
                    if line.startswith("PROBE_RESULT "):
                        probe = json.loads(line.removeprefix("PROBE_RESULT "))
                write_json(folder / "timing.json", {"elapsed_seconds": seconds, **probe,
                                                    "baseline_reused": name != "no_intervention"})
                shutil.copy2(log, folder / "run.log")
                result = save_checkpoint(folder, saved / f"{name}.zip", jobs[name], scene in visual_scenes)
                state[scene][name] = {"status": "complete", **result}
                write_json(state_path, state)
                summarize(rows, state, output, meta)
                print(f"  Completed {name} ({seconds / 60:.1f} min); saved to Drive", flush=True)
            if args.preflight_only:
                stage = "fast_reference_comparison"
                print("  Checking fast scalar against a fresh reference run (once)", flush=True)
                config_path = scratch / "reference_scalar.json"
                write_json(config_path, {"mode": "fixed", "alpha": args.alpha})
                reference_out = scratch / "reference_scalar"
                reference_folder = reference_out / f"output_{args.eta}" / scene
                seconds = run_logged([sys.executable, "-u", Path(__file__).with_name("paired_eval.py"),
                    "--model", "vggt", "--root", data, "--masks-dir", masks, "--scene", scene,
                    "--eta", args.eta, "--out", reference_out, "--num-images", args.num_views,
                    "--seed", args.seed, "--strength-config", config_path,
                    "--work-dir", scratch / "staging_reference"], saved / "reference_scalar.log", source, env)
                reference_metrics = read_json(reference_folder / METRICS)
                reference_audit = read_json(reference_folder / "inter_evaluation.json")
                with zipfile.ZipFile(saved / "scalar.zip") as handle:
                    fast_metrics = json.loads(handle.read(METRICS))
                    fast_audit = json.loads(handle.read("inter_evaluation.json"))
                    fast_base_audit = json.loads(handle.read("base_evaluation.json"))
                compare_results(reference_metrics, fast_metrics, reference_audit, fast_audit)
                if read_json(reference_folder / "base_evaluation.json")["prediction_sha256"] != fast_base_audit["prediction_sha256"]:
                    raise ValueError("Fast/reference baseline predictions differ")
                reference_check = {"status": "passed", "identity": identity,
                    "reference_seconds": seconds, "fast_seconds": state[scene]["scalar"]["elapsed_seconds"],
                    "measured_scalar_speedup": seconds / state[scene]["scalar"]["elapsed_seconds"],
                    "note": "One-scene measurement, not a dataset-wide speedup guarantee."}
                write_json(saved / "reference_check.json", reference_check)
                write_json(saved / "reference_metrics.json", reference_metrics)
                print(f"  Equivalence passed. Scalar condition: reference {seconds:.1f}s, "
                      f"fast {reference_check['fast_seconds']:.1f}s "
                      f"({reference_check['measured_scalar_speedup']:.2f}x on this scene)", flush=True)
                # Keep the preflight's reference visual examples; bulk jobs omit exports.
                with zipfile.ZipFile(saved / "reference_examples.zip", "w", zipfile.ZIP_DEFLATED) as handle:
                    for file in reference_folder.iterdir():
                        if file.suffix in (".json", ".glb", ".png") and file.is_file():
                            handle.write(file, file.name)
        except Exception as exc:
            message = sanitize(f"{type(exc).__name__}: {exc}", env.get("HF_TOKEN", ""))
            for name in CONDITIONS:
                if state[scene].get(name, {}).get("status") != "complete":
                    state[scene][name] = {"status": "failed", "reason": f"{stage}: {message}"}
            print(f"  FAILED {stage}: {message}", flush=True)
            if args.preflight_only:
                write_json(preflight_path, {"status": "failed", "identity": identity,
                                           "scene": scene, "stage": stage, "reason": message})
            # A failed zero control or first prepared scene must not trigger hundreds of jobs.
            if stage == "no_intervention" or index == 1:
                raise RuntimeError("Control/readiness gate failed; full run stopped. Inspect saved logs.") from None
            if "out of memory" in message.lower() or "No space left" in message:
                raise
        finally:
            write_json(state_path, state)
            summarize(rows, state, output, meta)
            clear_owned_scratch(scratch, scratch_root)
    write_json(state_path, state)
    summarize(rows, state, output, meta)
    if args.preflight_only:
        scene = ready[0]["scene"]
        if not all(state[scene].get(name, {}).get("status") == "complete" for name in CONDITIONS):
            raise RuntimeError("GPU preflight did not complete every condition")
        measurements = [(name, "inter_gb") for name in CONDITIONS] + [("no_intervention", "base_gb")]
        if any(not isinstance(state[scene][name].get(key), (int, float)) or
               not math.isfinite(state[scene][name][key]) or state[scene][name][key] <= 0
               for name, key in measurements):
            raise RuntimeError("GPU preflight is missing memory measurements; inspect the saved logs")
        write_json(preflight_path, {
            "status": "passed", "identity": identity, "scene": scene,
            "completed_conditions": list(CONDITIONS),
            "reference_equivalence": read_json(output / "scenes" / scene / "reference_check.json"),
            "total_vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "peak_allocated_gb": {name: {k: state[scene][name].get(k) for k in ("base_gb", "inter_gb")}
                                  for name in CONDITIONS},
            "note": "Measured on this scene; other scenes can still require more GPU or system memory.",
        })
        print("GPU preflight passed. Its results are saved and will be reused. Run the final cell to continue.", flush=True)
        return
    print(json.dumps(read_json(output / "progress.json"), indent=2), flush=True)
    print("Finished this pass. Inspect results.csv for failures/blocked scenes; rerun to retry failures.", flush=True)


if __name__ == "__main__":
    main()
