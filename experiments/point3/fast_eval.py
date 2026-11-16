"""Opt-in study acceleration without altering the reference metric formulas.

Cache only identical ground truth and the validated, unmodified baseline.
Visualization-only exports are omitted. The GPU preflight compares the fast
path with a fresh reference execution before permitting the dataset run.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path


def baseline_identity(job):
    # Strength settings do not affect the baseline; scene inputs and software do.
    keys = ("protocol", "source_sha256", "revision", "num_views", "seed", "scene",
            "anchor", "views", "torch_version", "gpu")
    return {key: job[key] for key in keys}


def validate_baseline(cache, identity):
    if cache.get("identity") != identity:
        raise ValueError("Cached baseline belongs to different inputs")
    if cache.get("audit", {}).get("status") != "ok" or not cache["audit"].get("prediction_sha256"):
        raise ValueError("Cached baseline has no successful prediction audit")
    return cache


def compare_results(reference, accelerated, reference_audit, accelerated_audit):
    import math
    if reference_audit["prediction_sha256"] != accelerated_audit["prediction_sha256"]:
        raise ValueError("Fast/reference predictions differ; do not launch the full run")
    for arm in ("base", "inter"):
        if reference[arm].keys() != accelerated[arm].keys():
            raise ValueError("Fast/reference metric fields differ")
        for key, value in reference[arm].items():
            if not math.isclose(value, accelerated[arm][key], rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(f"Fast/reference metric differs: {arm}/{key}")


def cached_ground_truth(original, directory):
    """Lossless NumPy cache in runner-owned per-scene scratch, never pickle."""
    import numpy as np
    directory = Path(directory)
    signature = inspect.signature(original)

    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        paths = values["image_paths"]
        masks = values.get("validity_masks_by_path") or {}
        identity = {
            "images": [str(Path(p).resolve()) for p in paths],
            "reconstruction": str(Path(values["recon_path"]).resolve()),
            "masks": [hashlib.sha256(np.asarray(masks[p]).tobytes()).hexdigest()
                      if p in masks else None for p in paths],
        }
        marker = directory / "complete.json"
        if marker.exists():
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if metadata["identity"] != identity:
                raise ValueError("Ground-truth cache inputs changed")
            current_paths = {Path(p).name: p for p in paths}
            samples = []
            for index, record in enumerate(metadata["samples"]):
                sample = {}
                for key, value in record.items():
                    if isinstance(value, dict) and "array" in value:
                        # Native evaluation libraries require writable buffers.
                        # Copy-on-write permits that without modifying the cache.
                        sample[key] = np.load(directory / value["array"], mmap_mode="c", allow_pickle=False)
                    else:
                        sample[key] = value
                sample["image_path"] = current_paths[Path(sample["image_path"]).name]
                samples.append(sample)
            print("Reused identical ground truth for this scene", flush=True)
            return np.load(directory / "points.npy", mmap_mode="c", allow_pickle=False), samples
        result = original(*args, **kwargs)
        if result is None or result[0] is None:
            return result
        points, samples = result
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "points.npy", points, allow_pickle=False)
        records = []
        for index, sample in enumerate(samples):
            record = {}
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    filename = f"sample_{index}_{key}.npy"
                    np.save(directory / filename, value, allow_pickle=False)
                    record[key] = {"array": filename}
                else:
                    record[key] = value
            records.append(record)
        marker.write_text(json.dumps({"identity": identity, "samples": records}), encoding="utf-8")
        return result

    return wrapped


def install(module, directory):
    """Patch only this isolated worker; original repository functions stay intact."""
    for name in ("save_prediction_glb", "save_aligned_combined_glb", "save_inter_error_colored_glb",
                 "visualize_intervention_patches", "pointcloud_to_glb"):
        setattr(module, name, lambda *args, **kwargs: None)
    module.generate_gt_by_unproject = cached_ground_truth(module.generate_gt_by_unproject, directory)
    # Reference print statements assume exports exist; make omissions explicit.
    def status_print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith(("Saved GT GLB", "Saved pred_")):
            args = ("Omitted 3D export in fast mode",)
        print(*args, **kwargs)
    module.print = status_print


def prepare_baseline_environment():
    """Validate cache before the reference pass reads environment settings."""
    path = os.environ.get("POINT3_BASELINE_CACHE", "")
    if not path:
        return None
    identity = json.loads(os.environ["POINT3_BASELINE_IDENTITY"])
    cache = validate_baseline(json.loads(Path(path).read_text(encoding="utf-8")), identity)
    scene = identity["scene"]
    root = Path(path).parent / "metric_baseline"
    folder = root / scene
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(cache["metrics"]), encoding="utf-8")
    os.environ["METRIC_EVAL_BASE"] = str(root)
    return cache


def attach_baseline_audit(module, cache):
    original = module.main

    def main(*args, **kwargs):
        result = original(*args, **kwargs)
        folder = Path(kwargs["work_dir"])
        # The reference pass copies baseline metrics itself; audit their origin.
        if (folder / "inter_evaluation.json").exists():
            audit = {**cache["audit"], "evaluation_source": "reused_validated_zero_control",
                     "baseline_identity": cache["identity"]}
            (folder / "base_evaluation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return result

    module.main = main
