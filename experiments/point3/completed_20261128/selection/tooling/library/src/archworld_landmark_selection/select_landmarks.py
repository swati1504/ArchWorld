"""CPU selection of complete training landmarks from frozen encoder outputs.

Run with --encodings PATH --out PATH. No model, photo download, or intervention.
See README.md for the explicitly heuristic size rule and unmeasured properties.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
VERSION = "0.4.0"
CATEGORIES = ("building_type", "region", "anchor")
NUMERICS = ("extent_m", "image_count", "duplicate_fraction", "median_megapixels")
LIMITATIONS = [
    "Coverage describes the 317 training landmarks only; the 79 evaluation landmarks are excluded.",
    "The chosen size is a coverage-curve elbow heuristic, not a proven minimum or representativeness certificate.",
    "VGGT features describe whole photographs, including backgrounds; mean/std can miss multimodal appearance.",
    "Anchor labels are supplied metadata, not measured anchor visibility or complete architectural-element inventories.",
    "Blur, exposure, camera/viewpoint coverage, depth/mask completeness and anchor visibility are not measured here.",
    "Coverage and balance are optimized on these descriptors. Random percentiles are descriptive, not p-values or independent validation.",
    "No reconstruction or intervention outcomes are used or tested. Outcome stability needs a separate experiment.",
    "Selected manifests keep every original image path, including duplicate content; raw photos are not copied by this selector.",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def emit(message):
    print(message, flush=True)


@dataclass(frozen=True)
class SelectionConfig:
    seed: int = 1729
    curve_points: int = 12
    refinement_points: int = 9
    curve_candidates: int = 4
    final_candidates: int = 16
    swap_attempts: int = 256
    random_subsets: int = 1000
    # Equal weight per descriptor family; each landmark also has equal weight.
    appearance_weight: float = 1.0
    composition_weight: float = 1.0
    scale_weight: float = 1.0
    collection_weight: float = 1.0
    knee_min_gain: float = 0.05
    # Extra landmarks permitted beyond each anchor's floor/ceiling target.
    # Zero reproduces the strict 0.2.0 bounds; presence is always mandatory.
    anchor_count_slack: int = 1

    def __post_init__(self):
        for name in ("seed", "curve_points", "refinement_points", "curve_candidates", "final_candidates", "swap_attempts", "random_subsets", "anchor_count_slack"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name in ("seed", "anchor_count_slack") else 1):
                raise ValueError(f"Invalid {name}")
        if self.curve_points < 3 or self.refinement_points < 3 or self.final_candidates < 2:
            raise ValueError("At least 3 curve/refinement points and 2 final candidates required")
        weights = self.family_weights
        if not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("Family weights must be finite, nonnegative, with a positive total")
        if not 0 <= self.knee_min_gain < 1:
            raise ValueError("knee_min_gain must be in [0, 1)")

    @property
    def family_weights(self):
        return [self.appearance_weight, self.composition_weight, self.scale_weight, self.collection_weight]

    @property
    def anchor_policy(self):
        return ("Every supplied label including __MISSING__: "
                "max(1,floor(n*p)-s) <= count <= min(training_count,max(1,ceil(n*p)+s)); "
                f"s={self.anchor_count_slack} extra landmarks beyond rounding")


def load_inputs(folder=HERE / "inputs"):
    folder = Path(folder)
    checks = read_json(folder / "checksums.json")
    for name, digest in checks.items():
        if Path(name).name != name or sha(folder / name) != digest:
            raise ValueError(f"Frozen input checksum mismatch: {name}")
    train = read_json(folder / "training_landmarks.json")
    evaluation = read_json(folder / "evaluation_landmarks.json")
    metadata = read_json(folder / "training_metadata.json")
    sources = read_json(folder / "archive_sources.json")
    if len(train) != 317 or len(set(train)) != 317 or len(evaluation) != 79 or len(set(evaluation)) != 79:
        raise ValueError("Expected frozen 317/79 split")
    if set(train) & set(evaluation) or set(metadata) != set(train) or set(sources) != set(train):
        raise ValueError("Split overlap or incorrect metadata/archive population")
    for scene in train:
        if scene in {".", ".."} or not scene or any(c in scene for c in "/\\\x00:"):
            raise ValueError("Unsafe landmark identifier")
        row = metadata[scene]
        if any(not row.get(k) for k in CATEGORIES) or not np.isfinite(row["extent_m"]) or row["extent_m"] <= 0:
            raise ValueError(f"Missing required metadata: {scene}")
    return train, evaluation, metadata, sources, checks


def check_completion(encodings, train, evaluation):
    path = Path(encodings) / "progress.json"
    if not path.is_file():
        raise RuntimeError("No progress.json yet. Finish the encoder notebook before selecting.")
    p = read_json(path)
    done = p.get("completed", [])
    if (set(p.get("requested", [])) != set(train) or len(done) != len(train)
            or set(done) != set(train) or p.get("failures")
            or p.get("all_317_training_landmarks_complete") is not True):
        raise RuntimeError(f"Encodings are not complete: {len(set(done) & set(train))}/{len(train)} completed; "
                           f"{len(p.get('failures', []))} recorded failures. Finish/resume extraction first.")
    if set(done) & set(evaluation) or p.get("holdout_encoded") is not False:
        raise ValueError("Holdout exclusion was not established")
    return p


def validate_landmark(folder, scene, settings, expected_dimension=1024):
    """Read only one landmark's features at a time, checking identities and rows."""
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    summary = read_json(folder / "summary.json")
    identity = {k: manifest[k] for k in ("scene", "settings", "images")}
    fp = signature(identity)
    images = manifest["images"]
    names = [r["name"] for r in images]
    if (manifest["scene"] != scene or manifest["settings"] != settings
            or manifest.get("fingerprint") != fp or summary.get("fingerprint") != fp
            or summary.get("scene") != scene or summary.get("complete") is not True
            or summary.get("image_count") != len(images) or not images
            or len(set(names)) != len(names) or manifest.get("image_policy") != "all_original_images"):
        raise ValueError(f"Invalid manifest/completion: {scene}")
    if summary.get("source") != "sha256_verified_pinned_archive":
        raise ValueError(f"Original archive completeness unverified: {scene}. Use the original download-mode encodings.")
    for row in images:
        p = PurePosixPath(row["name"])
        if p.is_absolute() or ".." in p.parts or any(c in row["name"] for c in "\\\x00:"):
            raise ValueError(f"Unsafe image name: {scene}")
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]) or row["bytes"] <= 0:
            raise ValueError(f"Invalid image identity: {scene}")
    with np.load(folder / "encodings.npz", allow_pickle=False) as saved:
        if saved["fingerprint"].item() != fp or saved["names"].tolist() != names:
            raise ValueError(f"Feature/image identity mismatch: {scene}")
        z = saved["embeddings"].astype(np.float64)
        wh = saved["original_wh"].astype(np.float64)
    if z.shape != (len(images), expected_dimension) or wh.shape != (len(images), 2):
        raise ValueError(f"Wrong feature dimensions: {scene}")
    if not np.isfinite(z).all() or not np.isfinite(wh).all() or (wh <= 0).any():
        raise ValueError(f"Nonfinite features or invalid image dimensions: {scene}")
    if not np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-4):
        raise ValueError(f"Unnormalized features: {scene}")
    mean, std = z.mean(axis=0), z.std(axis=0)
    if not np.allclose(mean, summary["embedding_mean"], atol=1e-7) or not np.allclose(std, summary["embedding_std"], atol=1e-7):
        raise ValueError(f"Feature/summary mismatch: {scene}")
    pixels = wh.prod(axis=1) / 1e6
    aspect = np.log(wh[:, 0] / wh[:, 1])
    duplicates = len(images) - len({row["sha256"] for row in images})
    if duplicates != summary["duplicate_content_count"]:
        raise ValueError(f"Duplicate count mismatch: {scene}")
    collection = {"image_count": len(images), "duplicate_fraction": duplicates / len(images),
                  "median_megapixels": float(np.median(pixels)),
                  "p10_megapixels": float(np.quantile(pixels, .1)),
                  "p90_megapixels": float(np.quantile(pixels, .9)),
                  "portrait_fraction": float(np.mean(wh[:, 0] < wh[:, 1])),
                  "mean_log_aspect": float(aspect.mean()), "std_log_aspect": float(aspect.std())}
    return mean, std, collection, fp


def load_descriptors(encodings, inputs, output):
    train, evaluation, metadata, sources, checks = inputs
    encodings, output = Path(encodings), Path(output)
    check_completion(encodings, train, evaluation)
    settings = read_json(encodings / "settings.json")
    if settings.get("dataset_revision") != "5d4404323a21aa3de948f8bc57a053b29fa3d921":
        raise ValueError("Dataset revision differs from the frozen archive manifest")
    for name in ("training_landmarks.json", "evaluation_landmarks.json", "archive_sources.json"):
        if settings.get("input_hashes", {}).get(name) != checks[name]:
            raise ValueError(f"Encoder input split/source mismatch: {name}")
    # No evaluation directory is opened, inventoried or used in preprocessing.
    provenance = {"encoder_settings": settings, "inputs": checks, "landmarks": {}}
    means, stds, rows = [], [], []
    for index, scene in enumerate(train):
        folder = encodings / "landmarks" / scene
        paths = [folder / name for name in ("manifest.json", "summary.json", "encodings.npz")]
        if not all(p.is_file() for p in paths):
            raise RuntimeError(f"Missing complete saved artifacts for {scene}")
        identity = {p.name: sha(p) for p in paths}
        cached = output / "descriptor_cache" / f"{scene}.json"
        if cached.is_file() and read_json(cached).get("source_sha256") == identity:
            record = read_json(cached)
        else:
            mean, std, collection, fingerprint = validate_landmark(folder, scene, settings)
            record = {"source_sha256": identity, "fingerprint": fingerprint,
                      "mean": mean.tolist(), "std": std.tolist(), "collection": collection}
            save_json(cached, record)
        provenance["landmarks"][scene] = identity
        means.append(record["mean"])
        stds.append(record["std"])
        rows.append({"scene": scene, **metadata[scene], **record["collection"]})
        if (index + 1) % 25 == 0 or index + 1 == len(train):
            emit(f"Validated encodings: {index + 1}/{len(train)} landmarks")
    return rows, np.asarray(means), np.asarray(stds), provenance


def scaled_distance(matrix):
    """Median of positive training-pair distances fixes family scale."""
    d = np.asarray(matrix, dtype=np.float64)
    values = d[np.triu_indices(len(d), 1)]
    positive = values[values > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return d / scale, scale


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    return (x - x.mean(axis=0)) / scale


class Population:
    def __init__(self, rows, means, stds, config):
        self.rows, self.config = rows, config
        self.scenes = [r["scene"] for r in rows]
        self.n = len(rows)
        if self.n < 4 or len(set(self.scenes)) != self.n or np.shape(means) != np.shape(stds) or len(means) != self.n:
            raise ValueError("Invalid landmark descriptor population")
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError("Invalid appearance summaries")
        self.codes, self.levels, self.counts = {}, {}, {}
        for field in CATEGORIES:
            levels = sorted({r[field] for r in rows})
            codes = np.asarray([levels.index(r[field]) for r in rows])
            self.codes[field], self.levels[field] = codes, levels
            self.counts[field] = np.bincount(codes, minlength=len(levels))
        self.groups = [np.flatnonzero(self.codes["building_type"] == i) for i in range(len(self.levels["building_type"]))]
        self.numeric = {key: np.asarray([r[key] for r in rows], dtype=float) for key in NUMERICS}
        if not all(np.isfinite(v).all() for v in self.numeric.values()):
            raise ValueError("Missing or nonfinite continuous metadata")
        self.numeric_sorted = {key: np.sort(value) for key, value in self.numeric.items()}
        dm, sm = scaled_distance(cdist(means, means))
        ds, ss = scaled_distance(cdist(stds, stds))
        appearance = (dm + ds) / 2
        composition = sum((v[:, None] != v[None, :]).astype(float) for v in self.codes.values()) / len(CATEGORIES)
        extent = np.log1p(self.numeric["extent_m"])
        scale = np.abs(extent[:, None] - extent[None, :])
        collection = np.array([[np.log1p(r["image_count"]), r["duplicate_fraction"],
            np.log1p(r["median_megapixels"]), np.log1p(r["p10_megapixels"]), np.log1p(r["p90_megapixels"]),
            r["portrait_fraction"], r["mean_log_aspect"], r["std_log_aspect"]] for r in rows])
        collection = cdist(standardized(collection), standardized(collection)) / np.sqrt(collection.shape[1])
        self.families, self.scales = {}, {"appearance_mean": sm, "appearance_std": ss}
        for name, d in zip(("appearance", "composition", "scale", "collection"), (appearance, composition, scale, collection)):
            self.families[name], self.scales[name] = scaled_distance(d)
        weights = np.asarray(config.family_weights) / sum(config.family_weights)
        self.d = sum(w * d for w, d in zip(weights, self.families.values()))
        np.fill_diagonal(self.d, 0)

    def allocation(self, size):
        """Closest squared-error integer type counts, at least one of each type."""
        counts = self.counts["building_type"]
        if not len(counts) <= size <= self.n:
            raise ValueError("Size must include all building types and not exceed training population")
        ideal = size * counts / self.n
        q = np.ones(len(counts), dtype=int)
        while q.sum() < size:
            costs = 2 * (q - ideal) + 1
            costs[q >= counts] = np.inf
            q[int(np.argmin(costs))] += 1
        return q

    def random_subset(self, quota, rng):
        return np.sort(np.concatenate([rng.choice(group, int(k), replace=False) for group, k in zip(self.groups, quota)]))

    def region_complete(self, selected):
        return len(np.unique(self.codes["region"][selected])) == len(self.levels["region"])

    def anchor_bounds(self, size):
        """Every anchor remains; counts allow explicit slack beyond floor/ceil.

        Integer arithmetic avoids floating-point errors at exact quota boundaries.
        Missing labels stay a separate stratum rather than disappearing silently.
        """
        counts = self.counts["anchor"]
        slack = self.config.anchor_count_slack
        low = np.maximum(1, size * counts // self.n - slack)
        high = np.minimum(counts, np.maximum(1, (size * counts + self.n - 1) // self.n + slack))
        return low, high

    def anchors_valid(self, selected):
        low, high = self.anchor_bounds(len(selected))
        actual = np.bincount(self.codes["anchor"][selected], minlength=len(low))
        return bool(np.all(actual >= low) and np.all(actual <= high))

    def valid_selection(self, selected, quota=None):
        selected = np.asarray(selected, dtype=int)
        if not len(selected) or len(np.unique(selected)) != len(selected) or min(selected) < 0 or max(selected) >= self.n:
            return False
        quota = self.allocation(len(selected)) if quota is None else quota
        actual = np.bincount(self.codes["building_type"][selected], minlength=len(quota))
        return bool(np.array_equal(actual, quota) and self.region_complete(selected) and self.anchors_valid(selected))

    def feasible_start(self, quota, rng):
        anchor_low, anchor_high = self.anchor_bounds(int(sum(quota)))
        if anchor_low.sum() > sum(quota) or anchor_high.sum() < sum(quota):
            return None
        indicators = [self.codes["building_type"] == i for i in range(len(quota))]
        indicators += [self.codes["region"] == i for i in range(len(self.levels["region"]))]
        indicators += [self.codes["anchor"] == i for i in range(len(self.levels["anchor"]))]
        low = np.r_[quota, np.ones(len(self.levels["region"])), anchor_low]
        high = np.r_[quota, np.full(len(self.levels["region"]), np.inf), anchor_high]
        result = milp(rng.random(self.n), integrality=np.ones(self.n), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(np.asarray(indicators, dtype=float), low, high),
                      options={"time_limit": 15})
        if result.x is None:
            if result.status == 2:
                return None
            raise RuntimeError("Type/region/anchor feasibility solver did not finish; cannot label this size infeasible")
        chosen = np.flatnonzero(result.x > .5)
        if not self.valid_selection(chosen, quota):
            raise RuntimeError("Invalid solver allocation")
        return chosen

    def minimum_feasible_size(self):
        lower = max(len(self.groups), len(self.levels["region"]), len(self.levels["anchor"]))
        for size in range(lower, self.n + 1):
            if self.feasible_start(self.allocation(size), seeded(self.config, 5, size)) is not None:
                return size
        raise RuntimeError("Even the full training population failed its own constraints")

    def balance(self, selected):
        result = {}
        for field in CATEGORIES:
            proportion = np.bincount(self.codes[field][selected], minlength=len(self.levels[field])) / len(selected)
            result[field + "_tv"] = float(np.abs(proportion - self.counts[field] / self.n).sum() / 2)
        for field, full in self.numeric_sorted.items():
            sub = np.sort(self.numeric[field][selected])
            result[field + "_ks"] = float(np.max(np.abs(np.searchsorted(full, full, side="right") / self.n
                                                       - np.searchsorted(sub, full, side="right") / len(sub))))
        return result

    def metrics(self, selected):
        selected = np.asarray(selected, dtype=int)
        if len(selected) == 0 or len(np.unique(selected)) != len(selected) or min(selected) < 0 or max(selected) >= self.n:
            raise ValueError("Invalid selection indices")
        unselected = np.setdiff1d(np.arange(self.n), selected)
        balance = self.balance(selected)
        result = {"size": int(len(selected)), "unselected_count": int(len(unselected)), **balance,
                  "all_regions_present": self.region_complete(selected),
                  "all_anchors_present": len(np.unique(self.codes["anchor"][selected])) == len(self.levels["anchor"]),
                  "anchor_bounds_satisfied": self.anchors_valid(selected)}
        if len(unselected):
            near = self.d[np.ix_(unselected, selected)].min(axis=1)
            visual = self.families["appearance"][np.ix_(unselected, selected)].min(axis=1)
            result.update(coverage_mean=float(near.mean()), coverage_median=float(np.median(near)),
                          coverage_p95=float(np.quantile(near, .95)), coverage_worst=float(near.max()),
                          appearance_mean=float(visual.mean()), appearance_p95=float(np.quantile(visual, .95)))
        else:
            result.update({k: None for k in ("coverage_mean", "coverage_median", "coverage_p95", "coverage_worst", "appearance_mean", "appearance_p95")})
        tv = float(np.mean([balance[k + "_tv"] for k in CATEGORIES]))
        ks = float(np.mean([balance[k + "_ks"] for k in NUMERICS]))
        result["score"] = float(.35 * (result["coverage_mean"] or 0) + .35 * (result["coverage_p95"] or 0) + .15 * tv + .15 * ks)
        return result

    def improve(self, selected, rng, attempts):
        selected = np.sort(selected).copy()
        if not self.valid_selection(selected):
            raise ValueError("Optimization requires a type/region/anchor-feasible starting subset")
        current = self.metrics(selected)
        for _ in range(attempts):
            position = int(rng.integers(len(selected)))
            group = self.groups[self.codes["building_type"][selected[position]]]
            replacement = int(rng.choice(group))
            if replacement in selected:
                continue
            proposal = selected.copy()
            proposal[position] = replacement
            if not self.valid_selection(proposal):
                continue
            value = self.metrics(proposal)
            if value["score"] < current["score"] - 1e-12:
                selected, current = proposal, value
        return np.sort(selected), current


def seeded(config, phase, size, index=0):
    return np.random.default_rng(np.random.SeedSequence([config.seed, phase, size, index]))


def isotonic_decreasing(values):
    """Unweighted pool-adjacent-violators; avoids noise inventing extra knees."""
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            right, left = blocks.pop(), blocks.pop()
            count = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / count, count])
    return np.asarray([mean for mean, count in blocks for _ in range(count)])


def choose_size(curve, population_size, min_gain=.05, tail_weight=.5):
    rows = sorted((r for r in curve if r.get("feasible") and r["size"] < population_size), key=lambda r: r["size"])
    if len(rows) < 3:
        return {"size": population_size, "status": "no_clear_elbow_use_full_training", "points": []}
    x = np.asarray([r["size"] for r in rows], dtype=float)
    raw = np.asarray([(1 - tail_weight) * r["metrics"]["coverage_mean"] + tail_weight * r["metrics"]["coverage_p95"] for r in rows])
    fitted = isotonic_decreasing(raw)
    if fitted[0] - fitted[-1] <= 1e-12:
        return {"size": population_size, "status": "no_clear_elbow_use_full_training", "points": []}
    gains = (fitted[0] - fitted) / (fitted[0] - fitted[-1]) - (x - x[0]) / (x[-1] - x[0])
    best = int(np.argmax(gains))
    clear = 0 < best < len(rows) - 1 and gains[best] >= min_gain
    return {"size": int(x[best]) if clear else population_size,
            "status": "heuristic_coverage_elbow" if clear else "no_clear_elbow_use_full_training",
            "elbow_gain": float(gains[best]), "minimum_gain": min_gain, "tail_weight": tail_weight,
            "points": [{"size": int(n), "raw_coverage": float(a), "fitted_coverage": float(b), "gain": float(c)}
                       for n, a, b, c in zip(x, raw, fitted, gains)]}


def study_size(pop, size, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "result.json"
    if cached.is_file():
        previous = read_json(cached)
        if not previous["feasible"] or (folder / "random_subsets.npz").is_file():
            return previous
    quota = pop.allocation(size)
    start = pop.feasible_start(quota, seeded(pop.config, 1, size))
    if start is None:
        result = {"size": size, "feasible": False, "reason": "Type quotas, all-region coverage and proportional anchor bounds cannot all be satisfied"}
        save_json(cached, result)
        return result
    baseline_rng = seeded(pop.config, 2, size)
    selections, baseline_metrics = [], []
    best_eligible, best_score = start, pop.metrics(start)["score"]
    for _ in range(pop.config.random_subsets):
        chosen = pop.random_subset(quota, baseline_rng)
        metrics = pop.metrics(chosen)
        selections.append(chosen)
        baseline_metrics.append(metrics)
        if pop.valid_selection(chosen, quota) and metrics["score"] < best_score:
            best_eligible, best_score = chosen, metrics["score"]
    columns = [k for k, v in baseline_metrics[0].items() if isinstance(v, (int, float, bool))]
    save_npz(folder / "random_subsets.npz", training_landmarks=np.asarray(pop.scenes),
             selected_indices=np.asarray(selections, dtype=np.int32), metric_columns=np.asarray(columns),
             metrics=np.asarray([[m[k] for k in columns] for m in baseline_metrics], dtype=float))
    candidates = []
    for i in range(pop.config.curve_candidates):
        rng = seeded(pop.config, 3, size, i)
        initial = best_eligible if i == 0 else pop.feasible_start(quota, rng)
        chosen, metrics = pop.improve(initial, rng, pop.config.swap_attempts)
        candidates.append({"indices": chosen.tolist(), "metrics": metrics})
    best = min(candidates, key=lambda r: (r["metrics"]["score"], r["indices"]))
    result = {"size": size, "feasible": True, "quota": quota.tolist(), "metrics": best["metrics"],
              "indices": best["indices"], "curve_candidates": candidates,
              "random_subsets": pop.config.random_subsets,
              "random_region_complete_fraction": float(np.mean([m["all_regions_present"] for m in baseline_metrics])),
              "random_anchor_bounds_satisfied_fraction": float(np.mean([m["anchor_bounds_satisfied"] for m in baseline_metrics]))}
    save_json(cached, result)
    return result


def anchor_matched_reference(pop, selected, output):
    """Independent draws conditional on the winner's type-by-anchor counts and all regions.

    Uniform draws within disjoint type/anchor cells followed by region rejection
    are uniform on that conditional space, not on all feasible landmark subsets.
    """
    keys = [(r["building_type"], r["anchor"]) for r in pop.rows]
    strata = sorted(set(keys))
    groups = [np.asarray([i for i, k in enumerate(keys) if k == level]) for level in strata]
    counts = [sum(keys[i] == level for i in selected) for level in strata]
    rng = seeded(pop.config, 6, len(selected))
    draws, metrics = [], []
    attempts = 0
    while len(draws) < pop.config.random_subsets:
        attempts += 1
        if attempts > pop.config.random_subsets * 1000:
            raise RuntimeError("Could not collect the requested region-complete matched reference; no partial random benchmark accepted")
        chosen = np.sort(np.concatenate([rng.choice(group, count, replace=False) for group, count in zip(groups, counts) if count]))
        if not pop.region_complete(chosen):
            continue
        if not pop.valid_selection(chosen):
            raise RuntimeError("Matched random draw violated an anchor/type constraint")
        draws.append(chosen)
        metrics.append(pop.metrics(chosen))
    columns = [key for key, value in metrics[0].items() if isinstance(value, (int, float, bool))]
    save_npz(Path(output) / "anchor_matched_random_subsets.npz", training_landmarks=np.asarray(pop.scenes),
             selected_indices=np.asarray(draws, dtype=np.int32), metric_columns=np.asarray(columns),
             metrics=np.asarray([[m[key] for key in columns] for m in metrics], dtype=float))
    return {"draws": len(draws), "attempts": attempts,
            "distinct_memberships": len({tuple(x) for x in draws}),
            "conditioning": "same building-type by anchor counts as selected subset, and all training regions present",
            "all_anchor_constraints_passed": True}


def compare_random(metrics, folder, filename="random_subsets.npz"):
    with np.load(Path(folder) / filename, allow_pickle=False) as data:
        columns, values = data["metric_columns"].tolist(), data["metrics"]
    result = {}
    for name in columns:
        if name in {"size", "unselected_count", "all_regions_present", "all_anchors_present", "anchor_bounds_satisfied"}:
            continue
        v = values[:, columns.index(name)]
        result[name] = {"selected": metrics[name], "random_median": float(np.median(v)),
                        "random_p05": float(np.quantile(v, .05)), "random_p95": float(np.quantile(v, .95)),
                        "fraction_random_strictly_worse": float(np.mean(v > metrics[name])),
                        "fraction_tied": float(np.mean(np.isclose(v, metrics[name], rtol=0, atol=1e-12)))}
    return result


def detailed_audit(pop, chosen):
    unselected = np.setdiff1d(np.arange(pop.n), chosen)
    coverage = []
    for index in unselected:
        neighbor = chosen[int(np.argmin(pop.d[index, chosen]))]
        coverage.append({"scene": pop.scenes[index], "nearest_selected": pop.scenes[neighbor],
                         "distance": float(pop.d[index, neighbor]),
                         **{k: pop.rows[index][k] for k in CATEGORIES},
                         "family_nearest_distances": {k: float(d[index, chosen].min()) for k, d in pop.families.items()}})
    coverage.sort(key=lambda r: (-r["distance"], r["scene"]))
    distributions = {}
    for fields in [(k,) for k in CATEGORIES] + [("building_type", "region"), ("building_type", "anchor")]:
        full = Counter(tuple(r[f] for f in fields) for r in pop.rows)
        sub = Counter(tuple(pop.rows[i][f] for f in fields) for i in chosen)
        distributions[" × ".join(fields)] = [{"labels": list(level), "training_count": count,
            "subset_count": sub[level], "training_share": count / pop.n, "subset_share": sub[level] / len(chosen),
            "difference_pp": 100 * (sub[level] / len(chosen) - count / pop.n)} for level, count in sorted(full.items())]
    low, high = pop.anchor_bounds(len(chosen))
    actual = np.bincount(pop.codes["anchor"][chosen], minlength=len(low))
    anchor_rows = [{"anchor": label, "training_count": int(total), "ideal_subset_count": len(chosen) * int(total) / pop.n,
                    "minimum": int(lo), "selected_count": int(count), "maximum": int(hi), "passed": bool(lo <= count <= hi)}
                   for label, total, lo, count, hi in zip(pop.levels["anchor"], pop.counts["anchor"], low, actual, high)]
    return {"metrics": pop.metrics(chosen), "coverage_unselected_landmarks": coverage,
            "anchor_constraints": {"all_passed": pop.anchors_valid(chosen), "counts": anchor_rows,
                                   "anchor_count_slack": pop.config.anchor_count_slack,
                                   "policy": pop.config.anchor_policy},
            "worst_covered": coverage[:20], "distributions": distributions,
            "omitted_levels": {field: [row["labels"] for row in values if row["subset_count"] == 0] for field, values in distributions.items()},
            "limitations": LIMITATIONS}


def write_csv(path, rows):
    import csv
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(pop, curve, decision, audit, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = decision["points"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if points:
        ax.plot([p["size"] for p in points], [p["raw_coverage"] for p in points], "o-", label="Observed coverage loss")
        ax.plot([p["size"] for p in points], [p["fitted_coverage"] for p in points], "--", label="Monotone fit")
    ax.axvline(decision["size"], color="tab:red", label=f"Chosen size: {decision['size']}")
    ax.set(xlabel="Selected training landmarks", ylabel="Mean of average and 95th-percentile distance",
           title="Coverage of unselected training landmarks (lower is better)")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "coverage_curve.png", dpi=180); plt.close(fig)
    values = audit["distributions"]["building_type"]
    y = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y - .18, [r["training_share"] * 100 for r in values], .36, label="Training pool")
    ax.barh(y + .18, [r["subset_share"] * 100 for r in values], .36, label="Chosen subset")
    ax.set(yticks=y, yticklabels=[r["labels"][0] for r in values], xlabel="Landmarks (%)", title="Building-type proportions")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "building_type_balance.png", dpi=180); plt.close(fig)


def write_report(pop, curve, decision, audit, comparison, output):
    m = audit["metrics"]
    lines = ["# Whole-landmark subset report", "", f"Selected **{m['size']} of {pop.n} training landmarks**; 79 evaluation landmarks reserved.", "",
             f"Size decision: `{decision['status']}`. The count is derived from measured coverage, not a fixed landmark budget.", "",
             "The primary comparison averages nearest-selected distances over **unselected** landmarks; selected landmarks do not contribute zero self-distances.", "",
             "## Representation metrics", "", "| Metric | Chosen subset | Random median | Random 5–95% |", "|---|---:|---:|---:|"]
    for name, row in comparison.items():
        lines.append(f"| {name} | {row['selected']:.4f} | {row['random_median']:.4f} | {row['random_p05']:.4f}–{row['random_p95']:.4f} |")
    lines += ["", f"The main table uses {pop.config.random_subsets} independent random draws conditioned on the selected subset's exact building-type by anchor counts and on all training regions appearing. Every draw satisfies the hard anchor bounds. This is a conditional reference, not uniform over all possible feasible subsets. An additional type-only random reference is retained at each tested size for comparison with version 0.1.0. A full-training fallback has no unselected-landmark coverage statistic or distinct same-size alternatives.", "",
              "## Hard anchor-count checks", "",
              f"Minimum size satisfying all hard constraints: **{decision['minimum_feasible_size']}**. All anchor checks passed: **{audit['anchor_constraints']['all_passed']}**.", "",
              f"Every named anchor and the separate missing-label category must be present. Anchor counts may extend **{pop.config.anchor_count_slack} landmark(s)** below the floor or above the ceiling of their proportional target, capped by availability and with a minimum of one. Building-type quotas, region presence, the coverage size rule and the evaluation split are unchanged.", "",
              "| Anchor label | Training count | Proportional target | Allowed count | Selected |", "|---|---:|---:|---:|---:|"]
    for row in audit["anchor_constraints"]["counts"]:
        lines.append(f"| {row['anchor']} | {row['training_count']} | {row['ideal_subset_count']:.2f} | {row['minimum']}–{row['maximum']} | {row['selected_count']} |")
    lines += ["",
              "## Gaps and interpretation", ""] + [f"- {s}" for s in LIMITATIONS]
    lines += ["", "Omitted anchor labels: " + (", ".join(x[0] for x in audit["omitted_levels"]["anchor"]) or "none") + ".", "",
              "## Worst covered unselected landmarks", "", "| Landmark | Closest selected landmark | Distance |", "|---|---|---:|"]
    for row in audit["worst_covered"]:
        lines.append(f"| {row['scene']} | {row['nearest_selected']} | {row['distance']:.4f} |")
    lines += ["", "## Outputs", "", "`subset_manifest.json` and `selected_landmarks.json` define membership. `selected_image_manifests/` preserves every image path/hash. `evaluation_landmarks.json` is the unchanged held-out ID list. `audit.json`, `size_decision.json`, `size_curve.csv` and candidate/random manifests document selection. No raw photographs are copied.", "",
              "![Coverage curve](coverage_curve.png)", "", "![Building type balance](building_type_balance.png)", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


@contextmanager
def output_lock(output):
    lock = output / ".selector.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Another selector may be running. If a previous session was terminated, remove {lock} only after confirming it stopped.") from None
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_selection(encodings, output_dir, config=None, inputs_dir=HERE / "inputs"):
    config = config or SelectionConfig()
    inputs = load_inputs(inputs_dir)
    train, evaluation, metadata, sources, checks = inputs
    encodings, root = Path(encodings).resolve(), Path(output_dir).resolve()
    if root == encodings or root.is_relative_to(encodings) or encodings.is_relative_to(root):
        raise ValueError("Selection outputs must be separate from encoder outputs")
    check_completion(encodings, train, evaluation)
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root):
        rows, means, stds, provenance = load_descriptors(encodings, inputs, root)
        import scipy
        protocol = {"version": VERSION, "selector_sha256": sha(__file__), "config": asdict(config),
                    "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
                    "provenance": provenance, "reference": "317_training_landmarks_only", "image_policy": "all_original_images",
                    "population_weighting": "one_equal_weight_per_landmark", "distance_scaling": "positive training-pair median per family",
                    "score": "0.35*unselected_mean_distance + 0.35*unselected_p95_distance + 0.15*mean_marginal_TV + 0.15*mean_numeric_KS",
                    "type_rounding": "minimum squared deviation from ideal type counts, minimum one per type; ties in sorted type order",
                    "anchor_hard_bounds": config.anchor_policy,
                    "anchor_failure_policy": "Reject infeasible sizes and swaps; never exceed configured slack or omit a label",
                    "size_rule": "geometric size grid + local refinement; isotonic decreasing coverage; maximal normalized gain above endpoint chord; full training if no clear elbow",
                    "random_baseline": "main: uniform within selected type-by-anchor cell counts, conditional on all regions; secondary: type-only stratified reference at every tested size",
                    "limitations": LIMITATIONS}
        run_id = signature(protocol)[:16]
        output = root / f"run_{run_id}"
        output.mkdir(exist_ok=True)
        save_json(output / "selection_protocol.json", protocol)
        required = ("result.json", "selected_landmarks.json", "subset_manifest.json", "audit.json", "REPORT.md",
                    "evaluation_landmarks.json", "size_decision.json", "coverage_curve.png", "building_type_balance.png")
        if all((output / name).is_file() for name in required):
            emit(f"Reusing completed selection: {output}")
            result = read_json(output / "result.json")
            selected = read_json(output / "selected_landmarks.json")
            if (selected and len(set(selected)) == len(selected) == result["selected_landmarks"]
                    and set(selected) <= set(train)
                    and (output / "evaluation_landmarks.json").read_bytes() == (Path(inputs_dir) / "evaluation_landmarks.json").read_bytes()
                    and all((output / "selected_image_manifests" / f"{scene}.json").is_file()
                            and sha(output / "selected_image_manifests" / f"{scene}.json") == provenance["landmarks"][scene]["manifest.json"]
                            for scene in selected)):
                save_json(root / "latest_result.json", result)
                return result
        pop = Population(rows, means, stds, config)
        save_json(output / "training_descriptors.json", rows)
        save_npz(output / "landmark_descriptors.npz", training_landmarks=np.asarray(train), means=means, stds=stds,
                 composite_distance=pop.d, **{k + "_distance": v for k, v in pop.families.items()})
        save_json(output / "distance_scales.json", pop.scales)
        low = pop.minimum_feasible_size()
        emit(f"Minimum size satisfying building-type, region and anchor constraints: {low}")
        sizes = sorted(set(np.rint(np.geomspace(low, pop.n - 1, config.curve_points)).astype(int).tolist())) if low < pop.n else []
        curve = []
        for size in sizes:
            emit(f"Coverage study: {size} landmarks; {config.random_subsets} random comparisons")
            curve.append(study_size(pop, size, output / "sizes" / str(size)))
        provisional = choose_size(curve, pop.n, config.knee_min_gain)
        if provisional["size"] < pop.n:
            index = sizes.index(provisional["size"])
            refinements = sorted(set(np.rint(np.linspace(sizes[max(0, index - 1)], sizes[min(len(sizes) - 1, index + 1)], config.refinement_points)).astype(int)) - set(sizes))
            for size in refinements:
                emit(f"Refining coverage: {size} landmarks")
                curve.append(study_size(pop, int(size), output / "sizes" / str(size)))
        curve.sort(key=lambda r: r["size"])
        decision = choose_size(curve, pop.n, config.knee_min_gain)
        decision["minimum_feasible_size"] = low
        decision["sensitivity"] = {label: choose_size(curve, pop.n, config.knee_min_gain, weight)["size"]
                                   for label, weight in (("mean_only", 0), ("mean_and_tail", .5), ("tail_only", 1))}
        decision["evaluated_sizes"] = [r["size"] for r in curve]
        save_json(output / "size_decision.json", decision)
        save_json(output / "size_curve.json", curve)
        write_csv(output / "size_curve.csv", [{"size": r["size"], "feasible": r["feasible"],
            "coverage_mean": r.get("metrics", {}).get("coverage_mean"), "coverage_p95": r.get("metrics", {}).get("coverage_p95"),
            "coverage_worst": r.get("metrics", {}).get("coverage_worst"), "score": r.get("metrics", {}).get("score")} for r in curve])
        size = decision["size"]
        candidates = []
        if size == pop.n:
            chosen = np.arange(pop.n)
            comparison = {}
            type_only_comparison = {}
            matched_reference = {"draws": 0, "reason": "full_training_population"}
        else:
            best_curve = next(r for r in curve if r["size"] == size)
            quota = pop.allocation(size)
            for i in range(config.final_candidates):
                rng = seeded(config, 4, size, i)
                initial = np.asarray(best_curve["indices"]) if i == 0 else pop.feasible_start(quota, rng)
                selected, metrics = pop.improve(initial, rng, config.swap_attempts)
                candidate = {"candidate": i, "indices": selected.tolist(), "landmarks": [train[j] for j in selected], "metrics": metrics}
                candidates.append(candidate)
                save_json(output / "candidates" / f"candidate_{i:03d}.json", candidate)
            winner = min(candidates, key=lambda r: (r["metrics"]["score"], r["indices"]))
            chosen = np.asarray(winner["indices"], dtype=int)
            type_only_comparison = compare_random(winner["metrics"], output / "sizes" / str(size))
            emit("Comparing against anchor-preserving random subsets")
            matched_reference = anchor_matched_reference(pop, chosen, output)
            comparison = compare_random(winner["metrics"], output, "anchor_matched_random_subsets.npz")
        if not pop.valid_selection(chosen):
            raise RuntimeError("Refusing to export a subset that violates type, region or anchor constraints")
        audit = detailed_audit(pop, chosen)
        audit["random_comparison"] = comparison
        audit["matched_random_reference"] = matched_reference
        audit["type_only_random_comparison"] = type_only_comparison
        audit["distinct_final_candidates"] = len({tuple(c["indices"]) for c in candidates})
        audit["candidate_count"] = len(candidates)
        save_json(output / "audit.json", audit)
        write_csv(output / "anchor_counts.csv", audit["anchor_constraints"]["counts"])
        selected_scenes = [train[i] for i in chosen]
        manifest = {"training_population": 317, "evaluation_holdout": 79, "image_policy": "all_original_images",
                    "selected_landmarks": selected_scenes, "selected_count": len(chosen), "size_decision": decision["status"],
                    "image_count": sum(rows[i]["image_count"] for i in chosen), "archives": {s: sources[s] for s in selected_scenes},
                    "dataset_revision": provenance["encoder_settings"]["dataset_revision"],
                    "encoder_settings_sha256": signature(provenance["encoder_settings"]),
                    "selection_run": run_id, "representation_reference": "training_only", "limitations": LIMITATIONS}
        manifest["anchor_constraints"] = audit["anchor_constraints"]
        for scene in selected_scenes:
            dest = output / "selected_image_manifests" / f"{scene}.json"
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(encodings / "landmarks" / scene / "manifest.json", dest)
        save_json(output / "subset_manifest.json", manifest)
        save_json(output / "selected_landmarks.json", selected_scenes)
        shutil.copyfile(Path(inputs_dir) / "evaluation_landmarks.json", output / "evaluation_landmarks.json")
        write_csv(output / "selected_landmarks.csv", [rows[i] for i in chosen])
        figures(pop, curve, decision, audit, output)
        write_report(pop, curve, decision, audit, comparison, output)
        result = {"output_dir": str(output), "selected_landmarks": len(chosen), "selected_images": manifest["image_count"],
                  "evaluation_landmarks": 79, "size_status": decision["status"], "report": str(output / "REPORT.md"),
                  "all_anchor_constraints_passed": audit["anchor_constraints"]["all_passed"],
                  "minimum_feasible_size": low,
                  "interventions_run": False, "selection_complete": True}
        save_json(output / "result.json", result)
        save_json(root / "latest_result.json", result)
        emit(f"Saved {len(chosen)} complete landmarks and audit: {output}")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encodings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON object of SelectionConfig overrides")
    args = parser.parse_args(argv)
    config = SelectionConfig(**read_json(args.config)) if args.config else SelectionConfig()
    print(json.dumps(run_selection(args.encodings, args.out, config), indent=2))


if __name__ == "__main__":
    main()
