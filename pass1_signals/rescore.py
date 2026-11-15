#!/usr/bin/env python
"""Score saved VGGT predictions (inter_raw32.npz, written by seeded_run.py) exactly as the pass scores them, with a chosen seed.

The pass (intervention/models/vggt/run_pass.py) scores a prediction by aligning it to the ground
truth (random correspondence subsampling + RANSAC, from NumPy's global generator, then ICP) and
measuring Chamfer distance. It never seeds the generator, so the same prediction scores differently
each run. This script rebuilds the pass's scoring inputs from a run's work dir (image list,
scene_info.json, validity masks, ground truth by unprojection) with the pass's own functions, seeds
NumPy right before the alignment, and repeats only the scoring. No repository file is changed.

Differences from the in-run scoring, none of which changes the score:
  - the ground truth and its KD-tree are built once per scene and reused for every eta and draw;
  - no GLB export;
  - KD-trees are built unbalanced (balanced_tree=False, compact_nodes=False): identical nearest-neighbour
    distances (checked on Paul_Kruger_Church), queries ~10x faster on these clouds;
  - predictions are read from inter_raw32.npz (fp32), not the pass's inter_raw.npz: its fp16 confidences
    round values in (1, ~1.0005] to 1.0, and the scoring's confidence filter (> ~1.00001) then drops those pixels.
Also recorded per draw (alignment fit quality): open3d fitness and inlier RMSE of the coarse
(RANSAC) alignment and after ICP, at the ICP threshold.

Modes
  draws  --raw-root R --scene S --etas 0 1000 --seeds 0 1 2      score these draws (skip existing)
  label  --raw-root R --scene S --etas ... --scheme scheme.json     label every (scene, eta) with a raw file
                                                                    and no label yet, drawing seeds as needed
  collect --labels-dir L --out labels.csv                          all labels -> one CSV (swati's format)
  choose --draws D --scenes A B C --out scheme.json                phase-0 gate and scheme choice

Draw files: <draws dir>/<scene>/eta<eta>_s<seed>.json
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The pass's configuration, exactly as intervention/run.py sets it, before the pass is imported
# (it reads the environment at import time).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # CPU only; the GPU belongs to the GPU loop
os.environ.update({
    "NUM_IMAGES": "50",
    "USE_ICP": "1",
    "ICP_THRESHOLD": "0.1",
    "USE_VALIDITY_MASKS": "1",
    "VALIDITY_MASKS_BASE": str(REPO / "data" / "manual_filtration_masks"),
    "IMAGE_MANIFEST": "",
})

import numpy as np  # noqa: E402

FAIL_RATIO = 1.15          # a draw scoring > 1.15 x its scene's median counts as a failed alignment
AGREE = 0.03               # scheme A: two draws within 3% are averaged
N_PHASE0 = 8


def _pass():
    sys.path[:0] = [str(REPO / "intervention" / "models" / "vggt"), str(REPO / "eval"),
                    str(REPO / "intervention"), str(REPO)]
    import importlib
    return importlib.import_module("models.vggt.run_pass")


# ----------------------------------------------------------------------------- scene inputs
class Scene:
    """Scoring inputs shared by every eta of one scene (built once)."""

    def __init__(self, rp, scene, work_dir):
        t0 = time.time()
        self.rp, self.scene, work_dir = rp, scene, Path(work_dir)
        script_dir = Path(rp.__file__).resolve().parent
        self.image_paths = rp.get_image_paths(work_dir, scene_name_override=scene, csv_script_dir=script_dir)
        self.vmasks = rp.load_validity_masks(scene, self.image_paths) if rp.VALIDITY_MASKS_BASE else {}
        info = json.loads((work_dir / "scene_info.json").read_text(encoding="utf-8"))
        image_folder = info.get("image_folder", str(work_dir / rp.IMAGES_DIR))
        if not Path(image_folder).is_dir():          # e.g. the run folder was moved since
            image_folder = str(work_dir / rp.IMAGES_DIR)
        depth_candidates = [info["depth_folder"], str(work_dir / "depth")]
        tmp_glb = str(work_dir / f".rescore_{scene}_gt.glb")
        res = rp.generate_gt_by_unproject(self.image_paths, image_folder, depth_candidates, info["recon_path"],
                                          tmp_glb, verbose=os.environ.get("RESCORE_VERBOSE") == "1", validity_masks_by_path=self.vmasks)
        if Path(tmp_glb).exists():
            Path(tmp_glb).unlink()
        if res is None or res[0] is None or len(res[0]) < 100:
            raise RuntimeError(f"{scene}: ground truth could not be built")
        self.gt_points, self.gt_samples = res
        self.gt_samples.sort(key=lambda x: x["image_path"])
        from scipy.spatial import cKDTree
        self.gt_tree = cKDTree(self.gt_points, balanced_tree=False, compact_nodes=False)  # same distances, ~10x faster queries here
        self.use_masked = bool(self.vmasks) or any(
            isinstance(s, dict) and s.get("validity_mask") is not None for s in self.gt_samples)
        self.build_s = time.time() - t0


def load_pred(npz_path):
    import torch
    z = np.load(npz_path)
    arrays = {k: z[k] for k in ("depth_map", "depth_conf", "extrinsics", "intrinsics")}
    h = hashlib.sha1()
    for k in sorted(arrays):
        h.update(k.encode()); h.update(np.ascontiguousarray(arrays[k]).tobytes())
    pred = {k: torch.from_numpy(v.astype(np.float32)) for k, v in arrays.items()}
    return pred, h.hexdigest()


def score_once(sc, pred, seed):
    """One draw: the pass's _run_alignment_and_eval_worker scoring steps, seeded, with timings."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    rp, t = sc.rp, {}
    np.random.seed(seed)
    t0 = time.time()
    if sc.use_masked:
        c, R, tr = rp.compute_alignment_with_correspondence_masked(
            sc.gt_samples, pred, "", max_total_points=None, n_iterations=1000,
            image_paths=sc.image_paths, validity_masks_by_path=sc.vmasks)
    else:
        c, R, tr = rp.compute_alignment_with_correspondence(
            sc.gt_samples, pred, "", max_total_points=None, n_iterations=1000, image_paths=sc.image_paths)
    t["align_s"] = time.time() - t0
    if c is None or not np.isfinite(c):
        return {"status": "alignment_failed", **t}
    t0 = time.time()
    if sc.use_masked:
        pts = rp.get_pred_points_masked(pred, sc.image_paths, sc.vmasks)
    else:
        pts = rp.get_pred_points_clean(pred, sc.image_paths, cache_dir=str(Path("/tmp") / f"rescore_{os.getpid()}"))
    if pts is None:
        return {"status": "no_pred_points", **t}
    aligned = (c * R @ pts.T + tr).T
    thr = float(os.environ["ICP_THRESHOLD"])
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(np.asarray(aligned, dtype=np.float64))
    pcd_gt = o3d.geometry.PointCloud(); pcd_gt.points = o3d.utility.Vector3dVector(np.asarray(sc.gt_points, dtype=np.float64))
    coarse = o3d.pipelines.registration.evaluate_registration(pcd, pcd_gt, thr, np.eye(4))
    reg = o3d.pipelines.registration.registration_icp(
        pcd, pcd_gt, thr, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
    T = np.asarray(reg.transformation)
    aligned = (aligned @ T[:3, :3].T) + T[:3, 3]
    t["icp_s"] = time.time() - t0
    t0 = time.time()
    d_acc, _ = sc.gt_tree.query(aligned, workers=-1)               # eval_utils.accuracy, cached tree
    d_comp, _ = cKDTree(aligned, balanced_tree=False, compact_nodes=False).query(sc.gt_points, workers=-1)   # eval_utils.completion
    acc, comp = float(np.mean(d_acc)), float(np.mean(d_comp))
    t["chamfer_s"] = time.time() - t0
    return {"status": "ok", "Acc-mean": acc, "Acc-med": float(np.median(d_acc)),
            "Comp-mean": comp, "Comp-med": float(np.median(d_comp)), "Score": (acc + comp) / 2,
            "coarse_fitness": coarse.fitness, "coarse_rmse": coarse.inlier_rmse,
            "icp_fitness": reg.fitness, "icp_rmse": reg.inlier_rmse, **t}


# ----------------------------------------------------------------------------- draws on disk
class Draws:
    def __init__(self, raw_root, draws_dir, scene, rp=None):
        self.raw_root, self.dir, self.scene = Path(raw_root), Path(draws_dir) / scene, scene
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rp, self.sc, self._pred = rp, None, {}

    def path(self, eta, seed):
        return self.dir / f"eta{eta}_s{seed}.json"

    def get(self, eta, seed):
        p = self.path(eta, seed)
        if p.exists():
            return json.loads(p.read_text())
        rp = self.rp or _pass()
        self.rp = rp
        npz = self.raw_root / f"output_{eta}" / self.scene / "inter_raw32.npz"
        if eta not in self._pred:
            self._pred[eta] = load_pred(npz)
        pred, digest = self._pred[eta]
        # identical prediction already scored at another eta with this seed -> reuse
        for q in self.dir.glob(f"eta*_s{seed}.json"):
            d = json.loads(q.read_text())
            if d.get("pred_sha1") == digest and d.get("status") == "ok":
                d.update(eta=eta, reused_from=d.get("eta"))
                p.write_text(json.dumps(d, indent=1))
                return d
        if self.sc is None:
            self.sc = Scene(rp, self.scene, npz.parent)
        t0 = time.time()
        d = score_once(self.sc, pred, seed)
        d.update(scene=self.scene, eta=eta, seed=seed, pred_sha1=digest, total_s=time.time() - t0,
                 gt_build_s=self.sc.build_s)
        p.write_text(json.dumps(d, indent=1))
        print(f"  {self.scene} eta={eta} seed={seed}: {d['status']} "
              f"Score={d.get('Score', float('nan')):.5f} ({d['total_s']:.0f}s)", flush=True)
        return d


# ----------------------------------------------------------------------------- labelling schemes
def label_one(draws, eta, scheme):
    """Apply the scheme to one (scene, eta). Returns (label dict, list of draws used)."""
    def ok(d):
        return d["status"] == "ok"
    if scheme["scheme"] == "F":
        m, cut, fail_low = scheme["measure"], scheme["cutoff"], scheme["fail_is_low"]
        used = [draws.get(eta, 0)]
        looks_failed = ok(used[0]) and ((used[0][m] < cut) if fail_low else (used[0][m] > cut))
        if not ok(used[0]) or looks_failed:
            used += [draws.get(eta, 1), draws.get(eta, 2)]
    else:
        used = [draws.get(eta, 0), draws.get(eta, 1)]
        good = [d for d in used if ok(d)]
        if not (len(good) == 2 and abs(good[0]["Score"] - good[1]["Score"]) / min(g["Score"] for g in good) <= AGREE):
            used.append(draws.get(eta, 2))
    good = [d for d in used if ok(d)]
    if not good:
        return None, used
    keys = ["Acc-mean", "Acc-med", "Comp-mean", "Comp-med"]
    if len(good) == 2:            # two agreeing draws (scheme A): mean
        lab = {k: float(np.mean([d[k] for d in good])) for k in keys + ["Score"]}
    else:                          # one draw, or the median of three (the draw with the median Score)
        mid = sorted(good, key=lambda d: d["Score"])[len(good) // 2]
        lab = {k: mid[k] for k in keys + ["Score"]}
    return lab, used


def cmd_label(args):
    """Label each (scene, eta) whose raw file exists and has no label yet: <labels>/<scene>/eta<eta>.json."""
    scheme = json.loads(Path(args.scheme).read_text())
    dr = Draws(args.raw_root, args.draws, args.scene)
    out_dir = Path(args.labels_dir) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bad = 0
    for eta in args.etas:
        out = out_dir / f"eta{eta}.json"
        if out.exists() or not (dr.raw_root / f"output_{eta}" / args.scene / "inter_raw32.npz").exists():
            continue
        try:
            lab, used = label_one(dr, eta, scheme)
        except Exception as e:                      # record and move on; never retried forever
            import traceback; traceback.print_exc()
            out.write_text(json.dumps({"scene": args.scene, "eta": eta, "status": f"error: {e}"}, indent=1))
            n_bad += 1
            continue
        row = {"scene": args.scene, "eta": eta, "n_draws": len(used),
               "draw_scores": " ".join(f"{d.get('Score', float('nan')):.6g}" for d in used)}
        if lab is None:
            row["status"] = "no_successful_draw"; n_bad += 1
        else:
            row.update(lab, status="ok")
        out.write_text(json.dumps(row, indent=1))
        print(f"{args.scene} eta={eta}: {row['status']} Score={row.get('Score', float('nan')):.5f} "
              f"({len(used)} draws)", flush=True)
    return 1 if n_bad else 0


def cmd_collect(args):
    """All label files -> one CSV in swati's per_scene.csv format (+ n_draws, draw_scores)."""
    rows = [json.loads(p.read_text()) for p in sorted(Path(args.labels_dir).glob("*/eta*.json"))]
    rows = [r for r in rows if r.get("status") == "ok"]
    cols = ["scene", "eta", "Acc-mean", "Acc-med", "Comp-mean", "Comp-med", "Score", "n_draws", "draw_scores"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["scene"], r["eta"])))
    print(f"{len(rows)} labels -> {args.out}")


def cmd_draws(args):
    dr = Draws(args.raw_root, args.draws, args.scene)
    for eta in args.etas:
        for s in args.seeds:
            dr.get(eta, s)
    if args.repeat_seed is not None:               # determinism check: score a seed again, no cache
        eta = args.etas[0]
        pred, _ = dr._pred.get(eta) or load_pred(dr.raw_root / f"output_{eta}" / args.scene / "inter_raw32.npz")
        d = score_once(dr.sc or Scene(dr.rp or _pass(), args.scene,
                                      dr.raw_root / f"output_{eta}" / args.scene), pred, args.repeat_seed)
        (dr.dir / f"eta{eta}_s{args.repeat_seed}_repeat.json").write_text(json.dumps(d, indent=1))


# ----------------------------------------------------------------------------- phase-0 gate
PROBE_RANGE = {"Paul_Kruger_Church": (0.180, 0.239), "Shuri_Castle": (0.486, 0.671), "Dendera": (0.038, 0.040)}
MEASURES = ["coarse_fitness", "coarse_rmse", "icp_fitness", "icp_rmse"]


def simulate_label(scores, measure_vals, scheme, rng, n=2000):
    """Distribution of the scheme's label when draws are resampled from the observed ones."""
    out, idx = [], np.arange(len(scores))
    for _ in range(n):
        pick = rng.choice(idx, 3, replace=True)
        s = scores[pick]
        if scheme["scheme"] == "F":
            v = measure_vals[pick[0]]
            failed = (v < scheme["cutoff"]) if scheme["fail_is_low"] else (v > scheme["cutoff"])
            out.append(np.median(s) if failed else s[0])
        else:
            out.append(np.mean(s[:2]) if abs(s[0] - s[1]) / min(s[:2]) <= AGREE else np.median(s))
    return np.array(out)


def cmd_choose(args):
    rep = {"gate": "pass", "checks": [], "scheme": "A"}
    data = {}
    for sc in args.scenes:
        d = [json.loads(p.read_text()) for p in sorted((Path(args.draws) / sc).glob("eta0_s*.json"))
             if "_repeat" not in p.name]
        d = [x for x in d if x["status"] == "ok"]
        data[sc] = d
        scores = np.array([x["Score"] for x in d])
        med = float(np.median(scores)) if len(scores) else float("nan")
        lo, hi = PROBE_RANGE[sc]
        inside = lo * 0.95 <= med <= hi * 1.05
        rep["checks"].append({"scene": sc, "n_ok": len(d), "median": med, "range": [lo, hi], "inside": inside,
                              "scores": [round(float(s), 6) for s in scores]})
        if len(d) < N_PHASE0 or not inside:
            rep["gate"] = "fail"
        rp_ = Path(args.draws) / sc / "eta0_s0_repeat.json"
        if rp_.exists():
            r = json.loads(rp_.read_text())
            s0 = next((x for x in d if x["seed"] == 0), None)
            same = s0 is not None and r.get("Score") == s0["Score"]
            rep["checks"].append({"scene": sc, "determinism": same})
            if not same:
                rep["gate"] = "fail"
    # failures and fit-quality separation
    fails, goods = [], []
    for sc, d in data.items():
        med = np.median([x["Score"] for x in d])
        for x in d:
            (fails if x["Score"] > FAIL_RATIO * med else goods).append(x)
    rep["n_failed_draws"], rep["n_good_draws"] = len(fails), len(goods)
    best = None
    if len(fails) >= 2 and goods:
        for m in MEASURES:
            f, g = np.array([x[m] for x in fails]), np.array([x[m] for x in goods])
            if f.max() < g.min():
                gap, cut, low = g.min() - f.max(), (f.max() + g.min()) / 2, True
            elif f.min() > g.max():
                gap, cut, low = f.min() - g.max(), (f.min() + g.max()) / 2, False
            else:
                continue
            rel = gap / (np.abs(np.concatenate([f, g])).mean() + 1e-12)
            if best is None or rel > best[0]:
                best = (rel, m, float(cut), low)
    if best:
        rep.update(scheme="F", measure=best[1], cutoff=best[2], fail_is_low=best[3])
    # noise floor: spread of the chosen scheme's label, by resampling, largest over scenes
    rng = np.random.default_rng(0)
    floors = {}
    for sc, d in data.items():
        s = np.array([x["Score"] for x in d])
        mv = np.array([x[rep.get("measure", "icp_fitness")] for x in d])
        sim = simulate_label(s, mv, rep, rng)
        floors[sc] = float(np.std(sim) / np.median(sim))
    rep["noise_floor_by_scene"], rep["noise_floor"] = floors, max(floors.values()) if floors else None
    times = [x["total_s"] for d in data.values() for x in d if not x.get("reused_from")]
    rep["median_draw_s"] = float(np.median(times)) if times else None
    rep["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("draws", "label"):
        p = sub.add_parser(name)
        p.add_argument("--raw-root", required=True)
        p.add_argument("--draws", required=True, help="directory of draw files")
        p.add_argument("--scene", required=True)
        p.add_argument("--etas", type=int, nargs="+", required=True)
        if name == "draws":
            p.add_argument("--seeds", type=int, nargs="+", required=True)
            p.add_argument("--repeat-seed", type=int)
        else:
            p.add_argument("--scheme", required=True)
            p.add_argument("--labels-dir", required=True)
    p = sub.add_parser("choose")
    p.add_argument("--draws", required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"draws": cmd_draws, "label": cmd_label, "choose": cmd_choose, "collect": cmd_collect}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
