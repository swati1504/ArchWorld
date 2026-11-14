#!/usr/bin/env python
"""Pass 1 of ArchAttend on VGGT, instrumented: one forward pass per scene with
hooks on every global attention layer, recording the raw signals that the
step-size predictor will be built from.

Per scene it writes

    <out>/<scene>.json   per element and layer: SCVA (several definitions), token counts,
                         the anchor each definition picks, and output-based statistics
    <out>/<scene>.npz    per query token and layer: attention mass on architectural /
                         non-architectural / special tokens (same-view vs cross-view),
                         per non-architectural class, and entropy for a sample of queries

Nothing here is a summary feature yet; `features.py` (to come) turns these into
per-scene features, so definitions can change without re-running the GPU.

SCVA definitions (A = head-averaged attention at one global layer):
  code   sum over view pairs i != j of mean_{p in P_e^i, q in P_e^j} A[p,q]
         -- what intervention/models/vggt/run_pass.py:compute_feature_attention_sum does
  eq     sum over all p, q in P_e of A[p,q]                         -- the paper's Eq. 1-2
  eqx    as eq, restricted to p and q in different views
Each is computed twice: with RoPE (the attention the model really uses) and
without (what run_pass.py's Pass-1 hook computes).

The N x N attention matrix is never formed: passing a 0/1 indicator matrix as the
value tensor to scaled_dot_product_attention returns each query's attention mass
on the indicated keys (the trick in intervention/models/vggt/loss.py).

    conda activate archworld
    python pass1_signals/extract_features.py --root data --masks-dir pass1_out/masks \
        --out pass1_out/features --scene Dendera
"""
# --- imports resolve like the passes: repo root, eval/, and the VGGT pass dir ---
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "eval", REPO / "intervention" / "models" / "vggt", REPO / "intervention",
           Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# -------------------------------------------------------------------------------
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from run_pass import map_mask_to_patches, PATCH_SIZE, PATCH_START_IDX  # the pass's own mask->patch mapping
from make_masks_full import NON_ARCH

ASSETS = REPO / "intervention" / "assets"


def load_asset(name):
    return json.loads((ASSETS / name).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- masks
def load_patch_sets(mask_dir, stems, image_paths, elements, Hp, Wp):
    """patch_sets[e][i] = sorted patch indices of element e in view i (same rule as run_pass)."""
    out = {}
    for e in elements:
        per_view = []
        for stem, img in zip(stems, image_paths):
            f = mask_dir / f"{stem}_{e}.npy"
            if not f.exists():
                per_view.append(None)            # never segmented (distinct from "found nothing")
                continue
            arr = np.load(f)
            if arr.ndim == 4:
                arr = arr.sum(axis=(0, 1)) > 0
            elif arr.ndim == 3:
                arr = arr.sum(axis=0) > 0
            per_view.append(sorted(map_mask_to_patches(arr.astype(bool), Hp, Wp, image_path=str(img))))
        out[e] = per_view
    return out


def token_index(patch_sets_e, P):
    """Global token indices and their view ids for one element."""
    idx, view = [], []
    for i, ps in enumerate(patch_sets_e):
        if ps:
            idx.extend(i * P + PATCH_START_IDX + p for p in ps)
            view.extend([i] * len(ps))
    return np.asarray(idx, dtype=np.int64), np.asarray(view, dtype=np.int64)


# ----------------------------------------------------------------------------- attention math
def mass_on(q, k, V):
    """q (1,H,n,D), k (1,H,N,D), V (N,c) 0/1 -> (n,c): head-averaged attention mass on V's columns."""
    H, D = q.shape[1], q.shape[-1]
    out = []
    for s in range(0, V.shape[1], D):            # value width must equal head dim for the fused kernels
        Vc = V[:, s:s + D]
        c = Vc.shape[1]
        if c < D:
            Vc = F.pad(Vc, (0, D - c))
        Vc = Vc.to(q.dtype)[None, None].expand(1, H, -1, -1).contiguous()
        o = F.scaled_dot_product_attention(q, k, Vc)      # default scale = D**-0.5, as the model
        out.append(o[0].float().mean(0)[:, :c])
    return torch.cat(out, 1)


def entropy_stats(q, k, chunk=128):
    """For queries q (1,H,n,D): per-query mean-over-heads entropy, entropy of the head-mean
    distribution, and head-mean top-1 mass. Entropies in nats."""
    H, D = q.shape[1], q.shape[-1]
    scale = D ** -0.5
    ent_h, ent_m, top1 = [], [], []
    for s in range(0, q.shape[2], chunk):
        z = (q[0, :, s:s + chunk].float() @ k[0].float().transpose(-1, -2)) * scale   # (H,c,N)
        logp = torch.log_softmax(z, dim=-1)
        p = logp.exp()
        ent_h.append(-(p * logp).sum(-1).mean(0))
        pm = p.mean(0)
        ent_m.append(-(pm * torch.log(pm.clamp_min(1e-30))).sum(-1))
        top1.append(pm.max(-1).values)
        del z, logp, p, pm
    return torch.cat(ent_h), torch.cat(ent_m), torch.cat(top1)


# ----------------------------------------------------------------------------- per-scene run
def run_scene(model, scene, root, mask_root, out_dir, dtype, n_entropy, seed=0):
    t0 = time.time()
    views = load_asset("views.json")[scene]
    arch = list(load_asset("elements.json")[scene])
    nonarch = [e for e in NON_ARCH if e not in arch]
    shipped_anchor = load_asset("anchors.json").get(scene)

    img_dir = Path(root) / "archworld" / scene / "images"
    image_paths = [img_dir / v for v in views]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{scene}: {len(missing)} views missing (first {missing[0]})")
    stems = [Path(v).stem for v in views]

    images = load_and_preprocess_images([str(p) for p in image_paths]).to("cuda")[None]
    S, Himg, Wimg = images.shape[1], images.shape[-2], images.shape[-1]
    Hp, Wp = Himg // PATCH_SIZE, Wimg // PATCH_SIZE
    P = PATCH_START_IDX + Hp * Wp
    N = S * P
    assert S <= 64, "per-view value columns assume at most 64 views"

    mask_dir = Path(mask_root) / scene / "mask"
    patch_sets = load_patch_sets(mask_dir, stems, image_paths, arch + nonarch, Hp, Wp)
    never = [e for e, pv in patch_sets.items() if all(x is None for x in pv)]
    if never:
        raise FileNotFoundError(f"{scene}: no mask files at all for {never} under {mask_dir}")

    view_of = torch.arange(N, device="cuda") // P
    special = torch.zeros(N, device="cuda")
    for i in range(S):
        special[i * P:i * P + PATCH_START_IDX] = 1

    def union(names):
        m = torch.zeros(N, device="cuda")
        for e in names:
            idx, _ = token_index(patch_sets[e], P)
            if len(idx):
                m[torch.as_tensor(idx, device="cuda")] = 1
        return m

    arch_u, nonarch_u = union(arch), union(nonarch)
    onehot_view = F.one_hot(view_of, S).float()                         # (N,S)
    # group x view columns: mass split by key group and by the key's view
    G = torch.cat([onehot_view * arch_u[:, None], onehot_view * nonarch_u[:, None],
                   onehot_view * special[:, None]], 1)                  # (N, 3S)
    per_class = torch.stack([union([e]) for e in nonarch], 1) if nonarch else None

    elem_idx = {e: token_index(patch_sets[e], P) for e in arch + nonarch}
    off = ~torch.eye(S, dtype=torch.bool, device="cuda")
    elem_gpu = {}                     # per element: query index, query view, key indicator split by view
    for e, (idx, vw) in elem_idx.items():
        if len(idx) == 0:
            continue
        it = torch.as_tensor(idx, device="cuda")
        Ve = onehot_view * torch.zeros(N, device="cuda").index_fill_(0, it, 1)[:, None]
        vw_t = torch.as_tensor(vw, device="cuda")
        elem_gpu[e] = (it, vw_t, Ve, torch.bincount(vw_t, minlength=S).float())

    rng = np.random.default_rng(seed)
    patch_tokens = np.array([t for t in range(N) if t % P >= PATCH_START_IDX])
    ent_q = np.sort(rng.choice(patch_tokens, size=min(n_entropy, len(patch_tokens)), replace=False))
    ent_q_t = torch.as_tensor(ent_q, device="cuda")

    rec = {"elements": {}, "per_query": {}}
    layers = range(len(model.aggregator.global_blocks))

    def make_hook(l):
        def hook(module, args, kwargs, output):
            x = args[0]
            pos = kwargs.get("pos", args[1] if len(args) > 1 else None)
            B_, N_, C = x.shape
            qkv = module.qkv(x).reshape(B_, N_, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q0, k0, _ = qkv.unbind(0)
            q0, k0 = module.q_norm(q0), module.k_norm(k0)
            variants = {"norope": (q0, k0)}
            if module.rope is not None:
                variants["rope"] = (module.rope(q0, pos), module.rope(k0, pos))
            else:
                variants["rope"] = variants["norope"]

            # 1. SCVA per element, both variants
            for var, (q, k) in variants.items():
                for e, (it, vw_t, Ve, n) in elem_gpu.items():
                    m = mass_on(q[:, :, it], k, Ve)                          # (n_e, S)
                    M = torch.zeros(S, S, device="cuda").index_add_(0, vw_t, m)
                    pair = (n[:, None] > 0) & (n[None, :] > 0) & off
                    code = (M / (n[:, None] * n[None, :]).clamp_min(1))[pair].sum()
                    r = rec["elements"].setdefault(e, {}).setdefault(var, {})
                    r[l] = {"code": code.item(), "eq": M.sum().item(), "eqx": M[off].sum().item()}

            # 2. per-query group masses and 3. entropy -- true attention only
            q, k = variants["rope"]
            m = mass_on(q, k, G)                                              # (N, 3S)
            same = F.one_hot(view_of, S).bool()
            out = {}
            for g, name in enumerate(["arch", "nonarch", "special"]):
                blk = m[:, g * S:(g + 1) * S]
                out[f"{name}_same"] = blk[same].view(N)
                out[f"{name}_cross"] = blk.sum(1) - out[f"{name}_same"]
            if per_class is not None:
                out["nonarch_class"] = mass_on(q, k, per_class)               # (N, n_nonarch)
            eh, em, t1 = entropy_stats(q[:, :, ent_q_t], k)
            out["ent_heads"], out["ent_mean"], out["top1"] = eh, em, t1
            rec["per_query"][l] = {kk: v.half().cpu().numpy() for kk, v in out.items()}
        return hook

    hooks = [model.aggregator.global_blocks[l].attn.register_forward_hook(make_hook(l), with_kwargs=True)
             for l in layers]
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            tokens_list, ps_idx = model.aggregator(images)
            for h in hooks:
                h.remove()
            hooks = []
            depth, depth_conf = model.depth_head(tokens_list, images, ps_idx)
            pose_enc = model.camera_head(tokens_list)[-1]
            extr, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    finally:
        for h in hooks:
            h.remove()
    peak = torch.cuda.max_memory_allocated() / 1e9

    # ---- output-based statistics (no ground truth)
    E = extr[0].float().cpu().numpy()                         # (S,3,4) world->cam
    R, t = E[:, :, :3], E[:, :, 3]
    centers = -np.einsum("sij,si->sj", R, t)                  # camera centres in the view-0 frame
    axes = R[:, 2, :]                                         # optical axes in world coords
    d = depth[0, ..., 0].float().cpu().numpy()
    conf = depth_conf[0].float().cpu().numpy()
    med_depth = float(np.median(d))
    mean_axis = axes.mean(0) / np.linalg.norm(axes.mean(0))
    outputs = {
        "median_depth": med_depth,
        "cam_spread_rel": float(np.linalg.norm(centers - centers.mean(0), axis=1).mean() / med_depth),
        "viewdir_spread_deg": float(np.degrees(np.arccos(np.clip(axes @ mean_axis, -1, 1))).mean()),
        "depth_conf_logmean": float(np.mean(np.log(np.clip(conf, 1e-6, None)))),
        "depth_conf_median": float(np.median(conf)),
    }

    # ---- anchors under each definition, at the last layer
    last = max(layers)
    picks = {}
    for var in ("rope", "norope"):
        for metric in ("code", "eq", "eqx"):
            scores = {e: rec["elements"][e][var][last][metric] for e in arch if e in rec["elements"]}
            picks[f"{var}/{metric}"] = max(scores, key=scores.get) if scores else None

    summary = {
        "scene": scene, "n_views": S, "image_hw": [Himg, Wimg], "patch_grid": [Hp, Wp],
        "tokens_per_view": P, "n_tokens": N, "dtype": str(dtype), "peak_gpu_gb": round(peak, 2),
        "seconds": round(time.time() - t0, 1),
        "arch_elements": arch, "nonarch_elements": nonarch,
        "element_tokens": {e: int(len(elem_idx[e][0])) for e in arch + nonarch},
        "element_views": {e: int(sum(1 for x in patch_sets[e] if x)) for e in arch + nonarch},
        "shipped_anchor": shipped_anchor, "anchor_by_definition": picks,
        "scva": rec["elements"], "outputs": outputs,
        "entropy_queries": ent_q.tolist(),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{scene}.json").write_text(json.dumps(summary, indent=1))
    arrays = {f"L{l:02d}_{k}": v for l, d_ in rec["per_query"].items() for k, v in d_.items()}
    arrays["view_of_token"] = view_of.cpu().numpy().astype(np.int16)
    arrays["arch_union"] = arch_u.cpu().numpy().astype(bool)
    arrays["nonarch_union"] = nonarch_u.cpu().numpy().astype(bool)
    for e in arch + nonarch:
        arrays[f"tok_{e}"] = elem_idx[e][0].astype(np.int32)
    np.savez_compressed(out_dir / f"{scene}.npz", **arrays)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("ARCHWORLD_ROOT", "data"))
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--scenes-file")
    ap.add_argument("--full-precision-weights", action="store_true",
                    help="keep fp32 weights under bf16 autocast (run_pass default; more memory). "
                         "Default casts weights to bf16, like VGGT_HALF=1.")
    ap.add_argument("--entropy-queries", type=int, default=4096)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scenes = list(args.scene)
    if args.scenes_file:
        scenes += [l.strip() for l in open(args.scenes_file, encoding="utf-8") if l.strip()]
    if not scenes:
        ap.error("give --scene and/or --scenes-file")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    if not args.full_precision_weights:
        model = model.to(dtype)

    for i, scene in enumerate(scenes, 1):
        if (Path(args.out) / f"{scene}.json").exists() and not args.overwrite:
            print(f"[{i}/{len(scenes)}] {scene}: exists, skipping")
            continue
        torch.cuda.reset_peak_memory_stats()
        try:
            s = run_scene(model, scene, args.root, args.masks_dir, args.out, dtype, args.entropy_queries)
        except Exception as exc:
            print(f"[{i}/{len(scenes)}] {scene}: FAILED {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()
            continue
        picks = s["anchor_by_definition"]
        agree = [k for k, v in picks.items() if v == s["shipped_anchor"]]
        print(f"[{i}/{len(scenes)}] {scene}: {s['seconds']}s, peak {s['peak_gpu_gb']} GB, "
              f"shipped anchor '{s['shipped_anchor']}', matched by {agree or 'none'}; picks {picks}",
              flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
