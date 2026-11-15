#!/usr/bin/env python
"""Turn extract_features.py's raw per-scene output into one row of features per scene.

The anchor is the one Pass 2 actually steered (assets/anchors.json), since that is
what the eta labels were produced with. All attention quantities use the model's
real attention (with RoPE) unless the name says `norope`.

Features (L = global layer; each attention feature is reported at L in {4, 11, 17, 23}
and as the mean over all layers, suffix `_L4 ... _L23, _Lmean`):

  anchor dominance (from SCVA, authors' `code` definition)
    a_code          SCVA of the anchor (log10)
    a_share         anchor SCVA / sum of SCVA over architectural elements
    a_margin        anchor SCVA / runner-up architectural SCVA (log10)
    nonarch_share   sum SCVA over non-architectural elements / sum over all elements
  anchor coherence (paper's Eq. definition)
    a_self          mean attention mass an anchor query puts on anchor tokens (eq / n tokens)
    a_crossfrac     share of that mass that goes to other views (eqx / eq)
  leakage (per query, averaged over anchor queries `anc_` or all patch queries `all_`)
    *_leak          non-arch cross-view mass / (arch + non-arch cross-view mass)
    *_archx         attention mass on architectural tokens in other views
    *_sink          attention mass on special (camera/register) tokens
    all_sky         attention mass on sky tokens
  focus (sampled queries)
    ent             mean per-head attention entropy / log(N)   (1 = uniform)
    anc_ent         same, anchor queries only
    top1            mean head-averaged top-1 attention mass
  scene (no layer)
    anchor_tok_frac, anchor_view_frac, nonarch_tok_frac, sky_tok_frac,
    cam_spread_rel, viewdir_spread_deg, depth_conf_logmean,
    anchor_agrees    1 if our recomputed SCVA argmax (rope, code) equals the shipped anchor

    python pass1_signals/features.py --in pass1_out/features --out pass1_out/pass1_features.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY_LAYERS = (4, 11, 17, 23)
EPS = 1e-12


def _layers(js):
    any_e = next(iter(js["scva"].values()))
    return sorted(int(l) for l in any_e["rope"])


def scva_layer(js, l, var="rope", metric="code", names=None):
    names = names if names is not None else list(js["scva"])
    return {e: js["scva"][e][var][str(l)][metric] for e in names if e in js["scva"]}


def layer_feats(js, z, l, anchor):
    arch, nonarch = js["arch_elements"], js["nonarch_elements"]
    f = {}
    code_a = scva_layer(js, l, names=arch)
    code_n = scva_layer(js, l, names=nonarch)
    a = code_a.get(anchor, 0.0)
    others = sorted((v for e, v in code_a.items() if e != anchor), reverse=True)
    f["a_code"] = np.log10(a + EPS)
    f["a_share"] = a / (sum(code_a.values()) + EPS)
    f["a_margin"] = np.log10((a + EPS) / (others[0] + EPS)) if others else np.nan
    f["nonarch_share"] = sum(code_n.values()) / (sum(code_a.values()) + sum(code_n.values()) + EPS)

    n_a = js["element_tokens"].get(anchor, 0)
    eq = js["scva"].get(anchor, {}).get("rope", {}).get(str(l), {})
    f["a_self"] = eq.get("eq", np.nan) / n_a if n_a else np.nan
    f["a_crossfrac"] = eq.get("eqx", np.nan) / (eq.get("eq", np.nan) + EPS) if n_a else np.nan

    P, N = js["tokens_per_view"], js["n_tokens"]
    patch = np.arange(N) % P >= 5
    anc = np.zeros(N, bool)
    anc[z[f"tok_{anchor}"]] = True
    g = lambda k: z[f"L{l:02d}_{k}"].astype(np.float32)
    archx, nonx = g("arch_cross"), g("nonarch_cross")
    sink = g("special_same") + g("special_cross")
    leak = nonx / (archx + nonx + EPS)
    for name, q in (("anc", anc), ("all", patch)):
        if q.any():
            f[f"{name}_leak"], f[f"{name}_archx"], f[f"{name}_sink"] = leak[q].mean(), archx[q].mean(), sink[q].mean()
        else:
            f[f"{name}_leak"] = f[f"{name}_archx"] = f[f"{name}_sink"] = np.nan
    if "sky" in nonarch:
        f["all_sky"] = g("nonarch_class")[:, nonarch.index("sky")][patch].mean()

    eq_idx = np.asarray(js["entropy_queries"])
    ent = g("ent_heads") / np.log(N)
    f["ent"] = ent.mean()
    in_anc = anc[eq_idx]
    f["anc_ent"] = ent[in_anc].mean() if in_anc.any() else np.nan
    f["top1"] = g("top1").mean()
    return f


def scene_features(json_path, npz_path, anchor=None):
    js = json.loads(Path(json_path).read_text())
    z = np.load(npz_path)
    anchor = anchor or js["shipped_anchor"]
    if f"tok_{anchor}" not in z:
        raise KeyError(f"{js['scene']}: no tokens recorded for anchor '{anchor}'")
    layers = _layers(js)
    per_layer = {l: layer_feats(js, z, l, anchor) for l in layers}
    row = {"scene": js["scene"]}
    for k in per_layer[layers[0]]:
        for l in KEY_LAYERS:
            if l in per_layer:
                row[f"{k}_L{l}"] = per_layer[l][k]
        row[f"{k}_Lmean"] = float(np.nanmean([per_layer[l][k] for l in layers]))

    P, N, S = js["tokens_per_view"], js["n_tokens"], js["n_views"]
    n_patch = S * (P - 5)
    row["anchor_tok_frac"] = js["element_tokens"].get(anchor, 0) / n_patch
    row["anchor_view_frac"] = js["element_views"].get(anchor, 0) / S
    row["nonarch_tok_frac"] = float(z["nonarch_union"].sum()) / n_patch
    row["sky_tok_frac"] = js["element_tokens"].get("sky", 0) / n_patch
    for k in ("cam_spread_rel", "viewdir_spread_deg", "depth_conf_logmean"):
        row[k] = js["outputs"][k]
    row["anchor_agrees"] = int(js["anchor_by_definition"].get("rope/code") == anchor)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="directory of <scene>.json/.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for jp in sorted(Path(args.inp).glob("*.json")):
        npz = jp.with_suffix(".npz")
        if not npz.exists():
            print(f"skip {jp.stem}: no .npz")
            continue
        rows.append(scene_features(jp, npz))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} scenes x {df.shape[1] - 1} features -> {args.out}")


if __name__ == "__main__":
    main()
