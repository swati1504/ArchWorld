#!/usr/bin/env python
"""Phase 6 of denoise-plan.md: test the confirmation list on the new scenes.

  - every listed (signal, target): one-sided Spearman on the new scenes in the sign seen on dev,
    Holm-corrected across the whole list; confirmed = adjusted p < 0.05
  - the chosen selector, trained on all dev scenes, applied to the new scenes, against best-fixed
    (dev's best fixed eta): regret, wins/losses, catastrophes (regret > 50%)

    python pass1_signals/confirm_analysis.py --list pass1_out/heldout/confirm_list.json \
        --labels pass1_out/heldout/clean_labels.csv --features pass1_out/heldout/heldout_features.csv \
        --dev-labels pass1_out/denoise/clean_labels.csv --dev-features pass1_out/pass1_features.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection import load_curves, regret, sel_curve_ridge        # noqa: E402
from step_risk import sel_knn_curve                                # noqa: E402
import clean_analysis as ca                                         # noqa: E402
from clean_analysis import targets, scene_noise                    # noqa: E402


def curves_at(path, E):
    scenes, etas, X = load_curves(path)
    missing = [e for e in E if e not in etas]
    if missing:
        sys.exit(f"{path}: eta {missing} missing")
    return scenes, X[:, [etas.index(e) for e in E]]


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); adj[o] = np.minimum(1, np.maximum.accumulate(p[o] * (n - np.arange(n))))
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--dev-labels", required=True)
    ap.add_argument("--dev-features", required=True)
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    L = json.loads(Path(args.list).read_text())
    E = L["E"]
    Ep = [e for e in E if e != 0]

    scenes, XE0 = curves_at(args.labels, E)
    s0, XE = XE0[:, E.index(0)], XE0[:, [E.index(e) for e in Ep]]
    F = pd.DataFrame({"scene": scenes}).merge(pd.read_csv(args.features), on="scene", how="left")
    # same tie rule as on dev: each new scene's band = 2 x its own label noise, same clipping
    ca.NOISE_MIN, ca.LABEL_FACTOR = L["noise_min"], L["label_factor"]
    tie = 2 * scene_noise(args.labels, scenes, L["noise_cap"])
    T = targets(XE, Ep, s0, tie)
    print(f"tie band on new scenes: median {np.median(tie):.1%} (dev median {L['dev_tie_band_median']:.1%})")
    print(f"{len(scenes)} new scenes; E = {E}; list written {L['written']} from {L['n_dev_scenes']} dev scenes\n")

    rows = []
    for t, items in L["targets"].items():
        for it in items:
            s, sign = it["signal"], it["sign"]
            if t not in T or s not in F:
                rows.append({"target": t, "signal": s, "rho_dev": it["rho_dev"], "rho_new": np.nan, "p_one_sided": 1.0})
                continue
            x, y = F[s].to_numpy(float), T[t]
            ok = np.isfinite(x) & np.isfinite(y)
            r, p = spearmanr(x[ok], y[ok])
            p1 = p / 2 if np.sign(r) == sign else 1 - p / 2
            rows.append({"target": t, "signal": s, "rho_dev": it["rho_dev"], "rho_new": r, "p_one_sided": p1})
    R = pd.DataFrame(rows)
    R["p_holm"] = holm(R["p_one_sided"])
    R["confirmed"] = R["p_holm"] < 0.05
    print(f"== listed signals ({len(R)} tests, Holm-corrected, one-sided in the dev direction)")
    print(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nconfirmed: {int(R.confirmed.sum())} of {len(R)}\n")

    # selector trained on dev, applied to the new scenes
    sel = L["selector"]
    dsc, DX = curves_at(args.dev_labels, Ep)
    DF = pd.DataFrame({"scene": dsc}).merge(pd.read_csv(args.dev_features), on="scene", how="left")
    allF = pd.concat([DF, F], ignore_index=True)
    allX = np.vstack([DX, XE])
    tr = list(range(len(dsc)))
    jbf = Ep.index(sel["best_fixed_eta"])
    if sel.get("kind") == "knn":
        f = sel_knn_curve(sel["k"], sel["signals"])
    elif sel.get("kind") == "ridge":
        f = sel_curve_ridge(sel["signals"])
    else:
        f = None
    choice = np.array([f(tr, len(dsc) + i, allF, allX, Ep) if f else jbf for i in range(len(scenes))])
    Rg = regret(XE)
    rs, rb = Rg[np.arange(len(XE)), choice], Rg[:, jbf]
    print(f"== selector '{sel['name']}' (trained on dev) vs best-fixed eta={sel['best_fixed_eta']} on the new scenes")
    print(pd.DataFrame([
        {"": "selector", "mean regret": rs.mean(), "median regret": np.median(rs), "max regret": rs.max(),
         "regret>50%": int((rs > 0.5).sum())},
        {"": "best-fixed", "mean regret": rb.mean(), "median regret": np.median(rb), "max regret": rb.max(),
         "regret>50%": int((rb > 0.5).sum())},
        {"": "oracle", "mean regret": 0.0, "median regret": 0.0, "max regret": 0.0, "regret>50%": 0},
    ]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    d = rs - rb
    print(f"wins/losses vs best-fixed: {(d < -1e-9).sum()}/{(d > 1e-9).sum()}; "
          f"eta chosen: {dict(pd.Series(np.array(Ep)[choice]).value_counts().sort_index())}")


if __name__ == "__main__":
    main()
