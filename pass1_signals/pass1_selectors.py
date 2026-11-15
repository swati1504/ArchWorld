#!/usr/bin/env python
"""First test: do Pass-1 attention signals pick eta better than the best fixed eta?

1. Descriptive: Spearman correlation of every Pass-1 feature with two targets
   (log10 best eta; large_vs_small, see metadata_baseline.py), with Benjamini-Hochberg
   correction because there are ~90 features and 39 scenes.
2. Leave-one-scene-out selectors (selection.py), scored by regret against best-fixed:
     curve-ridge / logeta-ridge on all features, on the layer-mean features only,
     and on the top-k features chosen *inside each fold* by |Spearman| with the
     training scenes' large_vs_small (so the held-out scene never informs selection).

    python pass1_signals/pass1_selectors.py --labels swati/results_vggt_dev/per_scene.csv \
        --features pass1_out/pass1_features.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection import (load_curves, run_all, sel_best_fixed, sel_oracle, sel_fixed,  # noqa: E402
                       sel_curve_ridge, sel_logeta_ridge)


def targets(X, etas):
    big, small = [etas.index(e) for e in (1000, 5000)], [etas.index(e) for e in (1, 10, 100)]
    return {"log10_best_eta": np.log10(np.array(etas)[X.argmin(1)]),
            "large_vs_small": np.log(X[:, big].min(1) / X[:, small].min(1))}


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def sel_topk(k, cols, kind="curve"):
    """Pick the k features most rank-correlated with large_vs_small on the training rows, then ridge."""
    def f(tr, te, F, X, etas):
        y = targets(X[tr], etas)["large_vs_small"]
        r = [abs(spearmanr(F[c].iloc[tr], y)[0]) if F[c].iloc[tr].nunique() > 1 else 0 for c in cols]
        top = [cols[i] for i in np.argsort(np.nan_to_num(r))[::-1][:k]]
        inner = sel_curve_ridge(top) if kind == "curve" else sel_logeta_ridge(top)
        return inner(tr, te, F, X, etas)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--top", type=int, default=15, help="rows of the correlation table to print")
    args = ap.parse_args()

    scenes, etas, X = load_curves(args.labels)
    F = pd.DataFrame({"scene": scenes}).merge(pd.read_csv(args.features), on="scene", how="left")
    cols = [c for c in F.columns if c != "scene" and F[c].nunique() > 1]
    T = targets(X, etas)
    pd.set_option("display.width", 200)

    rows = []
    for c in cols:
        r = {"feature": c}
        for t, y in T.items():
            r[f"{t} rho"], r[f"{t} p"] = spearmanr(F[c], y)
        rows.append(r)
    C = pd.DataFrame(rows)
    for t in T:
        C[f"{t} q"] = bh(C[f"{t} p"].fillna(1))
    C = C.reindex(C["large_vs_small rho"].abs().sort_values(ascending=False).index)
    print(f"== Spearman with targets, {len(cols)} features, n={len(scenes)} "
          f"(q = BH-adjusted; large_vs_small < 0 means the scene prefers eta in {{1000, 5000}})")
    print(C.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"features with q<0.10: large_vs_small {(C['large_vs_small q'] < .1).sum()}, "
          f"log10_best_eta {(C['log10_best_eta q'] < .1).sum()}\n")

    lmean = [c for c in cols if c.endswith("_Lmean") or "_L" not in c]
    sel = [(f"eta={e}", sel_fixed(j)) for j, e in enumerate(etas) if e in (1000, 5000)]
    sel += [("best-fixed", sel_best_fixed), ("oracle", sel_oracle),
            ("curve-ridge all", sel_curve_ridge(cols)), ("logeta-ridge all", sel_logeta_ridge(cols)),
            ("curve-ridge Lmean+scene", sel_curve_ridge(lmean)),
            ("logeta-ridge Lmean+scene", sel_logeta_ridge(lmean))]
    for k in (1, 3, 5, 10):
        sel += [(f"curve-ridge top{k}", sel_topk(k, cols)), (f"logeta-ridge top{k}", sel_topk(k, cols, "logeta"))]
    print("== leave-one-scene-out (ref = eta=1)")
    print(run_all(sel, F, X, etas).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
