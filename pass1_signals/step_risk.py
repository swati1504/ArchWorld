#!/usr/bin/env python
"""Plan A+B (fixed in eta-from-pass1-log.md before running): step risk and curve pooling.

A  Does anchor coverage predict how badly the largest step fails?
     risk_large = log( Score(5000) / min Score over eta <= 1000 )
   Hypothesis: negative Spearman with anchor_view_frac and anchor_tok_frac.
B  Selectors that pool whole regret curves, on 5 pre-chosen Pass-1 features:
     knn-curve k   average the training regret curves of the k nearest scenes, pick the lowest eta
     curve-ridge / logeta-ridge restricted to the same 5 features
   Control: 200 permutations of the feature rows; p = share of shuffles with mean regret <= real.

    python pass1_signals/step_risk.py --labels swati/results_vggt_dev/per_scene.csv \
        --features pass1_out/pass1_features.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection import (load_curves, design, regret, loso, metrics, run_all,  # noqa: E402
                       sel_best_fixed, sel_oracle, sel_curve_ridge, sel_logeta_ridge)

FEATURES = ["anchor_view_frac", "anchor_tok_frac", "a_share_Lmean", "nonarch_share_Lmean", "ent_L4"]
N_PERM = 200


def sel_knn_curve(k, cols):
    def f(tr, te, F, X, etas):
        A, B = design(F, tr, [te], cols)
        d = np.linalg.norm(A - B[0], axis=1)
        nn = np.asarray(tr)[np.argsort(d)[:k]]
        return int(np.argmin(regret(X[nn]).mean(0)))
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--seed", type=int, default=0, help="permutation seed")
    args = ap.parse_args()

    scenes, etas, X = load_curves(args.labels)
    F = pd.DataFrame({"scene": scenes}).merge(pd.read_csv(args.features), on="scene", how="left")
    j5000, small = etas.index(5000), [etas.index(e) for e in etas if e <= 1000]
    risk = np.log(X[:, j5000] / X[:, small].min(1))
    big, sm = [etas.index(e) for e in (1000, 5000)], [etas.index(e) for e in (1, 10, 100)]
    lvs = np.log(X[:, big].min(1) / X[:, sm].min(1))
    pd.set_option("display.width", 220)

    print(f"{len(scenes)} scenes, eta grid {etas}\n")
    print("== A: step risk. risk_large = log(Score(5000) / best Score at eta<=1000); >0 means 5000 is worse")
    order = np.argsort(-risk)
    print("   worst 6:", ", ".join(f"{scenes[i]} {np.exp(risk[i]):.2f}x" for i in order[:6]))
    for c in ["anchor_view_frac", "anchor_tok_frac"]:
        r, p = spearmanr(F[c], risk)
        print(f"   pre-registered: Spearman({c}, risk_large) = {r:+.3f}, p = {p:.3f}  (hypothesis: negative)")
    print("   descriptive, the other 3 features (not tested):")
    for c in FEATURES[2:]:
        r, p = spearmanr(F[c], risk)
        r2, p2 = spearmanr(F[c], lvs)
        print(f"     {c:22s} risk_large {r:+.3f} (p {p:.3f})   large_vs_small {r2:+.3f} (p {p2:.3f})")
    for c in FEATURES[:2]:
        r2, p2 = spearmanr(F[c], lvs)
        print(f"     {c:22s} large_vs_small {r2:+.3f} (p {p2:.3f})")
    print("   anchor coverage of the worst 6 vs the rest (median):")
    for c in FEATURES[:2]:
        print(f"     {c:22s} worst6 {np.median(F[c].values[order[:6]]):.4f}   rest {np.median(F[c].values[order[6:]]):.4f}")

    print("\n== B: selectors on the 5 pre-chosen features (LOSO; ref = eta=1)")
    sel = [("best-fixed", sel_best_fixed), ("oracle", sel_oracle),
           ("knn-curve k=5", sel_knn_curve(5, FEATURES)), ("knn-curve k=10", sel_knn_curve(10, FEATURES)),
           ("curve-ridge 5f", sel_curve_ridge(FEATURES)), ("logeta-ridge 5f", sel_logeta_ridge(FEATURES))]
    res = run_all(sel, F, X, etas)
    R = regret(X)
    cat = {}
    for name, s in sel:
        ch = loso(s, F, X, etas)
        cat[name] = (int((R[np.arange(len(X)), ch] > 0.5).sum()),
                     dict(pd.Series(np.array(etas)[ch]).value_counts().sort_index()))
    res["n regret>50%"] = [cat[n][0] for n in res.selector]
    res["eta chosen"] = [str(cat[n][1]) for n in res.selector]
    print(res.drop(columns=["median Score"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n== control: {N_PERM} permutations of the feature rows (seed {args.seed})")
    rng = np.random.default_rng(args.seed)
    for k in (5, 10):
        real = metrics(loso(sel_knn_curve(k, FEATURES), F, X, etas), X, etas, 0)["mean regret"]
        null = []
        for _ in range(N_PERM):
            Fp = F.copy()
            Fp[FEATURES] = F[FEATURES].values[rng.permutation(len(F))]
            null.append(metrics(loso(sel_knn_curve(k, FEATURES), Fp, X, etas), X, etas, 0)["mean regret"])
        null = np.array(null)
        print(f"   knn-curve k={k}: real mean regret {real:.4f}; shuffled median {np.median(null):.4f} "
              f"(5-95%: {np.percentile(null, 5):.4f}-{np.percentile(null, 95):.4f}); "
              f"p = {(null <= real).mean():.3f}")


if __name__ == "__main__":
    main()
