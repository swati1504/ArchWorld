#!/usr/bin/env python
"""Step 2 (plan fixed in eta-from-pass1-log.md before running): a one-feature threshold rule.

Inside each leave-one-scene-out fold: for each candidate threshold on the feature (midpoints between
sorted training values, >= MIN_SIDE training scenes per side), give each side the eta (any of the grid)
with the lowest mean regret on that side; keep the threshold with the lowest total training regret.
The held-out scene gets the eta of its side.

    python pass1_signals/threshold_rule.py --labels swati/results_vggt_dev/per_scene.csv \
        --features pass1_out/pass1_features.csv
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection import load_curves, regret, loso, metrics, run_all, sel_best_fixed, sel_oracle  # noqa: E402

FEATURES = ["anchor_tok_frac", "anchor_view_frac"]
MIN_SIDE = 5
N_PERM = 200


def fit_stump(x, R):
    """x: feature of training scenes; R: their regret curves. Returns (threshold, eta_low, eta_high)."""
    order = np.argsort(x)
    xs = x[order]
    best = None
    for i in range(MIN_SIDE, len(x) - MIN_SIDE + 1):
        if xs[i - 1] == xs[i]:
            continue
        lo, hi = order[:i], order[i:]
        jl, jh = R[lo].mean(0).argmin(), R[hi].mean(0).argmin()
        cost = R[lo, jl].sum() + R[hi, jh].sum()
        if best is None or cost < best[0]:
            best = (cost, (xs[i - 1] + xs[i]) / 2, int(jl), int(jh))
    return best[1:]


def sel_stump(col, log=None):
    def f(tr, te, F, X, etas):
        x = F[col].to_numpy(float)
        thr, jl, jh = fit_stump(x[tr], regret(X[tr]))
        if log is not None:
            log.append((thr, etas[jl], etas[jh]))
        return jl if x[te] <= thr else jh
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes, etas, X = load_curves(args.labels)
    F = pd.DataFrame({"scene": scenes}).merge(pd.read_csv(args.features), on="scene", how="left")
    R = regret(X)
    pd.set_option("display.width", 220)

    print(f"{len(scenes)} scenes, eta grid {etas}\n")
    print("== the rule fitted on all 39 scenes (for reading only; scoring below is leave-one-out)")
    for c in FEATURES:
        thr, jl, jh = fit_stump(F[c].to_numpy(float), R)
        lo = F[c] <= thr
        print(f"   {c}: threshold {thr:.4f} -> eta={etas[jl]} below ({lo.sum()} scenes), "
              f"eta={etas[jh]} above ({(~lo).sum()} scenes)")

    logs = {c: [] for c in FEATURES}
    sel = [("best-fixed", sel_best_fixed), ("oracle", sel_oracle)]
    sel += [(f"stump {c}", sel_stump(c, logs[c])) for c in FEATURES]
    res = run_all(sel, F, X, etas)
    extra = {}
    for name, s in sel:
        ch = loso(s, F, X, etas)
        extra[name] = (int((R[np.arange(len(X)), ch] > 0.5).sum()),
                       dict(Counter(int(e) for e in np.array(etas)[ch])))
    res["n regret>50%"] = [extra[n][0] for n in res.selector]
    res["eta chosen"] = [str(dict(sorted(extra[n][1].items()))) for n in res.selector]
    print("\n== leave-one-scene-out (ref = eta=1)")
    print(res.drop(columns=["median Score"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n== stability across the 39 folds")
    for c in FEATURES:
        L = logs[c][:len(scenes)]
        thr = np.array([t for t, _, _ in L])
        sides = Counter((a, b) for _, a, b in L)
        print(f"   {c}: threshold median {np.median(thr):.4f}, range {thr.min():.4f}-{thr.max():.4f}; "
              f"(eta below, eta above) per fold: {dict(sides)}")

    print(f"\n== control: {N_PERM} permutations of the feature (seed {args.seed})")
    rng = np.random.default_rng(args.seed)
    for c in FEATURES:
        real = metrics(loso(sel_stump(c), F, X, etas), X, etas, 0)["mean regret"]
        null = []
        for _ in range(N_PERM):
            Fp = F.copy()
            Fp[c] = F[c].values[rng.permutation(len(F))]
            null.append(metrics(loso(sel_stump(c), Fp, X, etas), X, etas, 0)["mean regret"])
        null = np.array(null)
        print(f"   stump {c}: real {real:.4f}; shuffled median {np.median(null):.4f} "
              f"(5-95%: {np.percentile(null, 5):.4f}-{np.percentile(null, 95):.4f}); p = {(null <= real).mean():.3f}")


if __name__ == "__main__":
    main()
