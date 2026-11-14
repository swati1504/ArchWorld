#!/usr/bin/env python
"""Step 0: how does reconstruction error vary with eta, per scene?

Reads swati's sweep (one row per scene x eta, `Score` = (Acc-mean + Comp-mean) / 2,
the raw Chamfer distance of the intervened prediction) and reports:
  - which eta wins per scene, and how often each eta wins;
  - how much the choice matters per scene (worst / best eta);
  - for every fixed eta: median Score and regret against the per-scene oracle.
Scores are compared only within a scene, so the scale of CD does not matter.

    python pass1_signals/eta_landscape.py swati/results_vggt_dev/per_scene.csv \
        --out pass1_out/eta_landscape.csv
"""
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_scene_csv")
    ap.add_argument("--out", help="write the per-scene table here")
    args = ap.parse_args()

    score = defaultdict(dict)
    for r in csv.DictReader(open(args.per_scene_csv, encoding="utf-8")):
        score[r["scene"]][int(r["eta"])] = float(r["Score"])
    etas = sorted({e for s in score.values() for e in s})
    scenes = sorted(s for s in score if set(score[s]) == set(etas))
    print(f"{len(scenes)} scenes with all of eta = {etas}\n")

    X = np.array([[score[s][e] for e in etas] for s in scenes])     # (scenes, etas)
    best = X.min(1)
    best_eta = [etas[i] for i in X.argmin(1)]
    rel = X / best[:, None]                                          # 1.0 = oracle

    print("wins per eta:", dict(sorted(Counter(best_eta).items())))
    spread = X.max(1) / best
    print(f"worst/best eta per scene: median {np.median(spread):.2f}x, "
          f"min {spread.min():.2f}x, max {spread.max():.2f}x")
    gap1 = X[:, 0] / best
    print(f"eta=1 vs oracle per scene: median {np.median(gap1):.3f}x; "
          f"scenes where oracle beats eta=1 by >5%: {(gap1 > 1.05).sum()}/{len(scenes)}\n")

    print(f"{'eta':>6} {'median Score':>13} {'mean regret':>12} {'median regret':>14} {'wins':>5}")
    for j, e in enumerate(etas):
        print(f"{e:>6} {np.median(X[:, j]):13.5f} {np.mean(rel[:, j] - 1):12.1%} "
              f"{np.median(rel[:, j] - 1):14.1%} {best_eta.count(e):5d}")
    print(f"{'oracle':>6} {np.median(best):13.5f} {0:12.1%} {0:14.1%}")

    jfix = int(np.argmin(np.mean(rel, 0)))
    print(f"\nbest single eta by mean regret: {etas[jfix]} "
          f"(in-sample: chosen and scored on the same 39 scenes)")
    gain_oracle = 1 - np.median(best) / np.median(X[:, 0])
    gain_fixed = 1 - np.median(X[:, jfix]) / np.median(X[:, 0])
    print(f"median Score vs eta=1: fixed eta={etas[jfix]} {gain_fixed:+.1%}, oracle {gain_oracle:+.1%}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scene"] + [f"score_eta{e}" for e in etas] + ["best_eta", "worst_over_best"])
            for s, row, be, sp in zip(scenes, X, best_eta, spread):
                w.writerow([s, *[f"{v:.6g}" for v in row], be, f"{sp:.4f}"])
        print(f"\nper-scene table -> {args.out}")


if __name__ == "__main__":
    main()
