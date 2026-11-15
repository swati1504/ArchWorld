#!/usr/bin/env python
"""Evaluation harness for step-size selectors: leave-one-scene-out on the dev scenes.

Labels are the per-scene error curves from an eta sweep (one Score per scene and eta).
A selector sees the features and curves of the training scenes and picks one eta for
the held-out scene, from its features only. It is scored by **regret**: how much worse
the chosen eta's Score is than the scene's best, Score(chosen) / Score(best) - 1.
A near-miss on a flat scene costs almost nothing, so noisy "best eta" labels on flat
scenes don't dominate.

Selectors
  eta=<e>          a fixed eta for every scene
  best-fixed       the eta with the lowest mean regret on the training scenes
  oracle           the per-scene best (needs ground truth; the upper bound)
  group:<col>      best eta for the scene's category on the training scenes
                   (falls back to best-fixed if fewer than 3 training scenes share it)
  curve-ridge      ridge regression of log(Score/best) at every eta; pick the lowest prediction
  logeta-ridge     ridge regression of log10(best eta); round to the nearest grid value

    python pass1_signals/selection.py --labels swati/results_vggt_dev/per_scene.csv \
        --features pass1_out/meta_features.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import pandas as pd

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_GROUP = 3


# ----------------------------------------------------------------------------- data
def load_curves(path):
    score = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        score[r["scene"]][int(r["eta"])] = float(r["Score"])
    etas = sorted({e for s in score.values() for e in s})
    scenes = sorted(s for s in score if set(score[s]) == set(etas))
    X = np.array([[score[s][e] for e in etas] for s in scenes])
    return scenes, etas, X


def design(F, train, test, cols):
    """Standardised numeric + one-hot categorical design matrix; fit on train rows only."""
    parts_tr, parts_te = [], []
    for c in cols:
        v = F[c]
        if pd.api.types.is_numeric_dtype(v):
            a = v.to_numpy(float)
            mu, sd = np.nanmean(a[train]), np.nanstd(a[train]) or 1.0
            a = np.nan_to_num((a - mu) / sd)                 # missing -> training mean
            parts_tr.append(a[train, None]); parts_te.append(a[test, None])
        else:
            cats = sorted(set(v.iloc[train]))
            oh = np.array([[x == k for k in cats] for x in v], float)
            parts_tr.append(oh[train]); parts_te.append(oh[test])
    return np.hstack(parts_tr), np.hstack(parts_te)


def ridge_fit_predict(A, Y, B):
    """Ridge with intercept; alpha picked by closed-form leave-one-out on the training rows."""
    mu_a, mu_y = A.mean(0), Y.mean(0)
    A0, Y0 = A - mu_a, Y - mu_y
    best = None
    for alpha in ALPHAS:
        G = np.linalg.solve(A0.T @ A0 + alpha * np.eye(A.shape[1]), A0.T)
        H = A0 @ G
        resid = (Y0 - H @ Y0) / (1 - np.diag(H) - 1.0 / len(A))[:, None]   # LOO residuals incl. intercept
        err = float((resid ** 2).mean())
        if best is None or err < best[0]:
            best = (err, G @ Y0)
    return (B - mu_a) @ best[1] + mu_y


# ----------------------------------------------------------------------------- selectors
def regret(X):
    return X / X.min(1, keepdims=True) - 1


def sel_fixed(j):
    return lambda tr, te, F, X, etas: j


def sel_best_fixed(tr, te, F, X, etas):
    return int(np.argmin(regret(X[tr]).mean(0)))


def sel_oracle(tr, te, F, X, etas):
    return int(np.argmin(X[te]))


def sel_group(col):
    def f(tr, te, F, X, etas):
        g = F[col].iloc[te]
        rows = [i for i in tr if F[col].iloc[i] == g]
        if len(rows) < MIN_GROUP:
            return sel_best_fixed(tr, te, F, X, etas)
        return int(np.argmin(regret(X[rows]).mean(0)))
    return f


def sel_curve_ridge(cols):
    def f(tr, te, F, X, etas):
        A, B = design(F, tr, [te], cols)
        Y = np.log(X[tr] / X[tr].min(1, keepdims=True))
        return int(np.argmin(ridge_fit_predict(A, Y, B)[0]))
    return f


def sel_logeta_ridge(cols):
    def f(tr, te, F, X, etas):
        A, B = design(F, tr, [te], cols)
        y = np.log10(np.array(etas)[X[tr].argmin(1)])[:, None].astype(float)
        p = ridge_fit_predict(A, y, B)[0, 0]
        return int(np.argmin(np.abs(np.log10(etas) - p)))
    return f


# ----------------------------------------------------------------------------- evaluation
def loso(selector, F, X, etas):
    n = len(X)
    return np.array([selector([i for i in range(n) if i != t], t, F, X, etas) for t in range(n)])


def metrics(choice, X, etas, ref):
    R = regret(X)[np.arange(len(X)), choice]
    S = X[np.arange(len(X)), choice]
    ref_S = X[:, ref]
    # regret as in swati/proxy_select.py (there as a ratio, 1.00 = perfect; here minus 1),
    # plus the paper's style (intervention/aggregate.py): % change of the median Score
    # across scenes against a reference, and the share of scenes improved.
    return {"mean regret": R.mean(), "median regret": np.median(R), "max regret": R.max(),
            "n regret>10%": int((R > 0.10).sum()),
            "hit oracle": float((choice == X.argmin(1)).mean()),
            "median Score": np.median(S),
            "median vs ref": (np.median(S) - np.median(ref_S)) / np.median(ref_S),
            "improved vs ref": float((S < ref_S).mean()), "_R": R}


def run_all(selectors, F, X, etas, ref=0):
    rows, base_R = [], None
    for name, sel in selectors:
        m = metrics(loso(sel, F, X, etas), X, etas, ref)
        if name == "best-fixed":
            base_R = m["_R"]
        rows.append((name, m))
    out = []
    for name, m in rows:
        d = {k: v for k, v in m.items() if not k.startswith("_")}
        if base_R is not None:
            diff = m["_R"] - base_R
            d["vs best-fixed W/L"] = f"{(diff < -1e-9).sum()}/{(diff > 1e-9).sum()}"
        out.append({"selector": name, **d})
    return pd.DataFrame(out)


def default_selectors(etas, F=None, feature_cols=(), group_cols=()):
    s = [(f"eta={e}", sel_fixed(j)) for j, e in enumerate(etas)]
    s += [("best-fixed", sel_best_fixed), ("oracle", sel_oracle)]
    for c in group_cols:
        s.append((f"group:{c}", sel_group(c)))
    if feature_cols:
        s.append(("curve-ridge", sel_curve_ridge(list(feature_cols))))
        s.append(("logeta-ridge", sel_logeta_ridge(list(feature_cols))))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--features", help="CSV with a 'scene' column; other columns are features")
    ap.add_argument("--cols", nargs="*", help="feature columns for the ridge selectors (default: all)")
    ap.add_argument("--group", nargs="*", default=[], help="categorical columns for group selectors")
    args = ap.parse_args()

    scenes, etas, X = load_curves(args.labels)
    F = pd.DataFrame({"scene": scenes})
    cols = []
    if args.features:
        f = pd.read_csv(args.features)
        F = F.merge(f, on="scene", how="left")
        missing = F.drop(columns="scene").isna().all(axis=1).sum()
        if missing:
            print(f"warning: {missing} scenes have no features")
        cols = args.cols if args.cols is not None else [c for c in F.columns if c != "scene"]
    pd.set_option("display.width", 200)
    print(f"{len(scenes)} scenes, eta grid {etas}; leave-one-scene-out.\n"
          f"regret = Score(chosen)/Score(best eta) - 1. 'ref' = eta={etas[0]} (stand-in for no "
          f"intervention, which this sweep lacks).\n")
    print(run_all(default_selectors(etas, F, cols, args.group), F, X, etas)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
