#!/usr/bin/env python
"""Phase 3 of denoise-plan.md: what the clean dev labels say, and the confirmation list for new scenes.

Inputs: clean labels for every (scene, eta) incl. eta=0 (rescore.py collect), the scheme file (noise
floor), the Pass-1 feature table, and swati's original labels for comparison.

Ties: two scores whose ratio is below 1 + 2 x noise floor are tied.

Report (printed):
  3.1  per-scene best eta with ties, wins, regret of each fixed eta; each eta vs eta=0; how many of
       swati's best-eta labels change; scenes where eta does not matter; earlier noisy-label findings
       re-checked (anchor size vs eta=5000 risk, layer-4 entropy vs best eta, knn-curve k=10, threshold rule)
  3.2  the eta set E carried to the new scenes (rule in the plan)
  3.3  targets per scene on E
  3.4  Spearman of every Pass-1 feature x target, Benjamini-Hochberg over all pairs, bootstrap
       intervals for the top 10 per target
  3.5  selectors, leave-one-scene-out within E+, signals chosen inside each fold
  3.6  the confirmation list (--list-out), written with a timestamp

    python pass1_signals/clean_analysis.py --labels pass1_out/denoise/clean_labels.csv \
        --scheme pass1_out/denoise/scheme.json --features pass1_out/pass1_features.csv \
        --swati swati/results_vggt_dev/per_scene.csv --list-out pass1_out/heldout/confirm_list.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection import load_curves, regret, loso, metrics, sel_best_fixed, sel_curve_ridge  # noqa: E402
from step_risk import sel_knn_curve, FEATURES as OLD5                                           # noqa: E402
from threshold_rule import sel_stump                                                            # noqa: E402

MIN_WINS = 5
MAX_E = 4
# Ties (amendment of 2026-11-16, made before any clean label existed): each scene's tie band is twice its
# own label noise, estimated from its own draws; label noise = LABEL_FACTOR x pooled within-label draw sd
# (a label is the mean of 2 or the median of 3 draws), clipped to [NOISE_MIN, phase-0 noise floor].
NOISE_MIN = 0.01
LABEL_FACTOR = 0.7
TOP_PER_TARGET = 3
N_BOOT = 1000
SELECTOR_TARGETS = ("T_best", "T_slope", "T_regret")   # signals for the selectors come from these


# ----------------------------------------------------------------------------- shared helpers
def scene_noise(labels_csv, scenes, cap):
    """Per-scene label noise from the scene's own draws (pooled within-label relative sd x LABEL_FACTOR).

    Labels without a draw_scores column (e.g. a dry run on single-run labels) fall back to the cap."""
    df = pd.read_csv(labels_csv)
    out = {}
    for s in scenes:
        g = df[df.scene == s]
        ss, dof = 0.0, 0
        for v in (g["draw_scores"] if "draw_scores" in g else []):
            d = np.array([float(x) for x in str(v).split() if x not in ("nan",)])
            d = d[np.isfinite(d)]
            if len(d) >= 2:
                r = (d - d.mean()) / d.mean()
                ss, dof = ss + float((r ** 2).sum()), dof + len(d) - 1
        sd = np.sqrt(ss / dof) if dof else np.nan
        out[s] = float(np.clip(LABEL_FACTOR * sd, NOISE_MIN, cap)) if np.isfinite(sd) else float(cap)
    return np.array([out[s] for s in scenes])


def tied_best(row, tie):
    """Indices tied with the row's minimum (ratio < 1 + tie)."""
    return np.where(row / row.min() < 1 + tie)[0]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def targets(XE, etasE, s0, tie):
    """Targets per scene on E+ (columns of XE are the eta values in etasE, all > 0); s0 = score at eta 0.
    tie: one tie band per row (or a scalar)."""
    etasE = np.asarray(etasE); le = np.log10(etasE)
    tie = np.broadcast_to(np.asarray(tie, float), (len(XE),))
    T = {}
    T["T_best"] = np.array([np.median(le[tied_best(r, t)]) for r, t in zip(XE, tie)])
    n = len(etasE)
    if n >= 2:
        small, large = np.arange(n // 2), np.arange(n - n // 2, n)
        T["T_slope"] = np.log(XE[:, large].min(1) / XE[:, small].min(1))
    best = XE.min(1)
    for j, e in enumerate(etasE):
        T[f"T_regret({e})"] = np.log(XE[:, j] / best)
    T["T_matters"] = np.log(XE.max(1) / best)
    if s0 is not None:
        T["T_gain"] = np.log(best / s0)
    return {k: v for k, v in T.items() if np.nanstd(v) > 0}


def rank_signals(F, T, cols):
    rows = []
    for t, y in T.items():
        for c in cols:
            x = F[c].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 8 or np.std(x[ok]) == 0:
                continue
            r, p = spearmanr(x[ok], y[ok])
            rows.append({"target": t, "signal": c, "rho": r, "p": p})
    R = pd.DataFrame(rows)
    R["q"] = bh(R["p"].fillna(1))
    return R


def top_signals(F, T, cols, k=TOP_PER_TARGET, which=None):
    """{target: [(signal, rho), ...]} with the k largest |rho| per target."""
    R = rank_signals(F, T, cols)
    out = {}
    for t, g in R.groupby("target"):
        if which and not any(t.startswith(w) for w in which):
            continue
        g = g.reindex(g["rho"].abs().sort_values(ascending=False).index).head(k)
        out[t] = [(s, float(r)) for s, r in zip(g["signal"], g["rho"])]
    return out


def selector_signals(F, XE, etasE, tie, cols, rows=None):
    rows = np.arange(len(XE)) if rows is None else np.asarray(rows)
    T = targets(XE[rows], etasE, None, np.broadcast_to(np.asarray(tie, float), (len(XE),))[rows])
    Fr = F.iloc[rows].reset_index(drop=True)
    tops = top_signals(Fr, T, cols, which=SELECTOR_TARGETS)
    return sorted({s for v in tops.values() for s, _ in v})


def make_selector(kind, k, etasE, tie, cols):
    """A selector whose signals are chosen on the training rows only."""
    def f(tr, te, F, X, etas):
        sig = selector_signals(F, X, etasE, tie, cols, rows=tr)
        if kind == "knn":
            return sel_knn_curve(k, sig)(tr, te, F, X, etas)
        return sel_curve_ridge(sig)(tr, te, F, X, etas)
    return f


def boot_ci(x, y, rng, n=N_BOOT):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    rs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            rs.append(spearmanr(x[i], y[i])[0])
    return np.percentile(rs, [2.5, 97.5]) if rs else (np.nan, np.nan)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--swati")
    ap.add_argument("--list-out")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    ff = lambda v: f"{v:.4f}"

    cap = json.loads(Path(args.scheme).read_text()).get("noise_floor") or 0.0
    scenes, etas_all, Xall = load_curves(args.labels)
    assert etas_all[0] == 0, "labels must include eta=0"
    s0, X, etas = Xall[:, 0], Xall[:, 1:], etas_all[1:]
    noise = scene_noise(args.labels, scenes, cap)
    tie = 2 * noise
    F = pd.DataFrame({"scene": scenes}).merge(pd.read_csv(args.features), on="scene", how="left")
    cols = [c for c in F.columns if c != "scene" and pd.api.types.is_numeric_dtype(F[c]) and F[c].nunique() > 1]
    print(f"{len(scenes)} scenes with all of eta {etas_all}; per-scene tie band = 2 x own label noise, "
          f"noise clipped to [{NOISE_MIN:.0%}, {cap:.1%}]")
    print(f"tie band across scenes: median {np.median(tie):.1%}, range {tie.min():.1%}-{tie.max():.1%}; "
          f"at the cap: {(noise >= cap - 1e-12).sum()}, at the minimum: {(noise <= NOISE_MIN + 1e-12).sum()}\n")

    # ---- 3.1 landscape
    print("== 3.1 what the clean labels say")
    ties = [tied_best(r, t) for r, t in zip(X, tie)]
    wins = np.zeros(len(etas), int)
    for t in ties:
        wins[t] += 1
    R = regret(X)
    print(pd.DataFrame({"eta": etas, "wins (ties count for each)": wins, "mean regret": R.mean(0),
                        "median regret": np.median(R, 0)}).to_string(index=False, float_format=ff))
    flat = sum(len(t) == len(etas) for t in ties)
    print(f"scenes where every eta > 0 is tied (eta does not matter): {flat}/{len(scenes)}")
    print(f"scenes with a single clear best eta: {sum(len(t) == 1 for t in ties)}/{len(scenes)}\n")
    print("each eta vs eta=0 (relative change of CD per scene)")
    for j, e in enumerate(etas):
        d = X[:, j] / s0 - 1
        print(f"  eta={e:5d}: median {np.median(d):+.2%}, mean {d.mean():+.2%}, better on {(d < -tie).sum()} "
              f"/ tied {(np.abs(d) <= tie).sum()} / worse {(d > tie).sum()}, Wilcoxon p={wilcoxon(X[:, j], s0).pvalue:.3f}")
    d = X.min(1) / s0 - 1
    print(f"  best eta per scene: median {np.median(d):+.2%}, better than eta=0 beyond the tie band on "
          f"{(d < -tie).sum()}/{len(scenes)}")
    if args.swati:
        sw_sc, sw_etas, SW = load_curves(args.swati)
        sw_best = {s: sw_etas[int(np.argmin(r))] for s, r in zip(sw_sc, SW)}
        changed = [s for s, t in zip(scenes, ties) if s in sw_best and sw_best[s] not in [etas[i] for i in t]]
        print(f"\nswati's best eta is outside the clean tied-best set on {len(changed)}/{len(scenes)} scenes")

    print("\nearlier noisy-label findings, re-checked on clean labels")
    if 5000 in etas and 1000 in etas:
        r = np.log(X[:, etas.index(5000)] / X[:, etas.index(1000)])
        for c in ("anchor_tok_frac", "anchor_view_frac"):
            print(f"  Spearman({c}, log S5000/S1000) = {spearmanr(F[c], r)[0]:+.3f} (p {spearmanr(F[c], r)[1]:.4f})"
                  f"   [noisy: about -0.47 / -0.40]")
        tiny, blow = F["anchor_tok_frac"].values < 0.02, np.exp(r) > 1.5
        print(f"  tiny anchors (<2%): {blow[tiny].sum()}/{tiny.sum()} blow-ups (>1.5x); others {blow[~tiny].sum()}/{(~tiny).sum()}"
              f"; Fisher one-sided p {fisher_exact([[blow[tiny].sum(), (~blow[tiny]).sum()], [blow[~tiny].sum(), (~blow[~tiny]).sum()]], alternative='greater')[1]:.4f}"
              f"   [noisy: 4/8 vs 1/31]")
    tb = np.array([np.median(np.log10(np.array(etas))[t]) for t in ties])
    print(f"  Spearman(ent_L4, best log10 eta) = {spearmanr(F['ent_L4'], tb)[0]:+.3f} (p {spearmanr(F['ent_L4'], tb)[1]:.4f})"
          f"   [noisy: -0.47]")
    bf = metrics(loso(sel_best_fixed, F, X, etas), X, etas, 0)
    kn = metrics(loso(sel_knn_curve(10, OLD5), F, X, etas), X, etas, 0)
    st = metrics(loso(sel_stump("anchor_tok_frac"), F, X, etas), X, etas, 0)
    print(f"  LOSO mean regret: best-fixed {bf['mean regret']:.2%}, knn-curve k=10 on the old 5 features "
          f"{kn['mean regret']:.2%}, threshold rule (anchor_tok_frac) {st['mean regret']:.2%}"
          f"   [noisy: 4.38% / 4.20% / 7.93%]\n")

    # ---- 3.2 eta set E
    jbf = int(np.argmin(R.mean(0)))
    keep = [j for j in range(len(etas)) if wins[j] >= MIN_WINS or j == jbf]
    if len(keep) > MAX_E - 1:
        keep = sorted(keep, key=lambda j: (j != jbf, -wins[j]))[:MAX_E - 1]
    keep = sorted(keep)
    E_plus = [etas[j] for j in keep]
    E = [0] + E_plus
    print(f"== 3.2 eta set E = {E} (best-fixed eta = {etas[jbf]}; wins {dict(zip(etas, wins.tolist()))})\n")

    # ---- 3.3 targets on E
    XE = X[:, keep]
    T = targets(XE, E_plus, s0, tie)
    print(f"== 3.3 targets on E+ = {E_plus}: {', '.join(T)}\n")

    # ---- 3.4 signal ranking
    print(f"== 3.4 Spearman, {len(cols)} Pass-1 signals x {len(T)} targets, BH over all {len(cols) * len(T)} pairs")
    Rk = rank_signals(F, T, cols)
    rng = np.random.default_rng(0)
    for t in T:
        g = Rk[Rk.target == t]
        g = g.reindex(g["rho"].abs().sort_values(ascending=False).index).head(10).copy()
        g["95% CI"] = [("[%+.2f, %+.2f]" % tuple(boot_ci(F[s].to_numpy(float), T[t], rng))) for s in g.signal]
        print(f"\n-- {t}  (pairs with q < 0.10: {(Rk[Rk.target == t].q < 0.10).sum()})")
        print(g[["signal", "rho", "p", "q", "95% CI"]].to_string(index=False, float_format=ff))
    print(f"\nall pairs with q < 0.10: {(Rk.q < 0.10).sum()}; q < 0.05: {(Rk.q < 0.05).sum()}\n")

    # ---- 3.5 selectors
    print("== 3.5 selectors (leave-one-scene-out within E+, signals chosen inside each fold)")
    menu = [("best-fixed", sel_best_fixed, None)]
    for kind, k in (("knn", 10), ("knn", 5), ("ridge", None)):
        name = f"knn-curve k={k}" if kind == "knn" else "curve-ridge"
        menu.append((name, make_selector(kind, k, E_plus, tie, cols), (kind, k)))
    res = []
    for name, sel, spec in menu:
        ch = loso(sel, F, XE, E_plus)
        m = metrics(ch, XE, E_plus, 0)
        RE = regret(XE)[np.arange(len(XE)), ch]
        res.append({"selector": name, "mean regret": m["mean regret"], "median regret": m["median regret"],
                    "max regret": m["max regret"], "regret>50%": int((RE > 0.5).sum()), "_spec": spec})
    Rs = pd.DataFrame(res)
    base = Rs.loc[Rs.selector == "best-fixed", "mean regret"].item()
    print(Rs.drop(columns="_spec").to_string(index=False, float_format=ff))
    best_row = Rs.loc[Rs["mean regret"].idxmin()]
    print(f"chosen selector: {best_row.selector} (best-fixed {base:.2%})\n")

    # ---- 3.6 confirmation list
    tops = top_signals(F, T, cols)
    spec = best_row["_spec"]
    selector = {"name": best_row.selector, "best_fixed_eta": int(E_plus[int(np.argmin(regret(XE).mean(0)))])}
    if spec is not None:
        selector.update(kind=spec[0], k=spec[1], signals=selector_signals(F, XE, E_plus, tie, cols))
    lst = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "n_dev_scenes": len(scenes), "E": E,
           "tie_rule": "per-scene: 2 x own label noise", "noise_cap": cap, "noise_min": NOISE_MIN,
           "label_factor": LABEL_FACTOR, "dev_tie_band_median": float(np.median(tie)),
           "targets": {t: [{"signal": s, "rho_dev": r, "sign": int(np.sign(r))} for s, r in v] for t, v in tops.items()},
           "selector": selector, "dev_loso_mean_regret": {r["selector"]: r["mean regret"] for r in res}}
    print("== 3.6 confirmation list")
    print(json.dumps(lst, indent=1))
    if args.list_out:
        Path(args.list_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.list_out).write_text(json.dumps(lst, indent=1))
        print(f"-> {args.list_out}")


if __name__ == "__main__":
    main()
