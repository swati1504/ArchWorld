#!/usr/bin/env python
"""Does a drift summary that is sensitive to local damage beat the median?

The locked rule summarizes the per-pixel relative depth change (eta=5000 vs eta=1) by its median
over all pixels of all views. On the one test miss (Degutale Temple) the change was concentrated
in one region, so the median stayed low. This script computes several summaries of the same
per-pixel change for every dev and test scene and evaluates the guarded-5000 rule with each:

  median      median over all pixels (the locked rule)
  p75, p90    75th / 90th percentile over all pixels
  view_max    largest per-view median
  view_p75    75th percentile of per-view medians
  frac05      share of pixels that moved more than 5%
  frac10      share of pixels that moved more than 10%

Protocol (same as the locked rule): tau tuned leave-one-out on the 39 dev scenes, frozen as the
median of the LOO choices, then applied once to the 40 test scenes. Regret on both splits is
against the better of eta=1000 and eta=5000 so the two splits are comparable. CPU only.

    python swati/drift_variants.py --csv swati/results_drift_variants.csv
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

STATS = ["median", "p75", "p90", "view_max", "view_p75", "frac05", "frac10"]
SAFE, BOLD, REF = 1000, 5000, 1


def depth(p):
    z = np.load(p)["depth_map"]
    return z.reshape(z.shape[1], *z.shape[2:4]).astype(np.float32)


def cd(p):
    if not p.exists():
        return None
    m = json.loads(p.read_text()).get("inter") or {}
    return 0.5 * (m["Acc-mean"] + m["Comp-mean"]) if "Acc-mean" in m and "Comp-mean" in m else None


def summaries(d5, d1):
    ok = (d1 > 1e-4) & (d5 > 1e-4)
    rel = np.where(ok, np.abs(d5 - d1) / np.maximum(d1, 1e-4), np.nan)
    flat = rel[ok]
    per_view = np.array([np.nanmedian(rel[v]) for v in range(len(rel)) if ok[v].any()])
    return {"median": float(np.median(flat)), "p75": float(np.percentile(flat, 75)),
            "p90": float(np.percentile(flat, 90)), "view_max": float(per_view.max()),
            "view_p75": float(np.percentile(per_view, 75)),
            "frac05": float(np.mean(flat > 0.05)), "frac10": float(np.mean(flat > 0.10))}


def collect(results, scenes, split):
    rows = []
    for s in scenes:
        run = lambda e: results / f"output_{e}" / s
        if not all((run(e) / "inter_raw.npz").exists() for e in (REF, BOLD)):
            print(f"  [{split}] {s}: missing raw runs, skipped"); continue
        c = {e: cd(run(e) / "eval_metrics_0_23.json") for e in (SAFE, BOLD)}
        if None in c.values():
            print(f"  [{split}] {s}: missing scores, skipped"); continue
        r = {"split": split, "scene": s, "cd_1000": c[SAFE], "cd_5000": c[BOLD]}
        r.update(summaries(depth(run(BOLD) / "inter_raw.npz"), depth(run(REF) / "inter_raw.npz")))
        rows.append(r)
        print(f"  [{split}] {s[:40]:<40} median {r['median']:.4f}  p90 {r['p90']:.4f}  view_max {r['view_max']:.4f}", flush=True)
    return rows


def evaluate(dev, test, stat):
    def regs(rows, t):
        out = []
        for r in rows:
            pick = r["cd_5000"] if r[stat] <= t else r["cd_1000"]
            out.append(pick / min(r["cd_1000"], r["cd_5000"]))
        return np.array(out)
    taus = sorted({0.0, np.inf} | {r[stat] for r in dev})
    loo, chosen = [], []
    for i, r in enumerate(dev):
        others = dev[:i] + dev[i + 1:]
        t = min(taus, key=lambda t: (regs(others, t).mean(), t))
        chosen.append(t); loo.append(regs([r], t)[0])
    tau = float(np.median(chosen))
    tr = regs(test, tau)
    pick_cd = np.array([r["cd_5000"] if r[stat] <= tau else r["cd_1000"] for r in test])
    safe_cd = np.array([r["cd_1000"] for r in test])
    d = pick_cd / safe_cd - 1
    bad = [r for r in test if r["cd_5000"] > 1.2 * r["cd_1000"]]
    caught = sum(r[stat] > tau for r in bad)
    kept = sum(r[stat] <= tau for r in test)
    deg = [r for r in test if r["scene"].startswith("Degutale")]
    deg_s = ("caught" if deg[0][stat] > tau else "missed") if deg else "n/a"
    return (f"  {stat:<9} tau {tau:<9.4g} dev LOO regret {np.mean(loo):.4f} | test regret {tr.mean():.4f}"
            f"  mean CD {pick_cd.mean():.4f}  vs1000 W/T/L {int((d < -1e-12).sum())}/{int((abs(d) <= 1e-12).sum())}/{int((d > 1e-12).sum())}"
            f"  kept5000 {kept}/{len(test)}  failures caught {caught}/{len(bad)}  Degutale {deg_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, default=Path("results/vggt_dev"))
    ap.add_argument("--test", type=Path, default=Path("results/vggt_test"))
    ap.add_argument("--dev-scenes", type=Path, default=Path("scenes_dev.txt"))
    ap.add_argument("--test-scenes", type=Path, default=Path("scenes_test.txt"))
    ap.add_argument("--csv", type=Path, default=Path("swati/results_drift_variants.csv"))
    a = ap.parse_args()
    read = lambda p: [l.strip() for l in p.read_text().splitlines() if l.strip()]
    dev = collect(a.dev, read(a.dev_scenes), "dev")
    test = collect(a.test, read(a.test_scenes), "test")
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dev[0].keys())); w.writeheader(); w.writerows(dev + test)
    print(f"\n{len(dev)} dev, {len(test)} test scenes -> {a.csv}")
    safe = np.array([r["cd_1000"] for r in test]); best = np.array([min(r["cd_1000"], r["cd_5000"]) for r in test])
    print(f"reference: fixed 1000 test mean CD {safe.mean():.4f}, test regret {(safe / best).mean():.4f}; oracle mean CD {best.mean():.4f}\n")
    for s in STATS:
        print(evaluate(dev, test, s))


if __name__ == "__main__":
    main()
