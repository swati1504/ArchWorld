#!/usr/bin/env python
"""Does scene metadata explain which eta a scene wants?

Builds a per-scene metadata table and relates it to the sweep. This is the bar the
Pass-1 attention signals have to clear: if "it's a mosque" predicts eta as well as the
attention does, the attention adds nothing.

Columns written to --out:
  region, arch_type, anchor, construction_year, n_views     -- known without reconstruction
  gt_scale, gt_extent, gt_model_diagonal                     -- derived from the ground-truth
        reconstruction: fine for understanding, NOT valid predictor inputs at inference

Targets for the descriptive analysis:
  log10_best_eta   log10 of the per-scene best eta
  large_vs_small   log( min Score over eta in {1000, 5000} / min Score over eta in {1, 10, 100} );
                   negative = the scene prefers large steps. Continuous, uses the whole curve.

    python pass1_signals/metadata_baseline.py --labels swati/results_vggt_dev/per_scene.csv \
        --out pass1_out/meta_features.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection import load_curves, regret, run_all, default_selectors  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "intervention" / "assets"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--metadata", default=str(REPO / "data" / "ArchWorld_metadata.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenes, etas, X = load_curves(args.labels)
    md = pd.read_csv(args.metadata).set_index("scene")
    anchors = json.loads((ASSETS / "anchors.json").read_text(encoding="utf-8"))
    views = json.loads((ASSETS / "views.json").read_text(encoding="utf-8"))

    F = pd.DataFrame({
        "scene": scenes,
        "region": [md.loc[s, "Region"] for s in scenes],
        "arch_type": [md.loc[s, "Architecture Type"] for s in scenes],
        "anchor": [anchors[s] for s in scenes],
        "construction_year": [float(md.loc[s, "Construction Year"]) for s in scenes],
        "n_views": [len(views[s]) for s in scenes],
        "gt_scale": [md.loc[s, "Scale"] for s in scenes],
        "gt_extent": [md.loc[s, "Extent"] for s in scenes],
        "gt_model_diagonal": [md.loc[s, "Model_Diagonal"] for s in scenes],
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    F.to_csv(args.out, index=False)

    big, small = [etas.index(e) for e in (1000, 5000)], [etas.index(e) for e in (1, 10, 100)]
    T = pd.DataFrame({
        "log10_best_eta": np.log10(np.array(etas)[X.argmin(1)]),
        "large_vs_small": np.log(X[:, big].min(1) / X[:, small].min(1)),
    })
    R = regret(X)
    pd.set_option("display.width", 200)

    print("== categorical: per-category mean regret at each eta (lower = better), and best eta counts")
    for col in ["region", "arch_type", "anchor"]:
        rows = []
        for g, idx in F.groupby(col).groups.items():
            idx = list(idx)
            rows.append({col: g, "n": len(idx),
                         **{f"eta{e}": R[idx, j].mean() for j, e in enumerate(etas)},
                         "large_vs_small": T["large_vs_small"].iloc[idx].mean(),
                         "best eta counts": dict(pd.Series(np.array(etas)[X[idx].argmin(1)]).value_counts().sort_index())})
        print(pd.DataFrame(rows).sort_values("n", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"), "\n")

    print("== numeric: Spearman correlation with the targets (n=%d)" % len(scenes))
    rows = []
    for col in ["construction_year", "n_views", "gt_scale", "gt_extent", "gt_model_diagonal"]:
        r = {"feature": col}
        for t in T:
            rho, p = spearmanr(F[col], T[t])
            r[f"{t} rho"], r[f"{t} p"] = rho, p
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"), "\n")

    print("== leave-one-scene-out selectors using metadata available at inference")
    sel = default_selectors(etas, F, feature_cols=["region", "arch_type", "construction_year", "n_views"],
                            group_cols=["region", "arch_type", "anchor"])
    keep = {"best-fixed", "oracle"}
    sel = [s for s in sel if s[0] in keep or not s[0].startswith("eta=")]
    print(run_all(sel, F, X, etas).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
