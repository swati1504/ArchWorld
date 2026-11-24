#!/usr/bin/env python
"""Demo figure: what depth drift looks like on a caught failure, a safe scene, and the one miss.

For each scene it shows the input photo, the predicted depth at eta=1 and eta=5000, and the
per-pixel relative depth change between them (the quantity whose median is the drift score).
The view shown is the one with the largest per-view drift. Runs on CPU from saved predictions.

    python swati/make_demo_figure.py --results results/vggt_test --out swati/figures/drift_demo.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SCENES = [
    ("Gangaikonda_Cholapuram_Temple", "Caught: drift 0.48, fell back to η=1000\n(η=5000 was 18.7× worse)"),
    ("Archiepiscopal_Palace_of_Lima", "Safe: drift 0.002, kept η=5000\n(η=5000 slightly better)"),
    ("Degutale_Temple,_Patan", "Missed: drift 0.005, kept η=5000\n(η=5000 was 65% worse)"),
]


def depth(path):
    z = np.load(path)["depth_map"]
    return z.reshape(z.shape[1], *z.shape[2:4]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/vggt_test"))
    ap.add_argument("--data", type=Path, default=Path("data/archworld"))
    ap.add_argument("--views", type=Path, default=Path("intervention/assets/views.json"))
    ap.add_argument("--out", type=Path, default=Path("swati/figures/drift_demo.png"))
    a = ap.parse_args()
    views = json.load(open(a.views))

    fig, axes = plt.subplots(len(SCENES), 4, figsize=(13, 3.3 * len(SCENES)))
    titles = ["Input photo", "Depth, η=1 (near untouched)", "Depth, η=5000", "Relative depth change"]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=12)

    for r, (scene, label) in enumerate(SCENES):
        d1 = depth(a.results / "output_1" / scene / "inter_raw.npz")
        d5 = depth(a.results / "output_5000" / scene / "inter_raw.npz")
        ok = (d1 > 1e-4) & (d5 > 1e-4)
        rel = np.where(ok, np.abs(d5 - d1) / np.maximum(d1, 1e-4), np.nan)
        scene_drift = float(np.nanmedian(rel))
        per_view = [np.nanmedian(rel[v]) if np.isfinite(rel[v]).any() else -1 for v in range(len(rel))]
        v = int(np.argmax(per_view))

        names = views.get(scene, [])
        photo = None
        if v < len(names):
            p = a.data / scene / "images" / names[v]
            if p.exists():
                photo = Image.open(p).convert("RGB")
        ax = axes[r, 0]
        if photo is not None:
            ax.imshow(photo)
        else:
            ax.text(0.5, 0.5, "photo not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(scene.replace("_", " ") + "\n\n" + label, fontsize=10, rotation=0,
                      ha="right", va="center", labelpad=10)

        lo, hi = np.nanpercentile(np.where(ok[v], d1[v], np.nan), [2, 98])
        for c, dm in ((1, d1[v]), (2, d5[v])):
            axes[r, c].imshow(np.where(dm > 1e-4, dm, np.nan), cmap="viridis", vmin=lo, vmax=hi)
        im = axes[r, 3].imshow(rel[v], cmap="magma", vmin=0, vmax=0.3)
        axes[r, 3].text(0.02, 0.04, f"scene drift {scene_drift:.4f}\nview {v}: {per_view[v]:.4f}",
                        transform=axes[r, 3].transAxes, color="white", fontsize=9, va="bottom",
                        bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
        print(f"{scene}: scene drift {scene_drift:.4f}, shown view {v} ({names[v] if v < len(names) else '?'}), "
              f"view drift {per_view[v]:.4f}")

    cb = fig.colorbar(im, ax=axes[:, 3], fraction=0.046, pad=0.02)
    cb.set_label("|depth(5000) - depth(1)| / depth(1)  (clipped at 0.3)")
    fig.suptitle("Depth drift flags when a large step breaks the reconstruction (no ground truth used)",
                 fontsize=14, y=0.995)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
