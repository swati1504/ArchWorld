#!/usr/bin/env python
"""Segment everything Pass 1 needs: the full architectural vocabulary plus a
non-architectural list, so leakage onto sky, vegetation, etc. can be measured.

This reuses intervention/make_masks.py unchanged; it only swaps in a longer
element list per scene. Output layout is identical to make_masks.py:

    <out>/<scene>/mask/<view stem>_<element>.npy

Run in the SAM 3 environment:

    conda activate archworld-sam3
    python pass1_signals/make_masks_full.py --root data --out pass1_out/masks \
        --scenes-file scenes_dev.txt

Existing files are skipped, so it resumes after an interruption.
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "intervention"))
import make_masks  # noqa: E402  (the repo's own SAM 3 segmenter)

# Our own non-architectural vocabulary. The paper's list (its Appendix C.1) is
# not public; these are the "common non-landmark elements in internet images"
# it names (sky, trees, water) plus frequent foreground clutter.
NON_ARCH = ["sky", "tree", "bush", "grass", "water", "person", "car", "road"]


def full_elements(scene, all_elements=True):
    arch = make_masks.load_asset("elements.json").get(scene, [])
    return list(arch) + [e for e in NON_ARCH if e not in arch]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("ARCHWORLD_ROOT", "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--scenes-file", help="one scene name per line")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true",
                    help="cast SAM 3 to bfloat16 (halves memory; masks may differ slightly)")
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    args = ap.parse_args()

    scenes = list(args.scene)
    if args.scenes_file:
        scenes += [l.strip() for l in open(args.scenes_file, encoding="utf-8") if l.strip()]
    if not scenes:
        ap.error("give --scene and/or --scenes-file")

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device=args.device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    processor = Sam3Processor(model, confidence_threshold=make_masks.CONFIDENCE)
    if torch.cuda.is_available():
        print(f"SAM 3 loaded, GPU memory allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    make_masks.elements_for = full_elements   # the only change to make_masks' behaviour
    for i, scene in enumerate(scenes, 1):
        t0 = time.time()
        n = make_masks.segment_scene(processor, args.root, scene, args.out,
                                     overwrite=False, verbose=args.verbose, all_elements=True)
        print(f"[{i}/{len(scenes)}] {scene}: wrote {n} files in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
