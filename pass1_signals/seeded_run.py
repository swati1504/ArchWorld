#!/usr/bin/env python
"""Run intervention/run.py unchanged, but reseed the random generators before every scene,
and optionally trim work whose output is not needed.

The VGGT pass aligns prediction to ground truth with random subsampling and RANSAC drawn
from NumPy's global generator, which run.py never seeds; the draw for a scene also depends
on every scene run before it in the same process. This wrapper replaces the pass's `main`
with one that first calls np.random.seed / random.seed / torch.manual_seed(SEED), then
hands every argument to the real run.py. Nothing in the repository is modified: the
wrapping happens on the imported module object.

Options (before the `--`):
  --seed N              seed set before every scene (required)
  --trim                skip work that does not affect the saved prediction, for runs with
                        SKIP_EVAL=1 and SAVE_RAW_TENSORS=1: the prediction GLB export, the in-run
                        ground-truth build (only scoring uses it), and the patch montage after
                        the first run of each scene in this process. The attention plot is kept.
  --skip-existing-raw   skip a (scene, eta) whose work dir already has inter_raw.npz
                        (run.py itself only skips when an eval_metrics file exists)

    python pass1_signals/seeded_run.py --seed 0 -- --model vggt --root data \
        --masks-dir pass1_out/masks --scene Dendera --eta 0 --out pass1_out/noise_probe/s0_r1
"""
import importlib
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def parse(argv):
    if "--" not in argv:
        sys.exit(__doc__)
    own, run_args = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    if "--seed" not in own:
        sys.exit(__doc__)
    seed = int(own[own.index("--seed") + 1])
    return seed, "--trim" in own, "--skip-existing-raw" in own, run_args


def main():
    seed, trim, skip_raw, run_args = parse(sys.argv[1:])
    sys.path[:0] = [str(REPO / "intervention"), str(REPO)]

    # The pass reads its configuration from the environment at import time, and run.py sets
    # that environment just before importing it. So wrap the module at that import, not earlier.
    real_import = importlib.import_module
    montage_done = set()

    def import_and_wrap(name, *a, **k):
        module = real_import(name, *a, **k)
        if not name.endswith(".run_pass") or getattr(module.main, "_seeded", False):
            return module
        real_main = module.main

        if trim:
            real_vis = module.visualize_intervention_patches

            def no_glb(*a2, **k2):
                print("  [seeded_run --trim] prediction GLB export skipped", flush=True)

            def no_gt(*a2, **k2):
                print("  [seeded_run --trim] in-run GT build skipped (scoring is done separately)", flush=True)
                return None

            def vis_once(*a2, **k2):
                scene = module.os.environ.get("SCENE_NAME", "")
                if scene in montage_done:
                    print("  [seeded_run --trim] patch montage skipped (already made for this scene)", flush=True)
                    return None
                montage_done.add(scene)
                return real_vis(*a2, **k2)

            module.save_prediction_glb = no_glb
            module.generate_gt_by_unproject = no_gt
            module.visualize_intervention_patches = vis_once

        def seeded_main(*a2, **k2):
            work_dir = k2.get("work_dir")
            if skip_raw and work_dir is not None and (Path(work_dir) / "inter_raw.npz").exists():
                print(f"[seeded_run] {k2.get('scene_name')}: inter_raw.npz exists, skipping", flush=True)
                return k2.get("model")
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            print(f"[seeded_run] seed={seed} trim={trim} for {k2.get('scene_name')}", flush=True)
            return real_main(*a2, **k2)

        seeded_main._seeded = True
        module.main = seeded_main
        return module

    importlib.import_module = import_and_wrap
    sys.argv = [str(REPO / "intervention" / "run.py"), *run_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
