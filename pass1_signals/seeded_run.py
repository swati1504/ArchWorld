#!/usr/bin/env python
"""Run intervention/run.py unchanged, but reseed the random generators before every scene.

The VGGT pass aligns prediction to ground truth with random subsampling and RANSAC drawn
from NumPy's global generator, which run.py never seeds; the draw for a scene also depends
on every scene run before it in the same process. This wrapper replaces the pass's `main`
with one that first calls np.random.seed / random.seed / torch.manual_seed(SEED), then
hands every argument to the real run.py. Nothing in the repository is modified.

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


def main():
    argv = sys.argv[1:]
    if "--" not in argv or argv[0] != "--seed":
        sys.exit(__doc__)
    seed = int(argv[1])
    run_args = argv[argv.index("--") + 1:]

    # The pass reads its configuration from the environment at import time, and run.py sets
    # that environment just before importing it. So wrap the module at that import, not earlier.
    real_import = importlib.import_module

    def import_and_wrap(name, *a, **k):
        module = real_import(name, *a, **k)
        if name.endswith(".run_pass") and not getattr(module.main, "_seeded", False):
            real_main = module.main

            def seeded_main(*a2, **k2):
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                print(f"[seeded_run] seed={seed} for {k2.get('scene_name')}", flush=True)
                return real_main(*a2, **k2)

            seeded_main._seeded = True
            module.main = seeded_main
        return module

    importlib.import_module = import_and_wrap
    sys.argv = [str(REPO / "intervention" / "run.py"), *run_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
