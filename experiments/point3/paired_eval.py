"""Study-only entry point: identical evaluation RNG for baseline and intervention.

The reference pass and metric formulas are unchanged. This wrapper resets and
restores NumPy's RNG around each evaluation and records prediction fingerprints.
It must not be mixed with the earlier pilot's unpaired evaluation results.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

PROTOCOL = "point3-paired-eval-v1"


def paired_worker(original, seed):
    import numpy as np

    def wrapped(*args, **kwargs):
        # Signature of the existing VGGT evaluation worker; no tensor mutations.
        prediction, folder, pass_name = args[2], Path(args[4]), args[8]
        fingerprints = {}
        for name in ("depth_map", "depth_conf", "extrinsics", "intrinsics"):
            value = prediction[name]
            array = value.detach().cpu().float().numpy()
            fingerprints[name] = hashlib.sha256(array.tobytes()).hexdigest()
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            result = original(*args, **kwargs)
        finally:
            np.random.set_state(state)
        (folder / f"{pass_name}_evaluation.json").write_text(json.dumps({
            "protocol": PROTOCOL, "seed": seed, "prediction_sha256": fingerprints,
            "status": result[0],
        }, indent=2), encoding="utf-8")
        return result

    return wrapped


def main():
    source = Path(os.environ["POINT3_SOURCE"])
    sys.path[:0] = [str(source / "intervention"), str(source)]
    fast = os.environ.get("POINT3_FAST_MODE") == "1"
    cache = None
    if fast:
        from fast_eval import prepare_baseline_environment
        cache = prepare_baseline_environment()
    module = importlib.import_module("models.vggt.run_pass")
    if module.EVAL_TIMEOUT_SEC != 0:
        raise RuntimeError("Paired study requires in-process evaluation (EVAL_TIMEOUT_SEC=0)")
    module._run_alignment_and_eval_worker = paired_worker(
        module._run_alignment_and_eval_worker, int(os.environ["POINT3_EVAL_SEED"]))
    if fast:
        from fast_eval import install, attach_baseline_audit
        install(module, os.environ["POINT3_GT_CACHE"])
        if cache is not None:
            attach_baseline_audit(module, cache)
    from intervention.run import main as run_main
    run_main()


if __name__ == "__main__":
    main()
