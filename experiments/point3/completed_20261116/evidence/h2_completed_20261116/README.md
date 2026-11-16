# Completed H2 internal confirmation

Downloaded from the completed H2 Drive run on November 16, 2026. Completion was
recorded at 16:06:57 UTC. `drive_provenance.json` records canonical source URLs,
file IDs, modification times, MIME types, and SHA-256 checksums. It contains no
signed download URLs or credentials.

The study completed all 18 calibration and 36 confirmation landmarks, 198 planned
scene/condition jobs, and 2,622 selected views across 54 landmarks. Confirmation
uses 1,776 views. These counts exclude additional reference checks. The recorded
failure table is empty. Numerical completion does not certify semantic mask quality.

The frozen choices are global `scalar_eta1000` and H2 `view_eta100`. These are
separately calibrated policies, not an equal-base-strength ablation. The same
confirmation scenes, images, and masks are used for every condition. The
79-landmark final holdout was not run. H1 and H3 results are not included here.

## Reproduction

From the report source folder, run:

```bash
python evidence/h2_completed_20261116/reproduce_h2_results.py
```

Python and Matplotlib are required. The script checks the frozen plan and
calibration hashes, calibration-only selections, disjoint splits, all 198
condition records, and the full 36-row paired table. It then writes
`verified_statistics.json` and `figures/point3_h2_confirmation.pdf` / `.png`.
No model inference is performed, and the numerical results are not changed.

The primary mean paired normalized-CD difference is +0.003943852850879824
(H2 minus scalar; negative favors H2), with median +0.0517673207405156 and
13 wins, zero ties, and 23 losses. Mean paired difference versus no intervention
is +0.8427159389740467. The report makes no significance, equivalence, or
full-benchmark generalization claim from this single-seed confirmation.

`run_execution_environment.json` is the coordinator's saved environment record;
its flags are not asserted to describe every worker's hook-level execution.
The runtime study code hash is preserved in `run_study.json`. It is a different
identity from the Git commit or notebook ZIP checksum. The active experiment
retained its original source bundle when the implementation was published.
