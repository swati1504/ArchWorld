# Completed H1 internal confirmation

The H1 run completed on November 16, 2026 at 20:02:32 UTC. These snapshots
contain all 18 calibration and 36 confirmation landmarks, 198 planned
scene/condition jobs, and no recorded failures. Extra reference checks are not
counted as planned jobs. The 54 landmarks use 2,622 views; confirmation uses
1,776 views. The separate 79-landmark final holdout was not evaluated.

`drive_provenance.json` records canonical Drive URLs, file IDs, modification
times, MIME types, and SHA-256 checksums. It contains no temporary download
URLs or credentials. `run_session_1_0.json` confirms both run and report completion.

Calibration separately selected `scalar_eta1000` and `layer_alpha0.02`.
The coefficient grids were scalar eta in {100, 1000} and layer alpha in
{0.01, 0.02}. They were frozen using calibration scores only.

## Results and scope

The primary mean paired normalized-CD difference is -0.6224070615122497
(H1 minus scalar; negative favors H1), with median -0.04327081474448885:
23 wins, zero ties, and 13 losses. Against no intervention, mean difference
is +0.21636502461091725, median difference is -0.029649546862844267,
and 21 landmarks improve while 15 worsen. Thus H1 improves mean error over
the calibrated scalar in this cohort, but not over the unmodified model.

This is descriptive, single-seed internal confirmation with a two-candidate
layer grid. It does not establish statistical significance or performance on
the complete benchmark. Numerical mask checks do not certify semantic quality.
H3 remains unfinished and is not reported here.

## Reproduction

From the report source folder, using Python and Matplotlib:

```bash
python evidence/h1_completed_20261116/reproduce_h1_results.py
python evidence/plot_completed_hypotheses.py
```

The first script checks snapshot hashes, completion, disjoint splits, frozen
selection hashes, calibration-only choices, all condition records, and all
36 paired rows. It writes `verified_statistics.json` and a standalone H1 plot.
The second checks H1/H2 snapshot integrity and exact agreement of their
independently run baseline and scalar controls, selected-view counts, and
confirmation scenes. It produces the combined table data and two-panel figure.
The two panels sort their landmarks independently. No GPU inference is rerun.

`run_execution_environment.json` is the coordinator's saved environment record;
it is not asserted to describe every worker's hook-level flags. The resumed
runtime's package inventory is preserved separately. The study source hash is
distinct from the published Git commit and the notebook source ZIP checksum.
