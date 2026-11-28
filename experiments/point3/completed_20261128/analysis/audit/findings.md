# Evidence audit and overlap of improvements

All 35 evidence files match their archived SHA-256 checksums. The three paired-result files reproduce the combined 36-landmark table, including anchors, image counts, both controls and selected outcomes.

The 18 calibration landmarks and 36 confirmation landmarks are disjoint. All 54 are in the 317-landmark training partition and outside the 79-landmark final holdout. Full metadata covers 396 landmarks. No holdout outcomes were read.

Each statistical observation is a landmark. The 1,776 confirmation images are not 1,776 independent evaluation cases.

| Reference | At least one H improves | All three improve | None improves |
|---|---:|---:|---:|
| baseline | 28 | 14 | 8 |
| scalar | 25 | 1 | 11 |
| both | 19 | 0 | 17 |

“Both” means lower error than both unmodified VGGT and the scalar control. Counts are descriptive post hoc overlaps. They do not evaluate a selector: picking the winning method after inspecting a landmark’s ground-truth error would leak the evaluation outcome.

H3 often reuses the scalar prediction, so agreement with H3 is not independent replication. Shared controls also make correlations among method-minus-control scores mathematically dependent.

Inputs: the frozen files in `analysis/sources`; exact hashes and audit details are in `audit.json`.
