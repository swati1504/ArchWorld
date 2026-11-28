# Landmark-selection provenance

The experiment subset was constructed in two stages. The original selector chose
40 using a coverage-elbow heuristic. The final repair kept those exact 40 and added
14 landmarks for missing supplied-anchor categories. Neither stage used
intervention or reconstruction errors. The separate 79-ID partition was excluded.

## Original 40

The archived [executed notebook](original40_run.ipynb) records library version
0.1.0 and output run `run_3181fc0dfdf53b02`. It validated encodings for all 317
development landmarks and selected 40 containing 10,544 original photograph IDs.
The exact membership is [original_40_landmarks.json](tooling/seeds/original_40_landmarks.json).
Its SHA-256 is `4f8e8885f1cf3bc0d48e92f662555c76293cf4ebf683941f27c8e232b6c36d2c`.

The initial size grid was 9, 12, 17, 24, 33, 45, 63, 87, 120, 165, 229, 316.
Refinement evaluated 37, 40, 44, 48, 52, 56, 59. Four optimization starts per
size and 1,000 random subsets with matching building-type counts were used.
Optimized candidates preserved approximate building-type proportions and covered
all nine types and eight regions. Anchor balance entered the score, but all-anchor
presence was not required by this historical version.

Landmark distances equally combined saved full-image VGGT feature means/variation,
categorical metadata, log physical extent, and photo-collection descriptors. Scales
were fitted only on development landmarks. Coverage used nearest selected distance
for unselected landmarks, excluding self-distances. A nonincreasing fit to the
average of mean and 95th-percentile coverage loss was normalized, and the largest
gain above the endpoint chord defined the elbow. The minimum prominence was 0.05,
not a representativeness probability or tolerance. The chosen size was 40.

At size 40, 16 candidate subsets were optimized with up to 256 local swap attempts
per candidate. The best used the score

`0.35 mean nearest distance + 0.35 p95 nearest distance + 0.15 mean categorical TV + 0.15 mean numeric KS`.

| Metric | Chosen 40 | Random median |
| --- | ---: | ---: |
| Mean nearest distance | 0.5107 | 0.5530 |
| P95 nearest distance | 0.6932 | 0.8009 |
| Region TV | 0.0518 | 0.1440 |
| Anchor TV | 0.1276 | 0.1999 |
| Score | 0.4406 | 0.5089 |

These were optimized descriptive features, not independent outcome validation.
The original 40 omitted 13 named anchor labels and the missing-label category.

## Final 54

The [final selector source and protocol](tooling/README.md) use version 0.4.0's
fixed-base repair. They exhaustively compare all 336 ways to add one landmark for
each missing category, keeping the original 40. The lowest score wins. Thus 54 is
the minimum repair size under these constraints, not an independent optimum.
Proportional quotas are not enforced in this final repair mode.

The [saved selection summary](evidence/point3_selection_summary.json) contains
all 54 IDs, the 14 additions, scores, and absence of overlap with the 79 evaluation
IDs. [The audit](evidence/point3_subset_audit.json) records representation checks.
The saved full-distance matrices are included for the appearance-neighbor analysis.
The two-dimensional coordinates are only visualization outputs and were not used
to select landmarks or methods.

The archived [version 0.3 protocol](historical_selector_v0.3.md) records intermediate
policies. It is not the original-40 procedure or the final fixed-54 repair. The
current tooling cannot be relabeled as the original version 0.1.0 implementation.
Full selection reruns require the external archive-verified image encodings.

The final selector is also supplied as an installable source package. From the
repository root, install it with

```bash
python -m pip install ./experiments/point3/completed_20261128/selection/tooling/library
```

The existing selector checks can be run from `selection/tooling/` with
`python -m unittest test_selection test_repair -v`. They use synthetic descriptors
and frozen metadata, not GPU inference or new benchmark outcomes.

The later 18/36 metadata split and manual Forth Bridge/Washington Monument anchor
corrections are recorded in the experiment plans under `../report/evidence/`.
Original supplied labels remain in selection provenance. The final study has
24 executed anchor labels overall and 23 in confirmation.
