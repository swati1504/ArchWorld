# Whole-landmark subset selection

Open the [Colab subset-selection notebook](https://colab.research.google.com/drive/1zR6LwlRXFPGHOQM1_AxkJ2HBOKlYJVr2).

**Version 0.3.0 relaxes anchor counts by one additional landmark.** Every supplied
anchor label, plus the separate `__MISSING__` category, must still appear. Its count
may extend one below the floor or one above the ceiling of its proportional target,
capped by training availability and with a minimum of one. Building-type quotas,
region presence, the coverage size rule and the evaluation split are unchanged.
On the frozen 317-landmark metadata, the smallest jointly feasible size is **48**
with `anchor_count_slack=1`. Set this to 0 to restore the strict version 0.2.0 rule
and its minimum of 120. These are feasibility boundaries, not selected sizes or
representativeness certificates. The earlier 40-landmark result remains historical;
it omits anchor labels and fails both policies.
Reuse the existing encodings; no GPU extraction is needed. Refresh the notebook and
rerun its cells to install 0.3.0. The setup reloads an already-imported older version.

This CPU stage consumes the completed `point3_encodings` directory. It never runs an
encoder, reconstruction, intervention, or new photo download. It selects landmarks,
keeps all their original image paths, and assesses representation against **317 training
landmarks only**. The frozen **79 evaluation IDs** are copied unchanged to the results.

Use the separate `Point3_Select_Landmark_Subset.ipynb` notebook after extraction finishes.
The installable `archworld-landmark-selection` package is independent of the encoder
package. Its readable source and frozen inputs are included in the wheel.

```python
from archworld_landmark_selection import SelectionConfig, run_selection
result = run_selection(
    '/content/drive/MyDrive/point3_encodings',
    '/content/drive/MyDrive/point3_subset_selection',
    SelectionConfig(),
)
print(result)
```

Standalone, from this repository:

```text
python experiments/point3/landmark_selection/select_landmarks.py --encodings PATH_TO_ENCODINGS --out PATH_TO_SELECTION_OUTPUT
```

Install `numpy`, `scipy`, and `matplotlib` first for standalone use. Optional `--config`
accepts a JSON object of `SelectionConfig` overrides. The notebook installs dependencies.

## Validation before selection

The full frozen training request must be complete, with no recorded failures and explicit
holdout exclusion. The selector checks input checksums, archive revision, manifest hashes,
image names and counts, per-image normalized features, and summary consistency. It accepts
the original archive-verified download mode, not incompletely inventoried local directories.
Evaluation feature directories are never opened. A missing anchor is a real missing-label
category, not a reason to exclude a candidate. No old 50-view eligibility rule is used.

It reads one landmark's feature file at a time. Reproducible summaries and source hashes
are cached; the search then works with 317-by-317 matrices. A resumed run validates source
hashes before reusing work. Parameters, code hash, input hashes, encoder settings and
source identities identify each output directory. Avoid concurrent writers to one output
root; if a process is forcibly killed, its `.selector.lock` must be removed after confirming
that the process stopped. Normal errors and interrupts release the lock.

## Descriptors and distances

Every landmark receives equal weight regardless of its image count. Four distance
families have equal default weight:

1. **Appearance:** Euclidean distances between full-collection feature means and between
   feature standard deviations. Their pairwise median scales are fitted on training only;
   the two distances are averaged. This retains variation but does not describe all modes.
2. **Composition:** average mismatch across building type, region and supplied anchor.
3. **Physical scale:** absolute difference in log(1 + extent in meters).
4. **Photo collection:** Euclidean distance between standardized log image count,
   duplicate fraction, log median/10th/90th-percentile megapixels, portrait fraction,
   and mean/standard deviation of log aspect ratio. Photos are not dropped or deduplicated.

Each family is divided by the median of its positive pairwise training distances.
Constant dimensions contribute zero; missing values are not invented. Extent metadata
comes from the existing training table. Supplied anchor labels are not anchor-visibility
measurements or a complete architectural-element inventory.

## Size selection, then equally sized candidates

The number of selected landmarks is not fixed to 20, 62, or a runtime budget.

For every tested size, type quotas minimize squared deviation from ideal proportional
counts, subject to at least one of each type and available counts. Incremental allocation
uses smallest marginal squared error, breaking ties in sorted type order. Optimized
candidates must also include every training region. Exact unweighted proportions require
the full population; rounding differences are always reported.

For each anchor label a, let p_a = N_a / 317 and s = `anchor_count_slack` (default 1).
Require `max(1, floor(n*p_a)-s) <= selected_count_a <= min(N_a, max(1, ceil(n*p_a)+s))`.
The slack is a count of **additional landmarks beyond rounding**, not a percentage
or a guarantee of coverage. It is validated as a nonnegative integer and recorded
in the run settings, audit and report. It applies to all 24 named labels and the
missing-label category. For example, at size 48 the arch target is 14.69 and its
permitted count is 13–16; dome and minaret labels must each still appear. Actual
joint constraints may further restrict a label's count. Rare one-landmark labels
force that landmark to appear; the report exposes their increased subset shares.
These are constraints on supplied labels, not proof of visible elements or usable masks.

Scan candidate sizes in ascending order to find the first jointly feasible type,
region and anchor allocation. Sizes ruled out by summed anchor bounds or a proven
infeasible solver result are rejected. A solver timeout is an error, never evidence
of infeasibility. The initial grid has 12 geometrically spaced sizes from the first
feasible size through 316. Other infeasible grid sizes are reported and excluded.

At each feasible size, retain the earlier 1,000 type-only random draws as a secondary
reference, recording how often they satisfy anchor constraints. Four optimization
starts use hard-feasible type/region/anchor solutions. A random candidate may seed
optimization only if all hard constraints pass. Every proposed within-type swap is
checked against region coverage and anchor bounds, and every export is checked again.
The configured slack is never silently increased to retain a small subset.

Candidate score (lower is better):

`0.35 mean nearest distance + 0.35 p95 nearest distance + 0.15 mean marginal TV + 0.15 mean numeric KS`.

Nearest-selected distance uses **only unselected training landmarks**. Selected landmarks
never inflate coverage with zero self-distances. TV averages building type, region and
anchor gaps. KS averages extent, image count, duplicate fraction and median megapixels.

To propose size, average the mean and p95 nearest distances at each curve point. Fit a
nonincreasing curve using pool-adjacent-violators. Normalize size and coverage gain to
[0,1], then find the largest coverage gain above the straight endpoint chord. Refine
the two neighboring size intervals with nine linearly spaced points and recompute once.
Require an interior maximum at least 0.05 above the chord; otherwise use all 317 training
landmarks rather than claim an unsupported small subset. The raw curve, monotone fit,
evaluated sizes, and mean-only/p95-only sensitivity choices are saved.

This elbow is a **declared heuristic**, not a statistical acceptance bound, proof of
minimum size, or certificate of representativeness. The monotone fit summarizes noisy
optimized coverage values; raw values remain visible. Different grids/weights can change
the result, and sensitivity disagreement should be discussed in the report.

At the chosen proper-subset size, optimize 16 candidates (the first starts from the best
curve candidate; other starts are randomized feasible solutions). Choose the lowest score,
breaking ties by sorted landmark indices. Save all candidates, and the number of distinct
memberships, since independent randomized starts may converge to identical sets.

The primary final random comparison now uses 1,000 **anchor-preserving** draws.
Fix the winner's exact building-type × anchor cell counts, sample uniformly without
replacement within each cell, and reject a draw if any region is missing. Accepted
draws are independent and uniform over that conditional space, not over every
possible feasible subset. Anchor/type counts match exactly; region proportions can
vary while every region remains present. Every accepted draw is checked against all
hard constraints. Report attempted/accepted draws and distinct memberships; repeats
are possible. The original type-only baseline remains a separately labeled reference.

## Parameters to decide before intervention results

| Parameter | Default | Effect |
|---|---:|---|
| seed | 1729 | Reproducible baseline draws and candidate starts |
| anchor_count_slack | 1 | Additional landmarks below floor / above ceiling of each anchor target; 0 restores strict counts; every label remains required |
| curve_points | 12 | Initial size grid resolution |
| refinement_points | 9 | Local size refinement resolution |
| curve_candidates | 4 | Search starts per tested size |
| final_candidates | 16 | Candidates at the selected common size |
| swap_attempts | 256 | Local search effort per candidate |
| random_subsets | 1000 | Protocol's random reference count per tested size |
| four family weights | 1 each | Appearance/composition/scale/collection tradeoff |
| knee_min_gain | 0.05 | Minimum normalized elbow prominence, **not** a representation tolerance |

These defaults are algorithm choices. They must not be tuned against held-out data or
intervention outcomes. Score coefficients are fixed and recorded in the protocol.

## Saved outputs

- `selected_landmarks.json`, `selected_landmarks.csv`, `subset_manifest.json`: chosen IDs,
  all-image policy, photo counts, pinned original archives, and selection provenance.
- `selected_image_manifests/`: byte-identical complete encoder manifests, including all
  image names and hashes. **Raw images are not copied.**
- `evaluation_landmarks.json`: byte-identical frozen 79-ID holdout, with no feature audit.
- `selection_protocol.json`: settings, source hashes, explicit score and size rule.
- `anchor_counts.csv` and `audit.json.anchor_constraints`: every anchor's training
  count, proportional target, permitted range, selected count and pass/fail check.
- `size_curve.json/csv`, `size_decision.json`, `coverage_curve.png`: measured size tradeoff.
- `candidates/`: all final candidates and scores.
- `sizes/N/random_subsets.npz`: every random selection as indices into the saved training
  ID order, with its metrics. This is the secondary type-only reference.
- `anchor_matched_random_subsets.npz`: the primary final reference with the same
  type-by-anchor counts and all regions; every draw preserves the anchor constraints.
- `audit.json`, `REPORT.md`, `building_type_balance.png`: marginal/joint composition,
  numerical distribution gaps, worst covered unselected landmarks, omitted labels, and
  descriptive random comparisons. Family-specific coverage uses each family's nearest
  selected neighbor, which can differ from the composite nearest neighbor.

No raw-image materialization or intervention is run. These files define the complete
photo collections for later experiments; memory strategy for reconstruction is separate.

## What this does not establish

Appearance includes backgrounds. Mean/variance pooling can miss rare views. Image quality
is represented only by dimensions and aspect ratios: blur/exposure are unmeasured. Camera
coverage, anchor visibility, depth/mask completeness and intervention behavior are not
available from the saved encoder outputs. The audit explicitly lists these omissions.
Optimized metrics and random percentiles are descriptive, not independent validation or
p-values. Later experiments must test whether method effects survive landmark resampling.

## Checks

```text
python -m unittest discover -s experiments/point3/landmark_selection -p test_selection.py -v
python experiments/point3/landmark_selection/build_delivery.py
```

Offline checks use synthetic features, not intervention outcomes. A separate compatibility
check can validate a completed real Colab landmark without selecting a subset prematurely.
