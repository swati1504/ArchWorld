# Fixed 54-landmark anchor repair

Open the [Colab selection notebook](https://colab.research.google.com/drive/1zR6LwlRXFPGHOQM1_AxkJ2HBOKlYJVr2).
Refresh it and run all cells. The setup prints version **0.4.0**. CPU is sufficient.
Existing encodings are reused; no encoder, segmentation, reconstruction or intervention runs.

## What the notebook produces

Exactly **54 training landmarks**, retaining all original photographs:

- The original 40 IDs are frozen in `seeds/original_40_landmarks.json`.
- Exactly one landmark is added for each of 13 missing named anchor labels and the
  separate `__MISSING__` category: 14 additions.
- All nine building types, eight regions and 25 anchor-label categories must appear.
- The 79 evaluation IDs are copied unchanged and never used in selection or distance fitting.
- There is no elbow search, no automatic increase in size, and no fallback to 317.

The original seed comes from `run_3181fc0dfdf53b02`, not from either later 317-landmark
output. Its exact file hash is
`4f8e8885f1cf3bc0d48e92f662555c76293cf4ebf683941f27c8e232b6c36d2c`.
It is bundled in the library, so moving old result folders will not change the seed.

**54 is the requested minimum repair size while keeping these 40 and covering every
missing category. It is not an independently established representative sample size.**
Each landmark has one supplied anchor label, so it takes at least 14 additions.

## Which 14 additions?

There are **336 possible combinations**. The selector evaluates all of them and chooses
the lowest score, breaking ties by sorted indices in the frozen training list.
This is an exhaustive minimum within this repair space, not among all 54-landmark sets.

Every candidate keeps the original 40 and adds exactly one for each missing category.
**Proportional count quotas are not enforced in this mode.** Building-type, region and
anchor distribution differences remain part of the score and are fully reported.
Retaining the original 40 can prevent exact proportional quotas from being satisfied.

The score is unchanged:
`0.35 mean distance + 0.35 p95 distance + 0.15 mean categorical TV + 0.15 mean numeric KS`.

Distances combine equally weighted appearance (full-image feature mean and variation),
type/region/anchor metadata, physical extent, and photo-collection characteristics.
Scales are fitted on the 317 training landmarks only. Each landmark has equal weight.
Coverage is measured on the 263 unselected landmarks, excluding zero self-distances.
All candidate sets have the same size, although their unselected identities differ.

The before/after plot instead uses the SAME 263 landmarks outside the winning repaired
set for both the original40 and repaired54 distances. Adding examples cannot increase
their nearest-selected distances; this is a descriptive comparison, not evidence of
improved intervention performance.

## Random comparisons

Both references contain 1,000 independently generated draws; memberships may repeat.

1. **Main reference:** random 54-landmark sets with the winner's exact building-type
   by anchor cell counts, conditional on all regions appearing. The original40 are
   not locked in these draws. This compares coverage at matched composition and is
   not uniform sampling over every possible 54-landmark set.
2. **Repair reference:** keep the original40 and independently choose one landmark
   uniformly for each missing label. There are only 336 possible memberships, so
   repeated sets are expected. The output records the distinct count.

All descriptors are used for selection and comparison. Random percentiles are descriptive,
not significance tests or independent validation. No intervention outcome is used.

## Settings and implementation

```python
from archworld_landmark_selection import RepairConfig, run_repair_selection

result = run_repair_selection(
    '/content/drive/MyDrive/point3_encodings',
    '/content/drive/MyDrive/point3_subset_selection',
    RepairConfig(seed=1729, random_subsets=1000),
)
assert result['selected_landmarks'] == 54
```

The four distance-family weights default to 1 each; the seed controls random benchmarks.
Use these decisions before looking at intervention results. No landmark-count parameter
is needed: the frozen base and missing-label count determine 54.
The earlier `SelectionConfig/run_selection` API is retained for reproducibility,
but the current notebook uses `RepairConfig/run_repair_selection`.

The shared input checks still require all 317 complete, archive-verified encodings.
They validate image identities, descriptor summaries, fingerprints and frozen input
checksums, one landmark at a time. There are no 50-view restrictions or image deduplication.
A concurrent writer is rejected. Old output folders are retained; the new run is saved
under `repair54_<fingerprint>`. Rerunning identical inputs/settings reuses completed work.

## Outputs

- `selected_landmarks.json/csv`: exactly54 final IDs and metadata.
- `fixed_base_landmarks.json`: byte-identical original40.
- `added_landmarks.json/csv`: 14 additions, with each category identified in the CSV.
- `selected_image_manifests/`: every original photo identity for each selected landmark.
- `evaluation_landmarks.json`: byte-identical frozen79.
- `subset_manifest.json`: source archives, full image counts and provenance.
- `repair_candidates.json`: all336 combinations and their metrics.
- `random_repairs.npz`, `anchor_matched_random_subsets.npz`: actual random draws and metrics.
- `audit.json`, `anchor_counts.csv`: marginal/joint distribution gaps, presence checks,
  nearest representatives, worst covered landmarks and both random comparisons.
- `size_decision.json`: explicit fixed-size reason, without an elbow claim.
- `REPORT.md`, `coverage_comparison.png`, `composition_balance.png`: readable findings.
- `selection_protocol.json`: settings, code/input hashes and limitations.

Raw photos are not copied. Metadata labels do not prove element visibility or usable
masks. Camera coverage, blur, exposure and reconstruction accuracy are not measured.
Test intervention hypotheses separately, and retain the held-out evaluation set.

## Checks

`python -m unittest test_selection test_repair -v`

These check original40 retention, all336 repairs, all-category presence, fixed54 output,
1,000-draw references, same-target coverage, seed integrity, held-out exclusion, complete
image manifests and resumability. Historical policy documentation is retained in
`snapshots/README_before_fixed54.md`.
