# Point 3: faster shared experiments on the 54-landmark subset

For **one independently executable notebook per hypothesis**, see
[Three hypothesis experiment notebooks](HYPOTHESIS_NOTEBOOKS.md). Each includes its
own controls, calibration, confirmation and report and can run in a separate GPU
session. First run the dedicated `Point3_00_Prepare_Masks_UpTo50.ipynb` once; it imports
compatible existing masks and generates only missing ones. All three hypotheses
read the completed 54-landmark cache, whose actual mask count is recorded. The four-notebook workflow also
selects CUDA 12.8 environments for G4/Blackwell, subject to the real GPU pilot.
The shared workflow below remains available as a separate option.

This implementation tests whether ArchAttend's intervention strength should vary
per layer (H1), per view (H2), or per architectural anchor (H3). One GPU execution
notebook produces all three experiments; three separate CPU notebooks analyze
their results. It retains the user's 54 selected landmarks and **all eligible distinct
photographs up to 50 per reconstruction**. On 2026-11-15 the user explicitly chose
to keep this exact cohort when runtime inspection found scenes with fewer views.

CPU regression tests cover the implementation. The independent H1 and H2 notebooks
have passed their 50-view memory and exact reference checks on NVIDIA L4 and have
produced calibration checkpoints. The complete three-hypothesis confirmation study
is still in progress; no full-cohort benefit or measured parallel speedup is claimed.
Each new runtime must pass its own reference/memory pilot. The session limit is
eight hours, with resume support; it is not a runtime estimate or a promise that
the study fits within an hour.

## Run in Colab

Use `dist/Point3_Shared54_Run.ipynb`. Keep it and the matching, checksum-verified
`point3_shared54_source_<hash>.zip` in `MyDrive/Point3_Shared54`.

1. Select a GPU runtime. Fifty-view intervention must pass the memory pilot;
   the earlier five-view T4 result does not establish feasibility at 50 views.
2. Add `HF_TOKEN` to Colab Secrets and enable notebook access. The account must
   already have approved SAM 3 access. The token is passed only to child process
   environments, redacted from captured logs, and never embedded in the notebooks.
3. Run the settings and setup cells. Compatible VGGT/SAM environments are reused;
   otherwise isolated environments are installed. Source/model-code revisions
   are fixed and installed package versions are saved.
4. Run the pilot cell. It executes seven calibration conditions on one landmark
   and compares a fresh scalar reference against a cached run with a reused model.
   Prediction fingerprints must match exactly; every numeric metric must match
   within relative `1e-6`, absolute `1e-9`. The zero control also checks identical
   predictions and nearly identical metrics. Failure blocks the bulk run.
5. Run the shared study cell. Calibration finishes before any confirmation begins.
   Keep the same source bundle, calibration size, run label and output folder to resume.
6. Open `Point3_H1_Report.ipynb`, `Point3_H2_Report.ipynb` and
   `Point3_H3_Report.ipynb`. They only read shared outputs and require no GPU.

An OOM stops the affected worker. The code never silently reduces the number of
views, changes precision, weakens evaluation, or drops a failed landmark.

## Data and frozen split

`inputs/selection54.json` copies the saved 54-landmark selection audit, with
14,333 source images reported by that audit. It is a frozen ID selection rather
than an assertion that all archives have already been downloaded or validated.
All 54 IDs belong to the 317-landmark training partition and are disjoint from
the 79 final-evaluation IDs in `inputs/evaluation_landmarks.json`.

Default: **18 calibration + 36 internal confirmation landmarks**. A deterministic
metadata-only search considers 512 seeded allocations. It allocates at least two
calibration landmarks to each of six supported anchors while retaining at least
one confirmation example, then favors building-type/region coverage and smaller
marginal total variation from the 54-landmark set. No reconstruction outcomes
enter selection. This does not establish statistical representativeness of all
396 landmarks. The final 79 are not automatically run or tuned on.

Supported H3 calibration types: **arch, column, facade, statue, tower, window**.
Other types use a declared global-scalar fallback. There is only one dome and one
minaret in this subset, and no spire; this study cannot independently establish
dome/minaret/spire-conditioned improvements. Two calibration examples per type
also provide limited evidence even for supported types.

Forth Bridge lacks a supplied intervention anchor. On 2026-11-15 the user explicitly
authorized **bridge** as a manual anchor. `protocol.py` records both supplied and
effective labels and the override provenance. It does not rewrite the original
anchor asset or portray this override as the paper's attention-selected anchor.

Washington Monument's supplied **altar** anchor produced **0 nonempty masks out of
50 selected views** in the first mask run. On 2026-11-15 the user authorized the
manual **obelisk** correction. It is a documented data/anchor correction, not an
attention-selected label or a reconstruction-result-driven choice. Preserve the
original 18 calibration / 36 confirmation allocation (Washington remains in
confirmation); H3 uses the global-scalar fallback for obelisk. All 50 overlays
require review. The other 53 verified preparations are reused from the shared
cache without downloading their inputs or rerunning SAM. The failed altar masks
and original run remain available for the audit trail.

### Up to 50 photographs and mask validation

Dataset archives are pinned to revision
`5d4404323a21aa3de948f8bc57a053b29fa3d921` and verified against their LFS SHA256.
For each landmark, reconstruction inputs must be registered RGB photographs with
unambiguously matched depth and whole-landmark validity masks. Byte-identical RGB
copies are deduplicated **only for reconstruction input selection**. The original
54-landmark dataset remains unchanged. Select `min(50, eligible_distinct_count)`
by seeded uniform sampling, then sort original paths; view 0 is the reference.
When fewer than 50 eligible images exist, all are selected. At least two are required.

This is an explicit new view-selection protocol, not reuse of the supplied
basename-only view lists. It avoids ambiguous nested-path matches. There is no padding,
duplication or replacement of landmarks. Actual counts appear in `input_manifest.json`,
`view_counts.csv`, checkpoint identities, per-condition and paired results, and the
combined comparison. Metrics are paired on identical views for each landmark and
averaged with equal landmark weights, not weighted by image count.
Collision-safe aliases are mapped to original paths
in `input_manifest.json`; only COLMAP image names are rewritten, preserving poses,
cameras, tracks, RGB/depth/validity contents. This protocol must not be pooled with
the earlier five-view pilot or represented as an exact reproduction of paper views.

Zero-filled depth/validity maps are preserved and audited as views with no ground-truth
support, matching the reference evaluator's pixel filtering. Malformed or nonfinite
arrays still fail. No missing ground truth is fabricated; scene-level alignment and
evaluation must still succeed. A UTF-8-safe COLMAP writer preserves non-ASCII names
for unselected images along with all poses and tracks; the reference metric code is unchanged.
Dataset downloads use the supplied `HF_TOKEN` privately instead of forcing anonymous access.

SAM 3 runs once per anchor/image set, at confidence 0.5. Every selected-view mask must exist,
be finite binary masks at the original image dimensions (empty arrays explicitly
represent absence), and at least two views must have nonempty masks. Every
intervention layer must then have nonzero anchor tokens. `mask_validation.json`
records per-view coverage; `mask_preview.jpg` shows six views normally and **all
selected Forth Bridge views**. Numeric checks do not verify semantic correctness: review
the overlays before drawing architectural conclusions. Forth Bridge's actual
SAM outputs have not yet been generated/visually validated by this delivery.

## Hypotheses and decision variables

Let `T_l` be the token activations, `g_l` the existing VGGT attention-objective
gradient, and `M` the anchor mask. Preserve the reference loss scale (`1e5`),
mean-based objective, layers 0–23, all-token update support and accept/revert rule.
Weights are float32 with the reference autocast policy; `VGGT_HALF` is disabled
for this study. The reference's q/k normalization and rotary conventions are retained.

| Policy | Rule | Calibration decision | Fixed choices |
| --- | --- | --- | --- |
| No intervention | eta = 0 | None | Same inputs and paired evaluation |
| Global scalar | `T_l + eta * g_l` | eta in {100, 1000}, minimizing mean calibration CD_norm | One eta for all layers/views/anchors |
| H1: layer | `eta_l = min(1e5, alpha * norm(T_l[M]) / max(norm(g_l[M]), 1e-8))` | alpha in {0.01, 0.02} | Same accept/revert rule, all-token update |
| H2: view | `eta_v = eta * q_v / RMS(q)`; `q_v = clip(((n_v+1)/(mean(n)+1))^0.5, 0.5, 2)` | eta in {100, 1000} | Anchor-token counts at first hook; view coefficients stay fixed across layers |
| H3: anchor | `eta = table[anchor]` | Per-type eta in {100, 1000}, using only that type's calibration landmarks | At least two calibration landmarks; otherwise global scalar fallback |

These two-point grids are preliminary choices, not the paper's complete
`{1,10,100,1000,5000}` eta sweep or proof that an optimum was located. Ties choose
the smaller eta/alpha. Hyperparameters are frozen before confirmation, never
selected separately on a confirmation landmark. H3 shares scalar predictions
with the global comparison whenever their selected values coincide.

Parameters that could be tuned in a later, separately versioned study are eta/alpha
grids, H2 exponent/clipping, minimum H3 support, the 18/36 allocation, SAM threshold,
and random seed. Altering them after inspecting confirmation results spends that
confirmation set; use a new held-out assessment for a final claim. The current
implementation deliberately does not search all these dimensions.

## What saves time

| Execution design | Scene/condition jobs |
| --- | ---: |
| Three independent full-54 sweeps | 702 |
| One shared full-54 sweep | 378 |
| Default 18/36: calibration | 126 |
| Default 18/36: confirmation | At most 180 |
| Default total | **At most 306**, plus reference checks/replays |

This cuts scheduled jobs by at least 56.4% versus 702, before gate overhead.
Jobs have different costs; this is **not a measured 56.4% wall-clock speedup**.
The 18/36 design also changes the evaluation design versus sweeping all 54.

- Load VGGT once per landmark and retain it across conditions; reset hooks,
  controller/environment state and RNG. A fresh reference model is extra gate work.
- Reuse the baseline metric/audit from the zero control for nonzero conditions.
- Cache exactly matching ground truth and processed image tensors within a scene.
- Reuse SAM masks across every hypothesis and across interrupted sessions.
- Deduplicate H3/global-scalar jobs where the selected eta is identical.
- Skip visualization-only 3D exports; preserve the original reconstruction metric
  computation, alignment and ICP settings. Scientific plots are created from results.
- Validate checkpoint identity, ZIP integrity, trace completeness and normalized
  score consistency before resume. Completed scene inputs are removed only from
  runner-owned local scratch; persistent masks/checkpoints remain in Drive.

Changing source/settings creates a new study identity; incompatible caches are
rejected. Incomplete calibration or missing candidate scores cannot freeze a
selection. Failures remain explicit in reports. Do not remove a stale lock until
the owning process has stopped. Notebook interruption signals cancel each active
worker process group before the coordinator exits; tests verify child termination.

## Multiple GPUs or independent workers

On a machine with matching GPUs, use the evaluation environment:

```bash
python experiments/point3/shared54/run.py --output /persistent/point3_run \
  --work /local/point3_work --sam-python /envs/sam3/bin/python \
  --devices 0,1 --hours 8 --phase all
```

Each GPU runs one landmark at a time and owns all its conditions. The scene bins
are disjoint and balanced by source-archive bytes, a rough workload proxy.
No extra GPUs are provisioned by this code. Colab usually exposes a single device;
the implementation does not combine multiple GPUs for one 50-view forward pass.

Independent workers can share persistent results using `--shards N --shard-index i`.
Initialize the output folder once with `--phase plan`, and run one pilot on the
coordinator to establish the GPU/software profile. Run all shards with
`--phase calibration`, wait for all of them, then run
`--phase freeze` **once** with one coordinator. Run disjoint confirmation shards,
wait for them, then `--phase report` once. Do not use `--phase all` with shards or
run overlapping shard assignments. Use separate local scratch directories. All
workers must use identical source/config and matching GPU models/software; each
job records its hardware. The saved execution profile rejects changes across sessions.

`--calibration-size 54` is an optional 378-job shared sweep, with a different output
folder. It provides calibration-only data and **no internal confirmation cohort**;
its reports must not be presented as held-out evidence.

## Outputs, metric and interpretation

Primary metric, matching `intervention/aggregate.py`:

`CD_norm = ((Acc-mean + Comp-mean) / 2) * Scale * 1000 / Extent`.

Use the paired landmark difference `structured − global_scalar` on the 36
confirmation landmarks; negative is better. Also compare against no intervention.
Report mean/median differences, wins/ties/losses and completeness, and inspect
accepted steps and effective strengths. H3 separately reports supported-anchor
and fallback cases; identical predictions are ties, not independent experiments.
Evaluation resets the existing NumPy RNG identically for baseline/intervention.
The final selection rule differs from the paper's per-scene oracle-best aggregate;
do not compare those two aggregates as though they were the same protocol.

Persistent outputs:

- `study.json`, `selection.json`, `calibration_scores.json`, `job_budget.json`:
  frozen inputs, choices and provenance.
- `results.csv`, `failures.csv`, `strength_diagnostics.csv`, `summary.json`:
  all conditions, completeness and effective-strength diagnostics.
- `reports/H1`, `reports/H2`, `reports/H3`: paired CSV, summary JSON, Markdown report
  and paired-difference PNG when results exist.
- `scenes/<hash>/`: per-condition metric/trace/audit ZIPs, input manifest, reusable
  SAM mask ZIP, overlays, mask validation, reference checks, timing and failure logs.

The first milestone can report this implementation and preliminary completed
results, including negative or partial findings. It must state which cohort ran,
actual runtime/hardware, remaining mask checks and failures. This does not yet
establish significance, sensitivity across seeds, full-dataset performance or a
benefit for every fine-grained architectural element. Team members should each
own a concrete part; assign names/net IDs in the milestone report rather than
inventing ownership in this repository.

## Local validation and files

`protocol.py` freezes split/policies; `prepare.py` verifies archives/images/masks;
`worker.py` executes conditions with one model; `run.py` coordinates resume/shards;
`report.py` creates the three reports; `build_delivery.py` packages notebooks/source.
`intervention/strength.py` implements opt-in structured coefficients. Existing
reference entry points keep their default behavior without a strength configuration.

```bash
python -X utf8 -m unittest discover -s experiments/point3 -p "test_*.py"
python -X utf8 -m unittest discover -s experiments/point3/shared54 -p "test_*.py"
python experiments/point3/shared54/build_delivery.py
```

CPU tests use synthetic archives/models to verify data identity, split leakage,
selection, caching, reference gates, failure handling and resume. They are not
reconstruction results. The notebook itself performs the required real-GPU gate.
