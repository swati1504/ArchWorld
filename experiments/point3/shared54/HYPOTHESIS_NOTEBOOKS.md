# Mask preparation plus three hypothesis notebooks

Use these when the team wants one executable notebook per hypothesis, each on its
own GPU runtime. All three notebooks contain setup, a memory/equivalence pilot,
calibration, confirmation, results and diagnostics. They are execution notebooks;
the earlier `Point3_H1_Report.ipynb` / H2 / H3 files only read a shared run.

Run `Point3_00_Prepare_Masks_UpTo50.ipynb` once before these experiments. It imports
compatible old masks, generates missing masks, and validates the **same 54 landmarks,
with all eligible distinct images up to 50 each**. It performs no VGGT reconstruction or intervention tuning.
All three experiment notebooks read its completed cache.

**Washington Monument correction (authorized 2026-11-15):** its supplied `altar`
anchor yielded 50 empty masks. The effective anchor is now the documented manual
`obelisk`; the original label stays in the provenance. The original 18/36 split
and all 54 landmark IDs remain fixed. Washington stays in confirmation and uses
H3's declared global-scalar fallback. Review all 50 new overlays. The other 53
completed preparations are verified and reused without regenerating masks or
downloading their inputs again. New source/output folders preserve the failed run.

This view-count policy was explicitly authorized on 2026-11-15 after real archives
showed fewer than 50 eligible images for some landmarks. No landmarks are replaced
or padded. Existing completed masks remain reusable. Stop the earlier preparation
cell before running this corrected notebook; its logs and saved outputs are preserved.

| Notebook | Experiment | Calibration jobs | Confirmation jobs | Total upper bound |
| --- | --- | ---: | ---: | ---: |
| `Point3_H1_Run_54.ipynb` | Layer-dependent coefficient | 90 | 108 | 198 |
| `Point3_H2_Run_54.ipynb` | View-dependent coefficient | 90 | 108 | 198 |
| `Point3_H3_Run_54.ipynb` | Anchor-to-coefficient lookup | 54 | At most 108 | At most 162 |

Reference checks/replays add work. H3 deduplicates identical global and anchor
choices, so its actual confirmation count can be lower. There are up to **558
jobs in total** across the notebooks, versus at most 306 for the
single shared design. Reconstruction controls are repeated; SAM masks are shared.
The separate preparation notebook imports or generates masks. All three
experiments check readiness before installing VGGT; missing masks stop immediately.
On three GPUs, elapsed time is governed by
the slowest notebook, with additional setup, download and validation overhead;
three notebooks do not imply a measured 3× speedup. H1 and H2 have passed the
50-view reference/memory gate on L4 and produced calibration checkpoints. The
complete confirmation cohorts and H3 experiment remain unfinished. Scene timings
vary substantially, so pilot timings are not full-study completion estimates.

## Build from this repository

Run the builder from the repository root with Python 3.12 or newer:

```bash
python experiments/point3/shared54/build_hypothesis_delivery.py
```

It writes the four execution notebooks, a checksum-named source ZIP, and
`hypothesis_delivery.json` into `experiments/point3/shared54/dist/`. Copy those
four notebooks and the exact ZIP named in the manifest to the Drive folder below.
The builder uses the Python standard library; model dependencies are installed by
the Colab setup cells. Source ZIPs, private Drive-link inventories, runtime logs,
model weights, masks, and reconstruction outputs are not committed to Git.

The checked-in notebooks are clean, unexecuted copies. Rebuild after changing
source files or documentation included in the bundle. A new bundle has a new
identity and output directory: retain the original notebook, source ZIP, run
label, and matching GPU/software when resuming an existing experiment. Publishing
or rebuilding this repository does not update notebooks already running in Colab.

## Run

1. Keep the four notebooks and the matching source ZIP in
   `MyDrive/Point3_Hypothesis54`. They load the same verified source revision.
2. Run `Point3_00_Prepare_Masks_UpTo50.ipynb` first. Add `HF_TOKEN` in its Colab Secrets,
   enable access, and use a token with approved SAM 3 access. Wait for
   **READY: 54 landmarks / N masks**, inspect the saved overlays, and review `view_counts.csv`.
3. Open one experiment notebook per independently allocated GPU runtime. Use matching GPU
   models/software if you plan to combine results. Each GPU must individually
   pass the 50-image memory check; memory is not pooled across sessions.
   H1, H2 and H3 can use `HF_TOKEN` for public VGGT downloads but require no SAM
   access or installation.
4. Run H1/H2/H3 concurrently or in any order. Run cells in order in each
   notebook. Each pilot checks its own runtime and cached-model equivalence.
   None of these notebooks generates masks.
5. Use the same run label and settings to resume that hypothesis. Results are
   stored separately under `results/hypotheses54_<source hash>_cal18_v1/H1`, `H2`,
   and `H3`; local scratch directories are separate as well.
6. Each notebook shows its own report. When all three finish, rerun the final
   comparison cell in **H3** to verify comparability and create a combined table.

The default session cap is eight hours, not a completion-time estimate. With one
GPU, finish mask preparation first, then run each hypothesis. Concurrent sessions require independently
allocated resources under the provider's usage rules; opening extra browser tabs
does not allocate a GPU. These notebooks do not provision or purchase compute.

After the session cap expires, wait until the previous execution has stopped.
On a still-connected runtime, rerun the first settings cell to renew `DEADLINE`,
then rerun **Run or resume H1/H2/H3**. Completed validated checkpoints are reused.
After a runtime reset, rerun setup and the pilot before the resume cell. Keep the
same source bundle and run label; never run two copies of one hypothesis into
the same output directory. The mask-preparation notebook is needed again only
if its readiness checks report missing or incompatible masks.

G4/Blackwell detection selects PyTorch 2.7.0 with CUDA 12.8 in separate environment
directories, avoiding the older CUDA wheels used by the L4/A100/T4 profile.
The GPU setup performs an actual CUDA operation and saves the selected package
profile. The real 50-view reference/memory gate is still required; compatibility
and runtime have not been measured on G4. Do not mix GPU/software profiles within
a study or the combined comparison. See [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/).

## Shared experimental design

Each hypothesis uses the same frozen 54-landmark IDs, up to 50 eligible distinct
photographs per landmark, ordered image identities, seed 1729 and metadata-only
18-calibration/36-confirmation split. The final 79 held-out IDs remain untouched.
Settings are selected on calibration only and frozen before that notebook's
confirmation. All notebooks include no-intervention and scalar controls.

Actual view counts are recorded in all result tables and checkpoints. Every
condition/hypothesis uses identical selected images per landmark. Calibration
selection and aggregate differences weight landmarks equally, not by their number
of views. A smaller scene is not a 50-image result. The pilot prefers a calibration
landmark with the largest selected view count, without reducing images to fit memory.

Decisions and fixed parameters:

- **H1:** select alpha from `{0.01, 0.02}` for the per-layer relative activation/
  gradient norm rule. Coefficient cap is `100000`, denominator epsilon `1e-8`.
  Separately select the global scalar from eta `{100, 1000}`.
- **H2:** select eta from `{100, 1000}` separately for the view rule and global
  scalar. View visibility exponent `0.5`, clipping `[0.5, 2]` and RMS normalization
  stay fixed. View weights are determined at the first hook and held across layers.
- **H3:** select eta from `{100, 1000}` per anchor type with at least two calibration
  landmarks; otherwise use the calibration-selected global scalar. Supported
  types are arch, column, facade, statue, tower and window. Dome/minaret use the
  fallback, and the cohort contains no spire.

Keep the original VGGT attention objective, loss scale `1e5`, layers 0–23,
all-token update support, accept/revert rule, paired evaluation and metric formula.
Both fresh-reference and accelerated executions require deterministic PyTorch
algorithms (`warn_only=False`) inside the attention intervention hooks,
deterministic cuDNN without benchmarking, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before CUDA initialization. Outside the hooks, the original inference kernels
remain in use: global deterministic mode in PyTorch 2.3.1 switches no-grad
bilinear resizing to a decomposition that exhausted L4 memory. These settings are
recorded in every result's hardware metadata. Seeding alone did not make the
intervention gradients repeatable on L4/PyTorch 2.3.1: the baseline predictions
matched exactly, but the intervened predictions diverged and correctly stopped
the pilot. The exact prediction-hash and metric-equivalence gate remains required;
unsupported deterministic operations stop the run rather than silently falling back.
The fresh reference runs in a separate process that exits before the persistent
model starts, releasing its complete CUDA context. The zero control and scalar
check still run consecutively on the same retained model, so the gate continues
to test model reuse as well as cached inputs and evaluation.
Each notebook retains model reuse and caches within each landmark, plus resumable
reconstruction checkpoints. Controls are computed separately; SAM outputs are reused
from `MyDrive/Point3_Hypothesis54/common_masks` across all three notebooks and across
source-bundle versions. Mask preparation is the only writer; all hypotheses are readers.
Do not run duplicate preparation sessions against the same cache. The local lock
is a guard, not a distributed lock across Drive mounts. Completed sets are published only after
all selected masks pass validation. Preparation saves partial sets every ten generated images,
so a resumed run generates only the remaining images.

## Reusing masks already generated

The preparation notebook automatically checks earlier mask-preparation and hypothesis result directories,
`MyDrive/Point3_Shared54/results`, `MyDrive/AdaptiveAttend_Point3`, and the old local
`/content/adaptiveattend_point3/results` and `masks` locations. Additional known
locations can be added to `MASK_SEARCH_ROOTS` in the settings cell.

A documented archive is imported by **RGB content hash, anchor and recorded SAM
revision/confidence**, with checksum, binary-array and image-dimension checks. This
works even if filenames, view order, code-bundle identity or the selected image set
changed: matching individual images can be reused and only missing masks generated.
Cached empty masks remain valid records of SAM finding no anchor. At least two views
must have nonempty masks before a completed scene is usable.

Older archives lacking image hashes or SAM provenance, and loose `.npy` files, are
reported in `mask_discovery.json` for review. They are not silently matched by
filename. Whole-landmark validity masks are not SAM architectural-anchor masks.
The earlier five-view Akdamar Church pilot is outside the current 54. The first
shared-cache run has also saved completed scenes, which this version reuses by
content identity. Full coverage is asserted only after all 54 landmarks validate;
the total image count is computed from actual selected views, not assumed to be 2,700.

`mask_reuse.json` in each scene reports imported, reused and newly generated counts
for its latest attempt, source provenance and rejected inputs. Each notebook displays totals. Existing
source archives are read without modification. The shared cache identity excludes
notebook, intervention strengths and source-bundle hashes, preventing needless
regeneration when those change. Each hypothesis keeps its own validated archive
copy for reproducible result comparison.

Forth Bridge uses the user-authorized manual **bridge** anchor, with provenance
preserved. Check all selected-view mask overlays before interpreting its effect. Numeric
validation and cross-notebook agreement do not establish semantic mask correctness.

## Comparison and outputs

Within each notebook, report paired `structured − global scalar` normalized Chamfer
distance on the 36 confirmation landmarks, plus the no-intervention comparison.
Negative differences favor the structured rule. Include mean/median differences,
wins/ties/losses, failures, mask review, effective strengths and accepted layers.
Each notebook produces `results.csv`, `failures.csv`, `strength_diagnostics.csv`,
per-scene metric/trace/audit checkpoints, and `reports/H1`, `H2` or `H3`.
Per-condition, paired and combined result tables include `num_views`.

The optional combined comparison revalidates all checkpoints and frozen selections,
then verifies identical source/cohort, image/mask identities, GPU/software profiles,
global scalar choice, and shared control predictions/metrics. It refuses a mismatch.
It creates `comparison/paired_hypotheses.csv` and `comparison/summary.json` only
after all three experiments pass those checks. The comparison result is preliminary
internal confirmation, not a 396-landmark claim or a significance test.

Full method and dataset details remain in [README.md](README.md). This delivery
extends `run.py` with `--hypothesis H1`, `H2` or `H3`; omitting it preserves the
shared all-hypothesis runner. Build the four notebooks with:

```bash
python experiments/point3/shared54/build_hypothesis_delivery.py
```

The previous shared-run notebook and report notebooks remain available as a
separate option. Do not mix their checkpoints with this separately versioned study.
