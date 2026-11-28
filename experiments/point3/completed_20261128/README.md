# Completed Point 3 contribution

H1, H2, and H3 have completed the same 54-landmark study, using 18 calibration
and 36 internal confirmation landmarks. This package supersedes the November 16
H1/H2 milestone snapshot. It publishes Anastasios Gerasis's completed experiments,
later CPU analyses, verified reconstruction examples, and the shared final-report
template. It adds no model runs and changes no model or evaluation implementation.

The report template deliberately leaves Eric's and Swati's sections, the abstract,
and the combined discussion/conclusion empty for team integration. It is not a claim
that the combined team submission or full-benchmark evaluation is finished.

## Read the results

- [Final-report template PDF](report/main.pdf) and [editable LaTeX](report/main.tex)
- [Extended effects, anchor groups, and correlations](analysis/Analysis.md)
- [Additional findings and failure-case interpretation](FINDINGS.md)
- [Per-landmark errors for all three hypotheses](report/evidence/paired_hypotheses.csv)
- [Calibration-only group and appearance selectors](selection_audit/READ_ME.txt)
- [Verified reconstruction comparisons](qualitative/Comparisons.md)
- [How the original 40 became the final 54](selection/README.md)

## Main results

Errors are whole-landmark normalized Chamfer distances. Lower is better. Changes
below are ratios of cohort means, not averages of per-landmark percentage changes.

| Method | Mean error | Change vs original | Change vs scalar | Wins/ties/losses vs scalar |
| --- | ---: | ---: | ---: | ---: |
| Original VGGT | **13.92426** | — | — | — |
| Scalar, eta 1000 | 14.76303 | +6.02% | — | — |
| H1, layer alpha 0.02 | 14.14062 | +1.55% | **−4.22%** | 23/0/13 |
| H2, view base eta 100 | 14.76697 | +6.05% | +0.03% | 13/0/23 |
| H3, anchor lookup | 14.40769 | +3.47% | **−2.41%** | 2/33/1 |

H1 and H3 improve the calibrated scalar mean, but none improves the original-model
mean. All six pointwise paired-bootstrap mean-effect intervals include zero. No
primary association survives global correction across the 46 group/feature tests.
These findings do not establish equivalence or population-wide superiority.

The three facade landmarks favor H1 and the two pillar landmarks favor H2. H3's
entire scalar-relative change comes from two statues and one column. These are
small, exploratory groups. Seventeen confirmation anchor groups contain only one
landmark. Scores do not separately measure geometry inside the anchor masks.

All 24 layers accepted updates in every selected run, including the 15 H1, 16 H2,
and 14 H3 landmarks that worsened relative to original VGGT. Thus the monitored
attention increase does not guarantee improved reconstruction in this cohort.

## Reproduce without a GPU

From the repository root, use Python 3.10 or newer in a virtual environment.

```bash
python -m pip install -r experiments/point3/completed_20261128/requirements.txt
python experiments/point3/completed_20261128/verify_package.py
python experiments/point3/completed_20261128/verify_package.py --full
```

The first command after installation validates immutable SHA-256 fingerprints,
original-40/54 selection identity, the saved figure verification records, and PDF
text. The `--full` command also reproduces the numerical reports, 100,000 paired
bootstrap samples, 49,999-permutation association analyses, group and appearance
selection audits, and deletion sensitivities in a temporary copy. It compares all
regenerated analysis CSV values with the committed tables. It requires no network,
Google Drive, Colab, images, model weights, or new inference. Allow several minutes.
The committed evidence and report sources are not overwritten by verification.

Individual analysis entry points remain available under `analysis/`, `findings/`,
and `selection_audit/`. Running them directly writes derived outputs beside their
scripts. The package verifier is preferred for a read-only review.

To compile the paper or produce an Overleaf ZIP, install a LaTeX distribution with
`acmart`, then run

```bash
python experiments/point3/completed_20261128/build_report.py
```

The current PDF contains no visible colons or semicolons, following the author's
formatting preference. LaTeX keys, URLs, and source-code syntax retain required
punctuation. Importing the generated ZIP creates an Overleaf project, not a shared
editing link. Sharing remains a separate action for the project owner.

## Scope and provenance

- All 54 selected landmarks come from the 317-ID development partition. The same
  frozen inputs, masks, controls, source identity, and L4 environment were checked
  across H1/H2/H3. There are 2,622 study views, including 1,776 confirmation views.
- The 79-ID evaluation partition is unused by Point 3. Fifteen IDs overlap earlier
  team development/test lists, so it is not established as untouched team-wide.
  See [the recorded overlap audit](report/evidence/team_split_overlap.json).
- No new GPU experiments are included. H2 and the global scalar have different
  calibrated base coefficients. The three available matched-base confirmation
  comparisons are nonrepresentative counterexamples, not a full H2 ablation.
- Oracle choices and deleting the worst regressions use observed outcomes. They
  are explicitly labeled hindsight/sensitivity diagnostics, never revised headline
  results or achieved inference-time selectors. All 36 scenes remain primary.
- Complete semantic mask review, multi-seed evaluation, and full-396 evaluation
  remain outside the completed scope. Numeric validation does not prove masks are
  semantically correct or identify the causes of regressions.
- Report figure PDFs and GitHub-preview PNGs are included with their recorded
  hashes and export checks. Raw photos, masks, point clouds, feature archives, and
  model weights are not redistributed. Rendering from geometry requires the
  separately retained exports. Existing experiment runners remain in `../shared54/`.
- [Original report-delivery validation](provenance/delivered_report_validation.json)
  describes the earlier Overleaf ZIP. The package's current immutable files are
  covered separately by [immutable_sha256.json](provenance/immutable_sha256.json).

## Ownership

Anastasios Gerasis owns the Point 3 selection audit, structured-strength rules,
mask workflow, matched experiments, and analyses in this package. The pretrained
models, benchmark, reference objective, and original evaluation conventions are
inherited from ArchWorld/ArchAttend. The historical milestone remains available
at [completed_20261116](../completed_20261116/README.md).
