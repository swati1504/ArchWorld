Selection audit: 27 November 2026

The existing 18 calibration and 36 confirmation landmarks were reused. No new
model runs were performed. Three simple categorical selectors were examined:
anchor, building type and region. Within a category with at least two calibration
landmarks, choose the lowest-mean-error frozen intervention among scalar, H1,
H2 and H3. Otherwise use the globally best calibration intervention (H3).
The minimum support of two follows the existing H3 convention; it was not tuned.

This is an exploratory retrospective audit. The analysis began after inspecting
confirmation outcomes. Calibration-only fitting does not make this a new blinded
validation. Original-VGGT scores are available on confirmation but absent from
these calibration-score files; original therefore serves as a benchmark, not a
learned no-intervention choice in these selectors. H1/H2 settings and H3's lookup
were themselves already fitted on calibration. Fixed-candidate omission checks
are sensitivity diagnostics, not nested cross-validation.

Observed confirmation mean error (lower is better):
  Original VGGT:          13.924255
  H1 everywhere:          14.140620
  Building-type selector: 14.157398
  Anchor selector:        14.239599
  Region selector:        14.527417
  Scalar everywhere:      14.763027

Some category preferences agree between calibration and confirmation, including
facade/H1, statue/H3, column/H3 and window/H1. However, facade calibration has one
H1 scalar win and one loss; the confirmation set has three H1 wins. Column has
only one confirmation case. Window's confirmation advantage over scalar is tiny.
The Misc/H1 and North America/H1 means also agree across the split, but these are
broad, overlapping categories, not identified causal properties. Pillar/H2 has no
pillar calibration cases. Tomb confirmation favors H2, while the single tomb in
calibration favors H1. Arch calibration strongly favors H1, while confirmation
favors original VGGT overall and scalar/H3 among interventions.

All three selectors beat the scalar mean but none beats H1 everywhere or original
VGGT. This motivates testing narrow rules; it does not establish a reliable
method-selection policy or justify choosing confirmation winners after the fact.
No new significance claim is made, and the earlier 46-test correction does not
cover a new, outcome-guided search over additional subgroups.

Files:
  analyze_selection.py: complete reproducible standard-library computation
  calibration_candidates.csv: 18 rows of frozen candidate calibration scores
  group_calibration_confirmation.csv: all categories, both splits and caveats
  policy_confirmation_predictions.csv: all 108 policy/landmark decisions
  selection_audit.json: protocol, aggregate results, limitations, source hashes

Verification: all three policies contain exactly the same 36 unique confirmation
landmarks. Each selected error equals its recorded method error. Independently
weighted group means match direct per-landmark means to 1e-12.


COVERAGE EXTENSION (27 November 2026)
The four highlighted favorable anchor groups cover 17/54 landmarks overall,
including 9/18 calibration and 8/36 confirmation. They leave 28/36 confirmation
cases outside those favorable examples. They were not omitted from full-cohort
policy scores: exact-anchor selection uses category support on 17/36 and the
global-calibration fallback on 19/36; building-type selection uses support on
24/36 and fallback on 12/36; region uses category support on all 36.

One additional fixed exploratory selector uses the three nearest calibration
scenes in the saved image-derived appearance distance. It chooses the lowest
mean calibration error among scalar/H1/H2/H3 on those three neighbors. No k
search, category merging, test-error fitting, or new inference was performed.
The composite distance was not used because it incorporates physical extent;
2D visualization coordinates were not used either.

This rule assigns all 36 confirmation scenes a choice, including rare anchors.
It selects H1 for 28, H2 for 1 and H3 for 7. Mean confirmation error is 14.405449:
2.42% lower than scalar but 1.87% higher than H1 everywhere and 3.46% higher than
original VGGT. It therefore improves assignment coverage without establishing
better selection. Its eight departures from H1 produce one improvement and
seven regressions. There is still no learned no-steering option because the
saved calibration tables used here lack original-model scores.

This is a post-hoc exploratory extension, not a new independent validation.
The appearance descriptors come from dataset selection, and equivalence to the
reconstruction's sampled views or compute budget has not been established.

Additional files:
  coverage_and_similarity.py
  similarity_protocol.json
  similarity_confirmation_predictions.csv
  coverage_all_anchor_groups.csv (all 24 executed labels, including cal-only wall)
  coverage_similarity_audit.json

Verification: all 36 decisions were independently recomputed from their three
listed calibration neighbors; selected scores and aggregate means agree.
