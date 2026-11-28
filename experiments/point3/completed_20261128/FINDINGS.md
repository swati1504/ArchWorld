# Additional findings and interpretation

This note complements the report using existing results only. The full 36-landmark
confirmation cohort remains primary. Proposed failure explanations are hypotheses,
not identified causal mechanisms.

## Anchor groups

| Group | Confirmation n | H1 change vs original | H2 | H3 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Facade | 3 | −16.80% | −1.11% | −4.55% | H1 best among all five methods on all three |
| Pillar | 2 | +1.14% | −2.14% | −0.48% | H2 best on both |
| Statue | 2 | +1.76% | −9.22% | −15.90% | H3 best structured method on both, scalar best on one |
| Window | 2 | −3.01% | −1.80% | −2.82% | Small H1 mean advantage, mixed scalar comparisons |
| Arch | 8 | +1.90% | +8.73% | +1.26% | Original best group mean |
| Bridge | 2 | +2.86% | +7.43% | +1.30% | Original best group mean, individual winners differ |

These are ratios of group means, not anchor-region scores. H3 duplicates scalar
predictions in these groups except statues. Exact recorded labels are retained,
including separate pillar, column, and solomonic-column groups. The other 17
confirmation groups contain one landmark each. No primary association survives
global correction. The facade/H1 and pillar/H2 patterns are exploratory leads.

Calibration and confirmation preferences sometimes disagree. H1 reduces the
scalar mean by 19.18% on six calibration arches but increases it by 0.63% on eight
confirmation arches. This counters a general anchor-to-method rule. The retrospective
anchor selector improves scalar mean but does not beat H1 everywhere or original.

## Large regressions

| Landmark | Saved observation | Possible interpretation |
| --- | --- | --- |
| Heian-jingū | Scalar error 17.01 with ~0.20% relative update, H1 error 24.61 at ~2% | H1 may amplify an unhelpful update |
| Place Jacques-Cartier | Scalar error 60.36 at ~9.24%, H1 error 48.69 at ~2%, original 55.33 | Controlling an aggressive scalar update may help |
| Palacio Legislativo, Uruguay | Scalar eta 100 error 16.95, scalar eta 1000 error 21.53, H2 base eta 100 error 23.07 | Strength and allocation matter, and H2 can hurt at matched base |
| Saint Demetrius Church | Original 22.65, scalar 22.72, H2 27.75, ~4-fold view coefficient spread | Weighting may be unhelpful, but absent scalar-100 scores prevent isolation |
| Itmad-Ud-Daulah's Tomb | H2 15.15 vs original 17.79, mixed wide/close-up views, 10 of 50 railing masks empty | Uneven visibility may favor view adaptation, confounded with base strength |
| Palácio Nereu Ramos | Every intervention worsens, H2 worst despite smaller relative update | Step magnitude alone is insufficient, objective/mask/geometry/alignment sensitivity remains possible |

Relative update is the mean anchor-masked activation norm ratio across layers,
not geometric displacement or a cumulative percentage. Empty masks can reflect
true anchor absence and do not establish segmentation errors. Attention can rise
without matching the correct physical points. Nonlinear propagation and alignment
sensitivity are plausible but have not been isolated in these saved runs.

## Outcome-based deletion sensitivity

The three worst regressions account for 74.24%, 74.32%, and 86.44% of gross harm
for H1/H2/H3 respectively. Gross harm sums positive changes from original, not total
error. Worst cases were identified after observing the results.

| Retained cohort | n | H1 change vs original | H2 | H3 |
| --- | ---: | ---: | ---: | ---: |
| Full official cohort | 36 | +1.55% | +6.05% | +3.47% |
| Omit Palácio Nereu Ramos | 35 | −1.42% | +2.30% | +0.32% |
| Omit union of each method's worst three | 30 | −4.91% | −1.02% | −2.48% |

The common union is the six landmarks in the preceding table. Every row compares
all methods on exactly the same retained IDs and recomputes the original mean on
those IDs. This demonstrates sensitivity to severe regressions, not superiority
after cleaning invalid data. The omitted cases have not been shown to be invalid.
Choosing omissions from errors favors the chosen methods by construction.
Method-specific worst-three omissions use different cohorts and are not a fair
ranking. No filtered significance tests or revised headline scores are claimed.

Full calculations are in [outlier_sensitivity.csv](findings/outlier_sensitivity.csv).
The original report already includes single-landmark omission analyses. This note
also records the later multi-landmark sensitivity without silently changing the PDF.

## Hindsight choices are diagnostic only

Selecting the best recorded intervention per scene yields mean error 13.3201.
Allowing original VGGT as a fifth choice yields 12.8996. Restricting hindsight
choice to H1 or original yields 13.1615. These require ground-truth outcomes and
are not achieved selectors. The four tested retrospective group/appearance rules
do not beat H1 everywhere or original VGGT.
