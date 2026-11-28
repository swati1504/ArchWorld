# Anchor and categorical scene analysis

The 36 confirmation landmarks contain useful local patterns, but most anchor groups are too small to establish general anchor-dependent performance. All calculations below use frozen H1/H2/H3 settings and one equally weighted observation per landmark.

## Main findings

- H1 improves all three facade-anchor landmarks against both controls: Herald Square, Jahangiri Mahal (Agra Fort), and Patan Durbar Square. The error reduction calculated from group means is **16.80% versus unmodified VGGT** and **12.83% versus scalar**. The sign survives removing any one of the three cases, but n=3 remains a descriptive pattern.
- The largest repeated anchor group, arch (n=8), does not improve on average against either control under H1 or H2. H2 is 8.73% worse than unmodified VGGT and 7.37% worse than scalar, with only 2/8 and 1/8 wins, respectively.
- H3 has **19 unsupported-anchor fallback cases, 14 supported-anchor cases selecting the same scalar, and only 3 cases selecting a different strength**. The 33 equal-to-scalar cases must not be counted as independent evidence that anchor-dependent selection helped.
- A regional H1-versus-scalar association survives correction within the 18 categorical tests (permutation p=0.00206; BH q=0.03708; Bonferroni p=0.03708). The corresponding H1-versus-original regional test is not persuasive (p=0.41068). This is comparator-specific exploratory evidence, not a claim of geographic superiority or causality. Broader correction across all parallel analyses is reported by the combined report.

## Group sizes and limits

There are **23 executed anchor categories**, of which **17 contain one landmark**. Only arch has at least four observations. The repeated categories are arch (8), facade (3), bridge (2), pillar (2), statue (2), and window (2). Region has eight categories with four or five landmarks each. Building type has eight categories with two to nine landmarks.

The executed frozen-plan anchor matches the paired-result table for all 36 landmarks. Two original supplied labels differ from that executed anchor: Forth Bridge has a missing supplied label and uses bridge; Washington Monument has supplied altar and uses obelisk. These are already recorded in the study plan, not changes made by this analysis. Both values are retained in the joined data.

Anchor labels identify the semantic class used for steering. The scores measure whole-landmark reconstruction, not error restricted to that semantic region. These tabular results do not establish segmentation-mask quality or replace a manual audit of all input masks.

## Repeated anchor groups

Percent changes below are `100 × (mean(method) / mean(control) − 1)`. Negative values favor the method. W/T/L counts use paired landmark differences. Singleton groups remain in the complete CSV but are omitted from this compact table.

| Anchor | n | Method | Change vs original | W/T/L | Change vs scalar | W/T/L |
|---|---:|---|---:|---|---:|---|
| arch | 8 | H1 | +1.90% | 3/0/5 | +0.63% | 3/0/5 |
| arch | 8 | H2 | +8.73% | 2/0/6 | +7.37% | 1/0/7 |
| arch | 8 | H3 | +1.26% | 4/0/4 | +0.00% | 0/8/0 |
| bridge | 2 | H1 | +2.86% | 1/0/1 | +1.54% | 1/0/1 |
| bridge | 2 | H2 | +7.43% | 1/0/1 | +6.05% | 0/0/2 |
| bridge | 2 | H3 | +1.30% | 1/0/1 | +0.00% | 0/2/0 |
| facade | 3 | H1 | -16.80% | 3/0/0 | -12.83% | 3/0/0 |
| facade | 3 | H2 | -1.11% | 3/0/0 | +3.60% | 0/0/3 |
| facade | 3 | H3 | -4.55% | 3/0/0 | +0.00% | 0/3/0 |
| pillar | 2 | H1 | +1.14% | 1/0/1 | +1.63% | 1/0/1 |
| pillar | 2 | H2 | -2.14% | 2/0/0 | -1.66% | 2/0/0 |
| pillar | 2 | H3 | -0.48% | 1/0/1 | +0.00% | 0/2/0 |
| statue | 2 | H1 | +1.76% | 1/0/1 | -3.03% | 1/0/1 |
| statue | 2 | H2 | -9.22% | 2/0/0 | -13.49% | 1/0/1 |
| statue | 2 | H3 | -15.90% | 2/0/0 | -19.86% | 1/0/1 |
| window | 2 | H1 | -3.01% | 2/0/0 | -0.20% | 1/0/1 |
| window | 2 | H2 | -1.80% | 2/0/0 | +1.05% | 0/0/2 |
| window | 2 | H3 | -2.82% | 2/0/0 | +0.00% | 0/2/0 |

H2 illustrates why the comparator matters: all three facade cases improve slightly over original VGGT, yet all three are worse than the scalar control. For the two statue cases, H3 beats original VGGT twice, but beats scalar only once. Its mean improvement over scalar is therefore not a universal statue result.

## H1 regional pattern

| Region | n | Change vs original | Wins vs original | Change vs scalar | Wins vs scalar |
|---|---:|---:|---:|---:|---:|
| Africa | 4 | +1.67% | 1/4 | -0.14% | 3/4 |
| East Asia | 4 | +15.22% | 3/4 | +33.39% | 2/4 |
| Europe | 5 | +2.96% | 2/5 | +1.35% | 3/5 |
| Latin America | 5 | +13.41% | 3/5 | -3.04% | 3/5 |
| North America | 4 | -8.16% | 3/4 | -13.49% | 4/4 |
| SE Asia | 4 | -8.66% | 4/4 | -7.53% | 1/4 |
| South Asia | 5 | -1.75% | 3/5 | -9.97% | 5/5 |
| West/Central Asia | 5 | +0.28% | 2/5 | +0.46% | 2/5 |

North America and South Asia have 4/4 and 5/5 wins against scalar, respectively, and their mean direction remains favorable after any one case is removed. They have only 3/4 and 3/5 wins against original VGGT. East Asia has two scalar wins and two losses; Heian-jingu supplies 87.1% of that region's total absolute H1-minus-scalar difference. SE Asia has only 1/4 scalar wins despite its favorable mean: Prambanan supplies 95.5% of the total absolute difference, and removing it changes the group mean from favorable to unfavorable.

For the H1-versus-scalar regional omnibus association, leave-one-landmark-out sensitivity gives unadjusted permutation p values from 0.00030 to 0.00595. In 14/36 deletions, the result no longer clears the original 18-test Bonferroni threshold. These are follow-up influence diagnostics, not 36 independent discoveries. Each uses 99,999 permutations and a fixed seed; equal group-size configurations share the same cached permutation null.

Region, architecture, anchor, original difficulty, and intervention response can overlap in this small sample. The omnibus result identifies separation of outcome ranks among the observed regions; it does not isolate geography as the cause or validate a region-based steering policy.

## Building types

No building-type omnibus comparison survives the 18-test correction. H1 beats scalar on all seven Misc landmarks with a 12.62% lower group-mean error, but Misc is a broad residual category and should not be converted into a semantic selection rule. Temple/Shrine has four H1 scalar wins out of five, yet a slightly worse mean (+0.10%): the Heian-jingu regression offsets the wins. The full table includes every building type and leave-one-out mean range.

## H3 mechanism

| Executed rule category | n | Relation to scalar |
|---|---:|---|
| Unsupported anchor fallback | 19 | Exactly equal to scalar |
| Supported anchor, same selected strength | 14 | Exactly equal to scalar |
| Supported anchor, different selected strength | 3 | Two scalar wins, one scalar loss |

The three changed cases use eta=100 instead of global eta=1000:

| Landmark | Anchor | H3 minus original | H3 minus scalar |
|---|---|---:|---:|
| Mansudae Grand Monument | statue | -2.67575 | +0.20137 |
| Marine Corps War Memorial | statue | -3.59097 | -8.41346 |
| Palacio Legislativo, Uruguay | column | -0.14015 | -4.57996 |

Across these three cases, mean error is 11.34% lower than original and 20.34% lower than scalar. This is the subgroup defined by the executed rule, not a new independently selected test set. Its sample is too small for a general column/statue claim. The full H3 mean remains worse than original: the 19 fallback cases inherit scalar's 7.76% worse group mean, while the 14 supported-but-unchanged cases are 0.40% better than original. The result is consistent with limited incremental coverage of the anchor-specific policy.

## Omnibus tests and method

The analysis family was defined as three fixed categorical fields × three frozen methods × two controls = 18 exploratory tests. For each comparison, compute log(method/control), assign average ranks to exact ties, and calculate the fraction of centered rank sum-of-squares explained by the category. This is a rank-based between-group separation statistic, not a predictive R-squared. Shuffle the 36 landmark outcome vectors jointly 49,999 times (seed 2026112602), preserving dependence between methods. The permutation p value is `(1 + number of permuted statistics at least as large as observed) / 50000`. Apply Benjamini-Hochberg and Bonferroni across all 18 p values. No data-driven anchor merges, outcome cutoffs, or singleton removal are used in the tests.

| Grouping | Comparison | Permutation p | BH q (18 tests) |
|---|---|---:|---:|
| anchor | H1 vs baseline | 0.11320 | 0.28665 |
| anchor | H1 vs scalar | 0.12740 | 0.28665 |
| anchor | H2 vs baseline | 0.01246 | 0.11214 |
| anchor | H2 vs scalar | 0.30752 | 0.42901 |
| anchor | H3 vs baseline | 0.04974 | 0.22349 |
| anchor | H3 vs scalar | 0.90884 | 0.90884 |
| region | H1 vs baseline | 0.41068 | 0.52796 |
| region | H1 vs scalar | 0.00206 | 0.03708 |
| region | H2 vs baseline | 0.17464 | 0.31435 |
| region | H2 vs scalar | 0.46930 | 0.52796 |
| region | H3 vs baseline | 0.05462 | 0.22349 |
| region | H3 vs scalar | 0.30984 | 0.42901 |
| building_type | H1 vs baseline | 0.06208 | 0.22349 |
| building_type | H1 vs scalar | 0.09188 | 0.27564 |
| building_type | H2 vs baseline | 0.17242 | 0.31435 |
| building_type | H2 vs scalar | 0.46606 | 0.52796 |
| building_type | H3 vs baseline | 0.21152 | 0.34612 |
| building_type | H3 vs scalar | 0.81056 | 0.85824 |

Permutation inference assumes exchangeable landmark outcome ranks under the null. The confirmation subset was deliberately selected, not randomly sampled from the entire dataset; category sizes are small and some categories are highly sparse. The p values therefore characterize exploratory associations in this analyzed subset and do not establish population-wide effects. Within-family correction does not account for additional analyses outside these 18 tests; use the combined report's broader multiplicity sensitivity when drawing overall conclusions.

## Reproduction and audit

`python analyze_groups.py --source <evidence_directory> --output <result_directory>` regenerates the numeric outputs. `--prior-anchor-summary <anchor_comparisons.csv>` optionally checks the previous anchor table. Required inputs are paired_hypotheses.csv, h1_run_study.json, h2_run_study.json, h3_study.json and h3_selection.json. Outputs include data_audit.json (source checksums and exceptions), group_effects.csv, categorical_omnibus_tests.csv, h1_region_leave_one_out.csv, h3_rule_groups.csv and h3_changed_landmarks.csv.

Checks passed: 36 unique confirmation rows; all three study plans agree on split/category labels; 18 calibration rows are excluded; H3 rule choices and fallback flags match their frozen selection file; equality-to-scalar cases are exact; and the existing anchor-summary means/counts were reproduced to numerical precision. The independent unit is the landmark, so the 1,776 input views are not treated as additional observations. No model reruns, setting changes or final-holdout outcomes were used.
