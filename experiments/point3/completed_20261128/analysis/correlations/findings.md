# Scene and intervention-strength correlations

No continuous-feature association survived correction for the 28 primary tests. The smallest Benjamini–Hochberg adjusted p value was **0.188**. An additional 28 log-ratio sensitivity tests did not change this result: none survived either their own correction or a joint correction over all 56 tests. The data contain possible leads, but no validated rule for predicting where steering will improve reconstruction.

## Scope and definitions

The analysis uses the **36 confirmation landmarks only**, with the frozen H1 `layer_alpha0.02`, H2 `view_eta100`, and anchor-selected H3 condition. Calibration scenes were excluded. All scene-condition diagnostics were unique, all 36 landmark joins succeeded for each hypothesis, and physical extent agreed between the two metadata sources. No held-out outcomes or new model runs were used.

The primary outcome is `hypothesis error − comparator error`, calculated once per landmark. Negative values indicate improvement. Therefore, **negative Spearman correlation** means that larger feature values tend to accompany better paired effects, while **positive correlation** means larger feature values tend to accompany worse effects. Both the original VGGT and calibrated scalar comparator were analyzed. The reported error is whole-landmark normalized Chamfer error.

## Strongest exploratory associations

| Hypothesis and comparator | Feature | Spearman rho | Raw permutation p | BH q across 28 | Pointwise bootstrap interval |
|---|---|---:|---:|---:|---|
| H3 versus original | Mean applied relative update | +0.445 | 0.00672 | **0.188** | [0.114, 0.671] |
| H2 versus scalar | Maximum/minimum coefficient ratio | −0.411 | 0.01412 | **0.198** | [−0.645, −0.126] |
| H2 versus scalar | Maximum coefficient | −0.364 | 0.03030 | **0.283** | [−0.619, −0.037] |
| H2 versus scalar | Mean applied relative update | −0.342 | 0.04140 | **0.290** | [−0.604, −0.028] |

These intervals are **unadjusted, pointwise** intervals. Their exclusion of zero does not override the multiple-testing result. The confirmation subset was deliberately selected rather than randomly sampled from the full dataset; the resampling intervals describe stability within this selected collection, not a guarantee of population coverage.

The strongest H3 association stays positive in every leave-one-landmark-out calculation, with rho ranging from **0.399 to 0.491**. The strongest H2 association stays negative, ranging from **−0.472 to −0.378**. Thus these two patterns are not produced by a single landmark, but they still do not pass the declared multiplicity threshold.

For H2's coefficient spread, **21 of 36** landmarks attain the exact ratio 4, and the other 15 have smaller ratios. Average ranks handle those ties. The log-ratio outcome gives rho **−0.425**, with sensitivity-family q **0.181** and joint-56 q **0.198**. This supports describing a repeatable exploratory pattern within these data, not a confirmed selection criterion.

For H3, **33 of 36 predictions are identical to the scalar control**. An association between H3 update size and error relative to original VGGT therefore mostly describes scalar steering behavior; it is not specific evidence that anchor-dependent selection works. H3-versus-scalar change is nonzero at only three landmarks. About 4.0–4.6% of bootstrap samples omit all three and have undefined correlation. Those rows retain 3,823, 3,839, or 3,817 of 4,000 resamples, depending on the feature. Their stored intervals are conditional on correlation being defined and should not be interpreted as unconditional 95% confidence intervals.

## Landmark size

Physical extent did not show a corrected association with improvements. For H1, larger landmarks had a weak tendency toward better differences versus original VGGT (rho **−0.284**, q **0.341**) and versus scalar (rho **−0.241**, q **0.434**). H2's extent correlation was **+0.063** versus original and **+0.252** versus scalar; H3's was **−0.028** and **+0.160**, respectively. None supplies a defensible landmark-size rule.

## Important exclusions and interpretation limits

- All selected conditions accepted **24 layers**, and all reported **zero clipped view steps**. Those constant diagnostics cannot explain variation between landmarks.
- H1's mean applied relative update is effectively fixed at **0.02**. Its approximately 1e−9 variation is numerical noise and was excluded. Large H1 coefficients should not be interpreted as larger relative steps: the policy explicitly normalizes the relative update.
- Actual view count is 50 for 35 landmarks and 26 for one. A correlation with this variable would primarily describe that one landmark, so it was not tested as a continuous predictor.
- The assigned metadata does not contain usable total-image counts, duplicate fractions, or image-resolution measurements. Missing features were not replaced by actual run view count. Reconstruction-coordinate scale/diagonal and numeric category IDs were not added as extra tests.
- Baseline error and improvement share a mathematical term. Baseline-error correlations are provided separately as descriptive diagnostics without inferential p values. They cannot establish that difficult landmarks benefit more. Baseline-adjusted rank correlations are also descriptive sensitivities, not causal adjustments.
- Diagnostics are observed during the selected intervention. They are not independently randomized predictors and are not necessarily available before an intervention is run. Anchor category, architecture, region, gradient behavior, and scene difficulty may confound these associations.
- H2's frozen base eta is 100, while the scalar comparator uses 1000. Associations against that scalar control cannot isolate the contribution of view weighting from a change in effective steering strength. Confirmation data do not include a complete matched `scalar_eta100` comparison for this purpose.

## Method and reproducibility

The protocol was saved before computing correlations. It lists five H1, six H2, and three H3 features, each compared with two outcomes: **28 primary tests**. Rank correlations use average ranks for ties. Two-sided permutation tests shuffle outcome ranks 49,999 times, using `(exceedances + 1) / 50,000`. All resampling uses fixed seed 20261126. Paired bootstrap calculations use 4,000 landmark resamples, reranking ties within each sample. Leave-one-landmark-out correlations identify influence without deleting cases from the primary analysis.

The sensitivity outcome is `log(hypothesis error / comparator error)` for the same 28 tests. `primary_correlations.csv` contains the primary p values, BH correction across those 28 tests, the joint-56 correction, pointwise resampling intervals, and leave-one-out ranges. `log_ratio_sensitivity.csv` contains the corresponding sensitivity results. `joined_confirmation_features.csv`, `excluded_features.csv`, `protocol.json`, and `results.json` preserve the data join, exclusions, methods, and source SHA-256 hashes.

Run `analyze_correlations.py --source <evidence-directory> --output <analysis-directory>` using Python with NumPy. Original evidence is read-only. The broader report may also pool these 28 p values with other exploratory analysis families; no correlation here passes its own declared correction.

The useful next step is to explain these tentative patterns and preserve them as ideas for future independent evaluation. These results do not justify changing the frozen model-selection rule or claiming a successful predictor on the same 36 landmarks.
