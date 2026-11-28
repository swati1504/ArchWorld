# Paired effects and robustness

The analysis uses the same 36 confirmation landmarks for each method. A negative change means lower error and favors the hypothesis. All numbers below are whole-landmark CD_norm results.

## Primary paired results

| Method | Control | Mean change | 95% bootstrap interval | Change of means | Wins / ties / losses | Sign p (Holm) |
|---|---|---:|---:|---:|---:|---:|
| H1 | baseline | +0.2164 | [-0.7849, +1.4040] | +1.55% | 21 / 0 / 15 | 1.0000 |
| H1 | scalar | -0.6224 | [-1.5247, +0.1813] | -4.22% | 23 / 0 / 13 | 0.7950 |
| H2 | baseline | +0.8427 | [-0.1018, +2.2123] | +6.05% | 20 / 0 / 16 | 1.0000 |
| H2 | scalar | +0.0039 | [-0.7783, +0.6863] | +0.03% | 13 / 0 / 23 | 0.7950 |
| H3 | baseline | +0.4834 | [-0.3451, +1.6351] | +3.47% | 22 / 0 / 14 | 0.9719 |
| H3 | scalar | -0.3553 | [-0.9500, +0.0112] | -2.41% | 2 / 33 / 1 | 1.0000 |

Every pointwise mean-change interval crosses zero. None of the six win/loss sign tests rejects equal win/loss probabilities after Holm correction. These results do not establish equivalence or prove that effects are absent.

## Influence and robust summaries

| Method / control | Median paired change | Trimmed mean change | Range after omitting any one landmark |
|---|---:|---:|---:|
| H1 / baseline | -0.0296 | -0.1266 | [-0.1884, +0.4122] |
| H1 / scalar | -0.0433 | -0.4277 | [-0.8575, -0.3069] |
| H2 / baseline | -0.0059 | +0.2329 | [+0.3054, +0.9438] |
| H2 / scalar | +0.0518 | +0.1923 | [-0.1397, +0.2463] |
| H3 / baseline | -0.0297 | -0.0249 | [+0.0421, +0.5998] |
| H3 / scalar | +0.0000 | +0.0000 | [-0.3712, -0.1251] |

The trimmed sensitivity removes the three smallest and three largest paired differences (8.33% from each tail), leaving 30 landmarks. It is exploratory and does not replace the prespecified untrimmed mean.

- **H1 versus baseline:** largest improvement: Place_Jacques-Cartier (-6.6365); largest regression: Palácio_Nereu_Ramos (+14.3823). Omitting the largest improvement gives mean change +0.4122; omitting the largest regression gives -0.1884.
- **H1 versus scalar:** largest improvement: Place_Jacques-Cartier (-11.6646); largest regression: Heian-jingū (+7.6066). Omitting the largest improvement gives mean change -0.3069; omitting the largest regression gives -0.8575.
- **H2 versus baseline:** largest improvement: Prambanan (-2.6956); largest regression: Palácio_Nereu_Ramos (+19.6498). Omitting the largest improvement gives mean change +0.9438; omitting the largest regression gives +0.3054.
- **H2 versus scalar:** largest improvement: Itmad-Ud-Daulah's_Tomb (-8.4794); largest regression: Saint_Demetrius_Church_(Vladimir) (+5.0310). Omitting the largest improvement gives mean change +0.2463; omitting the largest regression gives -0.1397.
- **H3 versus baseline:** largest improvement: Marine_Corps_War_Memorial (-3.5910); largest regression: Palácio_Nereu_Ramos (+15.9304). Omitting the largest improvement gives mean change +0.5998; omitting the largest regression gives +0.0421.
- **H3 versus scalar:** largest improvement: Marine_Corps_War_Memorial (-8.4135); largest regression: Mansudae_Grand_Monument (+0.2014). Omitting the largest improvement gives mean change -0.1251; omitting the largest regression gives -0.3712.

## Distinct summaries answer distinct questions

| Method / control | Change of means (%) | Mean landmark change (%) | Median landmark change (%) |
|---|---:|---:|---:|
| H1 / baseline | +1.55 | +0.03 | -1.31 |
| H1 / scalar | -4.22 | -1.08 | -1.04 |
| H2 / baseline | +6.05 | +2.51 | -0.30 |
| H2 / scalar | +0.03 | +1.47 | +1.27 |
| H3 / baseline | +3.47 | +0.15 | -0.82 |
| H3 / scalar | -2.41 | -1.12 | +0.00 |

The prespecified absolute mean error gives each landmark equal weight in error units. The percentage change of means equals `100 × (mean(method) / mean(control) − 1)`, and implicitly weights per-landmark relative changes by control error. Mean landmark percentage change instead averages `100 × (method_i / control_i − 1)` equally. Neither may be substituted for the other. A positive mean paired change can coexist with more individual wins than losses because regressions can be larger than improvements.

## Calibration remains separate

| Method | Selected calibration mean | Scalar calibration mean | Selected minus scalar |
|---|---:|---:|---:|
| H1 | 10.2625 | 9.6894 | +0.5731 |
| H2 | 9.9655 | 9.6894 | +0.2761 |
| H3 | 9.2678 | 9.6894 | -0.4216 |

The 18 calibration landmarks selected the settings. Their scores above are descriptive and optimistic for selection purposes; they are not an additional independent test, are not pooled with the 36 confirmation landmarks, and are not used for confirmation confidence intervals.

## Method and limits

- Input: `analysis/sources/paired_hypotheses.csv`, checked against the frozen study split and existing verified condition means. Scene keys are unique and all 36 scenes match the confirmation split.
- Bootstrap: 100,000 draws of 36 landmarks with replacement, seed 20261126, using identical sampled indices for all paired conditions. Intervals are the 2.5th and 97.5th percentiles and are pointwise, not simultaneous.
- Exact two-sided sign tests condition on non-ties and test equal probabilities of improvement/regression. Ties use absolute tolerance 1e-12. Holm correction covers six comparisons. This is a complementary direction-of-effect test, not a test of the mean-error estimand.
- The 54-landmark study subset was deliberately selected for coverage. Resampling these 36 confirmation landmarks describes sensitivity under exchangeability of this observed set; it does not justify population-wide confidence coverage for all 396 landmarks.
- H3 changes only three confirmation predictions relative to scalar. Its 33 ties reflect the learned policy, not 33 independent demonstrations of H3-specific benefit. Only three non-ties inform its sign test; leave-one-out and bootstrap estimates are correspondingly discrete and fragile.
- No per-view replication, holdout outcomes, additional model runs, or post hoc retuning was used. Deleting difficult landmarks is not proposed as a new official score.

Reproduce by running `analyze_effects.py` with Python and NumPy. The default source is the existing `analysis/sources` directory relative to the script, and default output is the script directory. Explicit `--source PATH --output PATH` arguments support relocation. `provenance.json` records source SHA-256 hashes and the resampling protocol; CSV files expose all estimates and every landmark contribution.
