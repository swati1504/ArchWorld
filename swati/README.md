# Inference-time eta selection for ArchAttend (VGGT)

Goal: pick ArchAttend's step size eta per scene at inference time, by optimizing an
unsupervised objective computed from the model's own output (no ground truth).

## Setup used
- Scenes: 80 downloaded, split 40 dev / 40 test (`scenes_dev.txt`, `scenes_test.txt`; one dev scene,
  Bridge_of_San_Martín, is not in the benchmark views and was dropped -> 39 dev).
- Data fix: `python tools/flatten_scene_images.py --root data` (images live in category subfolders).
- Sweep: `swati/run_sweep.sh` (VGGT, layers 0-23, eta in {1,10,100,1000,5000}, SAVE_RAW_TENSORS=1,
  SKIP_BASELINE_PASS=1 because base+intervention does not fit in 24 GB at 50 views).
- Scores per scene and eta: `swati/results_vggt_dev/eta_*/<scene>.json`, table in `per_scene.csv`.

## Scripts
| script | what it does |
|---|---|
| `proxy_select.py` | first proxies (confidence, multi-view consistency) vs fixed eta and oracle |
| `picker.py` | computes per-(scene, eta) features incl. depth/rotation drift vs eta=1; writes `picker_features.csv` |
| `analyze_picker.py` | compares pickers (leave-one-out tau) with bootstrap CI and Wilcoxon test |
| `pick_eta.py` | the locked rule, to apply to new scenes (test set) |

## Dev results (39 scenes)
Regret = CD(chosen eta) / CD(best eta for the scene); 1.00 = perfect.

| picker | median | mean | worst | >1.2 |
|---|---|---|---|---|
| fixed eta=1000 (best single eta) | 1.019 | 1.044 | 1.22 | 1 |
| fixed eta=5000 | 1.014 | 1.389 | 6.25 | 7 |
| confidence argmax | 1.016 | 1.076 | 2.14 | 4 |
| **guarded-5000 (depth drift, LOO tau)** | **1.011** | **1.037** | 1.22 | 1 |

guarded-5000 vs fixed 1000: 10 wins / 27 ties / 2 losses, Wilcoxon p = 0.064, bootstrap 95% CI of the
mean regret difference [-0.018, +0.004]; closes 15% of the gap to the oracle. Depth drift at eta=5000
correlates with eta=5000's regret (Spearman +0.58), i.e. it flags the runs where the large step breaks
the reconstruction.

## Locked rule (for the test set)
pick eta=5000 if median relative depth change(eta=5000 vs eta=1) <= 0.00626, else eta=1000.
Needs three runs per scene (eta = 1, 1000, 5000).

## Test results (40 held-out scenes, rule frozen before the run)
Run: `run.py --scenes-file scenes_test.txt --eta 1,1000,5000`, then
`python swati/pick_eta.py results/vggt_test --scenes scenes_test.txt --csv swati/results_vggt_test/picks.csv`.
Per-scene drift, pick and CD for eta=1000/5000 are in `results_vggt_test/picks.csv`.
Regret here is against the better of eta=1000 and eta=5000 (the two values the rule chooses between).

| picker | mean CD | median CD | mean regret | worst regret | >1.2 |
|---|---|---|---|---|---|
| **fixed eta=1000** | **0.0632** | 0.0215 | **1.017** | 1.11 | 0 |
| fixed eta=5000 | 0.0735 | 0.0252 | 1.545 | 18.67 | 6 |
| guarded-5000 (ours) | 0.0641 | 0.0215 | 1.033 | 1.65 | 1 |
| oracle (best of 1000/5000) | 0.0623 | | 1.000 | | |

- The rule picked eta=5000 on 8 scenes. Against fixed eta=1000: 4 wins / 33 ties / 3 losses,
  mean relative change +1.5% (Wilcoxon p = 1.0). Almost all of the gap is one miss:
  Degutale_Temple,_Patan (drift 0.0049, under the threshold; eta=5000 is 65% worse there).
- The drift guard does detect failures: eta=5000 is >20% worse than eta=1000 on 6 test scenes and the
  rule falls back to 1000 on 5 of them (e.g. Gangaikonda_Cholapuram_Temple, drift 0.48, eta=5000 18.7x worse).
- Headroom is small: a perfect choice between 1000 and 5000 is only 1.4% better than fixed 1000, and no
  threshold applied on the test set does better than 0.0632. That is below the scoring noise
  (unseeded RANSAC alignment moves CD by 2-15% per scene; see eric's eta-from-pass1-report.md).

**Conclusion.** Depth drift at a large step is a reliable label-free failure detector, but per-scene
selection between eta=1000 and eta=5000 does not beat a fixed eta=1000 on held-out scenes. The dev-set
gain (mean regret 1.037 vs 1.044, p = 0.064) did not hold on test.

## Post-hoc robustness: other ways to summarize drift
`drift_variants.py` re-runs the guarded-5000 rule with seven summaries of the same per-pixel relative
depth change (eta=5000 vs eta=1), each with the same protocol (tau tuned leave-one-out on dev, frozen,
applied once to test). Regret here is against the better of eta=1000 and eta=5000 on both splits.
Per-scene values: `results_drift_variants.csv`.

| summary | dev LOO regret | test mean CD | vs fixed 1000 (W/T/L) | test failures caught | Degutale |
|---|---|---|---|---|---|
| median (locked rule) | 1.0259 | 0.0641 | 4/33/3 | 5/6 | missed |
| p75 | 1.0230 | 0.0640 | 8/25/7 | 5/6 | missed |
| p90 | 1.0264 | 0.0632 | 6/29/5 | 6/6 | caught |
| largest per-view median | 1.0258 | 0.0641 | 4/31/5 | 5/6 | missed |
| 75th pct of per-view medians | 1.0262 | 0.0641 | 5/29/6 | 5/6 | missed |
| share of pixels moved > 5% | 1.0280 | 0.0632 | 6/28/6 | 6/6 | caught |
| share of pixels moved > 10% | 1.0364 | 0.0632 | 5/31/4 | 6/6 | caught |
| fixed eta=1000 (reference) | 1.0324 | 0.0632 | | 0 failures | |

- On dev every summary flags all 7 eta=5000 failures, so dev cannot tell them apart, and none beats
  the median there. We therefore keep the median as the locked rule and report this as post-hoc.
- On test, the summaries that count strongly moved pixels (p90, share > 5%, share > 10%) catch all 6
  failures, including Degutale_Temple,_Patan, which the median missed. Its change is concentrated in
  part of the image (see `figures/drift_demo.png`), so the median stays low. Degutale clears the p90
  threshold only narrowly (0.0378 vs 0.0336).
- No summary beats fixed eta=1000 on test (best: a tie at 0.0632), consistent with the 1.4% headroom.

`make_demo_figure.py` draws `figures/drift_demo.png`: input photo, depth at eta=1 and eta=5000, and the
per-pixel change for a caught failure, a safe scene, and the missed scene.
