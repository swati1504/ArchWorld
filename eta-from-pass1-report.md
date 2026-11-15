# Predicting ArchAttend's step size from Pass 1: report

Branch `eric/predict-ita-from-pass-1-signal`, 2026-11-14 to 2026-11-15. VGGT-1B, the 39 dev scenes
(`scenes_dev.txt`), NVIDIA L4. The full step-by-step record, including dead ends, is in
[`eta-from-pass1-log.md`](eta-from-pass1-log.md).

## Summary

- **Question.** ArchAttend picks its step size η per scene by sweeping η ∈ {1, 10, 100, 1000, 5000}
  and keeping the best by ground truth. We asked whether η can be predicted beforehand, from signals
  Pass 1 already computes, without ground truth.
- **Pass 1 rebuilt and checked.** Pass 1 isn't in the repository, so we rebuilt it. It reproduces the
  shipped anchors on 34 of 39 scenes, but only with the anchor score defined as in the authors' *code*;
  the paper's equation gives 17 of 39.
- **The bar is high.** A fixed η=1000 is already within 4.4% of the per-scene best on average. No
  predictor built from 88 Pass-1 features, or from scene metadata, beat it.
- **The labels are noisy.** The scoring step lines the prediction up with the ground truth using
  unseeded random sampling. On two hard scenes, re-scoring the same untouched prediction moved the
  error by 10–15% (standard deviation); on Dendera by 2%. The typical difference between a scene's best
  and second-best η is 1.7%. For many scenes the "best η" label is partly luck.
- **One real signal.** Scenes whose anchor is small (few patches, seen in few views) are the ones
  where η=5000 fails badly (3–6× worse). We wrote this hypothesis down before testing it, and it held
  (Spearman ρ ≈ −0.40, p ≈ 0.01).
- **A selector built on it** narrowly beats η=1000 (mean regret 4.20% vs 4.38%, never picks a
  catastrophic η), but the margin is too small to call decisive on 39 scenes. A simpler threshold
  rule on the same signal fails.
- **Next.** Test the anchor-size finding on new scenes (about 14 GPU-hours for 40 scenes). Separately,
  get noise-free labels, which needs swati's saved raw predictions.

## 1. The question

ArchAttend (arXiv 2610.22514) improves a 3D foundation model's reconstruction in two passes:

- **Pass 1** finds the *anchor*: the architectural element (facade, window, tower, …) whose image
  patches attend to each other most strongly across views (Semantic Cross-View Attention, SCVA).
- **Pass 2** pushes the model's tokens one step of size η in the direction that increases attention on
  the anchor.

The best η differs by scene, so the paper sweeps five values and keeps the best by ground-truth
Chamfer distance. That costs five reconstructions and needs ground truth, which a user wouldn't have.
**Our goal:** pick η before Pass 2, from what Pass 1 already computes. (swati works on the
complementary route: scoring the five swept outputs with unsupervised proxies.)

We kept η as an ordered choice over the whole grid, scored by **regret**: how much worse the chosen
η's error is than the scene's best, Score(chosen)/Score(best) − 1, averaged over scenes. Every
selector was evaluated leave-one-scene-out: trained on 38 scenes, tested on the 39th.

## 2. What the labels look like

swati's sweep (39 scenes × 5 η, VGGT, fp32, L4) gives the labels. Before predicting anything, we
looked at them:

| η | mean regret | scenes where it is best |
|---|---|---|
| 1 | 7.6% | 4 |
| 10 | 8.1% | 5 |
| 100 | 7.4% | 7 |
| **1000** | **4.4%** | 7 |
| 5000 | 38.9% | 16 |

- **Most scenes are flat.** The median scene's worst η is only 10% worse than its best.
- **η=5000 is bimodal.** It is the most frequent winner, but it wrecks a few scenes: Santa Prisca 6.3×
  worse than the best η, Theater of Petra 3.9×, Kumano Hongu Taisha 3.4×, Disney California Adventure 3.0×.
- **η=1000 is the bar.** It is within 2% of the best η on 20 scenes and never worse than 22%. Every
  selector below has to beat its 4.4% mean regret.

## 3. Rebuilding Pass 1, and a discrepancy with the paper

The repository ships Pass 1's *result* (`intervention/assets/anchors.json`) but not the code that
computes the signals. We rebuilt it:

- **Masks** (`pass1_signals/make_masks_full.py`): SAM 3 on every view, using the scene's architectural
  vocabulary plus non-architectural classes (sky, tree, bush, grass, water, person, car, road). The
  paper's non-architectural list isn't public; we asked the authors and used our own meanwhile.
- **Attention features** (`pass1_signals/extract_features.py`): one VGGT forward pass per scene with
  hooks on all 24 global attention layers. It records SCVA per element and layer, attention leaking
  to non-architectural regions, attention to special tokens, and attention entropy, without forming
  the full attention matrix. It is unit-tested against the authors' function, matching to six digits.
- **88 summary features** per scene (`pass1_signals/features.py`).

To check the rebuild, we recomputed each scene's anchor and compared it with the shipped one:

| SCVA definition | anchors matching `anchors.json` |
|---|---|
| authors' code (cross-view pairs only, mean attention per view pair) | **34 / 39** |
| paper's Eq. 1–2 (sum over all patch pairs) | 17 / 39 |

**The shipped anchors were computed with the code's definition, not the paper's equation.** The
equation sums over all patch pairs, so it grows with the square of an element's size and favours big
elements. On Paul Kruger Church, for example, it prefers the belfry over the window that was actually
steered. Rotary position embeddings never changed the pick.

Four of the five misses are near-ties: the shipped anchor scores 85–99% of our pick. The fifth,
Disney, turned out to matter later. SAM 3 finds almost no architecture there: the candidate anchors
cover 83–397 patches in only 2–3 of 50 views, so their scores are near zero and the pick is fragile.

## 4. First attempt: nothing beats η=1000

With 88 features in hand, we tried ridge regression on the whole regret curve, ridge on log η, and
feature subsets chosen inside each fold:

- **None beat η=1000.** The best were statistical ties (4.36–4.41% mean regret) because they mostly
  fell back to choosing 1000. Models with many features did worse (7–20%): occasionally they chose
  η=5000 on a scene where it is catastrophic.
- **The strongest single signal** was attention entropy at layer 4: scenes with more diffuse early
  attention prefer smaller η (ρ = −0.47, p = 0.003). But with 88 features tested, nothing survived
  multiple-testing correction.
- **Scene metadata** (region, architecture type, anchor type, construction year, number of views)
  did no better: no significant correlations, and no metadata selector beat η=1000.

So either Pass 1 carries little information about η, or something about the labels makes it hard to
learn. The next two sections are about the labels.

## 5. A proper "before": how much does ArchAttend help?

swati's sweep has no no-intervention run. Eric's first baseline (`eval/evaluate.py`) turned out to
score on a different scale (about 12× higher Chamfer distance on Dendera, because it scores with a
different procedure). So Eric reran untouched VGGT through the same pipeline as the sweep
(`intervention/run.py` at η=0; `eric/vggt_base_eta0_dev_results.csv`). It matches swati's η=1 closely:
per-scene rank correlation 0.997.

Against that baseline, per scene:

| η | median change in error vs η=0 | scenes improved | Wilcoxon p |
|---|---|---|---|
| 1 | −0.2% | 20 / 39 | 0.82 |
| 10 | 0.0% | 19 / 39 | 0.91 |
| 100 | −0.5% | 22 / 39 | 0.65 |
| **1000** | **−1.2%** | **27 / 39** | **0.046** |
| 5000 | +0.1% (mean +31%) | 19 / 39 | 0.42 |
| best η per scene | −3.0% | 35 / 39 | |

**Only η=1000 helps measurably, and modestly.** Median CD_norm across scenes goes from 3.89
(η=0) to 3.63 with the best η per scene. That per-scene best is the most a perfect predictor could
reach: it improves the median scene by 3.0%, against 1.2% for η=1000. The difference, about 1.8
percentage points, is the whole prize.

One detail stood out: η=1 should barely change VGGT, yet it differed from η=0 by a median 1.4% per
scene and up to 29%. That looked like noise, so we looked for it.

## 6. The twist: the scoring is noisy

**Source.** To score a prediction, the pipeline aligns it to the ground truth (subsampling, then
RANSAC, then ICP) and measures Chamfer distance. The sampling and RANSAC draw from NumPy's random
generator, which the intervention pipeline never seeds. In a multi-scene run, each scene's draw also
depends on the scenes before it. The authors note this in `intervention/aggregate.py` ("the unseeded
RANSAC").

**Test** (`pass1_signals/seeded_run.py` wraps `run.py` and seeds it before each scene; no repository
code changed). We ran untouched VGGT repeatedly:

- **Same seed, twice: identical to the last digit,** including the predicted point clouds. VGGT itself
  is deterministic; all run-to-run variation comes from the scoring draw, and fixing the seed removes it.
- **Different seeds: large spread.**

  | scene | untouched VGGT, 5 runs | spread | swati's 5 ArchAttend runs |
  |---|---|---|---|
  | Paul Kruger Church | 0.180 – 0.239 | ±10% | 0.192 – 0.227, all inside the untouched range |
  | Shuri Castle | 0.486 – 0.671 | ±15% | 0.595 – 0.751, 2 of 5 inside |
  | Dendera | 0.0386, 0.0394 (2 runs) | ~2% | 0.0382 – 0.0405 |

  On Shuri Castle a bad draw roughly doubles the completion error: the alignment occasionally lands
  clearly wrong rather than slightly off. Shuri Castle at first looked like a scene ArchAttend makes
  16–47% worse; most of that is scoring luck.

**What this means:**

- **For predicting η:** for scenes like Paul Kruger Church, the "best η" label is indistinguishable
  from luck. No predictor can learn that, which may explain much of section 4. Large effects, such as
  η=5000's 3–6× failures, are real.
- **For the paper's before/after numbers:** *before* is averaged over the five sweep runs, which
  cancels the noise, while *after* takes the best of the five, which picks the luckiest draw. Picking
  the best of five noisy runs looks better than average even with no real effect. So the reported
  improvement is likely inflated by some amount we haven't measured. We tested only VGGT, on three
  scenes, two of them chosen because they looked noisy. That's a reason to check, not a demonstration
  that the paper's numbers are wrong.

**Removing the noise from the labels** means rescoring each run with a fixed seed, or averaging
several draws. It needs VGGT's raw per-view predictions. swati's sweep saved them (`SAVE_RAW_TENSORS=1`
in `swati/run_sweep.sh`) but didn't upload them. Rescoring then costs about 2.5–6 CPU-minutes per draw
on our 4-core machine, so a full set of clean labels is a substantial job. **We parked this thread**
and continued, treating differences of a few percent as unreadable.

## 7. Using what rises above the noise: anchor size and big steps

The Disney case suggested a hypothesis: if the anchor is tiny, pushing hard on it should wreck the
reconstruction. We fixed the test, the features and the selectors in the log before running anything:

- **Hypothesis:** the smaller the anchor, the worse η=5000 does relative to the best η ≤ 1000.
- **Result: confirmed.**

  | anchor size measure | Spearman ρ with η=5000's failure | p |
  |---|---|---|
  | share of views containing the anchor | −0.40 | 0.012 |
  | share of image patches covered by the anchor | −0.41 | 0.010 |

  The six scenes where η=5000 fails worst have anchors covering a median 0.7% of patches in 68% of
  views; the others cover 8.0% in 98%. Since Disney inspired the hypothesis, we re-tested without it:
  ρ ≈ −0.35, p ≈ 0.03. Weaker, but it holds.
- **A selector built on it** (`pass1_signals/step_risk.py`). For each new scene it takes the 10 most
  similar training scenes on five pre-chosen Pass-1 features (the two anchor-size measures, anchor
  share of SCVA, non-architectural share, layer-4 entropy), averages their regret curves, and picks
  the η with the lowest average:

  | | always η=1000 | 10-neighbour selector |
  |---|---|---|
  | mean regret | 4.38% | **4.20%** |
  | median regret | 1.92% | **1.21%** |
  | picks the scene's best η | 18% | **36%** |
  | times it chooses η=5000 | 0 | 12 |
  | catastrophes (regret > 50%) | 0 | **0** |

  It uses η=5000 on 12 scenes and never on one where 5000 fails. It met our pre-set bar: lower mean
  regret, and better than the same selector given shuffled features (permutation p = 0.025). But
  the margin is 0.18 points. Head to head it wins 10 scenes and loses 6 (sign test p = 0.45). With
  5 neighbours instead of 10 it fails (6.8%).
- **A simpler rule fails** (`pass1_signals/threshold_rule.py`): "use η=5000 only if the anchor is
  large, with the threshold learned in each fold" gives 7–8% mean regret. With only about six failure
  scenes, the threshold depends on which of them are in training. When Isa Khan Niyazi's tomb is held
  out, the rule loses the very example that would protect it and sends it to η=5000 (108% regret).

## 8. Where we are

1. **Pass 1 is reproduced,** and the shipped anchors use the code's SCVA, not the paper's equation.
   Anyone reimplementing ArchAttend from the paper will pick different anchors on more than half the
   scenes.
2. **On VGGT the attainable gain is small.** ArchAttend at η=1000 improves 27 of 39 scenes by a
   median 1.2%; a perfect per-scene choice would reach 3.0%, and part of that extra is luck.
3. **The scoring noise is large enough to blur most per-scene η labels.** It comes entirely from the
   unseeded alignment and would disappear with a fixed seed.
4. **Anchor size predicts when big steps fail.** This is the clearest Pass-1 signal we found: tested
   as planned, physically sensible, and robust to dropping the scene that inspired it.
5. **No Pass-1 selector clearly beats η=1000 yet.** The best one ties or narrowly wins and avoids
   catastrophes; on 39 noisy scenes, that's as far as the evidence goes.

**Limitations:** one model (VGGT), 39 scenes, one run per (scene, η); noise measured on three scenes
only; we looked at many features and selectors on the same 39 scenes, so everything beyond the
pre-registered anchor-size test should be read as suggestive.

## 9. Next steps

| step | what it answers | cost |
|---|---|---|
| Anchor size on 40 new scenes, η ∈ {1000, 5000} | Does "small anchor ⇒ big step fails" hold on unseen scenes? Adds ~6 failure examples. | ~14 GPU-hours (353 extracted scenes available) |
| Clean labels from swati's `inter_raw.npz` | Which per-scene η differences are real; a fair before/after comparison | CPU only, but 2.5–6 min per draw; a subset first |
| DA3 or π³ sweep | η may matter more there, above the noise | many GPU-hours; memory on the L4 untested |
| Full sweep on more scenes | Enough data to train and validate a real selector; continuous η | days of GPU time |

## Appendix: reproducing

Environments: `archworld` (VGGT, analysis), `archworld-sam3` (masks). Outputs go to `pass1_out/`
(git-ignored).

```bash
# Pass 1 signals (GPU, run alone: features peak at ~21 GB)
conda activate archworld-sam3
python pass1_signals/make_masks_full.py --root data --out pass1_out/masks --scenes-file scenes_dev.txt --quiet
conda activate archworld
python pass1_signals/extract_features.py --root data --masks-dir pass1_out/masks --out pass1_out/features \
       --scenes-file scenes_dev.txt
python pass1_signals/features.py --in pass1_out/features --out pass1_out/pass1_features.csv

# Analyses (CPU, seconds to minutes)
L=swati/results_vggt_dev/per_scene.csv; F=pass1_out/pass1_features.csv
python pass1_signals/eta_landscape.py $L --out pass1_out/eta_landscape.csv          # section 2
python pass1_signals/metadata_baseline.py --labels $L --out pass1_out/meta_features.csv
python pass1_signals/pass1_selectors.py --labels $L --features $F                    # section 4
python pass1_signals/eta0_analysis.py                                                # section 5
python pass1_signals/step_risk.py --labels $L --features $F                          # section 7
python pass1_signals/threshold_rule.py --labels $L --features $F

# Seeded run of the unchanged pipeline (section 6)
python pass1_signals/seeded_run.py --seed 0 -- --model vggt --root data \
       --masks-dir pass1_out/masks --scene Dendera --eta 0 --out <out dir>
```

Inputs from collaborators: swati's sweep (`swati/results_vggt_dev/`), Eric's η=0 baseline
(`eric/vggt_base_eta0_dev_results.csv`, procedure in `eric-outcome-editing.md`).
