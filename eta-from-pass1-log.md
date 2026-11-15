# Predicting ArchAttend's η from Pass-1 signals: steps and outcomes

Branch: `eric/predict-ita-from-pass-1-signal` (worktree, based on `516379d`). Started 2026-11-14.

## Goal

ArchAttend's step size η is chosen per scene by an oracle: sweep η ∈ {1, 10, 100, 1000, 5000} and keep
the best by ground-truth Chamfer distance. That costs 5 runs and needs ground truth. This work picks η
**before Pass 2**, from signals Pass 1 already produces, with no ground truth. (swati covers the other
route: scoring the swept outputs with unsupervised proxies, `swati/proxy_select.py`.)

## Background (from the project PDF and arXiv 2610.22514)

- **SCVA**: for element *e*, S(e) = Σ_{i,j ∈ P_e} A[i,j], attention averaged over heads, at a global
  attention layer. P_e spans all views; same-view pairs appear to be included. No normalisation.
- **Pass 1**: anchor e\* = argmax over architectural elements of S at the last layer (VGGT: layer 23).
- **Pass 2**: at each global layer (VGGT: 0–23 per the `run.py` table), T ← T + η·∇_T S(e\*), one step.
- **Masks**: SAM 3 (`facebook/sam3`), text prompts, confidence 0.5. Architectural vocabulary is in
  `intervention/assets/elements.json`. The paper's non-architectural list (sky, trees, …; Appendix C.1)
  is not in the repo.
- **What's known about η**: VGGT's best fixed η gives −6.2% median CD; the per-scene oracle gives about
  −13%. π³ diverges at η=1000 on 61/395 scenes and at η=5000 on 251/395. The paper shows the SCVA
  "leakage" link to error only qualitatively (Fig. 4), with no correlations.
- **Implementation note**: `intervention/models/vggt/loss.py:143-146` passes the mask as the value tensor
  to `scaled_dot_product_attention`, which gives each query's attention mass on the mask without building
  the N×N matrix. One call can score many elements (one per value column).

## Plan

0. Look at η against error per scene from swati's VGGT sweep: how often each η wins, how flat the curves
   are, and the share of oracle gain the best fixed η recovers.
1. Pass-1 feature extractor: anchor SCVA and its share, margin over runner-up, leakage onto
   non-architectural elements, attention entropy, layer profile (4/11/17/23 or 10–23), predicted camera
   dispersion, depth confidence, response to one η=1 step.
2. Check the signals: correlate each feature with base error and log η\*, leave-one-scene-out on the
   39 dev scenes.
3. Selectors: (a) a small predictor from features to η; (b) a threshold gate in the style of
   Attend-and-Excite (smallest η reaching a target S(e\*)). Compare with swati's proxies.
4. Evaluate on `scenes_test.txt` once: no intervention, best fixed η chosen on dev, oracle. Report
   regret and the share of oracle gain recovered.

## Conventions

- Large data (`data/`, masks, results) stays outside git. `data` is a symlink to
  `/mnt/disks/archworld-data/data`.
- New folders get descriptive names, not `eric/` or `results/`.

## Log

### 2026-11-14: setup

- Created this worktree. Linked `data` to `/mnt/disks/archworld-data/data`.
- Read the project PDF, arXiv 2610.22514, 2511.22686 and 2301.13826 (findings summarised above).
- Found that Pass 1 is not in the repo: `run.py` reads the anchor from `assets/anchors.json` and writes
  a placeholder score. The Pass-1 signals have to be computed by new code.

### 2026-11-14: inventory, starting without swati's sweep

swati's sweep results aren't available (no reply yet). Checked what exists locally:

- **No η-sweep outputs anywhere** on this machine (no `eval_metrics_*.json`, no `output_<η>/`).
- **No masks** in either checkout (`/home/eric/ArchWorld/masks` is empty).
- **SAM 3 is usable**: conda env `archworld-sam3` exists, and the HF account has access to the gated
  `facebook/sam3` (checked with `huggingface_hub.auth_check`). The weights aren't downloaded yet (3.45 GB).
- **GPU (L4, 23 GB) is busy** with the VGGT base reproduction in the main checkout: 18.7 GB used,
  14/39 dev scenes done at about 6 min per scene, so about 2.5 h left.
- **Base CD per scene** from that run (`/home/eric/ArchWorld/results/vggt_base/metrics_vggt.csv`) can be
  used as a label-free target: does a Pass-1 signal predict how badly the scene reconstructs?

Decision: start with the steps that need no η labels (masks, feature extractor, signal against base
error), and produce our own η labels on a subset if swati's don't arrive.

### 2026-11-14: do published results give η labels? No

Checked whether the paper's release already holds per-scene, per-η numbers we could use instead of
swati's sweep:

- **Paper (arXiv 2610.22514)**: all tables are aggregates (medians, IQRs, per region or type). The
  VGGT/π³ ArchAttend results are only in a figure (Fig. 13). No per-scene or per-η numbers.
- **Project page** (cornell-vailab.github.io/ArchWorld): example images and one DA3 figure, no metrics files.
- **HF dataset** `ArchWorld/ArchWorld`: 399 files, all scene tarballs plus `ArchWorld_metadata.csv` and
  `README.md`. No results.
- **Repo**: ships `anchors.json`, `elements.json`, `views.json` only.

So reproducing the paper doesn't give swati per-scene η labels from the authors either; she has to
run the sweep herself. The labels have to come from her runs, from our own sweep, or from the authors
(Evan Zhang) on request.

**Papers saved locally** in `papers/` (git-ignored): the project PDF and arXiv 2610.22514, 2511.22686,
2503.11651, 2511.16719, 2208.01626, 2301.13826. Read them from there, not the web.

### 2026-11-14: code reading, mask script, email to the authors

Findings from the code:

- **The authors' SCVA in code is not the paper's Eq. 1–2.** `compute_feature_attention_sum`
  (`intervention/models/vggt/run_pass.py:345`) sums over view pairs i≠j only (cross-view), and takes
  the **mean** attention per view pair. Its hook (`compute_per_layer_most_attended_features`, :379)
  applies the q/k norms but **not RoPE**, while the model's real attention does use RoPE. We compute
  both versions and check which one reproduces `anchors.json`; this replaced asking the authors.
- **VGGT layer range for Pass 2 is 0–23** (`run.py` MODELS table), not 10–23 (the `loss.py` default).
- **Token layout**: 5 special tokens per view (camera + 4 registers), then patches of 14 px; images are
  518 wide (crop mode). Masks map to patches with `map_mask_to_patches` (patch counts if >50% covered).
- **GPU**: VGGT on 50 views needs about 10–12 GB (weights plus all 24 layer outputs), so the feature
  extractor cannot run alongside the base run (18.7 GB held). SAM 3 may fit in the remaining ~4 GB.

Done:

- Wrote `pass1_signals/make_masks_full.py`: reuses `intervention/make_masks.py` unchanged, swapping in
  the scene's full architectural vocabulary plus our non-architectural list
  `sky, tree, bush, grass, water, person, car, road`. Output goes to `pass1_out/masks/` (git-ignored).
- First test on Dendera **failed at import**: the `archworld-sam3` env lacks `decord`, which SAM 3
  imports even for images. Installed `decord==0.6.0` there; the import then failed on `pycocotools`.
  Both come from SAM 3's `notebooks` extra (pyproject), which `setup/sam3.sh` doesn't install.
  The remaining install is left to Eric (command in the conversation).
- **Workflow change**: Eric runs the main steps (masks, VGGT runs, sweeps) in his own terminal;
  Claude runs only trivial steps, gives the commands and reads the outputs.
- Drafted an email to Evan Zhang asking for (1) per-scene, per-η results and (2) the non-architectural
  vocabulary. Eric sends it.

Observed: the base run in the main checkout stalled at 13/39 on `San_Francisco_Church,_Mexico_City`
(more than 30 min, against about 6 min per scene before).

### 2026-11-14: masks for Dendera, feature extractor written and unit-tested

- Eric finished the SAM 3 env (`pycocotools` etc.) and ran
  `python pass1_signals/make_masks_full.py --root data --out pass1_out/masks --scene Dendera`.
  **Outcome**: 2150 files (50 views × 43 elements) in 296 s, so about 3.2 h for 39 scenes. SAM 3 uses
  about 5 GB. Sanity check of coverage (views found / mean image fraction):
  sky 49/50 (0.27), stone 43 (0.23), **facade 42 (0.39)** (= shipped anchor), doorway 30, grass 28,
  wall 28, column 28, tree 27, triumphal arch 15 (0.47, likely the gateway). Elements foreign to
  Dendera (minaret, dome, pyramid, …) were never found, as expected.
- Wrote `pass1_signals/extract_features.py`. One VGGT forward per scene with hooks on all 24 global
  layers. It records:
  - SCVA per element and layer in 3 definitions (`code` = authors' cross-view per-pair mean;
    `eq` = paper Eq. 1–2; `eqx` = Eq. restricted to cross-view), each with and without RoPE;
  - the anchor each definition picks at layer 23, against `anchors.json`;
  - per query token: attention mass on architectural / non-architectural / special tokens, same-view
    and cross-view, and per non-architectural class;
  - entropy (per-head mean, of the head-mean) and top-1 mass for 4096 sampled queries;
  - output statistics: camera spread / median depth, view-direction spread, depth confidence.

  Raw outputs go to `pass1_out/features/<scene>.{json,npz}`; summary features will be a separate
  script. Weights are cast to bf16 by default (like `VGGT_HALF=1`).
- **Unit test** (random q, k; 4 views, one without the element): all three SCVA definitions match the
  explicit attention matrix, and `code` matches the authors' `compute_feature_attention_sum`
  (0.102301 vs 0.102301). Entropy error 5e-7, top-1 error 6e-9.
- First real run on Dendera: **OOM**. The base run had started its next scene and held 16.5 GB.
  Feature extraction must wait for the base run to end.
- Started masks for all 39 dev scenes in the background (fits next to the base run):
  `python pass1_signals/make_masks_full.py --root data --out pass1_out/masks --scenes-file scenes_dev.txt --quiet`
  → log `pass1_out/logs/masks_dev.log`.
- **Stopped at Eric's request** (23:41): mask job killed during scene 2, `Paul_Kruger_Church` (382 files
  written, scene incomplete; rerunning the same command resumes it, since existing files are skipped).
  The watcher that would have started feature extraction after the base run was stopped too.
  Nothing of ours is running.

### 2026-11-14: swati's sweep arrived; step 0 (η landscape) done

- swati pushed `dde4336`: `swati/results_vggt_dev/` with `eta_<η>/<scene>.json` and `per_scene.csv`.
  It covers **39 dev scenes × η ∈ {1, 10, 100, 1000, 5000}, complete, no NaN**. It is the `inter` arm only
  (`SKIP_BASELINE_PASS=1`); `Score` = (Acc-mean + Comp-mean)/2. The intervention anchor is recorded per
  scene (e.g. Dendera → facade). Eric pulled it (`pull.rebase true` rebased our commits on top).
- Wrote `pass1_signals/eta_landscape.py`, ran
  `python pass1_signals/eta_landscape.py swati/results_vggt_dev/per_scene.csv --out pass1_out/eta_landscape.csv`.

**Outcomes** (scores compared within a scene only):

| η | median Score | mean regret vs oracle | median regret | wins |
|---|---|---|---|---|
| 1 | 0.02814 | 7.6% | 2.9% | 4 |
| 10 | 0.02685 | 8.1% | 2.7% | 5 |
| 100 | 0.02809 | 7.4% | 3.1% | 7 |
| 1000 | 0.02660 | **4.4%** | 1.9% | 7 |
| 5000 | 0.03234 | 38.9% | **1.4%** | **16** |
| oracle | 0.02622 | 0 | 0 | |

- **Most scenes are flat.** Median worst/best η ratio is 1.10×; for 13/39 scenes all five η are within 5%.
- **η = 5000 is bimodal.** It wins most often (16/39) and is best by median regret, but it is
  catastrophic on a few scenes (Santa Prisca 6.3×, Theater of Petra 3.9×, Kumano Hongu Taisha 3.4×,
  Disney California Adventure 3.0× worse than the best η), hence the 38.9% mean regret.
  Against η=1000: 5000 better on 20, worse by >10% on 10, worse by >50% on 5.
- **Best fixed η = 1000** (in-sample): median Score +5.5% vs η=1; oracle +6.8%.
- **The decision reduces to "is η=5000 safe for this scene?"** An oracle restricted to {1000, 5000} has
  mean regret 1.1% (median 0) against 4.4% for fixed 1000. So most of the attainable gain is a
  **binary prediction**, a much sharper target for Pass-1 signals than regressing η.

**Caveats**
- One run per (scene, η). The paper averages 5 runs; RANSAC/ICP alignment noise is unknown, so
  differences of a few percent may be noise (relevant for the flat scenes).
- No base (no-intervention) arm. Eric's `evaluate.py` base Score is on a different scale
  (Dendera: 0.449 vs 0.038 here), so it can't serve as the "before" without matching pipelines.
- "Best fixed η" was chosen on the same scenes it is scored on (optimistic for the baseline).

### 2026-11-15: target definition, evaluation harness, metadata baseline, summary features

**Target (decided with Eric):** keep the full η choice; don't reduce it to "is 5000 safe". η is treated
as ordinal (log η), selectors are scored by **regret** Score(chosen)/Score(best) − 1 under
leave-one-scene-out, plus a whole-curve predictor. The 1000-vs-5000 finding is a check, not the target.

**Evaluation harness**: `pass1_signals/selection.py`. Selectors: fixed η, best-fixed (chosen on training
folds only), oracle, `group:<col>` (best η of the scene's category), `curve-ridge` (ridge on log regret
at every η, pick the lowest), `logeta-ridge` (ridge on log10 η\*, round to grid). Ridge alpha is chosen by
closed-form LOO, verified against brute-force refits (max diff 4e-16). No sklearn (not installed);
numpy only.

Baselines (LOSO, 39 dev scenes):

| selector | mean regret | median regret | max regret | n regret>10% | oracle gain recovered (median Score, from η=1) |
|---|---|---|---|---|---|
| η=1 | 7.6% | 2.9% | 59% | 9 | 0 |
| **best-fixed** (= η=1000 in every fold) | **4.4%** | 1.9% | 22% | 7 | 80% |
| oracle | 0 | 0 | 0 | 0 | 100% |

Best-fixed out-of-sample equals the in-sample choice, so the earlier in-sample caveat didn't bite.
**Bar for any selector: mean regret below 4.4%.**

**Metadata baseline**: `pass1_signals/metadata_baseline.py` → `pass1_out/meta_features.csv`.
- Numeric metadata: no correlation with log10 η\* or with `large_vs_small` (the log ratio of the best
  large-η Score to the best small-η Score). All |ρ| ≤ 0.27, all p ≥ 0.09 (construction year, n views,
  and the GT-derived scale/extent/diagonal).
- Categories: suggestive but tiny groups. SE Asia 5/5 best at η=5000; mosques prefer large η
  (large_vs_small −0.084, n=4); the single Temple/Shrine and Latin America scenes are hurt badly by 5000.
- LOSO with metadata-only selectors: **none beats best-fixed on mean regret.** group:region 13.6%,
  group:arch_type 12.7%, group:anchor 5.8% (lower median 1.1% and 9 wins / 3 losses vs best-fixed, but one
  90% miss), curve-ridge 9.4%, logeta-ridge 4.5% (≈ best-fixed).
- Conclusion: metadata doesn't explain η\*. Pass-1 signals have to beat best-fixed (4.4%), not a strong
  metadata model.

**Summary features**: `pass1_signals/features.py` turns extractor output into one row per scene: anchor
dominance (SCVA, share, margin), non-arch share, anchor self-attention and cross-view fraction, leakage
(anchor queries and all queries), attention sinks, sky mass, entropy, top-1, each at layers 4/11/17/23
plus the mean over layers; and scene-level token fractions, camera spread, depth confidence, and
whether our recomputed anchor matches the shipped one. 88 features in total.
- Added `--device cpu` and `--max-views` to `extract_features.py` (smoke tests only).
- **Smoke test** (CPU, fp32, Dendera, first 3 views, 92 s; output in `pass1_out/features_smoke/`):
  extractor and feature script run end to end. Values are plausible: facade is in only 1 of the
  3 views, so its cross-view SCVA is 0; sky gets 22% of attention against 28% of tokens. Layer-23
  attention is close to uniform (normalised entropy 0.98) — to watch at 50 views. All definitions pick
  `wall` here, which says nothing about the real anchor check (3 views only).

### 2026-11-15: correction to "oracle gain recovered"; η beyond the grid

- **Correction.** Earlier entries said best-fixed (η=1000) recovers **80%** of the oracle's gain. That
  was a ratio of *medians across scenes* (0.02814 → 0.02660 vs oracle 0.02622), which is fragile:
  by the same formula η=10 scores 67% despite worse regret than η=1. A per-scene log version gives
  39% for η=1000. Neither number is a measure the authors or swati use, so the column was **dropped**.
- **Metrics now follow the collaborators** (agreed with Eric):
  - *swati* (`swati/proxy_select.py`): regret = CD(chosen)/CD(best η), median and mean, plus how often
    the oracle's η is hit. Ours is the same minus 1. **Primary metric.**
  - *Authors* (`intervention/aggregate.py`): median CD across scenes, before → after, % change, and the
    share of scenes improved. Reported here against η=1 as a stand-in reference, since the sweep has
    no no-intervention arm.

  Baselines in the new format (LOSO): η=1000 / best-fixed: mean regret 4.4%, hits the oracle on 18%,
  median Score −5.5% vs η=1, improves 69% of scenes. Oracle: −6.8%, improves 90% (η=1 is itself
  best on 4 scenes).
- Why η=1000 has low regret despite winning only 7/39 scenes: it is within 2% of the best η on 20
  scenes, within 5% on 29, within 10% on 32; worst 22%.
- **η is not limited to {1, 10, 100, 1000, 5000}** (Eric). Decision: evaluate on the 5-point grid for
  now, since that's where labels exist. Continuous η is a later step once a selector beats best-fixed
  on the grid: selectors output log η; scoring off the grid needs either interpolation in log η
  (risky between 1000 and 5000, where errors jump) or extra sweep runs (e.g. 2000, 3000).

### 2026-11-15: our GPU jobs caused 2 base-run OOMs; first full-size features

- **Incident.** Two base-run scenes in the main checkout were recorded `oom`, and both coincide with our
  GPU jobs: Santa_Prisca_Church (23:30:55) with the fp32 SAM 3 Dendera test (23:27–23:32), and
  Basílica_Colegiata…Guanajuato (23:34:34) with the first VGGT feature attempt. Eric reran them; the
  base run now has 39/39 `ok`. **Lesson: no GPU jobs alongside the base run** (base peaks at ~18.7 GB).
- **Dendera features at full size** (GPU, bf16 weights, 50 views, 68,700 tokens): 112 s, **peak 21.0 GB**.
  Dendera already has the maximum token count of all dev scenes (50 views at 518×518), so this is the
  worst case. Features need the GPU alone (SAM 3 + VGGT doesn't fit).
- **Anchor check, first scene:** all six SCVA definitions (code/eq/eqx × RoPE on/off) pick `facade`,
  matching `anchors.json`. Consistent, but it doesn't yet separate the definitions.
- Started `pass1_out/run_dev_pipeline.sh` at 01:21: masks for all 39 dev scenes, then features for all
  39, sequentially. Logs in `pass1_out/logs/{pipeline_dev,masks_dev,features_dev}.log`. Expected about
  3.2 h for masks plus about 75 min for features.

### Open questions

- [x] swati's VGGT sweep results: `swati/results_vggt_dev/` (all 39 dev scenes).
- [ ] Run-to-run noise of a (scene, η) score: repeat a few to know which differences are real.
- [ ] A base arm on the same pipeline (`run.py` without `SKIP_BASELINE_PASS`) for "no intervention".
- [ ] The paper's exact non-architectural vocabulary: asked Evan Zhang; using our own list meanwhile.
- [ ] Which SCVA version reproduces `anchors.json` (paper's Eq. vs code's cross-view mean; RoPE or not).
