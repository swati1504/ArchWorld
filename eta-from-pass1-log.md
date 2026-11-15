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

### 2026-11-15: early batch; anchor check on 6 scenes

- Restarted the pipeline in two stages so the check could start early (Eric's request): batch 1 =
  `scenes_dev.txt` lines 2–6, then the remaining 33 (`pass1_out/scenes_{batch1,rest}.txt`). On the
  restart, a `pkill -f` pattern matched its own command line and killed the driver script but not
  the mask process, which I stopped by PID; all 4,968 recent mask files were checked readable.
- Masks ran faster than estimated: about 1.5–3.5 min per scene. Batch 1 features: 98–161 s per scene,
  peak 20.8–21.0 GB.
- **Anchor check, 6 scenes (Dendera + batch 1):**

  | scene | shipped | code | eq | eqx |
  |---|---|---|---|---|
  | Basilique … (Ouidah) | arch | arch | belfry | belfry |
  | Dendera | facade | facade | facade | facade |
  | Hassan_Tower | pillar | pillar | pillar | pillar |
  | Kumano_Hongu_Taisha | gate | gate | gate | gate |
  | Paul_Kruger_Church | window | window | belfry | belfry |
  | Red_House_Theater | window | window | facade | facade |

  **code definition: 6/6 matches** (with and without RoPE); **paper's Eq. (eq, eqx): 3/6.** RoPE on/off
  never changed the pick. Where they disagree, Eq. picks a large element (belfry, facade): it sums over
  all patch pairs, so it grows with element size squared. Example, Paul_Kruger_Church: window has 2,182
  patches in 43 views (code 0.0269, eq 65); belfry has 7,494 patches in 16 views (code 0.0056, eq 1,269).
- **Interpretation:** the shipped anchors were produced with the code definition (cross-view pairs,
  mean per view pair), not the paper's Eq. 1–2, and our pipeline (masks, token mapping, attention,
  SCVA) reproduces them so far. To confirm on all 39.

### 2026-11-15: all 39 dev scenes; anchor check; first Pass-1 selector test (negative)

- Pipeline finished: masks and features for all 39 dev scenes, no failures. Features for the last 33
  scenes took about 70 min. `features.py` built `pass1_out/pass1_features.csv` (39 scenes x 88 features).
- **Anchor check, 39 scenes:** code definition **34/39**, paper's Eq. (eq and eqx) **17/39**; RoPE never
  changes the pick. Of the 5 code mismatches, 4 are near-ties (the shipped anchor's SCVA is 85-99% of our
  top pick: Hermesvilla 0.99, Stadhuis Gouda 0.92, Baguio 0.91, San Francisco Mexico City 0.85), so a
  bf16 or mask difference can flip them. Disney is a real miss (shipped tower = 0.2x arch). Conclusion:
  the shipped anchors come from the code definition, and our pipeline reproduces Pass 1.
- **Correlations (`pass1_signals/pass1_selectors.py`, log `pass1_out/logs/pass1_selectors.log`):**
  strongest is early-layer attention entropy, layer 4. Scenes with more diffuse attention there prefer
  smaller eta: `anc_ent_L4` rho = -0.49 with log10 best eta (p = 0.001), `ent_L4` -0.47 (p = 0.003).
  Next: `nonarch_share` (share of SCVA on non-architectural elements) at L4/L17/L23, rho about -0.35 to
  -0.40: more attention on sky/trees, smaller eta. **None survives multiple-testing correction** over 88
  features (best BH q = 0.11).
- **LOSO selectors vs best-fixed (eta=1000, mean regret 4.38%):** none beats it. Best are
  logeta-ridge top1 (4.36%, W/L 7/9), curve-ridge top10 (4.41%), logeta-ridge Lmean+scene (4.39%,
  W/L 9/11). These are ties: they mostly fall back to choosing 1000. Curve-ridge with many features is
  worse (7-20%) because it sometimes predicts 5000 on a scene where 5000 is catastrophic.
- **Reading:** with 39 scenes, one noisy run per (scene, eta), and a best-fixed that is already within
  4.4% of the oracle on average, Pass-1 summary features give at most a weak signal (early entropy,
  non-arch share), not yet a usable selector. The oracle headroom is small on average (median Score
  -6.8% vs eta=1 for oracle, -5.5% for eta=1000) and concentrated in a few scenes.
- Not done: the leakage-vs-base-error test needs a base arm on the same pipeline (open question).

### 2026-11-15: η=0 base on the same pipeline (Eric's run); gain, noise floor, leakage

Eric reran untouched VGGT through `run.py` at η=0 (fp32; `eric/vggt_base_eta0_dev_results.csv`, commit
75f52b1). Same scale as swati's sweep (Dendera CD 0.0386 at η=0 vs 0.0382 at η=1). Analysis:
`pass1_signals/eta0_analysis.py`, log `pass1_out/logs/eta0_analysis.log`; combined labels with η=0 as a
sixth grid point: `pass1_out/labels_with_eta0.csv`.

- **ArchAttend vs no intervention, per scene (CD relative to η=0):**

  | η | median | mean | scenes better | Wilcoxon p |
  |---|---|---|---|---|
  | 1 | −0.17% | +0.73% | 20/39 | 0.82 |
  | 10 | +0.01% | +0.63% | 19/39 | 0.91 |
  | 100 | −0.52% | +0.43% | 22/39 | 0.65 |
  | 1000 | **−1.23%** | **−1.64%** | **27/39** | **0.046** |
  | 5000 | +0.09% | +31.3% | 19/39 | 0.42 |
  | oracle η | −2.98% | | 35/39 | |

  Only η=1000 helps, and only modestly. The oracle's −3% is partly selection on noise (next point).
- **Noise floor.** η=1 should barely move VGGT, yet |CD(η=1)/CD(η=0) − 1| has median 1.37%, 75th
  percentile 7.6%, max 29%. Either the pipeline is not deterministic run to run or η=1 matters more
  than expected; either way that is the size of differences we cannot trust from one run. Gap between
  a scene's best and runner-up η: median 1.71%, and in 17/39 scenes it is below 1.37%. **For about half
  the scenes the "best η" label is within noise.**
- **LOSO with η=0 allowed:** best-fixed is still η=1000 (mean regret 5.3%; median Score −1.4% vs η=0,
  better on 69% of scenes). Oracle −2.8%. So the whole headroom a selector could capture over η=1000
  is about 1.4% median CD, comparable to the noise floor.
- **Paper's leakage claim** (more attention leaking to non-architecture → larger base error): the
  leakage ratios themselves are weak (rho +0.07 to +0.28, all p > 0.08). Sky attention is clearer
  (`all_sky_*` rho +0.33 to +0.41, p 0.01–0.04), but the plain share of non-architectural pixels is
  stronger still (`nonarch_tok_frac` rho +0.47, p 0.003; `sky_tok_frac` +0.39). So scenes with much
  sky/vegetation reconstruct worse, and the attention signal does not obviously add to just counting
  those pixels. Directionally consistent with the paper, not evidence for an attention mechanism.

### 2026-11-15: where does the noise come from? (code reading; probe started)

- **Code reading.** Scoring aligns the prediction to the ground truth before measuring Chamfer
  distance: random subsampling of correspondences (`np.random.choice`), then RANSAC Umeyama
  (`eval_utils.umeyama_ransac`, `seed=None`, so NumPy's global generator), then ICP (open3d). The VGGT
  pass never seeds NumPy (`evaluate.py` does, `np.random.seed(42)`, but `run.py` does not import it), and
  `run.py` runs all scenes in one process, so a scene's draw also depends on the scenes before it. The
  authors know: `intervention/aggregate.py` says baseline runs "differ only by the noise of the unseeded
  RANSAC" and therefore averages *before* over the sweep, while *after* is the best of the 5 runs per
  scene, which favours a lucky draw. (Not yet quantified; needs the probe's noise size.)
- **swati's setup (from Eric):** NVIDIA L4 24 GB on GCP g2-standard-8 (8 vCPU, 32 GB RAM), VGGT-1B fp32,
  50 views, `SKIP_BASELINE_PASS=1`. Same GPU, precision and settings as Eric's η=0 run and the probe, so
  hardware/precision does not explain differences. Only the CPU count differs (ours 4); ICP and KD-tree
  run on CPU, unlikely to matter at this size.
- **Probe** (`pass1_out/noise_probe/run_probe.sh`): η=0 on Dendera, Shuri_Castle, Aguinaldo_Shrine,
  Paul_Kruger_Church (the last three have the largest |η=1 − η=0| gaps: 29%, 20%, 14%), 5 rounds:
  seed 0 twice (is the pipeline deterministic given a seed?), seeds 1–3 (spread from alignment).
  Seeding via `pass1_signals/seeded_run.py`: wraps the pass's `main` at the moment `run.py` imports it
  (the pass reads its env config at import, so it must not be imported earlier) and calls
  `np.random.seed / random.seed / torch.manual_seed` before each scene. No repository code changed.
- **First numbers (round 1, seed 0):** Dendera 0.0394 (Eric's η=0 0.0386; swati η=1 0.0382): ~2%
  between two untouched runs. Shuri_Castle 0.486 (Eric's η=0 0.512, ~5%); swati's five ArchAttend runs
  are 0.595–0.751, all worse than both untouched runs. Noise or real harm: to be settled by seeds 1–3.

- **Seed-0 repeat (rounds 1 and 2): bit-identical.** Same Score to the last digit on all 4 scenes and
  byte-identical `pred_inter.glb`. Given a seed the whole pipeline, GPU forward included, is
  deterministic; all run-to-run noise comes from the unseeded random draws in scoring. A fixed seed
  removes it.
- **Two untouched (η=0) runs per scene, differing only in the draw** (ours seed 0 vs Eric's run):
  Dendera 0.0394 vs 0.0386 (2%), Shuri_Castle 0.486 vs 0.512 (5%), Aguinaldo_Shrine 0.0460 vs 0.0520
  (13%), Paul_Kruger_Church 0.228 vs 0.180 (27%). On Paul_Kruger all five of swati's η scores
  (0.192–0.227) lie between the two untouched runs. Shuri_Castle is the exception (swati 0.595–0.751,
  worse than both). Seeds 1–3 to give a proper spread.
- Disney anchor mismatch explained: SAM 3 finds almost no architecture there (arch 83 patches in 3/50
  views, tower 91 in 2, facade 397 in 2); SCVA ~1e-4 vs ~2e-5, so the 5x ratio is between noise-level
  numbers. Likely their masks found the tower in a few more views.

- **Seeds 1–3, trimmed to 2 scenes at Eric's request** (Dendera/Aguinaldo dropped mid-round; ~9 min per
  run). Five untouched (η=0) scores per scene, differing only in the scoring draw:

  | | Eric | seed 0 | seed 1 | seed 2 | seed 3 | range | sd/mean | swati η=1…5000 |
  |---|---|---|---|---|---|---|---|---|
  | Shuri_Castle | 0.512 | 0.486 | 0.660 | 0.671 | 0.537 | 0.49–0.67 (38%) | 15% | 0.595–0.751 (2/5 inside) |
  | Paul_Kruger_Church | 0.180 | 0.228 | 0.239 | 0.210 | 0.213 | 0.18–0.24 (33%) | 10% | 0.192–0.227 (5/5 inside) |

  On Shuri the bad draws double Comp-med (~0.54 → ~1.0): the alignment occasionally lands clearly
  wrong, not slightly off. Paul_Kruger: every ArchAttend score is inside the untouched spread, so its
  "best η" is indistinguishable from luck. Shuri: swati's mean (0.679) is ~1.2 sd above the untouched
  mean (0.573); at most weak evidence that ArchAttend hurts there, not the 16–47% it first looked like.
- **Conclusion:** the noise is not small. On these two hard scenes one scoring draw moves CD by 10–15%
  (sd), far above the ~1.7% median best-vs-runner-up η gap. Cause: unseeded alignment only (seeded runs
  are bit-identical). These two were picked for large gaps, so they overstate the typical scene;
  Dendera's two draws differed by 2%.

### 2026-11-15: noise parked; plan A+B written before running (step-risk and curve pooling)

**Noise thread paused, not solved.** Rescoring needs swati's `inter_raw.npz` (her sweep set
`SAVE_RAW_TENSORS=1`; not uploaded) and costs ~2.5–6 CPU-min per draw here. Parked by Eric's decision.
Working assumption from now on: per-scene differences of a few % are unreadable; large ones (η=5000
blow-ups of 3–6x) are real.

**Plan, fixed before looking at results.** Labels: swati's sweep, η ∈ {1, 10, 100, 1000, 5000}
(η=0 excluded: the choice is among intervention step sizes). Bar: best-fixed (η=1000 in every fold).

- *Features, chosen by hypothesis (5):* `anchor_view_frac`, `anchor_tok_frac` (anchor coverage: the
  Disney hypothesis, tiny anchor + big step = wreckage), `a_share_Lmean` (anchor dominance),
  `nonarch_share_Lmean` (attention on non-architecture), `ent_L4` (early attention spread).
  Honesty note: `ent_L4` and `nonarch_share` were already seen to correlate with log10 best η in the
  first test; the anchor-coverage features were not examined then.
- *A, step risk:* `risk_large = log(Score(5000) / min Score over η ≤ 1000)`. Prediction: negative
  Spearman with `anchor_view_frac` and `anchor_tok_frac`. Reported with the plain p-value (2 tests).
- *B, smooth target:* `large_vs_small` as before.
- *Selectors (LOSO, same harness):* `knn-curve k=5` and `k=10`: average the training regret curves of
  the k nearest scenes (standardized features, fit on training rows), pick the η with the lowest
  average. Averaging pools noise (B), and a neighbour's blow-up raises that η's average (A).
  Plus `curve-ridge` and `logeta-ridge` restricted to the 5 features.
- *Control:* permutation test, 200 shuffles of the feature rows across scenes, rerun `knn-curve`;
  p = share of shuffles with mean regret ≤ the real one. A selector counts as beating best-fixed only
  if its mean regret is lower **and** the permutation p < 0.05.
- *Also reported:* number of scenes with regret > 50% (catastrophes avoided or caused).

### 2026-11-15: plan A+B results (`pass1_signals/step_risk.py`, log `pass1_out/logs/step_risk.log`)

- **A, step risk: hypothesis confirmed.** Spearman(`anchor_view_frac`, risk_large) = −0.40 (p = 0.012);
  Spearman(`anchor_tok_frac`, risk_large) = −0.41 (p = 0.010). The 6 scenes where η=5000 fails worst
  (Santa Prisca 6.3x, Petra 3.9x, Kumano 3.4x, Disney 3.0x, Isa Khan 2.1x, Azadi 1.25x) have a median
  anchor of 0.7% of patches in 68% of views; the rest 8.0% in 98%. Small anchors break under big
  steps. Descriptive (not tested): `nonarch_share_Lmean` +0.40 and `ent_L4` +0.34 with risk_large.
- **B, selectors on the 5 features (bar: best-fixed, mean regret 4.38%):**

  | selector | mean regret | median | max | regret>50% | W/L vs best-fixed | perm. p |
  |---|---|---|---|---|---|---|
  | best-fixed (η=1000) | 4.38% | 1.92% | 21.7% | 0 | – | – |
  | **knn-curve k=10** | **4.20%** | **1.21%** | 21.7% | 0 | 10/6 | **0.025** |
  | knn-curve k=5 | 6.78% | 1.67% | 90.6% | 1 | 11/9 | 0.115 |
  | curve-ridge 5f | 6.54% | 1.67% | 90.6% | 1 | 7/5 | |
  | logeta-ridge 5f | 4.53% | 3.24% | 21.7% | 0 | 5/8 | |

  knn-curve k=10 meets the pre-set criterion (lower mean regret and permutation p < 0.05). It picks
  η=5000 on 12 scenes yet causes no blow-up (max regret equal to best-fixed), hits the oracle on 36%
  vs 18%. **But the margin is small:** 0.18 points of mean regret; W/L 10/6 has sign-test p = 0.45; the
  permutation test shows the features beat *random* neighbours (shuffled median 7.7%), not that the
  selector beats η=1000 decisively. k=5 fails, so the result depends on k (2 values tried).
- **Check without Disney** (the scene that inspired the hypothesis): `anchor_view_frac` ρ −0.35 (p 0.031),
  `anchor_tok_frac` ρ −0.36 (p 0.025). Weaker, still holds.
- **Reading:** the first Pass-1 signal with a pre-registered confirmation: anchor coverage predicts
  large-step failure. As a selector it lets us use η=5000 where it is safe, roughly tying η=1000 on
  mean regret with a better median. Promising, not yet decisive on 39 noisy scenes.

### 2026-11-15: step 2 plan, threshold rule (written before running)

- *Rule ("stump"):* one feature, one threshold. Inside each LOSO fold, for every candidate threshold
  (midpoints between sorted training values, at least 5 training scenes per side), give each side the
  η (any of the 5) with the lowest mean regret on that side's training scenes; keep the threshold with
  the lowest total training regret. The held-out scene gets its side's η.
- *Features:* `anchor_tok_frac` (primary), `anchor_view_frac` (secondary). No others.
- *Report:* same metrics and W/L vs best-fixed, permutation p (200 shuffles of the feature), and how
  stable the chosen threshold and the two sides' η are across folds.
- *Counts as better than best-fixed:* lower mean regret and permutation p < 0.05 (as before).

### 2026-11-15: step 2 result, threshold rule fails (`pass1_signals/threshold_rule.py`)

- Fitted on all 39 (reading only): use η=5000 above `anchor_tok_frac` 0.256 (9 scenes) or above
  `anchor_view_frac` 0.99 (13 scenes), else η=1000. Same direction as the step-risk finding.
- **LOSO:** stump on `anchor_tok_frac` mean regret 7.9% (W/L 5/9, perm. p 0.48); on `anchor_view_frac`
  7.1% (W/L 8/6, p 0.38). Both worse than best-fixed (4.38%), each with one blow-up: Isa Khan Niyazi's
  tomb (η=5000, regret 108%).
- **Why:** only ~6 scenes show the failure, so the threshold depends on which of them are in training.
  When Isa Khan (one of the 6) is held out, its fold's threshold drops and the rule sends it to 5000:
  holding a bad scene out removes the evidence that would have protected it. A one-feature cut is too
  brittle for 6 cases; knn-curve k=10 (5 features, averaging 10 neighbours' curves) is not.
- **Reading:** the step-risk *correlation* stands (it is a property of the data); a simple *rule* built
  on it does not generalize across folds. Not pursued further on these 39 scenes.

### 2026-11-15: report

- Wrote [`eta-from-pass1-report.md`](eta-from-pass1-report.md): the storyline version of this log for
  collaborators. New scenes (40 × η ∈ {1000, 5000}, ~14 GPU-hours) postponed by Eric.

### 2026-11-15: plan for held-out GPU runs (branch `eric/anchor-size-heldout`; written before running)

**Hypothesis under test** (found on dev, so it needs new scenes): *if the anchor is tiny, η=5000 is risky.*
On dev, with the outcome measured as r = Score(η=5000) / Score(η=1000): Spearman(`anchor_tok_frac`,
log r) = −0.47 (p = 0.003); tiny anchors (< 2% of patches) blow up (r > 1.5) in 4/8 scenes, others 1/31
(Fisher one-sided p = 0.004). The 2% and 1.5 cut-offs were chosen after looking at dev; they are fixed now.

**Scenes.** The 313 extracted scenes in neither `scenes_dev.txt` nor `scenes_test.txt` (test stays
untouched for a final evaluation). Screening order: random permutation, numpy seed 0.

**Design: screen cheaply, then run Pass 2 on a stratified set.**
1. *Screen* (Pass 1, anchor only): SAM 3 masks for the shipped anchor with the repo's
   `intervention/make_masks.py` (exactly the masks Pass 2 needs), then anchor coverage
   (`anchor_tok_frac`, same definition as `features.py`) by a CPU script. Stop when 24 tiny scenes are
   found or 120 scenes are screened.
2. *Select:* every tiny scene found (up to 24) + the same number of non-tiny scenes, the first ones
   in the screening order.
3. *Pass 2:* η = 1000 and 5000 on the selected scenes, via `pass1_signals/seeded_run.py` with seed 0
   (repeatable scoring; the same seed for both η). Output `pass1_out/heldout_pass2/`.

Power (simulated, Fisher one-sided, α = 0.05) if the effect is weaker than on dev (35% vs 5%):
8+32 random 55%, 20+20 69%, 24+24 80%. As strong as on dev: ≥ 92% for all.

**Pre-registered tests** on the new scenes:
- *Primary:* blow-up rate (r > 1.5), tiny vs non-tiny, Fisher exact, one-sided, p < 0.05.
- *Secondary:* Spearman(`anchor_tok_frac`, log r) < 0, one-sided, over all Pass-2 scenes (sample is
  stratified, so read the size of ρ with care).
- Reported regardless: rates with 95% intervals, every scene's r and anchor size.

**Step 0, before the long run (~20 min):** (a) on one dev scene, time anchor-only SAM 3 and check its
anchor mask equals the full-vocabulary one; (b) check the coverage script reproduces `anchor_tok_frac`
in `pass1_features.csv` for all 39 dev scenes (CPU).

**Cost estimate:** screening 100–120 scenes at ~1.5 min (to be measured in step 0) ≈ 2.5–3 h; Pass 2
48 scenes × 2 runs × ~8 min ≈ 13 h. Total ≈ 15–16 h for 24+24 (≈ 13 h for 20+20). GPU alone the whole
time; check every log for `Intervention token mask: N` with N > 0.

**Noise note:** single seeded run per (scene, η). Scoring noise is up to ~15% per run on hard scenes, so
a ratio r > 1.5 can occasionally come from luck; the blow-ups on dev were 2–6x.

### 2026-11-15: de-noising and confirmation run started (branch `eric/denoise-labels`)

Plan: `denoise-plan.md` (local, git-ignored). Scripts: `pass1_signals/{seeded_run,rescore,clean_analysis,confirm_analysis}.py`
and `pass1_signals/denoise_pipeline.sh` (detached; progress in `pass1_out/denoise/STATUS.md`).
`pass1_out/` and `results/` now live on the data disk (`/mnt/disks/archworld-data`), linked.

- **Phase 0, GPU:** seeded in-run scores at η=0 reproduce the noise probe's seed-0 values exactly
  (Paul Kruger 0.2283, Shuri Castle 0.4861, Dendera 0.0394). **Trim check passed:** a trimmed run (no
  GLB export, no in-run GT build) saves an `inter_raw.npz` identical array for array to the untrimmed
  one, in ~1.6 min per run instead of ~5–7.
- **First rescoring draw:** Paul Kruger 0.2232 (in the expected 0.180–0.239 range), but 428 s per draw:
  Chamfer 297 s, ICP 102 s, alignment 28 s.
- **Fix (exact):** KD-trees built unbalanced (`balanced_tree=False, compact_nodes=False`). On Paul Kruger
  (16.8M GT points, 1.76M predicted) the nearest-neighbour distances are identical, and the queries
  take 3.3 s instead of 42.6 s (accuracy) and 7.8 s instead of 66 s (completion). ICP (open3d, ~100 s)
  is left as is. Pipeline stopped and resumed at 22:24 (11 raw files kept, all checked readable).

- **Bug found and fixed: fp16 raw files.** Shuri Castle's rescored draws came out 0.88–0.91 against 0.486
  in-run (accuracy 0.037 vs 0.183, completion 1.79 vs 0.79): the rescored cloud was missing many points.
  Cause: the pass's `inter_raw.npz` stores depth confidence as fp16, which rounds every confidence in
  (1, ~1.0005] to exactly 1.0, and the scoring keeps a pixel only if its confidence exceeds ~1.00001.
  So low-confidence pixels vanish on rescoring (Shuri has many; Paul Kruger few, so its draws had
  looked plausible). Fix: `seeded_run.py` saves `inter_raw32.npz` (fp32 depth, confidence and cameras)
  from the exact tensors the pass hands to its GLB export; the rescorer and the driver use only that
  file. All raw files, draws and the trim check were deleted and Phase 0 restarted at 22:44.

### Open questions

- [x] swati's VGGT sweep results: `swati/results_vggt_dev/` (all 39 dev scenes).
- [x] Run-to-run noise: all from unseeded alignment (seeded runs bit-identical); sd 10–15% on two hard scenes, ~2% on Dendera. Typical size over all scenes still unknown.
- [x] A base arm on the same pipeline: Eric's η=0 run (commit 75f52b1).
- [ ] The paper's exact non-architectural vocabulary: asked Evan Zhang; using our own list meanwhile.
- [x] Which SCVA version reproduces `anchors.json`: the code definition (34/39; 4 of 5 misses are near-ties), RoPE irrelevant.
