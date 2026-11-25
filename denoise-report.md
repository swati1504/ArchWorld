# Clean labels and a confirmation test: which Pass-1 signals predict η?

Branch `eric/denoise-labels`, 2026-11-15 to 2026-11-25. VGGT-1B, NVIDIA L4, 4 CPU cores. Follows
[`eta-from-pass1-report.md`](eta-from-pass1-report.md); the full step-by-step record, including every
fix and amendment, is in [`eta-from-pass1-log.md`](eta-from-pass1-log.md).

## Summary

- **Goal.** Find the Pass-1 signals that best predict ArchAttend's step size η per scene, on labels
  free of scoring luck, and check them on scenes they were not chosen on.
- **Clean labels.** We rescored every prediction with fixed seeds, using 2–3 alignment draws per label
  instead of one unseeded draw: 39 dev scenes × 6 η, then 40 new scenes × 4 η (394 labels, 915
  draws). Along the way we found that the raw files the pass saves (`inter_raw.npz`, fp16) silently
  change the score when rescored, and saved full-precision copies instead.
- **ArchAttend's gain on VGGT is small.** On dev, no fixed η beats no intervention (η=0) reliably;
  η=1000 comes closest (median −0.6%). On the new scenes, η=1000 changes nothing (median +0.1%). The
  per-scene best η beats η=0 beyond noise on 14 of 39 dev and 9 of 40 new scenes.
- **On dev, Pass-1 signals looked strong.** The anchor's self-attention and the anchor's size predicted
  the best η and the risk of η=5000 (ρ ≈ 0.57–0.59, false-discovery q < 0.01). Attention entropy
  predicted how much the best η gains.
- **On the new scenes, most of it did not hold.** We wrote down 21 (signal, target) pairs before
  scoring any new scene; **2 of 21 were confirmed**, both saying the same thing: scenes whose anchor
  attends strongly to itself are scenes where the choice of η matters less (ρ ≈ −0.46 and −0.49).
  The headline dev signals for *which* η is best were near zero on the new scenes (ρ between −0.01 and
  +0.05).
- **Why, as far as we can tell.** The dev correlations were carried by a handful of catastrophic η=5000
  failures on tiny anchors. The new scenes had none of those: η=5000 is at worst 2.2× worse there (6×
  on dev), and none of their 4 tiny-anchor scenes blew up.
- **Answer to the research question, for VGGT:** no Pass-1 signal we found predicts the best η on
  unseen scenes. The one replicated signal (anchor self-attention) predicts *whether η matters*, not
  which η to use. Always using η=1000 remains the best rule we have.

## 1. Why clean labels

The previous report found that the score of a run depends on luck. To compare a prediction with the
ground truth, the pipeline aligns them using random subsampling and RANSAC from NumPy's global
generator, which `run.py` never seeds. Rescoring one untouched prediction moved its error by about 10–15%
on hard scenes and 2% on easy ones, while the typical gap between a scene's best and second-best η is
about 2%. Many "best η" labels were partly luck, and any signal found on them, including the anchor-size
finding, needed rechecking.

So the plan for this branch (written 2026-11-15, before any run) was:

1. regenerate every prediction once, saving the raw tensors;
2. score each one several times with fixed seeds and combine the draws into a clean label;
3. analyse the clean dev labels and, by fixed rules, write a short list of signals to test;
4. run 40 new scenes and test that list on them.

Eric asked that nothing be carried over from the noisy results: neither the η values to test nor the
signals to look at.

## 2. How the labels were made

**Predictions** (GPU). `pass1_signals/seeded_run.py` runs the authors' `intervention/run.py` unchanged,
wrapping it from outside: it seeds every scene, saves the raw predictions, and skips outputs that scoring
doesn't need (prediction GLB, in-run ground truth). We checked that the skipped steps don't change the
prediction: the arrays are identical with and without them. About 1.5 minutes per run.

**Scoring** (CPU). `pass1_signals/rescore.py` scores a saved prediction exactly as the pass does
(validity masks, ground truth by unprojection, masked correspondences, RANSAC with 1,000 iterations,
ICP, Chamfer distance), using the authors' functions, with a given seed. Same seed, same score, to the
last digit.

**Two problems found on the way:**

- **The pass's own raw file changes the score.** `inter_raw.npz` stores depth and confidence in fp16.
  Confidences slightly above 1 round to exactly 1.0, and the scoring keeps a pixel only if its
  confidence exceeds 1.00001, so rescoring from that file silently drops pixels. On Shuri Castle it gave
  0.91 instead of the in-run 0.49. We now save a full-precision copy (`inter_raw32.npz`) at the moment
  the pass has the tensors. Rescoring it with the in-run seed reproduces the in-run score exactly (Paul
  Kruger 0.22828, Shuri 0.48608, Dendera 0.03937). **This matters for anyone rescoring swati's saved
  `inter_raw.npz` files.**
- **Speed.** A draw first took 7 minutes, mostly in the nearest-neighbour search for the Chamfer distance.
  Building the search trees without balancing gives identical distances about 10× faster; a draw now
  takes about 2–2.5 minutes.

**How many draws** (phase 0, decided by a rule fixed beforehand). Eight draws on each of three scenes:
Paul Kruger 0.194–0.240, Shuri Castle 0.486–0.896, Dendera 0.0378–0.0394. Only one of the 24 draws was a
clear alignment failure (Shuri, 0.896), too few to learn to recognise failures from fit quality. So
the rule chose: **score with seeds 0 and 1; if they agree within 3%, the label is their mean;
otherwise add seed 2 and take the median**, which ignores a single failed draw. Two thirds of labels
needed 2 draws, one third 3.

**When two η count as tied** (amendment 1, agreed with Eric before any clean label existed). Each
scene's tie band is twice that scene's own label noise, estimated from its draws (clipped to 1–12.3%).
The original rule used the noisiest scene's noise for every scene, which would have made a 24.6% band
and declared almost everything tied on quiet scenes. The median band came out at 2%.

## 3. What the clean dev labels say (39 scenes)

**ArchAttend helps little.** Change in error against η=0, per scene ("tied" = within the scene's band):

| η | median change | better / tied / worse | Wilcoxon p |
|---|---|---|---|
| 1 | +0.3% | 1 / 36 / 2 | 0.27 |
| 10 | +0.1% | 3 / 31 / 5 | 0.91 |
| 100 | −0.0% | 3 / 31 / 5 | 0.59 |
| **1000** | **−0.6%** | 3 / 34 / 2 | 0.053 |
| 5000 | −0.2% (mean +29%) | 13 / 14 / 12 | 0.60 |
| best η per scene | −2.8% | better on 14 / 39 | |

- On 8 scenes every η > 0 is tied: η doesn't matter there.
- η=5000 is a coin flip: it is the scene's single clear best on 9 scenes, and blows up on others
  (Santa Prisca 6×, Theater of Petra 3.8×, Disney 2.6×, Kumano Hongu Taisha 2.5×).
- The noisy labels weren't far off on average: swati's best η falls outside the clean tied-best set on
  only 2 of 39 scenes. The noise mostly blurred the *size* of differences, and so the correlations.

**η values carried forward.** By the pre-set rule (amendment 2: each scene's win is shared among its
tied best η), E = {0, 100, 1000, 5000}; the best fixed η is 1000.

**Signals.** We correlated all 88 Pass-1 features with 7 targets (best η, preference for large over
small η, each η's regret, how much η matters, gain of the best η over η=0), 616 pairs, with
false-discovery correction. 39 pairs reached q < 0.05. The strongest:

| target | signal | ρ on dev |
|---|---|---|
| best η (higher = larger) | anchor self-attention (`a_self`, mean over layers; layers 4, 17, 23 similar) | +0.59 |
| best η | anchor size (`anchor_tok_frac`, share of patches) | +0.57 |
| regret of η=5000 | anchor self-attention, layer 17 | −0.59 |
| gain of best η over η=0 | mean attention entropy (`ent_Lmean`) | +0.55 |
| how much η matters | anchor self-attention, layer 17 | −0.39 |

`a_self` is the share of attention an anchor patch gives to other anchor patches, normalised by the
anchor's size. Large, self-attending anchors went with larger best η and with η=5000 being safe.

**Earlier noisy findings, rechecked on dev:** anchor size vs η=5000's failure held and strengthened
(ρ −0.51; tiny anchors under 2% of patches blew up in 4 of 8 cases vs 1 of 31). The earlier "entropy at
layer 4 predicts the best η" did not hold (ρ −0.22, p 0.18; noisy −0.47).

**Selectors.** Despite those correlations, no per-scene selector beat "always η=1000" in
leave-one-scene-out testing (mean regret 3.4%; 10-neighbour 3.8%, 5-neighbour 4.1%, ridge 5.1%). So the
chosen selector was best-fixed η=1000.

**Confirmation list.** The 3 strongest signals per target (21 pairs), with the direction seen on dev,
written to `pass1_out/heldout/confirm_list.json` before any new scene was scored.

## 4. The new scenes (40)

Drawn at random (seed 0) from the 313 extracted scenes in neither the dev nor the test list. Pass 1 was
run in full on each (masks, attention features); predictions at η ∈ {0, 100, 1000, 5000}; scored with
the same scheme and tie rule (median band 2.2%).

**Confirmation test** (one-sided Spearman in the dev direction, Holm-corrected over the 21 pairs;
confirmed = adjusted p < 0.05):

| target | signal | ρ dev | ρ new | confirmed |
|---|---|---|---|---|
| how much η matters | anchor self-attention, layer 23 | −0.35 | **−0.49** | **yes** (p 0.014) |
| how much η matters | anchor self-attention, layer 17 | −0.39 | **−0.46** | **yes** (p 0.032) |
| gain of best η over η=0 | mean attention entropy | +0.55 | +0.28 | no (raw p 0.04) |
| regret of η=5000 | anchor self-attention (3 variants) | −0.57 to −0.59 | −0.15 to −0.22 | no |
| best η | anchor self-attention (3 variants) | +0.58 to +0.59 | −0.00 to +0.05 | no |
| preference for large η | anchor self-attention (3 variants) | −0.59 to −0.60 | −0.04 to −0.11 | no |
| regret of η=100 / η=1000 | entropy, attention-sink signals (6 pairs) | ±0.34 to 0.53 | −0.15 to +0.16 | no |

The selector (always η=1000) had mean regret 5.5% on the new scenes and median 0.7%. Its one bad
scene is Palacio de Bellas Artes, where η=5000 halves the error (0.0135 vs 0.028 for every other η, both
draws agreeing): the biggest single effect on the new scenes is a *gain* from a large step.

**What the new scenes look like** (descriptive, not pre-registered):

| | dev (on E) | new |
|---|---|---|
| η=1000 vs η=0, median change | −0.6% | +0.1% |
| best η per scene beats η=0 beyond noise | 14 / 39 | 9 / 40 |
| η > 0 all tied (η doesn't matter) | 11 / 39 | 13 / 40 |
| η=5000, worst failure | 6.0× | 2.2× |
| scenes where η=5000 fails > 1.5× | 5 | 1 |
| tiny anchors (< 2% of patches) | 8, of which 4 blew up | 4, none blew up |
| anchor size vs η=5000's failure (ρ) | −0.51 (p 0.001) | −0.27 (p 0.10) |
| median anchor size | 5.6% of patches | 8.4% |

## 5. What this means

1. **The confirmed signal.** When the anchor's patches attend mostly to each other (high `a_self` in
   the late layers), the reconstruction barely changes with η. It held on dev and on new scenes with
   similar strength. Read physically: a self-contained anchor gives the step little to redistribute.
   It tells you when sweeping η is a waste, not which η to choose.
2. **The best-η signals did not generalise.** On dev, the correlations with the best η were driven by a
   few scenes where η=5000 is catastrophic, all with tiny anchors. The random draw of new scenes
   contained fewer tiny anchors and no catastrophes, and without those the relation disappears. The
   direction of anchor size vs η=5000's failure is the same on the new scenes (ρ −0.27), but weak and
   not significant. "Small anchor ⇒ avoid η=5000" may still be true as a safety rule for rare failures;
   40 random scenes can neither confirm nor refute it.
3. **For VGGT, the prize is small.** On a fresh random sample, the best fixed η does nothing measurable
   on average, and a perfect per-scene oracle improves 9 of 40 scenes beyond noise. Picking η per scene
   could help at most on those few scenes, and nothing from Pass 1 identifies which ones in advance.
4. **The paper's VGGT numbers likely overstate the gain.** With one unseeded draw per run and the best
   of five η taken as *after*, part of the improvement is the luckiest draw. Our clean labels put the
   attainable gain (best η per scene, over η ∈ {100, 1000, 5000}) at a median 2.2% on dev and 0.9% on
   new scenes.

**Limitations.** One model (VGGT). 79 scenes in all. The tie band rests on 2–3 draws per label. Power:
with 40 scenes, a true correlation of about 0.4 is detected about 75% of the time, so weaker real
signals could have been missed. Phase 6 tested only the 21 pre-listed pairs; the descriptive table
above is exploratory.

## 6. Possible next steps

| step | what it answers |
|---|---|
| Pool dev and new scenes (79) and re-rank signals, then test on a third set | a list chosen on more varied scenes; a one-sample discovery set was the weak point here |
| Run DA3 or π³ | η may matter more for those models, above the noise floor |
| Target catastrophes directly: oversample small-anchor scenes | whether "small anchor ⇒ avoid η=5000" holds as a safety rule |
| Tell swati about the fp16 rescoring problem | any rescoring of `inter_raw.npz` drops low-confidence pixels |

## Appendix: reproducing

Everything runs from one detached script; outputs go to `pass1_out/denoise/` (dev) and
`pass1_out/heldout/` (new scenes), both git-ignored. Progress: `pass1_out/denoise/STATUS.md`.

```bash
conda activate archworld
(setsid nohup bash pass1_signals/denoise_pipeline.sh >> pass1_out/denoise/pipeline.out 2>&1 < /dev/null &)
```

It is resumable: restarting skips finished predictions, draws and labels. Its parts:

| phase | script | output |
|---|---|---|
| 0, 1, 4b: predictions | `pass1_signals/seeded_run.py --seed 0 --trim --skip-existing-raw -- <run.py args>` | `raw/output_<eta>/<scene>/inter_raw32.npz` |
| 0, 2, 5: scoring | `pass1_signals/rescore.py {draws,label,collect,choose}` | `draws/`, `labels/`, `clean_labels.csv`, `scheme.json` |
| 3: dev analysis | `pass1_signals/clean_analysis.py` | `pass1_out/denoise/analysis.log`, `pass1_out/heldout/confirm_list.json` |
| 4a: new-scene Pass 1 | `make_masks_full.py`, `extract_features.py`, `features.py` | `pass1_out/heldout/heldout_features.csv` |
| 6: confirmation | `pass1_signals/confirm_analysis.py` | `pass1_out/heldout/confirmation.log` |

Time on this machine: predictions about 1.5 GPU-minutes per run (≈ 6 h for dev, 4 h for new scenes),
scoring about 5 CPU-minutes per label (≈ 19 h for dev, 13 h for new scenes), Pass 1 for the new scenes
about 4 GPU-hours.
