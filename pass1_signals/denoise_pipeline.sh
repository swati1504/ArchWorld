#!/bin/bash
# De-noising and confirmation pipeline (plan: denoise-plan.md). Two resumable loops:
#   GPU: phase 0 (raw eta=0 on 3 scenes + trim check), 1 (dev raw), 4a (new scenes' Pass 1), 4b (new scenes' raw at E)
#   CPU: phase 0 (draws, gate, scheme), 2 (dev labels), 3 (clean analysis + confirmation list),
#        5 (new-scene labels), 6 (confirmation)
# Start detached:   setsid nohup bash pass1_signals/denoise_pipeline.sh > pass1_out/denoise/pipeline.out 2>&1 &
# Progress:         cat pass1_out/denoise/STATUS.md
# Safe to rerun: every step skips work already done.
cd "$(dirname "$0")/.." || exit 1
REPO=$PWD
D=pass1_out/denoise; H=pass1_out/heldout
mkdir -p $D/logs $H/logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate archworld
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SKIP_BASELINE_PASS=1 SAVE_RAW_TENSORS=1
unset VGGT_HALF
ETAS_ALL="0 1 10 100 1000 5000"
P0_SCENES="Paul_Kruger_Church Shuri_Castle Dendera"

ev() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $D/events.log; status; }
status() {
  local tmp=$D/.STATUS.md.$BASHPID
  python - "$D" "$H" <<'EOF' > $tmp 2>/dev/null && mv $tmp $D/STATUS.md; rm -f $tmp
import sys, json, glob, os, shutil
D, H = sys.argv[1:3]
dev = [l.strip() for l in open("scenes_dev.txt") if l.strip()]
def n(p): return len(glob.glob(p))
etas = [0, 1, 10, 100, 1000, 5000]
held = [l.strip() for l in open(f"{H}/scenes.txt")] if os.path.exists(f"{H}/scenes.txt") else []
E = json.load(open(f"{H}/confirm_list.json"))["E"] if os.path.exists(f"{H}/confirm_list.json") else None
print("# Pipeline status\n")
for f, lab in [("PHASE0_GPU_DONE", "phase 0 GPU"), ("scheme.json", "phase 0 gate/scheme"), ("PHASE1_DONE", "phase 1 (dev raw)"),
               ("clean_labels.csv", "phase 2 (dev labels)"), ("PHASE4A_DONE", "phase 4a (new-scene Pass 1)")]:
    print(f"- {lab}: {'done' if os.path.exists(f'{D}/{f}') else 'not yet'}")
print(f"- phase 3 (confirmation list): {'done' if E is not None else 'not yet'}")
for f, lab in [("PHASE4B_DONE", "phase 4b (new-scene raw)"), ("clean_labels.csv", "phase 5 (new-scene labels)"), ("confirmation.log", "phase 6")]:
    print(f"- {lab}: {'done' if os.path.exists(f'{H}/{f}') else 'not yet'}")
print(f"\n- dev raw predictions: {n(D + '/raw/output_*/*/inter_raw.npz')} / {len(dev) * 6}")
print(f"- dev labels: {n(D + '/labels/*/eta*.json')} / {len(dev) * 6}")
if held:
    print(f"- new scenes: {len(held)}; features: {n(H + '/features/*.json')}; raw: {n(H + '/raw/output_*/*/inter_raw.npz')}"
          f"{' / ' + str(len(held) * len(E)) if E else ''}; labels: {n(H + '/labels/*/eta*.json')}")
if os.path.exists(f"{D}/scheme.json"):
    s = json.load(open(f"{D}/scheme.json"))
    fmt = lambda v, f: format(v, f) if isinstance(v, (int, float)) else "n/a"
    print(f"\n- gate: {s.get('gate')}; scheme: {s.get('scheme')}"
          f"{' (' + str(s.get('measure')) + ' cut-off ' + fmt(s.get('cutoff'), '.4g') + ')' if s.get('scheme') == 'F' else ''}; "
          f"noise floor {fmt(s.get('noise_floor'), '.2%')}; median draw {fmt(s.get('median_draw_s'), '.0f')}s")
free = shutil.disk_usage("/mnt/disks/archworld-data").free / 1e9
print(f"- data disk free: {free:.0f} GB")
print("\n## Events (latest last)\n")
print("".join(open(f"{D}/events.log").readlines()[-25:]) if os.path.exists(f"{D}/events.log") else "")
EOF
}
disk_wait() {
  while [ "$(df --output=avail -BG /mnt/disks/archworld-data | tail -1 | tr -dc 0-9)" -lt 50 ]; do
    ev "GPU paused: data disk below 50 GB free"; sleep 600; done
}
mask_check() {  # warn about zero token masks in a log
  grep -h "Intervention token mask: 0 " "$1" >/dev/null 2>&1 && ev "WARNING: zero token mask in $1"
  grep -c "Intervention token mask" "$1" 2>/dev/null
}

# ------------------------------------------------------------------------------------------- GPU loop
gpu_loop() {
  # Phase 0: untrimmed, in-run scoring on, on 3 scenes; then Dendera trimmed for the trim check
  if [ ! -f $D/PHASE0_GPU_DONE ]; then
    ev "phase 0 GPU: eta=0 raw on $P0_SCENES (untrimmed)"
    args=(); for s in $P0_SCENES; do args+=(--scene "$s"); done
    python pass1_signals/seeded_run.py --seed 0 --skip-existing-raw -- --model vggt --root data \
      --masks-dir pass1_out/masks "${args[@]}" --eta 0 --out $D/raw --work-dir $D/work >> $D/logs/phase0_gpu.log 2>&1
    mask_check $D/logs/phase0_gpu.log >/dev/null
    ev "phase 0 GPU: trim check (Dendera, trimmed)"
    SKIP_EVAL=1 python pass1_signals/seeded_run.py --seed 0 --trim --skip-existing-raw -- --model vggt --root data \
      --masks-dir pass1_out/masks --scene Dendera --eta 0 --out $D/trimcheck --work-dir $D/work >> $D/logs/phase0_trim.log 2>&1
    python - <<EOF >> $D/events.log 2>&1
import numpy as np
a = np.load("$D/raw/output_0/Dendera/inter_raw.npz"); b = np.load("$D/trimcheck/output_0/Dendera/inter_raw.npz")
same = all(np.array_equal(a[k], b[k]) for k in a.files)
open("$D/TRIM_OK" if same else "$D/TRIM_BAD", "w").write("")
print("trim check:", "identical arrays -> trimmed runs" if same else "DIFFERENT -> untrimmed runs")
EOF
    touch $D/PHASE0_GPU_DONE; ev "phase 0 GPU done"
  fi
  TRIM=""; [ -f $D/TRIM_OK ] && TRIM="--trim"

  # Phase 1: dev raw predictions, all 6 eta
  if [ ! -f $D/PHASE1_DONE ]; then
    for attempt in 1 2; do
      disk_wait
      ev "phase 1: dev raw, attempt $attempt ($TRIM)"
      SKIP_EVAL=1 python pass1_signals/seeded_run.py --seed 0 $TRIM --skip-existing-raw -- --model vggt --root data \
        --masks-dir pass1_out/masks --scenes-file scenes_dev.txt --eta 0,1,10,100,1000,5000 --out $D/raw \
        --work-dir $D/work >> $D/logs/phase1_attempt$attempt.log 2>&1
      mask_check $D/logs/phase1_attempt$attempt.log >/dev/null
      missing=$(python -c "
import os
dev=[l.strip() for l in open('scenes_dev.txt') if l.strip()]
m=[f'{s}@{e}' for s in dev for e in (0,1,10,100,1000,5000) if not os.path.exists(f'$D/raw/output_{e}/{s}/inter_raw.npz')]
print(' '.join(m))")
      [ -z "$missing" ] && break
      ev "phase 1: missing after attempt $attempt: $missing"
    done
    touch $D/PHASE1_DONE; ev "phase 1 done"
  fi

  # Phase 4a: new scenes, full Pass 1
  if [ ! -f $D/PHASE4A_DONE ]; then
    python - <<EOF
import json, os, numpy as np
v = json.load(open("intervention/assets/views.json")); a = json.load(open("intervention/assets/anchors.json"))
used = {l.strip() for f in ("scenes_dev.txt", "scenes_test.txt") for l in open(f) if l.strip()}
ok = sorted(s for s in v if s in a and s not in used
            and all(os.path.isdir(f"data/archworld/{s}/{d}") for d in ("images", "depth", "sparse")))
pick = [ok[i] for i in np.random.default_rng(0).permutation(len(ok))[:40]]
if not os.path.exists("$H/scenes.txt"):
    open("$H/scenes.txt", "w").write("\n".join(pick) + "\n")
print(len(ok), "candidates; 40 drawn")
EOF
    disk_wait
    ev "phase 4a: masks for 40 new scenes"
    conda run -n archworld-sam3 --no-capture-output python pass1_signals/make_masks_full.py --root data \
      --out $H/masks --scenes-file $H/scenes.txt --quiet >> $H/logs/masks.log 2>&1
    ev "phase 4a: Pass-1 features for new scenes"
    python pass1_signals/extract_features.py --root data --masks-dir $H/masks --out $H/features \
      --scenes-file $H/scenes.txt >> $H/logs/features.log 2>&1
    python pass1_signals/features.py --in $H/features --out $H/heldout_features.csv >> $H/logs/features.log 2>&1
    touch $D/PHASE4A_DONE; ev "phase 4a done: $(tail -1 $H/logs/features.log)"
  fi

  # Phase 4b: new scenes' raw predictions at E (all 6 eta if the gate failed and there is no list)
  if [ ! -f $H/PHASE4B_DONE ]; then
    while [ ! -f $H/confirm_list.json ] && [ ! -f $D/GATE_FAILED ]; do status; sleep 300; done
    if [ -f $H/confirm_list.json ]; then
      E=$(python -c "import json; print(','.join(str(e) for e in json.load(open('$H/confirm_list.json'))['E']))")
    else
      E=0,1,10,100,1000,5000
    fi
    for attempt in 1 2; do
      disk_wait
      ev "phase 4b: new-scene raw at eta $E, attempt $attempt"
      SKIP_EVAL=1 python pass1_signals/seeded_run.py --seed 0 $TRIM --skip-existing-raw -- --model vggt --root data \
        --masks-dir $H/masks --scenes-file $H/scenes.txt --eta $E --out $H/raw --work-dir $H/work \
        >> $H/logs/phase4b_attempt$attempt.log 2>&1
      mask_check $H/logs/phase4b_attempt$attempt.log >/dev/null
    done
    touch $H/PHASE4B_DONE; ev "phase 4b done"
  fi
  ev "GPU loop finished"
}

# ------------------------------------------------------------------------------------------- CPU loop
label_pending() {  # $1 raw root, $2 labels dir, $3 scenes file, $4 etas (space-separated), $5 draws dir
  local did=0
  while read -r s; do
    [ -z "$s" ] && continue
    for e in $4; do
      if [ -f "$1/output_$e/$s/inter_raw.npz" ] && [ ! -f "$2/$s/eta$e.json" ]; then
        mkdir -p "$2/$s"; att=$(( $(cat "$2/$s/.attempts" 2>/dev/null || echo 0) + 1 ))
        if [ $att -gt 3 ]; then
          [ -f "$2/$s/.gave_up" ] || { touch "$2/$s/.gave_up"; ev "rescoring $s: gave up after 3 crashed attempts"; }
          break
        fi
        echo $att > "$2/$s/.attempts"
        nice -n 10 python pass1_signals/rescore.py label --raw-root "$1" --draws "$5" --scene "$s" \
          --etas $4 --scheme $D/scheme.json --labels-dir "$2" >> "$2.log" 2>&1 < /dev/null
        did=1; status; break
      fi
    done
  done < "$3"
  return $((1 - did))
}

cpu_loop() {
  while [ ! -f $D/PHASE0_GPU_DONE ]; do sleep 60; done
  if [ ! -f $D/scheme.json ]; then
    ev "phase 0 CPU: 8 draws on each of $P0_SCENES"
    for s in $P0_SCENES; do
      rep=""; [ "$s" = Dendera ] && rep="--repeat-seed 0"
      nice -n 10 python pass1_signals/rescore.py draws --raw-root $D/raw --draws $D/draws --scene "$s" \
        --etas 0 --seeds 0 1 2 3 4 5 6 7 $rep >> $D/logs/phase0_cpu.log 2>&1
    done
    python pass1_signals/rescore.py choose --draws $D/draws --scenes $P0_SCENES --out $D/scheme.json >> $D/logs/phase0_cpu.log 2>&1
    gate=$(python -c "import json; print(json.load(open('$D/scheme.json'))['gate'])" 2>/dev/null || echo fail)
    ev "phase 0 CPU: gate $gate"
    if [ "$gate" != pass ]; then touch $D/GATE_FAILED; ev "GATE FAILED: no rescoring; GPU phases continue"; return; fi
  fi

  # Phase 2: dev labels as raw files appear
  while true; do
    label_pending $D/raw $D/labels scenes_dev.txt "$ETAS_ALL" $D/draws && continue
    if [ -f $D/PHASE1_DONE ]; then
      label_pending $D/raw $D/labels scenes_dev.txt "$ETAS_ALL" $D/draws || break
    else
      sleep 120
    fi
  done
  python pass1_signals/rescore.py collect --labels-dir $D/labels --out $D/clean_labels.csv >> $D/logs/collect.log 2>&1
  ev "phase 2 done: $(tail -1 $D/logs/collect.log)"

  # Phase 3
  while [ ! -f pass1_signals/clean_analysis.py ]; do ev "waiting for clean_analysis.py"; sleep 600; done
  if [ ! -f $H/confirm_list.json ]; then
    python pass1_signals/clean_analysis.py --labels $D/clean_labels.csv --scheme $D/scheme.json \
      --features pass1_out/pass1_features.csv --swati swati/results_vggt_dev/per_scene.csv \
      --list-out $H/confirm_list.json > $D/analysis.log 2>&1
    ev "phase 3 done (exit $?): confirmation list $( [ -f $H/confirm_list.json ] && echo written || echo MISSING)"
  fi
  [ -f $H/confirm_list.json ] || return

  # Phase 5: new-scene labels at E
  E=$(python -c "import json; print(' '.join(str(e) for e in json.load(open('$H/confirm_list.json'))['E']))")
  while [ ! -f $H/scenes.txt ]; do sleep 120; done
  while true; do
    label_pending $H/raw $H/labels $H/scenes.txt "$E" $H/draws && continue
    if [ -f $H/PHASE4B_DONE ]; then
      label_pending $H/raw $H/labels $H/scenes.txt "$E" $H/draws || break
    else
      sleep 120
    fi
  done
  python pass1_signals/rescore.py collect --labels-dir $H/labels --out $H/clean_labels.csv >> $H/logs/collect.log 2>&1
  ev "phase 5 done: $(tail -1 $H/logs/collect.log)"

  # Phase 6
  while [ ! -f pass1_signals/confirm_analysis.py ]; do ev "waiting for confirm_analysis.py"; sleep 600; done
  python pass1_signals/confirm_analysis.py --list $H/confirm_list.json --labels $H/clean_labels.csv \
    --features $H/heldout_features.csv --dev-labels $D/clean_labels.csv \
    --dev-features pass1_out/pass1_features.csv > $H/confirmation.log 2>&1
  ev "phase 6 done (exit $?): see $H/confirmation.log"
  touch $D/ALL_DONE
}

ev "pipeline start (pid $$)"
gpu_loop &
GPU_PID=$!
cpu_loop &
CPU_PID=$!
wait $GPU_PID $CPU_PID
ev "pipeline end"
