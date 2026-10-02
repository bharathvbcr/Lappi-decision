#!/bin/bash
# Post-F queue item 0 (campaign/f-j7prime-preregistered.json): F's J7' CPU stage, from qd-lane10
# (main 3460afc, >= 600b241). CPU only, niced, no gpu.lock: the ruling runs it "parallel to 1-2",
# so it starts on f.done, not avgnp.done.
#   1. rule (ii) on F's three seeds decides the seed set. If it fires, J7' averages five and a
#      three-seed average is not built: this waits for item 2S (s34.done), then refuses unless
#      seeds 3 and 4 trained to completed, non-quick ft rows of F's recipe.
#   2. the fp32-masters average (ckpt_average --from masters, as box_j7.sh);
#   3. avg-np: base + lambda * mean(theta_i - base), lambda = sqrt(n / (1 + (n-1)c)), c measured
#      by the tool from F's own seeds (as box_q_avgnp.sh's (c)). Built whatever rule (i) says;
#      item 3 scores it only if J4's avg-np qualified;
#   4. the bf16 release export of the average (qd-export, as box_j7.sh), whose parity item 3
#      checks on the GPU. Then j7pavg.done: item 3 needs nothing below;
#   5. delta_cosine.py on F's masters (Fable I-2 diagnostic 2, as J4's AUDIT run);
#   6. calibration fits on F's per-seed verdicts, all and two-fold (as box_j7.sh step 3).
# Marker j7pcpu; j7pavg is touched after step 4 and, like j7pcpu.done, on every exit.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j7pavg.done /home/ubuntu/queue/j7pcpu.done' EXIT
touch $Q/j7pcpu.queued
until [ -f $Q/f.done ]; do sleep 30; done
touch $Q/j7pcpu.started
pin "$RULES" "$RULES_SHA256" || exit 3
lane "$LANE10" "$LANE10_AT" || exit 3
say "item 0 (J7' CPU) at $(git rev-parse --short HEAD)"

SEEDS=$(f_seeds item0)
if [ -z "$SEEDS" ]; then
  say "item 0: rule (ii) refused, so F's J7' seed set is undecided: no average, avg-np, export, delta cosine or calibration is built (escalate; nothing defaults to three)"
  exit 3
fi
if [ "$SEEDS" = "0 1 2 3 4" ]; then
  say "item 0: rule (ii) fired: J7' is over five seeds; waiting for item 2S (s34.done)"
  until [ -f $Q/s34.done ]; do sleep 60; done
fi
# shellcheck disable=SC2086 # $SEEDS is a list of seeds
IDS=$(f_ft_ids $SEEDS) || { say "item 0: F's ft rows for seeds $SEEDS are not one averageable configuration (the binary's reason is above); stopping"; exit 3; }
read -r -a IDA <<< "$IDS"
CKPTS=()
for s in $SEEDS; do CKPTS+=("$F_CKPT/epoch-seed$s-cuda.json"); done
AVG=$(avg_path "$SEEDS")
AVGNP=$(avgnp_path "$SEEDS")
RELEASE=/home/ubuntu/release/p4-v4-avg-masters-v1
mkdir -p "$F_CKPT/avg" /home/ubuntu/release /home/ubuntu/calib "$PF"
say "item 0: seeds $SEEDS, ft rows ${IDA[*]}"

say "2. masters average -> $AVG"
nice -n 10 "$PY" -u tools/ckpt_average.py --from masters --ft-row-ids "${IDA[@]}" --out "$AVG" "${CKPTS[@]}"
say "2. masters average done (exit $?)"

say "3. avg-np -> $AVGNP"
nice -n 10 "$PY" -u tools/ckpt_average.py --from masters --norm-preserving --base-snapshot "$BACKBONE" \
  --ft-row-ids "${IDA[@]}" --out "$AVGNP" "${CKPTS[@]}"
say "3. avg-np done (exit $?)"

if [ -f "$AVG" ] && [ ! -e "$RELEASE" ]; then
  say "4. release export -> $RELEASE"
  nice -n 10 /home/ubuntu/bin/qd-export --source "$AVG" --base-snapshot "$BACKBONE" \
    --tokenizer-sha256 "$TOKENIZER_SHA256" --expect-vocab-size 248320 --out "$RELEASE"
  say "4. export done (exit $?)"
elif [ -f "$AVG" ]; then
  say "4. export NOT RUN: $RELEASE already exists"
else
  say "4. export NOT RUN: no average at $AVG"
fi
touch $Q/j7pavg.done

if [ -f "$AVG.manifest.json" ] && pin "$DELTA_COSINE" "$DELTA_COSINE_SHA256"; then
  SIDECARS=()
  for s in $SEEDS; do
    mapfile -t found < <(ls "$F_CKPT"/epoch-seed"$s"-cuda.????????????????.safetensors 2>/dev/null)
    if [ "${#found[@]}" -ne 1 ]; then say "5. seed $s has ${#found[@]} sidecars, not 1"; SIDECARS=(); break; fi
    SIDECARS+=("${found[0]}")
  done
  if [ "${#SIDECARS[@]}" -gt 0 ]; then
    OUTJ=$PF/f-delta-cosine-seed$(echo "$SEEDS" | tr -d ' ').json
    say "5. delta cosine on F's masters -> $OUTJ"
    nice -n 10 "$PY" -u "$DELTA_COSINE" --avg-manifest "$AVG.manifest.json" \
      --base "$BACKBONE/model.safetensors-00001-of-00001.safetensors" "${SIDECARS[@]}" > "$OUTJ"
    say "5. delta cosine done (exit $?)"
  else
    say "5. delta cosine NOT RUN: a seed's sidecar is not unique"
  fi
else
  say "5. delta cosine NOT RUN: no average manifest, or $DELTA_COSINE is not the pinned copy"
fi

for s in $SEEDS; do
  V=$F_OUT/verdicts-s$s.jsonl
  if [ ! -f "$V" ]; then say "6. calibration seed $s NOT RUN: no $V"; continue; fi
  nice -n 10 "$PY" -u tools/calib_fit_row.py --bin /home/ubuntu/bin/qd-calib-fit \
    --verdicts "$V" --population all --name "p4-v4-s$s-all-v1" \
    --eval-ledger "$F_LEDGER" --ledger "$CALIB_LEDGER" --out-dir "/home/ubuntu/calib/p4-v4-s$s-all"
  say "6. calibration seed $s all (exit $?)"
  nice -n 10 "$PY" -u tools/calib_fit_row.py --bin /home/ubuntu/bin/qd-calib-fit \
    --verdicts "$V" --population two-fold --split-key j7-calib \
    --name "p4-v4-s$s-2fold-v1" --eval-ledger "$F_LEDGER" --ledger "$CALIB_LEDGER" \
    --out-dir "/home/ubuntu/calib/p4-v4-s$s-2fold"
  say "6. calibration seed $s two-fold (exit $?)"
done
say "item 0 all done"
