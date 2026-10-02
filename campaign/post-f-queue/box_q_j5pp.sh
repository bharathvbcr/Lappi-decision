#!/bin/bash
# Idle queue j5pp (Fable's idle-queue ruling, Q1 item 1 "Then j5pp" and Q2; pre-registration
# campaign/f-successor-preregistered.json follow_on_j5pp): J5'', the shuffled_label control x3 on
# F', in box_q_j5p.sh's form. Without it F' is not a candidate.
# Runs only if fsucc wrote three completed F' ft rows (runs_iff):
#   - fsucc's word ($Q/fsucc.word) is fires:j6f or fires:j6dv4 (absent or any other word: skip);
#   - the three ids in F''s training logs pass `qd-post-f-rules ft-rows` on the F' ledger:
#     completed, quick false, tag epoch, the seed claimed, one recipe hash and data snapshot.
# Otherwise it logs that it was skipped and why, and touches .done.
# Per seed 0 1 2: TARGET = `qd-post-f-rules eval-row` (F''s epoch-score-val row of that seed);
# then the firing arm's recipe flags (idle_common.sh, the ones fsucc trained with) with
# --score-val --shuffled-label TARGET, cap 32,400 s ($20.61, the human's 2026-10-01 yes via
# approved()), rows into the F' ledger, from qd-lane8 at a502670. real_ft_run refuses before any
# tower loads a run whose planned recipe differs from the target's ft row in anything but the
# shuffle and the span weight. Caps 3 x 32,400 s = $61.83.
# Queue: after fsucc's GPU part (fsuccgpu.done, touched after F''s last needle control or on any
# exit of fsucc). Holds gpu.lock for all three seeds.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j5pp.done' EXIT
touch $Q/j5pp.queued
until [ -f $Q/fsuccgpu.done ]; do sleep 60; done
WORD=$(read_succ_word)
if ! fp_arm "$WORD"; then
  say "j5pp SKIPPED: fsucc's word is '${WORD:-absent or not one of: $SUCC_WORDS}', not fires:<arm>; follow_on_j5pp runs only on three completed F' ft rows"
  exit 0
fi
pin "$RULES" "$RULES_SHA256" || exit 3
IDS=()
for SEED in 0 1 2; do IDS+=("$(log_ft_row "$FP_OUT/train-s$SEED.log")"); done
if ! "$RULES" ft-rows --ledger "$FP_LEDGER" --ft-row "0=${IDS[0]}" --ft-row "1=${IDS[1]}" --ft-row "2=${IDS[2]}"; then
  say "j5pp SKIPPED: fsucc fired ($WORD) but F''s ft rows '${IDS[*]}' in $FP_LEDGER are not three completed F' ft rows (the binary's reason is above)"
  exit 0
fi
exec 9>$Q/gpu.lock
flock 9
touch $Q/j5pp.started
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
say "j5pp (J5'' on F' $FP_ARM) at $(git rev-parse --short HEAD): ${FP_RECIPE[*]} -> $FP_LEDGER"
for SEED in 0 1 2; do
  FT=${IDS[$SEED]}
  LOG=$FP_OUT/j5pp-s$SEED.log
  if [ -e "$LOG" ]; then say "J5'' seed $SEED: $LOG exists; refusing to run it twice"; continue; fi
  TARGET=$("$RULES" eval-row --ledger "$FP_LEDGER" --ft-row "$SEED=$FT") || {
    say "J5'' seed $SEED NOT RUN: F' seed $SEED has no single completed epoch-score-val eval row scored from ft row '${FT}' (the binary's reason is above)"
    continue
  }
  say "J5'' seed $SEED -> eval row $TARGET (ft row $FT)"
  timeout 34200 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" "${FP_RECIPE[@]}" --devices cuda --seeds $SEED \
    --score-val --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$FP_LEDGER" \
    --approved-by "$(approved "J5'' shuffled_label on F' ($FP_ARM) seed $SEED (idle queue j5pp)")" \
    --shuffled-label "$TARGET" \
    2>&1 | tee "$LOG"
  say "J5'' seed $SEED done (exit $?)"
done
say "j5pp all done"
