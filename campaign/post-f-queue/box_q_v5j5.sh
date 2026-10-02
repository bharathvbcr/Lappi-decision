#!/bin/bash
# v5 queue: J5', the shuffled_label control x3 on v5's exact path (campaign/v5-preregistered.json
# j5prime; campaign/f-successor-preregistered.json follow_on_j5pp's form, box_q_j5pp.sh's script
# form). Without it v5 is not a candidate.
# Runs only if v5 wrote three completed ft rows (runs_iff): the ids in v5's training logs pass
# `qd-post-f-rules-v5 ft-rows` on v5's ledger (completed, quick false, tag epoch, the seed claimed,
# one recipe hash and data snapshot). Otherwise it logs that it was skipped and why.
# Per seed 0 1 2: TARGET = `qd-post-f-rules-v5 eval-row` (v5's epoch-score-val row of that seed);
# then v5's recipe flags as v5 ran them (V5_RECIPE: --batch-order seed, so J5' seed s trains on v5
# seed s's batch order) with --score-val --shuffled-label TARGET, cap 32,400 s, rows into v5's
# ledger. No checkpoint retention, no needle or OOD scoring. real_ft_run refuses before any tower
# loads a run whose planned recipe differs from the target's ft row in anything but the shuffle
# and the span weight.
# Waits on v5.done, then on v5s34 and v5nw if they are queued. Holds gpu.lock for all three seeds.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
# shellcheck source=v5_common.sh
source /home/ubuntu/post-f/v5_common.sh || exit 3
trap 'touch /home/ubuntu/queue/v5j5.done' EXIT
touch $Q/v5j5.queued
UNSET_PINS=$(v5_pins_unset)
if [ -n "$UNSET_PINS" ]; then say "v5j5 deferred: pins UNSET:$UNSET_PINS; J5' NOT RUN"; exit 3; fi
PIN_PROBLEMS=$(v5_pins_check)
if [ -n "$PIN_PROBLEMS" ]; then say "v5j5 deferred: pins malformed: $(echo "$PIN_PROBLEMS" | tr '\n' ';'); J5' NOT RUN"; exit 3; fi
v5_split_check || exit 3
v5_recipe || exit 3
until [ -f $Q/v5.done ]; do sleep 60; done
for n in v5s34 v5nw; do
  if [ -f "$Q/$n.queued" ]; then say "v5j5: v5 is done; $n is queued, so v5j5 waits for $n.done"; fi
done
wait_queued v5s34 v5nw
v5_verify || { say "v5j5 refused before gpu.lock; J5' NOT RUN"; exit 3; }
v5_ft_args v5 --ft-row 0 1 2 || exit 3
if ! v5_rules_call ft-rows --ledger "$V5_LEDGER" "${V5_FT_ARGS[@]}"; then
  say "v5j5 SKIPPED: v5's ft rows '${V5_FT_ARGS[*]}' in $V5_LEDGER are not three completed v5 ft rows (the binary's reason is above); runs_iff not met"
  exit 0
fi
v5_lock
touch $Q/v5j5.started
say "v5j5 (J5' on v5) at $(git rev-parse --short HEAD): ${V5_RECIPE[*]} -> $V5_LEDGER"
v5_total_line
for SEED in 0 1 2; do
  FT=$(log_ft_row "$V5_OUT/train-s$SEED.log")
  LOG=$V5_OUT/j5p-s$SEED.log
  if [ -e "$LOG" ]; then say "J5' seed $SEED: $LOG exists; refusing to run it twice"; continue; fi
  TARGET=$(v5_rules_call eval-row --ledger "$V5_LEDGER" --ft-row "$SEED=$FT") || {
    say "J5' seed $SEED NOT RUN: v5 seed $SEED has no single completed epoch-score-val eval row scored from ft row '${FT}' (the binary's reason is above)"
    continue
  }
  if ! echo "$TARGET" | grep -Eqx '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'; then
    say "J5' seed $SEED NOT RUN: eval-row printed '$(printf '%s' "$TARGET" | head -c 200)', not one row id"
    continue
  fi
  v5_budget_ok "$V5_EST_J5_USD" "$V5_EST_J5_H" "J5' seed $SEED" || continue
  say "J5' seed $SEED -> eval row $TARGET (ft row $FT)"
  v5_cost_line "J5' seed $SEED" "$V5_TRAIN_CAP_S" "~ ${V5_EST_J5_H} h, \$${V5_EST_J5_USD}"
  T0=$(date +%s)
  timeout $((V5_TRAIN_CAP_S + 1800)) "$PY" -u tools/real_ft_run.py "${V5_SPLIT[@]}" "${V5_RECIPE[@]}" \
    --devices cuda --seeds "$SEED" \
    --score-val --wall-clock-cap-s "$V5_TRAIN_CAP_S" "${COST[@]}" --ledger "$V5_LEDGER" \
    --approved-by "$(v5_approved "J5' shuffled_label on v5 seed $SEED (v5j5)")" \
    --shuffled-label "$TARGET" \
    2>&1 | tee "$LOG"
  RC=$?
  v5_spend_add "J5' seed $SEED" $(( $(date +%s) - T0 ))
  say "J5' seed $SEED done (exit $RC)"
done
v5_total_line
say "v5j5 all done; the v5 block STOPS here"
