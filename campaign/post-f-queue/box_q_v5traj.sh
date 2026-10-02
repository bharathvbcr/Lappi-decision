#!/bin/bash
# v5 post-seed waiter <run>traj-s<N> (recipe.added[0].scoring: "a separate waiter per seed
# (v5traj-s<N>) after its eval row, holding gpu.lock, cap 3,600 s ... its .done never gates the
# next seed"). Started by its parent (box_q_v5.sh, box_q_v5s34.sh or box_q_v5nw.sh through
# v5_post_seed) after the seed's train+score and needle control, with the parent's gpu.lock
# handed to it:
#   usage: box_q_v5traj.sh RUN SEED FT_ROW    (RUN v5 | v5nw; SEED 0-4; FT_ROW the seed's ft row)
# 1. Under gpu.lock: the 180-case OOD suite on each retained snapshot
#    (<ckpt>/epoch-seed<N>-cuda-step<S>.json, --retain-tower-every 1000), one OOD-only
#    --score-checkpoint call each (--ood without --score-val: tag trajectory-ood,
#    metrics.checkpoint_step, report-only), into the run's ledger. The final step first, then
#    the rest ascending, all inside 3,600 s from taking the lock; what the cap leaves out is
#    logged NOT RUN. The last-3 average row is NOT RUN: ckpt_average.py --same-seed-trajectory
#    builds it but real_ft_run.py --score-checkpoint refuses to score it
#    (GAP-V5TRAIN-TRAJECTORY-AVERAGE-NOT-SCORABLE-2026-10-02); the reason is printed from the
#    lane's own ckpt_average.TRAJECTORY_NOT_SCORABLE.
# 2. gpu.lock released, then the CPU letter and option controls (box_f_controls.sh's form through
#    idle_common.sh controls_block) on v5's split, niced: report-only, overlapping the next seed.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
# shellcheck source=v5_common.sh
source /home/ubuntu/post-f/v5_common.sh || exit 3
RUN=$1 SEED=$2 FT=$3
case "$RUN:$SEED" in
  v5:[0-4]|v5nw:[0-2]) ;;
  *) say "usage: box_q_v5traj.sh RUN SEED FT_ROW (RUN v5 with SEED 0-4, or v5nw with SEED 0-2); got '$RUN' '$SEED'"; exit 2 ;;
esac
NAME=${RUN}traj-s$SEED
trap 'touch "$Q/$NAME.done"' EXIT
touch "$Q/$NAME.queued"
if ! echo "$FT" | grep -Eqx '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'; then
  say "$NAME: ft row '$FT' is not a row id; NOT RUN"; exit 3
fi
UNSET_PINS=$(v5_pins_unset)
if [ -n "$UNSET_PINS" ]; then say "$NAME deferred: pins UNSET:$UNSET_PINS; NOT RUN"; exit 3; fi
PIN_PROBLEMS=$(v5_pins_check)
if [ -n "$PIN_PROBLEMS" ]; then say "$NAME deferred: pins malformed: $(echo "$PIN_PROBLEMS" | tr '\n' ';'); NOT RUN"; exit 3; fi
v5_split_check || exit 3
pin "$V5_PREP_BIN" "$V5_PREP_BIN_SHA256" || exit 3
export QD_PREP_BIN=$V5_PREP_BIN
lane_full "$V5_LANE" "$V5_LANE_AT" || exit 3
pin "$V5_PREREG" "$V5_PREREG_SHA256" || exit 3
v5_read_prereg || exit 3
v5_run_vars "$RUN" || exit 3

# --- 1. the trajectory, under gpu.lock -------------------------------------------------------
v5_lock
touch "$Q/$NAME.started"
T0=$(date +%s)
STEPS=$(for f in "$V5R_CKPT/epoch-seed$SEED-cuda-step"*.json; do
          [ -f "$f" ] || continue
          s=${f##*-step}; s=${s%.json}
          echo "$s" | grep -Eqx '[0-9]+' && echo "$s"
        done | sort -n)
if [ -z "$STEPS" ]; then
  say "$NAME: no retained snapshot epoch-seed$SEED-cuda-step*.json in $V5R_CKPT; trajectory NOT RUN"
else
  FINAL=$(echo "$STEPS" | tail -1)
  ORDER="$FINAL $(echo "$STEPS" | grep -vx "$FINAL" | tr '\n' ' ')"
  say "$NAME: $(echo "$STEPS" | wc -l | tr -d ' ') snapshots (steps $(echo "$STEPS" | tr '\n' ' ')); scoring order $ORDER; cap ${V5_TRAJ_CAP_S}s"
  v5_cost_line "$NAME trajectory" "$V5_TRAJ_CAP_S" "(inside the seed's ~\$${V5_EST_SEED_USD})"
  SKIPPED=""
  for S in $ORDER; do
    LEFT=$(( V5_TRAJ_CAP_S - ($(date +%s) - T0) ))
    if [ "$LEFT" -lt "$V5_TRAJ_MIN_LEFT_S" ]; then SKIPPED="$SKIPPED $S"; continue; fi
    LOG=$V5R_OUT/traj-s$SEED-step$S.log
    if [ -e "$LOG" ]; then say "$NAME step $S: $LOG exists; refusing to score it twice"; continue; fi
    timeout "$LEFT" "$PY" -u tools/real_ft_run.py "${V5_SPLIT[@]}" --devices cuda --seeds "$SEED" \
      --ood --ood-general-record "$REC" \
      --score-checkpoint "$V5R_CKPT/epoch-seed$SEED-cuda-step$S.json" \
      --ft-ledger "$V5R_LEDGER" --ft-row-id "$FT" \
      "${COST[@]}" --wall-clock-cap-s "$LEFT" \
      --ledger "$V5R_LEDGER" --suite-verdicts-out "$V5R_OUT/traj-s$SEED-step$S.jsonl" \
      2>&1 | tee "$LOG"
    say "$NAME step $S done (exit $?)"
  done
  if [ -n "$SKIPPED" ]; then say "$NAME: the ${V5_TRAJ_CAP_S}s cap left steps$SKIPPED NOT RUN"; fi
fi
REASON=$("$PY" -c 'import sys; sys.path.insert(0, "tools"); import ckpt_average; print(ckpt_average.TRAJECTORY_NOT_SCORABLE)' 2>&1 | tail -1)
say "$NAME: last-3 average row NOT RUN (GAP-V5TRAIN-TRAJECTORY-AVERAGE-NOT-SCORABLE-2026-10-02): $REASON"
v5_spend_add "$NAME trajectory" $(( $(date +%s) - T0 ))
exec 9>&-
say "$NAME: gpu.lock released"

# --- 2. the CPU controls, outside the lock ---------------------------------------------------
v5_controls "$NAME ($RUN seed $SEED)" "$V5R_LEDGER" "$V5R_OUT/verdicts-s$SEED.jsonl" "$Q/$NAME.letter.started"
say "$NAME all done"
