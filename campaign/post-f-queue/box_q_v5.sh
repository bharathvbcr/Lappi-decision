#!/bin/bash
# v5 queue: v5 seeds 0, 1, 2 (campaign/v5-preregistered.json seeds.v5; the human's yes, answer 1
# at d24c865, on ~$137 / ~60 GPU-h). Appended after the post-F chain, never inserted:
# launch.projected_gpu_hours.slot says the v5 waiter waits on j5pp.done, then wait_queued j6a,
# then wait_queued j6g. Then it waits for the human's launch marker $Q/V5_LAUNCH_YES (non-empty,
# the human's words; bounded).
# Per seed, under gpu.lock, from the v5 lane (v5_common.sh):
#   - train + score (--score-val --needle --ood) with recipe.base + recipe.added (--min-lr 0,
#     --batch-order seed, --checkpoint-every 100000 --retain-tower-every 1000) and the pinned
#     conditionals, cap 32,400 s (+1,800 s timeout margin), with its cost line;
#   - F's needle control on its checkpoint, cap 5,400 s;
#   - then gpu.lock passes to the seed's own post-seed waiter v5traj-s<N> (trajectory-ood rows,
#     cap 3,600 s, then the CPU controls); its .done never gates the next seed.
# R9 (readings.R9_pause_after_seed_0): right after seed 0's train+score, qd-post-f-rules-v5
# v5-pause. continue: seeds 1-2 run. Anything else: HOLD, and $Q/v5.paused says why, as exactly
# one of pause, refused, unknown:<word> or no-seed-0-row (v5_common.sh v5_rule's V5_SAID); seeds
# 1-2 start only once the human writes $Q/V5_CONTINUE (bounded; $Q/V5_STOP ends the block).
# The hold acts after seed 0's needle control and trajectory hand-off, which are seed 0's own
# (already approved) spend.
# Every run's estimate is checked against the approved total first (v5_budget_ok).
# Downstream: box_q_v5s34.sh, box_q_v5nw.sh and box_q_v5j5.sh wait on v5.done. LAUNCH THEM
# FIRST: wait_queued skips a waiter whose .queued is absent.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
# shellcheck source=v5_common.sh
source /home/ubuntu/post-f/v5_common.sh || exit 3
trap 'touch /home/ubuntu/queue/v5.done' EXIT
touch $Q/v5.queued
UNSET_PINS=$(v5_pins_unset)
if [ -n "$UNSET_PINS" ]; then
  say "v5 deferred: pins UNSET:$UNSET_PINS (fail-closed until the lead fills them); v5 NOT RUN"
  exit 3
fi
PIN_PROBLEMS=$(v5_pins_check)
if [ -n "$PIN_PROBLEMS" ]; then
  say "v5 deferred: pins malformed: $(echo "$PIN_PROBLEMS" | tr '\n' ';'); v5 NOT RUN"
  exit 3
fi
v5_split_check || { say "v5 deferred: V5_SPLIT refused; v5 NOT RUN"; exit 3; }
v5_recipe || { say "v5 deferred: no recipe; v5 NOT RUN"; exit 3; }
for m in "$V5_PAUSED" "$V5_CONTINUE" "$V5_STOP"; do
  if [ -e "$m" ]; then say "v5: $m exists before v5 started (stale; the lead clears it); v5 NOT RUN"; exit 3; fi
done
until [ -f $Q/j5pp.done ]; do sleep 60; done
for n in j6a j6g; do
  if [ -f "$Q/$n.queued" ]; then say "v5: j5pp is done; $n is queued, so v5 waits for $n.done"
  else say "v5: j5pp is done; $n is NOT queued, so v5 does not wait for it"; fi
done
wait_queued j6a j6g
v5_wait_marker "$V5_LAUNCH_YES" "the human's v5 launch yes" || { say "v5 NOT RUN"; exit 3; }
v5_verify || { say "v5 refused before gpu.lock; v5 NOT RUN"; exit 3; }
if [ -e "$V5_LEDGER" ]; then say "v5: $V5_LEDGER exists; v5's rows go into a new ledger, refusing"; exit 3; fi
say "v5 recipe (C1=$V5_C1 C2a=$V5_C2A C2b=$V5_C2B lower=$V5_LOWER lr=$V5_LRSET): ${V5_RECIPE[*]}"
say "v5 data: ${V5_SPLIT[*]}"
v5_total_line

# --- seed 0, then R9 --------------------------------------------------------------------------
v5_lock
touch $Q/v5.started
if [ -e "$V5_LEDGER" ]; then say "v5: $V5_LEDGER appeared while waiting for the lock; refusing"; exit 3; fi
v5_train_seed v5 0 "v5 seed 0"
FT0=$V5_FT
# R9_WHY: continue, or the hold's reason as $Q/v5.paused carries it (Fable's ruling A): pause (the
# reading), refused (the binary refused or could not be read or run), unknown:<word> (it said a
# word R9 does not have), no-seed-0-row (seed 0 wrote no ft row, so v5-pause was not run).
if [ -n "$FT0" ]; then
  v5_rule v5-pause "continue pause" v5-pause --preregistration "$V5_PREREG" --ledger "$V5_LEDGER" --ft-row "0=$FT0"
  R9_WHY=$V5_SAID
else
  R9_WHY=no-seed-0-row
  say "R9: v5 seed 0 wrote no ft row; v5-pause NOT RUN; no-seed-0-row"
fi
v5_needle_control v5 0 "$FT0" "v5 seed 0"
v5_post_seed v5 0 "$FT0"
if [ "$(v5_pause_action "$R9_WHY")" = run ]; then
  say "R9: continue: seeds 1-2 start"
else
  write_atomic "$V5_PAUSED" "$R9_WHY" || exit 3
  say "R9: $R9_WHY: HOLD (v5-pause on ft row ${FT0:-none}). $V5_PAUSED says '$R9_WHY'; seeds 1-2 start only once the human writes $V5_CONTINUE ($V5_STOP ends v5 here)"
  v5_wait_marker "$V5_CONTINUE" "R9's hold" || { say "R9: no continue: v5 seeds 1-2 NOT RUN"; exit 3; }
fi

# --- seeds 1, 2 -------------------------------------------------------------------------------
for SEED in 1 2; do
  v5_lock
  v5_train_seed v5 "$SEED" "v5 seed $SEED"
  FT=$V5_FT
  v5_needle_control v5 "$SEED" "$FT" "v5 seed $SEED"
  v5_post_seed v5 "$SEED" "$FT"
done
v5_total_line
say "v5 all done (seeds 0-2); v5s34, v5nw and v5j5 read its rows next"
