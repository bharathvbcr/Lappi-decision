#!/bin/bash
# v5 queue, one GPU lane of the Lambda 2x H100 box: box_q_v5.sh LANE (LANE 0 or 1). One canonical
# script serves both lanes; launch it twice, as box_q_v5.sh 0 and box_q_v5.sh 1.
# The scheduling rule (Fable, 2026-10-03; campaign/v5-preregistered.json hardware.lanes): each
# lane takes the earliest job in the human's order whose inputs are ready. The order is 11 jobs:
# v5 seeds 0-4 (seeds.v5; unconditional since the amendment a29bca1, so seeds34 is not read), the
# noul-weight arm x3 (seeds 0-2) iff v5nw.room is room or V5NW_HUMAN_YES is written, then J5' x3
# (seeds 0-2). The inputs: v5 seed 0 none; v5 seeds 1-4 only V5_CONTINUE or R9's continue, and
# only when R9 is kept (V5_R9); the arm v5nw.launch run, decided with v5nw.room from v5 seeds 0-2's
# eval rows; every J5' seed v5 seeds 0-2's three completed ft rows (j5prime.runs_iff verbatim,
# Fable's ruling via the lead, 2026-10-03), then its own seed's eval row (J5' trains on v5 seed
# s's batch order and reads no checkpoint). So J5' runs ahead of the arm when a lane would
# otherwise idle: that interleaving is a change to the human's order (answer 1 at 0b559bb), named
# in the amendment (hardware.lanes, projected_gpu_hours.slot) for the human's yes. R9 waived, the
# rule gives [s0|s1] [s2|s3] [s4|nw0] [nw1|nw2] [J5'0|J5'1] [J5'2|idle]; with no room, round 3 is
# [s4|J5'0]; R9 kept, seed 0 runs alone until R9 speaks, and R9's hold holds both lanes (nothing
# J5' or the arm needs can exist while seeds 1-2 are held). A decision whose inputs are done (the
# room, the arm's reading) is made, once, by the first lane to pick after that, before its pick;
# each decision word is written once to its file (v5r9.word, v5.paused, v5nw.room, v5nw.launch,
# v5nw.word) and read by whoever needs it.
# Lanes: lane N holds $Q/gpu<N>.lock (gpu0.lock or gpu1.lock) for every GPU step and runs it with
# CUDA_VISIBLE_DEVICES=N; a seed's post-seed waiter (box_q_v5traj.sh) takes the same lane's lock,
# handed over as before. Picks are made under the lanes' scheduling mutex $Q/v5.sched.lock, taken
# after the lane's lock. Each job is the GH200 block's per-seed form (v5_common.sh v5_job_seed and
# v5_job_j5), its estimate checked against the approved 11-run total beside what the other lane's
# running job may still spend (v5_budget_ok).
# Start: the human's $Q/V5_LAUNCH_YES (non-empty; bounded) and v5_verify; on this box there is no
# post-F chain to wait on. Then the memory probe (hardware.probe), once, on lane 0 under
# gpu0.lock, before any seed: tools/real_ft_run.py --probe-shapes 12 with v5's recipe
# (--epoch --no-memorise) and data argv, no scoring flags, cap 1,800 s, into its own probe ledger.
# A non-zero exit is retried once with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for that
# call only; a second fail writes $Q/v5.probe-failed and both lanes wait (bounded) for the human's
# $Q/V5_PROBE_YES. Lane 1 waits for the probe's result before its first pick.
# Markers: $Q/v5lane<N>.queued / .started / .done; per job $Q/v5job-<job>.claimed / .done /
# .spend. $Q/V5_STOP ends both lanes at their next pick or wait. A lane refuses to start over
# markers an earlier launch left (stale, or decided once); the lead clears them.
# The GH200's three chained waiters (box_q_v5s34.sh, box_q_v5nw.sh, box_q_v5j5.sh) are retired:
# their runs and readings are these lanes' jobs and decisions.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
# shellcheck source=v5_common.sh
source /home/ubuntu/post-f/v5_common.sh || exit 3
LANE=$1
case "$LANE" in
  0|1) ;;
  *) say "usage: box_q_v5.sh LANE (0 or 1, the box's GPU); got '$LANE'"; exit 2 ;;
esac
# Before the trap: a second copy of a lane, or a lane over an earlier launch's markers, must not
# touch the markers it found.
v5_lane_stale "$LANE" || exit 3
trap 'touch "$Q/v5lane$LANE.done"' EXIT
touch "$Q/v5lane$LANE.queued"
UNSET_PINS=$(v5_pins_unset)
if [ -n "$UNSET_PINS" ]; then
  say "v5 lane $LANE deferred: pins UNSET:$UNSET_PINS (fail-closed until the lead fills them); v5 NOT RUN"
  exit 3
fi
PIN_PROBLEMS=$(v5_pins_check)
if [ -n "$PIN_PROBLEMS" ]; then
  say "v5 lane $LANE deferred: pins malformed: $(echo "$PIN_PROBLEMS" | tr '\n' ';'); v5 NOT RUN"
  exit 3
fi
v5_split_check || { say "v5 lane $LANE deferred: V5_SPLIT refused; v5 NOT RUN"; exit 3; }
v5_recipe || { say "v5 lane $LANE deferred: no recipe; v5 NOT RUN"; exit 3; }
v5_lane_set "$LANE" || exit 3
v5_wait_marker "$V5_LAUNCH_YES" "the human's v5 launch yes" || { say "v5 lane $LANE: v5 NOT RUN"; exit 3; }
v5_verify || { say "v5 lane $LANE refused before gpu$LANE.lock; v5 NOT RUN"; exit 3; }
say "v5 lane $LANE on GPU $V5_GPU: recipe (C1=$V5_C1 C2a=$V5_C2A C2b=$V5_C2B lower=$V5_LOWER lr=$V5_LRSET): ${V5_RECIPE[*]}"
say "v5 lane $LANE data: ${V5_SPLIT[*]}"
v5_total_line

if [ "$LANE" = 0 ]; then
  for f in "$V5_LEDGER" "$V5NW_LEDGER" "$V5_PROBE_LEDGER"; do
    if [ -e "$f" ]; then say "v5 lane 0: $f exists; v5's rows go into a new ledger, refusing"; exit 3; fi
  done
  v5_lock || exit 3
  touch "$Q/v5lane0.started"
  v5_probe
  case $? in
    0) ;;
    1) v5_wait_marker "$V5_PROBE_YES" "the human's yes after the failed probe" || { say "v5 lane 0: the probe failed; v5 NOT RUN"; exit 3; } ;;
    *) say "v5 lane 0: the probe could not run; v5 NOT RUN"; exit 3 ;;
  esac
  exec 9>&-
else
  v5_wait_probe || { say "v5 lane 1: v5 NOT RUN"; exit 3; }
fi

v5_lane_loop
RC=$?
v5_total_line
if [ "$RC" = 0 ]; then say "v5 lane $LANE all done"; fi
exit "$RC"
