#!/bin/bash
# Post-F queue item 8 (campaign/f-j7prime-preregistered.json "no_mask"): the no-mask Tier-B P2
# screen, sessions T1 then T2, by the staged box script (tools/perf_nomask_p2.sh at main 669550d,
# on the box as /home/ubuntu/perf/perf_nomask_p2.sh). Each session takes
# /home/ubuntu/queue/gpu.lock itself under `timeout 1200`, then computes the P2 statistic on CPU
# outside the lock, so this waiter must not hold the lock. T2 runs whatever T1's exit was: the two
# shapes are separate screens, and perf_tierb_outcome.sh nomask --run (item 9) refuses unless both
# have a pass verdict.
# Queue: in the gap after item 3 (j7p.done: its rows exist), before the composed slice (item 4
# waits on this marker). Marker nomaskp2; the EXIT trap releases the chain whatever happens here.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/nomaskp2.done' EXIT
touch $Q/nomaskp2.queued
until [ -f $Q/j7p.done ]; do sleep 30; done
# Fable's P2 amendment (campaign/f-j7prime-preregistered.json "no_mask.p2_timing", 1de0a34): no
# P2 session runs before the amended rule is on main with its calibration .out and the box overlay
# is rebuilt at that commit, so that no retired-rule verdict row ever enters p2_gate's input. The
# lead touches $Q/nomask-p2-ruled once that holds; without it item 8 waits for a later gap (run by
# hand) and the chain moves on.
if [ ! -f $Q/nomask-p2-ruled ]; then
  say "item 8 not run: amended P2 rule not yet on the box (no $Q/nomask-p2-ruled); run T1/T2 by hand in a later gap"
  exit 0
fi
touch $Q/nomaskp2.started
for S in T1 T2; do
  say "item 8: no-mask P2 screen $S"
  bash /home/ubuntu/perf/perf_nomask_p2.sh "$S" 2>&1
  say "item 8: P2 $S exit $? (2 usage, 3 refused before the GPU: no overlay, dirty overlay or an existing result file)"
done
say "item 8 all done"
