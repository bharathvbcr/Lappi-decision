#!/bin/bash
# Post-F queue item 9 (campaign/f-j7prime-preregistered.json "no_mask"): the no-mask Tier-B
# outcome run, by the staged box script (tools/perf_tierb_outcome.sh at main 669550d, on the box
# as /home/ubuntu/perf/perf_tierb_outcome.sh): `nomask --run`. That script refuses with exit 5,
# before taking the lock, unless both shapes' latest P2 verdict is pass, and takes
# /home/ubuntu/queue/gpu.lock itself for each GPU stage, so this waiter must not hold it.
# The P2 rule is being amended by Fable, so the run is admitted only when the marker
# $Q/nomask-p2-ruled exists (touched by the lead once the amended rule is on the box); it is
# read when item 9's turn comes, not at launch. Absent, item 9 is logged NOT RUN and the chain moves on: this
# waiter never blocks the queue (its EXIT trap touches nomask.done on every path).
# Queue: after J5' (j5p.done), and after J6(f) when item 6 took the early slot; J6(f) in the late
# slot, tierb (item 7) and J6(d) (item 10) wait on this marker.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/nomask.done' EXIT
touch $Q/nomask.queued
until [ -f $Q/j5p.done ]; do sleep 30; done
after_early_j6f "item 9"
if [ ! -f $Q/nomask-p2-ruled ]; then
  say "item 9 not run: P2 rule awaiting Fable (no $Q/nomask-p2-ruled)"
  exit 0
fi
touch $Q/nomask.started
say "item 9: $Q/nomask-p2-ruled present; no-mask Tier-B outcome run"
bash /home/ubuntu/perf/perf_tierb_outcome.sh nomask --run 2>&1
say "item 9: outcome run exit $? (5: the P2 gate refused it before the lock; 3/4: refused or no ft row)"
