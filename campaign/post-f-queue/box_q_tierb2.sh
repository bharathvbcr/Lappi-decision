#!/bin/bash
# Post-F queue item 7 (campaign/f-j7prime-preregistered.json): tierb, the fused-AdamW Tier-B
# outcome run, requeued as filler after J5'. Its run call is box_q_tierb.sh's, unchanged:
# /home/ubuntu/perf/perf_tierb_fused.sh --run, which takes /home/ubuntu/queue/gpu.lock itself for
# each GPU stage (so this waiter must not hold it). The code it runs is /home/ubuntu/perf/p3fused,
# pinned here at f6a0928; perf_tierb_fused.sh refuses a dirty tree itself. Nothing in it enters a
# phase-5/6 run of this campaign (the ruling's tier_b_changes).
# Order: after J5' (j5p.done), and after J6(f) when item 6 took the early slot -- read from the
# position item 6 wrote ($Q/j6f.position), never from a second reading of rule (iii).
# Marker tierb2 (the old box_q_tierb.sh's EXIT trap owns tierb.done; kill that waiter first).
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/tierb2.done' EXIT
touch $Q/tierb2.queued
until [ -f $Q/j5p.done ]; do sleep 30; done
until [ -f $Q/j6f.position ] || [ -f $Q/j6f.done ]; do sleep 30; done
if [ "$(cat $Q/j6f.position 2>/dev/null)" = early ]; then
  say "item 7: J6(f) took the early slot; waiting for it (j6f.done)"
  until [ -f $Q/j6f.done ]; do sleep 30; done
fi
P3=/home/ubuntu/perf/p3fused
if [ "$(cd $P3 && git rev-parse --short=7 HEAD)" != f6a0928 ]; then
  say "item 7: $P3 is not at f6a0928; tierb NOT RUN"
  exit 3
fi
touch $Q/tierb2.started
say "item 7 (tierb) at $(cd $P3 && git rev-parse --short HEAD)"
bash /home/ubuntu/perf/perf_tierb_fused.sh --run
say "Tier-B fused run done (exit $?)"
