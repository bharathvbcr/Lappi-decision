#!/bin/bash
# Perf session 2 (GPU): Tier-A parity for --checkpoint-skip-layers under --deterministic, through
# the real real_ft_run._train loop, then F's default-kernel speed and memory for the chosen N.
#   baseline = /home/ubuntu/qd-lane4 (main 7ed66d7, read only)
#   modified = /home/ubuntu/perf/overlay (the train-step branch, built by perf_mkoverlay.sh)
# Order is priority: shape B (F's) parity first, then A (span-carrying), then speed/memory, then
# the determinism floor. Every arm appends its own result line, so a timeout loses only what had
# not started. Takes the GPU lock for at most 1,200 s. Writes only under /home/ubuntu/perf.
set -o pipefail
PERF=/home/ubuntu/perf
flock /home/ubuntu/queue/gpu.lock timeout 1200 bash $PERF/perf_session2_body.sh
echo "=== session 2 done (exit $?) at $(date -u +%H:%M:%S)"
touch $PERF/session2.done
