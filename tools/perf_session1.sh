#!/bin/bash
# Perf session 1 (GPU): profile the UNMODIFIED real step (qd-lane4, main 7ed66d7) at J4's shape
# (A: v3 shards, --batch-tokens 16384) and F's stand-in shape (B: J2's probe-8k repo-history set,
# batches 8,001-8,441 wide at --batch-tokens 35403). Deterministic mode (F's path) and default
# kernels (J4's path); checkpointing full / partial / off; fused AdamW and scoped torch.compile,
# all toggled in-process. Waits for j7g.done, then takes the GPU lock for at most 1,200 s and runs
# perf_session1_body.sh, which is read only once the lock is held. Writes only under
# /home/ubuntu/perf.
set -o pipefail
PERF=/home/ubuntu/perf
until [ -f /home/ubuntu/queue/j7g.done ]; do sleep 20; done
echo "=== j7g.done seen at $(date -u +%H:%M:%S); waiting for the GPU lock"
flock /home/ubuntu/queue/gpu.lock timeout 1200 bash $PERF/perf_session1_body.sh
echo "=== session 1 done (exit $?) at $(date -u +%H:%M:%S)"
touch $PERF/session1.done
