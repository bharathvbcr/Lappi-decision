#!/bin/bash
# Body of perf session 1, run under the GPU lock by perf_session1.sh. See that file.
set -o pipefail
PERF=/home/ubuntu/perf
PY=/home/ubuntu/qd-venv/bin/python
BASE=/home/ubuntu/qd-lane4
BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
RES=$PERF/session1.jsonl
export HF_HUB_OFFLINE=1
# What real_ft_run.py sets for --deterministic, before cuBLAS initialises. Exported for the whole
# process so the in-process +det toggle is exactly --deterministic; every row records it.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "=== lock held at $(date -u +%H:%M:%S)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
$PY -u $PERF/perf_step.py --code-root $BASE --out /home/ubuntu/phase4-v3-2026-10-01 \
  --backbone $BACKBONE --shape A --batch-tokens 16384 --width-max 5383 --n-batches 24 \
  --warmup 4 --steps 20 --budget-s 450 --results $RES \
  --configs off:0+profile+syncdebug off:0+det+profile off:0+det+fused off:12+det off:all+det \
            off:0+det+compile off:12+det+fused+compile
echo "=== shape A exit $? at $(date -u +%H:%M:%S)"
$PY -u $PERF/perf_step.py --code-root $BASE --out /home/ubuntu/probe-8k-2026-09-30 \
  --backbone $BACKBONE --shape B --batch-tokens 35403 --width-min 8001 --width-max 8441 \
  --allow-stale-shards --n-batches 24 --warmup 3 --steps 20 --budget-s 700 --results $RES \
  --configs off:0+det+profile+memhist+n50 off:0 off:6+det off:12+det off:all+det \
            off:0+det+fused off:0+det+compile
echo "=== shape B exit $? at $(date -u +%H:%M:%S)"
