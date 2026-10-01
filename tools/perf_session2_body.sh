#!/bin/bash
# Body of perf session 2, run under the GPU lock by perf_session2.sh. See that file.
set -o pipefail
PERF=/home/ubuntu/perf
PY=/home/ubuntu/qd-venv/bin/python
BASE=/home/ubuntu/qd-lane4
OVER=$PERF/overlay
BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
V3=/home/ubuntu/phase4-v3-2026-10-01
P8K=/home/ubuntu/probe-8k-2026-09-30
LEDGER=$PERF/ledger-parity-2026-10-01.jsonl
RES=$PERF/parity.jsonl
N=6
export HF_HUB_OFFLINE=1
echo "=== lock held at $(date -u +%H:%M:%S)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

A=(--out $V3 --backbone $BACKBONE --batch-tokens 16384 --width-max 5383 --n-batches 50
   --passes 1 --deterministic --ledger $LEDGER --result $RES)
B=(--out $P8K --backbone $BACKBONE --batch-tokens 35403 --width-min 8001 --width-max 8441
   --n-batches 5 --allow-fewer --passes 10 --allow-stale-shards --deterministic
   --ledger $LEDGER --result $RES)
arm() { echo "=== arm $1 start $(date -u +%H:%M:%S)"; shift; $PY -u $PERF/perf_parity.py "$@" 2>&1 \
  | grep -v -i "warn" | tail -3; }

arm B-base   --code-root $BASE "${B[@]}" --tag B-base
arm B-skip$N --code-root $OVER "${B[@]}" --tag B-skip$N --checkpoint-skip-layers $N
arm A-base   --code-root $BASE "${A[@]}" --tag A-base
arm A-skip$N --code-root $OVER "${A[@]}" --tag A-skip$N --checkpoint-skip-layers $N
arm A-over0  --code-root $OVER "${A[@]}" --tag A-over0

# F's mode: default kernels. Interleaved A/B min-of-3 at 4 x 8,441, then peak memory at the
# 3,905 bucket (9 x 3,905 = 35,145 tokens, the probe set's nearest to 35,403).
echo "=== speed B default kernels start $(date -u +%H:%M:%S)"
$PY -u $PERF/perf_step.py --code-root $OVER --out $P8K --backbone $BACKBONE --shape B \
  --batch-tokens 35403 --width-min 8001 --width-max 8441 --allow-stale-shards --n-batches 5 \
  --warmup 2 --steps 8 --rounds 3 --budget-s 240 --results $PERF/bench.jsonl \
  --configs off:0 skip:$N 2>&1 | grep '"config"'
echo "=== memory B3905 default kernels start $(date -u +%H:%M:%S)"
$PY -u $PERF/perf_step.py --code-root $OVER --out $P8K --backbone $BACKBONE --shape B3905 \
  --batch-tokens 35403 --width-min 3905 --width-max 3905 --allow-stale-shards --n-batches 4 \
  --warmup 1 --steps 4 --budget-s 150 --results $PERF/bench.jsonl \
  --configs skip:$N skip:7 skip:8 2>&1 | grep '"config"'

arm A-base2  --code-root $BASE "${A[@]}" --tag A-base2
echo "=== compare $(date -u +%H:%M:%S)"
$PY $PERF/perf_parity.py --compare $RES:B-base $RES:B-skip$N
$PY $PERF/perf_parity.py --compare $RES:A-base $RES:A-skip$N $RES:A-over0 $RES:A-base2
$PY $PERF/perf_step.py --summarize $PERF/bench.jsonl
