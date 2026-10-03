#!/bin/bash
# Runs ON the 2x H100 box: experiment 3, the short-row end of exp 2's width curve (shape S,
# widths 512-2,999 at v5's 35,403 batch tokens). Throughput and memory only; v5 unchanged.
#   GPU0: unmodified tree at 8e6a009, skip:6 and skip:6+nomask, 3 interleaved rounds
#   GPU1: exp 2's cuDNN overlay (mask kept), skip:6, 3 rounds
# Usage: bash exp3_run.sh <timeout s>
set -u
T=$1
E=/home/ubuntu/exp-2026-10-03
C=$E/Lappi-decision
O=$E/overlay-cudnn
PY=/home/ubuntu/qd-venv/bin/python
BB=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$C" || exit 2
echo "=== exp3 at $(git rev-parse HEAD) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -e "$O" ] || { echo "no cuDNN overlay at $O"; exit 2; }
common=(--out "$E/data" --backbone "$BB" --shape S --batch-tokens 35403 --width-min 512 --width-max 2999)
( CUDA_VISIBLE_DEVICES=0 timeout --kill-after=60 "$T" "$PY" tools/perf_step.py --code-root "$C" "${common[@]}" \
    --configs skip:6 skip:6+nomask --rounds 3 --budget-s $((T - 120)) \
    --results "$E/exp3-S.jsonl" > "$E/exp3-S.log" 2>&1; echo "=== S exit $? $(date -u +%H:%M:%SZ)" >> "$E/exp3-S.log" ) &
( CUDA_VISIBLE_DEVICES=1 timeout --kill-after=60 "$T" "$PY" tools/perf_step.py --code-root "$O" "${common[@]}" \
    --configs skip:6 --rounds 3 --budget-s $((T - 120)) \
    --results "$E/exp3-S-cudnn.jsonl" > "$E/exp3-S-cudnn.log" 2>&1; echo "=== S cudnn exit $? $(date -u +%H:%M:%SZ)" >> "$E/exp3-S-cudnn.log" ) &
wait
for f in exp3-S exp3-S-cudnn; do
  "$PY" tools/perf_step.py --summarize "$E/$f.jsonl" > "$E/$f.summary.jsonl" 2>&1
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== exp3 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
