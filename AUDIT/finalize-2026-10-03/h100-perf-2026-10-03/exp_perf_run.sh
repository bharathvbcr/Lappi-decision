#!/bin/bash
# Runs ON the 2x H100 box (launched by exp_stage.sh's `run`, nohup setsid). The human's
# "meanwhile optimize the code and GPU and run experiments" (~23:10Z 2026-10-03); Fable's scope:
# tools/perf_step.py only (throughput and memory, NOT loss parity), both GPUs at once, a hard stop
# before the v5 launch. Nothing here changes v5: results inform v6.
#   GPU0: shape W (batch 35,403 tokens, widths 9,000-10,240: v5's new widest bins)
#   GPU1: shape M (batch 35,403 tokens, widths 3,000-5,383)
#   configs, interleaved over 2 rounds (min-of-2 by --summarize):
#     off:6 (~ v5's --checkpoint-skip-layers 6; perf_step spaces the layers evenly),
#     off:6+fused, off:10, off:14, off:all, off:6+profile (the fp32 GEMM owner)
# Usage: bash exp_perf_run.sh <timeout seconds> <budget seconds>
set -u
T=$1
BUDGET=$2
E=/home/ubuntu/exp-2026-10-03
C=$E/Lappi-decision
BB=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$C" || exit 2
echo "=== exp at $(git rev-parse HEAD) $(date -u +%Y-%m-%dT%H:%M:%SZ) timeout ${T}s budget ${BUDGET}s"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
run() {
  local gpu=$1 shape=$2 wmin=$3 wmax=$4
  CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=60 "$T" /home/ubuntu/qd-venv/bin/python tools/perf_step.py \
    --code-root "$C" --out "$E/data" --backbone "$BB" --shape "$shape" \
    --batch-tokens 35403 --width-min "$wmin" --width-max "$wmax" \
    --configs off:6 off:6+fused off:10 off:14 off:all off:6+profile --rounds 2 \
    --budget-s "$BUDGET" --results "$E/perf-$shape.jsonl" > "$E/perf-$shape.log" 2>&1
  echo "=== $shape exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$E/perf-$shape.log"
}
run 0 W 9000 10240 &
run 1 M 3000 5383 &
wait
for s in W M; do
  /home/ubuntu/qd-venv/bin/python tools/perf_step.py --summarize "$E/perf-$s.jsonl" > "$E/perf-$s.summary.jsonl" 2>&1
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== exp done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
