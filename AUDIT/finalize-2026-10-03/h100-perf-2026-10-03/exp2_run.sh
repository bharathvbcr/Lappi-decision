#!/bin/bash
# Runs ON the 2x H100 box: experiment 2 (after exp 1's 23:16Z profile put the padded attention's
# sm80 mem-efficient kernel at 51%/38% of CUDA time at shapes W/M). Throughput and memory only
# (perf_step), never loss parity; nothing here changes v5 (the human kept v5 as pre-registered).
#   per GPU, process 1, unmodified tree at 8e6a009, interleaved over 3 rounds:
#     skip:6 (the product rule = v5's --checkpoint-skip-layers 6), skip:6+nomask (SDPA flash,
#     the Tier-B lever), skip:6+fused, skip:6+nomask+fused
#   per GPU, process 2, a throwaway overlay whose padded attention is forced onto SDPA's cuDNN
#     backend (mask kept): skip:6 over 3 rounds
# Usage: bash exp2_run.sh <timeout s, process 1> <budget s, process 1> <timeout s, process 2>
set -u
T1=$1
B1=$2
T2=$3
E=/home/ubuntu/exp-2026-10-03
C=$E/Lappi-decision
O=$E/overlay-cudnn
PY=/home/ubuntu/qd-venv/bin/python
BB=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$C" || exit 2
echo "=== exp2 at $(git rev-parse HEAD) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ! -e "$O" ]; then
  git clone -q /home/ubuntu/suite-8e6a009/repo.bundle "$O" && (cd "$O" && git -c advice.detachedHead=false checkout -q "$(git -C "$C" rev-parse HEAD)") \
    && "$PY" "$E/exp2_patch_cudnn.py" "$O" || { echo "overlay build failed"; exit 3; }
fi
(cd "$O" && echo "overlay at $(git rev-parse HEAD), changed: $(git status --porcelain | tr '\n' ' ')")
run() {
  local gpu=$1 shape=$2 wmin=$3 wmax=$4
  local common=(--out "$E/data" --backbone "$BB" --shape "$shape" --batch-tokens 35403
                --width-min "$wmin" --width-max "$wmax")
  CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=60 "$T1" "$PY" tools/perf_step.py --code-root "$C" "${common[@]}" \
    --configs skip:6 skip:6+nomask skip:6+fused skip:6+nomask+fused --rounds 3 \
    --budget-s "$B1" --results "$E/exp2-$shape.jsonl" > "$E/exp2-$shape.log" 2>&1
  echo "=== $shape exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$E/exp2-$shape.log"
  CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=60 "$T2" "$PY" tools/perf_step.py --code-root "$O" "${common[@]}" \
    --configs skip:6 --rounds 3 --budget-s $((T2 - 120)) \
    --results "$E/exp2-$shape-cudnn.jsonl" > "$E/exp2-$shape-cudnn.log" 2>&1
  echo "=== $shape cudnn exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$E/exp2-$shape-cudnn.log"
}
run 0 W 9000 10240 &
run 1 M 3000 5383 &
wait
for f in exp2-W exp2-M exp2-W-cudnn exp2-M-cudnn; do
  "$PY" tools/perf_step.py --summarize "$E/$f.jsonl" > "$E/$f.summary.jsonl" 2>&1
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== exp2 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
