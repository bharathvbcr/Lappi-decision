#!/usr/bin/env bash
# Is width-512 short of nothing, or just over-stepping?
#
# The diff-task capacity sweep found accuracy falling monotonically in width -- 76.3% at
# 128, 70.9% at 256, 58.1% at 512 -- which reads as "more capacity is worse". But the seed
# spread went 5.1 -> 10.3 -> 29.7 points, and the width-512 arm held seeds at 37.2% and
# 66.9% side by side. A model that is too large for its data degrades smoothly; a model
# whose optimiser is over-stepping degrades erratically, which is what this looks like.
#
# The two stories make the same prediction about the MEAN and opposite predictions about
# what happens when the rate comes down. So: width 512, four rates, everything else pinned.
#
# If 512 recovers toward 128's 76.3% at a lower rate, "capacity hurts" was a statement about
# the LR schedule and the capacity question is still open -- which would be the first
# measured thing arguing for more hardware. If it does not, the finding stands on its own
# and the answer to the cluster question is settled.
#
# 3e-3 is the rate every arm so far used and is included as the control point, not assumed.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache
LEDGER=/home/ubuntu/qwen-decision/ledger/gh200-lr-sweep-2026-09-22.jsonl
REV=0632f693d3b765b726499e7b4bf19c67959b75cb

mkdir -p /home/ubuntu/lrsweep-out

for LR in 3e-3 1e-3 3e-4 1e-4; do
  LOG=/home/ubuntu/lrsweep-"$LR".log
  echo "######## width 512, lr=$LR, n=8, started $(date -u +%FT%TZ) ########" > "$LOG"
  "$PY" -u "$TOOL" \
    --examples "$CORPUS"/examples.jsonl \
    --manifest-in "$CORPUS"/manifest.json \
    --rev "$REV" \
    --context-source diff \
    --control-cache "$CACHE" \
    --out /home/ubuntu/lrsweep-out \
    --ledger "$LEDGER" \
    --device cuda \
    --instance lambda-1xGH200 \
    --usd-per-hour 1.49 \
    --seeds 8 \
    --epochs 3 \
    --batch-size 16 \
    --span-weight 0.05 \
    --lr "$LR" \
    --width 512 --heads 4 --layers 2 \
    >> "$LOG" 2>&1
  echo "=== lr $LR exited $? ===" >> "$LOG"
done
echo "LR SWEEP DONE"
