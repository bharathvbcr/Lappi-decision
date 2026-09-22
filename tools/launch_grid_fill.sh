#!/usr/bin/env bash
# Find each width's own optimum, so "capacity helps" rests on best-vs-best.
#
# The grid so far tests width 512 at six rates and widths 128 and 256 at two -- 3e-3 and
# 1e-4, which bracket the optimum rather than finding it. Best-of-tested currently reads
# 76.33% / 78.71% / 80.26% for 128 / 256 / 512, but 128's best is almost certainly at a rate
# between the two it has been given, so the capacity gain is an upper bound on a comparison
# that is not yet fair in 128's favour.
#
# Four arms fill it: the two intermediate rates for each of the two smaller widths. After
# this every width has four points on the same curve and the comparison is between three
# optima rather than between three arbitrary choices.
#
# This is the honest version of the claim the cluster question turns on. It is cheap --
# these are the two fastest widths -- and the alternative is reporting a capacity effect
# measured with the smallest model handicapped.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache
LEDGER=/home/ubuntu/qwen-decision/ledger/gh200-tuned-capacity-2026-09-22.jsonl
REV=0632f693d3b765b726499e7b4bf19c67959b75cb

mkdir -p /home/ubuntu/tunedcap-out

for WIDTH in 128 256; do
  for LR in 1e-3 3e-4; do
    LOG=/home/ubuntu/tunedcap-w"$WIDTH"-lr"$LR".log
    echo "######## width=$WIDTH lr=$LR n=8 started $(date -u +%FT%TZ) ########" > "$LOG"
    "$PY" -u "$TOOL" \
      --examples "$CORPUS"/examples.jsonl \
      --manifest-in "$CORPUS"/manifest.json \
      --rev "$REV" \
      --context-source diff \
      --control-cache "$CACHE" \
      --out /home/ubuntu/tunedcap-out \
      --ledger "$LEDGER" \
      --device cuda \
      --instance lambda-1xGH200 \
      --usd-per-hour 1.49 \
      --seeds 8 \
      --epochs 3 \
      --batch-size 16 \
      --span-weight 0.05 \
      --lr "$LR" \
      --width "$WIDTH" --heads 4 --layers 2 \
      >> "$LOG" 2>&1
    echo "=== width $WIDTH lr $LR exited $? ===" >> "$LOG"
  done
done
echo "GRID FILL DONE"
