#!/usr/bin/env bash
# Capacity at a TUNED rate, which is the comparison nobody has made.
#
# Every capacity number this programme has recorded ran at lr 3e-3, and the LR sweep showed
# what that cost width 512: 59.82% at 3e-3 against 80.26% at 1e-4, with the margin improving
# 3.5x on the rate alone. Comparing widths at one rate compares each width against how well
# 3e-3 happens to suit it, which is not a capacity measurement.
#
# Two things here, in this order:
#
# 1. Does the rate keep paying below 1e-4? The sweep had not turned over at its lowest
#    point, so the optimum for width 512 may be lower still. Two more rates at that width.
# 2. Widths 128 and 256 at 1e-4, so the three widths can finally be compared at a rate that
#    suits the largest of them rather than the smallest.
#
# Only then is "does capacity help" a question this repository has an answer to. The 8-seed
# width-512 arm at 1e-4 already exists in gh200-lr-sweep-2026-09-22.jsonl and is not rerun;
# it is the third point of the comparison and carries the same recipe.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache
LEDGER=/home/ubuntu/qwen-decision/ledger/gh200-tuned-capacity-2026-09-22.jsonl
REV=0632f693d3b765b726499e7b4bf19c67959b75cb

mkdir -p /home/ubuntu/tunedcap-out

run() {
  local width="$1" lr="$2"
  local log=/home/ubuntu/tunedcap-w"$width"-lr"$lr".log
  echo "######## width=$width lr=$lr n=8 started $(date -u +%FT%TZ) ########" > "$log"
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
    --lr "$lr" \
    --width "$width" --heads 4 --layers 2 \
    >> "$log" 2>&1
  echo "=== width $width lr $lr exited $? ===" >> "$log"
}

# 1 -- has the rate stopped paying?
run 512 3e-5
run 512 1e-5

# 2 -- the other two widths at the rate that suits 512
run 128 1e-4
run 256 1e-4

echo "TUNED CAPACITY DONE"
