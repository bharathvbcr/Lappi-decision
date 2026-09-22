#!/usr/bin/env bash
# Does capacity help once the task has signal in it?
#
# `b72aa54` found accuracy MONOTONICALLY DECREASING in width on rung 0: 128 -> 512 cost 3.9
# points across 24 seeds. That is a strange result, and it was measured on the post-image
# task -- the one we now know carries almost no evidence that a change occurred. A model
# given more capacity to fit noise fitting more noise is not a statement about the
# architecture.
#
# On `--context-source diff` the same control goes from 57.8% to 88.5%, so the signal is
# there. This re-takes the capacity question on the task that has it. Three widths, 8 seeds
# each, everything else pinned: same corpus, same schedule, same span weight, same cached
# control, so width is the only thing that moves.
#
# Two outcomes and both are worth having. If accuracy still falls with width, the earlier
# finding was about the architecture and not about the corpus, and a larger cluster buys
# nothing. If it rises, the earlier finding was an artifact of a degenerate task and the
# capacity question is open again -- which is the one thing measured so far that would
# argue for more hardware.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache
LEDGER=/home/ubuntu/qwen-decision/ledger/gh200-diff-capacity-2026-09-22.jsonl
REV=0632f693d3b765b726499e7b4bf19c67959b75cb

mkdir -p /home/ubuntu/diffcap-out

# 128 is already measured at 8 seeds in gh200-fourway-2026-09-22.jsonl (75.93% mean, margin
# -0.1258). Re-run here anyway: pooling across ledger files is where a reader silently mixes
# two populations, and one file that holds the whole sweep cannot be mixed by accident.
for WIDTH in 128 256 512; do
  LOG=/home/ubuntu/diffcap-w"$WIDTH".log
  echo "######## diff capacity, width=$WIDTH, n=8, started $(date -u +%FT%TZ) ########" > "$LOG"
  "$PY" -u "$TOOL" \
    --examples "$CORPUS"/examples.jsonl \
    --manifest-in "$CORPUS"/manifest.json \
    --rev "$REV" \
    --context-source diff \
    --control-cache "$CACHE" \
    --out /home/ubuntu/diffcap-out \
    --ledger "$LEDGER" \
    --device cuda \
    --instance lambda-1xGH200 \
    --usd-per-hour 1.49 \
    --seeds 8 \
    --epochs 3 \
    --batch-size 16 \
    --span-weight 0.05 \
    --width "$WIDTH" --heads 4 --layers 2 \
    >> "$LOG" 2>&1
  echo "=== width $WIDTH exited $? ===" >> "$LOG"
done
echo "DIFF CAPACITY SWEEP DONE"
