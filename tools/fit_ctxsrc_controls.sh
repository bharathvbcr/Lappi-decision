#!/usr/bin/env bash
# Fit both controls off the GPU, so the paired margin has an opponent.
#
# The in-process control refused itself: "fitting the linear control on 31042 training
# document(s) projects to 39.8 hours, over the 15 minute budget". That projection is the
# SPARSE cost at the runner's conservative in-process memory budget. This job owns the
# machine, so it passes a 24 GB dense budget -- 31042 x 65536 x 8 = 16.3 GB, under it --
# and the dense path is ~86x faster than the gather it replaces.
#
# Every flag that decides WHICH documents exist must match the arm's, or the cache key
# differs and the fit is silently useless. The defaults are imported from the runner
# (val-share, context-bytes, batch-size, max-iter); width/heads/layers are passed because
# the arm passes them, even though only max_context_bytes reaches the documents.
#
# Runs sequentially, not in parallel: two dense fits would contend for BLAS threads and
# neither would finish sooner.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/fit_linear_control.py
CORPUS=/home/ubuntu/commitpackft-mutated
CACHE=/home/ubuntu/control-cache

mkdir -p "$CACHE"

for SOURCE in diff after; do
  LOG=/home/ubuntu/ctxsrc-control-"$SOURCE".log
  echo "######## control fit, context-source=$SOURCE, started $(date -u +%FT%TZ) ########" > "$LOG"
  "$PY" -u "$TOOL" \
    --examples "$CORPUS"/examples.jsonl \
    --control-cache "$CACHE" \
    --context-source "$SOURCE" \
    --width 128 --heads 4 --layers 2 \
    --dense-budget-gb 24.0 \
    >> "$LOG" 2>&1
  echo "=== control $SOURCE exited $? ===" >> "$LOG"
done
echo "BOTH CONTROLS DONE"
