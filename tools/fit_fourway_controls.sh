#!/usr/bin/env bash
# The two v2 controls, fitted off the GPU while the GPU trains something else.
#
# `rung0_real_run.py` refuses an in-process fit at this corpus size -- 39.8 hours projected
# against a 15-minute budget -- so without these the four-way arms would burn GPU time only
# to record `paired_margin_vs_linear: not_run`. The projection is the SPARSE cost at the
# runner's conservative in-process budget; this job owns the machine and passes a 24 GB dense
# budget, under which the same fit took 331s and 349s on the three-class corpus.
#
# Sequential, not parallel: two dense fits contend for BLAS threads and neither finishes
# sooner. Diff first, because that is the arm whose number is in question.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/fit_linear_control.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache

mkdir -p "$CACHE"

for SOURCE in diff after; do
  LOG=/home/ubuntu/fourway-control-"$SOURCE".log
  echo "######## v2 control, context-source=$SOURCE, started $(date -u +%FT%TZ) ########" > "$LOG"
  "$PY" -u "$TOOL" \
    --examples "$CORPUS"/examples.jsonl \
    --control-cache "$CACHE" \
    --context-source "$SOURCE" \
    --width 128 --heads 4 --layers 2 \
    --dense-budget-gb 24.0 \
    >> "$LOG" 2>&1
  echo "=== v2 control $SOURCE exited $? ===" >> "$LOG"
done
echo "BOTH V2 CONTROLS DONE"
