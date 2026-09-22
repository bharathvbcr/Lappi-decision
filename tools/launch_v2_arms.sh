#!/usr/bin/env bash
# The four-way pair on the repaired corpus, under the corrected span metric.
#
# An earlier attempt ran under `d0de223`, which decomposed the span START pointer over
# pointing rows and left the END pointer over every scored row. The five rows it wrote
# before it was stopped carry that pair -- `start 1.6% end 17.9%` -- and are superseded by
# these, not corrected: the ledger is append-only. `050980a` decomposes both.
#
# `commitpackft-corpus-v2` is the first four-class corpus on which `--context-source diff`
# means anything: before `1b75f58` a clean example carried an empty diff, so an empty
# context meant `clean` and nothing else.
#
# Both controls are already cached -- 57.8% for `after` over 12,770 rows, 88.5% for `diff`
# over 12,792 -- so neither arm spends GPU time waiting on a CPU fit that the runner would
# refuse anyway at 39.8 hours against its 15-minute budget.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache
LEDGER=/home/ubuntu/qwen-decision/ledger/gh200-fourway-2026-09-22.jsonl
REV=0632f693d3b765b726499e7b4bf19c67959b75cb

mkdir -p /home/ubuntu/fourway-out

for SOURCE in after diff; do
  LOG=/home/ubuntu/v2-"$SOURCE".log
  echo "######## v2 four-class, context-source=$SOURCE, n=8, started $(date -u +%FT%TZ) ########" > "$LOG"
  "$PY" -u "$TOOL" \
    --examples "$CORPUS"/examples.jsonl \
    --manifest-in "$CORPUS"/manifest.json \
    --rev "$REV" \
    --context-source "$SOURCE" \
    --control-cache "$CACHE" \
    --out /home/ubuntu/fourway-out \
    --ledger "$LEDGER" \
    --device cuda \
    --instance lambda-1xGH200 \
    --usd-per-hour 1.49 \
    --seeds 8 \
    --epochs 3 \
    --batch-size 16 \
    --span-weight 0.05 \
    --width 128 --heads 4 --layers 2 \
    >> "$LOG" 2>&1
  echo "=== v2 $SOURCE exited $? ===" >> "$LOG"
done
echo "V2 PAIR DONE"
