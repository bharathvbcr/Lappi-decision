#!/usr/bin/env bash
# The controlled pair: same corpus, same schedule, same capacity, one variable.
#
# The linear control says reading the diff instead of the post-image is worth +27.22
# points on rows where the empty-diff leak cannot operate (AUDIT/after-vs-diff-leak.md).
# That is a statement about the corpus, not about ByteDecider. These two arms ask whether
# the model can take the signal the control found.
#
# Both run on `commitpackft-mutated` -- the three mutated classes, 41,728 rows, no empty
# diff anywhere -- so the ONLY difference between them is `--context-source`. Running the
# diff arm against the existing full-corpus post-image rows instead would have differed in
# two ways at once and answered neither question.
#
# Capacity is 128x4x2: the capacity sweep found accuracy monotonically DECREASING in width
# on this task (128 -> 512 cost 3.9 points), so the narrowest arm is the strongest one and
# a wider arm would understate the model.
#
# Cost: the last 8-seed commitpackft arm took 277s and $0.115. Two arms is well under the
# $20 single-GPU threshold in CLAUDE.md rule 4, so no human sign-off is required -- and the
# wall-clock cap is passed anyway rather than relying on that.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CORPUS=/home/ubuntu/commitpackft-mutated
CACHE=/home/ubuntu/control-cache
LEDGER=/home/ubuntu/qwen-decision/ledger/gh200-context-source-2026-09-22.jsonl
STAMP=$(date -u +%FT%TZ)

# The mutation-engine revision the examples were generated at, which is what `rev` names
# and what the commitpackft arms of 2026-09-22 already carry. Not this checkout's HEAD:
# the corpus is supplied pre-generated, so the code that produced these rows is 0632f69,
# and `what_ran_state` -- not `rev` -- is what pins the trainer that reads them.
REV=0632f693d3b765b726499e7b4bf19c67959b75cb

mkdir -p "$CACHE" /home/ubuntu/ctxsrc-out

for SOURCE in after diff; do
  LOG=/home/ubuntu/ctxsrc-"$SOURCE".log
  echo "######## context-source=$SOURCE, mutated-only, n=8, started $STAMP ########" > "$LOG"
  "$PY" -u "$TOOL" \
    --examples "$CORPUS"/examples.jsonl \
    --manifest-in "$CORPUS"/manifest.json \
    --rev "$REV" \
    --context-source "$SOURCE" \
    --control-cache "$CACHE" \
    --out /home/ubuntu/ctxsrc-out \
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
  echo "=== $SOURCE exited $? ===" >> "$LOG"
done
echo "BOTH ARMS DONE"
