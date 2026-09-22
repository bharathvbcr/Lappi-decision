#!/usr/bin/env bash
# The four-way pair on the repaired corpus, plus the `after` re-run its own control missed.
#
# Three jobs, in this order, because each one's cost depends on the one before:
#
# 1. The `after` arm on `commitpackft-mutated` again. Its first run recorded
#    `paired_margin_vs_linear` as NOT RUN -- correctly: the control had not been fitted yet
#    and the in-process fit projects to 39.8 hours against a 15-minute budget. The cache is
#    warm now, so this run gets a real margin against the 69.3% control. The first run's rows
#    stay in the ledger; they are honest rows that say the gate did not run.
#
# 2. and 3. The four-way pair on `commitpackft-corpus-v2` -- 50,177 examples, every class,
#    no empty diff anywhere. This is the task rung 0 is actually specified to do, and it is
#    runnable for the first time: before `1b75f58` a clean example had no diff to read, so
#    `--context-source diff` on a four-class corpus was measuring the length of its input.
#
# Each arm fits its own control on a cache MISS, which at this corpus size is refused rather
# than attempted. So the v2 controls are fitted first, by fit_fourway_controls.sh, and this
# refuses to start until both cache files exist -- a run whose gate cannot report is a run
# that burns the GPU to say `not_run`.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CACHE=/home/ubuntu/control-cache
REV=0632f693d3b765b726499e7b4bf19c67959b75cb

run_arm() {
  local tag="$1" corpus="$2" source="$3" ledger="$4"
  local log=/home/ubuntu/fourway-"$tag".log
  echo "######## $tag  corpus=$corpus  context-source=$source  started $(date -u +%FT%TZ) ########" > "$log"
  "$PY" -u "$TOOL" \
    --examples "$corpus"/examples.jsonl \
    --manifest-in "$corpus"/manifest.json \
    --rev "$REV" \
    --context-source "$source" \
    --control-cache "$CACHE" \
    --out /home/ubuntu/fourway-out \
    --ledger "$ledger" \
    --device cuda \
    --instance lambda-1xGH200 \
    --usd-per-hour 1.49 \
    --seeds 8 \
    --epochs 3 \
    --batch-size 16 \
    --span-weight 0.05 \
    --width 128 --heads 4 --layers 2 \
    >> "$log" 2>&1
  echo "=== $tag exited $? ===" >> "$log"
}

mkdir -p /home/ubuntu/fourway-out

run_arm mutated-after-rerun /home/ubuntu/commitpackft-mutated after \
  /home/ubuntu/qwen-decision/ledger/gh200-context-source-2026-09-22.jsonl

run_arm v2-after /home/ubuntu/commitpackft-corpus-v2 after \
  /home/ubuntu/qwen-decision/ledger/gh200-fourway-2026-09-22.jsonl

run_arm v2-diff /home/ubuntu/commitpackft-corpus-v2 diff \
  /home/ubuntu/qwen-decision/ledger/gh200-fourway-2026-09-22.jsonl

echo "ALL THREE ARMS DONE"
