#!/usr/bin/env bash
# Measure the newly supervised span head at real scale, against the null that makes it
# readable.
#
# Until 6e57512 there was no mode in which both heads were supervised: `after` mode has a
# choice task where 54% of the corpus shows no evidence a change happened, and `diff` mode
# gave the span head nothing, which is what "span NOT MEASURED (all N scored rows abstain)"
# has been reporting on every diff arm. --span-in-diff resolves the span into the diff's own
# byte space; 83456 of 83456 endpoints over both corpora land on the exact source line.
#
# Two arms, differing in one flag:
#
#   nospan   the current diff protocol, span head unsupervised -- the 120-plus rows already
#            in the ledger. Re-run here rather than cited, so both arms are the same eight
#            seeds on the same box on the same day.
#   span     --span-in-diff, span head supervised, span gradient reaching the shared trunk.
#
# Two questions, and the second is the one that decides whether the flag is worth using.
#
# 1. Does supervising the span cost the choice head? The gradient is shared, and
#    --span-weight 0.05 is a guess inherited from the arms that trained the head on nothing.
#    If choice top-1 falls, the flag buys a span at the expense of the task that matters.
#
# 2. Is the span number worth anything? On a 16-wide one-epoch smoke run the head reached
#    92.3% against a uniform-over-candidate-lines chance of 8.8% -- and 90.8% against a
#    policy that reads no code at all and points at a '+' line, because a unified diff marks
#    its own answer and both corpora average about 1.5 added lines per hunk. Every row here
#    records val_span_pointing_added_line_chance beside the two span gates, and the arm is
#    read against the LARGER of the two nulls.
#    GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD.
#
# Config is the grid's best cell (AUDIT/capacity-and-learning-rate.md): width 256, 2 layers,
# 4 heads, lr 3e-4. Cost: 2 arms x 8 seeds at ~92s, about 30 minutes, roughly $0.75 on one
# GH200 -- well under the $20 line in rule 4.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache
LEDGER=/home/ubuntu/qwen-decision/ledger/gh200-span-in-diff-2026-09-22.jsonl
OUT=/home/ubuntu/spandiff-out
ARM_CAP=3600

REV="${1:?usage: launch_span_in_diff.sh <full-sha-of-the-synced-HEAD>}"
if [ "${#REV}" != "40" ]; then
  echo "refusing to launch: REV '$REV' is not a 40-character sha" >&2
  exit 2
fi

mkdir -p "$OUT"

run_arm() {
  local label="$1"; shift
  local log=/home/ubuntu/spandiff-"$label".log
  echo "######## $label started $(date -u +%FT%TZ) ########" > "$log"
  timeout --signal=TERM --kill-after=120 "$ARM_CAP" \
    "$PY" -u "$TOOL" \
      --examples "$CORPUS"/examples.jsonl \
      --manifest-in "$CORPUS"/manifest.json \
      --rev "$REV" \
      --context-source diff \
      --control-cache "$CACHE" \
      --out "$OUT" \
      --ledger "$LEDGER" \
      --device cuda \
      --instance lambda-1xGH200 \
      --usd-per-hour 1.49 \
      --seeds 8 \
      --epochs 3 \
      --batch-size 16 \
      --span-weight 0.05 \
      --lr 3e-4 \
      --width 256 --heads 4 --layers 2 \
      "$@" \
      >> "$log" 2>&1
  echo "=== $label exited $? ===" >> "$log"
  echo "$label exited $?"
}

run_arm nospan
run_arm span --span-in-diff

echo "SPAN IN DIFF DONE"
