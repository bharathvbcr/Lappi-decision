#!/usr/bin/env bash
# Ask rung 0 the question AUDIT/operator-holdout.md already answered for the control.
#
# The char-n-gram control collapses when one mutation operator is removed from training --
# stub.panic 99.10% to 3.08%, logic.change_constant 91.12% to 19.63%, cosmetic.rename_local
# 58.71% to 0.00% -- so it classifies by recognising the generator rather than by reading
# the change. That audit's last section says, in as many words, that it measures the control
# and not the model, and that the same holdout against a trained rung 0 needs a GPU and has
# not been run. This runs it.
#
# Three arms per operator, differing in one thing each:
#
#   reference  --measure-operator OP           everything trained on; the operator's rows
#                                              scored separately. The number the other two
#                                              are read against.
#   holdout    --hold-out-operator OP          the operator's rows removed from TRAINING
#                                              ONLY, after the split, so they survive in
#                                              validation and there is something to ask.
#   sizematch  --drop-random-train N           N rows dropped at random instead, N taken
#              --measure-operator OP           from the holdout arm's own count. Without it
#                                              a collapse is explained as well by "the
#                                              training set shrank" -- stub.panic alone is
#                                              43% of the rows.
#
# All three score the IDENTICAL validation rows, because validation is never filtered and
# the split is a function of the corpus and --val-share alone.
#
# The confound the size-matched arm cannot reach is that every operator here produces
# exactly one class, so removing one also moves the class prior. Each run therefore also
# reports val_choice_top1_on_measured_operator_siblings: the same class from OTHER
# operators, which training did see and which sits under the identical shifted prior. That
# pair is what separates "the generator was the signal" from "the arm moved the prior".
#
# Config is the best cell of the tuned-capacity grid (AUDIT/capacity-and-learning-rate.md):
# width 256, 2 layers, 4 heads, lr 3e-4, --context-source diff, which read 80.97% top-1.
# Deliberately NOT retuned per arm: a holdout arm tuned separately would confound the
# effect with the tuning, and rule 2 makes these read-only anyway.
#
# Cost: 9 arms x 8 seeds, ~90s per seed on the GH200, so roughly 3.5-4 hours at $1.49/h --
# about $6, a single-GPU job under the $20 line in rule 4, which needs no human yes.
#
# The linear control CANNOT be fitted for the holdout and sizematch arms: their training
# sets differ from every cached fit and fit_linear_control.py has no holdout flags, so
# paired_margin_vs_linear is NotRun on those rows and must be read as not measured rather
# than as a loss. The reference arm shares the grid's training set, so its margin is real.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/rung0_real_run.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache
LEDGER=/home/ubuntu/qwen-decision/ledger/gh200-operator-holdout-model-2026-09-22.jsonl
OUT=/home/ubuntu/ophold-out
# Passed in, never hardcoded and never read from the box's own git. sync_box.sh rsyncs
# python/ and tools/ only, so the box HEAD can name a commit older than the files that will
# actually train -- a rev read there would be confidently wrong. The caller passes the local
# HEAD it just verified, and a missing one stops the launch rather than writing nine rows
# that cannot be traced to code.
REV="${1:?usage: launch_operator_holdout_model.sh <full-sha-of-the-synced-HEAD>}"
if [ "${#REV}" != "40" ]; then
  echo "refusing to launch: REV '$REV' is not a 40-character sha" >&2
  exit 2
fi
# Per-arm ceiling. An arm that has not finished in this long has stopped doing what it was
# launched to do, and the remaining arms matter more than waiting on it.
ARM_CAP=3600

mkdir -p "$OUT"

# The three the control experiment measured, so the model's numbers sit beside them.
OPERATORS="stub.panic logic.change_constant cosmetic.rename_local"

run_arm() {
  local label="$1"; shift
  local log=/home/ubuntu/ophold-"$label".log
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
  local rc=$?
  echo "=== $label exited $rc ===" >> "$log"
  echo "$label exited $rc"
  return $rc
}

for OP in $OPERATORS; do
  echo "======== $OP ========"

  run_arm "ref-$OP" --measure-operator "$OP"
  run_arm "holdout-$OP" --hold-out-operator "$OP"

  # N from the holdout arm's OWN count, not from a second implementation of the split that
  # could drift from it. A size-matched control that dropped the wrong number of rows would
  # look exactly like a correct one and would silently stop controlling for volume.
  HOLD_LOG=/home/ubuntu/ophold-holdout-"$OP".log
  MATCHES=$(grep -cE "^  holding out ${OP}: [0-9]+ of [0-9]+ training row" "$HOLD_LOG")
  if [ "$MATCHES" != "1" ]; then
    echo "!! $OP: expected exactly one row-count line in $HOLD_LOG, found $MATCHES."
    echo "!! Skipping the size-matched control rather than guessing N: an arm that drops"
    echo "!! the wrong number of rows is indistinguishable from one that drops the right"
    echo "!! number and controls for nothing."
    continue
  fi
  N=$(grep -E "^  holding out ${OP}: [0-9]+ of [0-9]+ training row" "$HOLD_LOG" \
      | sed -E 's/^  holding out .*: ([0-9]+) of [0-9]+ training row.*/\1/')
  echo "$OP: size-matching against $N dropped training row(s)"
  run_arm "sizematch-$OP" --drop-random-train "$N" --drop-random-seed 0 --measure-operator "$OP"
done

echo "OPERATOR HOLDOUT MODEL DONE"
