#!/usr/bin/env bash
# Fit the char-n-gram control on each holdout arm's OWN training set.
#
# Without this every holdout and size-matched row carries paired_margin_vs_linear=not_run:
# the arm trains on a reduced set, and the only cached fits are over the full corpus, so the
# key the arm computes has no entry. A control fitted on data the arm never saw would not be
# its opponent even if it were used.
#
# This is the comparison the whole operator-holdout experiment turns on. The control scores
# 99.10% on stub.panic with it in training and 3.08% with it held out; the model's first
# seeds read 81.97% and 15.84%. Those pairs come from different tools on different splits
# and cannot be subtracted. Fitting here puts both on the same training set and the same
# validation rows, and if the model's smaller collapse survives that, it is the first
# configuration in this project where rung 0 beats the control.
#
# CPU only, off the GPU, and cheap: the full-corpus four-way fit took 378.9s on this box.
# Nine fits at that scale is under an hour of CPU with the GPU free to run arms.
#
# Run AFTER the arms finish: it costs no GPU, but `sync_box.sh` refuses while training is
# live and the box must be carrying this script's own filter_train_rows before it can fit
# the reduced sets.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/fit_linear_control.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache

# Matched to launch_operator_holdout_model.sh. The control is a property of the corpus, the
# split and the training set -- not of the model -- but the SPLIT depends on --val-share and
# the scored order depends on --batch-size, so those must match the arms or the key differs.
COMMON=(
  --examples "$CORPUS"/examples.jsonl
  --control-cache "$CACHE"
  --context-source diff
  --width 256 --heads 4 --layers 2
  --batch-size 16
  --dense-budget-gb 24.0
)

fit() {
  local label="$1"; shift
  local log=/home/ubuntu/opctl-"$label".log
  echo "######## $label started $(date -u +%FT%TZ) ########" > "$log"
  timeout --signal=TERM --kill-after=120 3600 \
    "$PY" -u "$TOOL" "${COMMON[@]}" "$@" >> "$log" 2>&1
  echo "=== $label exited $? ===" >> "$log"
  tail -2 "$log"
}

# The reference arm's control is the full-corpus fit every diff arm already shares; it is
# listed so a cache hit is confirmed rather than assumed, and so the three arms of each
# operator are visibly fitted the same way.
fit full

for OP in stub.panic logic.change_constant cosmetic.rename_local; do
  fit "holdout-$OP" --hold-out-operator "$OP"

  # N from the ARM's own log, exactly as the launcher took it, so the control's training set
  # is the arm's training set rather than a number retyped here.
  HOLD_LOG=/home/ubuntu/ophold-holdout-"$OP".log
  MATCHES=$(grep -cE "^  holding out ${OP}: [0-9]+ of [0-9]+ training row" "$HOLD_LOG")
  if [ "$MATCHES" != "1" ]; then
    echo "!! $OP: no single row-count line in $HOLD_LOG (found $MATCHES); skipping the"
    echo "!! size-matched control rather than fitting one against a guessed N, which would"
    echo "!! key to a training set no arm ever trained on and hit for nothing."
    continue
  fi
  N=$(grep -E "^  holding out ${OP}: [0-9]+ of [0-9]+ training row" "$HOLD_LOG" \
      | sed -E 's/^  holding out .*: ([0-9]+) of [0-9]+ training row.*/\1/')
  fit "sizematch-$OP" --drop-random-train "$N" --drop-random-seed 0
done

echo "OPERATOR HOLDOUT CONTROLS DONE"
