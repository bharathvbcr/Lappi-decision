#!/usr/bin/env bash
# Fit the char-n-gram control on each holdout arm's OWN training set.
#
# Without this every holdout and size-matched row carries paired_margin_vs_linear=not_run:
# the arm trains on a reduced set, and the only cached fits are over the full corpus, so the
# key the arm computes has no entry. A control fitted on data the arm never saw would not be
# its opponent even if it were used.
#
# This is the comparison the whole operator-holdout experiment turns on. The control scores
# 99.10% on stub.panic with it in training and 3.08% with it held out; the model's eight
# seeds read 81.97% and 15.44%. Those pairs come from different tools on different splits
# and cannot be subtracted. Fitting here puts both on the same training set and the same
# validation rows, and if the model's smaller collapse survives that, it is the first
# configuration in this project where rung 0 beats the control.
#
# ## What this costs, measured rather than estimated
#
# An earlier version of this header said the full-corpus fit took 378.9s and that "nine fits
# at that scale is under an hour of CPU with the GPU free to run arms". Both claims were
# wrong, and the script was built on them:
#
#   * Measured 2026-09-22, the `full` fit over 37,385 training documents scored on 12,792
#     rows projected 101.4 MINUTES, not 378.9s. Nine fits is therefore about 15 hours of
#     this box, roughly $22 at the GH200's hourly rate.
#   * The fit is NOT single-threaded, whatever `control_cache`'s docstring says. It is a
#     dense GEMM through BLAS: one fit drove load average to 64.8 on a 64-core box. So the
#     fits cannot be overlapped to buy wall clock -- they already own every core -- and they
#     are not free of the GPU either. A span arm's seed went from ~92s to 140.5s while one
#     fit ran beside it.
#
# Because the spend is real and crosses the $20 line rule 4 draws around a job needing a
# human yes, this script refuses to start without an explicit acknowledgement:
#
#     QD_CONTROL_FIT_ACK=1 bash tools/fit_operator_holdout_controls.sh
#
# ## Why the cache is cold at all
#
# Six fits sit in the cache and none of them can be served. `control_key` folds in a sha256
# of `python/qd_train/baseline.py`, deliberately, so that an edit to the control's
# implementation can never serve a verdict produced by code that no longer exists. Commit
# 20cc7f5 moved `context_texts` into that file. The move was a relocation, but the digest
# cannot see intent -- it sees different bytes, and it fails closed. That is the behaviour
# the digest is FOR, and the cost of rebuilding the cache is the price of it. Do not weaken
# the digest to reclaim the six entries.
#
# One consequence worth stating plainly, because it decides whether running this is worth
# anything: warming the cache does NOT retrofit rows already written. `rung0_real_run.py`
# reads the cache once, before its seed loop, and records not_run on a miss; it never fits
# inline. The 72 rows in gh200-operator-holdout-model-2026-09-22.jsonl are final. To get
# margins on them the arms must be RE-RUN after this completes -- another ~2.4 GPU-hours.
#
# ## The cap
#
# FIT_CAP_MIN feeds both the `timeout` and the tool's own --max-fit-minutes, so the two
# cannot drift. The tool refuses up front when its projection exceeds the cap, rather than
# being killed 60 minutes in with nothing cached -- which is what happened here, and what
# 5fd0ea8 ("The linear control could not finish inside the cap it was launched under")
# already recorded once before.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/fit_linear_control.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache

#: Above the 101.4-minute measured projection with headroom for a slower box or a larger
#: training set. Raise it if a projection legitimately exceeds it; never lower --max-iter
#: to fit inside it, which would weaken the opponent rather than afford the fit.
FIT_CAP_MIN=150
FIT_CAP_S=$(( FIT_CAP_MIN * 60 ))

if [ "${QD_CONTROL_FIT_ACK:-0}" != "1" ]; then
  echo "refusing to start: this queues up to 9 fits at ~101 minute(s) each -- about 15"
  echo "hours of this box, roughly \$22. Measured, not estimated; see the header."
  echo "Re-run with QD_CONTROL_FIT_ACK=1 if you mean to spend it."
  echo
  echo "Note that warming the cache does NOT fill in the margins on rows already written."
  echo "The 72 operator-holdout rows are final; the arms must be re-run afterwards."
  exit 2
fi

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
  --max-fit-minutes "$FIT_CAP_MIN"
)

fit() {
  local label="$1"; shift
  local log=/home/ubuntu/opctl-"$label".log
  echo "######## $label started $(date -u +%FT%TZ) ########" > "$log"
  timeout --signal=TERM --kill-after=120 "$FIT_CAP_S" \
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
