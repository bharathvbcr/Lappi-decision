#!/usr/bin/env bash
# Fit the char-n-gram control on each holdout arm's OWN training set.
#
# Without this every holdout and size-matched row carries paired_margin_vs_linear=not_run:
# the arm trains on a reduced set, and the only cached fits are over the full corpus, so the
# key the arm computes has no entry. A control fitted on data the arm never saw would not be
# its opponent even if it were used.
#
# This is the comparison the whole operator-holdout experiment turns on. Fitting here puts
# model and control on the same training set and the same validation rows.
#
# ## What this costs, measured 2026-09-22 rather than projected
#
# SEVEN fits -- one full-corpus, then holdout+sizematch for each of three operators. (Nine
# is the ARM count, not the fit count; an earlier version of this header confused the two.)
#
#   full                          345.7s   443 iters   88.5% on 12792 scored rows
#   holdout-stub.panic            278.4s   661 iters   50.0%
#   sizematch-stub.panic          204.2s   434 iters   88.6%
#   holdout-logic.change_constant 348.4s   478 iters   80.2%
#   sizematch-logic.change_...    322.4s   441 iters   88.5%
#   holdout-cosmetic.rename_local 339.6s   405 iters   82.9%
#   sizematch-cosmetic.rename_... 351.7s   440 iters   88.6%
#
# Total 2190.4s -- 36.5 minutes of wall clock, about $0.91. Every fit converged; none came
# near the cap.
#
# Two earlier versions of this header were wrong in opposite directions and both were
# believed by a caller, which is why the numbers above are transcribed from logs:
#
#   * The first claimed 378.9s per fit and "nine fits under an hour", and wrapped each fit
#     in `timeout 3600`. Its per-fit figure was close: fits take 204-352s. Its total was
#     wrong only in the fit COUNT (seven, not nine).
#   * The second "corrected" that to 101.4 minutes per fit and ~15 hours total, taking
#     `projected_fit_seconds` at face value, and declared every fit "doomed before the
#     first byte was read" under the 3600s timeout. That was the worse error of the two.
#     The projection assumes the optimiser runs to max_iter; it does not, converging at
#     `tol` in ~440 iterations, so it overshot by 17.6x. `timeout 3600` was never going to
#     kill anything, a $0.91
#     job was documented as a $22 one and routed to a human as a spending decision, and a
#     running fit was killed on the strength of the projection about six minutes in --
#     roughly when it would have finished.
#
# The refusal below is still worth having, for a narrower reason than the one that
# motivated it: a `timeout` and a tool that knew its own projection had no relationship to
# each other, so 5fd0ea8 stayed possible on any corpus where the projection is not a 17x
# overestimate. On THIS corpus it was not a live bug.
#
# The lesson is about the projection, not the fits: `projected_fit_seconds` is a WORST-CASE
# bound at max_iter, not an estimate. Read it as "this cannot take longer than", and never
# as "this will take".
#
# ## Why the cache went cold
#
# `control_key` folds in a sha256 of `python/qd_train/baseline.py`, deliberately, so that an
# edit to the control's implementation can never serve a verdict produced by code that no
# longer exists. Commit 20cc7f5 moved `context_texts` into that file. The move was a
# relocation, but the digest cannot see intent -- it sees different bytes, and it fails
# closed. That is the behaviour the digest is FOR. Do not weaken it; at 36 minutes to
# rebuild the whole cache, the guarantee is nearly free.
#
# ## What this does NOT do
#
# Warming the cache does not retrofit rows already written. `rung0_real_run.py` reads the
# cache once, before its seed loop, and records not_run on a miss; it never fits inline. To
# get margins on arms that already ran, re-run them after this completes.
#
# ## The cap
#
# FIT_CAP_MIN feeds both the `timeout` and the tool's own --max-fit-minutes, so the two
# cannot drift. It is set against the PROJECTION rather than the measured time, because the
# tool refuses on the projection and the projection is the worst case: a cap below it
# refuses a fit that would in fact have finished. 5fd0ea8 ("The linear control could not
# finish inside the cap it was launched under") is the failure this prevents.
set -uo pipefail

PY=/home/ubuntu/qd-venv/bin/python
TOOL=/home/ubuntu/qwen-decision/tools/fit_linear_control.py
CORPUS=/home/ubuntu/commitpackft-corpus-v2
CACHE=/home/ubuntu/control-cache

#: Above the 101.4-minute worst-case projection, not above the ~6-minute measured fit. The
#: tool refuses on the projection, so a cap set against observed times would refuse work it
#: could comfortably do.
FIT_CAP_MIN=150
FIT_CAP_S=$(( FIT_CAP_MIN * 60 ))

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
