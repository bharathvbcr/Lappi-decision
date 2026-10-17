#!/bin/bash
# Span-refit arm B launcher (campaign/span-refit-mac-preregistered.json), rebuilt as a tracked
# script after the untracked build/w3-rescore/b.sh was lost
# (GAP-SPAN-REFIT-B-LAUNCHER-B-SH-GONE-2026-10-14) and extended per
# AUDIT/board-sweep-2026-10-12/fable-span-refit-b-recovery-ruling.md (main's build/ was wiped, so the
# caches are redone and now live outside any build/ directory).
#
#   SPAN_CODE_REPO=<pin tree> SPAN_CODE_PIN=<full sha> bash tools/span_refit_b.sh [--redo] [--print]
#
# Without --redo the chain is the continuation of a finished avg-train extraction (HANDOFF/
# span-rescore-2026-10-12.md's order):
#
#   0. verify the finished avg-train cache (12,000 rows) -- fail closed, never re-extract silently
#   1. span_refit_extract.py --tower seed1 --split val --device mps   (resumable: rows already in
#      cache/seed1-val/rows.jsonl are skipped, and the layout must match or the script refuses)
#   2. span_refit_fit.py --arm c
#   3. span_refit_fit.py --arm e
#
# With --redo (the ruling's S1-S4, staged so the gate is bought first):
#
#   1. span_refit_extract.py --tower avg --split val        S1
#   2. span_refit_score.py --gate-only                      the prereg's arm (a) gate; it exits 3
#                                                           on failure and this launcher exits 68
#                                                           with nothing after it run
#   3. span_refit_extract.py --tower avg --split train      S2 (resumable), then the same cache
#                                                           verification as step 0 above
#   4. seed1 val extraction, fit c, fit e                   S4, S3
#
# Every heavy step runs under tools/mac_heavy.sh, one at a time, logging to
# $SPAN_REFIT_ROOT/logs/<date>/<step>.log (appended, so a resumed extraction keeps its history).
# Every step names its cache/head/result paths explicitly (--out/--cache/--val-cache); the four
# span_refit_*.py scripts keep their C.BUILD defaults untouched. Scoring and the five ledger rows are
# NOT here: tools/span_refit_rows.py does that afterwards (--val-cache, --head-c, --head-e,
# --seed1-val-cache under $SPAN_REFIT_ROOT).
#
# Rule 3: every path this script reads or writes goes through qd_train.data_access
# assert_path_not_held_out (the repo's own check, with DataConfig's markers) before anything runs.
# Rule 5: --print runs nothing heavy; it shows each step's argv and verifies nothing on disk (the
# pin included).
# Rule 8: the rows these caches feed are quick; this script writes no ledger row.
#
# Settings (environment):
#   SPAN_CODE_REPO       required. The checkout whose python/ and tools/ the extraction imports
#                        (span_refit_common.CODE_REPO); the val shard's code is not main's, so the
#                        choice is the caller's and is never defaulted here.
#   SPAN_CODE_PIN        required in run mode, not defaulted: the full 40-hex sha the pin tree's HEAD
#                        must equal, with a tree clean of tracked changes, else exit 67
#                        (GAP-SPAN-CHAIN-UNPINNED-TREE-MAIN-CHANGED-UNDER-IT-2026-10-06).
#   SPAN_REFIT_ROOT      cache/ heads/ results/ logs/ root. A root that resolves under a repo build/
#                        directory is refused (exit 64): build/ was wiped once.
#   SPAN_REFIT_B_DATE    log directory date, default 2026-10-14.
#   SPAN_REFIT_B_PY      python for the steps and the held-out check, default the ml venv.
# Exit codes: 64 usage or bad setting, 65 held-out path, 66 avg-train cache missing or short,
# 67 pin tree not at the pin or dirty, 68 the arm (a) gate failed.
set -euo pipefail

REPO=/Users/bharath/Code/research/Lappi-decision
ROOT=${SPAN_REFIT_ROOT:-/Users/bharath/qd-campaign/span-refit-2026-10-06}
ROOT=${ROOT%/}
CACHE=$ROOT/cache
AVG_VAL=$CACHE/avg-val
AVG_TRAIN=$CACHE/avg-train
SEED1_VAL=$CACHE/seed1-val
HEADS=$ROOT/heads
RESULTS=$ROOT/results
MAC_HEAVY=/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh
PY=${SPAN_REFIT_B_PY:-/Users/bharath/.venvs/ml/bin/python}
DATE=${SPAN_REFIT_B_DATE:-2026-10-14}
AVG_TRAIN_ROWS=12000
HIDDEN=2048
USAGE="usage: SPAN_CODE_REPO=<checkout> SPAN_CODE_PIN=<sha> bash tools/span_refit_b.sh [--redo] [--print]"

PRINT=0
REDO=0
for arg in "$@"; do
  case "$arg" in
    --print) [ "$PRINT" = 0 ] || { echo "$USAGE" >&2; exit 64; }; PRINT=1 ;;
    --redo) [ "$REDO" = 0 ] || { echo "$USAGE" >&2; exit 64; }; REDO=1 ;;
    *) echo "$USAGE" >&2; exit 64 ;;
  esac
done

case "$DATE" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
  *) echo "span_refit_b: SPAN_REFIT_B_DATE must be YYYY-MM-DD; got '$DATE'. Refusing" >&2; exit 64 ;;
esac
if [ -z "${SPAN_CODE_REPO:-}" ]; then
  echo "span_refit_b: SPAN_CODE_REPO is not set. It names the checkout the val shard's code is pinned to (span_refit_common.CODE_REPO); main's checkout has drifted from it, so this script will not guess. Refusing" >&2
  exit 64
fi
case "$ROOT" in
  /*) ;;
  *) echo "span_refit_b: SPAN_REFIT_ROOT must be an absolute path; got '$ROOT'. Refusing" >&2; exit 64 ;;
esac
LOG_DIR=$ROOT/logs/$DATE

# A root under a repo build/ directory is refused, as resolved (symlinks followed), before any step
# is shown: build/ is gitignored and was wiped once, taking the caches with it.
CHECK_ROOT=$(dirname "$(dirname "$0")")
"$PY" -I -c '
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
for repo in sys.argv[2:]:
    build = (Path(repo) / "build").resolve()
    if root == build or build in root.parents:
        print(f"span_refit_b: SPAN_REFIT_ROOT {root} is under {build}; build/ is not durable. Refusing", file=sys.stderr)
        sys.exit(64)
' "$ROOT" "$REPO" "$CHECK_ROOT"

# Rule 3, before anything else: the repo's own path check on every path this chain touches.
"$PY" -I -c '
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "python"))
from qd_data.config import DataConfig
from qd_data.errors import HeldOutViolation
from qd_train.data_access import assert_path_not_held_out
for p in sys.argv[2:]:
    try:
        assert_path_not_held_out(Path(p), config=DataConfig(), repo_root=root)
    except HeldOutViolation as exc:
        print(f"span_refit_b: held-out path refused (rule 3): {exc}", file=sys.stderr)
        sys.exit(65)
' "$CHECK_ROOT" "$SPAN_CODE_REPO" "$ROOT" "$AVG_VAL" "$AVG_TRAIN" "$SEED1_VAL" "$LOG_DIR" "$HEADS" "$HEADS/arm-c" "$HEADS/arm-e" "$RESULTS"

verify_pin() {
  local head
  if [ -z "${SPAN_CODE_PIN:-}" ]; then
    echo "span_refit_b: SPAN_CODE_PIN is not set. Run mode requires the full sha the pin tree must be at; it is never defaulted. Refusing" >&2
    exit 67
  fi
  case "$SPAN_CODE_PIN" in
    *[!0-9a-f]* | "") echo "span_refit_b: SPAN_CODE_PIN must be a full lowercase 40-hex sha; got '$SPAN_CODE_PIN'. Refusing" >&2; exit 64 ;;
  esac
  [ "${#SPAN_CODE_PIN}" -eq 40 ] || { echo "span_refit_b: SPAN_CODE_PIN must be a full 40-hex sha; got '$SPAN_CODE_PIN'. Refusing" >&2; exit 64; }
  head=$(git -C "$SPAN_CODE_REPO" rev-parse HEAD 2>/dev/null) || { echo "span_refit_b: $SPAN_CODE_REPO is not a git checkout; the pin cannot be verified. Refusing" >&2; exit 67; }
  if [ "$head" != "$SPAN_CODE_PIN" ]; then
    echo "span_refit_b: $SPAN_CODE_REPO HEAD is $head, not SPAN_CODE_PIN $SPAN_CODE_PIN. Refusing" >&2
    exit 67
  fi
  if [ -n "$(git -C "$SPAN_CODE_REPO" status --porcelain --untracked-files=no)" ]; then
    echo "span_refit_b: $SPAN_CODE_REPO is not clean (tracked changes against $SPAN_CODE_PIN). Refusing" >&2
    exit 67
  fi
  echo "span_refit_b: pin tree $SPAN_CODE_REPO is at $SPAN_CODE_PIN and clean"
}

step() { # <label> <log name> <command and args...>
  local label=$1 logname=$2
  shift 2
  if [ "$PRINT" = 1 ]; then
    echo "STEP $label >> $LOG_DIR/$logname.log :: bash $MAC_HEAVY $label env SPAN_CODE_REPO=$SPAN_CODE_REPO $PY -u $*"
    return 0
  fi
  bash "$MAC_HEAVY" "$label" env "SPAN_CODE_REPO=$SPAN_CODE_REPO" "$PY" -u "$@" >> "$LOG_DIR/$logname.log" 2>&1
}

verify_avg_train() {
  local rows bytes want=$((AVG_TRAIN_ROWS * HIDDEN * 4))
  [ -f "$AVG_TRAIN/layout.json" ] || { echo "span_refit_b: $AVG_TRAIN/layout.json is missing; the avg-train extraction is not on disk (pass --redo to extract it). Refusing" >&2; exit 66; }
  [ -f "$AVG_TRAIN/rows.jsonl" ] || { echo "span_refit_b: $AVG_TRAIN/rows.jsonl is missing. Refusing" >&2; exit 66; }
  rows=$(wc -l < "$AVG_TRAIN/rows.jsonl" | tr -d ' ')
  if [ "$rows" -ne "$AVG_TRAIN_ROWS" ]; then
    echo "span_refit_b: $AVG_TRAIN/rows.jsonl holds $rows rows, the extraction writes $AVG_TRAIN_ROWS. Refusing" >&2
    exit 66
  fi
  [ -f "$AVG_TRAIN/hq.f32" ] || { echo "span_refit_b: $AVG_TRAIN/hq.f32 is missing. Refusing" >&2; exit 66; }
  bytes=$(stat -f %z "$AVG_TRAIN/hq.f32")
  if [ "$bytes" -ne "$want" ]; then
    echo "span_refit_b: $AVG_TRAIN/hq.f32 is $bytes bytes, expected $want ($AVG_TRAIN_ROWS x $HIDDEN fp32). Refusing" >&2
    exit 66
  fi
  echo "span_refit_b: avg-train cache present ($rows rows, hq.f32 $bytes bytes)"
}

EXTRACT=$REPO/tools/span_refit_extract.py
FIT=$REPO/tools/span_refit_fit.py
SCORE=$REPO/tools/span_refit_score.py

if [ "$PRINT" = 0 ]; then
  verify_pin
  mkdir -p "$LOG_DIR" "$RESULTS"
fi

if [ "$REDO" = 1 ]; then
  step span-refit-b-extract-avg-val extract-avg-val \
    "$EXTRACT" --tower avg --split val --device mps --out "$AVG_VAL"
  gate_rc=0
  step span-refit-b-gate gate \
    "$SCORE" --gate-only --val-cache "$AVG_VAL" --out "$RESULTS/gate.json" || gate_rc=$?
  if [ "$gate_rc" -eq 3 ]; then
    echo "span_refit_b: the arm (a) gate failed (scorer exit 3; see $RESULTS/gate.json). Every arm is void by the prereg: nothing after the gate was run. Record a gap" >&2
    exit 68
  fi
  [ "$gate_rc" -eq 0 ] || { echo "span_refit_b: the gate scorer exited $gate_rc (not a gate verdict). Stopping" >&2; exit "$gate_rc"; }
  step span-refit-b-extract-avg-train extract-avg-train \
    "$EXTRACT" --tower avg --split train --device mps --out "$AVG_TRAIN"
  [ "$PRINT" = 1 ] || verify_avg_train
elif [ "$PRINT" = 1 ]; then
  echo "SKIP avg-train extraction: verified, not redone: rows.jsonl == $AVG_TRAIN_ROWS lines and hq.f32 == $((AVG_TRAIN_ROWS * HIDDEN * 4)) bytes in $AVG_TRAIN"
else
  verify_avg_train
fi

step span-refit-b-extract-seed1-val extract-seed1-val \
  "$EXTRACT" --tower seed1 --split val --device mps --out "$SEED1_VAL"
step span-refit-b-fit-c fit-c "$FIT" --arm c --cache "$AVG_TRAIN" --out "$HEADS/arm-c"
step span-refit-b-fit-e fit-e "$FIT" --arm e --cache "$AVG_TRAIN" --out "$HEADS/arm-e"
