#!/bin/bash
# Span-refit arm B launcher (campaign/span-refit-mac-preregistered.json), rebuilt as a tracked
# script after the untracked build/w3-rescore/b.sh was lost
# (GAP-SPAN-REFIT-B-LAUNCHER-B-SH-GONE-2026-10-14).
#
#   SPAN_CODE_REPO=<checkout the val shard was written by> bash tools/span_refit_b.sh [--print]
#
# The chain, in HANDOFF/span-rescore-2026-10-12.md's order ("steps: seed-1 val extraction, fit c,
# fit e"; the avg-train extraction finished before this launcher was lost and is not redone):
#
#   0. verify the finished avg-train cache (12,000 rows) -- fail closed, never re-extract silently
#   1. span_refit_extract.py --tower seed1 --split val --device mps   (resumable: rows already in
#      cache/seed1-val/rows.jsonl are skipped, and the layout must match or the script refuses)
#   2. span_refit_fit.py --arm c
#   3. span_refit_fit.py --arm e
#
# Every heavy step runs under tools/mac_heavy.sh, one at a time, logging to
# build/span-refit-b-<date>/<step>.log (appended, so a resumed extraction keeps its history).
# Scoring and the five ledger rows are NOT here: tools/span_refit_rows.py does that afterwards.
#
# Rule 3: every path this script reads or writes goes through qd_train.data_access
# assert_path_not_held_out (the repo's own check, with DataConfig's markers) before anything runs.
# Rule 5: --print runs nothing heavy; it shows each step's argv and verifies nothing on disk.
# Rule 8: the rows these caches feed are quick; this script writes no ledger row.
#
# Settings (environment):
#   SPAN_CODE_REPO       required. The checkout whose python/ and tools/ the extraction imports
#                        (span_refit_common.CODE_REPO); the val shard's code is not main's, so the
#                        choice is the caller's and is never defaulted here.
#   SPAN_REFIT_B_DATE    log directory date, default 2026-10-14.
#   SPAN_REFIT_B_PY      python for the steps and the held-out check, default the ml venv.
set -euo pipefail

REPO=/Users/bharath/Code/research/Lappi-decision
BUILD=/Users/bharath/Code/research/Lappi-decision/build/needle-span-mac-2026-10-06
AVG_TRAIN=/Users/bharath/Code/research/Lappi-decision/build/needle-span-mac-2026-10-06/cache/avg-train
SEED1_VAL=/Users/bharath/Code/research/Lappi-decision/build/needle-span-mac-2026-10-06/cache/seed1-val
MAC_HEAVY=/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh
PY=${SPAN_REFIT_B_PY:-/Users/bharath/.venvs/ml/bin/python}
DATE=${SPAN_REFIT_B_DATE:-2026-10-14}
AVG_TRAIN_ROWS=12000
HIDDEN=2048

PRINT=0
case "${1-}" in
  "") ;;
  --print) PRINT=1 ;;
  *) echo "usage: SPAN_CODE_REPO=<checkout> bash tools/span_refit_b.sh [--print]" >&2; exit 64 ;;
esac
[ $# -le 1 ] || { echo "usage: SPAN_CODE_REPO=<checkout> bash tools/span_refit_b.sh [--print]" >&2; exit 64; }

case "$DATE" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
  *) echo "span_refit_b: SPAN_REFIT_B_DATE must be YYYY-MM-DD; got '$DATE'. Refusing" >&2; exit 64 ;;
esac
if [ -z "${SPAN_CODE_REPO:-}" ]; then
  echo "span_refit_b: SPAN_CODE_REPO is not set. It names the checkout the val shard's code is pinned to (span_refit_common.CODE_REPO); main's checkout has drifted from it, so this script will not guess. Refusing" >&2
  exit 64
fi
LOG_DIR=/Users/bharath/Code/research/Lappi-decision/build/span-refit-b-$DATE

# Rule 3, before anything else: the repo's own path check on every path this chain touches.
CHECK_ROOT=$(dirname "$(dirname "$0")")
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
' "$CHECK_ROOT" "$SPAN_CODE_REPO" "$BUILD" "$AVG_TRAIN" "$SEED1_VAL" "$LOG_DIR" "$BUILD/heads"

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
  [ -f "$AVG_TRAIN/layout.json" ] || { echo "span_refit_b: $AVG_TRAIN/layout.json is missing; the completed avg-train extraction is not on disk, and this launcher does not redo it silently. Refusing" >&2; exit 66; }
  [ -f "$AVG_TRAIN/rows.jsonl" ] || { echo "span_refit_b: $AVG_TRAIN/rows.jsonl is missing. Refusing" >&2; exit 66; }
  rows=$(wc -l < "$AVG_TRAIN/rows.jsonl" | tr -d ' ')
  if [ "$rows" -ne "$AVG_TRAIN_ROWS" ]; then
    echo "span_refit_b: $AVG_TRAIN/rows.jsonl holds $rows rows, the finished extraction wrote $AVG_TRAIN_ROWS. Refusing" >&2
    exit 66
  fi
  bytes=$(stat -f %z "$AVG_TRAIN/hq.f32")
  if [ "$bytes" -ne "$want" ]; then
    echo "span_refit_b: $AVG_TRAIN/hq.f32 is $bytes bytes, expected $want ($AVG_TRAIN_ROWS x $HIDDEN fp32). Refusing" >&2
    exit 66
  fi
  echo "span_refit_b: avg-train cache present ($rows rows, hq.f32 $bytes bytes); extraction step skipped"
}

if [ "$PRINT" = 1 ]; then
  echo "SKIP avg-train extraction: verified, not redone: rows.jsonl == $AVG_TRAIN_ROWS lines and hq.f32 == $((AVG_TRAIN_ROWS * HIDDEN * 4)) bytes in $AVG_TRAIN"
else
  verify_avg_train
  mkdir -p "$LOG_DIR"
fi

step span-refit-b-extract-seed1-val extract-seed1-val \
  "$REPO/tools/span_refit_extract.py" --tower seed1 --split val --device mps
step span-refit-b-fit-c fit-c "$REPO/tools/span_refit_fit.py" --arm c
step span-refit-b-fit-e fit-e "$REPO/tools/span_refit_fit.py" --arm e
