#!/bin/bash
# The release replay as one command: build the requests, replay them through the product socket,
# check the results. `make release-replay` runs this; `make preflight-release` then reads the
# results file it wrote (REPLAY_RESULTS).
#
#   RELEASE_DIR=... VAL_ROWS=... MIN_ANSWERED=N REPLAY_RESULTS=... SOCKET=... \
#     bash tools/release_replay.sh
#
# Required, none with a default (a missing one is NotRun, exit 3, and nothing runs):
#   RELEASE_DIR     the qd-export release directory
#   VAL_ROWS        JSONL of val rows: {"row_id", "split": "val", "request": <wire request>},
#                   written by qd_data.val_requests.write_val_rows; its .binding.json sidecar must
#                   name the release's train_manifest_sha256
#   MIN_ANSWERED    distinct answered requests required per (admitted family, trained shape); it is
#                   also how many requests are built per pair. GAP-QD-RELEASE-CHECK-MIN-ANSWERED-HAS-NO-RECORD-2026-10-17
#                   is the owner's: this script never picks one.
#   REPLAY_RESULTS  where the replay results are written
#   SOCKET          the socket a qd-metal-serve bound to RELEASE_DIR listens on
# Optional:
#   VERDICTS CALIB_ROWS CALIBRATION_REPORT  passed to `check`; without VERDICTS the release's
#                   headline figures are not computed, and then the flow is NotRun, never PASS.
#   QD (default `qd`), QD_PYTHON, QD_RELEASE_CHECK_BIN (default: cargo run -p qd-preflight),
#   WORK_DIR (default: the results file's directory; the built requests are kept there).
#
# Exit codes: 0 PASS, 1 FAILED, 3 NotRun; the last line is `RESULT: PASS|FAILED|NotRun`.
# FAILED outranks NotRun. PASS needs: every pair full, every replay served over the socket, every
# `check` line passing, and the headline figures computed from real eval verdicts.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0
FAILED=1
NOT_RUN=3

finish() {
  case "$1" in
    "$PASS") echo "RESULT: PASS" ;;
    "$FAILED") echo "RESULT: FAILED" ;;
    *) echo "RESULT: NotRun" ;;
  esac
  exit "$1"
}

need() {
  if [ -z "${2:-}" ]; then
    echo "NotRun: release-replay - $1 is not set; this flow has no default for it and did not run."
    finish "$NOT_RUN"
  fi
}

need RELEASE_DIR "${RELEASE_DIR:-}"
need VAL_ROWS "${VAL_ROWS:-}"
need MIN_ANSWERED "${MIN_ANSWERED:-}"
need REPLAY_RESULTS "${REPLAY_RESULTS:-}"
need SOCKET "${SOCKET:-}"

case "$MIN_ANSWERED" in
  '' | *[!0-9]* | 0)
    echo "NotRun: release-replay - MIN_ANSWERED=$MIN_ANSWERED is not a positive integer."
    finish "$NOT_RUN"
    ;;
esac

PYTHON="${QD_PYTHON:-$REPO/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "NotRun: release-replay - no interpreter at $PYTHON (set QD_PYTHON); the val rows were not read."
  finish "$NOT_RUN"
fi
WORK="${WORK_DIR:-$(dirname "$REPLAY_RESULTS")}"
mkdir -p "$WORK" || finish "$NOT_RUN"
VAL_REQUESTS="$WORK/val_requests.jsonl"
REQUESTS="$WORK/replay_requests.jsonl"

rc_bin() {
  if [ -n "${QD_RELEASE_CHECK_BIN:-}" ]; then
    "$QD_RELEASE_CHECK_BIN" "$@"
  else
    cargo run --quiet --manifest-path "$REPO/Cargo.toml" -p qd-preflight --bin qd-release-check -- "$@"
  fi
}

# Anything but 0 or 1 from a step is NotRun: a step that could not run says nothing about the release.
step_status() {
  case "$1" in
    1) return "$FAILED" ;;
    *) return "$NOT_RUN" ;;
  esac
}

echo "======== 1/4 val requests ========"
# VAL_ROWS must carry the binding the tracked writer (qd_data.val_requests.write_val_rows)
# records, naming the train manifest this release was exported from: a hand-made file or another
# shard set's is refused (no binding: NotRun; a different one: FAILED).
TRAIN_SHA="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_manifest"]["ledger_binding"]["train_manifest_sha256"])' "$RELEASE_DIR/release_manifest.json" 2>&1)"
case "$TRAIN_SHA" in
  *[!0-9a-f]* | '')
    echo "NotRun: release-replay - $RELEASE_DIR/release_manifest.json records no train_manifest_sha256, so VAL_ROWS cannot be bound to this release."
    finish "$NOT_RUN"
    ;;
esac
PYTHONPATH="$REPO/python" "$PYTHON" -m qd_data.val_requests --val-rows "$VAL_ROWS" --out "$VAL_REQUESTS" \
  --expect-train-manifest-sha256 "$TRAIN_SHA"
rc=$?
if [ "$rc" -ne 0 ]; then step_status "$rc"; finish $?; fi

echo "======== 2/4 build-requests (per shape: $MIN_ANSWERED) ========"
rc_bin build-requests --release "$RELEASE_DIR" --val-requests "$VAL_REQUESTS" --per-shape "$MIN_ANSWERED" --out "$REQUESTS"
rc=$?
if [ "$rc" -ne 0 ]; then
  [ "$rc" -eq 1 ] && echo "FAILED: release-replay - a (family, shape) has fewer than $MIN_ANSWERED val requests; nothing was replayed."
  step_status "$rc"; finish $?
fi

echo "======== 3/4 replay through $SOCKET ========"
rc_bin replay --release "$RELEASE_DIR" --socket "$SOCKET" --out "$REPLAY_RESULTS" \
  --requests "$REQUESTS" --qd "${QD:-qd}" --repo "$REPO"
rc=$?
if [ "$rc" -ne 0 ]; then step_status "$rc"; finish $?; fi

echo "======== 4/4 check ========"
set -- check --release "$RELEASE_DIR" --results "$REPLAY_RESULTS" --min-answered "$MIN_ANSWERED" --repo "$REPO"
[ -n "${VERDICTS:-}" ] && set -- "$@" --verdicts "$VERDICTS"
[ -n "${CALIB_ROWS:-}" ] && set -- "$@" --calib-rows "$CALIB_ROWS"
[ -n "${CALIBRATION_REPORT:-}" ] && set -- "$@" --calibration-report "$CALIBRATION_REPORT"
out="$WORK/check_output.txt"
rc_bin "$@" > "$out" 2>&1
rc=$?
# `check`'s own RESULT: line is the step's, not the flow's: only the last line of this script's
# output is the flow's verdict, and a grep for `^RESULT: PASS` must not match a step.
sed 's/^RESULT: /check step RESULT: /' "$out"
if [ "$rc" -ne 0 ]; then step_status "$rc"; finish $?; fi
if grep -q '^NotRun headline metrics' "$out"; then
  echo "NotRun: release-replay - every check line passed, but the headline figures were not computed (no VERDICTS from a real eval); a replay alone is not a pass."
  finish "$NOT_RUN"
fi
finish "$PASS"
