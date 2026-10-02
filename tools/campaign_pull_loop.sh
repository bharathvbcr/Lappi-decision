#!/usr/bin/env bash
# Run on the MAC for the length of a campaign: pull every snapshot the box publishes, until
# the final one is pulled and acknowledged.
#
# tools/campaign_driver.py runs on the box and will not terminate the instance until the
# final snapshot's `pulled-<n>.ok` exists -- it waits a bounded grace window, then keeps the
# box alive rather than destroy the only copy. That acknowledgement is written by
# `tools/sync_box.sh pull`, which this loops. So this loop is the reason the box is allowed
# to stop billing; if it is not running at the end of the campaign, the box waits, and then
# stays up.
#
# Bounded twice: QD_PULL_LOOP_MAX_HOURS (default 80, above the 72 h campaign cap plus its
# grace window) ends the loop loudly; a pull that fails is logged and retried at the next
# tick rather than ending the loop, because a transient ssh failure is not a reason to stop
# acknowledging.
#
# Exit status: 0 the final snapshot was pulled and verified; 1 the loop hit its time bound
# without seeing it (the box is waiting or kept alive -- go and look).
#
# Usage (same variables as `sync_box.sh pull`):
#   QD_BOX=ubuntu@1.2.3.4 QD_BOX_KEY=~/.ssh/key QD_PULL_DEST=~/qd-campaign \
#   QD_PULL_SYNC_DIR=/home/ubuntu/campaign/state/sync bash tools/campaign_pull_loop.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${QD_PULL_INTERVAL_S:-60}"
MAX_HOURS="${QD_PULL_LOOP_MAX_HOURS:-80}"
START=$(date +%s)
LIMIT=$(awk -v h="$MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }')
FAILURES=0

while :; do
  NOW=$(date +%s)
  if [ $((NOW - START)) -ge "$LIMIT" ]; then
    echo "!!! pull loop: ${MAX_HOURS} h elapsed without the FINAL snapshot. The box is waiting"
    echo "!!! for pulled-<n>.ok or has been kept alive and is billing. Go and look."
    exit 1
  fi
  OUT=$(bash "$REPO/tools/sync_box.sh" pull 2>&1)
  RC=$?
  printf '%s pull rc=%s\n%s\n' "$(date '+%H:%M:%S')" "$RC" "$OUT"
  if [ "$RC" -eq 0 ] && printf '%s\n' "$OUT" | grep -q "FINAL SNAPSHOT PULLED"; then
    echo "=== pull loop: final snapshot acknowledged after $FAILURES failed pull(s); done ==="
    exit 0
  fi
  if [ "$RC" -ne 0 ] && [ "$RC" -ne 4 ]; then
    FAILURES=$((FAILURES + 1))
    echo "!!! pull loop: pull failed ($FAILURES so far); retrying in ${INTERVAL}s"
  fi
  sleep "$INTERVAL"
done
