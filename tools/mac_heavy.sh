#!/bin/bash
# Run one heavy job on the Mac under the machine's one-heavy-job lock.
#
#   bash tools/mac_heavy.sh <label> <command> [args...]
#
# "Heavy" means a cargo build or test, a full pytest/torch suite, a data build or containment
# scan, or any GPU/Metal/MPS job. The Mac kernel-panicked on 2026-10-02 at a load of about 350,
# with several sessions building and testing at once
# (GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-2026-10-02). This script makes the rule mechanical
# for every Lappi lane:
#
# - The lock is an atomic `mkdir` of $MAC_HEAVY_LOCK. It defaults to the main checkout's
#   gitignored .claude/mac-heavy.lock.d, so every worktree shares it.
# - The script waits up to $MAC_HEAVY_MAX_WAIT_S (default 4 h). Then it fails with exit 75 and
#   never runs unlocked.
# - It refuses with exit 74 when the data volume has under $MAC_HEAVY_MIN_FREE_GB (default 15) GB
#   free.
# - It sets CARGO_BUILD_JOBS=4 unless the caller set it.
# - It releases the lock on every exit path, and only a lock it owns.
# - It exits with the command's own status.
#
# It never breaks another holder's lock. A holder whose pid is gone is reported as STALE every
# 5 minutes, and the lead clears it by hand. A hold taken by hand (for another session's run)
# carries no pid and is never reported stale.
set -u

if [ $# -lt 2 ]; then
  echo "usage: bash tools/mac_heavy.sh <label> <command> [args...]" >&2
  exit 64
fi
LABEL=$1
shift

LOCK=${MAC_HEAVY_LOCK:-/Users/bharath/Code/research/Lappi-decision/.claude/mac-heavy.lock.d}
MAX_WAIT=${MAC_HEAVY_MAX_WAIT_S:-14400}
MIN_FREE_GB=${MAC_HEAVY_MIN_FREE_GB:-15}
POLL=${MAC_HEAVY_POLL_S:-20}
case "$MAX_WAIT$MIN_FREE_GB$POLL" in
  *[!0-9]*) echo "mac_heavy: MAC_HEAVY_MAX_WAIT_S, MAC_HEAVY_MIN_FREE_GB and MAC_HEAVY_POLL_S must be integers" >&2; exit 64 ;;
esac
[ "$POLL" -ge 1 ] || POLL=1

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

waited=0
last_note=-300
until mkdir "$LOCK" 2>/dev/null; do
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "mac_heavy: $(now) gave up after ${waited}s waiting for $LOCK; '$LABEL' did NOT run" >&2
    [ -f "$LOCK/owner" ] && sed 's/^/mac_heavy:   holder: /' "$LOCK/owner" >&2
    exit 75
  fi
  if [ $((waited - last_note)) -ge 300 ]; then
    holder=$(tr '\n' ' ' < "$LOCK/owner" 2>/dev/null)
    hpid=$(sed -n 's/^pid=//p' "$LOCK/owner" 2>/dev/null)
    stale=""
    if [ -n "$hpid" ] && ! kill -0 "$hpid" 2>/dev/null; then stale=" STALE (pid $hpid is gone; the lead clears it)"; fi
    echo "mac_heavy: $(now) '$LABEL' waiting ${waited}s for the lock; holder: ${holder:-unknown}$stale" >&2
    last_note=$waited
  fi
  sleep "$POLL"
  waited=$((waited + POLL))
done

release() {
  if [ "$(sed -n 's/^pid=//p' "$LOCK/owner" 2>/dev/null)" = "$$" ]; then
    rm -f "$LOCK/owner"
    rmdir "$LOCK" 2>/dev/null
  fi
}
trap release EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
  echo "label=$LABEL"
  echo "pid=$$"
  echo "start=$(now)"
  echo "cwd=$PWD"
} > "$LOCK/owner"

vol=/System/Volumes/Data
[ -d "$vol" ] || vol=/
free_kb=$(df -k "$vol" | awk 'NR == 2 { print $4 }')
case "$free_kb" in
  ''|*[!0-9]*) echo "mac_heavy: could not read free space on $vol; '$LABEL' did NOT run" >&2; exit 74 ;;
esac
if [ "$free_kb" -lt $((MIN_FREE_GB * 1024 * 1024)) ]; then
  echo "mac_heavy: $vol has $((free_kb / 1024 / 1024)) GB free, under ${MIN_FREE_GB} GB; '$LABEL' did NOT run" >&2
  exit 74
fi

export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
echo "mac_heavy: $(now) '$LABEL' holds the lock (waited ${waited}s); running: $*" >&2
"$@"
rc=$?
echo "mac_heavy: $(now) '$LABEL' finished with exit $rc" >&2
exit "$rc"
