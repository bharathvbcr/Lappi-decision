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
#   gitignored build/mac-heavy.lock.d, so every worktree shares it. (.claude/ is a path the
#   harness protects from writes.)
# - The script waits up to $MAC_HEAVY_MAX_WAIT_S (default 4 h). Then it fails with exit 75 and
#   never runs unlocked.
# - It refuses with exit 74 when the data volume has under $MAC_HEAVY_MIN_FREE_GB (default 15) GB
#   free.
# - It sets CARGO_BUILD_JOBS=2 unless the caller set it (4 until the second panic on 2026-10-02;
#   the human then chose "resume, gentler": cargo at -j 2).
# - It releases the lock on every exit path, and only a lock it owns.
# - It exits with the command's own status.
#
# Three guards were added after the second launchd-SIGBUS panic (14:24 local, at load ~30 with one
# job under this lock; AUDIT/mac-stability-2026-10-02/report.md). JetsamEvent reports showed
# single test processes at 36-120 GiB resident on this 64 GiB Mac.
# - Memory cap. The job's whole process tree (children and grandchildren) is summed every
#   $MAC_HEAVY_RSS_POLL_S (default 2) seconds. Above $MAC_HEAVY_MAX_RSS_MB (default 32768) the
#   tree gets SIGTERM, then SIGKILL after 10 s, and the script exits 70.
# - Busy machine. After taking the lock, it waits while the 1-minute load average exceeds
#   $MAC_HEAVY_MAX_LOAD (default 12) or the process count exceeds $MAC_HEAVY_MAX_PROCS (default
#   2500). If that does not clear within the wait budget, it exits 76 and the job never runs.
# - Clean stop. On SIGTERM or SIGINT it stops the job's tree before it exits, so no job outlives
#   its wrapper. The job runs in the background of this script, so its stdin is /dev/null.
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

LOCK=${MAC_HEAVY_LOCK:-/Users/bharath/Code/research/Lappi-decision/build/mac-heavy.lock.d}
MAX_WAIT=${MAC_HEAVY_MAX_WAIT_S:-14400}
MIN_FREE_GB=${MAC_HEAVY_MIN_FREE_GB:-15}
POLL=${MAC_HEAVY_POLL_S:-20}
MAX_RSS_MB=${MAC_HEAVY_MAX_RSS_MB-32768}
RSS_POLL=${MAC_HEAVY_RSS_POLL_S-2}
MAX_LOAD=${MAC_HEAVY_MAX_LOAD-12}
MAX_PROCS=${MAC_HEAVY_MAX_PROCS-2500}
for v in "$MAX_WAIT" "$MIN_FREE_GB" "$POLL" "$MAX_RSS_MB" "$RSS_POLL" "$MAX_LOAD" "$MAX_PROCS"; do
  case "$v" in
    ''|*[!0-9]*) echo "mac_heavy: MAC_HEAVY_MAX_WAIT_S, _MIN_FREE_GB, _POLL_S, _MAX_RSS_MB, _RSS_POLL_S, _MAX_LOAD and _MAX_PROCS must be non-negative integers" >&2; exit 64 ;;
  esac
done
[ "$POLL" -ge 1 ] || POLL=1
[ "$RSS_POLL" -ge 1 ] || RSS_POLL=1

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

# The pids of $1 and every descendant, one per line, from one process-table snapshot.
tree_pids() {
  ps -Ao pid=,ppid= | awk -v root="$1" '
    { parent[$1] = $2 }
    END {
      if (!(root in parent)) exit
      seen[root] = 1
      grew = 1
      while (grew) {
        grew = 0
        for (p in parent) if (!(p in seen) && (parent[p] in seen)) { seen[p] = 1; grew = 1 }
      }
      for (p in seen) print p
    }'
}

# The summed resident size, in KiB, of $1 and every descendant.
tree_rss_kb() {
  ps -Ao pid=,ppid=,rss= | awk -v root="$1" '
    { parent[$1] = $2; rss[$1] = $3 }
    END {
      if (!(root in parent)) { print 0; exit }
      seen[root] = 1
      grew = 1
      while (grew) {
        grew = 0
        for (p in parent) if (!(p in seen) && (parent[p] in seen)) { seen[p] = 1; grew = 1 }
      }
      total = 0
      for (p in seen) total += rss[p]
      print total
    }'
}

# SIGTERM the job's tree, give it 10 s, then SIGKILL whatever is left.
JOB=""
stop_job() {
  [ -n "$JOB" ] || return 0
  local pids
  pids=$(tree_pids "$JOB")
  [ -n "$pids" ] || return 0
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null
  local i=0
  while [ "$i" -lt 10 ] && kill -0 "$JOB" 2>/dev/null; do
    sleep 1
    i=$((i + 1))
  done
  pids=$(tree_pids "$JOB")
  # shellcheck disable=SC2086
  [ -z "$pids" ] || kill -KILL $pids 2>/dev/null
  wait "$JOB" 2>/dev/null
  JOB=""
}

trap release EXIT
trap 'stop_job; exit 130' INT
trap 'stop_job; exit 143' TERM

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

# Busy machine: hold the lock and wait for quiet; a job never starts into a storm.
load1() { sysctl -n vm.loadavg | awk '{ print $2 }'; }
nprocs() { ps -A -o pid= | wc -l | tr -d ' '; }
busy_note=-300
while :; do
  load=$(load1)
  procs=$(nprocs)
  over=$(awk -v l="$load" -v ml="$MAX_LOAD" -v p="$procs" -v mp="$MAX_PROCS" \
    'BEGIN { print ((l + 0 > ml + 0) || (p + 0 > mp + 0)) ? 1 : 0 }')
  [ "$over" = 0 ] && break
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "mac_heavy: $(now) the machine stayed busy (load $load > $MAX_LOAD or $procs processes > $MAX_PROCS) for the whole wait; '$LABEL' did NOT run" >&2
    exit 76
  fi
  if [ $((waited - busy_note)) -ge 300 ]; then
    echo "mac_heavy: $(now) '$LABEL' holds the lock but waits: machine busy (load $load, max $MAX_LOAD; $procs processes, max $MAX_PROCS)" >&2
    busy_note=$waited
  fi
  sleep "$POLL"
  waited=$((waited + POLL))
done

export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
echo "mac_heavy: $(now) '$LABEL' holds the lock (waited ${waited}s; load $load, $procs processes); running: $*" >&2
"$@" &
JOB=$!
cap_kb=$((MAX_RSS_MB * 1024))
while kill -0 "$JOB" 2>/dev/null; do
  rss_kb=$(tree_rss_kb "$JOB")
  if [ "${rss_kb:-0}" -gt "$cap_kb" ]; then
    echo "mac_heavy: $(now) '$LABEL' is over the RSS cap: its process tree holds $((rss_kb / 1024)) MiB > $MAX_RSS_MB MiB; stopping it (SIGTERM, then SIGKILL after 10 s). The largest:" >&2
    for p in $(tree_pids "$JOB"); do ps -o rss=,pid=,command= -p "$p"; done \
      | sort -rn | head -5 | awk '{ printf "mac_heavy:   %d MiB  pid %s  %s\n", $1 / 1024, $2, substr($0, index($0, $3), 120) }' >&2
    stop_job
    echo "mac_heavy: $(now) '$LABEL' was killed over the RSS cap (exit 70); its result does not count" >&2
    exit 70
  fi
  sleep "$RSS_POLL"
done
wait "$JOB"
rc=$?
JOB=""
echo "mac_heavy: $(now) '$LABEL' finished with exit $rc" >&2
exit "$rc"
