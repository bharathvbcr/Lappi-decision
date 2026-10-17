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
# - It exports MAC_HEAVY_LABEL=<label> to the job, so the job can tell it runs under the lock:
#   python/tests/conftest.py runs cargo on darwin only when it is set
#   (GAP-PYTEST-QD-PREP-FIXTURE-BUILDS-OUTSIDE-THE-MAC-LOCK-2026-10-06).
# - It releases the lock on every exit path, and only a lock it owns.
# - It exits with the command's own status.
#
# Three guards were added after the second launchd-SIGBUS panic (14:24 local, at load ~30 with one
# job under this lock; AUDIT/mac-stability-2026-10-02/report.md). JetsamEvent reports showed
# single test processes at 36-120 GiB resident on this 64 GiB Mac.
# - Memory cap, on phys_footprint (not RSS: Metal/MPS memory is in the footprint and not in `ps`
#   RSS; a 3 GiB MPS allocation moved RSS +0.04 GiB and the footprint +3.06 GiB). The job's whole
#   process tree (children and grandchildren) is summed every $MAC_HEAVY_RSS_POLL_S (default 2)
#   seconds by tools/mac_footprint.py (stdlib ctypes, proc_pid_rusage). Above
#   $MAC_HEAVY_MAX_FOOTPRINT_MB (default 53248 = 52 GiB: 64 GiB minus a 12 GiB floor for the OS,
#   the harness and a second light session) the tree gets SIGTERM, then SIGKILL after 10 s, and the
#   script exits 70. Summing over a tree can double-count shared IOKit memory; a GPU job is one
#   process and the cap is a safety bound, so over-counting is the side to err on. If the reader
#   fails on 3 consecutive polls the job is stopped and the script exits 71 ("memory guard could
#   not run"): a running job with no guard is a check that could not run. A caller that still sets
#   the old MAC_HEAVY_MAX_RSS_MB gets exit 64, so it is never silently uncapped. A job that
#   legitimately needs more sets the new variable on its launch line, and the line is recorded.
#   $MAC_HEAVY_FOOTPRINT_CMD replaces the reader (it is run with the tree's pids and prints the
#   summed MiB) and is refused with exit 64 unless MAC_HEAVY_TEST_HOOK=1 is also set. Every numeric
#   setting is refused above 9 digits. This script does not export PYTORCH_MPS_HIGH_WATERMARK_RATIO; a torch
#   job that wants it sets it itself.
#   A tighter cap binds first and is not this script's: memguard (user LaunchAgent
#   com.bharath.memguard, source ~/Code/devtools/memguard, verified 2026-10-17 from main.go, policy.go
#   and the plist) samples ri_phys_footprint of every process the user owns every 2 s and SIGTERMs any
#   unprotected single process over 32 GiB, SIGKILL 10 s later; under critical memory pressure for 3
#   consecutive samples it does the same to the largest process of 2 GiB or more. Per process, not per
#   tree, so a 33 GiB job dies to memguard long before this 52 GiB tree cap. It posts a notification
#   and writes a JSON "terminate" line to ~/Library/Logs/memguard/memguard.log by its plist, but that
#   directory does not exist on this Mac (checked 2026-10-17), so the kill leaves no log line here or
#   there: the job ends with signal 15 and no footprint message from this script.
# - Busy machine. After taking the lock, it waits while the 1-minute load average exceeds
#   $MAC_HEAVY_MAX_LOAD (default 12) or the process count exceeds $MAC_HEAVY_MAX_PROCS (default
#   2500). If that does not clear within the wait budget, it exits 76 and the job never runs.
# - Clean stop. On SIGTERM or SIGINT it stops the job's tree before it exits, so no job outlives
#   its wrapper. The job runs in the background of this script in its own process group (set -m),
#   with stdin /dev/null. The "tree" is the job's descendants plus every process still in that
#   group, so a child reparented to init when the root exits is still found; when the root exits,
#   whatever remains is stopped (SIGTERM, then SIGKILL) and reported as "left behind", and the
#   exit status stays the command's. A process that calls setsid leaves the group and the
#   tree, and is not covered.
#
# It never breaks another holder's lock. A holder whose pid is gone is reported as STALE every
# 5 minutes, and the lead clears it by hand. A hold taken by hand (for another session's run)
# carries no pid and is never reported stale.
# The whole body is one function and the last line calls it. Bash reads a running script
# incrementally, so a script edited in place while a wrapper runs (this one runs for hours) resumed
# at shifted offsets and could execute a fragment such as `stop_job; exit 143`
# (2026-10-12: four re-score jobs died that way). A function is parsed whole before it runs, and
# the final line is already read, so a later rewrite cannot reach a live wrapper. Install an
# edited version by writing a temp file and `mv`, which also keeps the old inode for live readers.
main() {
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
if [ -n "${MAC_HEAVY_MAX_RSS_MB+x}" ]; then
  echo "mac_heavy: MAC_HEAVY_MAX_RSS_MB is retired; the cap is now on phys_footprint, so use MAC_HEAVY_MAX_FOOTPRINT_MB (default 53248). Refusing rather than running uncapped" >&2
  exit 64
fi
MAX_FP_MB=${MAC_HEAVY_MAX_FOOTPRINT_MB-53248}
FOOTPRINT_PY=$(dirname "$0")/mac_footprint.py
RSS_POLL=${MAC_HEAVY_RSS_POLL_S-2}
MAX_LOAD=${MAC_HEAVY_MAX_LOAD-12}
MAX_PROCS=${MAC_HEAVY_MAX_PROCS-2500}
# Every numeric setting is a canonical decimal: "0", or a digit 1-9 then digits, at most 9 of them.
# A leading zero is refused, not normalised: bash arithmetic reads 09 as an invalid octal (the
# lock loop then fell through and the job ran without the lock) and 0900 as 0 (a free-space floor
# of nothing). Bounded too: a value too wide for bash's integer tests makes `[ -gt ]` error, and
# an error read as "not over" is a guard that fails open.
for v in "$MAX_WAIT" "$MIN_FREE_GB" "$POLL" "$MAX_FP_MB" "$RSS_POLL" "$MAX_LOAD" "$MAX_PROCS"; do
  case "$v" in
    0|[1-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9]|[1-9][0-9][0-9][0-9][0-9]|[1-9][0-9][0-9][0-9][0-9][0-9]|[1-9][0-9][0-9][0-9][0-9][0-9][0-9]|[1-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]|[1-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) ;;
    *) echo "mac_heavy: MAC_HEAVY_MAX_WAIT_S, _MIN_FREE_GB, _POLL_S, _MAX_FOOTPRINT_MB, _RSS_POLL_S, _MAX_LOAD and _MAX_PROCS must each be a canonical non-negative decimal integer of at most 9 digits (no leading zeros, no sign, no fraction); got '$v'. Refusing" >&2; exit 64 ;;
  esac
done
if [ -n "${MAC_HEAVY_FOOTPRINT_CMD:-}" ] && [ "${MAC_HEAVY_TEST_HOOK:-}" != 1 ]; then
  echo "mac_heavy: MAC_HEAVY_FOOTPRINT_CMD replaces the memory guard and is honoured only for tests (MAC_HEAVY_TEST_HOOK=1); refusing rather than running unguarded" >&2
  exit 64
fi
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
  waited=$((waited + 10#$POLL))
done

release() {
  if [ "$(sed -n 's/^pid=//p' "$LOCK/owner" 2>/dev/null)" = "$$" ]; then
    rm -f "$LOCK/owner"
    rmdir "$LOCK" 2>/dev/null
  fi
}

# The pids of the job rooted at $1: every process in process group $1 (the job's own group, from
# set -m), and every descendant of those. The root pid $1 itself counts only while it is alive
# ($2 = 1): once it has exited its number may belong to an unrelated process.
tree_pids() {
  ps -Ao pid=,ppid=,pgid= | awk -v root="$1" -v alive="${2:-1}" '
    { parent[$1] = $2; pg[$1] = $3 }
    END {
      for (p in parent) if (pg[p] == root && (alive == 1 || p != root)) seen[p] = 1
      grew = 1
      while (grew) {
        grew = 0
        for (p in parent) if (!(p in seen) && (parent[p] in seen) && (alive == 1 || p != root)) { seen[p] = 1; grew = 1 }
      }
      for (p in seen) print p
    }'
}

# The summed phys_footprint, in MiB, of $1 and every descendant. Fails (non-zero, nothing on
# stdout) when the reader fails or prints anything but an integer.
tree_footprint_mb() {
  local pids out
  pids=$(tree_pids "$1" "${JOB_ALIVE:-1}")
  [ -n "$pids" ] || { echo 0; return 0; }
  # shellcheck disable=SC2086
  if [ -n "${MAC_HEAVY_FOOTPRINT_CMD:-}" ]; then
    out=$($MAC_HEAVY_FOOTPRINT_CMD $pids) || return 1
  else
    out=$(python3 -I "$FOOTPRINT_PY" $pids) || return 1
  fi
  case "$out" in ''|*[!0-9]*) return 1 ;; esac
  echo "$out"
}

# SIGTERM the job's tree, give it 10 s, then SIGKILL whatever is left. $JOB_ALIVE is 1 while the
# root has not been reaped.
JOB=""
JOB_ALIVE=1
stop_job() {
  [ -n "$JOB" ] || return 0
  local pids i=0
  pids=$(tree_pids "$JOB" "$JOB_ALIVE")
  [ -n "$pids" ] || return 0
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null
  while [ "$i" -lt 10 ] && [ -n "$(tree_pids "$JOB" "$JOB_ALIVE")" ]; do
    sleep 1
    i=$((i + 1))
  done
  pids=$(tree_pids "$JOB" "$JOB_ALIVE")
  # shellcheck disable=SC2086
  [ -z "$pids" ] || kill -KILL $pids 2>/dev/null
  [ "$JOB_ALIVE" = 1 ] && wait "$JOB" 2>/dev/null
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
if [ "$free_kb" -lt $((10#$MIN_FREE_GB * 1024 * 1024)) ]; then
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
  waited=$((waited + 10#$POLL))
done

export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
# This job's own label, replacing any the caller inherited from an outer mac_heavy.sh.
export MAC_HEAVY_LABEL=$LABEL
echo "mac_heavy: $(now) '$LABEL' holds the lock (waited ${waited}s; load $load, $procs processes); running: $*" >&2
set -m
"$@" < /dev/null &
JOB=$!
set +m
job_pgid=$(ps -o pgid= -p "$JOB" | tr -d ' ')
# An empty answer means the job already exited; its group is then swept below like any other.
if [ -n "$job_pgid" ] && [ "$job_pgid" != "$JOB" ]; then
  echo "mac_heavy: the job is not in its own process group (pgid '$job_pgid', pid $JOB); the guard cannot cover it" >&2
  kill -TERM "$JOB" 2>/dev/null
  exit 71
fi
reader_fails=0
while kill -0 "$JOB" 2>/dev/null; do
  if fp_mb=$(tree_footprint_mb "$JOB") && [ "${#fp_mb}" -le 12 ]; then
    reader_fails=0
    if [ "$fp_mb" -gt "$MAX_FP_MB" ]; then
      echo "mac_heavy: $(now) '$LABEL' is over the footprint cap: its process tree holds $fp_mb MiB of phys_footprint > $MAX_FP_MB MiB; stopping it (SIGTERM, then SIGKILL after 10 s). The largest by RSS:" >&2
      for p in $(tree_pids "$JOB" 1); do ps -o rss=,pid=,command= -p "$p"; done \
        | sort -rn | head -5 | awk '{ printf "mac_heavy:   %d MiB RSS  pid %s  %s\n", $1 / 1024, $2, substr($0, index($0, $3), 120) }' >&2
      stop_job
      echo "mac_heavy: $(now) '$LABEL' was killed over the footprint cap (exit 70); its result does not count" >&2
      exit 70
    fi
  elif kill -0 "$JOB" 2>/dev/null; then
    reader_fails=$((reader_fails + 1))
    echo "mac_heavy: $(now) '$LABEL': the footprint reader failed ($reader_fails of 3)" >&2
    if [ "$reader_fails" -ge 3 ]; then
      stop_job
      echo "mac_heavy: $(now) '$LABEL' was stopped: the memory guard could not run (exit 71); its result does not count" >&2
      exit 71
    fi
  fi
  sleep "$RSS_POLL"
done
wait "$JOB"
rc=$?
JOB_ALIVE=0
left=$(tree_pids "$JOB" 0)
if [ -n "$left" ]; then
  echo "mac_heavy: $(now) '$LABEL': the job's root exited with $(echo "$left" | wc -l | tr -d ' ') process(es) left behind (pids $(echo $left)); stopping them (SIGTERM, then SIGKILL after 10 s)" >&2
  stop_job
fi
JOB=""
echo "mac_heavy: $(now) '$LABEL' finished with exit $rc" >&2
exit "$rc"
}

main "$@"; exit $?
