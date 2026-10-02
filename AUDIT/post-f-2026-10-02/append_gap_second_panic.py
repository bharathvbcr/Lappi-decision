"""The lead's record of the Mac's second kernel panic on 2026-10-02.

Appended once to GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-2026-10-02 through
qd_train.gaps.append_gap; refuses a second run. It states that the one-heavy-job lock did not
prevent the second panic, so the morning's "concurrent heavy load" diagnosis is unverified.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

GAP = "GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-2026-10-02"
PANIC = "/Library/Logs/DiagnosticReports/panic-base+socd-2026-10-02-142456.000.panic"
if any(r.get("id") == GAP and PANIC in str(r.get("evidence", "")) for r in read_gaps()):
    sys.exit("already appended")

append_gap(
    {
        "id": GAP,
        "opened": "2026-10-02",
        "lane": "lead",
        "owner": "human",
        "status": "open",
        "tool": (
            "the panic file's panicString and Compressor line; kern.boottime; "
            "the lock's owner file"
        ),
        "question": (
            "The Mac panicked a second time, at 14:24:56 local (19:24:56Z). Did the one-heavy-job "
            "lock prevent it, and is 'concurrent heavy load' the cause?"
        ),
        "answer": (
            "No, and unverified. Same panicString as 08:53: 'initproc exited -- exit reason "
            "namespace 2 subcode 0xa', i.e. launchd (PID 1) ended by signal 10 (SIGBUS). The "
            "Compressor line reads '4% of compressed pages limit (OK) ... with 4 swapfiles and OK "
            "swap space'; the data volume had 224 GiB free. Load was ~30-36, not ~350. "
            "Exactly one "
            "Lappi heavy job held the lock: 'L-v5-queue mutations round 2 (27 x one test file)', "
            "pid 20686, started 19:24:05Z (the stale owner file, removed by the lead after "
            "kern.boottime 19:24:39Z proved it stale). Outside the lock: several non-Claude `agy` "
            "agents (vitest, golangci-lint in Portfolio, ScholarLM, Manvi) and ~90 Claude "
            "sessions' "
            "MCP servers. The lock was reasonable hygiene, not the fix: CPU load does not end "
            "launchd, and SIGBUS in PID 1 points at a failed fault on a mapped page, a launchd bug "
            "or hardware [I]. One correlate both times, a hypothesis only: rapid process spawning "
            "(a 3,500-test pytest with torch at 08:53; mutation testing plus vitest forks plus MCP "
            "server launches at 14:24). Mac heavy work is paused pending the human; the box was "
            "unaffected (GPU 100%, 19 waiters alive at 19:28Z)."
        ),
        "evidence": PANIC + "; uptime 3 min at 14:27 local",
    }
)
print("appended", GAP)
