"""The lead's resolutions of J6(g)'s prelude and staging gaps (2026-10-02), appended once through
qd_train.gaps.append_gap. Kept as the record of what was appended; it refuses to append a second
time once a gap's current record is no longer open.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps  # noqa: E402

COMMON_EVIDENCE = (
    "commit 32593c7; AUDIT/j6g-rule-2026-10-02/{j6g-prelude-record.json,j6g-prelude-run.log,run_prelude.sh}"
)
RESOLUTIONS = {
    "GAP-J6G-MAC-PRELUDE-NOT-RUN-2026-10-02": (
        "Run. Executed 2026-10-02 at 14:06-14:10 UTC through tools/mac_heavy.sh, holding the machine's one "
        "heavy-job lock. Script: AUDIT/j6g-rule-2026-10-02/run_prelude.sh, which carries out HANDOFF "
        "section 5. Result: exit 0, PRELUDE OK in 219 s. 'Option permutation seed 20260919: 183252 choice "
        "rows re-permuted per pass'. 0 permutation refusals. 9,683 batches, width 7,936, 363,950 rows. "
        "Shard 8bcf56ad, data ea3215c4, tokenizer fe000e3e, code_commit a5026707. Record sha256 "
        "ef17d56c. Two measurements differ from the handoff's [I]: wall time 219 s against 10-20 min, "
        "and max RSS 20.2 GB (20,190,789,632 bytes) against 5-12 GB, about twice the upper estimate; "
        "both are now [V]. --locked refused a502670's stale Cargo.lock, so qd-prep was built from a "
        "second throwaway worktree at a502670 without --locked (offline), per the handoff's fallback. "
        "The prelude's --root stayed the clean worktree."
    ),
    "GAP-J6G-NOT-STAGED-BLOCKS-V5-C1-2026-10-02": (
        "Staged and launched. The pins in box_q_j6g.sh were filled with J6G_PRELUDE="
        "/home/ubuntu/post-f/j6g-prelude-record.json and J6G_PRELUDE_SHA256=ef17d56c; the result was "
        "committed at 32593c7 (sha256 bc0dfeaa). On the box, verified by sha256sum: checker "
        "/home/ubuntu/bin/qd-post-f-rules-j6g 4090d565 (a new path; qd-post-f-rules cadbdc74 and "
        "qd-post-f-rules-succ be020297 unchanged), pre-registration 8df45d23, record ef17d56c, waiter "
        "bc0dfeaa. post_f_common.sh 4867546e and idle_common.sh 1a449f7a are unchanged. The box's F "
        "train header names shard 8bcf56ad and data ea3215c4, equal to the record's. Launched "
        "~14:13 UTC while j6a.queued existed (GAP-J6G-WAITER-LAUNCH-ORDER-2026-10-02 respected). "
        "j6g.queued is present, and the log carries no deferral line. The waiter's order: j5pp.done, "
        "then wait_queued j6a, then gpu.lock. v5's C1 will have a J6(g) row to read once it runs."
    ),
}

current = {}
for r in read_gaps():
    current[r.get("id")] = r
for gap, answer in RESOLUTIONS.items():
    rec = current.get(gap)
    if rec is None:
        sys.exit(f"{gap} is not in gaps.jsonl")
    if rec.get("status") != "open":
        sys.exit(f"{gap} is already {rec.get('status')}")
for gap, answer in RESOLUTIONS.items():
    rec = dict(current[gap])
    rec.update(
        {
            "status": "resolved",
            "resolved": "2026-10-02",
            "owner": "lead",
            "answer": answer,
            "evidence": COMMON_EVIDENCE,
        }
    )
    append_gap(rec)
    print("resolved", gap)
