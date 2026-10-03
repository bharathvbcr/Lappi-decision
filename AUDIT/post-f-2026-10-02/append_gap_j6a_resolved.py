"""The lead's resolution of GAP-J6A-RUN-CONDITION-IDLES-GPU-ON-ROOM-REFUSAL-2026-10-02, appended
once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. Running it again would append a duplicate
resolution; it refuses if the gap's current record is already resolved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

GAP = "GAP-J6A-RUN-CONDITION-IDLES-GPU-ON-ROOM-REFUSAL-2026-10-02"

current = [r for r in read_gaps() if r.get("id") == GAP]
if not current:
    sys.exit(f"{GAP} is not in gaps.jsonl")
if current[-1].get("status") != "open":
    sys.exit(f"{GAP} is already {current[-1].get('status')}; refusing a duplicate resolution")

append_gap(
    {
        "id": GAP,
        "opened": "2026-10-02",
        "resolved": "2026-10-02",
        "lane": "fable-postf",
        "owner": "lead",
        "status": "resolved",
        "tool": "bash j6a_cond_test.sh (16 fixtures); sha256sum on the Mac and the box; pgrep on "
                "the box",
        "question": current[-1]["question"],
        "answer": (
            "Amended per Fable's post-F ruling 2(a), before any fsucc row exists. At 266950a, "
            "box_q_j6a.sh (sha256 90cdeefe...) runs J6(a) when fsucc's word is quiet, or when it "
            "is refused and four things hold: the newest fsucc decision JSON has detail.arms as an "
            "object; every detail.refused_because entry begins with \"(c)\"; neither F' ledger "
            "exists; and $Q/j6a-on-room-refusal-yes is non-empty. j6a-preregistered.json (sha256 "
            "5e6b10ab...) carries the same text in arm.what and adds readings.R4_room_refusal. The "
            "human's advance yes (\"Yes, run J6(a) then (Recommended)\") is recorded at 0b559bb. "
            "It is pinned on the box at /home/ubuntu/queue/j6a-on-room-refusal-yes (389 bytes). "
            "The fixture test gives 16/16 on the amended block; the old block fails exactly "
            "room_only_with_pin (15/16). On the box, ~13:26 UTC, the old waiter (pid 1076141) was "
            "killed with -9, the two files were replaced (box shas equal the Mac shas above), and "
            "the waiter was relaunched (pid 499082, alive at 13:31 UTC). j6a.queued is present and "
            "j6a.done is absent. Residual: the relaunch truncated /home/ubuntu/logs/q-j6a.log "
            "('>' where '>>' was intended), so the old waiter's log is gone. The waiter prints "
            "nothing until fsucc decides, so that log was most likely empty (inferred, not "
            "verified). The two F'-ledger checks use box paths, and the fixture test could not "
            "exercise them on the Mac."
        ),
        "evidence": (
            "commits 266950a and 0b559bb; AUDIT/post-f-2026-10-02/j6a_cond_test.sh and "
            "j6a_cond_test.log; AUDIT/post-f-2026-10-02/human-answers-post-f.md"
        ),
    }
)
print(f"appended resolution for {GAP}")
