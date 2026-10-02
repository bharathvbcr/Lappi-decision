"""The lead's resolution of GAP-L-V5DATA-OWN-PROSE-HUMAN-REVIEW-2026-10-02 on main (2026-10-02).

The gap was opened on lane L-v5-data's branch, which lands only on v5-build. This record carries
the human's answer to main as well, so citations on main resolve. When v5-build merges main,
the union puts this record after the lane's open one, so it becomes current. Appended once
through qd_train.gaps.append_gap; refuses a second run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps  # noqa: E402

GAP = "GAP-L-V5DATA-OWN-PROSE-HUMAN-REVIEW-2026-10-02"
if any(r.get("id") == GAP and r.get("status") == "resolved" for r in read_gaps()):
    sys.exit(f"{GAP} already resolved here")
append_gap(
    {
        "id": GAP,
        "opened": "2026-10-02",
        "resolved": "2026-10-02",
        "lane": "L-v5-data",
        "owner": "human",
        "status": "resolved",
        "tool": "AskUserQuestion (the lead)",
        "question": (
            "Is every own-prose-v1 unit text the human wants trained on, and does the question form meet "
            "the DRAFT's '~599 distinct question sentences'?"
        ),
        "answer": (
            "The human, ~16:45 UTC: 'Strike personal + business (Recommended)'. Struck by walker rule: the "
            "repos web/Lappi-BDay (98 units), web/WhimsicalLove (6) and web/bharathvbcr (14), and the paths "
            "scholarlm/docs/business/** (38) and research/BINN/writing/** (31). That is 187 of 2,150, so "
            "~1,963 remain, above the 1,500 floor. AGENTS.md/CLAUDE.md and CHANGELOG.md are kept. The "
            "question form measured 197 units, not ~599; the DRAFT records this as data.sources[3]."
            "form_measured, and the 1,500 floor is unchanged. Lane L-v5-data applies the strike on its "
            "branch for v5-build."
        ),
        "evidence": (
            "AUDIT/v5-plan-2026-10-02/human-answer-own-prose-strike.md; commit ba432c4; "
            "HANDOFF/v5-data-2026-10-02.md (v5-build)"
        ),
    }
)
print("resolved", GAP)
