"""Lane L-prep2's navigation gap, appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. It refuses to append a second copy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

GAP = "GAP-L-PREP2-NAVIGATION-2026-10-02"

if any(r.get("id") == GAP for r in read_gaps()):
    sys.exit(f"{GAP} is already in gaps.jsonl; refusing a duplicate")

append_gap(
    {
        "id": GAP,
        "opened": "2026-10-02",
        "lane": "L-prep2",
        "owner": "lead",
        "status": "open",
        "tool": (
            "devmap_status, devmap_explore/search, gitpulse_insights, ToolSearch for ListAgents"
        ),
        "question": (
            "Could DevMap, GitPulse Insights and ListAgents answer lane L-prep2's navigation "
            "questions (callers of qd_train.replay.decontaminate and prompt_content, who else "
            "holds tools/containment_scan.py, python/qd_train/exclusions.py and the parity test)?"
        ),
        "answer": (
            "Partly. DevMap: this worktree has no store (devmap_status: no readable DevMap store "
            "under .devmap/codeintel); the main checkout's index (generation 3062, fresh, 12,273 "
            "nodes) answered instead, with tools/real_ft_run.py a parse failure (520,787 bytes "
            "over the 5 s budget), so every walk was walk_incomplete and the callers of "
            "decontaminate and prompt_content are a lower bound. GitPulse: insights ran, every "
            "facet ok, but "
            "the collisions facet was truncated (16 of 50 worktrees scanned, 34 unscanned) and "
            "18 worktrees reported dirty_unknown, so 'no collision on these files' is not "
            "established. ListAgents: not exposed in this session (ToolSearch found no such "
            "tool), so a peer session in this worktree could not be checked; the worktree was "
            "fresh (no commits beyond main, nothing dirty) when this lane started. Every other "
            "read was direct (Read, sed, rg) and is reported as such, not as graph-confirmed."
        ),
    }
)
print(f"appended {GAP}")
