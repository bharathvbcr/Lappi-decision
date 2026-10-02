"""Lane L-prep3's navigation gap, appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. It refuses to append a second copy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

GAP = "GAP-L-PREP3-NAVIGATION-2026-10-02"

if any(r.get("id") == GAP for r in read_gaps()):
    sys.exit(f"{GAP} is already in gaps.jsonl; refusing a duplicate")

append_gap(
    {
        "id": GAP,
        "opened": "2026-10-02",
        "lane": "L-prep3",
        "owner": "lead",
        "status": "open",
        "tool": "devmap_status, devmap_impact, gitpulse_insights, ToolSearch for ListAgents",
        "question": (
            "Could DevMap, GitPulse Insights and ListAgents answer lane L-prep3's navigation "
            "questions (what reaches qd_train.containment_strip and qd_train.exclusions, who "
            "else holds the lane's files)?"
        ),
        "answer": (
            "Partly. DevMap: this worktree has no store (devmap_status: no readable DevMap "
            "store under .devmap/codeintel); the main checkout's index (generation 3074, fresh, "
            "12,446 nodes, 44,577 edges) answered instead. devmap_impact on "
            "python/qd_train/containment_strip.py (depth 3) named exclusions.py, "
            "tools/containment_scan.py, tools/real_tokenizer_pipeline.py and the three "
            "containment test files, with walk_incomplete: tools/real_ft_run.py failed to parse "
            "(520,787 bytes over the 5 s budget), so its callers are a lower bound. GitPulse: "
            "insights ran and every facet reported ok, but the collisions facet was truncated "
            "(16 of 56 worktrees scanned, 40 unscanned; the one overlap named was Cargo.lock), "
            "so 'no collision on the lane's files' is not established. ListAgents: not exposed "
            "in this session (ToolSearch finds no such tool), so a peer in this worktree could "
            "not be checked; the worktree was fresh and clean when the lane started (branched "
            "at c65d7da, fast-forwarded to main 6e62363 before any change). Every other read was "
            "direct (Read, sed, rg) and is reported as such, not as graph-confirmed."
        ),
    }
)
print(f"appended {GAP}")
