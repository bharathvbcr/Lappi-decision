"""Lane L-lint's two gap records, appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. It refuses to append a second copy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

NAVIGATION = "GAP-L-LINT-NAVIGATION-2026-10-02"
PINNED = "GAP-L-LINT-PINNED-FILES-KEEP-THE-GATE-RED-2026-10-02"

present = {r.get("id") for r in read_gaps()}
for gid in (NAVIGATION, PINNED):
    if gid in present:
        sys.exit(f"{gid} is already in gaps.jsonl; refusing a duplicate")

append_gap(
    {
        "id": NAVIGATION,
        "opened": "2026-10-02",
        "lane": "L-lint",
        "owner": "lead",
        "status": "open",
        "tool": "devmap_status, gitpulse_insights, ToolSearch for ListAgents",
        "question": (
            "Could DevMap, GitPulse Insights and ListAgents answer lane L-lint's navigation "
            "questions (who else holds the 18 files it reformats; what pins the bytes of the "
            "23 files the lint gate flags)?"
        ),
        "answer": (
            "Partly. DevMap: this worktree has no store (devmap_status: no readable DevMap "
            "store under .devmap/codeintel); the main checkout's index (generation 3090, "
            "fresh) indexes main at 4be5d87, not the merged v5-build files, and DevMap has no "
            "query for 'which record pins this file's sha256'. So the pin search is rg, not "
            "the graph: AUDIT/lint-2026-10-02/pin_search.py over AUDIT/, HANDOFF/, ledger/, "
            "campaign/ and gaps.jsonl, then rg -uu over the whole worktree (minus .git, .venv "
            "and the lane's own directory) for each file's sha256[:8] and git blob[:7] "
            "(pin-search.txt, pin-search-whole-tree.txt). It found four sha256 pins and no "
            "others. GitPulse: insights ran and every facet reported ok, but the collisions "
            "facet was truncated (16 of 60 worktrees scanned, 44 unscanned) and 28 worktrees "
            "reported dirty_unknown, so 'no other worktree holds these files' is not "
            "established by it. It rests on git instead: git log --all --not HEAD over the 18 "
            "files lists only eabcea8 (L-v5-queue's merge of v5-build), and git diff HEAD "
            "eabcea8 over them is empty. ListAgents was not loadable in this session (ToolSearch "
            "returned no such tool), so a peer editing the same files in this worktree was not "
            "ruled out by it; the worktree is the lane's own agent worktree, fast-forwarded "
            "from a clean state."
        ),
    }
)

append_gap(
    {
        "id": PINNED,
        "opened": "2026-10-02",
        "lane": "L-lint",
        "owner": "lead",
        "status": "open",
        "tool": "python/tests/test_lint_gate.py::test_the_repository_has_no_ruff_findings",
        "question": (
            "The lint gate still fails on 31 findings, all in five files whose bytes are "
            "pinned or declared a record. Leave them failing, or give them per-file-ignores "
            "in pyproject [tool.ruff.lint]?"
        ),
        "answer": (
            "Undecided; the lead decides, since a per-file-ignore loosens the gate's "
            "configuration. L-lint fixed the other 239 of 270 findings without changing any "
            "program (AUDIT/lint-2026-10-02/ast-equal.txt) and did not edit these five. "
            "sha256-pinned: AUDIT/post-f-2026-10-02/recency/recency_diagnostic.py (5: 4 E501, "
            "1 I001; tool_sha256 677c81c0... in recency-f-2026-10-02.json:1427, cited in "
            "HANDOFF/recency-2026-10-02.md:215); AUDIT/prep2-2026-10-02/clinc_oracle.py (8: 5 "
            "E702, 3 B007; sha256 638eb349... in fable-clinc-strip-ruling.md:3); "
            "tools/qd_train_oracle_optimizer_table.py (6: 5 E501, 1 I001; script_sha256 "
            "0cf0f7db... in crates/qd-train/tests/fixtures/optimizer-table-2b.json:5); "
            "tools/qd_train_oracle_head_digest.py (4 E501; script_sha256 79a47e72... in "
            "crates/qd-train/tests/fixtures/span-head-init-digest-tiny.json:5). Declared a "
            "record without a hash: AUDIT/post-f-2026-10-02/recency/append_gaps_recency.py "
            "(8 E501; HANDOFF/recency-2026-10-02.md:215 says it was left unfixed so it 'stays "
            "the record of what was appended'). Each of the two tools hashes its own bytes into "
            "the fixture it writes, so any edit to one needs that fixture regenerated, which is "
            "crates/** work and not this lane's."
        ),
    }
)
print(f"appended {NAVIGATION} and {PINNED}")
