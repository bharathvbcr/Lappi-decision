"""Append lane L-v5-rules' gap records to gaps.jsonl through qd_train.gaps.append_gap.

One O_APPEND + fsync line per record, validated first; never a read-modify-write. A record whose
id is already in the ledger is skipped, so a second run appends nothing.

Run from the worktree root:
    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python AUDIT/v5-rules-2026-10-02/append_gaps.py
"""

from qd_train.gaps import append_gap, current_records

LANE = "L-v5-rules"
OPENED = "2026-10-02"

RECORDS = [
    {
        "id": "GAP-V5-RULES-REBUILT-SUITE-NOT-ON-THE-ROW-2026-10-02",
        "owner": "lead",
        "status": "open",
        "tool": "read: ledger/gh200-p4-v4-2026-10-01.jsonl f4feac15 (recipe.needle, "
        "metrics.needle_suite_tokens); branch l-v5-train-build d4552f2 (needle.py, real_ft_run.py)",
        "question": "R9 and the noul-weight arm's needle guard read the 8K worst bucket 'on the "
        "rebuilt suite' (campaign/v5-preregistered.DRAFT.json readings.R9, arm_noul_weight.guards). "
        "Can qd-post-f-rules confirm from an eval row that it was scored on the rebuilt suite?",
        "answer": "No [V, read]. recipe.needle records cases, cases_per_depth, hit_rule, "
        "min_recall, suite_seed and target_tokens (8192 for F and for v5 alike) and no suite "
        "digest; metrics.needle_suite_tokens.value is the median real length, and the max appears "
        "only in free detail text. d4552f2 sizes the suite in real tokens and refuses an "
        "over-target case at scoring time, but adds no row field that tells the rebuilt suite from "
        "F's. v5-pause, v5-noulw and seeds34 --preregistration record recipe.needle and the median "
        "in their JSON and name this gap under not_checked; they do not refuse on it. A row field "
        "(the suite digest the DRAFT's amendments_pending names) would let them.",
    },
    {
        "id": "GAP-V5-RULES-AMENDMENTS-NOT-CROSS-CHECKED-2026-10-02",
        "owner": "lead",
        "status": "open",
        "tool": "read: campaign/v5-preregistered.DRAFT.json amendments_pending",
        "question": "Do the v5 checkers pin v5's rows to the build the pre-registration's "
        "amendments will name (build commit, data_snapshot_hash, shard hashes, needle suite "
        "digest)?",
        "answer": "No. amendments_pending is a list of strings with no filled structure yet, and "
        "this lane did not invent one (the j6a precedent's six-key `amendments` object was "
        "registered by its pre-registration, not by its checker). The v5 checkers pin v5's rows "
        "by the ft row ids the waiter passes, check them as one configuration (completed, quick "
        "false, tag epoch, no shuffled_label, one recipe hash and data snapshot) and hold the "
        "arm's code_commit, data_snapshot_hash and shard_hash equal to v5's same-seed ft row; "
        "each decision records code_commit, recipe_hash and data_snapshot_hash for the reader. "
        "Under Fable's seed-order ruling (merged at 4be5d87) v5-noulw also applies the pairing "
        "check identity.ft_rows names: corpus.plan_order_digest recorded on the arm's and v5's "
        "same-seed ft rows and equal, else refused. Not checked, because no bound text says it "
        "yet: that each v5 seed's digest is the one the Mac prelude printed for that seed, that "
        "corpus.plan_seed equals protocol.seed, and that recipe.batch_order = 'seed' is on every "
        "v5 ft row (amendments_pending lists the printed digests and the key; the arm inherits "
        "v5's batch_order through recipe equality, so an arm row without it refuses only if v5's "
        "rows carry it). Cross-checking them against the filled amendments needs the amendment's "
        "shape fixed first, then a checker change.",
    },
    {
        "id": "GAP-V5-RULES-DRAFT-KEY-REFUSES-UNTIL-THE-RENAME-DROPS-IT-2026-10-02",
        "owner": "lead",
        "status": "open",
        "tool": "qd-post-f-rules v5-pause / v5-noulw / seeds34 --preregistration "
        "(crates/qd-runtime/src/bin/qd_post_f_rules.rs not_a_draft)",
        "question": "What does the rename to campaign/v5-preregistered.json have to do for the "
        "v5 checkers to apply the file?",
        "answer": "Drop the top-level \"draft\" key. The DRAFT's own first words are that no "
        "checker, waiter or lane may read it as a rule, so every v5 subcommand refuses a file "
        "that still carries it (tested on the real DRAFT). The rename commit must also keep the "
        "texts the checkers read and phrase-check (readings.R9_pause_after_seed_0, seeds.v5, "
        "seeds.seeds_3_4, arm_noul_weight.*) or the checker in step with them; an amended text "
        "refuses until the checker follows it.",
    },
    {
        "id": "GAP-L-V5-RULES-NAVIGATION-2026-10-02",
        "owner": "lead",
        "status": "open",
        "tool": "DevMap MCP, GitPulse MCP, ToolSearch for ListAgents",
        "question": "What could DevMap, GitPulse Insights and ListAgents not answer for lane "
        "L-v5-rules?",
        "answer": "(1) DevMap has no store in this worktree (.devmap/codeintel/devmap.sqlite "
        "absent); skeleton, clones and affected_tests were answered from the main checkout's "
        "index, whose qd_post_f_rules.rs is byte-identical to the worktree's pre-change source "
        "(sha256 739355ba..., main a3c09de); affected_tests named only the bin's own unit tests "
        "and carried walk_incomplete (60,625 of 119,856 sites unresolved repo-wide). (2) "
        "GitPulse collisions scanned 16 of 57 worktrees (truncated, 41 unscanned), so 'no other "
        "worktree holds qd_post_f_rules.rs' rests on git: no commit on any branch outside main "
        "touches it or tests/post_f_rules_v5.rs (git log --all --not main). (3) ListAgents was "
        "not loadable in this session (ToolSearch returned SendMessage only), so a peer editing "
        "the same files in this worktree was not ruled out by it; this worktree is the lane's "
        "own agent worktree. (4) devmap_clones ran once, on the main index before any v5 code "
        "existed, budget-capped at 42 of 181 groups; it could not see the new code, so 'no "
        "near-copy' for the v5 section rests on reading it against the reused helpers, not on "
        "the graph. The pairing addition reuses one_configuration for the arm's three ft rows "
        "(the call that already checks v5's envelope) and adds no recipe-hash comparison of its "
        "own.",
    },
]


def main() -> None:
    have = current_records()
    for rec in RECORDS:
        if rec["id"] in have:
            print(f"skip {rec['id']}: already recorded")
            continue
        line = append_gap({"id": rec["id"], "opened": OPENED, "lane": LANE, **rec})
        print(f"appended {rec['id']} ({len(line)} bytes)")


if __name__ == "__main__":
    main()
