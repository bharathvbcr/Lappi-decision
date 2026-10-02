"""Lane L-v5-queue's gap records, appended once through qd_train.gaps.append_gap (O_APPEND + fsync,
one line each, never a read-modify-write). Run from the worktree root:

    /Users/bharath/.venvs/ml/bin/python AUDIT/v5-queue-2026-10-02/append_gaps.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from qd_train.gaps import append_gap, read_gaps  # noqa: E402

COMMON = {"opened": "2026-10-02", "lane": "L-v5-queue", "owner": "lead", "status": "open"}

RECORDS = [
    {
        "id": "GAP-L-V5-QUEUE-NAVIGATION-2026-10-02",
        "tool": "DevMap MCP, GitPulse MCP, ToolSearch for ListAgents",
        "question": "What could DevMap, GitPulse Insights and ListAgents not answer for lane "
        "L-v5-queue?",
        "answer": (
            "(1) DevMap has no store in this worktree (devmap_status: no readable store at "
            ".devmap/codeintel/devmap.sqlite). devmap_search on the main checkout's index (fresh, "
            "generation 3090) found idle_common.sh controls_block, which the v5 waiters reuse "
            "through bash's dynamic scope rather than copying, and no shell helper for a running "
            "cost total (query 'spend': 0 hits; 'cost', semantic: real_ft_run _cost, run_cost.py, "
            "run_control.cost_for, 48 more hidden). The new files were never indexed, so 'no "
            "near-copy' for them rests on reading them against post_f_common.sh and "
            "idle_common.sh, not on devmap_clones. (2) GitPulse collisions scanned 16 of 60 "
            "worktrees (truncated, 44 unscanned). That no other worktree holds a v5 waiter rests "
            "on `rg --files -uu` over .claude/worktrees for post-f-queue/*v5* (0 hits), not on the "
            "graph. (3) ListAgents was not loadable in this session (not in the deferred tool "
            "list); peers were reached through SendMessage to the lead only. This lane's files are "
            "all new paths in its own worktree."
        ),
    },
    {
        "id": "GAP-V5-QUEUE-WAITERS-NOT-RUN-ON-BOX-2026-10-02",
        "tool": "campaign/post-f-queue/box_q_v5*.sh, v5_common.sh; "
        "python/tests/test_v5_queue_scripts.py",
        "question": (
            "Do the v5 waiters behave on the box as they do in the Mac dry runs, and do their "
            "real_ft_run.py argv parse under the v5 lane's own main()?"
        ),
        "answer": (
            "Not run on the box, and the argv were never parsed by real_ft_run.main. The dry runs "
            "ran under the Mac's /bin/bash 3.2.57 (the box has bash 5) with fakes for python, the "
            "rules binary, flock, timeout, sleep, setsid, nohup and git, and assert on the "
            "recorded argv. The v5 lane commit (build + rename) does not exist yet, so the "
            "L-waiters argv check (each recorded argv through main() of the pinned commit with "
            "resolve_rev stubbed) could not run: it is the first thing to run once V5_LANE_AT is "
            "known. Unexercised on any machine: --batch-order (not on any branch yet), the needle "
            "control and trajectory-ood argv against v5 code, ft_linear_control.py on v5's split "
            "with --exclude-identity-keys, and the qd-post-f-rules-v5 build itself (stubbed)."
        ),
    },
    {
        "id": "GAP-V5-QUEUE-DESIGN-READINGS-2026-10-02",
        "tool": "campaign/post-f-queue/v5_common.sh and the v5 waiters",
        "question": "Which readings did lane L-v5-queue make where the DRAFT leaves a choice (lead "
        "to confirm)?",
        "answer": (
            "(1) R9: refused, an unknown word, or no seed-0 ft row holds like pause (the DRAFT "
            "names only pause). (2) The arm's launch: a refused --room never runs the arm even "
            "with V5NW_HUMAN_YES (the yes covers no_room; the arm's own reading decides room from "
            "the same envelope first, so every arm row would read refused). (3) The post-seed "
            "waiter <run>traj-s<N> is started by its parent through setsid nohup, and the parent "
            "hands it gpu.lock (waits <= 900 s for its .started); it scores the final step first, "
            "then the rest ascending, inside 3,600 s per seed, then runs the CPU letter/option "
            "controls outside the lock. (4) The running total counts each GPU step's wall clock at "
            "COST's rate, not idle box time (R9 holds, waits): the box bills those hours too. (5) "
            "A run whose pre-registration estimate would carry the running total past the approved "
            "total (+ if_seeds_3_4 once seeds34 fired) is refused unless V5_OVER_BUDGET_YES holds "
            "the human's words (the lead's instruction, 2026-10-02); the estimate, not the cap, is "
            "compared, and USD and GPU-h each bind (at $2.29/h a 7.0 h seed is $16.03, so the USD "
            "side binds first near the end of the block). (6) Stale V5_CONTINUE / V5_STOP / "
            "v5.paused at v5's start refuse. (7) V5_SPLIT accepts data flags only (an allowlist): "
            "a v5 build flag outside it refuses until the allowlist is amended. (8) Every human "
            "marker must be non-empty; its first 300 bytes are logged and the arm's yes is carried "
            "into --approved-by."
        ),
    },
    {
        "id": "GAP-V5-QUEUE-PRELUDE-BLOCKED-ON-ORDER-DIGEST-2026-10-02",
        "tool": "campaign/post-f-queue/v5_prelude_mac.py (not written)",
        "question": (
            "The v5 Mac prelude prints per-seed corpus.plan_order_digest for plan seeds 0-4 and "
            "asserts the seed-invariant plan shape by calling L-v5-train's plan_order_digest and "
            "seed_arms (tools/real_ft_run.py). Is that code committed and merged into v5-build?"
        ),
        "answer": (
            "Not when this lane closed: L-v5-train's --batch-order seed, plan_seed_for, "
            "plan_order_digest and seed_arms were written but uncommitted (the lead's relay, "
            "2026-10-02), and the lane was told not to merge l-v5-train-build itself. The prelude "
            "is therefore not written. The waiters already pin and check its record "
            "(v5_prelude_check: ok, code_commit = V5_LANE_AT, batch_order seed, five distinct "
            "64-hex plan_order_digest entries for plan seeds 0-4, shape_equal_across_seeds true, "
            "the box train header's shard and data hashes, the box tokenizer, "
            "option_permutation_seed iff C1, and argv recipe flags equal to V5_RECIPE), so a "
            "prelude that writes those fields satisfies them."
        ),
    },
]


def main() -> int:
    known = {r["id"] for r in read_gaps(ROOT / "gaps.jsonl")}
    for rec in RECORDS:
        if rec["id"] in known:
            print(f"{rec['id']} already recorded; not appended twice")
            continue
        append_gap({**COMMON, **rec}, path=ROOT / "gaps.jsonl")
        print(f"appended {rec['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
