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
            "known. Unexercised on any machine: these argv under the real main() (--batch-order "
            "itself is on v5-build at 3223417 with L-v5-train's own tests; the waiters' argv were "
            "not parsed by it), the needle control and trajectory-ood argv against v5 code, "
            "ft_linear_control.py on v5's split with --exclude-identity-keys, and the "
            "qd-post-f-rules-v5 build itself (stubbed)."
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
        "id": "GAP-V5-QUEUE-PRELUDE-NOT-RUN-ON-V5-DATA-2026-10-02",
        "tool": "campaign/post-f-queue/v5_prelude_mac.py; python/tests/test_v5_prelude_mac.py",
        "question": (
            "Does the v5 Mac prelude pass on v5's real shard set and argv: five distinct plan "
            "order digests, one shape across plan seeds 0-4, C1's sentence, the noul-weight "
            "count, its peak memory, and its run time?"
        ),
        "answer": (
            "Not run: v5's shard set does not exist yet (build_order steps 1-4). It ran only on "
            "L-v5-train's toy corpus (test_real_ft_rungd_flags's corpus and _patch_module, CPU, "
            "no tower), where its record carries plan_order_digest at plan seeds 0-4 (five "
            "distinct), one SeedArms.shape(), and passes v5_common.sh v5_prelude_check once two "
            "fields the toy cannot supply are given (the argv's v5 recipe, which needs the 2B "
            "tower's layers, and a tokenizer sha: the tiny snapshot has no tokenizer.json). On "
            "v5's real plan it holds main's seed-0 plan beside one more seed's at a time "
            "(Fable's figure: ~1.57 GB and ~38 s per plan; [I] for v5's 1.16x larger set); "
            "C1's validation loop was not exercised (the toy run has no "
            "--option-permutation-seed). Run it under tools/mac_heavy.sh."
        ),
    },
    {
        "id": "GAP-V5-QUEUE-MUTATION-ROUND-2-NOT-RUN-2026-10-02",
        "tool": "AUDIT/v5-queue-2026-10-02/mutations.py",
        "question": (
            "Does the per-gate wait test (test_v5_waits_for_each_gate_on_its_own) catch the "
            "mutants that survived round 1 (M18 no j5pp wait, M19 no j6g wait) and M27 (no j6a "
            "wait)?"
        ),
        "answer": (
            "Not measured. Round 2 took the heavy-job lock at 19:24:05Z and the Mac "
            "kernel-panicked at 19:24:56Z; the lead then barred mutation testing on this Mac "
            "for host stability after the 2026-10-02 launchd panics "
            "(AUDIT/mac-stability-2026-10-02/report.md). Its log stops after M1 and M2 "
            "(mutations-round2-NOT-RUN-interrupted.txt) and is not a result. Round 1 stands: 24 "
            "of 26 caught. The four per-gate cases pass on the real waiter; that each fails "
            "with its own wait removed is inferred, not run."
        ),
    },
    # Fable's rulings A (the R9 reason marker) and B (the read-race retry), 2026-10-02.
    {
        "id": "GAP-L-V5-QUEUE-AB-NAVIGATION-2026-10-02",
        "tool": "DevMap MCP, GitPulse MCP",
        "question": "What could DevMap and GitPulse Insights not answer for rulings A and B?",
        "answer": (
            "(1) devmap_search 'v5_rule' on the main checkout's built index returned 0 items "
            "(truncated false): the shell functions are not indexed, and this worktree has no "
            "store. The callers of v5_rule and v5_rules_call (box_q_v5.sh R9; box_q_v5s34.sh "
            "seeds34; box_q_v5nw.sh --room and the arm's reading; box_q_v5j5.sh ft-rows and "
            "eval-row) rest on rg over campaign/post-f-queue, re-run with -uu (same files plus "
            "the test file and the handoff), not on the graph. (2) gitpulse_insights: every "
            "facet ok, 6 of 6 worktrees scanned, the only cross-worktree overlap Cargo.lock "
            "(not touched). (3) ListAgents was not loadable in this session; the lead was "
            "reached with SendMessage."
        ),
    },
    {
        "id": "GAP-V5-QUEUE-READ-RACE-RETRY-NOT-RUN-ON-BOX-2026-10-02",
        "tool": "campaign/post-f-queue/v5_common.sh v5_rules_run, v5_read_race",
        "question": (
            "Does the bounded retry on a half-written last ledger line (Fable's ruling B) behave "
            "on the box with the qd-post-f-rules-v5 build and a real post-seed waiter appending?"
        ),
        "answer": (
            "Not run on the box. Measured on the Mac only: the refusal text from the v5 build's "
            "debug qd-post-f-rules (build/v5-build-wt, sha256 6ec9d128..., read-only on temp "
            "ledgers; AUDIT/v5-queue-2026-10-02/read-race-probe.txt), and the retry against a "
            "stub and a fake that writes and completes a half row in the real ledger file "
            "(python/tests/test_v5_queue_scripts.py). Residual, inferred: if the writer "
            "completes line N and appends another row between the binary's read and the "
            "waiter's line count, N is no longer the last line and the reading holds as "
            "refused (fail-closed). Ledger.append takes LOCK_EX while it writes "
            "(python/qd_train/ledger.py _take_write_lock), so a shared flock around each read "
            "would close the race at its source; not done, the ruling is the retry."
        ),
    },
    {
        "id": "GAP-V5-QUEUE-BOX-WALL-CLOCK-NOT-TRACKED-2026-10-02",
        "tool": "campaign/post-f-queue/v5_common.sh v5_spend_add, v5_total_line",
        "question": "Does the v5 block's running total say what the box bills?",
        "answer": (
            "No, by design (reading 4, confirmed by Fable): $Q/v5.spend and the running total "
            "count GPU-step wall time only. The box bills its wall clock, which adds up to 72 h "
            "(V5_HUMAN_WAIT_MAX_S) for the launch yes and up to 72 h for an R9 hold (each <= "
            "$164.88 at $2.29/h), the hand-offs, the last seed's CPU controls (no timeout in "
            "idle_common.sh controls_block), and whatever follows v5j5.done until the box is "
            "stopped. No waiter tracks it. HANDOFF/v5-queue-2026-10-02.md section 2 reading 4 "
            "has the two numbers side by side and the box command for the bill; projected, not "
            "measured."
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
