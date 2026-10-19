# Fable rulings, 2026-10-18

Asked by session lappi-decision-ce after the owner said "Commit and merge all changes and next wave; for advice ask fable" (2026-10-18). Fable answered read-only. The body below is Fable's answer, copied verbatim.

Read-only ruling under the standing delegation (`docs/promotion-decisions.json:14`, "For decisions ask fable and proceed with them"). Navigation: DevMap on the main checkout is fresh (`devmap_status`: 20,662 symbols, generation 3939) and answered `fit_span_entry` and `with_backend`'s callers (83 edges, truncated at 80, `walk_incomplete` set). Everything else below is `rg`/read, not graph-confirmed.

## Q1. Unbuilt Rust: (a) hold, and the real blocker is not the lock

**Decision.** W2-a and W2-c stay unmerged until `cargo test -p qd-preflight` / `-p qd-mutate` ran. No "merge labelled unverified": `make gates` runs `cargo test --workspace` (Makefile:152), so one compile error on main turns every lane's gate red, which is v1's shape, not a label. "Merge all changes" is honoured by merging the moment the compile is green, not by merging first. `cargo check -p` **is** heavy under the header's own definition ("a cargo build or test", mac_heavy.sh:6); no exemption today.

**Reason.** The contamination premise is false but the conclusion holds for a worse reason: the Mac is at load 122–264 against mac_heavy's `MAX_LOAD=12` (gdn-mem.log, 00:03Z–01:14Z), from tessl/GitPulse/Go builds in *other* repos that this lock does not cover (`ps`: rustc in `tessl/.claude/worktrees/*`, `tessl-eg2`, GitPulse clippy, Go `-race` tests). The 2026-10-02 panic was at ~350. Adding a build there is the risk; the lock is incidental.

**Evidence.** mac_heavy.sh:6, :59, :123; Makefile (17cc878):152; W2-c commit 37e6df1 ("UNBUILT ... not wired into lib.rs"); GAP-W2C-CARGO-NOT-RUN-MAC-HEAVY-LOCK-HELD-2026-10-17.

## Q2. The holder: do not ask to pause; ask the owner to look

**Decision.** Wait; never break the lock. But tell the owner the facts, because gdn-mem has **done no work**: its only child is `sleep 20`, `mem-status.txt` does not exist (the script's first line, `: > "$STATUS"`, never ran), and the log is 3 h of "holds the lock but waits: machine busy". At `MAX_WAIT_S=14400` it exits 76 at ~02:08Z and frees the lock without running. Pausing it changes nothing; the owner's decision is whether to stop the sibling repos' builds, which no Lappi session may kill (hard constraint: explicit PIDs only). Record a gap: the one-heavy-job lock binds Lappi only.

**Evidence.** `pgrep -P 97046` → one `sleep`; `ojas/target-baseline/bench-prep/gdn-mem.log` tail; mem_runs.sh:25; mac_heavy.sh:89, :264.

## Q3. K = 2 rows per (cluster_key, class), as a data-build parameter

**Decision.** K = 2. One function contributes at most 2 rows of each label class (stub, logic, cosmetic, clean), so ≤ 8 rows total, inside the existing 32-per-file cap (generate.rs:69). Carried in the v6 cap config beside the stated zero for `knowledge.multiple_choice` (plan §230), passed to `--max-per-function` which keeps no default. This is a construction cap, not an eval gate, so rule 2 is not moved; the clean twin counts as its class so a body cannot be "clean" three times.

**Reason.** 2 rather than 1 lets two operators of one class appear on one body, so the model cannot learn body ⇒ operator; more than 2 reinstates the M4 weight problem the key exists to fix.

**Evidence needed, not to set K but to revise it.** The generator writes the per-key histogram (rows per key before and after the cap, rows dropped per class) into the pool manifest. The first v6 build is the measurement; revising K after reading it is a human decision.

**Evidence.** GAP-W2C-FUNCTION-CAP-K-NEEDS-A-HUMAN-2026-10-17; cluster.rs header (37e6df1); HANDOFF/data-clean-v6-plan-2026-10-06.md:62.

## Q4. Span check: per-pointer ECE at the registered threshold

**Decision.** The span check is the letter check applied to the thing a pointer is: a choice over rows. Confidence = each pointer's top probability after the fitted temperature; correct = that pointer hit its gold line; both pointers pooled (2n samples, exactly as `fit_span_entry` already pools `p_gold` for the conformal cutoff, calibration_fit.rs:806–811). `ece_state` over it: 15 bins, `ECE_THRESHOLD` 0.05, `ECE_MIN_SAMPLES` 100 or NOT_RUN. Recorded at `/span/scored/ece_after` with the same `EceState` shape, so `calibration_gate.rs:200` reads it unchanged. No new threshold is set; the existing registered one is applied to a second slot kind. `pointer_conformal_coverage` stays reported, not gated (its target 1−α is not registered as a gate).

**Must be built first.** `SpanScore` carries `p_gold_*` but not `p_top_*` (calibration_fit.rs:724–744); `score_span_rows` has to expose them. Failing pre-fix test: a well-calibrated synthetic span fit is refused by qd-export today; after the fix it passes and a miscalibrated one is refused.

**Evidence.** GAP-QD-CALIB-FIT-RECORDS-NO-ECE-FOR-A-FITTED-SPAN-ENTRY-2026-10-17; qd_calib_fit.rs:614–647; calibration_fit.rs:54–62, :504–521.

## Q5. Calibration-pin proposal: proceed

**Decision.** Proceed under delegation. The security rule asks first only when a change **widens** access; this one narrows (fewer runtimes build, no refusal becomes an admission). The remaining "needs the human" reason in the proposal is file ownership (`runtime.rs`, `fixtures.rs`), which is a ListAgents/SendMessage matter, not a ruling.

**Conditions.** `assemble` is the backstop; keep `from_release`/`from_ensemble` checks for their messages. DevMap lists the callers to re-examine (truncated 80/83): `calib_fit_roundtrip.rs::runtime_with` (a fitted table over the reference backend: the "passes only because nothing checks" case), `calibration_hash_pin.rs::the_pin_of_a_table_that_is_not_being_served_is_refused`, `refusal_is_not_noul.rs::a_hash_mismatch_does_not_read_as_model_humility`. Prefer making the test backend declare the fitted table's hash over adding an unchecked constructor; a new seam is the defect the proposal removes. [W8-a citation tag, not Fable's text: GAP-FABLE-RULINGS-FIGURES-FROM-SOURCES-OUTSIDE-THE-REPO-2026-10-18]

**Evidence.** AUDIT/harden-2026-10-17/calib-pin-proposal.md; `devmap_impact with_backend`.

## Q6. The checkout hook: install it

**Decision.** Install a project `PreToolUse` Bash hook that exits 2 on `git checkout|switch|branch -f|stash|add -A` when `git status --porcelain --untracked-files=no` is non-empty. Memory is not enough: the 15:21 actor was unattributed, and this memory lives under `.claude-lappi`, so a session on another config dir never reads it. A hook binds every Claude Code session in the repo; it cannot bind a human shell or other tools, and say so in its message. Note `.claude/` is harness-protected from writes (mac_heavy.sh:13), so the file may need the owner's hand.

**Evidence.** memory `commit-before-switching.md`; `.claude/` holds no settings.json (`ls`).

## Q7. Next wave, ranked by likelihood of recurring in the next train

1. **`started` row refused by the Rust ledger mirror and campaign_driver** (GAP-STARTED-STATUS-UNKNOWN-TO-RUST-LEDGER-MIRROR-AND-CAMPAIGN-DRIVER-2026-10-17). The new "started before run" guard writes a row `qd-train/src/ledger.rs` `Status` cannot parse; the first run under the guard breaks the trainer's ledger read. Certain.
2. **Span null absent from FT rows** (GAP-FT-ROWS-CARRY-NO-SPAN-NULL-2026-10-17). Human-ruled gate; `real_ft_run` writes no `val_span_pointing_added_line_chance`, so every next eval row is NOT_RUN on it and nothing can promote. Certain.
3. **Red suite on main**: `ckpt_average` default refusal breaks `test_backbone` callers (GAP-CKPT-AVERAGE-DEFAULT-REFUSAL-BREAKS-OTHER-CALLERS-2026-10-17) and `test_campaign_driver` is load-sensitive (GAP-CAMPAIGN-DRIVER-TESTS-FAIL-UNDER-MACHINE-LOAD-2026-10-17). A gate that is red for unrelated reasons teaches people to ignore red, v1's habit. Certain at next `make gates`.
4. **Shard-header fingerprint drift** (GAP-LDATA-ALL-89-SHARD-HEADERS-DRIFT-FROM-QD-DATA-HEAD-2026-10-17). Every `qd_data/*.py` edit makes every pinned shard set unreadable; the v6 rebuild then any fix repeats it. Fix the class: fingerprint the modules that determine shard content, as a named list. Very likely.
5. **Paid-box launchers outside the driver**: `post_f_common.sh`, `v5_common.sh` (all `box_q_*.sh`), `tools/perf_tierb_outcome.sh` (real_ft_run at $2.29/h, :57–64) do not source `tools/require_driver.sh`. One `. require_driver.sh` line each plus a test that the pre-fix script runs unguarded. Medium.
6. **Release-check seam**: train manifest records slot shapes per family (canonical owner), qd-export reads them instead of `--trained-slot-shapes`; the ensemble table through the same gate; a tracked request builder with `--min-answered` recorded (four L-release gaps). Release-time, so last.

## Lead's notes on acting on these (lappi-decision-ce)

- **Q7.1** is already owned by the other session (lappi-decision-10). Its wave-2 lane adds `Status::Started` to the Rust mirror on wave-integration-2026-10-17. The campaign_driver half was merged with L-ledger (f4cbc8f).
- **Q7.3, test_backbone half:** already fixed on main by W2-b (a084326, merged fd0a9d6). Under torch it passes (258 passed, 3 skipped on merged main). The load-sensitive campaign_driver half is still open.
- **Gap ids in Q7:** Fable's answer gave truncated ids for items 1, 2, 3 and 4. They are expanded above to the full ids in gaps.jsonl, and every id was checked to exist there. Nothing else in Fable's text was changed. The two GAP-W2C ids (Q1, Q3) were written by lane W2-c on its unmerged commit 37e6df1; their two lines were appended to main's gaps.jsonl so the citations resolve.
