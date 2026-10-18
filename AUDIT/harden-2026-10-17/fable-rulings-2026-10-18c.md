# Fable rulings, 2026-10-18 (third round)

Asked by session lappi-decision-ce after the owner said "Commit and merge all changes and next wave and delete old and stale branches; for advice ask fable". The body below is Fable's answer, verbatim. Earlier rounds are in fable-rulings-2026-10-18.md and fable-rulings-2026-10-18b.md.

Read-only, under the standing delegation. Navigation: `devmap_status` fresh (generation 3944, 20,836 symbols); `gitpulse_insights` returned every facet `ok:false` ("git worktree deferred under load"), so nothing on worktrees or collisions is tool-confirmed. The harness refused `git status` in the two foreign worktrees; their dirty state below is the coordinator's claim, not mine.

## Q15. Merge criteria: W4-e merges now; W5-a waits

**Rule.** "Ships with a test that fails against the pre-fix code" (CLAUDE.md:98) is a merge precondition, read together with rule 5 (:73). Satisfied by any of: the new test run on the pre-fix tree; or a run on the pre-fix tree of a test that asserted the defect. A reading of the diff is not a proof.

**W4-e: merge.** The pre-fix evidence already ran: main's own `the_pin_of_the_table_being_served_is_answered` asserted `assert_ne!(table.hash(), runtime.identity().calibration_hash)`, i.e. "mismatch admitted", and `cargo test -p qd-runtime` passed at 1092e60. d7b8307 flips it to `assert_eq!` and adds `with_backend_refuses_a_fitted_table_over_a_backend_declaring_another`, whose `Ok(_) => panic!` arm is the pre-fix path. Merge commit cites the 1092e60 run as the proof; the queued re-run stays queued but no longer gates. Run it against 879933c, not af4bf6d, from a worktree beside `../tessl`.

**W5-a: hold.** Its diff adds `crates/qd-train-metal/tests/run_door.rs` (+37) that has not run; rule 5 binds before the proof question does. Its fails-before cannot be the new test file (it does not compile against main: new `shard_content_modules`, `code_fingerprint_version`). The proof is behavioural: main's `qd-train` from `build/verify-879933c` opening a Python-written v2 set is refused naming an unlisted module; W5-a's reader opens it. One lock window: run_door, then that. Then merge; W2-d's (b) clears.

## Q16. W2-a: merge main into it and compile; do not retire

**Decision.** Keep. `git merge-tree --write-tree main w2a-preflight-2026-10-17` yields a clean tree (no conflicts); main touched only `crates/qd-preflight/src/prereg_bind.rs` (+4/−2) in its files since base 4b1f037; all five binaries its Makefile needs exist (`crates/qd-preflight/src/bin/`: qd-cite, qd-pin, qd-prereg-bind, qd-release-check, qd-train-data). Merge main in as W5-a did (b06e15c), so 17cc878 stays citable from the handoff; rebasing orphans the citation. Then `cargo test -p qd-preflight` plus `test_pin_gate.py`, `test_preflight_targets.py`. `make preflight-train` is the one target that sequences every gate before a launch; v1 launched without it.

## Q17. Load: keep 48; make mac_heavy see the slot gate; two asks

**Decision.** `MAX_LOAD=48` stands. Load at 95 is 5.3/core on 18 cores and memory pressure is the guarded failure; but the real cost of running under load is false timeouts (`stress_v6.rs:34` CLI_LIMIT 1800 s; W5-d's d9bfbcf-vs-879933c timing is not an A/B unless interleaved, so treat its result as `quick`). Raising 48 buys nothing but more such failures. [W8-a citation tag, not Fable's text: GAP-FABLE-RULINGS-FIGURES-FROM-SOURCES-OUTSIDE-THE-REPO-2026-10-18]

**Lappi's own fix.** The slot gate is `~/.claude-bin/cargo`: 2 lockf slots (:48), waits ≤1800 s then runs ungated (:49, :77-79), active only under `CLAUDECODE=1` (:25). So a Lappi holder waits up to 30 min holding the lock, then builds anyway, which is W5-b's defect one layer down (GAP-MAC-HEAVY-HOLDER-WAITS-ON-MACHINE-WIDE-CARGO-SLOT-GATE-2026-10-18). Add "no cargo slot free" to mac_heavy's busy predicate, probed before `mkdir`, with a test that a waiter at 2/2 does not own `$LOCK/owner`. **Ask the owner:** (a) whether a job under the mac_heavy lock should run with `CARGO_SLOT_HELD` set, since the lock already serializes Lappi to one build; (b) sibling-repo builds (GAP-MAC-HEAVY-LOCK-AND-LOAD-GATE-BIND-LAPPI-ONLY-2026-10-18).

## Q18. Branch hygiene

**span-pin-8a531e8-verdicts: keep, ask.** One commit (1d42f37), only `tools/real_ft_run.py` (+21/−20); its source hunk a1b1799 is in main, so it is a recreated cherry on an old base, "local only" (HANDOFF/span-rescore-2026-10-12.md:6). Worktree outside the repo, owner unknown, dirty state unreadable: that is an ask, not a delete.

**build/main-gates-2026-10-15: remove after capture.** fe42017 is an ancestor of main (0 commits outside it); its ledger chain is kept as `ledger/runs.main-gates-2026-10-15.jsonl` (GAP-LEDGER-FORK-MAIN-GATES-2026-10-15; test_ledger.py:1577-1584). Save the three dirty files' diff to `AUDIT/harden-2026-10-17/main-gates-worktree-dirty-2026-10-18.diff`, confirm the ledger rows are in the kept chain, then `git worktree remove --force`. Ask only if a dirty file is not reproducible from the diff.

**Other.** `git branch --merged main` lists only w5d-synth-timeout-2026-10-18, a live lane: delete nothing today. w2c-mutate-cluster-2026-10-17 is in w4d (ancestor); delete it when W4-d merges. The wave-l* branches are the peer's.

## Q19. Next wave, ranked

1. **W5-a to main, then W2-d.** Every rebuilt set is refused until it lands (round 2 Q9, GAP-W2D-V6-SHARD-REBUILD-NOT-RUN-2026-10-17). Certain.
2. **Q14 P1 pairwise rebuild, with the band in a committed config.** `allocate.rs:916` builds `pairwise_length_band: None`; a rebuild without the key reproduces the refused pool (GAP-W3B-NO-COMMITTED-PRODUCER-WRITES-PAIRWISE-LENGTH-BALANCE-2026-10-17, GAP-W3B-V5-DECISION-POOL-FAILS-THE-V6-LENGTH-BAND-2026-10-17). Certain.
3. **real_ft_run.py's 1,800 s default cap** for a non-quick run (GAP-RUN-CONTROL-DEFAULT-CAP-CALLERS-OUTSIDE-L-OPS-2026-10-17): the silent stop of v1, still reachable. Certain on the first Python-driven run.
4. **W2-a compiled and merged** (Q16).
5. **exclusions.py closure fingerprint** (GAP-W4C-CONTAINMENT-ATTESTATION-STILL-FINGERPRINTS-WHOLE-QD-DATA-2026-10-18): fires at the first attestation after any `qd_data` edit, which the v6 build is. Likely.
6. **Box-queue guard deploy** (GAP-BOX-QUEUE-WAITERS-NEED-THE-GUARD-DEPLOYED-2026-10-18): every `box_q_*` waiter exits 78 on the next box unless `require_driver.sh` ships and the waiters run under a campaign unit. Likely at the next rental.

Release-check seam (GAP-QD-EXPORT-NO-TRACKED-PER-FAMILY-SLOT-SHAPE-REGISTRY-2026-10-17, GAP-QD-EXPORT-ENSEMBLE-TABLE-NOT-GATED-2026-10-17, GAP-QD-RELEASE-CHECK-NO-TRACKED-REQUEST-BUILDER-2026-10-17): release-time, below the cut. Pushing main (302 ahead, GAP-AGENT-WORKTREE-ISOLATION-CUTS-FROM-STALE-ORIGIN-MAIN-2026-10-17): the owner's, one command.

## Q20. W2-d data build at MAX_LOAD=48: yes, bounded; memory pressure is the new condition

(Asked after lane W2-d's rebuild waited 4 h at the default MAX_LOAD=12 and exited 76. Fable's answer, verbatim.)

**Decision.** (a) Yes. `MAC_HEAVY_MAX_LOAD=48` in the wrapper's environment (mac_heavy.sh:162), footprint cap untouched, no wait for the owner. (b) Conditions below. (c) Content does not depend on load; timings and failure modes do.

**Reason.** Q8 excluded data builds because their guarded failure is memory, and this one is large: v5's identical argv peaked at ~25 GiB RSS, `ft_split_rows` 26.0-26.5 GiB (HANDOFF/v6-startup-2026-10-04.md:152, :217) against the 28,672 MiB default cap (mac_heavy.sh:34-35), footprint ≥ RSS. That margin is load-independent. What load adds is memguard's critical-pressure rule (policy.go:83-85: after 3 critical samples, the largest process ≥ 2 GiB dies), and a 25 GiB build is the largest process on the box. Both kills are loud (exit 70/79, no row: the row is written only after `run()` returns, real_tokenizer_pipeline.py:4688-4706), so the worst case is a rerun, not a wrong shard. Note the first launch did get the lock at load 10.55 after 65 min and failed at stage 3 on a missing input (`commitpackft-corpus-v3/examples.jsonl`, launch.log 13:41-13:42Z); only chain2 waited 4 h (launch2.log, exit 76).

**Conditions.**
1. `sysctl kern.memorystatus_vm_pressure_level` reads normal at launch; mac_heavy has no pressure gate (:51 is a comment).
2. Stage 0 checks every `CORPUS` path exists before a slot is spent.
3. The full launch line, with the raised value and the worktree HEAD, as the launch log's first line and in the HANDOFF; the ledger row carries neither.
4. Exit 70/76/79 or a hang-detector timeout (PREP_TIMEOUT_S 600 s, pipeline:1939; containment 4 h, containment_scan.py:105; PHASE_LIMIT 3600 s, team.rs:41) means rerun; raising any limit or the cap is the owner's.
5. The row is already `quick=True` (pipeline:4715); the HANDOFF states wall_clock_s was taken under load 17-177 and is not comparable to v5's 2729/2888 s.

**(c) Verified:** containment output is oracle-ordered and "does not depend on `threads`" (containment.rs:16-19, :564); threads default to cores, not load (main.rs:210); minhash canaries re-signed by the Python reference refuse drift (pipeline:1940-1944); two separate v5 processes gave the same canonical digest (v6-startup:217). **Unverified:** `train_examples_export.py` exists only in the worktree, untested (HANDOFF/harden-2026-10-17.md:55); check it orders rows by input, not by time. Proof is a quiet-machine rerun matching `data_snapshot_hash` and shard shas.

## Lead's notes (lappi-decision-ce)

- **Q15, W4-e:** verified before the merge. At 879933c, `crates/qd-runtime/tests/calibration_hash_pin.rs` asserted `assert_ne!(table.hash(), runtime.identity().calibration_hash)`, and the `cargo test -p qd-runtime` run at 1092e60 (verify-main-1092e60, exit 0 for qd-runtime) passed with it. W4-e's diff flips it to `assert_eq!`.
- **Q18, build/main-gates-2026-10-15:** this session's harness refuses git commands in another worktree, so the dirty diff cannot be captured from here. The worktree stays until the owner, or a session rooted in it, captures the diff.
- **Q17 (a) and (b) and pushing main** are owner asks. They are not done under the delegation.
