# W2c gap triage, 2026-10-21 (lane lane/ft-5fdca0a2, card ft-5fdca0a2e8de09d94238221453e129f2)

Session lappi-decision-0f (Opus) lead; this lane Sonnet. Base d09222d. Only `gaps.jsonl` and this file change; no code.
Every append went through `qd_train.gaps.append_gap` (validated, one line, `O_APPEND`, fsync).

## Counts, same function before and after

Rule used: `qd_train.gaps.read_gaps` / `current_records` (last line wins), status in `OPEN_STATUSES`, `opened` in 2026-10-17..19,
minus ids another record `supersedes` or `follows` (the card's own words). Same code for both numbers (scratch script, not committed).

| | open ids dated 10-17..19 | all open ids in the ledger |
|---|---|---|
| before (d09222d, 1815 lines) | **78** | 540 |
| after (this branch, 1872 lines) | **47** | 511 |

The card says 74 at 5e1a2d3. The same rule gives 78 at d09222d, so four ids were opened or reopened between the two commits
(not identified; the counting rule is the card's). All 78 are in the table below.
Delta: 21 folded and 10 closed (1 resolved, 9 resolved-with-residual) leave the count; 24 kept open (23 owner records plus the re-verified oracle-shards gap) and 23 owned by other lanes stay in it.
The all-open column falls by 29, not 31, because two new open canonicals were created (below).

## What is NOT claimed

- 23 ids are owned by lanes L2, L3, L4, L5, W2a and W2b. They were not triaged here and count as kept open with that owner.
- 23 kept-open records were given an owner and a note only. Their own answers were not re-verified by a command, and each record says so.
- Nine of the ten closures are resolved-with-residual; each record's `residual` names what the evidence does not cover.
- The card's "74" and this lane's "78" differ; see above.

## Evidence (every command run this session, 2026-10-21, on lane/ft-5fdca0a2 at d09222d)

- R-LORA: `bash tools/mac_heavy.sh w2c-lora-driver-only make torch-pytest` with `PYTEST_ADDOPTS=-k lora_reference_driver`: exit 0, 71 passed, 1 skipped, 6153 deselected.
- R-TORCH: `bash tools/mac_heavy.sh w2c-torch-pytest make torch-pytest` with `PYTEST_ADDOPTS=-k "lora_reference or ckpt_average_lora or test_backbone"`: exit 0, 218 passed, 1 skipped, 6006 deselected.
- R-CARGO: `MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh w2c-qd-mutate cargo test -p qd-mutate`: exit 0, 275 passed, 0 failed; tests/cluster_key.rs 7 passed.
- R-GUARD: `uv run ... python -m pytest python/tests/test_launch_driver_guard.py python/tests/test_native_minhash_memory.py python/tests/test_pin_gate.py` under mac_heavy: exit 0, 42 passed (three files together).
- R-BAND: `load_decision_pool(build/pools/pool-v6-decisions-v1, config=DataConfig())` plus `check_pairwise_length_balance` under mac_heavy: exit 0, 214407 rows; train P 0.5499 (3584/2933), val P 0.5484 (136/112), band (0.45, 0.55). A first attempt with the system python3 failed on `dataclass(slots=True)` (wrong interpreter) and was rerun under the ml python.
- R-SHA: python sha256 of build/pools/pool-v6-decisions-v1/examples.jsonl in main's checkout: exit 0, 25a88f6e46af92e82e7edb26c0745552badc9b7ea2f009e10e0f4197948d6e0e, 437636535 bytes, equal to the pin in data/decisions/v6-allocation.json.
- R-LEDGER: python read of ledger/mac-v6-lora2b-shards-2026-10-18.jsonl: 4 rows, eaac82e0 and 6bf4abfc completed (both quick=True), commit da5b27b.
- R-RG: `rg -n pairwise_length data/decisions`: data/decisions/v6-decisions.json:6. R-RG2: `rg -c 'require_driver|CAMPAIGN_DRIVER' tools --glob 'launch_*.sh'`: 9 files, 1 match each.
- Re-verified OPEN: `tools/qd_train_oracle_shards.py --out target/w2c-oracle-tiny` under mac_heavy: exit 1, the same not_run ValueError the record names.

## New records created

- GAP-GITPULSE-INSIGHTS-FACETS-SHED-UNDER-GIT-SPAWN-LOAD-2026-10-19 and GAP-DEVMAP-INDEX-STALE-AGAINST-SOURCE-TREE-IN-LANE-NAVIGATION-2026-10-19:
  open canonicals for two navigation limits no ft-5c92df4c canonical covered (insights shed under load; index stale against the tree).
  Six duplicates fold into them. The other folds go to GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 and its
  component canonicals; the three shell-not-indexed records fold into GAP-DEVMAP-DOES-NOT-INDEX-SHELL-MAC-HEAVY-CARGO-SLOTS-2026-10-18, which stays open for the Fable Q17 a ruling.
- Not folded and why: GAP-W2-LANES-NAVIGATED-BY-RG-WITHOUT-DEVMAP-UNRECORDED-2026-10-17 (its open question is enforcement) and
  GAP-DECIDE-NO-DEVMAP-STORE-2026-10-18 (a sibling repo, not an agent worktree).
- The wave-2 collision-risk-deferred record (dated 2026-10-21, appended on main's checkout): absent from this base (rg over gaps.jsonl), skipped.
- The two Fable figure gaps stay open as unverified: no source file for 2729/2888, the gdn-mem log, the 80/83 envelope or the 48.28 load is in the repo; none was copied in.

## Owner and card routing

Owner-only items point at ft-8139b5b1f204f034a4642dbd02d76123 (item numbers in each record). GAP-W5A-ORACLE-SHARDS-SCRIPT-REFUSES-NOT-RUN-MANIFEST-AT-MAIN-2026-10-18 goes to ft-cc9fac73e1b078b5ab412748e0134f64.
Three live items (raw containment CLI control, train-data guard shard sets, prereg-bind shard hash) fit no item
on ft-c5ea3e71d03046023cfc3be19c93bc89 exactly; each record names the nearest card. The tape-provider gap points at ft-d066187fdc2e8fab5ada677e516113c8. Adding items is the lead's call.

## Table of the 78 ids

| id | action | evidence or owner | commit |
|---|---|---|---|
| GAP-AGENT-WORKTREE-ISOLATION-CUTS-FROM-STALE-ORIGIN-MAIN-2026-10-17 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 1 (push main)) |  |
| GAP-BOX-QUEUE-WAITERS-NEED-THE-GUARD-DEPLOYED-2026-10-18 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 9 (deploy require_driver.sh)) |  |
| GAP-CKPT-AVERAGE-DEFAULT-REFUSAL-BREAKS-OTHER-CALLERS-2026-10-17 | closed (resolved-with-residual) | R-TORCH (full text in the appended record) | a084326 |
| GAP-D1-EVAL-PROMPT-VARIANT-MUST-MATCH-SERVED-2026-10-18 | kept open | owner: human (owner of the Vani local shadow; d1 eval amendment on ft-3791917025307190c9112b614973a50b) |  |
| GAP-DECIDE-LAPPI-BACKEND-NAME-HIDES-MODEL-2026-10-18-NARROWED | kept open | owner: human (third-party decider runtime, docs/third-party-decider-runtime-2026-10-18.md) |  |
| GAP-DECIDE-NO-DEVMAP-STORE-2026-10-18 | kept open | owner: human (index /Users/bharath/Code/devtools/decide in DevMap) |  |
| GAP-DEVMAP-AND-GITPULSE-UNAVAILABLE-LANE-LEDGER-2026-10-17 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-NO-STORE-IN-AGENT-WORKTREE, GAP-GITPULSE-INSIGHTS-FACETS-SHED-UNDER-GIT-SPAWN-LOAD-2026-10-19, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-DEVMAP-DOES-NOT-INDEX-SHELL-MAC-HEAVY-CARGO-SLOTS-2026-10-18 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 4 (machine load; Fable Q17 a, CARGO_SLOT_HELD)) |  |
| GAP-DEVMAP-DOES-NOT-INDEX-SHELL-MAC-HEAVY-FIFO-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-DOES-NOT-INDEX-SHELL-MAC-HEAVY-CARGO-SLOTS-2026-10-18 |  |
| GAP-DEVMAP-NO-INDEX-FOR-AGENT-WORKTREES-2026-10-17 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-NO-STORE-IN-AGENT-WORKTREE, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-DEVMAP-NO-INDEX-W11A-2026-10-19 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-NO-STORE-IN-AGENT-WORKTREE, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-DEVMAP-NOT-USEFUL-FOR-SHELL-MAC-HEAVY-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-DOES-NOT-INDEX-SHELL-MAC-HEAVY-CARGO-SLOTS-2026-10-18 |  |
| GAP-DEVMAP-STALE-INDEX-CANNOT-ANSWER-SYNTH-CAREER-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-INDEX-STALE-AGAINST-SOURCE-TREE-IN-LANE-NAVIGATION-2026-10-19, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-FABLE-RULINGS-C-V5-WALL-CLOCK-FIGURE-NOT-IN-REPO-2026-10-18 | kept open | owner: human (copy the v5 build's launch log or ledger row into AUDIT/ with a [V: path] tag) |  |
| GAP-FABLE-RULINGS-FIGURES-FROM-SOURCES-OUTSIDE-THE-REPO-2026-10-18 | kept open | owner: human (copy the ojas gdn-mem.log line, the DevMap envelope behind 80/83 and a load reading into AUDIT/ with [V: path] tags) |  |
| GAP-FT-ROWS-CARRY-NO-SPAN-NULL-2026-10-17 | not triaged here | owned by lane L2, kept open with that owner |  |
| GAP-GITPULSE-DEFERRED-AMENDMENT2-DRAFT-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-GITPULSE-INSIGHTS-FACETS-SHED-UNDER-GIT-SPAWN-LOAD-2026-10-19, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-L1-D17-GATES-UNRUN-2026-10-17 | not triaged here | owned by lane L2, kept open with that owner |  |
| GAP-L2-LORA-TESTS-NOT-RUN-2026-10-17 | not triaged here | owned by lane L2, kept open with that owner |  |
| GAP-L2-LORA-TRAINER-DISPATCH-AND-RUN-WIRING-2026-10-17 | not triaged here | owned by lane L5, kept open with that owner |  |
| GAP-L3-LORA-DRIVER-BUILT-2026-10-17 | not triaged here | owned by lane L2, kept open with that owner |  |
| GAP-L3-LORA-DRIVER-GITPULSE-LISTAGENTS-2026-10-17 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-GITPULSE-INSIGHTS-FACETS-SHED-UNDER-GIT-SPAWN-LOAD-2026-10-19, GAP-LISTAGENTS-UNAVAILABLE-TO-SUBAGENT-2026-10-06, GAP-DEVMAP-WALK-INCOMPLETE-OR-TRUNCATED-IN-LANE-NAVIGATION-2026-10-03, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-LAUNCH-SCRIPTS-RUN-A-PAID-BOX-OUTSIDE-THE-DRIVER-2026-10-17 | closed (resolved-with-residual) | R-RG2 + R-GUARD (full text in the appended record) | a084326 |
| GAP-LDATA-ALL-89-SHARD-HEADERS-DRIFT-FROM-QD-DATA-HEAD-2026-10-17 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 14 (stale shard headers: rebuild, or rule allow_stale_code)) |  |
| GAP-LDATA-MUTANT-CLUSTER-KEY-CAP-NOT-BUILT-2026-10-17 | closed (resolved) | R-CARGO (full text in the appended record) | 50d43b5, 59edb10 |
| GAP-LDATA-RAW-CONTAINMENT-CLI-CONTROL-IS-TARGET-SELF-PLANT-2026-10-17 | kept open | owner: unowned live engineering: no item on ft-c5ea3e71d03046023cfc3be19c93bc89 or ft-cc9fac73e1b078b5ab412748e0134f64 covers it; nearest card named in triage_note |  |
| GAP-LDATA-TRAIN-DATA-GUARD-SHARD-SETS-HAVE-NO-TEXT-2026-10-17 | kept open | owner: unowned live engineering: no item on ft-c5ea3e71d03046023cfc3be19c93bc89 or ft-cc9fac73e1b078b5ab412748e0134f64 covers it; nearest card named in triage_note |  |
| GAP-LOPS-NAVIGATION-RG-AND-FILE-READS-ONLY-2026-10-17 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-LORA-FT-ROW-HAS-NO-PRODUCER-AND-MUST-CARRY-TRAIN-MANIFEST-SHA-2026-10-18 | not triaged here | owned by lane L5, kept open with that owner |  |
| GAP-LORA2B-AUDIT2-LANE-NO-LISTAGENTS-DEVMAP-FRESHNESS-DISAGREES-2026-10-17 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-LISTAGENTS-UNAVAILABLE-TO-SUBAGENT-2026-10-06, GAP-DEVMAP-MCP-AND-CLI-DISAGREE-ON-ANALYZER-FRESHNESS-2026-10-14, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-LORA2B-QUANT-GATE-TOOL-MISSING-2026-10-17 | not triaged here | owned by lane L4, kept open with that owner |  |
| GAP-LORA2B-REDUCED-RUN-DEFAULT-CAP-1800S-2026-10-17 | not triaged here | owned by lane L2, kept open with that owner |  |
| GAP-LORA2B-TAPE-PROVIDER-MISSING-2026-10-17 | kept open | owner: ft-d066187fdc2e8fab5ada677e516113c8 (switch qd-train onto ojas-train at ojas 4123dfb, in progress) and ft-3163ce2315f5d63c03d05de888434469 |  |
| GAP-LORA2B-TORCH-DRIVER-MISSING-2026-10-17 | not triaged here | owned by lane L2, kept open with that owner |  |
| GAP-M12-TASK-HOLDOUT-CONTENT-OVERLAP-IS-EVERY-COMMITPACKFT-TRAINING-ROW-2026-10-17 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 2 (held-out partition ratio)) |  |
| GAP-MAC-HEAVY-HOLDER-WAITS-ON-MACHINE-WIDE-CARGO-SLOT-GATE-2026-10-18 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 4 (machine load)) |  |
| GAP-MAC-HEAVY-LOCK-AND-LOAD-GATE-BIND-LAPPI-ONLY-2026-10-18 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 4 (machine load)) |  |
| GAP-MEMORY-HOST-CEILING-CHECK-COVERS-ONLY-TORCH-MPS-ADMISSION-2026-10-17 | not triaged here | owned by lane W2a/W2b, kept open with that owner |  |
| GAP-MEMORY-PREDICTOR-UNDER-PREDICTED-EVERY-RECORDED-GH200-FOOTPRINT-2026-10-17 | kept open | owner: human (GH200 box session, ft-12da8e67e32f4439bbb286ee0ec2acd4: needs the cost yes) |  |
| GAP-OJAS-PIN-4123DFB-SOURCE-DIFF-AND-PIN-GATE-PENDING-2026-10-19 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 11 (ojas pin: make pin / qd-pin run)) |  |
| GAP-POOL-GUARD-OVERLONG-LINE-UNSCANNED-PYTHON-AND-RUST-2026-10-17 | not triaged here | owned by lane W2a/W2b, kept open with that owner |  |
| GAP-PREREG-BIND-SHARD-HASH-IS-THE-RECORDED-HEADER-FIELD-2026-10-17 | kept open | owner: unowned live engineering: no item on ft-c5ea3e71d03046023cfc3be19c93bc89 or ft-cc9fac73e1b078b5ab412748e0134f64 covers it; nearest card named in triage_note |  |
| GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17 | not triaged here | owned by lane W2a/W2b, kept open with that owner |  |
| GAP-QD-EXPORT-TRAIN-MANIFEST-BINDING-BLOCKS-RE-EXPORT-OF-EXISTING-MODELS-2026-10-18 | not triaged here | owned by lane W2a/W2b, kept open with that owner |  |
| GAP-QD-EXPORT-TRAIN-MANIFEST-FILE-SHA-NOT-IN-ANY-TAMPER-EVIDENT-RECORD-2026-10-18 | not triaged here | owned by lane W2a/W2b, kept open with that owner |  |
| GAP-QD-RELEASE-CHECK-MIN-ANSWERED-HAS-NO-RECORD-2026-10-17 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 8 (N for --min-answered)) |  |
| GAP-STRESS-V6-CLI-LIMIT-1800S-LOAD-SENSITIVE-2026-10-18 | not triaged here | owned by lane L4, kept open with that owner |  |
| GAP-W10A-DEVMAP-NO-INDEX-2026-10-19 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-NO-STORE-IN-AGENT-WORKTREE, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W10A-GITPULSE-INSIGHTS-SHED-2026-10-19 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-GITPULSE-INSIGHTS-FACETS-SHED-UNDER-GIT-SPAWN-LOAD-2026-10-19, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W11B-DEVMAP-NO-INDEX-IN-WORKTREE-2026-10-19 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-NO-STORE-IN-AGENT-WORKTREE, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W11B-VAL-ROWS-PRODUCER-NEEDS-REAL-FT-RUN-2026-10-19 | not triaged here | owned by lane W2a/W2b, kept open with that owner |  |
| GAP-W2-2-LORA-A-DIGEST-DEPENDS-ON-MODULE-REGISTRATION-ORDER-2026-10-18 | not triaged here | owned by lane L2, kept open with that owner |  |
| GAP-W2-2-LORA-DRIVER-TORCH-SUITE-NOT-RUN-LOCK-NEVER-GRANTED-2026-10-18 | closed (resolved-with-residual) | R-LORA (full text in the appended record) | 08436f8 |
| GAP-W2-LANES-NAVIGATED-BY-RG-WITHOUT-DEVMAP-UNRECORDED-2026-10-17 | kept open | owner: human (nothing enforces CLAUDE.md rule 1; it rests on the lane prompt) |  |
| GAP-W2C-CARGO-NOT-RUN-MAC-HEAVY-LOCK-HELD-2026-10-17 | closed (resolved-with-residual) | R-CARGO (full text in the appended record) | 50d43b5 |
| GAP-W2C-FUNCTION-CAP-K-NEEDS-A-HUMAN-2026-10-17 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 7 (K)) |  |
| GAP-W2D-CLINC-TEST-ITEMS-LABEL-CONFLICT-WITH-TRAIN-2026-10-19 | kept open | owner: human (owner ruling 2026-10-19: REPORT ONLY, test set never edited) |  |
| GAP-W2D-NATIVE-MINHASH-MEMORY-MEASURED-2026-10-18 | kept open | owner: human (owner decision on ft-8139b5b1f204f034a4642dbd02d76123, item 5 (disk under mac_heavy's 15 GB floor) then live engineering) |  |
| GAP-W2D-V6-SCAN-FOOTPRINT-OVER-MAC-HEAVY-CAP-2026-10-18 | not triaged here | owned by lane L3, kept open with that owner |  |
| GAP-W2D-V6-SHARD-REBUILD-NOT-RUN-2026-10-17 | closed (resolved-with-residual) | R-LEDGER (full text in the appended record) | da5b27b |
| GAP-W3A-DEVMAP-INDEX-STALE-NOT-CONSULTED-2026-10-17 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-INDEX-STALE-AGAINST-SOURCE-TREE-IN-LANE-NAVIGATION-2026-10-19, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W3B-DEVMAP-AND-LISTAGENTS-NOT-CONSULTED-2026-10-17 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W3B-NO-COMMITTED-PRODUCER-WRITES-PAIRWISE-LENGTH-BALANCE-2026-10-17 | closed (resolved-with-residual) | R-BAND + R-RG (full text in the appended record) | 275989b |
| GAP-W3B-V5-DECISION-POOL-FAILS-THE-V6-LENGTH-BAND-2026-10-17 | closed (resolved-with-residual) | R-BAND (full text in the appended record) | 7e20585 (merge W6-c), 275989b |
| GAP-W4A-DEVMAP-CALLERS-ANSWERED-BY-GREP-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-WALK-INCOMPLETE-OR-TRUNCATED-IN-LANE-NAVIGATION-2026-10-03, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W4A-SPAN-REFIT-ROWS-CARRY-NO-SPAN-NULL-2026-10-18 | not triaged here | owned by lane L2, kept open with that owner |  |
| GAP-W4B-NAVIGATED-BY-RG-NOT-DEVMAP-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W4B-PIN-GATE-BASELINE-NOT-RUN-IN-AN-EXTRACTED-TREE-2026-10-18 | closed (resolved-with-residual) | R-GUARD (full text in the appended record) | b3291dc |
| GAP-W4C-CONTAINMENT-ATTESTATION-STILL-FINGERPRINTS-WHOLE-QD-DATA-2026-10-18 | not triaged here | owned by lane L3, kept open with that owner |  |
| GAP-W4C-DEVMAP-NOT-CONSULTED-IMPORT-CLOSURE-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W4C-V1-HEADERS-STILL-REFUSED-UNDER-V1-RULE-2026-10-18 | not triaged here | owned by lane W2a/W2b, kept open with that owner |  |
| GAP-W4D-DEVMAP-NOT-CONSULTED-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W4E-DEVMAP-NO-TRAVERSAL-START-FOR-RUNTIME-WITH-BACKEND-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-DEVMAP-WALK-INCOMPLETE-OR-TRUNCATED-IN-LANE-NAVIGATION-2026-10-03, GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W5A-ORACLE-SHARDS-SCRIPT-REFUSES-NOT-RUN-MANIFEST-AT-MAIN-2026-10-18 | kept open | owner: ft-cc9fac73e1b078b5ab412748e0134f64 (v6-default fallout: red Python tests since DataConfig() became the v6 rules) |  |
| GAP-W6C-DEVMAP-AND-GITPULSE-NOT-CONSULTED-2026-10-18 | folded (closed, consolidated-duplicate) | fold, no re-run needed: duplicate_of GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03 |  |
| GAP-W6C-QDPREP-CARGO-TEST-FAILS-3-OF-287-IN-BUILD-ROW-2026-10-18 | not triaged here | owned by lane L4, kept open with that owner |  |
| GAP-W6C-V6-POOL-LIVES-IN-WORKTREE-TARGET-NOT-CAMPAIGN-DIR-2026-10-18 | closed (resolved-with-residual) | R-SHA + R-BAND (full text in the appended record) | 7e20585 |
| GAP-W9B-NO-TRACKED-VAL-ROWS-FILE-WRITER-2026-10-18 | not triaged here | owned by lane W2a/W2b, kept open with that owner |  |

## Open gaps from this lane

None new beyond the two canonicals. Gap ids above are the status of record.

## First command for the next lane

`/Users/bharath/Code/research/Lappi-decision/.venv/bin/python -m pytest python/tests/test_gaps_ledger.py -o addopts= -q`
