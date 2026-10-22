# W3a: the 17 "v6-default fallout" reds (ft-cc9fac73), 2026-10-21

Branch `lane/ft-cc9fac73` (from d09222d, not rebased onto main). Session lappi-decision-0f.

## Measured (logs under build/w3a/out/, git-ignored; the host has no ledger rows for these)

- Before, unmodified d09222d, make torch-pytest at load 12, QD_PREP_BIN pre-built:
  - `base.txt`, 11 files: 8 failed, 259 passed, 10 skipped. The worktree has no data, so four
    files skipped (commitpackft download absent).
  - `base3.txt`, after linking the host's `data/pool` files as L3's worktree does (untracked):
    7 failed (3 `test_pipeline_val_shards`, 4 `test_qd_prep_spancheck_parity`), 112 passed, 3
    skipped (`test_real_ft_pieces`: pool-v2 file not yet linked).
  - Load-48 ruling (Fable, 2026-10-21), pre-built QD_PREP_BIN, per-file wrapper `build/w3a/perfile.py`
    (stdlib, killpg, 1800 s per file counted from "holds the lock", exits 70/76/79/timeout = not
    run): `before-summary.txt`.
    `test_split_cache_main.py` RAN, passed on d09222d (load1 46.91 at launch). So it is not red on
    the base tree here; L3's sweep failure is not reproduced (order dependence or the 133-file
    environment, not diagnosed).
- After: see "After" below. At the time of writing the load-48 queue had not delivered.

## Per-test table

| Test | Class | Cause | Fix (commit) | Fails before / after |
|---|---|---|---|---|
| test_pipeline_val_shards (3) | (a) fixture, v6 | `pipeline.run` default v6; R1 refuses a build with no `--defect-class`: "defect_token_share not_run ... Refusing to write a shard set (Fable R1)" (base3.txt) | `v6_data_rules=False` + `legacy_v5()` readers, rule set named (f1514da) | fails before (base3); after: NOT RUN |
| test_qd_prep_spancheck_parity (4) | (a) fixture, v6 | same refusal | `v6_data_rules=False`, docstring names it (f1514da) | fails before; after: NOT RUN |
| test_pre_dedupe_drops | (a) half-migrated fixture | 3 `pipeline.run` calls left on v6 (d4d99ed moved the rest). Measured: same identity keys under v5/v6; v5 dedupe drops val v0 for train twin t0, v6 dedupe drops t0 and keeps v0 | `v6_data_rules=False`, docstring names legacy v5 (f1514da) | fails before (base.txt); after: NOT RUN |
| test_ft_linear_control_split_cache (3) | (c) stale expectation | `row()` expected 1 ledger line; file holds `started` + `completed` (read, build/w3a/lines.py). Two more asserts differ only in `lifecycle.started_row_id` | `row()` takes the non-started row; `comparable()` drops that id (1cd4c1b) | fails before; after (first edit): 8 passed 2 failed on the lifecycle id; second edit rerun NOT RUN |
| test_option_control | (c) stale expectation | recorder writes `decision_record.*` x3 and `lifecycle.started_row_id` on ft/eval rows (ledger.py `__enter__`) | `RECORDER_OWN_METRICS` whitelist (1cd4c1b) | fails before; after: NOT RUN |
| test_ledger_provenance | (c) stale expectation | 2 `started` rows of `mac-v6-lora2b-shards-2026-10-18.jsonl` cannot carry a digest (written before it); their terminal rows do | exclude a started row once a terminal row names it; 2 new tests (1191554) | fails before; after: NOT RUN |
| test_batch_order | (c) helper | `_patch_module` stub requires `config`; base ref f168576's tool does not pass it (real_ft_run_base.py:10408) | stub takes `**_kw` in test_real_ft_rungd_flags.py (3796406) | fails before; after: NOT RUN (the differential may expose a further difference) |
| test_wall_clock_cap_required | (c) cross-lane | expects `header.json` in the missing-shard-set error; L2's 154dfaa changes `shards.py` to raise that, on main | none: needs L2's `python/qd_train/shards.py`, which this lane may not touch and has not merged | red on this branch until main's shards.py is merged; before-arm run: NOT RUN |
| test_split_cache_main | not reproduced | passes on d09222d (before-summary.txt) | none | passes before |
| test_real_ft_pieces | (a) fixture, v6 | on d09222d (load-48 arm, before-test_real_ft_pieces.txt): 1 failed, 44 passed; R1 refuses the 20-pair / 40-defect-row build: defect_token_share 0.387384 against floor 0.558 | `v6_data_rules=False` + `legacy_v5()`, docstring names it (a528a37) | fails before; after: NOT RUN |

## Flake candidate: test_ledger.py::test_an_interrupt_during_a_suite_takes_its_process_group_down

UNCLASSIFIED (one arm at most). Nothing run yet: the 5 under-load runs (load48, load1 recorded) and the
5 alone runs (load 12, `flakealone*`) are queued. Reading: `run_suite` kills the process group on
any BaseException (ledger.py `run_suite`), so the product path looks correct (inferred, stays inferred
until the alone arm runs). The test starts a 1.0 s timer (SIGINT) and then reads `child.pid`; a child
whose Python start-up takes longer than 1 s is killed before it writes the file, which would read as
FileNotFoundError on `pid_file.read_text()` (inferred; not observed, no failure message was kept by
L3's sweep).

## Deviation from the load-48 conditions (confirmed by the lead)

The 1800 s per-file bound is counted from the "holds the lock" line of the job's log, i.e. on the
job's own runtime. Lock wait is bounded only by MAC_HEAVY_MAX_WAIT_S=28800. Counting the wait inside
the 1800 s made every file time out unrun on a host above the load gate (the first attempt did).

## Follow-up for card ft-cc9fac73 (file in its log)

v6-valid fixture for `test_pre_dedupe_drops`: under v6 rules the pool's val twin is not knocked out
(v6 dedupe keeps `v0`, drops `t0`), so the drop list has no v6 test. The test runs under legacy v5
until a fixture with a v6 knock-out exists.

## Open

- After-edit results for the table, the 5+5 flake runs, and test_real_ft_pieces: read
  `build/w3a/out/*-summary.txt`; first command: `cat build/w3a/out/*summary.txt`.
- A v6-valid fixture for the pre-dedupe drop list (under v6 there is no knock-out to undo).
- No gaps.jsonl records appended, so `test_gaps_ledger.py` and `test_lint_gate.py` were not run.
