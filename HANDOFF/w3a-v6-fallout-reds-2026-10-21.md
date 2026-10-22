# W3a: the 17 "v6-default fallout" reds (ft-cc9fac73), 2026-10-21/22

Branch `lane/ft-cc9fac73` (from d09222d, not rebased onto main). Session lappi-decision-0f.
Logs: `build/w3a/out/` (git-ignored). Result lines: `build/w3a/out/*-summary.txt`, each with the
1-minute load at launch. Wrapper: `build/w3a/perfile.py` (stdlib, killpg, one mac_heavy job per file).

## Deviation from the load-48 conditions (confirmed by the lead)

The 1800 s per-file bound is counted from the "holds the lock" line of the job's log, i.e. on the
job's own runtime. Lock wait is bounded only by MAC_HEAVY_MAX_WAIT_S=28800. Counting the wait inside
the 1800 s made every file time out unrun on a host above the load gate (the first attempt did).

## Measured

Before = unmodified d09222d. After = this branch (fix commits f1514da, 1cd4c1b, 1191554, 3796406,
a528a37). Every run: make torch-pytest under tools/mac_heavy.sh, QD_PREP_BIN pre-built, host data
linked (untracked). Load-48 grant for the before-files, the after-files and test_ledger.py only.

- Before: `base.txt` (11 files, load-12 grant: 8 failed 259 passed 10 skipped; four files skipped for
  missing data), `base3.txt` (data linked: 7 failed in val_shards + spancheck), then the load-48 arm:
  test_split_cache_main passed; test_real_ft_pieces 1 failed 44 passed; test_wall_clock_cap_required
  1 failed 5 passed.
- After, all 12 files RAN (after-summary.txt): test_ft_linear_control_split_cache (10), test_pipeline_val_shards,
  test_qd_prep_spancheck_parity, test_split_cache_main, test_pre_dedupe_drops, test_real_ft_pieces,
  test_option_control, test_batch_order (6 passed), test_ledger_provenance, test_real_ft_rungd_flags,
  test_ledger (104 passed): all passed. test_wall_clock_cap_required: 1 failed, 5 passed (below).
- Not run: test_gaps_ledger.py and test_lint_gate.py (no gap record was appended). `ruff check python/tests/`: clean.

## Per-test table

| Test | Class | Cause | Fix (commit) | Before / after |
|---|---|---|---|---|
| test_pipeline_val_shards (3) | (a) fixture, v6 | `pipeline.run` default v6; R1 refuses a build with no `--defect-class`: "defect_token_share not_run ... Refusing to write a shard set (Fable R1)" (base3.txt) | `v6_data_rules=False` + `legacy_v5()` readers, rule set named (f1514da) | 3 fail / pass |
| test_qd_prep_spancheck_parity (4) | (a) fixture, v6 | same refusal | `v6_data_rules=False`, docstring names it (f1514da) | 4 fail / pass |
| test_pre_dedupe_drops | (a) half-migrated fixture | 3 `pipeline.run` calls left on v6 (d4d99ed moved the rest). Measured: same identity keys under v5/v6; v5 dedupe drops val v0 for train twin t0, v6 dedupe drops t0 and keeps v0 | `v6_data_rules=False`, docstring names legacy v5 (f1514da) | fail / pass |
| test_real_ft_pieces | (a) fixture, v6 | R1 refuses the 20-pair / 40-defect-row build: defect_token_share 0.387384 against floor 0.558 | `v6_data_rules=False` + `legacy_v5()`, docstring names it (a528a37) | fail / pass |
| test_ft_linear_control_split_cache (3) | (c) stale expectation | `row()` expected 1 ledger line; the file holds `started` + `completed` (read, build/w3a/lines.py). Two cached-vs-uncached asserts differed only in `lifecycle.started_row_id` | `row()` takes the non-started row; `comparable()` drops that id (1cd4c1b) | 3 fail / 10 pass (re-run after the second edit) |
| test_option_control | (c) stale expectation | recorder writes `decision_record.*` x3 and `lifecycle.started_row_id` on ft/eval rows | `RECORDER_OWN_METRICS` whitelist (1cd4c1b) | fail / pass |
| test_ledger_provenance | (c) stale expectation | 2 `started` rows of `mac-v6-lora2b-shards-2026-10-18.jsonl` cannot carry a digest; their terminal rows do | exclude a started row once a terminal row names it; 2 new tests (1191554) | fail / pass |
| test_batch_order | (c) helper | `_patch_module` stub requires `config`; base ref f168576's tool does not pass it | stub takes `**_kw` in test_real_ft_rungd_flags.py (3796406) | fail / 6 pass (the differential against f168576 holds) |
| test_wall_clock_cap_required | (c) cross-lane | expects `header.json` in the missing-shard-set error; got "[Errno 2] ... shards/train". L2's 154dfaa makes `shards.py` raise it; it is on main | none: blocked, needs `python/qd_train/shards.py` (L2, now on main) | 1 fail / 1 fail until main is merged |
| test_split_cache_main | not reproduced | passes on d09222d and on the branch | none | pass / pass |

## Flake: test_ledger.py::test_an_interrupt_during_a_suite_takes_its_process_group_down

Classification: NOT REPRODUCED in either arm (9 of 9 completed runs passed, plus 5 burner runs, 14 of 14).
It stays a flake candidate with no defect found; "the product code is correct" is INFERRED from
reading `run_suite` (kills the process group on any BaseException), now with 14 passes, not proof.

- Quiet host, load 12 gate (flakealone2..5): 4 of 4 passed; flakealone1 timed out waiting for a
  quiet host (NOT RUN).
- load-48 queue (flakeload1..5): ran after the host went quiet, so these are quiet runs too
  (1-minute load 4.6 to 4.9 at launch): 5 of 5 passed. They are NOT an under-load arm.
- Under load, with 18 self-terminating busy-loop python processes under mac_heavy at the load-48 grant
  (`flake_inner.sh load`, flakeburn.txt; 1-minute load printed per run, it lags the burners):
  5.89, 20.58, 24.84, 29.42, 44.30. 5 of 5 passed, each in about 1.6 s, so a 1.0 s timer
  was not outrun by the child's start-up at these loads.
- Hypothesis from reading (inferred, not observed): the test fires SIGINT from a 1.0 s timer and then reads
  `child.pid`; a child whose start-up exceeds 1 s is killed before it writes the file
  (FileNotFoundError on `pid_file.read_text()`). L3's sweep (load far above these, 133 files) kept
  no failure message, so the sweep failure is unexplained. No card for a product defect is warranted by this evidence.

## Follow-ups

- Card ft-cc9fac73 log: a v6-valid fixture for `test_pre_dedupe_drops`. Under v6 the pool's val twin is
  not knocked out (v6 dedupe keeps `v0`, drops `t0`), so the drop list has no v6 test; the test runs
  under legacy v5 and says so.
- Merge main (L2's shards.py) to turn test_wall_clock_cap_required green; I did not edit shards.py.
- Four legacy-v5 moves (val_shards, spancheck, pre_dedupe, real_ft_pieces) leave no test building a
  v6-valid shard set end to end from the plain commitpackft download; a bigger defect corpus would.
