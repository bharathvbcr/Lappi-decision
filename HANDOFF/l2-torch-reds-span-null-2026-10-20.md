# L2 handoff: torch reds, span null, gap re-verification (ft-57fc74844adf4754765a3083d4210a15, items b1, b4, b7)

Branch `lane/ft-57fc7484-l2` from main d09222d. Not merged, not pushed. Evidence files are in the
lane worktree's gitignored `build/`.

## Rulings this lane ran under

- Fable 2026-10-21, load: `MAC_HEAVY_MAX_LOAD=48` granted for the baseline only, on unmodified
  d09222d, over five files, with QD_PREP_BIN built beforehand at load 12 and passed as a literal
  path, no cargo inside the lock, exits 70/76/79 / TimeoutExpired / RunIncomplete = not run, no
  retry at 48, MPS or tower-timing tests -k-excluded and listed, result is a baseline citation
  only (rule 8), no amendment of Q8 or Q21.
- Fable 2026-10-21, condition 2 amended: five separate mac_heavy jobs, one file each, under the
  stdlib wrapper `build/l2_bound.py` (subprocess, start_new_session, os.killpg), 1800 s per file
  and 2700 s for test_real_ft_general_record.py. No pytest-timeout was installed. The wrapper is
  not committed. Nothing timed out.
- After-fix runs used the default load of 12.

## What was measured

Baseline on unmodified d09222d (load 48 grant), `make torch-pytest` with `TORCH_PYTEST_RUN`
narrowed to one file:

| file | result | evidence |
| --- | --- | --- |
| test_real_ft_score_plan.py | exit 2, 1 failed 41 passed | build/l2-base-1-score-plan.txt |
| test_real_ft_general_record.py | exit 2, 2 failed 14 passed | build/l2-base-2-general-record.txt |
| test_ft_span_null_row.py | exit 0, 8 passed | build/l2-base-3-span-null.txt |
| test_lora_reference.py | exit 0, 20 passed | build/l2-base-4-lora-ref.txt |
| test_lora_reference_driver.py `-k 'not the_mps_bf16_leg'` | exit 0, 32 passed 1 deselected | build/l2-base-5-lora-driver.txt |

NOT RUN: `test_lora_reference_driver.py::test_the_mps_bf16_leg_runs_with_a_bf16_base_f32_stored_tensors_and_f32_adapters`
(needs MPS; -k-excluded by condition 4, not skipped-as-pass).

After the fix, load 12:

- test_real_ft_shard_missing, test_real_ft_score_plan, test_real_ft_general_record: 61 passed, exit 0 (build/l2-after-final.txt)
- the same three plus test_shards and test_ft_span_null_row: 198 passed, exit 0 (build/l2-after-b1b.txt)
- test_gaps_ledger and test_lint_gate: 26 passed, exit 0 (build/l2-gaps-lint2.txt)
- test_lora_reference_driver_adversarial `-k digest_does_not_depend`: 1 passed, exit 0 (build/l2-b7-digest.txt)
- `cargo test -p qd-train` at d09222d, exit 0 (build/l2-cargo-qd-train.txt): lib 53, d17_objective 15, lora_init 8, lora_plan 10, lora_row 5 (1 ignored)

Interpreter and binary: torch runs use the Makefile's `uv run --python /Users/bharath/.venvs/ml/bin/python`;
the non-torch pair used the main checkout's `.venv` through an untracked `.venv` symlink in the
worktree (the harness would not let this lane remove it; it is not committed). QD_PREP_BIN was
this worktree's `target/release/qd-prep`, built by `cargo build -p qd-prep --release` under
mac_heavy at load 12 (build/l2-build-qd-prep.txt).

## b1: the three torch reds (commit 154dfaa plus the lint-line follow-up)

1. `test_the_committed_box_plan_passes_every_argv_check`: `assert_shard_files_trainable`
   (python/qd_train/shards.py, the `root.iterdir()` call) let the directory error escape before
   header.json was named. It now raises FileNotFoundError naming `<root>/header.json`. The
   test's match is unchanged. New test: python/tests/test_real_ft_shard_missing.py.
2. and 3. NOT a draw bug. `partition_replay` (python/qd_data/general.py:544) is correct: the same
   300 MMLU rows give 48 replay rows (16%) under `DataConfig.legacy_v5()`, and a 300-row
   CommonsenseQA train fixture gives 54 (18%) under v6. The baseline output shows
   `refused cais/mmlu benchmark_eval_split_is_a_target: 300`: `DataConfig()` is v6, where MMLU
   test/dev are decontamination targets (human ruling 2026-10-17), so zero MMLU rows reached
   the draw. The fixture now carries CommonsenseQA train rows (pinned to train), and a new test
   pins both halves of the diagnosis.

## b1 follow-up: Opus review breaks L2-1 and L2-2 (adversarial tests cherry-picked as f485470)

Load 12 throughout (the load-48 grant covered the baseline only).

- L2-1, `assert_shard_files_trainable` returned `{}` for an existing directory with no
  header.json. Callers, by `rg -uu` over tracked and untracked source (not DevMap): the only
  non-test callers are `ShardReader.__init__` (python/qd_train/shards.py:2123) and
  `tools/lora_reference_run.py:211` (`validate_inputs`, which calls it on its own when the
  directory exists and leaves a missing directory to the reader). An earlier search of mine was
  cut off by `head -20` and missed the second caller.
  Through the real caller: before the fix `ShardReader` on an empty directory did refuse, but with
  `HeldOutViolation: expected 'a member the rule-3 sweep checked'`, blaming rule 3 for an unbuilt
  set, and `validate_inputs` would have passed it. Fails before (build/l2-l21-before.txt, 4 failed
  30 passed): test_the_reader_refuses_a_directory_with_no_header_naming_it_not_as_a_held_out_violation
  (HeldOutViolation escaped), test_a_directory_of_other_files_but_no_header_is_still_a_missing_header
  (DID NOT RAISE), the adversarial test_an_existing_empty_directory_is_not_reported_as_a_trainable_shard_set
  (DID NOT RAISE). Fix: after the sweep (so held-out members are still refused as such) the
  sweep raises FileNotFoundError naming `<root>/header.json`.
- L2-2, test_replay_rows_are_marked_and_never_in_the_gold_train_split asserted only a negative.
  The checks are now one helper, `assert_replay_rows_exist_are_marked_and_stay_out_of_train`
  (test_real_ft_general_record.py): the slice exists (message EMPTY_DRAW_MESSAGE), every row is
  marked REPLAY_ONLY, and the rows the partition removed from train are exactly the marked ones.
  The adversarial zero-draw test (MMLU-only record, which v6 refuses) now asserts the helper
  raises the "replay draw is empty" AssertionError, in place of the red-by-construction version.
  Shown failing against the old checks: with the helper cut back to the old negative-only checks the
  test failed `Failed: DID NOT RAISE AssertionError` (build/l2-weak-demo.txt); the helper was then
  restored and the temporary line is not in any commit.
- Two mechanical changes to the cherry-picked adversarial files: `ruff --fix` sorted their import
  blocks (the lint gate failed on I001), nothing else in them changed beyond the zero-draw test.
- After: 288 passed, 1 deselected (the MPS bf16 leg, NOT RUN), exit 0
  (build/l2-l21-after2.txt): test_real_ft_shard_missing, test_adv_shard_root,
  test_adv_general_record_fixture, test_real_ft_general_record, test_real_ft_score_plan,
  test_shards, test_shard_reader_heldout_symlink, test_real_ft_heldout_readers,
  test_ft_span_null_row, test_lora_reference_driver.
- The untracked `.venv` symlink in the worktree is left uncommitted for the lead to clear.

## b4: span null

- FT rows: already fixed on main by 6fc79e8 (2026-10-17). `score_states` writes the three
  metrics and every FT scoring path goes through it. GAP-FT-ROWS-CARRY-NO-SPAN-NULL-2026-10-17
  was stale; appended as resolved-with-residual. NOT RUN: no recorded real FT row carries the
  null (needs a model run). The fails-before of 6fc79e8 was not independently reproduced.
- Span-refit rows: NOT done, needs an owner decision. The pinned scorer
  (tools/span_refit_score.py, sha256-pinned to the pre-registration) emits no per-pointer start
  and end top-1 over the pointing rows, so there is nothing to compute the verdict from without
  amending the pre-registration. GAP-W4A-SPAN-REFIT-ROWS-CARRY-NO-SPAN-NULL-2026-10-18 stays open.

## b7: gaps (appended to gaps.jsonl with qd_train.gaps.append_gap)

Resolved: GAP-W11A-THREE-TORCH-REDS-ON-MAIN-2026-10-20, GAP-W2-2-LORA-A-DIGEST-DEPENDS-ON-MODULE-REGISTRATION-ORDER-2026-10-18,
GAP-L1-D17-GATES-UNRUN-2026-10-17, GAP-LORA2B-REDUCED-RUN-DEFAULT-CAP-1800S-2026-10-17.
Resolved with residual: GAP-FT-ROWS-CARRY-NO-SPAN-NULL-2026-10-17, GAP-LORA2B-TORCH-DRIVER-MISSING-2026-10-17,
GAP-L2-LORA-TESTS-NOT-RUN-2026-10-17.
Left open, with the reason in the record: GAP-W4A-SPAN-REFIT-ROWS-CARRY-NO-SPAN-NULL-2026-10-18,
GAP-L3-LORA-DRIVER-BUILT-2026-10-17 (no real-model run), GAP-LORA-CKPT-LAYOUT-NOT-OJAS-VERIFIED-2026-10-12 (converter card ft-3b8126cd).

## Open

- Span-refit null (above). GAP-W4A-... ; owner decision on amending the pre-registration.
- A recorded real FT row carrying the span null: needs a model run.
- This lane did not touch GAP-W6C-QDPREP-CARGO-TEST-FAILS-3-OF-287-IN-BUILD-ROW-2026-10-18 (lane L4).
- `cargo test -p qd-train-metal` was not run.

## First command for the next lane

    bash tools/mac_heavy.sh next-torch make torch-pytest "TORCH_PYTEST_RUN=<uv run ... python -m pytest python/tests/test_real_ft_shard_missing.py python/tests/test_real_ft_score_plan.py python/tests/test_real_ft_general_record.py -o addopts= -q>"
