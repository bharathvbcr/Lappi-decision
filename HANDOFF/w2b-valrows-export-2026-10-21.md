# HANDOFF: w2b-valrows-export (rule 10)

Lane W2b of ft-c5ea3e71 (items 1, 5, 6). Branch `lane/ft-c5ea3e71-w2b`, based on d09222d. Commits: `fee42bc` (item 1), `70e4050` (items 5 and 6), plus the gaps and this file. Not merged, not pushed.

## What was measured (no ledger row: no gate or training run)

- **Item 1.** `python/tests/test_val_rows_producer.py`: on d09222d 6 of 10 fail (the Makefile default, the flag and its place in main, the produced file passing release_replay's binding check, the held-out refusal, the other-seed binding, overwrite refusal); the 2 interrupted-write cases pass on both, as regression guards. After: 10 passed; with `test_val_rows_writer.py` and `test_release_replay_flow.py`, 34 passed (ml venv, torch present, default lock).
- **Item 5.** `a_shape_spelled_so_the_runtimes_lookup_misses_it_is_refused`: fails on d09222d (15 passed, 1 failed in calibration_gate), passes after.
- **Item 6.** 5 new tests in `train_manifest_binding.rs`: do not compile on d09222d (`cannot find function verify_release_train_manifest`); pass after.
- `cargo test -p qd-export --no-fail-fast` after: every target ok (14 + 16 + 4 + 17 + 6 + 3 + 11 + 16 + 4 + 3 + 8 + 15 + 7 + 5 tests; 1 ignored in roundtrip). `ledger_real_producer` needs `QD_PYTHON` (unset it fails at its own `expect`; run with the ml venv python it passes). `cargo clippy -p qd-export --all-targets --no-deps -- -D warnings`: clean. Without `--no-deps`, clippy fails in qd-train (`collapsible_if`, frozen file, not touched).
- Not run: `--write-val-rows` on a real corpus rebuild.

## What changed

- `tools/real_ft_run.py`: `--write-val-rows PATH`, `write_run_val_rows`, `VAL_MANIFEST`; called after the split, before `relabel_train`.
- `python/qd_data/val_requests.py`: `write_val_rows` overwrites an existing rows file or sidecar only with identical bytes; rows first, so an interrupted write leaves a file with no binding (NotRun) and a rerun completes it.
- `Makefile`: `FT_OUT`, and `VAL_ROWS ?= $(FT_OUT)/data/pool/val_rows.jsonl` (empty, so NotRun, when neither is set).
- `crates/qd-export/src/calibration_gate.rs`: shape presence is decided by `qd_runtime::calibration::CalibrationTable::lookup`, not a string comparison of keys.
- `crates/qd-export/src/export.rs`, `lib.rs`: `verify_release_train_manifest(release, ft_ledger, manifest_override)`.

## Findings that changed the card

- Item 6's gap no longer held as written: the ft ledger row already records `train_manifest_sha256` (ledger.py:1046/1216, written by real_ft_run since 9536245) and export refuses a differing one. The remaining hole was after export, which is what the new function closes.
- Item 5's full fix is a qd-runtime change (admission does not bind family and slot shape). The gap stays open, narrowed. A sweep of undeclared table entries in the export gate was rejected: it would break `--exclude-family`.

## Security-impact statement

Every change narrows what can be released or replayed; none widens access. (1) A VAL_ROWS file is refused unless its sidecar binds it, as before; the writer now refuses to replace existing evidence with different bytes. (2) The export gate refuses more (shape spellings the runtime would never look up); it accepts nothing it refused before, because the runtime's own lookup is the test. (3) `verify_release_train_manifest` only adds a refusal path: it returns a sha only when the file, the release record, the binding and (given a ledger) every ft row agree. No auth, quota, crypto or secret handling is touched.

## Hot path

None of these is a hot path: `write_val_rows` runs once per run, the gate and verify once per export. DevMap callers were not consulted (not run; this was read with rg over crates/qd-export, python and tools, not graph-confirmed).

## Open

- GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17 (runtime admission binding; owner human).
- Wiring `verify_release_train_manifest` into `qd-release-check` (qd-preflight, lane W2a's files); suggested call: `qd_export::verify_release_train_manifest(&release, ledger.as_deref(), None)`.

## First command for the next lane

`bash tools/mac_heavy.sh next cargo test -p qd-export --no-fail-fast` with `QD_PYTHON` set to the ml venv python.
