# L5 handoff: lora_ft_row producer (ft-57fc7484 item b5), 2026-10-20/21

Branch `lane/ft-57fc7484-l5` from main d09222d. Lane L5 of the GitPulse backlog run.

## Decision: outcome (b), criterion b5 stays OPEN

A producer for `qd_train::ledger::lora_ft_row` cannot be wired from qd-train-metal today.

- ojas at the pin `4123dfb` (read with `git cat-file`, not the ojas working tree, which is at a
  later main): a LoRA `StepProvider` exists, `ojas-model/src/qwen35/step_provider.rs`
  `TapeStepProvider::open_lora` (`impl StepProvider` at line 1074). It implements
  `ojas_train::lappi::step::StepProvider`, ojas-train's copy of qd-train's trait (line 97 import).
- qd-train-metal depends on `ojas-qwen35` and `ojas-core` only (`crates/qd-train-metal/Cargo.toml:40-41`);
  its `Qwen35Provider` (`provider.rs:297`) wraps `Qwen35Step`, which is full fine-tuning
  (`ojas-qwen35/src/step.rs:660` at 4123dfb).
- Wiring needs a new dependency (ojas-model), a bridge between two trait copies, and the loop
  dispatching `lora_optimizer_table` (`trainer.rs:508` calls `optimizer_table`; trainer.rs is in the
  frozen moving set, moving to ojas-train under ft-d066187f). Anything less is a stub, refused.
- No flag or recipe in qd-train-metal selected LoRA. A LoRA run could only have been a full-FT run
  written as `optimizer_recipe: "f32"`.

## What changed

- `crates/qd-train-metal/src/args.rs`: new optional `--optimizer-recipe`. `f32` (or absent) is
  today's run, unchanged (`TrainArgs` equal). `lora` is `ArgsError::Refused` before the shard
  door, snapshot or GPU, naming the missing provider and the unwired `lora_ft_row`. Anything else
  is `Usage`. No `TrainArgs` field was added (args.rs was needed only because the refusal needs a
  way to state a LoRA recipe; provider.rs and run.rs are untouched).
- `crates/qd-train/src/ledger.rs`: `append_with`, the one locked append, now refuses a row labelled
  `lora` that lacks any of the seven metrics `lora_ft_row` adds, or that records no
  `train_manifest_sha256`. Rows not labelled `lora` skip the check.
- Tests: `crates/qd-train-metal/tests/lora_refusal.rs` (3), `crates/qd-train/tests/lora_row.rs` (2 new).
- `gaps.jsonl`: `GAP-L5-LORA-FT-ROW-PRODUCER-UNREACHABLE-FROM-QD-TRAIN-METAL-2026-10-21` (opened).

Unchanged by construction: run.rs still writes its row last, once, through `ledger::append_with`
after both exports, so a crash mid-run writes no completed row. A resume across the LoRA boundary
is refused (`check_resume_recipe` already reports a differing `optimizer_recipe` key; pinned by a test).

## Evidence (ledger-free; test runs, all under tools/mac_heavy.sh at MAC_HEAVY_MAX_LOAD=48)

Raw outputs are in `build/l5/` (gitignored): `fb2-metal.out`, `fb3-qd-train.out`, `after-metal.out`,
`after3-qd-train.out`, `after-qd-metal.out`.

- Characterization, on unmodified code: an f32 row's appended bytes (4357 bytes) hash to
  `8251ace8...446110e`; the test asserts it and passes after the change.
- Fails before: `cargo test -p qd-train-metal --test lora_refusal` on pre-fix args.rs: exit 101,
  0 passed, 3 failed (`unknown argument "--optimizer-recipe"`). `cargo test -p qd-train --test
  lora_row` on pre-fix ledger.rs: exit 101, 6 passed, 1 failed (the lora-labelled bare row was
  appended: `unwrap_err()` on `Ok(Stamp ..)`).
- After: `cargo test -p qd-train` exit 0, 267 passed, 0 failed, 5 ignored; `cargo test -p
  qd-train-metal` exit 0, 44 passed, 0 failed, 15 ignored (the 3 new lora_refusal tests included).
- `cargo test -p qd-metal --no-run`: exit 0.

## Not run

- GPU-ignored tests (`gpu_*`, 15 ignored in qd-train-metal) stay ignored: no GPU job was launched.
- `pythons_verifier_accepts_a_lora_row` (ignored, needs the ml venv) not run.

## Open

- `GAP-L5-LORA-FT-ROW-PRODUCER-UNREACHABLE-FROM-QD-TRAIN-METAL-2026-10-21` (and the two it refs).

## Adversarial pass (Opus review), fixes 2026-10-21

Cherry-picked b6dc9a6 (`adv_lora_label.rs`), 20dff34 (`adv_lora_args.rs`) and 16d9836
(`test_adv_lora_row_python.py`); all pass without weakening one.

1. The gate moved from `append_with` to `Row::line_with` (`check_optimizer_label`), which every ft
   line is built by (`FtRow::line_with`, `append_line` closures, `append_with`). Consequence:
   `FtRow::line` / `append` on a lora row without `train_manifest_sha256` now refuse, so the
   existing lora tests that used them (`lora_row.rs`: the row-shape test, the order test and the
   ignored Python-verifier test) now pass `manifest_extras()`; their assertions are unchanged. The
   f32 characterization hash is byte-identical (`8251ace8...446110e`).
2. Values, not presence: the five digest metrics are passed `ran` records holding lowercase hex
   sha256; `corpus.plan_seed` an unsigned integer; `lora.param_groups` a `name xN at lr_scale S`
   summary (derived from what `lora_ft_row` writes); a `not_run`, null, failed or malformed metric
   is refused. Pinned by `every_lora_metric_is_checked_for_what_lora_ft_row_writes`.
   The lora evidence is demanded of `completed` rows only (Rust and Python): a failed or killed
   row records an unfinished run and must stay writable when the run died before its digests.
3. Closed label set `OPTIMIZER_RECIPE_LABELS` = bf16, master, kahan, f32, lora (Python's
   `OPTIMIZER_RECIPES`; Rust's own tests use f32 and master). Anything else, including case,
   whitespace and homoglyph variants, is refused at row-build time (`the_optimizer_label_set_is_closed`;
   the empty label was already refused by `Protocol::for_recipe`). All 99 committed ledgers (2211
   rows) still verify with the gate in place (`build/l5/verify_all.py`, read-only).
4. Python parity: `LedgerRow.check_optimizer_label()` runs in `Ledger.append` before the write and in
   `LedgerRow.from_json`, so `rows()` and `verify_chain()` refuse what a raw writer put in the file.
   It is deliberately not in `__post_init__`: the adversarial test builds a bare row object before it
   appends it. New `python/tests/test_lora_label_gate.py` (pins the label set equal across Rust,
   `ledger.py` and `real_ft_run.py`, and every bad value). Rust-to-Python direction: the extended
   `pythons_verifier_accepts_a_lora_row` (ignored; run by name, passes). `python/qd_train/ledger.py`
   was granted by the lead.
5. Metric names diverge and the pre-registration does not decide: OPEN, owner decision,
   `GAP-L5-LORA-METRIC-NAMES-DIVERGE-RUST-VS-REFERENCE-DRIVER-2026-10-21`. Neither writer was
   edited. Because `tools/lora_reference_run.py`'s rows (and `test_lora_reference_driver.py`) use the
   `init.*` names and record no manifest sha, the Python gate accepts a completed lora row with either
   complete validated set (never a mixture), and requires `train_manifest_sha256` only for the Rust
   set. So Python accepts driver-spelled rows that Rust would refuse; the release export refuses them
   (no manifest sha). Resolve by choosing one set and deleting the other branch.
6. L5-5 (`adv_lora_then_f32_is_still_the_lora_refusal`, error-message ordering) is cosmetic: dropped,
   not required. `--optimizer-recipe lora --optimizer-recipe f32` is still refused (as a usage error
   for the repeat).

Runs after the fixes (mac_heavy, MAX_WAIT_S=28800): `cargo test -p qd-train` 274 passed, 0 failed,
5 ignored; `-p qd-train-metal` 49 passed, 0 failed, 15 ignored; `-p qd-metal --no-run` exit 0;
`cargo test -p qd-train --test lora_row -- --ignored` 1 passed; `make torch-pytest` narrowed to the
lora and ledger tests (10 files) 303 passed. Raw output in `build/l5/r3-*.out` (gitignored).

## Fable's ruling on the metric names (2026-10-21): Rust set canonical, implemented

Supersedes item 5 and the dual-set paragraph above; the GAP
`GAP-L5-LORA-METRIC-NAMES-DIVERGE-RUST-VS-REFERENCE-DRIVER-2026-10-21` is closed with a resolved
record citing the ruling.

- `tools/lora_reference_run.py` writes `train.span_head_init_digest`, `lora.d17_init_digest`,
  `lora.a_init_digest`, `lora.base_safetensors_sha256`, `lora.dtype_map_sha256`, `corpus.plan_seed`
  (an integer now) and `lora.param_groups`, plus `train_manifest_sha256` on the row. The manifest
  is a new required `--manifest`/`LoraRunConfig.manifest`, hashed before and after the shard door
  opens (`read_with_manifest_sha256`, as `open_training_data` and qd-train-metal's W2-4 do); changed
  bytes are refused. `init.base_snapshot`, `init.f32_names_sha256`, `corpus.plan_order_digest` and
  `corpus.train_shard_hash` stay as extra metrics the gate ignores. Its digests dict gains
  `train_manifest_sha256`.
- `python/qd_train/ledger.py`: the "either complete set" branch is gone; one set, Rust's rule.
- `python/tests/test_lora_reference_driver.py` pins the Rust names (and the manifest sha, and the
  refusal on changed manifest bytes); the old `init.*` digest names must be absent. Its `_cfg`
  gained `manifest=`; no other assertion changed.
- Cross-language parity on the SAME row bytes: fixture
  `crates/qd-train/tests/fixtures/lora-gate-cases.json` (14 cases, written by
  `tools/gen_lora_gate_cases.py`; `--check` and a Python test fail if it is stale). Rust:
  `crates/qd-train/tests/lora_gate_parity.rs` through the new `qd_train::ledger::check_row_line`.
  Python: `python/tests/test_lora_gate_parity.py` through `verify_chain` and `Ledger.append`. Cases
  include a valid Rust-named row (accept), an `init.*`-named row (refuse, with and without a manifest
  sha), a mixed row (refuse), bare, `not_run`, string seed, upper-case digest, bad group summary,
  upper-case and homoglyph labels (refuse), and an f32 row and a failed lora row without evidence
  (accept). The Rust-writes/Python-verifies direction remains the ignored
  `pythons_verifier_accepts_a_lora_row`.
- Committed ledgers scanned (`RIPGREP_CONFIG_PATH= rg -uu` over `ledger/` and the repo, and
  `git ls-files '*.jsonl'`): NO lora row and no `init.d17_digest` / `init.lora_a_digest` metric
  exists in any committed ledger. So no frozen `lora_reference_v0` kind was added, no row ids to
  list, and no cited verdict rests on an `init.*` row.
- Runs: `cargo test -p qd-train` 276 passed, 0 failed, 5 ignored (mac_heavy 48);
  `make torch-pytest` narrowed to the lora, ledger and driver tests (11 files), default load 12,
  318 passed, 1 deselected. NOT RUN: `test_the_mps_bf16_leg_runs_with_a_bf16_base_f32_stored_tensors_and_f32_adapters`
  (`-k "not test_the_mps_bf16_leg"`, MPS leg). Committed-ledger verify (`build/l5/verify_all.py`): 99
  files, 2211 rows, 0 failed. `-p qd-train-metal` was not re-run after this change (no API it uses
  moved; last run above).

## First command for the next lane

After ojas ft-b5e64557 phase B puts a LoRA provider behind qd-train's loop (ojas-train, after
ft-d066187f): `rg -n "lora_ft_row|optimizer_recipe" crates` then remove the `lora` refusal in
`crates/qd-train-metal/src/args.rs`, dispatch from `run.rs` to `lora_ft_row` with
`row_extras(&manifest_sha256)` through `ledger::append_with` (the append guard already requires both).
