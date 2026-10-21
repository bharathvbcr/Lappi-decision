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

## First command for the next lane

After ojas ft-b5e64557 phase B puts a LoRA provider behind qd-train's loop (ojas-train, after
ft-d066187f): `rg -n "lora_ft_row|optimizer_recipe" crates` then remove the `lora` refusal in
`crates/qd-train-metal/src/args.rs`, dispatch from `run.rs` to `lora_ft_row` with
`row_extras(&manifest_sha256)` through `ledger::append_with` (the append guard already requires both).
