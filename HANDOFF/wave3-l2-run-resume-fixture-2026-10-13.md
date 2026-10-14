# Handoff: wave 3 lane W3-2, a 64-vocab shard fixture and run()-level ledger and resume tests

Card ft-3163ce2315f5d63c03d05de888434469 (d) and the CLI `--resume` path. Branch
`wave3-l2-run-resume-fixture`, cut from 174452f. Ruling: the fixture, not a `run()` reader seam
(`AUDIT/board-sweep-2026-10-13/fable-wave3-ruling.md` section 2). `run.rs` is unchanged.

## What was measured

CPU only. No GPU test was run by this lane (orchestrator instruction: the Metal lock cannot be
taken from a lane worktree). No ledger row id: nothing here wrote a ledger row.

- `cargo test -p qd-train --test fixture_pins`: red before the set existed
  (`shards-tiny-v64: cannot read SHA256SUMS`), 13 passed after.
- `cargo test -p qd-train-metal`: `run_door` 4 passed, `args` 11 passed; `gpu_tiny` 0 passed,
  9 ignored (5 new: three tests and the child).

## What changed

- `tools/qd_train_oracle_shards.py`: `--vocab 64` writes `shards-tiny-v64/`. The corpus uses more
  than 64 distinct bytes, so no lossless tokenizer fits: byte `b` is id `b % 64`, there is no
  `decode`, the span-decode check is `NotRun`, and the remap is the identity over 64 ids.
  `--vocab 64` also freezes `python/qd_data/*.py` into `<out>/qd_data/`, because `run()` has no
  `allow_stale_code` and the live package drifts from the header's fingerprint.
- `crates/qd-train/tests/fixtures/shards-tiny-v64/` (new, 58 files + `SHA256SUMS`) and its entry in
  `fixture_pins.rs`. `shards-tiny` and its `SHA256SUMS` (sha256 9d192512...a6321) are not modified
  (`git status` shows no change under `shards-tiny/`).
- `crates/qd-train-metal/tests/run_v64/mod.rs` (shared support), `tests/run_door.rs` (CPU),
  `tests/gpu_tiny.rs` (3 new `#[ignore = "GPU"]` tests + child), `tests/args.rs` (1 CLI test).
- `gaps.jsonl`: one append, `GAP-TESSL-TINY-SNAPSHOT-LACKS-ATTN-OUTPUT-GATE-2026-10-13`.

## Finding that changes the ruling's premise

`run()` cannot read tessl's `qwen35_train` snapshot as it stands: its `config.json` has no
`attn_output_gate`, and `Layout::from_config_json` refuses it (run.rs:151). The weights do carry
the gate. The tests copy the two files to a scratch directory and add the key; tessl is untouched.

## Fixture pins

SHA256SUMS 0293a982fd9b68515688ee7e1eafc2c9361f53af8da9f9ba48353096f77ae75d;
header.json 46638b06...ce1e0; tokens.u32 d3fe02d7...e235; remap.npz a5e3cb10...ddeb8e.
Head init pins used: sha256 3c1cdbb7...9555b0, content digest 81bd953e...1d6.

## Open

- GPU tests not run (see first command). The kill tests use 24 steps of a 59-batch epoch and
  kill when `ckpt-00000002/trainer.json` appears; if the child finishes first the test panics
  ("the trigger never fired"), it does not pass.
- Byte equality of `run()`'s final checkpoint (resumed vs uninterrupted) is asserted but unrun.
- `write_checkpoint` refusing an existing `.ckpt-N.partial`: `run()` requires a new or empty
  `--out`, so it cannot reach it. Recorded, not changed.
- The legacy generator path (no `--vocab`) was only run to completion into a scratch dir; its
  output differs from the tracked `shards-tiny` because `qd_data` has moved since 2026-10-01
  (as `allow_stale_code` in `common::fixture_open` anticipates). Not regenerated.
- DevMap and GitPulse Insights were not queried by this lane.
- `make lint`: NotRun (the worktree `.venv` has no ruff). `ruff 0.15.20` from PATH, same
  config: `ruff check --config pyproject.toml .` passed.

## First command for the next lane

```
cargo test --release -p qd-train-metal --test gpu_tiny -- --ignored --test-threads=1 gpu_run
```
