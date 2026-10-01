# Perf lane: the `--score-checkpoint` prelude (2026-09-30)

Branch `perf-prelude-rust` (local, not pushed, not merged). Everything here ran on the Mac's
CPU. Nothing ran on MPS or on the GH200.

## What was measured

No ledger row was written. These are timings of the prelude, not results of a model, so no row
is the right place for them. Every number below is quoted verbatim in
`AUDIT/perf-prelude-2026-09-30.md`.

**Profile.** This is stdlib cProfile over `main()` of `tools/real_ft_run.py` at 63f5b68. It
runs up to the needle-worker launch, which is where the prelude hands off to `_checkpoint_step`.
The inputs are the phase-3 set: `--no-repo-history`, the defect class, `--score-val --needle
--ood`, CPU. The total was 461.9 s profiled.

| function | cumulative s | note |
| --- | ---: | --- |
| `ft_splits` (2 calls: the train/val rebuild and the OOD general-record rebuild) | 390.7 | |
| `qd_data.minhash.MinHasher.signature` (151,812 calls) | **346.1** | pure-Python bigint `(a*h+b) mod p`, min over shingles |
| `split._cross_split_near_duplicates` (2) | 233.8 | re-signs every row |
| `dedupe` (2) | 133.2 | signs every content unit |
| `prepare_ood` | 74.8 | contains one `ft_splits` |
| `prepare_needle` | 45.1 | |
| `encode_slot` → `_span_token_positions` (480) | 28.3 | |
| `qd_train.shards._token_index_for_char` (202,415) | 21.6 (tottime) | O(lines × tokens) scan |
| `render._escape` | 26.1 | |
| `build_mixture` (2) | 18.6 | |
| `candidate_pairs` (4) | 11.9 | |
| tokenizer load ×2 (`_matching_tokenizer`) | 9.9 | |

**Unprofiled baseline.** At 0f4a18b, wall clock was 238.8 s. Of that, `dedupe` + `split` took
175.0 s.

**Interleaved A/B, full prelude, same inputs, same code.** Arm A replaces `native_minhash` with
a pass-through, so `qd_data` signs with the Python code it used before. Arm B uses `QD_PREP_BIN`.
The rounds ran in the order A, B, A, B, A, B. The machine was shared, so A varies a lot.

| arm | runs (s) | min (s) |
| --- | --- | ---: |
| reference (Python) | 285.46, 322.37, 175.68 | **175.7** |
| qd-prep (Rust) | 102.12, 66.67, 70.75 | **66.7** |

The speedup is 2.6× on the whole prelude (min over min). In the round-3 phase split, `dedupe` +
`split` fell from 124.8 s to 9.0 s. The output digest was identical in all six runs:
`edcd6bacdb7d4bd32f85ca7fbbd46ca256f06799166c888652db73ffb8f95469`. It covers every row id in
both `ft_splits` results in order, the needle suite digest, the OOD case ids and record sha, and
the val plan shape.

**Committed kernel A/B.** `test_benchmark_reference_against_qd_prep_interleaved_min_of_n` signs
8,279 real sets (639,023 shingles). The reference took 20.907, 22.126 and 14.137 s; qd-prep took
0.401, 0.168 and 0.114 s, process and file I/O included. Min over min is 124×.

**Full-corpus parity.** Every distinct shingle set the prelude signs was signed by both. That is
66,669 sets and 3,038,587 shingles, from 85,133 rows across both mixtures. There were **0
mismatches**. Native took 0.93 s and the reference 60.2 s, in one pass.

**After the port.** These are the remaining costs in a native prelude (round 3):

| phase | s |
| --- | ---: |
| `prepare_needle` | 36.6 |
| `ft_splits` (mixture, loaders, shingling) | 23.7 |
| `prepare_ood` | 9.8 |
| `_labels` | 7.0 |

## What changed (commits)

- `32ffa44` adds `crates/qd-prep` and the `native_minhash` / `_prep_signatures` adapter in
  `tools/real_tokenizer_pipeline.py`. It wires the three dedupe/split call sites:
  `real_ft_run.ft_splits`, `real_tokenizer_pipeline.run` and `mac_zero_shot.main`.
- `1a92a29` merges main at a151483.
- `3c90201` points the six dedupe-reaching tests at a built qd-prep and adds the committed A/B
  benchmark. Its `data_fixtures.use_qd_prep` builder was replaced in `b2afca0`.
- `85ba1fb` adds this handoff and 4 `GAP-` records.
- `5a8e1ec` merges main at 10ce63f. The `gaps.jsonl` append conflict was resolved by keeping
  both sides: main's 596 lines, then this lane's 4.
- `b2afca0` builds qd-prep through `python/tests/conftest.py` (`qd_prep_bin` / `qd_prep`),
  next to main's margin-probe fixture.
- The commit that adds `AUDIT/perf-prelude-2026-09-30.md` holds every number in this file
  verbatim.

**Languages:**

- **`crates/qd-prep` is Rust.** The work is a CPU-bound hash-and-modular-arithmetic loop. It
  uses only std and clap, and BLAKE2b is hand-written because no blake2 crate is in the
  workspace. Threads come from `std::thread::scope`.
- **`tools/real_tokenizer_pipeline.py::native_minhash` / `_prep_signatures` is Python, in the
  thin-binding role.** It marshals shingle sets to the binary over a length-checked file
  protocol (`QDPMHIN1` / `QDPMHOK1`, see `crates/qd-prep/src/wire.rs`). It also installs the
  answers where `qd_data` signs.
- **`qd_data.minhash.MinHasher.signature` (with `shingle`) is Python, in the reference-oracle
  role.** The parity tests import it, and every run re-signs 8 canaries with it. It is no longer
  a runtime path. `qd_data` was not edited, because `qd_data.fingerprint` puts it in every
  shard header.

**Behaviour change:**

- `tools/real_ft_run.py` (every mode that reaches `ft_splits`; `--probe` returns before it),
  `tools/replay_decontam.py`, `tools/ft_linear_control.py`,
  `tools/real_tokenizer_pipeline.py` and `tools/mac_zero_shot.py` now **refuse** unless
  `QD_PREP_BIN` names an absolute path to the binary. That binary must also run, exit 0, reply in
  the exact shape, and agree with the reference canaries.
- The refusal message carries both build commands.
- There is no Python fallback.
- New ledger rows carry a new `code_that_ran` digest, as any change to the tool closure does.
- No recipe, threshold, suite or shard byte is touched: the output digest above is identical.

**Build:**

- On the Mac: `cargo build --release -p qd-prep` produces `target/release/qd-prep`.
- For the box (verified to build; DT_NEEDED is `libgcc_s.so.1` and `libc.so.6` only):

  ```
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh \
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc" \
  cargo build --release -p qd-prep --bin qd-prep --target aarch64-unknown-linux-gnu \
    --target-dir /Users/bharath/qd-campaign/target-aarch64-linux
  ```

  The result is `/Users/bharath/qd-campaign/target-aarch64-linux/aarch64-unknown-linux-gnu/release/qd-prep`.
  Copy it to the box and `export QD_PREP_BIN=<absolute path>` before the campaign driver starts.

## What is open

- `GAP-PERF-PRELUDE-TOKEN-INDEX-FOR-CHAR-IS-QUADRATIC`: the next hot spot. It needs a batched
  span-projection subcommand, not a per-slot subprocess.
- `GAP-PERF-PRELUDE-NEEDLE-WORKER-STILL-REBUILDS-THE-PRELUDE`: the worker still pays one
  native prelude, 67–102 s on the Mac.
- `GAP-PERF-PRELUDE-NOT-MEASURED-ON-THE-GH200`: the box binary was built but never run there.
- `GAP-PERF-PRELUDE-DEVMAP-ANSWERED-FROM-THE-MAIN-CHECKOUT`: some answers were not
  graph-confirmed.
- Not wired: no shard set was rebuilt with the native path. `pipeline.run` now signs through it,
  and `test_pipeline_val_shards` exercises that, but that is not a phase-3 rebuild.

## First command for the next lane

```
cargo build --release -p qd-prep && \
QD_PREP_BIN=$PWD/target/release/qd-prep QD_PREP_BENCH=1 \
uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \
  --with datasketch python -m pytest -o addopts= -q -s -p no:cacheprovider \
  python/tests/test_qd_prep_minhash_parity.py
```

The end-to-end A/B driver (`ab_prelude.py <worktree> reference|native`) and the full-corpus
parity script are throwaway analysis in this session's scratchpad. They are not committed,
because this lane was told to add no Python modules.
