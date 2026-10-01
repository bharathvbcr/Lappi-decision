# Perf lane: the shard pipeline, `tools/real_tokenizer_pipeline.py` (2026-09-30)

Branch `perf-pipeline-shards` was cut from main at 884b658. Every measurement below was made
at f164f35. After the measurements, main d34a936 (R2, noul rows) was merged in at 8c0214f.
R2 edits `qd_data` (`defect_class`, `mixture`, `split`). From that merge on, every shard set
gets a new `code_fingerprint` whichever MinHash path builds it, so the parity below is stated
for the pre-merge code. After the merge, these suites were run on CPU with `-k "not mps"`:
the parity suite, `test_pipeline_val_shards`, `test_real_ft_general_record`,
`test_real_ft_pieces`, `test_tool_call_sites`, `test_lint_gate`, `test_defect_noul` and
`test_ledger`. 224 tests passed and 5 were skipped: two because they are gated by
`QD_PREP_BENCH`, three because their noul corpus is not on this host.

The branch is local only: not pushed, not merged. Everything ran on the Mac's CPU. Nothing
ran on MPS or on the GH200.

## What this lane did

1. **Profiled** the pipeline with stdlib cProfile on the phase-3 inputs, and once with the
   clinc-only `--general-record`.
2. **Ported** the top hot spot, `qd_data.minhash.MinHasher.signature`, to Rust. Main then
   merged an equivalent port that was done independently: `crates/qd-prep`, bd47939, from the
   perf-prelude-rust lane. This lane's own port is therefore **superseded and must not be
   merged**. It is branch `perf-pipeline-rust`, commits 5a35c1a, 0853ff9 and 4d9242c.
3. **On main's port**, this lane did two things:
   - It closed the perf-prelude handoff's open item, "no shard set was rebuilt with the native
     path". Phase 3, clinc and phase 4 were rebuilt with it. The outputs are byte-identical to
     the pure-Python references, apart from the wall-clock timestamp fields.
   - It removed two of the three `shingle` passes inside `native_minhash`. This was the
     "one shingling pass shared by all three callers" step named in
     `GAP-PERF-PRELUDE-J1-REMAINDER-IS-SHINGLE-BANDING-AND-RENDER`.

## What changed (commits on `perf-pipeline-shards`)

| commit | change |
| --- | --- |
| 3b5678f | `native_minhash` now shingles each distinct `dedupe_text` once. It installs that table as `qd_data.dedupe.shingle` and `qd_data.split.shingle`, beside the native `MinHasher`, and restores both in `finally`. It refuses four things: a text outside the table, another `k` or `max_doc_bytes`, a module whose `shingle` is not `qd_data.minhash`'s, and a table that nobody read. Two tests come with it, and both fail on 884b658. |
| 95984e4 | The committed A/B benchmark for the shingle table: `test_benchmark_shingle_table_interleaved_min_of_n`. |
| f164f35 | A test that the precheck refuses a module that shingles through another function. It fails on 884b658. |

Nothing under `python/qd_data/` was edited, so the shard headers' `code_fingerprint` is
unchanged. `crates/` is identical to main. No Python module was added.

## What was measured

All runs were ledgered in `ledger/mac-perf-qd-prep-2026-09-30.jsonl`. All are `smoke`, all
are `quick`, and they ran on the Mac CPU. Other sessions kept the machine heavily loaded
throughout (load average 6–104). So the end-to-end wall times are **sequential and not
interleaved**. They are indicative only, and no speedup is claimed from them. The speedup
claims rest on the two committed interleaved benchmarks below.

| row | run | code (`real_tokenizer_pipeline.py` sha256) | wall s | data_snapshot_hash |
| --- | --- | --- | ---: | --- |
| 389888e8 | phase 3, pure-Python reference (main checkout e3477eba, before bd47939) | 9bbc2e1 | 510.1 | be74a513 |
| 814054a7 | phase 3, superseded port (`perf-pipeline-rust`) | f9bf2f9 | 222.1 | be74a513 |
| 75cbd709 | phase 3, **this branch** (qd-prep + shingle table) | c345882 | 204.4 | be74a513 |
| 5972b896 | clinc, pure-Python reference (main checkout 10ce63f5, before bd47939) | 9bbc2e1 | 574.5 | aceba6b1 |
| 548609d5 | clinc, superseded port | f9bf2f9 | 719.0 | aceba6b1 |
| 1cec1569 | clinc, **this branch** | c345882 | 251.2 | aceba6b1 |
| 585ae875 | phase 4, superseded port | f9bf2f9 | 1757.1 | 28f13185 |
| f5664ba2 | phase 4, **this branch** | c345882 | 600.9 | 28f13185 |

The phase-4 reference is `/Users/bharath/qd-campaign/phase4-fullvocab-2026-09-30`, row
1af4e762 in `ledger/mac-phase4-fullvocab-shards-2026-09-30.jsonl`. The rows above that name
this branch were written at HEAD 95984e4 or f164f35 (`-dirty`, which means the untracked ledger file and a `.venv` symlink). `tools/` and `crates/` are byte-identical
across 3b5678f..f164f35, and the `code_that_ran` digest c345882 is this branch's tool file.

### Shard-byte parity

Every file in the output directory (`data/` and `shards/`) was compared by sha256. The JSON
headers and manifests were compared with only `created_at` (shard headers) and
`provenance.built_at_utc` (data manifests) removed. Those fields are wall-clock timestamps.

| this branch | reference | files | identical bytes | identical but for the timestamp | `code_fingerprint` / `shard_hash` |
| --- | --- | ---: | ---: | ---: | --- |
| 75cbd709 phase 3 | 389888e8, and the unledgered pure-Python A1 build | 33 | 27 | 6 | equal |
| 1cec1569 clinc | 5972b896 | 33 | 27 | 6 | equal |
| f5664ba2 phase 4 | 1af4e762 (`phase4-fullvocab-2026-09-30`) | 44 | 36 | 8 | equal |

The phase-4 reference directory holds 45 files and this branch's rebuild holds 44. The extra
file is `replay-attestation-d43790d.json`, which was placed there at 22:37, twelve hours after
the build. `tools/real_ft_run.py` reads it through `--replay-attestation`, and the pipeline
neither reads nor writes it. All 44 files the pipeline writes match.

Two more datapoints are supporting evidence only, because neither was ledgered. A phase-3
rebuild with main's native path before the table was identical to A1. The clinc build made
under cProfile with this branch was identical to 5972b896.

### Committed benchmarks (interleaved A/B, min of N)

```
cargo build --release -p qd-prep
QD_PREP_BIN=$PWD/target/release/qd-prep QD_PREP_BENCH=1 \
uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \
  --with datasketch python -m pytest -o addopts= -q -s -p no:cacheprovider \
  -k "benchmark and not mps" python/tests/test_qd_prep_minhash_parity.py
```

These were run after N4 finished, at a load average of about 7, from `f164f35`.

| benchmark | input | reference arm, min | native arm, min | min over min |
| --- | --- | ---: | ---: | ---: |
| `test_benchmark_reference_against_qd_prep_interleaved_min_of_n` (main's: signatures only; qd-prep time includes the process and I/O) | 8,279 real sets, 639,023 shingles, 3 rounds | 10.645 s | 0.107 s | 99.5x |
| `test_benchmark_shingle_table_interleaved_min_of_n` (this lane: `dedupe` + `split` inside `native_minhash`, re-shingling arm vs table arm) | phase-3 corpus, 50,177 rows, 49,842 distinct texts, 5 rounds | 6.333 s | 4.586 s | 1.38x |

An earlier run of the shingle-table benchmark gave 6.946 s against 4.997 s (1.39x). That run
was taken while N2 was running, at a load average of about 14, so it is not the cited
number.

### Profiles (stdlib cProfile, Mac CPU; the dumps are scratch files and were not ledgered)

**Phase 3, pure Python, before the port.** This is main as it stood before bd47939; 626.1 s
were profiled.

| function | cumulative s | calls |
| --- | ---: | ---: |
| `qd_data.minhash.MinHasher.signature` | **268.2** | 100,018 |
| `qd_data.dedupe.dedupe` | 183.4 | 1 |
| `qd_train.shards.write_shards` | 171.4 | 3 |
| `real_tokenizer_pipeline._encode` (of which `tokenizers` `encode_batch` 125.9 s tottime, already Rust) | 163.4 | 463,704 |
| `qd_data.split.split` → `_cross_split_near_duplicates` | 120.4 | 1 |
| `real_tokenizer_pipeline.census` | 99.8 | 2 |
| `decode` | 37.0 | 1,929,150 |
| `qd_data.render.render` (`_escape` 27.0) | 36.4 | 191,057 |
| `qd_data.minhash.candidate_pairs` | 19.0 | 2 |
| `qd_data.minhash.shingle` | 13.6 | 100,018 |

**Clinc, this branch, after the port.** 358.2 s were profiled.

| function | cumulative s | calls |
| --- | ---: | ---: |
| `write_shards` | 167.6 | 3 |
| `_encode` (`encode_batch` 121.5 s tottime) | 159.9 | 729,409 |
| `census` | 102.0 | 2 |
| `render` (`_escape` 35.5) | 57.4 | 636,731 |
| `decode` | 30.3 | 2,098,288 |
| `candidate_pairs` | 12.3 | 2 |
| `split` + `dedupe` + `native_minhash` | 9.0 + 5.0 + 2.4 | |
| `shingle` (once per distinct text) | 1.7 | 73,687 |

After the port, dedupe and split are about 16 of 358 profiled seconds on clinc. The pipeline
is now dominated by three costs:

- tokenization: phase 3 makes 463,704 encode calls for the 95,474 sequences it tokenizes
  (4.9 per sequence);
- `census`, which tokenizes again;
- `render`.

## Language per component

| component | language | why |
| --- | --- | --- |
| `crates/qd-prep` (MinHash signatures; main's, unchanged here) | Rust | CPU-bound bigint loop, 43% of the phase-3 profile; std only, no C beyond libc |
| `native_minhash` / `_prep_signatures` in `tools/real_tokenizer_pipeline.py` | Python, a thin binding | It is the subprocess adapter at the existing call site. It marshals, and Rust computes. |
| the shingle table (this lane) | Python, in the same adapter | It caches the reference's results. It computes nothing new. |
| parity tests and benchmarks | Python | They are the reference-oracle harness. |

**The oracle roles.** `qd_data.minhash.MinHasher.signature` is the reference oracle for
qd-prep: the parity tests import it, and the block still signs its 8 canaries with it.
`qd_data.minhash.shingle` is **still the shipped implementation**. It is existing Python and
was not rewritten. It now runs once per distinct text instead of three times. Its port is
proposed below and was not done.

## What is open

- `GAP-PERF-SHARDS-NATIVE-TABLES-COVER-ONLY-DEDUPE-AND-SPLIT-NAMES` is the seam's blind spot,
  together with the fingerprint-versus-signer attestation question.
- `GAP-PERF-SHARDS-WORKTREE-CARGO-NEEDS-A-TESSL-SYMLINK` records that this lane left the
  symlink `.claude/worktrees/tessl -> /Users/bharath/Code/research/tessl` in place.
- `GAP-PERF-SHARDS-DEVMAP-HAS-NO-STORE-IN-THIS-WORKTREE` records that the callers came from the
  main checkout's graph (walk_incomplete) and from rg.
- `GAP-PERF-PRELUDE-J1-REMAINDER-IS-SHINGLE-BANDING-AND-RENDER` is still open for banding and
  render. Its shingle part is closed by `GAP-PERF-PRELUDE-J1-REMAINDER-SHINGLE-PART-RESOLVED`.
- `GAP-PERF-PRELUDE-NOT-MEASURED-ON-THE-GH200` still stands. The aarch64 binary was rebuilt
  from this branch's crate source, which is identical to main's. Its DT_NEEDED entries are
  `libgcc_s.so.1` and `libc.so.6`, and its highest symbol version is GLIBC_2.34. It has never
  been executed on the box. The verified cross-build command is:
  `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc" cargo build --release -p qd-prep --bin qd-prep --target aarch64-unknown-linux-gnu --target-dir /Users/bharath/qd-campaign/target-aarch64-linux`
  It produces sha256 c0de659c…62163.
- **Not done:**
  - Port `candidate_pairs` (LSH banding) into qd-prep. It was 121 s in the J1 cProfile and
    12–19 s here. It would patch `qd_data.dedupe.candidate_pairs` at the same seam.
    `max_pairs` early return, insertion-order buckets and two calls with different signature
    sets all need adversarial parity.
  - Port `shingle` to Rust, with a whitespace table that matches `str.isspace` exactly and
    truncation at 262,144 bytes.
  - Stop re-tokenizing the same text across `census`, `write_shards` and stage 8. This is
    Python orchestration in this file and in `qd_train/shards.py`. Tokenization itself cannot
    move to Rust here, because the `tokenizers` crate needs onig (C).
- `test_gaps_writer` fails in any `.claude/worktrees/<agent>` checkout, because the directory is
  not named `Lappi-decision`. That failure is environmental, and the test was not run green.

## First command for the next lane

```
cargo build --release -p qd-prep && \
QD_PREP_BIN=$PWD/target/release/qd-prep QD_PREP_BENCH=1 \
uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \
  --with datasketch python -m pytest -o addopts= -q -s -p no:cacheprovider \
  -k "not mps" python/tests/test_qd_prep_minhash_parity.py
```
