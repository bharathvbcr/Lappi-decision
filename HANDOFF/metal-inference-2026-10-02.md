# L-metal: Mac inference on qd-metal (2026-10-02)

Mandate: Fable's Mac-inference ruling
(`AUDIT/fable-optimize-2026-10-02/fable-improve-and-mac-inference-ruling.md`, Q2(a)-(d), to-do
items 3-6) and `AUDIT/fable-optimize-2026-10-02/human-decisions.md` item 7 (Mac GPU checks
allowed one at a time, only when the Mac is free). Labels: **[V]** run or read here, **[I]** inferred, **[U]** unverified.

**Every GPU number below is the base weights**: Hugging Face snapshot Qwen3.5-2B-Base
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`. No qd-export release exists on this Mac
(`GAP-QDM-NO-TRAINED-RELEASE-ON-MAC-2026-10-02`), so nothing here measures a trained model.

## Measured (ledger: `ledger/mac-qd-metal-2026-10-02.jsonl`)

Every row was written by the run itself, through `qd_metal::ledger::write_row` (Rust) or
`python -m qd_train.ledger record` (build rows), into one hash chain. Each Rust-written row
records the exe's sha256 (`code_that_ran`), the Lappi commit, and canonical tessl's HEAD
`cf65d9d5` + the sha256 of each of its 27 dirty paths (`tessl_dirty_files`; the ojas session's
kept, uncommitted exact-f32 GEMM panel-walk change). Release binaries were built at 01:46:42-55
from `c1fa18f`'s code; tessl's library inputs (`src/`, `kernels/`, `build.rs`, `Cargo.toml`)
have not changed since 01:32:30 [V, `find -newermt`].

| Row id | Kind | What | Result |
| --- | --- | --- | --- |
| `669e1ba9-bf81-40fe-a50a-0bbe3a58e189` | build | CPU suites at `c1fa18f`'s code | qd-metal 42/50 (8 ignored), snapshot_cpu `--ignored` 4/4, qd-train lib 50/50, ledger_oracle 5/5; exit 0 |
| `0112dade-0cd0-4034-be74-d20a398a4545` | smoke | parity gate, short set | **PASS** (numbers below) |
| `2a8bc443-6e62-4959-9b58-3d864f73c5f3` | build | `tests/gpu.rs --ignored`, first run ever | **FAILED** 3/4: `gpu_backend_readonly_decode_leaves_the_runtime_hash_unchanged` at gpu.rs:171 (below) |
| `a9ef3c3a-c262-4119-960d-217f90c5dad2` | build | `tests/gpu.rs --ignored` after correcting that test | 4/4 passed |
| `ebb17358-fbe6-4320-a659-630f2400bb8d` | throughput (`quick`) | `qd-metal-bench --decision`, both embed arms interleaved | numbers below |
| `2b858908-ca96-48b6-91ef-7a624ac4a3b5` | build | CPU suites at the final code | qd-metal 43/51 (8 ignored: 4 GPU, 4 snapshot), snapshot_cpu `--ignored` 4/4, qd-train lib 50/50, ledger_oracle 5/5; exit 0 |
| `c9452efc-c24e-49e2-9300-2d4e8e9d1ee1` | smoke | parity gate, long set | **PASS** (numbers below) |

The file was checked after the last row was written:
- `python -m qd_train.ledger verify`: "chain verifies: 7 row(s)", exit 0;
- `pytest python/tests/test_ledger.py python/tests/test_ledger_provenance.py`, which read every
  `ledger/*.jsonl`: 106 passed, exit 0.

The GPU test binary for row `a9ef3c3a` was rebuilt at 02:52 from `2bcfc03`'s test code. The parity
and bench executables are the 01:46 builds.

### Parity (rule 2's gate, thresholds read-only, main's committed `parity.rs`)

Short set: 24 real commitpackft prompts, 180-416 tokens (prefixes 119-355), torch 2.12.1 fp32 on
CPU, dumped this session into `/Users/bharath/qd-campaign/qd-metal-ref-2026-10-02/short/`
(manifest sha256 `e24847ae…`; dump exit 0, 365 s). Row `0112dade`:

| Check | Worst over 24 | Bound |
| --- | --- | --- |
| per-layer residual rel L2 vs bf16-input torch | 5.44e-3 | 2e-2 |
| per-layer residual rel L2 vs fp32 torch | 4.81e-3 | 5e-2 |
| letter log-softmax max abs diff vs bf16-input | 0.0341 | 0.10 |
| letter log-softmax max abs diff vs fp32 | 0.0394 | 0.25 |
| snapshot path (prefill + continue vs one pass) | 0.0345 | 0.05 |
| argmax agreement with fp32 | 24/24 | ≥ 95% |
| non-finite values in the gate's object or output | 0 | 0 |

Tokenizer: 24/24 prompts tokenize identically to the Python reference. Weight hash
`92f6bd1c…71fba2e1`, device Apple M5 Pro, weights loaded in 4.7 s.

Long set: 20 prompts, 716-1397 tokens (prefixes 655-1336). It was asked for 700-3000, and
`prompts()` takes the first 20 qualifying rows in pool order (`dump_torch_reference.py:79-116`),
so none happened to exceed 1397. It was dumped to `.../long/` (manifest sha256
`8961a3cc…`; dump exit 0, 396 s). Row `c9452efc`:

| Check | Worst over 20 | Bound |
| --- | --- | --- |
| per-layer residual rel L2 vs bf16-input / fp32 | 6.54e-3 / 5.44e-3 | 2e-2 / 5e-2 |
| letter log-softmax max abs diff vs bf16-input / fp32 | 0.0365 / 0.0308 | 0.10 / 0.25 |
| snapshot path | 0.0327 | 0.05 |
| argmax agreement with fp32 | 20/20 | ≥ 95% |
| non-finite values | 0 | 0 |

For scale, the torch reference's own fp32-vs-bf16-input gap on the letters is ≤ 0.0378 (short)
and ≤ 0.0282 (long) [V, dump logs]. qd-metal's distance from either reference is the same size
as the reference's distance from itself. The parity gate covers T ≤ 1397. The `--decision`
bench's T=2048/8192 decisions are checked for arm bit-identity and finiteness, **not** against
torch.

### GPU tests (`tests/gpu.rs`)

The first run of the suite on this Mac (row `2a8bc443`) **failed** one test:
`gpu_backend_readonly_decode_leaves_the_runtime_hash_unchanged` compared
`fresh.state_digest()` with the first snapshot's, at gpu.rs:171. That is the sha256 of the
52-byte `StateRecord`, which names the *entry*, and `Worker::snapshot` (`backend.rs:522-547`)
inserts a new entry for every snapshot by design (so a write-back on one snapshot never reaches
another). Two snapshots of one unchanged prefill therefore always hash differently: the test
encoded the wrong expectation. A diagnostic re-run of the same binary with `--nocapture`
(not recorded as a row; log `target/l-metal-logs/c-gpu-tests-diagnostic.out`) showed it and that
the other three pass. The corrected test compares the record's device-state `digest` and `tokens`
(what the comment says it checks, and a check that still fails if the write-back changed the
cached prefill's state) and asserts the two entries differ. Re-run, row `a9ef3c3a`: 4/4.

- `gpu_continuations_read_the_snapshot_and_never_write_it`: continuation vs whole pass, max abs
  diff in letter log-prob 0.0262 (1 token), 0.0101 (5), 0.0177 (61); 3-row batch vs batch-1:
  0.0 on every row; the snapshot's digest unchanged throughout.
- `gpu_write_back_moves_only_the_new_state`: prefix → write-back 30 → continue 31 vs whole:
  0.0215.
- `gpu_embed_paths_are_bit_identical` (new): Host vs Device embedding, bit-identical on 161
  gathered rows, the prefix state's digest, the continuation's letters and a whole pass's letters.
- `gpu_backend_readonly_decode_leaves_the_runtime_hash_unchanged` (corrected): passes.

### `--decision` latency (row `ebb17358`, `quick`, base weights)

One `choice` slot of k=4 options, the backend's own sequence: one prefill, 3 host SHA-256 state
digests, 2 passes of 61 tokens each (the rendered suffix and the permuted second pass). Min and
median of 7 interleaved samples after 2 warm-ups, ms. tessl unchanged during the run
(`tessl_unchanged_during_run` pass).

| T (prefix tokens) | Arm | Total min | Total median | Prefill | 2 passes | 3 digests | State |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 (493) | embed=host | 409.72 | 412.46 | 125.10 | 66.40 | 221.56 | 32.3 MB |
| 512 (493) | embed=device | 410.53 | 411.51 | 124.91 | 66.27 | 221.27 | 32.3 MB |
| 2048 (2026) | embed=host | 1059.36 | 1064.66 | 502.21 | 81.95 | 478.64 | 70.0 MB |
| 2048 (2026) | embed=device | 1060.67 | 1064.76 | 506.47 | 82.69 | 478.10 | 70.0 MB |
| 8192 (8187) | embed=host | 4275.68 | 4293.45 | 2635.96 | 148.24 | 1513.46 | 221.4 MB |
| 8192 (8187) | embed=device | 4281.09 | 4316.27 | 2654.14 | 148.31 | 1511.46 | 221.4 MB |

Per decision (median, `tessl::infer_trace`): 920 dispatches and barriers (923 with
embed=device), **5 commits** (host syncs), **101 cold allocations** (104), sync-wait 0.19 s /
0.58 s / 2.70 s at the three T. Arms bit-identical: 14/14 samples at every T (both passes'
logits and the prefix-state digest).

What the numbers say:

- **The host digest is the largest cost after the prefill**: 53.7% of the median decision at
  T=512, 45.0% at 2048, 35.3% at 8192. It hashes at a flat **0.44 GB/s** (3 × state bytes /
  digest time, 438-439 MB/s at all three T), not the 1.5-2.5 GB/s the ruling assumed [U there],
  so the ruling's 25-35% / 20-25% estimate was low. Whether `sha2` 0.10.9 is using the ARMv8 SHA
  extensions here, and how much of the time is `try_contents_u8()`'s copy rather than hashing,
  is **[U]**: not measured separately. The device digest (`AUDIT/metal-digest-2026-10-02/`)
  would remove most of it; a faster host SHA-256 is the no-tessl alternative.
- **`embed_rows` buys nothing measurable**: medians are within 0.3% at T=512 and 2048, and the
  device arm's prefill median is 0.7% slower at 8192, within this run's spread. It is exact, so
  it stays as a flag; the default stays `Host`.
- 101 cold allocations per decision: buffers are allocated per call. Pooling them is an
  unmeasured candidate [I].
- One process on one host, started on a clear pre-check (02:52:56); what other lanes started
  during its 110 s is not known. `quick`: it promotes nothing (rule 8).

## Changed (commits on `worktree-agent-a70ae6b7242dade65`, from main `276a475`)

| Commit | What |
| --- | --- |
| `14a4047` | `crates/qd-train/src/ledger.rs`: `Row`, a ledger-schema row for any non-build `run_kind` (validated kind, `quick_reason`, wall clock, non-empty recipe, `recipe_hash`), and `append_line` (flock, duplicate check, `prev_row_hash` chain, `O_APPEND` + `fsync`) without the trainer's path rule. `FtRow::line` is rebuilt on `Row` and writes the same bytes (`ledger_oracle` passes). |
| `c1fa18f` | `crates/qd-metal`: the `qd-metal-serve` bin (`src/serve.rs`, `src/bin/serve.rs`); `qd-metal-bench --decision` (`src/decision.rs`); the `EmbedPath { Host, Device }` flag on `Model` (`src/model.rs`); schema ledger rows (`src/ledger.rs`, restricted to `ledger/mac-qd-metal-*.jsonl`, recording the exe sha256, the Lappi commit and canonical tessl's HEAD + dirty-file sha256s); `qd-metal-record-parity` (`src/bin/record_parity.rs`, `src/parity_row.rs`); tests `serve_cpu.rs`, `ledger_rows.rs`, `gpu.rs::gpu_embed_paths_are_bit_identical`, `snapshot_cpu.rs::snapshot_decision_prompts_fill_each_t_from_below`. |
| `5f840dd` | `AUDIT/metal-digest-2026-10-02/`: the device-side state digest's design, the proposed tessl kernel and its CPU oracle (item 5 below). |
| `2bcfc03` | Two test corrections found by running them. `tests/ledger_rows.rs` fails closed without a Python: it printed SKIPPED and returned, which libtest counts as `ok`, and the `c1fa18f` binary read "4 passed" in this worktree with no venv and no `QD_PYTHON` [V]. It now tries `QD_PYTHON`, `<repo>/.venv`, then `python3`, bounds each call at 60 s, and `no_usable_python_fails_rather_than_skips` pins the panic. `tests/gpu.rs`: the fresh-snapshot check (below). Rows `a9ef3c3a` and `2b858908` ran this code before it was committed, so they record `5f840dd-dirty`. |
| the commit carrying this file | `ledger/mac-qd-metal-2026-10-02.jsonl` (the rows above), `gaps.jsonl` records, this HANDOFF. |

Not committed, on purpose: `Cargo.lock` (rule; its diff adds the `clap` and `qd-train` edges to
`qd-metal`, and main's committed lock was already missing `qd-train`, `qd-train-metal` and tessl's
`block2`/`dispatch2` before this lane). No frozen file was edited: `backend.rs`, `parity.rs`,
`tokenizer.rs`, `serve.rs`, `service.rs`, `tests/common/mod.rs`, `tests/lifecycle.rs`,
`tristate.rs` are byte-identical to main `276a475` here (`git diff --stat 276a475 --` on them is
empty [V]), and nothing was created under `crates/qd-metal/tests/fixtures/`.
Nothing in tessl was written; tessl was read with read-only git (`rev-parse`, `status`, `log`).

## Work items

1. **`--decision` bench harness** [V]. `qd-metal-bench --decision T=512,2048,8192 k=4
   --ledger <ledger/mac-qd-metal-*.jsonl>`. One decision = the backend's own sequence
   (`backend.rs:445-502`, `:548-657`): prefill, synchronize, digest; then two passes (the
   rendered slot suffix and the permuted second pass from `second_pass_permutation`), each a
   `Model::run` + `score` + digest, and the read-only digest compared after each. Prompts are
   rendered through `wire::parse_line` → `render` → `permuted_slot_suffix`, from `model.rs`'s own
   source repeated as context, and binary-searched to the largest prefix ≤ T tokens (T=512 → 493,
   2048 → 2026, 8192 → 8187 prefix tokens [V, `snapshot_cpu.rs`]). Arms are interleaved, the order
   rotating each round; min-of-7 and median after 2 warm-ups; `tessl::infer_trace` counts per
   decision. The row is `run_kind: throughput`, `quick: true` (one host, one session; it promotes
   nothing). Canonical tessl's tree state is read before and after each T, and a T is discarded
   and the run stops if it moved (the lead's rule: never interleave an A/B across a tessl change).
2. **`qd-metal-serve`** [V on CPU only]. `Release::open` → `MetalBackend::start` →
   `Runtime::from_release` → `Service::with_factory` → `Server::bind`, from qd-runtime's public
   API on main; qd-runtime still does not depend on qd-metal. The release is opened before the
   socket is bound, so a bad release refuses without a socket. Flags: `--release` (required),
   `--socket`, `--idle-timeout-ms` (default **qd-runtime's** `ServiceConfig::default()`, 600 000
   ms; a longer one is the human's), `--request-timeout-ms`, `--max-in-flight`,
   `--max-connections` 64, `--queue-capacity` 8, `--max-entries` 8, `--max-tokens` 16384,
   `--job-timeout-ms` 30 000. The factory takes a `BackendStarter`, so the CPU tests drive the same
   wiring with the reference backend. No GPU run: there is no release to open
   (`GAP-QDM-NO-TRAINED-RELEASE-ON-MAC-2026-10-02`).
3. **`embed_rows`**. `Model::set_embed_path(EmbedPath::Device)` replaces the host gather
   (`model.rs` `embed_rows_host`, main's 462-478) with `tessl::qwen35::embed_rows` over the
   resident bf16 embedding. Default stays `Host`: `MetalConfig` cannot carry the flag while
   `backend.rs` is frozen (`GAP-QDM-BACKEND-CANNOT-CARRY-AB-FLAGS-2026-10-02`), and moving the
   default moves the product path, which needs the parity suite re-passed with it (rule 2). The
   GPU test `gpu_embed_paths_are_bit_identical` compares the gathered rows, the prefix-state
   digest, the continuation's letter log-probs and the whole pass's logits bitwise.
4. **GPU runs**: below, under Measured.
5. **Device-side digest**: not implementable in `model.rs` on tessl's current exports (no
   hash or integer-reduction kernel; an overlay metallib needs a `build.rs`). Delivered as
   `AUDIT/metal-digest-2026-10-02/`: the design, `state_digest.metal` (proposed tessl kernel +
   wrapper signature), and `reference.rs`, a std-only oracle whose 5 tests pass (`rustc --test
   reference.rs`, then the binary: 5 passed, 0 failed, exit 0 [V]). Construction: 8 u32 lanes,
   each the XOR over words of `fmix32(word ^ key(lane, index))`: deterministic whatever the GPU's
   reduction order, and any one-bit flip changes every lane. Needs a tessl commit by its owner, or
   the human's yes to a qd-metal overlay build.
6. **Deferred**: items 3 and 4 of the ruling, with the exact lines, below.

## Deferred: the exact lines items 3 and 4 need

Line numbers are main `276a475`'s, and are this worktree's too. `backend.rs` is held by another
session's uncommitted work, which is why neither was started.

**Ruling item 3: batch the two passes and all slots as one `Model::run` of B rows**
(`GAP-QDM-ONE-SLOT-PER-DECODE-CALL`).

- `crates/qd-runtime/src/backend.rs:211-234`: `DecisionBackend` gains `decode_slots(&self,
  snapshot: &mut StateSnapshot, queries: &[SlotQuery<'_>], mode) -> Result<Vec<Logits>, _>`,
  with a default impl that loops over `decode_slot`, so `ReferenceBackend` and every test
  double keep working unchanged.
- `crates/qd-runtime/src/answer.rs:284-320` (`answer_generic`: the first pass at 301 and the
  permuted second pass at 316) and `:363`, `:406`, `:412`: each pair of `readonly_decode` calls
  becomes one `decode_slots` call. `readonly_decode` (`answer.rs:548-588`) checks the snapshot
  digest once around the batch rather than per call.
- `crates/qd-runtime/tests/answering_procedure.rs:436-497`: the recorder counts calls; it must
  count *queries* (the ruling's contract-pin change), and `docs/schema-api.md`'s decode table
  says the same.
- `crates/qd-metal/src/backend.rs:119-137` (`Job::Decode` carries one suffix/slot/rows/kind):
  a `Job::DecodeMany { snapshot, queries: Vec<(String, String, usize, QueryKind)>, mode, reply }`.
- `backend.rs:254-276` (`decode_slot` impl): add `decode_slots`, submitting one `DecodeMany`.
- `backend.rs:369-398` (`Worker::serve`'s dispatch): route `DecodeMany`.
- `backend.rs:548-657` (`Worker::decode`): the tokenization, boundary and `max_tokens` checks
  (`:587-617`) run per query; the run becomes `Model::run(&concat, B, seq, Some(&state), false,
  None)` over B rows sharing the stride-0 prefix, grouped by continuation length (or tessl's
  `*_varlen` kernels), then one `score` over B answer positions and **one** digest
  (`:625-628`) instead of B. Write-back stays one query per call.
- Oracle: logits equal to the sequential path to 1e-5, a new `tests/gpu.rs` test.

**Ruling item 4: span serving.**

- `backend.rs:558-564`: `Worker::decode` refuses any `kind != QueryKind::Letters` (the ruling
  cites 589-595; this is the same refusal on main). Span `PointerStart`/`PointerEnd` queries are
  served here once the head exists.
- `backend.rs:291-303` (`Entry`): keeps the prefill's line-start hidden rows (or their indices
  plus a device buffer); `Rc<PrefixState>` today holds only conv/GDN/K/V.
- `backend.rs:445-502` (`Worker::prefill`): `Model::prefill` must return the residual at the
  line-start positions (post-final-norm, `qwen35::rms_norm` with `w_offset` 1 [I, from the ruling,
  `backbone.py:1080`]) and the entry keeps it.
- `backend.rs:278-287` (`pooled_features`): stays `Unavailable` for the base weights
  (`GAP-QDM-POOLED-FEATURES-UNDEFINED`); span serving does not need it.
- `backend.rs:327-335` (`Worker::new`): loads `span_head.safetensors` from the release (two
  [H,H] matrices + abstain vectors) beside the model.
- Oracle: `heads.py` on dumped hidden states, bit-level; the abstain row last (`answer.rs:69-71`).

## Not run, skipped, and caveats

- **`qd-metal-serve` on the GPU: not run.** There is no release on this Mac. Its wiring is tested
  on the CPU only (`tests/serve_cpu.rs`, 5 tests, and 3 in the bin).
- **Device-side digest: not implemented** (item 5 above). It is the measured biggest lever.
- **Ruling items 3 and 4: deferred** (the lines are above).
- **`EmbedPath` default not moved**: no measured gain, and `MetalConfig` cannot carry the flag.
- **No trained weights measured.** Every number is the base snapshot `b1485b2f`.
- **Waiting.** The GPU sequence waited from 01:47 to 02:20 (and again 02:22-02:37) for the
  pre-check. It was blocked by the ojas session's `cargo test` builds (rustc) and its tessl
  `device` GPU tests, L-replay's `j6a` prelude job (python, 14 GB), and L-replay/L-prep pytest
  suites (python, 1.3-6 GB). Every run was gated by a logged pre-check run immediately before it.
  The pre-check checked for:
  - rustc;
  - any process whose arguments mention bench/tessl/matmul (the lead's rule);
  - cargo bench;
  - python/torch/mlx over 2 GB;
  - L-replay pipeline patterns;
  - free + inactive memory ≥ footprint + 4 GB.
  The logs are in `target/l-metal-logs/*.precheck` (not committed). The 2 GB python threshold is
  this lane's reading of rule (2): about 25 small MCP python servers are always present.
- **One overlap of my own:** a 0.7 s `cargo test -p qd-metal --test ledger_rows` build and run
  happened during the short torch dump. That dump is CPU torch, so its outputs are unaffected.
- **`tessl_state_predates_exe` reads failed on the parity and bench rows.** tessl's newest change
  (01:47:36) is after the binaries were linked (01:46:53 / 01:46:55), but the files that changed
  are `CHANGELOG.md` and `tests/gemm_{flag_paths,ragged_shapes}.rs`, which are not inputs to the
  tessl library qd-metal links [V, `find -newermt '2026-10-02 01:46:50'`]. The metric counts every
  dirty path, so it is conservative. It is reported as it ran, and the refinement is
  `GAP-QDM-TESSL-PREDATES-EXE-COUNTS-NON-BUILD-FILES-2026-10-02`.
- **The diagnostic GPU re-run** that showed which test failed is not a ledger row. The failure is
  row `2a8bc443`; the pass after the correction is row `a9ef3c3a`.

## Open gaps (appended this lane, via `qd_train.gaps.append_gap`)

| Id | Status | Owner |
| --- | --- | --- |
| `GAP-QDM-GPU-GATES-NOT-RUN` | resolved-with-residual (base weights only; re-run at the merge) | agent |
| `GAP-QDM-NO-LEDGER-ROW-ALL-MAC-INFERENCE-NUMBERS-BY-REPORT-2026-10-02` | resolved-with-residual (memory per request beyond state bytes; trained weights) | lead |
| `GAP-QD-SERVE-WIRES-NO-METAL-BACKEND-2026-10-02` | resolved-with-residual (no GPU end-to-end: no release on the Mac) | lead |
| `GAP-QDM-STATE-DIGEST-HOST-SHA256-PER-DECODE-2026-10-02` | open (measured 35-54% of a decision; design in AUDIT) | lead |
| `GAP-QDM-BACKEND-CANNOT-CARRY-AB-FLAGS-2026-10-02` | open | agent |
| `GAP-QDM-NO-TRAINED-RELEASE-ON-MAC-2026-10-02` | open | lead |
| `GAP-QDM-PARITY-LEDGER-ROW-NOT-SCHEMA-2026-10-02` | open (worked around by `qd-metal-record-parity`) | agent |
| `GAP-QDM-SERVE-COLD-START-UNDER-LIFECYCLE-LOCK-2026-10-02` | open | agent |
| `GAP-QDM-RUNTIME-OPENED-WITH-TIMESTAMP-HEAP-2026-10-02` | open | agent |
| `GAP-QDM-TESSL-PREDATES-EXE-COUNTS-NON-BUILD-FILES-2026-10-02` | open | agent |
| `GAP-LMETAL-NAVIGATION-2026-10-02` | open (DevMap store degraded/absent, GitPulse untrusted, no ListAgents) | lead |

Still open from before, and named above: `GAP-QDM-ONE-SLOT-PER-DECODE-CALL` (ruling item 3),
`GAP-QDM-POOLED-FEATURES-UNDEFINED`. `python/tests/test_gaps_ledger.py`: 10 passed, exit 0.

## First command for the next lane

This branch and the other session's uncommitted `backend.rs`/`parity.rs`/`tokenizer.rs` work
meet at merge. Whichever lands second re-runs the gates, because a `backend.rs` or `tokenizer.rs`
change can move a logit (rule 2). From the main checkout, after the merge, one process at a time
and after the pre-check:

```sh
cargo build --release -j 4 -p qd-metal --bins && \
target/release/qd-metal-record-parity --parity-bin target/release/qd-metal-parity \
  --ledger ledger/mac-qd-metal-<date>.jsonl \
  --raw-out /Users/bharath/qd-campaign/qd-metal-ref-2026-10-02/parity-raw-postmerge.jsonl \
  -- --fixtures /Users/bharath/qd-campaign/qd-metal-ref-2026-10-02/short
```

then `PYTHONPATH=python .venv/bin/python3 -m qd_train.ledger record --ledger
ledger/mac-qd-metal-<date>.jsonl --toolchain "<cargo/rustc/Metal>" --suite qd_metal_gpu="cargo
test --release -j 4 -p qd-metal --test gpu -- --ignored --test-threads=1"` (expect 4/4).

After that, by measured size:
1. The digest: tessl's owner takes `AUDIT/metal-digest-2026-10-02/state_digest.metal`, or the
   human allows a qd-metal overlay `build.rs`. Before either, split `digest_ms` into
   `try_contents_u8` copy and SHA-256 time, and check whether `sha2` uses the ARMv8 SHA
   instructions here.
2. Ruling item 3 (batched `decode_slots`, lines above), once `backend.rs` is free.
3. `GpuRuntime::new_inference()` as a bench arm
   (`GAP-QDM-RUNTIME-OPENED-WITH-TIMESTAMP-HEAP-2026-10-02`).
