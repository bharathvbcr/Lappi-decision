# HANDOFF v6-spancheck, 2026-10-03

Lane: v6 "spancheck". Worktree `build/v6-spancheck-wt`, branch `v6-spancheck`, based on
`8e6a009`. Local commits only; nothing pushed, nothing on main.

The lane ports the Python hot loop of shard writing's span work to Rust:
`qd_train.shards._span_token_positions`, every step of it that does not decode
(`AUDIT/perf-pipeline-shards-2026-09-30.md`). It is a new `qd-prep spancheck` subcommand. The
Python stays the definition and the unedited reference; `shards.py` and `python/qd_data/` are not
edited. The native path is off unless `--native-spancheck` is passed, so v5 is unaffected.

Labels: **[V]** produced by a command this lane ran from the worktree (cited below), or read at
file:line; **[I]** inferred; **[U]** not verified.

## Read this first

1. **The bar holds end to end on the real tokenizer [V].**
   - The test: `test_the_pipeline_writes_every_shard_set_the_reference_writes` runs the
     pipeline's `run()` with and without the flag, on Qwen3.5's tokenizer, over 60 commitpackft
     pairs plus the val set.
   - The cases: refuse-any and refuse-gold, each with single blank lines and with blank-line
     runs.
   - The result, in all four cases:
     - every file of `shards/train`, `shards/val` and `shards/train-no-decode` is the
       reference's (`header.json` without `created_at`; each `.npz` by its arrays; every other
       file byte for byte);
     - everything `run()` measured (metrics, gates, hashes, notes) is the reference's, except
       the one metric the flag adds.
   - Lookups answered from the reply, by case (88 lookups each: 42 train sequences read twice,
     at stages 6 and 8, plus 4 val):

     | Case | From the reply | Refused through the reference |
     |---|---|---|
     | refuse-any, single blank | 88 | 0 |
     | refuse-any, blank runs | 0 | 88 (`lines_collapse`) |
     | refuse-gold, single blank | 88 | 0 |
     | refuse-gold, blank runs | 84 | 4 (`lines_collapse`, the val slots) |

     - With blank-line runs, Qwen3.5's BPE merges each passage's blank-line run into one token
       holding several line starts, so under refuse-any the reference refuses every span slot
       whole.
     - The 4 under refuse-gold are val slots. `run()` writes a gate val set under refuse-any
       whatever the train policy (`tools/real_tokenizer_pipeline.py:3665-3668`).
2. **It is not faster. Keep `--native-spancheck` off [V].** The committed benchmark is an
   interleaved A/B, min of 5, over a fixed 600-row slice of `commitpackft-corpus-v2`, with the
   pre-pass, subprocess and canaries inside the native arm:
   - byte tokenizer: reference 0.279 s, qd-prep 0.297 s (**0.94x**);
   - Qwen3.5: reference 0.344 s, qd-prep 0.424 s (**0.81x**).

   The reason is in the AUDIT the port was planned from [V, read]. In its phase-3 profile
   (626.1 s, code of 2026-10-01), `_span_token_positions` is 101.2 s cumulative:
   - `offsets` (an encode under `--memo-limit 0`): 60.9 s;
   - the decode check: 35.2 s;
   - the pure-Python mapping this port replaces (`_token_index_for_char`, 11.9 s, plus its
     generator and the function's own 1.4 s): about 13 s.

   The port removes the mapping and the lookup's `offsets` encode, but adds a pre-pass that
   renders and encodes each span sequence once more. Decoding stays in Python. On the slice the
   pre-pass alone is 0.21 s of the native arm's 0.30-0.42 s. A speed-up needs both open items 1
   and 2 below; the port alone cannot give one.
3. **A process fault of mine; read before trusting any earlier run.**
   - Every heavy job I ran before 00:12Z on 2026-10-04 had cwd = main's checkout. That covers
     clippy, `cargo test --lib spancheck`, `cargo test --test input_bound`, the release build and
     the parity pytest runs.
   - Main, the worktree and their parents carry no `.cargo` config, and every cargo call used
     `--manifest-path`, so their results are inferred equal [I]. They are not evidence here.
   - Every [V] in this file comes from the reruns with cwd = the worktree (00:23Z onward).
   - Two side effects:
     - **A stale lock.** I stopped the cwd = main full-suite run with TaskStop, which killed
       `mac_heavy.sh` before its release trap ran. `build/mac-heavy.lock.d` stayed stale (pid
       58892) until the lead cleared it at about 00:21Z. The peer's `qdm-merge-test` had waited
       since 00:06:47Z.
     - **Cache files in main.** 22 files appeared in main's gitignored `.hypothesis/constants/`:
       17 at 00:02:20Z and 5 at 00:05:56Z, the start times of my two cwd = main pytest runs.
       They are hypothesis's example cache. Left in place for the lead. Main's `target/tmp`
       entries from that window are the peer's `wip-review` job (00:05:09Z), not mine.

## Commits

| Commit | What |
|---|---|
| `a9ddf61` | `qd-prep spancheck`: `crates/qd-prep/src/spancheck.rs` (new), the subcommand in `main.rs`/`lib.rs`, `wire::le_u32s` made `pub(crate)`, an input-bound test |
| `cbc0edc` | `tools/real_tokenizer_pipeline.py`: `--native-spancheck` (default off), `build_native_spancheck`, `NativeSpancheck`, `RealTokenizer.encode` |
| `363bb08` | `python/tests/test_qd_prep_spancheck_parity.py` (new): parity, refusals, `write_shards` byte-equality, Qwen BPE, the benchmark |
| `4e97e02` | the same test file: the end-to-end `run()` test; the collapse test's wrong expectation (failed on its first run, fixed, both numbers in the message); three ruff findings; `QD_PREP_DATA_ROOT` |
| this file's commit | this handoff; `gaps.jsonl` (`GAP-V6-SPANCHECK-NAVIGATION-DEVMAP-UNAVAILABLE-2026-10-03`) |

The commit messages of `a9ddf61`, `cbc0edc` and `363bb08` say "Not run". They were written
before the heavy-job lock was released. The runs are below.

Branch total `8e6a009..4e97e02`: 7 files, 2,933 lines added and 74 removed. With `git diff -w`,
2,869 added and 10 removed: the other 64 removals are lines re-indented under the new `with`
blocks. Of the 10, the pipeline has 7 (one import line and two comments re-wrapped at the new
depth), `wire.rs` 2 (`le_u32s` made `pub(crate)`) and `main.rs` 1 (the `use` line).

## Design as built

### Rust: `qd-prep spancheck --input IN --output OUT`

- **Footprint.** Std only, with threads via `std::thread::scope`; `Cargo.toml` and `Cargo.lock`
  are unchanged, so there is no new dependency [V: `git diff --stat 8e6a009`].
- **Request `QDPSCIN1`.** Per sequence:
  - flags (abstains, NFC-stable), policy (refuse-any 0, refuse-gold 1);
  - the UTF-8 text;
  - the id count;
  - the offsets, as code-point `(start, end)` u32 pairs;
  - the line starts;
  - the gold `(start, end)`.
- **Request refusals.** `parse` refuses:
  - bad magic, trailing bytes, an unknown flag or policy;
  - no line starts (the Python passes those straight to the reference);
  - an abstain carrying a span;
  - non-UTF-8 text;
  - any count past its bound.
- **Bounds.** 1 GiB a request, 2^24 sequences, 64 MiB a text. A run walk is capped at 2^22
  steps; over the cap, the runs are dropped and flagged, never sent in part.
- **Checks.** `check` (`spancheck.rs:372`) runs the reference's checks in the reference's order
  (`spancheck.rs:12-29`). So a non-ok status names the exception class the reference raises and
  the first index its message names:

  | Status | Exception class |
  |---|---|
  | `OffsetCount`, `Reach`, `CandidatePastEnd`, `GoldPastEnd` | ShardContractViolation |
  | `LineInNoToken`, `LinesCollapse`, `GoldInNoToken`, `GoldReversed`, `GoldInOneToken`, `GoldNotCandidate`, `GoldSharesToken` | UnencodableGold |

  `CandidatePastEnd` and `GoldPastEnd` cannot fire once `OffsetCount` passes, because a token
  index is always below the offset count. The reference has the same two checks and they are
  kept for fidelity [I].
- **The token holding a code point.** The lowest token index whose span contains it. It is found
  by one sweep over the tokens with a next-unclaimed pointer array (path halving), linear in
  code points plus tokens.
- **What an ok reply adds:**
  - each checked position's decode-check run `(run_end, first, last)`, in the reference's order
    (gold first, then candidates);
  - whether every line start lies on the text's own line grid (`artifacts.line_start_indices`:
    `\n` ends a line, `\r` is content, empty text has no lines);
  - the NFC flag, echoed and never decided.
- **Reply `QDPSCOK1`.** Columnar: 24-byte per-sequence headers, then all candidates, then all
  runs.

### Python: `tools/real_tokenizer_pipeline.py`

- **Pre-pass.** `build_native_spancheck(groups, ...)` (`:2930`) renders every span sequence the
  writes will ask about, with `training_texts(row, seed=config.seed)` and default caps. The four
  `write_shards` calls in `run()` render the same way: they pass neither `caps` nor `seed`
  (`shards.py:1452,1606,1636`). It encodes each sequence once (`RealTokenizer.encode`, one
  encode for ids and offsets) and streams requests of at most 256 MiB.
- **Reply checks.** Each reply is checked in full against its request.
- **Canaries.** After every call, the first and last 4 sequences re-run through the reference:
  the class and value with `decode=None`, and the whole answer (value, or exception class and
  text) with the run's `decode`.
- **Install.** `NativeSpancheck.installed()` swaps `qd_train.shards._span_token_positions` for
  `lookup` during one `write_shards` call, as `native_minhash` does for MinHash.
- **When the reply is the answer.** `lookup` returns it only when the reference provably returns
  the same (`_native_answer`, `:2682`):
  - Every non-ok status goes to the reference, so exception text is the reference's by
    construction.
  - With a `decode`, these also go to the reference:
    - every sequence whose decode check the reply does not settle (NFC-unstable text, a line
      start off the grid, runs over budget);
    - any sequence whose pieces or whole decode differ from what the offsets claim. Those
      decodes run in the reference's order.
- **Refusals by `lookup`:**
  - a sequence the table lacks;
  - a `token_offsets` callable not equal to the pre-pass's;
  - ids whose length or blake2b digest differ from the pre-pass's.
- **Refusals elsewhere:** a table never read (`finish()`); a second install.
- **Wiring in `run()`:**
  - stage 5c builds one table for train, val and replay;
  - stages 6, 6b, 6c and 8 each install it;
  - the census keeps the reference.
- **The metric.** `metrics["native_spancheck"]` is `Ran`, with `value` = the qd-prep binary's
  sha256 (which `code_fingerprint` cannot show) and `n`/`n_total` = lookups answered from the
  reply over all calls. Its `detail` carries the counts by reason.
- **Not recipe.** The flag is not a recipe key, like `--memo-limit`. Without it every write runs
  under `contextlib.nullcontext()`, with no metric and no recipe key [V: the end-to-end test
  asserts the reference arm has no `native_spancheck` metric and is otherwise equal].

### The one assumption the fast path rests on

`decode` is a pure function. With a `decode`, the native answer re-runs the reference's decode
checks except one: its second whole-sequence decode, used for the round-trip comparison, is
skipped and the first result reused. A `decode` that answers differently for the same ids on a
second call would pass here and fail the reference [U for arbitrary callables; Qwen3.5's
`tokenizer.decode` is deterministic by construction [I]].

## What was run (all from the worktree, each through `tools/mac_heavy.sh`) [V]

All times are UTC on 2026-10-04.

| Run | Command (cwd = worktree) | Result |
|---|---|---|
| 00:23:20-00:23:43 | `cargo clean -p qd-prep; cargo clean -p qd-prep --release; cargo test -p qd-prep -j 2; cargo build --release -p qd-prep --bin qd-prep -j 2` | 92 lib tests passed (13 of them spancheck); 3 `input_bound` tests passed, including `spancheck_refuses_past_its_own_bound_and_writes_nothing`; 0 failed. Release qd-prep sha256 `633fa7ccbf31ed093d6ccb82334538334f7460a8c47d6552923e113507dd9d64`, the binary every run below used |
| 00:24:03-00:24:05 | `cargo clean -p qd-prep; cargo clippy -p qd-prep --all-targets -j 2 -- -D warnings` | the log shows "Checking qd-prep"; exit 0, no findings |
| 00:27:07-00:36:27 | main's `.venv/bin/python -m pytest -p no:cacheprovider -rfEs --ignore=python/tests/test_needle_handoff.py python/tests` (the `.venv` convention of `HANDOFF/v5-data-2026-10-02.md:306`: `test_needle_handoff` imports torch, which `.venv` lacks) | **4 failed, 3049 passed, 198 skipped** in 559.76 s. None of the 4 involves this lane (see below) |
| 00:36:27-00:37:10 | the ML venv (`uv run --offline --no-project --python ~/.venvs/ml/bin/python --with pytest --with hypothesis`), `python -m pytest -p no:cacheprovider -rfEs -s python/tests/test_qd_prep_spancheck_parity.py`, with `QD_PREP_BIN=target/release/qd-prep` and `QD_PREP_DATA_ROOT=<main checkout>` (read only) | **68 passed, 2 skipped** (the benchmark's two arms) in 42.69 s, on the file as committed in `4e97e02`. This includes the Qwen BPE test and the four end-to-end cases |
| 00:37:10-00:37:23 | as the ML row, with `QD_PREP_BENCH=1 ... -s -k benchmark` | 2 passed in 11.62 s; the numbers are in "Benchmark" below |
| 00:22Z, 00:29Z | `.venv/bin/ruff check --config pyproject.toml <worktree>` (ruff 0.16.8, the lint gate's locked version) | this lane's two Python files: clean. 10 findings in `tools/qd_train_oracle_head_digest.py` and `tools/qd_train_oracle_optimizer_table.py`, both unchanged since `8e6a009` (last commit `99a7816`); not this lane's |

**The suite's 4 failures.** Each fails on a condition present at `8e6a009` or on the worktree
itself. The branch changes none of the files involved: its only `gaps.jsonl` change is the one
appended line, and the required-keys failure does not name it.

| Test | Cause |
|---|---|
| `test_gaps_ledger::test_every_current_record_carries_the_required_keys` | `GAP-V5-2GPU-COST-INSTANCE-RATE-REFUSED-AT-ONE-GPU-2026-10-03`, `GAP-V5-2GPU-J5-PER-SEED-VS-RUNS-IFF-2026-10-03` and `GAP-V5-2GPU-R9-HOLD-HOLDS-J5-2026-10-03` lack `question` in their last line, as at the base |
| `test_gaps_ledger::test_every_gap_id_cited_in_a_tracked_file_exists_in_the_ledger` | `campaign/v5-preregistered.DRAFT.json` cites a gap id (the dedupe-LSH-band one, dated 2026-10-03) that the base's `gaps.jsonl` does not hold (`git show 8e6a009:`). It is not spelled out here, because spelling it would make this file a second dangling citation |
| `test_gaps_writer::test_the_real_ledger_is_not_touched_by_any_of_this` | it asserts the checkout's directory is named `qwen-decision` or `Lappi-decision`; a worktree is `v6-spancheck-wt` |
| `test_lint_gate::test_ruff_is_installed_not_merely_declared` | it requires `REPO/.venv/bin/ruff`, and a worktree has no `.venv`. That is also why the gate's own ruff command is in the table above |

The suite ran the test file before a docstring-only edit that is in `4e97e02`. The ML and
benchmark runs used the committed file.

**Where the imports came from [V].** `qd_train`, `qd_data` and `real_tokenizer_pipeline` load
from `build/v6-spancheck-wt` under main's `.venv` with cwd = the worktree. Main's `python/`
holds no module the worktree's lacks, so main's editable `.pth`, later on `sys.path`, cannot
supply one (stock `rg --files`, untracked included). The ML venv carries no `.pth` into either
checkout.

## Benchmark [V]

`test_benchmark_span_projection_reference_against_qd_prep_interleaved_min_of_n`. Each round
times both arms, alternating which goes first, over `write_shards`' per-slot span work: render,
`_tokenize_checked`, and `_span_token_positions` with the decode check, under refuse-any. It runs
600 span slots from the sha256-ordered first 600 rows of `commitpackft-corpus-v2`. The native
arm's time includes the pre-pass (a second render and encode per slot), the request, the
`qd-prep` process, the reply checks, the canaries and every lookup. Every round's answers were
equal: values, and refusals by class and text.

| Tokenizer | Reference, 5 rounds (s) | qd-prep, 5 rounds (s) | min / min | Native |
|---|---|---|---|---|
| bytes | 0.324, 0.284, 0.279, 0.292, 0.331 | 0.307, 0.297, 0.301, 0.300, 0.304 | 0.279 / 0.297 = **0.94x** | 600 of 600 from the reply; pre-pass 0.217 s |
| Qwen3.5 | 0.431, 0.344, 0.386, 0.388, 0.360 | 0.437, 0.427, 0.432, 0.445, 0.424 | 0.344 / 0.424 = **0.81x** | 523 of 600 from the reply, 77 refused (`lines_collapse`) through the reference; pre-pass 0.212 s |

The slice is small and the machine was shared, with load 5-10 throughout. These numbers rank
the two paths at this size; they are not a v5-scale throughput [U].

## Not verified, and assumptions

- **[U] The full-size A/B.** The bar is proven at 60 pairs plus val. The v5 recipe has not been
  built with and without the flag. That needs the general record and hours of Mac time, for a
  path the benchmark says not to turn on.
- **[U] Stage 6c (replay) under the flag.** No test writes a replay set with the flag. Replay rows
  come only from MMLU and CSQA (`qd_data/general.py:106`), which are choice families with no span
  slots, so stage 6c's lookups would all be pass-through [I].
- **[I] Memo key set.** With the memo on, the pre-pass's texts are the strings `write_shards`
  renders (same `training_texts` call, same seed, same caps), so the memo's key set and its
  `--memo-limit` refusal point are unchanged.
- **[I] The decode-purity assumption** above.

## Open

1. **Rust-side decode needs the `tokenizers` crate**, which is a new dependency and the owner's
   call. Until then decoding stays in Python: the decode check was 35.2 s of the AUDIT's 626.1 s
   profile.
2. **The double encode under `--memo-limit 0`**, which v5-size builds use. The reference already
   encodes each span sequence twice per write, once in `tokenize` and once in `token_offsets`:
   - in the AUDIT's profile, `encode_batch` ran 463,704 times (125.9 s) for 281,759 `tokenize`
     and 181,945 `offsets` calls, the census's included;
   - with the table installed, the lookup's `token_offsets` encode is gone, but the pre-pass adds
     one, plus a render.

   Removing an encode means a write taking ids and offsets from one encode. That edits
   `shards.py` (outside this lane), or `RealTokenizer` keeps the last text's encode, a one-entry
   cache whose hit pattern does not depend on row order [I, from
   `real_tokenizer_pipeline.py:607-635`; not built or measured]. Of the two open items, this is
   the larger and cheaper win, and it needs no Rust.
3. **`GAP-V6-SPANCHECK-NAVIGATION-DEVMAP-UNAVAILABLE-2026-10-03`** (open, in this worktree's
   `gaps.jsonl`). DevMap could not answer this lane's navigation; the lane used reads and `rg`,
   reported as such.

## Commands for the next lane (cwd = `build/v6-spancheck-wt`; through `mac_heavy.sh`, one at a time)

```bash
# 1. Rust: tests, clippy, the release binary (the qd-prep that QD_PREP_BIN names).
bash ../../tools/mac_heavy.sh v6-rust cargo test -p qd-prep -j 2
bash ../../tools/mac_heavy.sh v6-clippy cargo clippy -p qd-prep --all-targets -j 2 -- -D warnings
bash ../../tools/mac_heavy.sh v6-release cargo build --release -p qd-prep --bin qd-prep -j 2

# 2. Parity, end to end and Qwen (the ML venv has transformers and tokenizers).
QD_PREP_BIN=$PWD/target/release/qd-prep QD_PREP_DATA_ROOT=/Users/bharath/Code/research/Lappi-decision \
  bash ../../tools/mac_heavy.sh v6-parity uv run --offline --no-project \
  --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \
  python -m pytest -p no:cacheprovider -rfEs python/tests/test_qd_prep_spancheck_parity.py

# 3. The benchmark (report-only).
QD_PREP_BENCH=1 QD_PREP_BIN=$PWD/target/release/qd-prep \
  QD_PREP_DATA_ROOT=/Users/bharath/Code/research/Lappi-decision \
  bash ../../tools/mac_heavy.sh v6-bench uv run --offline --no-project \
  --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \
  python -m pytest -p no:cacheprovider -rfEs -s -k benchmark \
  python/tests/test_qd_prep_spancheck_parity.py
```

The harness blocks `cd` in a direct command. An agent runs each line through a two-line wrapper
that `cd`s to the worktree and then `exec "$@"`; this lane's was
`build/v6-spancheck-msgs/in_wt.sh`, which is gitignored. Never run these with cwd = main's
checkout: hypothesis writes its cache relative to the cwd.

The next lane's first command, if open item 2 is taken up, is the benchmark above (step 3) on
this branch. It is the baseline any encode-once change has to beat.
