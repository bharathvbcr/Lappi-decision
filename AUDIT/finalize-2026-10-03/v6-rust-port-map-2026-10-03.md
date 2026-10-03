# v6 Rust port map: shard writing and scoring (2026-10-03, ~22:46Z)

This is a read-only mapping by an Explore subagent, run while the v5 rebuild wrote shards. It answers the
standing goal: "minimize python bottlenecks with rust". It does not change v5. v5 launches as
pre-registered, with no-mask and fused AdamW off (human answer `human-answers-2026-10-03-optimize.md`).

**Navigation.** DevMap answered "database disk image is malformed" for the agent. Every `file:line`
below came from `rg` and reading in `build/v5-build-wt` at 8e6a009, so none of it is graph-confirmed.
The gap is GAP-V6-PORT-MAP-NAVIGATION-DEVMAP-MALFORMED-2026-10-03.

Labels: **V** verified (read or measured), **I** inferred, **U** unverified.

## 1. Stage 6 shard writing (~29 min on one core in the first v5 build)

**Where the time goes**

- **[I]** About 50% of the time is Python (range 40-60%) and the rest is the Rust inside HF `tokenizers`.
  Both run on one core.
- **[V]** The basis is `AUDIT/perf-pipeline-shards-2026-09-30.md`.
  - Rust tottime was 134.4 s out of 271.2 s for census plus write_shards: `encode_batch` 125.9 s and
    `decode` 8.5 s.
  - About 77% of `decode` was the transformers Python wrapper: `to_py_obj` was called 65.6M times.
  - Caveats: that profile used an older `shards.py`, a different corpus, and cProfile, which inflates
    the Python side.

**The per-row loop**, `python/qd_train/shards.py:1625-1728` **[V]**

- Per span slot, it does pure-Python work:
  - `list(token_offsets())` at line 1174 and a reach generator at line 1186;
  - an NFC-check decode at lines 1200-1203;
  - `_token_indices_for_chars` at line 388, which is called twice (lines 1212 and 1241).
- `_assert_spans_decode_to_their_text` (line 1007) also runs twice per slot. Each run:
  - decodes once per candidate;
  - does a full decode;
  - runs `line_start_indices`, a per-character loop at `artifacts.py:1087`.

**The double encode** **[V]**

- `--memo-limit 0` is set at `campaign/v5-preregistered.DRAFT.json:258`.
- So `tokenize` and `offsets` each re-encode (`tools/real_tokenizer_pipeline.py:602-619`). Every span
  slot is encoded twice.

**A free measurement on the running build** **[V]** that the code exists; **[U]** what it will show

- Stage 8 (`real_tokenizer_pipeline.py:3138-3161`) rewrites the same corpus without `decode=`.
- Stage 6 wall time minus stage 8 wall time is the decode check's cost on the real v5 corpus.

**Proposed port: `qd-prep spancheck`** (std only, threads via `std::thread::scope`)

- It follows the existing `native_minhash` seam (`real_tokenizer_pipeline.py:1913-2312`).
- The request is batch-shaped and chunked. Chunking is required because 386M tokens at about 12 B each
  is roughly 4.6 GB.
- The pipeline encodes once per text, which also removes the double encode.
- For each sequence, Rust returns the status, the `(start_tok, end_tok)` span and the candidates.
- The result is installed as a lookup table in place of `_span_token_positions`, so `shards.py` stays
  the unedited oracle.
- Any slot that is not ok is re-run by the Python reference, so exception text is identical by
  construction.
- Canaries re-run N sequences from each end of every call.
- The line rule must match `crates/qd-runtime/src/context.rs:172/183` exactly.
- Tests and benchmark:
  - parity tests in a new `test_qd_prep_spancheck_parity.py`;
  - an interleaved A/B, min-of-N benchmark behind `QD_PREP_BENCH`;
  - sha256 equality of the full artifact set, ignoring timestamp fields.

**Dependencies and what stays the owner's call**

- `spancheck` needs **no new dependency**.
- Moving the encode or decode itself into Rust needs either:
  - the `tokenizers` crate (**a new dependency, owner's approval**; it may link onig C, against
    qd-prep's no-C rule **[U]**), or
  - a hand-written ByteLevel decoder whose U+FFFD behaviour matches **[U]**.
- `render` is in `qd_data`, so speeding it up changes the shard fingerprint. That is the owner's call.

**Bit-exactness**

- A false refusal in Rust drops a slot. That changes `sequence_index`, `coverage.json`,
  `n_sequences` and `shard_hash`.
- A false pass accepts a slot Python would have aborted on.
- `code_fingerprint` covers only `python/qd_data/*.py` (`fingerprint.py:81-97`). So the binary's
  sha256 must go into the metrics, because the header will not show it.

## 2. Scoring (the per-seed `epoch-score-val` and the needle control)

**What the ledger shows** **[V]** (`ledger/gh200-p4-v4-2026-10-01.jsonl`)

| What | Time |
|---|---|
| `epoch-score-val` per seed | 578-601 s |
| Needle control per seed | 187 s |
| Gap from each seed's ft row to its score row | about 4,050 s |
| Of that gap, not covered by any ledger timer | about 3,470 s per seed |

- The "0.82 h / 16% of a seed" figure used in this session was **not found** in any file.
- `HANDOFF/gh200-phase4-2026-10-01.md:838` says about 1.2 h of scoring per seed.

**Host-side Python**

- `int(argmax)`, `float(softmax[...])` and `.double().tolist()` in `_decode`
  (`real_ft_run.py:1540-1620`) each force a GPU sync.
- `_readout` does a per-letter-row host-to-device copy, index and clone (lines 1317-1319).
- **[I]** At about 36k rows times a few syncs each, this is low single-digit percent of the timed
  578 s.

**Recommendation.** Do not port scoring to Rust. Measure first:

- the box log's `checkpoint: ... in Ys` line (`real_ft_run.py:1919`);
- a timer around `_record_score`;
- a py-spy or torch-profiler capture of seed 0.

That matches the plan to measure seed 0's phases live for v6. Later, a pure-torch batching change is
possible:

- one gather at `target_index`;
- a device-side masked softmax and argmax;
- one device-to-host copy per batch.

It keeps the full-width `lm_head` GEMM for rounding parity, uses the current `_decode` as the oracle,
and compares the verdict JSONL byte for byte. No dependency is needed.

## What is open

- **Stage 6 minus stage 8** on the v5 rebuild (`build/v5-build/build-r.log`). Read it when the build
  finishes.
- **Seed 0's phase timings** on the H100 lanes, the scoring gap first.
- **`qd-prep spancheck`** as a v6 lane. It needs no dependency and no human approval, but needs a
  pre-registered parity bar.
- **A Rust `tokenizers` encode or decode.** It needs the owner's approval for the dependency.
