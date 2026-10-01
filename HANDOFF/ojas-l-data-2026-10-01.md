# HANDOFF: ojas L-data lane (qd-train shard reader behind the rule-3 door), 2026-10-01

Branch `worktree-agent-a20a87b9b547b3e3b`, fast-forwarded from main `6a06bd3`. Plan:
`AUDIT/ojas-training-2026-10-01/fable-advice.md`, Q1 rows 13-14 and Q5 (lane L-data).
No ssh, no box path, no GPU, no edits to ojas, tessl, or the foreign `qd-metal` /
`qd-preflight` / `qd-runtime` work. `Cargo.lock` is modified in the worktree and not committed.

## What was measured

This lane wrote **no ledger row**: nothing was trained. Every number below is a test outcome
or cites ledger row `d96409bd-4890-4d7f-9e7c-ca4cafbdb9a8` (`ledger/mac-phase4-v4-shards-2026-10-01.jsonl`).

- **Fail-first door evidence.** I temporarily replaced the door with a stub that permits
  everything: path layers, manifest split/family/status, shard split, `report_only` and
  manifest binding. Then I ran `cargo test -p qd-train --test door`. The result was
  `FAILED. 9 passed; 11 failed`, and every door test failed. Each held-out case read the
  whole `tokens.u32`: `410861 bytes were read (327496 of them token bytes) from a shard set
  under a held_out segment` for the marker case and `410861 bytes read through
  …/quarantine/train` for the root case. The manifest-split and family cases each read 351,248
  shard bytes, and the heldout-split header read 327,496 token bytes. The 9 tests that still
  passed are provenance and format refusals, not door tests. Both sources were restored
  byte-identical (sha256 checked).
- **Parity with Python.** Each item below is a test in `tests/parity.rs` against the oracle
  fixture, and all pass:
  - numpy `SeedSequence` pool and `generate_state`, `PCG64` raw draws, and `permutation(16)`.
  - Batch order and the consumed digest for **8 configs**: seeds 20260919 and 7, epochs 0
    and 1, batch_tokens 3192 and 6385. Membership, bucket, width, lengths, per-batch
    `_fold` digest and the running digest all match, over 32 and 19 batches.
  - The reader: shard_hash, each of the 98 sequences' token sha256, candidates, kinds,
    targets, spans, padding and recorded check states.
  - `ft_supervision` and `plan_span_batch` on all 51 batches of two configs. Six of those
    batches carry span rows, including abstaining ones.
  - Python `json.dumps`, the `DataConfig()` defaults, and the contract constants.
  - `RESERVED_NOUL_ROWS`, pinned against `crates/qd-runtime/src/{schema,answer}.rs`.
- **The parity suite is not vacuous.** I applied five single-point mutations: the order tag,
  the PCG multiplier, the fold byte order, span rows in the letter mask, and the abstain row
  first. The suite caught all five, and I restored the sources byte-identical after each run.
- Raw output of both runs: `AUDIT/ojas-training-2026-10-01/l-data-stub-door-run.txt` (the
  stub-door `cargo test` output) and `AUDIT/ojas-training-2026-10-01/l-data-parity-mutations.txt`.
- **Real v4.** `tests/v4.rs` is opt-in. It opens `shards/{train,val}` beside
  `data/pool/{train,val}.json` with `expect_rev` set to the row's `recipe.rev` and with no
  `allow_stale_code`. Every door, rev, code-fingerprint and remap check passes. It matches
  the row's counts, read from the row itself:
  - 363,950 sequences and total_tokens 306,926,895.
  - max_seq_len 7936 and its 32 buckets.
  - Slot coverage 363,950 of 364,006 (56 excluded), over 289,142 rows.
  - Padding 16,146,815 of 323,073,710 positions.
  - Val: 18,223 of 18,428 sequences over 16,124 rows.

  On the real epoch-0 plan (batch_tokens 15872, data seed), Rust and Python agree on all
  22,713 batches' membership (sha256 `20026a67…`). They also agree on the consumed digest
  over the first 8 batches (`26c94be4…`); the Python values come from
  `tools/qd_train_oracle_shards.py --v4-out`. Release build, 1.8 s, CPU only.
- **Python tests.** I ran test_gaps_ledger, test_gaps_writer, test_data_access, test_shards
  and test_heads: 202 passed, 1 failed. The failure is
  `test_gaps_writer.py::test_the_real_ledger_is_not_touched_by_any_of_this`, which asserts
  the repository directory is named `qwen-decision` or `Lappi-decision`. In any linked
  worktree it is `agent-…`, so this failure is environmental; it is already on record in
  `gaps.jsonl`.
  `ruff check tools/qd_train_oracle_shards.py` is clean.

## What changed

- `3400063`: the door (`held_out.rs`: four layers, manifest hash re-derivation, status
  refusals) and `ShardReader::open`, the reader's only constructor. It checks the shard
  path first, then the full door on the set's own manifest, then the header and
  `assert_shard_trainable`. It also adds `pyjson.rs`, `tristate.rs`, `files.rs` (the
  counting seam), `npy.rs`, `npz.rs` (stored zip; deflate refused by name), `np_random.rs`,
  `shards.rs` (every `ShardReader.__init__` refusal, `_plan`, `assemble_batch` with the
  `Batch` refusals, and `ConsumedPrefix`) and `supervision.rs`. The same commit adds the
  oracle `tools/qd_train_oracle_shards.py`, the fixture
  `crates/qd-train/tests/fixtures/shards-tiny/` (1.0 MB, deterministic apart from the
  headers' `created_at`) and `tests/door.rs` (20 tests).
- `8b84ce9`: `tests/parity.rs` (10 tests).
- `b27d59d`: `tests/v4.rs` (2 opt-in tests), the oracle's `--v4-out` mode, and ruff fixes.
- This handoff and two gap records.

The reader is deliberately stricter than Python's `ShardReader` in these ways:
- The source manifest is required and must pass the full door, and the header must pin its
  hash and split. Python never applies layers 3 and 4 to a shard set.
- `report_only` is refused outright.
- A header without `shard_hash` is refused.
- Every sequence's `row_id` must be in the admitted manifest.
- Slot kinds are checked at open.
- Token ids must be below `vocab_size`.
- The path layers run before `header.json` is opened.
- The code fingerprint is taken from an explicit, required `qd_data_dir`.

## What is open

- `GAP-L-DATA-REAL-FT-SOURCE-TRANSFORMS-NOT-PORTED-2026-10-01`. `real_ft_run`'s `source()`
  does several things this lane did not port: it re-indexes across `--passes`, filters by
  `--max-width`, can apply `ChoicePermutation`, and can interleave replay batches. Which of
  these F's recipe sets was not read.
- `GAP-L-DATA-CASEFOLD-TABLE-RECALLED-2026-10-01`. The non-ASCII case-fold list for path
  markers was recalled, not read from Unicode's case-folding table.
- `GAP-L-DATA-GITPULSE-UNTRUSTED-2026-10-01`, `GAP-L-DATA-NO-LISTAGENTS-2026-10-01` and
  `GAP-L-DATA-DEVMAP-CONSUMED-DIGEST-FIELD-2026-10-01`.
- Not ported: `DataConfig`'s source-registry and admission validation (it needs
  `qd_data.sources`). Instead, the oracle checks the Rust defaults against Python's
  `DataConfig()`. `ShardReader.to_json()`'s ledger shape is not ported either; that is the
  trainer lane's to own.

## First command for the next lane (L-trainer)

```text
cargo test -p qd-train && QD_TRAIN_V4_OUT=/Users/bharath/qd-campaign/phase4-v4-2026-10-01 \
    cargo test -p qd-train --release --test v4 -- --ignored --nocapture
```
