# Mac inference path audit: findings in files another lane holds uncommitted (2026-10-03)

The bench session (Lappi model benchmarking, f88b2d) ran a read-only DevMap audit of the Mac
inference path at the human's request. These findings sit in files that are **dirty and
uncommitted in main's checkout, held by another lane**: `crates/qd-metal/src/tokenizer.rs` and
`crates/qd-metal/src/backend.rs`. Neither the bench nor the lead edits those files. This note is
for their owner.

Navigation: the audit's file:line claims came from rg and Read, not from the graph (see
GAP-DEVMAP-MAC-INFERENCE-AUDIT-CALLERS-DEAD-SYMBOLS-2026-10-03).

Status labels:
- **verified (read)**: the lead read the code and the structure is as stated. No run, no
  measurement.
- **relayed**: the bench's claim; the lead has not checked it.

## HIGH

### H1: `encode_untrusted` is quadratic in the number of tag occurrences

Location: `tokenizer.rs:107-127, 141-152` (dirty file). Status: **verified (read)** for the loop
structure; the cost figures are **relayed**.

How it goes quadratic:
- `encode_text_without_added_tags` loops `while let Some(..) = next_added_tag(rest, &tags)`.
- `next_added_tag` runs `rest.find(tag)` for every added tag over the whole remainder, on every
  iteration.
- A tag that never occurs costs a full scan of the remainder each time. The bench says 20 of the
  22 `<|…|>` tags never occur, because the renderer escapes them.

Cost the bench gives:
- A 131,072-byte context packed with `<tool_call>` means ~12k iterations × 20 scans × up to 128 KB.
- This runs 3× per request on the single GPU worker: backend.rs:474 prefill, and :625 where each
  decode re-tokenizes prefix+suffix.
- The caller times out at 30 s, but the worker is not cancelled, so every queued request stalls.

Suggested fix: cache each tag's next match index, or use one multi-pattern pass, and cap
occurrences. Ship it with a test that fails on the current code: a bounded-time encode of a
packed 128 KB input.

## MEDIUM (relayed)

### M1: the deadline is checked only between backend calls

Location: `backend.rs` (dirty) and `answer.rs:590-597`.

`submit` waits the full `job_timeout` of 30 s, so a "30 s" request can take ~60 s.

Fix: validate `job_timeout <= request_timeout`, or pass the remaining deadline down.

### M2: a live request's snapshot can be evicted

Location: `backend.rs:429-441, 555-567` (dirty).

At the defaults, `max_entries` 8 equals `max_in_flight` 8. But each request inserts a prefill
**and** a snapshot, and `snapshot()` does not bump the prefill's `last_used`. A live request's
snapshot can then be evicted, which gives DecodeFailed "no longer held". It fails closed, but it
costs availability.

Fix: require `max_entries >= 2 × max_in_flight`, or pin the entries in use.

## LOW, in dirty `backend.rs` (relayed)

- `:362`: the label-set hash hard-codes `"ABCDEFGHIJKLMNOPZ"`.
- `:181`: `ready_rx.recv()` has no timeout.
- `:226-237`: Drop drains and joins without a bound.
- `:335`: the worker uses `GpuRuntime::new()`, not `new_inference()`.

## Coming behaviour change for `backend.rs`'s owner

This is at Fable's request (ruling 2, 2026-10-03).

When branch `qdm-digest-parallel` (item 1a) merges, `PrefixState::digest()` in
`crates/qd-metal/src/model.rs` becomes the **parallel** digest, with no line of `backend.rs`
changing.

What stays the same:
- The bytes are identical. The v1 pins match, and the A/B rows 147da0cc and 28505f4c are
  bit-identical 14/14.

What changes:
- **Threads:** each digest now copies the state into a thread-local host scratch and hashes the
  buffers on up to 8 threads (`qd_runtime::SHA256_PARALLEL_MAX_WORKERS`).
- **Scratch memory:**
  - A scratch up to 256 MiB stays resident on the calling thread, the GPU worker.
  - At 16K tokens (~423 MB) the scratch is freed after every digest and allocated again on the
    next one.
  - Ruling 2 #1, a hardening item, follows on the same branch.
- **Speed:** min-of-7 `digest_ms` is 5.3–6.1× faster at T = 131 to 8,192.

If `backend.rs` calls `digest()` from a context where extra threads or the scratch matter, read
this before merging.

## Measured for `backend.rs`'s and `tokenizer.rs`'s owner (Fable ruling 2, step 6)

The rows are on branch `qdm-digest-parallel` in `ledger/mac-qd-metal-eb-2026-10-03.jsonl`
(commit 9fac7be). All are `quick`, all record the clean commit 75c683b, and the reading for each
was declared before it ran.

### Item E: `GpuRuntime::new` vs `new_inference`

Rows: 94736e9b, a2ed6bc8, 552e6073, 91832401, run in ABBA order on the release weights.

Result: there is no reason to switch the worker (`backend.rs:328`) to `new_inference`.

| T | Δ min total_ms (inference − timestamps) | noise floor |
|---|---|---|
| 131 | +0.61 ms | 1.64 |
| 409 | +0.68 ms | 0.89 |
| 770 | +1.07 ms | 0.53 |

- At T = 770 the slowdown is beyond the noise floor.
- Dispatches (920) and commits (5) per decision are the same under both runtimes.
- The ruling's estimate of a 0–5 ms gain did not show up.

### Item B: product path (`MetalBackend`) vs the `Model`-driven path

Row 02156dc5, on the base snapshot. Only the in-process delta is the reading.

Equal work was checked: the product logits are bit-identical to the `Model` path's 5/5 at every
T, and every timed prefill missed the cache.

| T | Product path adds | Drift between the two `Model` phases |
|---|---|---|
| 131 | +1.86 ms | 0.39 |
| 409 | +1.13 ms | 1.56 |
| 8,192 | +12.20 ms | 3.03 |

- At T = 409 the gap is within the drift.
- At 8K, host tokenization alone is 10.64 ms of the 12.20 ms. That is
  `QwenTokenizer::encode` of the prefix once, then of prefix + suffix again for each decode.
- **Suggestion (the bench's, not verified):** encode only the suffix per decode and check the
  token boundary. That could remove about 2/3 of those 10.64 ms at 8K.
- **Caveat:** this measured the COMMITTED `backend.rs` and `tokenizer.rs`, not the dirty versions
  in main. H1's quadratic tag scan in the dirty `encode_untrusted` is not measured, and would add
  to this at long T.

## In clean files (the bench may fix these on its branch)

These are separate commits after 1a, each with a fail-first test.

- **M4**, `crates/qd-metal/src/bin/serve.rs:130,134`:
  - 0 is silently coerced to 1;
  - a bad `--queue-capacity`, `--max-entries` or `--max-tokens 0` is refused only on the first
    request, after bind.

  Fix: validate MetalBounds before `Server::bind`.
- **Tri-state**, `answer.rs:554-576`: `state_host_visible=true` with **empty** state bytes would
  report Ran, because sha256("") equals itself. Fix: refuse empty or unchanged-length state bytes
  when the backend claims host visibility.
