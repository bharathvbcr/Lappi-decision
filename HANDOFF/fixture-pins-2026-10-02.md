# HANDOFF: L-fixture-pins, "allow pinned fixtures only", 2026-10-02

The human's decision (2026-10-02): **"Allow pinned fixtures only."** Binary files are allowed
only under `crates/*/tests/fixtures/**` with a fixture extension (`.npy .npz .safetensors .u32`),
and each must be sha256-pinned in a tracked manifest. Every other tracked file stays NUL-free.
This lane implemented that rule in the gate. It pinned the 270 binaries that had no pin, and
made each fixture set's consumers verify their pins before use.

Branch `worktree-agent-ad15ff6b3ae75f557`, based on main `6305a70`.

## What was measured

No ledger rows: this lane ran no training or eval. Every number below comes from a command output
quoted here, run in this lane's worktree.

### Before (unmodified `6305a70`)

- `cargo test --workspace --no-fail-fast`: 814 passed, **1 failed**, 17 ignored. The one failure
  was `qd-runtime --test tracked_source_is_text`, `no_tracked_source_file_contains_a_nul_byte`:
  "scanned 1243 of 1243 tracked files (0 absent from the worktree, 0 over the 8388608-byte scan
  cap)", with 281 offenders, all under `crates/qd-train/tests/fixtures/`.
- Pin coverage (throwaway census: each binary's sha256 searched in tracked text under the same
  fixture root): 11 pinned and 270 unpinned. The new gate's exact-string matching gives the same
  11 (`281 contain a NUL byte: 11 binary-allowed (pinned fixtures), 270 offenders`), so
  substring and exact matching agree. All 270 failed on the pin alone, with no directory or
  extension failure.

### Binary fixtures per set (from `fixture_pins.rs`, `every_binary_fixture_is_in_a_set_a_consumer_verifies`)

| set | binaries | files the manifest verifies | manifest | pinned before this lane |
| --- | ---: | ---: | --- | --- |
| `span-head/` (9 cases) | 261 `.npy` | 270 (261 `.npy` + 9 `manifest.json`) | `SHA256SUMS` (new) | no (combined digest `3d05a73f…` only) |
| `shards-tiny/` | 9 (`door/` 1, `shards/train` 4, `shards/val-report-only` 4) | 38 (9 binary + 29 text) | `SHA256SUMS` (new) | no (shard content hashes only) |
| `tiny-published/` | 8 | 13 | `manifest.json` `files` | in the manifest, **checked by no Rust test** |
| `adamw-decay-sensitive/` | 2 | 4 | `manifest.json` `files` | yes, checked by `adamw_decay_sensitive.rs` |
| `span-head-init-seed0-h64.safetensors` | 1 | 1 | its `.manifest.json` `file` | yes, checked by `head_init.rs` via `load_span_head_pinned` |
| **total** | **281** | | | |

A correction to the brief. It said tiny-published "pins", which is true of its manifest. But
before this lane no Rust test read `manifest.json` `files`: `qd-train-metal/tests/rung_b_data.rs`
reads only `coverage`, and `gpu_rung_b.rs` reads `recipe` and `arms`. It is checked now, in
`head_init.rs` before use and in `fixture_pins.rs`.

The `SHA256SUMS` sets list every tracked file in the set, text included. A hand edit to, say,
`shards-tiny/door/train-tampered.json` therefore has to be re-pinned.

### After (`77aed64`)

- `cargo test --workspace --no-fail-fast`: **838 passed, 0 failed, 17 ignored**. This reconciles
  with before: +11 gate unit tests, +1 for the gate main test now passing, +12 `fixture_pins`.
- The gate: "scanned 1245 of 1245 tracked files (0 absent from the worktree, 0 over the
  8388608-byte scan cap)", "281 contain a NUL byte: 281 binary-allowed (pinned fixtures), 0
  offenders", "manifests read under fixture roots: 2 SHA256SUMS, 54 JSON, 0 unreadable; 1496
  pins". 12 of 12 tests pass.
- `cargo clippy -p qd-runtime -p qd-train --all-targets -- -D warnings`: clean.
- Independent check of the two manifests with Perl's `shasum -a 256 -c`: 270/270 OK and 38/38 OK.

### Fail-first evidence

The gate (old rule fails on `6305a70`, quoted above). The new gate, run before the two manifests
existed: 11 allowed and 270 offenders, each `fails: its sha256 is in no manifest under its own
fixture root`. Live probes after the fix, each a temporarily staged file, then `git rm --cached`
and deleted (`git status` clean afterwards):

| probe | gate result |
| --- | --- |
| (a0) exact copy of a pinned `.npy` into `span-head/one-candidate/` | **passes** (282 allowed). Its bytes are on record, which is the user's rule. qd-train's `fixture_pins` **fails** on it: `one-candidate/zz-probe.npy: under the set but not in SHA256SUMS` |
| (a) the same copy with one byte appended | **fails**: `fails: its sha256 is in no manifest under its own fixture root` |
| (b) a pinned `.npy` copied to `crates/qd-train/tests/zz-probe.npy` | **fails**: `not under crates/<crate>/tests/fixtures/`, and the pin is named as being under another root |
| (c) a binary `zz-binary-probe.rs` at the repo root (outside every crate, so cargo does not compile it) | **fails** on all three: directory, extension `.rs`, pin |
| (d) a malformed `crates/qd-train/tests/fixtures/zz-probe/SHA256SUMS` (one space) | **fails** as unreadable, before any offender |

The consumer check. Without the two manifests, qd-train fails in door 20, parity 7, ft_data 4,
span_head 11 and fixture_pins 7, all with `cannot read SHA256SUMS`. With them it is green. The
tamper tests in `fixture_pins.rs` run on scratch copies and never touch a tracked fixture: a byte
appended (SHA256SUMS and JSON forms), an unlisted file, a missing file, a missing manifest, a
symlink, the before-use panic, and grammar refusals.

## What changed

| commit | what |
| --- | --- |
| `2daba27` | `span-head/SHA256SUMS`, `shards-tiny/SHA256SUMS`; `crates/qd-train/tests/common/pins.rs` (new: the consumer-side check); `tests/fixture_pins.rs` (new, 12 tests); before-use checks in `common::fixture()`, `span_head.rs` `fixture_root()`, `head_init.rs` `tiny_published()`, `adamw_decay_sensitive.rs` `pinned_manifest()` (its private `sha256_hex` and pin loop converged onto `pins`); `span_head.rs`'s directory contract expects exactly `CASES` plus `SHA256SUMS` |
| `77aed64` | `crates/qd-runtime/tests/tracked_source_is_text.rs`: the rule, the pure `classify_binary` with 11 unit tests, and module docs recording the decision with its date, why it is a pin and not a skip, the accepted manifest forms, and the re-pin command |
| this commit | this file, and `gaps.jsonl` |

No fixture byte changed. `git status` shows no modified file under `tests/fixtures/`, and
`shasum -c` passes. `Cargo.lock` was touched by the build and is not committed. No dependency was
added: `sha2` and `serde_json` were already dependencies of both crates.

**How the manifests were generated, and why.** They were written with a documented one-line
`shasum -a 256` command, not a Rust helper. The digests then come from an implementation
(Perl's `Digest::SHA`) independent of the Rust that checks them, and no committed code can bless
whatever bytes are on disk. From the set's directory, after any regenerated files are tracked:

```text
git ls-files -z -- . ':!:SHA256SUMS' | xargs -0 shasum -a 256 > SHA256SUMS
shasum -a 256 -c SHA256SUMS
```

This lane ran the equivalent from the worktree root, stripping the set prefix with `sed`, because
the harness blocks `cd`. The output is the same bytes: same `git ls-files` order, paths relative
to the set.

**Two parsers, one grammar.** The gate (qd-runtime) asks "is this digest recorded under this
root?". `pins.rs` (qd-train) asks "does this path hash to its digest?". Sharing one parser would
need a cross-crate `#[path]` include, which this repo has never used. Instead the grammar is
defined once, in the gate's module docs, and `pins.rs` cites it. Both are strict, and a
disagreement fails one side's test rather than passing.

**Regenerating a fixture set now needs a re-pin.**
`tools/qd_train_oracle_span_head.py` refuses an `--out` holding anything but its cases, so delete
`span-head/SHA256SUMS` before regenerating and re-pin afterwards. Its message, "Remove them by
hand", now covers the pin as well. `tools/qd_train_oracle_shards.py --replace` deletes
`shards-tiny/` along with its pin. Either way the gate and the consumers fail loudly until the
re-pin. Neither oracle was edited, since no Python test covers them.

## What is open

- `GAP-QD-RUNTIME-NUL-BYTE-TEST-FAILS-ON-QD-TRAIN-BINARY-FIXTURES-2026-10-02`: **resolved** by
  this lane (appended record cites `77aed64`, `2daba27`).
- `GAP-L-FIXTURE-PINS-GITPULSE-TRUST-REQUIRED-2026-10-02`: GitPulse returned
  `REPOSITORY_TRUST_REQUIRED` on every facet. The cross-worktree check was done by hand with
  `shasum`, and the result is inferred, not GitPulse-confirmed. The id has two lines; the second
  corrects a worktree count (39, not 42).
- `GAP-L-FIXTURE-PINS-LISTAGENTS-UNAVAILABLE-2026-10-02`: there is no `ListAgents` tool in this
  session.
- `GAP-L-FIXTURE-PINS-DEVMAP-NO-WORKTREE-INDEX-2026-10-02`: there is no DevMap store in the
  worktree, so DevMap was answered from the main checkout. Fixture consumers were found with `rg`
  on path strings, not graph-confirmed.
- **Not done, by scope:** `crates/qd-train-metal/tests/tiny_published/mod.rs` reads tiny-published
  without a pin check of its own. It is another crate and cannot reach qd-train's
  `tests/common`. A change to tiny-published still fails `cargo test --workspace` through
  qd-train (`head_init.rs`, `fixture_pins.rs`). If qd-train-metal's tests are ever run alone, a
  `#[path]` include of `pins.rs` or a dev-dependency would be needed, and that is a human call.
- **Not done, by scope:** the gate allows an exact byte copy of a pinned fixture anywhere under the
  same fixture root (probe a0), because its bytes are on record. That is the user's rule as
  stated. Inside a `SHA256SUMS` or JSON-`files` set, the consumer's completeness check refuses
  it. A copy placed elsewhere under the root, outside every set, would also fail
  `fixture_pins.rs`'s `every_binary_fixture_is_in_a_set_a_consumer_verifies`.

## First command for the next lane

```text
cargo test -p qd-runtime --test tracked_source_is_text -- --nocapture && cargo test -p qd-train --test fixture_pins -- --nocapture
```
