# W2a: the five Vani gates before any real row pool (2026-10-21)

Task ft-951c360d848535417fd3fce9a4a16f78. Branch `lane/ft-951c360d`, from d09222d. Not merged, not pushed.

## Security-impact statement

Every change narrows access; none widens it.

- A `vani-agreed` row needs two agreeing, distinct teachers; is refused near a held-out posting
  (a missing, corrupt, wrong-version or empty signature file refuses the whole pool); is redacted,
  and is refused if an email, phone number, signature-block name or credential-shaped string
  remains or if redaction leaves under 8 words.
- A public adapter refuses BY-SA, research-only and `vani-agreed` rows. An absent
  `adapter_visibility` is read as public (the stricter reading: a forgotten field can cause a
  refusal, never a leak). An unrecognised value is refused.
- A run not stated to be on the Mac refuses `vani-agreed` rows (an unstated target is read as cloud).
- `cc-by-sa-3.0` stays in the global ALLOW tier, so v1 pools are unchanged; the scope is a second
  gate (`admit_licence_for`).
- The training process never reads the signatures: they live under `data/heldout/`, and
  `held_out.rs` has a test that its path door refuses that path in four spellings.
- Authorization, rate limits, billing, crypto and secret handling are untouched.

## What changed

| Gate | Where |
| --- | --- |
| 1 Agreement filter | `crates/qd-prep/src/vani_pool.rs` (`agree`, `admit`, `run`), CLI `qd-prep vani-pool` |
| 2 Content door | `crates/qd-prep/src/posting_door.rs`, `Family::from_seed` in `minhash.rs`, CLI `qd-prep posting-door-build`; pin in `crates/qd-train/src/held_out.rs` |
| 3 Redactor | `crates/qd-prep/src/redact.rs`; secrets go through `qd_runtime::caller_record::secret_like_prefix`, the repo's one secret screen |
| 4 Visibility | `python/qd_data/licences.py` (`AdapterVisibility`, `admit_licence_for`), `crates/qd-preflight/src/pool_policy.rs`, `adapter_visibility` line in `qd-release-check check` |
| 5 Mac only | `mac_only` tier of `qd-train-data` (`--run-target mac|cloud`, and the manifest's own `run_target`) |

Gate 5's earliest door that this lane owns is `qd-train-data` (preflight-train). The pool builders
and the trainer are in files other lanes own (`python/qd_data/manifest.py`, `Makefile`, the frozen
`qd-train` modules); see the gaps.

### Decisions to check

- "Agree" is the same chosen label (trimmed, exact), from two different teacher ids (trimmed,
  ASCII case-insensitive), exactly two readings. No margin (ruling 9).
- The v5 rule the door applies is the decontamination rule: lower-cased `\w+` word 8-grams,
  containment of the held-out posting's n-grams at 0.5 (`qd_train.replay`), plus v5's dedupe
  Jaccard 0.8 over 128 permutations (`qd_data.config`). Containment is estimated from the
  signature as `J (m + k) / (1 + J) / k`. Thresholds are constants guarded by a test that reads
  the Python files.
- Existing releases: `qd-release-check` now fails a release whose train manifest does not exist,
  does not hash to the recorded sha256, or holds BY-SA rows and is not marked `private`. v5's pool
  holds `rajpurkar/squad_v2` (`cc-by-sa-4.0`), so a v5 release is refused until its manifest says
  `private`. Fixtures in `crates/qd-preflight/tests/release_check.rs` and
  `python/tests/test_release_replay_flow.py` now record a private train manifest; their intent is
  unchanged.

## Measured

See the final report for command, exit and counts; the logs are in `build/w2a/` (gitignored):
`py-before.txt`, `vani-before.txt`, `pp-before2.txt`, `prep-after3.txt`, `preflight-after.txt`,
`train-after.txt`, `py-after2.txt`, `bench-before.txt`.

## Open (gaps)

- GAP-W2A-INSIGHTS-AND-LISTAGENTS-UNKNOWN-2026-10-21
- GAP-W2A-VISIBILITY-AND-RUN-TARGET-WRITERS-NOT-WIRED-2026-10-21
- GAP-W2A-DOOR-THRESHOLDS-ARE-CONSTANTS-WITH-PARITY-TEST-2026-10-21

## First command for the next lane

```
bash tools/mac_heavy.sh w2a-verify cargo test -p qd-prep --test vani_gates
```

then `qd-prep posting-door-build --postings P --out data/heldout/vani-postings/signatures.json`
once Vani exports the owner-labelled postings.
