# Proposal: bind the calibration-hash pin to the loaded table on every build path

For the human. Nothing here is implemented: it changes the hash-binding contract, which is
security-adjacent. Gap: `GAP-RT-CALIBRATION-HASH-PIN-BINDS-THE-BACKEND-NOT-THE-LOADED-TABLE`
(open, owner human, opened 2026-10-01).

## What the code does today [V: crates/qd-runtime/src/runtime.rs]

The gap's literal claim is stale for the product path; the residue is narrower. Read at this
worktree's base commit (f7cb935), not run:

- `Runtime::check_hashes` (runtime.rs:305-343) compares `expect.calibration_hash` with
  `self.calibration.hash()`, the table the runtime calibrates with. The 10-01 defect (comparing with
  the backend's *declared* string) was fixed by f75c199 ("G9(c): verify the calibration table's hash
  at load; pin the table that calibrates").
- `Runtime::from_release` runs `Release::check_backend` (release.rs:677-684): the backend's declared
  `calibration_hash` must equal the release table's hash, or the runtime is not built.
- `Runtime::from_ensemble` repeats that per member (runtime.rs:181-190).
- `Runtime::with_backend` (runtime.rs:210-224) does **not**: any backend identity is accepted with
  any table. Its callers are the fixtures and tests, and the runtime's own comment says no product
  path uses it (qd-metal builds through `from_release`). `assemble` has no equality check either.

So a pin cannot currently pass for a table other than the one calibrating on the product path.
What remains: (1) on the `with_backend` seam the identity a caller reads back (`status`,
`backend.calibration_hash`) can name a different table from the one used; (2) the guarantee is
enforced by each constructor separately, not by `assemble`, so a new constructor can omit it.

## Proposed change

In `assemble`, refuse with `BackendError::Unavailable` unless
`backend.identity().calibration_hash == calibration.hash()`. Then drop the duplicate checks in
`from_release`/`from_ensemble` (or keep them for their better messages and let `assemble` be the
backstop). Fix the fixtures' backend (`ReferenceBackend` already declares
`CalibrationTable::reference().hash()`, and `fixtures.rs:139` pairs it with the reference table, so
the fixture agrees) and find, with the graph, any test that builds `with_backend` with a fitted
table over the reference backend: such a test passes today only because nothing checks. None was
looked for beyond the fixture; unverified.

## Security-impact statement

- **Direction: narrows access.** Fewer runtimes can be built; no request that is refused today
  becomes admitted. No auth, quota, crypto or secret handling is touched.
- **What it protects.** The `expect.calibration_hash` pin and the identity echoed by `status` both
  mean "the table that produced this score". A path where they can disagree lets a caller believe
  it pinned the serving table when it pinned a label. That is the "approved means unexamined" shape.
- **Contract change.** `Runtime::with_backend` becomes fallible on identity mismatch (it already
  returns `Result`). Out-of-tree callers that pair a backend with a different table would stop
  building. In this repo that is test code only (to be enumerated with `devmap_impact` once an index
  exists for the worktree; none did on 2026-10-17, so the caller list above is from `rg`, not the
  graph).
- **Risk of the fix.** A test that deliberately mixes tables to exercise `hash_mismatch` would need
  a dedicated seam instead of the general one. No new dependency.
- **Needs the human's yes** because it changes what a pinned hash means at the one seam where it is
  currently unenforced, and because `runtime.rs`/`fixtures.rs` are outside this lane's files.

## Why this lane did not do the neighbouring check

`qd-release-check` (check c) compares the replay's `status.backend.calibration_hash` with the
release's `expected_identity.calibration_hash`. That value is the backend's declared hash, which is
the table's hash on every product path today only because `from_release` enforces it. If this
proposal is not taken, check (c) is exactly as strong as that constructor, and no stronger.
