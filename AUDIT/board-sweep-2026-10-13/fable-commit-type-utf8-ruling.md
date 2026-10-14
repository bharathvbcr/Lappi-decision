# Fable ruling: `commit_type` and non-UTF-8 diffs (2026-10-14)

Question: the 2026-10-12 ruling (`AUDIT/board-sweep-2026-10-12/fable-board-sweep-ruling.md:335-336` [V]) said the
builder uses the diff bytes unchanged and "invalid UTF-8 is not a drop rule". qd-prep instead drops such diffs as
`context_not_utf8` (`crates/qd-prep/src/commit_type.rs:415-418` [V]), a departure it documents (`:33-35` [V]),
because `decisions::Candidate.context` is a `String` (`decisions.rs:119` [V]) shared by every pool builder and the
pool row JSON carries context as a string; the wire schema already has `context_b64` (`python/qd_data/schema.py:582`
[V]) but the pool row format does not. GitPulse sends exact bytes (`context_b64`; reported, gap
`GAP-GITPULSE-LAPPI-CONTEXT-IS-LOSSY-UTF8-2026-10-06` is `closed` [V]). The built pool dropped 0 of 4,081 commits
on this rule [R, coordinator; not re-counted here].

## Decision: keep the counted drop; do not require the format change for v6

The departure stands as an admission rule, on three conditions. (1) It stays a counted tally row, and every pool
build report cites the count from the tally (a "0" that is assumed rather than read is the failure this repo
names). (2) The pre-registration for any pool that feeds a promoted run records the count, so a nonzero value
is visible before the result is read. (3) The departure text at `commit_type.rs:33-35` stays, and names this
ruling.

## Reason

The ruling's sentence was about the wire: the app must never see a context the model was trained on in a form the
app cannot send. Lossy conversion would break that (it trains a replacement-character string the app never
emits); dropping does not. Dropping leaves a different gap, a diff shape the app can send and the model never
saw, and today that gap is empty: 0 of 4,081 at the supply the pool actually draws from. A byte-carrying row
format is not a field rename: it is `Candidate.context` as bytes across every pool builder, the Python pool
reader and `byte_batch`, plus byte-level equivalents of the secret screen and the unassigned-code-point check,
which run on `&str` today. That is a cross-format change with a measured benefit of zero rows, in a window where
the P1 result and v6 are the work. Rule-wise it is clean: the drop is counted and documented, not silent, so
"approved" does not come to mean "unexamined".

## What triggers revisiting (any one of these reopens it as a format change, not a tolerance)

- A nonzero `context_not_utf8` count in any pool build that feeds a pre-registered run.
- A caller family admitted to training whose repositories carry non-UTF-8 source as a matter of course (vendored
  binaries excluded; the bound is text diffs that are not UTF-8, e.g. Latin-1 or mixed-encoding files).
- Serve-side evidence that callers send non-UTF-8 contexts: a `context_b64` request whose bytes are not UTF-8,
  counted where the schema decodes it. One such request is enough; it is the app sending what training never saw.

When triggered, the change is the format: a byte-carrying pool row (`context_b64` in the row, as on the wire),
`Candidate.context: Vec<u8>`, the reader and `byte_batch` on bytes, and the two `&str` screens re-stated on bytes
with tests that fail first on a non-UTF-8 fixture. Never a lossy conversion, and never an uncounted drop.
