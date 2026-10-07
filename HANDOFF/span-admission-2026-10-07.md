# HANDOFF span-admission, 2026-10-07

Task: GitPulse `ft-5af5e8ee45ef06d778887bfe37d85c94`, "Make code.defect_class servable (span
calibration, span head, admission vs composed diff shape)".

**State: written and verified on this Mac, and committed on `main` with this file** (the human
said "Commit it, leave gaps.jsonl out"). Not pushed. The three gap records below were appended to
`gaps.jsonl` but are **not** in the commit: that file also held other lanes' uncommitted records.
The Mac panicked on 2026-10-07 (GAP-MAC-WATCHDOG-PANIC-UNDER-A-LOCKED-JOB-2026-10-07). At
~16:17Z the human said "Resume, clear the stale lock and run it". The stale ojas hold
(pid 88638, gone) was removed. No ledger row: nothing was scored.

## Verification (2026-10-07, under tools/mac_heavy.sh; logs in build/span-admission-2026-10-07/)

**Part A, fail-first.** The new tests ran against HEAD 4453a10 in a detached worktree; every
step failed, each for the intended reason [V]:

| Step | Exit | Why it failed |
| --- | --- | --- |
| A-admission-rs | 101 | composed context refused `ContextLanguageNotInPool { path: "hoop33/limo" }`; doc example refused at line 3 (`--- a/src/add.rs`) |
| A-calib-roundtrip-rs | 101 | report `span.fitted` false; two-fold `scored.how` null; mixed inputs not refused |
| A-span-verdicts-py | 1 | `KeyError: 'start_logits'` |
| A-schema-doc-py | 1 | documented `context_len` 145 vs trained 111 |
| A-calib-parity-span-py | 1 | `tables.all.span: None`; mixed inputs not refused |

**Part B's first run measured nothing.** It shared Part A's cargo target dir. Cargo hashes
workspace path packages relative to the workspace root, so the worktree's pre-fix library
(newer mtimes) satisfied main's build. Those logs are kept as `run1-*` and are not results.
`run.sh` now gives the worktree its own target dir and takes `PART=A|B`.

**Part B on the fix** [V]:

| Check | Result |
| --- | --- |
| rustfmt --check, this lane's five Rust files | 0 |
| `cargo test -p qd-runtime` | exit 0: every suite `ok`, 0 failed; the corpus test is the only ignored one |
| corpus admission (`QD_DEFECT_EXAMPLES` = qd-campaign composed-v2 + data/pool corpus-v3, `--ignored`) | **admitted 74,953, refused 0, skipped 0** |
| `cargo clippy -p qd-runtime --all-targets -D warnings -A clippy::items_after_test_module` | 0, after a fix: it caught `SpanLine.rows` never read, now removed |
| ruff, the ten touched Python files | All checks passed |
| pytest (ml venv), eight touched files | **216 passed, 2 skipped**: the parity benchmark (`QD_CALIB_FIT_BENCH`) and the composed real slice (`QD_SLICE_*`), both opt-in |
| `test_calib_fit_parity.py` after the clippy fix | 21 passed, 1 skipped (the benchmark), including the real phase-3 verdicts at 2053 unfitted span rows |

**Left as found** (not this lane's):
- `clippy::items_after_test_module` at `crates/qd-runtime/src/lib.rs:240`. It is red on HEAD and
  stops `--all-targets -D warnings`.
- `cargo fmt` diffs in `crates/qd-runtime/src/bin/qd_caller_records.rs`.

**Not run:**
- the full workspace `cargo test`;
- the full torch pytest suite, which the memory rule forbids running unattended beside other
  work. Only the touched files ran.

Labels: **[V]** read or run, with the source named; **[I]** inferred; **[U]** unknown.

## Rulings (advisor, per the "decisions to Fable" rule)

- **Admission moves, not training.** Admission accepts exactly the two shapes training renders,
  and no third. `docs/schema-api.md`'s example is rewritten to the single-file shape.
- Criteria 1 and 2 cannot be met from this Mac, and the task ends at `review`, not `done`.
  - Criterion 1 needs a qd-metal span head loader (GAP-J7-EXPORT-SPAN-HEAD-UNSERVED; `rg span`
    in `crates/qd-metal/src` finds no span-head code) and a non-blended head
    (GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06).
  - Criterion 2 needs a re-score that writes the new verdict fields (GPU, or MPS; runner 7b's
    MPS extract OOM'd at 48 GiB).

## Findings

- **[V] Composed contexts render as `file: <owner/repo>`.** In
  `data/pool/commitpackft-composed-v2/examples.jsonl`, `path == repo` on 25,000 of 25,000 rows,
  and `mixture.py:748` renders `file: {raw.path}`.
  - So admission refused v5's composed training rows at the **language** check on line 1
    (`language_from_path("hoop33/limo")` is `None`). That happens before the
    `---`/`+++` check GAP-ADMISSION-REFUSES-COMPOSED-DEFECT-CLASS-ROWS named.
  - The language comes from `qd-lang/src/lib.rs:78-91` [V, read]. That the old code refuses on
    language is not yet run.
- **[V] Every composed file block is in a pool language.** All 483,595 file-block paths across
  those 25,000 rows have a pool language.
  - Lines begin only with ` `, `+`, `-`, `@` or `d`.
  - No block has an `index` line.
  - 245 rows contain `/dev/null` somewhere in the text.
  - 15,199 rows end in a whitespace-only line.
  - Measured by throwaway scripts in this session's scratchpad.
- **[V] `_decode` already wrote the pointer scores, behind a switch.** `real_ft_run.py`'s
  `_decode` wrote `start_logits`/`end_logits` only behind `pointer_scores`, which no val decode
  set, and it wrote no gold rows. That is why no `--verdicts-out` file could fit a span entry.

## What changed (in this commit)

### Admission

- `crates/qd-runtime/src/admission.rs`: line 3 picks the shape.
  - `diff --git ` starts a composed context. Each block must be exactly
    `diff --git a/P b/P`, `--- a/P`, `+++ b/P`, then hunks. The language is checked per block
    path, and the `file:` line is not language-checked.
  - Anything else is single-file, unchanged.
  - Still refused: a bare `---`/`+++` preamble, `index`/mode lines, `/dev/null`, renames, and a
    block with no hunk.
- `crates/qd-runtime/tests/admission.rs`:
  - The composed shape is admitted.
  - Eight composed malformations are refused, with their line numbers.
  - A Swift block path is refused.
  - `the_documented_request_is_admitted_and_answered` reads `docs/schema-api.md`.
  - An `#[ignore]`d test, `every_trained_defect_context_in_the_named_corpora_is_admitted`,
    takes `QD_DEFECT_EXAMPLES`.
- `docs/schema-api.md`:
  - The example is the single-file shape (`context_len` 111, the same bytes as the load test's
    `d-both.req`).
  - Both trained shapes are documented, along with what to strip from `git diff`.
  - The refusal table row is updated.
- `python/tests/test_schema_api_doc.py`: `DOC_DIFF` has no preamble, and `diff_span` is (3, 3).
- `python/tests/test_qd_data_wire_agreement.py`: the "diff --git preamble" case became the bare
  `---`/`+++` preamble (still refused), and the comment is updated.

### Span verdicts

- `tools/real_ft_run.py`:
  - Every span verdict carries `start_logits`, `end_logits`, `gold_start` and `gold_end`.
  - The `pointer_scores` switch is removed, and `VERDICT_DISTRIBUTION_KEYS` carries the four
    fields.
  - `--suite-logits` still decides which suite *output* lines copy the scores.
- Test fakes updated for the removed keyword:
  - `test_real_ft_suite_logits.py`
  - `test_needle_ft_contract.py`
  - `test_real_ft_composed_slice.py`
  - `test_real_ft_ensemble.py`
- New fail-first test: `test_every_span_verdict_carries_both_pointer_distributions_and_their_gold`
  in `test_real_ft_ensemble.py`. It uses the tiny real model.

### Span fit

- `crates/qd-runtime/src/calibration_fit.rs`:
  - `nll` and the search are generalised to ragged rows. The letter path's arithmetic is
    unchanged; parity will confirm it.
  - New: `SpanRows`, `fit_span_temperature`, `SpanScore`, `score_span_rows` and
    `fit_span_entry`.
  - Definitions:
    - One temperature over 2n pointer rows, interleaved start/end.
    - `answer.rs`'s rule: score = min of the two margins; it abstains when a pointer is on the
      abstention row or the end is before the start.
    - `noul_margin` comes from `fit_noul_margin` on that score against the verdict at margin 0.
    - The conformal cutoff is `1 - q̂` over both pointers' p(gold). A span answer carries no
      set, so nothing reads this cutoff yet.
  - New unit tests: the rule, the refusals, and a synthetic fit.
- `crates/qd-runtime/src/bin/qd_calib_fit.rs`:
  - Span lines are read with their four fields, all or none. Inputs that mix the two kinds are
    refused.
  - The span entry is fitted in the run's population (`all` or two-fold, by `row_id`) and
    written into each table.
  - The report's `span` section, the span rows in `rows.jsonl`, and `population_counts.span_rows`
    are added. The "not fitted" caveat appears only when the span entry is not fitted.
- `crates/qd-runtime/tests/calib_fit_roundtrip.rs`:
  - A fitted span entry makes a span request answer through `Runtime::answer` on the reference
    backend. This is the runtime half, not a release.
  - Two-fold puts span in both fold tables and keeps a row's slots in one fold.
  - Mixed inputs are refused.
- `python/qd_train/calibration_fit.py`: `fit_span_temperature`, the oracle's definition.
- `python/tests/calib_fit_oracle.py`: span reading, fit, score, summary, rows and tables, mirrored
  for bit-exact parity.
- `python/tests/test_calib_fit_parity.py`:
  - `synthesize(pointers=True)`.
  - `assert_parity` compares the span section and its counts.
  - Span parity runs in both populations.
  - A mixed-input refusal is checked on both sides.

## Open

- `gaps.jsonl`: this lane's three updated records are in the working copy only, beside other
  lanes' uncommitted lines. Whoever commits that file next commits them all.
- `build/` is gitignored, so `run.sh` and its logs exist on this Mac only.
- Criterion 1:
  - GAP-J7-EXPORT-SPAN-HEAD-UNSERVED and GAP-G9A-SPAN-HEAD-SERVING-NEEDS-MODEL-AND-CALIBRATION-WORK
    (the qd-metal span head).
  - GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06 (which head ships).
- Criterion 2: GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS. The fitter and the writer exist, but
  no verdict file carries the fields until a re-score.
- Updated gap records have been appended (2026-10-07):
  - GAP-ADMISSION-REFUSES-COMPOSED-DEFECT-CLASS-ROWS: fixed in the working copy.
  - GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS: the code is done; the fit on real data is not run.
  - GAP-MAC-WATCHDOG-PANIC-UNDER-A-LOCKED-JOB-2026-10-07: the human resumed heavy work.
- The fail-first worktree `build/failfirst-span-admission` was removed after Part A. A re-run of
  Part A recreates it.
- Torch-free `.venv`: `test_schema_api_doc.py` and `test_qd_data_wire_agreement.py` gave
  61 passed. The rest of the `.venv` suite was not run.

## First command for the next lane

Re-verify the fix (Part B only; Part A needs the worktree above):

`PART=B bash tools/mac_heavy.sh span-admission build/span-admission-2026-10-07/run.sh`

The corpus step reads composed-v2 from
`/Users/bharath/qd-campaign/v5-data-2026-10-02/data/pool/`. The in-repo
`data/pool/commitpackft-composed-v2/` holds only the manifest. The first run pointed at the
in-repo copy and failed on the missing file; run.sh is fixed.

After that comes the re-score with `--verdicts-out` (GPU, or MPS), then
`qd-calib-fit --population two-fold`.
