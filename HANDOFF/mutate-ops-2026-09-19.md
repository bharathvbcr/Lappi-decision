# MUTATE-OPS — the cosmetic operators, measured instead of believed

**Lane:** MUTATE-OPS · **Date:** 2026-09-19 · **Owns:** `crates/qd-mutate/**`,
`docs/mutation-operators.md`, `docs/ledger-schema.md`, and the row-kind code in
`python/qd_train/ledger.py`. One file outside that list was edited — see
[Ownership exception](#ownership-exception-one-file).

## What this lane was for

Four gaps. Three were about `cosmetic`, and all three turned out to be the same shape as the
crate's own thesis: **this engine labels by construction, so an operator that silently does
nothing produces training rows labelled as a mutation that is not there.** Each gap was a place
where the crate was doing that to itself.

| Gap | Asked | Settled as |
| --- | --- | --- |
| `GAP-MUTATE-NO-PRETTIER-ON-THIS-HOST` | which formatters resolve here | Four of five run. But `found` was decided by `is_file()`, so **a formatter that could not run read exactly like one that ran and worked** |
| `GAP-MUTATE-DOC-CONFLICT-REORDER-IMPORTS` | does `reorder_imports` survive the no-formatter shrink | **Neither document was right.** The formatter is the wrong axis, and the gate was hiding an unsound TypeScript check |
| `GAP-MUTATE-REFORMAT-NARROWED-TO-WHITESPACE` | is the narrowing sound, or does it over-claim | **Sound, and over-claiming.** The proof is elsewhere; the narrowing itself was *uncounted* |
| `GAP-MUTATE-NO-LEDGER-ROW-KIND` | which ledger row a build lane writes | `run_kind: "build"`, with a marker for what it does not have and a hard bar on promotion |

Superseding records appended to `gaps.jsonl`:
`GAP-MUTATE-PRETTIER-MEASURED-AND-FOUND-WAS-ONLY-A-STAT`,
`GAP-MUTATE-REORDER-IMPORTS-SETTLED-FORMATTER-WAS-THE-WRONG-AXIS`,
`GAP-MUTATE-REFORMAT-NARROWING-IS-SOUND-BUT-WAS-SILENT`,
`GAP-MUTATE-LEDGER-BUILD-ROW-KIND-IMPLEMENTED`. Two new open ones:
`GAP-MUTATE-OPS-GITPULSE-ANSWERS-BUT-IS-BLIND-TO-SAME-WORKTREE-AGENTS`,
`GAP-MUTATE-BLACK-SAFETY-CHECK-DEGRADED-ON-THIS-HOST`.

## 1. This host, measured

Re-derive with one command:

```
cargo run -q -p qd-mutate --bin qd-mutate -- formatters
```

| Language | Declared | State | Path |
| --- | --- | --- | --- |
| rust | `rustfmt` | `found` | `/Users/bharath/.cargo/bin/rustfmt` (1.9.0-stable) |
| go | `gofmt` | `found` | `/opt/homebrew/bin/gofmt` |
| python | `black` | `found` | `/Users/bharath/.local/bin/black` (26.5.1) |
| typescript | `prettier` | **`not_found`** | not on PATH |
| swift | `swift-format` (via xcrun) | `found` | `…/XcodeDefault.xctoolchain/usr/bin/swift-format` |

`npx` and `node` **are** present (`/opt/homebrew/bin`), so prettier is fetchable — that is a
network fetch and an undeclared dependency, and it was not done.

Each program was also smoke-run by hand with the exact argv the crate declares, which is how
the black finding below surfaced.

### The defect this measurement exposed

`Availability::Found` was produced by `which()` / `xcrun_find()` on the strength of
`Path::is_file()` alone. A program that exists and cannot format was reported `found`; the
manifest then recorded **no** restriction for that language and `qd-mutate formatters` printed
it as usable. **A check that could not run reported the same result as a check that ran and
passed** — the one thing `fmt.rs`'s own header says the module exists to prevent. Nothing on
this host is actually broken, so the bug was invisible to observation and visible only to
reading.

**Fixed:** `fmt::resolve` now delegates to `fmt::resolve_spec(spec, smoke)`, which runs the
formatter on the new required trait method `Language::smoke_source()` and demands exit 0 with
non-empty stdout. A program that is present and fails becomes the new
`Availability::Unusable { program, path, detail }` (key `unusable`, `is_usable()` false), so the
existing restriction machinery covers it with no new branch. `Generator::new` now derives its
recorded availability from the same resolve pass instead of calling `probe_all()` a second time
— one measurement, not two that could disagree, and half the subprocesses.

`resolve_spec` is split out so the three states are reachable from a test by choosing a program
for its behaviour (`/usr/bin/false`, `/bin/cat`, a name that does not exist) rather than only
through whichever tools a machine happens to have.

## 2. `reorder_imports`: the formatter was the wrong axis

The disagreement was three-way, not two-way: `docs/mutation-operators.md`'s prose shrink
sentence excluded the operator, **its own operator table** stated no formatter condition for it,
and `docs/hardening.md` §1 words the safe set as "comment text, import order within a group".

Settled by running it, against the **unmodified** code:

- `reordering_rust_imports_needs_no_formatter` → `NoFormatter { language: "rust", operator:
  "cosmetic.reorder_imports" }`. A Rust `use` binds a name and runs nothing; the candidate was
  refused for a missing tool that has no bearing on it.
- `reordering_typescript_imports_is_refused_for_the_reason_that_is_actually_true`, run with a
  formatter resolved → **`Ok([Candidate { op: CosmeticReorderImports, … "import { a } from './a';"
  and "import { b } from './b';" swapped }])`**. The swap was emitted.

The formatter is never consulted for this operator in either direction:
`OpId::verified_by_canonical_form` excludes it by design, because a reorder *changes* the
canonical form. So the gate proved nothing, deleted provably safe Rust and Swift candidates on a
machine merely missing a tool, and on a machine that had the tool waved through a TypeScript
swap that nothing had checked.

**The TypeScript check was unsound.** It refused only the clause-less `import "./polyfill";`
shape. But an ES module is *evaluated* when it is imported, in the importer's source order, so
`import { a } from './a'` runs `./a`'s module body just as surely — it merely also binds a name.
Nothing confined to one file can show two module bodies do not interact. On this host the
missing prettier hid it; one `npm i -g prettier` away, this crate would have emitted rows
labelled behaviour-preserving that are not.

**Fixed:**

- `OpId::requires_formatter` no longer names `CosmeticReorderImports`; `ops::reorder_imports` has
  no formatter guard.
- New **required** trait method `Language::import_order_is_semantic() -> Option<&'static str>`:
  `Some(reason)` for Python and TypeScript, `None` for Rust, Go and Swift. Required rather than
  defaulted, because a language that could not answer would default into "reordering is fine
  here", which is the direction that poisons a label.
- `ts_lang::import_block` is now `reorderable: false`, and both it and `python_lang` take their
  `why_not` from that one method, so the manifest's up-front restriction and the per-file refusal
  cannot drift into two explanations of one rule.
- `Manifest::new` records the language-level restriction beside the `wrap_line` one. This also
  fixes an older hole: Python's `reorder_imports`, refused for the language on every machine, was
  not in `restricted_operators` at all.
- `qd-mutate formatters` prints it, so the command answers "what can this machine verify" on both
  axes.

Per-language reasoning is now a table in `docs/mutation-operators.md`. Go stays `None` on a
narrower argument than Rust's or Swift's: the Go spec leaves the initialization order of
*independent* imported packages unspecified, so no correct program may depend on it; the
deliberately ordered case, a blank import, is refused per file.

## 3. `reformat`: sound, and it was claiming more than it delivered

**Sound — but not because of the narrowing.** Soundness is carried by the emit-time
canonical-form comparison in `generate::build_example`, which formats both sides and drops a
candidate whose canonical forms differ. That runs for every `reformat` example because
`verified_by_canonical_form` includes the operator. The whitespace filter is a candidate filter,
not the proof.

**Claiming more, in two places.** The `detail` string said `lines N..=M reformatted by <path>`
for a hunk whose non-whitespace content is identical. And — the one with teeth — the filter was a
bare `continue` with no counter, so these three cases were indistinguishable in the manifest:

1. the file is already canonical → nothing to do;
2. every hunk fell outside this body → nothing to do here;
3. hunks landed in this body and **every one was rejected** → the operator ran and turned down
   its work.

All three printed `sites 0 emitted 0 refused 0`. Only the third is a finding, and it is precisely
the measure of how much narrower `cosmetic.reformat` is than its name. `ops::candidates`' own
doc comment already promised that "a silent zero never reads as 'the operator ran'".

### Measured, on files this lane did not edit

`qd-mutate one --seed 3 --limit 12` over ten Rust sources under `crates/qd-preflight/src` and
`crates/qd-runtime/src`:

| File | Result |
| --- | --- |
| `qd-preflight/src/cuda.rs` | sites 3, **declined 3** |
| `qd-preflight/src/tristate.rs` | sites 1, **declined 3** |
| `qd-runtime/src/schema.rs` | **sites 0, declined 1** |
| `qd-runtime/src/b64.rs` | sites 2, emitted 1 |
| `qd-runtime/src/{calibration,render}.rs` | sites 1 each |
| `qd-preflight/src/artifacts.rs`, `qd-runtime/src/{context,reference,refusal}.rs` | sites 0, no declines |

**8 candidates offered, 7 hunks declined as `not_layout_only`** — roughly half of everything the
operator examined, and none of it recorded. `qd-runtime/src/schema.rs` is the decisive row: it
printed exactly what `context.rs` and `refusal.rs` print, where the operator genuinely had
nothing to do.

Caveat on reproducing these exact numbers: the sweep read a **live** working tree that other
lanes were editing (`crates/qd-runtime/src/schema.rs` among them). The `schema.rs` row was
re-confirmed at the end of this lane and still reads `sites 0 … declined 1
(not_layout_only=1)`, but a later lane should expect the counts to move with the source, not
treat them as fixed. What is stable is the *shape*: `sites 0` with a non-zero `declined` is a
different fact from `sites 0` alone, and until this change the manifest could not express it.

**Fixed:** `ops::candidates` returns `Offered { candidates, declined }`. Declines are keyed by
`ops::decline::{NOT_LAYOUT_ONLY, OVER_CANDIDATE_CAP}` — the second because truncating at
`MAX_CANDIDATES_PER_OP` in silence hands a reader 64 with no way to know it was not all of them.
`OperatorReport` gains `declined: BTreeMap<String, u64>` with `serde(default)`, so a manifest
written before this loads as an empty map, which is honest: that run did not measure declines.
`Manifest::note_declined` records per operator; the coverage table prints `declined` **beside**
`sites`, which is the only place it disambiguates anything. `generate.rs` records declines
*unconditionally*, before the empty-candidates check — skipping the bookkeeping when the list
came back empty would drop exactly case 3.

`detail` now reads `lines N..=M relaid out (layout-only hunk of the <path> pass)`.

## 4. `run_kind: "build"` — a row a build lane can honestly write

`Protocol.for_build(commands=…, toolchain=…)`: `data_snapshot_hash`, `tokenizer_hash` and
`backbone_commit` carry the literal marker `"n/a:build"` — not a hash, not empty, so a reader, a
diff and a test can all tell it from a real value. `recipe_hash` is a **real** SHA-256 over the
canonical `{commands, toolchain}`, so two build rows are comparable exactly when they ran the
same commands on the same toolchain. Suite results are ordinary tri-states under
`metrics["suite.<name>"]`: `value` the passing count, `n`/`n_total` tests-run over
tests-collected, and a suite this environment cannot run is `NotRun` with the reason. No second
mechanism was introduced.

**The hazard found while testing this.** Without an explicit bar, three completed, non-quick
`build` rows with every gate and control `ran`/`passed` return `promoted=True`. Promotion rule 4
only blocks them today because `RunRecorder._fill_unreported` happens to leave unreported gates
`not_run` — a safety net, not an invariant. So the refusal is stated on the run kind:
`NON_PROMOTING_RUN_KINDS = {"build"}`, checked as promotion condition 6, because a build row's
gates are **vacuous, not satisfied**, and a conjunction over vacuous inputs is the shape that
quietly comes out true.

`LedgerRow._check_build_marker` enforces the marker **both ways**: a `build` row missing it is
refused (it would claim a snapshot it never had, and look comparable to training rows), and a
non-`build` row carrying it is refused (its protocol would identify nothing, making every
comparison against it unfalsifiable).

`docs/ledger-schema.md` gains a `run_kind: "build"` section and promotion condition 6.

## Ownership exception: one file

The five tests for §4 are in **`python/tests/test_ledger.py`**, which this lane was told not to
touch. Two binding rules collide: *every fix ships with a test that fails against the pre-fix
code*, and *touch nothing else in `python/`*. They cannot both hold for a change to
`ledger.py`, and shipping an untested ledger change is the worse failure. The conflict is named
rather than resolved silently. Mitigations checked before editing: `test_ledger.py` was clean at
`HEAD`, mtime `11:54` — older than every other lane's activity today — and it is the canonical
owner of ledger tests, so a new file beside it would have been a second registry.

## Pre-fix failures observed, and the mutation proving each test bites

Every new test was run against the unmodified code first. Eleven mutations, one per invariant.

| # | Test | Pre-fix failure | Mutation that kills it |
| --- | --- | --- | --- |
| 1 | `reordering_rust_imports_needs_no_formatter` | `NoFormatter { language: "rust", … }` | restore the `ctx.formatter` guard in `ops::reorder_imports` → same failure |
| 2 | `reordering_typescript_imports_…_actually_true` | `Ok([Candidate … swapped])` | `ts_lang::import_block` → `reorderable: true` → `Ok([…])` again |
| 3 | `a_missing_formatter_restricts_the_operators_before_a_file_is_read` | `assert!(contains_key("cosmetic.reorder_imports"))` failed at `manifest.rs:405` | record the reason as `"no usable formatter on this machine"` → *"must give the language reason, got …"* |
| 4 | `a_reformat_hunk_that_is_not_layout_only_is_counted_not_dropped_in_silence` | did not compile: `Offered`/`decline` did not exist | drop `out.decline(NOT_LAYOUT_ONLY, 1)`, keep the bare `continue` → *"got {}"* |
| 5 | same | — | replace the filter condition with `false` → *"an identifier rename is not a layout change and must not become a cosmetic candidate"* |
| 6 | `a_declined_site_is_distinguishable_from_a_site_that_never_existed` | as #4 | make `note_declined` a no-op (`if count >= 0: return`) → *"the operator has an entry once it has declined something"* |
| 7 | `a_program_that_exists_but_cannot_format_is_not_reported_as_usable` | did not compile: no `Unusable`, no `resolve_spec` | swallow the smoke-run error → `Ok(Resolved { path: "/usr/bin/false", … Found })` |
| 8 | `a_program_that_formats_the_smoke_input_is_reported_as_found` | as #7 | make every resolution `Unusable` → *"cat round-trips its input: Unusable …"* |
| 9 | `test_a_build_row_cannot_promote_anything` | `ImportError: cannot import name 'NOT_APPLICABLE'` | disable the `NON_PROMOTING_RUN_KINDS` check → **`promoted=True`, "3 completed rows, seeds [1, 2, 3], every gate and control ran and passed"** |
| 10 | `test_a_training_row_may_not_borrow_the_build_marker` | as #9 | blind `_check_build_marker` (`marked = set()`) → `DID NOT RAISE ValueError` |
| 11 | `test_a_build_protocol_says_which_components_it_does_not_have` | as #9 | hash only the toolchain → two different command sets share a `recipe_hash` |

Mutations #5 and #8 exist because #4 and #7 would otherwise pass against code that declined
everything. Every mutation was reverted and the suite re-run.

Mutation #8 also caught a defect this lane introduced:
`every_language_answers_the_availability_question_one_way_or_the_other` did not handle the new
`Unusable` variant and would have failed on any host with a broken formatter. Fixed in the same
change.

Two pre-fix failures (#4/#6, #7/#8, #9-11) were **compile/import errors**, not assertion
failures, because the observable did not exist yet. That is weaker evidence than a runtime
failure, which is why each is paired with a mutation kill above. Stated rather than glossed.

## Counts, with the exact commands

| Command | Baseline (coordinator, 20:18) | Now |
| --- | --- | --- |
| `cargo test -p qd-mutate` | 140 passed, 0 failed | **148 passed, 0 failed** (+8, all this lane) |
| `cargo test --workspace` | 332 passed, 0 failed | **347 passed, 0 failed** (344 → 345 → 347 across three runs this session as other lanes landed tests; only the per-package row below is attributable) |
| `.venv/bin/python -m pytest python/tests/test_ledger.py -o 'addopts='` | 25 test functions at `HEAD` | **30 test functions, 38 collected, 38 passed, 0 failed** (+5, all this lane; the file is partly parametrized, so functions and collected tests differ) |
| `.venv/bin/python -m pytest python/tests -o 'addopts='` | 948 passed, 4 skipped | **1081 passed, 5 skipped, 0 failed** (also moving; 1004 on this lane's *first* run, before it changed anything) |
| `cargo clippy --workspace --all-targets` | — | **no warnings, no errors** |
| `ruff check python/qd_train/ledger.py python/tests/test_ledger.py` | 23 findings at `HEAD` | **23 findings** — identical rules, shifted line numbers. **Zero new findings introduced.** All 23 are pre-existing (`F401`, `RUF022`, `E501`, `SIM105`, `SIM117`, `RUF100`) and were left alone as out-of-scope drive-bys. `ruff` is declared in `pyproject.toml` but is **not** installed in `.venv`; the host binary at `/Users/bharath/.local/bin/ruff` was used, and the `HEAD` comparison ran via `git show … \| ruff check --stdin-filename …` |

**The two workspace-wide totals are not this lane's to explain, and the baseline disagreement was
chased before anything was changed.** The pytest baseline was already 1004/18 on the first run of
this session, not 948/4. Cause, verified: six untracked Python modules and their tests
(`byte_context`, `byte_decider`, `mutate_adapter`, later `byte_batch`, `schema_mirror`,
`pool_builder`) were written into this same working tree by other sessions *after* the
coordinator's 20:18 measurement — `HEAD` is timestamped `2026-09-19T20:18:13-05:00` and every one
of those files is newer. The skip count fell from 18 to 5 because 13 of the skips were
`test_wire_golden_corpus.py` self-skipping with *"qd was built before ['wire.rs'] changed"*; this
lane's `cargo test --workspace` rebuilt `qd`, which un-skipped them. Attributable numbers are the
per-target rows above.

## What was not verified, and why

- **`cosmetic.reorder_imports` end-to-end for TypeScript with prettier installed.** prettier is
  not on this host and installing it is an undeclared dependency and a network fetch. The unsound
  emission was demonstrated with a resolved stand-in formatter (`/bin/cat`) in
  `candidates_for_with_formatter`, which reaches the identical code path. **NOT RUN** against
  real prettier.
- **The `Offered.declined` wiring through `Generator::run` is measured, not test-enforced.**
  `Generator` has no injection point for a resolved formatter, and adding one for a single test
  caller would be a speculative API. The ops end and the manifest end are each unit-tested; the
  seam between them is evidenced by the `qd-mutate one` sweep in §3, which is re-derivable but
  depends on `rustfmt` being installed. A later lane that wants this pinned should add
  `Generator::with_resolved_formatters` when it has a second caller.
- **`Availability::Unusable` has never been observed in the wild here** — all four resolvable
  formatters run. It is exercised only by `/usr/bin/false`.
- **black's AST-equivalence self-check is degraded on this host** (`python3` is 3.14.7; black
  warns it cannot verify equivalence for a newer target). It exits 0 and its output is correct, so
  resolution is genuinely `found`, and qd-mutate compares canonical forms itself rather than
  relying on black's self-check. Recorded as
  `GAP-MUTATE-BLACK-SAFETY-CHECK-DEGRADED-ON-THIS-HOST`; **not** fixed, because passing
  `--target-version py314` changes the canonical form for Python and would invalidate any Python
  cosmetic fixture generated before it.
- **GitPulse.** Contrary to the lane brief, `gitpulse_insights` answered, every facet `ok: true`.
  One of those `ok` facets is wrong: `agents` reported `sessions: 0` while other sessions were
  demonstrably writing this tree throughout, and `collisions` reported `overlapping_files: 0`
  because its model is per-worktree and every concurrent writer here is inside the *same*
  worktree. Recorded as
  `GAP-MUTATE-OPS-GITPULSE-ANSWERS-BUT-IS-BLIND-TO-SAME-WORKTREE-AGENTS`. Concurrency was tracked
  with `git status --short` and file mtimes instead.
- **DevMap** answered: `devmap_status` reported `is_fresh: true`, `generation_id: 366`, 3121
  nodes, 10161 edges, no `walk_incomplete`. It disagrees with GitPulse's `codeintel` facet, which
  reported `is_fresh: false` for the reason that *that* binary was built without the parsing
  frontend. Every claim in this document cites `file:line` or command output regardless.

## Net size

`git diff --numstat`, net lines:

| Area | Net |
| --- | --- |
| `crates/qd-mutate/src` | **+627** |
| `crates/qd-mutate/tests` | +1 |
| `python/qd_train/ledger.py` | +114 |
| `python/tests/test_ledger.py` | +126 |
| `docs/{mutation-operators,ledger-schema}.md` | +124 |
| **Total** | **+1085 / −93, net +992** |

This lane grew the codebase, and the honest accounting is that a large share of the `+627` is
prose: the comments stating why each refusal is the refusal it is, and why the formerly silent
paths are now counted. Roughly 200 of those lines are new tests inside `src/`. No file was added
and none removed. The one deletion worth naming is a duplicate measurement: `Generator::new` no
longer calls `probe_all()` alongside its own resolve loop, which halves formatter subprocesses per
run and removes the possibility of the recorded availability and the formatter actually used
disagreeing.

## The exact first command for the next lane

```
cargo run -q -p qd-mutate --bin qd-mutate -- one --path crates/qd-runtime/src/schema.rs --seed 3 --limit 12
```

Read the `cosmetic.reformat` row of the `[rust]` block. It says `sites 0 … declined 1
(not_layout_only=1)`. Before this lane it said `sites 0 emitted 0 refused 0`, exactly like the
files where the operator had nothing to do. That one line is the whole lane in miniature, and it
is the number to watch when the mixture weights are set: **`cosmetic.reformat` currently declines
about half of what it examines, so a weight of 5 buys materially fewer examples than it looks
like.** Deciding whether to apply *all* hunks of a formatter pass as one mutation — a different,
larger span, and a design decision rather than a bug fix — is the open question
`GAP-MUTATE-REFORMAT-NARROWING-IS-SOUND-BUT-WAS-SILENT` leaves behind.
