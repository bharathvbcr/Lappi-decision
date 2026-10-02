# Ledger schema (S6)

The ledger is the decision record. **The decision reads the ledger, not the console.**

Two rules from the plan drive every design choice here:

- *"Every run writes to the protocol-hashed ledger before it is allowed to finish. A run whose row is
  missing is rerun, not remembered."*
- *"A run cannot exit 0 until its row is written. A deliberately broken run must produce a row that
  says so."*

Storage: `ledger/runs.jsonl`, append-only, one JSON object per line, UTF-8, newline-terminated.
Append-only is enforced by the writer (open in `O_APPEND`) and by a test that rewriting a byte of an
existing line is detected via the chain hash.

## The tri-state rule (this is the load-bearing invariant)

Every metric, gate and control is a **tri-state**, never a bare number or a bare boolean:

```json
{ "state": "ran", "passed": true, "value": 0.83, "n": 300 }
{ "state": "ran", "passed": false, "value": 0.41, "n": 300 }
{ "state": "not_run", "reason": "no GPU in this environment: MPS only, suite needs CUDA" }
```

`state` is `ran` | `not_run`. There is **no third option and no default**. A consumer that reads a
tri-state must branch on `state` before touching `passed` or `value`; the deserializer refuses a row
where `state == "ran"` and `passed` is absent, and refuses a row where `state == "not_run"` and
`reason` is empty.

This exists because *a check that could not run must never report the same result as a check that ran
and passed*. The common failure is an `Option<bool>` that a report renders as "no failures found".
Here, `not_run` has no `passed` field at all, so a report that prints it as passing has to fabricate
one — which the schema test catches.

**Aggregation rule.** A gate over several tri-states is `ran/passed` only if *every* input is
`ran/passed`. If any input is `not_run`, the aggregate is `not_run` and carries the union of reasons.
An aggregate is never allowed to be more confident than its least-informed input.

**Coverage rule.** Any metric computed over a sample carries both `n` (examined) and `n_total`
(eligible). A report renders `n/n_total`. Presenting a capped sample as complete coverage is a bug,
not a rounding choice.

## Row

| Field | Type | Notes |
| --- | --- | --- |
| `row_id` | string | UUIDv4 (`uuid.uuid4()`). Unique; the writer refuses a duplicate. This table said ULID until 2026-09-19; the code has always written uuid4 and the code wins |
| `written_at` | RFC3339 string | UTC |
| `prev_row_hash` | string \| null | SHA-256 of the previous line's bytes; `null` on the first row |
| `protocol_hash` | string | SHA-256 over the five components below, canonically serialized |
| `protocol` | object | `{data_snapshot_hash, tokenizer_hash, backbone_commit, recipe_hash, seed}` |
| `run_kind` | enum | `teacher` \| `lr_probe` \| `cpt` \| `prune_heal` \| `ft` \| `ablation` \| `eval` \| `calibration` \| `smoke` \| `throughput` \| `resume` \| `scale` \| `build` (see below) |
| `status` | enum | `completed` \| `killed` \| `failed`. A killed run still writes a row |
| `quick` | bool | true if <3 seeds, truncated schedule, or subsampled. **A `quick` row cannot promote anything**. Set by the caller; since 2026-09-29 `RunRecorder` also sets it on a `cpt`/`ft`/`prune_heal` row whose `train.termination` is not `steps_exhausted`, appending the reason (it never clears it). Seed count is judged per family by `promotion_verdict` (rule 3 below), since a one-seed-per-unit row cannot see its family |
| `quick_reason` | string \| null | Required non-empty when `quick` is true |
| `code_commit` | string | git HEAD of this repo at launch; `-dirty` suffix if the tree was dirty |
| `env` | object | `{torch, transformers_sha, fla_present, causal_conv1d_present, device, host}` |
| `metrics` | object | tri-states, keyed by name; see below |
| `noul_rate` | object | tri-state; in-domain abstain rate |
| `controls` | object | `{shuffled_label, privileged_hunk, degenerate_head, transfer_gate}` — all tri-states |
| `gates` | object | `{paired_margin_vs_linear, ood_abstain, needle_hunk_recall, permutation_consistency, ece}` — all tri-states |
| `wall_clock_s` | number | measured, not estimated |
| `wall_clock_source` | enum | `caller` \| `recorder` \| `unrecorded` — where the duration above came from. Added 2026-09-21 (`a409895`); rows written before it read back as `unrecorded`, which is what they say, rather than being assigned a source nobody wrote down |
| `cost_usd` | number | measured from instance rate x wall clock |
| `notes` | string | free text; never parsed by a gate |
| `recipe` | object \| null | the settings `protocol.recipe_hash` was taken of, stored as well as hashed. `null` on a row that does not record them, which is every row written before 2026-09-21 — never `{}`, which would claim the run had no settings |

### `recipe`

`recipe_hash` makes two runs of different settings incomparable, which is its job, and it
makes neither of them readable. Until 2026-09-21 every field a runner varied —
`train_subsample`, `epochs`, `batch_size`, `val_share`, `lr`, `span_weight`, `deterministic`
— went into the hash's input and was stored nowhere, so a row could say that two arms of a
sweep were not comparable and not say how they differed. Labelling the three points of a
learning curve that day took re-hashing four candidate `train_subsample` values against the
other seven fields pinned at the launch command's: correct, and it recovers the label from
the log by a longer route rather than from the row.

Rules:

* **It is the same object the hash was taken of.** Every runner builds the recipe once,
  names it, and passes that name to both. A stored recipe that does not reproduce the row's
  own `recipe_hash` is worse than no recipe, because it reads as an answer.
* **`null` means unrecorded, not empty.** `{}` would claim the run had no settings; a run
  with a `recipe_hash` always had some. `LedgerRow` refuses an empty mapping.
* **It does not enter any hash.** `protocol_hash` and `recipe_hash` are unchanged by its
  presence, so rows written before it stay comparable with rows written after — which
  matters: the capacity sweep and the learning curve of 2026-09-21 were written without it.
* **Each tool keeps its own spelling of the hash.** `tools/real_ft_run.py` uses
  `separators=(",", ":")` and `tools/rung0_real_run.py` does not, so the two produce
  different hashes for identical dicts. Unifying them would move every future
  `recipe_hash` and break comparability with every row already written, which is why it has
  not been done; see `GAP-SIX-SPELLINGS-OF-ONE-RECIPE-HASH`.

### `metrics` keys

Broken out **by k and by language**, because the plan's calibration gate is per-k and per-language and
an aggregate hides exactly the failure it is meant to catch:

- `accuracy.k{2,3,4,8,16}` — one tri-state each
- `accuracy.lang.{rust,go,swift,python,typescript}`
- `brier.k{...}`, `ece.k{...}`, `ece.lang.{...}`
- `span_iou`, `span_exact`
- `ordinal_mae`, `ordinal_spearman`
- Per family, on a `tools/real_ft_run.py` FT `eval` row, **report-only** (never aggregated into
  a gate): `permutation_consistency.family.{family_id}` and
  `ood_abstain.in_distribution.family.{family_id}`, whose `n`/`n_total` sum to the gate or pooled
  metric they break down; `ood_abstain.in_distribution.gold_noul.family.{family_id}`, the rows
  that bound excludes by contract; and `ece.family.{family_id}.{kind}.k{...}`

A key that was not computed is present with `state: "not_run"` and a reason, or absent entirely.
It is **never** present with a zero value.

## `fla_present` is not cosmetic

`env.fla_present` records whether `flash-linear-attention` was actually active, established by
grepping the startup log for the fallback warning (S1). Without it the GDN layers run an fp32 torch
chunk loop ~43% slower. A throughput or cost number recorded with `fla_present: false` is not
comparable to one recorded with `true`, and the comparison code refuses to pair them.

`fla_present` is itself a tri-state in the row (`ran/passed` = confirmed present; `ran/failed` =
confirmed fallback; `not_run` = the dry run did not execute, e.g. on the Mac where there is no CUDA).
An unchecked environment must not read as a clean one.

## `run_kind: "build"` — the row a build-and-test lane writes

Every other `run_kind` names a training or evaluation run, and their required protocol components —
`data_snapshot_hash`, `tokenizer_hash`, `backbone_commit` — have no meaning for a lane that compiles
the workspace and runs a test suite. Repo rule 5 says *a number in a report cites a ledger row or is
not in the report*, so before this kind existed a build lane had no honest way to comply: three lanes
in a row printed command output instead and flagged it (`GAP-MUTATE-NO-LEDGER-ROW-KIND`). Inventing a
value for the missing components would have been worse than the gap — a fabricated entry in the
decision record — so the kind carries a marker instead.

**Protocol.** `Protocol.for_build(commands=…, toolchain=…)` builds it:

| Component | Value |
| --- | --- |
| `data_snapshot_hash` | `"n/a:build"` |
| `tokenizer_hash` | `"n/a:build"` |
| `backbone_commit` | `"n/a:build"` |
| `recipe_hash` | **real**: SHA-256 over the canonical `{commands, toolchain}` — two build rows are comparable exactly when they ran the same commands on the same toolchain |
| `seed` | `0` unless the lane has a reason |

`"n/a:build"` is not a hash and not empty, so a reader, a diff and a test can all tell it from a real
value. The marker is checked **in both directions**: a `build` row missing it is refused, and a
non-`build` row carrying it is refused. A training row with an unidentified protocol component makes
every comparison against it meaningless, and a build row claiming a snapshot it never had looks
comparable to training rows it has nothing to do with.

**Metrics.** Suite results go in `metrics` under `suite.<name>`, as ordinary tri-states — there is no
second mechanism:

```json
{ "suite.cargo_test_workspace": { "state": "ran", "passed": true, "value": 344, "n": 344, "n_total": 344 },
  "suite.gpu_throughput":       { "state": "not_run", "reason": "no CUDA on this host: MPS only" } }
```

`value` is the passing count, `n`/`n_total` the coverage pair — tests that ran over tests that were
collected, so a suite whose collection silently shrank is visible. A suite the environment cannot run
is `not_run` with the reason, never absent and never a zero.

**`wall_clock_s` is measured** as for any other row. `cost_usd` is `0.0` for a local build; it is a
measured quantity, and zero is the measurement, not a placeholder.

**A `build` row never promotes.** See below.

## Recording a build run: the one command

```
PYTHONPATH=python .venv/bin/python -m qd_train.ledger record \
  --toolchain "cargo 1.98.0 / CPython 3.14.7 (.venv)" \
  --suite cargo_test_workspace="cargo test --workspace" \
  --suite pytest_python_tests=".venv/bin/python -m pytest python/tests -o addopts=" \
  --suite pytest_torch_python_tests="uv run --no-project --python ~/.venvs/ml/bin/python \
      --with pytest --with hypothesis --with datasketch python -m pytest python/tests -o addopts="
```

It runs each suite, writes one `build` row, and prints **the row id alone on stdout** — everything
else goes to stderr, so `ROW=$(… record …)` works. Cite that id. `--ledger` defaults to
`ledger/runs.jsonl`, `--repo` to the repo root.

**Run `make ledger-record`, not this, unless you are learning the interface.** The shape above is
the CLI's contract and is what the flags mean; the *commands* this repo gates on live in the
`Makefile`, in one `*_RUN` variable per suite that both `make <target>` and the `--suite` argument
are built from. That is not tidiness. Until 2026-09-21 this example named two suites while the gate
ran three, so a lane following it literally wrote a row with no model-path coverage at all — the
precise hole the third suite was added to close — and on the same day the Makefile's own two
spellings of the torch command drifted by one flag, leaving a row whose `detail` named a command
nobody had run. `python/tests/test_lint_gate.py` now asserts the recorded command is the command
its target runs; nothing can assert that about a code block in a document, which is why the
document should not be the source.

### Which reading of rule 5 this implements

Repo rule 5 has two clauses and they have different scopes. *"Nothing is green that was not run"*
governs runs. *"A number in a report cites a ledger row or is not in the report"* governs **numbers
in reports** — that is the clause a build lane trips, and it is read literally: a lane iterates as
freely as it likes while working, and records the one run whose numbers it is about to publish in a
handoff, a doc or a gap record. Every `pytest` during development does not get a row; the count you
paste into `HANDOFF/` does.

This is the wider of the two available readings — the narrow one would exempt test counts as
"command output". It was chosen because the counts *are* the load-bearing numbers here (the repo's
green/red state is nothing else), and because they have already been wrong in this repo in ways a
row would have caught: five vacuous tests, three false survivors from `.pyc` caching, and a briefed
baseline of "1094 passed" that measured 1104 an hour later with no way to tell whether the tree or
the measurement had moved. A row pins commit, dirty state, host, wall clock and the exact argv, and
that is what makes a count mean something.

### The four outcomes a suite can have

The parser never invents a number, and the three not-passed outcomes are kept distinct:

| What happened | Recorded as |
| --- | --- |
| binary missing, `OSError` on spawn | `not_run`, reason names the binary. **Never a zero** |
| killed at `--timeout` | `not_run`, reason names the limit. A suite cut off has no result |
| counts parsed | `ran`, `passed = exit 0 AND zero failures` — the exit code has a veto over clean counts |
| no parsable counts | exit ≠ 0 → `ran/failed` (a compile error is a real failure). exit 0 → **`not_run`**, because a pass with no counts cannot be told from a suite that collected nothing |

`value` is the count that ran and did not fail. `n`/`n_total` is **executed over collected**, so
skipped, ignored, deselected and filtered-out tests widen the denominator and a silently shrunk run
is visible as `12/344` rather than as a smaller, tidier pass count. `cargo test` prints one summary
per test binary and they are summed; a parser reading only the last line would report the doc-tests'
`0 passed` for the whole workspace.

**A run that was cut short has no denominator.** `cargo test` is fail-fast: after the first failing
test binary it stops and never builds the rest, and the tests in those binaries appear in no summary
line at all — not as `ignored`, not as `filtered out`. pytest's `-x` and a collection error do the
same. Summing what the harness printed then yields `n == n_total` for a run that covered a fraction
of the suite, which is this repo's forbidden shape with the numbers pointing the wrong way. So each
parser detects its harness's own abort announcement (`error: test failed, to rerun pass …`,
`stopping after N failures`, `Interrupted:`) and sets `collection_complete=False`; `as_tristate`
then keeps `value` and **drops the coverage pair**, because nobody measured the eligible population.
`coverage_str()` reads `coverage unstated` and `is_complete_coverage` is False.

This was found by the first real row, not by reasoning about the code: row
`06d15c1b-e643-4fae-a478-32b7581ef632` recorded
`suite.cargo_test_workspace: FAILED value=299 coverage=301/301` against a 347-test baseline. cargo
had not lost 46 tests; it had stopped. That row stays in the ledger — it is append-only, and a wrong
row is corrected by a later row, never by an edit.

Exit codes are three, for the same reason the tri-state is: `0` all suites ran and passed, `1` at
least one ran and failed, `3` none failed but at least one did not run. `$?` is a trace like any
other, and an unexamined suite must not leave the same one as an examined one.

Commands are argv, run without a shell. A pipeline or redirect is **refused**, not split: `cargo
test --workspace | tail -5` would hand `|` to cargo as a test-name filter and cargo would exit 0
having run nothing, which is a green row for a run that did not happen.

### Recording and the report that cites it

A report is written after the run it reports, so the tree always moves once more after the last row.
The terminating rule: **record after the last change to code, tests, or data a suite reads.** Prose
written afterwards — a `HANDOFF/` file, a `gaps.jsonl` append — is not part of the run. The one
exception is `python/tests/test_gaps_ledger.py`, which scans prose for gap ids; a handoff citing a
newly minted id should be followed by one more recorded run.

## Promotion

A result may promote a decision only if **all** hold:

1. `status == "completed"`
2. `quick == false`
3. Three rows exist sharing a `protocol_hash` that differs only in `seed`
4. Every gate in `gates` is `state: "ran"` — a `not_run` gate blocks promotion, it does not pass it
5. Every control in `controls` is `ran` and `passed`
6. `run_kind` is not in `NON_PROMOTING_RUN_KINDS` — currently `{build}`

Rule 4 is the one that is easy to get wrong and the reason the tri-state exists.

Rule 6 is stated on the run kind rather than left to rule 4 to catch. A build lane's gates are
*vacuous*, not satisfied: it has nothing to say about `paired_margin_vs_linear`, and the only reason
rule 4 would currently block it is that `RunRecorder._fill_unreported` marks unreported gates
`not_run`. That is a safety net, not an invariant — a build lane that set a gate for an unrelated
reason would promote. Measured: with rule 6 removed, three completed non-quick `build` rows with
every gate and control `ran`/`passed` return `promoted=True`
(`python/tests/test_ledger.py::test_a_build_row_cannot_promote_anything`).

7. No gate or control that **states** its coverage saw fewer than all eligible items —
   a `ran/passed` result carrying `n < n_total` refuses, it does not promote.
   Coverage that is unstated (`n` absent) is not treated as partial.

Rule 7 is the Coverage rule made enforceable on the verdict. `promotion_verdict` used to
ignore `n`/`n_total`, so a gate measured on 1 of 1000 eligible items promoted exactly like
one measured on 1000 of 1000. A `PROMOTE` states the weakest coverage it promoted on, so
"complete coverage" is never implied by silence. Unstated coverage stays promotable: many
gates are a single observation with no population, and refusing those would make promotion
unreachable rather than honest.

### The human decisions record: `docs/promotion-decisions.json`

The questions only a human may answer (rule 2) are **data a human edits**, not code. Every
verdict reads this file and carries what it read:

- **`promotion_population`** on every `PromotionVerdict`: the val population the gates are
  judged on, its `families` (`"all"` or a list of family ids), `status` (`open` until a human
  rules), the `source` it rests on, who decided it and where, and the record's sha256. Nothing
  in code infers it. Today it is `open`: the gates pool every val family's choice rows, as
  built (`GAP-GATES-POOL-THE-GENERAL-FAMILIES-INTO-DEFECT-CONTRACTS`).
- The other open questions: `average_may_promote`, `ece_population`, `degenerate_head_floor`,
  `privileged_hunk_pass_rule`, `transfer_gate_definition`, each naming its gap.

**Who edits it, and how.** A human, and only a human. To record a ruling, set the question's
`status` to `"decided"`, set `value` (and `families`, for the population) to what was decided,
and fill `decided_by`, `decided_on` (`YYYY-MM-DD`) and `decision_ref` (where the ruling is
written: a HANDOFF section, a gap answer, a message), in a commit of its own whose message
names the gap. `load_promotion_decisions` refuses a record missing a question, a `decided`
question without all three of who/when/where, an `open` one with any of them set, and an open
`average_may_promote` that says yes. A verdict whose record cannot be read refuses: it cannot
say which population it judged. Changing a gate's code to follow a ruling is a separate change;
the record is what says the ruling exists.

### The `avg` kind: one average of seeds

`Ledger.promotion_verdict_avg(avg_row_id, input_ledgers=...)`
(CLI: `verdict --kind avg --row-id AVG --input-ledger FT_LEDGER`) judges one averaged eval row
(`real_ft_run.py --score-checkpoint AVG.safetensors`, recipe `averaged`). Condition 3 is
replaced, never dropped:

- the average's **own** gate row, joined with the supplements naming it, passes conditions
  1, 2, 4, 5, 6 and 7 exactly as one seed's unit does;
- every input ft row `recipe.averaged.ft_row_ids` names is found, completed, non-quick, and
  ended by `steps_exhausted` (its full schedule); all share one `recipe_hash`,
  `data_snapshot_hash`, `tokenizer_hash` and `backbone_commit` (the average's own snapshot
  among them); and they are at least three distinct seeds, the ones the recipe names;
- then the human: **every question still `open` in the decisions record refuses**, and so
  does a recorded no on `average_may_promote`. While any question is open, no average
  promotes, whatever its gates say
  (`python/tests/test_promotion_decisions.py::test_an_average_is_not_promotable_under_the_current_decisions`,
  with its positive control beside it).

## Chain integrity

`prev_row_hash` chains rows.

```
PYTHONPATH=python .venv/bin/python -m qd_train.ledger verify
```

recomputes the chain and reports the first break, exit 1 if broken. (This section named a
`qd-ledger verify` binary from S6 until 2026-09-19. No such binary ever existed and nothing outside
the tests had ever called `verify_chain`; the command above is the one that does.)

An edited or reordered history is detectable; it is not cryptographically prevented (a local
append-only research log, not a tamper-proof audit log — stated here so nobody claims more).

**Concurrent writers.** `append` holds an exclusive `flock` on the ledger file across
read-predecessor → duplicate-id check → write, and waits at most 60s before refusing loudly.
Without it, two processes read the same `last_line_hash()` and write two rows claiming one
predecessor — and because the log is append-only, `verify_chain` then refuses it from that line
onward *permanently*. This is not hypothetical on this repo: 6 processes × 5 rows released from a
common start time produced `line 2 (row w2-0): prev_row_hash is None but the previous line hashes
to efbdd1ff…` before the lock existed
(`python/tests/test_ledger.py::test_concurrent_appends_do_not_break_the_chain`). O_APPEND makes each
write atomic; it says nothing about the read-then-write window. The lock is advisory: it binds
writers that go through `append`, which is every writer in this repo.
