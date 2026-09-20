# TRAINER-SEMANTICS — three undefined seams, settled in the contract

**Lane:** TRAINER-SEMANTICS · **Date:** 2026-09-19 · **Owns:** `python/qd_train/{trainer,heads,artifacts}.py`
and their three test files. Nothing under `crates/`, `docs/` or any other lane's file was edited.

## What this lane was for

Three gaps, all of the same shape: **a rule that two components read two ways, where both
readings produce a falling loss curve.** That is the failure mode `artifacts.py` exists to
prevent, and all three of these were inside it.

| Gap | The disagreement | Settled as |
| --- | --- | --- |
| `GAP-TRAINER-FT-SLOT-LM-ROW-UNDEFINED` | `Batch` permitted `SLOT_LM` in a batch also carrying `target_index`; `ft_supervision` refused it | **Refused at construction.** Neither reading. |
| `GAP-TRAINER-CPT-REFUSES-LENGTH-ONE-ROWS` | `Batch` permitted a one-token row; `cpt_supervision` refused it | **Refused at construction**, agreeing with the writer's existing floor |
| `GAP-RT-POINTER-HEAD-PAD-SHAPE-UNRECORDED` | Rust demands exactly `line_count + 1` pointer rows; whether training pads was unverified in *both* directions | **It pads** — to the batch max. Now stated, asserted, and sliceable |

## What each settlement says, and where the code says it

### 1. `SLOT_LM` cannot appear in `Batch.slot_kind`

`python/qd_train/artifacts.py:586` (the check), `:413` (the constant and its docstring),
`:489-501` (the reasoning, in `Batch`'s own docstring).

The gap named two readings — an LM row deliberately mixed into fine-tuning, versus a
mislabelled choice/score row — and asked which an FT loop applies. **Neither.** The
deciding evidence is inside the contract rather than outside it: `span_target` gained the
`NO_SPAN` sentinel the moment a row needed to say "not me", and `target_index` has no
equivalent. A contract that would have needed a do-not-care sentinel and never grew one is
a contract that never meant to express the row.

Verified that nothing in the repo ever writes one: `qd_train.shards.slot_kind_of`
(`shards.py:231-244`) returns only `SLOT_CHOICE`, `SLOT_SCORE` or `SLOT_SPAN` and raises
`UnencodableGold` otherwise; `rg -uu` over the whole tree (including gitignored and
generated files) finds `SLOT_LM` only in `artifacts.py`, `trainer.py` and their tests.

`SLOT_LM` deliberately **stays** in `_SLOT_KINDS`. Dropping it would refuse the same
batches with `unknown slot kind(s) [0]`, which sends the writer looking for a typo instead
of for the mislabelled row. `test_slot_lm_stays_a_known_kind_so_its_refusal_can_explain_itself`
pins the arrangement, because the two arrangements differ only in what the writer is told.

### 2. A `Batch` row carries at least two real tokens

`python/qd_train/artifacts.py:425` (`_MIN_ROW_TOKENS`), `:540` (the check),
`:503-513` (the reasoning).

The gap offered "make `Batch` refuse it too" or "document why both are right". The
evidence chose the first: **`qd_train.shards._tokenize_checked` has refused `ids.size < 2`
since S4 landed** (`shards.py:488-497`), with the same reasoning and the same number, and
its comment already names the trainer. So writer and consumer always agreed; `Batch` was
the only component out of step.

Also worth recording: the disagreement only ever existed on the **CPT** path. FT already
refused a one-token row as a side effect of the `target_index` range check — `[0, lengths-1)`
is empty at `lengths == 1`.

Consequence, stated rather than hidden: `trainer._refuse_unsupervised_rows` and
`trainer._prediction_grid`'s `width < 2` guard are now **unreachable through the
constructor** and carry `# pragma: no cover` with a reason, matching the three
`Batch refuses this` guards already in `ft_supervision`. They stay as postconditions on the
mask arithmetic; the contract statement moved upstream, and `test_artifacts.py` is where it
is now tested.

### 3. The pointer head pads to the batch maximum, and that is now checked

`python/qd_train/heads.py:40-64` (the section), `SpanPlan.runtime_rows`,
`_check_runtime_rows`, `serving_scores`.

**Answer to the xlang-rs lane's question: yes, the training side pads — but to *this
batch's* widest candidate set, not to a global maximum.** A ragged score matrix is not a
tensor. Measured on a counts `[3, 1]` batch: the one-line row is scored in a 4-wide matrix
while `qd-runtime` will demand 2. Handed over as-is, `backend::validate_logits`
(`crates/qd-runtime/src/backend.rs:240`, checks at `:250` and `:258`) refuses it twice —
`logit_shape_mismatch` against `query.rows`, and `non_finite_logit` for the `-inf` fill.

The padding is nevertheless **safe**, because serving is one context at a time: `K == 1`,
the batch maximum *is* that context's line count, and there is nothing to strip. Three
things make that a checked claim instead of an argument:

* `SpanPlan.runtime_rows` — the per-row number `answer.rs:193` puts in `query.rows`, pinned
  elementwise to `span_head_rows`, which is itself pinned to the Rust formula.
* `SpanPointerHead.forward` — a postcondition, on **both** pointers, that each row's finite
  columns are exactly `0 .. n_candidates[k]`. Equality in both directions: a padded column
  that became selectable *and* a dead or `NaN` column inside the range both fail the batch.
* `serving_scores(scores, plan)` — exactly `runtime_rows[k]` finite values per row: the
  slice a serve-time comparison would use.

A serve-time `logit_shape_mismatch` is now a train-time `ValueError`. The gap said "loud is
correct but late"; loud was already guaranteed by the Rust, early was the missing half.

**Not** written into `docs/training-contract.md`, which the gap's `action_required` asked
for: that file is outside this lane's ownership, and it states by design that it "deliberately
does not restate the formats" because the code is the thing that runs. The coordinator should
decide whether it wants a line there anyway.

## What was measured

No training ran, so **there are no ledger rows from this lane** and no number below is a
model number. Every number is a test count from the command beside it, run from the repo root.

| Suite | Command | Result |
| --- | --- | --- |
| Pre-change baseline, repo venv | `.venv/bin/python -m pytest python/tests -o 'addopts='` | **980 passed, 5 skipped** at 21:0x |
| Pre-change baseline, torch venv | `uv run --python ~/.venvs/ml/bin/python --with pytest --no-project python -m pytest python/tests/test_heads.py -o 'addopts='` | **16 passed** |
| This lane's three files, repo venv | `.venv/bin/python -m pytest python/tests/test_artifacts.py python/tests/test_trainer.py python/tests/test_heads.py -o 'addopts='` | **96 passed, 1 skipped** (`test_heads.py` `importorskip`s torch) |
| This lane's three files, torch venv | `uv run --python ~/.venvs/ml/bin/python --with pytest --no-project python -m pytest python/tests/test_artifacts.py python/tests/test_trainer.py python/tests/test_heads.py -o 'addopts='` | **116 passed** |
| Whole Python suite, 21:37 | `.venv/bin/python -m pytest python/tests -o 'addopts=' --ignore=python/tests/test_byte_batch.py` | **1059 passed, 5 skipped, 0 failed** |
| Whole Python suite, 21:40 | *same command* | **1046 passed, 18 skipped, 0 failed** |

**The whole-suite total is a moving target and should not be read as a number this lane
produced.** Three minutes apart, the same command gave 1059/5 and then 1046/18 — other lanes
were still landing files. Zero failures in both. This lane's own three files were re-measured
immediately after the second run and were unchanged at **96 passed, 1 skipped**, so the drift
is entirely outside the lane. The stable figures to hold this lane to are the two rows above
it: 96/1 and 116/0.

The coordinator's 20:18 baseline was 948 passed / 4 skipped. The 980/5 first run reconciles
exactly: `test_byte_context.py` + `test_byte_decider.py`, both untracked and written by a
concurrent session at 20:24, contribute **32 passed, 1 skipped** measured on their own. The
climb from there to 1059 is more concurrent lanes landing files during this session, not this
lane. **This lane added 8 tests net** — 10 new `def test_` minus 2 renamed-away — measured as
`test_heads.py` 16 → 20 and `test_artifacts.py` + `test_trainer.py` 92 → 96.

**`cargo test --workspace` is NOT reported as passing, because it did not run to completion.**
At 21:32 it failed to *compile*: `crates/qd-runtime/tests/wire_refusals.rs` referenced
`MAX_SLOT_NAME_BYTES`, which was not yet in scope (`error[E0425]`, 10 occurrences). That file
was written at 21:32:24, while the build was in flight. This lane edited no Rust — `git diff
--name-only` over its six files is entirely under `python/` — so the breakage is the wire
lane's in-flight edit and is reported as **not run**, not as failed-by-this-lane and not as green.

## Every fix ships with a test that failed against the pre-fix code

Pre-fix behaviour was captured by executing the unmodified code, not by reading it:

```
GAP 1: Batch CONSTRUCTED an FT batch holding a SLOT_LM row: kinds=[0, 1]
       ft_supervision then refuses it: TrainerContractViolation
GAP 2: Batch CONSTRUCTED a CPT batch holding a length-1 row: lengths=[3, 1]
       cpt_supervision then refuses it: TrainerContractViolation
GAP 3: n_candidates = [3, 1]; runtime demands [4, 2]; SpanPlan.max_rows = 4
       hasattr(plan, 'runtime_rows') = False;  'serving_scores' in heads = False
       head output shape = (2, 4)   <- PADDED to the batch max, not per row
       a plan whose n_candidates disagrees with its mask:
         row 1 finite columns = [0, 2]   <- a HOLE at column 1, and no error raised
```

Adding the two contract checks then broke exactly three pre-existing tests, all in
`test_trainer.py` and all because the refusal moved one layer earlier. Each was rewritten to
assert the refusal where it is now reachable, and each keeps a positive half so it cannot pass
by raising for the wrong reason.

## The mutation that proves each test bites

Ten mutations, applied one at a time by a harness that restores the file on every exit path
and asserts the restore. **All ten killed.** The harness was deleted; `rg MUTANT` over
`python/` is clean.

| # | Mutation | Test(s) that caught it |
| --- | --- | --- |
| M1 | `runtime_rows` drops `+ RESERVED_NOUL_ROWS` | `test_the_plan_states_the_row_count_the_runtime_will_demand` (`3 != 4`), + 10 more |
| M2 | `serving_scores` returns the unsliced padded matrix | `test_serving_scores_strips_the_padding...` (`At index 1 diff: 4 != 2`) |
| M3 | delete the `forward` postcondition call | `test_the_head_refuses_a_row_whose_selectable_set_is_not_the_runtimes` (DID NOT RAISE) |
| M4 | `isfinite != served` → `isfinite & ~served` (catch only extra columns) | same test — **this is the one that proves the check has to be two-directional**; the lying plan's failure is a *hole inside* the served range |
| M5 | point the Rust pin at a string absent from `backend.rs` | `test_serving_scores_pins_itself_to_what_validate_logits_actually_checks` |
| M6 | length floor made unreachable (`< 1`, already guaranteed) | `test_a_one_token_row_is_refused...` + 2 trainer tests |
| M7 | `_MIN_ROW_TOKENS = 1` | the above + `test_the_one_token_floor_agrees_with_what_the_writer_already_refuses` |
| M8 | delete the `SLOT_LM` refusal | `test_an_lm_row_cannot_sit_in_the_slot_channel` + the trainer-side one |
| M9 | drop `SLOT_LM` from `_SLOT_KINDS` | the above + `test_slot_lm_stays_a_known_kind...` |
| M10 | abstain row scattered first instead of last | 3 pre-existing tests — a control, confirming the new work did not blunt them |

Two deliberate negatives, because a mutation that kills everything proves nothing about any
one test: **M2 leaves `test_the_plan_states_the_row_count...` passing** (different concern,
own mutation), and the ragged fixture uses counts `[3, 1]` rather than `[2, 1]` — with `[2, 1]`
the narrow row's only padded column is the one the abstention is scattered into, so a deleted
mask is invisible. That vacuity was caught in this file once before; the new `_ragged_plan`
helper carries the reason in its docstring.

## What is open

| Gap id | State |
| --- | --- |
| `GAP-TRAINER-FT-SLOT-LM-ROW-SETTLED` | resolved — supersedes `...-UNDEFINED` |
| `GAP-TRAINER-BATCH-MIN-ROW-TOKENS-SETTLED` | resolved — supersedes `...-CPT-REFUSES-LENGTH-ONE-ROWS` |
| `GAP-RT-POINTER-HEAD-PADS-TO-BATCH-MAX-ANSWERED` | resolved — supersedes `...-PAD-SHAPE-UNRECORDED`. **One residual for the coordinator:** whether `docs/training-contract.md` should carry the sentence, which this lane may not write |
| `GAP-TRAINER-HEADS-IS-A-TORCH-HARD-DOOR-TO-A-TORCH-FREE-CONSTANT` | opened, then **superseded within ~90 seconds** by `GAP-TRAINER-HEADS-TORCH-DOOR-FIXED-BY-ANOTHER-LANE` |
| `GAP-TRAINER-GITPULSE-AVAILABLE-CONTRARY-TO-BRIEF` | open — a brief correction, below |

**This lane appended 6 records**, each as a single atomic `printf '%s\n' '<json>' >> gaps.jsonl`,
never a read-and-rewrite. The file went 123 (at `HEAD`) to 132, so **three of the nine new
records are other lanes'** — concurrent appends, which is exactly what the append-only rule is
for. The whole file was re-validated afterwards as one parseable JSON object per line: 132
lines, 0 malformed.

### Tooling, against what the brief said

**The brief was wrong about both tools, and this lane checked rather than believing it.**

* **GitPulse works.** `gitpulse_insights` returned **every facet `ok: true`** — worktrees,
  changes, collisions, agents, ledger `recording: true`. No `REPOSITORY_TRUST_REQUIRED`
  anywhere. Each `ok` was read individually, not inferred from the envelope.
  *But:* it counted **1 worktree and 0 agent sessions** while three sessions were
  demonstrably editing this same tree. Its collision facet is worktree-level and cannot see
  intra-worktree contention. File mtimes were the only signal that worked for that, and this
  lane used them three times.
* **DevMap is fresh.** `devmap_status` reported `is_fresh: true`, generation 365 (the brief
  said false). One coverage gap, already recorded as `GAP-RT-SOURCE-NUL-BYTE-UNGATED`:
  `crates/qd-runtime/src/wire.rs` is refused before parsing for a NUL byte at offset 3858.
  No result in this lane was reported as graph-confirmed that came from `rg`.

### Concurrent-lane state observed at 21:37, **not this lane's to fix**

`python/tests/test_byte_batch.py` aborts the whole repo-venv pytest run at collection:
`ImportError: attempted relative import with no known parent package` at line 11
(`from .test_mutate_adapter import row`). It is untracked, written at 21:31, and was still
being edited during this lane. The whole-suite figure above excludes it and says so.

## The exact first command for the next lane

```
.venv/bin/python -m pytest python/tests -o 'addopts=' -q
```

Expect it to **abort at collection** until the owner of `test_byte_batch.py` fixes the
relative import above; add `--ignore=python/tests/test_byte_batch.py` to get a number. Do not
expect 1059 or 1046 — both were true within three minutes of each other and neither is this
lane's. What this lane guarantees is `python/tests/test_artifacts.py python/tests/test_trainer.py
python/tests/test_heads.py` → **96 passed, 1 skipped**.

Then, because three of this lane's tests cannot run in the repo venv:

```
uv run --python /Users/bharath/.venvs/ml/bin/python --with pytest --no-project \
    python -m pytest python/tests/test_heads.py -o 'addopts=' -q
```

Expect **20 passed**. A `skipped` here is not a pass — it means torch was not found and the
four pointer-head/runtime agreement tests did not run.
