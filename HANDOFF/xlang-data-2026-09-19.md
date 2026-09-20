# HANDOFF — XLANG-DATA lane · 2026-09-19

**Read this first.** Three of the four gaps are closed. Closing two of them fires two
assertions in `crates/qd-runtime/tests/wire_context_crosslang.rs` that were written to
pin those gaps **open**. `crates/` is not this lane's, so they are left failing and the
six replacement lines are written out below. **`cargo test --workspace` is 345 passed /
2 failed until someone applies them.** Everything else in both suites is green.

That is not an accident of this lane. It is the finding five lanes missed: **each of the
four gaps was pinned open from the Rust side, so it cannot be closed from
`python/qd_data` alone.** Recorded as `GAP-XLANG-DATA-RUST-PINS-ASSERT-THE-GAP-OPEN`.

---

## The two Rust lines this lane may not write

Both in `crates/qd-runtime/tests/wire_context_crosslang.rs`. Both assertions carry
messages written for exactly this moment.

### 1. `:805`, in `both_lanes_refuse_the_same_payloads_and_name_the_context_checks_identically`

```
left: []   right: ["unknown top-level field"]
"the set of payloads Python accepts and Rust refuses changed"
```

Replace the comment at `:801-804` and the assertion at `:805-810` with:

```rust
    // Python refuses unknown top-level fields too, as of `GAP-XLANG-UNKNOWN-FIELD-LENIENCY`
    // closing: `qd_data.schema.Request.from_wire` checks `raw.keys()` against
    // `WIRE_REQUEST_KEYS`, which `python/tests/test_qd_data_wire_agreement.py` re-derives
    // from `REQUEST_KEYS` in `src/wire.rs` on every run. This list must stay empty: an
    // entry is a way for a caller to satisfy one lane and not the other.
    assert!(
        python_accepted.is_empty(),
        "Python accepts payloads this build refuses: {python_accepted:?}"
    );
```

`mapped.len() == KNOWN_VOCABULARY_DIFFERENCES.len()` is **unaffected** — the new refusal
uses `malformed_request`, the identifier Rust already uses, so the pair lands in the
equal-arm and the count stays 6. Verified: that assertion did not fire.

### 2. `:566-570`, in `python_has_no_hash_expectation_and_its_label_set_hash_is_a_different_quantity`

```
left: Some(Bool(true))   right: Some(Bool(false))
"Python grew a HashExpectation: cross-assert it instead of skipping it"
```

Flip the expected boolean and rename the test:

```rust
fn python_has_a_hash_expectation_and_its_label_set_hash_is_a_different_quantity() {
    …
    assert_eq!(
        inv.get("answer_side_types").and_then(|a| a.get("HashExpectation")),
        Some(&Value::Bool(true)),
        "qd_data lost its HashExpectation: the training lane can no longer pin a build"
    );
```

The second assertion in that test — `request_to_wire_emits_expect == false` — **stays as
it is and still passes**, deliberately. The probe builds a request that pins nothing, and
a request that pins nothing emits no `expect`. Leave it.

The cross-assertion the message asks for already exists, on the Python side and stronger
than a boolean: `test_the_five_pins_are_the_rust_structs_fields_and_not_a_retyped_list`
compares `dataclasses.fields(HashExpectation)` against
`qd_wire.contract.STRUCT_FIELDS["HashExpectation"]`, itself re-derived from `schema.rs`
every run. Extending the probe in `crosslang_fixtures.py` to emit a filled `expect` block
would be better still, but that file is under `crates/` too.

---

## What was measured

| Suite | Result | Command |
| --- | --- | --- |
| Python, whole suite | **1209 passed, 7 skipped, 0 failed** (it was 1206 twenty minutes earlier; see below) | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest python/tests -o 'addopts='` |
| — this lane's file | **54 passed, 0 skipped** | `pytest python/tests/test_qd_data_wire_agreement.py` |
| Rust workspace | **345 passed, 2 failed** (was 347 / 0) | `cargo test --workspace --no-fail-fast`, tallied over 23 binaries |
| — the 2 failures | both in `wire_context_crosslang`, both above | `cargo test -p qd-runtime --test wire_context_crosslang` |
| Ruff, the files this lane wrote | **clean** | `ruff check python/qd_data/rows.py python/tests/test_qd_data_wire_agreement.py` |
| Ruff, `python/qd_data` as a whole | 7 errors, **all pre-existing** | see below |
| Mutation | **25 mutations, 0 survivors** (1 survivor on the first pass, fixed) | `scratchpad/mutate.py` |

**The ruff finding is not this lane's, and that is verified rather than asserted.** All 7
are `RUF022`/`RUF023` (`__all__` and `__slots__` not sorted), in `config.py`,
`licences.py`, `render.py` ×2, `sources.py`, `errors.py`, `schema.py`. Four of those files
this lane never opened. For the two it did, `git show HEAD:…` copies were linted in
isolation and produced **the same 2 errors**, so `python/qd_data` was never ruff-clean
under this config — the previous handoffs' "ruff clean" covered `python/qd_wire` and
`test_wire_*.py` only. Left alone: sorting seven unrelated `__all__` lists is the
unrelated drive-by `CLAUDE.md` forbids.

### The baseline disagreed with the brief, and the tree moved under the lane

The brief stated 1094 passed / 7 skipped at 22:10. The first run measured **1104 / 7**,
and the final run collects **1216**. Not investigated away — attributed: `test_byte_train.py`
and `test_run_control.py` (mtime 22:17:38), `test_byte_batch.py` (22:08), `test_shards.py`
(22:35, 22:39, 22:44) and `test_mutate_adapter.py` all moved during this lane. `gaps.jsonl`
was appended by other lanes at 22:34 and 22:43. Of the final total, **54 are this lane's**,
by direct count of the one file it added. No other delta is claimed.

---

## What was settled, and where both sides now state it

### 1. `GAP-XLANG-UNKNOWN-FIELD-LENIENCY` — closed on this side

`Request.from_wire` now refuses any top-level key outside `WIRE_REQUEST_KEYS`, with a new
`MalformedRequestRefusal` whose `check` is **`malformed_request`** — the identifier Rust
already uses, so closing this gap did not widen the vocabulary one.

Two details are load-bearing:

* **Order.** The check sits *after* the retired-`context` check, mirroring
  `wire.rs::validate`. `context` is not on the allowlist, so the other order would answer
  `malformed_request` to a caller still sending the retired form instead of naming what
  replaced it. Pinned by
  `test_the_retired_context_field_is_still_named_before_the_unknown_field_check`.
* **The allowlist is not a transcription.**
  `test_the_allowlist_is_read_out_of_the_rust_and_is_not_a_second_copy` parses
  `const REQUEST_KEYS: &[&str]` out of `crates/qd-runtime/src/wire.rs` at test time and
  compares. The extractor **raises** if it cannot find the declaration or parses fewer
  than five keys, because an extractor returning `()` would make the comparison pass
  against nothing. Two of this session's defects were a transcribed constant.

Pre-fix failure observed: `from_wire` accepted `{…, "totally_unknown_field": 1}`.

### 2. `GAP-XLANG-NO-PY-HASH-EXPECTATION` — closed on this side

`qd_data.schema.HashExpectation` carries the five optional pins; `Request.expect` defaults
to an empty one; `to_wire()` emits `expect` **only when something is pinned**;
`from_wire()` parses it back, refusing an unknown pin (`deny_unknown_fields`) and a
non-string pin (`Option<String>`).

*Empty emits nothing* is the detail worth keeping. `HashExpectation::default()` is what
the runtime substitutes for an absent `expect`, so an empty object or five nulls would be
a second spelling of one state — the defect shape this repo has shipped three times. It is
also why `test_wire_gap_pins.py::test_a_training_lane_request_still_pins_no_hashes`, a
pin belonging to another lane, **still passes** rather than needing an edit.

The class is named `HashExpectation` on purpose, knowing the probe searches for that name.
Calling it something else to dodge the probe would be one concept under two names, which
is `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS` exactly.

Also asserted: a request that pins hashes **renders byte-identical prompts** to one that
does not. The pins are a claim about the build, not part of what the model sees.

### 3. `GAP-XLANG-SPAN-THREE-SPELLINGS` — closed, at **zero** cost to the Rust suite

The count of spellings was never the defect, and reducing it would have been
over-unification: a `GoldAnswer` is a training label in a JSONL row and a
`SlotValue::Span` is a runtime answer on the wire. **`GoldAnswer.value` is still
`[41, 47]`**, and `the_three_span_spellings_stay_distinct_across_the_two_lanes` still
passes.

What was actually wrong: `GAP-XLANG-SPAN-BOUNDS-UNPINNED` made `1 <= start_line <=
end_line` a term of the format on both sides of the wire and left the gold spelling
checking **nothing** — no arity, no ordering, no floor, and `bool` passing as `int`
because `bool` subclasses `int`. `GoldAnswer(value=(47, 41))` was constructible and
survived dedupe, splitting and the shard writer.

Now `python/qd_data/rows.py` has one predicate, `_checked_span`, called by
`GoldAnswer.__post_init__` and by the named pair `gold_span_to_wire` /
`wire_span_to_gold` that the gap's `action_required` asked for — so the conversion
carries the bounds rather than only reshaping fields.

**The agreement is asserted by execution, not by a second copy of the rule.** The test
drives this module's output into `qd_wire.answer.parse_slot_value` and checks that what
one refuses the other refuses. The one honest difference is stated rather than omitted:
`41` is a legal `SlotValue` (a score bin) so the parser reads it, while the converter
refuses it as a span — a converter that coerced it would turn a score into the line range
`41..41`.

### 4. `GAP-XLANG-REFUSAL-VOCABULARIES` — **open**, and reframed

Not closed, and the reason is the finding. **Four of the six "pairs" are not renames.**
The Python classes are *coarser* than the Rust kinds, so renaming would attach a failure's
name to a class that also fires for other failures. Read off the raise sites:

| Python `check` | runtime kinds it actually stands for |
| --- | --- |
| `options_len` | `too_many_options`, **`too_few_options`** (zero options raises it too) |
| `slots_non_empty` | `empty_slots`, **`too_many_slots`**, **`malformed_request`** |
| `bins_range` | `bins_out_of_range`, **`slot_field_missing`**, **`malformed_request`** |
| `slot_type_known` | `unknown_slot_type`, **`unknown_route`**, **`empty_slot_name`**, **`slot_field_missing`**, **`malformed_request`** |
| `schema_version` | `unknown_schema_version` — **one to one** |
| `slot_names_unique` | `duplicate_slot_name` — **one to one** |

Only the last two are renameable as they stand, and
`test_which_of_the_pinned_pairs_are_renames_and_which_are_conflations` asserts exactly
that set so the claim cannot go stale. Renaming even those two touches three files this
lane does not own (`wire_context_crosslang.rs`, `qd_wire/refusal.py`, `docs/schema-api.md`
lines 89-90 and 311), so it is reported.

**And there is a seventh divergence the table never listed.** `ContextTooLargeRefusal`
(`context_bytes`) stands for `context_over_cap`, `context_not_utf8`, `question_over_cap`
**and** `task_over_cap`. It is missing from both copies of
`KNOWN_VOCABULARY_DIFFERENCES` because the cross-language negative set sends no over-cap
payload — the pin meant to make the divergence visible could not observe this one at all.

What did land: the mapping is **explicit and executable** instead of a comment. Every
`QdRefusal` subclass declares `rust_kinds`, and five tests hold it closed:

1. every subclass must **declare** it, not inherit it — an inherited default is
   indistinguishable from "checked, and there is none";
2. no declared kind may be one `refusal.rs` does not have, checked against the
   **generated** table;
3. the classes declaring `()` are exactly the five data-lane ones;
4. every runtime kind is either mapped or in a pinned `NO_QD_DATA_COUNTERPART` set of 10,
   so a kind added to `refusal.rs` lands in neither and fails;
5. each of the six pinned pairs must be explained by the class that produces it.

That is the gap's own instruction — *"do not add a refusal to one lane without checking
whether the other already names the same condition differently"* — enforced at the moment
the refusal is written.

---

## Mutation testing: 25 mutations, 1 survivor on the first pass, 0 after

`PYTHONDONTWRITEBYTECODE=1` throughout, one mutation at a time, file restored and
byte-compared after each. A mutation whose target text is not present **exactly once** is
a harness error, not a survivor.

**The survivor is the useful part.** `M15`, loosening the span arity check from `len != 2`
to `len < 2`, was reported killed — and was not. A three-element span still raised
`ValueError`, but from **tuple unpacking**, with Python's *"too many values to unpack"*
rather than anything naming the span. The test asserted the exception type only, so a
refusal firing for the wrong reason read the same as one firing for the right reason. Both
arity cases now assert the message, and `M15` plus a new `M15b` (drop the arity check
entirely) are killed.

The other 23, all caught first time: the unknown-key loop deleted; a key added to the
allowlist that `wire.rs` lacks; `expect` dropped from the allowlist; the refusal
identifier drifted off `malformed_request`; `weight_hash` respelled `weights_hash`;
`to_wire` always emitting `expect`; an unknown pin ignored; a non-string pin coerced;
unset pins emitted as null; `from_wire` dropping the block; `GoldAnswer` not checking its
span; the 1-based floor moved to 0; `>` → `>=` on the ordering; the `bool` exclusion
removed; start and end swapped in the converter; a converter field renamed; the reverse
converter's key-set check dropped; the seventh divergence losing a kind; a class claiming
a kind Rust lacks; a refusal not declaring its kinds; a conflation claimed one-to-one; a
data-lane refusal claiming a runtime kind; **and the Rust extractor blinded**, which
proves the allowlist comparison fails closed rather than passing against nothing.

The `>` → `>=` mutation matters specifically: `(1, 1)` is a legal one-line span, and an
off-by-one would refuse it while passing every "it refuses bad input" test.

---

## Tooling — what answered and what could not

- **DevMap**: `is_fresh: true`, generation 616, 3451 nodes / 11243 edges,
  `repository.root` correct, and **every** `coverage_gaps` bucket empty including
  `parse_failed` — the NUL-byte degradation the XLANG-FINISH lane fixed stayed fixed.
  `devmap_explore` on `from_wire` gave the one caller that mattered before tightening it
  (`qd_data.render.render_for_serving`). Its `walk_incomplete` was read, not ignored: the
  depth-3 radius is a lower bound, so `to_wire`'s callers were confirmed by ripgrep as
  well and the two agreed.
- **GitPulse Insights**: every facet `ok: true` — worktrees, changes, collisions, agents,
  ledger, codeintel. **And it could not answer the only question that mattered.**
  `collisions` scanned 2 worktrees and reported 0 overlapping files while at least two
  other lanes were writing `python/qd_train` and `python/tests` in *this* worktree;
  `agents` reported 1 session. Both facets are per-worktree, and same-tree concurrent
  edits are invisible to them.
- **Mtimes were the only working collision signal**, exactly as the brief said. They
  caught `test_shards.py` moving three times mid-run, `gaps.jsonl` appended by others
  twice, and the collected count moving 1104 → 1213.

Recorded as `GAP-XLANG-DATA-TOOLING-STATE`.

---

## Found in passing, not this lane's, and already fixed by its owner

For about twenty minutes the Python suite was **1196 passed / 1 failed**:
`test_wire_gap_pins.py::test_every_gap_id_this_lane_cites_exists_in_the_ledger`, on a
truncated `GAP-RT-WIRE-CONTEXT-` cited in `python/tests/test_shards.py`. Cause: that file
wraps `GAP-RT-WIRE-CONTEXT-ENCODING` across a line in a docstring, and the two
gap-citation guards applied **different rules** to it — `test_gaps_ledger.py` documents
and tolerates a trailing-hyphen prefix, `test_wire_gap_pins.py` carried its own regex that
could not match one. Two implementations of one rule, nothing asserting they agree.

(Writing this paragraph failed the guard once, for spelling the truncated token without
its trailing hyphen. Which is the guard working: a whole token must resolve exactly, and a
mistyped id that happens to be a prefix of a real record is precisely what it is for.)

Recorded as `GAP-XLANG-DATA-CITATION-GUARDS-DISAGREE`, then **closed in the same ledger**
under the same id: another lane fixed it at 22:43:22 by making `test_wire_gap_pins.py:50`
import `ID_RE_TEXT` from `test_gaps_ledger` — one regex, one owner. Verified by running,
not by reading the diff.

---

## Not done, and deliberately

- **`crates/**` and `docs/schema-api.md`** — not touched. The two required edits are at the
  top of this file.
- **Renaming the two one-to-one refusal checks** — it is correct and it is three unowned
  files. Reported with the exact pairs.
- **Splitting the four conflated refusal classes** — the real fix for
  `GAP-XLANG-REFUSAL-VOCABULARIES`, and a change large enough to be the coordinator's call.
- **`qd serve`** — still only `qd oneshot` is exercised. Unchanged, and still inferred from
  a file read rather than measured.

---

## Files changed

| Path | What |
| --- | --- |
| `python/qd_data/schema.py` | `WIRE_REQUEST_KEYS`; the unknown-key refusal in `from_wire`; `HashExpectation`; `Request.expect`; `to_wire`/`from_wire` carry it |
| `python/qd_data/errors.py` | `MalformedRequestRefusal`; `rust_kinds` on `QdRefusal` and every subclass |
| `python/qd_data/rows.py` | `_checked_span`; `gold_span_to_wire`; `wire_span_to_gold`; `GoldAnswer` validates a span |
| `python/qd_data/loaders.py` | `rust_kinds = ()` on the two data-lane refusals |
| `python/qd_data/mixture.py` | `rust_kinds = ()` on `RowRefused` |
| `python/tests/test_qd_data_wire_agreement.py` | **new**, 54 tests |
| `gaps.jsonl` | 7 records appended, one `printf` each, never rewritten |

Nothing outside `python/qd_data/`, `python/tests/` and `gaps.jsonl` was written.

**Size: +383 / -7 in `python/qd_data/`** (`git diff --numstat`), plus 540 lines of new
tests. The lane is net larger, and honestly so: most of the +383 is the reasoning attached
to each new invariant, and the only deletions are lines that were replaced. Nothing here
was dead code to remove — the defects were all *missing* checks, not surplus ones.

---

## The exact first command for the next lane

```bash
cargo test --workspace --no-fail-fast 2>&1 | grep -E '^test result:'
```

Expect **2 failed**, both in `wire_context_crosslang`, until the six lines at the top of
this file are applied. Then:

```bash
cargo test -p qd-runtime --test wire_context_crosslang && \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest python/tests -o 'addopts=' -q
```

Expect **16 passed** and **1209 passed / 7 skipped / 0 failed** — check `python/tests/`
mtimes before investigating a higher Python count, because at least two lanes were writing
that tree throughout this one.

After any change to `REQUEST_KEYS` in `crates/qd-runtime/src/wire.rs`, or to
`HashExpectation` or `SpanValue` in `schema.rs`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest python/tests/test_qd_data_wire_agreement.py -q
```

It re-derives all three from the Rust and names the one that drifted. `PYTHONDONTWRITEBYTECODE=1`
is not decoration: a lane got three false mutation survivors from `.pyc` caching when two
mutations changed a file by the same byte count in one second.
