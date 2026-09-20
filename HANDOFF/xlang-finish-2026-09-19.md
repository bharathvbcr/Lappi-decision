# HANDOFF — XLANG-FINISH lane · 2026-09-19

**Owns both sides of the wire.** `crates/qd-runtime/**`, `python/qd_wire/**`, `docs/schema-api.md`,
`fixtures/wire/**`, `python/tests/test_wire_*.py`. Nothing outside those was written; `git status`
confirms it, and `python/qd_data/schema.py` — which this lane mutated during a proof — is back to
untouched with an empty `git diff`.

Two previous lanes built the machinery and then correctly **declined** to use it on three gaps,
because each fix would have made one lane stricter than the other. Owning both sides is what made
those three settleable. That is most of this lane.

---

## What was measured

Every number below is from a run, with the command. Exit codes were never taken from a pipe into
`tail`.

| Suite | Result | Command |
| --- | --- | --- |
| Rust workspace | **347 passed, 0 failed** | `cargo test --workspace` |
| — of which `qd-runtime` | **173 passed, 0 failed** (166 at `HEAD`, so **+7 are this lane's**) | `cargo test -p qd-runtime` |
| Clippy, whole workspace | **0 warnings, 0 errors** | `cargo clippy --workspace --all-targets` |
| Python, whole suite | **1081 passed, 5 skipped, 0 failed** | `.venv/bin/python -m pytest python/tests -o 'addopts='` |
| — this lane's 7 wire files | **299 passed, 0 skipped** | `pytest python/tests/test_wire_*.py` |
| Ruff | clean | `ruff check python/qd_wire python/tests/test_wire_*.py crates/qd-runtime/tests/crosslang_fixtures.py` |
| Corpus seam, both directions | 6 passed + 32 passed | `cargo test -p qd-runtime --test wire_fixtures` then `pytest python/tests/test_wire_golden_corpus.py` |

`fixtures/wire/`: **58 → 59 files** (one new refusal). Refusal kinds: **36 → 37**.
`gaps.jsonl`: **139 → 149 records**, all 149 valid JSON, appended one at a time, never rewritten.

### The baseline disagreed with the brief, and why

The brief stated 948 passed / 4 skipped at 20:18. The first run measured **980 / 5**. Cause found
before anything was changed: `python/tests/test_byte_context.py` and `test_byte_decider.py`
(mtime 20:24, after the baseline) run to **exactly 32 passed, 1 skipped** on their own. 948+32=980,
4+1=5.

Separately, `test_mutate_adapter.py` collected **0** tests during that run and **37** two minutes
later — the total collected went 980 → 1018 at 21:17:55 while another lane was writing the file.
That file is excluded from any attribution here. The Rust side had the same hazard:
`cargo test --workspace` failed to compile at 21:31 with six errors in `crates/qd-mutate/src/ops.rs`
(mtime 21:30, mid-introduction of an `Offered` type). It compiled at 21:53 and the workspace number
above is from after that. **Attribution of the +15 workspace delta: +7 this lane, +8 the qd-mutate
lane**, from `#[test]` counts at `HEAD` versus now.

---

## The NUL byte — found before the main work, and it was degrading the main work

`crates/qd-runtime/src/wire.rs` carried a **literal NUL byte at offset 3858, line 65**, in a doc
comment where the author meant to *describe* the six-character JSON escape for a NUL and typed the
character. It compiled, and all 332 Rust tests passed. Four tools were silently degraded, each
measured before and after:

| | before | after |
| --- | --- | --- |
| `devmap_status` `coverage_gaps.parse_failed` | names `wire.rs` | **empty**; graph +59 nodes, +219 edges |
| `rg -l MAX_PAYLOAD_BYTES crates/` | 6 files, **not** the one defining it | 7 files, including it |
| `file(1)` | `data` | `Unicode text, UTF-8 text` |
| `git diff --numstat` | `-  -` | still `-  -`, **and the reason is verified** |

That last row is the one worth reading carefully. Git calls the pair binary because the **HEAD-side
blob still holds the NUL**; proven rather than assumed by diffing two clean copies of the same file,
which reports `1  1`. It becomes diffable on commit.

This is almost certainly why both previous lanes reported DevMap degraded and confirmed every Rust
fact by file read: every symbol in `wire.rs` — `MAX_PAYLOAD_BYTES`, `parse_slot`, `PayloadBudget`,
`known_keys` — was absent from the graph, and DevMap answers "nothing" for a file it never parsed,
which is indistinguishable from "does not exist". After the fix, `devmap_search parse_slot` returned
the function body that became the primary evidence for the slot-name gap.

**Fixed the class:** `crates/qd-runtime/tests/tracked_source_is_text.rs` enumerates every path
`git ls-files` reports and fails if any carries a NUL, naming file, offset and line. It fails closed
— an erroring or empty `git ls-files` panics rather than passing, a file over the 8 MiB scan cap is
a *named failure* rather than a partial scan reported clean, and the run prints `scanned 243 of 243`
so a capped sample can never read as complete coverage. No allowlist: all 243 tracked files are
text and the repo tracks no binary at all.

`GAP-RT-SOURCE-NUL-BYTE-UNGATED`, closed.

---

## What was settled, and where both sides now state it

### 1. `GAP-RT-SLOT-NAME-UNCAPPED` — closed

**`schema::MAX_SLOT_NAME_BYTES = 256`**, UTF-8, before escaping, refused never truncated, via a new
refusal kind `slot_name_over_cap`.

Why 256 and not a new number: a slot name is an **identifier**, and this repo already prices an
identifier at 256 (`RenderCaps::max_task_bytes`), against 512 for option text and 4096 for question.
Measured before choosing it — **the longest slot name anywhere in the repository is 8 bytes**
(`severity`, `evidence`), so the cap is 32× the largest thing that exists and refuses nothing.

It landed on **every reader of a slot name at once**, which is the whole point:

| Locus | Where |
| --- | --- |
| request | `wire::parse_slot` — the sole constructor of `SlotSpec` in `src/` |
| prompt | `render::slot_suffix` — also serves specs built in Rust (`qd oneshot`, fixtures, tests) |
| answer map, Rust reader | `schema::deserialize_slot_map` |
| answer map, Python reader | `qd_wire.answer.check_slot_name` |

One predicate owns the rule — `schema::slot_name_fault` — with `check_slot_name` formatting the
request locus and the deserializer formatting the map locus, so the threshold exists **once**.
Python does not transcribe the number: a new `rust_source.extract_const_usize` generates
`qd_wire.contract.MAX_SLOT_NAME_BYTES` from the Rust line, re-derived and compared on every run, the
way `ESCAPE_WORST_CASE_GROWTH` and `HEX_ESCAPED` are pinned.

The over-cap refusal carries **index, cap and actual — never the name**, so a request cannot choose
the size of its own error message.

**The residual, measured not assumed.** `qd_data.schema` — the training lane's request *builder*,
owned by neither wire lane — still checks only for emptiness, so it can build a request serving
would refuse. Probed directly: it has no cap, **and the longest name it actually produces is 8
bytes**. So the divergence is theoretical, and
`wire_context_crosslang.rs::the_slot_name_cap_is_one_number_here_and_in_qd_wire_and_qd_data_produces_nothing_over_it`
fails on the day it stops being.

### 2. `GAP-XLANG-SPAN-BOUNDS-UNPINNED` — closed

**`1 <= start_line <= end_line` is a term of the format.** Enforced by `SpanValue`'s
`TryFrom<SpanValueWire>` in Rust and `qd_wire.answer.parse_slot_value` in Python.

Both bounds were already true of everything the system emits and enforced in exactly one place — the
decode path, which abstains on a backwards span and builds `start_line` as `row + 1` so 0 is
unreachable — while both *readers* accepted `{"start_line": 47, "end_line": 41}`. The 1-based floor
was included because it is the same defect in the same struct: a rule the doc comment has asserted
since the type was written and nothing enforced. Measured before tightening: every span in
`fixtures/wire/` and every test is inside both bounds, minimum `start_line` 1.

`is_ordered` is **deleted**, not left returning `True` forever. It had one caller: the test that
asserted the gap.

The two failure modes stay distinct, and `docs/schema-api.md` now says so: a *decode* that points
backwards **abstains** (a model output, `noul` is honest); a *wire value* that runs backwards is
**refused** (a malformed envelope, with no model to abstain for).

### 3. `GAP-XLANG-EXPECT-FIELD-NAME-DOC-DRIFT` — closed, structurally

`docs/schema-api.md:158` said `weights_hash`; the struct declares `weight_hash` and is
`deny_unknown_fields`, so a caller copying the documented example was refused for an unknown field
and went looking for a typo in their own code. One word fixed.

The old pin asserted the drift was *still there* so that fixing it would fail and say "delete me" —
right for an unowned gap, wrong once fixed, and it only ever guarded that one word. Replaced by
`test_the_documented_expect_block_deserializes_against_the_struct`, which parses the field names out
of the document's own `expect` example and compares them against the **generated** contract table.
Document, Python table and Rust struct are now one set of names.

### 4. `GAP-XLANG-LABEL-SET-HASH-TWO-MEANINGS` — closed, but **not** on the observation

`qd_data.schema` exports no `label_set_hash` (probe: `False`; `slot_set_digest` is `True`). The two
quantities are still unrelated and always will be — one is a property of a loaded build, the other
the identity of a request's slot list. What made it a trap was the shared *name*, and the rename
removed it.

This lane did **not** close it on that observation. `crosslang_fixtures.py` imports
`slot_set_digest`, so a rename *back* would break the probe — but re-adding `label_set_hash`
**alongside** it, which is the shape the gap actually warns about, would have been caught by
nothing. The probe now reports both, and
`wire_context_crosslang.rs::python_has_no_hash_expectation_and_its_label_set_hash_is_a_different_quantity`
asserts both.

---

## Left open, each verified by a running probe rather than by reading

| Gap | Verified how | Why it stays open |
| --- | --- | --- |
| `GAP-XLANG-REFUSAL-VOCABULARIES` | `KNOWN_VOCABULARY_DIFFERENCES` still lists the same six pairs and its test passes | Unifying means renaming in `python/qd_data/errors.py`, not this lane's. **Note: the Rust vocabulary is now one identifier wider** — `slot_name_over_cap`. No seventh pair, because `qd_data` has no such check at all. |
| `GAP-XLANG-UNKNOWN-FIELD-LENIENCY` | fed `Request.from_wire` a valid payload plus `totally_unknown_field`; it **accepted** it. No allowlist loop exists in `from_wire`. | One loop in `qd_data/schema.py` mirroring `wire.rs::known_keys`; that file is the data lane's. |
| `GAP-XLANG-SPAN-THREE-SPELLINGS` | `GoldAnswer(...).to_json()["value"]` is `[41, 47]`; no `SpanValue` anywhere in `qd_data` | Still two spellings. **Narrowed:** the runtime spelling now validates itself and the gold array does not, so the conversion function the gap calls for must carry the bounds, not only reshape fields. |
| `GAP-XLANG-NO-PY-HASH-EXPECTATION` | `to_wire()` emits no `expect`; `hasattr(qd_data.schema, "HashExpectation")` is `False` | **Narrowed:** the five names a caller would copy are now correct, so the remaining work is adding the producer, not discovering the field names. |

---

## The payload budget moved. The cap did not.

```
total = 1,826,994   (was 1,777,842)     against MAX_PAYLOAD_BYTES = 1,572,864
shortfall = 254,130 (was 204,978)
context_b64=174,766  question=24,578  task=1,538  slot_list=1,625,088  envelope=1,024
uncapped = []        (was ["slots[].name"])
```

The delta is exactly `32 × 256 × 6 = 49,152` — `MAX_SLOTS` names at the cap, worst-case JSON escaped.

The important change is not the number, it is `uncapped` being empty: **`total()` is now the worst
case rather than a lower bound on it.** The field stays so the next uncapped term lands in it instead
of silently not being counted.

**`MAX_PAYLOAD_BYTES` is unchanged.** CLAUDE.md rule 2 makes it read-only; this lane reports the
arithmetic and does not move the gate. Recorded as `GAP-RT-PAYLOAD-CAP-SHORTFALL-WIDENED` for a
human. Printed by `cargo test -p qd-runtime --test wire_refusals -- --nocapture`, never transcribed
into a document.

---

## Mutation testing: 19 mutations, **3 initial survivors**, all 19 killed

Every new test was run against unmodified code first and observed to fail. `PYTHONDONTWRITEBYTECODE=1`
throughout.

The three survivors are the useful part of this section, because each was a branch that would have
shipped unverified:

1. **`render::slot_suffix`'s check** — deleting it left the entire `-p qd-runtime` suite green. No
   test reached it, because every test arrived through `parse_slot`. Fixed by
   `render_contract.rs::a_hand_built_slot_spec_cannot_carry_an_over_cap_name_into_the_prompt`, which
   builds a `SlotSpec` directly the way `qd oneshot` does, and also asserts the reported index is the
   slot's real position rather than a constant.
2. **`AnswerEnvelope`'s `deserialize_with`** — deleting it left the suite green. Fixed by
   `answer_envelope_contract.rs::the_answer_maps_keys_obey_the_same_slot_name_cap_the_request_side_does`.
3. **A vacuous assertion of my own.** `budget.slot_list > names_term` passed with the name term
   deleted, because the options term alone is 1,575,936 against a 49,152 names term. Replaced with a
   per-slot additive check that fails if *either* term is missing.

The rest, all caught first time: NUL re-injection; cap branch removed (Rust and Python); `>` → `>=`
on the cap; bytes → `chars().count()` on the cap; Rust constant desynced from Python (256→255); cap
lowered to 4 on *both* sides to isolate the `qd_data` divergence assertion; span ordering removed,
1-based removed, and `>` → `>=`, each on both sides; the original `weights_hash` typo reintroduced;
a *different* field misspelled (`calibration_hash` → `calibrations_hash`) to prove the doc pin guards
all five rather than the one that happened first; and `label_set_hash` re-added to `qd_data` as an
alias.

The `>` → `>=` mutations matter specifically: the corpus contains a legal one-line span `{2, 2}` and
a legal at-cap name, and an off-by-one would refuse both while passing every "it refuses bad input"
test.

---

## Tooling — both worked, and one only because this lane fixed it

- **DevMap**: `is_fresh: true`, `repository.root` correct. On the first call
  `coverage_gaps.parse_failed` named `wire.rs`; after the NUL fix it is empty. Graph answers for
  `qd-runtime` should now be trustworthy for the first time this session.
- **GitPulse**: **every facet returned `ok: true`** — worktrees, changes, collisions, agents, ledger,
  codeintel — where both previous lanes reported `REPOSITORY_TRUST_REQUIRED` on all of them.
  1 worktree, 0 overlapping files, 0 registered agent sessions.
- **One limitation worth not rediscovering:** `gitpulse_insights` scans *worktrees*, and the
  concurrent lanes in this session share **one** worktree. It reported 0 overlapping files while
  three other lanes were writing `python/qd_train/` and `crates/qd-mutate/` in the same tree.
  Same-tree concurrent edits are invisible to it. Mtime comparison is what actually caught them.
- `codeintel.is_fresh: false` is unrelated to source staleness: the binary was built without the
  parse feature so analyzer freshness cannot be decided. `source_freshness: true`.

Recorded as `GAP-XLANG-FINISH-TOOLING-STATE`.

---

## Not done, and deliberately

- **`MAX_PAYLOAD_BYTES`** — rule 2. Reported, not touched.
- **The three gaps owned by `qd_data`** — vocabularies, unknown-field leniency, the hash-expectation
  producer. All three fixes are small and all three are in another lane's module.
- **`qd serve`** — still only `qd oneshot` is exercised end to end. Unchanged from the previous
  lane's note and still **inferred from a file read, not measured**.
- **A non-reference backend** — none exists in this build, so every answer parsed still came from
  `reference-deterministic-v1`.

---

## The exact first command for the next lane

```bash
cargo test --workspace && \
  .venv/bin/python -m pytest python/tests -o 'addopts=' -q
```

Expect **347 passed / 0 failed** and **1081 passed, 5 skipped / 0 failed**. If the Python count is
higher, check `python/tests/` mtimes before investigating — three lanes were writing that tree
during this one, and the count moved twice mid-run.

If you touch the envelope in either direction:

```bash
cargo test -p qd-runtime --test wire_fixtures && \
  .venv/bin/python -m pytest python/tests/test_wire_golden_corpus.py -rs -q
```

Regenerate with `QD_WRITE_FIXTURES=1` **only** when the envelope change was intended, and say so in
`docs/schema-api.md` in the same change.

After any change to a constant or a struct field in `crates/qd-runtime/src/{schema,refusal}.rs`:

```bash
PYTHONPATH=python .venv/bin/python -m qd_wire.rust_source
```

and update `python/qd_wire/contract.py` to match — `test_wire_contract_matches_rust.py` re-derives
and compares on every run, and now covers scalar constants as well as field tables, so the
slot-name cap cannot drift between the lanes without a Python test failing and naming both numbers.
