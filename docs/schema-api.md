# The typed request: one interface, two routes

This is the contract shared by `qd-runtime`, the training prompt format in `python/qd_data`, and every
calling app. **The training format and the serving format are the same object.** If they drift, the
model is served a prompt shape it never saw, and nothing in the eval catches it — so both are
generated from this one schema, and a test asserts the training renderer and the runtime renderer
produce byte-identical prompts for the same request.

## Request

```json
{
  "schema_version": 1,
  "task": "devcouncil.verdict",
  "context_b64": "Zm4gYWRkKCkge30K",
  "context_len": 12,
  "question": "Does this diff implement what the commit message claims?",
  "slots": [
    {"name": "verdict", "type": "choice", "options": ["stub", "logic", "cosmetic", "clean"]},
    {"name": "severity", "type": "score", "bins": 5},
    {"name": "evidence", "type": "span"}
  ],
  "route": "generic"
}
```

- `context` crosses the FFI boundary as **bytes and a length, never a Swift `String`**. A `String`
  round-trip normalizes Unicode and would silently change token boundaries and therefore line spans.
- `route` is `generic` | `registered`. `registered` requires a head file hash-bound to the backbone.

### The wire encoding of `context` — exactly one form

**`context_b64` (base64, ASCII) plus a required `context_len` (the decoded byte count). They must
agree, or it is a typed refusal.** There is no second accepted form.

This was underspecified and the two lanes diverged on it: `qd_data` emitted `context_b64`, while
`qd-runtime` took `Context` as `#[serde(transparent)] Vec<u8>` — a JSON array of integers — and
*refused a string in that position*. No caller could satisfy both, so the system did not work end to
end even though both sides' own tests passed. Recorded as `GAP-RT-WIRE-CONTEXT-ENCODING`.

Base64 rather than an integer array, on three grounds:

- **Size.** An 8K context is ~11 KB of base64 against ~35 KB as `[104,101,...]`. At the plan's whole-diff
  sizes that is the difference between a request and a payload.
- **The rule's actual concern is normalization, and base64 is immune to it.** "Never a `String`"
  exists because a Unicode round-trip rewrites token boundaries and line spans. Base64 is ASCII-only
  and byte-exact, so it does not have the property the rule was written to prevent. An integer array
  is not *more* faithful, only more verbose.
- **Convention.** Base64 is how bytes travel in JSON; an integer array invites a well-meaning client
  to send a string and get refused.

`context_len` is kept even though base64 makes it derivable, because a disagreement between the two
is the signature of a **truncated payload that still decodes cleanly** — the failure a length field
is for. Equal lengths prove nothing was lost; unequal lengths refuse.

A cross-language test asserts the Python encoder and the Rust decoder agree on the same bytes, in the
style of `qd-preflight/tests/schema_compat.rs`. Two sides agreeing is an assumption until it is
asserted — which is precisely how this divergence survived in the first place.

#### Canonical base64, not merely decodable base64

"Base64" is not one encoding until the slack is nailed down, so:

- the standard alphabet (`A-Z a-z 0-9 + /`), never URL-safe `-`/`_`;
- padding required, so the length is always a multiple of 4; no whitespace, no line wrapping;
- the bits a partial quantum does not use must be **zero**.

The last rule earns its place: `Zg==` and `Zh==` carry the same data bits, and a permissive decoder
returns `b"f"` for both. Two distinct strings naming one context is the same "undefined" this
section exists to remove, and it would put a digest over the encoded form at odds with a digest over
the bytes. Python's `base64.b64decode(validate=True)` checks the alphabet but **not** the slack bits,
so `qd_data.schema.decode_context` re-encodes and compares; `qd_runtime::b64` rejects them directly.
Both lanes are strict in the same place, and the cross-language test asserts it rather than trusting
it.

#### The four refusals, named identically in both lanes

A refusal identifier is what a caller branches on, so these four are the same string on both sides —
`Refusal::kind()` in `crates/qd-runtime/src/refusal.rs`, `QdRefusal.check` in
`python/qd_data/errors.py`:

| Condition | Identifier |
| --- | --- |
| `context_b64` absent, not a JSON string, or a `context` field sent instead | `context_not_bytes` |
| `context_b64` present but not canonical base64 | `context_not_base64` |
| `context_len` absent, or not a non-negative integer | `context_len_missing` |
| `context_len` disagrees with the decoded byte count (names **both** numbers) | `context_length_mismatch` |

The rest of the two refusal vocabularies do **not** match — `options_len` vs `too_many_options`,
`slot_names_unique` vs `duplicate_slot_name`, and four more. That is pinned as a table in
`crates/qd-runtime/tests/wire_context_crosslang.rs` and tracked as
`GAP-XLANG-REFUSAL-VOCABULARIES`; it is a rename across both suites, not part of this fix.

**Status: implemented 2026-09-19, `GAP-RT-WIRE-CONTEXT-ENCODING` closed.** The single owner of the
encoding is `crates/qd-runtime/src/context.rs` on one side and
`qd_data.schema.encode_context` / `decode_context` on the other. The standing check is
`crates/qd-runtime/tests/wire_context_crosslang.rs`, which runs the real Python encoder
(`crates/qd-runtime/tests/crosslang_fixtures.py`) into the real Rust decoder and covers the whole
shared surface — envelope, all three slot kinds, both routes, `SpanValue`, `HashExpectation`, and the
malformed payloads both lanes must refuse.

## Slot types

| Type | Answers with | Decoded how (generic route) |
| --- | --- | --- |
| `choice` | one of k <= 16 named options, or `noul` | 1 token over a **17**-row `lm_head` slice: 16 option letters + 1 reserved `noul` row |
| `score` | an ordinal bin (<= 16 bins), or `noul` | letters again; bins are ordered, loss is cumulative (CORAL-style) |
| `span` | `{start_line, end_line}` in the context, or `noul` | pointer head over line-start tokens |
| `noul` | abstain | a reserved letter present in **every** option set |

`noul` is a first-class member of every option set, not a dump class. The in-domain `noul` rate is
capped and is a reported gate; the held-out task families must go to `noul`.

### The letter budget: 16 options + 1 reserved `noul` = 17 rows

The plan states three constraints that cannot all hold: *"k <= 16 named options"*, *"decoded over the
16 letter tokens through a 16-row slice of `lm_head`"*, and *"a `noul` letter in every option set"*.
Sixteen named options plus a mandatory `noul` needs **seventeen** rows.

**Resolved as: `k <= 16` named options over letters A–P, plus one reserved `noul` row outside that
block — 17 rows in the slice.** The slice width is the thing that gives, because it is an
implementation detail of how we carve a 248,320-row `lm_head`, with no hardware meaning at 16 or 17.
`k <= 16` is a user-facing capability the plan states, and a caller told "16" and then refused at 16
because an invisible slot was already taken has been lied to by the documentation.

The data and runtime lanes reached this reading **independently and agree**
(`qd_data/schema.py::MAX_CHOICE_OPTIONS = 16` with `NOUL_LETTER = "Z"` outside A–P;
`qd-runtime/src/schema.rs::MAX_OPTIONS = 16` with `RESERVED_NOUL_ROWS = 1`). Tracked as
`GAP-SCHEMA-NOUL-LETTER-BUDGET`. If it ever flips it is a one-constant change on each side — and the
two sides must not disagree silently, which is what the `render_contract` test exists to prevent.

Consequence for quantization: the calibration step fits **17** rows, not 16.

## Refusals — fail closed, never truncate

The runtime **refuses** rather than degrades when:

| Condition | Why a refusal and not a fallback |
| --- | --- |
| tokenizer / weight / head / label-set hash != build | A swapped tokenizer maps wrong ids silently. Answering would be confidently wrong |
| `options.len() > 16` | The letter slice holds 16 option rows plus the reserved `noul` row. Dropping an option changes the question |
| `context` over the configured cap | Truncating moves the answer out of the window without saying so. Line spans would point at the wrong lines |
| a `code.defect_class` `context` not in the trained shape (`file: <path>`, a blank line, unified-diff hunks whose bodies match their headers): `context_not_unified_diff`; or its file's language not one the pool held: `context_language_not_in_pool` (`crates/qd-runtime/src/admission.rs`) | The model was never shown such a context. An answer would be read from a shape it cannot read, and would look like any other |
| `slots` empty, or a duplicate slot name | The answer map would be ambiguous |
| `bins < 2` for `score`, or `bins > 16` | Same 16-letter limit; a 1-bin ordinal is not a question |
| `schema_version` unknown | Forward-compat guessing is how a field changes meaning silently |

A refusal is a typed error carrying which check failed and both values compared. It is never an
empty answer, and never `noul` — `noul` means *the model abstained*, a refusal means *the request was
not answerable as posed*. Collapsing the two would let a hash mismatch read as model humility.

## `expect` — the hash pins, and the one that meant two things

A request may carry an `expect` block. Every field is optional; each present field is compared
against the **loaded build**, and a disagreement is a typed refusal naming both values.

```json
"expect": {
  "tokenizer_hash": "…", "weight_hash": "…", "head_hash": "…",
  "label_set_hash": "…", "calibration_hash": "…"
}
```

**Every one of these is a property of the build, not of the request.** That sentence is the whole
section, because one of them was read the other way in the Python lane and nothing caught it:

> `qd_data.schema.label_set_hash(slots)` hashed the **request's own slot list**, so its value moved
> with the request. The runtime compares `expect.label_set_hash` against the loaded backend's
> `identity.label_set_hash` (`qd-runtime/src/runtime.rs::check_hashes`), which is fixed for a given
> set of weights. A caller computing one and sending it as the other earns a hash-mismatch refusal
> on **every** request — the two can never be equal except by collision.

Renamed to `slot_set_digest`, which is what it computes: the identity of a *task's* label set, used
to bind a registered head to the task it was fitted against. Tracked as
`GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`, and asserted by
`wire_context_crosslang.rs::python_has_no_hash_expectation_and_its_label_set_hash_is_a_different_quantity`
rather than left as prose.

`calibration_hash` had the same shape on the Rust side. It is compared with the table the runtime
**calibrates with** (`CalibrationTable::hash`), not with the hash the backend declares; until
2026-10-01 it was the latter, and a runtime serving a fitted table answered a pin of the reference
table (`GAP-RT-CALIBRATION-HASH-PIN-BINDS-THE-BACKEND-NOT-THE-LOADED-TABLE`,
`crates/qd-runtime/tests/calibration_hash_pin.rs`). A runtime built from a release takes its table
only from the release, verified against `release_manifest.json` (`qd_runtime::release`).

A runtime serving an N-tower ensemble (`Runtime::from_ensemble`, `qd_runtime::ensemble`) reports
**the ensemble's** `weight_hash`: `ensemble_weight_hash` over the members' weight hashes in
manifest order, recorded as `expected_identity.weight_hash` in `ensemble_manifest.json`. A pin of
one member's weight hash is refused by the ensemble, as it should be: the answer is not that
tower's. `calibration_hash` is the ensemble's table, fitted on the ensemble's decode (the mean of
the members' row log-softmax, `qd_runtime::release::ENSEMBLE_DECODE`).

This is the second instance of one shape found in a single audit — `GAP-RT-WIRE-CONTEXT-ENCODING`
was the first. **One name meaning two things, with both lanes' suites green**, is the failure mode
this document exists to prevent, and neither instance was caught by either lane testing itself. The
cross-language fixture is the standing answer: it drives the real Python encoder into the real Rust
decoder, and its `inventory` block *fails* a test when a shared type appears on one side only rather
than silently skipping it.

## The answer side is a corpus, not an example

Everything below this line was, until 2026-09-19, **one example object and no spec** — while the
request side above ran to six sections. Four audit questions went unanswered for that reason alone.
The asymmetry is now closed the way the training-side asymmetry was closed
(`python/qd_train/artifacts.py`, `docs/training-contract.md`): by making the seam **executable**.

**`fixtures/wire/` is the contract; this section is its reading guide.** It holds one JSON file per
envelope variant this runtime can emit — every answer shape, every refusal, every backend error —
generated from the real types by the real code paths, plus an `index.json` manifest naming every
file, its `status`, its `kind` and the request that produced it. The Python lane's parser reads the
same files. When a field here is added, renamed or removed, the corpus changes and the other side
**breaks loudly** instead of silently accepting an envelope it no longer understands.

```text
cargo test -p qd-runtime --test wire_fixtures                        # compare; fails on any drift
QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures    # regenerate
```

Comparison is the default deliberately. A corpus that regenerates on every run records whatever the
code currently does and can never disagree with it, which is not a gate. The generator is
`crates/qd-runtime/src/fixtures.rs`; `tests/wire_fixtures.rs` also asserts the corpus covers **every**
variant of both error enums, so a new variant fails a test rather than quietly going unexported.

The answer fixtures are produced by the reference backend, whose answers are a function of the
rendered prompt's bytes, so a change to the prompt (prompt format 2 was one) re-rolls them. Each
answer fixture's intended shape -- which slots answer, which abstain, whether it is degraded -- is
therefore pinned by name in `fixtures.rs`'s `tests::every_answer_fixture_shows_its_intended_shape`:
a re-roll that loses an answered or an abstained shape fails there, and the fix is a new fixture
input, not a new expected shape. Under format 2 the three-slot example and the score-only fixture
read `fn add(a: i32, b: i32) -> i32 { // 6` as their first context line, and the span fixture
`alpha 2`; the search that chose them is `AUDIT/v5-fmt-2026-10-02/fixture_search.py`.

**Both halves exist.** `python/qd_wire/` is an independent parser for the answer side, and
`python/tests/test_wire_golden_corpus.py` reads this corpus and re-derives every claim below from
the bytes. That closes `GAP-XLANG-NO-PY-ANSWER-PARSER`, which had stood because there was no second
implementation to disagree with — the condition that let `GAP-RT-WIRE-CONTEXT-ENCODING` and
`GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS` survive with both suites green.

## Answer

Three replies are possible and a caller must match all three. They are distinguished by a
`status` discriminant, and their payloads **share no field name** beyond `schema_version`:

| `status` | Carries | Means |
| --- | --- | --- |
| `ok` | `backend`, `degraded`, `slots` | the model was asked and answered — possibly by abstaining |
| `refused` | `refusal`, `message` | the request was not answerable as posed; the model was never asked |
| `error` | `error`, `message` | the backend could not serve; the model was never asked |

```json
{
  "status": "ok",
  "schema_version": 1,
  "backend": "reference-deterministic-v1",
  "degraded": true,
  "slots": {
    "verdict": {"value": "stub", "conformal_set": ["stub", "logic"], "score": 0.71,
                "noul": false, "degraded": true},
    "severity": {"value": 3, "conformal_set": [2, 3, 4], "score": 0.55,
                 "noul": false, "degraded": true},
    "evidence": {"value": {"start_line": 41, "end_line": 47}, "conformal_set": null,
                 "score": 0.62, "noul": false, "degraded": true}
  }
}
```

**Five envelope keys, not two.** The example in this document used to show `schema_version` and
`slots`, and a caller written from it would never have read `backend` — which is the field that says
an answer came from `reference-deterministic-v1` rather than from a model, and is the single most
important field in the envelope while no Metal backend exists. `GAP-RT-SPEC-ANSWER-ENVELOPE-FIELDS`.

**Five slot keys, always all five present.** `value` and `conformal_set` are emitted as explicit
`null`s rather than omitted: a missing key and a null key would be two spellings of one state.

### `value` is absent **if and only if** `noul`

This was the contract's own open question, and it mattered more than it reads. `value` and `noul`
are *one fact spelled twice* — precisely the shape this repo has now shipped three times with both
suites green (`GAP-RT-WIRE-CONTEXT-ENCODING`, `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`). A caller
branching on `noul` and a caller branching on `value === null` must never disagree.

So the biconditional is **enforced, not described**:

- *producing* — `SlotAnswer::answered` and `SlotAnswer::abstained` are the only two constructors,
  and each sets both fields together. There is no struct literal left in `crates/qd-runtime/src/`;
- *parsing* — a hand-written `Deserialize` refuses `{"value": …, "noul": true}` and
  `{"value": null, "noul": false}` outright. `tests/answer_envelope_contract.rs` fails against the
  derived deserializer that preceded it.

A parser on the other side should do the same. The corpus contains both legal shapes and neither
illegal one.

### Unknown fields are **refused**, in both directions

This document previously recorded *"whether unknown fields are refused or ignored — the two lanes
chose differently"*. The request side refused (`wire::known_keys`) and the answer side silently
ignored, so one protocol had opposite rules depending on which way a message was travelling.
Settled: **refused**, on the answer side too. `AnswerEnvelope`, `RefusalEnvelope`, `ErrorEnvelope`
and `SlotAnswer` all carry `deny_unknown_fields`.

One consequence worth stating, because it is a trap: the whole reply must be read as the
three-variant `Response`, not as a bare `AnswerEnvelope`. `status` belongs to the discriminated
union, and an envelope read on its own will now reject it as an unknown key.

### `score`, `conformal_set`, `degraded`

- `score` is a **margin** on temperature-scaled probabilities, `p_top − p_second`, always in
  `[0, 1]`. Never entropy: the audits attribute "entropy as confidence" to the Laya spec as a
  defect, and there is no code path in `crates/qd-runtime/src/calibration.rs` that computes one.
  For a `span` it is the **weaker of the two pointer margins** — a span is only as good as its less
  certain end.
- `conformal_set` is a split-conformal set on those calibrated scores, holding *values* and never
  the reserved abstain row. It is `null` for a `span`.
- `degraded` has exactly **three** sources in this build, and thermal pressure is not one of them.
  The previous wording — *"or under thermal pressure with a raised threshold"* — described a
  mechanism that does not exist anywhere in the repository. The real three are: the backend
  declares `is_model: false`; the runtime was rebuilt after a poison (`Runtime::mark_degraded`);
  or the slot-isolation check could not run because the backend's state is not host-visible.
- **`degraded` is mirrored, not per-slot.** The envelope's flag and every slot's flag are the same
  bit. The contract placed `degraded` on the slot, but its stated meaning is a property of the
  answer as a whole, so the runtime sets one and copies it down. A caller may read either; what it
  may not do is expect them to differ. Asserted in `tests/answer_envelope_contract.rs`.

### The refusal identifiers, all of them

This document used to name four and leave the other 32 undocumented. There are **36** refusal kinds
and **12** backend-error kinds, and rather than a table that goes stale, the complete set is the
corpus: one file per kind under `fixtures/wire/`, named `refusal-<kind>.json` and `error-<kind>.json`,
each carrying a real instance with both compared values filled in and its `message` rendered.
`fixtures/wire/index.json` lists them. The authority in code is `Refusal::kind()` and
`BackendError::kind()` in `crates/qd-runtime/src/refusal.rs`, and the counts are pinned by
`tests/wire_fixtures.rs` and `tests/refusal_is_not_noul.rs`, whose exhaustive matches fail to
compile when a variant is added.

The two lanes still spell six of these differently (`options_len` vs `too_many_options` and five
more). That is unchanged and still tracked as `GAP-XLANG-REFUSAL-VOCABULARIES`.

## Answering procedure

1. Render the request's prefix in **prompt format 2** (below) and prefill it **once**. The prefix
   is everything through the question line; each slot's suffix starts after it.
2. **Snapshot** the recurrent GDN state and the attention KV.
3. Answer each slot as a 1-token query **from the snapshot**. Never write back — the `readonly` flag
   on the decode kernel (K2) is the slot-isolation mechanism, one flag instead of a mask kernel.
   This is what makes the prefix-LM mask expressible on a recurrent model at all: the snapshot *is*
   the prefix cache.
4. For `choice`: run a **second pass with the options permuted** and require agreement. Disagreement
   is `noul`. This turns letter-position bias from a silent in-domain win into an abstention.

   **The permutation must be a derangement** — no option may keep its position. A uniform shuffle
   (Fisher-Yates) leaves fixed points, and when the winning row happens to be one, a purely
   position-biased model *agrees with itself* across both passes and the abstention never fires. The
   check then passes precisely on the cases it exists to catch. Use Sattolo's algorithm, which
   generates only cyclic permutations.

   This applies to the **training gate as well as the runtime**: a permutation-consistency figure
   measured over uniform shuffles is measuring something weaker than the runtime enforces, so the
   >= 95% gate must be computed over derangements or it is not the same quantity.

   The permutation is seeded from `DecisionRequest::digest()` and the slot name, so it is
   **reproducible for a given request**: a caller that sees a `noul` from permutation disagreement
   can replay the exact pair of passes that disagreed. `render::second_pass_permutation`.
5. Return per-slot `{value, conformal_set, score, noul, degraded}`.

Step 4 is why permutation consistency (>= 95%) is a training gate and not only a runtime check: a
model that fails it makes the second pass fire constantly and the abstain rate blows the cap.

### The prompt the model sees: format 2

One renderer per lane writes it, byte for byte the same: `python/qd_data/render.py` for training
and `crates/qd-runtime/src/render.rs` for serving. Prompt format 2 (v5) puts the context **before**
the question:

```text
<|qd_begin|>
<|qd_prompt_format|>2
<|qd_schema_version|>1
<|qd_task|>T
<|qd_route|>R
<|qd_context_begin|>
C
<|qd_context_end|>
<|qd_question|>Q
```

Then, for each slot, its suffix: `<|qd_slot|>`, `<|qd_type|>`, the option block with the reserved
`Z. noul` row last, and `<|qd_answer|>`, the position the one-token query is answered from.

- **Only the question moved.** Format 1 (v4) had no format line and rendered the question line
  between the route and the context. Task and route stay ahead of the context, so N questions with
  one task over one context share every byte through `<|qd_context_end|>`. The slot suffixes did
  not change.
- **The format is not the wire version.** The request's `schema_version` stays 1, so callers send
  the same requests. The format is a property of the *model*: what it was trained on.
- **Every marker has the shape `<|qd_...|>`, the format line's included**, so the one escape rule
  covers it. A context or question carrying `<|qd_prompt_format|>` or `<|qd_question|>` is
  rendered as text (`<\|...`), and the renderer's own lines stay the only markers.
- **A model is never served another format than it was trained on.** The release manifest
  (`qd-release.v2`) binds `expected_identity.prompt_format`, and `qd_runtime::release` refuses a
  release whose format is not `render::PROMPT_FORMAT` (a v4 release, `qd-release.v1`, is refused
  by its format, and a v4-era runtime refuses a `qd-release.v2` one). `qd-export` stamps the format
  from the checkpoint's source manifest (`<avg>.manifest.json` `prompt_format`, an integer; absent
  means 1), never from its own constant, so a v4 checkpoint exported now is refused, not
  mislabelled. The shard header records the format its sequences were rendered in
  (`ShardHeader.prompt_format`), and the shard reader refuses a set of another format.

### How many decodes each slot kind costs

"Answer each slot as a 1-token query" and "a span answers `{start_line, end_line}`" cannot both be
literally true for a two-ended answer, so the count is stated here and **measured** in
`tests/answering_procedure.rs`, which counts what the runtime actually asks the backend for:

| Slot kind | Decodes | Rows per decode | Query kinds |
| --- | --- | --- | --- |
| `choice` | **2** — the pass and step 4's permuted pass, different suffixes, same snapshot | `k + 1` | `letters` |
| `score` | **1** — ordinal bins are never permuted | `bins + 1` | `letters` |
| `span` | **2** — one per end, *same* suffix, same snapshot | `line_count + 1` | `pointer_start`, then `pointer_end` |

`GAP-RT-SPEC-SPAN-DECODE`. Two decodes is the reading that keeps every decode a 1-token query; a
one-decode reading, with both ends read from a single distribution, is a different head, and the
kernel (K2) and training lanes must build the one written here.

### The `span` slot, completely

- The pointer head ranges over **this context's line starts**, plus the reserved `noul` row. A line
  is defined by `Context::line_starts`: `\n` terminates a line, a trailing `\n` does not open an
  empty final line, and `\r` is an ordinary content byte. That definition is shared with the
  renderer, whose `escape_block` preserves the newline count *exactly* so the escaped context has
  the same line numbering as the raw bytes.
- The row count is therefore **per request**, which is exactly why a fixed-shape registered head
  cannot answer a `span`. The runtime refuses any other row count as `logit_shape_mismatch` rather
  than reading a padded head — so a head padded to a fixed maximum fails loudly here.
- `{start_line, end_line}` are **1-based and inclusive**.
- An end before its start **abstains**. A backwards span is not a low-confidence span, it is not a
  span, and silently reordering it would turn a broken answer into a plausible-looking one.
- Either end landing on the abstain row abstains.
- **`1 <= start_line <= end_line` is a term of the format, not only a property of the producer.**
  Both bounds were true of every span this runtime emits and enforced in exactly one place — the
  decode path above — while `SpanValue` derived its `Deserialize` with no check, so both *readers*
  accepted `{"start_line": 47, "end_line": 41}` and the type carried no guarantee anything
  downstream could rely on. A consumer computing span overlap or slicing context lines was trusting
  a producer-side check, and a second producer — a registered head, a different backend, a
  hand-written fixture — would satisfy the format while breaking it. The invariant now belongs to
  the type, on both sides: `schema::SpanValue`'s `TryFrom<SpanValueWire>` and
  `qd_wire.answer.parse_slot_value`. `GAP-XLANG-SPAN-BOUNDS-UNPINNED`.
  Note the two failure modes stay distinct: a *decode* that points backwards abstains (it is a model
  output, and `noul` is the honest answer); a *wire value* that runs backwards is refused (it is a
  malformed envelope, and there is no model to abstain for).

### The reserved abstain row is **last**, for every slot kind

Row `rows - 1`, always: option index `k` for a `choice`, bin index `bins` for a `score`, and index
`line_count` for a `span`'s pointer head. `python/qd_train/heads.py` places the abstain column at
index `n_candidates`, which is the same position.

This is written down because it is an off-by-one with **no observable symptom during training**. A
head trained with the abstain column first, served by a runtime reading it last, would put every
"no evidence" answer on the context's final line, or on option A, and no loss curve would show it.
`tests/answering_procedure.rs` asserts both directions: a backend that wins the last row must read
as an abstention, and a backend that wins row 0 must read as line 1 / bin 1 / option A.

## Registered route

An app ships a head file, hash-bound to the backbone, and gets a single GEMV on pooled features
instead of option text in the prompt. Same prefill, same snapshot, same calibration table.

### "Same abstain rule" is **not** literally true, and that changes what the agreement test means

The abstain rule has two halves — the calibrated-margin threshold, and step 4's permuted second
pass. **The registered route runs the margin half only.** `GAP-RT-SPEC-REGISTERED-ROUTE`.

It is not an omission that could be fixed: there is no option text in the prompt to permute, the
head's rows *are* the option order, and permuting the rows of a GEMV is exactly equivariant — so the
check would agree with itself by construction and abstain on nothing. The runtime issues **no
decode at all** on this route, which `tests/answering_procedure.rs` measures; there is no second
pass to run.

The table is `answer::abstain_rule(route, kind)`, which the runtime itself consults:

| Route | `choice` | `score` | `span` |
| --- | --- | --- | --- |
| `generic` | margin + permuted second pass | margin | margin + span ordering |
| `registered` | margin | margin | **not served** |

**The consequence, stated rather than left to be discovered:** the registered route has one fewer
abstention mechanism, so its `noul` rate will not match the generic route's *even on a perfectly
fitted head*. A registered task is a speed-up, never a different model — but the agreement test this
section asks for ("generic-route and registered-route answers must agree on the held-out set at a
stated rate") must account for that difference, or it will report a route difference as a head bug.
The rate itself is still not stated; see below.

### A registered head cannot answer a `span`

A `span`'s pointer head ranges over *this* context's line starts, so its row count changes per
request, and a head file has a fixed row count. The runtime refuses with the typed
`registered_route_span_unsupported` rather than quietly answering it on the generic route — which
would make `route` advisory, and would make the generic-vs-registered agreement test compare the
generic route with itself. If registered-route spans are ever wanted, the head file needs a
per-context row shape, which is a different design and a contract change.

## The payload cap does not cover a maximal slot list

`wire::MAX_PAYLOAD_BYTES` bounds one JSON line. Its derivation lived in a comment, was written
against the retired integer-array `context` encoding, and stopped describing the number when the
encoding became base64. `GAP-RT-PAYLOAD-CAP-VS-MAXIMAL-SLOTS`.

The derivation is now a function, `wire::payload_budget(caps)`, which returns the worst-case wire
size term by term — context, question, task, slot list, envelope. The numbers are deliberately
**not reproduced here**, because a number in a document is how the last derivation went stale; run
`cargo test -p qd-runtime --test wire_refusals the_payload_cap_does_not_cover -- --nocapture` to
print the current ones. What is asserted, and therefore true as of any green run:

- the terms the cap's comment *was* derived from — context, question, task, envelope — **fit**
  inside the cap;
- a request legal at **every** documented limit at once does **not**. The slot list
  (32 slots × 16 options × 512 bytes, `\uXXXX`-dense) is larger on its own than everything else
  combined;
- so such a request is refused by `payload_over_cap`, which names the payload rather than the slot
  list that caused it. The refusal is correct; it just costs a debugging cycle.

The number is **left alone on purpose**. Raising it is a joint decision with the Python lane's
`RenderCaps`; shrinking it would refuse payloads that are legal today. What has changed is that the
shortfall is now a failing assertion away from anyone who edits either side.

**A slot name is capped at `schema::MAX_SLOT_NAME_BYTES` = 256 bytes**, UTF-8, measured before
escaping, and refused rather than truncated (`slot_name_over_cap`). Until 2026-09-19 it had no cap
anywhere: `wire::parse_slot` checked it non-empty and stopped, `render::slot_suffix` did not check
at all, and `MAX_PAYLOAD_BYTES` was the only bound — which made `payload_budget().total()` a *lower
bound* on the worst case rather than the worst case. `GAP-RT-SLOT-NAME-UNCAPPED`.

Two lanes each declined to add the cap alone, correctly: a cap on one side refuses requests the
other accepts, which is this repository's recurring defect rather than a fix for it. So it landed on
every side that reads a slot name at once — the request (`wire::parse_slot`, the sole constructor of
`SlotSpec`), the prompt (`render::slot_suffix`, which also serves specs built in Rust), and both
readers of the answer map (`schema::deserialize_slot_map` and `qd_wire.answer.check_slot_name`).
Python does not transcribe the number: `qd_wire.contract.MAX_SLOT_NAME_BYTES` is generated from
`schema.rs` and re-derived and compared on every test run.

Why 256: a slot name is an **identifier**, and this document already prices an identifier at 256
(`max_task_bytes`), against 512 for option text and 4096 for the question. The longest slot name
anywhere in this repository is 8 bytes, so the cap refuses nothing that exists.

One reader remains uncapped and it is **not** on the wire: `qd_data.schema`, the training lane's
request *builder*, still checks only for emptiness. That divergence is measured rather than assumed
— the longest name `qd_data` actually produces is 8 bytes, so nothing it can build is refused today
— and `wire_context_crosslang.rs::the_slot_name_cap_is_one_number_here_and_in_qd_wire_and_qd_data_produces_nothing_over_it`
fails on the day that stops being true.

`payload_budget().uncapped` is now empty, so `total()` is the worst case rather than a lower bound
on it. The field stays so the next uncapped term lands in it instead of silently not being counted.
Counting the names **widened** the shortfall against `MAX_PAYLOAD_BYTES`, by 32 × 256 × 6 = 49,152
bytes. `MAX_PAYLOAD_BYTES` is unchanged: that number is a human decision, and this lane may report
the arithmetic but not move the gate.

## What this document still does not specify

Named here rather than left to be discovered the same way. The list is shorter than it was; what
remains is either owned by another lane or is a decision two lanes must take together.

- **The refusal vocabularies still differ** across six identifiers. `GAP-XLANG-REFUSAL-VOCABULARIES`.
- **`MAX_PAYLOAD_BYTES` is not re-derived from the slot list** — see above. The joint decision with
  the Python lane's `RenderCaps` is outstanding. The slot-name cap that was also outstanding here
  has landed; capping it widened the shortfall rather than closing it.
- **The byte caps still live in code, in two copies**, one per language — a pin, not a single
  owner. The pin is real as of 2026-09-19:
  `wire_context_crosslang.rs::the_escape_growth_bound_and_every_byte_cap_are_the_same_numbers_on_both_sides`
  reads the live `qd_data.render` and compares all five caps, the escape-growth bound, and the
  floor *formula*. It was added because the claim that `render_contract.rs` already did this was
  **false** — that file compares frozen golden prompt bytes and never reads a cap number, so the
  escape-growth divergence (6 in Python, 2 in Rust) could have recurred unseen. Reproducing that
  split now fails the new test, which was checked rather than assumed.
- **The pointer head's training-time row shape is not recorded anywhere.** The runtime requires
  exactly `line_count + 1` rows and refuses anything else, so a head padded to a fixed maximum
  fails loudly rather than silently — but whether the training lane pads is a decision no document
  states.
- **The generic-vs-registered agreement rate is "a stated rate" with no number**, and per the
  section above it cannot be 100% even in principle, because the two routes run different halves of
  the abstain rule.
