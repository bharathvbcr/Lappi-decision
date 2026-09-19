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
  "context": "<bytes>",
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

## Slot types

| Type | Answers with | Decoded how (generic route) |
| --- | --- | --- |
| `choice` | one of k <= 16 named options, or `noul` | 1 token over the 16 letter-token slice of `lm_head` |
| `score` | an ordinal bin, or `noul` | letters again; bins are ordered, loss is cumulative (CORAL-style) |
| `span` | `{start_line, end_line}` in the context, or `noul` | pointer head over line-start tokens |
| `noul` | abstain | a reserved letter present in **every** option set |

`noul` is a first-class member of every option set, not a dump class. The in-domain `noul` rate is
capped and is a reported gate; the held-out task families must go to `noul`.

## Refusals — fail closed, never truncate

The runtime **refuses** rather than degrades when:

| Condition | Why a refusal and not a fallback |
| --- | --- |
| tokenizer / weight / head / label-set hash != build | A swapped tokenizer maps wrong ids silently. Answering would be confidently wrong |
| `options.len() > 16` | The 16-row `lm_head` slice cannot express a 17th option. Dropping one changes the question |
| `context` over the configured cap | Truncating moves the answer out of the window without saying so. Line spans would point at the wrong lines |
| `slots` empty, or a duplicate slot name | The answer map would be ambiguous |
| `bins < 2` for `score`, or `bins > 16` | Same 16-letter limit; a 1-bin ordinal is not a question |
| `schema_version` unknown | Forward-compat guessing is how a field changes meaning silently |

A refusal is a typed error carrying which check failed and both values compared. It is never an
empty answer, and never `noul` — `noul` means *the model abstained*, a refusal means *the request was
not answerable as posed*. Collapsing the two would let a hash mismatch read as model humility.

## Answer

```json
{
  "schema_version": 1,
  "slots": {
    "verdict": {"value": "stub", "conformal_set": ["stub", "logic"], "score": 0.71,
                "noul": false, "degraded": false},
    "severity": {"value": 3, "conformal_set": [2, 3, 4], "score": 0.55,
                 "noul": false, "degraded": false},
    "evidence": {"value": {"start_line": 41, "end_line": 47}, "conformal_set": null,
                 "score": 0.62, "noul": false, "degraded": false}
  }
}
```

- `conformal_set` is a **split-conformal** set on calibrated scores — margin, not entropy. The audits
  attribute "entropy as confidence" to the Laya spec as a defect; entropy is not used here.
- `degraded` is true when the runtime answered from a rebuilt-after-poison `GpuRuntime`, or under
  thermal pressure with a raised threshold. A degraded answer is still an answer, but a caller that
  treats it as equal to a clean one is ignoring the flag it asked for.

## Answering procedure

1. Prefill the context **once**.
2. **Snapshot** the recurrent GDN state and the attention KV.
3. Answer each slot as a 1-token query **from the snapshot**. Never write back — the `readonly` flag
   on the decode kernel (K2) is the slot-isolation mechanism, one flag instead of a mask kernel.
   This is what makes the prefix-LM mask expressible on a recurrent model at all: the snapshot *is*
   the prefix cache.
4. For `choice`: run a **second pass with the options permuted** and require agreement. Disagreement
   is `noul`. This turns letter-position bias from a silent in-domain win into an abstention.
5. Return per-slot `{value, conformal_set, score, noul, degraded}`.

Step 4 is why permutation consistency (>= 95%) is a training gate and not only a runtime check: a
model that fails it makes the second pass fire constantly and the abstain rate blows the cap.

## Registered route

An app ships a head file, hash-bound to the backbone, and gets a single GEMV on pooled features
instead of option text in the prompt. **Same prefill, same calibration table, same abstain rule.**

A registered task is a speed-up, never a different model. The test that keeps this honest: for a task
with a registered head, generic-route and registered-route answers must agree on the held-out set at
a stated rate; a registered head that disagrees with the generic route is a bug in the head, not a
better answer.
