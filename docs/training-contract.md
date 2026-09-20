# S2, S4 and the trainer: why the contract is code

Three lanes meet here. S2 produces a vocabulary remap, S4 produces token shards, the trainer
consumes both. The shared formats live in `python/qd_train/artifacts.py` as **executable**
definitions — dataclasses with validating constructors — and not in this document.

This file explains why, and what each invariant is defending against. It deliberately does
**not** restate the formats: a second description of a format is a second thing to keep in
sync, and the code is the one that runs.

## Why not a document

This repo has produced two cross-lane failures, both on 2026-09-19, both with the same shape:

| | What each lane built | Why it broke |
| --- | --- | --- |
| `GAP-RT-WIRE-CONTEXT-ENCODING` | `qd_data` emitted `context_b64`; `qd-runtime` accepted only a JSON array of integers, and *refused* a string in that position | No caller could satisfy both. The system did not work end to end |
| `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS` | Python hashed the request's slots; Rust compared the build's identity | One name, two quantities. A caller pinning one earns a refusal on every request |

**Both lanes' test suites were green in both cases.** Each lane tested itself against its own
reading of a paragraph, and the paragraph was the only thing holding the system together.

So for this next round the seam is a module the lanes import. A lane does not implement
`ShardHeader` from a description; it constructs one and is bound by `__post_init__`. There is
nothing left to read differently. `Batch` is defined there for the same reason — it is the seam
between S4's reader and the trainer, and a seam specified in two prompts is a seam two lanes
implement two ways.

Where a shared *decision* cannot be a type, it is a shared function: `bucket_for` is imported
by the writer and the sampler, because two plausible implementations of "which bucket is this
length in" that disagree at a boundary produce a padding-waste figure measured against a
bucketing nobody trains with.

## The invariants that matter, and the failure each blocks

**`packed` is always `False`** — `ShardHeader` raises on `packed=True`. `docs/plan-corrections.md`
SAFETY-2: a GDN layer carries recurrent state *along* the sequence, so two examples concatenated
into one row bleed state across the boundary, and no attention mask expresses "reset the
recurrence here". This needs a structural block rather than a code review note because **the
damage is silent — loss still falls.**

**A dropped token is an error, never a substitution.** `RemapTable.encode` raises and names the
offending id. The tempting alternative is an `UNK` fallback, which converts a coverage bug into
a corpus where some fraction of tokens mean "something was here" — and trains the model to
predict exactly that. It shows up as nothing worse than slightly worse loss.

**Shards carry their own provenance, and rule 3 is checked again at their boundary.**
`open_training_data` is the one door onto training data and it refuses held-out input — but it
guards *manifests*. A shard is derived from a manifest and read directly by the trainer, so
without `assert_shard_trainable` the held-out rule would be satisfied on paper and bypassed by
indirection. Same check, new door.

**A `SLOT_LM` row never carries a `target_index`.** `Batch` refuses it rather than letting the
loops decide. Two readings were both defensible — an LM row deliberately mixed into fine-tuning,
or a mislabelled choice row — and they train *different objectives* while the loss curve looks
fine under either. What settled it was the contract's own history rather than an argument about
intent: `span_target` grew the `NO_SPAN` sentinel the moment a row needed to say "not me", and
`target_index` never grew an equivalent. A field that would have needed a do-not-care value and
never got one was never meant to express the row.

**A row of length 1 is refused where it is built, not tolerated where it is used.** `shards`
has refused `ids.size < 2` since S4 landed, for the same reason and at the same number; `Batch`
was the odd one out, and only on the CPT path. A length-1 row has no next-token pair, so it is
loaded, padded and paid for while contributing zero gradient — the effective batch is quietly
smaller than the recipe says. That is training on padding, in the opposite direction.

**The pointer head's row count is checked in both directions.** The head pads to the *batch*
maximum; the runtime demands exactly `line_count + 1`. Those agree only because serving runs one
context at a time, which was true but argued rather than enforced. It is now a postcondition on
both pointers, so a serve-time `logit_shape_mismatch` surfaces as a train-time `ValueError`. The
check has to be two-directional: the dangerous shape is a finite column *inside* the served
range, which a one-sided bound does not see.

**An empty measurement is `NotRun`, not a pass.** `padding_waste([])` is `0/0`; reporting it as
0% waste would let an empty shard set clear the ≤15% gate. This is `all([]) == True` one level
down, and it is the reason the tri-state exists.

## The torch boundary

torch is an optional `mac` extra and is **not** installed in the repo venv. That is a
constraint worth keeping rather than working around, because it forces the split that makes
this testable at all:

- Format, ordering, bucketing, coverage policy, schedules, the wall-clock cap and checkpoint
  bookkeeping are **torch-free**, and run in CI on any machine.
- Model loading, the fused cross-entropy and the optimizer step need torch, and their tests
  `importorskip` so they **skip** rather than pass when it is absent.

A test that could not run must never report the same result as one that ran and passed — the
same rule the image gates follow, applied to the suite itself.

## Wiring

```
qd_data rows ──▶ open_training_data ──▶ S4 write_shards ──▶ uint32 memmap + ShardHeader
                       (rule 3)              │                        │
S2 build_remap ──▶ RemapTable ───────────────┘                        │
       │              (encode raises on a dropped id)                 ▼
       │                                              ShardReader.batches(seed, epoch)
       ▼                                                              │
apply_remap_to_model ──▶ model with an ~80K lm_head                   ▼
                                                        trainer(Iterable[Batch])
                                                              │
                                                     RunRecorder ──▶ ledger row
```

The trainer takes `Iterable[Batch]` rather than a `ShardReader`. That is dependency inversion
for its own sake only in part; the practical reason is that it makes the trainer testable
against synthetic batches, and it let the three lanes be built concurrently without one waiting
on another's file to exist.
