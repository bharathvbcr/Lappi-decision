# Lappi

**Lappi** is the model this repository builds. The repository keeps the name `qwen-decision`
and the crates keep the `qd-` prefix; the product is Lappi.

## Why the name is not `qwen-decision`

`qwen-decision` names a dependency. The moment the base model changes — and for a decision
model at this size that is likely rather than hypothetical — the name is wrong. Lappi carries
no such expiry, and after 2026-09-19 it also stopped being merely cosmetic: the strongest
public peer, `Mapika/decider-2b`, is *literally* Qwen3.5-2B-Base, so "the Qwen decision
model" is now ambiguous in a way it was not a week ago.

Two collisions are known and accepted rather than overlooked. **"Lappy" is common
Indian-English slang for laptop**, so a substantial share of readers will hear "laptop"
first. **Lappi is the Finnish name for Lapland**, so search results skew to tourism. Neither
is disqualifying; both dilute searchability, and neither was discovered after the fact.

## What Lappi is

A typed decision model. Context, a question, and a schema of slots go in; the slots come back
answered, with a calibrated score and an explicit abstention. It does not generate text and
it does not write reasoning first. The wire contract is `docs/schema-api.md`; this document
does not restate it.

The category has a name now — **System 1 models** — and four public members as of
2026-09-19. `AUDIT/prior-art-jev-nimble-2026-09-19.md` holds the evidence.

## The ladder

Lappi is not one model. It is a ladder, and the rungs exist because the prior art showed the
cost of each and they disagree by three orders of magnitude.

| Rung | Approach | Public evidence | Status here |
| --- | --- | --- | --- |
| **0** | byte-level, from scratch, ~600K params | `cua-s1-forms`: 706K params, near-ceiling on a *narrow* task | **Trained & evaluated.** 80.97% best vs 88.5% control (`AUDIT/capacity-and-learning-rate.md`). Measured all 144 margins across 18 arms (`ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl`); model trails control in 17/18 arms, establishing floor |
| **1** | frozen base, option logits, no training | SemIf: 0.813 balanced accuracy on a frozen 4B | Planned as Step 1 in `docs/train-plan-2026-09-28.md` |
| **2** | LoRA r=16, one epoch | Nimble: 90.12% raw agreement, 2,676 examples, two days | Planned as Step 6 control arm |
| **3** | 2B Supervised FT / CPT | `decider-2b`: 0.805 in-task / 0.755 held-out | **Initial FT run completed.** 2k commitpackft rows on GH200 (`ledger/gh200-ft-commitpackft-2026-09-22.jsonl`); score slot learned on 2/3 seeds; fast-training plan in `docs/train-plan-2026-09-28.md` |

The ordering is deliberate and was a decision, not a default: **the cheap rungs run first**
so the expensive one is justified by measurement rather than by plan. Rung 3 is the only one
that needs the 2B base model weights (now cached on disk), and its training trajectory is
governed by `docs/train-plan-2026-09-28.md`.

Rungs are not exclusive. A rung that clears the gates ships; the ladder exists to find the
cheapest one that does.

## Where Lappi is actually different

Three things, and they are the reason this is worth building next to four existing systems.
Each is load-bearing because **none of them can be added afterwards**.

**1. Calibrated abstention.** Every published peer either disclaims calibration or fits a
single temperature. Nimble's README says plainly that 0.9 does not mean right 90% of the
time. `decider-2b` does fit one, and its ECE goes **0.037 in-task to 0.084 held-out** — it
more than doubles exactly where a caller needs it most. Lappi uses split-conformal prediction
on the margin, which yields a coverage *guarantee* rather than a number that looks like one.
`python/qd_train/calibration_fit.py` fits the table; `crates/qd-runtime/src/calibration.rs`
serves it.

**2. Abstention as a reserved row, not a catch-all option.** Nimble documents adding an
answer meaning "no match". `decider-2b` uses "optional catch-all options". Both put
abstention into competition with the real answers for probability mass, which entangles the
abstain rate with the label distribution. Lappi reserves a row — scored in the same softmax
so it genuinely competes, but never occupying an option slot. `RESERVED_NOUL_ROWS` is pinned
against the Rust source, and the row is **last**, which `answer.rs` and `heads.py` both
assert.

**3. Span grounding.** *Which line is the evidence* is a different question from *which
label*, and no public System-1 model answers it. `crates/qd-mutate` produces span labels by
construction from a tree-sitter edit, so the supervision is a property of the edit rather
than something a separate model has to verify.

## What rung 0 changed about the hardest open problem

The worst open gap family in this repo was `GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED` and
`GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE`: does "the token that starts line N" survive a BPE
tokenizer that merges across whitespace? S4 predicted its refusal would fire widely the first
time a real tokenizer ran.

In byte space the question does not arise. A line start **is** a byte offset.
`python/qd_train/byte_context.py` is exact and total, and its tests are exhaustive rather
than sampled. The gap family is **dissolved for rung 0** — and it remains open for rung 3,
which still tokenizes.

That is the clearest argument for the ladder: rung 0 is not only cheaper, it avoids a problem
rung 3 has to solve.

## What is honestly not claimed

`cua-s1-forms` reaches near-ceiling on **form filling** — a 224-byte context over a
55-concept catalogue. "Is this commit a refactor, and which line proves it" is a wider
problem. Nothing here claims 600K parameters covers it. The claim is narrower and testable:
rung 0 is trainable on this Mac this week, it removes the line-mapping problem, and it
establishes a floor that rung 3 must beat to justify its cost.

If rung 0 underperforms, the first thing to raise is context capacity, and
`ByteDeciderConfig` says so at the point where someone would otherwise rediscover it.

## Licence discipline

`bespokelabsai/nimble` carries **no licence**, so its 2,676-row training set and 324-row eval
set are visible and unusable. `qd_data.licences` refuses the source and that refusal is
correct. `cua-s1-forms` (MIT) and `decider-2b` (Apache-2.0) are usable; Nimble is readable for
method only. This distinction is recorded because the tempting mistake is to treat "public on
GitHub" as "available".
