# The control classifies by generator, not by change

Measured 2026-09-22 by `tools/operator_holdout.py` on `commitpackft-corpus-v2`. Raw output
in `AUDIT/operator-holdout.log` (untracked per `.gitignore`); every number below is quoted
from it.

## Why this was run

A char-n-gram logistic regression reaching **88.5%** on the four-way diff task should be
suspicious. Its per-class ordering said why: stub 98.1%, logic 89.5%, cosmetic 69.3%, clean
68.5% — which tracks how distinctive the mutation operator's textual fingerprint is, and
puts the two classes with no fingerprint at the bottom.

`qd-mutate` applies exactly one operator per example, so `logic.widen_comparison` leaves
`-  x > 0` / `+  x <= 0` in the diff. Hashed char n-grams can read that directly without
representing anything about the change.

## The design

Three conditions, all scoring the **same** validation rows — the rows produced by one
operator, on the far side of a repo-disjoint split:

| condition | training set |
| --- | --- |
| **seen** | the full repo-disjoint training set, as every arm so far |
| **matched** | the same set with an equal *number* of rows dropped at **random** |
| **unseen** | the same set with every row produced by that operator removed |

`matched` is the load-bearing control. `stub.panic` alone is 43% of this corpus, so without
it a collapse is explained just as well by "half the training data is gone" as by "the
fingerprint is gone". Other operators of the same class stay in training in all three
conditions, so the label remains reachable: this asks whether the control can recognise
`logic` from a `logic` operator it has never seen.

## The result

| operator held out | seen | matched | unseen | drop vs matched | volume effect |
| --- | --- | --- | --- | --- | --- |
| `stub.panic` (43% of train) | 99.32% | 99.10% | **3.08%** | **−96.02** | +0.23 |
| `logic.change_constant` (14%) | 91.12% | 91.12% | **19.63%** | **−71.50** | 0.00 |
| `cosmetic.rename_local` (9%) | 58.33% | 58.71% | **0.00%** | **−58.71** | −0.38 |

Removing up to 43% of the training set at random costs the control **0.23 points at most,
and on one operator nothing at all**. Removing the operator costs it everything. On
`cosmetic.rename_local` the unseen condition classifies **0 of 264 rows** correctly — not
degraded, not at chance, zero.

## What follows

**The control has memorised generator fingerprints.** It is not representing "this change
stubs out a function"; it is matching the string the stub operator inserts.

That reframes the result this lane spent its GPU time on. `paired_margin_vs_linear` compares
rung 0 against this control, and rung 0 loses by 8–12 points on every arm. Read literally
that says the model is worse than logistic regression at the task. Read against this table
it says the model is worse than logistic regression **at memorising mutation operators**,
which is a much weaker claim and not obviously one worth acting on.

It also means **this corpus cannot separate the two hypotheses it was built to separate**.
A model that scored 95% here might have learned what a logic defect is, or might have
learned seventeen regular expressions. Nothing in the current gate set can tell those apart
on synthetic data — which is precisely what the **transfer gate** was specified to catch:
*"a model trained on mutations only must beat the char-n-gram baseline on the natural
held-out set"* (`docs/teacher-plan.md` §7). That gate has never run, because the natural
held-out set does not exist
(`GAP-TRANSFER-GATE-IS-NAMED-NOT-SPECIFIED-AND-HAS-NO-DATA`).

## What this does NOT say

It measures the **control**, not the model. The model may be doing exactly the same thing,
and this experiment does not check — the same holdout against a trained rung 0 needs a GPU
and has not been run. Until it is, "the model generalises better than the control across
operators" is unverified in both directions.

It also does not say the corpus is worthless. Operator-fingerprint memorisation is a real
shortcut but the corpus still carries genuine signal about hunks and spans; what it cannot
do is license a claim about semantic understanding from a top-1 number.
