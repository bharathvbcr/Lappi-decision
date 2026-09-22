# The diff carries the change — and, for one class, the label

Measured 2026-09-22 by `tools/after_vs_diff.py`. Raw output in `AUDIT/after-vs-diff-leak.log`
(untracked, per `.gitignore`); every number below is quoted from it.

## What was being checked

`GAP-RUNG0-CHOICE-SEES-THE-POST-IMAGE-NOT-THE-DIFF` recorded that rung 0's choice head is
asked *"what kind of change is this?"* while being shown only `example.after` — the file the
change produced, never the change. It carried a measurement: same control, same rows,
**post-image 55.8%, unified diff 93.8%, a 38-point gap**.

That number was taken over all four classes, and over all four classes it cannot mean what it
appears to mean.

## The leak

`crates/qd-mutate/src/generate.rs:293` emits every `clean` example with
`before == after` and `diff: String::new()`. Over all 50,178 examples of the commitpackft
corpus:

```
empty diff, whole corpus, by class: {'clean': 8450}
clean rows WITH a non-empty diff:   0
```

So `diff == ""` ⟺ `class == "clean"` — an exact biconditional, measured, not assumed. In
`--context-source diff` a model can answer `clean` by measuring the **length** of its input
without reading a byte of it. The four-class comparison confirms it does exactly that:

```
      diff / unified: top-1 94.72% of 2293 rows
          clean 100.00%  (353/353)
      [353 validation row(s) carry an EMPTY context]
```

353 of 353. Not 99%.

This is not a corpus bug. A clean row genuinely has no change, so its unified diff is
genuinely empty. It is a defect in what the four-class comparison can be read to prove.

## The number the leak cannot reach

Restricting to the three mutated classes — every one of which has a non-empty diff, so an
empty context does not exist and the length signal carries no information:

| mutated-only, 5,601 train / 1,940 val | post-image | unified diff | delta |
| --- | --- | --- | --- |
| cosmetic | 20.18% (66/327) | **87.16%** (285/327) | +66.98 |
| logic | 42.23% (231/547) | **92.14%** (504/547) | +49.91 |
| stub | 95.03% (1013/1066) | 98.41% (1049/1066) | +3.38 |
| **overall** | **67.53%** | **94.74%** | **+27.22** |

Majority-class prior on that validation set: **54.95%** (`stub`).

So the post-image control clears the prior by 12.6 points; the diff control clears it by
39.8. **The finding survives the correction.** The recorded 38-point headline was inflated
by the leak — the honest figure is **+27.22 points** — but the direction and the magnitude
both hold, and they hold most strongly exactly where the post-image is least informative:
`cosmetic`, where renaming a local or reflowing a comment leaves a post-image
indistinguishable from any other file, goes from 20% to 87%.

## Both halves were reported

The four-class comparison is reproduced above rather than merely doubted, so the number in
the gap record can be checked against its replacement. `after_vs_diff.py` prints both
sections unconditionally; a measurement that only runs when it is expected to confirm
something is not a measurement.

## What was changed as a result

* `qd_train.mutate_adapter.is_void_diff` — one predicate for "this diff encodes to nothing".
* `to_decision(..., context_source="diff")` raises `EmptyDiffContext` on a void diff rather
  than encoding it. Its own exception type, because `decisions_of` counts refusals by name
  and this one is a statement about the corpus, not about a row.
* `refuse_leaky_diff_corpus` — a boundary pre-flight run before any training, so a rented
  GPU does not spend forty minutes parsing its way to the same refusal one row at a time.
  It names each affected class and flags any class that is *entirely* empty as
  perfectly separable.
* `recipe_of` puts `context_source` in the recipe hash — but only when it is not the
  default, so the 61 capacity rows and the learning curve already in the ledger stay
  comparable with new post-image rows.
* `tools/filter_corpus.py` derives `commitpackft-mutated` (41,728 rows, no `clean`), which
  the pre-flight passes. **A diagnostic subset, not the shipping task**: the option list is
  still four wide, so no run over it can discharge the four-way gate.

## After the repair: the style shortcut did not replace the length one

Re-taken on `commitpackft-corpus-v2` with the same tool
(`AUDIT/after-vs-diff-v2.log`), 9,000-row sample, repo-disjoint split. Every clean row now
carries the agent's own commit diff; `empty diff, whole corpus, by class: {}`.

| four classes, diff mode | v1 (leaky) | v2 (repaired) |
| --- | --- | --- |
| clean | **100.00%** (353/353) | **68.45%** (256/374) |
| cosmetic | 83.79% | 69.28% |
| logic | 91.04% | 89.47% |
| stub | 98.22% | 98.12% |
| **overall** | 94.72% | **87.15%** |

Confirmed at full scale by the cached v2 controls on the box: **after 57.8%** (12,770 rows)
against **diff 88.5%** (12,792 rows), a **+30.7 point** gap on the four-way task.

**This is the measurement the style caveat needed.** Clean diffs are real human commits and
mutated diffs are synthetic single-operator edits, so the worry was that a model would
separate them by style rather than by defect -- trading one shortcut for another. If that
were happening, `clean` would have stayed near 100%. It fell to 68.45%, which makes it the
*hardest* class in the set alongside `cosmetic`, and the overall figure fell by 7.6 points
-- almost exactly the clean class's share of the corpus times the accuracy it lost. The
leak's contribution is gone and nothing took its place.

The caveat is weakened, not discharged: 68.45% is well above the 16.8% prior for that class,
and none of this says *what* the control keys on. It says the trivially separable case is no
longer trivially separable.

## What is still open

The repair that makes the *four-way* task valid is to give `clean` rows the agent's own
commit diff — which is what `generate.rs`'s own comment has always claimed a clean example
is: *"the original agent diff, unmodified"*. It cannot do that today because `PoolRecord`
carries only `source` (the post-image) and `hunks`; the before-image is dropped by
`tools/build_commitpackft_pool.py`, which reads `row.old_contents` to compute the hunks and
then discards it.

That repair introduces a *different* asymmetry worth stating before anyone relies on it:
clean diffs would be real human commits while mutated diffs are synthetic single-operator
edits, so a model could learn to separate them by style rather than by defect. That is a
known weakness of mutation-derived corpora and it is a human's call, not an agent's.
