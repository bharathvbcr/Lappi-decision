# Handoff — context-source lane, 2026-09-22

## One paragraph

Rung 0's choice head is asked *"what kind of change is this?"* and was shown only the file
the change produced. Fixing that turned out to require fixing the corpus first: the obvious
fix — read the diff — was contaminated, because `clean` examples were emitted with an empty
diff and on this corpus an empty diff meant `clean` and nothing else. The leak is now closed
at the engine, the corpus is rebuilt, and the effect survives its own correction at **+25.1
points** on the linear control at full scale. The model, meanwhile, loses to that control.

## What was measured

All numbers from `AUDIT/after-vs-diff-leak.md` (sample) and the box's cached controls (full
scale). Ledger rows in `ledger/gh200-context-source-2026-09-22.jsonl`.

### The leak, established before it was acted on

`crates/qd-mutate/src/generate.rs` emitted every `clean` row with `before == after` and
`diff: String::new()`. Over all 50,178 examples of the v1 commitpackft corpus:

| | count |
| --- | --- |
| rows with an empty diff | 8,450 |
| of those, `clean` | 8,450 |
| `clean` rows with a non-empty diff | 0 |

An exact biconditional. The four-class control reading diffs scored `clean` **353/353 =
100.00%**. The 38-point gap recorded in `GAP-RUNG0-CHOICE-SEES-THE-POST-IMAGE-NOT-THE-DIFF`
was measured over all four classes and could not have meant what it said.

### The number the leak cannot reach

Three mutated classes only — every row has a non-empty diff, so context length carries no
information. 9,000-row sample, repo-disjoint split:

| | post-image | unified diff | delta |
| --- | --- | --- | --- |
| cosmetic | 20.18% | 87.16% | +66.98 |
| logic | 42.23% | 92.14% | +49.91 |
| stub | 95.03% | 98.41% | +3.38 |
| **overall** | **67.53%** | **94.74%** | **+27.22** |

Majority prior on that set: 54.95% (`stub`).

Confirmed at full scale by the two cached controls on `commitpackft-mutated` (41,728 rows):

| control | rows scored | top-1 | fit |
| --- | --- | --- | --- |
| `--context-source after` | 10,613 | **69.3%** | 349.1s, 399 iters, grad 9.961e-05 |
| `--context-source diff` | 10,635 | **94.4%** | 331.0s, 519 iters, grad 9.963e-05 |

**+25.1 points**, on a corpus where the leak cannot operate. The finding survives.

### The gate: 16 seeds, two representations, 0 wins

`paired_margin_vs_linear`, `ledger/gh200-context-source-2026-09-22.jsonl`, 10,000 bootstrap
resamples per row:

| mode | n | mean margin | range | positive | CI touches zero |
| --- | --- | --- | --- | --- | --- |
| `after` | 8 | **−0.0866** | −0.0922 … −0.0798 | **0/8** | never |
| `diff` | 8 | **−0.1047** | −0.1234 … −0.0926 | **0/8** | never |

Every row reads *"the whole interval is below zero, so this is not an inconclusive result:
the baseline beats the model by N and the comparison separates them"* — the three-outcome
verdict added in `e952af3`, doing the job it was added for.

**The margin does not close when the model is given the diff. It widens.** Reading the
change is worth ~+24 points to the model and ~+25 to the control, so the deficit is
untouched: the diff fixes the *task*, not the model. That is the finding this lane ends on.

A further 8 `after` rows carry `paired_margin_vs_linear` as **not_run** — the first arm ran
before the control cache existed and the in-process fit was refused at 39.8 hours against a
15-minute budget. Those rows are honest and are not counted above.

### The four-way task, runnable for the first time

`commitpackft-corpus-v2` (50,177 examples, all four classes, no empty diff anywhere),
`ledger/gh200-fourway-2026-09-22.jsonl`:

| mode | n | paired margin | range | positive | model top-1 | control |
| --- | --- | --- | --- | --- | --- | --- |
| `after` | 13 | **−0.0764** | −0.088 … −0.065 | **0/13** | 50.20% | 57.8% |
| `diff` | 8 | **−0.1258** | −0.148 … −0.102 | **0/8** | 75.93% | 88.5% |

Mean choice accuracy over the 46.0% majority prior: `after` **+4.1%** (spread 2.3), `diff`
**+30.0%** (spread 4.7).

Five of the 13 `after` rows come from a superseded run under `d0de223`, whose span END
metric was only half decomposed. That does not touch the paired margin or the span START
figure, which are identical in both; the rows are pooled for those and nothing else.

**Across all four arms — two corpora, two representations, 29 completed rows — the model
wins zero times and no confidence interval touches zero.**

### Capacity, re-taken on a task that has signal in it

`b72aa54` found accuracy falling monotonically in width and it was measured on the
post-image task, which carries almost no evidence a change occurred. More capacity fitting
more noise says nothing about the architecture. Re-taken on `--context-source diff`, where
the control reaches 88.5% — `ledger/gh200-diff-capacity-2026-09-22.jsonl`:

| width | n | top-1 | paired margin | worst | positive | seed spread |
| --- | --- | --- | --- | --- | --- | --- |
| 128×4 | 8 | 76.33% | −0.1218 | −0.1470 | 0/8 | 5.1 pts |
| 256×4 | 8 | 70.89% | −0.1761 | −0.2464 | 0/8 | 10.3 pts |
| 512×4 | 8 | **58.14%** | **−0.3037** | −0.5129 | 0/8 | **29.7 pts** |

**The earlier finding was not an artifact of a degenerate task.** More capacity is worse on
a task with abundant signal, and the margin degrades 2.5× from 128 to 512. That is the
measured answer to *"do we need a larger cluster"*: **no**.

With one confound still open, which is why the LR sweep is running. The seed spread grows
with width — 5.1 → 10.3 → 29.7 points, with width-512 seeds at 37.2% and 66.9% in the same
arm. A model too large for its data degrades *smoothly*; an optimiser over-stepping degrades
*erratically*. The peak LR was a literal in two places (`"lr": 3e-3` in the recipe and
`peak_lr=3e-3` in `train_once` — so the recipe stated a rate nothing obliged the run to
use), which is why this could not be told apart before. `--lr` now exists, defaulting to the
old value so no recorded row moves.

### The span head collapses on the four-class corpus

On `commitpackft-mutated`, where every row points, the span head reached 9.5% against 3.2%
chance — the one head in rung 0 that was generalising. On `commitpackft-corpus-v2`, where
17% of rows abstain, it reaches **1.50%**: *below* chance, on 13 rows. Span loss rises
during training (2.581 → 3.951) rather than falling.

Asked to learn both where to point and when not to, at `--span-weight 0.05`, it learns the
abstention and stops pointing. This was invisible until the pointing decomposition landed:
the undecomposed figure read 9.5% on the same runs, because the abstentions it got right
were counted as pointers it placed correctly.

In `--context-source diff` the span metric is `NotRun` by construction — span offsets are
into `after` and encoding the diff moves every byte, so there is no pointer to score. The
log says `span NOT MEASURED (all 12792 scored rows abstain)`.

### The model, on the same corpus

`after` arm, 8 seeds, 128x4x2, 3 epochs, span-weight 0.05. First seeds:

```
seed=0  choice val 46.7% -> 60.3%  (baseline 55.4%)  span start 9.5% (chance 3.2%)
seed=1  choice val 51.8% -> 60.7%  (baseline 55.4%)  span start 9.8% (chance 3.2%)
```

~60.5% against its own control's **69.3%**. The choice head clears the majority prior by ~5
points and loses to a char-n-gram logistic regression by ~9. The span head continues to be
the one that generalises: 9.5% against 3.2% chance.

**Caveat on pairing, stated because it is not zero.** `after` mode refuses 51 train / 22 val
rows on `PhantomFinalLine`; diff mode refuses none, because it suppresses span supervision.
So the two arms' validation sets differ by 22 rows in 10,635 (0.2%). Immaterial against a
25-point effect, but the arms are not strictly paired and no paired statistic should be
computed across them.

## What changed

| commit | what |
| --- | --- |
| `839a8ce` | `--context-source {after,diff}`, `is_void_diff`, `EmptyDiffContext`, `refuse_leaky_diff_corpus`, `recipe_of`, `tools/after_vs_diff.py`, `tools/filter_corpus.py` |
| `2c84587` | `RunRecorder` records the closure digest in `__enter__`, so killed and failed runs carry it too |
| `1b75f58` | `PoolRecord.prior_source`; `clean` becomes the agent's own diff; pre-image normalized before diffing; `Refusal::CleanDiffEmpty` |

`--context-source` defaults to `after` and is **omitted from the recipe hash at its
default**, so every post-image row stays comparable with the 61 capacity rows and the
learning curve already in the ledger, and no `diff` run can collide with one.

The v2 corpus (`data/pool/commitpackft-corpus-v2/`, 50,177 examples, sha256
`9ad8f17fad5e71f0190eae4b148959c2740d82832ba895c68fd2cf85f21b9824`) passes
`refuse_leaky_diff_corpus` on all **four** classes: zero empty diffs, zero rows with CR
surviving in `before`, clean-diff lengths overlapping the mutated classes.

## What is open

| gap | owner | what it blocks |
| --- | --- | --- |
| `GAP-RUNG0-EMPTY-DIFF-IS-THE-CLEAN-LABEL` | human | whether rung 0 reads diffs at all. The mechanism is built and defaults off. |
| `GAP-RUNG0-CHOICE-SEES-THE-POST-IMAGE-NOT-THE-DIFF` | human | superseded in magnitude by the above; its 38-point figure is corrected to +25.1 |
| `GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE` | human | unchanged |
| `GAP-LEDGER-A-KILLED-RUN-LOSES-ITS-PROVENANCE` | agent | **closed** by `2c84587` |

**The style asymmetry, which is now the one that matters.** With the leak closed, clean
diffs are real human commits and mutated diffs are synthetic single-operator edits. A model
may separate them by style rather than by defect. That is the known weakness of
mutation-derived corpora; it was previously hidden behind the length leak and is now the
load-bearing caveat on any four-way diff number. Nobody has measured it.

**Four gates still never reach a row**: `ood_abstain` (unspecified), `needle_hunk_recall`
(built, blocked on a span→hunk-index mapping a human must define), `privileged_hunk`
(measured near-vacuous on this corpus — the window is the whole file on 48.0% of examples),
`transfer_gate` (specified in `docs/hardening.md` §2, implementable).

## Is it ready for final train?

No, and the blocker is not one this lane can clear.

* **The model loses to a char-n-gram logistic regression in every configuration measured**:
  two corpora, two representations, three widths, 53 completed rows, **0 wins**, no
  confidence interval touching zero.
* **Reading the diff does not close the gap.** It is worth ~+25 points to the model and
  ~+31 to the control. The task was mis-encoded *and* the model is beaten; fixing the first
  does not touch the second.
* **Scale makes it worse**, monotonically, on a task with signal in it.
* **The span head — the one head that was generalising — collapses** below chance on any
  corpus it must also abstain on.

What this lane did clear: the corpus is now correct, the encoding is selectable and
recorded, the gates report `not_run` instead of vacuous passes, and every row says what code
produced it. Those were real defects and they were masking the result rather than causing
it.

What remains is a design decision about the choice objective or the architecture, which
`CLAUDE.md` rule 2 makes a human's call and which no further arm of this shape will inform.

## The exact first command for the next lane

The v2 corpus is already on the box at `/home/ubuntu/commitpackft-corpus-v2/`. Fit the
four-way diff control off the GPU first — the in-process one refuses itself at this corpus
size (39.8 hours projected against a 15-minute budget), which is the guard working:

```bash
/home/ubuntu/qd-venv/bin/python -u /home/ubuntu/qwen-decision/tools/fit_linear_control.py \
  --examples /home/ubuntu/commitpackft-corpus-v2/examples.jsonl \
  --control-cache /home/ubuntu/control-cache \
  --context-source diff --width 128 --heads 4 --layers 2 --dense-budget-gb 24.0
```

Then the four-way pair, which is what `tools/launch_context_source_arms.sh` does with
`CORPUS` repointed at `commitpackft-corpus-v2`. Sync first (`bash tools/sync_box.sh`); it
refuses while a training process is live, which is correct and is why the arms above ran on
the `839a8ce` closure rather than today's.
