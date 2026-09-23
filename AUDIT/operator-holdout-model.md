# The model classifies by generator too

Measured 2026-09-22 on `commitpackft-corpus-v2`, `--context-source diff`, width 256 /
2 layers / 4 heads at `lr 3e-4` — the best cell of the tuned-capacity grid
(`AUDIT/capacity-and-learning-rate.md`) — 3 epochs, batch 16, span-weight 0.05, 8 seeds per
arm. 144 `ft` rows in `ledger/gh200-operator-holdout-model-2026-09-22.jsonl`, rendered by
`tools/operator_holdout_report.py`: rows 1–72 hold out the dominant operator of each class,
rows 73–144 a minor one (see "The minor operators" below).

Reproduce the tables below with:

```bash
python tools/operator_holdout_report.py ledger/gh200-operator-holdout-model-2026-09-22.jsonl
```

## Provenance

| field | rows 1–72 (dominant operators) | rows 73–144 (minor operators) |
| --- | --- | --- |
| launch rev (`recipe.rev`) | `ecbd370a4d40a6ddc3d308b389cc1112a5e3cd30` | `728562b3da609a3aa3ad35efaa597c870e7ba1fe` |
| `code_that_ran` (what_ran_state) | `5bad419d1bb1b7ea…` | `7318cbb3212723c3…` |
| `code_commit` | `8b9df39bf816d149efc108e3581ab4ba8192688f-dirty` on both (see GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN) | |
| `data_snapshot_hash` | `728eee3743dbed57…` (61,193 pool records) | same |
| host / device | `192-222-58-240`, cuda (1×GH200) | same |
| torch / transformers | `2.10.0+cu128` / `5.17.0` | same |
| fla / causal_conv1d | `0.5.2` (triton 3.7.1) / `1.7.0` | same |
| rows / seeds | 72 = 9 arms × 8 seeds, all `completed` | 72 = 9 arms × 8 seeds, all `completed` |
| cost / wall clock | $2.39, 5,781.4s summed across rows | $2.96, 7,158.1s (seeds slowed while control fits shared the CPU) |

The two runs ran different code (`5bad419d` vs `7318cbb3`), and no comparison below crosses
them: every operator's three arms come from one run. The report checks this rather than
relying on it — an operator whose arms ran different code gets an `ARMS RAN DIFFERENT CODE`
line above its differences — and on these 144 rows it prints none.

**Every one of the 144 rows carries `quick: true`.** Under rule 8 they promote nothing, and
this document does not ask them to. The flag is not describing a truncated run — the
recorded reason says so in as many words:

> no rule-8 condition this run can determine stands: 8 seeds, full schedule, no subsample,
> real labels, corpus supplied as a pre-generated set (data_snapshot 728eee3743dbed57,
> 61193 pool record(s)). Recorded quick nonetheless — whether that corpus is the pool the
> plan names is not something this run can verify, and clearing the flag is a promotion
> decision that rule 2 makes a human's, not an agent's.

So: this is a **diagnostic**, and a strong one. Turning it into a promotion or a kill is a
human's decision, and clearing the `quick` flag is part of that decision rather than a
precondition for reading the result.

## The question this closes

`AUDIT/operator-holdout.md` showed the char-n-gram control collapsing when one mutation
operator is removed from its training set, and said in its own last section what it could
not say:

> This measures the CONTROL, not the model. If the control collapses, the 88.5% is inflated
> by generator recognition and the margin against it is not a clean statement about the
> architecture — but the model may be doing exactly the same thing, and this does not check.
> The same experiment against the trained model needs a GPU and is not this tool.

This is that experiment. The design is deliberately the control's, arm for arm, so the two
tables can be read side by side.

## The design

Three arms per operator, all scoring the **identical** validation rows — validation is never
filtered, so the split is a function of the corpus and `--val-share` alone:

| arm | training set |
| --- | --- |
| **reference** | the full repo-disjoint training set |
| **sizematch** | the same set with an equal *number* of rows dropped at random |
| **holdout** | the same set with every row that operator produced removed |

`sizematch` controls volume. Without it a collapse is explained as well by "the training set
shrank" as by "the fingerprint went", and `stub.panic` alone is 43% of this corpus.

### The fourth number, which is what makes the other three readable

Every operator in this corpus produces exactly one class, so holding one out also moves the
**class prior** — removing `stub.panic` takes `stub` from 46% of training to about 7%. A
model that then under-predicts `stub` would score badly on the held-out rows for a reason
with nothing to do with recognising generators, and `sizematch` cannot separate the two
because dropping uniformly *preserves* the prior it would need to disturb.

So every arm also reports its operator's **siblings**: validation rows of the same class
from a *different* operator, which training did see and which sit under the identical
shifted prior. Siblings holding while the operator collapses is the reading that means
"generator"; both collapsing would mean "the arm moved the prior".

### The language confound, ruled out separately

`stub.panic` suggests Rust. It is not: measured, it is 89% Python. And the two arms are
matched in language, not merely in row count — `tools/operator_holdout_language_check.py`
on the v2 training split:

| language | in train | cut by holdout | cut by sizematch |
| --- | --- | --- | --- |
| python | 31,774 | 14,368 | 13,678 |
| go | 2,213 | 486 | 919 |
| rust | 1,804 | 507 | 749 |
| typescript | 1,594 | 676 | 691 |
| **total** | **37,385** | **16,037** | **16,037** |

Both arms cut 16,037 rows; the Python difference is 690 rows, 2% of the Python training
set, and the holdout arm cuts *fewer* Go and Rust rows than its control.

## Results

Choice top-1 on the operator's own validation rows, mean ± sd over 8 seeds. Validation is
identical across all three arms of an operator.

### stub.panic — 5,448 validation rows, 427 sibling rows

| arm | on its own rows | siblings | overall | paired margin vs control |
| --- | --- | --- | --- | --- |
| reference | 81.97% ± 2.05 | 60.57% ± 6.45 | 80.57% ± 1.03 | −7.94% ± 1.03 (0 of 8 positive) |
| **holdout** | **15.44% ± 3.66** | **75.09% ± 5.18** | 51.74% ± 2.07 | not measured (8 seeds) |
| sizematch | 80.50% ± 3.36 | 59.63% ± 5.29 | 78.92% ± 1.43 | not measured (8 seeds) |

Total effect **+66.53pp**; siblings **−14.52pp** (they *rose*); volume alone **+1.47pp**.
The operator fell **81.05pp further than its siblings** under the identical prior.

### logic.change_constant — 1,786 validation rows, 1,128 sibling rows

| arm | on its own rows | siblings | overall | paired margin vs control |
| --- | --- | --- | --- | --- |
| reference | 86.90% ± 3.01 | 84.93% ± 3.87 | 80.60% ± 1.42 | −7.91% ± 1.42 (0 of 8 positive) |
| **holdout** | **28.86% ± 4.98** | **78.86% ± 4.57** | 71.48% ± 1.69 | not measured (8 seeds) |
| sizematch | 88.49% ± 1.74 | 87.10% ± 2.76 | 81.36% ± 0.76 | not measured (8 seeds) |

Total effect **+58.03pp**; siblings **+6.07pp**; volume alone **−1.59pp** (dropping rows at
random slightly *helped*). The operator fell **51.96pp further than its siblings**.

### cosmetic.rename_local — 1,127 validation rows, 719 sibling rows

| arm | on its own rows | siblings | overall | paired margin vs control |
| --- | --- | --- | --- | --- |
| reference | 77.56% ± 3.27 | 66.88% ± 3.15 | 80.39% ± 1.74 | −8.12% ± 1.74 (0 of 8 positive) |
| **holdout** | **0.45% ± 0.14** | **70.60% ± 4.53** | 76.70% ± 1.28 | not measured (8 seeds) |
| sizematch | 77.47% ± 2.20 | 67.39% ± 4.72 | 80.64% ± 1.77 | not measured (8 seeds) |

Total effect **+77.11pp**; siblings **−3.72pp**; volume alone **+0.09pp**. The operator fell
**80.83pp further than its siblings**.

`0.45% ± 0.14` over 1,127 rows and 8 seeds is the sharpest number in this table. It is not
confusion — a confused model scores near the class prior. The model systematically assigns
these rows elsewhere the moment it has not seen the generator that made them.

## The minor operators

The three operators above are each the dominant operator of their class — 42.9%, 14.2% and
9.0% of the 37,385 training rows — so holding one out moves the class prior as hard as this
corpus allows, and the sibling metric has to do real work to rule the prior out. Holding out
a **minor** operator leaves its class's dominant operator in training and barely moves the
prior: `logic.negate_condition` is 1,995 training rows (5.3%), `cosmetic.edit_comment` 1,837
(4.9%), `stub.default_return` 670 (1.8%). A collapse there has little else to be.

Rows 73–144, same recipe, same validation rows, launched at `728562b3`.

### logic.negate_condition — 737 validation rows, 2,177 sibling rows

| arm | on its own rows | siblings | overall | paired margin vs control |
| --- | --- | --- | --- | --- |
| reference | 88.91% ± 3.79 | 85.38% ± 3.50 | 80.88% ± 1.15 | not measured (8 seeds) |
| **holdout** | **45.56% ± 5.03** | **84.99% ± 2.55** | 79.66% ± 0.95 | not measured (8 seeds) |
| sizematch | 88.52% ± 4.05 | 85.42% ± 3.43 | 79.79% ± 2.07 | not measured (8 seeds) |

Total effect **+43.35pp**; siblings **+0.40pp**; volume alone **+0.39pp**. The operator fell
**42.96pp further than its siblings**. The reference arm has no margin for a reason unrelated
to the holdout: it started at 17:25:25, four minutes before the full-corpus control landed in
the cache at 17:29:13, missed it, and recorded the fallback's refusal (a 25.7-hour worst-case
projection against the 900s inline budget).

### cosmetic.edit_comment — 600 validation rows, 1,246 sibling rows

| arm | on its own rows | siblings | overall | paired margin vs control |
| --- | --- | --- | --- | --- |
| reference | 78.56% ± 2.69 | 71.86% ± 1.98 | 80.60% ± 1.88 | −7.91% ± 1.88 (0 of 8 positive) |
| **holdout** | **1.00% ± 1.04** | **76.59% ± 4.48** | 77.58% ± 1.05 | not measured (8 seeds) |
| sizematch | 77.77% ± 3.45 | 73.01% ± 2.62 | 80.85% ± 1.93 | not measured (8 seeds) |

Total effect **+77.56pp**; siblings **−4.73pp** (they rose); volume alone **+0.79pp**. The
operator fell **82.29pp further than its siblings** — the largest gap of the six, from an
operator that is under 5% of training.

### stub.default_return — 226 validation rows, 5,649 sibling rows

| arm | on its own rows | siblings | overall | paired margin vs control |
| --- | --- | --- | --- | --- |
| reference | 81.69% ± 3.52 | 80.33% ± 3.05 | 80.42% ± 1.75 | −8.09% ± 1.75 (0 of 8 positive) |
| **holdout** | **76.71% ± 3.39** | 78.27% ± 3.28 | 79.58% ± 1.41 | not measured (8 seeds) |
| sizematch | 80.37% ± 5.36 | 79.74% ± 5.08 | 80.40% ± 2.79 | not measured (8 seeds) |

Total effect **+4.98pp**; siblings **+2.06pp**; volume alone **+1.33pp**. The operator fell
only **2.91pp further than its siblings**, inside either arm's seed spread. **This is the one
operator of six the model still recognises without having seen it.** Its siblings are 5,649
rows of which 5,448 are `stub.panic` (the two sibling counts reconcile: 427 − 226 = 5,649 −
5,448 = 201 rows from the class's other operators), so the holdout arm still trained on
`stub.panic`. What these rows cannot say is *why* the stub signal transfers — whether the
model reads "a body replaced by one line" or whatever textual features the two stub
generators happen to share. Both readings predict this table.

## The reading

Five of the six operators say the same thing, and the three controls each rule out the
competing explanation independently:

- **Volume is not it.** Across all six operators the size-matched arm moves the score by at
  most 1.59pp in either direction, against total effects of 43–78pp on the five that
  collapse.
- **The class prior is not it.** Siblings — same class, same shifted prior, seen in
  training — hold in every case: the largest sibling drop is 6.07pp, and in three of the
  five they *rise*. The minor operators make this sharper still, because holding one out
  barely shifts the prior to begin with: `logic.negate_condition` collapsed 43.35pp with its
  siblings moving 0.40pp.
- **Language is not it — for `stub.panic`.** Measured above, on that operator's arms' own
  training sets, because it was the operator for which a language confound was plausible.
  It was not measured for the other five.

What is left is the generator's textual fingerprint. **The model classifies by recognising
the mutation operator that produced the row, not by reading the change.** It shares the
char-n-gram control's shortcut.

The bound on that statement is `stub.default_return`: held out, it still scores 76.71%
against 81.69%, because its class's dominant operator stayed in training and whatever the
model learned from `stub.panic` transfers to it. So the shortcut is not "one memorised
string per operator". It is "whatever text the training generators share". That is no
better news for the claim that the model reads changes: the one transfer found is between
two generators of the same stub-insertion family.

### What this does and does not license

It **does** mean the 120+ ledger rows reading "the model loses to the control by ~7.5
points" were comparing two fingerprint-matchers on a corpus that cannot distinguish
fingerprint-matching from understanding. Those margins are reproduced here — −7.94%,
−7.91%, −8.12%, 0 of 8 seeds positive in every reference arm, on a different day and a
different code version from the grid that first produced them. They are real and they are
stable; they are just not about architecture.

It **does not** mean rung 0 is the wrong architecture. This is a **task and data finding**.
Nothing here measures what the model would do on a corpus whose changes were not produced
by a small set of programmatic operators.

### The comparison that is still missing

Every holdout and size-matched arm carries `paired_margin_vs_linear: not_run` — 48 of rows
1–72 and 48 of rows 73–144, plus the 8 `ref-logic.negate_condition` rows that missed the cache
by four minutes: 104 of 144. That is not a loss and must never be read as one. It means no
control was cached for those arms' training sets when they started, so the model had no
opponent at all; each row records the reason, and the report prints it.

This is the one number that could still overturn the headline. Under holdout the control
collapses on `stub.panic` while the model holds 15.44%. A control fitted on the arm's own
training set and scored on its own validation rows is what makes the comparison real, and it
would be the first configuration in this project where rung 0 wins.

**The fits are now done.** `tools/fit_operator_holdout_controls.sh` ran on 2026-09-22 —
seven fits, 204.2s to 351.7s each, 2,190.4s total (36.5 minutes, ~$0.91). Every one
converged; all seven verdicts are in `/home/ubuntu/control-cache`. Overall control accuracy
on the same 12,792 validation rows:

| training set | control overall | model overall (from the table above) |
| --- | --- | --- |
| full | 88.5% | 80.57% / 80.60% / 80.39% (the three reference arms) |
| holdout `stub.panic` | 50.0% | 51.74% |
| sizematch `stub.panic` | 88.6% | 78.92% |
| holdout `logic.change_constant` | 80.2% | 71.48% |
| sizematch `logic.change_constant` | 88.5% | 81.36% |
| holdout `cosmetic.rename_local` | 82.9% | 76.70% |
| sizematch `cosmetic.rename_local` | 88.6% | 80.64% |

**Do not read the right-hand column as a margin.** Those are two columns from two tools;
`paired_margin_vs_linear` is a per-seed paired statistic and only the runner computes it.
The `holdout stub.panic` row is the one to watch — 51.74% against 50.0% is the first time
the model's number has been the larger one — but it is a difference of means across tools,
not a measured margin, and it is well inside the seed spread (±2.07).

Getting the real number needs the arms **re-run**. `rung0_real_run.py` reads the cache once,
before its seed loop. On a miss it fits inline only when the fit projects under its 900s
`LINEAR_CONTROL_TIME_BUDGET_S`, and on this corpus every arm projected 14.7 to 25.7 hours, so
every miss recorded `not_run`. The 144 rows above are final and can never gain margins
retroactively.

### Run 3: the comparison, measured

The re-run landed: `ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl`, 144 `ft` rows
= 18 arms × 8 seeds, all `completed`, the chain verifies. Launch rev (`recipe.rev`)
`52c8fe9fc39acbd5e3aa2b859c38637d2696a586`, `code_that_ran` `7318cbb3…` on every row (the
same closure as rows 73–144). One `data_snapshot_hash`, `728eee3743dbed57…`, the same as
rows 1–144. $5.49, 13,261.8 s summed across rows. Every row carries `quick: true` with the
same reason quoted under Provenance. It is its own ledger, so its seeds never pool with rows
1–144; the report refuses to pool them if both are passed together. Reproduce with:

```bash
python tools/operator_holdout_report.py ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl
```

**`paired_margin_vs_linear` ran on all 144 rows**, against 40 of 144 before. The model's
paired margin against a linear control fitted on each arm's own training set (mean over 8
seeds, ± the report's seed spread, seeds with a positive margin):

| operator | reference | holdout | sizematch |
| --- | --- | --- | --- |
| `stub.panic` | −7.62% ± 1.07 (0/8) | **+1.92% ± 2.04 (6/8)** | −9.65% ± 1.33 (0/8) |
| `logic.change_constant` | −7.87% ± 1.47 (0/8) | −8.43% ± 2.47 (0/8) | −7.14% ± 0.98 (0/8) |
| `cosmetic.rename_local` | −7.79% ± 1.59 (0/8) | −6.38% ± 1.15 (0/8) | −8.45% ± 2.32 (0/8) |
| `logic.negate_condition` | −7.74% ± 1.75 (0/8) | −9.99% ± 1.16 (0/8) | −8.62% ± 1.86 (0/8) |
| `cosmetic.edit_comment` | −8.04% ± 2.22 (0/8) | −5.92% ± 0.63 (0/8) | −7.73% ± 1.97 (0/8) |
| `stub.default_return` | −7.72% ± 1.30 (0/8) | −8.78% ± 1.25 (0/8) | −8.69% ± 2.62 (0/8) |

**The headline stands.** The model trails its linear control in 17 of 18 arms, by 5.92 to
9.99 points, and in every one of those arms all 8 seeds trail. That includes every holdout
and size-matched arm the earlier rows could not score. The one arm it leads is holdout
`stub.panic`: +1.92 points, 6 of 8 seeds positive. That is the arm where the control
collapses to 50.0% overall when its dominant generator is removed, and the model holds 51.97%.
It is a lead the size of its own seed spread, in the single configuration where the
opponent had lost its shortcut. It is not evidence that rung 0 reads changes the control
cannot. The earlier "difference of means across tools" (51.74% against 50.0%) was the right
sign. The paired statistic confirms the sign and not a margin anyone should act on.

The per-arm accuracies reproduce rows 1–144: the dominant operators collapse under holdout
(`stub.panic` 82.44% → 15.52%, `logic.change_constant` 86.72% → 28.77%,
`cosmetic.rename_local` 77.33% → 0.50%) while siblings hold. Two of the three minor
operators collapse too (`logic.negate_condition` 88.70% → 43.39%, `cosmetic.edit_comment`
77.73% → 0.92%). `stub.default_return` transfers (82.36% → 76.66%, 3.44 points beyond its
siblings). The reading above does not change.

## Gaps this leaves open

- `GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD` — unrelated to the choice head
  measured here, being addressed by the span-in-diff arms.
- `GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN` — `code_commit` on these rows is a dirty
  sha from an older commit; `recipe.rev` is what pins the launch.
- `GAP-CONTROL-CACHE-GOES-COLD-ON-ANY-BASELINE-EDIT` — why the cache had to be rebuilt at
  all, and why the digest that forces it must not be weakened. The rebuild turned out to
  cost 36.5 minutes, not the ~15 hours first recorded there: that figure came from
  `projected_fit_seconds`, which prices `max_iter` iterations and overshot by 17.6×.
- `GAP-THE-CONTROL-CLASSIFIES-BY-GENERATOR-NOT-BY-CHANGE` — moved to
  `resolved-with-residual` by this experiment. The residual was the 104 unmeasured margins,
  and run 3 measured them (see "Run 3: the comparison, measured").
