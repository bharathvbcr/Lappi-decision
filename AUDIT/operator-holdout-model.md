# The model classifies by generator too

Measured 2026-09-22 on `commitpackft-corpus-v2`, `--context-source diff`, width 256 /
2 layers / 4 heads at `lr 3e-4` — the best cell of the tuned-capacity grid
(`AUDIT/capacity-and-learning-rate.md`) — 3 epochs, batch 16, span-weight 0.05, 8 seeds per
arm. 72 `ft` rows in `ledger/gh200-operator-holdout-model-2026-09-22.jsonl`, rendered by
`tools/operator_holdout_report.py`.

Reproduce the tables below with:

```bash
python tools/operator_holdout_report.py ledger/gh200-operator-holdout-model-2026-09-22.jsonl
```

## Provenance

| field | value |
| --- | --- |
| launch rev (`recipe.rev`) | `ecbd370a4d40a6ddc3d308b389cc1112a5e3cd30` |
| `code_commit` | `8b9df39bf816d149efc108e3581ab4ba8192688f-dirty` (see GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN) |
| `data_snapshot_hash` | `728eee3743dbed57…` (61,193 pool records) |
| host / device | `192-222-58-240`, cuda (1×GH200) |
| torch / transformers | `2.10.0+cu128` / `5.17.0` |
| fla / causal_conv1d | `0.5.2` (triton 3.7.1) / `1.7.0` |
| rows / seeds | 72 rows = 9 arms × 8 seeds, all `status: completed` |
| cost / wall clock | $2.39, 5,781.4s summed across rows |

**Every one of the 72 rows carries `quick: true`.** Under rule 8 they promote nothing, and
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

## The reading

All three operators say the same thing, and the three controls each rule out the competing
explanation independently:

- **Volume is not it.** The size-matched arm moves the score by at most 1.59pp in either
  direction, against total effects of 58–77pp. Same number of rows, same languages.
- **The class prior is not it.** Siblings — same class, same shifted prior, seen in
  training — hold in every case, and in two of three they *rise*. Whatever the holdout did
  to the prior, the model absorbed it fine on rows whose generator it had seen.
- **Language is not it.** Measured above, on the arms' own training sets.

What is left is the generator's textual fingerprint. **The model classifies by recognising
the mutation operator that produced the row, not by reading the change.** It shares the
char-n-gram control's shortcut.

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

Every holdout and size-matched arm carries `paired_margin_vs_linear: not_run` — 48 of the 72
rows. That is not a loss and must never be read as one. It means no control was fitted on
those arms' reduced training sets, so the model had no opponent at all.

This is the one number that could still overturn the headline. Under holdout the control
collapses to 3.08% on `stub.panic` while the model holds 15.44%. Those two come from
different tools on different splits and **cannot be subtracted**. A control fitted on the
arm's own training set and scored on its own validation rows would make the comparison real,
and it would be the first configuration in this project where rung 0 wins.

`tools/fit_operator_holdout_controls.sh` exists to fit them. It is not cheap and it does not
finish the job on its own — see `HANDOFF/rung0-control-2026-09-22.md` for the measured cost
and the reason the 72 rows above can never gain margins retroactively.

## Gaps this leaves open

- `GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD` — unrelated to the choice head
  measured here, being addressed by the span-in-diff arms.
- `GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN` — `code_commit` on these rows is a dirty
  sha from an older commit; `recipe.rev` is what pins the launch.
- `GAP-CONTROL-CACHE-GOES-COLD-ON-ANY-BASELINE-EDIT` — why the 48 margins above cost ~15
  hours of CPU plus a re-run of the arms, and why the digest that makes them cost that
  must not be weakened to make them cheaper.
- `GAP-THE-CONTROL-CLASSIFIES-BY-GENERATOR-NOT-BY-CHANGE` — moved to
  `resolved-with-residual` by this experiment; the residual is those 48 margins.
