# Capacity did not hurt, and it does not help much either

Measured 2026-09-22 on `commitpackft-corpus-v2`, `--context-source diff`, 8 seeds per cell,
3 epochs, batch 16, span-weight 0.05. 120 completed `ft` rows across
`ledger/gh200-diff-capacity-2026-09-22.jsonl`, `gh200-lr-sweep-2026-09-22.jsonl` and
`gh200-tuned-capacity-2026-09-22.jsonl`.

## The claim this replaces

`b72aa54` recorded *"capacity is monotonic in the wrong direction"* — width 128 → 512 cost
3.9 points over 24 seeds — and it was re-confirmed earlier today on the diff task, where
width 128/256/512 read 76.3% / 70.9% / 59.0%. That was used to answer *"do we need a larger
cluster"* with **no**.

Every one of those arms ran at `lr 3e-3`, which was a literal in **two** places: `"lr": 3e-3`
in the recipe and `peak_lr=3e-3` in `train_once`. The recipe therefore printed a rate the
schedule was not obliged to use, and the two could have disagreed silently. Comparing widths
at one rate measures how well that rate suits each width, not what capacity is worth.

## The grid

Validation top-1, four-way task, control at 88.5%:

| lr | width 128 | width 256 | width 512 |
| --- | --- | --- | --- |
| 3e-3 | 76.33% | 70.89% | 58.98% |
| 1e-3 | 78.33% | 76.87% | 72.47% |
| 3e-4 | **78.69%** | **80.97%** | 78.33% |
| 1e-4 | 70.09% | 78.71% | **80.26%** |
| 3e-5 | — | — | 74.07% |
| 1e-5 | — | — | 65.06% |

Every column is an inverted U with a single interior peak, so the curves are real rather
than noise. The peak rate falls as the model widens — 3e-4 for 128 and 256, 1e-4 for 512 —
which is the ordinary width/rate interaction and is exactly what a single fixed rate hides.

## Best per width

| width | best lr | top-1 | paired margin | positive |
| --- | --- | --- | --- | --- |
| 128×4 | 3e-4 | 78.69% | −0.0982 | 0/8 |
| 256×4 | 3e-4 | **80.97%** | **−0.0754** | 0/8 |
| 512×4 | 1e-4 | 80.26% | −0.0825 | 0/8 |

## What follows

**"Capacity is monotonic in the wrong direction" is withdrawn.** At 3e-3 the three widths
read 76.3 / 70.9 / 59.0 — monotonically down. At each width's own best rate they read
78.7 / 81.0 / 80.3. The direction was an artifact of the fixed rate.

**But capacity buys almost nothing.** 16× the parameters moves top-1 by **1.6 points**, and
the best cell in the grid is the *middle* width, not the largest. 256 → 512 is −0.7 points.
The curve is flat-to-declining above 256.

So the answer to *"do we need a larger cluster"* is still **no** — but it now rests on a
grid in which every width was given a rate that suits it, rather than on a comparison that
handicapped the large models. The earlier **no** was right by accident.

**Nothing in the grid wins.** 120 rows, 0 positive margins, best single cell −0.0754 against
the char-n-gram control. Tuning moved the margin 3.5× at width 512 (−0.2869 → −0.0825) and
did not close it.

## What this does not settle

The control this margin is measured against has memorised the mutation operators
(`AUDIT/operator-holdout.md`), so "loses by 7.5 points" is a statement about
fingerprint-matching rather than about judging code changes. A grid that is entirely
negative against a compromised opponent does not establish that the architecture is
inadequate — it establishes that it is worse at the shortcut.

Widths above 512 were not tested, and 4 of the 6 rates were only run at width 512. A
capacity claim beyond this range is not supported by this table.
