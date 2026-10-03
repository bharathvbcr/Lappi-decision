# Hallucination: what it means for Lappi, what v4 measures, and what a reward system can do

The human asked, 2026-10-03, in the lead's session: "Also try to make the model less
hallucinating", then "Use reward system or some advice from fable on these". Fable's ruling and
the measurements it asked for are below. Everything here is CPU and offline, over files F already
wrote. No GPU job ran.

## 1. What hallucinating means for a typed decision model

Lappi answers a slot from a fixed option set that always includes `noul`, the abstain row (Z).
Hallucinating means one of four things.

| form | what it is | state |
| --- | --- | --- |
| answering instead of Z on input it cannot judge | OOD or unanswerable input | the suite half of `ood_abstain` measures this, and it fails on every v4 row |
| answering wrong in-distribution with high confidence | confident-wrong | measured in section 4 |
| position bias | the answer moves when the options are reordered | 99.4% permutation agreement on the defect family: not the problem |
| inventing evidence spans | the span points at the wrong lines | span slots refuse on the release (GAP-J7-EXPORT-SPAN-HEAD-UNSERVED), so no v5 lever |

The mapping is the lead's. Fable agreed with it, and the human may correct it.

## 2. Does training teach the letter channel to abstain?

Yes, from v4 on. Two gaps said otherwise, and both are now updated in `gaps.jsonl`.

- **`GAP-DATA-NO-LETTER-ROW-EVER-ABSTAINS`: resolved.**
  - `python/qd_data/mixture.py:726-734` gives every noul-corpus row a choice gold of Z.
  - `:541-543` does the same for CLINC out-of-scope rows.
  - v4's build row `d96409bd` consumed the noul corpus (`defect_noul_examples_sha256` b194f14c...;
    5,004 rows of defect-noul-v3b).
- **`GAP-OOD-ABSTAIN-FAILS-WITHOUT-LETTER-NOUL-SUPERVISION`: resolved-with-residual.**
  - F seed 0 abstains on 54/60 prose and 58/60 scrambled cases.
  - The phase-3 corpus, with no letter noul, abstained on 22/180 in total.
  - The residual is unseen-language: 20/60.

## 3. Would a confidence threshold fix the OOD suite? No: the model is confidently wrong

`abstain-threshold-curve-f-seeds.json` holds the full curves, written by `abstain_curve.py`.
`defect-selective-risk-f-seeds.json` is written by `selective_risk.py`. Both scripts sit beside
this file. They are throwaway analysis (Python's analysis-only role) and never ship.

- Inputs: F's val and suite verdicts for seeds 0-2, rows f4feac15, aeca8d69 and 8c3a774a, pulled
  2026-10-01/02 to `~/qd-campaign/rescore-excluded-2026-10-02/box/`.
- Method: uncalibrated top probability, with the margin as well. Each threshold is added on top
  of the runtime's existing rule, and checked against the gate's bounds unchanged (rule 2).
  - OOD: Wilson lower bound at least 0.90 over 180 cases.
  - In-distribution: Wilson upper bound at most 0.05 on the defect family's 2,304 val choice rows
    (the decided population).

| seed (row) | suite as built (prose / unseen-lang / scrambled) | defect in-dist abstained | p_top of answered suite cases, median | best suite under the in-dist cap |
| --- | --- | --- | --- | --- |
| 0 (f4feac15) | 132/180 (54 / 20 / 58) | 13 / 2,304 | unseen-lang 0.985; prose 0.559 (6 answered) | 136 (p_top >= 0.66) |
| 1 (aeca8d69) | 56/180 (0 / 0 / 56) | 21 / 2,304 | prose 0.976; unseen-lang 0.984 | 58 |
| 2 (8c3a774a) | 68/180 (1 / 9 / 58) | 26 / 2,304 | prose 0.943; unseen-lang 0.967 | 77 (79 on the margin) |

- No threshold, on either statistic, passes both bounds on any seed.
- The OOD cases the model answers sit at a median p_top of 0.94-0.98. Its in-distribution wrong
  answers sit at 0.77-0.82, and its correct ones at 0.9997.
- So on OOD input the model is not uncertain; it is confidently wrong. A threshold cannot separate
  those cases from correct answers. Temperature scaling would not either, since it moves every
  row's p_top in the same direction.
- Prose abstention is bimodal across seeds: 54, 0 and 1 of 60 (GAP-OOD-PROSE-ABSTENTION-BIMODAL-ACROSS-SEEDS-2026-10-02).

**Consequence.** OOD hallucination is a training-side problem. v5 already carries both levers
aimed at it:
- the defect-schema noul data: v3b, own-repo prose 2,000, and the commitpackft G6 unseen-language
  source of real files in new languages, which is the category that fails;
- the noul-weight arm, x3.

## 4. In-distribution, a threshold does cut confident-wrong answers

The data is in `defect-selective-risk-f-seeds.json`: the defect family's val choice rows,
uncalibrated p_top.

| p_top >= | seed 0: abstained / wrong answers | seed 1 | seed 2 |
| --- | --- | --- | --- |
| (as built) | 13 / 90 (risk 3.9%) | 21 / 84 (3.7%) | 26 / 79 (3.5%) |
| 0.7 | 92 / 58 (2.6%) | 77 / 58 (2.6%) | 85 / 51 (2.3%) |
| 0.9 | 172 / 35 (1.6%) | 167 / 27 (1.3%) | 185 / 28 (1.3%) |
| 0.99 | 393 / 8 (0.4%) | 381 / 10 (0.5%) | 447 / 8 (0.4%) |

Every threshold trades abstentions for fewer wrong answers. Above about 0.7, in-distribution
abstention passes the gate's 5% cap. The gate stays as it is. Where to sit on this curve is a
product cost, so it belongs to the human (section 5).

## 5. The reward system (Fable's ruling)

The reward is +1 for a correct answer, 0 for an abstention, and -c for a wrong answer. With an
explicit abstain row it has a closed-form optimal policy: abstain unless the calibrated p_top
exceeds c/(1+c). So no sampling and no RL/GRPO are needed. A policy gradient on a softmax decision
head under this reward is the gradient of a closed-form expected reward, which is just a loss
term. "Reward system" therefore has exactly two implementations here.

- **Training side:** a cost-weighted loss on abstention errors. v5's noul-weight arm already is
  this, in row-weight form. v5 gets no second training arm. If the human wants the exact
  expected-reward loss, it becomes a pre-registered v6 arm.
- **Inference side:** a calibrated abstain threshold set by c. It is the runtime's
  calibrated-margin slot, currently unfitted (GAP-RT-CALIBRATION-NOT-FITTED).
  - Section 3 shows it does not fix OOD.
  - Section 4 shows it does cut in-distribution confident-wrong answers.
  - The runtime thresholds the margin, `crates/qd-runtime/src/calibration.rs:223`
    (`probs[top] - second`), while the reward-optimal rule is on p_top. This is noted, not
    changed.
  - Fitting it needs the calibration fit's objective defect fixed first: CPU work.
  - The human scoped the calibration bind out of v5 in the bench session ("no calibration bind in
    v5"), so reopening it is the human's call.

**Rule-2 guard.** These curves show whether the mechanism is plausible. They must not pick c or a
threshold. c is the human's product number, the cost of a wrong answer against an abstention. It
is pre-registered before v5's suite is read, and the fit runs on val, never on the suite.

## 6. What v5 pre-registers for this (report-only, no gate moves)

- Per seed, the suite half per category, against F's three rows above.
- In-distribution abstention per family.
- The selective-risk table of section 4, per seed. It will be computed offline from the verdict
  files, as here, unless it lands as a metric.
- The data lane's risk: about 130k prose-shaped decision rows that should be answered must not
  make defect-schema prose less likely to abstain. Report the prose category per seed against F's.
