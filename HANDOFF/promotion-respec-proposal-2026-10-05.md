# Proposal for the human, 2026-10-05: two promotion re-specifications to record before v6

**Status: a proposal, not a decision.** `docs/promotion-decisions.json` is edited by a human only
(its `who_edits`); nothing here has been written to it. This came from the human's ~20:40Z "loosen
some gates" and their "Choose hte best". The choice made then was a v5 preview release with the gates
unchanged (`HANDOFF/lead-pipeline-2026-10-03.md`, the ~21:00Z section). These two items were set
aside for v6, because both will block v6 exactly as they block v5, for reasons that are about the
gates' wiring rather than about any model.

Fable's ruling (lead lane, ~20:45Z): these two are principled re-specifications, consistent with
decisions already recorded. Re-thresholding ood_abstain or needle_hunk_recall to clear a seed's
number is not, and is not proposed here.

## 1. paired_margin_vs_linear judged on the promotion population

**Why.** The gate is pooled over every family, and the pool includes synth.general, whose linear
control cannot run. In v5 seed 0's control row 0b0ac9a8, "234 of 332 inputs did not run", for
example `synth.general/access_route`, because the family has no training rows. So the pooled gate
is `not_run` on every seed of every model trained on this mixture, whatever the model does. A gate
that cannot run blocks; it does not pass (`docs/ledger-schema.md`, Promotion).

**What v5 shows on the promotion population** [V, row 0b0ac9a8, H100 ledger]:
`metrics.paired_margin_vs_linear.choice.code.defect_class` is +0.3898, 95% CI [+0.3694, +0.4110],
on 2304/2304 rows. 21 of the 22 families ran: 20 are above zero, and intent.in_scope's interval
lies wholly below zero (-0.0172). synth.general did not run.

**This mirrors** the recorded `promotion_population` decision. That decision already judges
permutation_consistency and ood_abstain's in-distribution bound on code.defect_class, with every
general family report-only.

**Entry to add** (fill `decided_by`, `decided_on` and `decision_ref`; the commit is your own and
names the gap):

```json
"paired_margin_population": {
  "question": "Which families does paired_margin_vs_linear judge promotion on: every family pooled (as built, which includes synth.general, whose linear control cannot run for lack of training rows), or the promotion population's families with every other family report-only?",
  "gap": "GAP-PAIRED-MARGIN-POOL-INCLUDES-A-FAMILY-WITH-NO-TRAINING-ROWS-2026-10-05",
  "status": "decided",
  "value": "the promotion population: each seed's metrics.paired_margin_vs_linear.choice.<family> for the promotion population's families must run and pass; every other family is report-only",
  "source": "the pooled gate as built (tools/ft_linear_control.py via the joined control row); v5 seed 0's row 0b0ac9a8 shows it not_run on synth.general",
  "decided_by": "",
  "decided_on": "",
  "decision_ref": ""
}
```

## 2. shuffled_label required on three seeds, not every seed

**Why.** shuffled_label is a leakage control: it asks whether the scores could come from the
labels' format rather than their content. v5 pre-registered J5' on three seeds (seeds 0-2: chance
at 0.2370, 0.2296 and 0.2209 against a 0.3049 ceiling; rows ba375781, daf4ec89, 8a47dd8a). J5' for
seed 3 is running on the human's yes (19:43Z). The verdict reads a missing control as blocking on
every seed, so seed 4 alone holds v5 back, and any 5-seed v6 needs 5 extra ~6.3 h runs.
PROMOTION_MIN_SEEDS is 3.

**Entry to add:**

```json
"shuffled_label_seeds": {
  "question": "On how many of a family's seeds must the shuffled_label control run and pass: every seed (as built) or PROMOTION_MIN_SEEDS (3), with any further seed's control still required to pass where it ran?",
  "gap": "GAP-SHUFFLED-LABEL-REQUIRED-ON-EVERY-SEED-2026-10-05",
  "status": "decided",
  "value": "3: at least three seeds of the family carry a ran-and-passed shuffled_label control; a control that ran and failed on any seed still blocks",
  "source": "the verdict as built requires it per row; v5 pre-registered J5' on seeds 0-2",
  "decided_by": "",
  "decided_on": "",
  "decision_ref": ""
}
```

## What has to happen with them

- **Code first, on a branch.** `python/qd_train/ledger.py` applies only the decisions it knows. A
  decided value it cannot apply reads `not_run`, never "as built". So each key needs code and tests
  that fail before the change. Adding a key to `REQUIRED_DECISIONS` makes the verdict refuse a
  record without it, so the record entry has to land with, or before, the code's merge. The lead
  writes the code once these are wanted; the merge is yours.
- **What they would not change for v5.** v5 still fails ood_abstain (152-162 of 180 on every seed)
  and needle_hunk_recall (seeds 0 and 2). It stays unpromoted, and its preview release does not
  depend on these entries.
