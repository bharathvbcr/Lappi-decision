# The human's answers on the v5 launch, 2026-10-03 ~18:42Z

## Verbatim, in the lead session's chat

> Yes to all, waive R9, approve ~$400

## What it answers

These were the open items in the lead's messages of ~17:40Z and ~18:35Z
(`report-to-human-2026-10-03-pipeline.md`, items 1–5). The interpretation is the lead's, and it
was stated back to the human in the same chat.

| Item | Answer | Applied as |
|---|---|---|
| The cost of 11 runs on one Lambda 2× H100 box: ~$310 at $4.19/GPU-h, ~$344 (R9 waived) or ~$400 (R9 kept) for the box | **approve ~$400** | **V5_HUMAN_YES** = the verbatim line above. The approved box spend ceiling is ~$400. |
| R9, the seed-0 pause | **waive** | **V5_R9 = waive** |
| The undecided conditionals C1, C2a, C2b, C3 | yes (off) | V5_C1=off, V5_C2A=off, V5_C2B=off; C3 off means V5_LOWER=keep and V5_LRSET=f (F's recipe plus v5's additions). The GH200 rows that would decide them are read afterwards, report-only. |
| J5′ ahead of the arm when a GPU would otherwise idle (a change to the human's order) | yes | The greedy two-lane rule. J5′ still waits for v5 seeds 0–2 (`j5prime.runs_iff` verbatim, Fable). |
| The box setup downloads | yes | 70 pinned wheels from PyPI and download.pytorch.org/whl/cu128, plus causal_conv1d 1.7.0's sdist built on the box (`build/v5-h100/h100_setup.sh`). No rustup: the musl cross-build needs none. |
| VitaminC, which carries the BIG-bench canary field | yes (in) | `tals/vitaminc` at its cap of 5,000 seeded rows (was 0). The canary GUID's identity is [U]. A model trained on it is contaminated for any BIG-bench task built from VitaminC. |
| Moving the data build to the box CPU if the Mac panics | yes (pre-approved) | Used only after a panic. Fable's one-attempt-on-the-Mac rule otherwise stands. |
| Optional decontamination downloads: BoolQ validation and VitaminC test | yes | Fetched at the revisions the train files came from, then enforced decontamination targets like ARC test. This closes GAP-DECISION-POOL-BOOLQ-VAL-AND-VITAMINC-TEST-ARE-NOT-DECONTAMINATION-TARGETS-2026-10-03. |
| The ship gate's 300 hand-labelled diffs | no action asked | Unmet until labelled or the gate is amended (rule 2). |

## Not covered by this answer

- **Launching the box** stays the human's action, when the lead says. That is about 2 h before
  the data attestation, estimated at 00:00–01:00Z.
- Terminating the box is likewise the human's action.
- Anything over the approved ~$400 needs a new yes (`V5_OVER_BUDGET_YES`).
