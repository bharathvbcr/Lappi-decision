# Supervising the span in diff space: free, and worth nothing

Measured 2026-09-22 on `commitpackft-corpus-v2`, `--context-source diff`, width 256 /
2 layers / 4 heads at `lr 3e-4` — the grid's best cell — 3 epochs, batch 16, span-weight
0.05, 8 seeds per arm. 16 `ft` rows in `ledger/gh200-span-in-diff-2026-09-22.jsonl`,
$0.69, 1,665.0s, launch rev `084b7f1d69c9aeeb8ef2a351dcf8fb9a03f06171`. All 16 rows carry
`quick: true` and promote nothing.

Two arms differing in one flag: `nospan` is the current diff protocol with the span head
unsupervised; `span` adds `--span-in-diff`, which resolves the gold span into the diff's own
byte space so the span head has a target at all.

## Question 1 — does supervising the span cost the choice head?

**No, not measurably.** The gradient is shared and `--span-weight 0.05` was inherited from
arms that trained the head on nothing, so this was the real risk.

| arm | val choice top-1 (8 seeds) | train choice top-1 |
| --- | --- | --- |
| `nospan` | 80.46% ± 1.61 | 83.16% ± 1.57 |
| `span` | 79.83% ± 1.11 | 82.73% ± 1.18 |

`tools/ledger_arms.py` on the two arms:

```
+0.63pp   pooled sd 1.38pp   floor at n=8 known-sd / estimated-sd 1.94 / 2.08pp
          inside the floor -- NOT a difference
```

Paired per-seed, which is the stronger form since both arms share seeds and split:
**−0.63pp ± 2.07, 3 of 8 seeds positive** (`−1.62 −3.23 +0.05 −3.18 −0.19 +0.26 −0.26
+3.13`). The sign is negative and the spread covers it. At this n the design cannot resolve
a difference this small, so the honest statement is *not demonstrated*, not *zero*.

## Question 2 — is the span number worth anything?

**No.** This is the finding.

| | value | over 10,635 pointing rows |
| --- | --- | --- |
| span start top-1 | **63.61% ± 6.72** | |
| span end top-1 | **59.68% ± 7.49** | |
| uniform-over-candidate-lines chance | 8.44% | what the two gates are scored against |
| **point at a random added (`+`) line** | **89.91%** | 10,429 of 10,635 rows have an added-line gold |

A unified diff marks its own answer. Both corpora average about 1.5 added lines per hunk, so
a policy that reads no code at all and aims at a `+` line scores 89.91%. The supervised span
head scores 63.61%. **It loses to reading nothing by 26.3 points.**

### The gate says this passed

It does, and the gate is not wrong about what it measures:

> span start top-1 60.6% of 10635 held-out rows, against a uniform-pointer chance over the
> candidate line starts of 8.4%. The gap is +52.1%; a model at the baseline has learned the
> prior and nothing else.

`passed: True`, on all 8 seeds. The uniform rate is the right null only when nothing marks
the changed line, and in a unified diff something does. Rule 2 makes a gate's baseline
read-only to an agent, so the gates were left alone and
`val_span_pointing_added_line_chance` is recorded **beside** them on every row, carrying its
own instruction:

> Read the pointer's accuracy against the LARGER of the two.

This is the whole reason that metric exists. A 16-wide one-epoch smoke run earlier in this
programme read 92.3% against 8.8% and would have shipped as a result; at real scale the same
comparison reads 63.61% against 89.91% and is a failure. The two nulls disagree by 81
points, and which one is quoted decides whether this document says "works" or "worse than
nothing."

`GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD`.

## What this licenses

`--span-in-diff` is **not worth turning on** for the span capability. It is close to free
for the choice head, so it is not harmful — but it buys a pointer materially worse than a
two-line heuristic, and shipping it as "the span head works" would rest entirely on quoting
the weaker null.

It does **not** say the span task is impossible. It says this corpus cannot show a span head
is doing anything, because the diff format hands the answer to any pointer that looks for a
`+`. A span capability that means something needs either a context where the changed line is
not marked, or a null that already includes the marker and a head that beats it.

## Other gates on these rows, reported because they ran

- `ece` **FAIL on all 16 rows**: 0.0746 mean against a 0.0500 bar, 15 bins of which 11
  carried mass. Both arms, so it is not caused by the span flag.
- `permutation_consistency` pass: the choice head agreed with itself across a derangement on
  12,792 of 12,792 rows (100.0%) against a 95% floor.
- `degenerate_head` control pass: entropy 0.2945, top-class share 0.382.
- `paired_margin_vs_linear` **not_run on all 16** — no cached control for this training set,
  so there was no opponent. Not a loss. Same cause as the operator-holdout arms; see
  `GAP-CONTROL-CACHE-GOES-COLD-ON-ANY-BASELINE-EDIT`.
- `needle_hunk_recall`, `ood_abstain`, `privileged_hunk`, `shuffled_label`, `transfer_gate`
  all `not_run`; each is human-owned and none is wired to a guessed contract.
