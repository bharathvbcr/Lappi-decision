# Proposal for the human, 2026-10-06: the v6 plan (drafted by Fable, written by the lead)

**Status: a proposal.** Nothing is launched, and no box is requested. Every run named here needs
the human's yes, with a wall-clock cap, auto-terminate and a cost estimate (rule 4). The reading
rules go into `campaign/` as a pre-registration before the first run (rule 10's sibling: written
before a result is read).

**Labels.** [V] is measured: a ledger row or a launcher line. [I] is inferred. [U] is unmeasured.

## Is v6 needed at all?

Fable's answer (lead lane, ~02:05Z): **v5 is a complete first milestone, and stopping at "Lappi
v0.1 preview" is a legitimate choice.** v5 does the task [V, `h100x2-v5-2026-10-03.jsonl`]:
- val choice top-1 ≥ 0.8278 and span ≥ 0.9066 on every seed (these are the worst seeds' values, from `read_v6x.py`'s v5 minimums);
- permutation_consistency ≥ 0.9288 on every seed;
- needle worst bucket 1.000 on seeds 1, 3 and 4;
- shuffled-label controls at chance on all four that ran: seeds 0-3, rows ba375781, daf4ec89, 8a47dd8a and deaf6717 (0.2370, 0.2296, 0.2209, 0.2183 against the 0.3049 ceiling). The model learned content, not label format.

What v5 fails:
- one gate it was trained toward, `ood_abstain` (152-162 of 180);
- `needle_hunk_recall` on seeds 0 and 2;
- one check that cannot run on this mixture, `paired_margin_vs_linear`.

A v6 is worth funding only if the human wants a promotable model or better "I don't know"
behaviour.

## What v6x taught (the evidence the plan rests on)

v6x (`--noul-weight 4`, 5 seeds, report-only) reads `no_signal` [V, `read_v6x.py`;
`HANDOFF/lead-pipeline-2026-10-03.md` ~01:55Z]:
- OOD abstention rose on 4 of 5 seeds: +6 to +16 of 180, and +3 to +13 of 60 on unseen languages.
- In-distribution abstention rose too: worst seed 8.69% vs v5's 8.19%.
- Choice accuracy fell: worst seed 0.8233 vs 0.8278.

**The binding half of `ood_abstain` is now the in-distribution cap** (5%). v5 sits at 8.2% and
v6x at 8.7% [V]. Two levers move both halves the same way, which is the wrong direction for the cap:
- **The noul weight:** v6x moved both up [V].
- **A fitted `noul_margin`:** the gate's own detail says it "can only add abstentions" [V, the gate detail; the size of the effect is U].

Fable's correction to its own earlier advice: "weight 4 plus a fitted noul_margin" is not a gate
strategy, because it pushes in-distribution abstention further over the cap.

**So the open v6 question is a lever that raises OOD abstention and lowers in-distribution
abstention at once. That is data, not dose.** [I]

## The plan

### 0. Prerequisite, before any v6 run: the two re-spec entries (a yes/no for the human)

`HANDOFF/promotion-respec-proposal-2026-10-05.md`:
- `paired_margin_population`, judged on code.defect_class;
- `shuffled_label_seeds`, 3 seeds.

Without the first, `paired_margin_vs_linear` reads `not_run` on every seed of every model trained
on this mixture, so v6 cannot promote whatever it learns [V, row 0b0ac9a8]. The lead then writes
the `ledger.py` code and tests on a branch; the merge is the human's.

### Arm A: dose (`--noul-weight 2`)

- 5 seeds on v5's recipe with `--noul-weight 2`.
- Reading rule as v6x's: at least 4 of 5 seeds strictly above v5's same seed on both
  `gates.ood_abstain` and `metrics.ood_abstain.unseen-language`, with the same guards.
- It measures the dose-response between weight 1 (v5) and weight 4 (v6x).
- [I] It may keep part of v6x's OOD gain at a smaller in-distribution and choice cost. It is unlikely on its own to bring in-distribution under 5%.

### Arm B: discrimination data (the untested lever)

- Add in-distribution hard cases, labelled *answer*, so the model learns when not to abstain.
- Source [I]: train-split rows on which v5 abstains, mined from the train split only. Held-out data and the two task-holdout families are never read (rule 3, the qd-train path check).
- Target: in-distribution abstention ≤ 5% (`metrics.ood_abstain.in_distribution`, Wilson upper) while OOD abstention holds at v5's level or above.
- The reading rule is pre-registered before the first row. The mining, dedupe and leakage checks are CPU work whose size is [U].

### Needle: long inputs in the mixture

- v5's needle worst bucket fails on seeds 0 and 2 (0.705 at 80-100% depth) [V]. The gate's detail names the shape: "A recurrent model losing early context".
- Add long-input examples. Target: worst depth bucket ≥ 0.95 on every seed.
- The data prep cost is [U].
- It can ride on Arm B's seeds rather than a third arm [I]. If it does, Arm B's reading rule names both targets.

### The shipped artifact

v5's 5-seed average was scored at row 6af73bef, 02:23Z [V]. Against v5's seeds it was:
- **Better on every tower metric:**
  - choice 0.857 vs 0.828-0.847;
  - permutation_consistency 0.957, the first to pass the 0.95 gate;
  - in-distribution abstention 4.6%, the first under the 5% cap;
  - needle 1.000.
- **Worse on two:**
  - span 0.773 vs 0.907-0.915;
  - OOD 149/180 vs 152-162.

The span head collapse is GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06: the seeds' span
heads are near-orthogonal (norm and cosine ≈ 1/√5).

**v6 pre-registers:**
- the tower averaged over seeds, with **one seed's span head** (named before scoring), or a span head initialised identically on every seed;
- the averaged artifact as the release candidate.

[I] Averaging may also be a lever on the in-distribution cap: it took 8.2% to 4.6% here. Arm B's
reading should include an averaged row.

### Rust targets (the goal's "minimize python bottlenecks with rust")

Each is profiled before it is ported. Each port ships with a parity test against the Python as the
reference oracle, a committed interleaved min-of-N benchmark, and the Python deleted in the same
change. No speedup is claimed before the benchmark exists.
- **The GH200 post-training tail:** ~10 min of single-thread Python, then ~5 min idle before the needle control, on each J6 run [V, the 10-05 J6 logs].
- **`tools/ft_linear_control.py`'s CPU phase:** the GH200's GPU idled 01:26:26-01:38:24Z during tierb3's control [V, `q-tierb3.log` launcher lines]. It runs on ~50 cores, so it is parallel, but not on the GPU.

**Correction (read-only analysis, ~03:20Z 10-06; grep-based, because DevMap's DB was malformed). Neither window is a Rust-port target:**
- **The linear control's Rust port already exists on main** (`qd-prep linfit`, 8f08c52). tierb3 ran a pre-port clone (884b658, inferred), and its 50-core phase is numpy GEMM. The fix is to run it from current main.
- **The J6 tail is per-row GPU-to-host syncs in `_decode`, plus `--needle-control`'s startup** (a full split rebuild and a model reload). The fix is batching, and folding or caching that startup.
- The GitPulse cards `lappi-rust-ft-linear-control` and `lappi-rust-gh200-post-train-tail` were rewritten to say so.

### Pre-registration

`campaign/v6-<date>.md`, before any run. It carries:
- the arms;
- the reading rules;
- the guards;
- the seeds;
- the caps;
- the cost.

## Cost

Per-unit figures [V] are at $4.19/GPU-h on Lambda's 2× H100 SXM5 ($8.38/h for the box):

| Unit | Wall | Cost | Source |
|---|---|---|---|
| One training seed, scored | ~6.5 h | ~$27 | v6x seed 4: 23,500 s |
| One J5′ shuffled-label control | ~6.3 h | ~$26 | J5′ s3: 19:43:40-02:00:24Z |
| Box overhead | | ~10% | 10-03 to 10-06: box ~$455 against ~$431 of GPU-hours |

| Plan | GPU cost | Wall on one 2× H100 box | Box cost |
|---|---|---|---|
| One arm: 5 seeds + 3 J5′ (with the re-spec) | ~$215 + 10% ≈ **$235** | ~28-32 h | ~$235-265 |
| Two arms (A and B) | ≈ **$470-530** | ~2.5 days | |
| Without the shuffled_label re-spec | +2 J5′ per arm, ≈ +$50 | | |

Not in these figures [U]:
- the box's setup time;
- Arm B's and the needle data's CPU preparation.

A new box is the human's yes (rule 4).

## The free measurement: a fitted margin is not a lever for `ood_abstain` (measured)

The human asked for it at ~02:17Z. It was pre-registered at `campaign/v6x-margin-probe-preregistered.json` (58c2bf5) before any report existed.

**Setup.** `qd-margin-probe` at target precision 0.95, on all 10 eval rows (v5 and v6x, seeds 0-4), each with two split keys: 20 reports in `/Users/bharath/qd-campaign/margin-probe-2026-10-06/`. Summary from `build/v6x/summarize_margin_probe.py`.

**Pre-registered word: `trade`. The trade is adverse.**

| Row | OOD abstentions the margin adds (of 180) | Extra in-distribution abstentions (val half B) |
|---|---|---|
| v5 seeds | +0 to +8 | 30.0% to 39.0% |
| v6x seeds | +1 to +5 | 30.3% to 39.1% |

- Both keys agree on every row.
- The fitted margins are 0.73-0.89.
- The OOD rows the margin misses are confidently wrong: the margin quantiles on unseen-language and scrambled sit near 1, as on 09-30.

**So at 0.95 a fitted `noul_margin` is not a lever for `ood_abstain`** [V]. It buys at most 8 of 180 OOD abstentions for roughly a third of normal questions. Arm B (data) stays the open lever.

**What it changed for the preview: the precision target is a serving decision.**
- A sweep of the practice table (v5 seed 1, in-sample; `build/v6x/coverage.py`) showed the trade:

  | Target | Answered | Precision |
  |---|---|---|
  | 0.95 | 68.1% | 95.1% |
  | 0.90 | 78.1% | 91.4% |
  | 0.85 | 84.2% | 88.3% |
  | 0.80 | 90.9% | 86.6% |

- At 0.95, 16-option questions are refused about 98% of the time.
- The human chose **0.80** at ~02:22Z.

## Budget (the human, ~02:17-02:22Z)

- The human allows ≤ $100 more, or $200 only for a genuine improvement.
- Fable: nothing under $200 qualifies tonight. Arm A at 3 seeds (~$110-130) would confirm a partial gain already shown; a v6x average (~$10) is a different point on the same trade and would be selection after reading.
- The human's answer: "Spend $0, shut down". Arm B (~$235 at 5 seeds) waits for its data. At 3 seeds (PROMOTION_MIN_SEEDS) with 3 J5′ controls it is ≈ 3 × $27 + 3 × $26 + 10% ≈ $175.
