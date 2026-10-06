# Lappi v0.1 preview: release notes (2026-10-06)

**Status: a preview. Not promoted.** It was released because the human decided to release it:
- ~20:45Z: "lappi is my first model, I am okay if it's not great", then "Choose hte best";
- ~02:17Z: "lets finalize the model";
- ~02:25Z: "Ship the average", after the bar below missed on two lines.

No gate, threshold or seed moved (rule 2).

**Scope** (the record's `transfer_gate_definition`): promotion under this record certifies the
mutation-labelled distribution only. Transfer to naturally occurring defects is unmeasured
(GAP-TRANSFER-GATE-IS-NAMED-NOT-SPECIFIED-AND-HAS-NO-DATA).

## The artifact

| | |
|---|---|
| Release directory | `/Users/bharath/qd-campaign/preview-2026-10-05/release-lappi-v0.1-preview/` |
| Built by | `qd-export` at a766fcf (branch `preview-export-train-manifest-bound`; merged into main 2026-10-06 as f23c890) |
| weight_hash | `6f7b9ba73bca3a0b2e4ec9b2c0fc2aa6e7459f7e3286b26386633a222667dfcb`: 320 tower tensors, equal to the 10-05 trial export's |
| tokenizer_hash | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` (Qwen3.5-2B-Base, snapshot b1485b2f) |
| calibration_hash | `7e56b34e378907adccc7cfc0a086507d81c762ed6627694a0a59c3c977aeb677`: `qd-calib-fit --population all --target-precision 0.80` |
| model.safetensors | `5ade5349…` |
| span_head.safetensors | `7a187a1e…` |
| Source | `v5-avg-s0-4.safetensors` (sha256 `0a19a1f6…`): a plain mean of v5's five seeds' fp32 masters (`tools/ckpt_average.py --from masters`), ft rows 9a00cfd0, b0d9454e, 8079c966, 1c8a0ab5, f2bd50e7 |
| Train manifest | `phase4-v5r-2026-10-03/data/pool/train.json` (sha256 `6c2afc81…`) |

`trained_families`: 23, recorded in the release manifest. The runtime refuses every other task.

**Load-tested: yes, on the Mac's GPU, 2026-10-06 02:56Z** [V]. `qd-metal-serve` at a766fcf
(`build/v6x/loadtest.sh`, `loadtest_defect.sh`) bound to this release directory, and requests went
through `qd oneshot --socket`, the product path. Outputs are in
`/Users/bharath/qd-campaign/preview-2026-10-05/loadtest/`.

**What worked:**
- The server confirmed weight_hash 6f7b9ba7… and calibration 7e56b34e… at bind.
- The backend is `qd-metal/qwen3.5-2b-base/tessl`, not degraded, and every request was served through the socket.
- **Latency:** the first request took 3.37 s (it loads the model); after that the median was 0.12 s and the maximum 0.76 s.

**The replay** was 90 val choice rows from the 15 decision-pool families scored at 6af73bef, 6 per family, deterministic. `build/v6x/replay_requests.py` built them through training's own request builder.
- Served correctness equals the CUDA scoring's correctness on 89 of 90 rows.
- The CUDA scoring shuffled option order, so answers are compared by correctness, not by option.
- **Answered:** 85, of which 70 were correct.
- **Abstained:** 5. On 4 of them CUDA was wrong; the fifth was a correct low-margin NLI row (score 0.117), which is the one disagreement.

**Refusals behaved as specified:**
- An untrained task gets `task_not_trained`.
- A wrong calibration-hash pin gets `hash_mismatch`; the right pin answers.

**`code.defect_class` is not servable by this release.** This is the core family and DevCouncil's use.
- **Its trained request asks for a choice slot and a span slot.** The span slot is refused with `calibration_entry_missing` (no span entry, GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS).
- **Asking for the choice slot alone** is a prompt the model never saw, because slots are rendered into it. On `a + b` → `a - b` it answered `noul`, score 0.405.
- **The multi-file `diff --git` / `---` / `+++` shape** is refused as `context_not_unified_diff`. v4's composed-v1 defect rows use it, per the 10-01 gap; v5's composed-v2 is assumed to as well, not verified here. The documented example request, which carries a `---`/`+++` preamble, is refused the same way. Both are GAP-ADMISSION-REFUSES-COMPOSED-DEFECT-CLASS-ROWS, open since 10-01, now measured.

So the preview serves the general decision families' choice questions only. This is GAP-PREVIEW-DOES-NOT-SERVE-CODE-DEFECT-CLASS-2026-10-06.

## What it serves

- **Choice (letters) slots of the general decision families:** 15 entries, 3 to 17 options. **Not `code.defect_class`** (see the load test above).
- **Span requests refuse** (`calibration_entry_missing`): span verdicts carry no pointer distribution, so no span entry can be fitted (GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS).
- **Score requests refuse.**

**Serving behaviour at the human's 0.80 precision target** [V, `build/v6x/coverage.py` on the
average's own fits, 17,254 val choice rows]:

| Estimate | Answers | Precision on answered | Refuses | 16-option questions refused |
|---|---|---|---|---|
| **Two-fold (honest)**: each row scored with the other fold's parameters | **91.9%** | **87.0%** | 8.1% | 51% |
| In-sample (`all`, the shipped table; optimistic) | 93.8% | 87.1% | 6.2% | 31% |

- **Not modelled here:** the choice slot's permuted second pass. Serving also abstains where the two passes disagree, so real refusals run somewhat higher. The row's in-distribution abstention, the noul and permutation halves before any margin, is 4.6%.
- **Calibration error (two-fold ECE, 15 bins, threshold 0.05):**
  - 0.0056-0.0379 on 8 of the 9 question types with ≥ 100 rows;
  - **0.079 on 16-option questions (fails)**, from 0.173 before calibration;
  - 6 small types (< 100 rows) are not measured.

## Measured on the artifact itself

Score row **6af73bef-aa56-49b3-8cdb-e0b0b56e9181**, `h100x2-v5avg-2026-10-05.jsonl` (2× H100, 02:23:25Z):

| Measure | The average | v5's five seeds | Gate |
|---|---|---|---|
| val choice top-1 | **0.857** (14779/17254) | 0.828-0.847 | (not a gate) |
| val span top-1 | **0.773** (5596/7238) | 0.907-0.915 | (not a gate) |
| permutation_consistency | **0.957** (16514/17254) | 0.929-0.949 | ≥ 0.95: **passes** |
| ood_abstain, OOD half | **149/180** (Wilson lower 0.766) | 152-162 | ≥ 0.9: **fails** |
| ood_abstain, in-distribution half | **4.6%** (770/16905; Wilson upper 0.049) | 5.6-8.2% | ≤ 5%: passes |
| unseen-language / scrambled / prose | 40/60, 49/60, 60/60 | | (reported) |
| needle_hunk_recall, worst bucket | **1.000** (300/300) | 0.705-1.000 | ≥ 0.95: passes |

## The bar it missed, and the human's choice

The bar was pre-registered before this row existed (`HANDOFF/lead-pipeline-2026-10-03.md`, commit
96838a1, 01:58:41Z). The average had to be no worse than v5's worst seed on six lines.

| Line | The average | v5's worst seed | Result |
|---|---|---|---|
| choice | 0.857 | 0.8278 | pass |
| permutation_consistency | 0.957 | 0.9288 | pass |
| in-distribution | 0.046 | 0.0819 | pass |
| needle | 1.000 | 0.7049 | pass |
| **span** | **0.773** | **0.9066** | **miss** |
| **ood_abstain** | **149** | **152** | **miss** |

The rule routed the misses to the human, who chose to ship (~02:25Z). Span is not served by this
release; OOD abstention is 3 of 180 under the worst seed.

## The span head is degraded: do not serve it

The release carries `span_head.safetensors`, the mean of the five seeds' span heads.

**Measured [V, `build/v6x/span_head_norms.py` against seed 1's snapshot]:**
- The projection norms are 0.447 of a seed's.
- The cosine to seed 1 is 0.448.
- The abstain vectors are 0.58 of a seed's, cosine 0.59.

**Inferred:** 0.447 is 1/√5, the norm and cosine of the mean of five mutually orthogonal vectors of
equal norm. So the seeds' span heads are near-orthogonal, and the mean blends them. The tower
averages well because every seed starts from the same pretrained weights. The span head is trained
from a seed-dependent start.

A future table that fits span would serve this head at ~77%. v6 should average the tower and take
one seed's span head (`HANDOFF/v6-plan-proposal-2026-10-06.md`).

## Controls

- **shuffled_label ran on v5 seeds 0-3, at chance on all four:** 0.2370, 0.2296, 0.2209, 0.2183 against the 0.3049 ceiling. Rows ba375781, daf4ec89, 8a47dd8a, deaf6717.
- It did not run on seed 4.
- No control ran on the average itself. The average inherits nothing formally.

## Why it is not promoted

Read-only, `Ledger.promotion_verdict` over v5's eval-row family 408d7387 (eval rows 166f7ebd,
e1bafd6f, 2af3c80d, e09b652a, a51c5bf4), re-run 02:01Z. **Not promoted:**
- `paired_margin_vs_linear` did not run on any seed. The pooled control cannot run on synth.general (row 0b0ac9a8). A re-spec is proposed: `HANDOFF/promotion-respec-proposal-2026-10-05.md`.
- `ood_abstain` fails on every seed: 158, 162, 152, 156, 155 of 180. Wilson lower 0.784-0.848 against 0.9.
- `needle_hunk_recall` fails on seeds 0 and 2.
- `shuffled_label` did not run on seed 4.

The record's `average_may_promote` is false, so an average cannot promote under the record.

## Open

- GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS
- GAP-TRANSFER-GATE-IS-NAMED-NOT-SPECIFIED-AND-HAS-NO-DATA
- GAP-PAIRED-MARGIN-POOL-INCLUDES-A-FAMILY-WITH-NO-TRAINING-ROWS-2026-10-05
- GAP-SHUFFLED-LABEL-REQUIRED-ON-EVERY-SEED-2026-10-05
- GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06
- GAP-ADMISSION-REFUSES-COMPOSED-DEFECT-CLASS-ROWS (measured 10-06)
- GAP-PREVIEW-DOES-NOT-SERVE-CODE-DEFECT-CLASS-2026-10-06
- The `qd-export` fix branch awaits the human's merge.
