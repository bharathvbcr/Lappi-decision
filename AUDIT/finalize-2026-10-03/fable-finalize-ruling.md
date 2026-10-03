# Fable ruling: finalizing the model (2026-10-03, ~12:30Z)

The human asked at ~12:25Z: "Ask fable for advice and work on it, and finalizing the model". The
lead consulted the advisor (Fable) with the full session, including the evidence below, and
recorded the ruling verbatim.

## Evidence the lead put in front of Fable, and verified afterwards

### The combined rows' gates

Rows from `ledger/gh200-p6-f-j7prime-2026-10-01.jsonl`:

| Row | ood_abstain | permutation_consistency | needle_hunk_recall | ece / paired_margin |
|---|---|---|---|---|
| avg c962cdd9 | FAIL, 62/180 = 34.4% (floor 0.9) | FAIL, 91.9% (floor 95%) | pass | not_run |
| ens5 b45406b5 | FAIL, 61/180 = 33.9% | FAIL, 93.7% | pass | not_run |

### Each F seed's gates

Rows from `ledger/gh200-p4-v4-2026-10-01.jsonl`. The lead checked these after the ruling, as the
ruling required before anything was sent:

| Seed | Row | ood_abstain | permutation_consistency | needle_hunk_recall |
|---|---|---|---|---|
| 0 | f4feac15 | FAIL 0.733 | FAIL 0.932 | FAIL 0.656 |
| 1 | aeca8d69 | FAIL 0.311 | FAIL 0.930 | pass 1.0 |
| 2 | 8c3a774a | FAIL 0.378 | FAIL 0.927 | FAIL 0.770 |
| 3 | ebc83be6 | FAIL 0.694 | FAIL 0.922 | FAIL 0.918 |
| 4 | d8c8300a | FAIL 0.339 | FAIL 0.931 | FAIL 0.836 |

On every seed, ece and paired_margin_vs_linear are not_run.

### Rules

- **G8** (HANDOFF/gh200-phase4-2026-10-01.md:664): "An average is promotable iff its own gate row
  passes and all inputs are non-quick, full-schedule seeds of one recipe and snapshot."
- **F-candidate rule** (campaign/f-j7prime-preregistered.json `f_candidate.rule`): "a three-seed
  artifact, weight average or logit ensemble, or nothing; the human chooses between measured rows
  under G8".

## The ruling, verbatim

**Where you are:** the human has handed you the decisions you've been queuing ("ask Fable, work on it, finalize"). You've done the orientation. Here is the ruling; record it verbatim in AUDIT as you did the tierb consult, commit it, and cite it in your report.

## 1. "Finalizing the model" — v4 cannot be it

G8 (gh200-phase4 HANDOFF:664): promotable iff its own gate row passes. You just read both combined rows:
- c962cdd9 (avg): ood_abstain **fail** 62/180 vs floor 0.9; permutation_consistency **fail** 91.9% vs 95%; ece and paired_margin **not_run**.
- b45406b5 (ens5): ood_abstain **fail** 61/180; permutation_consistency **fail** 93.7%; same two not_run.

Neither is promotable. Rule 2 means nobody moves those floors. So "which v4 to keep" is not a release decision — it's at most which interim artifact to serve internally, and under the F-candidate rule "nothing" is the only answer consistent with G8. **Verify before you write this:** pull each F seed's gate row (ood_abstain, permutation_consistency) the same way. The per-seed ood_abstain totals you tabulated (56–132 of 180) say they all fail the 0.9 floor too, but confirm permutation_consistency per seed. If every row fails, tell the human plainly: *no v4 artifact passes the release gates; the final model is v5.* Your last two updates said "the numbers lean toward the ensemble" without disclosing that both fail two gates — correct that now, with the row ids.

Recommendation to record: ens5 is the best *measured* v4 (83.8/91.2, composed 96.3%, best ECE); it may serve as the internal dev artifact; nothing from v4 is promoted.

## 2. No-mask: recommend **drop**

- No honest tau exists. Making all eight applications conclusive needs tau = 0.89 — a waiver. Even retiring letter/all as broken instrument (masked baselines with identical code end at letter loss 2.35 vs 3.66 — that observable isn't stable at 100 steps) leaves span/all inconclusive on both shapes, and B's span/all has a candidate arm at +3.4%, past the original 2%. Any tau that passes it is chosen after seeing it; p2_amendment_2 is the plan's own warning against that.
- Cost of dropping: shape B (bt 35,403 = the v5 batch size per box_q_j5p.sh) ran 33% faster unmasked, so ~20 GPU-h / ~20 h wall on v5. Real, but the cleaner speed lever is §4.
- **Lapse equals drop.** If the human says nothing by ~19:00Z, `nomask --run` exits 5 and C2A reads off — identical outcome. Stop presenting 19:00Z as forcing action. Ask for one word ("drop" / "amend") so the pre-registration records a decision rather than a lapse; don't amend tau yourself under any reading of "work on it".
- Prepare now: the C2A=off pin text and the pre-reg amendment line recording "dropped per Fable, 2026-10-03"; apply on the word or on the lapse.

## 3. The critical path, and the one human lever

From your waiter scan: C1 (last pin) = j6g ← j5pp ← fsucc ← rungd ← cudadev ← rung0 ← j6dv4. tierb2 (C2B) and nomask branch off j5p; j6f/j6a/j6ctl feed no pin you've named but compete for gpu.lock. Rough: j5p done ~22:00Z today; the chain to j6g.done is ~2.5 days of GPU; v5 is ~60 GPU-h after that. A trained v5 is **~Oct 8–9 at best** — and only if the data build is finished before j6g.done, because lead notes :596 already records that the GPU idles from j6g.done until the v5 deploy.

The data build is gated on Apple Diagnostics by the human's own instruction. That gate is now the single item most likely to idle the GPU. Put it **first** in the report, with the consequence and the two ways out: report the result, or say "release the gate" and the build starts under the Mac lock.

Queue trim: before proposing anything, label every remaining waiter with the pin or verdict it feeds (C1, C2A, C2B, LOWER, LRSET, fsucc's human verdict, or none) and its cap hours. Items feeding none are the only candidates to move behind v5. You cannot stop or edit box waiters; that's the human's action or a human-yes'd new launch. Present it as a table with GPU-h saved, not a recommendation to cut.

## 4. Do now, in order

1. AUDIT: this ruling verbatim; commit on main.
2. Verify the per-seed gate rows (§1), then the report: (a) Diagnostics gate + consequence; (b) no promotable v4, ens5 interim; (c) no-mask: drop, one word, lapse==drop; (d) timeline + trim table as a question. Push-notify once — the Diagnostics item changes what they'd do today.
3. Prep that needs no human: C2A=off pin draft; keep the rules-v5/qd-prep-v5 cross-build targets warm; the tierb checker is ready for tierb2's rows (~Oct 4); plan to run `qd-post-f-rules j6g` on j6g.done as noted.
4. Drop sha2-direct-yes and batched-pass wording from the recurring ask. They don't bear on finalizing; mention once at the bottom, then stop.

Blocking: §1's per-seed check. Don't send "no v4 passes" until you've read the five seed gate rows.
