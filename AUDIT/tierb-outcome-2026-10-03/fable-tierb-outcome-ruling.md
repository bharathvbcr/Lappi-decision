# Fable on finishing training faster: the Tier-B outcome rule gets a checker, and clause 1 is amended (2026-10-03)

The human asked the lead at ~02:00 UTC 2026-10-03, verbatim: "Ask Fable for model tuning decision
before a run or changes to finish training faster and implement it." The advisor is Fable,
through the lead's session (the advisor tool). There were three consults, at ~02:02, ~02:10 and
~02:14 UTC.

The lead follows the rulings under the human's standing instruction to follow Fable. Nothing here
moves a gate (rule 2). The outcome rule is Fable's own admission rule for a speed-up. It is not a
gate, and the human did not write it. No outcome row existed when it was amended: at ~02:12 UTC,
`/home/ubuntu/perf` held no `ledger-tierb-*.jsonl`, no `tierb-*` directory and no
`nomask-p2-verdicts.jsonl` (read-only ssh by the lead).

## Consult 1: what speeds training up

- No new tuning is implemented. The only speed-up worth the campaign's time is no-mask (20-40%,
  inferred) plus fused AdamW (~3%). Both are already v5's conditionals C2a and C2b, decided by
  tests queued ahead of v5 (nomaskp2, then the nomask outcome run, then tierb2).
- These are recorded as "not measured, not adopted", because each is under 3% or unmeasured, and
  measuring any of them is a box launch that needs the human's yes:
  - a larger batch_tokens;
  - the owner of the fp32 GEMM;
  - skip 7/8;
  - changes to scoring.
- Verify that what is pre-registered cannot fail silently:
  - the C2a/C2b wiring;
  - tierb2's code pin;
  - whether the v5 trajectory waiter takes gpu.lock.
- Queue order is unchanged. Launched waiters are frozen.

## Consult 2: two gaps the lead found while verifying

Found:
- **(a)** No machine checker exists for the outcome run's three-part pass rule, the rule that
  decides C2a and C2b. Its third clause is ambiguous.
- **(b)** The GPU idles between j6g.done and the v5 deploy, because C1's deciding row is the last
  in the queue.

Ruling:
- **(a)** Write a `tierb <fused|nomask>` subcommand of `qd-post-f-rules` on v5-build, in Rust, with
  fail-first tests, before the first outcome row lands. Its words are `pass`, `fail` and
  `refused`; it never passes by absence. Pin one reading of each clause in the DRAFT before any
  outcome row exists. Fix the DRAFT line that says the rule is "printed by
  `perf_tierb_outcome.sh fused --print`": `--print` prints argv (`tools/perf_tierb_outcome.sh:73-78`),
  not a rule.
- **(b)** Accept the idle (~30-60 min, ~$1-2 of ~60 GPU-h). Prepare everything except C1's line
  ahead of time, and keep the cross-build target dirs warm. Do not pre-build at a different
  commit: `V5_LANE_AT` means what it says.
- **P2.** If either shape reads `inconclusive`, the hold is the human's (amend τ, recorded, or
  drop the candidate). Have the numbers ready; recommend no τ.

## Consult 3: clause 1 as first pinned fails an equivalent candidate (the lead's evidence, Fable's amendment)

**The lead's evidence:**
- The phase-3 envelope rows are ledger/gh200-seed0-weights-2026-09-30.jsonl 6d170b3c, 60f29b07
  and 5c19c0e8:
  - `val_top1.choice` = 2327, 2326 and 2326 of 2332;
  - `val_top1.span` = 2050, 2048 and 2048 of 2053.
- Seeds 1 and 2 tie, and the fp32 rows 58fd1532, 116d2af2 and efed7b19 carry the same counts.
- Under "inside [min, max], inclusive", a 4th exchangeable draw lands inside at least half the
  time, and only at equality with ties. Clause 3 then asks the same of the fp32 rescore. An
  equivalent candidate would fail about half the time or more: the defect class that retired
  P2's first rule.

**Fable, verbatim (the ruling's operative part):**

> Drop my "[min, max] inclusive" reading of clause 1 — with seeds 1 and 2 tied and the band one or
> two counts wide, it fails an equivalent candidate by construction. That's the defect that retired
> P2's first rule, and the avgnp precedent ("the cited row's count governs") would lock it in rather
> than fix it, so this needs an explicit recorded amendment, not a reinterpretation.
>
> **Who decides.** The outcome rule's pass criteria are mine (the Tier-B design doc), not a gate
> under rule 2 and not a threshold the human wrote. I amend it; you record it in the DRAFT's
> `amendments_pending` for C2a and C2b before any outcome row exists — you verified the box has no
> `ledger-tierb-*.jsonl` yet, cite that. The human is *told* in the report, not asked: one line
> saying this is an amendment to Fable's own admission rule, made before data, and that if they'd
> rather v5 run masked regardless they can say so before the rows land (~20 h). Don't block the
> checker on them.
>
> **Amended clause 1.** The candidate's `val_top1.choice` and `val_top1.span` are each "inside the
> envelope" iff they land neither above nor below the three phase-3 rows by more than the envelope's
> own range — the exact complement of the successor's `clears`/`loses` test, so reuse
> `envelope_range`, `clears`, `loses` rather than writing a fourth comparator. On the rows you
> printed that is choice 2325–2328 of 2332 (range 1), span 2046–2052 of 2053 (range 2), computed
> from the rows at run time, integer cross-multiplication, never the float. If a range ever computes
> to 0, refuse — don't invent a floor. The design doc's printed decimals (99.7–99.8 / 99.8–99.9) go
> into the JSON as `literal`, report-only, the way lines 87–100 already do it.
>
> **Clause 3** takes the same envelope for the fp32 rescore's choice/span, plus: every gate with
> `state: ran` in 58fd1532 must be `ran` with the same `passed` in the candidate's all-gates row; a
> gate `ran` there and `not_run` in the candidate refuses. Flags only — no value comparison on
> needle/ood/permutation beyond what the envelope already covers. 58fd1532 is what I assumed (seed
> 0, fp32, needle+ood); pin its full id, and pin `shard_hash` d773b876… and `val_shard_hash`
> 105513b9… through `comparable`/`same_recipe_keys` so a candidate scored on different data refuses.
>
> **Clause 2 — check one thing before you spec it.** You showed eeda5db4's CI only in the gate's
> `detail` string. Read what `paired_margin()` at :1670 actually parses, and dump
> `metrics["paired_margin_vs_linear.choice"]` for lo/hi fields. If the bounds exist structured, use
> them. If they live only in the detail string, the parse is a strict regex tested against the exact
> eeda5db4 string, refusing on any deviation — and note it as a row-format gap.
>
> **Record, then build.** The amendment text in `amendments_pending` must match the checker's doc
> comment, and the checker refuses if the pre-registration's recorded reading disagrees with the
> code — the file's existing pattern. Fix the `--print` line in the same edit. Say plainly in the
> report that clause 1 is a low-power instrument at a 99.7% ceiling; the amendment makes it
> satisfiable, not sharp — the control CI and the gate flags carry the test. Don't add an
> instrument.

## What the lead checked for clause 2

- `paired_margin()` (`crates/qd-runtime/src/bin/qd_post_f_rules.rs:1670`) reads `n`, `n_total` and
  `value` only. eeda5db4 records its CI only in a `detail` string:
  - `gates.paired_margin_vs_linear.detail` and `metrics["paired_margin_vs_linear.choice"].detail`
    are both "paired margin +0.1501, 95% CI [+0.1359, +0.1651] over 10000 bootstrap resamples";
  - there are no structured lo/hi fields.
- The candidate's control tool is `tools/ft_linear_control.py` at qd-lane2 884b658, a descendant
  of eeda5db4's code_commit d43790d. It writes the same text through
  `qd_train.eval_harness.paired_margin_test` (884b658 `python/qd_train/eval_harness.py:282-287`):
  - the format is `paired margin {point:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] over {n_boot}
    bootstrap resamples`;
  - that is followed by an empty suffix when `lo > 0`, otherwise by one of two fixed suffixes.
- Recorded as GAP-LINEAR-CONTROL-CI-ONLY-IN-DETAIL-STRING-2026-10-03.

## Recipe keys the candidates record (read at the commits the outcome code carries)

- **fused:** `recipe.optimizer_fused = true` (e19dcf6 `tools/real_ft_run.py:336-337`).
- **nomask:** `recipe.train_attention_mask = "none"` (1cf8e6d `tools/real_ft_run.py:360`).

## Where it is applied

- **The DRAFT:** `campaign/v5-preregistered.DRAFT.json`:
  - `recipe.conditionals` C2a and C2b gain `outcome_rule`, and their `on_iff` names the checker;
  - `amendments_pending` replaces "C2b's printed rule text";
  - `amendments_applied` appends this amendment.
- **The checker:** the `tierb` subcommand of `crates/qd-runtime/src/bin/qd_post_f_rules.rs`, on
  v5-build.
