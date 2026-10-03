# Fable rulings: does the model need more data, and can anything promote? (2026-10-03, ~12:40Z)

The human asked at ~12:35Z: "See if it needs more data, ask fable". The lead consulted the
advisor (Fable) twice. The first ruling asked for three checks. The second came after those
checks found that promotion is structurally blocked. Both are recorded verbatim below, after the
evidence each one rested on.

## Evidence, checked by the lead

### Refusal across versions

v3 → v4, the campaign's only dose-response on refusal data.

| Run | prose /60 | scrambled /60 | unseen-language /60 |
|---|---|---|---|
| J4 seeds (v3) | 2–9 | 3–8 | 20–56 |
| F seeds (v4) | 0–54 | 56–58 | 0–25 |

Sources: HANDOFF/gh200-phase4-2026-10-01.md:720-727, and ledger/gh200-p4-v4-2026-10-01.jsonl rows
f4feac15, aeca8d69, 8c3a774a, ebc83be6 and d8c8300a.

v4's refusal data, per HANDOFF/v5-plan-2026-10-02.md:55:
- 2,010 prose rows;
- 2,160 scrambled rows;
- 834 unseen-language template rows (csharp, elixir, kotlin, php, scala, shell).

### What v5 already adds

From HANDOFF/v5-plan-2026-10-02.md §1 and campaign/v5-preregistered.DRAFT.json data.sources on
v5-build:
- prose-noul goes from 2,010 to 6,010 rows (3.24%);
- the commitpackft G6 unseen-language source (≥1,000 rows over ≥5 languages, download approved at
  9f6b1fa);
- the CLINC oos re-key;
- longer composed rows, about 9.7k real tokens;
- a noul-weight arm.

After the human's strike, about 1,963 own-prose units remain, above the 1,500 floor
(DRAFT data.sources[3].human_strike).

The general sources are nearly exhausted (v5 plan §1):

| Source | Used | Available |
|---|---|---|
| MMLU | 14,200 | 14,327 |
| CSQA | 9,619 | 9,741 |
| SQuAD | 141,752 | 142,192 |

### The OOD gate's two halves

From python/qd_train/ood.py:55-57 and 205-230, with Wilson bounds at z=1.96 (needle.py:194):
- **Suite half:** passes if the Wilson lower bound on suite refusals is ≥ 0.90. At n=180 that
  needs at least 170 refusals. ens5 has 61.
- **In-distribution half:** passes if the Wilson upper bound on in-distribution abstentions is
  ≤ 0.05. At n=10,985 that allows at most 504. ens5 has 814; the seeds have 843–945.
- An in-distribution abstention includes a permuted pass that names another option.

### Per family on ens5 (b45406b5)

| Family | permutation_consistency | in-distribution abstention |
|---|---|---|
| code.defect_class | 99.44% (2291/2304) | 13/2304, Wilson upper 0.0096 |
| intent.in_scope | 99.5% | — |
| intent.domain | 97.5% | — |
| intent.classification | 97.0% | — |
| intent.within_domain | 90.7% | — |
| commonsense | 90.3% | — |
| knowledge (MMLU) | 77.2% | 339/1485 |

Pooled permutation_consistency is 10,288/10,985 = 93.7%, against a 95% floor.

### Gates and controls that never ran

On every v4 row:
- gates `ece` and `paired_margin_vs_linear` are not_run;
- controls `degenerate_head`, `privileged_hunk` and `transfer_gate` are not_run.

The ece reason recorded on b45406b5: 8,681 of 10,985 letter rows carry no language, so ece.lang
cannot be computed.

**Correction (added after the wiring checks).** The `degenerate_head` part of the list above is
wrong, and ruling 2 below repeats the error. That control ran on every score-val eval row: seed 0
passed it; ens5, avg and seeds 1–4 failed it. See the "Correction to the record" section of
`promotion-decision-brief.md`, in this directory.

docs/ledger-schema.md:261-292, Promotion: every gate must be `ran`, and every control `ran` and
`passed`.

docs/promotion-decisions.json (human-only edits) holds every one of these definitions as
**open**:
- promotion_population;
- average_may_promote;
- ece_population;
- degenerate_head_floor;
- privileged_hunk_pass_rule;
- transfer_gate_definition.

## Ruling 1, verbatim

**Ruling: the question splits by failing gate, and v5 already carries the data answer for the two gates that are data-shaped. Don't add data to v5 now; make v5's result readable as a dose-response, and name the gate that data cannot fix.** Record this verbatim in AUDIT like the last one.

## 1. Where data demonstrably moves the needle — refusal

Your v3→v4 table is the only dose-response evidence in the campaign and it's strong:
- **Scrambled:** ~2,160 noul rows took refusal from 3–8/60 to 56–58/60. Data works for this gate.
- **Prose:** ~2,010 rows (same dose) gives 0–54/60 across seeds. Same dose, unstable outcome — that's a near-threshold dose, not a ceiling. v5 triples it (6,010, 3.24%) plus the noul-weight arm. That *is* the "more data" experiment, pre-registered.
- **Unseen-language:** v4 capped at 834 *synthetic template* rows in 6 languages and got **worse** than v3 (20–56 → 0–25). Templates in csharp/elixir/kotlin didn't transfer to the suite's c/java/ruby/haskell/sql/lua. The fix is distinct real languages, which the human already approved (G6 commitpackft, ≥1,000 rows, ≥5 languages, none overlapping the suite). Also in v5.

**Check before you report:** compute what the gate needs. Its detail reads "Wilson lower … vs floor 0.9" at n=180. Verify, but a Wilson lower bound of 0.9 at n=180 needs roughly 170 abstentions — with scrambled at ~57, prose and unseen must each reach ~56–57/60. State that number; it tells the human how far "enough" is.

**Check:** the post-strike prose-noul count. The human struck personal+business own-prose; the 6,010 figure predates that. Find the applied count (the data strike landed as 5e4fa73 on v5-build). If own-prose fell below its 1,500 floor, there is **local** supply to backfill before the build: SQuAD paragraphs, which v4's 2,010 prose-noul rows already came from (~19k train paragraphs, cc-by-sa-4.0). That's an amendment before any row exists — the same pattern as the tierb one.

## 2. Where data cannot fix it — permutation consistency

Your per-family pull settles this. ens5 (b45406b5): code 99.4%, intent.in_scope 99.5%, domain 97.5%, classification 97.0%, within_domain 90.7%, commonsense 90.3%, **knowledge 77.2%**. The gate is the aggregate (10,288/10,985 = 93.7% vs 95%). Knowledge alone contributes 339 of the 697 disagreements; the floor allows ~549. MMLU train is exhausted (14,200/14,327) and CSQA too (9,619/9,741) — the v5 plan §1 already says "nearly nothing left." And the property — same answer under option reordering — is a training-procedure property, not a knowledge property. The lever is **C1, train-time option permutation**, decided by j6g.

Tell the human plainly: **more data does not address this gate; if j6g does not fire C1, v5 very likely fails permutation consistency regardless of data.** That's a risk to the Oct 7–9 estimate, not a data purchase.

## 3. Two gates nobody has run — this blocks "finalize" independent of data

On every gate row you read — avg, ens5, all five seeds — `ece` and `paired_margin_vs_linear` are **not_run**. House rule: a check that could not run is not a pass. G8 says "its own gate row passes"; five of seven gates ran. The v5 plan lists `ece_population` among items "already open." Find out, before the report: what each needs to run (a defined calibration population; linear-control rows, which j6ctl writes), and whether v5's row would run them. If not, v5 cannot be promoted either, and that is a bigger finalize blocker than data.

## 4. Pre-register the contingency, now

Add to the DRAFT's amendments_pending, before the data build: *"If v5's `ood_abstain.prose` or `.unseen-language` seed_holds target misses, the next step is [SQuAD-paragraph prose-noul at 2× / additional G6 languages at 2×], not a recipe change."* Written now, "more data" becomes a decision rule the result triggers, not a reaction to it — the discipline p2_amendment_2 taught.

## Do

1. §1 and §3 checks. 2. The three prose/post-strike/not_run numbers in the report; the permutation sentence verbatim. 3. The contingency amendment. 4. Don't touch v5's data otherwise; it is the experiment.

## Ruling 2 (reconciled after the checks), verbatim

**Reconciled ruling. Your §3 check found something larger than §3, and it goes at the top of the report, above data.**

## 0. What changed: promotion is structurally unreachable for any model, v5 included

ledger-schema.md Promotion: every gate `ran`, every control `ran` **and** `passed`. On every v4 row, `ece` and `paired_margin_vs_linear` are not_run, and the controls `degenerate_head`, `privileged_hunk`, `transfer_gate` are not_run. Their definitions sit as **open** items in `docs/promotion-decisions.json` (human-only edits, rule 2). v5's DRAFT does not touch any of them. So v5 would score and could not promote — regardless of data, C1, or seeds. This is the finalize blocker; the data question is downstream of it.

Lead with it. Everything below is how to clear it.

## 1. "More data?" — refined, and now conditional on one human decision

The OOD gate has two halves, and both fail: the suite half needs **170/180** refusals (you computed it; prose and unseen must each reach ~56–57/60), *and* the in-distribution half caps abstention at **≤504/10,985**, with ens5 at 814 and seeds at 843–945. The in-dist rule counts a permuted-pass disagreement as an abstention, so knowledge/MMLU drives this half *and* the permutation gate. One cause, two gate failures.

That makes "more data" depend on `promotion_population`:
- **Pooled (as built):** more noul data pushes in-dist abstention *up*, away from 504. The plan already flags w=5 as in-dist risk. More refusal data here is counterproductive until C1 lands. Say so plainly.
- **code.defect_class, general report-only:** permutation 99.4% and in-dist 13/2,304 (Wilson upper 0.0096) both pass. What remains failing is the suite half — the one gate the v3→v4 dose-response shows data moves, and the one v5's 6,010 prose-noul + G6 languages target. Under this reading "more data" is correct and v5 already carries it.

Post-strike check: ~1,963 own-prose units ≥ 1,500 floor; no backfill. Question form is 197, not ~599 — the 2,000 contrast rows are question-form, so note it, don't act on it.

One more non-data lever, from the gate's own detail string: "the calibrated-margin half is not applied (no fitted calibration table): it can only add abstentions." Calib-fit rows exist (ea9aa136). **Check** whether v5's scoring applies the fitted table; if the margin half runs, suite refusal rises — and so does in-dist, which again points at the population decision.

## 2. The human decision brief (your deliverable now)

One table, six rows, for each open item: the as-built value; what each option reads on ens5 (b45406b5) and on seed 0 with row ids; Fable's recommendation; and that the human writes the record, in its own commit, **before v5's first ft row's written_at** — ideally before the data build, so v5's gates are known when it's launched. Deciding now with v4 in view must be stated in `decision_ref`, with both readings' numbers; that transparency is what separates it from the post-hoc trap, and the gap predates the results.

Recommendations to carry:
- `promotion_population`: code.defect_class, general families report-only. The product's contract (CLAUDE.md: DevCouncil and DevType) is the code decision [wrong for DevType, and no caller is wired: see the Correction at the end]; the gates were defined when val was code-only; a pooled bar across families with different base rates is a contract no caller invokes. State the counter-argument too: the human may want knowledge held to a bar, in which case "each family separately" and C1 become the path, and no v4 passes.
- `ece_population`: run the gate pooled over all letter rows; per-language report-only where language exists. **Check** `ECE_THRESHOLD` against ens5's pooled 0.043 before claiming it would pass.
- `average_may_promote`: leave false; v5 promotes a three-seed recipe, not an average.
- `degenerate_head_floor`, `privileged_hunk_pass_rule`, `transfer_gate_definition`: each needs a definition or an explicit human retirement; until then no row promotes. Give the human what each control measures and the as-built reason it's not_run; don't propose values yourself without a Fable pass on each.

## 3. Wiring checks before v5 deploy — yours, no human needed

- `paired_margin_vs_linear` ran on the Sept 22 rows (gh200-tuned-capacity) and on nothing since. Find what flag or input produced it there, and whether `box_q_v5.sh`'s command carries it. If not, v5's rows are not_run on it by construction → an amendment on v5-build now, while the waiter is unlaunched.
- Same for `degenerate_head` ("never evaluated by this run").
- Confirm v5 ×3 yields three rows sharing `protocol_hash` differing only in seed (the `batch_order='seed'` constant was designed for recipe_hash; check protocol_hash too).

## 4. Still do

The §4 contingency amendment, now worded on the population decision: *"If v5's suite-half target misses under the recorded population, next is 2× SQuAD-paragraph prose-noul / 2× G6 languages, not a recipe change."* And record this ruling verbatim beside the last one.

## Correction (2026-10-03, the lead): no caller calls the code decision, and DevType's decision is untrained

The case for the code-only population said the product's contract is "the code decision that
DevCouncil and DevType call". That reason is wrong in two ways.

- **No caller calls Lappi today.** [V, by grep, not graph-confirmed: DevMap's store was malformed,
  GAP-DEVMAP-DATABASE-MALFORMED-2026-10-03] No DevCouncil, DevType or GitPulse code sends Lappi a
  request (GAP-SCHEMA-API-DOC-REQUEST-IS-NOT-A-TRAINED-REQUEST-2026-10-03, impact). The two callers
  are the plan's (CLAUDE.md, first paragraph), not wired.
- **DevType's decision is routing, and nothing trains it.** [V] The runtime names DevType as the
  socket caller (`crates/qd-runtime/src/serve.rs:3`). Its task id `devtype.route` appears only in
  runtime refusal fixtures (`crates/qd-runtime/src/fixtures.rs:526-530`). No training family is
  `devtype.*` or routing: `python/qd_data/sources.py` defines code.change_scope,
  code.commit_intent, code.defect_class, code.language_id, commonsense.multiple_choice,
  intent.{classification,domain,in_scope,within_domain}, knowledge.multiple_choice,
  qa.answer_span and qa.answerability. Routing is not a held-out family either. Those are
  code.language_id and qa.answerability (`crates/qd-train/src/held_out.rs:50-51`), so routing is
  simply untrained.
- **What remains true.** [I] DevCouncil's verdict is the defect decision: code.defect_class is
  the family it would call, and it is 84% of v4's tokens.

**The decision stands.** The record's `promotion_population` keeps its value
(docs/promotion-decisions.json). Of the three reasons given for it, two do not rest on DevType:

1. The gates were defined when val was code.defect_class only (the record's `source`).
2. A pooled bar across families with different base rates is a contract no caller invokes. That
   reason holds more strongly now that no caller invokes any contract.

code.defect_class is also the only family with a planned caller, DevCouncil.

DevType routing goes to the v6 caller-family lane with DevCouncil relevance, GitPulse commit type,
severity and commit_intent.
