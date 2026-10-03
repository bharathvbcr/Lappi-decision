# Does Lappi need more data, and what must be decided before anything can promote? (2026-10-03)

This answers the human's question "See if it needs more data, ask fable", using Fable's two rulings
(`fable-data-and-promotion-ruling.md`, main afcfee9) and the checks those rulings asked for.

Every number below is a ledger value. The rows used:

| Label | Row | Ledger |
|---|---|---|
| seed 0 | f4feac15 | `ledger/gh200-p4-v4-2026-10-01.jsonl` line 2 |
| ens5 (five-seed ensemble) | b45406b5 | `ledger/gh200-p6-f-j7prime-2026-10-01.jsonl` line 7 |
| avg (averaged weights) | c962cdd9 | `ledger/gh200-p6-f-j7prime-2026-10-01.jsonl` line 6 |

## The answer

1. **No more data now.** The refusal gate is the one gate that data has demonstrably moved, and v5
   already carries more of it:
   - prose-noul goes from 2,010 rows to 6,010;
   - G6 adds real unseen languages (commitpackft, at least 1,000 rows over at least 5 languages).
   - If that still falls short, the next step is pre-registered: 2× prose-noul from SQuAD
     paragraphs and 2× G6 languages, not a recipe change. That is v5-build a14ba6f, written
     before any v5 row exists.
2. **One gate is not about data.** In Fable's words: "more data does not address this gate; if j6g
   does not fire C1, v5 very likely fails permutation consistency regardless of data."
   - The MMLU and CSQA supply is used up (14,200 of 14,327 and 9,619 of 9,741 rows).
   - The lever is C1, train-time option permutation, which j6g decides.
3. **Nothing can promote, v4 or v5, until six items in `docs/promotion-decisions.json` are
   decided.**
   - Promotion requires every gate to have run, and every control to have run and passed
     (`docs/ledger-schema.md`, Promotion).
   - On every v4 row, `ece`, `privileged_hunk` and `transfer_gate` are `not_run`, and their
     definitions are open human items.
   - `paired_margin_vs_linear` and `shuffled_label` also read `not_run` on these rows. Both are
     agent-owned wiring rather than open human decisions, and both are wired for v5 (see
     "Wiring checks" below).
   - This is the finalize blocker. The data question sits downstream of it.
4. **No own-prose backfill is needed.** About 1,963 own-prose units remain after the human's
   strike, above the 1,500 floor.

## How far "enough" is: the OOD gate

`ood_abstain` has two halves, and both must pass.

### The suite half

The model must abstain on at least 170 of 180 out-of-distribution prompts. That is a Wilson lower
bound of at least 0.90.

| Suite | seed 0 | ens5 |
|---|---|---|
| prose | 54/60 | 3/60 |
| scrambled | 58/60 | 57/60 |
| unseen-language | 20/60 | 1/60 |
| **total** | **132/180** ✗ | **61/180** ✗ |

The suite is out-of-distribution by construction, so this half reads the same under any
population choice.

### The in-distribution half

The model may abstain on at most 504 of 10,985 val rows (a Wilson upper bound of at most 0.05).
A permuted second pass that names a different option counts as an abstention.

| Population | seed 0 | ens5 |
|---|---|---|
| pooled | 843 ✗ | 814 ✗ |
| code.defect_class only | 13/2,304 ✓ (Wilson upper 0.0096) | 13/2,304 ✓ |

Under the pooled population, more refusal data pushes this half the wrong way. The knowledge
family (MMLU) alone abstains on 317 rows (seed 0) and 339 rows (ens5).

## The six decisions

The human writes each decision in `docs/promotion-decisions.json` (human-only edits), in a commit
of its own that names the gap. The deadline is **before v5 seed 0's ft row is written**, and
ideally before the data build, so v5's gates are known at launch. Because v4 is in view while
deciding, `decision_ref` must say so and give both readings' numbers. The v5 DRAFT now pins this
dependency (v5-build a14ba6f).

| # | Item | As built | Seed 0 reads | ens5 reads | Fable's recommendation |
|---|---|---|---|---|---|
| 1 | `promotion_population` | All families pooled | Pooled: permutation 93.2% ✗, in-dist 843 ✗. Code-only: 99.44% ✓, 13/2,304 ✓ | Pooled: 93.7% ✗, 814 ✗. Code-only: 99.44% ✓, 13/2,304 ✓ | Code-only, with the general families report-only (see the counter-argument below) |
| 2 | `average_may_promote` | `false` | (n/a: a seed row) | (n/a: an ensemble). avg c962cdd9: permutation 91.9% ✗, in-dist 955 ✗, degenerate_head ✗, pooled ECE 0.0578 ✗ | Keep `false`. v5 promotes a three-seed recipe, not an average |
| 3 | `ece_population` | Per slot shape and per language. 8,681 of 10,985 letter rows have no language, so the gate is `not_run` | One pooled ECE: 0.0498 ✓. Per shape: k10 0.081 ✗, k15 0.190 ✗. Code-only: 0.0207 ✓ | One pooled ECE: 0.0431 ✓. Per shape: k10 0.099 ✗, k15 0.177 ✗. Code-only: 0.0135 ✓ | Run over all letter rows, with per-language report-only where a language exists |
| 4 | `degenerate_head_floor` | Mean entropy ≥ 0.15 nats and no class above 0.95, per slot shape over all families | Ran, **passed**. Lowest is k16 at 0.1509 | Ran, **failed**: k2 0.1415, k16 0.1383 | Define it or retire it. No value proposed without a Fable pass |
| 5 | `privileged_hunk_pass_rule` | Not set; the control stays `not_run` | `not_run` | `not_run` | Define it or retire it |
| 6 | `transfer_gate_definition` | Not specified; the control stays `not_run` | `not_run` | `not_run` | Define it (it also needs data) or retire it |

Bar for rows 1 and 3: permutation consistency must be at least 95%, and ECE at most 0.05.

### Notes on each row

**1. `promotion_population`**

Under "each family separately", no v4 row passes. Permutation consistency per family:

| Family | seed 0 | ens5 |
|---|---|---|
| knowledge | 78.7% | 77.2% |
| commonsense | 88.0% | 90.3% |
| intent.within_domain | 90.8% | 90.7% |

Five of the seven families also exceed the in-distribution cap.

- **Fable's case for code-only:** the product contract is the code decision that DevCouncil and
  DevType call. [Wrong for DevType, and no caller is wired: see the Correction at the end.] The gates were defined when val was code-only. A pooled bar across families with
  different base rates is a contract no caller invokes.
- **The counter-argument:** the human may want knowledge held to a bar. Then "each family
  separately" plus C1 is the path, and no v4 row passes.

**2. `average_may_promote`**

ens5 is shown for reference only. This item concerns averaged weights.

**3. `ece_population`**

Fable's wording, "over all letter rows", fits two readings:

| Reading | seed 0 | ens5 |
|---|---|---|
| One pooled ECE | 0.0498, under the bar by 0.0002 | 0.0431 |
| Per slot shape | fails | fails |

- The per-shape failures are both CLINC families: k10 is intent.domain and k15 is
  intent.within_domain.
- v5 re-keys CLINC, and the DRAFT already marks those ECEs as not comparable with F.
- The rust per-language ECE is 0.0525 (seed 0) and 0.0600 (ens5). Under the recommendation it is
  report-only.

**4. `degenerate_head_floor`**

The code family alone (report-only today) has:
- entropy of 0.0726 nats (seed 0) and 0.0724 nats (ens5);
- a top-class share of 0.270;
- ECE of 0.0207 and 0.0135.

That is a balanced, well-calibrated head, and it would fail the floor. The control takes no
labels, so it cannot tell confidently right from confidently wrong
(GAP-DEGENERATE-HEAD-FAILS-ON-BINARY-SLOTS-TOO). The pooled per-shape check passes on k4 only
because knowledge's 0.85 nats is mixed in.

**5. `privileged_hunk_pass_rule`**

The control is meant to be a ceiling: the model is shown the changed hunk. On commitpackft,
though, the "privileged" window is the whole file on 48.0% of mutated examples, and 76.9% of the
file at the median. As built, it would measure almost nothing
(GAP-PRIVILEGED-HUNK-IS-NEARLY-VACUOUS-ON-COMMITPACKFT). The human sets the eligibility share
and the pass rule, or retires the control.

**6. `transfer_gate_definition`**

The only statement of this control is in `docs/teacher-plan.md:126`: a model trained only on
mutations must beat the char-n-gram baseline on a natural held-out set. No natural held-out set
with these labels exists. Producing one means a human judging real commits
(GAP-TRANSFER-GATE-IS-NAMED-NOT-SPECIFIED-AND-HAS-NO-DATA).

## Wiring checks (done; no human needed)

Three read-only code checks, by Grep and Read. DevMap was not available to the agents, so
nothing here is graph-confirmed.

**`paired_margin_vs_linear` is not `not_run` by construction in v5.**
- It is written on a separate supplement row by `tools/ft_linear_control.py`, and promotion merges
  that row with its eval row (`python/qd_train/ledger.py:1588-1648`).
- In v4, supplement rows exist for seeds 0–2: c89b89a1, c0e438e6 and 3f72112a. All three read
  `not_run` because the intent.domain linear baseline "did not converge in 6000 iterations".
- Seeds 3–4, avg and ens5 have no supplement row.
- v5-build raises the cap to 8,000 iterations with step-halving
  (`build/v5-build-wt/tools/ft_linear_control.py:144-153`). v5's per-seed waiter runs the
  control.
- No amendment is needed. Not verified: that the fit converges on v5's data.
- Separately, this gate's opponent is disputed. The linear control recognises mutation operators
  rather than judging the change (GAP-THE-CONTROL-CLASSIFIES-BY-GENERATOR-NOT-BY-CHANGE).

**`shuffled_label` is wired, and it is the seventh control the promotion check needs.** It reads
`not_run` on seed 0, ens5 and avg because J5′ writes it on a separate row.
- `_record_shuffled_label` writes that row with the target eval row's own protocol and
  `eval_row_id` (`build/v5-build-wt/tools/real_ft_run.py:8852-8900`), so the row joins the
  target's seed family and is merged with it at promotion.
- On v5, `box_q_v5j5.sh` writes those rows into v5's own ledger (`--ledger "$V5_LEDGER"`, line 63).
- The J5′ now running on the box does the same for F seeds 0–2.
- No human decision is needed for this control. It passes or fails on J5′'s rows.

**The calibrated-margin half of `ood_abstain` is not a lever.**
- v5 does not apply it; the scoring path has no input for it (`tools/real_ft_run.py`).
- The quick, in-sample calibration fits show a fitted margin would abstain on 4,454 of seed 0's
  10,985 val rows (row ab966074 in `ledger/gh200-calib-fit-2026-10-01.jsonl`). That is about nine
  times the 504 cap.
- So the fitter's coverage target and the gate's in-distribution cap are incompatible as built.
  That is a defect in what the fit optimises, and it must be reconciled with the gate before any
  export ships a calibration table. The reconciliation touches a gate, so the human owns it. It
  is recorded here and not fixed.

**v5's three seeds form one promotable seed family.**
- `protocol_hash` includes the seed. Promotion compares `Protocol.hash_without_seed()`
  (`python/qd_train/ledger.py:517-521`, `:1251`).
- F's five seeds share all four non-seed components.
- In v5, `batch_order` enters the recipe as the constant `"seed"`, and the Mac prelude refuses a
  launch whose per-seed batch shapes differ.
- `docs/ledger-schema.md:267` says "protocol_hash", which is out of step with the code.

## Correction to the record

The data-ruling AUDIT (afcfee9) and Fable's second ruling say `degenerate_head` was `not_run` on
every v4 row. That is wrong.
- It ran on every score-val eval row: seed 0 passed; ens5, avg and seeds 1–4 failed.
- It reads `not_run` only on needle-control, linear-control and option-control rows, which do not
  evaluate it (`python/qd_train/ledger.py:1829-1842`).

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
  qa.answer_span and qa.answerability. Routing is not one of the two task-holdout families,
  which are code.language_id and qa.answerability (`crates/qd-train/src/held_out.rs:50-51`).
  It is held out in a different sense: the plan's held-out data is "300 hand-labelled diffs ...
  plus real DevType queries", never trained on (README.md:106-107 at HEAD). Neither set exists:
  nobody has labelled the diffs, and no caller logs queries (docs/train-plan-2026-09-28.md:265-267 at HEAD). So a v6 routing family must train
  on some other source, and real DevType queries judge it once they exist. Today DevType routes
  palette queries to `resolveDate`, `runMacro` and `findSnippet` with Apple's on-device model
  (`~/Code/apps/DevType/Sources/ExpanderEngine/AI/PaletteToolRouter.swift:7-10`).
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
