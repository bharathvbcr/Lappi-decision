# Outreach drafts for the Lappi v0.1 preview (2026-10-06)

Drafts only. **Nothing here has been posted, pushed, uploaded, deployed or shared.** Each outward
action needs the author's explicit yes (list at the end).

| File | What it is |
|---|---|
| `linkedin-article.md` | The LinkedIn article, in the author's voice |
| `hf-model-card.md` | The Hugging Face model card (the HF repo's `README.md`) |
| `hf-release-plan.md` | The upload plan and the licence/attribution checklist |
| `evidence/served_families.py`, `evidence/served_families.txt` | The one analysis this lane added: what the headline numbers cover |
| `../../README.md` | The repository README, rewritten as a public overview |

Also from this lane, outside this repository:

- the Portfolio's Lappi pages: branch `lappi-v0.1-preview` in the Portfolio repository;
- a private claude.ai page telling the Lappi story.

## Two corrections to the brief this lane was given

1. **"Only the 15 general decision families' choice slots are served" conflates two counts.**
   - **15** is the number of calibration entries (`choice:3` ... `choice:17`, one per slot width,
     so 2 to 16 options), and also the number of decision-pool families the load test replayed.
   - The runtime admits any of the 23 trained families (`crates/qd-runtime/src/admission.rs:103-125`).
     21 of them have choice slots and calibration, so **21 are served**.
   - 6 of those 21 were never sent through the runtime: `intent.classification`, `intent.domain`,
     `intent.in_scope`, `intent.within_domain`, `knowledge.multiple_choice` and
     `commonsense.multiple_choice` (the load test's `requests.jsonl` holds the other 15 only).
   - The drafts say 21 served, 15 load-tested.
2. **The headline numbers include a family the release does not serve.**
   - Choice top-1 0.857 and two-fold coverage 91.9% at 87.0% are over all 17,254 val choice rows.
     That includes 2,304 `code.defect_class` rows at 0.954.
   - On the 21 served families: **0.842** (12,580 / 14,950), and **90.7% answered at 85.6%
     precision** (`evidence/served_families.txt`). The script reproduces the published figures
     exactly before it splits them.
   - The card and README give both, labelled. The article uses the served figures, rounded.

## Where each number comes from

Paths are relative to this repository unless they name the campaign directory.

| Number | Source |
|---|---|
| 0.857, 0.957, 4.6%, 149/180, 1.000, 0.773; the six-line bar | `HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md` (score row 6af73bef) |
| 40/60, 49/60, 60/60 by OOD kind | same file |
| 91.9% / 87.0% two-fold, 93.8% / 87.1% in-sample, 51% of 16-option refused | same file |
| ECE 0.0056-0.0379 on 8 of 9 widths; 0.079 on 16 options | same file |
| 0.842; 90.7% / 85.6%; per-family top-1 | `evidence/served_families.txt`, from the row's verdict file and two-fold rows |
| 89 of 90 served equal; 85 answered, 70 correct, 5 abstained; 0.12 s median, 3.37 s first | release notes, load test |
| JevArena table, +0.035 [-0.039, +0.094], +0.200 [+0.133, +0.257], 0.111, 53/43/12 of 108 | `HANDOFF/merge-and-bench-2026-10-06.md` section 6 |
| 72 ms at 124 tokens, 127 ms at 408, 0.41-0.44 s at 2,042, 2.3 s at 8,185; PyTorch MPS 1.6-3.2x slower | same file, section 5 |
| 12,176 steps, ~6.1 h per seed; ~$463 for the 2x H100 box | `HANDOFF/lead-pipeline-2026-10-03.md` lines 19 and 1681 |
| AdamW fp32 masters, lr 1e-5, lower 8 layers at 0.1x | `campaign/v5-preregistered.json` `recipe.base`; `HANDOFF/article-brief-2026-10-05.md` |
| PyTorch (torch 2.10, fla) on 2x H100 | `campaign/v5-preregistered.json` `hardware.environment` |
| shuffled labels 0.2183-0.2370 vs 0.3049 on seeds 0-3 | release notes, Controls |
| linear baseline beaten on 20 of 22 families (v5 seed 0) | `HANDOFF/article-brief-2026-10-05.md` (row 0b0ac9a8) |
| 487,407 train rows, 23 families, rows per family, licence histogram, held-out families | the release's `release_manifest.json` and the v5 train manifest |
| 606K params, 80.97% vs 88.5%, 17 of 18 arms | `docs/lappi.md` at HEAD |
| 80.8 GB peak on 64 GB | `docs/train-plan-2026-09-28.md:331` |
| 215 overlapping validation rows | `docs/lappi.md`, "What the 2B campaign has taught" |
| 1,087 commits, 1,060 with a Claude co-author trailer; commits by day | `git log` at 6d265c0 |
| two kernel panics on 10-02 | `AUDIT/mac-stability-2026-10-02/report.md` |
| peers' figures | `AUDIT/prior-art-jev-nimble-2026-09-19.md`, `docs/build-order-2026-09-19.md` (README-level claims, not measured here; decider-2b's JevArena run below is the one peer measurement) |
| prompt format 2 text, the 21 trained questions | rendered by `python/qd_data/render.py` from load-test request r023; `python/qd_data/sources.py:716-770` |
| Gemma 4 E4B reasoning pilot: 191 pairs; 0.708 / 0.816 / 0.853 / 14.0 s; Lappi 0.529 / 0.571 / 0.639 / 0.24 s on the same pairs; +0.245 [+0.122, +0.347] | campaign `lappi-bench-2026-10-06/jev-compare-pilot.txt` (`compare_all.sh`); the run's status `complete`, 382 calls, in `logs/jev-gemma4-think-pilot-run2.log` |
| decider-2b v11 (rev 533964da) on JevArena, CPU float32: 849 pairs, 0 failed calls; macro 0.612, micro 0.719 [0.661, 0.767], consistency 0.749, median 5.7 s; minus Lappi +0.110 [+0.071, +0.157]; RM-Bench code 0.546 (6 groups); on the 191-pair pilot 0.625 / 0.707 [0.590, 0.796] / 0.764 / 6.9 s, minus Lappi +0.136 [+0.048, +0.226] | `external/jevarena/runs/decider-2b-v11-cpu-test-g6` (status `complete`, 1,698 calls, `implementation_sha256` `d39ba018…`; the harness's own `report.md` agrees on pairs, consistency, latency and the code slice). Scored by this lane with the bench lane's `compare.py` and `compare_all.sh`'s run list, into this lane's scratch space, not the bench folder; every other row reproduced the committed `jev-compare-*.txt` files line for line. Not yet in `HANDOFF/merge-and-bench-2026-10-06.md` section 6, which says its row "lands in a follow-up commit" |
| decider-2b trains pairwise response preference and the Jev API's input shapes; holds out (the first) RewardBench; one temperature per answer type | its model card (`README.md`, Training and Evaluation) and `decider_config.json` in the snapshot; the request shape the harness sends it: `lappi-bench-2026-10-06/decider_stdio.py` docstring |
| RM-Bench code pairs reach Lappi through `pairwise.helpfulness` (2% of training, weakest family) | GitPulse card `ft-882ed8fcd29cfacd3e1cf14748a0e9de` (Fable's diagnosis, revision 2) |
| Defect probe: C1 0.956 on 2,304 val rows; C2 0.906 [0.835, 0.965] of 85 (logic 0.960, stub 1.000, cosmetic 0.708); RM-Bench 0.536 [0.448, 0.624] of 125; 206 programs refused by language | campaign `lappi-bench-2026-10-06/defect-probe/`: `report-c1full.txt` (C1, PASS), `report.txt` (C2 and A, first pass, primary), `REREGISTRATION-c1full.md`, design in `build_requests.py`'s docstring |
| MMLU test and dev trained on; CLINC150 test utterances of training intents trained on; CommonsenseQA val is a curated internal val | `python/qd_data/sources.py:196-204`, `:219-227`, `:486-493`; GAP-DECISION-INDEX-PANEL-CLINC-AND-MMLU-TEST-ITEMS-ARE-LAPPI-TRAINING-DATA-2026-10-03 |
| Decision families checked against ARC test, BoolQ val, VitaminC test and JevArena's sources; 1 `pairwise.helpfulness` row hit jevjudge; remaining hits 0 | campaign `v5-decisions-data-2026-10-03/pool-v5-decisions-v4/containment/attestation.json` |
| A random added line scores 90.8% where the span head scored 92.3% (diff mode, an earlier run) | GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD (2026-09-22) |
| Span rows: `qa.answer_span` 3,753 / 5,139 (0.730), `code.defect_class` 1,843 / 2,099 (0.878); 5,596 / 7,238 in all | campaign `preview-2026-10-05/v5-avg-verdicts/verdicts-avg.jsonl`, `kind == span` rows counted by family by this lane |
| 1,881,825,088 parameters, 320 tensors | the release's `model.safetensors` header, summed by this lane |

**Left out on purpose:**
- the GH200 cost floor ($129.35 in the story notes; not found in the cited files by this lane);
- any "no gate moved", "trained without PyTorch", "calibration guarantee", "span grounding
  shipped" or "DevCouncil uses it" claim (the story notes' do-not-claim list);
- "picks the longer program on 37 of 55 calls": the GitPulse card asks for the style prior to be
  measured on more than 6 prompt groups before any public claim;
- calling the RM-Bench 0.111 "a defect in its own domain": the same card says the pairs never asked
  the code-defect question. The defect probe now asks it, and the drafts report that instead;
- the needle misses' diagnosis (one hunk early, Swift and TypeScript only;
  `HANDOFF/needle-span-mac-2026-10-06.md`): no public claim depends on it, and the averaged
  weights miss none.

## Re-audit against the 2026-10-06 afternoon results

What the new results changed in the drafts:
- **The Gemma 4 E4B reasoning pilot finished**, so every "pending" now gives its result, on its own
  191-pair table and never mixed into the 849-pair one.
- **The RM-Bench framing.** The drafts had called 0.111 a defect in Lappi's own domain. They now say
  the pairs arrive as a helpfulness question, and they report the defect probe: it separates
  planted mutations but not RM-Bench's correct and broken programs.
- **A disclosure the card lacked:** MMLU's and CLINC150's test items are training data, so those
  scores are not benchmark results.
- **The span caveat:** span top-1 pools SQuAD answer spans (5,139 val rows, 0.730 for the average)
  and code-change evidence lines (2,099, 0.878). On the code rows it is weak evidence, because the
  diff's `+` markers show the changed lines. The caveat is scoped to those rows only.
- **The abstain row** is described as a design choice, not a measured advantage (the literature in
  `AUDIT/training-audit-2026-10-06.md` 4.1 says a reserved abstain class adds no separating
  information, matching the OOD miss).
- **decider-2b's full JevArena run finished** (14:49Z to 17:44Z, CPU, every call answered). It is
  better than Lappi: +0.110 [+0.071, +0.157] on the 849 pairs, +0.136 on the pilot's 191. The card
  carries its rows in both tables, the story page its 849-pair row and its pilot score, and the
  README, article and Portfolio branch the 849-pair result;
  the "not run head to head" paragraph in the article and story page now says only Jev and Nimble
  are unmeasured. Each surface states the asymmetry once, as fact, not as an excuse: pairwise
  preference is one of decider-2b's trained tasks, asked in its own request format, and for Lappi it
  is neither. Each also says its time is a CPU time, so speed is not compared. Left out on purpose:
  - any ranking of decider-2b against Gemma (0.612 against 0.632 macro): `compare.py` computes
    paired intervals against Lappi only;
  - "decider-2b handles the code pairs Lappi cannot": its 0.546 on RM-Bench code is 6 prompt groups,
    as exploratory as Lappi's 0.111;
  - any contamination claim about decider-2b: its README holds out the first RewardBench, not
    RewardBench 2, JudgeBench or RM-Bench, and whether its training data overlaps those three was
    not checked. Lappi's decision pool was checked (8-gram containment, above).

Unchanged, and re-confirmed:
- the 849-pair JevArena numbers (`jev-compare-all-test-g6.txt`, regenerated 09:29, same values;
  re-run with decider-2b added, every other row is identical);
- the served-families split (the training audit, 2.5, cites this lane's evidence file);
- 487,407 training rows: the data-clean plan's "558,790" counts shard sequences, which are not 1:1
  with rows (`HANDOFF/v6-startup-2026-10-04.md:360-361`).

## The README conflicts with held work in main's checkout

Main's checkout holds 16 uncommitted paths whose owner is unknown, `README.md` and `CLAUDE.md` among
them (GAP-MAIN-UNCOMMITTED-BATCHED-DECODE-FAILS-ITS-DIGEST-PIN-2026-10-06). This branch rewrites
`README.md`, so merging it will conflict with that held edit.

- The held edit is a 2026-10-04 status update. Among other things it:
  - says v5 runs "24,000 steps per seed" (the ledger says 12,176);
  - describes the batched decode as landed, though that work fails its own digest-pin test;
  - puts a local absolute path in the test command.
- Nothing in main's checkout was touched, stashed or committed by this lane.

## Actions that need the author's yes

Each is separate; a yes to one is not a yes to another.

1. **Post the LinkedIn article** (`linkedin-article.md`), after you edit it.
2. **Use the derived served-families numbers in public text** (0.842; ~91% at ~86%), or drop them
   for the release notes' all-families figures. The same question applies to the defect-probe,
   reasoning-pilot and decider-2b results: all three sit in the campaign bench directory or the
   harness's run directory, and their owning lane has not written them up in a handoff yet.
3. **Make `bharathvbcr/Lappi-decision` public.** The README, card and article point at it.
4. **Push the `outreach-2026-10-06` branch,** or merge it into main (resolving the `README.md`
   conflict above).
5. **Share the JevArena page** (claude.ai/artifact/Hdq9CDPFQrqN6DJLnkKNEd, owned by the lead
   session) or give a public location for it; the drafts mark the link pending.
6. **Share the Lappi story page** this lane published privately.
7. **Create the Hugging Face repo and upload** the release (`hf-release-plan.md`: redacted manifest,
   no span head, `LICENSE` and `LICENSE-Qwen`), with your own token.
8. **Settle the licence questions** in `hf-release-plan.md`: ShareAlike on the CC BY-SA rows,
   DeepSeek's output terms for the synthetic source, and OpenAI's terms for WANLI's GPT-3 seeds
   (behind Open-Jev's NLI rows).
9. **Push and deploy the Portfolio** branch (`lappi-v0.1-preview`), after its local CI.
