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
| peers' figures | `AUDIT/prior-art-jev-nimble-2026-09-19.md`, `docs/build-order-2026-09-19.md` (README-level claims; none measured here) |
| prompt format 2 text, the 21 trained questions | rendered by `python/qd_data/render.py` from load-test request r023; `python/qd_data/sources.py:716-770` |

**Left out on purpose:** the GH200 cost floor ($129.35 in the story notes; not found in the cited
files by this lane), and any "no gate moved", "trained without PyTorch", "calibration guarantee",
"span grounding shipped" or "DevCouncil uses it" claim (the story notes' do-not-claim list).

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
   for the release notes' all-families figures.
3. **Make `bharathvbcr/Lappi-decision` public.** The README, card and article point at it.
4. **Push the `outreach-2026-10-06` branch,** or merge it into main (resolving the `README.md`
   conflict above).
5. **Share the JevArena page** (claude.ai/artifact/Hdq9CDPFQrqN6DJLnkKNEd, owned by the lead
   session) or give a public location for it; the drafts mark the link pending.
6. **Share the Lappi story page** this lane published privately.
7. **Create the Hugging Face repo and upload** the release (`hf-release-plan.md`: redacted manifest,
   no span head, `LICENSE` and `LICENSE-Qwen`), with your own token.
8. **Settle the two licence questions** in `hf-release-plan.md`: ShareAlike on the CC BY-SA rows,
   and DeepSeek's output terms for the synthetic source.
9. **Push and deploy the Portfolio** branch (`lappi-v0.1-preview`), after its local CI.
