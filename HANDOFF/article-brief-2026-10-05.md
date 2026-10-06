# Brief for the LinkedIn-article session, 2026-10-05

The human, ~15:20Z: "I want to write a LinkedIn article about it in a new session". This file is
that session's source of truth. Every number cites a ledger row: rule 5 says a number without a
row stays out. The [V] items were checked on the H100 box's ledger
`/home/ubuntu/ledger/h100x2-v5-2026-10-03.jsonl` (68.209.74.244) at ~15:15Z.

**First command for the article session:** read this file, then
`HANDOFF/lead-pipeline-2026-10-03.md`, the top sections (newest first), for anything newer. The
v6x experiment's verdict lands ~19:30Z or ~02:00Z on 10-06, and its section there will say which.

## Update, 2026-10-06 ~02:35Z (supersedes the lines below where they disagree)

**Lappi v0.1 preview is released, not promoted.** The human chose this; the notes are
`HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md`.
- **What it is:** the average of v5's five seeds. Score row 6af73bef.
- **How it scores against the seeds:**
  - choice 85.7%, better than every seed;
  - option-order consistency 95.7%, the first to pass the gate;
  - "I don't know" on normal questions 4.6%, the first under the 5% cap;
  - unfamiliar input 149/180, below the seeds' 152-162.
- **What it serves:** choice questions only, at the human's 80% precision target. On held-out folds it answers 91.9% of questions at 87.0% precision; 16-option questions are refused about half the time.
- **Not load-tested.**

**v6x reads `no_signal`.**
- "I don't know" on unfamiliar input rose on 4 of 5 seeds (+6 to +16 of 180).
- Two guards lost: choice accuracy, and over-hedging on normal questions.
- It is report-only; nothing moved.

**Other findings:**
- **Shuffled-label control:** seed 3 now passes too (0.2183, deaf6717), so 4 of 5 seeds are covered.
- **A free calibration probe** showed that a confidence threshold cannot fix "I don't know" on unfamiliar input. It adds at most 8 of 180, at the cost of 30-39% of normal questions.
- **Averaging five seeds blurred the span head.** The seeds' span heads point in near-independent directions, so span accuracy fell to 77%. Span is not served.

**Cost:** the H100 box was ~$459 at shutdown-ready (02:29Z). The human chose to spend no more tonight.

**Weights:** the release, v5 seed 1 and v6x seed 3 are on the Mac; the box's other checkpoints go with the box.

**Rust and Python:** the "Rust cut Python bottlenecks" angle stays limited to what was measured
earlier. Tonight's two newly measured Python/CPU stalls (the GH200 post-training tail, and the
linear-control phase) are v6 targets, not done.

## What Lappi is

- One typed decision model derived from **Qwen3.5-2B-Base**. Any app calls it through a schema;
  DevCouncil and DevType are the first two callers (`CLAUDE.md`, `docs/lappi.md`).
- It answers a decision with a typed choice, or a span pointing into the input. It is built to
  say **"I don't know"** (abstain) rather than guess on input it was not trained for.
- **Rust does the systems work:**
  - `qd-prep`: MinHash/LSH dedupe and the linear-baseline fits. The box logs show it signing
    ~272k sets in 43 s and fitting the controls on 26 threads.
  - `qd-runtime`: the schema API and serving.
  - `qd-metal`: the Mac backend, on tessl's Metal kernels.
  - CUDA kernels in a sibling project (ojas). Its GH200 checks passed today: 174/174 rung-0
    checks, and 11 of 11 device-test binaries.
- Python orchestrates training (PyTorch).

## How it was trained (v5, the current model)

- 2× H100 on Lambda, $8.38/h. Five seeds, each one epoch: 12,176 steps, ~22,500 s of training.
- The recipe, as the lanes ran it at 142a67c: AdamW with fp32 master weights, lr 1e-5, lower 8
  layers at 0.1× the learning rate.
- Data: 22 task families. The core one is code-defect classification, built from CommitPackFT
  commits with mutation-made labels. The general decision families include VitaminC, BoolQ,
  ARC, commonsense and intent classification.
- Held-out data is never read by a training process; a path check in the trainer refuses it.

## Results (v5, five seeds; [V])

| Seed | Eval row | Span top-1 | Choice top-1 | "I don't know" on unfamiliar input | 8K long-context |
|---|---|---|---|---|---|
| 0 | 166f7ebd | 6593/7238 (91.1%) | 14498/17254 (84.0%) | 158/180 | 0.918 (fails 0.95) |
| 1 | e1bafd6f | 6626/7238 (91.5%) | 14591/17254 (84.6%) | 162/180 | 1.000 (passes) |
| 2 | 2af3c80d | 6606/7238 (91.3%) | 14470/17254 (83.9%) | 152/180 | 0.705 (fails) |
| 3 | e09b652a | 6570/7238 (90.8%) | 14616/17254 (84.7%) | 156/180 | 0.967 (passes) |
| 4 | a51c5bf4 | 6562/7238 (90.7%) | 14283/17254 (82.8%) | 155/180 | 0.966 (passes) |

**On the code-defect family**, the population the human's promotion rules judge
(`docs/promotion-decisions.json`):
- Every seed gives the same answer when the options are reordered on 99.2-99.5% of questions
  (2286-2293 of 2304; the floor is 95%).
- It wrongly says "I don't know" on 0.5-0.8% (11-18 of 2304; the cap is 5%).

**Controls**, the checks that the numbers are real:
- **Shuffled labels.** Trained on randomly shuffled answers, the model scores at chance: 0.2370
  (ba375781), 0.2296 (daf4ec89) and 0.2209 (8a47dd8a), against a ceiling of 0.3049. So the real
  scores are not leakage. All three passed.
- **A simple linear baseline,** seed 0 (row 0b0ac9a8):
  - Lappi beats it on 20 of 22 families, by +0.08 to +0.86 (code defects: +0.39).
  - It is behind on intent "in scope" by 0.017.
  - One synthetic family cannot be measured.

## What is NOT true yet: do not claim it

- **It is not released or promoted.** The repo's own verdict (`Ledger.promotion_verdict`, run
  ~17:30Z over the H100 ledger and `docs/promotion-decisions.json`) is **not promoted**.
  Promotion needs every gate to have run and passed on three seeds, and the decision is the
  human's alone. The blockers it names:
  - **"I don't know" on unfamiliar input** (unseen programming languages above all): FAIL on all
    5 seeds, 152-162 of 180 against a bar of ~170. Even the best seed answers instead of
    abstaining about 1 time in 10.
  - **The pooled baseline comparison did not run** on all 5 seeds (synthetic tasks with no
    training rows). A gate that did not run blocks; it does not pass.
  - **The long-context test fails** on seeds 0 and 2.
  - **The shuffled-label control was run on seeds 0-2 only**, so seeds 3 and 4 lack it.
- **What passes under the human's recorded rules:** option-order consistency (99.2-99.5% on code
  defects), calibration error (ECE, per slot shape) and the degenerate-head check. Two gates are
  retired for v5: privileged hunk and transfer.
- **Serving still needs a fitted calibration table.** The loader refuses a model without one, and
  none has been fitted for v5. That is a separate thing from the ECE gate.
- **No comparison with any other model** (GPT, Claude, Llama or the like) was run. Do not imply one.
- **v6x is an exploratory, report-only experiment** (`campaign/v6x-noulw-explore-preregistered.json`):
  it weights "I don't know" examples 4× in training. Nothing promotes on it.
  - Seed 0 did not move (158/180); seed 1 rose to 168/180 (c26eb4aa).
  - Its verdict comes tonight. Until then, only "an experiment is under way".
- **Cost.** The H100 box's spend is $363 at 15:08Z, ~$466 projected to 02:00Z on 10-06. That
  covers v5's training, its controls and v6x. Earlier phases (GH200, Mac) are not in that figure.

## Angles that are true and checkable

- The rigour: experiments pre-registered before results were read, five seeds, gates nobody may
  loosen after the fact, and controls that would expose a fake result.
- The model is built to abstain, and the article can say honestly that this is still its weakest
  point.
- Rust where the CPU work is, which cut the Python bottlenecks. Each claim must stay a measured one.

## Open for the human

- **Links.** Lappi's repo is private for now and planned to go public (memory: repo visibility).
  Whether the article links it is the human's call.
- **Hosting.** The model weights live only on the H100 box until they are copied off; the human
  has been asked.
