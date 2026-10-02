# The human's decisions on Fable's 2026-10-02 optimize ruling

Asked by the lead in chat on 2026-10-02 at about 05:15 UTC, through AskUserQuestion, after the ruling (`fable-optimize-ruling.md`, beside this file) named these items as the human's. Questions and answers are recorded verbatim. Rows that already exist are not changed: F's rows stand as measured, on the suites and populations they were measured on.

## 1. J6(g) rental hours

- Question: "J6(g) (F plus shuffled answer options during training, one seed) adds about 5 hours of GH200 rental after the idle queue, about $11. Run it?"
- Answer: **"Yes, after j6a (Recommended)"**
- Effect: J6(g) is queued as the last idle-queue item (… → j6a → j6g → STOP) under `campaign/j6g-preregistered.json`. Its waiter's `J6G_HUMAN_HOURS_YES` pin carries these words. Staging still needs the checker, the Mac prelude STOP gate and its pinned record (lane L-j6g).

## 2. CLINC out-of-scope keying in v5

- Question: "In v5, should CLINC's 1,350 out-of-scope examples get individual keys so they spread across train, validation and held-out? This changes the CLINC validation and held-out populations, so v5 numbers aren't directly comparable to F's on CLINC."
- Answer: **"Re-key in v5 (Recommended)"**
- Effect: the v5 build commit keys CLINC oos utterances individually (GAP-CLINC-OOS-ONE-REPO-KEY-ONE-SPLIT-2026-10-02). It is the one `python/qd_data` edit v5 carries, and it lands in the v5 build commit after the last v4 reader on main has run (Fable Q2). v5's CLINC val and held-out populations differ from v4's, and v5 arms are not read against F's envelope.

## 3. Needle 8K suite sizing

- Question: "The 8K long-context test cases average about 8,473 tokens, not the intended 8,192. Should the test be resized, or the training data lengthened to cover it?"
- Answer: **"Resize test to 8,192"** (option text: "Rebuild the test suite to its stated length. That changes the gate's test cases, and can be combined with longer training.")
- Effect: the 8K needle suite is rebuilt so its cases measure to the 8,192-token target in real tokens, not the 3-chars/token guess (GAP-NEEDLE-SUITE-EXCEEDS-TRAINED-WIDTH-2026-10-02). This is a gate-suite change made by the human, which rule 2 allows; no agent moves the 0.95 threshold. It applies to runs scored after the rebuilt suite is committed, starting with v5. F's needle rows (f4feac15 and its seeds) stand as measured on the current suite and are not re-scored as gate rows. The answer's option text allows combining it with longer training, so v5 still composes to the 9-10k target Fable named (data-only).

## 4. v5 rental

- Question: "After the idle queue, should I plan the v5 run (decontaminated data, longer examples, recipe fixes): 3 seeds + J5 + controls, roughly $70 on the GH200?"
- Answer: **"Plan it; ask before launch (Recommended)"**
- Effect: the lead prepares the v5 build, its pre-registration and its queue, then brings the human the final cost and timeline for a yes before any GPU time. No v5 GPU job launches on this answer alone.

## Still open (the human's, unchanged by these answers)

- G1 / `promotion_population`: on MMLU, permutation consistency and the in-distribution abstention cap cannot both pass under the pooled population (GAP-ABSTAIN-GATE-IN-DIST-EQUALS-PERMUTATION-DISAGREEMENT-2026-10-02).
- `ece_population`: whether general-family rows count in `ece.lang` (GAP-ECE-LANG-REASON-STRING-MISATTRIBUTES-ROWS-2026-10-02).
- `degenerate_head_floor`.
- G6 (unseen-language abstention data), G8, G12 and H4/G5.
- Content-disjointness of the two task-holdout families (GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN).
- Which linear-control row the margin gate reads after the refit (GAP-LINEAR-CONTROL-RERUN-WOULD-DOUBLE-THE-SELECTED-ROW-2026-10-02).
- The first commit of the untracked `ojas-qwen35-cuda/` crate into ojas git, raised by the ojas session with the human.
