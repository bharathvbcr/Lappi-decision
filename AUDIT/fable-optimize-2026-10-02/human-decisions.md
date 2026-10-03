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

## Second round: on Fable's improve/Mac-inference ruling (asked ~06:20 UTC 2026-10-02)

The ruling is `fable-improve-and-mac-inference-ruling.md`, beside this file. Questions and answers are verbatim.

### 5. Source of the extra prose noul rows (G6)

- Question: "Where should the extra prose 'decline to answer' training examples come from (G6)? Fable ranks this the top fix for seed-to-seed instability."
- Answer: **"Both"** (option text: "Use both sources. Gives the strongest signal, but carries the test-meaning change too.")
- Effect: v5's noul supply adds at least 3,000 prose-noul rows, question form favoured, from two sources:
  - (i) prose from the human's own repositories (READMEs, commit bodies, issue text), approved for training on 2026-09-28;
  - (ii) MMLU/CSQA **train** questions rendered under `code.defect_class` with gold Z, beside their own letter-gold rows.
- Rule 3 holds: no val or held-out question is used, and the v5 exclusion list removes E_val hits from both uses.
- Consequence the human accepted: under (ii), the OOD suite's prose category (val MMLU/CSQA questions under `code.defect_class`, `python/qd_train/ood.py:10-12`) is near-in-distribution by form for v5-trained models. v5's `ood_abstain.prose` is therefore not comparable to F's, and v5's pre-registration says so.

### 6. The noul-weight arm

- Question: "Fund an abstention loss-weight arm in v5 (abstain examples weighted 3–5×, three seeds, about $34 extra)?"
- Answer: **"Yes, 3 seeds in v5 (Recommended)"**
- Effect: a new `--noul-weight` flag; the arm is pre-registered before any row and runs on three seeds with v5. It launches only with v5, which still needs the human's final yes (item 4).

### 7. Mac GPU checks for qd-metal

- Question: "May lanes run the Mac GPU checks for the Metal backend (parity against the PyTorch reference, the snapshot-hash test, a latency benchmark)? An earlier session's permission check blocked these, so no Mac inference number has been measured with the real model."
- Answer: **"Yes, one at a time (Recommended)"** (option text: "Only when the Mac is free: no other heavy process, swap not near its limit. It loads the 3.8 GB model, plus the PyTorch reference once. Results go to a new Mac ledger.")
- Effect: this answers GAP-QDM-GPU-GATES-NOT-RUN's permission question. Lanes run `qd-metal-parity`, the snapshot-hash GPU test and the `--decision` benchmark one process at a time, only when no other heavy process runs and swap has room, and write rows to `ledger/mac-qd-metal-2026-10-02.jsonl`.

### 8. Prompt order in v5

- Question: "Should v5 put the code context before the question in the prompt? Then N questions about the same diff share one prefill (DevCouncil asks up to 128). It's a prompt-format change, so renderers, test goldens and v5's data all change."
- Answer: **"Context first in v5"** (option text: "Large serving savings for multi-question callers. It adds a format change to v5, so its effect is mixed with v5's other changes.")
- Effect: the v5 build renders the context before the question, in both renderers (Rust `crates/qd-runtime/src/render.rs` and the Python renderer). The `render_contract` goldens are regenerated for the v5 format version. The Python side lands in the v5 build commit with the other `python/qd_data` edit (Fable Q2), and the prompt's version marker changes so a v4-format model and a v5-format runtime refuse each other. v5's results carry this confound with its other changes.

### 9. More training data

- The human, in chat at ~06:25 UTC 2026-10-02, unprompted: "Train on more data if needed"
- Effect: the v5 plan may grow the training corpus beyond v4's sources where Fable's ruling or the gates call for it. The candidates are more own-repo code and prose (approved 2026-09-28), more noul sources, more long composed rows for the 9-10k target, and more rows from the already-downloaded MMLU/CSQA/CLINC/SQuAD train splits. Four limits still hold:
  - Rule 3: no val, held-out or task-holdout row ever trains.
  - The v5 decontamination scan covers every added row.
  - Every third-party source needs its licence checked.
  - A new download needs the human's explicit yes, with filename, source and size stated; this message does not cover downloads.
- The v5 pre-registration names each added source, its row count and its licence.

### 10. Decisions to Fable; the budget ceiling lifted (2026-10-03)

- The human, in the lead's session, in chat:
  - ~13:14Z: "For decisions ask fable and proceed with them";
  - ~14:40Z: "Ask fable for decision, I don't care about budget if it's gonna be a great model".
- Effect:
  - The open rulings are Fable's, recorded with decided_by "the human, by delegation (...), ruled
    by Fable".
  - The cost ceiling on v5 ($137 projected) is lifted. Fable's rulings of 2026-10-03 under it:
    - the full admitted decision-data set, still quality-capped;
    - v5 main-arm seeds 3-4;
    - the calibration fit reopened as a CPU lane, with c = 2.
    The record is AUDIT/hallucination-2026-10-03/ and the v5 draft's amendments.
  - The calibration fit was put to the human directly, as item 3 of
    AUDIT/finalize-2026-10-03/report-to-human-2026-10-03.md, and the human handed it to Fable. That
    is why it reopens despite the bench-session "no calibration bind in v5".
- Four limits still hold. The sentence does not cover them:
  - Rule 4 is unchanged. Every job carries its wall-clock cap, auto-terminate and cost estimate.
  - The draft's launch rule still fires. At launch the human sees the final plan, cost and
    timeline and says go; the cost half is answered in advance.
  - Downloads still need the human's per-action yes, with filename, source and size. Fable cannot
    grant them, and neither can this sentence.
  - The hold on the v5 data build until the human reports the Apple Diagnostics result stands.
- What stays v6 whatever the budget, per Fable: soft-label loss, score slots, the exact
  expected-reward loss, a p_top runtime rule, and self-labelled abstention. The human's "data
  only" for v5 (bench session) was a scoping answer, and cost does not reopen it. Stacking
  untested recipe changes into one run would also make its result unattributable.

## Still open (the human's, unchanged by these answers)

- G1 / `promotion_population`: on MMLU, permutation consistency and the in-distribution abstention cap cannot both pass under the pooled population (GAP-ABSTAIN-GATE-IN-DIST-EQUALS-PERMUTATION-DISAGREEMENT-2026-10-02).
- `ece_population`: whether general-family rows count in `ece.lang` (GAP-ECE-LANG-REASON-STRING-MISATTRIBUTES-ROWS-2026-10-02).
- `degenerate_head_floor`.
- G6's unseen-language half (the prose half is item 5), G8 (ens3 serving cost), G12 and H4/G5.
- A 4B base (Fable recommends against until v5 reads); the qd serve idle-timeout default for a resident 3.8 GB model.
- Content-disjointness of the two task-holdout families (GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN).
- Which linear-control row the margin gate reads after the refit (GAP-LINEAR-CONTROL-RERUN-WOULD-DOUBLE-THE-SELECTED-ROW-2026-10-02).
- The first commit of the untracked `ojas-qwen35-cuda/` crate into ojas git, raised by the ojas session with the human.
