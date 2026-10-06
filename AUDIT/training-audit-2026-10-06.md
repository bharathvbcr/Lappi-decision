# Training audit for the next Lappi model: gaps, literature, mitigations (2026-10-06)

**Status: read-only audit.** Nothing was trained, launched, rented or merged. No gate, threshold,
seed or held-out set was touched (rule 2). No code, `gaps.jsonl`, `HANDOFF/` or `docs/` file was
edited; the proposed gap records in section 8 are text for the lead to append.

**Labels.** Repo claims: **[V]** read or run this session (file:line, ledger row or command),
**[I]** inferred, **[U]** unknown. External claims: **[fetched]** the primary page was read by a
research agent this session; **[snippet]** search-result text only. Research agents read
abstracts and HTML pages through a summarising fetcher, not full PDFs, so every external number
needs a spot-check against the paper before it backs a gate or a ledger row.

## 1. Scope and method

- **Question (the human, 2026-10-06):** a deeper audit of training advancements, Lappi's current
  gaps, and how to mitigate them, for a larger decision model serving jevgrep, Jarvis (computer
  use, forms), the email sorter, ScholarLM (reranking), and tool calling for DevCouncil, GitPulse
  and MANVI.
- **What ran:**
  - DevMap (generation 3433, `source_freshness: fresh`) and `gitpulse_insights` (4 of 4
    worktrees scanned, 0 overlapping files) [V].
  - `ListAgents`: two live Lappi peers (merge-and-bench lane; outreach lane). This file was
    claimed with both by `SendMessage`, and both acknowledged.
  - Six agents: one classified every open gap; five researched the literature (abstention and
    calibration; long context and weight averaging; efficient 4B-9B training; app-task recipes;
    data and evaluation).
  - The repo checks in section 2, by this session.
- **What did not run:**
  - **ScholarLM MCP was unavailable:** the cloud connection returned 503, and the local one
    refused the connection. Literature came from web search and fetches instead. WisDev was not
    used as a fallback.
  - No full PDFs were read; no model was loaded; no eval was re-run.

## 2. Repo findings verified in this session

These are what the literature cannot tell you, and several change the v6 plan.

### 2.1 The needle failure is at the END of the context; the gate's diagnosis is boilerplate [V]

- **Depth convention:** depth is 0.0 at the start of the context and 1.0 at the end
  (`python/qd_train/needle.py:134`, `:148-155`).
- **The "losing early context" wording is fixed text.** `needle.py:477-479` appends "A recurrent
  model losing early context looks exactly like this" to every failing verdict, whichever bucket
  failed.
- **What v5 actually measured** (`/Users/bharath/qd-campaign/ledgers-h100-2026-10-05/ledger/h100x2-v5-2026-10-03.jsonl`,
  `metrics.needle_hunk_recall.depth.*`; seed read from each row's notes). Buckets 0-20%, 20-40%
  and 40-60% are 1.000 on every seed. Every miss is in the last 40%, the part nearest the
  question:

  | Seed (row) | 60-80% | 80-100% | Gate (worst ≥ 0.95) |
  |---|---|---|---|
  | 0 (166f7ebd) | 0.966 | 56/61 = 0.918 | fails |
  | 1 (e1bafd6f) | 1.000 | 1.000 | passes |
  | 2 (2af3c80d) | 51/59 = 0.864 | 43/61 = 0.705 | fails |
  | 3 (e09b652a) | 1.000 | 0.967 | passes |
  | 4 (a51c5bf4) | 0.966 | 0.967 | passes |

- **Correction to the prose record:** "fails on seeds 0 and 2 (0.705 at 80-100% depth)" in the
  v6 proposal was imprecise; the merge-and-bench lane has since corrected it. The v0.1 release
  notes ("0.705-1.000" across seeds; "fails on seeds 0 and 2") were already correct. 0.705 is seed 2 alone; seed 0's worst bucket is
  0.918. "1.000 on seeds 1, 3 and 4" holds for seed 1 only; seeds 3 and 4 pass at 0.967 and
  0.966.

- **Consequence:** the v6 proposal's needle item ("the gate's detail names the shape: 'A
  recurrent model losing early context'", `HANDOFF/v6-plan-proposal-2026-10-06.md`, Needle
  section) rests on that boilerplate. Long-input training data aimed at early context may not
  address this failure.
- **The cause is [U].** Candidates, none tested:
  - (a) A structural edge case. At depth 1.0, `needle.py:241` gives `position == len(hunks)`, so
    the needle becomes the last hunk, adjacent to the question.
  - (b) Recency or position bias in the readout.
  - (c) Seed-dependent damage to the few retrieval heads in the 6 attention layers (Bick, Xing &
    Gu, arXiv 2504.18574 [fetched]).
  - (d) Abstention rather than wrong picks.
- **Discriminating checks** (eval-only, Mac): split seeds 0 and 2's 60-100% misses into abstain
  vs wrong pick, and compare depth-1.0 cases with 0.8-0.95 cases.
- **Also:** the gate's verdict string should name the failing bucket's position, not a canned
  cause. That is a wording fix to an eval detail, not a threshold change, and it still needs the
  lead's review.

### 2.2 v5 trained the full vocabulary with fused cross-entropy [V]

- **Vocabulary:** v5's row records `recipe.backbone_vocab = 248320`.
- **Loss:** the training loss is `fused_linear_cross_entropy` (`python/qd_train/backbone.py:1248`).
  Full-vocabulary logits are materialised only in evaluation (`backbone.py:1224-1227`).
- **Consequence:** the concern that full fp32 logits cost ~35 GB per step does not apply to
  Lappi's training path.
- **Correction:** the "13,787-token vocabulary" figure in `backbone.py:611` describes an earlier
  remapped setup, not v5.

### 2.3 The trainable size is smaller than the Hugging Face count [V]

- **Text tower:** 1,881,825,088 parameters (`HANDOFF/j7-avg-scoring-2026-10-01.md`, "float64
  accumulators" line). The Hugging Face safetensors count for Qwen3.5-2B-Base, 2,274,069,824
  [fetched], includes the vision encoder Lappi drops.
- **Base config:** `vision_config` is present; layers are 3 linear-attention to 1
  full-attention, 24 layers; `max_position_embeddings` is 262144 [V, cached `config.json`].
- **Scaling to 4B/9B:** use the text-tower count of each model, not the Hugging Face total [I].
  The real answer is `tools/memory_budget.py` against the 4B and 9B configs, then one measured
  probe step. The memory predictor has erred in the unsafe direction before (section 3, item 14).

### 2.4 The trainer is single-GPU, full fine-tune only [V]

- **No multi-GPU:** `tools/memory_budget.py:51` reads "There is no DDP, no FSDP, no `torchrun`".
- **No LoRA:** `rg` (default ignores) for `lora|peft` over `python/`, `tools/`, `crates/` and
  `~/Code/research/ojas` finds no LoRA implementation.
- **No sharding in Rust:** `rg` for `fsdp|DistributedDataParallel|deepspeed|torch.distributed`
  finds no sharding in the Rust crates or ojas. That search used default ignore rules; it was not
  re-run with `-uu`.
- **Consequence:** 4B full FT is a memory question (section 5), and 9B is a trainer project.

### 2.5 The headline accuracy includes a family the preview does not serve [V]

- **Source:** the outreach lane (`.claude/worktrees/outreach-2026-10-06/docs/outreach/evidence/served_families.txt`,
  read this session).
- **Headline vs served:** 0.857 choice top-1 and 91.9% answered at 87.0% includes 2,304
  `code.defect_class` rows, which are not served. On served families only, it is 0.842, and
  90.7% answered at 85.6%.
- **Per-family top-1, the weak tail** (v5 average, row 6af73bef):

  | Family | top-1 |
  |---|---|
  | `pairwise.helpfulness` | 0.473 |
  | `knowledge.multiple_choice` | 0.632 |
  | `openjev.game` | 0.710 (31 rows) |
  | `openjev.nli` | 0.715 |
  | `intent.within_domain` | 0.740 |
  | `commonsense.multiple_choice` | 0.784 |

- **Interpretation [I]:** `knowledge` and `commonsense` are the families a larger base should
  lift. `pairwise` and `nli` are data-weak first.

### 2.6 RM-Bench code result: out of task, through the weakest family [V on the card; probe still running]

- **Source:** GitPulse card `ft-882ed8fcd29cfacd3e1cf14748a0e9de`, read this session.
- **Result:** Lappi v0.1 scores 0.111 on 54 RM-Bench code pairs, against 0.361 for v4 and 0.639
  for Gemma 4 E4B.
- **Why:**
  - The pairs route through `pairwise.helpfulness`, which is 2% of rows and the lowest margin
    over its control.
  - There is a length and style prior: Lappi picks the longer program on 37 of 55 non-TIE calls.
  - The defect-class skill was never asked.
- **Pending:** the merge-and-bench lane is running a defect-class probe on these programs. This
  audit does not predict its result.

### 2.7 No caller sends Lappi a request, and no caller logs queries [V]

- **Integration:** `campaign/v6-caller-families.DRAFT.json`, `callers_today` reads "No
  DevCouncil, DevType or GitPulse code sends Lappi a request" (grep-based, not graph-confirmed).
- **Query logs:** none; `docs/train-plan-2026-09-28.md:265-266`.
- **Consequence:** for a "heavy apps" model, there is no real-traffic data to train on or to
  hold out. This is the largest gap for the human's stated goal.

### 2.8 Serving gaps that block the code families [V]

- **No span head on the Mac backend:** `rg -i span crates/qd-metal/src` matches only
  byte-range code (`model.rs:217-231`, `safetensors.rs:237`). There is no span head.
- **Span calibration cannot be fitted:** GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS, open.
- **Admission refuses the composed defect rows:** GAP-ADMISSION-REFUSES-COMPOSED-DEFECT-CLASS-ROWS,
  measured 10-06.

### 2.9 Span accuracy in diff mode is close to a free pass [V for the recorded run; I for v5]

- **Recorded result:** GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD (open). A pointer to a
  uniformly random `+` line scores 90.8% on the same 10,516 rows where the head scored 92.3%.
- **For v5 [I]:** v5's 0.907-0.915 probably has the same shape. That has not been measured on
  v5's rows.
- **Consequence:** "pick one seed's span head" (the v6 plan) would choose on a weak signal.
- **Decision needed:** what the span head is asked in diff mode is a contract decision for the
  human.

### 2.10 Plan and record disagreements [V]

- **Promotion re-spec not recorded.** The paired_margin / shuffled_label re-spec is not in
  `docs/promotion-decisions.json` or `python/qd_train/ledger.py` (`rg`, gap classifier). The v6
  plan calls it a prerequisite.
- **Release candidate vs promotion.** The v6 plan names the averaged artifact as the release
  candidate, while `docs/promotion-decisions.json:18-26` records `average_may_promote: false`.
  Releasing and promoting are different; the human should say which v6's average is for.
- **Two release-blocking failures have no gap id:** the 16-option calibration failure, and v5's
  needle failure on seeds 0 and 2. The gap classifier searched gap ids for both [V]. That no
  differently-named record covers them is [I].

## 3. Gap inventory

- **Collapse method:** `gaps.jsonl` (working tree, which includes uncommitted appends) has 1,113
  rows and 777 unique ids. Each row is applied over earlier rows with the same id, in file order.
  457 ids are `open` at their latest row [V].
- **Full table:** the classifier's per-gap tables (id, summary, judgment, evidence pointer, blocker
  touched) are in this session's scratchpad (`gaps-classified.md`). They are not committed here,
  because 457 rows of judgment are evidence for the lead to triage, not records.

| Category | Open | Confirmed open | Unclear | Likely stale |
|---|---|---|---|---|
| A. Model quality / recipe | 34 | 3 | 8 | 23 |
| B. Data | 34 | 3 | 16 | 15 |
| C. Gates / promotion | 53 | 5 | 24 | 24 |
| D. Serving / runtime / export | 41 | 13 | 22 | 6 |
| E. Trainer / kernels / infra / cost | 118 | 10 | 57 | 51 |
| F. Tooling / process | 177 | n/a | n/a | n/a |

- **Depth of review (not complete coverage):**
  - All 330 non-tool records were read, with each field cut at 600 characters.
  - The 127 tool-named F records were reviewed by their question line only.
  - 105 of the 280 A-E judgments cite evidence the classifier read itself; the rest are [I] or [U].
  - "Likely stale" means the evidence points that way; no gap's check was re-run.
- **Stale share:** about 43% of A-E (119 of 280) look stale. They cluster in rung-0, v4/F,
  Mac-torch and never-run-device records that v5 overtook.
- **F themes:**

  | Theme | Count |
  |---|---|
  | ListAgents unavailable to sub-lanes | 68 |
  | No DevMap index in agent worktrees | 55 |
  | GitPulse repo trust | 52 |
  | DevMap stale-generation complaints | ~50 |
  | GitPulse collision check blind or truncated | 46 |

  DevMap is healthy today (generation 3433, fresh), so the 10-03 to 10-06 "DB malformed"
  records describe a past state.

**The 15 that matter for the next model** (the classifier's ranking; spot-checked here where
marked):

1. **GAP-PAIRED-MARGIN-POOL-INCLUDES-A-FAMILY-WITH-NO-TRAINING-ROWS-2026-10-05.** As built, no
   v6 seed can promote.
2. **GAP-OOD-CLINC-NOUL-DOES-NOT-TRANSFER-TO-CODE-SHAPED-OOD.** `ood_abstain` fails on every seed
   and on the average.
3. **GAP-ABSTAIN-GATE-IN-DIST-EQUALS-PERMUTATION-DISAGREEMENT-2026-10-02.** The in-distribution
   cap is largely permutation disagreement.
4. **GAP-AVERAGE-PROMOTION-IS-THE-HUMANS-DECISION.** See 2.10.
5. **GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06.**
6. **v5 needle failure on seeds 0 and 2:** no gap id. See 2.1.
7. **GAP-SHUFFLED-LABEL-REQUIRED-ON-EVERY-SEED-2026-10-05.**
8. **GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS.**
9. **GAP-ADMISSION-REFUSES-COMPOSED-DEFECT-CLASS-ROWS** and
   **GAP-PREVIEW-DOES-NOT-SERVE-CODE-DEFECT-CLASS-2026-10-06.**
10. **Span head unserved:** GAP-J7-EXPORT-SPAN-HEAD-UNSERVED,
    GAP-G9A-SPAN-HEAD-SERVING-NEEDS-MODEL-AND-CALIBRATION-WORK and
    GAP-QDM-SPAN-HEAD-NEEDS-POST-NORM-HIDDEN-AND-QUERY-ROW-2026-10-02. qd-metal has no span head
    [V, 2.8].
11. **16-option calibration:** no gap id. Two-fold ECE is 0.079 against 0.05, and 51% of these
    questions are refused at the 0.80 target.
12. **GAP-QD-DATA-DEDUPE-KEEP-RULE-LEXICAL-SPLIT-BLIND-2026-10-03.** `dedupe.py:444` keeps
    `members[0]`, which can drop a val or held-out row in favour of its train twin.
13. **GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD.** See 2.9 [V].
14. **GAP-MEMORY-PREDICTS-084-TO-097X-OF-MEASURED-BACKBONE-FOOTPRINT-ON-GH200** and
    **GAP-MEMORY-CAPACITY-PREDICTOR-ADMITS-MORE-ROWS-THAN-FIT.** The predictor errs in the unsafe
    direction.
15. **GAP-V5-2GPU-BOX-CEILING-NOT-ENFORCED-2026-10-03** and
    **GAP-V5-QUEUE-BOX-WALL-CLOCK-NOT-TRACKED-2026-10-02.** Rule 4's auto-terminate is not
    enforced by the queue.

**Runners-up:**
- GAP-TRANSFER-GATE-IS-NAMED-NOT-SPECIFIED-AND-HAS-NO-DATA
- GAP-DECISION-INDEX-PANEL-CLINC-AND-MMLU-TEST-ITEMS-ARE-LAPPI-TRAINING-DATA-2026-10-03
- GAP-QDM-TOKENIZER-UNTRUSTED-ENCODE-ONE-SIDED-2026-10-02
- GAP-SCORE-VAL-DECODE-RUNS-AT-A-QUARTER-OF-THE-GPU-2026-10-03
- GAP-BOX-GPU-IDLE-REAL-FT-RUN-PREAMBLE-2026-10-03

## 4. Problem → lever map

Each entry gives:
- **Measured:** the failure, with its source.
- **Levers:** the literature-backed options, cheapest first.
- **Cost:** CPU (Mac), plumbing (code in eval/runtime), or GPU (a training run).
- **Decides:** who owns the call. Every GPU run needs pre-registration; a box needs rule 4.

### 4.1 `ood_abstain`, OOD half: 149/180 (average), 152-162 (seeds); the gate needs a Wilson lower bound ≥ 0.9

- **Measured:** release notes. The margin probe shows OOD rows are confidently wrong (margins
  near 1).
- **Levers:**
  1. **A hidden-state OOD gate**, thresholded on in-distribution data only with conformal
     p-values, so its in-distribution false-refusal rate is bounded by construction.
     - Detectors: Relative Mahalanobis or kNN-cosine on hidden states.
     - Evidence: Uppaal et al. ACL'23, arXiv 2305.13282 [fetched]; Liu et al. COLING'24,
       arXiv 2308.10261 [fetched]; Ren et al. ICLR'23, arXiv 2209.15558 [fetched]; Sun et al.
       ICML'22, arXiv 2204.06507 [fetched]; Bates et al. Ann. Stat. 2023, arXiv 2104.08279
       [fetched].
  2. **A fine-tuned vs base likelihood ratio** as a second signal. Zhang et al., arXiv
     2404.08679 [fetched].
  3. **Retrain with a ranking loss** (OOD confidence below ID confidence) over synthetic
     code-shaped OOD, instead of a heavier noul weight. CoNAL, arXiv 2211.15718 [fetched];
     Abbas et al. COLM'25, arXiv 2502.03323 [fetched]. Or LogitNorm, arXiv 2205.09310 [fetched].
- **Why noul dose fails:** Feng et al. ICLR'23, arXiv 2206.09034 [fetched], and Verma &
  Nalisnick ICML'22, arXiv 2202.03673 [fetched]: a reserved abstain class adds no separating
  information, and a K+1 defer softmax is not calibrated. This matches v6x's measured result.
- **Cost:** lever 1 is plumbing, not free.
  - `ood_abstain` is scored by the Python eval path. A detector moves the gate only if it is
    mirrored in eval and in `_decode`'s output, and served by `qd-runtime`/`qd-metal`. That is
    the same class of work as span calibration.
  - The threshold must be fitted on a split that training never read and that is not the
    180-row gate population. Today's table is fitted with `--population all` on val.
  - Lever 3 is GPU.
- **Open [U]:** which layer and token position to read. There is no OOD literature on Gated
  DeltaNet hybrids. X-Mahalanobis (NeurIPS'25 [fetched]) mixes layers.
- **Selection leak to avoid:** choosing the detector, layer or position by its separation on
  the 180 gate rows is selection on the gate population.
  - Design it on a dev OOD set instead: code-shaped OOD built by qd-mutate from train repos, or
    unseen-language rows that are not among the 180.
  - Pre-register the design, then read the gate once.
- **Decides:** a new score in the abstain path is a serving contract change, so it is the
  human's. Thresholds stay read-only.

### 4.2 `ood_abstain`, in-distribution cap: 4.6% average; 5.6-8.2% seeds; cap 5%

- **Measured:** release notes. v6x raised it to 8.7%.
- **Levers:**
  - **Averaging,** which took the cap from 8.2% to 4.6% [V].
  - **Use permutation agreement as a score, not a refusal.**
    - PriDe, ICLR'24, arXiv 2309.03882 [fetched]: estimates a per-option-ID prior and removes it.
    - Average probabilities over k cyclic permutations and calibrate the averaged margin.
    - The basis is GAP-ABSTAIN-GATE-IN-DIST-EQUALS-PERMUTATION-DISAGREEMENT.
  - **Arm B's in-distribution hard-positive data** (the v6 plan).
- **Cost:** plumbing (the score change), then GPU (Arm B).
- **Decides:** changing what "abstain" means on disagreement is a contract change for the human.

### 4.3 16-option calibration: ECE 0.079 against 0.05; 51% refused at the 0.80 target

- **Measured:** release notes.
- **Levers:**
  - Conformal calibration conditioned on option count, pooling sparse cells: clustered conformal
    (Ding et al. NeurIPS'23, arXiv 2306.09335 [fetched]), or group-conditional guarantees
    (Gibbs, Cherian & Candès, arXiv 2305.12616 [fetched]).
  - Per-option-count temperature, or regularised Dirichlet calibration (Kull et al. NeurIPS'19,
    arXiv 1910.12656 [fetched]).
  - CROQ: prune to the conformal set, then re-ask with fewer options (ICML'25, arXiv 2501.00555
    [fetched]).
- **Caution:** calibration can improve ECE without improving which rows are refused (Zhu et al.
  ECCV'22, arXiv 2303.02970 [fetched]). Judge on refusals at the target precision.
- **Cost:** CPU (fitting) and plumbing (the table key and a second decode for CROQ).
- **Decides:** the lead, then the human for the serving table.

### 4.4 Span head after averaging: 0.773 vs 0.907-0.915 per seed

- **Measured:** release notes. Norm and cosine ≈ 1/√5.
- **Levers:**
  1. **Refit the span head by linear probe** on the frozen averaged tower's cached features,
     using train rows only. Compare it against one seed's head and against the mean of the five
     heads' logits.
  2. **Future seeds share one span head, initialised by linear probe** (LP-FT). WARM, arXiv
     2401.12187 [fetched, PDF pp. 5-6]: linear mode connectivity fails for weights trained from
     scratch even when the random initialization is shared, so it initialises the classifier by
     linear probe. Model soups, arXiv 2203.05482
     [fetched, PDF pp. 3-5]: LP initialisation is their default. LP-FT, arXiv 2202.10054
     [fetched].
- **Caveat:** 2.9 says span top-1 in diff mode is close to a free pass. Measure the refit against
  the random-`+`-line null, not the uniform null.
- **Cost:** CPU for the refit. Each future seed needs only a probe fit.

### 4.5 Span calibration and serving: no span entry can be fitted; no Mac span head

- **Levers:**
  - **Calibration:** record pointer distributions in verdicts, then calibrate a line set with
    conformal risk control on a line-level miss rate (Angelopoulos et al., arXiv 2208.02814
    [fetched]).
  - **Serving:** a span head in qd-metal (2.8).
- **Cost:** plumbing in eval, export and runtime.
- **Decides:** the lead, after the span contract decision in 2.9.

### 4.6 Needle: 60-100% depth misses on seeds 0 and 2

- **Levers:**
  - **First:** the discriminating checks in 2.1, which are eval-only.
  - **If retrieval heads are implicated:** ablate the 6 attention layers' heads in a good seed
    and a bad seed (2504.18574 [fetched]).
  - **If data is implicated:** add synthetic needles with depth sampled uniformly. IN2/FILM,
    arXiv 2404.16811 [fetched]; Xiong et al., arXiv 2406.19292 [fetched].
- **Not indicated:** the length-generalisation remedies (Buitrago & Gu, arXiv 2507.02782
  [fetched]; Stuffed Mamba, arXiv 2410.07145 [fetched]). They apply when fine-tuning sequences
  are much shorter than the eval. v5's build allowed `--max-seq-len 10240`
  (`HANDOFF/v5-plan-2026-10-02.md:315`), while the eval cases are ~7.6-8.8K tokens. The per-row
  length distribution was not read [U].
- **Architecture:** 3:1 is the recall-friendly end of the 3:1-6:1 range (Wang et al., arXiv
  2507.06457 [fetched]). It is fixed by the base model.

### 4.7 Transfer to natural defects: unmeasured; scope-limited by record

- **Levers:**
  - **A natural-bug held-out set** of about 100-300 single-statement fix commits from the
    human's own held-out repos. Use the existing holdout rule in
    `campaign/v6-caller-families.DRAFT.json`, `holdout_rule.split`, and only commits dated after
    the base model's cutoff (cutoff [U]). Never train on it.
  - **Two-phase training,** synthetic then real (He et al., arXiv 2204.10049 [fetched]: over 90%
    accuracy on synthetic bugs vs under 10% precision on real repositories).
  - **More realistic mutants:** contextual or LLM-proposed (DeepMutants, arXiv 2107.06657
    [snippet]; arXiv 2406.09843 [snippet]).
  - **More distinct repos:** SWE-smith, arXiv 2504.21798v2 [fetched]. AST-procedural bugs
    resolved 8.6% vs 9.2% for PR-mirrored bugs, and performance grew roughly with the log of the
    number of distinct repositories.
- **Cost:** CPU (mining) and human labelling time.
- **Decides:** the human specifies the transfer gate (GAP-TRANSFER-GATE).

### 4.8 Decontamination: MinHash/LSH covers exact and near-exact copies only

- **Levers:**
  - An embedding top-k plus LLM-judge pass for rephrased duplicates (Yang et al. LMSYS, arXiv
    2311.04850 [snippet]; github.com/lm-sys/llm-decontaminator [snippet]).
  - Repo- or function-level splits, so mutants of one function cannot cross the split
    (Allamanis, arXiv 1812.06469 [snippet]: duplication inflates metrics by up to 100%).
  - Fix the dedupe keep rule (gap 12).
- **Evidence for tiers:** Nourbakhsh et al. GEM 2026 [fetched]: no detector is reliable across
  contamination tiers.
- **Cost:** CPU.

### 4.9 Evaluation statistics

- **Levers:**
  - Cluster standard errors by source function or file, because many mutants share a source.
    Clustered SEs can exceed naive ones by more than 3x (Miller, arXiv 2411.00640 [fetched];
    anthropic.com/research/statistical-approach-to-model-evals [fetched]).
  - Paired differences and a power analysis before thresholds are set.
  - Vary data order as well as init across seeds (Dodge et al., arXiv 2002.06305 [snippet]).
  - "≥ 3 seeds" is this repo's convention. The researchers found no source that sets a seed count.
- **Cost:** CPU.
- **Decides:** thresholds and gates are the human's (rule 2). This audit proposes none.

### 4.10 Base-knowledge ceiling: knowledge MC 0.632, commonsense 0.784

- **Lever:** a larger base (section 5).
- **Weaker levers:**
  - Data amount per ability matters more than the mixture ratio (Dong et al. ACL'24 [snippet]).
  - Continued pretraining has no evidence of helping decision accuracy at this scale. AdaptLLM
    reports raw continued pretraining hurting prompted QA (ICLR'24 [snippet]).

### 4.11 Caller families (the human's apps); none trained, none integrated (2.7)

| App | Shape the literature supports | Evidence |
|---|---|---|
| ScholarLM reranker | Pointwise yes/no token probability with a task-instruction field; hard negatives mined from several retrievers; scientific evaluation probes | Qwen3-Reranker, arXiv 2506.05176 [fetched] (MTEB-R 0.6B 65.80, 4B 69.76, 8B 69.02); jina-reranker-v3, arXiv 2509.25085v4 [fetched]; SciRerankBench, arXiv 2508.08742 [fetched] |
| jevgrep | Embedding retrieval, then a Lappi choice over candidates by identifier, then a line span inside the chosen function. No published line-level baseline was found. | SweRank, arXiv 2505.07849 [fetched]; CoRNStack, arXiv 2412.01007 [fetched]; CodeScout, arXiv 2603.17829 [fetched card] |
| Tool selection (DevCouncil, GitPulse, MANVI) | Explicit call / ask / unable / answer options trained with preference optimisation; irrelevance rows made by removing the gold tool; names masked; retrieve then choose for large tool sets | When2Call, arXiv 2504.18851 [fetched] (tool hallucination at 4B 4.3% → 1.9%); Hammer, arXiv 2410.04587 [fetched] (irrelevance does not rise with size); ToolRet, arXiv 2503.01763 [snippet] |
| Jarvis forms | Choose among enumerated fields rather than localising free-form | FormGym, arXiv 2506.14079 [snippet] |
| Jarvis computer use | A pointer head over elements or patches with multi-positive KL supervision; a separate risk slot with a conformal execute/abstain threshold for irreversible actions | GUI-Actor, arXiv 2506.03143 [fetched]; CORA, arXiv 2604.09155 [fetched]. Screenshot input needs a vision path. The base has `vision_config` [V, 2.3]; that no Lappi crate implements vision is [I], from the text-only tower export and trainer. |
| Email sorter | Fine-tune, don't prompt, at 1-3B. Untrusted content in a typed field. Train with label-flipping injections at random positions; gate on label-flip rate under injection. | arXiv 2505.16078 [fetched]; Meta SecAlign, arXiv 2507.02735v3 [fetched]; LLMail-Inject, arXiv 2506.09956 [fetched] (its metric is tool calls, not label flips) |
| Shared tower | One tower conditioned by a task or definition field; gate each family against a single-task baseline to detect interference | Granite Guardian, arXiv 2412.07724 [fetched]; Ortho-LoRA, arXiv 2601.09684 [fetched]; LoRI, arXiv 2504.07448 [fetched] |

**Rule 7 applies to every tool and computer-use slot:** Lappi admits; it never authorises an
irreversible action or discharges a requirement.

## 5. Sizing update (supersedes this session's earlier chat answer where they differ)

- **Memory per parameter** [I; standard accounting per the efficient-training research]:

  | Setup | Bytes/param |
  |---|---|
  | bf16 weights + fp32 masters + fp32 grads + fp32 AdamW state | ~16-18 |
  | 8-bit optimizer state | ~10 |
  | Optimizer CPU-offloaded | ~4 on GPU |
  | LoRA on a frozen base | ~2 |

- **How 2B fits:** at 1.88B trainable parameters:
  - **Measured on earlier runs:** F peaked at 61.6 GiB (`HANDOFF/v5-plan-2026-10-02.md:315`),
    and the 10-01 step test at 60.51 GiB with skip:6 (`HANDOFF/train-step-perf-2026-10-01.md:36`).
  - **v5's own row** carries only an estimate: `metrics.device_budget`, 63.66 GiB against
    69.80 GiB available. That is not a measurement.
- **4B full FT** will not fit one H100 80GB at v5's batch with today's recipe [I]. Options: an
  H200, 2× H100 (which needs FSDP; the trainer has none), 8-bit optimizer state, or LoRA.
- **9B full FT** fits no single GPU without offload [I].
- **LoRA, per the literature:**
  - It matches full fine-tuning when applied to all layers including the MLP, at rank ≥ 8-16,
    with a learning rate ~10x the full-FT rate. Thinking Machines, "LoRA Without Regret",
    2025-09-29 [fetched]; Baseten on Qwen3-4B, Oct 2025 [fetched].
  - It "learns less and forgets less" in code continued pretraining (Biderman et al., arXiv
    2405.09673 [fetched]).
  - No study measures LoRA vs full-FT seed variance on classification heads, so Lappi would have
    to measure it.
- **Avoid** QLoRA on Qwen3.5 (Unsloth's Qwen3.5 page [fetched]) and Muon for full fine-tuning of
  an Adam-pretrained model (arXiv 2605.10468 [fetched]).
- **Kernels:**
  - fla 0.5.2 (2026-07-27) for Gated DeltaNet training [fetched].
  - Liger 0.8.4 (2026-09-30) `apply_liger_kernel_to_qwen3_5` for RMSNorm, SwiGLU and
    cross-entropy; it has no GDN kernel [fetched].
  - Lappi already has its own fused cross-entropy (2.2).
- **Prices fetched 2026-10-06:**

  | Provider | GPU | $/h |
  |---|---|---|
  | Lambda | GH200 | 2.29 |
  | Lambda | H100 SXM | 4.29 (1×), 4.19 (2×) |
  | Lambda | B200 | 6.99 |
  | RunPod Secure | H100 SXM | 3.49 |
  | RunPod Community | H100 SXM | 2.69 |
  | RunPod Secure | H200 | 4.59 |
  | Modal | H100 | ~3.95 |
  | Modal | H200 | ~4.54 |

  GH200 is aarch64, so it needs ARM builds. The repo has run a GH200 before
  (`ledger/gh200-*`).
- **Cost per promotable candidate** [I]: ≥ 3 seeds plus a control per seed is ≥ 6 runs. The
  base is $27 per 2B seed on an H100 [V, v6 plan].

  | Option | Est. per candidate |
  |---|---|
  | 2B on GH200 | ~$90, if throughput matches an H100 [U] |
  | 4B LoRA | ~$120-225 |
  | 4B full FT with 8-bit state | ~$185-335 |
  | 9B LoRA | ~$250-460 |
  | 9B full FT | ~$700+, plus trainer work |

- **Recommendation, unchanged in direction:**
  - Fix the non-capacity failures on 2B first.
  - Run one `quick` 4B probe (which cannot promote) on the base-bound families.
  - Go to 9B only if 4B measurably beats 2B.

## 6. Recommended order (no GPU first)

1. **Span-head refit on the averaged tower** (4.4). CPU on the Mac. Score it against the
   random-`+`-line null.
2. **Needle per-case diagnosis** (2.1). Eval only.
3. **Feasibility of the OOD gate** (4.1). Extract hidden states for train and val rows and a
   dev OOD set (not the 180 gate rows) on the Mac. Measure separation there, and scope the eval
   and runtime plumbing. Pre-register before the gate population is read. Report only; no gate
   change.
4. **The human decisions in section 7.**
5. **Data and evaluation infrastructure, on CPU:**
   - the natural-bug held-out set;
   - repo-level splits and the embedding decontamination pass;
   - clustered and paired statistics in the ledger reader;
   - per-option-count calibration fits.
6. **Caller-family data and integration** (4.11). This needs the first real request path and
   query logging, without which there is nothing to hold out.
7. **GPU, pre-registered,** after the above:
   - a 2B v6 with a shared LP-initialised span head and the data arms;
   - in parallel, one `quick` 4B probe (LoRA on all layers vs full FT with 8-bit state) on a
     probed box.
   - Rule 4 applies to any box.

## 7. Decisions only the human can make

1. Record the two promotion re-spec entries (`paired_margin_population`, `shuffled_label_seeds`),
   or decline them.
2. Is v6's average meant to **promote**, or only to be released? (`average_may_promote: false` is
   on record.)
3. What does the span head answer in diff mode, given that a random `+` line scores 90.8%?
4. Specify the transfer gate: population, size, and whether it binds.
5. Whether a hidden-state OOD score, or permutation-agreement-as-score, may enter the abstain
   path. Both are serving contract changes.
6. Which caller families v6 trains, and whether callers may log real queries (privacy: emails and
   form profiles).
7. Base size beyond 2B, after the `quick` 4B probe; any box, with its cap, auto-terminate and
   cost (rule 4).

## 8. Proposed gap records (text for the lead; not appended by this session)

**Status, 2026-10-06:** the merge-and-bench lane re-verified and appended the first three
(needle verdict string, v5 end-of-context needle misses, 16-option calibration). The lane left
the fourth (feature-space OOD score) unappended as an unverified design proposal; it is for the
human's v6 decision.

- **GAP-NEEDLE-VERDICT-STRING-NAMES-A-CAUSE-THE-BUCKET-CONTRADICTS-2026-10-06.**
  `needle.py:477-479` appends "losing early context" whichever bucket fails. v5's misses are all
  at 60-100% depth, the end of the context (2.1). The v6 needle item cites the string.
- **GAP-V5-NEEDLE-END-OF-CONTEXT-MISSES-SEEDS-0-2-HAVE-NO-ID-2026-10-06.** The 0.705 and 0.918
  worst buckets have no gap id. The cause is [U]; the discriminating checks are in 2.1.
- **GAP-16-OPTION-CALIBRATION-FAILS-AND-HAS-NO-ID-2026-10-06.** Two-fold ECE is 0.079 against
  0.05, and 51% of these questions are refused at 0.80. No record owns it.
- **GAP-OOD-ABSTAIN-HAS-NO-FEATURE-SPACE-SCORE-2026-10-06.** Every lever tried so far (noul
  weight, fitted margin) is an output-space score. The literature in 4.1 says those cannot
  separate confidently wrong OOD inputs. The feature-space gate is unmeasured, and its eval and
  runtime plumbing is unscoped.

## 9. Not verified or not run

- Every external figure is from an abstract or HTML page read through a summarising fetcher.
  No PDF bodies were read, except WARM pp. 5-6 and Model soups pp. 3-5.
- Snippet-only items are marked [snippet].
- Two items are 2026 unreviewed preprints: HG-CRC (arXiv 2607.24562) and bilevel calibration
  (arXiv 2608.07419).
- **Repo claims not checked by this session:**
  - the v5 fine-tuning length distribution;
  - the base model's data cutoff;
  - the 4B and 9B text-tower parameter counts;
  - GH200 throughput and ARM wheel availability;
  - whether v5's span rows share the diff-marker shape.
- No 4B/9B memory figure here was measured. Every one is [I] until `tools/memory_budget.py` and
  a probe step run.
- The RM-Bench defect-class probe was still running when this was written.
- Qwen3.8 small Base sizes are [U]. The cache holds only `Qwen3.8-27B` variants. A generation
  change would mean a kernel-parity lane in tessl and ojas, with fixtures labelled `published`
  (rule 9).

## 10. Sources

The full per-topic lists, with [fetched]/[snippet] labels, are in this session's scratchpad notes
(`lit-abstention-calibration.md`, `lit-longctx-averaging.md`, `lit-efficient-training.md`,
`lit-app-tasks.md`, `lit-data-eval.md`). Every external number in this file carries its arXiv id
or URL inline above. Non-arXiv sources cited:

- https://thinkingmachines.ai/blog/lora/
- https://labs.baseten.co/articles/practical-lora-research
- https://unsloth.ai/docs/models/qwen3.5/fine-tune
- https://github.com/fla-org/flash-linear-attention
- https://pypi.org/project/liger-kernel/
- https://lambda.ai/pricing
- https://www.runpod.io/pricing
- https://modal.com/pricing
- https://huggingface.co/Qwen/Qwen3.5-2B-Base
- https://huggingface.co/api/models/Qwen/Qwen3.5-4B-Base
- https://huggingface.co/api/models/Qwen/Qwen3.5-9B-Base
- https://www.anthropic.com/research/statistical-approach-to-model-evals
- https://aclanthology.org/2026.gem-main.50/
- https://github.com/lm-sys/llm-decontaminator
- https://proceedings.neurips.cc/paper_files/paper/2025/hash/d033407a1a4b7a75cd5f9e4575ad9fb5-Abstract-Conference.html
