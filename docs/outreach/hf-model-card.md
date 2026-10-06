---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B-Base
base_model_relation: finetune
language:
  - en
tags:
  - typed-decisions
  - classification
  - abstention
  - calibration
  - preview
datasets:
  - ZefanCai/Open-Jev-v1.1
  - tasksource/procedural-typed-decisions
  - LocalLLaMA/typed-decisions
  - n4ze3m/typed-decisions-synth
  - nvidia/HelpSteer2
  - google/boolq
  - allenai/ai2_arc
  - tals/vitaminc
  - bigcode/commitpackft
  - cais/mmlu
  - tau/commonsense_qa
  - clinc/clinc_oos
  - rajpurkar/squad_v2
---

<!--
DRAFT for the Hugging Face model repo's README.md. Not uploaded. Uploading, creating the repo and
choosing its name are the author's actions. Links to github.com/bharathvbcr/Lappi-decision work
only once that repository is public; until then remove them or keep the repository private and
hold this card. The JevArena results page link is private until the author shares it.
-->

# Lappi v0.1 preview

Lappi is a typed decision model fine-tuned from
[Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base). You give it a context, a
question and a set of named options. It returns one option, a score, and an explicit abstention
("noul") when it should not answer. It does not generate text and does not reason before it answers:
each answer is one token read from a 17-row slice of the output layer.

**This is a preview, not a promoted model.** The author released it after it missed its own
pre-registered bar on two of six lines (span accuracy and out-of-distribution abstention). Read
[What it does not do](#what-it-does-not-do) before you use it.

## What it serves

- **Choice questions** with 2 to 16 named options, plus the reserved abstain row.
- **21 trained task families** are admitted and calibrated:
  - **15 were load-tested end to end** through the product runtime: `arc.science`, `boolq.reading`,
    `decider.commands`, `decider.routing`, `openjev.evidence`, `openjev.game`, `openjev.nli`,
    `openjev.policy`, `openjev.routing`, `openjev.rubric`, `pairwise.helpfulness`,
    `procedural.decisions`, `synth.general`, `typed.workflow`, `vitaminc.nli`;
  - **6 are admitted and calibrated but were not exercised** through the runtime in the load test:
    `intent.classification`, `intent.domain`, `intent.in_scope`, `intent.within_domain`,
    `knowledge.multiple_choice`, `commonsense.multiple_choice`.
- **The calibration table has 15 entries**, one per option count from 2 to 16.
- **Every other task is refused** (`task_not_trained`): the runtime admits only the 23 families the
  weights were trained on, and two of those (below) are not usable.

## What it does not do

- **`code.defect_class` is not served.** This is the code-change family the project was built
  around. Its trained request asks for a choice slot and a span slot together:
  - the span slot is refused, because the release has no span calibration;
  - asking for the choice slot alone is a prompt the model never saw (on `a + b` changed to `a - b`
    it abstained);
  - multi-file `diff --git` / `---` / `+++` contexts are refused at admission.
- **Span answers are not served.** Neither is the span-only `qa.answer_span` family. The span head
  in this release is the mean of five seeds' heads, which point in nearly unrelated directions, and
  it measures 0.773 span top-1 against 0.907-0.915 for single seeds. It is not used. The span
  figure pools SQuAD answer spans (5,139 validation rows; 0.730 for this release) and code-change
  evidence lines (2,099; 0.878). On the code rows it is weak evidence that the head locates
  changes: the diff's `+` markers show the changed lines, and on an earlier run a pointer to a
  random added line scored 90.8% where the head scored 92.3%. v5's rows were not measured against
  that baseline.
- **Score (ordinal) slots** are refused. No family trained one.
- **Out-of-distribution abstention fails its bar.** On 180 inputs it was never trained for, it
  abstained on 149 (bar: at least 90%, so 162). By kind: unseen programming languages 40/60,
  scrambled text 49/60, prose 60/60. On the other inputs it answers, and often confidently.
- **It is not a general judge or chat model.** On JevArena's pairwise-judge benchmark it is near
  chance outside one domain (below).

## Results on validation

All on the v5 validation split, score row `6af73bef` (2x H100). The weights are a plain mean of the
five v5 seeds.

| Measure | v0.1 preview | v5's five seeds | Bar |
|---|---|---|---|
| Choice top-1, all 22 choice families | 0.857 (14,779 / 17,254) | 0.828-0.847 | none |
| Choice top-1, the 21 served families | 0.842 (12,580 / 14,950) [derived] | | none |
| Permutation consistency | 0.957 (16,514 / 17,254) | 0.929-0.949 | >= 0.95, passes |
| Abstention on in-distribution rows | 4.6% (770 / 16,905) | 5.6-8.2% | <= 5%, passes |
| Abstention on out-of-distribution rows | 149 / 180 | 152-162 | >= 0.9, **fails** |
| Needle recall, worst long-context bucket | 1.000 (300 / 300) | 0.705-1.000 | >= 0.95, passes |
| Span top-1 (span is not served) | 0.773 | 0.907-0.915 | none |

**[derived]:** the release notes report the all-families figure, which includes 2,304
`code.defect_class` rows (0.954) that this release does not serve. The served-families figure is
recomputed from the same row's per-question verdicts (`docs/outreach/evidence/` in the source
repository); the script reproduces the published all-families numbers first.

**Answering at an 80% precision target**, with the shipped calibration:

| Estimate | Answers | Precision on answered |
|---|---|---|
| Two-fold, all 22 choice families (each row scored with the other fold's parameters) | 91.9% | 87.0% |
| Two-fold, the 21 served families [derived] | 90.7% | 85.6% |
| In-sample (the shipped table; optimistic) | 93.8% | 87.1% |

- Serving also abstains when the two passes (below) disagree. That is not modelled in this table,
  so real refusal rates run somewhat higher.
- 16-option questions are refused about half the time (51% two-fold).
- **Calibration error** (two-fold ECE, 15 bins): 0.006-0.038 on 8 of the 9 option counts with at
  least 100 rows; **0.079 on 16-option questions, which fails** the 0.05 threshold. Six option
  counts have fewer than 100 rows and are not measured.

**Per family** (choice top-1, validation):

| Family | Rows | Top-1 | Family | Rows | Top-1 |
|---|---|---|---|---|---|
| arc.science | 145 | 0.903 | openjev.evidence | 588 | 0.952 |
| boolq.reading | 288 | 0.882 | openjev.game | 31 | 0.710 |
| commonsense.multiple_choice | 1,197 | 0.784 | openjev.nli | 701 | 0.715 |
| decider.commands | 188 | 0.936 | openjev.policy | 701 | 0.862 |
| decider.routing | 411 | 0.973 | openjev.routing | 292 | 0.938 |
| intent.classification | 1,572 | 0.945 | openjev.rubric | 219 | 0.982 |
| intent.domain | 1,572 | 0.868 | pairwise.helpfulness | 372 | 0.473 |
| intent.in_scope | 1,572 | 0.964 | procedural.decisions | 700 | 0.956 |
| intent.within_domain | 1,571 | 0.740 | synth.general | 512 | 0.914 |
| knowledge.multiple_choice | 1,485 | 0.632 | typed.workflow | 133 | 0.902 |
| | | | vitaminc.nli | 700 | 0.863 |

`knowledge.multiple_choice` is MMLU-derived and `commonsense.multiple_choice` is
CommonsenseQA-derived; both are bounded by what a 2B base model knows. `pairwise.helpfulness`
(which of two responses is more helpful) is near chance.

**These are internal validation numbers, not benchmark results.** MMLU's test and dev splits are
training data for this model (its validation split is the family's val), and CLINC150's train,
validation and test splits were all read and re-split by intent, so the published test utterances
of the training intents are training data. CommonsenseQA's validation split is used as a curated
internal val, with overlapping training rows removed. Do not report this model's MMLU or CLINC150
scores as benchmark results.

**Served through the runtime:** 90 validation questions from the 15 load-tested families, 6 each,
were replayed through the product path on Apple silicon. Served correctness equalled the training
cluster's scoring on 89 of 90. It answered 85 (70 correct) and abstained on 5.

## JevArena (JevJudge-Bench)

[JevArena](https://github.com/chenmingtang830/jevarena) asks a judge which of two answers is better
(A / B / tie) over 849 pairs from JudgeBench, RM-Bench and RewardBench 2, each asked in both orders.
**It is not Lappi's training task**, and Lappi's letter readout was used uncalibrated. Local models
only.

| Judge | Size | Domain macro | All pairs (micro) | Order consistency |
|---|---|---|---|---|
| Lappi v0.1 preview | 2B | 0.534 | 0.610 | 0.716 |
| Qwen3.5-2B (post-trained), as a chat judge | 2B | 0.522 | 0.646 | 0.523 |
| Qwen3.5-2B-Base, as a chat judge | 2B | 0.468 | 0.569 | 0.531 |
| Gemma 4 E4B, no reasoning | 8.0B | 0.632 | 0.812 | 0.801 |
| decider-2b v11 (same base model), on CPU | 2B | 0.612 | 0.719 | 0.749 |

- **Against Qwen3.5-2B it ties:** the paired difference is +0.035 in Qwen's favour, 95% interval
  [-0.039, +0.094].
- **Gemma 4 E4B is clearly better:** +0.200 [+0.133, +0.257].
- **decider-2b is better:** +0.110 [+0.071, +0.157], all 849 pairs answered in both orders.
  Pairwise response preference is one of decider-2b's trained tasks, and the harness asks it in its
  own request format with its own calibration. Neither holds for Lappi. Its README holds out the
  first RewardBench; whether its training data overlaps these three benchmarks was not checked
  here. It ran on the CPU (float32, 6 threads) because its Mac GPU path returned NaN, so its time
  per call (median 5.7 s) is not comparable with the others. Only differences against Lappi were
  computed, so decider-2b and Gemma are not ranked against each other.
- **Outside RewardBench 2's Ties domain, Lappi is near chance.**
- **On RM-Bench's code pairs it scores 0.111** (54 pairs, 6 prompt groups, exploratory). Of 108
  calls it answered "tie" on 53, picked the broken program on 43 and the correct one on 12. These
  pairs reach Lappi as a "which answer is better" question, through `pairwise.helpfulness` (its
  weakest family, 2% of training), not through the code-defect question it was trained on.
- Lappi is more order-consistent than Qwen3.5-2B and had no failed calls.
- **The decision families' training rows were checked against JevArena's three source
  benchmarks** (8-gram containment); one overlapping `pairwise.helpfulness` row was removed. The
  code family was not checked against RM-Bench's code.

**The code-defect question on RM-Bench's programs** (a diagnostic probe, not a gate). Each program
was rendered as a whole-file diff, the shape the training corpus uses for added files, and asked
the trained `code.defect_class` request; a pair counts as separated when the correct program gets
the higher probability of `clean`. Scored with these weights through the project's Metal scorer,
not the serving runtime, which does not serve this family.

| Check | Pairs | Separated [95% interval] |
|---|---|---|
| The probe reads the model correctly: top-1 on all 2,304 `code.defect_class` validation rows | 2,304 rows | 0.956 (ledger: 0.950-0.958) |
| A commit's real file vs the same file with a planted mutation | 85 | **0.906** [0.835, 0.965] |
| ... planted `logic` / `stub` / `cosmetic` mutations | 25 / 36 / 24 | 0.960 / 1.000 / 0.708 [0.500, 0.875] |
| RM-Bench: correct vs broken program (Go, Python, Rust) | 125 | **0.536** [0.448, 0.624], no separation |

- It separates the kinds of mutation it was trained on (not cosmetic ones), and it does not
  separate RM-Bench's correct and broken programs: on both, it calls most of them `clean`.
- RM-Bench programs in C++, Java and JavaScript (206) are refused: those languages are not in the
  training pool.
- The readout was written down before the probe ran. Its fidelity check was re-specified once,
  before the full-validation run, because the first version compared a stratified sample with a
  population figure; under the first version the probe was uninterpreted.

**Gemma 4 E4B with reasoning on**, on a 191-pair subset of the same data (no tie answers in it;
8,192-token output cap), against the other judges scored on the same 191 pairs:

| Judge | Domain macro | All pairs (micro) [95%] | Order consistency | Median time per call |
|---|---|---|---|---|
| Lappi v0.1 preview | 0.529 | 0.571 [0.466, 0.671] | 0.639 | 0.24 s |
| Qwen3.5-2B, as a chat judge | 0.547 | 0.616 [0.512, 0.709] | 0.484 | 0.37 s |
| Gemma 4 E4B, no reasoning | 0.635 | 0.721 [0.599, 0.815] | 0.729 | 0.59 s |
| Gemma 4 E4B, reasoning on | 0.708 | 0.816 [0.728, 0.882] | 0.853 | 14.0 s |
| decider-2b v11, on CPU | 0.625 | 0.707 [0.590, 0.796] | 0.764 | 6.9 s (CPU) |

- Gemma with reasoning minus Lappi, paired: **+0.245** [+0.122, +0.347]; decider-2b minus Lappi:
  +0.136 [+0.048, +0.226].
- Times are the harness's wall clock per call over different transports, not a kernel
  measurement. decider-2b's is a CPU time in float32.

Full results: [JevArena results page](https://claude.ai/artifact/Hdq9CDPFQrqN6DJLnkKNEd)
<!-- private until the author shares it; remove the link otherwise -->

## Speed

On an Apple M5 Pro, through the project's Metal backend (tessl kernels), a full decision (prefill
plus the two answer passes) takes about 72 ms with a 124-token prompt, 127 ms at 408 tokens,
0.41-0.44 s at about 2,000 and 2.3 s at about 8,200. PyTorch on MPS is 1.6 to 3.2 times slower at
the same lengths. Lappi costs the same as its base model: only the weights differ. In the load test
the median request took 0.12 s after a 3.37 s first load.

## How to call it

### The supported path

The runtime is Rust, in the [source repository](https://github.com/bharathvbcr/Lappi-decision):
`qd-metal-serve` loads this release directory on Apple silicon, and `qd oneshot --socket` sends it
one JSON request. The server checks the weight, tokenizer and calibration hashes when it loads the
release, and refuses requests it cannot answer as asked. Other runtimes (transformers, MLX, CUDA)
are **not tested** for this release.

A request from the load test (its `example_id`, which the runtime accepts and ignores, is dropped
here):

```json
{
  "schema_version": 1,
  "task": "decider.routing",
  "context_b64": "V2hhdCBpcyB0aGUgbWFpbiByZWFzb24gZm9yIHlvdXIgbWVzc2FnZT8KCmhvdyBkbyBpIGV4cG9ydCBteSByZXBvcnQgYXMgcGRmPw==",
  "context_len": 76,
  "question": "Which option fits the message, as the question above asks?",
  "slots": [
    {
      "name": "answer",
      "type": "choice",
      "options": ["account_login", "billing_issue", "feature_request", "inquiry", "misc", "technical_bug"]
    }
  ],
  "route": "generic",
  "metadata": {}
}
```

The context decodes to `What is the main reason for your message?\n\nhow do i export my report as pdf?`.
The reply the server sent (line-wrapped here):

```json
{"status":"ok","schema_version":1,"backend":"qd-metal/qwen3.5-2b-base/tessl","degraded":false,
 "slots":{"answer":{"value":"inquiry","conformal_set":["inquiry"],"score":0.985749987903814,"noul":false,"degraded":false}}}
```

### Rules a request must follow

- **`task` is the family id**, one of the 21 served families above.
- **`question` is that family's trained question, verbatim.** The question is rendered into the
  prompt, so different wording is a prompt the model never saw:

  | Family | Question |
  |---|---|
  | `arc.science` | Answer the science question above. |
  | `boolq.reading` | Answer the question above about the passage. |
  | `decider.commands` | Answer the question above about the shell command. |
  | `decider.routing` | Which option fits the message, as the question above asks? |
  | `openjev.evidence` | Which option do the cited facts support, as the question above asks? |
  | `openjev.game` | Which option fits the game state, as the question above asks? |
  | `openjev.nli`, `vitaminc.nli` | Which option does the evidence support, as the question above asks? |
  | `openjev.policy` | Which option does the policy select for this record, as the question above asks? |
  | `openjev.routing` | Where should this item be routed, as the question above asks? |
  | `openjev.rubric` | Which option does the rubric assign, as the question above asks? |
  | `pairwise.helpfulness` | Compare the two responses to the user's prompt. |
  | `procedural.decisions`, `synth.general` | Answer the question above about the record. |
  | `typed.workflow` | Answer the question above about this workflow record. |
  | `intent.classification` | Which intent does the utterance express? |
  | `intent.domain` | Which domain does the utterance belong to? |
  | `intent.in_scope` | Is the utterance in scope for the assistant at all? |
  | `intent.within_domain` | Which of these intents does the utterance express? |
  | `knowledge.multiple_choice` | Which option answers the question? |
  | `commonsense.multiple_choice` | Which option is the most sensible answer to the question? |

- **For the 15 load-tested families, the context is the row's own question line, a blank line,
  then the material** (the message, record, passage or command), as in the example above. The six
  CLINC/MMLU/CommonsenseQA-derived families were built by a different request builder, and their
  context shape is not documented here.
- **`context_b64` is the UTF-8 context in standard, padded base64**, and `context_len` is its byte
  count. They must agree.
- **One choice slot, at most 16 options.** The abstain row is added by the runtime; do not add a
  "none of these" option to stand in for it.
- **`noul: true` means the model abstained.** A refusal is a different thing: an error naming the
  check that failed (an untrained task, a hash mismatch, too many options).

### The prompt (format 2)

The runtime renders the request above into this text and reads the next token:

```text
<|qd_begin|>
<|qd_prompt_format|>2
<|qd_schema_version|>1
<|qd_task|>decider.routing
<|qd_route|>generic
<|qd_context_begin|>
What is the main reason for your message?

how do i export my report as pdf?
<|qd_context_end|>
<|qd_question|>Which option fits the message, as the question above asks?
<|qd_slot|>answer
<|qd_type|>choice
<|qd_options_begin|>
A. account_login
B. billing_issue
C. feature_request
D. inquiry
E. misc
F. technical_bug
Z. noul
<|qd_options_end|>
<|qd_answer|>
```

- The prompt ends right after `<|qd_answer|>`, with no newline. The markers are plain text for the
  base tokenizer, not added special tokens.
- **The answer is read from the logits of the letter tokens A-P and Z** (the abstain row) at the
  last position. No text is generated.
- **The question is asked twice,** the second time with the options in a permuted order. If the
  two passes disagree, the answer is an abstention.
- **The calibration table** (`calibration.json`) sets a temperature, a conformal quantile and an
  abstention margin per slot width. A width is the option count plus the abstain row, so
  `choice:7` is a six-option question.
- Escaping of marker-like text inside the context is part of the renderer. If you reproduce this
  outside the runtime, use the runtime's renderer, not a hand-written template.

## Training

- **Base:** Qwen/Qwen3.5-2B-Base.
- **Recipe:** supervised fine-tuning only, with no continued pretraining and no RL. AdamW with fp32
  master weights, learning rate 1e-5, the lower 8 layers at 0.1x, one epoch of 12,176 optimizer steps
  per seed. Five seeds (0-4), averaged into these weights.
- **Hardware:** PyTorch on one 2x H100 node (about 6.1 hours of training per seed).
- **Data:** 487,407 training rows across 23 task families. Every run was pre-registered before its
  result was read.

| Family group | Training rows | Source |
|---|---|---|
| `code.defect_class` (not served) | 71,435 | Mutations of CommitPackFT files, labelled by construction; abstain rows from prose, scrambled text and unseen languages, including the author's own repositories |
| `qa.answer_span` (not served) | 105,857 | SQuAD v2 |
| `procedural.decisions` | 91,955 | tasksource/procedural-typed-decisions |
| `intent.*` (4 families) | 84,021 | CLINC150 (clinc_oos) |
| `openjev.*` (6 families) | 54,298 | Open-Jev v1.1, redistributable subset |
| `synth.general` | 14,996 | n4ze3m/typed-decisions-synth |
| `knowledge.multiple_choice` | 13,124 | MMLU |
| `decider.*` (2 families) | 14,688 | decider teacher data (Mapika/decider) |
| `commonsense.multiple_choice` | 9,619 | CommonsenseQA |
| `pairwise.helpfulness` | 9,636 | HelpSteer2 |
| `boolq.reading` | 7,279 | BoolQ |
| `vitaminc.nli` | 4,995 | VitaminC |
| `arc.science` | 3,178 | ARC |
| `typed.workflow` | 2,326 | LocalLLaMA/typed-decisions |

- **Held out:** two task families (`code.language_id`, `qa.answerability`) were never read by a
  training process. A path check in the trainer refuses them.
- **Deduplication and splits:** MinHash/LSH (threshold 0.8) across the pool. Training rows that
  overlap validation rows were excluded, after an earlier corpus was found to share 215 validation
  rows with MMLU and CommonsenseQA training rows.
- **Public test splits in the training data:** MMLU's test and dev splits, and CLINC150's test
  utterances for the training intents, were trained on (see the note under the per-family table).
  The decision families were checked against ARC's test split, BoolQ's validation split, VitaminC's
  test split and JevArena's three source benchmarks, and overlapping rows were removed.
- **Controls:** a model trained on shuffled labels scored at chance on v5 seeds 0-3 (0.218-0.237
  against a 0.305 ceiling); it was not run on seed 4. A per-family linear baseline on v5 seed 0
  trailed Lappi on 20 of 22 families. No control was run on the averaged weights themselves.

## Limitations

- A preview. It is not promoted under the project's own rules, and its two missed lines (span and
  OOD abstention) are real.
- It answers confidently on some inputs it should refuse: unseen programming languages above all.
- It does not serve the code-review use it was built for. Asked its own code-defect question, it
  separates planted mutations of the kinds it was trained on, but not RM-Bench's correct and broken
  programs.
- Its MMLU and CLINC150 scores are not benchmark results: their test items are in the training data.
- Calibration on 16-option questions fails its own threshold.
- The training text is English (the code is in several programming languages). Contexts and
  option text outside the trained families' shapes are untested.
- Six of the 21 served families were not exercised through the runtime.

## Files

| File | What |
|---|---|
| `model.safetensors` | The language tower in bf16 (320 `model.language_model.*` tensors, 1,881,825,088 parameters), the mean of five seeds. No vision tower, so the count is below the base model's published total |
| `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | The base model's files, byte for byte. `config.json` still describes the base's multimodal architecture; only its text tower is in this release |
| `calibration.json` | Per slot width: temperature, conformal quantile, abstention margin. Span is `null` |
| `release_manifest.json` | The hashes the runtime checks at load, and the trained families |

Identity: weight hash `6f7b9ba7…dfcb`, tokenizer hash `fe000e3e…2927`, calibration hash
`7e56b34e…b677` (full values in `release_manifest.json`).

## Licence and attribution

The weights are released under **Apache-2.0**.

- **Base model:** Qwen3.5-2B-Base, Copyright 2026 Alibaba Cloud, Apache-2.0. Its licence is included
  as `LICENSE-Qwen`.
- **Training data** (licences as declared by each source; rows were admitted only under these):

| Source | Licence | Cite |
|---|---|---|
| CommitPackFT (code files kept only under MIT, Apache-2.0, BSD-2/3-Clause, ISC, CC0-1.0, Unlicense) | per file; dataset MIT | Muennighoff et al., *OctoPack: Instruction Tuning Code Large Language Models*, 2023 |
| SQuAD v2 | CC BY-SA 4.0 | Rajpurkar, Jia, Liang, *Know What You Don't Know: Unanswerable Questions for SQuAD*, ACL 2018 |
| CLINC150 (clinc_oos) | CC BY 3.0 | Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019 |
| MMLU | MIT | Hendrycks et al., *Measuring Massive Multitask Language Understanding*, ICLR 2021 |
| CommonsenseQA | MIT | Talmor, Herzig, Lourie, Berant, *CommonsenseQA*, NAACL 2019 |
| BoolQ | CC BY-SA 3.0 | Clark et al., *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions*, NAACL 2019 |
| VitaminC | CC BY-SA 3.0 | Schuster, Fisch, Barzilay, *Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence*, NAACL 2021 |
| ARC | CC BY-SA 4.0 | Clark et al., *Think you have Solved Question Answering? Try ARC*, 2018 |
| HelpSteer2 | CC BY 4.0 | Wang et al., *HelpSteer2*, 2024 |
| Open-Jev v1.1 (redistributable subset; rows kept only if marked CC BY 4.0 or CC0) | per row | Open-Jev; its NLI rows derive from WANLI (Liu, Swayamdipta, Smith, Choi, 2022) |
| tasksource/procedural-typed-decisions | Apache-2.0 | Sileo, *tasksource*, LREC-COLING 2024 |
| LocalLLaMA/typed-decisions | Apache-2.0 | |
| n4ze3m/typed-decisions-synth (generated by DeepSeek V4.1 Flash) | MIT | Nazeem, *Hmm: a small open model for typed decisions*, 2026 |
| decider teacher data (labelled by a local Qwen3.5-27B teacher) | Apache-2.0 (the repository's licence) | Marosi, *decider*, 2026 |
| The author's own repositories (prose, used as abstain rows) | the author's own | |

- **ShareAlike.** About 123,000 training rows come from CC BY-SA sources (SQuAD v2, ARC, BoolQ,
  VitaminC). This card attributes them. Whether ShareAlike reaches trained weights is not settled,
  and this card does not claim either way.

## Citation

```bibtex
@misc{lappi2026,
  title  = {Lappi v0.1 preview: a typed decision model},
  author = {Vaddaram, Bharath Chandra},
  year   = {2026},
  note   = {Fine-tuned from Qwen3.5-2B-Base}
}
```
