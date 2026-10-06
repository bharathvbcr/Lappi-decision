# Lappi: a typed decision model

[![Website](https://img.shields.io/badge/website-lappi.vbcr.dev-B91C1C?style=flat&logo=safari&logoColor=white)](https://lappi.vbcr.dev/)

Lappi answers typed questions. You give it a context, a question and named options. It returns one
option with a score, or an explicit abstention when it should not answer. It does not generate text
and does not reason first: each answer is one token read from a 17-row slice of the output layer of
a fine-tuned [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base).

This repository is `qwen-decision`, and the crates keep the `qd-` prefix. The model is Lappi, a name
chosen so it does not expire when the base model changes (`docs/lappi.md`). Code is Apache-2.0.

## Status: v0.1 preview (2026-10-06)

**Released as a preview, not promoted.** The weights are the plain mean of five training seeds. A
bar was written down before the average was scored: no worse than the weakest seed on six measures.
It passed four and missed two, span accuracy and out-of-distribution abstention, and the author
chose to release it anyway (`HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md`).

| Measure (validation) | v0.1 preview | Bar |
|---|---|---|
| Choice top-1, all 22 choice families | 0.857 (14,779 / 17,254) | none |
| Choice top-1, the 21 families it serves | 0.842 (12,580 / 14,950) | none |
| Permutation consistency | 0.957 | >= 0.95, passes |
| Abstention on in-distribution questions | 4.6% | <= 5%, passes |
| Abstention on out-of-distribution inputs | 149 / 180 | >= 0.9, **fails** |
| Needle recall, worst long-context bucket | 1.000 | >= 0.95, passes |
| Span top-1 (span is not served) | 0.773 | none |

The served-families row is recomputed from the same score row's verdicts by
`docs/outreach/evidence/served_families.py`, which first reproduces the all-families figure.

### What the preview serves

- **Choice questions with 2 to 16 options** in 21 trained task families: routing, policy, evidence,
  NLI, reading comprehension, science, intent and multiple-choice questions. Fifteen of them were
  load-tested through the runtime on Apple silicon.
- At an 80% precision target it answers about 91% of validation questions at about 86% precision
  (two-fold estimate, served families). 16-option questions are refused about half the time.

### What it does not serve

- **`code.defect_class`**, the code-change family this project was built around. Its trained
  request needs a span slot, which has no calibration in this release. A choice-only request is a
  prompt the model never saw. Multi-file diffs are refused at admission.
- **Span and score slots.** The release's span head is the mean of five seeds' heads, which do not
  agree, and it is not served. Span top-1 pools SQuAD answer spans (5,139 val rows) and
  code-change evidence lines (2,099). On the code rows it is weak evidence, because the diff's `+`
  markers show the changed lines (GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD).
- **Benchmark scores on MMLU or CLINC150.** Their test items are training data
  (`python/qd_data/sources.py`, `benchmark_reportable=False`).
- **Anything outside the 23 trained families**: refused with `task_not_trained`.

### How it compares

On [JevArena](https://github.com/chenmingtang830/jevarena), a pairwise-judge benchmark it was not
trained for (849 pairs, local models only), Lappi ties Qwen3.5-2B: domain macro 0.534 against
0.522, with an interval on the difference that includes zero. Gemma 4 E4B (8B) is clearly better
(0.632), and so is decider-2b (0.612), a typed decision model on the same base model: paired, it
leads Lappi by +0.110 [+0.071, +0.157]. Pairwise response preference is one of decider-2b's trained
tasks, asked in its own request format; for Lappi it is not. Lappi is near chance outside
RewardBench 2's Ties domain and scores 0.111 on RM-Bench's
code pairs, where it calls most pairs a tie or prefers the broken program. Those pairs reach it as a
"which answer is better" question, through `pairwise.helpfulness`, its weakest family. Asked its own
code-defect question in a diagnostic probe, it separates a commit's real file from the same file
with a planted mutation (0.906 of 85 pairs), but not RM-Bench's correct programs from the broken
ones (0.536 of 125 pairs, chance). On a 191-pair subset with no ties, Gemma 4 E4B with reasoning on
scores 0.816 against Lappi's 0.571 on the same pairs, at about 14 s per verdict. JevArena details:
`HANDOFF/merge-and-bench-2026-10-06.md` section 6.

On an M5 Pro, a full decision takes about 72 ms at a 124-token prompt and 2.3 s at 8,185 tokens. That
is the same as the base model: only the weights differ (section 5 of the same file).

## How it works

```
request  { schema_version, task, context_b64, context_len, question, slots: [{name, type, options}], route }
answer   { slot -> { value, conformal_set, score, noul, degraded } }
```

- **The context travels as bytes** (`context_b64` plus `context_len`, which must agree), so Unicode
  normalisation can never move a line.
- **A choice slot is decoded over 17 rows:** up to 16 option letters, plus one reserved abstain row
  (`noul`), always last. The abstain row competes in the same softmax but is never an option, so the
  abstain rate is not entangled with the label distribution. That is the design's intent; on its own
  the row did not make out-of-distribution abstention pass (149 / 180).
- **Each choice is asked twice,** the second time with the options permuted. If the two passes
  disagree, the answer is `noul`.
- **Calibration** is a fitted table per slot width (temperature, conformal quantile, abstention
  margin). A release without its table does not serve.
- **Refusals are not abstentions.** An untrained task, a hash mismatch or too many options is a
  typed error that names the check. `noul` means the model abstained.

`docs/schema-api.md` is the full contract. Note that its worked example is a `code.defect_class`
request, which this preview refuses.

## Run it

Apple silicon only for now: the backend runs on the author's
[tessl](https://github.com/bharathvbcr/tessl) Metal kernels, which `crates/qd-metal` takes as a
path dependency from a sibling checkout (`../tessl` next to this repository).

```bash
cargo build --release -p qd-metal -p qd-runtime
```

```bash
target/release/qd-metal-serve --release <release-dir> --socket /tmp/lappi.sock
```

```bash
target/release/qd oneshot --socket /tmp/lappi.sock < request.json
```

The server checks the release's weight, tokenizer and calibration hashes when it binds. The release
directory holds `model.safetensors`, the base model's config and tokenizer files, `calibration.json`
and `release_manifest.json`. The weights will be published on Hugging Face [pending].

A request must use a trained family's id as `task` and that family's trained question verbatim,
because both are rendered into the prompt. The model card lists them.

## How it was built

- **Cheapest rung first.** A 606K-parameter byte model trained from scratch set the floor; a linear
  control beat it in 17 of 18 arms (best 80.97% against 88.5%), which is what justified the 2B
  (`docs/build-order-2026-09-19.md`).
- **Labels by construction.** `crates/qd-mutate` edits real files with tree-sitter, so a defect's
  class and line span are facts about the edit, not a model's guess.
- **Data:** 487,407 training rows across 23 task families. Code-change rows from CommitPackFT
  (permissively licensed files only), the general families from open datasets, and abstain rows from
  prose, scrambled text and unseen languages. MinHash/LSH dedupe in Rust (`crates/qd-prep`).
  Training rows that overlap validation rows are excluded.
- **Recipe:** supervised fine-tuning only, in PyTorch on 2x H100. AdamW with fp32 master weights,
  learning rate 1e-5, lower 8 layers at 0.1x, one epoch (12,176 steps), five seeds averaged.
- **Held out:** two task families are never read by a training process; `qd-train` refuses their
  paths rather than trusting convention.
- **Controls:** shuffled-label training ran on four of the five seeds and scored at chance on all
  four; a linear baseline is fitted per family. The pooled paired-margin check against the linear
  baseline could not run, which is one reason the preview is not promoted.

## The rules this repository runs on

- **Pre-registration.** A run's bar and risks are committed in `campaign/` before its result is read.
- **Gates are read-only.** An agent may report that a gate failed; it may not move a threshold, drop
  a seed, shrink a held-out set or relabel a run as `quick` to pass it. Gate definitions changed
  only by recorded decisions of the author.
- **Nothing is green that was not run.** A check that could not run reports `not_run`, never a pass.
- **A number cites a ledger row.** `ledger/` is append-only; a run that did not write its row is
  rerun.
- **Handoff is a file.** Each lane ends with a `HANDOFF/<lane>-<date>.md`.

`CLAUDE.md` holds the full rules, which bind every agent working here.

## Layout

| Path | Holds |
|---|---|
| `crates/qd-mutate/` | Rust + tree-sitter mutation engine: labels and spans by construction |
| `crates/qd-runtime/` | Rust: schema API, admission, rendering, calibration serving, `qd serve` / `qd oneshot` |
| `crates/qd-metal/` | Rust: the Mac decision backend on tessl's Qwen3.5 Metal kernels, `qd-metal-serve` |
| `crates/qd-prep/` | Rust: MinHash/LSH dedupe and splits, the decision-data normaliser, linear controls |
| `crates/qd-train/`, `crates/qd-train-metal/` | Rust: a trainer over a model-generic step provider (not used for the preview) |
| `crates/qd-export/`, `crates/qd-lang/`, `crates/qd-preflight/` | Rust: release export, language admission, tri-state preflight |
| `python/qd_data/`, `python/qd_train/` | Data filters, prompt rendering, splits; fine-tuning, evaluation, calibration fitting, ledger |
| `campaign/` | Pre-registrations |
| `ledger/` | Append-only JSONL run records |
| `HANDOFF/` | One file per completed lane; the newest is the current state |
| `AUDIT/` | Evidence for every external claim the plan rests on |
| `docs/outreach/` | Drafts of the public write-ups and the model card |

## Read in this order

| Document | What it settles |
|---|---|
| `HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md` | What the preview is, what it serves and what it misses |
| `docs/lappi.md` | What Lappi is and how it differs from its peers |
| `docs/schema-api.md` | The request and answer contract |
| `HANDOFF/merge-and-bench-2026-10-06.md` | Speed on the Mac and the JevArena comparison |
| `AUDIT/prior-art-jev-nimble-2026-09-19.md` | The public peers and their licences |
| `docs/build-order-2026-09-19.md` | Why the cheap rungs ran first |
| `campaign/v5-preregistered.json` | The pre-registration the preview was trained under |
| `CLAUDE.md` | The rules every agent follows |

## Tests

```bash
cargo test --workspace
```

```bash
.venv/bin/python -m pytest python/tests -o 'addopts='
```

The GPU tests in `crates/qd-metal` are `#[ignore]`d and run with `--ignored` on Apple silicon. The
newest `HANDOFF/` file records the last measured totals; re-run rather than quote them.

## Licence

The code is Apache-2.0 (`LICENSE`). The v0.1 preview weights are released under Apache-2.0; the base
model is Apache-2.0, and the training data's sources, licences and attributions are listed in the
model card (`docs/outreach/hf-model-card.md`).
