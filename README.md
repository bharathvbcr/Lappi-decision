# qwen-decision — the repository behind **Lappi**

[![Website](https://img.shields.io/badge/website-lappi.vbcr.dev-B91C1C?style=flat&logo=safari&logoColor=white)](https://lappi.vbcr.dev/)

**Lappi** is the model. `qwen-decision` is the repository and `qd-` stays the crate prefix; the
product name is deliberately independent of the base model, because a name that encodes a dependency
expires when the dependency changes. See `docs/lappi.md`.

A general **typed decision model**: give it context bytes, a question, and a schema of slots, and it
answers the slots. Apache-2.0.

One model, one interface, two routes. DevCouncil and DevType are the first two callers; the interface
is not specific to either.

```
request  { context, question, slots: [{name, type, options?|bins?}], route }
answer   { slot -> { value, conformal_set, score, noul, degraded } }
```

Slot types: `choice` (k <= 16 named options), `score` (ordinal bins), `span` (start/end line in the
context), and `noul` (abstain, present in every option set).

- **Generic route** — options are written into the prompt with letter labels and decoded over a
  **17-row** slice of `lm_head`: 16 option letters plus one reserved `noul` row, which is **last**
  (`schema.rs` sizes a slot at `options.len() + RESERVED_NOUL_ROWS`; `answer.rs` reads the
  abstention back at `plan.rows - RESERVED_NOUL_ROWS`). Any app, any new task, no training.
- **Registered route** — an app registers a task once; a small head reads pooled features directly.
  Same prefill and the same calibration table — but **not the same abstain rule**, and it is not a
  drop-in for every slot. `answer::abstain_rule` is the one place that says so: a generic `choice`
  abstains on margin *and* a permuted second pass, a registered `choice` on margin only, and a
  registered `span` is refused outright. Faster, and narrower.

## Status — 2026-10-02

No model has shipped. Rung 0 is trained and evaluated, the first 2B fine-tunes have run, and the
**full-vocabulary GH200 campaign has produced its first gate-complete 2B run (F, on corpus v4)**.
It fails the gates below, and the gates are read-only — so the linear baseline remains the shipping
candidate until a 2B run clears them. Data lives under `data/pool/` (including the mutation corpus
manifests) and the base model `Qwen/Qwen3.5-2B-Base` is cached on disk.

Suites, measured rather than remembered — **re-run them instead of quoting a count**, because the
last counts written here went stale inside a day. The most recent recorded totals are in the
newest `HANDOFF/` files (for example `HANDOFF/gh200-phase4-2026-10-01.md`):

```
cargo test --workspace
.venv/bin/python -m pytest python/tests -o 'addopts='
```

On 2026-09-18/19 four public System-1 decision models appeared, one of them on this exact base model.
That did not change the destination, but it changed the order of the work — Lappi is built as a
**ladder**, cheapest rung first, so the expensive rung is justified by measurement rather than by plan:

| Rung | Approach | Status |
| --- | --- | --- |
| 0 | byte-level, from scratch, 606,336 params | **Trained & evaluated.** 80.97% best vs 88.5% control; 144 margins measured across 18 arms (`ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl`), trailing control in 17/18 arms and justifying higher rungs |
| 1 | frozen base, option logits | Planned as Step 1 in `docs/train-plan-2026-09-28.md` |
| 2 | LoRA r=16 | Planned as Step 6 control arm |
| 3 | 2B Supervised FT / CPT | **Campaign running; no run clears the gates yet.** The 2k-row pilot (`ledger/gh200-ft-commitpackft-2026-09-22.jsonl`) learned the score slot on 2/3 seeds. The full-vocabulary campaign then ran phases 0–4 on one GH200 (corpus v1 → v4, noul-row reforms, multi-hunk diffs, composed long-context rows); see "Where the 2B stands" below |

### Where the 2B stands

The campaign is a sequence of pre-registered runs (`campaign/*.json`), each recorded in a ledger row
before its result is read. The latest gate-complete run, **F** (corpus v4, seed 0, eval row
`f4feac15`; seed 1 is row `aeca8d69`), reads:

| Gate / measure | F seed 0 | Reading |
| --- | --- | --- |
| Pooled choice top-1 | 83.1% | defect-class slots near 99%; MMLU 0.61 and CSQA 0.76 are base-model-bound for a 2B |
| Permutation consistency | 93.2% | **fails** the 95% floor |
| In-distribution abstention | 7.7% | **fails** the 5% cap; equals permutation disagreement, so on MMLU the two gates cannot both pass under the pooled population (a human decision, `promotion_population`) |
| Needle-hunk recall | 1K / 2K / 4K buckets 1.000; 8K bucket fails | the 8K suite averages ~8,473 real tokens against an 8,192 target; the human decided to resize it for v5 |
| OOD abstention | prose 54/60, scrambled 58/60, unseen-language 20/60 | unseen-language needs more *distinct* languages in training (G6) |
| ECE | domain k10 0.081, within-domain k15 0.190 | overconfident on 10/15-way slots |

Earlier phases are why these are the open items: J4 (corpus v3, three seeds) showed needle recall
varies widely by seed, and a three-seed weight average (`1d93b3ee`) lifted pooled top-1 to 83.2% but
abstained on 0/180 OOD rows, which is why a logit ensemble is scored beside it. A report-only
re-score that holds out the 215 validation rows (183 MMLU, 32 CSQA) found to overlap F's MMLU/CSQA training rows
(`ledger/mac-rescore-excluded-2026-10-02.jsonl`, rows `e663248b` for seed 0 and `9e521e77` for seed 1;
seed 2 follows) moved no failing gate to passing on seed 0.

What is queued, and under whose yes:

- **Idle queue on the GH200:** J6(a) and then **J6(g)** (F plus shuffled answer options during
  training, one seed, quick, ~$11) — approved 2026-10-02 (`AUDIT/fable-optimize-2026-10-02/human-decisions.md`).
- **v5:** decontaminated data (train rows excluded against every val row), CLINC out-of-scope
  re-keyed across splits, longer composed examples, recipe fixes. **Planned only; it launches on a
  later explicit yes** with the final cost (about $70 for three seeds plus controls).
- **Training without PyTorch:** a Rust trainer (`crates/qd-train`) over tessl's Qwen3.5 Metal step
  on the Mac, with a CUDA provider started in ojas (`HANDOFF/ojas-training-2026-10-01.md`). Until it
  moves into ojas, results read "a Rust trainer over tessl". The GH200 campaign stays on PyTorch.

Repository is hosted on GitHub: [bharathvbcr/Lappi-decision](https://github.com/bharathvbcr/Lappi-decision).

Read in this order:

| Document | What it settles |
| --- | --- |
| `CLAUDE.md` | The rules that bind every agent. Read first |
| `docs/lappi.md` | What Lappi is, the ladder, and the three things that are actually differentiated |
| `docs/train-plan-2026-09-28.md` | Fast 2B training plan based on seven external models and 2026-09-28 status |
| `docs/schema-api.md` | The typed request/answer contract, shared by training and serving |
| `docs/ledger-schema.md` | The decision record, and the tri-state that keeps "not run" from reading as "passed" |
| `docs/hardening.md` | The adversarial contract each lane tests against before calling a feature done |
| `docs/build-order-2026-09-19.md` | Why the cheap rungs run first |
| `docs/schedule-2026-09-28.md` | What can and cannot be finished, with the 2026-09-28 verification |
| `AUDIT/prior-art-jev-nimble-2026-09-19.md` | The four public peers, their licences, and what is reusable |
| `AUDIT/` | Evidence for every external claim the plan rests on |
| `gaps.jsonl` | What DevMap and GitPulse could not answer |
| `campaign/` | Pre-registrations: the bar and the risk, written before a run's result is read |
| `AUDIT/fable-optimize-2026-10-02/` | The latest advisor ruling (training speed, decontamination, model improvement) and the human's decisions on it |
| `HANDOFF/` | One file per lane completion; the newest is the current state |

## The one gate that decides it

The 2B ships only if it beats the char-n-gram linear baseline by a paired margin on three seeds on the
natural held-out set, keeps in-domain `noul` under the cap, sends the held-out task families to `noul`,
**and** passes the 8K needle-hunk recall test. If it fails any of them, the linear baseline ships and
the GDN kernels stay a research artifact.

Gates are read-only. An agent may report that a gate failed; it may not move a threshold, drop a seed,
shrink a held-out set, or reclassify a run as `quick` to pass it.

## Layout

```
crates/qd-mutate/     Rust + tree-sitter. Labels by construction, with exact span labels
crates/qd-runtime/    Rust. Graph, schema API, `qd serve` (socket) and `qd oneshot` (sidecar), calibration
                      serving, N-tower ensemble serving, gate reports
crates/qd-prep/       Rust. MinHash/LSH dedupe and splits, linear controls (the CPU prelude)
crates/qd-train/      Rust. Trainer over a model-generic step provider (tessl first, ojas later)
crates/qd-metal/      Rust. Mac decision backend on tessl's Qwen3.5 Metal kernels
crates/qd-export/     Rust. Release exporter to qd-metal's loader layout
crates/qd-lang/       Rust. Language admission checks
crates/qd-preflight/  Rust. Tri-state preflight checks ("not run" never reads as "passed")
python/qd_data/       Pool filters, prompt format, splits, dedupe
python/qd_train/      CPT, FT, eval harness, ledger, calibration fitting
  byte_context.py       Rung 0: exact byte-offset line grid, torch-free
  byte_decider.py       Rung 0: the 606K byte-level model, torch-gated
  calibration_fit.py    Temperature, conformal cutoff and ECE -- the gate nothing computed
python/qd_wire/       The Rust->Python answer contract, parsed against a golden corpus
ledger/runs.jsonl     Append-only. A run whose row is missing is rerun, not remembered
```

Kernels (K1-K7, `tests/gdn.rs`, fixtures) land in the **canonical** tessl crate at
`~/Code/research/tessl`, never the nested copy under `MLSystemsLab/Rust_MLKit/`. CUDA kernels and the
CUDA Qwen3.5 provider live in ojas (`ojas/ojas-qwen35-cuda/`), because tessl is Metal-bound; Lappi
itself stays kernel-free.

## Provenance

The model card lists the licence filter applied to the pool. Held-out data is 300 hand-labelled diffs
from the author's own repositories plus real DevType queries; it is never trained on, and `qd-train`
refuses a held-out path rather than trusting convention.
