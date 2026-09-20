# qwen-decision — the repository behind **Lappi**

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
  16-row slice of `lm_head`. Any app, any new task, no training.
- **Registered route** — an app registers a task once; a small head reads pooled features directly.
  Same prefill, same calibration table, same abstain rule. A speed-up, never a different model.

## Status — 2026-09-19

Code is substantial; **no model has been trained.** `ledger/runs.jsonl` does not exist, there is no
training data on disk, and no base-model weights are cached. Every test number below is about code.

Suites: **981 Python tests, 0 failed, 5 skipped** (4 torch-gated modules, 1 network); 317 Rust passed.

On 2026-09-18/19 four public System-1 decision models appeared, one of them on this exact base model.
That did not change the destination, but it changed the order of the work — Lappi is now built as a
**ladder**, cheapest rung first, so the expensive rung is justified by measurement rather than by plan:

| Rung | Approach | Status |
| --- | --- | --- |
| 0 | byte-level, from scratch, 606,336 params | model built and verified, **untrained** |
| 1 | frozen base, option logits | not started |
| 2 | LoRA r=16 | not started |
| 3 | CPT + FT + vocabulary remap (the original plan) | S2/S4/trainer built, blocked on HF terms |

Rung 0 needs no Hugging Face terms and no GPU block. Weeks 1-3 run entirely on the Mac at $0, and
**no Metal kernel is written before the 8xH100 gate reports.**

Read in this order:

| Document | What it settles |
| --- | --- |
| `CLAUDE.md` | The rules that bind every agent. Read first |
| `docs/lappi.md` | What Lappi is, the ladder, and the three things that are actually differentiated |
| `docs/schema-api.md` | The typed request/answer contract, shared by training and serving |
| `docs/ledger-schema.md` | The decision record, and the tri-state that keeps "not run" from reading as "passed" |
| `docs/hardening.md` | The adversarial contract each lane tests against before calling a feature done |
| `docs/build-order-2026-09-19.md` | Why the cheap rungs run first |
| `docs/schedule-2026-09-28.md` | What can and cannot be finished, with the evidence |
| `AUDIT/prior-art-jev-nimble-2026-09-19.md` | The four public peers, their licences, and what is reusable |
| `AUDIT/` | Evidence for every external claim the plan rests on |
| `gaps.jsonl` | What DevMap and GitPulse could not answer |
| `HANDOFF/` | One file per lane completion |

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
crates/qd-runtime/    Rust. Graph, schema API, `qd serve` (socket) and `qd oneshot` (sidecar)
python/qd_data/       Pool filters, prompt format, splits, dedupe
python/qd_train/      CPT, FT, eval harness, ledger, calibration fitting
  byte_context.py       Rung 0: exact byte-offset line grid, torch-free
  byte_decider.py       Rung 0: the 606K byte-level model, torch-gated
  calibration_fit.py    Temperature, conformal cutoff and ECE -- the gate nothing computed
python/qd_wire/       The Rust->Python answer contract, parsed against a golden corpus
ledger/runs.jsonl     Append-only. A run whose row is missing is rerun, not remembered
```

Kernels (K1-K7, `tests/gdn.rs`, fixtures) land in the **canonical** tessl crate at
`~/Code/research/tessl`, never the nested copy under `MLSystemsLab/Rust_MLKit/`.

## Provenance

The model card lists the licence filter applied to the pool. Held-out data is 300 hand-labelled diffs
from the author's own repositories plus real DevType queries; it is never trained on, and `qd-train`
refuses a held-out path rather than trusting convention.
