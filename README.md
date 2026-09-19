# qwen-decision

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

## Status

Greenfield as of 2026-09-19. Weeks 1-3 of the plan run entirely on the Mac at $0. The model is judged
on one 8xH100 block in week 4; **no Metal kernel is written before that gate reports.**

Read in this order:

| Document | What it settles |
| --- | --- |
| `CLAUDE.md` | The rules that bind every agent. Read first |
| `docs/schema-api.md` | The typed request/answer contract, shared by training and serving |
| `docs/ledger-schema.md` | The decision record, and the tri-state that keeps "not run" from reading as "passed" |
| `docs/hardening.md` | The adversarial contract each lane tests against before calling a feature done |
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
python/qd_train/      CPT, FT, eval harness, ledger
ledger/runs.jsonl     Append-only. A run whose row is missing is rerun, not remembered
```

Kernels (K1-K7, `tests/gdn.rs`, fixtures) land in the **canonical** tessl crate at
`~/Code/research/tessl`, never the nested copy under `MLSystemsLab/Rust_MLKit/`.

## Provenance

The model card lists the licence filter applied to the pool. Held-out data is 300 hand-labelled diffs
from the author's own repositories plus real DevType queries; it is never trained on, and `qd-train`
refuses a held-out path rather than trusting convention.
