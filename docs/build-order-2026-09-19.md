# Build order after the prior-art check — 2026-09-19

Two systems with this thesis went public on 2026-09-18/19 (see
`AUDIT/prior-art-jev-nimble-2026-09-19.md`). Bespoke Nimble reached 90.12% raw agreement with
LoRA r=16 for one epoch on 2,676 examples, built in two days on a 9B. That changes the build
order here, not the destination.

## The position

Matching them is not the goal; they are ahead on a path we would be following. The product worth
building is the one neither system has:

> **A typed decision model that knows when it does not know, and can point at its evidence.**

Two capabilities, both load-bearing because **neither can be added afterwards**:

- **Calibrated abstention.** Nimble's own README disclaims calibration — softmax over the supplied
  answers, temperature fixed at 1.0, and a stated warning that 0.9 does not mean right 90% of the
  time. Abstention has to live in the head as a reserved row (`RESERVED_NOUL_ROWS`, pinned last in
  `answer.rs`, matched at column `n_candidates` in `heads.py`). Nimble's documented alternative — add
  an answer meaning "no match" — puts abstention into competition with the real options for
  probability mass, which entangles the abstain rate with the label distribution.
- **Span grounding.** Which *line* is the evidence. That has to be in the training data, and
  `qd-mutate` already emits it by construction.

Everything else — enum / boolean / score, flat schemas, per-field isolation, a 2K prompt cap — is
table stakes. Nimble has usefully demonstrated those are sufficient; match them and move on.

## Decision: the LoRA control runs first

Before committing to CPT + FT with the ~80K vocabulary remap and fused cross-entropy.

**It runs locally. It costs nothing.**

| | |
| --- | --- |
| Host | Mac17,8, 64 GiB, 18 cores, macOS 27.0 arm64 |
| Runtime | `mlx` 0.31.2 / `mlx-lm` 0.31.3, **already installed** — no new dependency |
| Fallback | torch 2.12.1 with MPS available, `transformers` 5.12.1, `accelerate` 1.14.0 |
| Not installed | `peft`, `trl`, `bitsandbytes` — not needed on the MLX path |
| GPU approval | **Not required.** Rule 4 gates 8xH100 jobs and single-GPU spend over $20; this is neither |

This is the point of a control: if it were expensive it would not be one.

### What it decides

The control answers one question — **does the heavy path earn its cost?** — and it is only
meaningful if it is judged against a bar fixed *before* it runs:

- If LoRA on a 2B lands within a small margin of what CPT + FT is projected to give, the remap and
  fused-CE work is not where the remaining effort belongs. Redirect it to calibration and span,
  which is where the differentiation actually is.
- If it falls far short, the heavy path is justified and we proceed with evidence instead of
  assumption.

Either outcome is useful, which is what makes it worth running. The result is recorded as an
`ablation` row; `quick` is true if fewer than 3 seeds, and a `quick` row cannot promote anything
(rule 8).

### What it does *not* decide

Nothing about calibration, abstention or span. Those are the moat and they are **not blocked by the
control** — they proceed in parallel.

## Data: contrastive pairs

Nimble's result is a data-efficiency finding before it is a modelling one: 2,676 examples moved a 9B
from roughly 66% to 90%. Their method is paired examples differing in one *focus fact* that flips the
label, edits capped around 8 words, four validation passes, and a pair kept only when every check
passes and the two labels differ.

`crates/qd-mutate` is a **stronger** instrument for this on code: the label is a property of the
tree-sitter edit, so it is correct by construction and needs no separate model call to re-verify.
Nimble needs four validation steps because an LLM wrote the pair; we do not, because a parser did.

Consequences:

1. Reframe the mutation corpus explicitly as contrastive pairs, and report pair counts, not row
   counts — an unpaired row carries much less signal and averaging the two hides that.
2. The 300 commissioned hand labels are worth more as **150 contrastive pairs** than as 300
   independent samples.
3. Nimble's `data/train.jsonl` and `data/eval.jsonl` are **unlicensed and unusable** — all rights
   reserved. `qd_data.licences` refuses the source and that refusal is correct. Method only.

## Gates

The differentiators have to be gated or they are just intentions:

| Gate | Bar |
| --- | --- |
| Conformal coverage | Empirical coverage within tolerance of the stated level, on held-out data the training process never read (rule 3) |
| Abstain rate | Tri-state, in-domain and OOD reported separately; `noul_rate` already in the ledger row |
| Span grounding | Blocked: `GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED`. The settling check is against **decoded text** — assert the token at a gold position decodes to the start of the intended line — not against the candidate set, which shares the bug |
| Agreement | Cohen's κ with bootstrap CI, **not** raw agreement |

On that last row: Nimble reports raw agreement and we report κ. On skewed labels raw agreement
flatters heavily, so 90.12% and a κ are not the same quantity. Any public comparison must say so or
it reads as a loss that is not one.

## Blocked on a human

1. **Hugging Face terms.** Blocks Qwen3.5-2B-Base weights *and* the commitpackft pool. This is the
   critical path for the control — everything else is ready.
2. **300 hand labels**, now best spent as 150 contrastive pairs. Cannot start: no pool exists yet,
   which is the same blocker as (1).
3. **GitPulse trust** for this repo — still `REPOSITORY_TRUST_REQUIRED`; the repo has never been
   added to GitPulse's workbench (4 repositories listed, this is not one).

**Lambda is still not needed.** Nothing has launched, and the control does not require it.
