# Prior art: TypeSafe Jev and Bespoke Nimble — checked 2026-09-19

The build plan is dated 2026-09-19. On the same day, two systems with the same thesis became
public. This file is the evidence for what they are, what they do that this plan also does, and the
three places where this plan is actually different — so that the difference is a design position
held on purpose rather than an assumption nobody checked.

**Method.** `WebFetch` of the repository README and the GitHub API metadata, plus a web search for
coverage. **Not checked: the source code.** Everything below about internals comes from the README,
the API metadata, or press coverage, and is labelled accordingly. Nothing here is code-confirmed.

## Verdict table

| Claim the plan implicitly rests on | Verdict |
| --- | --- |
| A typed, non-generative decision model over a schema is novel | **REFUTED.** Two shipped, one commercial and one open, both public 2026-09-18/19 **[V]** |
| "Read the option letter's logits instead of generating" is our trick | **REFUTED.** `mini-jev` does exactly this on a frozen Qwen3-4B **[V]** |
| A calibrated probability is table stakes for this category | **CONFIRMED, and it is the open gap.** Nimble explicitly disclaims calibration **[V]** |
| A first-class abstain row is the obvious design | **REFUTED as obvious.** Nimble recommends a catch-all *answer* instead **[V]** |
| Span slots are a normal part of the category | **UNSUPPORTED.** Neither system offers one **[V]** |
| Their training data could seed our pool | **REFUTED. No licence — all rights reserved** **[V]** |

## What they are

**Jev** (TypeSafe, commercial). Transformer-based but not an LLM; does not generate text. Send a
state plus typed questions, receive typed decisions with probabilities that code branches on.
Released 2026-09-19.

**Nimble** (Bespoke Labs, open). Built as an open rival **in two days**. Repository created
2026-09-18, last push 2026-09-20, 443 stars / 34 forks / 3 open issues at time of check. Description:
*"Local typed decisions, contrastive data curation, and model evaluation."*

Recipe, from its README:

| | |
| --- | --- |
| Base | Qwen3.5-9B, LoRA rank 16 |
| Optimiser | lr 5e-5, effective batch 8, 1 epoch (chosen from a 3-epoch schedule), bf16 |
| Data | 2,676 train / 324 held out |
| Prompt cap | 2,048 tokens, schema and field names included |
| Slots | `enum` (1–26 string choices), `boolean`, `score`. **Flat only — no nested fields** |
| Isolation | *"Each field is scored separately, so one field cannot see the answer to another field"* |
| Latency (median) | 106 ms on H100; **444 ms on an M5 Pro 64 GB Mac** |

Reported accuracy on their 324-example held-out set, metric is **raw reference-label agreement**:

| System | Agreement |
| --- | --- |
| Jev 1.13.0 | 93.21% (302/324) |
| Bespoke-Nimble-9B | 90.12% (292/324) |
| Untuned baseline | ~66% — read via a summariser, the table carries several baseline rows; **treat as approximate** |

An evaluation ecosystem is already forming around that held-out set (AmberTrace is running
Jev-vs-Nimble comparisons).

## The three differences that are real

**1. Calibration — the one that matters.** Nimble's README states it outright: the scorer softmaxes
the logits and rescales them to sum to 1 across the supplied answers, temperature fixed at 1.0 and
untuned, and *a probability of 0.9 does not mean the answer is right 90% of the time*. That is a
normalised score, not a calibrated one. This plan's split-conformal-on-margin gives an actual
coverage guarantee, which is precisely what a caller branching on a confidence score needs and
precisely what the incumbent disclaims.

**2. Abstain.** Nimble's guidance is to add an answer meaning "no match". That catch-all competes
with the real options for probability mass, so the abstain rate is entangled with the label
distribution. This plan reserves a `noul` row (`RESERVED_NOUL_ROWS`, pinned last in `answer.rs` and
matched at column `n_candidates` in `heads.py`), which is separate from the options and cannot be
crowded out by them. The competitor's choice is evidence that this was a design decision, not the
default.

**3. Span.** Nimble has enum / boolean / score. No span. *Which line is the evidence* is a different
question from *which label*, and neither system answers it.

Plus the envelope: 2B against 9B, and 444 ms median on an M5 Pro is a concrete number to beat.

## Hard constraint: no licence

The GitHub API reports **no licence** for `bespokelabsai/nimble`. No licence means all rights
reserved. `data/train.jsonl` (2,676 rows) and `data/eval.jsonl` (324 rows) are visible in the
repository and **may not be copied, redistributed, or used to train anything here.**
`qd_data.licences` would refuse the source, and that refusal is correct. Read it for method only.

This also means their 324-example held-out set is **not** a benchmark this project can adopt
directly; a comparison would have to be run against our own set and reported as such.

## Consequences for the plan

1. **Contrastive curation is the transferable lesson.** 2,676 examples moved a 9B from roughly 66%
   to 90%. Their method: paired examples differing in one *focus fact* that flips the label, edits
   capped at ~8 words, four validation passes, and a pair kept only when every check passes **and**
   the two labels differ. `crates/qd-mutate` is already a stronger instrument for this on code — it
   labels *by construction* from a tree-sitter edit, so the label is a property of the mutation
   rather than something a separate model call has to re-verify. The framing is worth adopting even
   though the implementation here is better.

2. **The 300 commissioned hand labels are probably worth more as contrastive pairs** than as
   independent samples, on this evidence.

3. **A LoRA control run is now cheap information.** Nimble reached 90% with rank 16 for one epoch in
   two days. This plan's CPT + FT with an ~80K vocabulary remap and fused cross-entropy is far
   heavier surgery, and S2's parity gate has already cost real effort. A LoRA baseline would say
   whether the heavy path earns its cost before the GPU budget is committed. **This is a
   recommendation to a human, not a plan change.**

4. **Metrics will not be comparable.** They report raw agreement; this plan uses Cohen's κ with a
   bootstrap CI. On skewed labels raw agreement flatters heavily, so 90.12% and a κ are different
   quantities. Any public comparison must say so explicitly or it will read as a loss that is not
   one.

5. **`GAP-SCHEMA-NOUL-LETTER-BUDGET` gets a data point.** Nimble supports 1–26 choices where
   `MAX_CHOICE_OPTIONS` here is 16. That does not settle whether our 16 rows are 16 options or 16
   plus `noul`, but it does say 16 is not an externally forced ceiling.

## Open

- Source not read. Every internal claim above is README-level, not code-confirmed.
- Whether Jev or Nimble has any abstain mechanism beyond the documented catch-all is **unverified** —
  absence from a README is not absence from the system.
- The ~66% untuned baseline came through a summariser rather than the raw table. Re-read before it is
  quoted anywhere that matters.

Sources: `github.com/bespokelabsai/nimble`, its GitHub API metadata,
`marktechpost.com/2026/09/19/typesafe-ai-releases-jev/`, `github.com/r-ms/mini-jev`,
`github.com/ambertrace-labs/ambertrace-rlvr/issues/123`.
