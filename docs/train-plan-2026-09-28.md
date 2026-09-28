# Training the 2B fast — plan, 2026-09-28

Asked on 2026-09-28: *get the 2B trained fast; read seven external decision models and plan it.*
Every external number below is a README / model-card claim, read by a subagent from the page and
**not reproduced here**. Every internal number cites a ledger row or a file.

## Where the 2B actually stands (verified 2026-09-28)

| Fact | Evidence |
| --- | --- |
| `Qwen/Qwen3.5-2B-Base` is cached on this Mac | `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base` |
| The 2B has had exactly one real FT: ~2,000 commitpackft rows, lr 1e-5, 1,860 steps/epoch, 3 seeds | `ledger/gh200-ft-commitpackft-2026-09-22.jsonl` |
| **choice** (`code.commit_intent`, yes/no) = 46/90 on every seed = answer-'A' baseline | rows `144ee025`, `6172cb03`, `c33dc423` |
| **score** (`code.change_scope`) = 38 / 54 / 55 of 90 vs 37 constant — **learns on 2 of 3 seeds** | same rows |
| span: 4 val rows. Uninformative | same rows |
| The 17-of-18 "trails its control" result is the **small from-scratch model** (256×2), not the 2B | `AUDIT/operator-holdout-model.md` |
| Best small model 80.97% vs char-n-gram control 88.5%; 0 of 120 cells positive | `AUDIT/capacity-and-learning-rate.md` |
| The 50,177-example mutation corpus (41,728 mutated, labels by construction) is on this Mac | `data/pool/commitpackft-corpus-v2/manifest.json` |
| **The 2B has never seen it.** `qd_data.mixture` only builds commitpackft / SQuAD families | `python/qd_data/mixture.py` |
| No rung-1 (zero-shot letter logits) number and no LoRA path exist | `rg -i 'lora|peft'` over `python/` `tools/` → none |
| The 300 hand-labelled held-out diffs the README names are **not on disk** | `data/` holds only `pool/` |

Reading: the FT pipeline is **not** broken — score learned. The choice collapse is a yes/no
decoy task on a 90-row val, which is too small to say much either way. The model has simply
never been pointed at the corpus that the whole experiment programme was built on.

## What the seven links say, reduced to what we can use

| Source | Base | What transfers | Reusable? |
| --- | --- | --- | --- |
| RSI-Jev | Qwen3.5-**2B**-Base | full FT, soft-label distillation, **bottom 8 layers at 0.1× LR** (beat freezing), **15% general replay**, per-question confidence head; "~50 min/seed on one H100" | MIT code, Apache ckpts |
| Mapika/decider | Qwen3.5-**2B**-Base | CE on option-letter logits at the slot (≈ our generic route), 50/50 prompt layouts, **bf16 AdamW — fp32 master weights reported to hurt small models**, per-type temperature; 1 epoch = 1.47M ex / 5.3 h GH200 | Apache-2.0, `train.sh full` |
| Contrastive-LM | 8B frozen | frozen backbone + 20M head, precomputed embeddings; hard negatives only as a **short late stage** | no licence stated |
| Lumma-fev, Drex, NeoHorse | 4–10B | output contract only (noul / choice / score, prefill-only) | no training recipe |
| GLiNER2.5-Decide | 340M encoder | public eval set `fastino/fast-decisions` | Apache-2.0 |

Recurring pattern: **no CPT, no vocabulary surgery** — letter logits read from the stock
`lm_head`, a plain supervised FT, replay, then a fitted temperature. Our plan's rung 3
(CPT + remap + fused CE) is the heaviest path in the field, and nothing in these sources says
it is needed.

## The plan — cheapest decisive experiment first

Rules 2, 5 and 8 stand. "Fast" means cheap experiments in the right order, not fewer seeds.

### Step 1 — Zero-shot number (rung 1). ~0.5 d agent, ~$0 (Mac MPS) or <$1 GPU

Base weights, no training, read the 17-row letter slice on the existing val set **and** on a
mutation-corpus val (step 3). Every external model reports against this baseline; without it
no FT number means anything. Blocker: none.

### Step 2 — Recipe fix on the existing 2k set. 1 run × 3 seeds × ~9 min, ~$0.70

Rerun `tools/real_ft_run.py` epoch arm, one change per arm:
(a) `--optimizer` bf16 AdamW instead of `master` (Decider's claim);
(b) bottom-8-layers at 0.1× LR (RSI's claim);
(c) lr 3e-5 (one step up from 1e-5 — the small-model grid had interior optima).
Also dump the val **prediction distribution**, not only top-1, to see whether choice is
constant-'A' or near-miss. Blocker: (b) needs a param-group split in `qd_train.optim` — small.

### Step 3 — Point the 2B at the mutation corpus. ~1–1.5 d agent — **the critical path**

Add a `code.defect_class` family to `qd_data.mixture` that renders qd-mutate `Example`s
(before/after or diff, 4-way class choice + span) into the same `Request` format, so
`real_tokenizer_pipeline.py` shards it. Repo-disjoint split as the small-model runs used.
This gives thousands of val rows instead of 90, labels by construction (the recurring
technique across the links), and the same task the 88.5% control is measured on.
Note: a `qd_data` edit moves the shard fingerprint — do it in the same change as the
`language` metadata (`GAP-FT-ECE-HAS-NO-LANGUAGE-TO-SPLIT-BY`) and the permutation hook, per
the rung-3 handoff, so shards are rebuilt once.

### Step 4 — Linear control for FT rows. ~1 d agent, parallel with step 3

`paired_margin_vs_linear` is *not built for FT*. Without it the run yields an accuracy and no
verdict. Reuse `tools/fit_linear_control.py` over the same FT train split. Also add an
**operator-holdout arm** from day one — the control is known to fingerprint generators
(`AUDIT/operator-holdout.md`), so the 2B's margin must be read on held-out operators too.

### Step 5 — The real run. 1× GH200, 3 seeds, ~1–3 h each, est. $10–25 total

FT on mutation corpus (+ commitpackft families), bf16, 1 epoch, 15% replay of general
text or of the base model's own answers, layer-wise LR from step 2's winner, then
`calibration_fit.py` per slot type. Single GPU, under $20 per job → rule 4 needs no human yes.
Needs a box: `192.222.58.240` was marked safe-to-delete on 2026-09-23 — **status unverified**.

### Step 6 — Only if step 5 clears its control: LoRA r=16 control arm, then consider more

LoRA is the Nimble / Decider-v11 recipe and a cheap check that full FT earns its cost.

## Deliberately cut from this run

- **CPT** — no driver, no `ShardReader` format (`GAP-TRAIN-CPT-HAS-NO-DRIVER`), and none of
  the seven sources with a recipe uses it.
- **Vocabulary remap** — the train-only remap encodes 5 of 184 val rows (row `d9b461ca`).
  Whether `train_ft` can run unremapped is **unverified**; checking it is the first task of
  step 3. If it cannot, keep the val-covering remap (`74dfe7b1`) for this run.
- **8×H100** — nothing here needs it.
- **Adopting RSI-Jev / decider code** — a new dependency; needs a human yes. Neither has rule 3's
  held-out path check, so any use needs pre-filtered data with the filter recorded.

## Human decisions needed

1. A GPU box for step 5 (one GH200 or H100; ~$25 cap).
2. Whether to pull RSI-Jev or decider code in, or keep to `qd_train` (recommended: keep, borrow
   the recipe only).
3. The 300 hand-labelled held-out diffs — the ship gate needs them and they do not exist. Not a
   blocker for training; a blocker for promotion.
4. Replay source for step 5 (general text needs a licensed corpus; self-distillation on the
   base's own answers needs none).

## Timeline

| Day | Work |
| --- | --- |
| D0 (today) | Step 1 zero-shot, step 2 recipe arms launched |
| D1–D2 | Step 3 mutation family + shards; step 4 FT linear control in parallel |
| D2–D3 | Step 5 on one GPU, 3 seeds, + operator-holdout arm |
| D3 | Calibration, report, handoff |

A first real 2B number on the mutation task is ~3 days out, for under $30.
