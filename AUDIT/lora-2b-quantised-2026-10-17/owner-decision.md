# LoRA-2B quantised arm: the owner's decision record (2026-10-17)

Recorded by the Lappi lane `lora2b-audit2-2026-10-17` from the owner's messages in the 2026-10-16/17
session, quoted verbatim below. Fable's ruling that these words answer is
`AUDIT/lora-2b-audit2-2026-10-17/fable-final-ruling.md` (section 3 "The quantised arm", section 6
"Owner-only items"); the four audits it was ruled on sit beside it. Per that ruling (section 3, item 3)
the memory-need waiver is valid as the owner's decision but was recorded only on the ojas board
(ft-a1a4ba4d criterion 5); Lappi rule 10 and the pre-registration's `not_decided_here` require it in
Lappi, in the owner's words, before the quantised arm's sibling pre-registration is written. This file
is that record. Nothing has been trained, quantised or measured under it.

Labels: **[owner]** the owner's words, verbatim; **[Fable]** Fable's ruling text or arithmetic, not the
owner's; **[lane]** what this lane did with them. No words are invented for the owner.

## 1. The owner's words [owner]

In order, verbatim, 2026-10-16/17:

1. "Is lappi v2 reday for trainging on ojas on my mac"
2. "Add these gaps to GitPulse tasks with high priority"
3. "I want to train a quantised version too, so update the cards too"
4. "Do a second deeper audit and fianlize with fable"
5. "procced with fable's advice"

Message 5 is the owner's answer to the ruling's section 6 (nine yes/no items). It adopts Fable's
recommendation wherever the ruling states one.

## 2. What the words resolve to [lane, reading message 5 against section 6]

| Section 6 item | Fable's recommendation [Fable] | Resolved as |
|---|---|---|
| 1. Lappi ojas pin bump 132b46e -> 5b67cdd now, delta report attached | "My recommendation: now." | **YES** (this lane's pin commit; `AUDIT/ojas-pin-5b67cdd-2026-10-17/delta-report.md`) |
| 2. Amendment 1 to `campaign/lora-2b-preregistered.json` | ruled default: the step definition in words; the window binds; N derived from the measured LoRA tokens/s row (ft-2bbced18 criterion 7), cap 400, no floor, fixed before step 0 | **YES**, with Fable's ruled default (this lane's amendment commit) |
| 3. Quantised arm owner-decision record in Lappi | (asked) | **YES**: this file |
| 4. 4-bit: written override of Unsloth's "no QLoRA on Qwen3.5", build q4 if it clears the gate | "My recommendation: no; q4 is measured in the gate regardless." | **NO**: q4 is measured in the gate, not built; no written override exists |
| 5. Sibling pre-registration `campaign/lora-2b-q8-preregistered.json`: approve its content once drafted | (asked) | **OPEN**: approve once drafted; nothing drafted here |
| 6. Extra Mac GPU time for the quantised arm beyond the 2026-10-09 approval | (asked) | **OPEN** |
| 7. Stage 2 (resident q8 storage + dequant-to-scratch kernel in canonical tessl) | "My recommendation: not now." | **NOT NOW** |
| 8. Verify to done the review cards whose runs are recorded | (asked; "the done gate is yours") | **OPEN** |
| 9. peft for the PEFT export round-trip test | (asked) | **OPEN** |

Items 5, 6, 8 and 9 remain the owner's; this lane did not act on them.

## 3. The quantised arm as decided

Message 3 is the request: a quantised version is trained too. Message 5 adopts how Fable scoped it.
What that fixes for the arm [Fable, section 3, adopted by message 5]:

- **The memory-need waiver.** Fable's D3 (2026-10-09) made 8-bit conditional on an mlx quality
  measurement *and* a demonstrated memory need. The owner's request is the owner's spending decision
  and waives the memory-need condition. The quality gate is not waived (section 4 below).
- **q8 only.** The arm is "LoRA over a q8 (MLX affine, group 64) frozen base". 4-bit is not built
  (item 4 above): q4 is measured in the gate because it is cheap and informative, and nothing more.
- **The first quantised runs run over a fake-quant base file.** The bf16 trainer, unchanged, over a
  safetensors file whose 150 wrapped linears are dequantize(quantize(W)) from mlx (q8, group 64, bf16
  scale and bias, mlx version recorded) and whose other tensors are the originals; written by the
  Lappi gate tool for 2B-Base and 0.8B-Base and pinned by sha256. No quantised dtype, kernel or host
  quantiser is on this run's path.
- **Stage 2 is not now.** Resident q8 storage and a dequant-to-bf16-scratch kernel in canonical tessl
  (1-2 weeks), and stage 3 (fused dequant GEMM), start only on a later owner yes after the fake-quant
  reduced 2B run is scored.
- **No quantised run starts before the bf16 reduced run's ledger row exists** (ruling section 1,
  critical path; ft-da2c4ab1's 2026-10-17 ordering criterion).

## 4. The arithmetic the waiver is recorded with [Fable]

From the ruling (section 3, item 1; evidence list, `bf16_param_bytes` of the tape-bf16 arm):

- bf16 wrapped linears at 2B: **2,745,434,112 bytes (2.745 GB)**; the same 150 linears as q8 g64:
  **about 1.46 GB**; the saving is **about 1.29 GB**.
- The LoRA step at 2B is **estimated at 12-14 GB** (unmeasured; the LoRA tokens/s and memory rows
  are ft-2bbced18 criterion 7, the first job on the path).
- **No pre-registered shape needs the saving**: the full-gradient upper bound at T=9,638 is 18.02 GB
  (bench rows `2026-10-11-bf16-frozen-base`, `2026-10-14-bf16-tied-table`), under memguard's 32 GiB
  per-process cap, and a LoRA step is smaller.

So the quantised arm is trained because the owner asked for it, not because memory requires it; the
fake-quant route gives the arm's numerics (does LoRA over the q8 base train and score as the bf16 base
does) at zero kernel work, and gives up the 1.29 GB saving, which nothing needs.

## 5. What stays as it was

- **The quality gate stays as D3 set it** [Fable]: argmax agreement >= 99.0% and mean |delta log p| of
  the gold letter <= 0.02 nats over >= 500 val prompts (never held-out), bf16 vs q8 vs q4, bars written
  before the run, mlx version named. Rule 2: no threshold moves.
- **The bf16 pre-registration is not edited for the quantised arm** [Fable, section 3 item 3]: its
  `base.precision` ("No 8-bit or 4-bit ... not built here") and `not_decided_here` stay true of the
  bf16 arm. The quantised arm gets a **sibling** pre-registration (item 5, open), with family id
  `lora-2b-q8`, `base.format`, the base-file sha256 in every row, and a pooling rule so a reader keyed
  on `optimizer_recipe` alone never pools it with the bf16 rows.
- **Merge target** [Fable]: dequant(Q(W)) + (alpha/r) B A, rounded once to bf16; merging into the
  original bf16 W is refused by base-sha mismatch; release is a dense bf16 tower through qd-export and
  qd-metal unchanged.

## 6. Gap records

`GAP-LORA2B-QUANT-ARM-UNRECORDED-2026-10-17` (the request was in no Lappi file) is resolved by this
file; `GAP-LORA2B-QUANT-GATE-TOOL-MISSING-2026-10-17` (no tool runs the D3 gate or writes the
fake-quant base file) stays open. Both are in `gaps.jsonl`.
