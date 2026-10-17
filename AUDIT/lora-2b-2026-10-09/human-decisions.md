# LoRA-2B: the human's answers to Fable's asks (2026-10-09)

Recorded by the ojas fine-tune audit session (ojas-7c) from the human's answers in that session. The
ruling they answer is `fable-ft-decisions.md` in this folder. The ojas side of the track is on the ojas
GitPulse board under `gp-ft-*` (umbrella `gp-ft-end-to-end-acceptance`). This file records decisions only;
nothing has been trained or measured under them yet.

| Ask (Fable) | The human's answer |
|---|---|
| Q0 + D6: framing and pre-registration | **Approved Fable's table.** LoRA-2B is a new, separately pre-registered recipe family compared with v5 at the gates only. It uses Lappi's data shape (v5 width, rows up to 9,638 tokens; 35,403-token budget; grad_accum 1; the 8K needle gate), not seq 2048 or batch 8x4. LoRA r=16 alpha=16 dropout 0 on all linears including GDN `linear_attn`; lr 2e-4 (D17 and span head 1e-4); weight decay 0.01; Lappi's `LRSchedule` (warmup steps//20, cosine to 0); no 0.1x layer split; `optimizer_recipe = "lora"`. The pre-registration file (`campaign/lora-2b-preregistered.json`) is still to be written, before any smoke result is read. |
| D2: head | **17-row readout + trainable delta.** 17-way CE over the answer rows with `D17 [17,H]` (init 0, f32, lr 1e-4); full-vocab letter CE logged report-only. No Clef head for this model. |
| D4 + D8: base, downloads, Mac GPU time | **2B-Base for the run, 0.8B-Base for the smoke.** Both downloads approved. GPU plan approved: the gating run on the local 2B instruct weights (~1 h), memory rows at T=2048 and T=9,638 (~1 h), the 0.8B smoke with its torch arm (~2 h), then a reduced 2B run inside the approved 3-6 h. Every Mac row is `quick` and cannot promote. 4B stays for a box. |
| D8: trainer owner | **Move into `ojas-train` now.** `qd-train`'s loop, schedule, groups, host AdamW and `StepProvider` move to ojas; the Tape Qwen3.5 tower implements `StepProvider`; Lappi keeps the rule-3 door, shards, supervision, objective, span head, ledger and export. Per Fable's reversal condition, the move waits for any live lane holding `crates/qd-train`. |

Decided by Fable as its own (reported, not asked): D1 Tape path for LoRA with the provider kept full-FT
plus correctness fixes and a `Precision::Bf16` passthrough; D3 bf16 frozen base with 8/4-bit not built
unless the mlx measurement clears its bar and a memory need exists; D5 merged dense bf16 release with
seed averaging on merged weights only; D7 plain-torch LoRA reference arm in `python/qd_train`, no new
dependency (peft only if the human later wants the PEFT export round-trip test; not asked yet).
