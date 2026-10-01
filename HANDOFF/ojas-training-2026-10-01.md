# HANDOFF — training Lappi without PyTorch (ojas / tessl), 2026-10-01

Lane: ojas-training (lead). Plan: `AUDIT/ojas-training-2026-10-01/fable-advice.md` (Fable, read-only
advice; gap table Q1, parity ladder Q3, lanes Q5). Gap records: `GAP-OJAS-ADVICE-*` in
`gaps.jsonl` (`38a4061`).

## The request and what it became

- **The user's request (2026-10-01):** "Replace pytorch with ojas, build kernel and features needed on the fly and bring parity with pyTorch. I want to try to train or finetune lappi using ojas. ask fable for advice". Then: "Spin multiple agents to close ojas gap".
- **Fable's finding:** ojas cannot express Qwen3.5 today. It lacks:
  - GDN;
  - attention at head_dim 256 / GQA (it is capped at 128);
  - Qwen's (1+w) RMSNorm;
  - partial RoPE;
  - a chunked CE over 248K vocab;
  - bf16 compute;
  - a CUDA `Backend`.
- **What already exists:** canonical tessl has the whole Qwen3.5-2B training step on Metal, forward and backward, checked against transformers' autograd.
- **So the path is:** tessl's kernels + a Rust trainer on the M5 Pro, climbing a quick-only parity ladder (a: op level, b: tiny tower, c: real 2B, d: reduced-schedule real fine-tune), while the GH200 campaign stays on PyTorch.

## The user's decisions (AskUserQuestion, 2026-10-01)

| Ask | Decision |
| --- | --- |
| Trainer placement (Fable ask 2) | **Lappi `crates/qd-train` now, ojas later.** It is built over a model-generic step-provider trait; tessl's `Qwen35Model` is the first provider, and ojas's Tape model a later one. Until the trainer moves into ojas, results read "a Rust trainer over tessl", not "ojas". |
| Mac GPU (ask 4) | **Approved:** the ~3–6 h rung (d) reduced-schedule run (F's recipe, seed 0, 10% of the schedule), once rungs a–c pass, in an agreed GPU window. Full-schedule or 3-seed Mac runs need a separate yes. |
| Scorer input (ask 7) | **Approved:** an additive `real_ft_run --score-checkpoint` change to score a Metal-trained artifact. It changes no gate, threshold or seed rule. |
| CUDA (ask 6) | **Start scoping now.** This is CPU-only design work with no GPU time. Fable had advised deferring it. |
| GH200 comparison arms (ask 5) | Informational: a few dollars, run after post-F item 10, under rule 4's $20. |

## Coordination

- **ojas coordinator session** ("Rust/Go ML packages alternative to PyTorch"):
  - It holds `ojas-core/src/{backend.rs,lib.rs}`, README, `docs/{status,op-coverage,pytorch-parity-plan,framework-design}.md`, `.gitignore`, and the bench lane's `ojas-metal/{benches,examples}`, `ojas-wgpu/examples`, `bench/` and `docs/bench-gpu-vs-torch.md`.
  - The ojas-cpu session ("Ojas performance optimization audit") owns `ojas-cpu`.
  - **No Lappi lane edits ojas** until a file split is agreed. An additive `ojas-qwen35` crate was proposed to the coordinator; its answer is pending.
- **Mac GPU queue** (agreed with the coordinator): (a) its Metal/wgpu-vs-torch bench → (b) the ojas-cpu session's GPU tests → (c) **the Lappi window**: tessl's rung-(a) GPU suites and the 8K memory measurement → (d) the coordinator's optimization round. Nothing Lappi-side touches the GPU before (c).
- **Trait:** the coordinator agrees the seam sits above both providers. Neither session defines it in ojas until the users decide.

## Lanes

| Lane | Where | Owns |
| --- | --- | --- |
| L-data | Lappi `crates/qd-train` (agent) | the rule-3 door first (fail-first: a held-out path raises before any token byte is read); the npy and stored-zip readers; bucketed batches with byte-identical order and consumed digest vs Python; the `ft_supervision` and `plan_span_batch` ports |
| L-head | `crates/qd-train/src/span_head.rs` (agent) | the f32 span head, forward and backward, with exact parity to the Python head and the `noul_row` layout |
| L-oracle | `tools/qd_train_oracle_tiny.py` (agent) | the rung (b) fixtures in tessl's tiny shape family on CPU torch; F's optimizer spec; the ft-row and manifest contract |
| L-trainer | `crates/qd-train` + new `crates/qd-train-metal` (agent) | the step-provider trait, schedule, loop, clip, ledger row, export; the tessl provider and bin |
| L-scorer | `tools/real_ft_run.py` (agent) | the additive Metal-artifact scoring (ask 7) |
| L-cuda | `AUDIT/ojas-training-2026-10-01/cuda-backend-scoping.md` (agent) | the CUDA backend design and scoping for the GH200 (sm_90, CUDA 12.8, aarch64), no code |
| L-tessl | canonical tessl (its own session, spawned by the lead) | per-entry AdamW `lr_scale`; `mrope_*` parse and refuse; the rung (a) GPU command list. It runs no GPU work until window (c). |

## Invariants for every lane

- Every row written goes to `ledger/mac-ojas-*.jsonl` with `quick=True` and a `quick_reason` (rule 8). It never shares a ledger, a `.done` marker or a box path with the campaign.
- Kernels live in canonical tessl (rule 6), and GDN fixtures say `published` (rule 9).
- Held-out data never reaches the Rust reader (rule 3).
- New code is Rust; Python appears only as reference oracles. No new dependencies.
- The PyTorch GH200 campaign (run F and the post-F queue) is untouched.

## Window (c) on the Mac GPU (22:43–22:53 UTC; released to the ojas coordinator after it)

Both results are logs only, so they cannot be cited as rows yet. The bench has no shard header, which a protocol needs. The quick `smoke`/`throughput` rows are written once L-trainer's ledger writer lands.

- **Rung (a), tessl's training GPU suites** (`AUDIT/ojas-training-2026-10-01/rung-a-tessl-gpu-suites.log`):
  - attn_train 4, cross_entropy 6, gdn_fixtures 11, gdn_train 5, qwen35_adamw 8, qwen35_bwd 20, qwen35_train 8;
  - 62 passed, 0 failed, 3 ignored (the real-2B oracle tests); 25 s wall.
  - The run was on canonical tessl's working tree, which has ~20 uncommitted files that are not ours (the ojas coordinator says they are not its files either).
- **8K memory and throughput** (`mac-8k-train-step-bench.log`): `bench_qwen35_train --bf16 --step=8192` on the real Qwen3.5-2B snapshot `b1485b2f…`:
  - `train_step` at T=8192 takes 18.141 s (452 tok/s), loss 14.1156;
  - peak memory footprint 30.78 GB. Adding 16 GB for AdamW's moments gives ≈47 GB of the M5 Pro's 64 GB, so it fits. The Metal working-set ceiling has not been read.
  - **Consequence:** at 452 tok/s, v4's 323.07M positions take ≈198 h per seed, so rung (d)'s "10% of steps" is ≈20 h, not Fable's 3–6 h, which was sized on J1's 74.7M. Rung (d) goes back to Fable for re-sizing. The user's approval covers 3–6 h.

## Open

- The coordinator's answer on the `ojas-qwen35` crate and the ojas file split.
- Whether any of tessl's ~20 dirty files belong to a live session.
- The 8K-token memory footprint on the M5 Pro (`GAP-OJAS-ADVICE-MAC-8K-FOOTPRINT-UNMEASURED-2026-10-01`).

## First command for the next lane

`cat AUDIT/ojas-training-2026-10-01/fable-advice.md`, then `git log --oneline main -- crates/qd-train` to see which lanes have merged.
