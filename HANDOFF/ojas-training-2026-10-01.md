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

**Second round: the CUDA implementation asks** (AskUserQuestion, 2026-10-01, after Fable's ruling in `AUDIT/ojas-training-2026-10-01/fable-cuda-asks.md`):

| Ask | Decision |
| --- | --- |
| CUDA home and trait owner (Fable blocks A and 5) | **Fable's layout.** <ul><li>The CUDA provider starts in the standalone sibling crate `ojas/ojas-qwen35-cuda/`, with its own `[workspace]` and no member edit. It writes no file outside that directory.</li><li>The coordinator merges it into `ojas-cuda`/`ojas-kernels` later.</li><li>The step-provider trait is L-trainer's, in Lappi `crates/qd-train`. Lappi adapter crates implement it for `Qwen35Step` and for the CUDA provider.</li><li>The rule-6 clarification is in `CLAUDE.md`.</li></ul> |
| cudarc and cuBLAS (Fable block B) | **Pin `cuda-12080`, and use cuBLAS.** <ul><li>This overrides Fable's "no cuBLAS" and buys back about 3 days.</li><li>The sibling crate declares cudarc `=0.19.10` with the `cuda-12080` and `cublas` features.</li><li>`ojas-cuda/Cargo.toml:28` is repinned to `cuda-12080` through the coordinator, so there is one pin workspace-wide.</li><li>The bf16-in/f32-out GemmEx combination is still [U]. The rung-0 probe falsifies it, and the hand-written FFMA ExactF32 tier is the fallback.</li></ul> |
| CUDA target (Fable block C) | **The parity ladder first:** rungs a–d, about 40 days [I]. <ul><li>GDN: port tessl's scan now; it is the on-device oracle. Build fla's chunked form only if the trigger, written before the measurement, fires: K2(i) alone exceeds PyTorch's whole step (1.2–1.9 s per 4×8K) after batching.</li><li>The f64 host oracle uses the published rule, and its fixtures are named `published`.</li><li>Campaign speed (+29–35 days) is decided after rung d.</li><li>Calibration point: M0+M1, about 9 days. If it runs past 2×, re-cut the plan before M2.</li></ul> |
| GH200 time (Fable block D) | **No insert into the post-F queue.** <ul><li>The ~5 min rung-0 probe (≈$0.08, `flock gpu.lock`) is the first item after post-F item 10, or runs on the next rental.</li><li>The lead may run three read-only commands on the box now, with no lock and no GPU: `nvidia-smi | head -4`, an `ls` of the venv's `nvidia/{cublas,cuda_nvrtc}/lib`, and `ls /usr/local/cuda/lib64 | grep -E 'nvrtc|cublas'`.</li></ul> |

## Coordination

- **ojas coordinator session** ("Rust/Go ML packages alternative to PyTorch"):
  - It holds `ojas-core/src/{backend.rs,lib.rs}`, README, `docs/{status,op-coverage,pytorch-parity-plan,framework-design}.md`, `.gitignore`, and the bench lane's `ojas-metal/{benches,examples}`, `ojas-wgpu/examples`, `bench/` and `docs/bench-gpu-vs-torch.md`.
  - The ojas-cpu session ("Ojas performance optimization audit") owns `ojas-cpu`.
  - **No Lappi lane edits ojas outside the agreed split.**
    - The coordinator approved the additive `ojas-qwen35` crate: "yes, you own it as an additive crate". It is provider-level and not an ojas `Backend`.
    - Off-limits: `ojas-core`, `ojas-cpu`, `ojas-metal`, `ojas-wgpu` and `ojas-kernels`. The one exception is new files under `ojas-kernels/src/cuda/`.
    - `Cargo.toml` members, `Cargo.lock` and `ojas-cuda/Cargo.toml` change only through the coordinator, as exact diffs.
    - The coordinator also relayed CUDA grants, attributed to the users. Those grants are **not** this user's approval; they are put to the user in the CUDA decision round. Fable's ruling is in `AUDIT/ojas-training-2026-10-01/fable-cuda-asks.md`.
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
| L-ojas-qwen35 | `ojas/ojas-qwen35/` (agent, untracked, standalone `[workspace]`) | the tessl Qwen3.5 step as an ojas provider; not a `Backend` |
| L-cuda-M0 | `ojas/ojas-qwen35-cuda/` minus the oracle's paths (agent) | `CudaRuntime`, `CudaBuffer`, the NVRTC cache, the K0 plumbing kernels, K1 GEMM via cuBLAS, and the rung-0 smoke binary, cross-built for aarch64 |
| L-cuda-oracle | `ojas/ojas-qwen35-cuda/tests/{reference,fixtures}/**` and `tests/reference_*.rs` (agent) | float64 host references: GDN at the published rule (a port of tessl's `tests/common/gdn_train.rs`, T ∈ {1, 63, 64, 65, 130}), then K3/K4/K6/K7/K9/K10, then K11 |

**Lane status at 2026-10-01 ~23:30 UTC:**

- **L-data: merged** at `4a5ebda`.
  - The held-out door works: every held-out case is refused with zero bytes read.
  - Batch order, consumed digest and supervision are byte-identical to Python.
  - The v4 counts match row d96409bd.
- **L-head: merged** at `d5dfc3d`.
  - The span head passes the Amendment 1 gate on all 9 cases; row `8e447aea` (completed, quick) supersedes `6d6ca078`.
  - qd-train on main: 28 unit + 20 door + 10 parity + 14 span-head tests pass; 2 opt-in v4 tests are ignored.
- **L-scorer: merged** at `1651fdf`.
- **L-ojas-qwen35: done.**
  - 23 CPU tests pass.
  - The coordinator applied both diffs: `ojas-qwen35` is now a workspace member, and `ojas-cuda` is pinned to `cuda-12080`.
  - **Its GPU parity tests ran in a Mac window the coordinator cleared, at 23:16–23:17 UTC by the logs' mtimes.** Conditions: AC power, memory 84–87% free. Logs: `AUDIT/ojas-training-2026-10-01/ojas-qwen35-gpu-*.log`.
    - `gpu_tiny`: 5/5 pass. Loss 5.82468344 vs transformers 5.82468319; bf16 operands 5.82425679.
    - `gpu_name_map`: 1/1 passes.
    - `gpu_real_2b` **fails, deterministically (2/2 runs).** The forward matches transformers (3.81034020 vs 3.81016684, rel 4.55e-5, 128 tokens). Then `read_gradients` on `embed_tokens.weight` hits a Metal 4 command buffer fault.
  - The fault is `GAP-OJAS-QWEN35-2B-READ-GRADIENTS-METAL-FAULT-2026-10-01`. L-ojas-qwen35 has been resumed to diagnose it. Until it is fixed, no 2B gradient or state read-back from the provider stands.
- **L-trainer and L-oracle: running.**
  - L-trainer reports one blocker. tessl has a single lr, so F's two-group recipe cannot run until tessl's per-entry `lr_scale` lands.
  - The trainer refuses any lr_scale ≠ 1.0, with no workaround.
- **L-tessl** is the user's pending chip "Add per-entry lr_scale and mrope handling to tessl". It needs a session rooted in tessl, because this harness blocks git in other repos. **It is the one blocker for rung (b)'s two-group arm and for rung (d).**
- **L-cuda-M0 and L-cuda-oracle: running.**

## Invariants for every lane

- Every row written goes to `ledger/mac-ojas-*.jsonl` with `quick=True` and a `quick_reason` (rule 8). It never shares a ledger, a `.done` marker or a box path with the campaign.
- Kernels live in canonical tessl (rule 6), and GDN fixtures say `published` (rule 9).
- Held-out data never reaches the Rust reader (rule 3).
- New code is Rust; Python appears only as reference oracles. No new dependencies.
- The PyTorch GH200 campaign (run F and the post-F queue) is untouched.

## Window (c) on the Mac GPU (~22:33–22:37 UTC by the logs' mtimes; released to the ojas coordinator after it)

Both results are logs only, so they cannot be cited as rows yet. The bench has no shard header, which a protocol needs. The quick `smoke`/`throughput` rows are written once L-trainer's ledger writer lands.

- **Rung (a), tessl's training GPU suites** (`AUDIT/ojas-training-2026-10-01/rung-a-tessl-gpu-suites.log`):
  - attn_train 4, cross_entropy 6, gdn_fixtures 11, gdn_train 5, qwen35_adamw 8, qwen35_bwd 20, qwen35_train 8;
  - 62 passed, 0 failed, 3 ignored (the real-2B oracle tests); 25 s wall.
  - The run was on canonical tessl's working tree, which has ~20 uncommitted files that are not ours (the ojas coordinator says they are not its files either).
- **8K memory and throughput** (`mac-8k-train-step-bench.log`): `bench_qwen35_train --bf16 --step=8192` on the real Qwen3.5-2B snapshot `b1485b2f…`:
  - `train_step` at T=8192 takes 18.141 s (452 tok/s), loss 14.1156;
  - peak memory footprint 30.78 GB. Adding 16 GB for AdamW's moments gives ≈47 GB of the M5 Pro's 64 GB, so it fits. The Metal working-set ceiling has not been read.
  - **Consequence:** at 452 tok/s, v4's 323.07M positions take ≈198 h per seed, so rung (d)'s "10% of steps" is ≈20 h, not Fable's 3–6 h, which was sized on J1's 74.7M. Rung (d) goes back to Fable for re-sizing. The user's approval covers 3–6 h.

## Rung (d), re-sized by Fable (2026-10-01)

Fable's full text, with every citation, is in `AUDIT/ojas-training-2026-10-01/fable-rung-d-resize.md`. Its gaps are `GAP-OJAS-RUNGD-*`.

**Sizing.**
- Fable's "10% of steps ≈ 3–6 h" was sized on J1's 74.7M positions.
- On v4 (row d96409bd: 306,926,895 unpadded positions, 9,683 steps), the approved hours buy ≈2% of the steps. 10% would be ≈20 h.
- **Rung (d) is therefore 200 optimizer steps of F's recipe, unchanged.** That is 6.34M tokens: ≈3.9 h at the 452 tok/s floor, ≈2.9 h inferred.
- **Hard cap 21,600 s, with auto-terminate.** The eta rule is checked at step 10. N is fixed before step 0 and never raised on the day.
- **This fits the user's approved 3–6 h.**

**Recipe and data.**
- **Batches:** the first 200 batches of `reader.batches(batch_tokens=35403, seed=20260919, epoch=0)`. That is F's own order: the epoch arm plans with the protocol seed, so all F seeds share it.
- **Schedule:** `LRSchedule(peak_lr=1e-5, total_steps=200, warmup_steps=10, min_lr=1e-6)`.
- **AdamW (F's):** eps 1e-8 and weight_decay 0.01 on **every** parameter (`optim.py:337-338,403`).
  - **Not** tessl's default exclusions.
  - The per-group lr scale is 0.1 for decoder layers 0–7 only (`--lower-layers-n 8 --lower-layers-lr-scale 0.1`, `campaign/f-v4-preregistered.json:6`). This needs tessl's `lr_scale`.
  - **Everything else trains at 1.0** (`python/qd_train/optim.py:288-316`, `layerwise_param_groups`): `embed_tokens` and the tied lm_head, the final norm and the span head.
  - An earlier draft of this line put the embeddings in the 0.1 group. That was the lead's error, which L-trainer caught by reading the code.
  - F's optimizer is `--optimizer master`, i.e. fp32 masters and fp32 moments.
  - tessl's per-entry wd vector is f32, so it holds f32(0.01) = 0.009999999776482582. That is 2.2e-8 relative to torch's 0.01. It is recorded, not approximated around.
- **Span head:** initialised from `span_head_init-seed0.safetensors` (L-oracle's `QwenDecisionStep(seed=0)`).

**Preconditions.**
- tessl's working set (0.9 × `recommendedMaxWorkingSetSize`) is ≥ 47 GB. If not, the host-AdamW contingency is costed first, and the moments are never bf16.
- The Mac is on AC power.
- GPU exclusivity is agreed.
- Rungs a–c are green.
- `lr_scale` has landed.
- The schedule bit-equality and consumed-digest parity tests are green.
- Rule 3 is enforced: `shards/train` only, opened through the door.

**Outputs of the Mac window.**
- the bf16 export (4 GB);
- the f32 masters (8 GB);
- the `ft` row in `ledger/mac-ojas-rung-d-<date>.jsonl`, quick ("1 seed, 200 of 9,683 steps, Metal/tessl trainer");
- a load-and-decode smoke run on ≥ 24 val prompts. It claims no number.

**Scoring on the GH200** after post-F item 10. The Mac's torch-MPS scorer is ≈18× slower. This is the one deviation from ask 7's "on the Mac".
- **GH200 arms** (quick, ≈$3.5–4.5, ask 5):
  - **T-bf16:** F's flags plus `--max-steps 200 --train-attention-mask none --deterministic`.
  - **T-fp32:** needs an additive `--train-dtype fp32`, on the non-master AdamW whose eps/wd must be read first.
- **Additive flags:** `--max-steps` and `--train-dtype` enter the recipe only when on.

**Pass criteria** (pre-registered, quick):
0. The consumed digest and the head-init digest are identical across arms. Otherwise the result is `digest_mismatch`.
1. Per-step loss, tessl vs T-fp32: |Δ| ≤ 0.02 nats for steps 0–49, and EMA(0.9) ≤ 0.05 for steps 50–199. Loose bound against T-bf16: 0.05 / 0.10.
2. The grad-norm |Δ| is report-only.
3. Any non-finite loss kills that arm, with no comparison.
4. Per parameter group, over the f32 masters:
   - cos(Δ_tessl, Δ_fp32) ≥ cos(Δ_bf16, Δ_fp32);
   - the norm-ratio bound with +0.01 slack;
   - a sanity floor of cos ≥ 0.9.
5. The export loads in `--score-checkpoint`.
6. Val deltas are ≤ F's across-seed range (report-only until F's rows exist), and the gate verdicts are the same.

**What a green (d) may claim:** "a Rust trainer over canonical tessl trained Lappi for 200 steps of F's recipe on the M5 Pro; its per-step losses and master deltas track torch fp32 within the pre-registered bounds; the campaign scorer scored its export at X (rows …), quick". It may not claim parity with F, a candidate, or any gate verdict.

## CUDA scoping (L-cuda, merged `226f74f`)

- **Doc:** `AUDIT/ojas-training-2026-10-01/cuda-backend-scoping.md`, plus seven `GAP-L-CUDA-*` records.
- **Recommendation:**
  - Design (B) first: a whole-step Qwen3.5 provider on CUDA, the counterpart of `ojas-qwen35`.
  - Design (A), a general CUDA `Backend` over the trait plus T1–T6, is the long-term target. It still lacks about ten Qwen ops, and those are trait edits the coordinator owns.
- **Effort:**
  - About 40 specialist days to reach the parity ladder (inferred).
  - At that point the faithful GDN port is estimated at 5–10× slower than PyTorch, because tessl's GDN kernel walks tokens one at a time.
  - Campaign speed, which needs fla's chunked GDN on CUDA, costs another 29–35 days.
- **GPU time:** the post-F queue has no gaps. Only the first two stages (a toolchain smoke test and op-level checks, about 25 min, about $1) are short enough to insert.
- **Build status:** ojas type-checks and links for aarch64 Linux, but has never run there.
- **Human asks:**
  1. Place the kernels and the provider in ojas.
  2. Insert the ~25 min item into the queue, or wait until after item 10.
  3. Enable cudarc's `cuda` feature and repin it to `cuda-12080`. That is a coordinator-owned file.
  4. Enable cudarc's `cublas` feature.
  5. Settle ojas file ownership and the trait owner.
  6. Choose the target: the parity ladder now, or campaign speed.
- The doc was sent to the ojas coordinator for `ojas/docs`.

## Open

- The coordinator's answer on the `ojas-qwen35` crate and the ojas file split.
- Whether any of tessl's ~20 dirty files belong to a live session.
- The 8K-token memory footprint on the M5 Pro (`GAP-OJAS-ADVICE-MAC-8K-FOOTPRINT-UNMEASURED-2026-10-01`).

## First command for the next lane

`cat AUDIT/ojas-training-2026-10-01/fable-advice.md`, then `git log --oneline main -- crates/qd-train` to see which lanes have merged.
