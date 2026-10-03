# Pre-launch GPU experiments on the 2x H100 box (2026-10-03, 23:16-23:37Z)

The human asked at ~23:10Z: "meanwhile optimize the code and GPU and run experiments". Fable's scope
(advisor, ~23:12Z):

- use `tools/perf_step.py` only, not the P2 screen, which a cap would cut to not_run;
- run both GPUs at once;
- stop hard before launch;
- use a separate experiment directory;
- **nothing changes v5**.

**These are THROUGHPUT and MEMORY numbers only, not loss parity.** perf_step drifts weights across
configs inside one process. Its parity mode was not run. No number here is an equivalence claim.

## Setup [V]

**Box.** ubuntu@68.209.74.244, 2x H100 80GB HBM3 (79.18 GiB usable), torch 2.10.0+cu128, cuDNN
9.10.2.

**Code.** A clean clone at 8e6a009, under `/home/ubuntu/exp-2026-10-03`.

**Data.** The FIRST v5 build's shards (`/Users/bharath/qd-campaign/phase4-v5-2026-10-03`, without
`data/heldout` or `shards/train-no-decode`). Throughput does not depend on which corpus.

**Backbone.** Qwen3.5-2B-Base snapshot b1485b2f… (24 layers, sdpa).

**Batches.** 35,403 batch tokens, v5's `--batch-tokens`, at two shapes:
- **W:** widths 9,000-10,240; the 24 picked batches are 9,638 wide.
- **M:** widths 3,000-5,383.

**Runs.**
- **Run 1** (23:16-23:24Z): `off:N` (perf_step's even spread), `+fused`, `+profile`, two rounds.
- **Run 2** (23:27-23:37Z): `skip:6`, the product rule, which is exactly v5's
  `--checkpoint-skip-layers 6`. Configs `+nomask`, `+fused`, `+nomask+fused`, three interleaved
  rounds.
- **Run 2's cuDNN overlay:** the same clone with the padded training attention forced onto
  `SDPBackend.CUDNN_ATTENTION`, **mask kept** (`exp2_patch_cudnn.py`), three rounds.

**Results.** Min-of-N summaries are in `h100-perf-2026-10-03/*.summary.jsonl`. The raw JSONL is in
the ignored `build/v5-h100/exp-2026-10-03/`.

## Results [V]

All values are max pos/s over the rounds that ran. Run 2's rounds agree to within 0.2%. The two
exceptions are M's round 1 in run 1 and in the cuDNN overlay, which were warm-up and are excluded
by the max.

| Config (run 2) | W pos/s | vs skip:6 | M pos/s | vs skip:6 | Peak GiB W / M |
|---|---|---|---|---|---|
| skip:6 (v5 as launched) | 14,122 | — | 18,543 | — | 56.5 / 60.8 |
| skip:6+fused | 14,377 | +1.8% | 18,886 | +1.9% | 56.5 / 60.8 |
| skip:6+nomask | 25,536 | **+80.8%** | 26,679 | **+43.9%** | 53.1 / 58.4 |
| skip:6+nomask+fused | 26,362 | **+86.7%** | 27,410 | **+47.8%** | 53.1 / 58.4 |
| skip:6, cuDNN attention, mask kept | 21,954 | **+55.5%** | 24,638 | **+32.9%** | 57.2 / 61.6 |

**Run 1 (off:N).**

| Config | W | M |
|---|---|---|
| off:6 | 13,411 pos/s, 57.9 GiB | 18,046 pos/s, 64.4 GiB |
| off:10 | 13,895 pos/s (+3.6%), 74.1 GiB | **OOM** at 75.9 GiB |
| off:14 | **OOM** | not run |
| off:all | not run | not run |

So widening selective checkpointing is not available on the 80 GB card at v5's batch.

**Profile** (`off:6+profile`, 3 steps; `h100-perf-2026-10-03/perf-*-off6_profile-top.txt`).

| Share of CUDA time | W | M |
|---|---|---|
| Padded attention, total | **51.2%** | **38.1%** |
| Its backward (`fmha_cutlassB_bf16_aligned_128x64_k65536_sm80`) | 40.95% | 31.37% |
| Its forward (`fmha_cutlassF_bf16_aligned_32x128_gmem_sm80`) | 10.28% | 6.72% |
| `aten::mm` | — | 22.0% |
| `aten::mul` | — | 12.6% |
| `aten::copy_` | — | 7.7% |
| `Optimizer.step#AdamW` | — | 2.8% |

- The attention kernels are SDPA's memory-efficient ones, built for sm80. With an explicit padding
  mask, SDPA does not pick flash.
- CPU self-time is 38% `Command Buffer Full`, meaning the step is GPU-bound.
- The fp32 GEMM's owner, Fable's open item, is **not identified**. The top table has no dtype or
  stack column (`record_shapes=False`). The cuBLASLt `nvjet_*` kernels carry no dtype in their
  names. A stack-recording profile is still needed.

## What it means [I]

1. **v5 is unchanged.** The human answered "Keep v5 as pre-registered (Recommended)" at ~23:26Z
   (`human-answers-2026-10-03-dirty-row.md`). That answer came before the run-2 numbers; it rested
   on the profile's 51%/38%.
2. **On H100 at v5's widths, the attention kernel choice is the step.**
   - Training without the mask gives +44-81% throughput. That exceeds the GH200's +55% at shape B
     and the 20-40% epoch-wide estimate.
   - **cuDNN attention with the mask kept** gives +33-56%. That is the same attended positions with
     different kernel numerics: a Tier-B kernel change, but without no-mask's padding-semantics
     question. That may make its equivalence screen easier to pass than no-mask's, which was
     inconclusive twice.
   - Both are v6 candidates. The order is for Fable and the human.
3. **Fused AdamW** is +1.8-1.9% on H100. It is small, consistent with the ~3% estimate, and stays
   off (C2b).
4. **Rough v6 arithmetic [I].** At a blended +50-60% throughput, v5's projected 7.0 h per seed
   would be about 4.5 h. Across the plan's 74 GPU-hours, that is roughly 25 GPU-hours, or about $100
   at $4.19 per GPU-hour.

## Not done here

- Loss parity: perf_step `--mode parity`, and the P2 screen for either lever.
- The fp32 GEMM owner.
- cuDNN together with `+fused`.
- An epoch-weighted blend over v5's real width histogram.
