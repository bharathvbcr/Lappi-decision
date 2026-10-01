# Scoping: a fast deterministic attention backward (Fable Round J, item C). Scope only, no build

**Status:** scoping only. Nothing here is built, and no GPU time goes to it before J5′ (Fable J).
The one GPU measurement below is the ~2-minute no-mask probe the lead approved for the end of perf
session 2. The human picks a route. Evidence labels: **[V]** measured or read, **[I]** inferred,
**[U]** unverified.

## The cost being scoped

- Under `--deterministic`, shape B (4 × ≤8,441, `--batch-tokens 35403`, qd-lane4 7ed66d7) runs at
  **9,983 pos/s, 3.19 s/step** (session 1, per-step sync). Through the production loop
  (`perf_parity` → `real_ft_run._train`, session 2 arm B-base) it runs at **9,326 pos/s**. [V]
  `AUDIT/perf-train-step-2026-10-01/session1.jsonl`, `session2/parity.jsonl`.
- On default kernels the same shape runs at **15,280 pos/s, 1.934 s/step** (session 2, overlapped,
  min of 3 interleaved rounds). [V] `session2/bench.jsonl`.
- In the det profile at B, `aten::_efficient_attention_backward` (kernel
  `fmha_cutlassB_bf16_aligned_128x64_k65536_sm80`, PyTorch's mem-efficient SDPA) takes **1.69 s/step,
  53% of GPU time**. [V] `session1-B-off0_det_profile_memhist_n50-top.txt`.
- F runs on default kernels (Fable J), so this cost lands on the `--deterministic` Tier-A goldens
  and on any later det run, not on F.

## Where the nondeterminism is, and why det is slow

- **Why mem-efficient and not flash.** `QwenDecisionStep.hidden` passes a padding mask built from
  `Batch.lengths`, so SDPA receives an explicit mask. PyTorch's flash backend refuses any explicit
  `attn_mask` (`check_for_attn_mask`, `sdp_utils.cpp`). [V] PyTorch `main` source; [V] on the box,
  because removing the mask switches the kernel to flash (profile below).
- **Mem-efficient's nondeterminism.** Its backward splits over key blocks (`num_splits_key > 1`) and
  accumulates the partial gradients in no fixed order. Strict deterministic mode forces
  `num_splits_key <= 1`, which serialises the backward over keys. [V] source text; [I] that this is
  the whole 1.69 s.
- **Flash's backward under det is not refused.** It runs FA2 with `deterministic=true`, which fixes
  the order of the dQ accumulation. [V] source text. Its speed is now measured (route d).
- **Right padding.** Shard batches are right-padded and attention is causal, so no real position
  attends to a pad. On CPU, the tiny tower without the mask gives **bit-identical hidden states at
  real positions**, but **gradients that are not bit-identical** (max |Δ| 4.8e-7, `embed_tokens`).
  Dropping the mask therefore changes numerics: it is Tier B. [V]
  `AUDIT/perf-train-step-2026-10-01/nomask_cpu.py`, CPU only.

## Route d, measured (perf session 2, last step, speed only, one round, not an interleaved A/B)

| shape B, off:0 | mask | det | pos/s | s/step (window mean) | attention backward kernel |
| --- | --- | --- | --- | --- | --- |
| bench (3 rounds) | padding mask | off | 15,280 | 1.934 | mem-efficient |
| probe | none (`is_causal`) | off | **25,703** | **1.204** | flash |
| probe | none (`is_causal`) | **on** | **23,208** | **1.334** | `pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` |

[V] `session2/probe-nomask.jsonl` and `probe-nomask-B-off0_det_nomask_profile-top.txt`:
- `aten::_flash_attention_backward` took 427 ms over 3 profiled steps, about **0.14 s/step**,
  against mem-efficient det's 1.69 s/step.
- The det run raised no determinism error.
- Peak memory is unchanged at 42.2 GiB.

[U] Two things are not established:
- Whether two det no-mask runs are bit-identical. No repeat was run.
- How much of the default-kernel gain carries over to F's narrower buckets.

## What a custom kernel would still buy over route d

- **The ceiling is attention's share of the step, and after d that share is small.**
  - In the det no-mask profile, flash backward is 9.98% of self-CUDA time and flash forward
    3.41%. Together that is about **13% of the step, roughly 0.19 s of about 1.4 s**. [V] profile
    above.
  - The step's remaining time goes to cuBLAS GEMMs (`aten::mm`, 31%), elementwise `mul`/`copy_`
    (25%), det-mode `fill_` (6%), and fla's GDN (about 7%).
  - So a perfect attention kernel recovers at most ~13% under det. A realistic one recovers less.
  - The default-kernel share is unmeasured: there is no default no-mask profile. [U]
- **The realistic gain is Hopper-native code.** PyTorch's FA2 is sm80 code running on sm90. A
  wgmma/TMA backward in the FlashAttention-3 style is where a custom kernel could beat it. At a
  ~13% share, even a 2× faster backward saves about 0.07 s/step, or ~5%. [I] Specialist work
  (route a); not worth it on these numbers.
- **What d structurally cannot do:**
  - **Packed or varlen batches with no pad compute.** d still runs attention over the pad tail of
    every row. How much that buys depends on F-like padding waste, which these runs did not
    measure at the step level. [U] fla `parallel_attn` with `cu_seqlens` (route e) covers this
    without a custom kernel, if its layout fits.
  - **A batch layout that needs a non-causal or non-tail mask.** None is planned.
- **A larger lever than any attention kernel.** Elementwise fusion of `mul`/`copy_`/`fill_` is about
  31% of det GPU time. That is compiler territory, and compile was dropped for F (Fable J). Det's
  `fill_` (`fill_uninitialized_memory`) is a separate Tier-A cleanup candidate. [I]

## Routes, cheapest first

| route | what | language rule | dependency | effort | risk |
| --- | --- | --- | --- | --- | --- |
| **d. no mask → SDPA flash** | Call the tower with `attention_mask=None` (`is_causal`) for right-padded batches, as an opt-in flag that records itself in the recipe. | no new kernel code | none | small | Tier B (the kernel changes, and grads differ at ~1e-7). **Measured fast**: det 1.33 s/step, default 1.20, both at B. |
| **e. fla `parallel_attn`** | fla 0.5.2 ships a Triton FlashAttention with separate dq and dkv backward kernels and **no atomics**. It supports GQA (`HQ`, `G`) and head dim ≤ 256. It would plug in as a transformers attention function. | installed third-party kernel, like the fla GDN kernels F runs | none (fla installed) | small to medium | Tier B. [I] deterministic, from reading the source (no `tl.atomic`). [U] performance with head_dim-256 tiles. [U] whether its varlen path (`cu_seqlens`) fits our padded batch layout without a repack. |
| **c. FlexAttention under torch.compile** | `flex_attention` with a causal + length block mask. Its backward template runs one dQ loop per query block and one dK/dV loop per KV block. | compiler-generated Triton | none | medium | Tier B. [I] no `tl.atomic` for mask-only use. [U] determinism at runtime. compile was dropped for F (Fable J). |
| **b. own Triton FA backward** | a two-pass backward (dK/dV per KV block, dQ per query block), deterministic by construction | a Python DSL, so a reference oracle only under the language rule | none (triton 3.7.1 installed) | medium | Tier B. Duplicates d and e. |
| **a. Rust kernel via cudarc / ojas** | a CUDA C++ / PTX two-pass backward, compiled with nvrtc and launched on torch's stream, behind a C-ABI cdylib → ctypes → `torch.autograd.Function` | consistent with the language rule | **cudarc 0.19.10 is a new dependency and needs the human's yes**. ojas has **zero commits**, and its `ojas-cuda` is only an affine smoke test. | large (days). A correct, fast FA backward at head_dim 256 on sm90 is specialist work. | Highest. Tier B, and we would own a kernel no one else maintains. |

## Recommendation (for the human to accept or reject)

1. **No custom kernel.** Route d already gives a fast deterministic backward: det at B goes from
   9,326 to 23,208 pos/s, using PyTorch's own FA2 kernel. Default kernels also speed up, by about
   1.6×. That makes route d the largest single throughput lever measured in this lane, for F's
   successors.
2. Before any lane uses d, it needs Fable's Tier-B screen: P2 against ≥3 baseline repeats, plus a
   phase-3 seed-0 outcome run. It also needs a repeat check that det no-mask is bit-identical to
   itself, so that Tier-A goldens can move onto it. All of this is GPU work, so it waits until
   after J5′.
3. Route e is the fallback if d fails Tier B or a later shape needs a mask (for example, packed
   varlen). Route a only if both d and e fail, and only after the human says yes to cudarc.
