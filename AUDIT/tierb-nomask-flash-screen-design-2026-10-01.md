# Tier-B screen design: training without the padding mask (SDPA flash). DRAFT for Fable

**Status:** a draft for the lead to take to Fable. Nothing here is built or queued.
- The GPU parts are sized to run **after F**.
- Fable J said "no GPU time until after J5′" for attention-backward work. Whether this screen
  waits for J5′ too is Fable's call (open question 1).

Evidence labels: **[V]** measured or read, **[I]** inferred, **[U]** unverified.

## The candidate

Train with `attention_mask=None` (`is_causal`). SDPA then takes FA2 instead of mem-efficient.

Measured at shape B (4 × ≤8,441, bt 35,403), one probe round [V]
(`AUDIT/perf-train-step-2026-10-01/session2/probe-nomask*`):

| kernels | pos/s with mask | pos/s without mask | change |
| --- | --- | --- | --- |
| default | 15,280 | 25,703 | about 1.6× |
| det | 9,326 | 23,208 | about 2.5× |

- Peak memory is unchanged at 42.2 GiB.
- It is Tier B because the attention kernel changes. On the CPU tiny tower, hidden states at real
  positions are bit-identical, but the gradients are not (max |Δ| 4.8e-7). [V]
  `nomask_cpu.py`, CPU only.
- It is **not** Tier C: no GDN kernel changes. fla's chunk kernels and `causal_conv1d` stay as
  they are.

Why dropping the mask is admissible at all:
- Shard batches are right-padded, and every real token sits in `[0, lengths[r])`.
- Attention and the GDN recurrence are both causal, so no real position reads a pad. [V] for
  SDPA and the torch GDN reference on CPU; [I] for fla's chunk kernels on CUDA (the chunked solve
  is lower-triangular).
- `QwenDecisionStep.hidden` is **shared by training and scoring**. Its callers are
  `accumulate`/`accumulate_span`, plus `real_ft_run._decode`, `_evaluate` and
  `replay._letter_logits`. [V] `rg -uu`; DevMap listed the scoring callers only as unresolved
  namesakes (GAP-TRAINSTEP-PERF-WORKTREE-HAS-NO-DEVMAP-STORE-GITPULSE-UNTRUSTED).
- So the switch is **training-only**. Scoring keeps the mask, and the outcome run's eval rows
  change for exactly one reason: the trained weights.

## Code it needs first (CPU only, about 1 h, my files; tests fail before the change)

1. `QwenDecisionStep(train_attention_mask="padding" | "none")`, default `"padding"`.
   - Only `accumulate`/`accumulate_span` pass `None`. `hidden()` keeps the mask for every scoring
     caller.
   - On CUDA, the `"none"` forward runs inside
     `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])`. That way an ineligible shape
     **raises** instead of quietly falling back to mem-efficient: fail closed. I'll check the
     current torch docs for this API before using it.
   - On CPU and MPS there is no flash backend, so the path records `math` rather than claiming
     flash.
2. Flag `--train-attention-mask none`.
   - Recipe key `train_attention_mask: "none"`, set only when on, and added to
     `RECIPE_PIECE_KEYS`. Every scored row then mirrors it (the 0b95adb pattern).
   - Recorded in `train.path`.
   - Refused without `--real-backbone`.
3. Tests:
   - real-position hidden states bit-identical, mask vs none (tiny tower, CPU);
   - scoring still masked;
   - the recipe key appears only when on, and is mirrored;
   - the argv refusal;
   - it composes with `--checkpoint-skip-layers` (as in `8b6e858`).

## (i) P2 screen: two GPU sessions, each ≤ 20 min under `flock gpu.lock timeout 1200`

**Harness:** `perf_parity.py` arms through `real_ft_run._train`, default kernels (no
`--deterministic`), the same batches in the same order. Each arm logs every per-step letter and
span loss (hex) and its `train.path`.

**T1, shape A** (v3 shards, bt 16,384, 100 steps, mostly span-carrying batches): about 15 min.
- A-mask ×3 and A-none ×3, about 2 min each, interleaved mask/none.
- A-none-det ×2, 50 steps each: is det no-mask bit-identical to itself? If yes, future Tier-A
  goldens can run about 2.5× faster on it, once Tier B passes.

**T2, shape B** (probe-8k, bt 35,403, 50 steps): about 16 min.
- B-mask ×3 and B-none ×3.
- Speed: mask vs none × skip 0 vs skip 6, 3 interleaved rounds, overlapped timing, min-of-3.
- Peak memory for none+skip6 at 9 × 3,905 = 35,145 tokens.

**Statistic** (proposed; Fable sets the rule; reported whatever it shows):
- Per step t and per channel:
  - the within-baseline spread is S(t) = max over pairs i<j of |B_i(t) − B_j(t)|;
  - the cross deviation is D(t) = max over pairs i,j of |M_i(t) − B_j(t)|.
- Report:
  - medians and maxima of D and S;
  - the fraction of steps with D(t) ≤ S(t);
  - the first step where D exceeds max_t S;
  - final-step losses.
- The P2 screen passes if D(t) ≤ S(t) on at least 90% of steps in both channels, at both shapes.
- Because P2 is weak (Fable: deviations compound and 50–100 steps see only the start), (ii)
  decides.

## (ii) Outcome run: one phase-3 seed-0 run on the candidate, default kernels

This copies `tools/perf_tierb_fused.sh`.
- **Code:** `/home/ubuntu/perf/p3nomask` = qd-lane2 884b658 (phase-3's qd_data, so the phase-3
  shards stay readable) + the train-step patch + 0b95adb + the change above. Each is committed
  locally, so rows carry a clean `code_commit`.
- **Steps:**
  1. Train plus `--score-val` (`flock … timeout 2400`, cap 3,600 s). That is phase-3's recipe
     with `--train-attention-mask none`, and **no** `--fused-adamw`, so the candidate is
     attributed alone.
  2. `ft_linear_control` (CPU).
  3. The fp32 all-gates re-score of the saved checkpoint (`flock … timeout 5400`).
- **Pass:** the eval row inside the phase-3 envelope (choice 99.7–99.8%, span 99.8–99.9%), the
  paired-margin CI overlapping `eeda5db4`'s +0.150 [+0.136, +0.165], and all-gates matching
  `58fd1532` within that envelope.
- **Rows** go to the perf ledger. Every row names `train_attention_mask` and the SDPA backend.
- **Size** [I] from phase-3's ~1,290 s at mask speed and the fused run's stages:
  - GPU about 1 h expected, 2.5 h at the caps;
  - $2.29/h → about $2.5 expected, ≤ $5.8 at the caps;
  - single GPU and under $20, so rule 4's human yes is not needed. The lead schedules it.

**Total:** about 30 min of screen plus about 1 h of outcome. All of it after F, which holds
the GPU for about 17 h.

## Open questions for Fable

1. Does this wait for J5′ (Fable J's rule for attention work), or may it follow F directly?
2. Is the P2 rule above (D ≤ S on ≥90% of steps, both channels, both shapes) acceptable as the
   screen, given that (ii) is the decider?
3. If det no-mask is bit-identical to itself (T1), may the Tier-A golden harness move onto it?
   That is its own Tier-A question: masked-det vs none-det are not bit-identical, so old goldens
   would need re-recording.
4. Fused AdamW's Tier-B run is already queued. Should a passing no-mask path be screened together
   with fused (one combined outcome run), or alone as designed here? Alone keeps the attribution
   clean.
