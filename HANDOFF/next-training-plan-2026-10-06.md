# The next training plan: refined data, a 16-bit recipe, a larger base (2026-10-06)

**Status: a plan.** Nothing is launched and no box is requested.

**Rules for any GPU run under this plan:**
- Every GPU-box run under it is a cost question for the human. That is the standing rule: Fable decides, and the human is asked only about cost.
- Rule 4 holds: every run needs a wall-clock cap, auto-terminate and a cost estimate.
- Reading rules go into `campaign/` before a run's result is read.

**Labels:**
- **[V]**: run or read this session (file:line, a command, a ledger row or a commit).
- **[I]**: inferred.
- **[U]**: not measured.
- **[fetched]**: external, read through a summarising fetcher.

**What this plan extends rather than replaces:**
- `HANDOFF/v6-plan-proposal-2026-10-06.md`: v6's arms and cost.
- `AUDIT/training-audit-2026-10-06.md`: the literature, gap triage and levers.
- `HANDOFF/data-clean-v6-plan-2026-10-06.md`: the data side, owned by session [764787].
- `AUDIT/ojas-tessl-for-next-plan-2026-10-06.md`: the Rust stacks, from [764787]'s read-only audit.
- `HANDOFF/caller-contract-2026-10-06.md`: callers and serving, owned by [32beaa].

This file adds four things: precision (16-bit), sizing for a larger base, the compute plan, and the training-side audit, fixes and stress tests.

**Ownership, agreed between the sessions on 2026-10-06:**
- This session: the training side (`python/qd_train/`, `tools/real_ft_run.py`, `crates/qd-train*`, the ledger and promotion, `crates/qd-metal` except main's 16 held paths) and this plan.
- [764787]: the data side, plus the ojas/tessl audit.
- [32beaa]: `crates/qd-runtime`, the caller contract and serving.

## 1. What changed in this lane

- **Commit `9819d16` (local, not pushed): the 16-bit recipe**
  - `qd_train.optim.KahanBf16AdamW` and `--optimizer kahan`.
  - Budgets that name only layouts the trainer builds.
  - Details in section 3.
- **Commit `8a531e8` (local): the Kahan checkpoint holds masters only.**
  - 14 B/param on disk, as the master recipe's; the compensation is recovered exactly on load.
  - State is allocated at the first step, so a scoring step pays nothing for it.
  - The split is canonical: it is exact and round-trips on 17.6M adversarial elements.
- **The audit-fix commit (section 10):** the DevMap audit's blocking findings, fixed with tests.
  - A larger base is now budgeted as itself, and an untied head is refused instead of silently replaced.
  - A checkpoint the sidecar bound would refuse is refused before step 0.
  - `gh200_rows` can measure P2's shape.
- **`AUDIT/next-train-2026-10-06/sizing.py` and `sizing.txt`:** text-tower sizes and memory for 2B/4B/9B (section 4).
  - Re-run after the value-head fix; the 4B/9B rows moved.
- **`gaps.jsonl`:** the records in section 8, appended through `qd_train.gaps.append_gap`.
- **Correction to my own records:** the GAP-DEVMAP-INDEX-MALFORMED-2026-10-06 update.
  - The store is not damaged: `sqlite3 PRAGMA quick_check` returns ok, and the CLI reads generation 3557.
  - Only the devmap MCP server answers "malformed".
  - So restarting that server fixes it; the index needs no rebuild.

## 2. The evidence the plan rests on

| Fact | Value | Source |
|---|---|---|
| v5 does the task | val choice top-1 ≥ 0.8278 and span ≥ 0.9066 on every seed; shuffled-label controls at chance | v6 proposal, "Is v6 needed" [V there] |
| v5 fails | `ood_abstain` on every seed; `needle_hunk_recall` on seeds 0 and 2, at the END of the context; `paired_margin_vs_linear` cannot run on this mixture | training audit §2.1, §3 [V there] |
| The released average | better on every tower metric; span 0.773 because the span heads are near-orthogonal | GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06 |
| A same-base, same-size decision model (decider-2b) beats Lappi on JevArena test-g6 | paired +0.110 [+0.071, +0.157]; RM-Bench +0.200; RewardBench 2 +0.074; JudgeBench +0.041 (n.s.) | `HANDOFF/merge-and-bench-2026-10-06.md` §6 [V]. Its training overlap with these benchmarks is [U] |
| The defect skill does not carry over to model-written bugs | RM-Bench chosen vs rejected separates at 0.536 [0.448, 0.624]; planted defects separate at 0.906 | merge-and-bench §7 [V] |
| Natural-bug held-out set from the human's own repos | 0 rows under the single-statement rule (9 of 1,838 fix commits qualify; 76% of fixes are agent-authored and multi-file) | [764787], data-clean plan, lane 3 [V there] |
| v5's optimizer | `master` (fp32 master + fp32 moments) on all 62 v5 ft rows | `rg optimizer_recipe` over `h100x2-v5-2026-10-03.jsonl` [V] |
| v5's batch | 35,403 tokens, widest bucket 9,638, `checkpoint_skip_layers` 6 | v5 ft row recipe [V] |

**What v6 is for [I, from the two proposals]:** better "I don't know" behaviour (`ood_abstain`, both halves) and the end-of-context needle failure. Those are data and contract problems, not capacity problems. A larger base addresses the knowledge ceiling (knowledge MC 0.632, commonsense 0.784; audit §4.10) and the gap to decider-2b.

## 3. Precision: what is already 16-bit, and the new 16-bit training recipe

| Path | Weights | Optimizer state | B/param (training) | Source |
|---|---|---|---|---|
| Release artifact | bf16, rounded to nearest-even from the fp32 master average | none | none | release notes line 25; `crates/qd-export` [V] |
| Mac serving (`qd-metal` on tessl) | bf16 weights and GEMM operands, f32 residual stream | none | none | `crates/qd-metal/src/model.rs:8-21` (doc comment) [V as written] |
| PyTorch `--optimizer master` (v5) | bf16 live + fp32 master | fp32 m, v; fp32 grad cast | **20**, measured on a GH200 | `memory.py` `bytes_per_param` [V] |
| PyTorch `--optimizer kahan` (**new**) | bf16 + bf16 Kahan compensation | fp32 m, v | **14** | `9819d16` [V, arithmetic and allocation test] |
| PyTorch `--optimizer bf16` | bf16 | bf16 m, v: the second moment freezes 50% low at step 384 | 8 | `optim.py` `moment_settling` [V]; refused for longer schedules |
| Rust `qd-train` over tessl/ojas | f32 masters (bf16 is a GEMM-operand option only) | f32 | 16 | ojas/tessl audit §2 [V there] |

**Why `kahan`.**
- Plain bf16 AdamW loses the updates. At the real tower's lr of 1e-5, an Adam step on a 0.02-magnitude weight is below bf16's half-spacing. On this host [V, `test_plain_bf16_adamw_loses_most_of_the_update_at_the_real_lr`], plain bf16 ends about 96% of the master's distance away from the master.
- The compensated recipe tracks the master to 5.2e-4 after 500 steps and 2.7e-4 after 2,000 steps (test bound 5e-3).
- Its fp32 parameters reproduce `torch.optim.AdamW` bit for bit.
- It resumes bit for bit through a real checkpoint file.
- Its checkpoints average through `ckpt_average --from masters` unchanged and score end to end.

**Not measured [U]. These are the first box's job (section 9, P1):**
- the optimizer step time on a GPU (a Python loop over parameters, with bounded slices of 16.8M elements; no foreach or fused kernel);
- the peak memory: `tools/gh200_footprint.py --optimizer kahan` now measures it;
- the loss trajectory on the real tower.

**Decision (Fable):**
- **2B v6 stays on `master`.** A recipe change makes v6's rows incomparable to v5's, and the master recipe fits.
- **`kahan` is for the larger base.** It is admitted there only after the parity probe P1 reads "admissible".

**What 16-bit does not reach yet:**
- Neither Rust stack has bf16 storage or compensated or 8-bit moments (ojas/tessl audit §2).
- ojas's bf16 tier (task `gp-bf16-compute-tier`, P0) is bf16 emulated in f32 storage (`1f26a4b`).
- Per the audit, that tier does not reach Lappi's Qwen3.5 path, which calls tessl directly.

## 4. A larger base: sizing (`AUDIT/next-train-2026-10-06/sizing.txt`)

**Method:**
- The text-tower parameter count is derived from each `config.json`. The formula reproduces the 2B checkpoint's safetensors header exactly: 1,881,825,088 [V].
- The 4B/9B configs are [fetched]. [764787]'s audit fetched the same official files and reports the same shapes.
- Memory comes from `qd_train.memory.estimate_step`. At v5's batch shape (35,403 tokens, 6 full-attention layers left un-checkpointed), it reproduces v5's recorded device budget to within 0.24%.
- **Activations are a floor.** The predictor under-predicts the backbone by 0.84-0.97x on a GH200 (GAP-MEMORY-PREDICTS-084-TO-097X-OF-MEASURED-BACKBONE-FOOTPRINT-ON-GH200). So the device tables budget against 0.84 × usable memory.
- **The loss head has never been measured** (GAP-LOSS-HEAD-TERM-NEVER-MEASURED-2026-10-06). Every GH200 figure drove a proxy loss, and code reading finds a bf16 `[V, H]` copy the estimate does not count: 1.27 GB at the 4B. P2 measures it with `gh200_rows.py --loss fused-ce`.
- **Corrected this lane: the value heads** (GAP-MEMORY-MODELSPEC-HAS-ONE-LINEAR-HEAD-COUNT-2026-10-06).
  - The 4B and 9B have 32 GDN value heads against 16 key heads. The memory model counted 16 for both.
  - That under-counted a 4B linear layer by 14% (54,784 against 64,000 saved elements per token) and halved its recurrent state.
  - The tables below are the corrected run. No 2B figure moved.

| Base | Text-tower params [I, derived] | Static GiB: master / kahan / bf16 | Step at v5's batch, GiB (kahan; ÷0.84) |
|---|---|---|---|
| 2B | 1,881,825,088 [V] | 35.1 / 24.5 / 14.0 | 53.0; 63.1 |
| 4B | 4,205,751,296 | 78.3 / 54.8 / 31.3 | 98.4; 117.2 |
| 9B | 8,953,803,264 (untied head) | 166.8 / 116.7 / 66.7 | 175.7; 209.2 |

**Tokens per batch that fit**, at width 9,638 with every layer checkpointed and a budget of 0.84 × usable memory (v5 trained 35,403). Values before the correction are in brackets:

| Base, recipe | H100 80 GB | GH200 96 GB | H200 141 GB | B200 180 GB |
|---|---|---|---|---|
| 2B master | 86,742 | 154,208 | 269,864 | 395,158 |
| 4B master | none | none | 57,828 | 125,294 [134,932] |
| **4B kahan** | none | **38,552** | 115,656 [125,294] | 192,760 [202,398] |
| 9B kahan | none | none | none | 9,638 [19,276] |

**A 4B checkpoint does not fit the sidecar bound** (GAP-SIDECAR-BOUND-REFUSES-4B-CHECKPOINTS-AFTER-TRAINING-2026-10-06).
- A full 4B checkpoint under master or kahan is 14 B/param, about 55 GiB. `MAX_SIDECAR_BYTES` is 32 GiB, derived for the 2B.
- `real_ft_run` now refuses such a run before step 0 rather than at its first save.
- The bound is a gate, so raising it is the human's decision. Until then a 4B run cannot save its weights. P2 saves none and is unaffected.

Usable memory per device:
- H100: 69.80 GiB, the free memory measured on v5's box [V].
- GH200: 87.95 GiB, the largest peak that completed [V].
- H200 and B200: nominal less 7% [U].

**Reading [I]:**
- **4B is a one-GPU run only under `kahan`.**
  - It fits a GH200 at v5's batch size. GH200 is $2.29/h [fetched 10-06, training audit §5], the cheapest box listed.
  - It fits an H200 with room to spare.
  - The master recipe needs an H200.
- **9B fits no single GPU at v5's batch** under any recipe that keeps faithful moments. It needs sharding (the trainer has no DDP or FSDP: `tools/memory_budget.py:51` [V]), 8-bit state, or LoRA. That makes it a trainer project, not a run.
- **The Rust stacks cannot train 4B or 9B** (ojas/tessl audit §1, [V there]). The blockers are GDN value heads ≠ key heads, an untied head and sharded safetensors. So the larger base is a **PyTorch-path run on a box**.
- **The 4B needs the 4B weights**, a download of at least 8.4 GB: the text tower alone in bf16 [I, from the parameter count]; the files also carry the vision encoder. That download needs the human's yes.
- **What the loader now checks on the 4B, on arrival.**
  - `spec_from_checkpoint` reads every field it needs from `config.json` or refuses.
  - The head must be tied by the top-level flag, by `text_config`'s flag where present, and by storage (no `lm_head.*` tensor).
  - `load_text_tower` compares every architecture number against that spec.
  - A refusal on the real 4B is information about the download, not a reason to relax a check.

## 5. Data

The data side is owned by [764787]; see `HANDOFF/data-clean-v6-plan-2026-10-06.md`. What this plan relies on from it:
- **Split-aware dedupe and the fixed keep rule.** They are opt-in for v6, so v5's pins do not move (GAP-QD-DATA-DEDUPE-KEEP-RULE-LEXICAL-SPLIT-BLIND-2026-10-03).
- **MMLU test/dev and CLINC test re-pinned as targets** (GAP-DECISION-INDEX-PANEL-CLINC-AND-MMLU-TEST-ITEMS-ARE-LAPPI-TRAINING-DATA-2026-10-03).
  - `knowledge.multiple_choice` gets a stated zero train cap.
  - v6's MMLU number will be honest, and may fall below v5's contaminated one.
- **New families.** Synthetic caller pools: `email.category`, `jarvis.*` and `apps.tool_select`. Converted sources: When2Call, ToolACE, SciRepEval, MNLI and CodeSearchNet.
  - **None is loadable yet.** The families and licences are registered in `qd_data` only when lane 2 merges, so nothing can train on them by accident [764787].
- **Natural-bug held-out set: 0 rows.** The transfer gate (GAP-TRANSFER-GATE-IS-NAMED-NOT-SPECIFIED-AND-HAS-NO-DATA) needs a different source or a looser rule, and that is the human's call.
- **From [764787] at 19:40Z, as reported** (not re-verified here; details in the data-clean plan §9 and `gaps.jsonl`):
  1. **The data lanes are integrated on branch `integrate-v6-data` (`22c38b6`), not yet on main.**
     - Its runner was green on qd-prep (clippy, test, release).
     - Its torch pytest had 69 errors and 4 failures that come from the worktree's environment and from gap ids that exist only in main's ledger (GAP-INTEG-WORKTREE-PYTEST-NEEDS-MAIN-ENVIRONMENT-2026-10-06).
     - So the Python suite is **owed on main after the merge, not passed**.
  2. **No synth or convert pool can enter a mixture until the v6 allocation exists.** The loader refuses a pool whose manifest allocation is not "applied" (GAP-V6-THREE-POOL-PRODUCERS-CAPS-APPLIED-IN-DECISIONS-ONLY-2026-10-06). The cap table over every pool is a rebuild-lane step that comes before any v6 training run (section 9).
  3. **The synthetic email pool has never been built, and had no held-out set** (GAP-SYNTH-EMAIL-HAS-NO-HELD-OUT-TARGET-2026-10-06).
     - The fix, on branch `stress-v6`, is a generator-held-out template partition, emitted under `heldout/` as both the eval set and the decontamination target.
     - An email caller is scored only on that set, once it is emitted: `~/qd-campaign/v6-data-2026-10-06/heldout/synth-email-v1/examples.jsonl`, split `heldout`.
  4. **Hardening on `stress-v6`, uncompiled until its runner runs:**
     - every qd-prep producer of training rows refuses a held-out-marked input (defence in depth under qd-train's own door);
     - every whole-file read is bounded and type-checked (a `/dev/zero` symlink used to read without end).
- From [32beaa]:
  1. v6's `gitpulse.commit_type` and `devtype.route` builders must render the exact question and slot strings in `docs/caller-contract.md` §4 (GAP-CALLER-REQUEST-SHAPES-PROPOSED-NOT-TRAINED-2026-10-06).
  2. Caller records are not training or eval data under the human's "Synthetic only" ruling. This plan lists none.

## 6. Compute plan

**Rule:** profile, then port, then benchmark (interleaved min-of-N, committed). No speedup is claimed before its benchmark exists.

| # | Where | What | Status | Owner |
|---|---|---|---|---|
| C1 | PyTorch box | Measure `kahan` against `master`: step time, peak memory, trajectory (P1). If the step is slow, move the Python per-parameter loop onto `torch._foreach_*` slices. | not measured [U] | this lane |
| C2 | PyTorch box | `--fused-adamw` (master): optimizer step 70 → 40 ms, peak 42.2 → 35.2 GiB at 4 × 8,441 | measured on the GH200 10-01 [V, `optim.py` comment]; a numerics change, so it rides in the recipe | the lead |
| C3 | PyTorch box | Score-val decode runs at a quarter of the GPU (GAP-SCORE-VAL-DECODE-RUNS-AT-A-QUARTER-OF-THE-GPU-2026-10-03); the J6 tail's per-row GPU-to-host syncs (v6 proposal) | profiled, not fixed | the lead |
| C4 | PyTorch box | GPU idle during the `real_ft_run` preamble (GAP-BOX-GPU-IDLE-REAL-FT-RUN-PREAMBLE-2026-10-03) | open | the lead |
| C5 | tessl (Mac) | Exact-size persistent allocations instead of power-of-two (about 35-40% padding at 2B/4B), plus a step-memory pre-flight | [I], ojas/tessl audit §4 | tessl `ft-4191d6df` (P1) |
| C6 | tessl (Mac) | Step waste: write the head gradient straight into the bank (`gemm.rs:1567` accumulate exists), drop CPU zeroing and forced per-layer drains. Free first test: `bench_qwen35_train --step` with `TESSL_MID_COMMIT=128` against unset, interleaved min-of-N | not run | tessl `ft-4dc74fad` (P2), under the lock |
| C7 | ojas/tessl | Blockers for any base above 2B: grouped GDN heads (Hv > Hk), untied head, sharded safetensors, newer config keys | open | tessl `ft-b7376d2e` (P1); ojas `ft-c57c98e0` (P1) |
| C8 | ojas/tessl | bf16 storage plus compensated or 8-bit moments: the Rust counterpart of `kahan`, with `KahanBf16AdamW` as its reference oracle | open | tessl `ft-a9661824` (P1); ojas `ft-c57c98e0` |
| C9 | ojas CUDA | `Qwen35Step` reports success without computing. **No plan counts ojas CUDA as a training path** until it is fixed; the cross-platform provider half (config, state, validators) is missing on CUDA | critical [V there] | ojas `ft-fa763e4f` (P0); `ft-e55afcb5` (P1) |
| C10 | tessl | Errors instead of panics on a busy or poisoned runtime; the training attention's fast-math `INFINITY` sites | open | tessl `ft-e2e677be` (P2); `ft-3ea6a4` (extended) |
| C11 | Mac memory | Is the Metal working set counted by `mac_heavy.sh`'s 32 GiB RSS cap? | [U] | whoever owns `tools/mac_heavy.sh`; tessl pre-flight in `ft-4191d6df` |

The ojas and tessl tasks are Mac work under the lock, except where one needs a box (ojas `ft-7162`'s GH200 work), which is a cost item.

## 7. Precautions: what v0.1 and this week taught, as rules for the next run

1. **Pre-register before reading.** This includes every reading rule for a probe. A probe that fails its pre-registered check stays on record (the C1 lesson, merge-and-bench §7).
2. **Gates and held-out sets are read-only (rule 2).** "Eliminate a gap" means fix the code or record the gap. It never means moving a threshold, dropping a seed or shrinking a held-out set. Decontamination that redefines v6's population *before any v6 result* is allowed and pinned in `campaign/` from the rebuild manifest.
3. **Measure the memory on the device before a long run.** The predictor errs in the unsafe direction (GAP-MEMORY-CAPACITY-PREDICTOR-ADMITS-MORE-ROWS-THAN-FIT, GAP-MEMORY-PREDICTS-084-TO-097X-OF-MEASURED-BACKBONE-FOOTPRINT-ON-GH200). `tools/gh200_rows.py` and `tools/gh200_footprint.py` sweep the intended shape on the rented device, and the run launches at the measured row count.
4. **Bound the box, not only the GPU steps.** v5's queue enforced `runs_usd` and never `box_usd`; the box bills while lanes wait for markers (GAP-V5-2GPU-BOX-CEILING-NOT-ENFORCED-2026-10-03, GAP-V5-QUEUE-BOX-WALL-CLOCK-NOT-TRACKED-2026-10-02).
   - **The v6 queue's first commit carries a box guard:**
     - it reads `launch.approved.box_usd` and refuses to start without it;
     - it takes the box's start from its uptime;
     - it caps every wait by the remaining box budget;
     - at the ceiling it stops all GPU work, writes a `BOX_CEILING` marker, and runs a termination command only if the human configured one.
   - The guard is written with that queue, not before. A guard with no queue to wire it into would be dead code. Its tests use a fake clock and uptime: budget reached mid-wait, a marker that never arrives, the guard process killed, and a misconfigured rate.
5. **Comparable rows only.** One recipe per comparison; `optimizer_recipe` is in every row. A `kahan` row and a `master` row are compared only inside P1's pre-registered reading.
6. **`quick` cannot promote (rule 8).** Every probe below is `quick` by construction.
7. **Shuffled-label controls on every seed** that could promote (GAP-SHUFFLED-LABEL-REQUIRED-ON-EVERY-SEED-2026-10-05), unless the human records the re-spec.
8. **Release and promotion are different decisions** (GAP-AVERAGE-PROMOTION-IS-THE-HUMANS-DECISION). The averaged artifact is a release candidate; `average_may_promote: false` stands until the human changes it.
9. **The span head:**
   - Averaging collapses it, so future seeds share one LP-initialised head (audit §4.4).
   - It is not served on the Mac (GAP-J7-EXPORT-SPAN-HEAD-UNSERVED).
   - In diff mode it is close to a free pass (GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD).
   - Its contract is the human's decision.
10. **Mac discipline:**
    - One heavy job at a time through `tools/mac_heavy.sh`; `cargo -j 2`; no mutation testing.
    - Every lane gets its own `CARGO_TARGET_DIR`; a shared one linked another checkout's code (caller-contract handoff, defect 5).
    - Test fixtures take `QD_PREP_BIN` from an existing build, so they never compile outside the lock (GAP-PYTEST-QD-PREP-FIXTURE-BUILDS-OUTSIDE-THE-MAC-LOCK-2026-10-06).
    - Agent worktrees branch from the stale `origin/main` (GAP-AGENT-WORKTREES-BRANCH-FROM-STALE-ORIGIN-MAIN-2026-10-06). A lane's green on its own base is not green on main.
11. **Pinned binaries and pinned data.** Every box binary and data input is pinned by sha256 in the queue, as in `post_f_common.sh`.
12. **A judge or helper that returns NaN is not a measurement.** Check backends at warm-up and fail closed (the decider-2b MPS lesson, merge-and-bench §6).

## 8. Gaps: what binds the next run

The 15 that matter (training audit §3) stand; this lane does not re-triage them. Status changes from this lane:

| Gap | Change | Evidence |
|---|---|---|
| GAP-DEVMAP-INDEX-MALFORMED-2026-10-06 | updated: the store checks ok; only the devmap MCP server reads it as malformed | quick_check, CLI status [V] |
| GAP-MEMORY-BUDGET-VERDICT-DEFAULTED-TO-A-LAYOUT-NOTHING-BUILDS-2026-10-06 | new, resolved | `9819d16`, `test_memory_budget.py` [V] |
| GAP-OPTIMIZER-SPEC-READ-AN-UNKNOWN-RECIPE-AS-BF16-2026-10-06 | new, resolved | `9819d16`, `test_every_recipe_name_maps_to_one_spec_and_an_unknown_one_is_refused` [V] |
| GAP-KAHAN-RECIPE-NOT-MEASURED-ON-A-GPU-2026-10-06 | new, open | section 3 |
| GAP-KAHAN-RECIPE-HAS-NO-RUST-COUNTERPART-2026-10-06 | new, open | section 3, C7 |
| GAP-QWEN35-4B-9B-TEXT-TOWER-COUNTS-ARE-DERIVED-2026-10-06 | new, open | section 4 |
| GAP-UNTIED-9B-HEAD-SILENTLY-REPLACED-BY-EMBEDDING-2026-10-06 | new, resolved with residual: an untied head is refused; training one is not built | section 10, #1 |
| GAP-SPEC-FROM-CHECKPOINT-READ-A-TIED-4B-AS-UNTIED-2026-10-06 | new, resolved | section 10, #3 |
| GAP-NO-MODELSPEC-REACHABLE-FOR-4B-9B-2026-10-06 | new, resolved | section 10, #3 |
| GAP-MEMORY-MODELSPEC-HAS-ONE-LINEAR-HEAD-COUNT-2026-10-06 | new, resolved | section 10, #4 |
| GAP-SIDECAR-BOUND-REFUSES-4B-CHECKPOINTS-AFTER-TRAINING-2026-10-06 | new, resolved with residual: refused before step 0; the bound is the human's | section 10, #2 |
| GAP-GH200-ROWS-PINNED-TO-2B-AND-BF16-2026-10-06 | new, resolved | section 10, #6 |
| GAP-GAPS-APPEND-IGNORED-A-SHORT-WRITE-2026-10-06 | new, resolved | section 10, #13 |
| GAP-LOSS-HEAD-TERM-NEVER-MEASURED-2026-10-06 | new, open: P2 measures it | section 10, #5 |
| GAP-SCORING-BUILDS-A-TRAINING-OPTIMIZER-2026-10-06 | new, open | section 10, #8 |
| GAP-QUEUE-WAIT-QUEUED-AND-FLOCK-UNBOUNDED-2026-10-06 | new, open: the v6 queue's box guard closes it | section 10, #9 |
| GAP-FUSED-CE-ROUNDS-LOGITS-TO-BF16-BEFORE-ITS-FP32-REDUCTION-2026-10-06 | new, open: a recipe decision | section 10, #11 |
| GAP-LOAD-CASTS-THE-BASES-FP32-TENSORS-TO-BF16-2026-10-06 | new, open: a recipe decision | section 10, #12 |
| GAP-SAFETENSORS-HEADER-READERS-NOT-ALL-ON-THE-BOUNDED-ONE-2026-10-06 | new, open (partly fixed) | section 10, clones |
| GAP-TESSL-GDN-VALUE-HEADS-UNVERIFIED-FOR-4B-9B-2026-10-06 | new, open | section 10, could not answer |

The training-side DevMap audit's findings are in section 10.

## 9. Order of work

**No GPU first (Mac, this lane and its peers):**
1. The DevMap audit's blocking findings (section 10), each fixed with a test that fails on the pre-fix code.
2. The data lanes integrated on main and re-tested on main by [764787] (the integration worktree's Python suite does not count; section 5). Then the rebuild lane's cap table over every pool, without which the loader refuses every synth and convert pool. v6's population is pinned from the rebuild manifest.
3. Eval-only diagnoses that need no box: the needle per-case split (audit §2.1), the span-head refit on the averaged tower (audit §4.4; pre-registered at `campaign/span-refit-mac-preregistered.json`), and OOD feature-space feasibility (audit §4.1).

**First box session (one pre-registration: `campaign/next-train-first-box-2026-10-06.DRAFT.json`):**
- **P1, recipe parity on 2B (quick).**
  - Arms `master` and `kahan`: same seed, same data, v5's flags except `--optimizer`, `--max-steps 600` (past the 384-step bf16 freeze, so the comparison is not on a schedule plain bf16 would also pass).
  - Then `gh200_footprint.py --optimizer {master,kahan}`.
  - The reading rule is in the draft.
- **P2, 4B footprint and step probe (quick), only if P1 reads "admissible".**
  - It needs the 4B download, which needs the human's yes.
  - The command: `gh200_rows.py --snapshot <4B snapshot> --optimizer kahan --width 9638 --loss fused-ce --out <path>`. The row count is measured, not predicted, and the loss head is in the measured step.
  - It writes no checkpoint, so the sidecar bound does not apply.
- **Box and cost [I]:**
  - A GH200 at $2.29/h.
  - P1 is about 2 h: 2 × (load + 600 steps at ~1.9 s + score) plus the footprint sweep.
  - P2 is about 1.5 h, plus the download.
  - About 3.5-4 h is about **$8-10**, under rule 4's $20 single-GPU line. It is still a **cost question for the human** under the standing rule, and GH200 cycles need the human's yes.

**Then v6 on 2B** (the v6 proposal's Arm B plus needle data, master recipe, ≥ 3 seeds with J5′ controls): about $175-235 (v6 proposal, cost table). A cost question for the human.

**Then, only if P1 and P2 pass, a `quick` 4B arm** on v6's data under `kahan`, to see whether a larger base moves the knowledge families and the gap to decider-2b.
- At 4B, a seed costs about 2.2× a 2B seed in GPU-hours [I, parameter ratio], but a GH200 hour costs 0.55× an H100's. The estimate goes in its own pre-registration.
- It needs the human's decision on the sidecar bound (section 4) before it can save weights.

## 10. Training-side DevMap audit

**Method.**
- A read-only subagent led with the `devmap` CLI, because the MCP server is down (GAP-DEVMAP-INDEX-MALFORMED-2026-10-06).
- It ran search, explore, impact, dependencies, dead and clones over the training path, at generations 3560-3594. Its full report is `build/next-train-2026-10-06/devmap-audit.md` (gitignored scratch; the findings are carried here and in `gaps.jsonl`).
- **Every impact walk stopped at depth 3 (`walk_incomplete`).** 74,784 of 147,767 attribution sites have no indexed target, and Rust call resolution is net 28.8%. So every caller list is a lower bound.
- Where a test file list was needed beyond DevMap's, it came from `rg` and is marked as such.

**Findings, and what this lane did.** "Pre-fix" means the test was run, or the defect was reproduced, on the code before the fix. [I] means the pre-fix failure is inferred from reading the code.

| # | Finding | Severity | Status | Evidence |
|---|---|---|---|---|
| 1 | A 9B (untied head) loads with the embedding as its head, silently; `ModelSpec.tied_embedding` was never read | blocks 9B | **fixed**: `head_is_tied` (config flags + storage, refuses a disagreement or an absent top-level flag); `load_text_tower` refuses untied; the spec check compares the tie | `test_backbone.py`: untied, contradictory and missing-flag loads refused [pre-fix: I, the old loader never read the tie] |
| 2 | 4B checkpoints exceed `MAX_SIDECAR_BYTES` and nothing checked before training | blocks 4B saving | **fixed in part**: `checkpoint_sidecar_bytes` refuses before step 0; its prediction equals the trained step's `state()` exactly (tiny tower, master and kahan); the bound stays, and is the human's | `test_backbone.py`, `test_optim.py`, `test_real_ft_device_budget.py` |
| 3 | No 4B/9B `ModelSpec` reachable from the trainer; the only builder lived in a CLI, read headers unbounded and defaulted the tie, gate and state dtype | blocks 4B | **fixed**: `qd_train.memory.spec_from_checkpoint`, one bounded reader, no defaults; `_real_step`, `memory_budget` and `gh200_rows` all use it; the 2B reproduces its constant field for field | pre-fix reproduced on `8a531e8`: a tied 4B read as untied, a contradictory one accepted (`build/.../prefix/prefix_4b.py`); `test_memory.py` |
| 4 | `ModelSpec` has one GDN head count; the 4B's 32 value heads were budgeted as 16 | degrades 4B (unsafe direction) | **fixed**: value heads and dims are required fields; checked against transformers' own module widths and a real recurrent state | pre-fix reproduced: 54,784 vs 64,000 elements, 1,048,576 vs 2,097,152 state bytes; `test_backbone.py`, `test_memory.py` |
| 5 | The loss-head term is unmeasured and misses a bf16 `[V, H]` copy | degrades | **open**, instrument built: `gh200_rows.py --loss fused-ce`; the estimate is not changed before a measurement | GAP-LOSS-HEAD-TERM-NEVER-MEASURED-2026-10-06 |
| 6 | `gh200_rows` pinned to the 2B, the bf16 recipe and a proxy loss | blocks P2 | **fixed**: `--snapshot --optimizer --width --loss --out`; no flags is the old sweep | `test_gh200_rows.py` |
| 7 | 9B fits no faithful single-card recipe | blocks 9B | a plan item (section 4): 9B is a trainer project | — |
| 8 | Scoring builds and budgets a training optimizer | degrades; blocks 9B scoring | **open**; the Kahan and torch AdamW states are lazy now, `MasterWeightAdamW`'s masters are not | GAP-SCORING-BUILDS-A-TRAINING-OPTIMIZER-2026-10-06 |
| 9 | Unbounded waits and `flock` without `-w` in the old box queue | cost | **open**: the v6 queue's box guard (precaution 4) bounds them; the old queue is not edited | GAP-QUEUE-WAIT-QUEUED-AND-FLOCK-UNBOUNDED-2026-10-06 |
| 10 | The Kahan checkpoint stored derivable masters (16 B/param) | degrades | **fixed** in `8a531e8` | `test_optim.py` |
| 11 | Fused CE rounds logits to bf16 before its fp32 reduction | unmeasured | **open**, a recipe decision: it changes numerics against v5 | GAP-FUSED-CE-ROUNDS-LOGITS-TO-BF16-BEFORE-ITS-FP32-REDUCTION-2026-10-06 |
| 12 | The base's 36 fp32 tensors are cast to bf16 at load | cosmetic | **open**, a recipe decision | GAP-LOAD-CASTS-THE-BASES-FP32-TENSORS-TO-BF16-2026-10-06 |
| 13 | `append_gap` ignored a short write | degrades | **fixed**: raises, naming the record; no retry | `test_gaps_writer.py` [pre-fix: I, the return value was discarded] |
| 14 | Stale figures in code text | cosmetic | **fixed**: the checkpoint size, the embedding row count, the `MAX_MASTER_PARAMS` and `MAX_KAHAN_PARAMS` comments, and the sidecar bound's derivation (the widest 2B checkpoint is 14 B/param, not 12; the bound is unchanged, and its test now derives the widest over every built layout) | `test_run_control.py` |

**Clones and dead code.**
- **Safetensors header readers.** Two Python copies are folded into `qd_train.memory.safetensors_header` (backbone's bounded one and `memory_budget`'s unbounded one). Two more remain (`run_control`'s sidecar reader and `verify_checkpoint_inventory.py`), plus the drifted Rust pair in qd-export and qd-metal, where qd-metal is a held path (GAP-SAFETENSORS-HEADER-READERS-NOT-ALL-ON-THE-BOUNDED-ONE-2026-10-06).
- **`replay_texts_from_prompts`** has two dead signals: DevMap at 0.9, and `rg -uu`. It is off the training path and not deleted. Its commit's intent has not been checked (Chesterton's fence).
- The other clone rows (`train_cpt`/`train_ft`, the TriState pair, the toy steps) are low-confidence and were not acted on.

**Lines.** The audit-fix commit's `git diff --stat` is in section 13.

## 11. Decisions

- **Fable (taken here):** 2B v6 stays on `master`; `kahan` is gated by P1; the box guard ships with the v6 queue; 9B is a trainer project.
- **The human: cost, downloads, dependencies, gates:**
  - the first box session (≈ $8-10) and any later box;
  - the 4B weights download (~8.4 GB);
  - `MAX_SIDECAR_BYTES` (32 GiB, a gate) for a 4B run that saves weights: a 4B full checkpoint is ~55 GiB. This is needed before the 4B arm, not before P1 or P2;
  - any new dependency (none is proposed here);
  - the seven decisions in the training audit §7, which stand.

## 12. Not verified

- The 4B/9B configs are [fetched] through a summarising fetcher. The audit's independent fetch agrees on shapes. No safetensors header for them was read.
- No GPU number for `kahan` exists.
- H200 and B200 usable memory is nominal.
- No 4B/9B safetensors header has been read; the loader's checks on the real 4B are first exercised at download.
- The pre-fix failures marked [I] in section 10 were inferred from the code, not run.
- DevMap's walk for the suite's test list stopped at depth 3 (`walk_incomplete`), and Rust call resolution is net 28.8%. Every DevMap list here is a lower bound. Thirteen more files came from `rg` (section 13).
- No GPU ran anything in this lane. Every GPU-only path changed here (`gh200_rows.attempt`, the `--loss fused-ce` step on CUDA) is **not run**; its CPU half is tested.

## 13. Regression and stress results

The Python suite runs single-process under `tools/mac_heavy.sh` (`build/next-train-2026-10-06/py_suite.sh`).

**Run 1** (2026-10-06 18:56-19:05Z, on `9819d16` plus the uncommitted masters-only Kahan change, which became `8a531e8`):
- 71 files DevMap names for the changed code (the log's own count; `py-suite-run1.log`).
- **1,440 passed, 30 skipped, 0 failed**, in 8 min 41 s.

**Run 2** (20:21-20:32Z, on `8a531e8` plus the audit fixes):
- 84 files: run 1's 71 plus 13 found by `rg` (the log's own count; `py-suite.log`).
- **2,123 passed, 30 skipped, 1 failed**, in 10 min 47 s.
- The failure was a test fixture, not the code under test:
  - `test_real_ft_pieces::test_a_cuda_run_whose_linear_attention_fell_back_to_torch_is_refused` called `_real_step` with `Path("snapshot")`, a path that names nothing, and stubbed the loader.
  - `_real_step` now reads the base's config and headers before loading, so it refused that path first: `--backbone snapshot: ... config.json is unreadable`.
  - The test now builds a header-only snapshot, and its assertion is unchanged. `test_real_ft_device_budget`'s near-copy of that helper is gone (it uses `test_memory._snapshot`).
- **Run 2 was not complete coverage.** DevMap fails closed on this tree (stale and degraded, GAP-DEVMAP-INDEX-MALFORMED-2026-10-06). So a static AST import closure over `python/tests`, `python/qd_train`, `python/qd_data` and `tools` was run, plus a text match for tests that read or run a changed file (`build/next-train-2026-10-06/coverage.py`; an analysis, not graph-confirmed).
  - It found 123 of the 180 test files reach a changed module or name a changed file.
  - 40 of those were not in run 2, among them `test_wire_gap_pins`, which reads `gaps.jsonl`.
  - Hence run 3.

**Run 3** (20:35:47-20:53:59Z, the whole `python/tests` directory, 180 files; `build/next-train-2026-10-06/py_full.sh`, `py-full.log`):
- **4,486 passed, 62 skipped, 1 deselected, 2 failed**, in 18 min 9 s. The deselected test is `test_loaders`' one `network` test: a download needs the human's yes.
- No cargo ran. `~/.cargo/bin`, the host's only cargo, was off PATH. qd-prep, qd-calib-fit and qd-gate-report were named by env var: the binaries runs 1-2 used, whose mtimes (13:15-13:59 CDT) predate both runs.
- **Runs 1 and 2 did invoke cargo in main's checkout.** conftest's `probe` and `noul_rows_bin` fixtures, and `calib_fit_bin`/`gate_report_bin` when their env var is unset, run `cargo build` whenever cargo is on PATH. Nothing was rebuilt (the binaries' mtimes are unchanged), but cargo ran. The run-2 script comment that said otherwise was wrong.
- **Two failures, neither from this change:**
  - `test_declared_dependencies`: this lane's launcher. `datasketch` is a declared dependency that `make torch-pytest` provides with `--with datasketch`, and `py_full.sh` did not. So `test_minhash.py`'s 29 tests counted as one module-level skip here. The pre-commit run below uses the Makefile's own launcher.
  - `test_lint_gate::test_the_repository_has_no_ruff_findings`: red on HEAD. 154 findings in 18 files, every one tracked and unmodified, and ruff is the locked 0.16.8, unchanged since 2026-09-19 (GAP-LINT-GATE-RED-ON-HEAD-154-FINDINGS-IN-18-COMMITTED-FILES-2026-10-06). Every file this commit carries is clean under the same ruff.
- **Skipped here and run in run 2:** `test_defect_noul` (17 tests: no qd-noul-rows without cargo) and `test_margin_probe_row` (3: no probe). Both files were in run 2, whose only failure was the fixture above.
- **Skipped here, and in no run of this lane:** `test_margin_probe_parity` (2, same cause) and `test_cargo_workspace_exclude` (1, `cargo metadata`). The rest are skips any host without the opt-in data or benchmarks takes. One of them is `test_qd_train_oracle_trainer`: its fixture was written under python 3.14.7, and this interpreter is 3.14.8.

**Pre-commit run** (the bytes committed): `test_real_ft_pieces` and `test_real_ft_device_budget` (ruff reordered their imports after run 3 collected them), `test_gaps_ledger`, `test_gaps_writer` and `test_wire_gap_pins` on the final `gaps.jsonl`, and `test_minhash` and `test_declared_dependencies` under the Makefile's launcher. Its result is recorded below.
