# ojas and tessl against the next training plan (read-only audit, 2026-10-06)

For the compute section of `HANDOFF/next-training-plan-2026-10-06.md`, which the training-plan session (cd80fb) owns. Two read-only subagent audits produced this, one per repo, with DevMap and GitPulse first. Nothing was edited, built or run, in either repo. Labels: **[V]** read in code at the cited line, **[I]** inferred, **[U]** unverified. Every memory or timing figure that comes from a repo's own docs is that doc's number; none was re-measured here.

DevMap caveat. Both graphs are fresh, but call resolution is low: tessl leaves 20,484 of 38,278 call sites unresolved, and ojas leaves 67,088 of 89,721. Every "nothing calls X" claim below rests on `rg` plus file reads, not on the graph. The gaps are listed at the end.

## 1. What the larger models need, and what blocks them

The configs are the official Hugging Face `config.json` files, fetched 2026-10-06 (external).

| | 2B | 4B | 9B |
|---|---|---|---|
| hidden / intermediate / layers | 2048 / 6144 / 24 | 2560 / 9216 / 32 | 4096 / 12288 / 32 |
| GDN key heads / value heads (dim 128) | 16 / 16 | **16 / 32** | **16 / 32** |
| attention q / kv heads, head_dim | 8 / 2, 256 | 16 / 4, 256 | 16 / 4, 256 |
| tied embeddings | yes | yes | **no** |
| checkpoint files | 1 | **2 shards** | **4 shards** |

**Blockers. Neither stack can train the 4B or the 9B today:**

1. **GDN value heads ≠ key heads** is refused in training. Both the 4B and the 9B have them.
   - tessl: `qwen35_train.rs:552-558`; `gdn_train` has no head grouping (`gdn_train.rs:25-27`). Inference already groups heads (`qwen35_gdn.metal:234`). [V]
   - ojas: `ojas-qwen35/src/config.rs:502-507`, and the CUDA `GdnPublishedPlan` takes a single head count (`ojas-cuda/src/gdn_plan.rs:111-140`). [V]
   - Existing task: tessl `ft-836fdba`.
2. **Untied LM head** (the 9B) is refused. One table serves as both embedding and head.
   - tessl: `qwen35_model.rs:165-180, 384-399`; `qwen35_train.rs:631-668`. [V]
   - ojas: `config.rs:417-427`. [V]
3. **Sharded safetensors** cannot be loaded. `SafeTensors::open` reads one file, and nothing reads `model.safetensors.index.json`.
   - tessl: `safetensors.rs:123`, `qwen35_model.rs:485`. [V]
   - ojas: noted at `ojas-qwen35/src/names.rs:294`.
4. **ojas config keys and the CUDA width cap.**
   - `output_gate_type` and other newer keys are refused by the key table (`config.rs:353`). [V]
   - The CUDA RMSNorm backward caps width at 4096 (`ojas-cuda/src/rmsnorm.rs:59,338`). That does not bind the 4B (hidden 2560) or the 9B (hidden 4096); it binds anything wider. [V]
5. **Validated shapes.** Only the 2B is validated. The compiled shape limits are attention head_dim 256 and GDN key dim 128, which all three sizes meet. The 16/4 attention heads of the 4B and 9B are in contract but tested only at 8/2 (`qwen35_model.rs:246-253, 325-332`). [V] Existing task: tessl `ft-d534aba`.

## 2. Memory: the Mac ceiling and 16-bit

- **Every training path is f32**, at 16 B/param: weights, gradient bank, and both AdamW moments.
  - ojas: `ojas-qwen35/src/step.rs:332-337`. [V]
  - tessl refuses bf16 for training at `qwen35_train.rs:541-545`. [V]
  - `Numerics::Bf16Operands` only rounds GEMM inputs, and it re-casts every weight on every call (`gemm.rs:169-175, 304-316, 1189-1192`; CUDA `gemm.rs:205-212`). [V]
  - No compensated, stochastic-rounded or 8-bit optimizer state exists anywhere; `rg -uu` finds none. [V]
- **tessl rounds every allocation up to the next power of two**, including persistent weights, gradients and moments (`runtime.rs:175-177, 187, 198-199, 1283`). [V]
  - That pads about 35-40% at 2B and 4B: roughly 2.5 GiB per 2B f32 table, so about 10 GiB across four tables. [I]
  - This plausibly explains most of the "~11.3 GB unattributed" in tessl `ft-dc1fa0`. [I]
- **f32 tables, real size and as allocated** (arithmetic from the code's shapes; not measured):

  | | one f32 table, real | as allocated (power-of-two) | four tables, real / allocated |
  |---|---|---|---|
  | 2B | 7.01 GiB | 9.50 GiB | 28 / 38 GiB |
  | 4B | 15.67 GiB | 22.00 GiB | 63 / 88 GiB |
  | 9B | 33.36 GiB | 42.00 GiB | 133 / 168 GiB |

- **The ceiling** is the M5's recommended GPU working set: 51.54 GB (48 GiB), measured by another session (`ft-dc1fa0`).
  - The ojas README measures 43.59 GB allocated after a 2B step at T=128 and 50.20 GB with staging, leaving 1.34 GB of headroom (`ojas-qwen35/README.md:186-218`). [U: README numbers]
- **Fitting the 4B on the Mac, before activations:** [I, arithmetic]

  | Variant | Size | Fits 48 GiB? |
  |---|---|---|
  | bf16 weights and gradients, f32 moments | 47.0 GiB | no |
  | everything in bf16 | 31.3 GiB | yes |
  | bf16 weights and gradients, 8-bit moments | about 23.5 GiB | yes |

  So the 4B on the Mac needs bf16 storage **and** a smaller optimizer state, or LoRA.
- **On ojas's bytes-per-parameter arithmetic** [I]:
  - An f32 master plus a bf16 working copy saves nothing: 4+2+2+4+4 = 16 B.
  - bf16 weights and gradients with f32 moments: 12 B.
  - bf16 weights with a compensation term and bf16 moments: about 10 B.
- **The "kahan" recipe** in the training-plan session is for the PyTorch path (`python/qd_train/optim.py`). It has no ojas or tessl counterpart.
- **No memory pre-flight.** A step's memory is logged, never enforced (`runtime.rs:705-711, 836-842`). Per-piece estimators exist (`GdnTrainWorkspace::bytes_for`, `CeWorkspace::bytes_for`), but nothing sums them, so an over-budget step runs until the OS stops it. [V] This is relevant given the two launchd SIGBUS panics on 2026-10-02.
- **`mac_heavy.sh` caps `ps` RSS** at 32 GiB. Whether shared Metal buffers count toward RSS is unverified, so the 32 GiB process cap and the 48 GiB GPU ceiling may be different limits. [U]

## 3. Correctness defects on or near the training path

1. **ojas CUDA `Qwen35Step` reports success without computing** (critical, [V]).
   - `forward` returns zero hidden rows and `letter_ce_sum = 0.0`; `backward` counts a gradient it never makes; `grad_sq_norm` returns 0.
   - `adamw_step` sets `BankState::Holds(1)` itself, getting past the empty-bank check and advancing the step count without an update (`ojas-cuda/src/step.rs:366-381, 383-409, 530-544`).
   - It is exported (`lib.rs:86`) and compiles without the `cuda` feature.
   - Nothing calls it today (rg; DevMap's walk was incomplete). Anything wired to it would log green steps that trained nothing.
2. **tessl panics instead of returning errors** when the runtime is busy or poisoned in the training path (`tensor.rs:218-220, 294-296, 320-322`, reached from `train_forward`, `attn_train.rs:113` and `cross_entropy.rs:269-270`). Only the C ABI catches panics; ojas calls Rust directly. [V]
3. **Fast-math `INFINITY` sentinels** in the training attention: `qwen35_attn_tiled.metal:181`, `qwen35_attn_bwd.metal:104, 187`. tessl `ft-3ea6a4` lists only the forward file. [V]
4. **CUDA input validation is weaker than Metal's.** `Sequence::validate` gets no maximum length (`ojas-cuda/src/step.rs:367`) and has no vision-token refusal (Metal: `ojas-qwen35/src/step.rs:91-115`). [V]
5. **GDN golden provenance is enforced only outside the ojas workspace.** `ojas-cuda/tests/` holds no `.rs` files. The `rule=published` guards live only in the out-of-workspace `ojas-qwen35-cuda/tests/`. The goldens themselves declare `"rule": "published"`. [V]

## 4. Compute: ranked opportunities (all [I]; nothing measured)

1. **Memory first** (tessl). Allocate persistent buffers at exact size instead of a power of two, then bf16 storage with smaller optimizer state. Without these, the 4B does not fit the Mac.
2. **Write the head gradient straight into the bank** (tessl, Lappi-specific). Every sequence with letter rows allocates a fresh vocab×hidden f32 head gradient: 2.0 GB at 2B, rounded to 4 GiB at 4B. It is zeroed on the CPU, filled by three full-vocab passes, then added into the bank (`qwen35_train.rs:631-635, 792, 801-822`). An accumulate-into GEMM already exists (`gemm.rs:1567`). Lappi pays this once per sequence, which may dominate at Lappi's short sequences.
3. **CPU zeroing and forced GPU drains** (tessl).
   - Every fresh allocation is zeroed on the CPU, even when a kernel overwrites it all (`runtime.rs:1284`): on the order of 20 GB per 2B step at T=2048. [I]
   - Each attention layer builds a workspace whose host writes force a full drain, in both the forward and the recompute (`qwen35_train.rs:1132`, `attn_train.rs:109-121`). A reusable scratch already exists (`:876`).
   - `TESSL_MID_COMMIT` overlap is off by default, and neither Lappi nor ojas sets it.
   - A free first experiment, with no code change: `bench_qwen35_train --step` with `TESSL_MID_COMMIT=128` against unset, interleaved min-of-N.
   - Caution: the drains also recycle temporaries, so removing them can raise peak memory.
4. **More than one sequence per Metal launch.** The CUDA GDN plan already batches variable lengths (`ojas-cuda/src/gdn_plan.rs:7-9`).
5. **On CUDA, keep bf16 weight shadows** refreshed once per optimizer step, instead of a fresh cast on every GEMM call (`ojas-cuda/src/gemm.rs:205-212`).
6. **Per-entry state staging.** Save and read stage a whole f32 table on device: 7.5 GB at 2B (`ojas-qwen35/src/state.rs:492-552`).

## 5. What Lappi's trainer needs to switch to ojas

`qd-train`'s `StepProvider` (`crates/qd-train/src/step.rs:190-243`) needs `parameters`, `hidden_size`, `vocab_size`, `accumulate`, `grad_sq_norm`, `supports_lr_scale`, `adamw_step`, `step_count`, `read_parameters`, `describe`, `save_state` and `load_state`.

- **Metal:** all of these exist through `qd-train-metal` over ojas-qwen35. Per-entry learning-rate scales wait on tessl (`groups.rs:278`). [V]
- **CUDA:** nearly everything is missing:
  - weight loading, the real step, the parameter table, read-back and state I/O;
  - `ojas-qwen35` is `#![cfg(target_os = "macos")]` (`lib.rs:33`), so its validator, name map, optimizer groups and state format vanish on Linux. [V]

**Plan consequence [I]:** a larger-model or bf16 run in the next cycle is a PyTorch-path run on a GPU box. The Rust stacks need the blocker tasks below first. That box run is a cost item for the human.

## 6. Tasks

Filed 2026-10-06 on each repo's GitPulse board, after reading the existing tasks and reviewing every one the board flagged as related. Grouped to keep the boards small.

| Repo | Item | Pri | What |
|---|---|---|---|
| ojas | `ft-fa763e4f…` | P0 | The CUDA `Qwen35Step` forges success; refuse until wired (section 3.1) |
| ojas | `ft-e55afcb5…` | P1 | The platform-neutral provider half, shared by Metal and CUDA; CUDA validator gaps (3.4, 5) |
| ojas | `ft-c57c98e0…` | P1 | Above 2B: GDN head grouping, untied head, new config keys, bf16 storage (1, 2) |
| tessl | `ft-b7376d2e…` | P1 | Load 4B/9B: sharded safetensors, untied LM head (1.2, 1.3) |
| tessl | `ft-4191d6df…` | P1 | Exact-size persistent allocation and a step-memory pre-flight (2) |
| tessl | `ft-a9661824…` | P1 | Split from `ft-836fdba`: bf16 storage plus bf16 or 8-bit moments (2) |
| tessl | `ft-4dc74fad…` | P2 | Step waste: gradients into the bank, CPU zeroing, forced drains; benchmark first (4) |
| tessl | `ft-e2e677be…` | P2 | Training path returns errors, not panics (3.2) |
| tessl | `ft-3ea6a4` (extended) | P2 | Adds the training attention's `INFINITY` sentinels (3.3) |

**Not filed separately:**
- **CUDA bf16 weight shadows** (section 4, item 5) and **K10's view-less GEMM**: these are ft-7162's existing GEMM-views criterion. Add the shadows there when that work starts.
- **Metal multi-sequence batching**: this is `ft-836fdba`'s first bullet.

The existing tasks the audit checked against the code:
- **Still match:** tessl `ft-836fdba`, `ft-dc1fa0`, `ft-4b1313`, `ft-3ea6a4`, `ft-a2151b5`, `ft-d534aba`, `ft-b1c4a86`; ojas `ft-69ef`, `ft-7162`, `ft-24d3`, `ft-301d`, `ft-d102`, `ft-8a69`.
- **Off Lappi's path:** ojas `ft-69ef` (the bf16 tier on the `Backend` trait) and `ft-9fc0` do not reach Lappi's Qwen3.5 path, which calls tessl directly. [V]
- **Closed:** the 120 GiB `go.test` blowup (ojas). The fixture is fixed (`go/profile_test.go:25-29`), and a regression cap is in place (`ojas-device/src/bandwidth.rs:633-650`). [V, code]

## 7. What DevMap and GitPulse could not answer

- **DevMap walks.** Every neighbours and impact answer in both repos carried `walk_incomplete`.
- **ojas `devmap_clones`.** It showed 48 of 396 groups; the two copies of `step.rs` were not surfaced.
- **tessl, missed caller.** DevMap missed `GpuBuffer::map_host` as a caller of `host_access`.
- **tessl, string-dispatched kernels.** Kernels dispatched by name read as dead: `qwen35_residual_add_f32` at 0.9 confidence. All `.metal` entry points are invisible this way.
- **Freshness disagreement.** `gitpulse_insights` reports `codeintel.is_fresh: false` for ojas while `devmap_status` reports fresh.
- **tessl collisions facet.** It reports `ok: false`, from the stale worktrees `target/wt_head` and `target/wt_norm`.
- **tessl HEAD.** Unverified: the harness blocks `git -C`.
