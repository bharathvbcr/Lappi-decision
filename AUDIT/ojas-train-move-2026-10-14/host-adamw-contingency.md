# Host-AdamW contingency, costed (2026-10-14, documentation only, no run claimed)

Spec: `fable-ruling.md` "The memory criterion" and step 6. Required by `HANDOFF/ojas-training-2026-10-01.md:200`.
Labels: [V] read this session (file:line / log line); [I] inferred, arithmetic shown; [U] unverified or unmeasured.

Tooling: `gitpulse_insights` ran first, all facets `ok:true`; two live claude sessions in this worktree (12401 busy, 47940 idle),
the only shared-worktree dirt is `BRANCH_REPORT.md`, `BUILD_REPORT.md` [V]. Codeintel `is_fresh:false` (analyzer and source),
which is `GAP-DEVMAP-MCP-AND-CLI-DISAGREE-ON-ANALYZER-FRESHNESS-2026-10-14` [V]. `devmap_search adamw_entry_step` answered
(`crates/qd-train/src/adamw.rs:82-115`) with `walk_incomplete` (1 file failed to parse): partial confirmation only [V]. Everything
else below is `rg` / Read, not graph-confirmed. The diag log was found with `find` (no ripgrep-ignored paths involved).

## 1. The criterion and the measured numbers

- Criterion: tessl's working set (0.9 x `recommendedMaxWorkingSetSize`) >= 47 GB, else "the host-AdamW contingency is costed
  first, and the moments are never bf16" [V `HANDOFF/ojas-training-2026-10-01.md:200`].
- Measured (Apple M5 Pro): recommended working set 51,539,607,552 B; wired budget 46,385,646,796 B; at enter allocated
  43,588,009,984 B [V `AUDIT/ojas-training-2026-10-01/ojas-qwen35-gpu-diag-none.log:21`].
- After a 7.5 GB staging alloc, allocated 52,436,942,848 B, above both the wired budget and the recommended set, `poisoned false`
  [V same log `:23`]. That run then faulted in `read_gradients` (`:25`, MTL4 command buffer fault). It was the 128-token parity
  test, so its 43.6 GB "enter" figure is not the 8K step footprint [V `:16-25`].
- Miss: 47,000,000,000 - 46,385,646,796 = 0.61 GB short [I]. Rule 2: the number stands; this doc does not move it.
- Arithmetic check: 64 GiB x 0.75 = 51,539,607,552 (recommended) and that x 0.9 = 46,385,646,796.8 (wired budget, 0.675 of
  64 GiB) [I, exact], so the Mac is 64 GiB = 68.72e9 B
  and the budget is a fixed fraction of it, not something a config can raise [I].
- The 47 was an inferred sum: 30.78 GB step peak + 16 GB moments [V `HANDOFF/ojas-training-2026-10-01.md:173`; step peak
  30,784,164,432 B "peak memory footprint" at T=8192, `AUDIT/ojas-training-2026-10-01/mac-8k-train-step-bench.log:35`]. The bench's
  step excludes the AdamW update, so no moments were resident when it was measured [V per `fable-rung-d-resize.md` Preconditions, citing `bench_qwen35_train.rs:38,673`; not re-read].
- Observation for the human, not a ruling: the 16 GB is a rounded figure. Two f32 moments of the 7.53 GB table (staging size,
  `HANDOFF/ojas-training-2026-10-01.md:91`) are 15.06 GB, and 30.78 + 15.06 = 45.84 GB, 0.54 GB under the wired budget [I].
  It would pass only with no staging, no checkpoint write and no growth, against a log (`:23`) where staging alone overshot
  the budget by 6 GB. Whether to re-specify the precondition as a measured peak-with-moments is the human's call (section 5).

## 2. Host-moment design

What `crates/qd-train/src/adamw.rs` already covers [V]:

- `Moments {m, v}` f32 vectors and `EntryState {moments, steps}` (`:38-70`); per-entry step count, skipped on a `None` gradient
  (`adamw_entry_step`, `:82-115`).
- The full torch decoupled AdamW update for one f32 tensor, incl. clip coefficient as `grad_scale`, per-entry lr scale and weight
  decay, bias correction in f64 (`adamw_update`, `:119-193`), with argument validation (`:128-159`).
- Tests: bit-identical to torch on 1070 of 1070 elements over the decay-sensitive golden (`HANDOFF/L-trainer-2026-10-01.md:41`);
  `tests/adamw_oracle.rs`, `tests/adamw_decay_sensitive.rs` [V names].
- Current use: only the span head. `trainer.rs:769` calls `provider.adamw_step` for the tower, `:772-779` calls
  `adamw_entry_step` for host entries; `step.rs:240-243` says host entries exist because the head can lack a gradient [V].
- Not covered [V by reading the file; no other site found with `rg adamw_entry_step crates`]: tower-sized tensors, any threading,
  SIMD or chunking (one scalar loop, `:172-191`), moment save/load for the tower (device `save_state` / `load_state` are the
  provider's, `step.rs:265,268`; the trainer checkpoints `host_state`, `trainer.rs:365,516-522`, sized for a head), and a
  `StepProvider` that omits device AdamW. The trait's `adamw_step` is mandatory (`step.rs:244`), so a host mode is a new
  provider path or a trait method, not a flag [V].

Design (new code is needed around the existing update; the update itself is reused unchanged):

1. Metal step runs forward/backward; the gradient bank and the f32 parameters stay device-resident (provider description:
   "f32 weights, gradients and AdamW state", `provider.rs:117` [V]).
2. `grad_sq_norm` stays on device (`provider.rs:377`), clip coefficient computed as today (`trainer.rs:742-746`).
3. Per tower entry i, in order: read gradient i to host, read parameter i to host, `adamw_entry_step(param_i, Some(grad_i),
   state_i, row_i, lr, coef)`, write parameter i back. Moments `m`,`v` live only in host memory, never bf16 (the ruling and
   `:200`). One entry in flight bounds the transient to the largest tensor (embed_tokens, ~2 GB, `HANDOFF/...:104` names a 2 GB d_embed [V]).
4. Per-entry host reads/writes of parameters, gradients and moments are exactly what the unapplied tessl patch adds
   (`read_parameter_host`, `write_parameter_host`, `read_gradient_host`, `read_adamw_moment_host`, `write_adamw_moment_host`;
   `AUDIT/ojas-training-2026-10-01/ojas-qwen35-tessl-host-reads.patch:24,36,46,192,204` [V]). The patch was type-checked, its test
   not run (`:6`) [V]. Without it only whole-table staging exists (`ojas-qwen35/src/state.rs:561 read_table`, `:569 read_entries`
   [V]), and that staging is what overshot the budget (diag `:23`). So the patch is a hard prerequisite [I].
5. Per-step data movement, 2B f32 table 7.53 GB [V `HANDOFF/...:91`]: gradients 7.5 read, parameters 7.5 read + 7.5 write,
   moments 15.1 read + 15.1 write = 52.7 GB of memory traffic per step [I, sum]. Unified memory, so these are CPU-mapped
   reads/writes of device buffers, not PCIe copies [I; whether tessl's buffers are CPU-mappable `StorageModeShared` is [U], the patch's name "host-mapped" suggests so, `patch:5` [V]].
6. Checkpoint: moments are host-owned, so the trainer's resume state must serialize them like `host_state` (`trainer.rs:516-522`) [I].

## 3. Physical-memory sum on the 64 GiB (68.72e9 B) machine

| Term | Bytes (GB, 1e9) | Source |
| --- | ---: | --- |
| Step working set, device (weights f32 + grads + activations) | 30.78 | [V] bench log `:35` |
| Host AdamW moments m+v, fp32 | 15.06 | [I] 2 x 7.53 (`HANDOFF/...:91`); criterion text says 16 |
| Provider host f32 table (`read_table` output) | 7.5 | [V] `provider.rs:97-100`; transient at export/readback, counted as resident for worst case |
| `HOST_MARGIN_BYTES` | 1.07 | [V] `provider.rs:74` (1 << 30) |
| OS and other apps | ~8.9 | [I] diag log `:3` "memory free 87%": 0.13 x 68.72 |
| Sum | ~63.3 | [I] |
| Headroom to 68.72 | ~5.4 | [I] |

- Variants [I]: without the 7.5 GB table (entry-at-a-time only) 55.8 GB, headroom 12.9; with the 16 GB criterion figure add 0.94.
- Peak resident set in the bench was 25.7 GB vs footprint 30.78 GB (`mac-8k-train-step-bench.log`, "maximum resident set size" [V]);
  the footprint is the larger and is used. Zero swaps in that run [V]. A positive headroom of ~5 GB on a desktop Mac that
  also runs an editor and browsers is thin; the moments are not wired by Metal, so they can be paged, and paging is the
  failure to rule out [I].
- Not counted: tessl scratch recycled only at the next waited commit (`HANDOFF/...:94`, why 52.44 GB happened) [V]; checkpoint
  write buffers (a prior 8.03 GiB checkpoint write per `HANDOFF/mac-gpu-lane-2026-09-29.md:38` [V]).
- Moments leave the Metal working set (device peak stays ~30.8 GB < 46.39 GB wired budget, 15.6 GB spare [I]) but not the 64 GiB.

## 4. s/step estimate for the host update

- **No measurement exists.** `rg -i 'adamw|host'` over `ledger/*.jsonl` and `AUDIT/**/*.{log,txt,jsonl}` finds only GH200 torch
  step times and nothing timing `adamw_entry_step` or any host AdamW; the 8K bench excludes the update. Run with default
  ignore settings (not `-uu`), so this is evidence about tracked files only [V].
- Only existing figure: Fable's "+5-10 s/step, +30 min over 200 steps" for the host update [I, unmeasured; `fable-rung-d-resize.md`
  Preconditions]. It is a guess, not a bench. Cross-check [I]: 52.7 GB / 10 s = 5 GB/s; a scalar single-thread loop with a
  `sqrt` and a divide per element over 1.88e9 elements (`adamw.rs:172-191`) is compute-bound and could exceed 10 s, so even
  this range is unproven. [U] M5 Pro memory bandwidth and single-core throughput are not recorded in the repo.
- Context [V]: nominal step 70.1 s at the 452 tok/s floor (`fable-rung-d-resize.md` item 1); a 5-10 s add is 7-14%, 200 steps
  3.9 h -> 4.2-4.5 h against the 6 h cap [I].
- What would measure it (not run here; the bench does not exist yet, a lane must add it): a release-mode timing of
  `adamw_entry_step` over the 2B entry shapes, min-of-N interleaved against a chunked-parallel variant, e.g.
  `cargo run --release -p qd-train --example adamw_host_bench -- --shapes 2b --reps 5` [proposed command, [U] until written],
  plus on the real device the patch's per-entry read/write round trip timed under `tools/mac_heavy.sh` (single GPU, $0, rule 4 needs no yes).
  Cite the ledger row, not a console figure.

## 5. What must be true for the window to start; what stays the human's

Must be true [V that each is stated as a precondition in `fable-rung-d-resize.md` / the handoff; the host-AdamW items are new]:

1. A decision on the criterion (below). 2. The tessl host-read patch applied in a tessl-rooted session and its GPU test run
   (`patch:6`); `GAP-OJAS-QWEN35-2B-STAGING-HEADROOM-2026-10-01` is open until then [V gaps.jsonl:742].
3. A host-moment `StepProvider` path (or trait method) with a fail-first test, `adamw_oracle` still bit-identical, tower
   moments in the checkpoint.
4. The section 4 bench row exists, so the cap/eta rule is sized on a measurement. 5. A step with moments measured, which is what
   closes `GAP-OJAS-ADVICE-MAC-8K-FOOTPRINT-UNMEASURED-2026-10-01` [V gaps.jsonl:674]. 6. Standing preconditions: AC power and powermode, GPU exclusivity,
   rungs a-c green, `lr_scale` landed, schedule and digest parity tests green (`HANDOFF/...:199-207`).
7. A physical-memory pre-flight on free RAM (sum in section 3), because the wired-budget check does not see host moments.

The human's, not a lane's: (a) choose between running this contingency and re-specifying the 47 GB precondition as a
measured peak-with-moments (the second changes a pre-registered precondition; `fable-ruling.md:164-165`); (b) approve the
pin bump and the cross-repo tessl patch; (c) the Mac window itself (GPU slot, AC power). Gates were not touched here.

## Not sourced

Host AdamW s/step; M5 Pro memory bandwidth and CPU throughput; whether tessl buffers are CPU-mappable; exact OS baseline
(only a 87%-free line); resident size of the 7.5 GB table during a step; whether Metal wired memory pressure triggers paging.
