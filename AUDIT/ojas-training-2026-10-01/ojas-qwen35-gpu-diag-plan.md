# ojas-qwen35 gpu_real_2b fault: pre-registered diagnosis runs

Lane: L-ojas-qwen35. Written 2026-10-01, before any of these runs. No GPU work
has been done for this diagnosis yet. Open gap:
`GAP-OJAS-QWEN35-2B-READ-GRADIENTS-METAL-FAULT-2026-10-01`.

## The fault

`Qwen35Model::read_gradients: embed_tokens.weight: Metal 4 command buffer fault:
(MTL4CommandQueueErrorDomain error 1.)`. It panics at `gpu_parity.rs:604`, 2 of 2
runs, at 23:16-23:17 UTC. Error 1 is `MTL4CommandQueueErrorTimeout`, which
objc2-metal 0.3.2 documents as "takes longer to execute than the system allows".

## Static evidence

All paths are under `~/Code/research/tessl/src`.

- **The failing call path.** `read_entries` calls `device_copies`. That allocates
  320 staging tensors (7.53 GB f32), zeroed on the host with no GPU work. Then
  `read_gradients` (`qwen35_params.rs:284`) does its first copy, the embedding:
  `gpu_copy` (`tensor.rs:610`), then `copy_f32` through `dispatch_1d`
  (`dispatch.rs:545`) over n = 508,559,360. With async encode off,
  `with_binder_sync` gives that copy a command buffer of its own and waits for it.
- **(a) One 2 GB dispatch exceeds the watchdog: falsified by the process's own
  history.** Seconds earlier in the same process, backward ran the same kernel,
  n and geometry: `deliver` (`qwen35_train.rs:227-243`, called at `:813-815`)
  with `accumulate=false` on an empty bank (ojas-qwen35 `step.rs`), copying
  d_embed into `bank.embed`. Its command buffer was waited on and its feedback
  observed, and `backward().unwrap()` passed.
- **(c) Index or geometry overflow: falsified by the same run.** Same n and
  geometry; `uint` indexing covers 508 M; 2.03 GB is under 2^31 bytes.
- **(e) Aliasing.** `gpu_copy` returns `Err` on overlap (`tensor.rs:619-624`); it
  does not fault.
- **(b) A late fault from an earlier command buffer: near-ruled-out.**
  - The feedback block sets `encode_failed` as soon as it runs
    (`runtime.rs:1666-1668`).
  - `alloc_buffer_kind` checks `encode_failed` on each of the 320 staging
    allocations (`:1085`), and `acquire_access` checks it (`:559`). All of those
    passed.
  - So the failing callback fired inside the embedding copy's own `with_binder`.
- **(d) Residency growth: open.** What is different about this copy:
  - It is the first `with_binder` after 320 `addAllocation`s.
  - `flush_residency` (`:969-987`) commits the set and calls `requestResidency`
    on it, so the 7.5 GB of growth lands on this copy's command buffer.
  - d_embed (2.03 GB) is still in the set, waiting for a cold recycle until the
    next waited commit.
  - Footprint is weights, bank and two moments (30.1 GB), plus staging (7.5 GB),
    plus d_embed (2 GB), about 40 GB. That is under `iogpu.wired_limit_mb=49152`
    on this 64 GB M5 Pro, so (d) cannot be closed on paper.
- **(f) Concurrent GPU work: open.** Reported by the lead: during both failing
  runs, the ojas coordinator's Metal and wgpu lanes were running short GPU tests.
  - Against it: the same-sized copy passed seconds earlier in the same runs, and
    both runs failed at the same place.
  - For it: GPU contention can stretch any one command buffer past the
    watchdog.

## Runs and pre-registered readings

All runs are serialized and each is under 1 min. Each is preceded by
`memory_pressure -Q`, and each runs only after the lead's "GPU cleared", with no
other GPU work. Each log goes to `ojas-qwen35-gpu-diag-<v>.log` in this
directory.

`OJAS_QWEN35_DIAG_FENCE=<v>` is a temporary switch in `device_copies`. It is
removed before shipping. It inserts an empty waited command buffer,
`rt.with_binder(|_| Ok(()))`, and prints `currentAllocatedSize` at each point.
The test is:
`cargo test -p ojas-qwen35 --release --test gpu_parity -- --ignored --test-threads=1 --nocapture gpu_real_2b`.

**R1, v=none (control, quiet GPU).** This run carries the most information.

| Outcome | Reading |
| --- | --- |
| Passes | The earlier fault was contention (f). The finding becomes "the read-back is too close to the watchdog under contention". The fix still shortens the work per command buffer, but it is not a residency bug. R2 to R4 are then run only to measure margins. |
| Same fault at `:604` | (f) is not needed to explain it, and (d) stands. Go on to R2. |

**R2, v=after.** Commit the staging growth on an empty waited command buffer
before tessl's copies.

| Outcome | Reading |
| --- | --- |
| Passes | The growth committed onto the copy's command buffer, or d_embed still being resident, is the cause. Go to R3 to tell them apart. |
| The fence itself faults | Establishing residency for the growth alone exceeds the watchdog. Go to R4. |
| The copy still faults | Footprint, not commit timing. Go to R4, and the fix is less staging (a tessl per-entry or host read). |

**R3, v=before** (only if R2 passes). Fence before staging (this evicts d_embed
and the step's scratch), and no fence after.

| Outcome | Reading |
| --- | --- |
| Faults | The growth committed onto the heavy command buffer is the cause. |
| Passes | Footprint, or d_embed's residency, is the cause. |

**R4, v=chunked** (only if R2 fails). Fence after every 1 GiB of staging, then
before the read.

| Outcome | Reading |
| --- | --- |
| Passes | Residency deltas need bounding. The fix is in ojas-qwen35 (chunked staging), with a tessl patch to bound them in `flush_residency`. |
| Faults | Footprint. The tessl patch for per-entry or host reads is required, and ojas-qwen35 can only avoid staging for gradients. |

**R5, gpu_tiny (5 tests).** Checks the save path through ojas-io
`replace_dir_with`, the streaming `SafeTensorsWriter` and the symlink refusal.

A pass on R1 does not count as a fix. A result counts only from a quiet-GPU run
whose log is in this directory.

## Amendment, 2026-10-02 00:17 UTC, written after R4 and before R3

Results so far:

- R1 (none) faulted at :604 on a quiet GPU: 43.59 GB allocated on entry, 52.44 GB after staging, against a recommended working set of 51.54 GB.
- R2 (after) faulted in the empty fence itself, at the same 52.44 GB.
- R4 (chunked) passed, but its first fence drained the pending cold recycles, so it finished staging at 50.20 GB, under the working set. R4 changed two things at once (footprint and delta size) and cannot separate them.

R3 (v=before) was pre-registered only for an R2 pass. It is run anyway, as the fifth and last run in the window, because it separates the two:

| Outcome | Reading |
| --- | --- |
| Passes | Footprint above the working set is the cause, and a 7.5 GB residency delta committed at once is not. |
| Faults | Delta size matters (possibly as well as footprint). |

R5 (gpu_tiny) follows R3.
