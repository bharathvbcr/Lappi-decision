# HANDOFF — L-cuda-M0: CUDA runtime (M0), K0 plumbing, K1 GEMM, rung-0 binary (2026-10-01)

Lane L-cuda-M0 of "train Lappi with ojas". Brief from the lead, 2026-10-01. The user's decisions
it implements are in `HANDOFF/ojas-training-2026-10-01.md:31-38`:
- the standalone sibling crate;
- design (B);
- cudarc pinned to `cuda-12080`, with cuBLAS;
- the parity ladder first;
- no GH200 time now.

The design followed is `AUDIT/ojas-training-2026-10-01/cuda-backend-scoping.md`: §3 K0/K1, §3.1,
§5.3 rung 0, and §6.1 M0.

**This is CPU-side work on a Mac with no NVIDIA GPU. No device code has run on a GPU.** Every
device test is NOT RUN (counts below).

Labels: **[V]** I ran it or read it (cited). **[I]** inferred, with the reasoning. **[U]**
unverified.

## What was built

Every file is in `/Users/bharath/Code/research/ojas/ojas-qwen35-cuda/`. None is outside it. No
git was run in ojas.

| File | What |
| --- | --- |
| `Cargo.toml` | My tables only: a `[[bin]] rung0` with `required-features = ["cuda"]`. `[dependencies]` and `[features]` are unchanged from the lead's scaffold. `[dev-dependencies]` (L-cuda-oracle's) is untouched |
| `README.md` | Numerics, library loading, the cross-build, **the rung-0 box command**, the device-test commands |
| `src/lib.rs` | Module map; `forbid(unsafe_code)` without `cuda` |
| `src/error.rs` | `CudaError`, the one error enum. Out-of-memory maps to `Capacity`, as in `ojas-cuda/src/lib.rs:126-141` |
| `src/bf16.rs` | f32→bf16 round-to-nearest-even and its inverse, bit-for-bit tessl's `tensor.rs:641-654` |
| `src/budget.rs` | `AllocBudget`: a lock-free bounded byte budget, with RAII `Reservation` |
| `src/nvrtc_cache.rs` | Bounded LRU compile cache. Key: module name, FNV-1a of the source, source length, NVRTC options (architecture and `--fmad` included), NVRTC version. A hash collision with different source text is refused |
| `src/kernels.rs` | CUDA-C for K0 (9 entries) and the FFMA GEMM (6 entries); `CompileSpec` `STRICT_SM90` / `STRICT_SM90A` |
| `src/k0_plan.rs` | Checked-arithmetic plans for every K0 op. Duplicate scatter rows are refused, naming both indices |
| `src/gemm_plan.rs` | `GemmShape`, `GemmLayout`, and `cublas_args`: the row-major→column-major `cublasGemmEx` mapping |
| `src/geometry.rs` | Grids from the shape only, never from the SM count |
| `src/host_ref.rs` | K0 references (bitwise); `gemm_f64`; `gemm_ffma_f32`, the FFMA kernel's exact order |
| `src/inputs.rs` | ojas's splitmix (`ojas-kernels/src/harness.rs:31-45`); tessl's ragged GEMM pattern (`tessl/src/gemm.rs:2349-2350`) |
| `src/check.rs`, `src/json.rs`, `src/libprobe.rs`, `src/rung0_cli.rs` | Check outcomes (a not-run or panicked check is never `pass`); a JSON writer; library diagnostics; rung-0 arguments and report file |
| `src/runtime.rs` (`cuda`) | `CudaRuntime` (below) |
| `src/buffer.rs` (`cuda`) | `CudaBuffer<T>` for f32 / bf16 bits / u32: typed and length-checked; downloads use a bounded event-poll before and after the copy |
| `src/k0.rs` (`cuda`) | The 9 K0 launches, each through its plan |
| `src/gemm.rs` (`cuda`) | K1: `gemm_ffma` (ExactF32, or bf16-rounding on load), `gemm_bf16_cublas` (raw `cublasGemmEx`), `gemm` (tessl's `GemmOperands` seam) |
| `src/smoke.rs` (`cuda`) | The device checks shared by rung 0 and the device tests, each under `catch_unwind` |
| `src/bin/rung0.rs` (`cuda`) | The rung-0 binary: JSON report, a watchdog cap of 280 s by default and 300 max, exit codes 0/1/2/3 |
| `tests/device_k0.rs`, `tests/device_k1.rs` (`cuda`) | 11 `#[ignore]` device tests |
| `tests/runtime_refusal.rs` (`cuda`, macOS) | The refusal-by-name test. It runs on the Mac |

**`CudaRuntime::open`** (`src/runtime.rs`), in order:
1. Probe `libcuda`, `libnvrtc` and `libcublas` with cudarc's own `is_culib_present` before any
   other cudarc call. cudarc panics on a missing library: `cudarc/src/lib.rs:199-201` [V, and
   seen: mutation M2 below].
2. `CudaContext::new`.
3. Refuse any compute capability other than 9.0.
4. Record the SM count, smem opt-in, `cuDriverGetVersion` and `nvrtcVersion`.
5. Create one non-default stream.
6. Create a cuBLAS handle, owned so that its destroy is not unwrapped (cudarc's `CudaBlas::drop`
   unwraps, `cublas/safe/mod.rs:91`):
   - bound to that stream;
   - a fixed 32 MiB workspace (lead's doc fact: 256-byte aligned, 32 MiB on sm_90+);
   - `CUBLAS_DEFAULT_MATH`;
   - atomics mode never set;
   - the math and atomics modes read back as raw u32s.
7. A bounded NVRTC cache: 32 modules, 64 MiB.

Every wait is an event poll bounded by `sync_timeout` (60 s by default; rung 0 uses
`min(cap, 60 s)`).

## Decisions taken, with the evidence

1. **ExactF32 is a hand-written fixed-k-order FFMA kernel, not cuBLAS.**
   - What it is: `acc = fmaf(a, b, acc)` over ascending k, from +0.0. No padded FMA, so a -0.0
     stays -0.0.
   - Why:
     - §3's convention (`cuda-backend-scoping.md:336`) and the K1 row (`:349`): "ExactF32 (FFMA,
       fixed k-order)".
     - The user's decision names "the hand-written FFMA ExactF32 tier" as the fallback
       (`HANDOFF/ojas-training-2026-10-01.md:36`).
     - Only a fixed-geometry kernel gives bits independent of the SM count (§3 determinism rule).
     - It admits a **bitwise** host test: Rust's `f32::mul_add` is one fused rounding. The test
       `mul_add_is_fused_on_this_host` proves this on the host [V].
   - The same kernel with `ROUND=true` rounds operands to bf16 on load. That gives the bf16
     tier's fallback and its on-device oracle for cuBLAS.
2. **Bf16 is raw `cublasGemmEx`:**
   - `CUDA_R_16BF` A/B, a `CUDA_R_32F` C, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT`, host
     alpha/beta f32.
   - cudarc's safe `Gemm<bf16>` writes a bf16 C (`cudarc/src/cublas/safe/gemm.rs:176`, as the
     scoping doc read it), so it is not used.
   - The cuBLAS determinism scope (same architecture and SM count; one stream; no atomics) is in
     `src/gemm.rs`'s module docs and the README.
3. **NVRTC → PTX at `compute_90` and a driver JIT, not CUBIN.**
   - cudarc 0.19.10's `nvrtc::result` has only `get_ptx` [V `cudarc/src/nvrtc/result.rs`, function
     list].
   - NVIDIA does not document PTX output for a real `sm_90` target (lead's doc facts).
   - `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. The options are passed verbatim
     as the strings in the cache key.
4. **`LD_LIBRARY_PATH` is set by the launching command.**
   - cudarc `dlopen`s bare sonames [V `cudarc/src/lib.rs:203-243`].
   - glibc reads `LD_LIBRARY_PATH` at process start [I: the glibc loader's documented behaviour,
     not tested here].
   - So the binary cannot point itself at the venv. It refuses by name instead, listing the
     searched names and any candidate file present but not loadable.
5. **Temporary device buffers drop right after their launch.** cudarc's drop is stream-ordered
   (`free_async` after event waits, `cudarc/src/driver/safe/core.rs:855-875`) [V read].
6. **Downloads are bounded on both sides of the copy.** cudarc's device-to-host copy into a `Vec`
   does not synchronise (`SyncOnDrop::Sync(None)`, `core.rs:1434-1450`) [V read].
7. **A fail-open in my own first draft of the watchdog, found and closed.**
   - The defect: a failed report write made the watchdog return without exiting, so the process
     could run past the cap.
   - The fix: `fire()` returns the timeout code whatever the write's result.
   - The test that fails against the draft: `a_failed_report_write_still_ends_the_run_at_the_cap`
     (`src/bin/rung0.rs`).

## Tests: run versus NOT RUN

Raw outputs are in `AUDIT/ojas-training-2026-10-01/l-cuda-m0-*.txt`.

| Suite | Ran on the Mac | Result | Evidence |
| --- | --- | --- | --- |
| lib unit tests, no feature | yes | **62 passed**, 0 failed | `l-cuda-m0-host-tests.txt` |
| lib unit tests, `--features cuda` | yes | **62 passed** | `l-cuda-m0-cuda-feature-tests.txt` |
| rung0 bin unit tests (watchdog), `--features cuda` | yes | **4 passed** | same |
| `tests/runtime_refusal.rs` (macOS, `--features cuda`) | yes | **1 passed** | same |
| `tests/device_k0.rs` | **NOT RUN** | 5 ignored (need an sm_90 GPU) | same |
| `tests/device_k1.rs` | **NOT RUN** | 6 ignored | same |
| rung0 on the Mac (refusal path) | yes | exit 2; names libcuda, libnvrtc, libcublas; report written, `status: refused` | `l-cuda-m0-mac-rung0-refusal.txt` |
| rung0 with an existing report / `--cap-secs 301` | yes | exit 2 / exit 2 | same |
| Mutation M1 (bf16 rounding to round-half-up) | yes | 2 bf16 tests **fail** | `l-cuda-m0-mutations.txt` |
| Mutation M2 (`probe_libraries()?` removed from `open`) | yes | the refusal test **fails** on cudarc's panic at `cudarc/src/lib.rs:200` | same |

**Totals for this lane:**
- Run on the Mac and passing: **67** distinct tests (62 + 4 + 1).
- Device tests: **11 NOT RUN**.
- rung0's device checks: **NOT RUN**.

L-cuda-oracle's 31 reference tests (4 + 8 + 1 + 3 + 15) also pass in the same runs. They are
that lane's to report.

**Clippy and format:**
- `cargo clippy --offline --all-targets --features cuda -- -D warnings` (the brief's command):
  **clean**, re-run in a fresh target dir to avoid cached results (`l-cuda-m0-clippy.txt`).
  - Earlier in the session it failed with 9 errors, all in L-cuda-oracle's `tests/reference/**`.
  - I relayed them to the lead. They were fixed by that lane before the final run.
- `rustfmt --check` on this lane's files: exit 0.
- `cargo fmt --check` on the whole crate reports diffs only in L-cuda-oracle's files and in
  `tessl/tests/common/gdn_train.rs`, which that lane includes by path (`l-cuda-m0-fmt.txt`).
  **Do not run `cargo fmt` in this crate:** it would rewrite a tessl file
  (`GAP-L-CUDA-M0-CARGO-FMT-WOULD-EDIT-TESSL-2026-10-01`).

## Cross-build artifacts (`l-cuda-m0-cross-build.txt`)

Built with the sysroot recipe (`cuda-backend-scoping.md` §1.5), target dir
`/Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda`.

- `cargo test --offline --release --no-run --target aarch64-unknown-linux-gnu --features cuda`
  links **10 test executables**. Mine are `ojas_qwen35_cuda-14579485aaa962c9`,
  `rung0-ec074d89f726fb62`, `device_k0-a1c65b0da4b54814`, `device_k1-c496f6df13da4818` and
  `runtime_refusal-bac9fcdc59c16410`.
- `cargo build --release --target aarch64-unknown-linux-gnu --features cuda --bin rung0`:
  - `.../aarch64-unknown-linux-gnu/release/rung0`;
  - ELF 64-bit aarch64 PIE, 1,575,016 bytes;
  - `NEEDED libgcc_s.so.1, libc.so.6` only;
  - glibc symbol versions GLIBC_2.17 to GLIBC_2.34, under the box's 2.39;
  - **sha256 `04ab105afcbf315d97f439f18c0baf73c659fd8d880ad34547419aa5ac9224f7`**.
- `device_k0-a1c65b0da4b54814` sha256 `e038f76dbc832b06661b1f208d4fb54b7dc533fb7038082d1e2c530e53e5374d`.
- `device_k1-c496f6df13da4818` sha256 `ce9e0a928f3152007e35e18498bc7eb62bd821d267845c5b45b0576577f4aa41`.

The sha256 depends on the checkout path, which the binary embeds. Copy this file; do not rebuild
it elsewhere.

## The rung-0 box command

The lead runs it after post-F item 10 (Fable Q4; rule 4 needs no yes for a ~5 min single-GPU
job). From the Mac:

    KEY=~/.ssh/bharath_m5_macbook_pro.pem; BOX=ubuntu@192.222.51.246
    ssh -i $KEY $BOX 'mkdir -p /home/ubuntu/bin /home/ubuntu/ojas-cuda'
    scp -i $KEY /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda/aarch64-unknown-linux-gnu/release/rung0 $BOX:/home/ubuntu/bin/ojas-qwen35-cuda-rung0

On the box, check `sha256sum /home/ubuntu/bin/ojas-qwen35-cuda-rung0` equals
`04ab105a…24f7`, then run one command:

    flock /home/ubuntu/queue/gpu.lock timeout 300 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib NVIDIA_TF32_OVERRIDE=0 /home/ubuntu/bin/ojas-qwen35-cuda-rung0 --out /home/ubuntu/ojas-cuda/rung0-$(date -u +%Y%m%dT%H%M%SZ); echo "rung0 exit $?"

**Exit codes:**
- 0: every check passed;
- 1: a check failed or panicked;
- 2: refused (a missing library, not sm_90, bad arguments, an existing report);
- 3: the binary's own cap (280 s);
- 124: the outer `timeout`.

The report `<out>/rung0-report.json` is one JSON object, `quick: true`, for a
`ledger/gh200-ojas-cuda-rung0-<date>.jsonl` row.

**Its `cublas.gemm_ex_bf16_f32_probe` check is the falsifier for
`GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01`.**
- If it passes, the gap closes positively.
- If it reports `CUBLAS_STATUS_NOT_SUPPORTED`, the gap closes negatively. The FFMA bf16 engine
  (`Bf16Engine::Ffma`) is then the bf16 tier, and its checks still run in the same report.

## Pre-registered: the local arm64 Linux container run

Approved by the lead, written here before running. Expectations:
- `rung0 --out /tmp/out` exits **2**, naming **libcuda, libnvrtc and libcublas**, and writes its
  report with `status: refused`;
- the lib unit-test binary passes **62** tests;
- the rung0 unit-test binary passes **4** tests.

Conditions:
- an arm64 glibc image already on disk (no pull);
- `--network none`;
- the binaries bind-mounted read-only and nothing else;
- no GPU;
- the image's glibc version reported against the box's 2.39;
- the podman machine stopped afterwards (it was stopped before).

**Results** (run 2026-10-02T00:27:44Z, after this section was committed in `af69512`; raw output in
`AUDIT/ojas-training-2026-10-01/l-cuda-m0-linux-container.txt`). **All three expectations held:**
- `rung0` exit **2** (the unpiped process status). It refused naming **libcuda, libnvrtc,
  libcublas**, and its report reads `status: refused`, `exit_code: 2`, `refusal.kind:
  library_missing`. The Linux candidate names it searched include `libcuda.so.1`,
  `libcublas.so.12` and `libnvrtc.so.12`, the box's sonames.
- The lib unit-test binary: `test result: ok. 62 passed; 0 failed`.
- The rung0 unit-test binary: `test result: ok. 4 passed; 0 failed`.
- `sha256sum` inside the container: `04ab105a…24f7`, the artifact above.

**Image:**
- `docker.io/library/python:3.11-trixie` (`b73116cf1823`), arm64, Debian 13. It was the only image
  on disk, and nothing was pulled.
- Its glibc is **2.41**, newer than the box's **2.39**. A newer glibc proves less: the binary's
  newest symbol version is GLIBC_2.34, so 2.39 should load it [I], but this run does not show it
  (`GAP-L-CUDA-M0-LINUX-GLIBC239-NOT-RUN-2026-10-01`).

**Isolation:** `--pull never --network none`, three read-only file bind mounts, no GPU.

**The podman machine:** stopped before; started for the run; stopped again at 00:27:58Z
(`podman machine inspect` → `stopped`).

## Open gaps

New, in `gaps.jsonl`:
- `GAP-L-CUDA-M0-NO-LISTAGENTS-2026-10-01`
- `GAP-L-CUDA-M0-CRATE-NOT-IN-DEVMAP-2026-10-01`
- `GAP-L-CUDA-M0-DEVICE-NOT-RUN-2026-10-01`
- `GAP-L-CUDA-M0-NVRTC-PTX-JIT-PATH-2026-10-01`
- `GAP-L-CUDA-M0-CAP-EXIT-PATH-NOT-EXERCISED-2026-10-01`
- `GAP-L-CUDA-M0-CARGO-FMT-WOULD-EDIT-TESSL-2026-10-01`
- `GAP-L-CUDA-M0-LINUX-GLIBC239-NOT-RUN-2026-10-01`

Still open, and falsified by rung 0: `GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01`.

## Calibration (M0+M1, about 9 specialist days)

**Finished in one agent session:**
- M0: runtime, buffer, cache, rung-0 binary;
- K0, all 9 entries;
- K1, both tiers, all three layouts;
- all host-tested and cross-built.

**Not started** (they are the rest of the M1 row in §6.1, outside this brief):
- K8 (`swiglu`, `residual_add`);
- K11 (AdamW, `grad_sq_norm`);
- the tiny-fixture loader.

**The calibration point is not yet measurable:**
- Device bring-up is where the scoping doc puts the risk (§7 risk 2, risk 3). None of it has
  happened.
- A first rung 0 that fails costs a box round trip per fix.
- The honest reading is "host side of M0/K0/K1 done; device side unvalidated", not "M0+M1 done".

## Notes for the next lanes

- The oracle lane owns `tests/reference/**`, `tests/reference_*.rs`, `tests/fixtures/**` and
  `[dev-dependencies]`. The device tests here use `src/host_ref.rs`. They can switch to
  `mod reference;` when it covers GEMM/K0.
- `ojas-core` is a `[dependencies]` entry from the lead's scaffold that no code here uses yet.
  It is left in place, not removed: it is the lead's, and it is likely meant for a later
  `DeviceBuffer` impl.

## First command for the next lane

After rung 0 has run on the box:

    scp -i ~/.ssh/bharath_m5_macbook_pro.pem 'ubuntu@192.222.51.246:/home/ubuntu/ojas-cuda/rung0-*/rung0-report.json' /Users/bharath/qd-campaign/ && python3 -m json.tool /Users/bharath/qd-campaign/rung0-report.json | rg -n '"status"|cublas_status|"detail"' | head -40

Before rung 0 (CPU work, next in M1):

    cargo test --offline --features cuda --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml
