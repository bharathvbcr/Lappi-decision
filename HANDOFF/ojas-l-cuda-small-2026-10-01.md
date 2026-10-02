# HANDOFF — L-cuda-small: K3, K4, K6, K7, K9, K10 on CUDA, forward and backward (2026-10-01)

Lane L-cuda-small of "train Lappi with ojas". The target is the Qwen3.5 training provider
`/Users/bharath/Code/research/ojas/ojas-qwen35-cuda/`, with names and boundaries from
`AUDIT/ojas-training-2026-10-01/cuda-backend-scoping.md` §3.

**Device status: none of these kernels has run on a GPU.** Every device test is NOT RUN.
The Mac has no NVIDIA GPU and no CUDA compiler. What ran on the Mac, in the ojas tree, is
listed below with its AUDIT file.

**Ledger rows: none.** Nothing here is a training run, and no number below is a ledger
measurement. The host bound measurements are in the AUDIT logs cited.

## What was built (all files new in ojas-qwen35-cuda; no existing file edited except `src/lib.rs`)

| Kernel | Host module (plans, CUDA-C source, host mirror) | Device module (`cuda`) | Test file |
| --- | --- | --- | --- |
| shared | `src/small_common.rs` | `src/small_common_cuda.rs`, `src/small_smoke.rs` | `tests/device_small_common/mod.rs` |
| K3 published GDN gates | `src/gates_published.rs` | `src/gates_published_cuda.rs` | `tests/device_k3_gates_published.rs` |
| K4 causal conv1d + SiLU | `src/conv1d.rs` | `src/conv1d_cuda.rs` | `tests/device_k4_conv1d.rs` |
| K6 q/k norm + partial RoPE, output gate | `src/qk_norm_rope.rs` | `src/qk_norm_rope_cuda.rs` | `tests/device_k6_qk_norm_rope.rs` |
| K7 rms_norm (`1 + w`), gated norm | `src/rmsnorm.rs` | `src/rmsnorm_cuda.rs` | `tests/device_k7_rmsnorm.rs` |
| K9 embedding | `src/embed.rs` | `src/embed_cuda.rs` | `tests/device_k9_embed.rs` |
| K10 chunked cross-entropy | `src/ce_rows.rs` | `src/ce_rows_cuda.rs` | `tests/device_k10_ce_rows.rs` |

**`src/lib.rs`**: one Edit, made after re-reading the file. It added 16 `pub mod` lines at
rustfmt-sorted positions and changed no other line.
- Host: `ce_rows`, `conv1d`, `embed`, `gates_published`, `qk_norm_rope`, `rmsnorm`, `small_common`.
- Under `#[cfg(feature = "cuda")]`: `ce_rows_cuda`, `conv1d_cuda`, `embed_cuda`,
  `gates_published_cuda`, `qk_norm_rope_cuda`, `rmsnorm_cuda`, `small_common_cuda`, `small_smoke`.

Not touched: `src/kernels.rs`, `src/bin/rung0.rs`, `src/bin/runga.rs`, `Cargo.toml`, `Cargo.lock`,
any `gdn*` file, and every L-cuda-M1 file.

**Every launch goes through a plan.** Each is validated against the actual buffer lengths, and
every size, window and grid is checked u64. Each `unsafe { launch }` (cudarc's) carries a
`// SAFETY:` comment naming the kernel signature and the checks that justify it.

## Determinism and the bitwise mirrors

The modules compile with `STRICT_SM90`: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`.

**Reductions have a fixed shape.** None depends on grid, SM count or scheduling:
- per-thread ascending partials;
- then a warp xor-butterfly (16, 8, 4, 2, 1);
- then a block stage: lane 0 publishes, a second butterfly folds, padded with `0.0` / `-FLT_MAX`;
- weight gradients come from per-block partials summed in block order by
  `qd_col_sum_blocks_f32`.

There are no atomics, and every element has one writer.

**The host mirrors replay the device operation for operation.** They use `small_common`'s
`warp_sum`, `block_sum`, `row_sum`, `col_sum_blocks`, and `k8_act`'s exp, log, sigmoid and SiLU
with their bit-identical host twins. So the device should equal the mirror bitwise for K3, K4,
K7, K9, the K6 output gate and K10 on the FFMA engine. That is **unverified until the box run**.

**Not bitwise:**
- K6 q/k (libdevice `cosf`/`sinf` against Rust's `cos`/`sin`);
- K10 on cuBLAS's bf16 engine (its own summation order).

## Bounds (written before any run; one table in `tests/device_small_common/mod.rs`, constants in `small_common`)

Kernels are compared to L-cuda-oracle's float64 references (`tests/reference/`), evaluated on
**the f32-rounded golden inputs**. Each bound is tessl's own bound for the same kernel on Metal.

| Output | Bound | Source |
| --- | --- | --- |
| rms_norm, gated norm and conv forwards | `abs 1e-6 + rel 1e-5` | `tessl/tests/qwen35_kernels.rs:1071,1118,1174` |
| output gate forward | `abs 1e-7 + rel 1e-5` | `tessl/tests/qwen35_kernels.rs:1617` |
| q/k norm + RoPE forward | `abs 2e-5` | `tessl/tests/qwen35_kernels.rs:1556,1562` |
| every backward, and the K3 gates forward | `1e-4 * max|ref|` | `tessl/tests/qwen35_bwd.rs:5,43-66` (gates `:1296-1307`, embed `:1091-1132`) |
| K9 gather | bit-exact | the f32 table's bits |
| K10 per-row loss and loss | `1e-5 + 1e-5|ref|` | `tessl/tests/cross_entropy.rs:12,185` |
| K10 dh, dW at f32 / bf16 operands | `1e-4` / `2^-7` of `max|ref|`; the bf16 reference is on bf16-rounded inputs | `tessl/tests/cross_entropy.rs:12-14,252-258` |

The device must also equal the mirror bitwise where the mirror is bitwise, and a repeat run
must be bit-identical.

**About the brief's bf16 tiers (≤2^-8 / 2^-7).** Only 2^-7 is used, and only for K10's
gradients at bf16 operands. That is tessl's bound, because the softmax gradient is rounded to
bf16 before dh and dW. tessl's 2^-8 tier is a bound on **bf16 outputs**
(`tessl/tests/qwen35_kernels.rs:1123,1178`), and no kernel in this lane emits one: every output
is f32. The only bf16-operand path here is K10's GEMMs. So 2^-8 has nothing to apply to; it was
not dropped.

## Per kernel: what ran on the Mac, and what is NOT RUN

The Mac runs are from the ojas tree, `cargo test --offline --no-fail-fast`, with and without
`--features cuda`: 0 failures. Logs: `AUDIT/ojas-training-2026-10-01/l-cuda-small-cargo-test-host.txt`
and `-cargo-test-cuda-feature.txt`. Lane unit tests in `src/`: 35, all pass.

The "worst" figures below come from the six host suites rerun with `--nocapture`:
`l-cuda-small-host-measurements.txt`, 113 `pass` lines and 0 `fail`.

### K3 — published GDN gates
- **Ran (3 host tests):**
  - the embedded goldens' sha256 is checked against the manifest;
  - the mirror is held to the reference on the torch golden;
  - the mirror is held to the reference at 300 rows × 16 heads, with softplus' linear branch
    and `a = 120`.
  - Worst: dA_log 4.2e-7 of peak, against a bound of 1e-4.
- **NOT RUN (2):** device vs reference and bitwise vs mirror; rung a's `k3_checks`.
- Rule 9: every K3 name says `published`, enforced by `rule_9_every_k3_name_says_published`.

### K4 — conv1d + SiLU
- **Ran (3):**
  - the goldens c0..c2 (including `T = 2 < K - 1` over B = 3);
  - the 2B's 6,144 channels at T = 300 in a projection window;
  - K = 2.
  - Worst: y at 7.0e-2 of its bound; dw 4.9e-7 of peak.
- **NOT RUN (2):** device vs reference and mirror; `k4_checks`.

### K6 — q/k norm + RoPE, output gate
- **Ran (4):**
  - the golden's 10 rows placed at their positions, up to 20,001, in a 20,002-token sequence;
  - the 2B head layout (8 q heads with gates, 2 kv heads, D = 256, rot 64);
  - the output-gate golden.
  - Worst: q at 3.0e-2 of the 2e-5 bound.
- **NOT RUN (3):** device q/k vs reference, device gate vs reference and mirror, `k6_checks`.
- The q/k device-vs-mirror bound in `k6_checks` is also 2e-5 abs. The backward is held to
  1e-4 of peak.

### K7 — rms_norm, gated norm
- **Ran (3):**
  - goldens rms c0/c1 and gated c0/c1;
  - rms 70 × 2048;
  - gated 9 × 16 × 128 with z and dz in a fused window.
  - Worst: y at 2.6e-2 of its bound; backward ≤ 2.4e-7 of peak.
- **NOT RUN (3):** device on goldens, device at the 2B shapes, `k7_checks`.

### K9 — embedding
- **Ran (3):**
  - the golden, where the gather is bit-exact;
  - 600 rows × 1,000 vocab × 2,048 hidden onto a prior gradient;
  - unread rows keep their bits.
- **NOT RUN (2):** device vs reference and mirror; `k9_checks`, which includes a NaN payload
  through the gather.

### K10 — cross-entropy
- **Ran (2):**
  - The goldens c0 (mean), c1 (sum, scale 0.7) and c2 (a dominant logit planted at column 299),
    each × chunk {7, 64, 300} × operands {f32, bf16}.
  - V = 16,384 and V = 248,320 in chunks of 8,192 (30 full chunks and a 2,560 tail).
  - Worst loss: 5.9e-3 of its bound. Gradients: f32 ≤ 1.1e-6 of peak; bf16 ≤ 2.0e-3 against 2^-7.
- **NOT RUN (3):**
  - the FFMA engine bitwise vs mirror (f32 and bf16) plus the reference;
  - cuBLAS bf16 vs reference;
  - `k10_checks`.

**Total NOT RUN: 15 device tests and `small_smoke::small_checks`.**

## Fail-first (each run against a planted mutation, captured, then reverted and re-run green)

All three ran in the lane's private copy before landing: the logs show `/private/tmp/.../dev/` paths,
and that copy predates the `ColWindow` delegation. That does not change what they show:
- the mutated lines (`yr[d] = xr[d] * inv * (1.0f + w[d]);` and its mirror twin, and the
  `carried = ... s[n] * qd_exp_nonpos(m_old - m_new)` pair) are byte-identical in the landed files;
- the tests that caught the mutations are the same tests that pass in the ojas logs above.

- **K7: `w` instead of `1 + w`**, planted in the CUDA source and the mirror. Log:
  `AUDIT/ojas-training-2026-10-01/l-cuda-small-failfirst-k7.txt`.
  - The source pin failed, and so did the f64 unit check.
  - Both host suites failed: y at 1.8e5 times the bound.
- **K10: carried sum not rescaled (`s_old` instead of `s_old * exp(m_old - m_new)`).** Log:
  `l-cuda-small-failfirst-k10.txt`.
  - Every multi-chunk case failed, with losses up to 1.1e6 times the bound.
  - The 18 single-chunk (chunk 300) checks still passed, which shows the test isolates the
    chunk boundary.
  - The source pin and the online-equals-one-pass unit test also failed.
- **Syntax check: a planted `prt`.** clang++ reported "use of undeclared identifier 'prt'". Log:
  `l-cuda-small-failfirst-syntax.txt`, run in the lane's private copy.

## Other evidence (ojas tree, after landing)

- clippy `--all-targets -D warnings` is clean with and without `--features cuda`:
  `l-cuda-small-clippy-cuda.txt`, `l-cuda-small-clippy-host.txt`.
- `cargo fmt --check` exits 0 (check mode only). `rustfmt --check -l` over the 21 lane files
  lists nothing: `l-cuda-small-fmt-check.txt`.
- The aarch64-linux cross-build of the device tests is in `l-cuda-small-crossbuild.txt`.
  sha256, all in
  `/Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda/aarch64-unknown-linux-gnu/release/deps/`:
  - `device_k3_gates_published-e0e9795d254ed675`: `c0347031…706c0e`
  - `device_k4_conv1d-748b11fa0d68887c`: `52677e36…fadb12d`
  - `device_k6_qk_norm_rope-00c1ad1cc5efdcc9`: `2fe43121…2413ec`
  - `device_k7_rmsnorm-7e04614e3c9fbae5`: `27d83682…bd3b7b`
  - `device_k9_embed-f53e30d1a601e180`: `2632615d…f6d355`
  - `device_k10_ce_rows-a5ced68f78e78122`: `5d202810…5377a`
  - Each is an ELF 64-bit aarch64 PIE (`file` output in the log). These are the binaries at
    landing time; a later lane's change rebuilds them, so check the sums on the box.

## Decisions taken

1. **K6 angle from the host, decided before any run (lead's approval).**
   - The host computes `rope_inv_freq(rot, theta)` with the same arithmetic as the oracle's
     `rope_angle_f32`. The device computes `angle = (float)pos * inv_freq[p]`.
   - So the device angle is bit-identical to the reference's f32 angle. No device `powf` is
     involved (GAP-L-CUDA-ORACLE-ROPE-DEVICE-POW-2026-10-01).
   - The q/k forward is therefore held to tessl's 2e-5, not the 4e-3 used against
     transformers' own angle.
2. **`cosf`/`sinf` are the only libdevice transcendentals left in the lane.**
   - The bound is 2 ulp each per the CUDA 13.4.2 table; that 12.8 has the same bound is inferred.
   - Everything else is `k8_act`'s explicitly rounded exp, log, sigmoid and SiLU, or plain `+ * /`
     and `sqrtf` under `--prec-*`.
   - The lane lint forbids `__expf`, `__logf`, `__sinf`, `__cosf`, `__fdividef`, `rsqrtf`,
     atomics and `#include`.
3. **K10 runs on K1's FFMA engine for the bitwise tier, and its cost is report-only.**
   - K1's `gemm` takes whole buffers. So K10 copies each head chunk in, and each dW chunk out,
     with K0 `deliver`, and holds two chunk-width scratch sets (the full width and the tail).
   - At V = 248,320 and H = 2048 that is roughly 10-14 GB of traffic per call, about 4-5 ms
     [inferred].
   - GAP-L-CUDA-SMALL-K10-VIEWLESS-GEMM-COPIES-2026-10-01: a view-taking GEMM removes the copies.
   - The exps are `qd_exp_nonpos`: their arguments are ≤ 0 by construction, since `s >= 1`.
     The log is `qd_log`.
   - The per-row loss is formed in f64 on the host from `(m, s, tlogit)`, as tessl does.
     `ce_rows` therefore waits once, and it refuses a non-finite row **before** the gradient walk.
4. **K0 primitives reused:**
   - `deliver` (K10's chunk copies);
   - `ce_gather_rows` (K10's row gather, f32 or bf16);
   - `grid_1d` / `THREADS_1D` and `QD_GRID_STRIDE` (via `device_prelude!`).
   - `small_common::Window::check` now delegates to M1's `k0_plan::ColWindow::new`. The
     semantics and error text were identical, so there is one window check in the crate.
     Plans keep the unvalidated `(ld, off)` `Window`.
   - **Not reusable:** a view-taking GEMM (it does not exist), and `smoke::twice`. That one is
     single-output and bitwise. `small_smoke::run_twice` compares several outputs per run, some
     within tolerance.
5. **M1 convergence.** `small_common`'s reductions are the crate's fixed-shape reductions, each
   with a host mirror. They are `qd_warp_sum`/`qd_warp_max`/`qd_block_sum`/`qd_block_max`, and
   `qd_col_sum_blocks_f32` with `col_sum_blocks`. M1's `grad_sq_norm` should build on them
   rather than a second tree; the message went to main.
6. **NaN bits are not canonicalised in this lane** (M1's K8 does canonicalise). The bitwise
   claims are for finite inputs: GAP-L-CUDA-SMALL-NAN-BITS-NOT-CANONICAL-2026-10-01.
7. **Test-file layout differs from K0/gdn.** `tests/device_k*.rs` are *not* file-level
   `#![cfg(feature = "cuda")]`.
   - Their host tests (mirror vs reference) run in plain `cargo test`.
   - The device tests sit in an inner `#[cfg(feature = "cuda")] mod device` and are `#[ignore]`.
   - Goldens are embedded with `include_bytes!` and checked against the embedded
     `manifest.json`'s sha256 (`Goldens::verify`), so a binary copied to the box needs no
     checkout.
8. **The host-C++ syntax check uses the system `clang++` (macOS only).**
   - It fails if clang++ is missing.
   - It says nothing about device semantics: barriers, shuffles, numerics
     (GAP-L-CUDA-SMALL-NO-LOCAL-CUDA-COMPILER-2026-10-01).
   - The first NVRTC compile happens on the box (`small_common_cuda::compile_checks`).

## Hooks for runga (L-cuda-M1's binary; signatures sent to main and to M1)

All in `small_smoke` (cuda), each `fn(rt: &CudaRuntime) -> Vec<Check>`:
- `small_checks`, which runs `compile_checks` and then K3 to K10;
- `k3_checks`, `k4_checks`, `k6_checks`, `k7_checks`, `k9_checks`, `k10_checks`;
- `small_common_cuda::compile_checks`.

## Open gaps (gaps.jsonl, all `open`, owner lead)

- GAP-L-CUDA-SMALL-LISTAGENTS-UNAVAILABLE-2026-10-01
- GAP-L-CUDA-SMALL-GITPULSE-TRUST-2026-10-01
- GAP-L-CUDA-SMALL-DEVMAP-OJAS-NOT-INDEXED-2026-10-01
- GAP-L-CUDA-SMALL-NO-LOCAL-CUDA-COMPILER-2026-10-01
- GAP-L-CUDA-SMALL-DEVICE-NOT-RUN-2026-10-01
- GAP-L-CUDA-SMALL-K10-VIEWLESS-GEMM-COPIES-2026-10-01
- GAP-L-CUDA-SMALL-NAN-BITS-NOT-CANONICAL-2026-10-01
- GAP-L-CUDA-SMALL-K6-LIBDEVICE-COSF-SINF-2026-10-01

## Commits

- **Lappi worktree:** this handoff, the `AUDIT/ojas-training-2026-10-01/l-cuda-small-*.txt` logs
  and the gaps.jsonl lines, committed by explicit path with no Cargo.lock. The commit id is in
  the lane report.
- **ojas:** no git was run there (rule of the brief). The landed files are in the working tree
  for the ojas coordinator's merge.

## First command for the next lane

Copy the six binaries above to the box. Check their sha256 against this handoff. Then run each
one under the GPU lock. The library paths are M0's (`HANDOFF/ojas-l-cuda-m0-2026-10-01.md`,
"The rung-0 box command"):

    flock /home/ubuntu/queue/gpu.lock timeout 900 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib NVIDIA_TF32_OVERRIDE=0 /home/ubuntu/bin/device_k7_rmsnorm --ignored --test-threads=1 --nocapture; echo "k7 exit $?"

Then do the same for `device_k3_gates_published`, `device_k4_conv1d`, `device_k6_qk_norm_rope`,
`device_k9_embed` and `device_k10_ce_rows`.

A pass is every check printing `pass` and exit 0. Any `fail` is a finding against a bound that
was written before the run, and it must not be loosened.
