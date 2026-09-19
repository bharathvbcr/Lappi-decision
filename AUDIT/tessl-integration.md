# tessl integration guide — K1–K7 + `tests/gdn.rs`

**Lane:** TESSL INTEGRATION AUDIT · **Date:** 2026-09-19 · **Target:** `/Users/bharath/Code/research/tessl` (canonical crate)

Every claim below is cited `file:line` and labelled **[verified]** (I ran it or read it), **[inferred]**
(derived from what I read), or **[unverified]**. Counts were re-measured today, never quoted from a doc —
`tessl`'s own docs quote stale numbers in at least one place (see CORRECTION 7).

---

## 0. TL;DR for the kernel author

| Thing | Answer | Cite |
| --- | --- | --- |
| Build health | `cargo build` **exit 0**, 5.39s, zero warnings | §E |
| Test health | **306 passed / 0 failed / 1 ignored**, 37 targets, exit 0 | §E |
| Threadgroup memory limit | **32768 bytes (32 KiB)** on Apple M5 Pro — measured | §B.5 |
| 128×128 fp32 GDN state | 65536 B = **2× the limit** → device buffer, mandatory | §B.5 |
| New `.metal` pickup | Drop in `kernels/`, nothing to register — glob is automatic | §A.1 |
| **But** TensorOps (K1) | Must be added to `tensorops_sources` or it silently falls back to metal3.2 | §A.3, CORRECTION 1 |
| **Landmine** | `promoted_kernels.rs` hard-asserts `PROMOTED.len() == 44` | §C.6, CORRECTION 5 |
| `.npy` fixtures | **No loader in tests, zero fixtures on disk, `read_npy` has zero callers** | §C.4, CORRECTION 3 |
| GDN in tessl today | **None.** 0 matches / 76 files with `rg -uu` | §G |

---

## A. Kernel authoring contract

### A.1 How a new `.metal` file is picked up

`build.rs` globs the directory. **There is no registry, no list to edit, no `mod` to add.**

`build.rs:161-171` — non-recursive `read_dir(kernels/)`, keeps `extension == "metal"`, minus a skip list:

```rust
let skip: &[&str] = &["matmul_tensorops.metal"];
let mut others: Vec<PathBuf> = fs::read_dir(&kernels_dir)
    .unwrap_or_else(|e| panic!("read kernels/: {e}"))
    .filter_map(|e| e.ok())
    .map(|e| e.path())
    .filter(|p| {
        p.extension().and_then(|s| s.to_str()) == Some("metal")
            && !skip.iter().any(|s| p.file_name().and_then(|n| n.to_str()) == Some(*s))
    })
    .collect();
others.sort();
```

Each source compiles to a `.air` in `OUT_DIR`, then all `.air` link into one `default-<pid>-<nanos>.metallib`
(`build.rs:211-226`). The path is exported as `TESSL_METALLIB` via `rustc-env` and as `DEP_TESSL_METALLIB`
to dependents (`build.rs:232-233`). **[verified]**

`kernels/tune/` is **not** picked up: `read_dir` is non-recursive and a directory has no `.metal` extension.
That is deliberate — `build.rs:110-113` says the 34 A/B kernels stay opt-in behind `TESSL_GEMM_TUNE`. **[verified]**

### A.2 Change tracking — why `.h` files matter

`track_kernel_sources` (`build.rs:353-374`) recurses and emits `rerun-if-changed` for **both** `.metal` and `.h`:

```rust
} else if matches!(p.extension().and_then(|s| s.to_str()), Some("metal") | Some("h")) {
```

`build.rs:346-352` explains why a bare directory watch is wrong: a directory's mtime only moves on
create/delete/rename, so editing a kernel in place would leave a stale metallib while `cargo test` reported
a pass. **Any new `.h` helper you add is tracked automatically.** **[verified]**

### A.3 Metal language version — two different contracts

This is the single most important authoring fact and it is **asymmetric**:

| Path | Flags | On failure |
| --- | --- | --- |
| `matmul_tensorops.metal` (+ `tune/` when enabled) | `-std=metal4.0 -O2 -isysroot <sdk> -mmacosx-version-min=26.0 -c` | **panic** — "refusing simdgroup-only metallib" (`build.rs:135-156`) |
| every other `.metal` | tries `metal4.0`, **falls back to `metal3.2`** | `cargo:warning` + prints the metal4.0 diagnostic, then panics only if 3.2 also fails (`build.rs:177-197`) |

`tensorops_sources` is a hard-coded vector at `build.rs:116`:

```rust
let mut tensorops_sources: Vec<PathBuf> = vec![kernels_dir.join("matmul_tensorops.metal")];
```

Compiler resolution: **never** `xcrun -sdk macosx metal` — `build.rs:10-12` records that `-sdk` breaks cryptex
Metal Toolchain resolution on Xcode 26+. It uses `xcrun -f metal` plus an explicit `-isysroot`
(`build.rs:102-103`, `298-311`). Requires `xcodebuild -downloadComponent MetalToolchain`. **[verified]**

`println!("cargo:rustc-cfg=metal_runtime_tensorops")` at `build.rs:157` — set unconditionally once TensorOps
compiles, so it is not a per-kernel capability signal. **[verified]**

### A.4 Metal 4 TensorOps — what K1 needs

`kernels/matmul_tensorops.metal:14-18`:

```cpp
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;
```

The canonical shape of a TensorOps GEMM (`matmul_tensorops.metal:44-90`, entry `matmul2d_tensorops_f32`):

```cpp
constexpr int SM = 32;
constexpr int SN = 32;
constexpr auto desc =
    matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false,
                        matmul2d_descriptor::mode::multiply);
matmul2d<desc, execution_simdgroup> op;

uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
if (tile.x >= tiles_n || tile.y >= tiles_m) return;
int tx = (int)tile.x * SN;
int ty = (int)tile.y * SM;

auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
if (interior) {
    auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM}, array<int, 2>{1, (int)K});
    auto tB = tensor(B + tx,          dextents<int, 2>{SN, (int)K}, array<int, 2>{1, (int)N});
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM}, array<int, 2>{1, (int)N});
    op.run(tA, tB, tC);
} else {
    auto tA = mA.slice(0, ty);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}
```

Four rules the file's own header states (`matmul_tensorops.metal:1-12`) **[verified]**:

1. **Device pointers must be non-`const`** — "`const` poisons MPP type matching". Note every TensorOps entry
   takes `device float *A`, not `device const float *A`. This is the opposite of every other kernel family.
2. `mode::multiply` **needs C zeroed once** by the host (packed with the matmul).
3. Compile-time tile extents via `offset + dextents{SN,SM}` is the bounds-check elision — pointer tensors lack
   `static_slice`.
4. Morton 1D threadgroup walk (`morton_decode_2d`, `tile_from_linear`, lines 21-38) for cache-friendly tile
   traversal; falls back to row-major when the grid is not square-power-of-two.

The 7 TensorOps entry points available to copy from **[verified, re-counted today]**:

| Entry | Line |
| --- | --- |
| `matmul2d_tensorops_f32` | `matmul_tensorops.metal:44` |
| `matmul2d_tensorops_tn_f32` | `:93` |
| `matmul2d_tensorops_nt_f32` | `:140` |
| `matmul2d_tensorops_tn_accum_f32` | `:186` |
| `matmul2d_tensorops_nt_accum_f32` | `:231` |
| `matmul2d_tensorops_tn_splitk_f32` | `:276` |
| `matmul2d_tensorops_tn_splitk_bf16_f32` | `:318` |

### A.5 Header / include pattern

Four shared headers exist. Two guard styles are in use — both work, follow the neighbouring file.

| Header | Guard | Exports | Included by |
| --- | --- | --- | --- |
| `gelu.h` | `#ifndef TESSL_GELU_H` (`:1-2`) | `tessl_gelu_pytorch_tanh(float)` (`:16`) | `gemv_q4_mlx.metal:9` |
| `softcap.h` | `#ifndef TESSL_SOFTCAP_H` (`:1-2`) | `tessl_apply_softcap(float,float)` (`:10`) | `softcap_sample.metal:4`, `utils.metal:3` |
| `reduce_tree.h` | `#pragma once` (`:13`) | row-reduction primitives, `REDUCE_MAX_TG = 1024` (`:20`) | `reduce.metal:14`, `rms_norm.metal:21` |
| `q4_mlx_dot.h` | `#pragma once` (`:10`) | `load_x16_qdot`, `qdot16` | `gemv_q4_mlx.metal:10`, `gemm_q4_mlx.metal:3` |

Every header opens `#include <metal_stdlib>` + `using namespace metal;` and is include-only — `build.rs`
compiles only `*.metal`, stated in-file at `reduce_tree.h:9-12` and `q4_mlx_dot.h:7-9`. **[verified]**

**The naming convention is `tessl_`-prefixed `static inline` for anything shared.** `reduce_tree.h:2-7` gives
the reason a helper goes in a header at all: each `.metal` is a separate translation unit, so without a header
the second file to need a reduction copies it, and "two copies of a reduction that must agree on lane
assignment and barrier placement is a defect waiting for one of them to be edited."

There is a live ODR hazard here worth knowing: `mlp_gelu_tanh.metal` in the **nested copy** still carries a
file-local `static inline gelu_pytorch_tanh_mlp` with the comment *"File-local (`static`) so metallib link
does not ODR-merge with `gelu_pytorch_tanh` in gemv_q4_mlx.metal"*. Canonical tessl already fixed this by
moving to the shared `tessl_gelu_pytorch_tanh` (§F). **Use the shared header; do not reintroduce a file-local
copy.** **[verified]**

---

## B. Host dispatch contract

### B.1 Module map

`src/lib.rs:192-205` — `pub mod nn` (`:201`) is where a kernel with a shape contract belongs;
`pub mod ops` (`:203`) is for contract-free elementwise ops. `src/ops.rs:7-9` states the split explicitly:

> Anything with a shape contract worth checking belongs in [`crate::nn`] instead, which validates operands
> before encoding. The split is by whether there is an invariant to enforce, not by kernel size.

**K1–K7 all have shape contracts → all go in `src/nn.rs`.** **[verified]**

### B.2 Pipeline state creation

`rt.pipeline(name)` → `Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String>`
(`src/runtime.rs:871-894`). It is **cached and name-keyed**; resolves the primary library first, then
registered overlays. Calling it records kernel use (`record_kernel_use(name)`, `:875`) and has a separate
`icb:` key namespace when ICB pipelines are enabled (`:877-889`). **[verified]**

There is no descriptor to build and no function-constant plumbing on this path — **the entry-point name is the
whole API.** A head-dim or tile-size variant is a *separate entry point*, not a parameter; `src/nn.rs:1075-1086`
makes this explicit for flash attention:

> Each kernel bakes `HEAD_DIM` in as a `constant`, so the head dimension selects the kernel rather than being
> passed to it. Calling one with the wrong `D` reads past the end of every head.

### B.3 Argument binding — there is no `setBytes`

Binding goes through `Binder` (`src/dispatch.rs:20`). Encode is **Metal 4 only** (`MTL4ArgumentTable`);
`Cargo.toml` prunes the classic `MTLCommandQueue/Encoder` features with the comment *"encode is Metal 4 only
(MTL4*)"*. Free functions wrap the methods (`src/dispatch.rs:539-558`) **[verified]**:

| Helper | Binds | Line |
| --- | --- | --- |
| `set_gpu_buf(bnd, buf, index)` | whole `GpuBuffer` | `dispatch.rs:543` |
| `set_gpu_buf_offset(bnd, buf, byte_offset, index)` | byte-offset view, no host round-trip | `:548` |
| `set_tensor(bnd, t, index)` | `Tensor` | `:539` |
| `set_u32(bnd, v, index)` | scalar → const arena | `:552` |
| `set_f32(bnd, v, index)` | scalar → const arena | `:556` |

Scalars are **materialised into a runtime-owned const arena**, not `setBytes`: `bind_u32`/`bind_f32`
(`dispatch.rs:323-330`) call `bind_bytes` (`:308`), which returns a GPU address. This is why every
`*_with_scalars` entry point is `unsafe` — the caller promises to bind only the documented slots, and the
runtime owns the storage through execution (see the SAFETY comment at `nn.rs:2441-2443`).

**Buffer index = the `[[buffer(n)]]` attribute in the `.metal` signature.** The mapping is positional and
manual; nothing checks it. `set_threadgroup_memory` shares the same slot space (`dispatch.rs:332-350`).

### B.4 Threadgroup sizing

Five dispatch helpers. Pick by grid shape **[verified]**:

| Helper | Grid | Threads/TG | TG memory | Line |
| --- | --- | --- | --- | --- |
| `dispatch_1d(rt, p, n, f)` | `n.div_ceil(tpt)` × 1 × 1 | `threadExecutionWidth().min(n)` | no | `dispatch.rs:565` |
| `dispatch_2d(rt, p, nx, ny, f)` | `nx.div_ceil(tx)` × `ny` × 1 | `threadExecutionWidth().min(nx)` | no | `:589` |
| `dispatch_3d(rt, p, nx, ny, nz, f)` | … × `ny` × `nz` | same | no | `:614` |
| `dispatch_2d_tg(rt, p, gx, gy, tptg, f)` | `gx` × `gy` × 1 | **caller-fixed** | no | `:666` |
| `dispatch_tg_1d(rt, p, groups, tptg, tg_mem, f)` | `groups` × 1 × 1 | **caller-fixed** | **yes** | `nn.rs:1051` |

`dispatch_tg_1d` is private to `nn.rs` and is **the only helper that takes threadgroup memory** — K1's device-
buffer design means it will likely use this with `tg_memory: None`, or `dispatch_2d_tg` for a 2D tile grid.

```rust
// src/nn.rs:1051-1071
fn dispatch_tg_1d(
    rt: &Arc<GpuRuntime>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    groups: usize,
    threads_per_tg: usize,
    tg_memory: Option<(usize, usize)>,   // (index, bytes)
    encode: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if groups == 0 || threads_per_tg == 0 {
        return Ok(());
    }
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode(bnd);
        if let Some((index, bytes)) = tg_memory {
            bnd.set_threadgroup_memory(index, bytes);
        }
        bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(threads_per_tg, 1, 1));
        Ok(())
    })
}
```

Every `dispatch()` (`dispatch.rs:356-403`) validates geometry against `pipeline.maxTotalThreadsPerThreadgroup()`
(latched at `dispatch.rs:183`) and **auto-inserts a `Dispatch→Dispatch` Device barrier afterwards** unless
`METAL_RUNTIME_HAZARD_BARRIERS=1`. A packed multi-dispatch op that needs RAW/WAR ordering inside one encoder
calls `bnd.barrier()` explicitly (`dispatch.rs:405`). **K7/K3/K2 chained on one encoder get ordering for free
by default.** **[verified]**

Two sizing helpers exist for reductions: `reduce_tptg(...)` and `reduction_tptg(max_threads, want, tg_array_len)`
(`nn.rs:2782`). The latter **rounds down to a power of two** — `nn.rs:2776-2781` explains that the halving
reduction loop reads `tg_val[lid + stride]`, so a non-power-of-two threadgroup "drops elements silently — it
produces a plausible token rather than an error."

### B.5 Threadgroup memory limit — MEASURED, and the plan is right

`GpuRuntime::max_threadgroup_memory()` (`src/runtime.rs:622-624`) returns `device.maxThreadgroupMemoryLength()`.

I measured it on this machine rather than trusting the docstring:

```
name: Apple M5 Pro
maxThreadgroupMemoryLength: 32768 bytes = 32 KiB
maxThreadsPerThreadgroup: MTLSize(width: 1024, height: 1024, depth: 1024)
supportsFamily(apple9): true    supportsFamily(metal4): true
```

**[verified — Swift/Metal probe against `MTLCreateSystemDefaultDevice()`, 2026-09-19]**

**The plan's premise holds:** a 128×128 fp32 GDN state is `128*128*4 = 65536` bytes = **exactly 2× the limit**.
It cannot live in threadgroup memory on this hardware. K1 must use a device buffer. **[verified]**

> Note the docstring at `src/runtime.rs:620-621` says *"Apple guarantees at least 32 KiB; current Apple silicon
> reports more."* On this M5 Pro it reports **exactly** 32768. See CORRECTION 7.

**The precedent to copy** — check before encoding, because the dispatch-time failure is useless.
`src/nn.rs:3591-3611`:

```rust
let (groups, tptg, tg_mem) = if variant == Q4MlxRowVariant::Tiled {
    (shape.rows as usize, GEMV_TILED_TPTG, None)
} else {
    let bytes = (shape.cols as usize).saturating_mul(4);
    let limit = rt.max_threadgroup_memory();
    if bytes > limit {
        return Err(format!(
            "{entry}: caching x needs {bytes} bytes of threadgroup memory but this \
             device allows {limit}; cols {} is too large for this kernel",
            shape.cols
        ));
    }
    ...
    ((shape.rows as usize).div_ceil(t), t, Some((0, bytes)))
};
```

`nn.rs:3195-3197` states the rule: *"`cols * 4` bytes must fit the device limit — checked against
`GpuRuntime::max_threadgroup_memory` before encoding, because the dispatch-time failure names neither this
kernel nor `cols`."* There is also a static ceiling `GEMV_X_TILE = 4096` (`nn.rs:3633`; 4096×4 = 16 KiB).

### B.6 Device buffer allocation

`rt.alloc_buffer(nbytes) -> Result<GpuBuffer, String>` (`src/runtime.rs:904-906`), defaulting to
`BufferKind::Cold`. Test helpers wrap it (`tests/common/mod.rs:239-262`): `buf`, `empty` (zeroed),
`buf_bf16`, `seeded(rt, elems, value)`. `GpuBuffer` carries `.zero()`, `.write_f32()`, `.read_f32()`,
`.write_bf16_bits()`, `.write_u32()`, `.read_u32()`, `.aliases(other)`, `.metal()`. **[verified]**

### B.7 Validation helpers — use these, the kernels have no bounds checks

`tests/nn_adversarial.rs:3-6` states the stakes: *"These kernels index device memory with no bounds check of
their own — that is deliberate and documented, and it is exactly why the host-side validation has to be right.
A missed check here is not a wrong number, it is a read or write of arbitrary device memory."*

| Helper | Does | Line |
| --- | --- | --- |
| `require::<T>(rt, buf, need, what)` | runtime match **+** capacity ≥ `need` elements of `T` | `nn.rs:107-110` |
| `require_disjoint_writes(entry, writes, reads)` | every write-vs-write and write-vs-read pair via `.aliases()` | `nn.rs:112-134` |
| `validate_rms_scalars(dim, eps, what)` | `dim != 0`, `eps` finite and `> 0` | `nn.rs:136-144` |
| `elems_product(&[..], what)` | checked multiply, named overflow error | used at `nn.rs:2534-2535` |

---

## B.8 ★ COMPLETE WORKED EXAMPLE — `rms_qkv_rope`, Rust call site → dispatch → `.metal`

Chosen over `flash_attn_swa_h256` because it is the kernel K5 extends, it exercises **all three** binding
styles (const-arena scalar, device-buffer scalar, buffer), and it shows the variant-selects-entry-point idiom
K1/K6 will both need. All line numbers are canonical `/Users/bharath/Code/research/tessl`.

### Step 1 — Public entry point (`src/nn.rs:2431-2470`)

```rust
pub fn rms_qkv_rope(
    rt: &Arc<GpuRuntime>,
    variant: QkvRopeVariant,
    qkv: QkvBuffers<'_>,
    dims: QkvRopeDims,
    pos_offset: u32,
    pos_offset_buf: Option<&GpuBuffer>,
    kv_store: Option<KvStoreTarget<'_>>,
    q_only: bool,
) -> Result<(), String> {
    // SAFETY: the closure binds only the scalar slots documented for the
    // selected variant, with the same values this call validates; the runtime
    // owns their const-arena storage through execution.
    unsafe {
        rms_qkv_rope_with_scalars(
            rt, variant, qkv, dims, pos_offset_buf, kv_store, q_only,
            |bnd, kv_capacity| {
                set_u32(bnd, dims.t, 6);
                set_u32(bnd, dims.heads_q, 7);
                set_u32(bnd, dims.heads_kv, 8);
                set_u32(bnd, dims.head_dim, 9);
                set_u32(bnd, dims.rotary_dim, 10);
                if variant == QkvRopeVariant::PosConst {
                    set_u32(bnd, pos_offset, 11);
                }
                set_f32(bnd, dims.theta, 12);
                set_f32(bnd, dims.eps, 13);
                if let Some(capacity) = kv_capacity {
                    set_u32(bnd, capacity, 17);
                }
            },
        )
    }
}
```

**The pattern: a safe `pub fn` that owns the scalar-binding closure, wrapping an `unsafe pub fn *_with_scalars`
that takes the closure.** The unsafe seam exists so an ICB/stable-scalar adapter can substitute its own pool.
Copy this shape for every K-kernel.

### Step 2 — Types (`src/nn.rs:2308-2400`, `2474-2487`)

```rust
pub struct QkvRopeDims {          // :2308
    pub t: u32, pub heads_q: u32, pub heads_kv: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,          // :2318  ← ALREADY A PARAMETER (K5 claim confirmed)
    pub theta: f32, pub eps: f32,
}

pub enum QkvRopeVariant {         // :2370
    PosConst, PosBuffer, PosBufferKvStore,
}
impl QkvRopeVariant {             // :2382-2390
    fn entry(self) -> &'static str {
        match self {
            Self::PosConst        => "rms_qkv_rope",
            Self::PosBuffer       => "rms_qkv_rope_posbuf",
            Self::PosBufferKvStore=> "rms_qkv_rope_kv_store",
        }
    }
}

pub struct QkvBuffers<'a> {       // :2474
    pub q: &'a GpuBuffer, pub k: &'a GpuBuffer, pub v: &'a GpuBuffer,
    pub q_weight: &'a GpuBuffer, pub k_weight: &'a GpuBuffer, pub v_weight: &'a GpuBuffer,
}
```

### Step 3 — Validation, separately callable (`src/nn.rs:2496-2560+`)

`pub fn validate_rms_qkv_rope(...)` runs **before** any pipeline lookup or scalar staging, so a stable-scalar
adapter can preflight without reserving slots (`nn.rs:2489-2494`). It rejects:

- `dims.validate("rms_qkv_rope")` — dimension sanity.
- `q_only` + `PosBufferKvStore` — *"its cache guard can suppress Q"* (`:2507-2511`).
- variant/operand mismatch both ways: `PosConst` with a buffer, or a `PosBuffer*` variant without one
  (`:2517-2527`) — *"Accepting a mismatched pair and ignoring the extra one is how a caller ends up reading a
  stale position for a whole session."*
- `kv_store` present without `PosBufferKvStore`, or absent with it (`:2528-2535`).
- capacity: `require::<f32>` on `q`, `q_weight`, and (unless `q_only`) `k`, `v`, `k_weight`, `v_weight`, using
  `elems_product` + `checked_mul` for extents (`:2536-2551`).
- cache capacity `< kv_elems` (`:2552-2558`).

### Step 4 — The unsafe seam, geometry, and encode (`src/nn.rs:2671-2751`)

```rust
pub unsafe fn rms_qkv_rope_with_scalars(
    rt: &Arc<GpuRuntime>, variant: QkvRopeVariant, qkv: QkvBuffers<'_>,
    dims: QkvRopeDims, pos_offset_buf: Option<&GpuBuffer>,
    kv_store: Option<KvStoreTarget<'_>>, q_only: bool,
    scalars: impl FnOnce(&mut Binder<'_>, Option<u32>),
) -> Result<(), String> {
    validate_rms_qkv_rope(rt, variant, qkv, dims, pos_offset_buf, kv_store, q_only)?;
    let n = dims.head_count(q_only)?;
    if n == 0 { return Ok(()); }                       // zero work is Ok, not Err
    let kv_capacity = kv_store.map(|target| target.capacity);
    let p = rt.pipeline(variant.entry())?;             // ← name-keyed pipeline
    let (rows_per_tg, threads_per_tg) = rope_row_geometry(&p)?;
    dispatch_tg_1d(rt, &p, n.div_ceil(rows_per_tg), threads_per_tg, None, |bnd| {
        set_gpu_buf(bnd, qkv.q, 0);
        set_gpu_buf(bnd, qkv.q_weight, 3);
        if q_only {
            // Fixed argument-table ABI: bind validated Q storage rather than
            // caller placeholders the q-only grid never reaches.
            set_gpu_buf(bnd, qkv.q, 1);
            set_gpu_buf(bnd, qkv.q, 2);
            set_gpu_buf(bnd, qkv.q_weight, 4);
            set_gpu_buf(bnd, qkv.q_weight, 5);
        } else {
            set_gpu_buf(bnd, qkv.k, 1);
            set_gpu_buf(bnd, qkv.v, 2);
            set_gpu_buf(bnd, qkv.k_weight, 4);
            set_gpu_buf(bnd, qkv.v_weight, 5);
        }
        if let Some(b) = pos_offset_buf { set_gpu_buf(bnd, b, 11); }   // device u32
        if let Some(t) = kv_store {
            set_gpu_buf(bnd, t.dst_k, 14);
            set_gpu_buf(bnd, t.dst_v, 15);
            set_gpu_buf(bnd, t.dst_offset, 16);
        }
        // The unsafe stable-scalar seam runs LAST so an adapter may replace
        // slot 16 with a validated byte-offset view of `dst_offset`.
        scalars(bnd, kv_capacity);
        if q_only {
            // Grid is T*Hq rounded up to whole threadgroups; the kernel guard is
            // T*Hq + 2*T*Hkv. With the real Hkv bound, padding simdgroups fall
            // into the K/V branches and re-normalize rows other simdgroups own.
            // Force Hkv = 0 AFTER the callback so the guard is exactly the Q grid.
            set_u32(bnd, 0, 8);
        }
    })
}
```

Two ordering facts worth copying verbatim: **the caller's scalar closure runs after the fixed buffer binds**
(so an adapter can override a slot), and **a correction that must not be overridable runs after the closure**
(the `q_only` `Hkv = 0` clamp).

### Step 5 — Geometry from pipeline introspection (`src/nn.rs:2739-2772`)

```rust
const ROPE_SIMD_WIDTH: usize = 32;   // :2739
const ROPE_ROWS_PER_TG: usize = 8;   // :2742

fn rope_row_geometry(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(usize, usize), String> {
    let width = pipeline.threadExecutionWidth();
    if width != ROPE_SIMD_WIDTH {
        return Err(format!(
            "rms_qkv_rope: kernel assumes a {ROPE_SIMD_WIDTH}-lane simdgroup, \
             pipeline reports an execution width of {width}"));
    }
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    if max_threads < ROPE_SIMD_WIDTH {
        return Err(format!(
            "rms_qkv_rope: pipeline allows {max_threads} threads per \
             threadgroup, fewer than one simdgroup"));
    }
    let rows = (max_threads / ROPE_SIMD_WIDTH).min(ROPE_ROWS_PER_TG);
    Ok((rows, rows * ROPE_SIMD_WIDTH))
}
```

**A shader assumption about lane width is asserted against the pipeline, not assumed.** Do this in K1/K2/K3.

### Step 6 — The `.metal` entry point (`kernels/rms_qkv_rope.metal:103-154`)

```cpp
kernel void rms_qkv_rope(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_weight [[buffer(3)]],
    device const float *k_weight [[buffer(4)]],
    device const float *v_weight [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &Hq [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant uint &rotary_dim [[buffer(10)]],
    constant uint &pos_offset [[buffer(11)]],
    constant float &theta [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    const ulong total_q  = (ulong)T * (ulong)Hq;
    const ulong total_kv = (ulong)T * (ulong)Hkv;
    const ulong gid64 = row_of(tg, sg, tptg);
    if (gid64 >= total_q + 2ul * total_kv) return;
    ...
}
```

Note the ABI contrast across variants **[verified]**:
`rms_qkv_rope` takes `constant uint &pos_offset [[buffer(11)]]`, while `rms_qkv_rope_posbuf` (`:159`) and
`rms_qkv_rope_kv_store` (`:217`) take `device const uint *pos_offset_ptr [[buffer(11)]]` — **same slot, different
storage class**, so an ICB that froze its binds can advance position without re-encoding. `rms_qkv_rope_kv_store`
adds slots 14/15 (`device float *dst_k/dst_v`), 16 (`device const uint *kv_dst_offset_ptr`) and 17
(`constant uint &kv_capacity`), and guards with widened arithmetic that cannot wrap
(`rms_qkv_rope.metal:250`): `if (kv_dst_offset > capacity || kv_span > capacity - kv_dst_offset) return;`

### Step 7 — Shared row helper (`kernels/rms_qkv_rope.metal:36-100`)

`template <bool STORE> inline void norm_rope_row(...)` — one simdgroup per head row, `simd_sum` for the sum of
squares, each lane owning exactly one RoPE pair so no cross-lane exchange is needed. The file header
(`:16-23`) records that this replaced a one-thread-per-row version and that the reassociated sum is why
`tests/qkv_rope.rs` compares against an **f64** reference rather than a matching f32 accumulation. **Relevant to
K7/K3: a GDN state update reassociated across lanes needs an f64 reference too.**

---

## C. Test harness style

### C.1 Location and shape

Integration tests are `tests/*.rs`, each its own binary, each starting `mod common;`. `tests/common/mod.rs`
is shared scaffolding (`tests/common/mod.rs:1-6`):

> Everything here goes through the public API only … Nothing in this module may reach into `pub(crate)`
> internals; if a check cannot be written from outside, it belongs in a `src/` unit test instead.

**`tests/gdn.rs` must therefore only use `tessl::nn::` public API.** **[verified]**

### C.2 GPU access — always `with_gpu`

```rust
// tests/common/mod.rs:22-39
static GPU: Mutex<()> = Mutex::new(());        // one GPU at a time regardless of --test-threads
pub fn with_gpu<R>(f: impl FnOnce(&Arc<GpuRuntime>) -> R) -> R {
    let _guard = gpu_lock();
    let rt = GpuRuntime::new().expect("GpuRuntime::new (needs a Metal 4 device + built metallib)");
    f(&rt)
}
```

A fresh runtime per test is deliberate (`:31-34`): `set_precision` / `set_relaxed_precision` /
`set_async_encode` are runtime-wide mutable state, so sharing would make failures order-dependent. The mutex
recovers from poisoning (`:25-28`) so one real failure does not cascade. `with_two_gpus` (`:46`) exists for
cross-runtime rejection tests. **[verified]**

> There is **no GPU-absent skip path** in canonical tessl — `with_gpu` *panics* if no Metal 4 device. Tests
> do run here (306 passed). Contrast the nested copy's `gemma-metal`, which has a `gpu_or_skip()`.

### C.3 Tolerances — three distinct idioms, pick deliberately

**(a) Derived error bound** — the rigorous one, for anything GEMM-shaped (`tests/common/mod.rs:160-234`):

```rust
pub const U_F32:  f64 = 5.960_464_477_539_063e-8;  // 2^-24
pub const U_BF16: f64 = 3.906_25e-3;               // 2^-8

fn gamma(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    assert!(nu < 1.0, "error bound degenerate at K={n}; shrink K");
    nu / (1.0 - nu)
}

pub fn tolerance(k: usize, mag: f64, operand_u: f64) -> f64 {
    (gamma(k + 8, U_F32) + 2.0 * operand_u) * mag
}

pub fn assert_within_bound(label: &str, got: &[f32], r: &Reference, k: usize, operand_u: f64)
```

Critically, the bound scales with `mag` = `sum_k |a_ik * b_kj|`, **not** `|c|` (`mod.rs:117-122`):

> The f32 accumulation bound is proportional to this, not to `|c|`: a row whose terms cancel has a small
> result and a large error budget, and judging it by `|c|` alone would demand accuracy the format cannot give.

`reference(layout, ...)` computes in **f64** (`mod.rs:126-158`) so the reference is not itself a source of
error. `assert_within_bound` reports the worst offender **scaled by its own budget** ("`is 1.7x its error
budget`") rather than just a raw delta. `operand_u` is 0 when the test rounds operands to bf16 before upload
via `round_trip_bf16` (`mod.rs:84-89`) — then the GPU reads exactly the reference's values.

**This is the right idiom for K7/K3/K2 GDN parity** — a delta-rule state update is a K-term accumulation.

**(b) Relative-plus-absolute** — the pragmatic one, for fused/activation kernels
(`tests/common/mod.rs:313-320`):

```rust
pub fn close_rel(what: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}]: non-finite {g}");
        let tol = rel * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{what}[{i}]: got {g} want {w}");
    }
}
```

Used e.g. `close_rel("gate_up_gelu_i4", &mid.read_f32()[..rows], &want, 5e-3)` — `5e-3` is the observed Q4
fused-path tolerance. Note `.max(1.0)` gives an absolute floor so near-zero values do not demand infinite
relative precision.

**(c) Bitwise** — for determinism only (`tests/gemm_determinism.rs:28-45`), `f32::to_bits()` not `==`.

### C.4 ★ `src/npy.rs` — NOT a test fixture loader. Do not plan around it.

**[verified]** Three independent checks:

1. `rg -n 'npy' tests/` → **zero hits**. The only mentions crate-wide are `src/lib.rs:76` (a doc table) and
   `src/lib.rs:202` (`pub mod npy`).
2. `rg --files -uu tessl --glob '*.npy'` → **0 files**. There are no `.npy` fixtures on disk anywhere in the crate.
3. `rg -n -uu '\bread_npy\b' /Users/bharath/Code` → the **only** source occurrence in the entire `~/Code` tree
   is its own definition at `research/tessl/src/npy.rs:49`. (Other hits are DevMap's own `code_graph.json`
   indexing a *different* `read_npy` in `MLSystemsLab/Rust_MLKit/arch_02_value_resid/metal-native/src/npy.rs`.)

So: **`read_npy` is dead code in canonical tessl, and `write_npy_f32` has exactly one caller** —
`src/bin/bench_gemm_sweep.rs:12,185-187`, which *writes* `parity_a.npy` / `parity_b.npy` for the Python
benchmark lanes to read.

`src/npy.rs:1-14` states the scope plainly: *"This exists for benchmark parity, not as a general-purpose
format library."* C-order only, `<f4` and `<i8` only, Fortran order is a hard error.

**Implication for `tests/gdn.rs`:** there is no existing fixture-loading path to follow. Either (a) generate
operands in-test with `random_f32(n, seed)` + `Rng` splitmix64 (`tests/common/mod.rs:53-79`) — which is what
**every** existing tessl test does — or (b) build a new `.npy` loading path, in which case `read_npy` is
already written and tested-by-nobody, and you would be its first caller. **Option (a) matches house style.**
If the plan needs a Python-generated golden (it does, per repo rule 9 about the `published` GDN operator),
that is a *new* seam, and its first job is to pin `rule="published"` in the fixture filename and header.

### C.5 Adversarial / aliasing / recycled-scratch idioms `tests/gdn.rs` must mirror

**Aliasing** (`tests/nn_aliasing.rs:32-43`) — the assertion is *refused before encoding*:

```rust
fn assert_alias_refused(rt: &GpuRuntime, label: &str, result: Result<(), String>) {
    let err = result.expect_err("aliased output must be refused");
    ...
    assert_eq!(
        rt.take_dispatch_count(), 0,
        "{label}: encoded GPU work before refusing the alias"
    );
}
```

`rt.take_dispatch_count()` (`src/runtime.rs:836-841`) reads and resets. **This is the mechanism that separates
"refused" from "refused after already dispatching."** `tests/nn_aliasing.rs:3-6` explains why the test observes
the host boundary rather than trying to detect corruption after the fact: the old implementation would launch
unordered reads/writes, which is nondeterministic.

Host-side aliasing is enforced by `require_disjoint_writes` (`nn.rs:112-134`) — K2/K3/K7 must call it, listing
the GDN state buffer as a **write** (it is read-modify-write across timesteps).

**Adversarial** (`tests/nn_adversarial.rs:8-11`) — threefold assertion:

> the call returns `Err`, it does **not** panic, and it encodes nothing.

Its six tests are a template for `gdn.rs`: `undersized_buffers_are_refused_across_the_whole_surface` (`:54`),
`zero_and_degenerate_dimensions_never_panic` (`:283`), `non_finite_scalars_are_refused_rather_than_propagated`
(`:312`), `rms_norm_refuses_non_positive_or_non_finite_eps` (`:372`),
`dimension_products_that_overflow_are_refused_not_wrapped` (`:396`),
`the_runtime_is_still_usable_after_the_whole_hostile_sweep` (`:441`).

**Recycled scratch** — the sentinel idiom, `seeded()` + a value no correct kernel would produce:

```rust
// tests/common/mod.rs:259-262
/// Buffer seeded with `value`, for detecting elements a kernel never wrote.
pub fn seeded(rt: &Arc<GpuRuntime>, elems: usize, value: f32) -> GpuBuffer
```

Used as `SENTINEL` in `tests/kv_safety.rs` (16 sites, e.g. `:61,66-67,88-89,148-149`) and `UNWRITTEN` in
`tests/attention.rs` (`:176,236-237,275,363-364,662`). This is exactly the bug class the SWA kernel's own
comment documents (`flash_attn_swa_h256.metal:~108-116`): a guarded `if (lid < BR*BC)` left threadgroup
entries 32..63 untouched, and `+=` then accumulated into whatever a *previous dispatch* left there —
*"plausible numbers, not NaN, so nothing looked wrong."*

**For K1's device-resident 128×128 state this is the #1 correctness risk.** A device buffer is recycled across
dispatches exactly like threadgroup memory. `tests/gdn.rs` must seed the state buffer with a sentinel and
assert every live element was overwritten, and must run the same shape twice in a row on one runtime.

**Determinism** (`tests/gemm_determinism.rs:1-14, 24-26`) — `REPEATS: usize = 48`, chosen to cross
command-buffer and allocator-slot boundaries, because "the Metal 4 package ping-pongs allocators, so a handful
of runs could all land in the same slot and prove nothing about the other one." Compare `to_bits()`.

### C.6 ★ Meta-tests that will break when you add kernels

**`tests/promoted_kernels.rs` hard-codes 44.** `promoted_count_matches_the_kernel_sources` (`:120-131`):

```rust
assert_eq!(
    PROMOTED.len(), 44,
    "PROMOTED drifted from the 44 entry points in the promoted .metal sources; \
     re-run: grep -hc '^kernel void' kernels/{{rms_norm,mlp_silu,mlp_gelu_tanh,\
flash_attn_swa_h128,flash_attn_swa_h256,flash_attn_global_h512,rms_qkv_rope,kv_store,\
gemv_q4,gemv_q8,gemv_q4_mlx,gemm_q4_mlx,embed_lookup,softcap_sample}}.metal"
);
```

I re-measured today — the number is currently **exactly right**, which is why it will bite:

| source | entries | | source | entries |
| --- | ---: | --- | --- | ---: |
| `rms_norm.metal` | 3 | | `kv_store.metal` | 3 |
| `mlp_silu.metal` | 1 | | `gemv_q4.metal` | 2 |
| `mlp_gelu_tanh.metal` | 2 | | `gemv_q8.metal` | 1 |
| `flash_attn_swa_h128.metal` | 1 | | `gemv_q4_mlx.metal` | 15 |
| `flash_attn_swa_h256.metal` | 1 | | `gemm_q4_mlx.metal` | 4 |
| `flash_attn_global_h512.metal` | 1 | | `embed_lookup.metal` | 3 |
| `rms_qkv_rope.metal` | 3 | | `softcap_sample.metal` | 4 |
| | | | **TOTAL** | **44** |

`PROMOTED` array length: **44**. **[verified — both sides counted today]**

**Consequence:** if K5 adds an entry point to `rms_qkv_rope.metal`, or K6 adds a SiLU sibling inside
`gemv_q4_mlx.metal`, or anything lands in `rms_norm.metal` / `mlp_silu.metal`, **this test fails** until you
add the name to `PROMOTED` *and* bump the literal `44`. Two other tests in the same file also apply:
`every_promoted_kernel_resolves_from_tessls_own_metallib` (`:89`) and `the_promoted_list_has_no_duplicates`
(`:106`).

**Escape hatch:** entry points in a **new** `.metal` file are not covered by this test at all. The 6 unlisted
sources today are `flash_attn_rows` (1 literal → **27** real), `flash_attn_decode` (2 → **36**),
`matmul_tensorops` (7), `matmul_simdgroup` (2), `reduce` (3), `utils` (11) — 26 literal, **86 real** (see
CORRECTION 7 on why those differ). **Putting K1–K7 in new files (e.g. `gdn.metal`) avoids the 44-assert
entirely** — and is the right call anyway for K1 (§CORRECTION 1).

Note the flip side: those 86 entry points have **no** automated "does this kernel resolve from the metallib"
guard. If K1–K7 land in a new file, add your own resolve test — a `gdn.rs` analogue of
`every_promoted_kernel_resolves_from_tessls_own_metallib` (`promoted_kernels.rs:89`) costs four lines and is
the only thing standing between a typo'd entry name and a `NotFound` at a downstream caller's first dispatch.

**`tests/docs_name_real_tools.rs`** — every `` `bench_*` `` identifier in README/docs markdown must be a real
`[[bin]]`. If you document a new `bench_gdn`, add the bin target or the test fails (`:34-45`).

**`tests/build_artifact_contract.rs`** — executes the real `build.rs` with fake compiler tools from two
concurrent build dirs and snapshots the source tree, asserting the build script never mutates
`CARGO_MANIFEST_DIR` (`:1-8`). Keep any new build-script logic out-dir-only.

### C.7 Fixture location

There is **no** fixture directory in tessl today (`.npy` count = 0, §C.4). The sibling repo has
`/Users/bharath/Code/research/qwen-decision/fixtures/` (empty as of this audit). Per `qwen-decision/CLAUDE.md`
("Kernel work (K1-K7, `tests/gdn.rs`, fixtures) lands in `~/Code/research/tessl`"), fixtures belong in tessl —
so this is a **new directory you will be creating**, with no precedent to match. Recommend `tessl/tests/fixtures/`
and a filename that names the operator rule (repo rule 9): e.g. `gdn_published_h128_t64.npy`.

---

## D. Existence check of every kernel the plan names

All six exist. Signatures re-read from source today. **[verified]**

| Plan reference | Exists? | File | Entry point(s) | Line |
| --- | --- | --- | --- | --- |
| `matmul_tensorops.metal` | **YES** | `kernels/matmul_tensorops.metal` | 7 entries (§A.4) | `:44,93,140,186,231,276,318` |
| `rms_qkv_rope.metal` | **YES** | `kernels/rms_qkv_rope.metal` | `rms_qkv_rope`, `rms_qkv_rope_posbuf`, `rms_qkv_rope_kv_store` | `:103,159,217` |
| `gemv_q4_mlx.metal` | **YES** | `kernels/gemv_q4_mlx.metal` | 15 entries | `:125`…`:961` |
| a `gate_up_gelu` entry point | **YES — but 3 of them, not 1** | `kernels/gemv_q4_mlx.metal` | `gemv_q4_mlx_blocked_gate_up_gelu`, `gemv_q4_mlx_simd_gate_up_gelu`, `gemv_q4_mlx_simd_gate_up_gelu_i4` | `:291,484,708` |
| `rms_norm.metal` | **YES** | `kernels/rms_norm.metal` | `rms_norm_f32`, `rms_norm_bf16`, `rms_norm_residual_add_f32` | `:45,72,98` |
| `mlp_silu.metal` | **YES** | `kernels/mlp_silu.metal` | `mlp_silu` (only) | `:11` |
| `flash_attn_swa_h256.metal` | **YES** | `kernels/flash_attn_swa_h256.metal` | `flash_attn_swa_h256` | `:18` |

### D.1 K5 claim — `rotary_dim` is already a parameter: **CONFIRMED**

Three independent places **[verified]**:

- Shader: `constant uint &rotary_dim [[buffer(10)]]` — `kernels/rms_qkv_rope.metal:113` (and `:169`, `:227`
  for the other two variants; same slot in all three).
- Host type: `pub rotary_dim: u32` — `src/nn.rs:2318`, documented *"Leading slice of each head that RoPE
  rotates. `<= head_dim`."*
- Bind site: `set_u32(bnd, dims.rotary_dim, 10);` — `src/nn.rs:2456`.

Semantics (`kernels/rms_qkv_rope.metal:50-58`): proportional NeoX / non-traditional RoPE. `rotary_dim == D` is
full NeoX; `rotary_dim < D` is global p-RoPE where only the first `rotary_dim/2` pairs rotate and the inv-freq
denominator uses the **full** head `D`. **K5 does not need to add this parameter.**

### D.2 K5 claim — `norm_mask` for QK-norm-without-V-norm: **genuinely new, but read this first**

There is **no `norm_mask`** anywhere in tessl (`rg` returns nothing). V-norm is currently **unconditional**:
all three variants call `norm_rope_row` on the V row with `v_weight` and `rotate=false`
(`kernels/rms_qkv_rope.metal:148-153`, `:204-209`, `:272-277`), commented *"V-norm: weight RMS only, no RoPE,
no attn scale."*

**The nearest existing precedent is `q_only: bool`**, which is threaded through `rms_qkv_rope`,
`validate_rms_qkv_rope` and `rms_qkv_rope_with_scalars`. But it is **not** what K5 wants — `q_only` skips K
*and* V. K5 needs "normalize Q and K, skip V," which is a third mode.

Two design cautions from the existing code, both load-bearing:

1. `q_only` is implemented by **clamping `Hkv = 0` after the scalar closure** (`nn.rs:2740-2748`) so the
   kernel's guard `gid64 >= total_q + 2ul*total_kv` becomes exactly the Q grid. A `norm_mask` that instead
   leaves the V rows in the grid but no-ops them wastes those simdgroups; one that shrinks the guard must keep
   the Q/K/V branch arithmetic consistent.
2. `q_only` is **refused** with `PosBufferKvStore` (`nn.rs:2507-2511`) because the cache guard can suppress Q.
   A new mask interacts with the same guard — validate the combination explicitly rather than allowing it by
   omission.

### D.3 K6 claim — clone `gate_up_gelu` swapping GELU→SiLU: exists, **but the plan undercounts by 3×**

The three entries differ by *dispatch strategy and weight layout*, not just activation
(`src/nn.rs:3877-3883`, the `_with_scalars` variant selector):

```rust
let entry = match dispatch {
    GateUpDispatch::Simd(l) => match l {
        Q4MlxLayout::RowMajor     => "gemv_q4_mlx_simd_gate_up_gelu",
        Q4MlxLayout::Interleaved4 => "gemv_q4_mlx_simd_gate_up_gelu_i4",
    },
    GateUpDispatch::Blocked => "gemv_q4_mlx_blocked_gate_up_gelu",
};
```

They also differ in **input dtype**: `Blocked` takes `device const float *x` and caches it in threadgroup
memory (`gemv_q4_mlx.metal:298`, host `require::<f32>` at `nn.rs:3897`); both `Simd` variants take
`device const bfloat *x` (`:491`, `:715`, host `require::<u16>` at `nn.rs:3894`). `Blocked` additionally
rejects `cols > GEMV_X_TILE` (4096) at `nn.rs:3884-3889`.

The activation call site is a single line in each — `float v = tessl_gelu_pytorch_tanh(partial_g[lid]) * partial_u[lid];`
(`:363`) and `float v = tessl_gelu_pytorch_tanh(gsum) * usum;` (`:539`, `:758`) — so the *swap* is trivial, but
**K6 is 3 kernels + 3 `PROMOTED` entries + a `44 → 47` bump, not 1**, if it targets parity with the GELU family.
Note also there is **no `silu.h`**: `mlp_silu.metal` computes SiLU inline and is the only SiLU in the crate.
A K6 that adds a shared `tessl_silu` helper should create `silu.h` following the `gelu.h` pattern (§A.5).

---

## E. Build and test health — measured 2026-09-19, not quoted

### E.1 `cargo build`

```
$ cargo build --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml
   Compiling tessl v0.2.0 (/Users/bharath/Code/research/tessl)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.39s
[exited with code 0]
```

**Compiles. Exit 0. Zero warnings** (`rg -n 'warning'` over the full run → no hits). No metal3.2 fallback
warnings, i.e. **every kernel compiles under `-std=metal4.0` today**. **[verified]**

### E.2 `cargo test --no-run`

Exit 0 — all test binaries link. **[verified]**

### E.3 `cargo test` (full suite)

**Exit 0. 306 passed, 0 failed, 1 ignored, 0 measured, 0 filtered out**, across 37 test targets.
The GPU was present and used — these are real dispatches, not skips. **[verified]**

The one ignored test is `gemm::stress_tests::gemm_fuzz_deep` (a `src/lib.rs` unit test, `#[ignore]`d as a deep
fuzz). **Nothing was skipped for lack of a GPU.**

| Target | pass | Target | pass |
| --- | ---: | --- | ---: |
| `src/lib.rs` unit | 94 (+1 ignored) | `nn_decode_hazards` | 3 |
| `attention` | 26 | `nn_kernels` | 15 |
| `bf16_cast` | 1 | `nn_runtime_ownership` | 4 |
| `build_artifact_contract` | 1 | `nn_wiring` | 14 |
| `docs_name_real_tools` | 3 | `promoted_kernels` | 4 |
| `f16` | 6 | `promoted_numeric` | 8 |
| `gemm_batched` | 9 | `public_api_errors` | 10 |
| `gemm_correctness` | 10 | `q4_interleaved` | 7 |
| `gemm_determinism` | 8 | `q4_shape_domain` | 2 |
| `gemm_epilogue` | 6 | `qkv_rope` | 4 |
| `gemm_flag_paths` | 3 | `quantized_gemv` | 2 |
| `gemm_i8` | 5 | `reductions` | 6 |
| `gemm_ragged_shapes` | 9 | `runtime_lifecycle` | 14 |
| `interop_mtl_buffer` | 3 | `shader_index_arithmetic` | 6 |
| `kv_safety` | 6 | Doc-tests | 4 |
| `nn_adversarial` | 6 | 5 `bench_*`/`probe_*` bins | 0 each |
| `nn_aliasing` | 7 | | |

Slowest: `attention` 12.24s, `src/lib.rs` 3.15s, `gemm_correctness` 1.76s. Whole suite well under a minute.

**Baseline for the kernel author: 306/0/1. Any number below 306 passing after your change is a regression.**

### E.4 Environment

Apple **M5 Pro**, unified memory, `recommendedMaxWorkingSetSize` 49152 MiB. `supportsFamily`: apple7/8/9 ✓,
metal3 ✓, **metal4 ✓**. `maxThreadsPerThreadgroup` 1024×1024×1024. **[verified — Metal probe]**

---

## F. The nested-copy hazard — GAP-TESSL-COPY-DRIFT

**Confirmed: `/Users/bharath/Code/research/MLSystemsLab/Rust_MLKit/crates/tessl` exists.** **[verified]**

It is **not** a submodule and **not** its own repo: `Rust_MLKit/.git` and `crates/tessl/.git` do not exist;
the enclosing git repo is `/Users/bharath/Code/research/MLSystemsLab/.git`. So it is a plain vendored directory
tracked by MLSystemsLab. Repo rule 6 (`qwen-decision/CLAUDE.md`) forbids touching it. **This audit only read it.**

### F.1 Drift, measured

`diff -rq` excluding `target/`, `.git/`, `.devmap/`, `.DS_Store`, `default.metallib`:

- **45 files differ**
- **6 paths only in canonical**: `.github`, `assets`, `docs`, `scripts/check_release_ready.sh`,
  `scripts/ci_local.sh`, `scripts/wave2_nn_transform.py`
- **41 paths only in nested**: `bench/__pycache__`, `scripts/__pycache__`, 7 `bench/*.py` files
  (`attn_paired.py`, `attn_tune.py`, `benchmark_evidence.py`, `flash_attn_torch_mlx.py`,
  `kernel_coverage.py`, `parity_ladder.py`, `test_parity_harness.py`), `src/bin/bench_flash_attn.rs`,
  `src/bin/bench_gemm_variants.rs`, `src/bin/common`, and ~29 `bench/results/*.json|txt` measurement artifacts

Canonical is the **published** crate (has `.github`, `assets`, `docs`, release scripts, `homepage`/`docs.rs`
metadata, the `exclude` list). Nested is an **older working copy with extra benchmark scaffolding**.
Timestamps: canonical `src/nn.rs` 2026-09-18, nested 2026-09-05. Canonical is ~2 weeks ahead.

### F.2 Does nested have kernels canonical lacks? **NO.**

Both `kernels/` directories contain the **identical 24 filenames** (21 `.metal`, 4 `.h`, plus `tune/`).
**Only 2 kernel files differ in content** **[verified]**:

| File | canonical LOC | nested LOC | changed lines |
| --- | ---: | ---: | ---: |
| `matmul_tensorops.metal` | 882 | 851 | 41 |
| `mlp_gelu_tanh.metal` | 47 | 35 | 18 |

The `mlp_gelu_tanh.metal` diff is instructive — canonical is the **older** shape here: nested moved to the
shared header while canonical still carries the file-local copy.

```diff
 #include <metal_stdlib>
+#include "gelu.h"
 using namespace metal;

-/// File-local (`static`) so metallib link does not ODR-merge with
-/// `gelu_pytorch_tanh` in gemv_q4_mlx.metal.
-static inline float gelu_pytorch_tanh_mlp(float x) { ... }
-
-    out[gid] = gelu_pytorch_tanh_mlp(gate[gid]) * up[gid];
+    out[gid] = tessl_gelu_pytorch_tanh(gate[gid]) * up[gid];
```

Read the direction carefully: the `-` side is **canonical**, the `+` side is **nested**. So **nested has the
header refactor and canonical does not** — on this file canonical is the *older* shape.

#### F.2.1 A live inconsistency in canonical, and it is in the exact family K6 clones

Canonical tessl has **two** GELU implementations that disagree for `|x| > 20`. Both read verbatim today:

```cpp
// canonical kernels/gelu.h:20          — outer factor is the ORIGINAL x
return 0.5f * x * (1.0f + t);

// canonical kernels/mlp_gelu_tanh.metal:22 — outer factor is the CLAMPED xc
return 0.5f * xc * (1.0f + t);
```

`gelu.h:11-15` states which one is correct: *"Only the approximation's cubic is bounded: the outer factor must
remain the original `x` so GELU approaches `x` for large positive inputs instead of clipping at the guard
value."* So `gelu.h` is the fixed form; `mlp_gelu_tanh.metal`'s file-local `gelu_pytorch_tanh_mlp`
(`:14-23`, kept `static` to dodge an ODR merge, `:12-13`) still clips.

Now the part that matters for K6. Canonical `gemv_q4_mlx.metal:9` **includes `gelu.h`**, so all three
`gate_up_gelu` entries call `tessl_gelu_pytorch_tanh` — the **unclamped** form (`:363`, `:539`, `:758`). But
canonical's test for that kernel asserts the **clamped** form:

```rust
// canonical tests/q4_interleaved.rs:443-451  (test: gate_up_gelu_i4_matches_two_gemvs_and_a_gelu)
// The kernel's GELU: clamp, tanh formulation, as `nn::mlp_gelu_tanh`.
let xc = (*g as f64).clamp(-20.0, 20.0);
let inner = 0.797_884_560_802_865_4 * (xc + 0.044715 * xc * xc * xc);
(0.5 * xc * (1.0 + inner.clamp(-10.0, 10.0).tanh()) * (*u as f64)) as f32
```

versus the nested copy of the same test, which asserts the unclamped form and says why:

```rust
// nested tests/q4_interleaved.rs:443-451
// The kernel's GELU: clamp the cubic input, but retain the original
// projection in the outer factor so large values are not clipped.
(0.5 * (*g as f64) * (1.0 + inner.clamp(-10.0, 10.0).tanh()) * (*u as f64)) as f32
```

**[verified — both files read directly, not via DevMap]**

**Conclusion: canonical's test encodes the wrong reference for the kernel it tests.** Its comment says *"as
`nn::mlp_gelu_tanh`"*, but the kernel under test is `gemv_q4_mlx_simd_gate_up_gelu_i4`, which uses `gelu.h`,
not `mlp_gelu_tanh`. The two agree only while `|gate| <= 20`; operands come from `random_f32` (uniform
`[-1,1)`) through a Q4 GEMV, so the projection never gets near 20 and `close_rel(..., 5e-3)` passes
**vacuously**. The test is green because it never exercises the disagreement.

This refines the existing `GAP-TESSL-COPY-DRIFT-SEMANTIC` (logged by another lane), which correctly spotted
that the two test copies disagree and correctly said *"at most one of the two matches the actual .metal
kernel"* — but left open which. **It is the nested one that matches**; canonical's is wrong. Since rule 6
forbids touching nested and the kernel lane reads canonical only, **the risk is inverted from what that gap
assumed**: reading canonical gives you the wrong golden, not the right one.

**For K6 specifically:** if you clone the `gate_up_gelu` family into a SiLU sibling and clone its test
alongside, you will inherit a reference that disagrees with `gelu.h` semantics — and for SiLU the analogous
question (does the outer factor use clamped or original `x`?) has to be answered deliberately, not inherited.
Logged as GAP-TESSL-GELU-DUP. Fixing canonical's test is out of this lane's scope.

### F.3 Risk to the kernel author

The real hazard is **editing the wrong `nn.rs`**: both files are 150-160 KB, both named `src/nn.rs`, both
contain `rms_qkv_rope` and `gemv_q4_mlx_gate_up_gelu`, and **DevMap returns both** for every symbol search —
in my searches the nested copy frequently ranked *first*. Always check `file_path` starts `research/tessl/`,
never `research/MLSystemsLab/`.

---

## G. Does anything GDN-related already exist in tessl? **No.**

`rg -in -uu` for `delta[_ ]?rule|gated.?delta|linear.?attn|linear.?attention|recurrent.?state|\bgdn\b|gated_delta|deltanet|delta_net|ssm|mamba|retnet|rwkv`
over `src/`, `kernels/`, `tests/`, `build.rs`:

```
0 matches / 0 matched lines / 0 files contained matches
76 files searched / 1346719 bytes searched
```

**[verified]** `-uu` was used, so `.gitignore` did not hide anything. **The plan's assumption is correct: tessl
has no GDN, no linear attention, no recurrent state, and no SSM of any kind.** Its only sequence-mixing
kernels are softmax flash attention.

### G.1 Prior art that does exist nearby (not in tessl — useful, not blocking)

1. **The MLX reference implementation of exactly this operator** is on disk:
   `/Users/bharath/Code/research/LocalModelBench/.venv/lib/python3.14/site-packages/mlx_vlm/models/qwen3_5/gated_delta.py`
   (plus `qwen3_5/language.py`, `qwen3_5_moe/language.py`). This is vendored site-packages, not user code, but
   it is **the numerical reference `tests/gdn.rs` should be pinned against** — and it is the natural place to
   confirm which operator variant is "published" per repo rule 9. **[verified — file exists]**
2. **Recurrent-mixer verification harness** in the nested repo:
   `research/MLSystemsLab/Rust_MLKit/arch_02_value_resid/metal-native/src/bin/verify_mixers.rs` defines
   `verify_mamba2`, `verify_mingru`, `verify_mingru_vr`, `verify_mingru_vr_layer`, plus
   `load_tensor_f32` / `load_matrix_python` / `assert_matrix_parity_python` and its **own** `npy.rs` with a
   *used* `read_npy`. That is a working Rust↔Python parity-fixture pattern for recurrent kernels — the exact
   thing §C.4 says tessl lacks. Worth reading before designing the `tests/gdn.rs` fixture seam.
   **[verified via DevMap `devmap_neighbors` + `code_graph.json`]** Read-only: it is under the rule-6 no-touch tree.

---

## GAPS

Also appended to `/Users/bharath/Code/research/qwen-decision/gaps.jsonl`.

| id | What could not be answered | Impact | Workaround used |
| --- | --- | --- | --- |
| **GAP-GITPULSE-TRUST** | `gitpulse_insights(repo_path=/Users/bharath/Code/research/tessl)` returned **every facet `ok: false`** — `worktrees`, `changes`, `collisions`, `agents`, `codeintel`, `ledger`, `branch` — all with `REPOSITORY_TRUST_REQUIRED: Open … in GitPulse and trust it before running repository commands.` **No facet came back clean; none came back at all.** | **Working-tree state of tessl is UNVERIFIED.** I do not know whether another worktree or agent session holds `src/nn.rs` or `kernels/`, whether there is uncommitted work, or whether files are contended. The kernel author must re-check before editing. | None available. I did not substitute `git status` — per navigation rules that would be reporting a shell result as graph-confirmed. **Action: open `/Users/bharath/Code/research/tessl` in GitPulse and trust it, then re-run `gitpulse_insights`.** |
| **GAP-DEVMAP-METAL-EDGES** | DevMap indexes `.metal` files as `Function`/`File` nodes (searches returned `kernel void` bodies with spans) but there is **no call edge from Rust `rt.pipeline("name")` to the `.metal` entry point** — the binding is a runtime string lookup. `devmap_trace`/`devmap_impact` cannot answer "what breaks if I rename this kernel". | Renaming or adding an entry point has a blast radius DevMap cannot compute. `promoted_kernels.rs` (§C.6) is the only automated guard, and it only covers 44 of 68 entry points. | Enumerated entry points by `rg -n '^kernel void'` over `kernels/*.metal` and cross-checked against `PROMOTED` and the `rt.pipeline("…")` string literals. **Labelled as grep-derived, not graph-confirmed.** |
| **GAP-DEVMAP-READNPY-AMBIG** | `devmap_neighbors(targets=["read_npy",…])` came back `truncated: true` (shown 60 of 62) with a long `walk_incomplete` (stopped at depth 2; 20 files failed to parse; 34 pattern-recovered with no calls extracted; 858960 unresolved attribution sites). It also **resolved to the wrong `read_npy`** — walking into `MLSystemsLab/…/metal-native/src/npy.rs` rather than tessl's. | DevMap cannot confirm "tessl's `read_npy` has zero callers". | Verified independently with `rg -n -uu '\bread_npy\b' /Users/bharath/Code` → single source occurrence (its own definition). **Two signals** (rg over the whole tree + zero `.npy` files on disk + zero `npy` mentions in `tests/`), per the deleting-requires-proof rule. Conclusion is **rg-confirmed, not graph-confirmed.** |
| **GAP-DEVMAP-DUPE-RANK** | Every DevMap symbol search returned the canonical and nested `tessl` copies interleaved, with the nested copy often ranked **first** (`flash_attn_swa_h256`, `rms_qkv_rope`, `gate_up_gelu` all did this). Searches were also `truncated: true` with `hidden: 3/12/18`. | An agent that trusts the first DevMap hit will read — or edit — the rule-6 forbidden copy. | Filtered every result on `file_path` prefix `research/tessl/`. **Every §D signature in this document was re-read from the canonical file directly.** |
| **GAP-DEVMAP-CORPUS-PARTIAL** | Every DevMap answer carried: *"the corpus behind this answer is not a complete read of the repository … 20 file(s) failed to parse, 34 recovered by pattern (no calls extracted), 0 refused by discovery; 7 file(s) in a language with no call extractor at all."* | Dead-code and unwired findings across `~/Code` are a **lower bound**. Does not affect Rust/Metal in tessl (no tessl file is on the failed/pattern-recovered lists — those are PowerShell, protobuf, SQL, and NUL-byte TS files elsewhere). | Noted; tessl itself parses cleanly. `devmap_status`: generation 25, 180660 nodes, 602286 edges, `pending_count: 0`, `is_fresh: true`, `degraded_reason: null`. |
| **GAP-TESSL-COPY-DRIFT** | *(the plan's own rule-6 item — answered, not a blocker)* Nested copy confirmed at `MLSystemsLab/Rust_MLKit/crates/tessl`, plain vendored directory, no own `.git`. **45 files differ; 6 only-in-canonical; 41 only-in-nested; only 2 of 24 kernel files differ; nested has NO kernel canonical lacks.** Canonical is ~2 weeks ahead on `src/`, nested has extra bench scaffolding. Full detail §F. | Low, if rule 6 is honoured. The live risk is misdirected edits (§F.3). | n/a — fully measured. |
| **GAP-PLAN-K6-THREE-VARIANTS** | *(supersedes another lane's `GAP-PLAN-K6-TWO-VARIANTS`)* That gap recorded **two** `gate_up_gelu` entry points from a `devmap_search` that returned `truncated: true, hidden: 12`. There are **three** — the missed one is `gemv_q4_mlx_simd_gate_up_gelu_i4` (`gemv_q4_mlx.metal:708`), the Interleaved4 layout, which is the variant canonical's own test actually exercises. | K6's estimate is understated a second time. The three differ in `x` dtype and threadgroup-memory use, so they are not a mechanical copy of one another. | Enumerated with `rg -n '^kernel void' kernels/gemv_q4_mlx.metal` on the canonical file. **Do not use `devmap_search` to count entry points** (GAP-DEVMAP-DUPE-RANK). |
| **GAP-TESSL-NO-NPY-FIXTURE-PATH** | Does tessl have an existing `.npy` fixture path for `tests/gdn.rs`? **No** — zero `.npy` files, zero `npy` references in `tests/`, `read_npy` has zero callers crate-wide. | The fixture seam is **new work** not in the plan's estimate: a fixture directory, a first caller for the untested `read_npy`, and a filename convention naming the operator rule (repo rule 9). | §C.4. Either generate in-test in house style, or build the seam deliberately and cover `read_npy`. |
| **GAP-TESSL-PROMOTED-44-ASSERT** | Do K1–K7 break a passing test by construction? **Yes, conditionally** — `tests/promoted_kernels.rs:120-131` hard-asserts `PROMOTED.len() == 44`, and both sides measure exactly 44 today. | K5 and K6 each break this test until `PROMOTED` gains the names *and* the literal `44` is bumped. | §C.6. New `.metal` files dodge it entirely — but then add your own resolve test, since those ~86 entry points have no guard. |
| **GAP-TESSL-GELU-DUP** | *(incidental, but it lands on K6)* Canonical tessl has **two** disagreeing GELUs — `gelu.h:20` `0.5f * x * (...)` (unclamped outer factor, the form `gelu.h:11-15` says is correct) and `mlp_gelu_tanh.metal:22` `0.5f * xc * (...)` (clamped). Worse: canonical `tests/q4_interleaved.rs:450` asserts the **clamped** reference against `gemv_q4_mlx_simd_gate_up_gelu_i4`, which includes `gelu.h` and uses the **unclamped** one. | Canonical's test is wrong for the kernel it tests; it passes only because `random_f32` operands never drive the projection past 20, so the disagreement is never exercised. **Refines GAP-TESSL-COPY-DRIFT-SEMANTIC: it is the nested copy that matches the kernel, canonical that does not** — so reading canonical (as rule 6 requires) gives the *wrong* golden. K6 clones this exact family. | Reported only; fixing canonical's test is outside this lane. **K6 must derive its SiLU reference from the kernel's actual helper, not by cloning this test.** |

---

## CORRECTIONS TO THE PLAN

Seven places where the plan's stated assumption about tessl does not match the crate.

### CORRECTION 1 — A new TensorOps kernel in a new file will **silently compile as metal3.2**, not fail
**Plan assumes:** dropping K1 into `kernels/` gets it Metal 4 TensorOps because `matmul_tensorops.metal` proves
the toolchain works.
**Actual:** `build.rs:116` hard-codes the TensorOps hard-fail list as exactly
`vec![kernels_dir.join("matmul_tensorops.metal")]`. Every other `.metal` goes through
`try_metal_compile(... "metal4.0")` and on failure **falls back to `-std=metal3.2`** with a `cargo:warning`
(`build.rs:177-197`). `#include <MetalPerformancePrimitives/…>` under metal3.2 would fail too, so you would get
a *panic* — but the diagnostic would be a confusing double error, and any TensorOps kernel that happens to
parse under 3.2 ships silently degraded.
**Do:** either **(a)** add K1's file to `tensorops_sources` at `build.rs:116` (a one-line, correct change), or
**(b)** append K1 into `matmul_tensorops.metal` itself. (a) is cleaner and avoids the §C.6 `44` assert.
**Also:** `cargo:warning` output is invisible under a plain `cargo build` unless the crate rebuilds — do not
rely on spotting the fallback by eye.

### CORRECTION 2 — "a `gate_up_gelu` entry point" is **three** entry points (not one, and not two)
**Plan assumes:** K6 clones one kernel, swapping GELU→SiLU.
**A prior lane logged `GAP-PLAN-K6-TWO-VARIANTS` saying there are two.** That is also wrong — it used
`devmap_search`, whose result for this query came back `truncated: true` with `hidden: 12`.
**Actual, from `rg -n '^kernel void' kernels/gemv_q4_mlx.metal` on the canonical file — three:**
`gemv_q4_mlx_blocked_gate_up_gelu` (`:291`), `gemv_q4_mlx_simd_gate_up_gelu` (`:484`),
`gemv_q4_mlx_simd_gate_up_gelu_i4` (`:708`) — selected by `GateUpDispatch::{Blocked, Simd(RowMajor),
Simd(Interleaved4)}` (`nn.rs:3877-3883`). The missed third is the **Interleaved4** layout, which is the one
canonical's own `tests/q4_interleaved.rs` actually exercises. They differ in **x dtype** (`float` vs `bfloat`),
weight layout, and threadgroup-memory use, not only activation.
**Do:** scope K6 as 3 kernels (+3 `PROMOTED` entries, + the `44` bump), **or** explicitly scope it to one
dispatch mode and say which — but then the SiLU path has no Interleaved4/Blocked coverage and the Q4 MLP
cannot use it at every shape. There is also **no `silu.h`**; add one following `gelu.h` if K6 shares the helper.

### CORRECTION 3 — There is **no `.npy` fixture path in tests**; `read_npy` is dead code
**Plan assumes:** `src/npy.rs` loads `.npy` fixtures for parity tests.
**Actual:** zero `.npy` files in the crate; zero `npy` references in `tests/`; `read_npy`'s only occurrence in
all of `~/Code` is its own definition (`src/npy.rs:49`). Only `write_npy_f32` is used, by
`src/bin/bench_gemm_sweep.rs` — tessl **writes** operands for Python to read, never the reverse.
`src/npy.rs:1-14`: *"This exists for benchmark parity, not as a general-purpose format library."*
**Do:** budget for building the fixture seam. Every existing tessl test generates operands in-process with
`random_f32(n, seed)`. If `tests/gdn.rs` needs a Python-generated golden (repo rule 9 says it does — the
`published` operator, not nanolab's `rule="repo"` default), that is **new work**: a fixture directory, a
loader call site, and a filename/header that names the rule. `read_npy` is written and will work; you are its
first caller, so it is also untested — cover it.

### CORRECTION 4 — The threadgroup limit is 32 KiB, but the crate's own docstring says otherwise
**Plan assumes:** 32 KB threadgroup limit → 64 KB state must go to a device buffer. **This is CORRECT** and I
confirm it: measured `maxThreadgroupMemoryLength = 32768` on this M5 Pro; 128×128 fp32 = 65536 B = 2× over.
**But** `src/runtime.rs:620-621` claims *"Apple guarantees at least 32 KiB; **current Apple silicon reports
more**."* On this machine it reports **exactly** 32768. Anyone who reads that comment instead of calling
`rt.max_threadgroup_memory()` will size a tile wrong.
**Do:** keep the plan's device-buffer design (it is right), always call `rt.max_threadgroup_memory()` at
runtime and error before encoding (§B.5 precedent), and fix the misleading docstring while you are there.

### CORRECTION 5 — Adding an entry point to an existing promoted kernel file breaks a passing test
**Plan does not mention:** `tests/promoted_kernels.rs:120-131` asserts `PROMOTED.len() == 44`, and I verified
both sides are currently exactly 44.
**Consequence:** K5 (new entry in `rms_qkv_rope.metal`) and K6 (new entries in `gemv_q4_mlx.metal`) each break
this test until `PROMOTED` gains the names **and** the literal `44` is bumped.
**Do:** treat it as required work, not a surprise. Note the effort estimate asymmetry: **new `.metal` files are
free of this** (6 existing sources with 26 entry points are unlisted), so putting GDN kernels in a new
`gdn.metal` avoids it — and is independently the right call per CORRECTION 1.

### CORRECTION 6 — `q_only` is not the `norm_mask` precedent it looks like
**Plan assumes:** K5 adds `norm_mask` for QK-norm-without-V-norm.
**Actual:** `rotary_dim` **is** already a parameter (D.1, confirmed — the plan is right here). `norm_mask` does
not exist and is genuinely new. But the existing `q_only: bool` skips **K and V together**, not V alone, and it
is implemented by clamping `Hkv = 0` *after* the caller's scalar closure (`nn.rs:2740-2748`) specifically so
padding simdgroups cannot fall into the K/V branches and re-normalize rows another simdgroup owns. It is also
**refused** in combination with `PosBufferKvStore` (`nn.rs:2507-2511`).
**Do:** do not model `norm_mask` as "another `q_only`". Decide whether it shrinks the grid guard (fast, but the
Q/K/V branch arithmetic must stay consistent) or no-ops V rows (simple, wastes `T*Hkv` simdgroups), and
explicitly validate its interaction with `PosBufferKvStore` and `q_only` rather than allowing combinations by
omission — that is the failure mode `validate_rms_qkv_rope` was written to prevent.

### CORRECTION 7 — Counting entry points by `grep '^kernel void'` undercounts by ~46%
**Plan should not quote any count from tessl's README/CHANGELOG.** Measured today:
**306 tests pass / 0 fail / 1 ignored** — not any number a doc states.

Kernel counts are subtler than they look, and the crate's own tooling gets this wrong. Across the 20 top-level
`.metal` files there are **70 literal `^kernel void` lines** (44 promoted + 26 unpromoted — that identity holds
exactly). But **three of those 70 are X-macro *definitions*, not entry points**, and they expand to 63:

| File | literal `^kernel void` | real entry points | expansion site |
| --- | ---: | ---: | --- |
| `flash_attn_rows.metal` | 1 (`:58`, `#define ROWS_KERNEL`) | **27** | 27 × `ROWS_KERNEL(...)` |
| `flash_attn_decode.metal` | 2 (`:48`, `:259`) | **36** | 27 × `DECODE_PARTIAL_KERNEL` + 9 × `DECODE_REDUCE_KERNEL` |
| all other 18 files | 67 | 67 | — |
| **TOTAL** | **70** | **130** | |

So the metallib really exports **130 entry points**, not 70. **[verified — re-counted today]**

This matters twice. First, `promoted_kernels.rs`'s own failure message tells you to re-derive the number with
`grep -hc '^kernel void' kernels/{...}.metal` — that command is **only** correct because none of the 14
promoted sources happens to be a macro file. Use it for those 14 and nowhere else. Second, if a GDN kernel is
templated over head-dim or tile size (very likely for K1/K2/K3, following `flash_attn_rows`'s
`ROWS_KERNEL(name, D, R, G)` pattern), **the literal grep will not count your kernels either**, and any
"how many kernels does tessl have" claim built on it is wrong.

**Do:** re-run the counts after any kernel change rather than citing this table; when templating, count
instantiation sites (`rg -c '^ROWS_KERNEL\('`), not `kernel void` lines. `flash_attn_rows.metal:220-246` is
the model to copy if K1–K3 want compile-time head-dim/tile variants — it is how tessl already gets 27 flash
kernels out of one body, and it sidesteps the fact that head-dim is baked in as a `constant` and therefore
selects the kernel rather than being passed to it (§B.2).

---

## Appendix — fast commands

```bash
# build + full suite (baseline: 306 passed, 0 failed, 1 ignored)
cargo build --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml
cargo test  --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml

# one test binary
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test gdn

# entry points (re-measure, never quote)
rg -n '^kernel void' /Users/bharath/Code/research/tessl/kernels/

# the 44-assert, both sides
rg -c '^kernel void' /Users/bharath/Code/research/tessl/kernels/{rms_norm,mlp_silu,mlp_gelu_tanh,flash_attn_swa_h128,flash_attn_swa_h256,flash_attn_global_h512,rms_qkv_rope,kv_store,gemv_q4,gemv_q8,gemv_q4_mlx,gemm_q4_mlx,embed_lookup,softcap_sample}.metal

# opt-in GEMM A/B kernels (kernels/tune/, 34 measurement-only kernels)
TESSL_GEMM_TUNE=1 cargo build --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml

# MUST DO FIRST: GitPulse trust, then re-run insights (GAP-GITPULSE-TRUST)
#   open /Users/bharath/Code/research/tessl in GitPulse and trust it
```

**Never edit** `/Users/bharath/Code/research/MLSystemsLab/Rust_MLKit/crates/tessl` (repo rule 6). Check every
DevMap hit's `file_path` starts `research/tessl/`.
