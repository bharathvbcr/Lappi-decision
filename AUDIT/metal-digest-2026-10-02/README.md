# Device-side prefix-state digest: design and proposed tessl patch (2026-10-02)

Lane L-metal. Fable's Mac-inference ruling, item 1
(`AUDIT/fable-optimize-2026-10-02/fable-improve-and-mac-inference-ruling.md`, Q2(c) row 1;
`GAP-QDM-STATE-DIGEST-HOST-SHA256-PER-DECODE-2026-10-02`). Labels: **[V]** read or run here,
**[I]** inferred, **[U]** unverified.

## Why this is a design and not a change

The brief allowed the change only if it fits in `crates/qd-metal/src/model.rs`
(`PrefixState::digest`, model.rs:132-143) on tessl's existing exports, without editing
`backend.rs` or tessl. It does not:

- tessl has no hash or integer-reduction kernel [V: `rg 'kernel void'` over `tessl/kernels`; the
  reductions there are f32 sums and maxes]. A float reduction cannot meet the ruling's oracle,
  "any one-bit flip in any buffer changes it": NaN payloads, `+0`/`-0`, and small values absorbed
  by a large sum all leave a float sum unchanged.
- tessl does export what an adopter needs to ship its own kernel (`GpuRuntime::add_metallib_bytes`,
  `GpuRuntime::pipeline`, `tessl::dispatch::{dispatch_1d, set_gpu_buf, set_u32}`) [V:
  `tessl/src/runtime.rs:1054-1091`, `tessl/src/dispatch.rs:523-569`]. But the metallib has to be
  compiled, which means a `.metal` file and a `build.rs` in qd-metal running `xcrun metal` /
  `metallib`. That is outside model.rs, and a second place Metal source is compiled for this
  model, beside tessl's own build.

So the kernel is proposed for canonical tessl (`state_digest.metal` here), and the qd-metal half
is described below. Nothing in tessl or qd-metal was changed for it.

## What it replaces, and what it buys [I, from the ruling; unmeasured]

`PrefixState::digest` maps every state buffer (18 conv, 18 GDN, 6 K, 6 V) and SHA-256s each on
the host: 32 MB at 500 tokens, about 224 MB at 8.5K. `backend.rs` calls it after the prefill and
after every decode (`backend.rs:473-476`, `:625-628`), so a one-slot `choice` request hashes the
state three times. The ruling estimates 25-35% of a 500-token decision and 20-25% of an 8.5K one
(SHA-256 at 1.5-2.5 GB/s, **[U]**). `qd-metal-bench --decision` times it as its own phase
(`digest_ms`), so the first measured number for this replaces the estimate (`HANDOFF/metal-inference-2026-10-02.md`).

A device digest reads the same bytes at GPU bandwidth (224 MB at an assumed ~200 GB/s is about
1 ms [I]), does ~16 32-bit integer mixes per word (native on Apple GPUs; 64-bit multiplies are
emulated, which is why the lanes are 32-bit), and returns 32 bytes per buffer: **one** readback
for all 48 buffers instead of 48 host maps.

## The construction (oracle: `reference.rs`)

A buffer is `n` little-endian `u32` words. Lane `l` of 8 is the XOR, over every word `w` at index
`i`, of `fmix32(w ^ key_l(i))` with `key_l(i) = fmix32(i ^ SEED[l]) ^ SEED[(l+1) % 8]`; `fmix32` is
murmur3's finalizer, a bijection; `SEED` is SHA-256's initial hash values.

- **Any one-bit flip changes every lane.** For a fixed index the term is injective in the word,
  so a changed word changes exactly one term in each lane's XOR.
- **Deterministic by construction.** XOR commutes and associates: how the GPU splits and orders
  the reduction does not change the result. No fixed reduction tree is needed, and none can be
  got wrong.
- **Position-sensitive.** The index is in every key, so a value moved to another word is seen.
- **Not collision-resistant against an adversary.** Several changed words could cancel in all 8
  lanes (256 bits). For the accidental write the runtime's read-only check
  (`qd_runtime::answer::readonly_decode`) exists to catch, that is negligible; for a chosen write
  it is not, and this digest is not used for anything chosen.
- Per state: SHA-256 on the host over `"qd-metal.prefix-state.v2\0"`, the token count, and per
  buffer its tag, byte length and 32-byte lane record (a few hundred bytes). The format version
  moves from v1, so a v1 record is never compared with a v2 one.

**CPU tests [V, this session]:** `reference.rs`'s five tests, compiled with `rustc --test` and
run. The command and its result are in `HANDOFF/metal-inference-2026-10-02.md`.

## Proposed tessl patch

1. `kernels/state_digest.metal` (this directory's file). `state_digest_u32x8` writes one 8-word
   partial per threadgroup. `state_digest_fold_u32x8` folds the partials with one threadgroup.
   Only `simd_xor` on `uint` and a threadgroup barrier are used.
2. `src/qwen35.rs` (or `ops.rs`) gets the wrapper, following `qwen35::embed_rows`'s checks:

   ```rust
   /// 8-lane order-independent digest of `nbytes` of `buf` (whole u32 words) into `out[0..8]`.
   /// `partials` holds `8 * groups` u32; `groups` is chosen by the wrapper and returned.
   pub fn state_digest_u32x8(
       rt: &Arc<GpuRuntime>,
       buf: &GpuBuffer,
       nbytes: usize,
       partials: &GpuBuffer,
       out: &GpuBuffer,
   ) -> Result<(), String>
   // refuses nbytes % 4 != 0, nbytes / 4 > u32::MAX, undersized partials/out, and an out that
   // aliases buf (require_disjoint_writes)
   ```
3. Its test, `tests/state_digest.rs`, compares the device lanes with a port of
   `reference.rs::buffer_lanes` on random buffers, a 1-word buffer, a buffer of 2^20 + 3 words
   (a partial last threadgroup), `+0`/`-0`, and two NaN payloads. Bit for bit.

The patch goes to tessl's owner (rule 6: canonical tessl, coordinated with the ojas sessions,
`bench_qwen35_layers` and the GDN fixtures green at the commit).

## The qd-metal half, once the kernel exists

- `model.rs`: a `DigestPath { Host, Device }` on `Model` (as `EmbedPath` is), copied into each
  `PrefixState` with an `Arc<GpuRuntime>`. `PrefixState::digest()` keeps its signature, so
  `backend.rs` does not change. `Device` dispatches all 48 buffers into one `[48, 8]` u32 output
  and maps it once.
- The default stays `Host` until the GPU oracle (device lanes == `reference.rs` on every buffer of
  a real prefill) and `tests/gpu.rs`'s three read-only tests pass with `Device`. The digest does
  not touch a logit, so the parity gate is unaffected. It is re-run anyway if the flag's default
  moves (rule 2).
- `qd-metal-bench --decision` gets `digest=host|device` arms beside `embed=…`. The A/B is
  `digest_ms` and `commits` per decision.

## What it needs

- A tessl commit by its owner (the kernel, the wrapper, the test). Or a human decision that
  qd-metal may compile its own overlay metallib (a `build.rs` running `xcrun`; no new crate). The
  overlay route works with tessl as it is, at the cost of a second Metal build.
- The `backend.rs` ordering is unchanged by this design. Merging the digest's readback into the
  score's readback would save one more wait per decode. That needs `backend.rs:620-628` to encode
  the digest before reading the scores, and that file is another session's today.
