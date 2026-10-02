// PROPOSED for canonical tessl (kernels/state_digest.metal). Not compiled anywhere yet; the lane
// that lands it compiles it with tessl's build.rs and runs the GPU oracle test in README.md.
//
// An order-independent digest of a buffer of 32-bit words: 8 lanes, each the XOR over every word
// i of fmix32(w[i] ^ key(lane, i)). XOR commutes, so the split over threads, simdgroups and
// threadgroups does not change the result: deterministic without a fixed reduction tree. The
// construction and its properties are in reference.rs (the oracle) beside this file.
#include <metal_stdlib>
using namespace metal;

constant uint QD_DIGEST_SEED[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

static inline uint qd_fmix32(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// The most simdgroups a threadgroup of up to 1024 threads holds.
#define QD_DIGEST_MAX_SG 32u

/// One partial per threadgroup: `partials[tg * 8 + lane]`. The host (or `state_digest_fold_u32x8`)
/// XORs the partials; any order gives the same 8 words.
///
/// Grid: 1D, any size; every thread strides over the words.
kernel void state_digest_u32x8(
    device const uint* words   [[buffer(0)]],
    constant uint&     n_words [[buffer(1)]],
    device uint*       partials [[buffer(2)]],
    uint gid   [[thread_position_in_grid]],
    uint grid  [[threads_per_grid]],
    uint tg    [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint n_sg  [[simdgroups_per_threadgroup]])
{
    uint acc[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    for (uint i = gid; i < n_words; i += grid) {
        const uint w = words[i];
        for (uint l = 0; l < 8u; ++l) {
            const uint key = qd_fmix32(i ^ QD_DIGEST_SEED[l]) ^ QD_DIGEST_SEED[(l + 1u) & 7u];
            acc[l] ^= qd_fmix32(w ^ key);
        }
    }
    threadgroup uint scratch[QD_DIGEST_MAX_SG * 8u];
    for (uint l = 0; l < 8u; ++l) {
        const uint s = simd_xor(acc[l]);
        if (lane == 0u) {
            scratch[sgid * 8u + l] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 8u) {
        uint x = 0u;
        for (uint s = 0; s < n_sg; ++s) {
            x ^= scratch[s * 8u + lid];
        }
        partials[tg * 8u + lid] = x;
    }
}

/// Fold `n_partials` partials into `out[0..8]` with one threadgroup of at least 8 threads.
kernel void state_digest_fold_u32x8(
    device const uint* partials   [[buffer(0)]],
    constant uint&     n_partials [[buffer(1)]],
    device uint*       out        [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (lid >= 8u) return;
    uint x = 0u;
    for (uint p = 0; p < n_partials; ++p) {
        x ^= partials[p * 8u + lid];
    }
    out[lid] = x;
}
