// cuda_emu.h - executes CUDA-C kernel sources on the CPU with the host C++ compiler.
//
// TEST-TIME EMULATION ONLY. NOT A GPU RUN, and never to be reported as one: a kernel that
// passes here has had its indexing, barrier placement, warp-collective use and reduction
// order checked against a CPU model of the CUDA execution model, nothing more.
//
// Execution model:
//   * blocks run one at a time (so `__shared__` can be a `static`), in the order the driver
//     asks for (forward, reverse or shuffled - an order-dependent result is an inter-block
//     race or an overlapping-write index bug);
//   * every CUDA thread of the running block is one std::thread;
//   * `__syncthreads` and every warp collective rendezvous through one mutex per block. A
//     thread that returns leaves every rendezvous (the arrive_and_drop of a std::barrier),
//     which is the sm_70+ rule that exited threads do not count. Under ThreadSanitizer that
//     mutex is hidden and only the orderings CUDA guarantees are reported to it (see
//     TsanQuiet below), so a missing barrier is a reported race rather than masked;
//   * std::barrier is NOT used: libc++ implements its arrive/wait in the shared library,
//     where ThreadSanitizer cannot see the synchronisation, so a correctly barriered kernel
//     reports false races (measured with clang 18 + libc++; libstdc++ does not). A mutex and
//     per-thread condition variables are visible to every sanitizer on every platform and
//     also give the exact deadlock detector below the state it needs.
//
// What is diagnosed (exit code in parentheses, mapped by the Rust harness):
//   (70) divergent __syncthreads (different call sites or call counts), a lane calling a
//        warp collective with a membermask that excludes itself, a shuffle that reads a lane
//        that exited / does not exist / is not in the mask, two lanes waiting for each other
//        at different collectives, __trap();
//   (71) out-of-bounds writes into the guard bytes around every device buffer and the
//        dynamic shared memory (reads need the Address sanitizer mode);
//   (72) deadlock: every live thread of the block is blocked and nothing can complete;
//   (73) wall-clock timeout per launch (a spin-wait that never ends);
//   (74) harness failure (thread creation, bad launch), (75) I/O failure.
// Shared memory (static and dynamic) is filled with 0xFF bytes before every block - NaN for
// float, -1 for int - so a read of a slot nothing wrote is visible in the output. The
// Address mode cannot poison static shared memory: see CUDA_EMU_SMEM_ATTR.

#pragma once

#ifndef CUDA_EMU_MODE_ASAN
#error "cuda_emu.h is included by the generated driver, which defines CUDA_EMU_MODE_ASAN"
#endif
#ifndef CUDA_EMU_ARCH
#define CUDA_EMU_ARCH 900
#endif

// Every system header a kernel might include is included here, before the keyword macros at
// the bottom exist: libstdc++ spells attributes like `__noinline__`, which the CUDA macro of
// that name would otherwise rewrite inside a header the kernel includes later.
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#define CUDA_EMU_PROVENANCE "cuda_emu: CPU emulation of CUDA-C on the host C++ compiler. NOT a GPU run."

namespace cuda_emu {

inline constexpr int kExitDiagnostic = 70;
inline constexpr int kExitOutOfBounds = 71;
inline constexpr int kExitDeadlock = 72;
inline constexpr int kExitTimeout = 73;
inline constexpr int kExitHarness = 74;
inline constexpr int kExitIo = 75;

[[noreturn]] __attribute__((format(printf, 2, 3))) inline void die(int code, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("cuda_emu: ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
  fflush(stderr);
  fflush(stdout);
  std::_Exit(code);
}

inline uint32_t f32_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}
inline float f32_from_bits(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}
inline uint64_t f64_bits(double d) {
  uint64_t u;
  memcpy(&u, &d, sizeof u);
  return u;
}
inline double f64_from_bits(uint64_t u) {
  double d;
  memcpy(&d, &u, sizeof d);
  return d;
}

// Round a double to the nearest value of a binary format with `mant_bits` stored mantissa
// bits and `exp_bits` exponent bits (ties to even, gradual underflow, overflow to infinity),
// returning its bit pattern. Every float is exactly a double, so float -> bf16 / half goes
// through here with a single rounding. NaN becomes 0x7fff, which is what cuda_fp16.hpp and
// cuda_bf16.hpp return for a NaN input.
inline unsigned short round_to_small_float(double d, int mant_bits, int exp_bits) {
  const uint64_t u = f64_bits(d);
  const unsigned short sign = static_cast<unsigned short>((u >> 63) << (mant_bits + exp_bits));
  const int dexp = static_cast<int>((u >> 52) & 0x7ff);
  const uint64_t dmant = u & ((uint64_t{1} << 52) - 1);
  const unsigned inf_bits = ((1u << exp_bits) - 1u) << mant_bits;
  if (dexp == 0x7ff) {
    if (dmant != 0) return 0x7fff;
    return static_cast<unsigned short>(sign | inf_bits);
  }
  if (dexp == 0) return sign;  // zero or a double subnormal: far below any target subnormal
  const int bias = (1 << (exp_bits - 1)) - 1;
  const int emin = 1 - bias;
  const uint64_t m = dmant | (uint64_t{1} << 52);  // value = m * 2^(dexp - 1075)
  const int e2 = dexp - 1075;
  const int big_e = dexp - 1023;  // floor(log2(value))
  if (big_e > bias + 1) return static_cast<unsigned short>(sign | inf_bits);
  const bool subnormal = big_e < emin;
  const int quantum_exp = (subnormal ? emin : big_e) - mant_bits;
  const int shift = quantum_exp - e2;  // >= 52 - mant_bits > 0
  uint64_t k;
  if (shift >= 64) {
    k = 0;
  } else {
    k = m >> shift;
    const uint64_t rem = m & ((uint64_t{1} << shift) - 1);
    const uint64_t half = uint64_t{1} << (shift - 1);
    if (rem > half || (rem == half && (k & 1))) ++k;
  }
  uint64_t bits;
  if (subnormal) {
    bits = k;  // k == 2^mant_bits encodes the smallest normal, which is what it rounded to
  } else {
    bits = (static_cast<uint64_t>(big_e + bias) << mant_bits) + (k - (uint64_t{1} << mant_bits));
  }
  if (bits >= inf_bits) return static_cast<unsigned short>(sign | inf_bits);
  return static_cast<unsigned short>(sign | bits);
}

inline unsigned short f32_to_bf16_rn(float f) { return round_to_small_float(static_cast<double>(f), 7, 8); }
inline unsigned short f64_to_bf16_rn(double d) { return round_to_small_float(d, 7, 8); }
inline unsigned short f32_to_f16_rn(float f) { return round_to_small_float(static_cast<double>(f), 10, 5); }
inline unsigned short f64_to_f16_rn(double d) { return round_to_small_float(d, 10, 5); }
inline float bf16_to_f32(unsigned short h) { return f32_from_bits(static_cast<uint32_t>(h) << 16); }
inline float f16_to_f32(unsigned short h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t e = (h >> 10) & 0x1fu;
  const uint32_t m = h & 0x3ffu;
  if (e == 0) {
    const float mag = static_cast<float>(m) * 0x1p-24f;  // exact: m < 2^10
    return f32_from_bits(f32_bits(mag) | sign);
  }
  if (e == 31) return f32_from_bits(sign | 0x7f800000u | (m << 13));
  return f32_from_bits(sign | ((e + 112u) << 23) | (m << 13));
}

}  // namespace cuda_emu

// ---------------------------------------------------------------------------------------
// CUDA built-in types.

struct uint3 {
  unsigned int x, y, z;
};
struct dim3 {
  unsigned int x = 1, y = 1, z = 1;
  constexpr dim3() = default;
  constexpr dim3(unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
  constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
};
struct int3 {
  int x, y, z;
};
struct float3 {
  float x, y, z;
};
struct alignas(8) float2 {
  float x, y;
};
struct alignas(16) float4 {
  float x, y, z, w;
};
struct alignas(8) int2 {
  int x, y;
};
struct alignas(16) int4 {
  int x, y, z, w;
};
struct alignas(8) uint2 {
  unsigned int x, y;
};
struct alignas(16) uint4 {
  unsigned int x, y, z, w;
};
struct alignas(16) double2 {
  double x, y;
};
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline int2 make_int2(int x, int y) { return {x, y}; }
inline int3 make_int3(int x, int y, int z) { return {x, y, z}; }
inline int4 make_int4(int x, int y, int z, int w) { return {x, y, z, w}; }
inline uint2 make_uint2(unsigned int x, unsigned int y) { return {x, y}; }
inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return {x, y, z}; }
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x, y, z, w}; }
inline double2 make_double2(double x, double y) { return {x, y}; }

inline thread_local uint3 threadIdx{};
inline thread_local uint3 blockIdx{};
inline thread_local dim3 blockDim{};
inline thread_local dim3 gridDim{};
inline constexpr int warpSize = 32;

// Storage types and conversions of cuda_fp16.h / cuda_bf16.h. Arithmetic operators on the
// 16-bit types are DELETED rather than provided: CUDA rounds each half/bf16 operation to 16
// bits, and letting `a + b` fall through the float conversion would compute in float without
// that rounding - a silent difference. Comparisons are exact through float and are allowed.
struct __half {
  unsigned short __x;
  __half() = default;
  __half(float f) : __x(::cuda_emu::f32_to_f16_rn(f)) {}
  __half(double d) : __x(::cuda_emu::f64_to_f16_rn(d)) {}
  __half(int i) : __x(::cuda_emu::f64_to_f16_rn(static_cast<double>(i))) {}
  __half(unsigned int i) : __x(::cuda_emu::f64_to_f16_rn(static_cast<double>(i))) {}
  __half(short i) : __x(::cuda_emu::f64_to_f16_rn(static_cast<double>(i))) {}
  __half(unsigned short i) : __x(::cuda_emu::f64_to_f16_rn(static_cast<double>(i))) {}
  operator float() const { return ::cuda_emu::f16_to_f32(__x); }
};
struct __nv_bfloat16 {
  unsigned short __x;
  __nv_bfloat16() = default;
  __nv_bfloat16(float f) : __x(::cuda_emu::f32_to_bf16_rn(f)) {}
  __nv_bfloat16(double d) : __x(::cuda_emu::f64_to_bf16_rn(d)) {}
  __nv_bfloat16(int i) : __x(::cuda_emu::f64_to_bf16_rn(static_cast<double>(i))) {}
  __nv_bfloat16(unsigned int i) : __x(::cuda_emu::f64_to_bf16_rn(static_cast<double>(i))) {}
  __nv_bfloat16(short i) : __x(::cuda_emu::f64_to_bf16_rn(static_cast<double>(i))) {}
  __nv_bfloat16(unsigned short i) : __x(::cuda_emu::f64_to_bf16_rn(static_cast<double>(i))) {}
  operator float() const { return ::cuda_emu::bf16_to_f32(__x); }
};
typedef __half half;
typedef __nv_bfloat16 nv_bfloat16;
static_assert(std::is_trivially_copyable_v<__half> && sizeof(__half) == 2, "__half layout");
static_assert(std::is_trivially_copyable_v<__nv_bfloat16> && sizeof(__nv_bfloat16) == 2, "bf16 layout");

#define CUDA_EMU_DELETE_16BIT_ARITH(T) \
  T operator+(T, T) = delete;          \
  T operator-(T, T) = delete;          \
  T operator*(T, T) = delete;          \
  T operator/(T, T) = delete;
CUDA_EMU_DELETE_16BIT_ARITH(__half)
CUDA_EMU_DELETE_16BIT_ARITH(__nv_bfloat16)
#undef CUDA_EMU_DELETE_16BIT_ARITH

inline __half __float2half(float f) { return __half(f); }
inline __half __float2half_rn(float f) { return __half(f); }
inline __half __double2half(double d) { return __half(d); }
inline float __half2float(__half h) { return ::cuda_emu::f16_to_f32(h.__x); }
inline unsigned short __half_as_ushort(__half h) { return h.__x; }
inline short __half_as_short(__half h) { return static_cast<short>(h.__x); }
inline __half __ushort_as_half(unsigned short u) {
  __half h;
  h.__x = u;
  return h;
}
inline __half __short_as_half(short s) { return __ushort_as_half(static_cast<unsigned short>(s)); }
inline __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16(f); }
inline __nv_bfloat16 __float2bfloat16_rn(float f) { return __nv_bfloat16(f); }
inline __nv_bfloat16 __double2bfloat16(double d) { return __nv_bfloat16(d); }
inline __nv_bfloat16 __float2bfloat16_rz(float f) {
  __nv_bfloat16 b;
  const uint32_t u = ::cuda_emu::f32_bits(f);
  b.__x = ((u & 0x7fffffffu) > 0x7f800000u) ? 0x7fff : static_cast<unsigned short>(u >> 16);
  return b;
}
inline float __bfloat162float(__nv_bfloat16 b) { return ::cuda_emu::bf16_to_f32(b.__x); }
inline unsigned short __bfloat16_as_ushort(__nv_bfloat16 b) { return b.__x; }
inline short __bfloat16_as_short(__nv_bfloat16 b) { return static_cast<short>(b.__x); }
inline __nv_bfloat16 __ushort_as_bfloat16(unsigned short u) {
  __nv_bfloat16 b;
  b.__x = u;
  return b;
}
inline __nv_bfloat16 __short_as_bfloat16(short s) { return __ushort_as_bfloat16(static_cast<unsigned short>(s)); }

// ---------------------------------------------------------------------------------------
// ThreadSanitizer: the rendezvous below takes one mutex per block, and TSan would read every
// lock/unlock as a happens-before edge between ANY two threads that touched it - masking a
// missing __syncthreads between warps (measured: the race went unreported). So the
// emulator's own bookkeeping is made invisible to TSan (accesses and synchronisation), and
// the happens-before edges CUDA actually guarantees are added explicitly: a __syncthreads
// orders every participant, a __syncwarp orders its participating lanes, and shuffles,
// votes and warp reductions order no memory at all (PTX gives them none).

#if defined(__SANITIZE_THREAD__)
#define CUDA_EMU_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define CUDA_EMU_TSAN 1
#endif
#endif
#ifndef CUDA_EMU_TSAN
#define CUDA_EMU_TSAN 0
#endif

#if CUDA_EMU_TSAN
extern "C" {
void __tsan_acquire(void* addr);
void __tsan_release(void* addr);
void AnnotateIgnoreReadsBegin(const char* file, int line);
void AnnotateIgnoreReadsEnd(const char* file, int line);
void AnnotateIgnoreWritesBegin(const char* file, int line);
void AnnotateIgnoreWritesEnd(const char* file, int line);
void AnnotateIgnoreSyncBegin(const char* file, int line);
void AnnotateIgnoreSyncEnd(const char* file, int line);
}
#endif

namespace cuda_emu {

inline void tsan_release(void* token) {
#if CUDA_EMU_TSAN
  __tsan_release(token);
#else
  (void)token;
#endif
}
inline void tsan_acquire(void* token) {
#if CUDA_EMU_TSAN
  __tsan_acquire(token);
#else
  (void)token;
#endif
}

// While alive, the calling thread's memory accesses and lock operations are not seen by TSan.
struct TsanQuiet {
  TsanQuiet() {
#if CUDA_EMU_TSAN
    AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
    AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
    AnnotateIgnoreSyncBegin(__FILE__, __LINE__);
#endif
  }
  ~TsanQuiet() {
#if CUDA_EMU_TSAN
    AnnotateIgnoreSyncEnd(__FILE__, __LINE__);
    AnnotateIgnoreWritesEnd(__FILE__, __LINE__);
    AnnotateIgnoreReadsEnd(__FILE__, __LINE__);
#endif
  }
  TsanQuiet(const TsanQuiet&) = delete;
  TsanQuiet& operator=(const TsanQuiet&) = delete;
};

}  // namespace cuda_emu

// ---------------------------------------------------------------------------------------
// The block rendezvous: __syncthreads and every warp collective.

namespace cuda_emu {

enum class Op : unsigned char {
  Running,
  Bar,
  ShflIdx,
  ShflUp,
  ShflDown,
  ShflBfly,
  SyncWarp,
  Ballot,
  Any,
  All,
  RedAdd,
  RedMinU,
  RedMaxU,
  RedMinI,
  RedMaxI,
  RedAnd,
  RedOr,
  RedXor,
  Exited,
};
enum class BarKind : unsigned char { Plain, Count, And, Or };

inline const char* op_name(Op op) {
  switch (op) {
    case Op::Running: return "running";
    case Op::Bar: return "__syncthreads";
    case Op::ShflIdx: return "__shfl_sync";
    case Op::ShflUp: return "__shfl_up_sync";
    case Op::ShflDown: return "__shfl_down_sync";
    case Op::ShflBfly: return "__shfl_xor_sync";
    case Op::SyncWarp: return "__syncwarp";
    case Op::Ballot: return "__ballot_sync";
    case Op::Any: return "__any_sync";
    case Op::All: return "__all_sync";
    case Op::RedAdd: return "__reduce_add_sync";
    case Op::RedMinU: return "__reduce_min_sync(unsigned)";
    case Op::RedMaxU: return "__reduce_max_sync(unsigned)";
    case Op::RedMinI: return "__reduce_min_sync(int)";
    case Op::RedMaxI: return "__reduce_max_sync(int)";
    case Op::RedAnd: return "__reduce_and_sync";
    case Op::RedOr: return "__reduce_or_sync";
    case Op::RedXor: return "__reduce_xor_sync";
    case Op::Exited: return "exited";
  }
  return "?";
}
inline const char* bar_kind_name(BarKind k) {
  switch (k) {
    case BarKind::Plain: return "__syncthreads";
    case BarKind::Count: return "__syncthreads_count";
    case BarKind::And: return "__syncthreads_and";
    case BarKind::Or: return "__syncthreads_or";
  }
  return "?";
}
inline bool is_warp_op(Op op) { return op != Op::Running && op != Op::Bar && op != Op::Exited; }

// PTX shfl.sync source-lane rule (ISA "Data Movement: shfl.sync"): returns the lane whose
// value is read; an out-of-segment source reads the caller's own lane.
inline unsigned shfl_source_lane(Op mode, unsigned lane, unsigned b, unsigned c) {
  const unsigned bval = b & 0x1fu;
  const unsigned cval = c & 0x1fu;
  const unsigned segmask = (c >> 8) & 0x1fu;
  const unsigned max_lane = (lane & segmask) | (cval & ~segmask);
  const unsigned min_lane = lane & segmask;
  int j = static_cast<int>(lane);
  bool pval = false;
  switch (mode) {
    case Op::ShflUp:
      j = static_cast<int>(lane) - static_cast<int>(bval);
      pval = j >= static_cast<int>(max_lane);
      break;
    case Op::ShflDown:
      j = static_cast<int>(lane + bval);
      pval = j <= static_cast<int>(max_lane);
      break;
    case Op::ShflBfly:
      j = static_cast<int>(lane ^ bval);
      pval = j <= static_cast<int>(max_lane);
      break;
    case Op::ShflIdx:
      j = static_cast<int>(min_lane | (bval & ~segmask));
      pval = j <= static_cast<int>(max_lane);
      break;
    default:
      die(kExitHarness, "shfl_source_lane called with %s", op_name(mode));
  }
  return pval ? static_cast<unsigned>(j) : lane;
}

class BlockSync {
 public:
  BlockSync(unsigned nthreads, const char* entry, uint3 block)
      : n_(nthreads), live_(nthreads), entry_(entry), block_(block), ts_(new ThreadSync[nthreads]),
        own_(new OwnCounts[nthreads]), lane_tokens_(new unsigned char[static_cast<size_t>(nthreads) * kTokenRing]) {}
  BlockSync(const BlockSync&) = delete;
  BlockSync& operator=(const BlockSync&) = delete;

  int syncthreads(unsigned tid, unsigned line, BarKind kind, int pred) {
    // The k-th barrier of every thread is the same phase (a thread that disagrees is
    // diagnosed below), so phase parity names a token no other in-flight phase uses.
    const unsigned long long k = own_[tid].bar_calls++;
    tsan_release(&bar_tokens_[k & 1]);
    int result;
    {
      TsanQuiet quiet;
      std::unique_lock<std::mutex> lk(mu_);
      result = syncthreads_locked(lk, tid, line, kind, pred);
    }
    tsan_acquire(&bar_tokens_[k & 1]);
    return result;
  }

  uint64_t warp_op(unsigned tid, Op op, unsigned line, unsigned mask, uint64_t value, unsigned b, unsigned c) {
    const bool orders_memory = op == Op::SyncWarp;
    const unsigned long long seq = own_[tid].warp_calls++;
    if (orders_memory) tsan_release(lane_token(tid, seq));
    uint64_t result;
    unsigned group = 0;
    unsigned long long peer_seq[32];
    {
      TsanQuiet quiet;
      std::unique_lock<std::mutex> lk(mu_);
      result = warp_op_locked(lk, tid, op, line, mask, value, b, c, seq);
      group = ts_[tid].group;
      if (orders_memory) memcpy(peer_seq, ts_[tid].peer_seq, sizeof peer_seq);
    }
    if (orders_memory) {
      const unsigned warp = tid / 32;
      for (unsigned j = 0; j < 32; ++j) {
        if ((group >> j) & 1u) tsan_acquire(lane_token(warp * 32 + j, peer_seq[j]));
      }
    }
    return result;
  }

  // The thread returned from the kernel: it leaves every rendezvous (arrive_and_drop).
  void exit(unsigned tid) {
    TsanQuiet quiet;
    std::unique_lock<std::mutex> lk(mu_);
    exit_locked(tid);
  }

  // Called by the watchdog. Never blocks for long: a thread holds the mutex only briefly.
  void dump(FILE* out) {
    TsanQuiet quiet;
    std::unique_lock<std::mutex> lk(mu_, std::defer_lock);
    for (int attempt = 0; attempt < 200 && !lk.owns_lock(); ++attempt) {
      if (!lk.try_lock()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!lk.owns_lock()) {
      fprintf(out, "cuda_emu:   (thread states unavailable: the block mutex stayed locked)\n");
      return;
    }
    dump_locked(out);
  }

 private:
  static constexpr unsigned kTokenRing = 4;

  // Touched only by the owning thread, outside the mutex.
  struct OwnCounts {
    unsigned long long bar_calls = 0;
    unsigned long long warp_calls = 0;
  };

  void* lane_token(unsigned tid, unsigned long long seq) {
    return &lane_tokens_[static_cast<size_t>(tid) * kTokenRing + seq % kTokenRing];
  }

  int syncthreads_locked(std::unique_lock<std::mutex>& lk, unsigned tid, unsigned line, BarKind kind, int pred) {
    ThreadSync& me = ts_[tid];
    me.op = Op::Bar;
    me.bar_kind = kind;
    me.line = line;
    ++me.bar_calls;
    me.value = pred != 0 ? 1 : 0;
    me.released = false;
    ++blocked_;
    ++at_bar_;
    // A lane of this warp already waiting at a warp collective whose mask names this thread
    // waits for it forever, and this barrier waits for that lane: deadlock on hardware.
    const unsigned warp = tid / 32, lane = tid % 32;
    for (unsigned j = 0; j < warp_threads(warp); ++j) {
      const ThreadSync& other = ts_[warp * 32 + j];
      if (j != lane && is_warp_op(other.op) && ((other.mask >> lane) & 1u)) {
        die(kExitDiagnostic,
            "%s block (%u,%u,%u): thread %u reached %s at line %u while lane %u of its warp waits at %s "
            "(line %u, mask 0x%08x) that needs it - they wait for each other (deadlock on hardware)",
            entry_, block_.x, block_.y, block_.z, tid, bar_kind_name(kind), line, j, op_name(other.op), other.line,
            other.mask);
      }
    }
    if (at_bar_ == live_) {
      complete_bar_locked();
    } else {
      check_deadlock_locked();
      while (!me.released) me.cv.wait(lk);
    }
    return static_cast<int>(me.result);
  }

  uint64_t warp_op_locked(std::unique_lock<std::mutex>& lk, unsigned tid, Op op, unsigned line, unsigned mask,
                          uint64_t value, unsigned b, unsigned c, unsigned long long seq) {
    const unsigned warp = tid / 32, lane = tid % 32;
    if (((mask >> lane) & 1u) == 0) {
      die(kExitDiagnostic,
          "%s block (%u,%u,%u): thread %u (warp %u lane %u) calls %s at line %u with membermask 0x%08x that "
          "excludes itself (undefined behaviour in CUDA)",
          entry_, block_.x, block_.y, block_.z, tid, warp, lane, op_name(op), line, mask);
    }
    ThreadSync& me = ts_[tid];
    me.op = op;
    me.line = line;
    me.mask = mask;
    me.value = value;
    me.b = b;
    me.c = c;
    me.seq = seq;
    me.released = false;
    ++blocked_;
    const unsigned wn = warp_threads(warp);
    for (unsigned j = 0; j < wn; ++j) {
      if (j == lane || ((mask >> j) & 1u) == 0) continue;
      const ThreadSync& other = ts_[warp * 32 + j];
      if (other.op == Op::Bar) {
        die(kExitDiagnostic,
            "%s block (%u,%u,%u): lane %u of warp %u waits at %s (line %u, mask 0x%08x) for lane %u, which waits at "
            "%s (line %u) for every live thread - they wait for each other (deadlock on hardware)",
            entry_, block_.x, block_.y, block_.z, lane, warp, op_name(op), line, mask, j,
            bar_kind_name(other.bar_kind), other.line);
      }
      if (is_warp_op(other.op) && ((other.mask >> lane) & 1u) && (other.op != op || other.mask != mask)) {
        die(kExitDiagnostic,
            "%s block (%u,%u,%u): warp %u lanes %u and %u wait for each other at different collectives: lane %u at "
            "%s (line %u, mask 0x%08x), lane %u at %s (line %u, mask 0x%08x) (deadlock on hardware)",
            entry_, block_.x, block_.y, block_.z, warp, lane, j, lane, op_name(op), line, mask, j, op_name(other.op),
            other.line, other.mask);
      }
    }
    if (!try_complete_group_locked(warp, lane)) {
      check_deadlock_locked();
      while (!me.released) me.cv.wait(lk);
    }
    return me.result;
  }

  void exit_locked(unsigned tid) {
    ts_[tid].op = Op::Exited;
    --live_;
    if (at_bar_ > 0 && at_bar_ == live_) complete_bar_locked();
    const unsigned warp = tid / 32;
    for (unsigned j = 0; j < warp_threads(warp); ++j) {
      if (is_warp_op(ts_[warp * 32 + j].op)) try_complete_group_locked(warp, j);
    }
    check_deadlock_locked();
  }

  struct ThreadSync {
    Op op = Op::Running;
    BarKind bar_kind = BarKind::Plain;
    bool released = false;
    unsigned line = 0;
    unsigned long long bar_calls = 0;
    unsigned mask = 0;
    unsigned b = 0, c = 0;
    uint64_t value = 0;
    uint64_t result = 0;
    unsigned long long seq = 0;            // the poster's warp-collective count, for its token
    unsigned group = 0;                    // set on release: the lanes that took part
    unsigned long long peer_seq[32] = {};  // set on release of a __syncwarp: each member's seq
    std::condition_variable cv;
  };

  unsigned warp_threads(unsigned warp) const { return std::min(32u, n_ - warp * 32); }

  void release_locked(unsigned tid, uint64_t result) {
    ThreadSync& t = ts_[tid];
    t.result = result;
    t.op = Op::Running;
    t.released = true;
    --blocked_;
    t.cv.notify_one();
  }

  void complete_bar_locked() {
    unsigned first = n_;
    unsigned long long count = 0;
    for (unsigned t = 0; t < n_; ++t) {
      const ThreadSync& s = ts_[t];
      if (s.op == Op::Exited) continue;
      if (s.op != Op::Bar) die(kExitHarness, "barrier completed while thread %u is %s", t, op_name(s.op));
      if (first == n_) {
        first = t;
      } else {
        const ThreadSync& f = ts_[first];
        if (s.line != f.line || s.bar_calls != f.bar_calls || s.bar_kind != f.bar_kind) {
          die(kExitDiagnostic,
              "%s block (%u,%u,%u): divergent barrier - thread %u reached %s at line %u (its call #%llu) while "
              "thread %u reached %s at line %u (its call #%llu); every live thread of a block must reach the same "
              "__syncthreads",
              entry_, block_.x, block_.y, block_.z, first, bar_kind_name(f.bar_kind), f.line, f.bar_calls, t,
              bar_kind_name(s.bar_kind), s.line, s.bar_calls);
        }
      }
      count += s.value;
    }
    if (first == n_) die(kExitHarness, "barrier completed with no live thread");
    uint64_t result = 0;
    switch (ts_[first].bar_kind) {
      case BarKind::Plain: result = 0; break;
      case BarKind::Count: result = count; break;
      case BarKind::And: result = count == live_ ? 1 : 0; break;
      case BarKind::Or: result = count > 0 ? 1 : 0; break;
    }
    for (unsigned t = 0; t < n_; ++t) {
      if (ts_[t].op == Op::Bar) release_locked(t, result);
    }
    at_bar_ = 0;
  }

  bool try_complete_group_locked(unsigned warp, unsigned lane) {
    const ThreadSync& me = ts_[warp * 32 + lane];
    const Op op = me.op;
    const unsigned mask = me.mask;
    const unsigned wn = warp_threads(warp);
    unsigned group = 0;
    for (unsigned j = 0; j < wn; ++j) {
      if (((mask >> j) & 1u) == 0) continue;
      const ThreadSync& s = ts_[warp * 32 + j];
      if (s.op == Op::Exited) continue;
      if (s.op != op || s.mask != mask) return false;
      group |= 1u << j;
    }
    uint64_t results[32] = {};
    switch (op) {
      case Op::ShflIdx:
      case Op::ShflUp:
      case Op::ShflDown:
      case Op::ShflBfly:
        for (unsigned i = 0; i < wn; ++i) {
          if (((group >> i) & 1u) == 0) continue;
          const ThreadSync& s = ts_[warp * 32 + i];
          const unsigned src = shfl_source_lane(op, i, s.b, s.c);
          if (((group >> src) & 1u) == 0) {
            const char* why = src >= wn                               ? "does not exist (partial warp)"
                              : ts_[warp * 32 + src].op == Op::Exited ? "has exited"
                                                                      : "is not in the membermask";
            die(kExitDiagnostic,
                "%s block (%u,%u,%u): %s at line %u: warp %u lane %u reads lane %u, which %s - the value is "
                "undefined in CUDA (warp has %u threads, mask 0x%08x, participating lanes 0x%08x)",
                entry_, block_.x, block_.y, block_.z, op_name(op), s.line, warp, i, src, why, wn, mask, group);
          }
          results[i] = ts_[warp * 32 + src].value;
        }
        break;
      case Op::SyncWarp:
        break;
      case Op::Ballot:
      case Op::Any:
      case Op::All: {
        unsigned ballot = 0;
        for (unsigned i = 0; i < wn; ++i) {
          if (((group >> i) & 1u) && ts_[warp * 32 + i].value != 0) ballot |= 1u << i;
        }
        const uint64_t r = op == Op::Ballot ? ballot : op == Op::Any ? (ballot != 0) : (ballot == group);
        for (unsigned i = 0; i < wn; ++i) results[i] = r;
        break;
      }
      default: {
        bool first = true;
        uint32_t acc = 0;
        for (unsigned i = 0; i < wn; ++i) {
          if (((group >> i) & 1u) == 0) continue;
          const uint32_t v = static_cast<uint32_t>(ts_[warp * 32 + i].value);
          if (first) {
            acc = v;
            first = false;
            continue;
          }
          switch (op) {
            case Op::RedAdd: acc += v; break;
            case Op::RedMinU: acc = std::min(acc, v); break;
            case Op::RedMaxU: acc = std::max(acc, v); break;
            case Op::RedMinI:
              acc = static_cast<uint32_t>(std::min(static_cast<int32_t>(acc), static_cast<int32_t>(v)));
              break;
            case Op::RedMaxI:
              acc = static_cast<uint32_t>(std::max(static_cast<int32_t>(acc), static_cast<int32_t>(v)));
              break;
            case Op::RedAnd: acc &= v; break;
            case Op::RedOr: acc |= v; break;
            case Op::RedXor: acc ^= v; break;
            default: die(kExitHarness, "unhandled warp collective %s", op_name(op));
          }
        }
        for (unsigned i = 0; i < wn; ++i) results[i] = acc;
        break;
      }
    }
    for (unsigned i = 0; i < wn; ++i) {
      if (((group >> i) & 1u) == 0) continue;
      ThreadSync& m = ts_[warp * 32 + i];
      m.group = group;
      if (op == Op::SyncWarp) {
        for (unsigned j = 0; j < wn; ++j) {
          if ((group >> j) & 1u) m.peer_seq[j] = ts_[warp * 32 + j].seq;
        }
      }
      release_locked(warp * 32 + i, results[i]);
    }
    return true;
  }

  void check_deadlock_locked() {
    if (live_ == 0 || blocked_ != live_) return;
    fprintf(stderr, "cuda_emu: DEADLOCK in %s block (%u,%u,%u): all %u live threads are blocked:\n", entry_, block_.x,
            block_.y, block_.z, live_);
    dump_locked(stderr);
    die(kExitDeadlock,
        "deadlock in %s block (%u,%u,%u): no barrier or warp collective can complete (on hardware: a hang)", entry_,
        block_.x, block_.y, block_.z);
  }

  void dump_locked(FILE* out) {
    struct Group {
      Op op;
      unsigned line, mask, count, example;
    };
    std::vector<Group> groups;
    for (unsigned t = 0; t < n_; ++t) {
      const ThreadSync& s = ts_[t];
      const unsigned line = (s.op == Op::Running || s.op == Op::Exited) ? 0 : s.line;
      const unsigned mask = is_warp_op(s.op) ? s.mask : 0;
      bool found = false;
      for (Group& g : groups) {
        if (g.op == s.op && g.line == line && g.mask == mask) {
          ++g.count;
          found = true;
          break;
        }
      }
      if (!found) groups.push_back(Group{s.op, line, mask, 1, t});
    }
    for (const Group& g : groups) {
      if (g.op == Op::Running || g.op == Op::Exited) {
        fprintf(out, "cuda_emu:   %u thread(s) %s (e.g. thread %u)\n", g.count, op_name(g.op), g.example);
      } else {
        fprintf(out, "cuda_emu:   %u thread(s) at %s line %u mask 0x%08x (e.g. thread %u)\n", g.count, op_name(g.op),
                g.line, g.mask, g.example);
      }
    }
  }

  std::mutex mu_;
  const unsigned n_;
  unsigned live_;
  unsigned blocked_ = 0;
  unsigned at_bar_ = 0;
  const char* entry_;
  uint3 block_;
  std::unique_ptr<ThreadSync[]> ts_;
  std::unique_ptr<OwnCounts[]> own_;
  std::unique_ptr<unsigned char[]> lane_tokens_;  // TSan sync objects: kTokenRing per thread
  unsigned char bar_tokens_[2] = {};               // TSan sync objects: barrier phase parity
};

namespace detail {
inline thread_local BlockSync* t_sync = nullptr;
inline thread_local unsigned t_tid = 0;
}  // namespace detail

inline BlockSync& current_block() {
  if (detail::t_sync == nullptr) die(kExitHarness, "a device intrinsic was called outside a kernel launch");
  return *detail::t_sync;
}

inline void syncthreads(unsigned line) { current_block().syncthreads(detail::t_tid, line, BarKind::Plain, 1); }
inline int syncthreads_count(unsigned line, int pred) {
  return current_block().syncthreads(detail::t_tid, line, BarKind::Count, pred);
}
inline int syncthreads_and(unsigned line, int pred) {
  return current_block().syncthreads(detail::t_tid, line, BarKind::And, pred);
}
inline int syncthreads_or(unsigned line, int pred) {
  return current_block().syncthreads(detail::t_tid, line, BarKind::Or, pred);
}
inline void syncwarp(unsigned line, unsigned mask = 0xffffffffu) {
  current_block().warp_op(detail::t_tid, Op::SyncWarp, line, mask, 0, 0, 0);
}

inline unsigned shfl_c(unsigned line, int width, bool with_clamp) {
  if (width < 1 || width > 32 || (width & (width - 1)) != 0) {
    die(kExitDiagnostic, "warp shuffle at line %u with width %d: width must be a power of two in [1, 32]", line,
        width);
  }
  const unsigned seg = static_cast<unsigned>(32 - width) << 8;
  return with_clamp ? (seg | 0x1fu) : seg;
}

template <class T>
inline T shfl(unsigned line, Op mode, unsigned mask, T var, unsigned b, unsigned c) {
  static_assert(std::is_trivially_copyable_v<T> && sizeof(T) <= 8, "warp shuffles move values of at most 8 bytes");
  uint64_t bits = 0;
  memcpy(&bits, &var, sizeof(T));
  const uint64_t r = current_block().warp_op(detail::t_tid, mode, line, mask, bits, b, c);
  T out;
  memcpy(&out, &r, sizeof(T));
  return out;
}
template <class T>
inline T shfl_sync(unsigned line, unsigned mask, T var, int src_lane, int width = 32) {
  return shfl(line, Op::ShflIdx, mask, var, static_cast<unsigned>(src_lane), shfl_c(line, width, true));
}
template <class T>
inline T shfl_up_sync(unsigned line, unsigned mask, T var, unsigned delta, int width = 32) {
  return shfl(line, Op::ShflUp, mask, var, delta, shfl_c(line, width, false));
}
template <class T>
inline T shfl_down_sync(unsigned line, unsigned mask, T var, unsigned delta, int width = 32) {
  return shfl(line, Op::ShflDown, mask, var, delta, shfl_c(line, width, true));
}
template <class T>
inline T shfl_xor_sync(unsigned line, unsigned mask, T var, int lane_mask, int width = 32) {
  return shfl(line, Op::ShflBfly, mask, var, static_cast<unsigned>(lane_mask), shfl_c(line, width, true));
}
inline unsigned ballot_sync(unsigned line, unsigned mask, int pred) {
  return static_cast<unsigned>(current_block().warp_op(detail::t_tid, Op::Ballot, line, mask, pred != 0, 0, 0));
}
inline int any_sync(unsigned line, unsigned mask, int pred) {
  return static_cast<int>(current_block().warp_op(detail::t_tid, Op::Any, line, mask, pred != 0, 0, 0));
}
inline int all_sync(unsigned line, unsigned mask, int pred) {
  return static_cast<int>(current_block().warp_op(detail::t_tid, Op::All, line, mask, pred != 0, 0, 0));
}
inline unsigned reduce_u(unsigned line, Op op, unsigned mask, unsigned v) {
  return static_cast<unsigned>(current_block().warp_op(detail::t_tid, op, line, mask, v, 0, 0));
}
inline unsigned reduce_add_sync(unsigned line, unsigned mask, unsigned v) { return reduce_u(line, Op::RedAdd, mask, v); }
inline int reduce_add_sync(unsigned line, unsigned mask, int v) {
  return static_cast<int>(reduce_u(line, Op::RedAdd, mask, static_cast<unsigned>(v)));
}
inline unsigned reduce_min_sync(unsigned line, unsigned mask, unsigned v) { return reduce_u(line, Op::RedMinU, mask, v); }
inline int reduce_min_sync(unsigned line, unsigned mask, int v) {
  return static_cast<int>(reduce_u(line, Op::RedMinI, mask, static_cast<unsigned>(v)));
}
inline unsigned reduce_max_sync(unsigned line, unsigned mask, unsigned v) { return reduce_u(line, Op::RedMaxU, mask, v); }
inline int reduce_max_sync(unsigned line, unsigned mask, int v) {
  return static_cast<int>(reduce_u(line, Op::RedMaxI, mask, static_cast<unsigned>(v)));
}
inline unsigned reduce_and_sync(unsigned line, unsigned mask, unsigned v) { return reduce_u(line, Op::RedAnd, mask, v); }
inline unsigned reduce_or_sync(unsigned line, unsigned mask, unsigned v) { return reduce_u(line, Op::RedOr, mask, v); }
inline unsigned reduce_xor_sync(unsigned line, unsigned mask, unsigned v) { return reduce_u(line, Op::RedXor, mask, v); }

// ---------------------------------------------------------------------------------------
// Atomics: relaxed, like CUDA's. Within a block the threads are real threads, so these are
// real atomics; across blocks they never contend because blocks run one at a time.

template <class T, class F>
inline T atomic_rmw(T* addr, F f) {
  T old;
  __atomic_load(addr, &old, __ATOMIC_RELAXED);
  T desired;
  do {
    desired = f(old);
  } while (!__atomic_compare_exchange(addr, &old, &desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
  return old;
}

// Static shared memory placement. Outside the Address mode every `__shared__` variable is put
// in one named section so the driver can fill all of it with 0xFF before each block. The
// Address sanitizer does not instrument globals in a user section (ELF), so in that mode the
// variables are plain statics: bounds-checked, but zero for the first block and carrying the
// previous block's values afterwards.
#if CUDA_EMU_MODE_ASAN
#define CUDA_EMU_SMEM_ATTR
inline void poison_static_smem() {}
#elif defined(__APPLE__)
#define CUDA_EMU_SMEM_ATTR __attribute__((section("__DATA,__cuemu_smem")))
extern unsigned char smem_section_begin[] __asm("section$start$__DATA$__cuemu_smem");
extern unsigned char smem_section_end[] __asm("section$end$__DATA$__cuemu_smem");
__attribute__((used)) CUDA_EMU_SMEM_ATTR static unsigned char smem_section_anchor[16];
inline void poison_static_smem() {
  memset(smem_section_begin, 0xff, static_cast<size_t>(smem_section_end - smem_section_begin));
}
#else
#define CUDA_EMU_SMEM_ATTR __attribute__((section("cuda_emu_smem")))
}  // namespace cuda_emu
extern "C" unsigned char __start_cuda_emu_smem[];
extern "C" unsigned char __stop_cuda_emu_smem[];
namespace cuda_emu {
__attribute__((used)) CUDA_EMU_SMEM_ATTR static unsigned char smem_section_anchor[16];
inline void poison_static_smem() {
  memset(__start_cuda_emu_smem, 0xff, static_cast<size_t>(__stop_cuda_emu_smem - __start_cuda_emu_smem));
}
#endif

// The dynamic shared memory of the running launch. `extern __shared__ T name[];` is rewritten
// by the harness into a reference to this pointer, reinterpreted as `T*`.
inline unsigned char* g_dyn_smem = nullptr;

inline constexpr size_t kGuardBytes = 256;
inline constexpr unsigned char kGuardByte = 0xa5;
inline constexpr unsigned char kPoisonByte = 0xff;

// A region with guard bytes on both sides (outside the Address mode, where the allocation is
// exact so the sanitizer sees every out-of-bounds access instead).
class Region {
 public:
  Region(size_t bytes, size_t align) : bytes_(bytes), align_(align) {
#if CUDA_EMU_MODE_ASAN
    base_ = static_cast<unsigned char*>(::operator new(bytes, std::align_val_t(align)));
    data_ = base_;
#else
    base_ = static_cast<unsigned char*>(::operator new(bytes + 2 * kGuardBytes, std::align_val_t(align)));
    data_ = base_ + kGuardBytes;
    memset(base_, kGuardByte, kGuardBytes);
    memset(data_ + bytes, kGuardByte, kGuardBytes);
#endif
  }
  Region(Region&& o) noexcept : bytes_(o.bytes_), align_(o.align_), base_(o.base_), data_(o.data_) {
    o.base_ = nullptr;
    o.data_ = nullptr;
  }
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;
  Region& operator=(Region&&) = delete;
  ~Region() {
    if (base_ != nullptr) ::operator delete(base_, std::align_val_t(align_));
  }
  unsigned char* data() const { return data_; }
  size_t bytes() const { return bytes_; }
  // Returns the offset (negative: before the start) of the first overwritten guard byte, or
  // false when both guards are intact.
  bool corrupted_guard(long long* offset) const {
#if CUDA_EMU_MODE_ASAN
    (void)offset;
    return false;
#else
    for (size_t i = 0; i < kGuardBytes; ++i) {
      if (base_[kGuardBytes - 1 - i] != kGuardByte) {
        *offset = -static_cast<long long>(i) - 1;
        return true;
      }
    }
    for (size_t i = 0; i < kGuardBytes; ++i) {
      if (data_[bytes_ + i] != kGuardByte) {
        *offset = static_cast<long long>(bytes_ + i);
        return true;
      }
    }
    return false;
#endif
  }

 private:
  size_t bytes_;
  size_t align_;
  unsigned char* base_;
  unsigned char* data_;
};

enum class Init { File, Poison, Zero };
enum class Order { Forward, Reverse, Shuffled };

struct LaunchSpec {
  const char* entry;
  dim3 grid;
  dim3 block;
  size_t dyn_smem;
  unsigned timeout_ms;
  Order order;
  unsigned long long seed;
};

// Kills the process if a launch outlives its wall-clock budget: a spin-wait or a livelock
// that the deadlock detector cannot see (some thread is still running) fails, never hangs.
class Watchdog {
 public:
  Watchdog(const char* entry, unsigned timeout_ms)
      : entry_(entry), timeout_ms_(timeout_ms),
        deadline_(std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms)) {
    try {
      th_ = std::thread([this] { run(); });
    } catch (const std::system_error& e) {
      die(kExitHarness, "cannot start the watchdog thread: %s", e.what());
    }
  }
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;
  ~Watchdog() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      done_ = true;
    }
    cv_.notify_one();
    th_.join();
  }
  void enter(BlockSync* sync, uint3 block) {
    std::lock_guard<std::mutex> lk(mu_);
    cur_ = sync;
    cur_block_ = block;
  }
  void leave() {
    std::lock_guard<std::mutex> lk(mu_);
    cur_ = nullptr;
    ++blocks_done_;
  }

 private:
  void run() {
    std::unique_lock<std::mutex> lk(mu_);
    if (cv_.wait_until(lk, deadline_, [this] { return done_; })) return;
    fprintf(stderr, "cuda_emu: TIMEOUT: launch of %s exceeded %u ms after %llu completed block(s)\n", entry_,
            timeout_ms_, blocks_done_);
    if (cur_ != nullptr) {
      fprintf(stderr, "cuda_emu: block (%u,%u,%u) thread states:\n", cur_block_.x, cur_block_.y, cur_block_.z);
      cur_->dump(stderr);
    }
    die(kExitTimeout, "timeout in %s (a thread that never reaches a barrier, e.g. a spin-wait)", entry_);
  }

  const char* entry_;
  unsigned timeout_ms_;
  std::chrono::steady_clock::time_point deadline_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  BlockSync* cur_ = nullptr;
  uint3 cur_block_{};
  unsigned long long blocks_done_ = 0;
  std::thread th_;
};

inline std::vector<uint3> block_order(dim3 grid, Order order, unsigned long long seed) {
  std::vector<uint3> blocks;
  blocks.reserve(static_cast<size_t>(grid.x) * grid.y * grid.z);
  for (unsigned z = 0; z < grid.z; ++z)
    for (unsigned y = 0; y < grid.y; ++y)
      for (unsigned x = 0; x < grid.x; ++x) blocks.push_back(uint3{x, y, z});
  if (order == Order::Reverse) {
    std::reverse(blocks.begin(), blocks.end());
  } else if (order == Order::Shuffled) {
    unsigned long long state = seed;
    auto next = [&state] {  // splitmix64
      state += 0x9e3779b97f4a7c15ull;
      unsigned long long z = state;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      return z ^ (z >> 31);
    };
    for (size_t i = blocks.size(); i > 1; --i) std::swap(blocks[i - 1], blocks[next() % i]);
  }
  return blocks;
}

class Driver {
 public:
  Driver(int argc, char** argv) {
    if (argc != 2) die(kExitHarness, "usage: driver <io-directory>");
    dir_ = argv[1];
    puts(CUDA_EMU_PROVENANCE);
    fflush(stdout);
  }

  template <class T>
  T* buffer(unsigned idx, const char* name, size_t count, Init init) {
    if (idx != bufs_.size()) die(kExitHarness, "buffers must be declared in order (got %u)", idx);
    const size_t bytes = count * sizeof(T);
    Region region(bytes, 256);
    if (init == Init::File) {
      const std::string path = dir_ + "/buf_" + std::to_string(idx) + ".in";
      FILE* f = fopen(path.c_str(), "rb");
      if (f == nullptr) die(kExitIo, "cannot open %s", path.c_str());
      const size_t got = bytes == 0 ? 0 : fread(region.data(), 1, bytes, f);
      const int extra = fgetc(f);
      fclose(f);
      if (got != bytes || extra != EOF) die(kExitIo, "%s: expected exactly %zu bytes for buffer '%s'", path.c_str(), bytes, name);
    } else {
      memset(region.data(), init == Init::Zero ? 0 : kPoisonByte, bytes);
    }
    bufs_.push_back(Buf{name, sizeof(T), count, std::move(region)});
    return reinterpret_cast<T*>(bufs_.back().region.data());
  }

  template <class F>
  void launch(const LaunchSpec& s, F&& body) {
    const unsigned long long nthreads = static_cast<unsigned long long>(s.block.x) * s.block.y * s.block.z;
    if (nthreads == 0 || nthreads > 1024 || s.grid.x == 0 || s.grid.y == 0 || s.grid.z == 0) {
      die(kExitHarness, "invalid launch of %s: grid (%u,%u,%u) block (%u,%u,%u)", s.entry, s.grid.x, s.grid.y,
          s.grid.z, s.block.x, s.block.y, s.block.z);
    }
    const unsigned n = static_cast<unsigned>(nthreads);
    Region dyn(s.dyn_smem, 128);
    g_dyn_smem = dyn.data();
    {
      Watchdog wd(s.entry, s.timeout_ms);
      for (const uint3& b : block_order(s.grid, s.order, s.seed)) {
        poison_static_smem();
        memset(dyn.data(), kPoisonByte, dyn.bytes());
        BlockSync sync(n, s.entry, b);
        wd.enter(&sync, b);
        {
          std::vector<std::thread> threads;
          threads.reserve(n);
          for (unsigned t = 0; t < n; ++t) {
            try {
              threads.emplace_back([&sync, &body, &s, b, t] {
                threadIdx = uint3{t % s.block.x, (t / s.block.x) % s.block.y, t / (s.block.x * s.block.y)};
                blockIdx = b;
                blockDim = s.block;
                gridDim = s.grid;
                detail::t_sync = &sync;
                detail::t_tid = t;
                body();
                sync.exit(t);
              });
            } catch (const std::system_error& e) {
              die(kExitHarness, "cannot create thread %u of block (%u,%u,%u) of %s: %s", t, b.x, b.y, b.z, s.entry,
                  e.what());
            }
          }
          for (std::thread& th : threads) th.join();
        }
        wd.leave();
        check_guards(s.entry, b, dyn);
      }
    }
    g_dyn_smem = nullptr;
  }

  void store_all() const {
    for (size_t i = 0; i < bufs_.size(); ++i) {
      const std::string path = dir_ + "/buf_" + std::to_string(i) + ".out";
      FILE* f = fopen(path.c_str(), "wb");
      if (f == nullptr) die(kExitIo, "cannot create %s", path.c_str());
      const size_t bytes = bufs_[i].region.bytes();
      const size_t put = bytes == 0 ? 0 : fwrite(bufs_[i].region.data(), 1, bytes, f);
      if (put != bytes || fclose(f) != 0) die(kExitIo, "cannot write %zu bytes to %s", bytes, path.c_str());
    }
  }

 private:
  struct Buf {
    std::string name;
    size_t elem_size;
    size_t count;
    Region region;
  };

  void check_guards(const char* entry, uint3 b, const Region& dyn) const {
    long long off = 0;
    for (const Buf& buf : bufs_) {
      if (buf.region.corrupted_guard(&off)) {
        die(kExitOutOfBounds,
            "out-of-bounds write in %s: buffer '%s' (%zu elements of %zu bytes = %zu bytes) has its guard byte at "
            "offset %lld overwritten; first seen after block (%u,%u,%u)",
            entry, buf.name.c_str(), buf.count, buf.elem_size, buf.region.bytes(), off, b.x, b.y, b.z);
      }
    }
    if (dyn.corrupted_guard(&off)) {
      die(kExitOutOfBounds,
          "out-of-bounds write in %s: dynamic shared memory (%zu bytes) has its guard byte at offset %lld "
          "overwritten by block (%u,%u,%u)",
          entry, dyn.bytes(), off, b.x, b.y, b.z);
    }
  }

  std::string dir_;
  std::vector<Buf> bufs_;
};

}  // namespace cuda_emu

// ---------------------------------------------------------------------------------------
// CUDA keywords, intrinsics and math. Defined after every system header is included, so no
// macro below can change a library declaration.

#define __CUDA_ARCH__ CUDA_EMU_ARCH
#define __CUDACC__ 1
#define __CUDACC_RTC__ 1
#define __global__
#define __device__
#define __host__
#define __constant__
#define __managed__
#define __grid_constant__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
#define __shared__ static CUDA_EMU_SMEM_ATTR

#define __syncthreads() ::cuda_emu::syncthreads(__LINE__)
#define __syncthreads_count(p) ::cuda_emu::syncthreads_count(__LINE__, (p))
#define __syncthreads_and(p) ::cuda_emu::syncthreads_and(__LINE__, (p))
#define __syncthreads_or(p) ::cuda_emu::syncthreads_or(__LINE__, (p))
#define __syncwarp(...) ::cuda_emu::syncwarp(__LINE__ __VA_OPT__(, ) __VA_ARGS__)
#define __shfl_sync(...) ::cuda_emu::shfl_sync(__LINE__, __VA_ARGS__)
#define __shfl_up_sync(...) ::cuda_emu::shfl_up_sync(__LINE__, __VA_ARGS__)
#define __shfl_down_sync(...) ::cuda_emu::shfl_down_sync(__LINE__, __VA_ARGS__)
#define __shfl_xor_sync(...) ::cuda_emu::shfl_xor_sync(__LINE__, __VA_ARGS__)
#define __ballot_sync(...) ::cuda_emu::ballot_sync(__LINE__, __VA_ARGS__)
#define __any_sync(...) ::cuda_emu::any_sync(__LINE__, __VA_ARGS__)
#define __all_sync(...) ::cuda_emu::all_sync(__LINE__, __VA_ARGS__)
#define __reduce_add_sync(...) ::cuda_emu::reduce_add_sync(__LINE__, __VA_ARGS__)
#define __reduce_min_sync(...) ::cuda_emu::reduce_min_sync(__LINE__, __VA_ARGS__)
#define __reduce_max_sync(...) ::cuda_emu::reduce_max_sync(__LINE__, __VA_ARGS__)
#define __reduce_and_sync(...) ::cuda_emu::reduce_and_sync(__LINE__, __VA_ARGS__)
#define __reduce_or_sync(...) ::cuda_emu::reduce_or_sync(__LINE__, __VA_ARGS__)
#define __reduce_xor_sync(...) ::cuda_emu::reduce_xor_sync(__LINE__, __VA_ARGS__)

inline void __threadfence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
inline void __threadfence_block() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
inline void __threadfence_system() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
[[noreturn]] inline void __trap() {
  ::cuda_emu::die(::cuda_emu::kExitDiagnostic, "__trap() in block (%u,%u,%u) thread (%u,%u,%u)", blockIdx.x,
                  blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}
inline void __nanosleep(unsigned int ns) { std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); }
inline long long clock64() {
  return static_cast<long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
template <class T>
inline T __ldg(const T* p) {
  return *p;
}

inline int atomicAdd(int* a, int v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
inline unsigned int atomicAdd(unsigned int* a, unsigned int v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) {
  return __atomic_fetch_add(a, v, __ATOMIC_RELAXED);
}
inline float atomicAdd(float* a, float v) { return ::cuda_emu::atomic_rmw(a, [v](float o) { return o + v; }); }
inline double atomicAdd(double* a, double v) { return ::cuda_emu::atomic_rmw(a, [v](double o) { return o + v; }); }
inline int atomicSub(int* a, int v) { return __atomic_fetch_sub(a, v, __ATOMIC_RELAXED); }
inline unsigned int atomicSub(unsigned int* a, unsigned int v) { return __atomic_fetch_sub(a, v, __ATOMIC_RELAXED); }
inline int atomicExch(int* a, int v) { return __atomic_exchange_n(a, v, __ATOMIC_RELAXED); }
inline unsigned int atomicExch(unsigned int* a, unsigned int v) { return __atomic_exchange_n(a, v, __ATOMIC_RELAXED); }
inline unsigned long long atomicExch(unsigned long long* a, unsigned long long v) {
  return __atomic_exchange_n(a, v, __ATOMIC_RELAXED);
}
inline float atomicExch(float* a, float v) {
  float old;
  __atomic_exchange(a, &v, &old, __ATOMIC_RELAXED);
  return old;
}
#define CUDA_EMU_ATOMIC_MINMAX(T)                                                                        \
  inline T atomicMin(T* a, T v) { return ::cuda_emu::atomic_rmw(a, [v](T o) { return v < o ? v : o; }); } \
  inline T atomicMax(T* a, T v) { return ::cuda_emu::atomic_rmw(a, [v](T o) { return v > o ? v : o; }); }
CUDA_EMU_ATOMIC_MINMAX(int)
CUDA_EMU_ATOMIC_MINMAX(unsigned int)
CUDA_EMU_ATOMIC_MINMAX(long long)
CUDA_EMU_ATOMIC_MINMAX(unsigned long long)
#undef CUDA_EMU_ATOMIC_MINMAX
#define CUDA_EMU_ATOMIC_BITS(T)                                                                   \
  inline T atomicAnd(T* a, T v) { return __atomic_fetch_and(a, v, __ATOMIC_RELAXED); }            \
  inline T atomicOr(T* a, T v) { return __atomic_fetch_or(a, v, __ATOMIC_RELAXED); }              \
  inline T atomicXor(T* a, T v) { return __atomic_fetch_xor(a, v, __ATOMIC_RELAXED); }            \
  inline T atomicCAS(T* a, T compare, T v) {                                                      \
    __atomic_compare_exchange_n(a, &compare, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);       \
    return compare;                                                                               \
  }
CUDA_EMU_ATOMIC_BITS(int)
CUDA_EMU_ATOMIC_BITS(unsigned int)
CUDA_EMU_ATOMIC_BITS(unsigned long long)
#undef CUDA_EMU_ATOMIC_BITS
inline unsigned short atomicCAS(unsigned short* a, unsigned short compare, unsigned short v) {
  __atomic_compare_exchange_n(a, &compare, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return compare;
}
inline unsigned int atomicInc(unsigned int* a, unsigned int v) {
  return ::cuda_emu::atomic_rmw(a, [v](unsigned int o) { return o >= v ? 0u : o + 1u; });
}
inline unsigned int atomicDec(unsigned int* a, unsigned int v) {
  return ::cuda_emu::atomic_rmw(a, [v](unsigned int o) { return (o == 0 || o > v) ? v : o - 1u; });
}
#define atomicAdd_block atomicAdd
#define atomicAdd_system atomicAdd
#define atomicMax_block atomicMax
#define atomicMin_block atomicMin
#define atomicCAS_block atomicCAS
#define atomicExch_block atomicExch

// Integer intrinsics.
inline int __popc(unsigned int x) { return __builtin_popcount(x); }
inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
inline int __clz(int x) { return x == 0 ? 32 : __builtin_clz(static_cast<unsigned int>(x)); }
inline int __clzll(long long x) { return x == 0 ? 64 : __builtin_clzll(static_cast<unsigned long long>(x)); }
inline int __ffs(int x) { return __builtin_ffs(x); }
inline int __ffsll(long long x) { return __builtin_ffsll(x); }
inline unsigned int __brev(unsigned int x) {
  unsigned int r = 0;
  for (int i = 0; i < 32; ++i) r |= ((x >> i) & 1u) << (31 - i);
  return r;
}
inline unsigned long long __brevll(unsigned long long x) {
  unsigned long long r = 0;
  for (int i = 0; i < 64; ++i) r |= ((x >> i) & 1ull) << (63 - i);
  return r;
}
inline int __mulhi(int a, int b) { return static_cast<int>((static_cast<long long>(a) * b) >> 32); }
inline unsigned int __umulhi(unsigned int a, unsigned int b) {
  return static_cast<unsigned int>((static_cast<unsigned long long>(a) * b) >> 32);
}
inline long long __mul64hi(long long a, long long b) {
  return static_cast<long long>((static_cast<__int128>(a) * b) >> 64);
}
inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
  return static_cast<unsigned long long>((static_cast<unsigned __int128>(a) * b) >> 64);
}
inline int __mul24(int a, int b) {
  const long long sa = static_cast<long long>(static_cast<int>(static_cast<unsigned int>(a) << 8) >> 8);
  const long long sb = static_cast<long long>(static_cast<int>(static_cast<unsigned int>(b) << 8) >> 8);
  return static_cast<int>(static_cast<unsigned int>(sa * sb));
}
inline unsigned int __umul24(unsigned int a, unsigned int b) { return (a & 0xffffffu) * (b & 0xffffffu); }

// CUDA's global min/max overloads.
inline int min(int a, int b) { return a < b ? a : b; }
inline unsigned int min(unsigned int a, unsigned int b) { return a < b ? a : b; }
inline unsigned int min(int a, unsigned int b) { return min(static_cast<unsigned int>(a), b); }
inline unsigned int min(unsigned int a, int b) { return min(a, static_cast<unsigned int>(b)); }
inline long min(long a, long b) { return a < b ? a : b; }
inline unsigned long min(unsigned long a, unsigned long b) { return a < b ? a : b; }
inline long long min(long long a, long long b) { return a < b ? a : b; }
inline unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
inline float min(float a, float b) { return fminf(a, b); }
inline double min(double a, double b) { return fmin(a, b); }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned int max(unsigned int a, unsigned int b) { return a > b ? a : b; }
inline unsigned int max(int a, unsigned int b) { return max(static_cast<unsigned int>(a), b); }
inline unsigned int max(unsigned int a, int b) { return max(a, static_cast<unsigned int>(b)); }
inline long max(long a, long b) { return a > b ? a : b; }
inline unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }
inline long long max(long long a, long long b) { return a > b ? a : b; }
inline unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
inline float max(float a, float b) { return fmaxf(a, b); }
inline double max(double a, double b) { return fmax(a, b); }

// Reinterpretation and rounding conversions (CUDA saturates out-of-range and maps NaN to 0).
inline int __float_as_int(float f) { return static_cast<int>(::cuda_emu::f32_bits(f)); }
inline unsigned int __float_as_uint(float f) { return ::cuda_emu::f32_bits(f); }
inline float __int_as_float(int i) { return ::cuda_emu::f32_from_bits(static_cast<uint32_t>(i)); }
inline float __uint_as_float(unsigned int u) { return ::cuda_emu::f32_from_bits(u); }
inline long long __double_as_longlong(double d) { return static_cast<long long>(::cuda_emu::f64_bits(d)); }
inline double __longlong_as_double(long long i) { return ::cuda_emu::f64_from_bits(static_cast<uint64_t>(i)); }
namespace cuda_emu {
template <class I>
inline I saturate_to(double r) {
  if (r != r) return 0;
  if (r <= static_cast<double>(std::numeric_limits<I>::min())) return std::numeric_limits<I>::min();
  if (r >= static_cast<double>(std::numeric_limits<I>::max())) return std::numeric_limits<I>::max();
  return static_cast<I>(r);
}
}  // namespace cuda_emu
inline int __float2int_rn(float x) { return ::cuda_emu::saturate_to<int>(nearbyint(static_cast<double>(x))); }
inline int __float2int_rz(float x) { return ::cuda_emu::saturate_to<int>(trunc(static_cast<double>(x))); }
inline int __float2int_rd(float x) { return ::cuda_emu::saturate_to<int>(floor(static_cast<double>(x))); }
inline int __float2int_ru(float x) { return ::cuda_emu::saturate_to<int>(ceil(static_cast<double>(x))); }
inline unsigned int __float2uint_rn(float x) {
  return ::cuda_emu::saturate_to<unsigned int>(nearbyint(static_cast<double>(x)));
}
inline unsigned int __float2uint_rz(float x) { return ::cuda_emu::saturate_to<unsigned int>(trunc(static_cast<double>(x))); }
inline long long __float2ll_rn(float x) { return ::cuda_emu::saturate_to<long long>(nearbyint(static_cast<double>(x))); }
inline long long __float2ll_rz(float x) { return ::cuda_emu::saturate_to<long long>(trunc(static_cast<double>(x))); }
inline int __double2int_rn(double x) { return ::cuda_emu::saturate_to<int>(nearbyint(x)); }
inline float __int2float_rn(int i) { return static_cast<float>(i); }
inline float __uint2float_rn(unsigned int u) { return static_cast<float>(u); }
inline float __ll2float_rn(long long i) { return static_cast<float>(i); }
inline float __double2float_rn(double d) { return static_cast<float>(d); }

// Correctly rounded operations: identical bits on the CPU (with -ffp-contract=off).
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fmaf_rn(float a, float b, float c) { return fmaf(a, b, c); }
inline float __frcp_rn(float a) { return 1.0f / a; }
inline float __fsqrt_rn(float a) { return sqrtf(a); }
inline double __dadd_rn(double a, double b) { return a + b; }
inline double __dmul_rn(double a, double b) { return a * b; }
inline double __fma_rn(double a, double b, double c) { return fma(a, b, c); }
inline float __saturatef(float x) {
  if (!(x > 0.0f)) return 0.0f;  // NaN and non-positive
  return x < 1.0f ? x : 1.0f;
}

// Approximate / transcendental intrinsics. These are libm's, NOT libdevice's: the ulps
// differ, and the Rust harness refuses a bitwise comparison for any kernel that calls them.
namespace cuda_emu::approx {
inline float rsqrt(float x) { return 1.0f / sqrtf(x); }
inline float rcbrt(float x) { return 1.0f / cbrtf(x); }
inline float exp10(float x) { return powf(10.0f, x); }
inline void sincos(float x, float* s, float* c) {
  *s = sinf(x);
  *c = cosf(x);
}
inline float frsqrt_rn(float x) { return static_cast<float>(1.0 / sqrt(static_cast<double>(x))); }
}  // namespace cuda_emu::approx
inline float rsqrtf(float x) { return ::cuda_emu::approx::rsqrt(x); }
inline float rcbrtf(float x) { return ::cuda_emu::approx::rcbrt(x); }
inline double rsqrt(double x) { return 1.0 / sqrt(x); }
#define exp10f(x) ::cuda_emu::approx::exp10(x)
#define sincosf(x, s, c) ::cuda_emu::approx::sincos((x), (s), (c))
#define __expf(x) expf(x)
#define __exp10f(x) ::cuda_emu::approx::exp10(x)
#define __logf(x) logf(x)
#define __log2f(x) log2f(x)
#define __log10f(x) log10f(x)
#define __powf(x, y) powf((x), (y))
#define __sinf(x) sinf(x)
#define __cosf(x) cosf(x)
#define __tanf(x) tanf(x)
#define __sincosf(x, s, c) ::cuda_emu::approx::sincos((x), (s), (c))
#define __fdividef(a, b) ((a) / (b))
#define __frsqrt_rn(x) ::cuda_emu::approx::frsqrt_rn(x)
