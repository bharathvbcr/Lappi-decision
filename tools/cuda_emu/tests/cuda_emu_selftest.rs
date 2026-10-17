//! Self-tests of the CUDA-C CPU emulator (`tests/cuda_emu/`).
//!
//! Each bug class the emulator exists to catch has a kernel that commits it and a test that
//! requires the emulator to report it, next to the corrected kernel passing. These run the
//! EMULATOR, on the CPU; nothing here is a GPU result.

mod cuda_emu;

use std::path::PathBuf;
use std::time::Duration;

use cuda_emu::{
    Arg, Bf16, BlockOrder, Dim3, EmuError, ErrorKind, F16, Kernel, Outputs, POISON_F32_BITS,
    PROVENANCE, Program, Sanitizer, assert_bitwise_eq_f32, assert_close_ulps_f32, ulp_distance_f32,
};

/// The options a bitwise-comparable kernel is compiled with by NVRTC in these tests.
const NVRTC_EXACT: &[&str] = &["--fmad=false"];

fn kernel(label: &str, src: &str) -> Kernel {
    Kernel::new(label, src).with_nvrtc_options(NVRTC_EXACT)
}

fn ok(result: Result<Outputs, EmuError>) -> Outputs {
    result.unwrap_or_else(|e| panic!("expected the emulator run to succeed:\n{e}"))
}

fn expect_err(result: Result<Outputs, EmuError>, kind: ErrorKind) -> EmuError {
    match result {
        Ok(_) => panic!("expected {kind:?}, but the run succeeded"),
        Err(e) if e.kind == kind => e,
        Err(e) => panic!("expected {kind:?}, got:\n{e}"),
    }
}

/// Deterministic test data with magnitudes spread over 2^-6..2^6, so that summation order
/// shows up in the rounding.
struct Data(u64);

impl Data {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
        let scale = ((self.next_u64() % 13) as i32 - 6) as f32;
        unit * scale.exp2()
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

// ------------------------------------------------------------------------------------------
// Floating point: contraction and reduction order.

#[test]
fn fp_contract_off_keeps_a_times_b_minus_c_unfused() {
    let k = kernel(
        "mul_sub",
        r#"extern "C" __global__ void mul_sub(const float* x, float* y) { y[0] = x[0] * x[0] - 1.0f; }"#,
    );
    let x = 1.0f32 + 2.0f32.powi(-12);
    let mut p = Program::new(k);
    let xb = p.input("x", &[x]);
    let yb = p.output::<f32>("y", 1);
    p.launch(
        "mul_sub",
        Dim3::x(1),
        Dim3::x(1),
        0,
        &[Arg::Buf(xb), Arg::Buf(yb)],
    );
    let y = ok(p.run()).get::<f32>(yb)[0];
    let fused = x.mul_add(x, -1.0);
    assert_eq!(
        y,
        2.0f32.powi(-11),
        "x*x-1 must round x*x first (NVRTC --fmad=false)"
    );
    assert_ne!(
        y.to_bits(),
        fused.to_bits(),
        "the fused value would mean contraction happened"
    );
}

const RMS_TEMPLATE: &str = r#"
extern "C" __global__ void rms_rows(const float* __restrict__ x, const float* __restrict__ w,
                                    float* __restrict__ y, float* __restrict__ ss, int cols, float eps) {
  __shared__ float partial[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  const float* xr = x + (size_t)row * cols;
  float acc = 0.0f;
  for (int i = tid; i < cols; i += blockDim.x) {
    const float v = xr[i];
    acc += v * v;
  }
  for (int o = OFFSET_START; OFFSET_COND; o = OFFSET_STEP) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) partial[warp] = acc;
  FIRST_BARRIER
  if (warp == 0) {
    float t = lane < nwarps ? partial[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) partial[0] = t;
  }
  __syncthreads();
  if (tid == 0) ss[row] = partial[0];
  const float inv = 1.0f / sqrtf(partial[0] / (float)cols + eps);
  for (int i = tid; i < cols; i += blockDim.x) y[(size_t)row * cols + i] = xr[i] * inv * w[i];
}
"#;

#[derive(Clone, Copy)]
enum Offsets {
    /// 16, 8, 4, 2, 1: the order the host mirror specifies.
    Descending,
    /// 1, 2, 4, 8, 16: also a full butterfly, but a different summation order.
    Ascending,
    /// 8, 4, 2, 1: the bug - each lane sums only its half of the warp.
    FromEight,
}

fn rms_source(first_barrier: bool, offsets: Offsets) -> String {
    let (start, cond, step) = match offsets {
        Offsets::Descending => ("16", "o > 0", "o >> 1"),
        Offsets::Ascending => ("1", "o < 32", "o << 1"),
        Offsets::FromEight => ("8", "o > 0", "o >> 1"),
    };
    RMS_TEMPLATE
        .replace("OFFSET_START", start)
        .replace("OFFSET_COND", cond)
        .replace("OFFSET_STEP", step)
        .replace(
            "FIRST_BARRIER",
            if first_barrier {
                "__syncthreads();"
            } else {
                ""
            },
        )
}

const RMS_ROWS: usize = 16;
const RMS_COLS: usize = 1000;
const RMS_BLOCK: usize = 128;
const RMS_EPS: f32 = 1e-6;

fn butterfly(v: &mut [f32; 32], offsets: &[usize]) {
    for &o in offsets {
        let old = *v;
        for l in 0..32 {
            v[l] = old[l] + old[l ^ o];
        }
    }
}

/// The host mirror: the kernel's arithmetic in the kernel's order, thread by thread.
/// Returns the normalised rows and each row's sum of squares.
fn rms_mirror(x: &[f32], w: &[f32], offsets: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; RMS_ROWS * RMS_COLS];
    let mut ss = vec![0.0f32; RMS_ROWS];
    for row in 0..RMS_ROWS {
        let xr = &x[row * RMS_COLS..(row + 1) * RMS_COLS];
        let mut acc = [0.0f32; RMS_BLOCK];
        for (t, a) in acc.iter_mut().enumerate() {
            let mut i = t;
            while i < RMS_COLS {
                *a += xr[i] * xr[i];
                i += RMS_BLOCK;
            }
        }
        let nwarps = RMS_BLOCK / 32;
        let mut partial = [0.0f32; 32];
        for (warp, p) in partial.iter_mut().enumerate().take(nwarps) {
            let mut lanes = [0.0f32; 32];
            lanes.copy_from_slice(&acc[warp * 32..warp * 32 + 32]);
            butterfly(&mut lanes, offsets);
            *p = lanes[0];
        }
        let mut lanes = [0.0f32; 32];
        lanes[..nwarps].copy_from_slice(&partial[..nwarps]);
        butterfly(&mut lanes, &[16, 8, 4, 2, 1]);
        ss[row] = lanes[0];
        let inv = 1.0f32 / (lanes[0] / RMS_COLS as f32 + RMS_EPS).sqrt();
        for i in 0..RMS_COLS {
            y[row * RMS_COLS + i] = xr[i] * inv * w[i];
        }
    }
    (y, ss)
}

struct Rms {
    program: Program,
    y: cuda_emu::BufId,
    ss: cuda_emu::BufId,
    x: Vec<f32>,
    w: Vec<f32>,
}

fn rms_program(src: &str) -> Rms {
    let mut data = Data(0x9e37_79b9_7f4a_7c15);
    let x = data.vec(RMS_ROWS * RMS_COLS);
    let w = data.vec(RMS_COLS);
    let mut p = Program::new(kernel("rms_rows", src));
    let xb = p.input("x", &x);
    let wb = p.input("w", &w);
    let yb = p.output::<f32>("y", RMS_ROWS * RMS_COLS);
    let ssb = p.output::<f32>("ss", RMS_ROWS);
    p.launch(
        "rms_rows",
        Dim3::x(RMS_ROWS as u32),
        Dim3::x(RMS_BLOCK as u32),
        0,
        &[
            Arg::Buf(xb),
            Arg::Buf(wb),
            Arg::Buf(yb),
            Arg::Buf(ssb),
            Arg::I32(RMS_COLS as i32),
            Arg::F32(RMS_EPS),
        ],
    );
    Rms {
        program: p,
        y: yb,
        ss: ssb,
        x,
        w,
    }
}

#[test]
fn block_reduction_matches_its_host_mirror_bitwise() {
    let r = rms_program(&rms_source(true, Offsets::Descending));
    let out = ok(r.program.run());
    let (want_y, want_ss) = rms_mirror(&r.x, &r.w, &[16, 8, 4, 2, 1]);
    let k = r.program.kernel();
    assert_bitwise_eq_f32(
        k,
        "rms_rows sum of squares vs mirror",
        &out.get::<f32>(r.ss),
        &want_ss,
    );
    assert_bitwise_eq_f32(
        k,
        "rms_rows output vs mirror",
        &out.get::<f32>(r.y),
        &want_y,
    );
}

#[test]
fn reduction_order_change_is_caught_bitwise_where_a_tolerance_would_pass() {
    let r = rms_program(&rms_source(true, Offsets::Ascending));
    let got = ok(r.program.run()).get::<f32>(r.ss);
    let (_, want) = rms_mirror(&r.x, &r.w, &[16, 8, 4, 2, 1]);
    let differing = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    let worst = got
        .iter()
        .zip(&want)
        .filter_map(|(g, w)| ulp_distance_f32(*g, *w))
        .max()
        .unwrap_or(0);
    assert!(
        differing > 0,
        "the ascending butterfly must not reproduce the descending one's bits"
    );
    assert!(
        worst <= 64,
        "the order change is a rounding difference ({worst} ulp), invisible to a loose tolerance"
    );
    // And the mirror of the order the kernel actually uses reproduces it exactly.
    let (_, ascending) = rms_mirror(&r.x, &r.w, &[1, 2, 4, 8, 16]);
    assert_bitwise_eq_f32(
        r.program.kernel(),
        "ascending kernel vs ascending mirror",
        &got,
        &ascending,
    );
}

#[test]
fn shuffle_offset_bug_loses_half_of_each_warp() {
    let r = rms_program(&rms_source(true, Offsets::FromEight));
    let got = ok(r.program.run()).get::<f32>(r.ss);
    let (_, want) = rms_mirror(&r.x, &r.w, &[16, 8, 4, 2, 1]);
    let worst_rel = got
        .iter()
        .zip(&want)
        .map(|(g, w)| ((g - w) / w).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst_rel > 0.05,
        "starting the butterfly at 8 must change the result grossly (got {worst_rel})"
    );
}

// ------------------------------------------------------------------------------------------
// Barriers.

#[test]
fn missing_syncthreads_is_reported_as_a_race_under_tsan() {
    let mut r = rms_program(&rms_source(false, Offsets::Descending));
    r.program.sanitizer(Sanitizer::Thread);
    let e = expect_err(r.program.run(), ErrorKind::Race);
    assert!(e.stderr.contains("data race"), "{e}");
}

#[test]
fn correctly_barriered_kernel_is_race_free_under_tsan() {
    let mut r = rms_program(&rms_source(true, Offsets::Descending));
    r.program.sanitizer(Sanitizer::Thread);
    let out = ok(r.program.run());
    let (want_y, _) = rms_mirror(&r.x, &r.w, &[16, 8, 4, 2, 1]);
    assert_bitwise_eq_f32(
        r.program.kernel(),
        "rms_rows under tsan",
        &out.get::<f32>(r.y),
        &want_y,
    );
}

const WARP_EXCHANGE: &str = r#"
extern "C" __global__ void warp_exchange(float* out, int use_syncwarp) {
  __shared__ float s[32];
  const unsigned lane = threadIdx.x;
  s[lane] = (float)lane;
  if (use_syncwarp) {
    __syncwarp();
  } else {
    (void)__shfl_xor_sync(0xffffffffu, 0, 1);
  }
  out[lane] = s[lane ^ 1];
}
"#;

fn warp_exchange(use_syncwarp: bool) -> (Result<Outputs, EmuError>, cuda_emu::BufId) {
    let mut p = Program::new(kernel("warp_exchange", WARP_EXCHANGE));
    let out = p.output::<f32>("out", 32);
    p.launch(
        "warp_exchange",
        Dim3::x(1),
        Dim3::x(32),
        0,
        &[Arg::Buf(out), Arg::I32(i32::from(use_syncwarp))],
    );
    p.sanitizer(Sanitizer::Thread);
    (p.run(), out)
}

#[test]
fn syncwarp_orders_shared_memory_between_lanes_under_tsan() {
    let (result, out) = warp_exchange(true);
    let want: Vec<f32> = (0..32).map(|l| (l ^ 1) as f32).collect();
    assert_eq!(ok(result).get::<f32>(out), want);
}

#[test]
fn a_shuffle_orders_no_memory_so_using_it_as_a_barrier_is_a_race() {
    let e = expect_err(warp_exchange(false).0, ErrorKind::Race);
    assert!(e.stderr.contains("data race"), "{e}");
}

#[test]
fn template_kernel_with_a_barrier_loop_runs_by_template_id_and_is_race_free() {
    let k = kernel(
        "scan",
        r#"
template <int BLOCK>
__global__ void inclusive_scan(const int* x, int* y) {
  __shared__ int s[BLOCK];
  const int t = threadIdx.x;
  s[t] = x[blockIdx.x * BLOCK + t];
  __syncthreads();
  for (int off = 1; off < BLOCK; off <<= 1) {
    const int v = t >= off ? s[t - off] : 0;
    __syncthreads();
    s[t] += v;
    __syncthreads();
  }
  y[blockIdx.x * BLOCK + t] = s[t];
}
"#,
    );
    let x: Vec<i32> = (0..128).map(|i| (i * 7 % 11) - 5).collect();
    let mut want = Vec::with_capacity(128);
    for blk in x.chunks(64) {
        let mut acc = 0;
        for v in blk {
            acc += v;
            want.push(acc);
        }
    }
    for mode in [Sanitizer::None, Sanitizer::Thread] {
        let mut p = Program::new(k.clone());
        let xb = p.input("x", &x);
        let yb = p.output::<i32>("y", 128);
        p.launch(
            "inclusive_scan<64>",
            Dim3::x(2),
            Dim3::x(64),
            0,
            &[Arg::Buf(xb), Arg::Buf(yb)],
        );
        p.sanitizer(mode);
        assert_eq!(ok(p.run()).get::<i32>(yb), want, "{mode:?}");
    }
}

#[test]
fn divergent_syncthreads_is_reported() {
    let k = kernel(
        "divergent_bar",
        r#"
extern "C" __global__ void divergent_bar(float* y) {
  if (threadIdx.x < 16) {
    __syncthreads();
  }
  __syncthreads();
  y[threadIdx.x] = 1.0f;
}
"#,
    );
    let mut p = Program::new(k);
    let y = p.output::<f32>("y", 32);
    p.launch("divergent_bar", Dim3::x(1), Dim3::x(32), 0, &[Arg::Buf(y)]);
    let e = expect_err(p.run(), ErrorKind::Diagnostic);
    assert!(e.stderr.contains("divergent barrier"), "{e}");
}

#[test]
fn threads_that_exit_early_do_not_block_a_barrier() {
    let k = kernel(
        "early_exit",
        r#"
extern "C" __global__ void early_exit(const float* x, float* y, int n) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  if (t >= n) return;
  s[t] = x[t];
  __syncthreads();
  y[t] = s[n - 1 - t];
}
"#,
    );
    let x: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let mut p = Program::new(k);
    let xb = p.input("x", &x);
    let yb = p.output::<f32>("y", 40);
    p.launch(
        "early_exit",
        Dim3::x(1),
        Dim3::x(64),
        0,
        &[Arg::Buf(xb), Arg::Buf(yb), Arg::I32(40)],
    );
    let want: Vec<f32> = x.iter().rev().copied().collect();
    assert_bitwise_eq_f32(p.kernel(), "reversed", &ok(p.run()).get::<f32>(yb), &want);
}

#[test]
fn three_lane_wait_cycle_is_a_deadlock_not_a_hang() {
    let k = kernel(
        "cycle",
        r#"
extern "C" __global__ void cycle(int* y) {
  const unsigned l = threadIdx.x;
  if (l == 0) __syncwarp(0x3u);       // waits for lane 1
  else if (l == 1) __syncwarp(0x6u);  // waits for lane 2
  else if (l == 2) __syncwarp(0x5u);  // waits for lane 0
  y[l] = 1;
}
"#,
    );
    let mut p = Program::new(k);
    let y = p.output::<i32>("y", 32);
    p.launch("cycle", Dim3::x(1), Dim3::x(32), 0, &[Arg::Buf(y)]);
    let e = expect_err(p.run(), ErrorKind::Deadlock);
    assert!(e.stderr.contains("DEADLOCK"), "{e}");
}

#[test]
fn lanes_waiting_at_different_collectives_are_reported() {
    let k = kernel(
        "mismatch",
        r#"
extern "C" __global__ void mismatch(const float* x, float* y) {
  float v = x[threadIdx.x];
  if (threadIdx.x < 16) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
  } else {
    __syncwarp();
  }
  y[threadIdx.x] = v;
}
"#,
    );
    let mut p = Program::new(k);
    let x = p.input("x", &[1.0f32; 32]);
    let y = p.output::<f32>("y", 32);
    p.launch(
        "mismatch",
        Dim3::x(1),
        Dim3::x(32),
        0,
        &[Arg::Buf(x), Arg::Buf(y)],
    );
    let e = expect_err(p.run(), ErrorKind::Diagnostic);
    assert!(e.stderr.contains("wait for each other"), "{e}");
}

#[test]
fn a_spin_wait_hits_the_wall_clock_timeout() {
    let k = kernel(
        "spin",
        r#"
extern "C" __global__ void spin(volatile int* flag) {
  if (threadIdx.x == 0) {
    while (flag[0] == 0) {
    }
  }
  __syncthreads();
}
"#,
    );
    let mut p = Program::new(k);
    let flag = p.zeroed::<i32>("flag", 1);
    p.launch("spin", Dim3::x(1), Dim3::x(32), 0, &[Arg::Buf(flag)]);
    p.launch_timeout(Duration::from_millis(1500));
    let e = expect_err(p.run(), ErrorKind::Timeout);
    assert!(e.stderr.contains("TIMEOUT"), "{e}");
    assert!(
        e.stderr.contains("31 thread(s) at __syncthreads"),
        "the dump names who waits where:\n{e}"
    );
}

// ------------------------------------------------------------------------------------------
// Warp collectives.

/// CUDA Programming Guide semantics of the four shuffles, written independently of the
/// PTX formulation the shim implements.
fn shfl_reference(mode: i32, lane: usize, b: usize, width: usize, vals: &[i32; 32]) -> i32 {
    let seg = lane / width * width;
    match mode {
        0 => vals[seg + b % width],
        1 => {
            if lane < seg + b {
                vals[lane]
            } else {
                vals[lane - b]
            }
        }
        2 => {
            if lane + b >= seg + width {
                vals[lane]
            } else {
                vals[lane + b]
            }
        }
        _ => {
            let src = lane ^ b;
            if src >= seg + width {
                vals[lane]
            } else {
                vals[src]
            }
        }
    }
}

#[test]
fn shuffles_follow_the_cuda_semantics_for_every_mode_and_width() {
    let k = kernel(
        "shfl_table",
        r#"
extern "C" __global__ void shfl_table(const int* modes, const int* bs, const int* widths, int ncases, int* out,
                                      double* dout) {
  const int lane = threadIdx.x;
  for (int k = 0; k < ncases; ++k) {
    const int v = 1000 * k + lane;
    int r;
    switch (modes[k]) {
      case 0: r = __shfl_sync(0xffffffffu, v, bs[k], widths[k]); break;
      case 1: r = __shfl_up_sync(0xffffffffu, v, (unsigned)bs[k], widths[k]); break;
      case 2: r = __shfl_down_sync(0xffffffffu, v, (unsigned)bs[k], widths[k]); break;
      default: r = __shfl_xor_sync(0xffffffffu, v, bs[k], widths[k]); break;
    }
    out[k * 32 + lane] = r;
  }
  const double d = 1.0 + lane * 0x1p-40;
  dout[lane] = __shfl_xor_sync(0xffffffffu, d, 1);
}
"#,
    );
    let mut modes = Vec::new();
    let mut bs = Vec::new();
    let mut widths = Vec::new();
    for width in [32, 16, 8, 4, 2, 1] {
        for b in [0, 1, 2, 3, 5, 8, 16, 31] {
            for mode in 0..4 {
                modes.push(mode);
                bs.push(b);
                widths.push(width);
            }
        }
    }
    let n = modes.len();
    let mut p = Program::new(k);
    let mb = p.input("modes", &modes);
    let bb = p.input("bs", &bs);
    let wb = p.input("widths", &widths);
    let ob = p.output::<i32>("out", n * 32);
    let db = p.output::<f64>("dout", 32);
    p.launch(
        "shfl_table",
        Dim3::x(1),
        Dim3::x(32),
        0,
        &[
            Arg::Buf(mb),
            Arg::Buf(bb),
            Arg::Buf(wb),
            Arg::I32(n as i32),
            Arg::Buf(ob),
            Arg::Buf(db),
        ],
    );
    let out = ok(p.run());
    let got = out.get::<i32>(ob);
    for k in 0..n {
        let vals: [i32; 32] = std::array::from_fn(|l| 1000 * k as i32 + l as i32);
        for lane in 0..32 {
            let want = shfl_reference(modes[k], lane, bs[k] as usize, widths[k] as usize, &vals);
            assert_eq!(
                got[k * 32 + lane],
                want,
                "mode {} b {} width {} lane {lane}",
                modes[k],
                bs[k],
                widths[k]
            );
        }
    }
    let d = out.get::<f64>(db);
    for (lane, v) in d.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            (1.0 + (lane ^ 1) as f64 * 2f64.powi(-40)).to_bits(),
            "double lane {lane}"
        );
    }
}

#[test]
fn shuffle_from_an_exited_lane_is_reported() {
    let k = kernel(
        "exited_source",
        r#"
extern "C" __global__ void exited_source(const float* x, float* y) {
  if (threadIdx.x >= 16) return;
  float v = x[threadIdx.x];
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  y[threadIdx.x] = v;
}
"#,
    );
    let mut p = Program::new(k);
    let x = p.input("x", &[1.0f32; 32]);
    let y = p.output::<f32>("y", 32);
    p.launch(
        "exited_source",
        Dim3::x(1),
        Dim3::x(32),
        0,
        &[Arg::Buf(x), Arg::Buf(y)],
    );
    let e = expect_err(p.run(), ErrorKind::Diagnostic);
    assert!(e.stderr.contains("has exited"), "{e}");
}

#[test]
fn full_warp_reduction_in_a_partial_warp_is_reported() {
    let k = kernel(
        "partial_warp",
        r#"
extern "C" __global__ void partial_warp(const float* x, float* y) {
  float v = x[threadIdx.x];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  y[threadIdx.x] = v;
}
"#,
    );
    let mut p = Program::new(k);
    let x = p.input("x", &[1.0f32; 48]);
    let y = p.output::<f32>("y", 48);
    p.launch(
        "partial_warp",
        Dim3::x(1),
        Dim3::x(48),
        0,
        &[Arg::Buf(x), Arg::Buf(y)],
    );
    let e = expect_err(p.run(), ErrorKind::Diagnostic);
    assert!(e.stderr.contains("does not exist"), "{e}");
}

#[test]
fn votes_block_counts_and_warp_reductions() {
    let k = kernel(
        "votes",
        r#"
extern "C" __global__ void votes(unsigned* ballots, int* anys, int* alls, unsigned* sums, int* block_results) {
  const unsigned t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const unsigned b = __ballot_sync(0xffffffffu, (lane % 3) == 0);
  const int a = __any_sync(0xffffffffu, lane == 7 && warp == 1);
  const int l = __all_sync(0xffffffffu, lane < 32);
  const int c = __syncthreads_count(t % 5 == 0);
  const int sa = __syncthreads_and(t < 64);
  const int so = __syncthreads_or(t == 63);
  const unsigned s = __reduce_add_sync(0xffffffffu, lane);
  if (lane == 0) {
    ballots[warp] = b;
    anys[warp] = a;
    alls[warp] = l;
    sums[warp] = s;
  }
  if (t == 0) {
    block_results[0] = c;
    block_results[1] = sa;
    block_results[2] = so;
  }
}
"#,
    );
    let mut p = Program::new(k);
    let ballots = p.output::<u32>("ballots", 2);
    let anys = p.output::<i32>("anys", 2);
    let alls = p.output::<i32>("alls", 2);
    let sums = p.output::<u32>("sums", 2);
    let block = p.output::<i32>("block_results", 3);
    p.launch(
        "votes",
        Dim3::x(1),
        Dim3::x(64),
        0,
        &[
            Arg::Buf(ballots),
            Arg::Buf(anys),
            Arg::Buf(alls),
            Arg::Buf(sums),
            Arg::Buf(block),
        ],
    );
    let out = ok(p.run());
    let want_ballot = (0..32)
        .filter(|l| l % 3 == 0)
        .fold(0u32, |m, l| m | (1 << l));
    assert_eq!(out.get::<u32>(ballots), vec![want_ballot; 2]);
    assert_eq!(out.get::<i32>(anys), vec![0, 1]);
    assert_eq!(out.get::<i32>(alls), vec![1, 1]);
    assert_eq!(out.get::<u32>(sums), vec![496, 496]);
    assert_eq!(out.get::<i32>(block), vec![13, 1, 1]);
}

// ------------------------------------------------------------------------------------------
// Memory: bounds, poison, block order.

const SHIFT_COPY: &str = r#"
extern "C" __global__ void shift_copy(const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i + 1] = x[i];
}
extern "C" __global__ void read_past(const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i + 1];
}
"#;

#[test]
fn out_of_bounds_write_trips_the_guard() {
    let mut p = Program::new(kernel("shift_copy", SHIFT_COPY));
    let x = p.input("x", &[1.0f32; 256]);
    let y = p.output::<f32>("y", 256);
    p.launch(
        "shift_copy",
        Dim3::x(2),
        Dim3::x(128),
        0,
        &[Arg::Buf(x), Arg::Buf(y), Arg::I32(256)],
    );
    let e = expect_err(p.run(), ErrorKind::OutOfBounds);
    assert!(
        e.stderr.contains("buffer 'y'") && e.stderr.contains("offset 1024"),
        "{e}"
    );
}

#[test]
fn out_of_bounds_read_is_caught_by_the_address_sanitizer() {
    let mut p = Program::new(kernel("read_past", SHIFT_COPY));
    let x = p.input("x", &[1.0f32; 256]);
    let y = p.output::<f32>("y", 256);
    p.launch(
        "read_past",
        Dim3::x(2),
        Dim3::x(128),
        0,
        &[Arg::Buf(x), Arg::Buf(y), Arg::I32(256)],
    );
    p.sanitizer(Sanitizer::Address);
    let e = expect_err(p.run(), ErrorKind::Memory);
    assert!(e.stderr.contains("heap-buffer-overflow"), "{e}");
}

#[test]
fn int32_index_overflow_is_caught_by_the_ub_sanitizer() {
    let k = kernel(
        "row_offset",
        r#"
extern "C" __global__ void row_offset(const int* shape, long long* y) {
  const int row = shape[0], cols = shape[1];
  const int off = row * cols;   // 40000 * 60000 overflows int
  y[threadIdx.x] = off;
}
"#,
    );
    let mut p = Program::new(k);
    let shape = p.input("shape", &[40_000i32, 60_000]);
    let y = p.output::<i64>("y", 1);
    p.launch(
        "row_offset",
        Dim3::x(1),
        Dim3::x(1),
        0,
        &[Arg::Buf(shape), Arg::Buf(y)],
    );
    p.sanitizer(Sanitizer::Address);
    let e = expect_err(p.run(), ErrorKind::Memory);
    assert!(e.stderr.contains("signed integer overflow"), "{e}");
}

#[test]
fn an_output_element_nothing_writes_reads_back_as_poison() {
    let k = kernel(
        "short_write",
        r#"
extern "C" __global__ void short_write(float* y, int n) {
  const int i = threadIdx.x;
  if (i < n - 1) y[i] = 2.0f;
}
"#,
    );
    let mut p = Program::new(k);
    let y = p.output::<f32>("y", 32);
    p.launch(
        "short_write",
        Dim3::x(1),
        Dim3::x(32),
        0,
        &[Arg::Buf(y), Arg::I32(32)],
    );
    let got = ok(p.run()).get::<f32>(y);
    assert!(got[..31].iter().all(|v| *v == 2.0));
    assert_eq!(got[31].to_bits(), POISON_F32_BITS);
    let mut want = vec![2.0f32; 32];
    want[31] = 0.0;
    let caught =
        std::panic::catch_unwind(|| assert_bitwise_eq_f32(p.kernel(), "short_write", &got, &want));
    assert!(
        caught.is_err(),
        "the poisoned element must fail a comparison against a written value"
    );
}

#[test]
fn shared_memory_nothing_wrote_is_poison_in_every_block() {
    let k = kernel(
        "smem_poison",
        r#"
extern __shared__ float dyn[];
extern "C" __global__ void static_smem(float* out) {
  __shared__ float s[33];
  s[threadIdx.x] = (float)threadIdx.x;
  if (blockIdx.x == 0 && threadIdx.x == 0) s[32] = 99.0f;
  __syncthreads();
  out[blockIdx.x * 32 + threadIdx.x] = s[threadIdx.x + 1];
}
extern "C" __global__ void dynamic_smem(float* out) {
  dyn[threadIdx.x] = (float)threadIdx.x;
  if (blockIdx.x == 0 && threadIdx.x == 0) dyn[32] = 99.0f;
  __syncthreads();
  out[blockIdx.x * 32 + threadIdx.x] = dyn[threadIdx.x + 1];
}
"#,
    );
    let mut p = Program::new(k);
    let a = p.output::<f32>("a", 64);
    let b = p.output::<f32>("b", 64);
    p.launch("static_smem", Dim3::x(2), Dim3::x(32), 0, &[Arg::Buf(a)]);
    p.launch(
        "dynamic_smem",
        Dim3::x(2),
        Dim3::x(32),
        33 * 4,
        &[Arg::Buf(b)],
    );
    let out = ok(p.run());
    for (name, buf) in [("static", a), ("dynamic", b)] {
        let got = out.get::<f32>(buf);
        for blk in 0..2 {
            for t in 0..31 {
                assert_eq!(
                    got[blk * 32 + t],
                    (t + 1) as f32,
                    "{name} block {blk} thread {t}"
                );
            }
        }
        assert_eq!(got[31], 99.0, "{name}: block 0 wrote slot 32");
        assert_eq!(
            got[63].to_bits(),
            POISON_F32_BITS,
            "{name}: block 1 never wrote slot 32 and must not see block 0's value"
        );
    }
}

#[test]
fn blocks_writing_overlapping_elements_are_caught_by_block_order() {
    let k = kernel(
        "overlap",
        r#"extern "C" __global__ void overlap(unsigned* out) { out[blockIdx.x * 31 + threadIdx.x] = blockIdx.x; }"#,
    );
    let mut p = Program::new(k);
    let out = p.zeroed::<u32>("out", 4 * 31 + 1);
    p.launch("overlap", Dim3::x(4), Dim3::x(32), 0, &[Arg::Buf(out)]);
    let e = expect_err(p.run_order_invariant(), ErrorKind::OrderDependent);
    assert!(e.message.contains("element 31"), "{e}");
}

#[test]
fn multi_dimensional_indices_cover_the_grid_exactly_once_in_any_block_order() {
    let k = kernel(
        "coords",
        r#"
extern "C" __global__ void coords(unsigned* hits, unsigned* enc) {
  const unsigned nthreads = blockDim.x * blockDim.y * blockDim.z;
  const unsigned block_lin = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const unsigned thread_lin = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const unsigned g = block_lin * nthreads + thread_lin;
  atomicAdd(&hits[g], 1u);
  enc[g] = (blockIdx.x << 24) | (blockIdx.y << 20) | (blockIdx.z << 16) | (threadIdx.x << 8) | (threadIdx.y << 4) |
           threadIdx.z;
}
"#,
    );
    let (grid, block) = (Dim3::xyz(3, 2, 2), Dim3::xyz(4, 3, 2));
    let total = (grid.count() * block.count()) as usize;
    let mut p = Program::new(k);
    let hits = p.zeroed::<u32>("hits", total);
    let enc = p.output::<u32>("enc", total);
    p.launch("coords", grid, block, 0, &[Arg::Buf(hits), Arg::Buf(enc)]);
    let out = ok(p.run_order_invariant());
    assert!(
        out.get::<u32>(hits).iter().all(|h| *h == 1),
        "every thread exactly once"
    );
    let mut want = Vec::with_capacity(total);
    for bz in 0..2u32 {
        for by in 0..2u32 {
            for bx in 0..3u32 {
                for tz in 0..2u32 {
                    for ty in 0..3u32 {
                        for tx in 0..4u32 {
                            want.push(
                                (bx << 24) | (by << 20) | (bz << 16) | (tx << 8) | (ty << 4) | tz,
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(out.get::<u32>(enc), want);
}

#[test]
fn atomics_are_atomic_within_a_block() {
    let k = kernel(
        "atomics",
        r#"
extern "C" __global__ void atomics(unsigned* counter, float* fsum, int* imax, unsigned* inc) {
  atomicAdd(counter, 1u);
  atomicAdd(fsum, 1.0f);
  atomicMax(imax, (int)(blockIdx.x * blockDim.x + threadIdx.x));
  atomicInc(inc, 9u);
}
"#,
    );
    let mut p = Program::new(k);
    let c = p.zeroed::<u32>("counter", 1);
    let f = p.zeroed::<f32>("fsum", 1);
    let m = p.zeroed::<i32>("imax", 1);
    let i = p.zeroed::<u32>("inc", 1);
    p.launch(
        "atomics",
        Dim3::x(4),
        Dim3::x(256),
        0,
        &[Arg::Buf(c), Arg::Buf(f), Arg::Buf(m), Arg::Buf(i)],
    );
    p.sanitizer(Sanitizer::Thread);
    let out = ok(p.run());
    assert_eq!(out.get::<u32>(c), vec![1024]);
    assert_eq!(out.get::<f32>(f), vec![1024.0]);
    assert_eq!(out.get::<i32>(m), vec![1023]);
    assert_eq!(out.get::<u32>(i), vec![1024 % 10]);
}

// ------------------------------------------------------------------------------------------
// Source handling.

#[test]
fn extern_and_static_shared_declarations_are_rewritten_in_code_only() {
    let src = r#"
// extern __shared__ float not_this_one[];
extern __shared__ float file_scope[];
extern "C" __global__ void file_scope_dyn(float* out) {
  file_scope[threadIdx.x] = (float)threadIdx.x;
  __syncthreads();
  out[threadIdx.x] = file_scope[blockDim.x - 1 - threadIdx.x];
}
extern "C" __global__ void function_scope_dyn(float* out) {
  extern __shared__ __align__(16) unsigned char raw[];
  static __shared__ float tail[4];
  float4* v = reinterpret_cast<float4*>(raw);
  if (threadIdx.x < 4) tail[threadIdx.x] = 10.0f * threadIdx.x;
  if (threadIdx.x == 0) v[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  __syncthreads();
  const float* f = reinterpret_cast<const float*>(raw);
  out[threadIdx.x] = f[threadIdx.x % 4] + tail[threadIdx.x % 4];
}
"#;
    let mut p = Program::new(kernel("rewrite", src));
    let a = p.output::<f32>("a", 32);
    let b = p.output::<f32>("b", 32);
    p.launch(
        "file_scope_dyn",
        Dim3::x(1),
        Dim3::x(32),
        32 * 4,
        &[Arg::Buf(a)],
    );
    p.launch(
        "function_scope_dyn",
        Dim3::x(1),
        Dim3::x(32),
        16,
        &[Arg::Buf(b)],
    );
    let driver = p.driver_source().unwrap_or_else(|e| panic!("{e}"));
    assert!(
        driver.contains("// extern __shared__ float not_this_one[];"),
        "comments are left alone"
    );
    assert!(
        driver.contains("static float* const& file_scope ="),
        "{driver}"
    );
    assert!(
        driver.contains("static unsigned char* const& raw ="),
        "{driver}"
    );
    assert_eq!(
        driver
            .lines()
            .position(|l| l.contains("file_scope_dyn(float* out)")),
        src.lines()
            .position(|l| l.contains("file_scope_dyn(float* out)"))
            .map(|i| i + 5),
        "line numbers survive the rewrite (the driver adds a 5-line prologue)"
    );
    let out = ok(p.run());
    let want_a: Vec<f32> = (0..32).rev().map(|i| i as f32).collect();
    let want_b: Vec<f32> = (0..32)
        .map(|t| (1 + t % 4) as f32 + 10.0 * (t % 4) as f32)
        .collect();
    assert_eq!(out.get::<f32>(a), want_a);
    assert_eq!(out.get::<f32>(b), want_b);
}

#[test]
fn inline_ptx_is_refused() {
    let k = kernel(
        "asm",
        r#"extern "C" __global__ void fence() { asm volatile("membar.gl;"); }"#,
    );
    let mut p = Program::new(k);
    p.launch("fence", Dim3::x(1), Dim3::x(1), 0, &[]);
    expect_err(p.run(), ErrorKind::Unsupported);
}

#[test]
fn launches_share_buffers_in_order() {
    let k = kernel(
        "pipeline",
        r#"
extern "C" __global__ void scale2(const float* x, float* t) { t[threadIdx.x] = 2.0f * x[threadIdx.x]; }
extern "C" __global__ void add1(const float* t, float* y) { y[threadIdx.x] = t[threadIdx.x] + 1.0f; }
"#,
    );
    let mut p = Program::new(k);
    let x = p.input("x", &[1.5f32; 8]);
    let t = p.output::<f32>("t", 8);
    let y = p.output::<f32>("y", 8);
    p.launch(
        "scale2",
        Dim3::x(1),
        Dim3::x(8),
        0,
        &[Arg::Buf(x), Arg::Buf(t)],
    );
    p.launch(
        "add1",
        Dim3::x(1),
        Dim3::x(8),
        0,
        &[Arg::Buf(t), Arg::Buf(y)],
    );
    assert_eq!(ok(p.run()).get::<f32>(y), vec![4.0; 8]);
}

#[test]
fn every_result_says_it_is_not_a_gpu_run() {
    let k = kernel(
        "noop",
        r#"extern "C" __global__ void noop(int* y) { y[0] = 7; }"#,
    );
    let mut p = Program::new(k);
    let y = p.output::<i32>("y", 1);
    p.launch("noop", Dim3::x(1), Dim3::x(1), 0, &[Arg::Buf(y)]);
    let out = ok(p.run());
    assert!(out.stdout().starts_with(PROVENANCE));
    assert!(out.provenance().contains("NOT a GPU run"));
    assert_eq!(out.get::<i32>(y), vec![7]);
}

#[test]
fn a_missing_compiler_is_a_failure_not_a_skip() {
    let k = kernel(
        "noop",
        r#"extern "C" __global__ void noop(int* y) { y[0] = 7; }"#,
    );
    let mut p = Program::new(k);
    let y = p.output::<i32>("y", 1);
    p.launch("noop", Dim3::x(1), Dim3::x(1), 0, &[Arg::Buf(y)]);
    p.compiler("/nonexistent/cuda-emu-no-such-clang++");
    let e = expect_err(p.run(), ErrorKind::CompilerMissing);
    assert!(e.message.contains("not a skip"), "{e}");
}

#[test]
fn an_invalid_launch_is_refused_before_compiling() {
    let k = kernel(
        "noop",
        r#"extern "C" __global__ void noop(int* y) { y[0] = 7; }"#,
    );
    let mut p = Program::new(k);
    let y = p.output::<i32>("y", 1);
    p.launch("noop", Dim3::x(1), Dim3::x(2048), 0, &[Arg::Buf(y)]);
    expect_err(p.run(), ErrorKind::Launch);
}

#[test]
#[should_panic(expected = "bitwise comparison refused")]
fn bitwise_comparison_is_refused_for_a_kernel_calling_transcendentals() {
    let k = kernel(
        "silu",
        r#"extern "C" __global__ void silu(const float* x, float* y) { y[0] = x[0] / (1.0f + expf(-x[0])); }"#,
    );
    assert_bitwise_eq_f32(&k, "silu", &[0.5], &[0.5]);
}

#[test]
#[should_panic(expected = "--fmad=false")]
fn bitwise_comparison_is_refused_without_fmad_false() {
    let k = Kernel::new(
        "plain",
        r#"extern "C" __global__ void plain(float* y) { y[0] = 1.0f; }"#,
    );
    assert_bitwise_eq_f32(&k, "plain", &[1.0], &[1.0]);
}

#[test]
fn a_transcendental_kernel_is_compared_in_ulps() {
    let k = kernel(
        "silu",
        r#"extern "C" __global__ void silu(const float* x, float* y) {
  const float v = x[threadIdx.x];
  y[threadIdx.x] = v / (1.0f + expf(-v));
}"#,
    );
    assert_eq!(
        k.transcendentals().unwrap_or_else(|e| panic!("{e}")),
        vec!["expf"]
    );
    let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 4.0).collect();
    let mut p = Program::new(k);
    let xb = p.input("x", &x);
    let yb = p.output::<f32>("y", 64);
    p.launch(
        "silu",
        Dim3::x(1),
        Dim3::x(64),
        0,
        &[Arg::Buf(xb), Arg::Buf(yb)],
    );
    let want: Vec<f32> = x.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    assert_close_ulps_f32("silu", &ok(p.run()).get::<f32>(yb), &want, 4);
}

fn scratch_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("cuda_emu_selftest")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("clearing {}: {e}", dir.display()));
    }
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("creating {}: {e}", dir.display()));
    dir
}

#[test]
fn kernel_sources_are_read_out_of_rust_files() {
    let dir = scratch_dir("extract");
    let rs = dir.join("kernels.rs");
    let body = r####"// const RAW: &str = "a commented-out decoy";
/* block /* nested */ comment */
pub const RAW: &str = r#"extern "C" __global__ void raw_k(int* y) {
  y[threadIdx.x] = 3;
}
"#;
pub(crate) const ESCAPED: &'static str = "a\tb \"q\" \u{41}\x42\
                                         continued";
static INCLUDED: &str = include_str!("included.cu");
const JOINED: &str = concat!("one ", "two",);
const BROKEN: &str = r#"extern "C" __global__ void broken(int* y) {
  y[0] = this_does_not_exist;
}
"#;
const COMPUTED: &str = make_source();
const LIFETIME_NOISE: char = '\'';
"####;
    std::fs::write(&rs, body).unwrap_or_else(|e| panic!("{e}"));
    std::fs::write(dir.join("included.cu"), "from the .cu file\n")
        .unwrap_or_else(|e| panic!("{e}"));

    let raw = Kernel::from_rust_const(&rs, "RAW").unwrap_or_else(|e| panic!("{e}"));
    assert!(
        raw.source()
            .starts_with("extern \"C\" __global__ void raw_k")
    );
    assert_eq!(raw.label(), "kernels.rs::RAW");
    let esc = Kernel::from_rust_const(&rs, "ESCAPED").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(esc.source(), "a\tb \"q\" ABcontinued");
    let inc = Kernel::from_rust_const(&rs, "INCLUDED").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(inc.source(), "from the .cu file\n");
    let joined = Kernel::from_rust_const(&rs, "JOINED").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(joined.source(), "one two");
    let computed = Kernel::from_rust_const(&rs, "COMPUTED")
        .map(|_| ())
        .unwrap_err();
    assert_eq!(computed.kind, ErrorKind::Source);
    let missing = Kernel::from_rust_const(&rs, "ABSENT")
        .map(|_| ())
        .unwrap_err();
    assert!(missing.message.contains("no `const ABSENT`"), "{missing}");

    // The extracted kernel runs.
    let mut p = Program::new(raw.with_nvrtc_options(NVRTC_EXACT));
    let y = p.output::<i32>("y", 4);
    p.launch("raw_k", Dim3::x(1), Dim3::x(4), 0, &[Arg::Buf(y)]);
    assert_eq!(ok(p.run()).get::<i32>(y), vec![3; 4]);

    // A compile error points at the line of the .rs file it is on.
    let broken = Kernel::from_rust_const(&rs, "BROKEN").unwrap_or_else(|e| panic!("{e}"));
    let mut p = Program::new(broken);
    let y = p.output::<i32>("y", 1);
    p.launch("broken", Dim3::x(1), Dim3::x(1), 0, &[Arg::Buf(y)]);
    let e = expect_err(p.run(), ErrorKind::Compile);
    assert!(
        e.stderr.contains("kernels.rs:12"),
        "the error must cite kernels.rs line 12:\n{e}"
    );
}

// ------------------------------------------------------------------------------------------
// 16-bit floating point.

fn decode_f16(h: u16) -> f64 {
    let e = i32::from((h >> 10) & 0x1f);
    let m = f64::from(h & 0x3ff);
    if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

fn decode_bf16(h: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(h) << 16))
}

/// Round-to-nearest-even by search over the format's positive values, with infinity
/// standing at 2^(emax+1) - an oracle independent of the shim's bit manipulation.
fn nearest(x: f32, inf: u16, top: f64, decode: fn(u16) -> f64) -> u16 {
    if x.is_nan() {
        return 0x7fff;
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = f64::from(x).abs();
    let value = |h: u16| if h == inf { top } else { decode(h) };
    if a >= top {
        return sign | inf;
    }
    let (mut lo, mut hi) = (0u16, inf);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if value(mid) <= a { lo = mid } else { hi = mid }
    }
    let (dl, dh) = (a - value(lo), value(hi) - a);
    let pick = if dl < dh {
        lo
    } else if dh < dl {
        hi
    } else if lo % 2 == 0 {
        lo
    } else {
        hi
    };
    sign | pick
}

#[test]
fn bf16_and_half_conversions_round_to_nearest_even() {
    let mut xs: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        65504.0,
        65519.99,
        65520.0,
        1e9,
        -1e9,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        f32::MIN_POSITIVE,
        1e-45,
        2f32.powi(-24),
        2f32.powi(-25),
        1.5 * 2f32.powi(-25),
        2f32.powi(-26),
        1.0 + 2f32.powi(-8),
        1.0 + 3.0 * 2f32.powi(-8),
        1.0 + 2f32.powi(-11),
        1.0 + 3.0 * 2f32.powi(-11),
    ];
    let mut data = Data(0x1234_5678_9abc_def1);
    for _ in 0..12_000 {
        let bits = data.next_u64() as u32;
        xs.push(f32::from_bits(bits));
    }
    // Exact midpoints and their neighbours, where ties-to-even decides.
    for _ in 0..2_000 {
        let h = (data.next_u64() % 0x7bff) as u16;
        let mid = ((decode_f16(h) + decode_f16(h + 1)) / 2.0) as f32;
        let b = (data.next_u64() % 0x7f7f) as u16;
        let bmid = ((decode_bf16(b) + decode_bf16(b + 1)) / 2.0) as f32;
        for v in [mid, bmid] {
            xs.extend([
                v,
                f32::from_bits(v.to_bits() + 1),
                f32::from_bits(v.to_bits() - 1),
                -v,
            ]);
        }
    }
    let n = xs.len();
    let k = kernel(
        "convert",
        r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
extern "C" __global__ void convert(const float* x, __nv_bfloat16* bf, __half* hf, float* bf_back, float* hf_back,
                                   __nv_bfloat16* bf_implicit, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bf[i] = __float2bfloat16(x[i]);
  hf[i] = __float2half(x[i]);
  bf_back[i] = __bfloat162float(bf[i]);
  hf_back[i] = __half2float(hf[i]);
  __nv_bfloat16 t = x[i];
  bf_implicit[i] = t;
}
"#,
    );
    let mut p = Program::new(k);
    let xb = p.input("x", &xs);
    let bf = p.output::<Bf16>("bf", n);
    let hf = p.output::<F16>("hf", n);
    let bfb = p.output::<f32>("bf_back", n);
    let hfb = p.output::<f32>("hf_back", n);
    let bfi = p.output::<Bf16>("bf_implicit", n);
    let blocks = n.div_ceil(256) as u32;
    p.launch(
        "convert",
        Dim3::x(blocks),
        Dim3::x(256),
        0,
        &[
            Arg::Buf(xb),
            Arg::Buf(bf),
            Arg::Buf(hf),
            Arg::Buf(bfb),
            Arg::Buf(hfb),
            Arg::Buf(bfi),
            Arg::I32(n as i32),
        ],
    );
    let out = ok(p.run());
    let (bf, hf, bfb, hfb, bfi) = (
        out.get::<Bf16>(bf),
        out.get::<F16>(hf),
        out.get::<f32>(bfb),
        out.get::<f32>(hfb),
        out.get::<Bf16>(bfi),
    );
    for (i, &x) in xs.iter().enumerate() {
        let want_bf = nearest(x, 0x7f80, 2f64.powi(128), decode_bf16);
        let want_hf = nearest(x, 0x7c00, 65536.0, decode_f16);
        assert_eq!(bf[i].0, want_bf, "bf16 of {x:e} ({:#010x})", x.to_bits());
        assert_eq!(hf[i].0, want_hf, "half of {x:e} ({:#010x})", x.to_bits());
        assert_eq!(bfi[i].0, want_bf, "implicit bf16 of {x:e}");
        if want_bf != 0x7fff {
            assert_eq!(
                f64::from(bfb[i]),
                decode_bf16(want_bf).copysign(f64::from(bfb[i])),
                "bf16 -> f32 of {x:e}"
            );
        }
        if want_hf != 0x7fff && want_hf & 0x7fff != 0x7c00 {
            let mag = decode_f16(want_hf & 0x7fff);
            let signed = if want_hf & 0x8000 != 0 { -mag } else { mag };
            assert_eq!(f64::from(hfb[i]), signed, "half -> f32 of {x:e}");
        }
    }
}

#[test]
fn arithmetic_on_16_bit_types_is_refused_rather_than_done_in_float() {
    let k = kernel(
        "bf16_add",
        r#"
#include <cuda_bf16.h>
extern "C" __global__ void bf16_add(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c) {
  c[0] = a[0] + b[0];
}
"#,
    );
    let mut p = Program::new(k);
    let a = p.input("a", &[Bf16(0x3f80)]);
    let b = p.input("b", &[Bf16(0x3f80)]);
    let c = p.output::<Bf16>("c", 1);
    p.launch(
        "bf16_add",
        Dim3::x(1),
        Dim3::x(1),
        0,
        &[Arg::Buf(a), Arg::Buf(b), Arg::Buf(c)],
    );
    let e = expect_err(p.run(), ErrorKind::Compile);
    assert!(e.stderr.contains("deleted"), "{e}");
}

#[test]
fn block_order_is_selectable() {
    let k = kernel(
        "last_writer",
        r#"extern "C" __global__ void last_writer(unsigned* out) { if (threadIdx.x == 0) out[0] = blockIdx.x; }"#,
    );
    let mut p = Program::new(k);
    let out = p.zeroed::<u32>("out", 1);
    p.launch("last_writer", Dim3::x(5), Dim3::x(1), 0, &[Arg::Buf(out)]);
    assert_eq!(ok(p.run()).get::<u32>(out), vec![4]);
    p.block_order(BlockOrder::Reverse);
    assert_eq!(ok(p.run()).get::<u32>(out), vec![0]);
}
