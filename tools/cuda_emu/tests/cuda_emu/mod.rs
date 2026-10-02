//! `cuda_emu`: executes CUDA-C kernel sources on the CPU with the host C++ compiler, so index,
//! barrier and reduction-order bugs are caught before a GPU run.
//!
//! **This is test-time emulation, not a GPU run, and must never be reported as one.** A
//! kernel that passes here was executed by `clang++` on the host against a CPU model of the
//! CUDA execution model. Device tests stay NOT RUN until they run on a device. Every
//! [`Outputs`] carries [`PROVENANCE`]; a report that quotes a result quotes that line too.
//!
//! # How a run works
//!
//! [`Program::run`] rewrites the kernel source (only `extern __shared__` declarations and
//! `static` next to `__shared__`), writes it after the shim header `cuda_emu.h` together
//! with a generated `main` that allocates the typed buffers, calls each entry point with
//! typed arguments, and writes every buffer back. The program is compiled with
//! `-std=c++20 -ffp-contract=off` (NVRTC's `--fmad=false`) and run in a temporary
//! directory; its outputs are read back as typed vectors.
//!
//! Inside the driver, blocks run one at a time and every CUDA thread of the running block
//! is a `std::thread`. `__syncthreads` and the warp collectives rendezvous through one mutex
//! per block, following the sm_70+ rules (exited threads leave every rendezvous; a warp
//! collective waits for the non-exited lanes of its mask). See `cuda_emu.h` for what is
//! diagnosed and how each failure maps to an [`ErrorKind`].
//!
//! # What it cannot model
//!
//! Warp divergence and independent thread scheduling (lanes are free-running threads, not
//! a SIMT warp), the GPU memory model (synchronisation is C++ happens-before: the Thread
//! mode gives ThreadSanitizer exactly the edges CUDA promises - `__syncthreads`,
//! `__syncwarp`, nothing for shuffles, votes or atomics - but in the other modes the
//! rendezvous mutex really does order memory between every thread that passes through it,
//! so a missing barrier can still produce the right values; weaker-than-C++ GPU behaviour
//! such as a stale read without a fence never happens here), concurrency between blocks
//! (blocks are serialised: inter-block races show only as block-order dependence, see
//! [`Program::run_order_invariant`]), NVRTC's own front end and headers (clang++ parses the
//! source), inline PTX, libdevice's transcendental functions (libm's are used and differ
//! in the last bits), and anything about performance.
//!
//! # Requirements
//!
//! A C++20 compiler (`clang++` by default, or `$CUDA_EMU_CXX`; extra flags from
//! `$CUDA_EMU_CXXFLAGS`). A missing compiler is an error, never a skip. The sanitizer modes
//! need the compiler's sanitizer runtimes; a missing runtime is a link error, never a skip.

#![allow(dead_code)]

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Printed by every driver as its first line and carried by every [`Outputs`].
pub const PROVENANCE: &str =
    "cuda_emu: CPU emulation of CUDA-C on the host C++ compiler. NOT a GPU run.";

const SHIM_HEADER: &str = include_str!("cuda_emu.h");
/// Headers an NVRTC source may include; each forwards to the shim, which provides the
/// subset of them that is emulated (storage types and conversions, no 16-bit arithmetic).
const FORWARDING_HEADERS: [&str; 2] = ["cuda_bf16.h", "cuda_fp16.h"];

/// CUDA's per-block thread limit.
pub const MAX_THREADS_PER_BLOCK: u64 = 1024;
/// The largest dynamic shared memory an sm_90 block can opt into (227 KiB).
pub const MAX_DYN_SMEM_BYTES: usize = 227 * 1024;
/// Upper bound on threads per launch: every CUDA thread is an OS thread here.
pub const MAX_THREADS_PER_LAUNCH: u64 = 1 << 22;
/// Upper bound on the total size of a program's buffers.
pub const MAX_BUFFER_BYTES: usize = 1 << 30;
/// The 0xFF fill of unwritten outputs and shared memory, read as an `f32`.
pub const POISON_F32_BITS: u32 = 0xffff_ffff;

const COMPILE_TIMEOUT: Duration = Duration::from_secs(300);
const HOST_SLACK: Duration = Duration::from_secs(60);
const STDERR_HEAD: usize = 48 * 1024;
const STDERR_TAIL: usize = 16 * 1024;

// Exit codes; the first six must match `cuda_emu.h`, the last two are set through the
// sanitizer runtime options below.
const EXIT_DIAGNOSTIC: i32 = 70;
const EXIT_OUT_OF_BOUNDS: i32 = 71;
const EXIT_DEADLOCK: i32 = 72;
const EXIT_TIMEOUT: i32 = 73;
const EXIT_HARNESS: i32 = 74;
const EXIT_IO: i32 = 75;
const EXIT_TSAN: i32 = 66;
const EXIT_ASAN: i32 = 67;

/// Calls that libm and libdevice do not round identically, or that CUDA computes
/// approximately. A kernel calling any of them is compared with a ulp tolerance, never
/// bitwise.
const TRANSCENDENTALS: &[&str] = &[
    "expf",
    "exp2f",
    "exp10f",
    "expm1f",
    "logf",
    "log2f",
    "log10f",
    "log1pf",
    "powf",
    "sinf",
    "cosf",
    "tanf",
    "sincosf",
    "sinpif",
    "cospif",
    "sincospif",
    "asinf",
    "acosf",
    "atanf",
    "atan2f",
    "sinhf",
    "coshf",
    "tanhf",
    "asinhf",
    "acoshf",
    "atanhf",
    "erff",
    "erfcf",
    "erfinvf",
    "erfcinvf",
    "normcdff",
    "normcdfinvf",
    "lgammaf",
    "tgammaf",
    "cbrtf",
    "rcbrtf",
    "rsqrtf",
    "hypotf",
    "rhypotf",
    "norm3df",
    "rnorm3df",
    "exp",
    "exp2",
    "exp10",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "pow",
    "sin",
    "cos",
    "tan",
    "sincos",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "erfc",
    "erfinv",
    "lgamma",
    "tgamma",
    "cbrt",
    "rsqrt",
    "hypot",
    "__expf",
    "__exp10f",
    "__logf",
    "__log2f",
    "__log10f",
    "__powf",
    "__sinf",
    "__cosf",
    "__tanf",
    "__sincosf",
    "__fdividef",
    "__frsqrt_rn",
];

static RUN_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// `format!` appended to a `String` (infallible, unlike `write!`, so nothing is discarded).
macro_rules! push_fmt {
    ($s:expr, $($arg:tt)*) => {
        $s.push_str(&format!($($arg)*))
    };
}

macro_rules! push_line {
    ($s:expr, $($arg:tt)*) => {{
        $s.push_str(&format!($($arg)*));
        $s.push('\n');
    }};
}

// ------------------------------------------------------------------------------------------
// Errors.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The C++ compiler could not be run. A failure, never a skip.
    CompilerMissing,
    /// The source uses something that cannot run on a CPU (inline PTX).
    Unsupported,
    /// The kernel source could not be extracted or rewritten.
    Source,
    /// The launch configuration is invalid for CUDA or exceeds the emulator's bounds.
    Launch,
    /// The driver did not compile (includes a missing sanitizer runtime).
    Compile,
    /// The emulator diagnosed undefined behaviour: divergent barrier, bad warp collective.
    Diagnostic,
    /// A guard byte around a buffer or the dynamic shared memory was overwritten.
    OutOfBounds,
    /// Every live thread of a block was blocked with nothing able to complete.
    Deadlock,
    /// A launch exceeded its wall-clock budget (the driver's watchdog fired).
    Timeout,
    /// The driver outlived the host-side backstop and was killed.
    HostTimeout,
    /// ThreadSanitizer reported a data race.
    Race,
    /// AddressSanitizer or UndefinedBehaviorSanitizer reported an error.
    Memory,
    /// The driver crashed or exited with an unexpected status.
    Crashed,
    /// Two block orders produced different outputs.
    OrderDependent,
    /// The emulator's own plumbing failed.
    Harness,
    /// Reading or writing the run directory failed.
    Io,
}

#[derive(Debug)]
pub struct EmuError {
    pub kind: ErrorKind,
    pub message: String,
    /// The compiler's or the driver's stderr, bounded.
    pub stderr: String,
    /// The run directory, kept for inspection when the run failed after creating it.
    pub dir: Option<PathBuf>,
}

impl EmuError {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        EmuError {
            kind,
            message: message.into(),
            stderr: String::new(),
            dir: None,
        }
    }

    fn io(context: &str, err: io::Error) -> Self {
        EmuError::new(ErrorKind::Io, format!("{context}: {err}"))
    }
}

impl fmt::Display for EmuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cuda_emu {:?}: {}", self.kind, self.message)?;
        if let Some(dir) = &self.dir {
            write!(f, " (run directory kept: {})", dir.display())?;
        }
        if !self.stderr.is_empty() {
            write!(f, "\n--- stderr ---\n{}", self.stderr)?;
        }
        Ok(())
    }
}

impl std::error::Error for EmuError {}

// ------------------------------------------------------------------------------------------
// Element types.

/// A buffer element: its C++ spelling in the driver and its native-endian byte image.
pub trait Element: Copy {
    const C_TYPE: &'static str;
    const SIZE: usize;
    fn put(self, out: &mut Vec<u8>);
    fn take(bytes: &[u8]) -> Self;
}

macro_rules! primitive_element {
    ($t:ty, $c:expr) => {
        impl Element for $t {
            const C_TYPE: &'static str = $c;
            const SIZE: usize = std::mem::size_of::<$t>();
            fn put(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_ne_bytes());
            }
            fn take(bytes: &[u8]) -> Self {
                let mut raw = [0u8; std::mem::size_of::<$t>()];
                raw.copy_from_slice(bytes);
                <$t>::from_ne_bytes(raw)
            }
        }
    };
}

primitive_element!(f32, "float");
primitive_element!(f64, "double");
primitive_element!(i8, "signed char");
primitive_element!(u8, "unsigned char");
primitive_element!(i16, "short");
primitive_element!(u16, "unsigned short");
primitive_element!(i32, "int");
primitive_element!(u32, "unsigned int");
primitive_element!(i64, "long long");
primitive_element!(u64, "unsigned long long");

/// A `__nv_bfloat16` element, as its bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bf16(pub u16);

/// A `__half` element, as its bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct F16(pub u16);

impl Element for Bf16 {
    const C_TYPE: &'static str = "__nv_bfloat16";
    const SIZE: usize = 2;
    fn put(self, out: &mut Vec<u8>) {
        self.0.put(out);
    }
    fn take(bytes: &[u8]) -> Self {
        Bf16(u16::take(bytes))
    }
}

impl Element for F16 {
    const C_TYPE: &'static str = "__half";
    const SIZE: usize = 2;
    fn put(self, out: &mut Vec<u8>) {
        self.0.put(out);
    }
    fn take(bytes: &[u8]) -> Self {
        F16(u16::take(bytes))
    }
}

// ------------------------------------------------------------------------------------------
// Kernels.

/// One NVRTC module's source: what the emulator compiles as one translation unit.
#[derive(Clone, Debug)]
pub struct Kernel {
    label: String,
    source: String,
    /// The file and line the source's first line comes from, when the mapping is exact;
    /// compiler errors and `__LINE__` in diagnostics then point into that file.
    origin: Option<(PathBuf, usize)>,
    nvrtc_options: Vec<String>,
    arch: u32,
}

impl Kernel {
    /// A source given directly, e.g. a crate's public `const` holding the NVRTC string.
    pub fn new(label: impl Into<String>, source: impl Into<String>) -> Self {
        Kernel {
            label: label.into(),
            source: source.into(),
            origin: None,
            nvrtc_options: Vec::new(),
            arch: 900,
        }
    }

    /// Reads the `const`/`static` string item `name` out of a Rust source file. The
    /// initializer must be a string literal (raw or not), `include_str!("...")`, or
    /// `concat!` of string literals; anything else is an error naming what was found.
    pub fn from_rust_const(path: impl AsRef<Path>, name: &str) -> Result<Self, EmuError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| EmuError::io(&format!("reading {}", path.display()), e))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let found = extract_rust_str_const(&text, name, base)
            .map_err(|m| EmuError::new(ErrorKind::Source, format!("{}: {m}", path.display())))?;
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        Ok(Kernel {
            label: format!("{file_name}::{name}"),
            source: found.value,
            origin: found
                .origin
                .map(|(file, line)| (file.unwrap_or_else(|| path.to_path_buf()), line)),
            nvrtc_options: Vec::new(),
            arch: 900,
        })
    }

    /// The options the crate passes to NVRTC for this module. Bitwise comparison requires
    /// `--fmad=false` among them and none of the fast-math options.
    pub fn with_nvrtc_options(mut self, options: &[&str]) -> Self {
        self.nvrtc_options = options.iter().map(|o| (*o).to_string()).collect();
        self
    }

    /// The `__CUDA_ARCH__` the source sees (default 900, the GH200's sm_90).
    pub fn with_arch(mut self, arch: u32) -> Self {
        self.arch = arch;
        self
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// The approximate or libm-dependent functions the source calls.
    pub fn transcendentals(&self) -> Result<Vec<&'static str>, EmuError> {
        let toks = lex_c(&self.source).map_err(|m| EmuError::new(ErrorKind::Source, m))?;
        let mut found: Vec<&'static str> = Vec::new();
        for (k, t) in toks.iter().enumerate() {
            if t.kind != CKind::Ident {
                continue;
            }
            let word = &self.source[t.start..t.end];
            let is_call = toks
                .get(k + 1)
                .is_some_and(|n| n.kind == CKind::Punct && self.source.as_bytes()[n.start] == b'(');
            if !is_call {
                continue;
            }
            match TRANSCENDENTALS.iter().copied().find(|n| *n == word) {
                Some(name) if !found.contains(&name) => found.push(name),
                _ => {}
            }
        }
        found.sort_unstable();
        Ok(found)
    }

    /// `Ok` when an emulator result can be compared bitwise against a host mirror that
    /// computes the same operations in the same order; otherwise every reason it cannot.
    pub fn bitwise_comparable(&self) -> Result<(), Vec<String>> {
        let mut reasons = Vec::new();
        match self.transcendentals() {
            Ok(t) if !t.is_empty() => reasons.push(format!(
                "it calls {}: libm is not libdevice and the last bits differ",
                t.join(", ")
            )),
            Ok(_) => {}
            Err(e) => reasons.push(format!("its source could not be scanned: {}", e.message)),
        }
        let mut fmad: Option<bool> = None;
        for opt in &self.nvrtc_options {
            let o = opt.trim_start_matches('-');
            match o {
                "fmad=false" => fmad = Some(false),
                "fmad=true" => fmad = Some(true),
                "use_fast_math" | "ftz=true" | "prec-div=false" | "prec-sqrt=false" => reasons
                    .push(format!(
                        "NVRTC option {opt} changes rounding away from IEEE, which the CPU computes"
                    )),
                _ => {}
            }
        }
        if fmad != Some(false) {
            reasons.push(
                "the NVRTC options do not end with --fmad=false: NVRTC contracts a*b+c into an FMA by \
                 default, the emulator compiles with -ffp-contract=off"
                    .to_string(),
            );
        }
        if reasons.is_empty() {
            Ok(())
        } else {
            Err(reasons)
        }
    }
}

// ------------------------------------------------------------------------------------------
// Programs.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn x(x: u32) -> Self {
        Dim3 { x, y: 1, z: 1 }
    }
    pub const fn xy(x: u32, y: u32) -> Self {
        Dim3 { x, y, z: 1 }
    }
    pub const fn xyz(x: u32, y: u32, z: u32) -> Self {
        Dim3 { x, y, z }
    }
    pub fn count(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sanitizer {
    /// No sanitizer: the mode for bitwise comparisons and the fastest.
    None,
    /// AddressSanitizer + UndefinedBehaviorSanitizer: out-of-bounds reads and writes of
    /// buffers and shared memory, signed overflow in index arithmetic. Static shared memory
    /// is not poisoned in this mode (see `cuda_emu.h`).
    Address,
    /// ThreadSanitizer: a missing `__syncthreads`/`__syncwarp` between a write and a read
    /// by another thread of the block is reported as a data race.
    Thread,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockOrder {
    Forward,
    Reverse,
    Shuffled(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufId(usize);

/// A kernel argument. Scalars are passed by their bit patterns, so a float argument is
/// exactly the value given.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Buf(BufId),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    /// A C++ expression passed verbatim, e.g. a parameter struct literal.
    Expr(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Init {
    File,
    Poison,
    Zero,
}

#[derive(Clone, Debug)]
struct BufSpec {
    name: String,
    c_type: &'static str,
    elem_size: usize,
    len: usize,
    init: Init,
    data: Vec<u8>,
}

#[derive(Clone, Debug)]
struct LaunchSpec {
    entry: String,
    grid: Dim3,
    block: Dim3,
    dyn_smem: usize,
    args: Vec<Arg>,
}

/// One translation unit, its device buffers, and a sequence of launches over them.
#[derive(Clone, Debug)]
pub struct Program {
    kernel: Kernel,
    buffers: Vec<BufSpec>,
    launches: Vec<LaunchSpec>,
    sanitizer: Sanitizer,
    order: BlockOrder,
    compiler: Option<PathBuf>,
    launch_timeout: Option<Duration>,
}

impl Program {
    pub fn new(kernel: Kernel) -> Self {
        Program {
            kernel,
            buffers: Vec::new(),
            launches: Vec::new(),
            sanitizer: Sanitizer::None,
            order: BlockOrder::Forward,
            compiler: None,
            launch_timeout: None,
        }
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// A buffer initialised from `data`.
    pub fn input<T: Element>(&mut self, name: &str, data: &[T]) -> BufId {
        let mut bytes = Vec::with_capacity(data.len() * T::SIZE);
        for v in data {
            v.put(&mut bytes);
        }
        self.push_buffer::<T>(name, data.len(), Init::File, bytes)
    }

    /// A buffer filled with 0xFF bytes (NaN for floats): an element the kernel never writes
    /// reads back as the poison pattern.
    pub fn output<T: Element>(&mut self, name: &str, len: usize) -> BufId {
        self.push_buffer::<T>(name, len, Init::Poison, Vec::new())
    }

    /// A zero-filled buffer, for accumulators.
    pub fn zeroed<T: Element>(&mut self, name: &str, len: usize) -> BufId {
        self.push_buffer::<T>(name, len, Init::Zero, Vec::new())
    }

    fn push_buffer<T: Element>(
        &mut self,
        name: &str,
        len: usize,
        init: Init,
        data: Vec<u8>,
    ) -> BufId {
        self.buffers.push(BufSpec {
            name: name.to_string(),
            c_type: T::C_TYPE,
            elem_size: T::SIZE,
            len,
            init,
            data,
        });
        BufId(self.buffers.len() - 1)
    }

    /// Appends a launch of `entry` (an `extern "C"` name, or a C++ template-id such as
    /// `"reduce<128>"`). Launches run in the order added and share the buffers.
    pub fn launch(
        &mut self,
        entry: &str,
        grid: Dim3,
        block: Dim3,
        dyn_smem_bytes: usize,
        args: &[Arg],
    ) -> &mut Self {
        self.launches.push(LaunchSpec {
            entry: entry.to_string(),
            grid,
            block,
            dyn_smem: dyn_smem_bytes,
            args: args.to_vec(),
        });
        self
    }

    pub fn sanitizer(&mut self, sanitizer: Sanitizer) -> &mut Self {
        self.sanitizer = sanitizer;
        self
    }

    pub fn block_order(&mut self, order: BlockOrder) -> &mut Self {
        self.order = order;
        self
    }

    /// The C++ compiler to use instead of `$CUDA_EMU_CXX` / `clang++`.
    pub fn compiler(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.compiler = Some(path.into());
        self
    }

    /// The wall-clock budget of each launch (default 60 s, 300 s under a sanitizer).
    pub fn launch_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.launch_timeout = Some(timeout);
        self
    }

    /// Compiles and runs the program once, in the configured block order.
    pub fn run(&self) -> Result<Outputs, EmuError> {
        self.run_in_order(self.order)
    }

    /// Runs the program in forward, reverse and a shuffled block order and requires every
    /// buffer to come back bit-identical. Blocks that write overlapping elements (an index
    /// bug) or read what another block wrote (an inter-block race) fail here.
    pub fn run_order_invariant(&self) -> Result<Outputs, EmuError> {
        let orders = [
            BlockOrder::Forward,
            BlockOrder::Reverse,
            BlockOrder::Shuffled(0x5eed_cafe_f00d),
        ];
        let first = self.run_in_order(orders[0])?;
        for &order in &orders[1..] {
            let other = self.run_in_order(order)?;
            for (i, (a, b)) in first.bytes.iter().zip(&other.bytes).enumerate() {
                if let Some(pos) = a.iter().zip(b).position(|(x, y)| x != y) {
                    let spec = &self.buffers[i];
                    return Err(EmuError::new(
                        ErrorKind::OrderDependent,
                        format!(
                            "buffer '{}' element {} differs between block orders {:?} and {:?}: blocks \
                             write overlapping elements or read each other's writes",
                            spec.name,
                            pos / spec.elem_size.max(1),
                            orders[0],
                            order
                        ),
                    ));
                }
            }
        }
        Ok(first)
    }

    /// The generated driver source, for inspection.
    pub fn driver_source(&self) -> Result<String, EmuError> {
        self.validate()?;
        let rewritten = rewrite_cuda_source(&self.kernel.source)?;
        Ok(self.render_driver(&rewritten, self.order))
    }

    fn launch_timeout_for_mode(&self) -> Duration {
        self.launch_timeout.unwrap_or(match self.sanitizer {
            Sanitizer::None => Duration::from_secs(60),
            Sanitizer::Address | Sanitizer::Thread => Duration::from_secs(300),
        })
    }

    fn validate(&self) -> Result<(), EmuError> {
        let launch_err = |m: String| Err(EmuError::new(ErrorKind::Launch, m));
        if self.launches.is_empty() {
            return launch_err("the program has no launch".to_string());
        }
        let total: usize = self
            .buffers
            .iter()
            .map(|b| b.len.saturating_mul(b.elem_size))
            .sum();
        if total > MAX_BUFFER_BYTES {
            return launch_err(format!(
                "buffers total {total} bytes, over the {MAX_BUFFER_BYTES}-byte bound"
            ));
        }
        for b in &self.buffers {
            if !b.name.chars().all(|c| c.is_ascii_graphic() || c == ' ')
                || b.name.contains(['"', '\\'])
            {
                return launch_err(format!(
                    "buffer name {:?} must be printable ASCII without quotes or backslashes",
                    b.name
                ));
            }
        }
        let timeout = self.launch_timeout_for_mode();
        if timeout.is_zero() || timeout.as_millis() > u128::from(u32::MAX) {
            return launch_err(format!("launch timeout {timeout:?} out of range"));
        }
        for l in &self.launches {
            let b = l.block;
            let g = l.grid;
            if !valid_entry(&l.entry) {
                return launch_err(format!(
                    "entry {:?} is not a C++ identifier or template-id",
                    l.entry
                ));
            }
            if b.x == 0
                || b.y == 0
                || b.z == 0
                || b.x > 1024
                || b.y > 1024
                || b.z > 64
                || b.count() > MAX_THREADS_PER_BLOCK
            {
                return launch_err(format!(
                    "{}: block {b:?} is not a valid CUDA block (x,y <= 1024, z <= 64, x*y*z <= 1024)",
                    l.entry
                ));
            }
            if g.x == 0
                || g.y == 0
                || g.z == 0
                || g.x > i32::MAX as u32
                || g.y > 65535
                || g.z > 65535
            {
                return launch_err(format!("{}: grid {g:?} is not a valid CUDA grid", l.entry));
            }
            if g.count().saturating_mul(b.count()) > MAX_THREADS_PER_LAUNCH {
                return launch_err(format!(
                    "{}: {} threads in one launch, over the emulator's bound of {MAX_THREADS_PER_LAUNCH}",
                    l.entry,
                    g.count().saturating_mul(b.count())
                ));
            }
            if l.dyn_smem > MAX_DYN_SMEM_BYTES {
                return launch_err(format!(
                    "{}: {} bytes of dynamic shared memory, over sm_90's {MAX_DYN_SMEM_BYTES}",
                    l.entry, l.dyn_smem
                ));
            }
            for a in &l.args {
                match a {
                    Arg::Buf(BufId(i)) if *i >= self.buffers.len() => {
                        return launch_err(format!(
                            "{}: argument names buffer {i}, which does not exist",
                            l.entry
                        ));
                    }
                    Arg::Expr(e) if e.contains('\n') || e.trim().is_empty() => {
                        return launch_err(format!(
                            "{}: expression argument {e:?} must be one non-empty line",
                            l.entry
                        ));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn render_driver(&self, rewritten: &str, order: BlockOrder) -> String {
        let mut s = String::new();
        push_line!(
            s,
            "// Generated by cuda_emu for {}. {PROVENANCE}",
            c_comment_safe(&self.kernel.label)
        );
        push_line!(
            s,
            "#define CUDA_EMU_MODE_ASAN {}",
            u8::from(self.sanitizer == Sanitizer::Address)
        );
        push_line!(s, "#define CUDA_EMU_ARCH {}", self.kernel.arch);
        s.push_str("#include \"cuda_emu.h\"\n");
        match &self.kernel.origin {
            Some((path, line)) => {
                push_line!(
                    s,
                    "#line {line} \"{}\"",
                    c_string_body(&path.display().to_string())
                );
            }
            None => {
                push_line!(s, "#line 1 \"{}\"", c_string_body(&self.kernel.label));
            }
        }
        s.push_str(rewritten);
        if !rewritten.ends_with('\n') {
            s.push('\n');
        }
        let next_line = s.lines().count() + 2;
        push_line!(s, "#line {next_line} \"cuda_emu_driver.cpp\"");
        s.push_str("int main(int argc, char** argv) {\n  ::cuda_emu::Driver drv(argc, argv);\n");
        for (i, b) in self.buffers.iter().enumerate() {
            let init = match b.init {
                Init::File => "File",
                Init::Poison => "Poison",
                Init::Zero => "Zero",
            };
            push_line!(
                s,
                "  {ty}* b{i} = drv.buffer<{ty}>({i}u, \"{name}\", {len}ull, ::cuda_emu::Init::{init});",
                ty = b.c_type,
                name = b.name,
                len = b.len
            );
        }
        let timeout_ms = self.launch_timeout_for_mode().as_millis();
        let (order_name, seed) = match order {
            BlockOrder::Forward => ("Forward", 0),
            BlockOrder::Reverse => ("Reverse", 0),
            BlockOrder::Shuffled(seed) => ("Shuffled", seed),
        };
        for l in &self.launches {
            let args: Vec<String> = l.args.iter().map(arg_expr).collect();
            push_line!(
                s,
                "  drv.launch(::cuda_emu::LaunchSpec{{\"{entry_lit}\", dim3({gx}u, {gy}u, {gz}u), dim3({bx}u, {by}u, \
                 {bz}u), {dyn}ull, {timeout_ms}u, ::cuda_emu::Order::{order_name}, {seed}ull}}, [&] {{ {entry}({args}); }});",
                entry_lit = c_string_body(&l.entry),
                gx = l.grid.x,
                gy = l.grid.y,
                gz = l.grid.z,
                bx = l.block.x,
                by = l.block.y,
                bz = l.block.z,
                dyn = l.dyn_smem,
                entry = l.entry,
                args = args.join(", ")
            );
        }
        s.push_str("  drv.store_all();\n  return 0;\n}\n");
        s
    }

    fn run_in_order(&self, order: BlockOrder) -> Result<Outputs, EmuError> {
        self.validate()?;
        let rewritten = rewrite_cuda_source(&self.kernel.source)?;
        let driver = self.render_driver(&rewritten, order);
        let compiler = self.resolve_compiler();
        let compiler_version = compiler_version(&compiler)?;
        let dir = make_run_dir(&self.kernel.label)?;
        let result = self.compile_and_execute(&dir, &driver, &compiler, &compiler_version, order);
        let keep = std::env::var_os("CUDA_EMU_KEEP").is_some();
        match result {
            Ok(outputs) => {
                if keep {
                    eprintln!("cuda_emu: run directory kept: {}", dir.display());
                } else {
                    fs::remove_dir_all(&dir)
                        .map_err(|e| EmuError::io(&format!("removing {}", dir.display()), e))?;
                }
                Ok(outputs)
            }
            Err(mut e) => {
                e.dir = Some(dir);
                Err(e)
            }
        }
    }

    fn resolve_compiler(&self) -> PathBuf {
        self.compiler
            .clone()
            .or_else(|| std::env::var_os("CUDA_EMU_CXX").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("clang++"))
    }

    fn compile_and_execute(
        &self,
        dir: &Path,
        driver: &str,
        compiler: &Path,
        compiler_version: &str,
        order: BlockOrder,
    ) -> Result<Outputs, EmuError> {
        let write = |name: &str, bytes: &[u8]| {
            let p = dir.join(name);
            fs::write(&p, bytes).map_err(|e| EmuError::io(&format!("writing {}", p.display()), e))
        };
        write("cuda_emu.h", SHIM_HEADER.as_bytes())?;
        for h in FORWARDING_HEADERS {
            write(h, b"#pragma once\n#include \"cuda_emu.h\"\n")?;
        }
        write("cuda_emu_driver.cpp", driver.as_bytes())?;
        for (i, b) in self.buffers.iter().enumerate() {
            if b.init == Init::File {
                write(&format!("buf_{i}.in"), &b.data)?;
            }
        }

        let exe = dir.join("cuda_emu_driver");
        let mut cc = Command::new(compiler);
        cc.current_dir(dir)
            .args([
                "-std=c++20",
                "-O1",
                "-g",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-fno-strict-aliasing",
            ])
            .args(["-pthread", "-Wall", "-I"])
            .arg(dir);
        match self.sanitizer {
            Sanitizer::None => {}
            Sanitizer::Address => {
                cc.args([
                    "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=undefined",
                    "-fno-omit-frame-pointer",
                ]);
            }
            Sanitizer::Thread => {
                cc.arg("-fsanitize=thread");
            }
        }
        if let Some(extra) = std::env::var_os("CUDA_EMU_CXXFLAGS") {
            cc.args(extra.to_string_lossy().split_whitespace());
        }
        cc.arg("cuda_emu_driver.cpp").arg("-o").arg(&exe);
        let status = run_bounded(
            &mut cc,
            COMPILE_TIMEOUT,
            &dir.join("compile.stdout"),
            &dir.join("compile.stderr"),
        )
        .map_err(|e| EmuError::io(&format!("running {}", compiler.display()), e))?;
        match status {
            None => {
                return Err(EmuError::new(
                    ErrorKind::HostTimeout,
                    format!(
                        "compiling {} exceeded {COMPILE_TIMEOUT:?}",
                        self.kernel.label
                    ),
                ));
            }
            Some(st) if !st.success() => {
                let mut e = EmuError::new(
                    ErrorKind::Compile,
                    format!(
                        "{} ({compiler_version}) rejected the driver for {}: {st}",
                        compiler.display(),
                        self.kernel.label
                    ),
                );
                e.stderr = read_bounded(&dir.join("compile.stderr"))?;
                return Err(e);
            }
            Some(_) => {}
        }

        let budget = self
            .launch_timeout_for_mode()
            .saturating_mul(u32::try_from(self.launches.len()).unwrap_or(u32::MAX))
            .saturating_add(HOST_SLACK);
        let mut run = Command::new(&exe);
        run.current_dir(dir).arg(dir);
        // Only the active mode's options, and none inherited: the TSan runtime also parses
        // UBSAN_OPTIONS, whose `exitcode` then overrides TSAN_OPTIONS' (measured).
        for var in ["TSAN_OPTIONS", "ASAN_OPTIONS", "UBSAN_OPTIONS"] {
            run.env_remove(var);
        }
        match self.sanitizer {
            Sanitizer::None => {}
            Sanitizer::Thread => {
                run.env(
                    "TSAN_OPTIONS",
                    format!("halt_on_error=1:exitcode={EXIT_TSAN}:report_signal_unsafe=0"),
                );
            }
            Sanitizer::Address => {
                run.env(
                    "ASAN_OPTIONS",
                    format!("halt_on_error=1:exitcode={EXIT_ASAN}:detect_leaks=0:abort_on_error=0"),
                )
                .env(
                    "UBSAN_OPTIONS",
                    format!("halt_on_error=1:exitcode={EXIT_ASAN}:print_stacktrace=1"),
                );
            }
        }
        let status = run_bounded(
            &mut run,
            budget,
            &dir.join("run.stdout"),
            &dir.join("run.stderr"),
        )
        .map_err(|e| EmuError::io(&format!("running {}", exe.display()), e))?;
        let stderr = read_bounded(&dir.join("run.stderr"))?;
        let stdout = read_bounded(&dir.join("run.stdout"))?;
        let st = match status {
            None => {
                let mut e = EmuError::new(
                    ErrorKind::HostTimeout,
                    format!(
                        "the driver for {} outlived the host backstop of {budget:?} and was killed",
                        self.kernel.label
                    ),
                );
                e.stderr = stderr;
                return Err(e);
            }
            Some(st) => st,
        };
        if !st.success() {
            let kind = classify_exit(st, self.sanitizer);
            let first_line = stderr
                .lines()
                .find(|l| {
                    l.starts_with("cuda_emu: ")
                        || l.contains("Sanitizer")
                        || l.contains("runtime error")
                })
                .unwrap_or("(no diagnostic line)")
                .to_string();
            let mut e = EmuError::new(
                kind,
                format!("{} exited with {st}: {first_line}", self.kernel.label),
            );
            e.stderr = stderr;
            return Err(e);
        }
        if !stdout.starts_with(PROVENANCE) {
            return Err(EmuError::new(
                ErrorKind::Harness,
                "the driver did not print the provenance line first; the shim and the harness disagree",
            ));
        }

        let mut bytes = Vec::with_capacity(self.buffers.len());
        for (i, b) in self.buffers.iter().enumerate() {
            let p = dir.join(format!("buf_{i}.out"));
            let data =
                fs::read(&p).map_err(|e| EmuError::io(&format!("reading {}", p.display()), e))?;
            if data.len() != b.len * b.elem_size {
                return Err(EmuError::new(
                    ErrorKind::Io,
                    format!(
                        "{} holds {} bytes, expected {}",
                        p.display(),
                        data.len(),
                        b.len * b.elem_size
                    ),
                ));
            }
            bytes.push(data);
        }
        Ok(Outputs {
            names: self.buffers.iter().map(|b| b.name.clone()).collect(),
            c_types: self.buffers.iter().map(|b| b.c_type).collect(),
            bytes,
            stdout,
            compiler_version: compiler_version.to_string(),
            sanitizer: self.sanitizer,
            order,
        })
    }
}

/// The buffers after the last launch, and what produced them.
#[derive(Clone, Debug)]
pub struct Outputs {
    names: Vec<String>,
    c_types: Vec<&'static str>,
    bytes: Vec<Vec<u8>>,
    stdout: String,
    compiler_version: String,
    sanitizer: Sanitizer,
    order: BlockOrder,
}

impl Outputs {
    /// The buffer's elements. Panics if `T` is not the type the buffer was declared with.
    pub fn get<T: Element>(&self, id: BufId) -> Vec<T> {
        let BufId(i) = id;
        assert_eq!(
            self.c_types[i],
            T::C_TYPE,
            "buffer '{}' holds {} elements, read as {}",
            self.names[i],
            self.c_types[i],
            T::C_TYPE
        );
        self.bytes[i].chunks_exact(T::SIZE).map(T::take).collect()
    }

    /// The driver's stdout: the provenance line, then anything the kernels printed.
    pub fn stdout(&self) -> &str {
        &self.stdout
    }

    /// What produced these numbers - quote it wherever they are reported.
    pub fn provenance(&self) -> String {
        format!(
            "{PROVENANCE} Compiler: {}; sanitizer: {:?}; block order: {:?}.",
            self.compiler_version, self.sanitizer, self.order
        )
    }
}

// ------------------------------------------------------------------------------------------
// Comparisons.

/// Bitwise equality against a host mirror. Refuses (panics) for a kernel whose result
/// cannot be bitwise-equal to the device's: see [`Kernel::bitwise_comparable`]. Two NaNs
/// compare equal (payloads are a platform property) unless the emulator's is the poison
/// pattern of an element nothing wrote and the expected one is not.
pub fn assert_bitwise_eq_f32(kernel: &Kernel, what: &str, got: &[f32], want: &[f32]) {
    if let Err(reasons) = kernel.bitwise_comparable() {
        panic!(
            "{what}: bitwise comparison refused for '{}': {}. Use assert_close_ulps_f32.",
            kernel.label,
            reasons.join("; ")
        );
    }
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs expected {}",
        got.len(),
        want.len()
    );
    let bad: Vec<usize> = (0..got.len())
        .filter(|&i| !same_f32(got[i], want[i]))
        .collect();
    if !bad.is_empty() {
        let mut msg = format!(
            "{what}: {} of {} elements differ bitwise from the host mirror",
            bad.len(),
            got.len()
        );
        for &i in bad.iter().take(8) {
            let ulps =
                ulp_distance_f32(got[i], want[i]).map_or("n/a".to_string(), |u| u.to_string());
            push_fmt!(
                msg,
                "\n  [{i}] got {:e} ({:#010x}) want {:e} ({:#010x}), {ulps} ulp",
                got[i],
                got[i].to_bits(),
                want[i],
                want[i].to_bits()
            );
        }
        panic!("{msg}");
    }
}

fn same_f32(got: f32, want: f32) -> bool {
    if got.to_bits() == want.to_bits() {
        return true;
    }
    got.is_nan()
        && want.is_nan()
        && (got.to_bits() != POISON_F32_BITS || want.to_bits() == POISON_F32_BITS)
}

/// Ulp-tolerance comparison, for kernels that call transcendentals. NaN matches only NaN.
pub fn assert_close_ulps_f32(what: &str, got: &[f32], want: &[f32], max_ulps: u64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs expected {}",
        got.len(),
        want.len()
    );
    let mut worst: Option<(usize, u64)> = None;
    for i in 0..got.len() {
        if got[i].is_nan() || want[i].is_nan() {
            assert!(
                got[i].is_nan() && want[i].is_nan(),
                "{what}: [{i}] got {} want {}",
                got[i],
                want[i]
            );
            continue;
        }
        let d = ulp_distance_f32(got[i], want[i]).unwrap_or(u64::MAX);
        if worst.is_none_or(|(_, w)| d > w) {
            worst = Some((i, d));
        }
    }
    if let Some((i, d)) = worst {
        assert!(
            d <= max_ulps,
            "{what}: [{i}] got {:e} want {:e}: {d} ulp > {max_ulps}",
            got[i],
            want[i]
        );
    }
}

/// Distance in units in the last place; `None` if either is NaN. +0 and -0 are 0 apart.
pub fn ulp_distance_f32(a: f32, b: f32) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    let ordered = |f: f32| {
        let u = f.to_bits();
        if u & 0x8000_0000 != 0 {
            -i64::from(u & 0x7fff_ffff)
        } else {
            i64::from(u)
        }
    };
    Some(ordered(a).abs_diff(ordered(b)))
}

// ------------------------------------------------------------------------------------------
// Process plumbing.

fn compiler_version(compiler: &Path) -> Result<String, EmuError> {
    let missing = |detail: String| {
        EmuError::new(
            ErrorKind::CompilerMissing,
            format!(
                "cannot run the C++ compiler {}: {detail}. The CUDA emulator cannot run without it; this is a \
                 failure, not a skip. Install the compiler or set CUDA_EMU_CXX.",
                compiler.display()
            ),
        )
    };
    let out = Command::new(compiler)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| missing(e.to_string()))?;
    if !out.status.success() {
        return Err(missing(format!("`--version` exited with {}", out.status)));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text
        .lines()
        .next()
        .unwrap_or("unknown compiler")
        .trim()
        .to_string())
}

fn make_run_dir(label: &str) -> Result<PathBuf, EmuError> {
    let base = option_env!("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("cuda_emu");
    fs::create_dir_all(&base)
        .map_err(|e| EmuError::io(&format!("creating {}", base.display()), e))?;
    let safe: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(48)
        .collect();
    let dir = base.join(format!(
        "{safe}-{}-{}",
        std::process::id(),
        RUN_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    if dir.exists() {
        fs::remove_dir_all(&dir)
            .map_err(|e| EmuError::io(&format!("removing stale {}", dir.display()), e))?;
    }
    fs::create_dir(&dir).map_err(|e| EmuError::io(&format!("creating {}", dir.display()), e))?;
    Ok(dir)
}

/// Runs a command with stdout/stderr redirected to files (so a chatty child cannot block on
/// a full pipe) and kills it at `limit`. `None` means it was killed.
fn run_bounded(
    cmd: &mut Command,
    limit: Duration,
    stdout: &Path,
    stderr: &Path,
) -> io::Result<Option<ExitStatus>> {
    let out = fs::File::create(stdout)?;
    let err = fs::File::create(stderr)?;
    let mut child = cmd.stdin(Stdio::null()).stdout(out).stderr(err).spawn()?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if start.elapsed() >= limit {
            kill_and_reap(&mut child)?;
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn kill_and_reap(child: &mut Child) -> io::Result<()> {
    match child.kill() {
        Ok(()) => {}
        // Already exited between try_wait and kill: reaping below collects it.
        Err(e) if e.kind() == io::ErrorKind::InvalidInput => {}
        Err(e) => return Err(e),
    }
    child.wait()?;
    Ok(())
}

fn read_bounded(path: &Path) -> Result<String, EmuError> {
    let bytes =
        fs::read(path).map_err(|e| EmuError::io(&format!("reading {}", path.display()), e))?;
    if bytes.len() <= STDERR_HEAD + STDERR_TAIL {
        return Ok(String::from_utf8_lossy(&bytes).into_owned());
    }
    let head = String::from_utf8_lossy(&bytes[..STDERR_HEAD]);
    let tail = String::from_utf8_lossy(&bytes[bytes.len() - STDERR_TAIL..]);
    Ok(format!(
        "{head}\n[... {} bytes elided by cuda_emu ...]\n{tail}",
        bytes.len() - STDERR_HEAD - STDERR_TAIL
    ))
}

fn classify_exit(status: ExitStatus, sanitizer: Sanitizer) -> ErrorKind {
    match status.code() {
        Some(EXIT_DIAGNOSTIC) => ErrorKind::Diagnostic,
        Some(EXIT_OUT_OF_BOUNDS) => ErrorKind::OutOfBounds,
        Some(EXIT_DEADLOCK) => ErrorKind::Deadlock,
        Some(EXIT_TIMEOUT) => ErrorKind::Timeout,
        Some(EXIT_HARNESS) => ErrorKind::Harness,
        Some(EXIT_IO) => ErrorKind::Io,
        Some(EXIT_TSAN) if sanitizer == Sanitizer::Thread => ErrorKind::Race,
        Some(EXIT_ASAN) if sanitizer == Sanitizer::Address => ErrorKind::Memory,
        _ => ErrorKind::Crashed,
    }
}

fn valid_entry(entry: &str) -> bool {
    let mut chars = entry.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && entry
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_:<>, ".contains(c))
}

fn arg_expr(arg: &Arg) -> String {
    match arg {
        Arg::Buf(BufId(i)) => format!("b{i}"),
        Arg::I32(v) => format!("static_cast<int>({:#010x}u)", *v as u32),
        Arg::U32(v) => format!("{v:#010x}u"),
        Arg::I64(v) => format!("static_cast<long long>({:#018x}ull)", *v as u64),
        Arg::U64(v) => format!("{v:#018x}ull"),
        Arg::F32(v) => format!("::cuda_emu::f32_from_bits({:#010x}u)", v.to_bits()),
        Arg::F64(v) => format!("::cuda_emu::f64_from_bits({:#018x}ull)", v.to_bits()),
        Arg::Expr(e) => format!("({e})"),
    }
}

/// The inside of a C string literal: `\` and `"` escaped, anything unprintable replaced.
fn c_string_body(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            c if c.is_ascii_graphic() || c == ' ' => out.push(c),
            _ => out.push('?'),
        }
    }
    out
}

fn c_comment_safe(s: &str) -> String {
    c_string_body(s).replace("*/", "*_/")
}

// ------------------------------------------------------------------------------------------
// The CUDA-C lexer: just enough to find code tokens outside comments and literals.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CKind {
    Ident,
    Punct,
    Other,
}

#[derive(Clone, Copy, Debug)]
struct CTok {
    kind: CKind,
    start: usize,
    end: usize,
}

fn line_at(text: &str, offset: usize) -> usize {
    text.as_bytes()[..offset]
        .iter()
        .filter(|&&c| c == b'\n')
        .count()
        + 1
}

fn lex_c(src: &str) -> Result<Vec<CTok>, String> {
    let b = src.as_bytes();
    let ident_start = |c: u8| c.is_ascii_alphabetic() || c == b'_' || c >= 0x80;
    let ident_continue = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80;
    let mut toks = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'/' && b.get(i + 1) == Some(&b'*') {
            let end = find_bytes(b, i + 2, b"*/")
                .ok_or_else(|| format!("unterminated /* comment at line {}", line_at(src, i)))?;
            i = end + 2;
        } else if c == b'"' || c == b'\'' {
            let end = skip_c_quoted(b, i)
                .ok_or_else(|| format!("unterminated literal at line {}", line_at(src, i)))?;
            toks.push(CTok {
                kind: CKind::Other,
                start: i,
                end,
            });
            i = end;
        } else if ident_start(c) {
            let s = i;
            while i < b.len() && ident_continue(b[i]) {
                i += 1;
            }
            let word = &src[s..i];
            if b.get(i) == Some(&b'"') && matches!(word, "R" | "u8R" | "uR" | "UR" | "LR") {
                let open = find_bytes(b, i + 1, b"(")
                    .ok_or_else(|| format!("malformed raw string at line {}", line_at(src, s)))?;
                let mut close = Vec::with_capacity(open - i + 1);
                close.push(b')');
                close.extend_from_slice(&b[i + 1..open]);
                close.push(b'"');
                let end = find_bytes(b, open + 1, &close).ok_or_else(|| {
                    format!("unterminated raw string at line {}", line_at(src, s))
                })?;
                i = end + close.len();
                toks.push(CTok {
                    kind: CKind::Other,
                    start: s,
                    end: i,
                });
            } else if matches!(b.get(i), Some(b'"') | Some(b'\''))
                && matches!(word, "u8" | "u" | "U" | "L")
            {
                let end = skip_c_quoted(b, i)
                    .ok_or_else(|| format!("unterminated literal at line {}", line_at(src, i)))?;
                toks.push(CTok {
                    kind: CKind::Other,
                    start: s,
                    end,
                });
                i = end;
            } else {
                toks.push(CTok {
                    kind: CKind::Ident,
                    start: s,
                    end: i,
                });
            }
        } else if c.is_ascii_digit() || (c == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit))
        {
            let s = i;
            i += 1;
            while i < b.len() {
                let d = b[i];
                let exponent_sign =
                    (d == b'+' || d == b'-') && matches!(b[i - 1], b'e' | b'E' | b'p' | b'P');
                let digit_separator =
                    d == b'\'' && b.get(i + 1).is_some_and(|n| n.is_ascii_alphanumeric());
                if !(exponent_sign || digit_separator || ident_continue(d) || d == b'.') {
                    break;
                }
                i += 1;
            }
            toks.push(CTok {
                kind: CKind::Other,
                start: s,
                end: i,
            });
        } else {
            toks.push(CTok {
                kind: CKind::Punct,
                start: i,
                end: i + 1,
            });
            i += 1;
        }
    }
    Ok(toks)
}

fn find_bytes(hay: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

/// End offset (exclusive) of the quoted literal starting at `start`.
fn skip_c_quoted(b: &[u8], start: usize) -> Option<usize> {
    let q = b[start];
    let mut j = start + 1;
    while j < b.len() {
        match b[j] {
            b'\\' => j += 2,
            b'\n' => return None,
            c if c == q => return Some(j + 1),
            _ => j += 1,
        }
    }
    None
}

/// The rewrites `__shared__ -> static` needs, done on tokens so comments and literals are
/// untouched and every line keeps its number:
///   * `extern __shared__ T name[];` (dynamic shared memory) becomes a reference to the
///     launch's dynamic shared memory pointer, reinterpreted as `T*`;
///   * a `static` next to `__shared__` is blanked, since the shim's `__shared__` is `static`.
///
/// Inline assembly is refused: PTX cannot execute on the CPU.
fn rewrite_cuda_source(src: &str) -> Result<String, EmuError> {
    let source_err = |m: String| EmuError::new(ErrorKind::Source, m);
    let toks = lex_c(src).map_err(source_err)?;
    let text = |t: &CTok| &src[t.start..t.end];
    let is_word = |t: &CTok, w: &str| t.kind == CKind::Ident && &src[t.start..t.end] == w;
    let is_punct = |t: &CTok, p: u8| t.kind == CKind::Punct && src.as_bytes()[t.start] == p;

    for (k, t) in toks.iter().enumerate() {
        if matches!(t.kind, CKind::Ident) && matches!(text(t), "asm" | "__asm__" | "__asm") {
            let next = toks.get(k + 1);
            if next.is_some_and(|n| {
                is_punct(n, b'(') || is_word(n, "volatile") || is_word(n, "__volatile__")
            }) {
                return Err(EmuError::new(
                    ErrorKind::Unsupported,
                    format!(
                        "inline assembly at source line {}: PTX cannot execute on the CPU",
                        line_at(src, t.start)
                    ),
                ));
            }
        }
    }

    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let mut k = 0;
    while k < toks.len() {
        let t = toks[k];
        if is_word(&t, "extern") && toks.get(k + 1).is_some_and(|n| is_word(n, "__shared__")) {
            let semi = (k + 2..toks.len())
                .find(|&j| is_punct(&toks[j], b';'))
                .ok_or_else(|| {
                    source_err(format!(
                        "line {}: `extern __shared__` without `;`",
                        line_at(src, t.start)
                    ))
                })?;
            let mut kept: Vec<CTok> = Vec::new();
            let mut j = k + 2;
            while j < semi {
                let d = toks[j];
                if is_word(&d, "static") {
                    j += 1;
                } else if is_word(&d, "__align__") || is_word(&d, "__attribute__") {
                    j += 1;
                    if j < semi && is_punct(&toks[j], b'(') {
                        let mut level = 0usize;
                        while j < semi {
                            if is_punct(&toks[j], b'(') {
                                level += 1;
                            } else if is_punct(&toks[j], b')') {
                                level -= 1;
                                if level == 0 {
                                    j += 1;
                                    break;
                                }
                            }
                            j += 1;
                        }
                    }
                } else {
                    kept.push(d);
                    j += 1;
                }
            }
            let n = kept.len();
            if n < 4
                || kept[n - 3].kind != CKind::Ident
                || !is_punct(&kept[n - 2], b'[')
                || !is_punct(&kept[n - 1], b']')
            {
                return Err(source_err(format!(
                    "line {}: cannot rewrite `{}`: expected `extern __shared__ <type> <name>[];`",
                    line_at(src, t.start),
                    &src[t.start..toks[semi].end]
                )));
            }
            let name = text(&kept[n - 3]);
            let ty = kept[..n - 3]
                .iter()
                .map(&text)
                .collect::<Vec<_>>()
                .join(" ");
            let newlines = src[t.start..toks[semi].end].matches('\n').count();
            edits.push((
                t.start,
                toks[semi].end,
                format!(
                    "static {ty}* const& {name} = reinterpret_cast<{ty}* const&>(::cuda_emu::g_dyn_smem);{}",
                    "\n".repeat(newlines)
                ),
            ));
            k = semi + 1;
            continue;
        }
        if is_word(&t, "__shared__") {
            if k > 0 && is_word(&toks[k - 1], "static") {
                edits.push((toks[k - 1].start, toks[k - 1].end, " ".repeat(6)));
            }
            if toks.get(k + 1).is_some_and(|n| is_word(n, "static")) {
                edits.push((toks[k + 1].start, toks[k + 1].end, " ".repeat(6)));
            }
        }
        k += 1;
    }

    edits.sort_by_key(|e| e.0);
    let mut out = String::with_capacity(src.len() + 128 * edits.len());
    let mut pos = 0;
    for (start, end, replacement) in edits {
        if start < pos {
            return Err(source_err(format!(
                "overlapping rewrites at line {}",
                line_at(src, start)
            )));
        }
        out.push_str(&src[pos..start]);
        out.push_str(&replacement);
        pos = end;
    }
    out.push_str(&src[pos..]);
    Ok(out)
}

// ------------------------------------------------------------------------------------------
// Reading a string constant out of a Rust source file.

#[derive(Clone, Debug, PartialEq)]
enum RTok {
    Ident(String),
    Punct(char),
    Str { value: String, exact_lines: bool },
    Other,
}

#[derive(Debug)]
struct Extracted {
    value: String,
    /// `(Some(file), line)` for an `include_str!`, `(None, line)` for a literal in the file
    /// itself; absent when escapes or `concat!` break the line mapping.
    origin: Option<(Option<PathBuf>, usize)>,
}

fn lex_rust(src: &str) -> Result<Vec<(RTok, usize)>, String> {
    let cs: Vec<(usize, char)> = src.char_indices().collect();
    let at = |i: usize| cs.get(i).map(|p| p.1);
    let off = |i: usize| cs.get(i).map_or(src.len(), |p| p.0);
    let ident_char = |c: char| c.is_alphanumeric() || c == '_';
    let mut toks = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i].1;
        let start = off(i);
        if c.is_whitespace() {
            i += 1;
        } else if c == '/' && at(i + 1) == Some('/') {
            while i < cs.len() && cs[i].1 != '\n' {
                i += 1;
            }
        } else if c == '/' && at(i + 1) == Some('*') {
            let mut depth = 1;
            i += 2;
            while depth > 0 {
                match (at(i), at(i + 1)) {
                    (Some('/'), Some('*')) => {
                        depth += 1;
                        i += 2;
                    }
                    (Some('*'), Some('/')) => {
                        depth -= 1;
                        i += 2;
                    }
                    (Some(_), _) => i += 1,
                    (None, _) => {
                        return Err(format!(
                            "unterminated block comment at line {}",
                            line_at(src, start)
                        ));
                    }
                }
            }
        } else if let Some((value, next)) = raw_string_at(&cs, i) {
            toks.push((
                RTok::Str {
                    value,
                    exact_lines: true,
                },
                start,
            ));
            i = next;
        } else if c == '"' || ((c == 'b' || c == 'c') && at(i + 1) == Some('"')) {
            let q = if c == '"' { i } else { i + 1 };
            let (value, exact_lines, next) = cooked_string_at(&cs, q)
                .ok_or_else(|| format!("unterminated string at line {}", line_at(src, start)))?;
            toks.push((RTok::Str { value, exact_lines }, start));
            i = next;
        } else if c == '\'' || (c == 'b' && at(i + 1) == Some('\'')) {
            let q = if c == '\'' { i } else { i + 1 };
            if at(q + 1) == Some('\\') {
                let mut j = q + 2;
                while j < cs.len() && cs[j].1 != '\'' {
                    j += if cs[j].1 == '\\' { 2 } else { 1 };
                }
                i = j + 1;
                toks.push((RTok::Other, start));
            } else if at(q + 2) == Some('\'') {
                i = q + 3;
                toks.push((RTok::Other, start));
            } else {
                i = q + 1;
                while i < cs.len() && ident_char(cs[i].1) {
                    i += 1;
                }
                toks.push((RTok::Other, start));
            }
        } else if c == 'r' && at(i + 1) == Some('#') && at(i + 2).is_some_and(ident_char) {
            i += 2;
            let s = i;
            while i < cs.len() && ident_char(cs[i].1) {
                i += 1;
            }
            toks.push((RTok::Ident(src[off(s)..off(i)].to_string()), start));
        } else if c.is_ascii_digit() {
            while i < cs.len()
                && (ident_char(cs[i].1)
                    || (cs[i].1 == '.' && at(i + 1).is_some_and(|n| n.is_ascii_digit())))
            {
                i += 1;
            }
            toks.push((RTok::Other, start));
        } else if ident_char(c) {
            let s = i;
            while i < cs.len() && ident_char(cs[i].1) {
                i += 1;
            }
            toks.push((RTok::Ident(src[off(s)..off(i)].to_string()), start));
        } else {
            toks.push((RTok::Punct(c), start));
            i += 1;
        }
    }
    Ok(toks)
}

/// `r"..."`, `r#"..."#`, `br#"..."#`, `cr"..."` starting at `i`: the value and the index
/// after the literal.
fn raw_string_at(cs: &[(usize, char)], i: usize) -> Option<(String, usize)> {
    let at = |k: usize| cs.get(k).map(|p| p.1);
    let mut j = i;
    if matches!(at(j), Some('b') | Some('c')) {
        j += 1;
    }
    if at(j) != Some('r') {
        return None;
    }
    j += 1;
    let mut hashes = 0;
    while at(j) == Some('#') {
        hashes += 1;
        j += 1;
    }
    if at(j) != Some('"') {
        return None;
    }
    j += 1;
    let mut value = String::new();
    while j < cs.len() {
        if cs[j].1 == '"' && (1..=hashes).all(|h| at(j + h) == Some('#')) {
            return Some((value, j + 1 + hashes));
        }
        if !(cs[j].1 == '\r' && at(j + 1) == Some('\n')) {
            value.push(cs[j].1);
        }
        j += 1;
    }
    None
}

/// A non-raw string whose opening quote is at `q`: its value, whether its lines map 1:1
/// onto the file's, and the index after the literal.
fn cooked_string_at(cs: &[(usize, char)], q: usize) -> Option<(String, bool, usize)> {
    let at = |k: usize| cs.get(k).map(|p| p.1);
    let mut value = String::new();
    let mut exact = true;
    let mut j = q + 1;
    while j < cs.len() {
        match cs[j].1 {
            '"' => return Some((value, exact, j + 1)),
            '\r' if at(j + 1) == Some('\n') => j += 1,
            '\\' => {
                let e = at(j + 1)?;
                j += 2;
                match e {
                    'n' => {
                        value.push('\n');
                        exact = false;
                    }
                    'r' => {
                        value.push('\r');
                        exact = false;
                    }
                    't' => value.push('\t'),
                    '\\' => value.push('\\'),
                    '0' => value.push('\0'),
                    '\'' => value.push('\''),
                    '"' => value.push('"'),
                    'x' => {
                        let hex: String = [at(j)?, at(j + 1)?].iter().collect();
                        value.push(char::from(u8::from_str_radix(&hex, 16).ok()?));
                        j += 2;
                    }
                    'u' => {
                        if at(j) != Some('{') {
                            return None;
                        }
                        let mut hex = String::new();
                        j += 1;
                        while at(j)? != '}' {
                            hex.push(at(j)?);
                            j += 1;
                        }
                        j += 1;
                        value.push(char::from_u32(
                            u32::from_str_radix(&hex.replace('_', ""), 16).ok()?,
                        )?);
                    }
                    '\n' | '\r' => {
                        exact = false;
                        while at(j).is_some_and(char::is_whitespace) {
                            j += 1;
                        }
                    }
                    _ => return None,
                }
            }
            c => {
                value.push(c);
                j += 1;
            }
        }
    }
    None
}

fn extract_rust_str_const(src: &str, name: &str, base_dir: &Path) -> Result<Extracted, String> {
    let toks = lex_rust(src)?;
    let is_ident = |k: usize, w: &str| matches!(toks.get(k), Some((RTok::Ident(s), _)) if s == w);
    let is_punct = |k: usize, p: char| matches!(toks.get(k), Some((RTok::Punct(c), _)) if *c == p);
    let mut sites = Vec::new();
    for k in 1..toks.len() {
        if !is_ident(k, name) || !is_punct(k + 1, ':') {
            continue;
        }
        let item = is_ident(k - 1, "const")
            || is_ident(k - 1, "static")
            || (is_ident(k - 1, "mut") && k >= 2 && is_ident(k - 2, "static"));
        if item {
            sites.push(k);
        }
    }
    let k = match sites.as_slice() {
        [] => return Err(format!("no `const {name}` or `static {name}` item found")),
        [k] => *k,
        many => {
            let lines: Vec<String> = many
                .iter()
                .map(|&s| line_at(src, toks[s].1).to_string())
                .collect();
            return Err(format!(
                "{} items named {name} (lines {}); pass the source with Kernel::new instead",
                many.len(),
                lines.join(", ")
            ));
        }
    };
    let eq = (k + 2..toks.len())
        .find(|&j| is_punct(j, '='))
        .ok_or_else(|| format!("{name} has no initializer"))?;
    let semi = (eq + 1..toks.len())
        .find(|&j| is_punct(j, ';'))
        .ok_or_else(|| format!("{name}'s initializer has no `;`"))?;
    let init = &toks[eq + 1..semi];
    let unsupported = || {
        format!(
            "the initializer of {name} (line {}) is not a string literal, include_str!(\"...\") or concat! of string \
             literals; pass the source with Kernel::new instead",
            line_at(src, toks[eq].1)
        )
    };
    match init {
        [(RTok::Str { value, exact_lines }, at)] => Ok(Extracted {
            value: value.clone(),
            origin: exact_lines.then(|| (None, line_at(src, *at))),
        }),
        [
            (RTok::Ident(m), _),
            (RTok::Punct('!'), _),
            (RTok::Punct('('), _),
            rest @ ..,
        ] if m == "include_str" => {
            let path = match rest {
                [(RTok::Str { value, .. }, _), (RTok::Punct(')'), _)]
                | [
                    (RTok::Str { value, .. }, _),
                    (RTok::Punct(','), _),
                    (RTok::Punct(')'), _),
                ] => value,
                _ => return Err(unsupported()),
            };
            let full = base_dir.join(path);
            let value = fs::read_to_string(&full)
                .map_err(|e| format!("include_str!({path:?}) -> {}: {e}", full.display()))?;
            Ok(Extracted {
                value,
                origin: Some((Some(full), 1)),
            })
        }
        [
            (RTok::Ident(m), _),
            (RTok::Punct('!'), _),
            (RTok::Punct('('), _),
            rest @ ..,
        ] if m == "concat" => {
            let mut value = String::new();
            let mut expect_str = true;
            let Some(((RTok::Punct(')'), _), parts)) = rest.split_last() else {
                return Err(unsupported());
            };
            for (t, _) in parts {
                match (t, expect_str) {
                    (RTok::Str { value: v, .. }, true) => {
                        value.push_str(v);
                        expect_str = false;
                    }
                    (RTok::Punct(','), false) => expect_str = true,
                    _ => return Err(unsupported()),
                }
            }
            Ok(Extracted {
                value,
                origin: None,
            })
        }
        _ => Err(unsupported()),
    }
}
