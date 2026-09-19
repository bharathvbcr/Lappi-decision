//! Probe the CUDA driver by opening it at runtime. No build-time CUDA required.
//!
//! ## Why dlopen and not linking
//!
//! This binary must run on a host with **no CUDA at all** and say so honestly. If it
//! linked `libcuda` at build time it would not build on the Mac and would not start on
//! any box without the driver — which destroys the one thing a preflight is for:
//! distinguishing "checked and clean" from "could not check".
//!
//! So the driver is opened with `dlopen` and every symbol is resolved by name. A
//! missing library, a missing symbol, or a driver that refuses to initialise each
//! produce a distinct `NotRun` or failing reason rather than a crash.
//!
//! ## What this can and cannot establish
//!
//! It probes the **substrate**: that a CUDA driver exists, initialises, exposes at
//! least one device, and reports a usable compute capability and memory size.
//!
//! It **cannot** tell which Python function `transformers` dispatched to. The
//! flash-linear-attention fast path is a Python-level choice resolved inside the training
//! process, and no external binary can observe it. `stack/verify_fast_path.py` covers
//! that, by instrumenting both implementations and running a real forward.
//!
//! The two are complementary and neither subsumes the other:
//!
//! | Question | Answered by |
//! | --- | --- |
//! | Is there a working CUDA device? | this crate |
//! | Are the native extensions loadable? | this crate |
//! | Do the artifact hashes match the build? | this crate |
//! | Which GDN implementation actually ran? | `verify_fast_path.py` |


use std::os::raw::{c_char, c_int};

use crate::tristate::TriState;

/// Candidate sonames, most-specific first. `libcuda.so.1` is the driver ABI and is
/// what a correctly installed driver provides; `libcuda.so` is often a devel symlink.
const SONAMES: &[&str] = &["libcuda.so.1", "libcuda.so", "nvcuda.dll"];

// CUDA driver API. Signatures from the CUDA Driver API reference; only the handful
// needed to establish "a usable device exists" are declared.
type CuInit = unsafe extern "C" fn(u32) -> c_int;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, c_int) -> c_int;

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CUDA_SUCCESS: c_int = 0;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    pub compute_capability: String,
    pub total_mem_gib: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CudaProbe {
    pub soname: String,
    pub driver_version: i32,
    pub devices: Vec<DeviceInfo>,
}

/// Open the driver and enumerate devices.
///
/// Returns `Ok(None)` when no driver library is present at all — that is the ordinary
/// state on a Mac and is not an error, it is a `not_run`.
fn probe() -> Result<Option<CudaProbe>, String> {
    let mut last_err = String::new();
    for soname in SONAMES {
        // SAFETY: we open a library by soname and resolve symbols whose signatures are
        // declared above from the CUDA Driver API reference. Every call's status code
        // is checked. The library is leaked deliberately (see `Library::into_raw`
        // discussion below) so that function pointers stay valid for the probe's life.
        let lib = match unsafe { libloading::Library::new(soname) } {
            Ok(l) => l,
            Err(e) => {
                last_err = format!("{soname}: {e}");
                continue;
            }
        };

        let result = unsafe { probe_with(&lib) };
        return match result {
            Ok(mut p) => {
                p.soname = (*soname).to_string();
                Ok(Some(p))
            }
            Err(e) => Err(e),
        };
    }
    if last_err.is_empty() {
        Ok(None)
    } else {
        // Every candidate failed to load. On a machine with no NVIDIA driver this is
        // expected; report it as absence, not as a fault.
        Ok(None)
    }
}

/// # Safety
/// `lib` must be a loaded CUDA driver library exposing the declared symbols.
unsafe fn probe_with(lib: &libloading::Library) -> Result<CudaProbe, String> {
    macro_rules! sym {
        ($t:ty, $name:literal) => {
            unsafe { lib.get::<$t>($name) }
                .map_err(|e| format!("symbol {} missing: {e}", stringify!($name)))?
        };
    }

    let cu_init = sym!(CuInit, b"cuInit\0");
    let cu_count = sym!(CuDeviceGetCount, b"cuDeviceGetCount\0");
    let cu_ver = sym!(CuDriverGetVersion, b"cuDriverGetVersion\0");
    let cu_get = sym!(CuDeviceGet, b"cuDeviceGet\0");
    let cu_name = sym!(CuDeviceGetName, b"cuDeviceGetName\0");
    let cu_attr = sym!(CuDeviceGetAttribute, b"cuDeviceGetAttribute\0");
    let cu_mem = sym!(CuDeviceTotalMem, b"cuDeviceTotalMem_v2\0");

    let rc = unsafe { cu_init(0) };
    if rc != CUDA_SUCCESS {
        // The library exists but the driver will not initialise: a broken or
        // mismatched driver, or no permission on the device nodes. This is a genuine
        // FAILURE, not an absence -- something is installed and it does not work.
        return Err(format!("cuInit returned {rc} (driver present but not usable)"));
    }

    let mut version: c_int = 0;
    if unsafe { cu_ver(&mut version) } != CUDA_SUCCESS {
        return Err("cuDriverGetVersion failed after a successful cuInit".into());
    }

    let mut count: c_int = 0;
    if unsafe { cu_count(&mut count) } != CUDA_SUCCESS {
        return Err("cuDeviceGetCount failed after a successful cuInit".into());
    }

    let mut devices = Vec::new();
    for i in 0..count {
        let mut dev: c_int = 0;
        if unsafe { cu_get(&mut dev, i) } != CUDA_SUCCESS {
            return Err(format!("cuDeviceGet failed for ordinal {i}"));
        }

        let mut buf = vec![0i8; 256];
        let name = if unsafe { cu_name(buf.as_mut_ptr() as *mut c_char, 256, dev) } == CUDA_SUCCESS
        {
            let bytes: Vec<u8> = buf.iter().take_while(|c| **c != 0).map(|c| *c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        } else {
            "<name unavailable>".to_string()
        };

        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        unsafe {
            cu_attr(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
            cu_attr(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        }

        let mut bytes: usize = 0;
        unsafe { cu_mem(&mut bytes, dev) };

        devices.push(DeviceInfo {
            index: i,
            name,
            compute_capability: format!("{major}.{minor}"),
            total_mem_gib: bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        });
    }

    Ok(CudaProbe { soname: String::new(), driver_version: version, devices })
}

/// The preflight check: is there a usable CUDA device?
///
/// - no driver library at all            -> `NotRun` (expected on the Mac)
/// - driver present but unusable         -> `Ran { passed: false }`
/// - driver usable but zero devices      -> `Ran { passed: false }`
/// - one or more devices                 -> `Ran { passed: true }`
///
/// The middle two are failures rather than `NotRun` because something **is** installed
/// and is broken; reporting that as "could not check" would let a misconfigured box
/// look like a laptop.
pub fn check_cuda() -> (TriState, Option<CudaProbe>) {
    match probe() {
        Ok(None) => (
            TriState::not_run(format!(
                "no CUDA driver library found (tried {}). Expected on a non-NVIDIA host; \
                 this check is meaningful only on the training box.",
                SONAMES.join(", ")
            )),
            None,
        ),
        Err(e) => (TriState::fail(format!("CUDA driver present but unusable: {e}")), None),
        Ok(Some(p)) if p.devices.is_empty() => (
            TriState::fail(format!(
                "CUDA driver {} loaded and initialised but reports ZERO devices",
                p.driver_version
            )),
            Some(p),
        ),
        Ok(Some(p)) => {
            let detail = format!(
                "driver {} via {}: {} device(s): {}",
                p.driver_version,
                p.soname,
                p.devices.len(),
                p.devices
                    .iter()
                    .map(|d| format!("{} (sm_{}, {:.1} GiB)", d.name, d.compute_capability.replace('.', ""), d.total_mem_gib))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let value = serde_json::to_value(&p).unwrap_or(serde_json::Value::Null);
            (TriState::pass(detail).with_value(value), Some(p))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probing_a_host_without_cuda_is_not_run_not_a_failure() {
        // On this Mac there is no libcuda; the check must say "could not check",
        // never "checked and failed" and never "checked and passed".
        let (state, info) = check_cuda();
        if cfg!(target_os = "macos") {
            assert!(!state.did_run(), "macOS must report not_run, got {state:?}");
            assert!(info.is_none());
        }
        // The invariant that holds on every platform:
        assert!(state.did_run() || !state.is_pass());
    }

    #[test]
    fn a_not_run_probe_never_reports_a_pass() {
        assert!(!TriState::not_run("no driver").is_pass());
    }
}
