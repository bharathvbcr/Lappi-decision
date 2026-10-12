//! Staging crate for `cuda_emu`, the CPU emulator of CUDA-C kernel sources.
//!
//! Everything lives under `tests/`, laid out exactly as it is meant to land in the ojas
//! crate: `tests/cuda_emu/mod.rs` (the harness), `tests/cuda_emu/cuda_emu.h` (the shim the
//! kernels compile against) and `tests/cuda_emu_selftest.rs` (what the emulator itself is
//! checked by). This library is empty because Cargo needs a target to hang the tests on.
//!
//! The emulator is test-time CPU emulation, not a GPU run, and is never reported as one.
