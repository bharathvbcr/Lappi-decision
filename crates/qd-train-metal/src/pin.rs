//! The ojas pin, at the run's door.
//!
//! This crate builds against the `ojas` checkout beside the repo by path, so what a row's
//! numbers were produced by is whatever commit that checkout was on. `upstream.lock.json` names
//! the commit; [`check_ojas`] compares the checkout's HEAD with it (through `qd-preflight`'s
//! `pin`, the one implementation of "what commit is that sibling on", which the `make pin` gate
//! also runs) and refuses a run that differs.
//!
//! `--unpinned-ojas` lifts the refusal and nothing else: the row is marked with the reason
//! `unpinned ojas` ([`mark_quick_reason`]), and the row is `quick` regardless (rule 8: every row
//! this binary writes is quick, so a pin can only make a row stricter). An unpinned run can
//! never read as a promotable one. The flag also applies when the lock or the checkout's HEAD
//! cannot be read: an unverifiable pin is an unpinned run, never a pinned one.
//!
//! Portable: this file reads two small files and compares two strings, so it is tested on any
//! host. Only the run that calls it is macOS-only.

use std::path::Path;

use qd_preflight::pin::{PinOutcome, check_pin};

/// The ojas workspace ojas-qwen35 is built from (`../../../ojas` from this manifest, the same
/// directory `Cargo.toml`'s path dependencies reach).
pub const OJAS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../ojas");
/// The lock at the repository root.
pub const LOCK_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../upstream.lock.json");
/// The lock's key for ojas.
pub const OJAS_KEY: &str = "ojas";

/// The reason a row carries when the run was not at the locked ojas commit.
pub const UNPINNED_REASON: &str = "unpinned ojas";

/// Whether the run was at the commit the lock names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinState {
    /// The checkout's HEAD is the locked commit.
    Pinned { sha: String },
    /// It is not (or could not be read), and `--unpinned-ojas` was given. `detail` says which.
    Unpinned { detail: String },
}

/// Check the checkout at `ojas_dir` against `lock_path`. `Err` is the refusal, naming both
/// commits when both are known.
pub fn check_ojas(
    lock_path: &Path,
    ojas_dir: &Path,
    allow_unpinned: bool,
) -> Result<PinState, String> {
    let (detail, why) = match check_pin(lock_path, OJAS_KEY, ojas_dir) {
        PinOutcome::Match { sha } => return Ok(PinState::Pinned { sha }),
        PinOutcome::Mismatch { locked, actual } => (
            format!("ojas HEAD {actual} differs from the locked {locked}"),
            format!("ojas is at {actual}, but upstream.lock.json pins {locked}"),
        ),
        PinOutcome::NotRun { reason } => (
            format!("the ojas pin could not be checked: {reason}"),
            format!("the ojas pin could not be checked ({reason})"),
        ),
    };
    if allow_unpinned {
        Ok(PinState::Unpinned { detail })
    } else {
        Err(format!(
            "{why}. A row must say which ojas produced it: check ojas out at the locked commit, \
             or pass --unpinned-ojas to run anyway, in which case the row is marked \
             \"{UNPINNED_REASON}\""
        ))
    }
}

/// `base` (a row's quick reason) with the pin state appended: unchanged when pinned, and
/// carrying [`UNPINNED_REASON`] and its detail when not.
pub fn mark_quick_reason(base: String, state: &PinState) -> String {
    match state {
        PinState::Pinned { .. } => base,
        PinState::Unpinned { detail } => format!("{base}; {UNPINNED_REASON}: {detail}"),
    }
}
