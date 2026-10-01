//! The binary's per-subcommand input bound, through the CLI. Each request file is sparse
//! (`set_len`), so a file past a bound costs no disk and the refusal happens on its size
//! before anything is read.

use std::path::PathBuf;
use std::process::Command;

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("qd-prep-input-bound-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Runs `qd-prep <command>` on a sparse request of `size` bytes; returns (success, stderr,
/// whether a reply was written).
fn run_on_sparse(command: &str, size: u64) -> (bool, String, bool) {
    let dir = scratch(command);
    let input = dir.join("request.bin");
    let output = dir.join("reply.bin");
    let file = std::fs::File::create(&input).expect("request file");
    file.set_len(size).expect("sparse request");
    drop(file);
    let out = Command::new(env!("CARGO_BIN_EXE_qd-prep"))
        .args([command, "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()
        .expect("qd-prep runs");
    let wrote = output.exists();
    std::fs::remove_dir_all(&dir).expect("scratch cleanup");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        wrote,
    )
}

#[test]
fn linfit_and_ngrams_refuse_only_past_their_own_bound() {
    let bound = qd_prep::linwire::MAX_INPUT_BYTES;
    for command in ["linfit", "ngrams"] {
        let (ok, stderr, wrote) = run_on_sparse(command, bound + 1);
        assert!(!ok && !wrote, "{command}: {stderr}");
        assert!(
            stderr.contains(&format!("{} bytes; the bound is {bound}", bound + 1)),
            "{command} refused at another bound: {stderr}"
        );
    }
}

#[test]
fn minhash_keeps_its_four_gib_bound() {
    let bound = qd_prep::wire::MAX_INPUT_BYTES;
    assert_eq!(bound, 4 << 30);
    let (ok, stderr, wrote) = run_on_sparse("minhash", bound + 1);
    assert!(!ok && !wrote, "{stderr}");
    assert!(
        stderr.contains(&format!("the bound is {bound}")),
        "{stderr}"
    );
}
