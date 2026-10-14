//! The ojas pin at the run's door, against a fake `OJAS_DIR` (a temp directory with a hand-made
//! `.git/HEAD`). CPU only, portable: nothing here opens a shard set, a snapshot or a GPU.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use qd_train_metal::args::{TrainArgs, parse};
use qd_train_metal::pin::{
    LOCK_PATH, OJAS_DIR, PinState, UNPINNED_REASON, check_ojas, mark_quick_reason,
};

const LOCKED: &str = "bc498c4375c2e4753d784e6e9fc267a8d18b038a";
const OTHER: &str = "0123456789abcdef0123456789abcdef01234567";

fn scratch(tag: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let d = std::env::temp_dir().join(format!(
        "qd-train-metal-pin-{}-{tag}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// A lock at `dir/upstream.lock.json` and a fake ojas checkout at `dir/ojas` on `head`.
fn fixture(dir: &Path, head: &str) -> (PathBuf, PathBuf) {
    let lock = dir.join("upstream.lock.json");
    fs::write(
        &lock,
        format!(r#"{{"schema_version":1,"ojas":{{"url":"u","revision":"{LOCKED}"}}}}"#),
    )
    .unwrap();
    let ojas = dir.join("ojas");
    fs::create_dir_all(ojas.join(".git")).unwrap();
    fs::write(ojas.join(".git/HEAD"), format!("{head}\n")).unwrap();
    (lock, ojas)
}

#[test]
fn an_ojas_at_the_locked_commit_is_pinned_and_leaves_the_reason_alone() {
    let d = scratch("pinned");
    let (lock, ojas) = fixture(&d, LOCKED);
    for allow in [false, true] {
        let state = check_ojas(&lock, &ojas, allow).unwrap();
        assert_eq!(
            state,
            PinState::Pinned {
                sha: LOCKED.to_owned()
            }
        );
        assert_eq!(mark_quick_reason("base".to_owned(), &state), "base");
    }
}

#[test]
fn an_ojas_at_another_commit_is_refused_naming_both_and_the_flag() {
    let d = scratch("refused");
    let (lock, ojas) = fixture(&d, OTHER);
    let why = check_ojas(&lock, &ojas, false).unwrap_err();
    assert!(why.contains(LOCKED) && why.contains(OTHER), "{why}");
    assert!(why.contains("--unpinned-ojas"), "{why}");
}

#[test]
fn the_flag_lets_the_run_through_and_the_row_says_unpinned_ojas() {
    let d = scratch("flag");
    let (lock, ojas) = fixture(&d, OTHER);
    let state = check_ojas(&lock, &ojas, true).unwrap();
    let reason = mark_quick_reason("Metal/tessl trainer, 1 seed".to_owned(), &state);
    assert!(reason.contains(UNPINNED_REASON), "{reason}");
    assert!(
        reason.starts_with("Metal/tessl trainer, 1 seed"),
        "{reason}"
    );
    assert!(
        reason.contains(OTHER) && reason.contains(LOCKED),
        "{reason}"
    );
}

#[test]
fn an_unreadable_pin_is_never_pinned() {
    let d = scratch("unreadable");
    let (lock, _) = fixture(&d, LOCKED);
    // Absent checkout.
    let absent = d.join("no-ojas");
    assert!(check_ojas(&lock, &absent, false).is_err());
    let state = check_ojas(&lock, &absent, true).unwrap();
    assert!(matches!(state, PinState::Unpinned { .. }), "{state:?}");
    assert!(mark_quick_reason(String::new(), &state).contains(UNPINNED_REASON));
    // Absent lock.
    let (_, ojas) = fixture(&scratch("nolock"), LOCKED);
    let no_lock = d.join("missing.json");
    assert!(check_ojas(&no_lock, &ojas, false).is_err());
    assert!(matches!(
        check_ojas(&no_lock, &ojas, true).unwrap(),
        PinState::Unpinned { .. }
    ));
}

#[test]
fn the_committed_lock_and_the_built_against_ojas_agree_on_where_to_look() {
    // The constants the run uses name the repo's own lock and the path dependencies' ojas.
    assert!(Path::new(LOCK_PATH).is_file(), "{LOCK_PATH}");
    assert!(
        OJAS_DIR.ends_with("/../../../ojas"),
        "OJAS_DIR must be the directory Cargo.toml's path dependencies reach: {OJAS_DIR}"
    );
}

fn base() -> Vec<String> {
    [
        "--snapshot",
        "/nonexistent/snap/b1485b2f",
        "--data-root",
        "/nonexistent/out",
        "--shards",
        "/nonexistent/out/shards/train",
        "--manifest",
        "/nonexistent/out/data/pool/train.json",
        "--qd-data",
        "/nonexistent/python/qd_data",
        "--head-init",
        "/nonexistent/span_head_init-seed0.safetensors",
        "--head-init-sha256",
        "8f10b58bb45af911e5e8524157f1898604c3bca7fb97ff66437614cda5d48230",
        "--head-init-content-digest",
        "81bd953e1f27e8e2b703264df663f15f239327c9523d3e85973687009db471d6",
        "--out",
        "/nonexistent/run",
        "--ledger",
        "/nonexistent/ledger/mac-ojas-rung-d-2026-10-01.jsonl",
        "--repo",
        "/nonexistent",
        "--seed",
        "0",
        "--lr",
        "1e-5",
        "--steps",
        "200",
        "--batch-tokens",
        "35403",
        "--span-weight",
        "1.0",
        "--cap-s",
        "21600",
        "--operands",
        "bf16",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

fn parsed(extra: &[&str]) -> Result<TrainArgs, qd_train_metal::args::ArgsError> {
    let mut v = base();
    v.extend(extra.iter().map(|s| (*s).to_owned()));
    parse(&v)
}

#[test]
fn the_flag_defaults_off_and_takes_no_value() {
    assert!(!parsed(&[]).unwrap().unpinned_ojas);
    assert!(parsed(&["--unpinned-ojas"]).unwrap().unpinned_ojas);
    assert!(parsed(&["--unpinned-ojas=true"]).is_err());
    assert!(parsed(&["--unpinned-ojas", "--unpinned-ojas"]).is_err());
}
