//! **Caller records cannot reach a training process** (CLAUDE.md rule 3, and the human's
//! "Synthetic only", 2026-10-06).
//!
//! Apps write what they saw and chose around a decision to
//! `$HOME/Library/Application Support/Lappi/heldout/caller-records/<app>/<day>.jsonl`
//! (`docs/caller-contract.md`; `qd_runtime::caller_record::STORE_RELATIVE_TO_HOME` holds the same
//! literal). The `heldout` segment is the whole mechanism: this test proves the door's existing
//! marker layer refuses that store with no caller-specific code, under the default config and with
//! no configured root that happens to cover it.

use std::path::{Path, PathBuf};

use qd_train::held_out::{
    assert_path_not_held_out, DataConfig, DEFAULT_HELD_OUT_FAMILIES,
    DEFAULT_HELD_OUT_PATH_MARKERS, DEFAULT_SEED,
};

/// Pinned twice on purpose: the runtime's constant and this literal must move together.
const STORE_RELATIVE_TO_HOME: &str = "Library/Application Support/Lappi/heldout/caller-records";

fn default_config() -> DataConfig {
    DataConfig::new(
        DEFAULT_HELD_OUT_FAMILIES.iter().map(|s| s.to_string()).collect(),
        // A root nowhere near the store, so only the marker layer can refuse it.
        vec![PathBuf::from("data/heldout")],
        DEFAULT_HELD_OUT_PATH_MARKERS.iter().map(|s| s.to_string()).collect(),
        DEFAULT_SEED,
    )
    .expect("the default config is valid")
}

#[test]
fn every_app_s_record_file_is_refused_by_the_marker_layer() {
    let config = default_config();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for app in ["devcouncil", "devtype", "gitpulse", "any-future-app"] {
        let path = PathBuf::from("/Users/someone")
            .join(STORE_RELATIVE_TO_HOME)
            .join(app)
            .join("2026-10-06.jsonl");
        let refused = assert_path_not_held_out(&path, &config, &repo_root);
        assert!(refused.is_err(), "{} must be refused by the training door", path.display());
        let message = refused.unwrap_err().to_string().to_ascii_lowercase();
        assert!(message.contains("rule 3") && message.contains("marker"), "{message}");
    }
}

#[test]
fn a_record_file_copied_into_the_training_tree_is_not_caught_by_the_marker() {
    // The limit, stated as a test: the door refuses by path, so a copy under data/pool/ would pass
    // the marker layer. The contract therefore forbids copying records anywhere (and
    // `qd-caller-records` refuses a record file outside a held-out segment). This case documents
    // that the path is the guard; it is not a hole this test claims to close.
    let config = default_config();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let copied = repo_root.join("data/pool/caller-copy/2026-10-06.jsonl");
    assert!(assert_path_not_held_out(&copied, &config, &repo_root).is_ok());
}
