//! ojas-qwen35 has no vocabulary remap, so a run needs the identity: the shards' ids must be
//! the snapshot's. CPU only, on qd-train's tiny shard fixture, whose remap keeps 256 of 512 ids.

use std::path::PathBuf;
use std::sync::Arc;

use qd_train::files::OsFiles;
use qd_train::held_out::DataConfig;
use qd_train::shards::{ShardOpen, ShardReader};
use qd_train_metal::remap::{check_identity, require_identity_remap, REMAP_CHECK};

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture() -> PathBuf {
    crate_dir().join("../qd-train/tests/fixtures/shards-tiny")
}

/// qd-train's fixture, through the door, as its own tests open it: its root anchors the
/// held-out roots, the revision is the one it was written at, and `allow_stale_code` because a
/// frozen fixture's `qd_data` fingerprint drifts (the check still runs and is recorded).
fn open(shards: PathBuf) -> ShardReader {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("oracle/meta.json")).unwrap()).unwrap();
    let mut opts = ShardOpen::new(DataConfig::default(), fixture(), crate_dir().join("../../python/qd_data"));
    opts.expect_rev = Some(meta["corpus_rev"].as_str().unwrap().to_owned());
    opts.allow_stale_code = true;
    opts.files = Arc::new(OsFiles);
    ShardReader::open(&shards, &fixture().join("data/pool/train.json"), &opts).expect("the fixture opens through the door")
}

#[test]
fn only_the_identity_passes() {
    assert_eq!(check_identity(&[0, 1, 2], &[0, 1, 2]), Ok(3));
    let swap = check_identity(&[1, 0, 2], &[1, 0, 2]).unwrap_err();
    assert!(swap.contains("new_to_old[0] = 1") && swap.contains("renumbers"), "{swap}");
    let slice = check_identity(&[0, -1, 1], &[0, 2]).unwrap_err();
    assert!(slice.contains("keeps 2 of 3 tokens"), "{slice}");
    // Inverse tables that agree with each other but not with the identity in one direction.
    assert!(check_identity(&[0, 1, 3], &[0, 1, 2]).unwrap_err().contains("old_to_new[2] = 3"));
    assert!(check_identity(&[], &[]).unwrap_err().contains("no tokens"));
}

#[test]
fn the_fixtures_slicing_remap_is_refused_by_name() {
    let reader = open(fixture().join("shards/train"));
    assert!(reader.check(REMAP_CHECK).unwrap().is_pass(), "the door verified the remap first");
    let err = require_identity_remap(&reader, &OsFiles, 512).unwrap_err();
    assert!(err.contains("keeps 256 of 512 tokens") && err.contains("ojas-qwen35 has no remap"), "{err}");
}

/// A shard set with no remap artifact is refused: the door records `not_run`, and an unknown
/// vocabulary is not the identity.
#[test]
fn a_shard_set_without_a_remap_is_refused() {
    let dir = std::env::temp_dir().join(format!("qd-train-metal-remap-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture().join("shards/train")).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("remap.") {
            std::fs::copy(entry.path(), dir.join(&name)).unwrap();
        }
    }
    let reader = open(dir.clone());
    assert!(!reader.check(REMAP_CHECK).unwrap().is_pass());
    let err = require_identity_remap(&reader, &OsFiles, 256).unwrap_err();
    assert!(err.contains(REMAP_CHECK) && err.contains("vocabulary is unknown"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}
