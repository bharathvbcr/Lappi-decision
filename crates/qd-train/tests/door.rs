//! CLAUDE.md rule 3 at the Rust reader: held-out data and the two task-holdout families are
//! refused **before any token byte is read**. Removing or weakening these tests is a refused
//! change.
//!
//! Every fail-first test runs the same shape: open through the door with a byte-counting
//! [`CountingFiles`]; if (wrongly) admitted, read every sequence; then assert the byte count
//! *before* asserting the error. So a door that admits held-out data fails on the count of
//! token bytes it let through, not only on a missing error -- which is how these tests were
//! shown to fail against a stub door that permits everything (see the L-data handoff).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::{
    CountingFiles, copy_dir, fixture, fixture_open, open_and_drain, oracle, qd_data_dir, scratch,
};
use qd_train::files::OsFiles;
use qd_train::held_out::{
    DEFAULT_HELD_OUT_FAMILIES, DEFAULT_HELD_OUT_PATH_MARKERS, DEFAULT_SEED, DataConfig, DoorError,
    HeldOutLayer, open_training_manifest,
};
use qd_train::shards::{ShardError, ShardReader, TOKENS_NAME};
use qd_train::tristate::TriState;

fn manifest(split: &str) -> PathBuf {
    fixture()
        .join("data")
        .join("pool")
        .join(format!("{split}.json"))
}

fn door_fixture(name: &str) -> PathBuf {
    fixture().join("door").join(name)
}

fn train_shards() -> PathBuf {
    fixture().join("shards").join("train")
}

fn layer(err: &ShardError) -> Option<HeldOutLayer> {
    err.held_out().map(|v| v.layer)
}

fn config_with_roots(roots: Vec<PathBuf>) -> DataConfig {
    DataConfig::new(
        DEFAULT_HELD_OUT_FAMILIES
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        roots,
        DEFAULT_HELD_OUT_PATH_MARKERS
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        DEFAULT_SEED,
    )
    .expect("a valid config")
}

// --- the four fail-first cases the task names ------------------------------------------------

#[test]
fn a_held_out_shard_path_is_refused_before_any_byte_is_read() {
    let tmp = scratch("marker");
    let shards = tmp.join("held_out").join("train");
    copy_dir(&train_shards(), &shards);
    let files = CountingFiles::new();
    let outcome = open_and_drain(&shards, &manifest("train"), &fixture_open(files.clone()));
    assert_eq!(
        files.total(),
        0,
        "{} bytes were read ({} of them token bytes) from a shard set under a held_out segment",
        files.total(),
        files.bytes_read(&shards.join(TOKENS_NAME))
    );
    let err = outcome.expect_err("a shard set under a held_out path segment must be refused");
    assert_eq!(layer(&err), Some(HeldOutLayer::PathMarker), "{err}");
}

#[test]
fn a_configured_held_out_root_is_refused_before_any_byte_is_read_even_through_a_symlink() {
    let tmp = scratch("root");
    let quarantine = tmp.join("quarantine");
    copy_dir(&train_shards(), &quarantine.join("train"));
    std::os::unix::fs::symlink(&quarantine, tmp.join("link")).expect("symlink");
    std::fs::create_dir_all(tmp.join("elsewhere")).expect("mkdir");
    for via in [
        quarantine.join("train"),
        tmp.join("link").join("train"),
        tmp.join("elsewhere")
            .join("..")
            .join("quarantine")
            .join("train"),
    ] {
        let files = CountingFiles::new();
        let mut opts = fixture_open(files.clone());
        opts.config = config_with_roots(vec![PathBuf::from("quarantine")]);
        opts.repo_root = tmp.clone();
        let outcome = open_and_drain(&via, &manifest("train"), &opts);
        assert_eq!(
            files.total(),
            0,
            "{} bytes read through {}",
            files.total(),
            via.display()
        );
        let err =
            outcome.expect_err("a shard set under a configured held-out root must be refused");
        assert_eq!(
            layer(&err),
            Some(HeldOutLayer::ConfiguredRoot),
            "via {}: {err}",
            via.display()
        );
    }
}

#[test]
fn a_manifest_declaring_split_heldout_is_refused_before_any_shard_byte_is_read() {
    // The held-out manifest, moved to a path with no held-out marker: only its own
    // declaration is left to refuse it.
    let files = CountingFiles::new();
    let outcome = open_and_drain(
        &train_shards(),
        &door_fixture("heldout-moved.json"),
        &fixture_open(files.clone()),
    );
    assert_eq!(
        files.bytes_read_under(&train_shards()),
        0,
        "shard bytes read behind a held-out manifest"
    );
    let err = outcome.expect_err("a manifest that declares split heldout must be refused");
    assert_eq!(layer(&err), Some(HeldOutLayer::ManifestSplit), "{err}");
}

#[test]
fn a_manifest_carrying_a_held_out_family_is_refused_before_any_shard_byte_is_read() {
    let files = CountingFiles::new();
    let outcome = open_and_drain(
        &train_shards(),
        &door_fixture("train-with-heldout-family.json"),
        &fixture_open(files.clone()),
    );
    assert_eq!(
        files.bytes_read_under(&train_shards()),
        0,
        "shard bytes read behind a held-out family"
    );
    let err = outcome.expect_err("a train manifest carrying a held-out family must be refused");
    assert_eq!(layer(&err), Some(HeldOutLayer::HeldOutFamily), "{err}");
    let v = err.held_out().expect("held-out violation");
    assert!(
        v.actual.contains("qa.answerability") || v.actual.contains("code.language_id"),
        "{v}"
    );
}

// --- rule 3 at the shard boundary -------------------------------------------------------------

#[test]
fn a_header_declaring_split_heldout_is_refused_before_any_token_byte_is_read() {
    let tmp = scratch("header-heldout");
    let shards = tmp.join("train");
    copy_dir(&train_shards(), &shards);
    std::fs::copy(
        door_fixture("header-split-heldout.json"),
        shards.join("header.json"),
    )
    .expect("copy");
    let files = CountingFiles::new();
    let outcome = open_and_drain(&shards, &manifest("train"), &fixture_open(files.clone()));
    assert_eq!(
        files.bytes_read(&shards.join(TOKENS_NAME)),
        0,
        "token bytes read from a heldout-split set"
    );
    let err = outcome.expect_err("a header declaring split heldout must be refused");
    assert_eq!(layer(&err), Some(HeldOutLayer::ShardSplit), "{err}");
}

#[test]
fn a_report_only_set_is_refused_by_the_training_reader_before_any_token_byte() {
    let shards = fixture().join("shards").join("val-report-only");
    let files = CountingFiles::new();
    let outcome = open_and_drain(&shards, &manifest("val"), &fixture_open(files.clone()));
    assert_eq!(files.bytes_read(&shards.join(TOKENS_NAME)), 0);
    let err = outcome.expect_err("a report_only set must be refused for training");
    assert!(err.to_string().contains("report_only"), "{err}");
    // Python's ShardReader admits it; the refusal here is deliberate (oracle/door.json).
    assert_eq!(
        oracle("door.json")["shards/val-report-only"]["admitted"],
        true
    );
}

#[test]
fn a_shard_set_beside_the_wrong_manifest_is_refused_before_any_token_byte() {
    let files = CountingFiles::new();
    let outcome = open_and_drain(
        &train_shards(),
        &manifest("val"),
        &fixture_open(files.clone()),
    );
    assert_eq!(files.bytes_read(&train_shards().join(TOKENS_NAME)), 0);
    let err = outcome.expect_err("the train set read beside the val manifest must be refused");
    assert!(err.to_string().contains("built from"), "{err}");
}

#[test]
fn the_held_out_manifest_at_its_real_path_is_refused_by_layer_one() {
    let files = CountingFiles::new();
    let path = fixture().join("data").join("heldout").join("heldout.json");
    let err = open_training_manifest(
        &path,
        &DataConfig::default(),
        &fixture(),
        false,
        files.as_ref(),
    )
    .expect_err("data/heldout is the default held-out root");
    assert_eq!(files.total(), 0, "the held-out manifest's bytes were read");
    assert!(
        matches!(err, DoorError::HeldOut(ref v) if v.layer == HeldOutLayer::ConfiguredRoot),
        "{err}"
    );
}

// --- the rest of the manifest door ------------------------------------------------------------

#[test]
fn every_door_fixture_gets_pythons_verdict_at_the_same_layer() {
    let verdicts = oracle("door.json");
    let cases: [(&str, PathBuf, Option<HeldOutLayer>); 9] = [
        ("pool/train", manifest("train"), None),
        ("pool/val", manifest("val"), None),
        (
            "heldout/heldout",
            fixture().join("data/heldout/heldout.json"),
            Some(HeldOutLayer::ConfiguredRoot),
        ),
        (
            "door/heldout-moved",
            door_fixture("heldout-moved.json"),
            Some(HeldOutLayer::ManifestSplit),
        ),
        (
            "door/train-with-heldout-family",
            door_fixture("train-with-heldout-family.json"),
            Some(HeldOutLayer::HeldOutFamily),
        ),
        (
            "door/train-other-holdout",
            door_fixture("train-other-holdout.json"),
            Some(HeldOutLayer::RecordedHoldout),
        ),
        (
            "door/train-not-run",
            door_fixture("train-not-run.json"),
            Some(HeldOutLayer::SnapshotNotRun),
        ),
        (
            "door/train-failed",
            door_fixture("train-failed.json"),
            Some(HeldOutLayer::SnapshotFailed),
        ),
        (
            "door/train-tampered",
            door_fixture("train-tampered.json"),
            None,
        ),
    ];
    for (name, path, want) in cases {
        let python = &verdicts[name];
        let rust =
            open_training_manifest(&path, &DataConfig::default(), &fixture(), false, &OsFiles);
        assert_eq!(
            rust.is_ok(),
            python["admitted"].as_bool().expect("admitted"),
            "{name}: {rust:?}"
        );
        match rust {
            Ok(admitted) => {
                assert_eq!(
                    Some(admitted.data_snapshot_hash.as_str()),
                    python["data_snapshot_hash"].as_str(),
                    "{name}: the re-derived data_snapshot_hash must equal Python's"
                );
                assert!(admitted.checks.iter().all(|(_, c)| c.is_pass()), "{name}");
            }
            Err(DoorError::HeldOut(v)) => {
                assert_eq!(Some(v.layer), want, "{name}: {v}");
                assert_eq!(python["error"], "HeldOutViolation", "{name}");
            }
            Err(DoorError::Manifest { detail, .. }) => {
                assert_eq!(name, "door/train-tampered", "{detail}");
                assert!(
                    detail.contains("does not match the hash re-derived"),
                    "{detail}"
                );
                assert_eq!(python["error"], "ValueError", "{name}");
            }
            Err(e) => panic!("{name}: unexpected {e}"),
        }
    }
}

#[test]
fn allow_not_run_snapshot_admits_not_run_but_nothing_admits_ran_and_failed() {
    let not_run = open_training_manifest(
        &door_fixture("train-not-run.json"),
        &DataConfig::default(),
        &fixture(),
        true,
        &OsFiles,
    )
    .expect("not_run is admitted when asked for deliberately");
    assert!(matches!(
        not_run.checks.iter().find(|(k, _)| k == "snapshot_status"),
        Some((_, TriState::NotRun { .. }))
    ));
    let failed = open_training_manifest(
        &door_fixture("train-failed.json"),
        &DataConfig::default(),
        &fixture(),
        true,
        &OsFiles,
    );
    assert!(
        matches!(failed, Err(DoorError::HeldOut(ref v)) if v.layer == HeldOutLayer::SnapshotFailed)
    );
}

#[test]
fn segment_markers_are_exact_so_a_lookalike_directory_is_not_a_holdout() {
    let tmp = scratch("lookalike");
    let shards = tmp.join("withheld_outputs").join("train");
    copy_dir(&train_shards(), &shards);
    let reader = open_and_drain(
        &shards,
        &manifest("train"),
        &fixture_open(Arc::new(OsFiles)),
    )
    .expect("withheld_outputs is not a held-out marker");
    assert_eq!(reader.len(), 98);
    let upper = tmp.join("HeldOut").join("train");
    copy_dir(&train_shards(), &upper);
    let err = ShardReader::open(&upper, &manifest("train"), &fixture_open(Arc::new(OsFiles)))
        .expect_err("markers are case-folded");
    assert_eq!(layer(&err), Some(HeldOutLayer::PathMarker));
}

// --- provenance checks the reader makes before the first token -------------------------------

#[test]
fn drifted_qd_data_is_refused_unless_read_deliberately() {
    let tmp = scratch("drift");
    let stale = tmp.join("qd_data");
    std::fs::create_dir_all(&stale).expect("mkdir");
    for entry in std::fs::read_dir(qd_data_dir()).expect("read qd_data") {
        let p = entry.expect("entry").path();
        if p.extension().is_some_and(|e| e == "py") {
            std::fs::copy(&p, stale.join(p.file_name().expect("name"))).expect("copy");
        }
    }
    std::fs::write(
        stale.join("render.py"),
        b"# edited after the shard set was written\n",
    )
    .expect("edit");
    let files = CountingFiles::new();
    let mut opts = fixture_open(files.clone());
    opts.qd_data_dir = stale.clone();
    opts.allow_stale_code = false;
    let err = open_and_drain(&train_shards(), &manifest("train"), &opts)
        .expect_err("drifted code must be refused");
    assert!(err.to_string().contains("render.py"), "{err}");
    assert_eq!(files.bytes_read(&train_shards().join(TOKENS_NAME)), 0);
    opts.allow_stale_code = true;
    let reader =
        ShardReader::open(&train_shards(), &manifest("train"), &opts).expect("read deliberately");
    assert!(
        reader
            .check("shard_code_current")
            .expect("recorded")
            .is_fail(),
        "recorded as a failure, not silenced"
    );
}

#[test]
fn the_corpus_revision_is_checked_when_named_and_not_run_when_not() {
    let mut opts = fixture_open(Arc::new(OsFiles));
    let reader = ShardReader::open(&train_shards(), &manifest("train"), &opts).expect("open");
    assert!(reader.check("shard_rev_matches").expect("rev").is_pass());
    opts.expect_rev = Some("a-different-revision".to_owned());
    let err =
        ShardReader::open(&train_shards(), &manifest("train"), &opts).expect_err("rev mismatch");
    assert!(err.to_string().contains("a-different-revision"), "{err}");
    opts.expect_rev = None;
    let reader = ShardReader::open(&train_shards(), &manifest("train"), &opts).expect("open");
    assert!(
        matches!(
            reader.check("shard_rev_matches"),
            Some(TriState::NotRun { .. })
        ),
        "unchecked is not a pass"
    );
}

fn tampered_copy(tag: &str, edit: impl FnOnce(&Path)) -> PathBuf {
    let tmp = scratch(tag);
    let shards = tmp.join("train");
    copy_dir(&train_shards(), &shards);
    edit(&shards);
    shards
}

fn refused_without_token_bytes(shards: &Path, expect: &str) {
    let files = CountingFiles::new();
    let err = open_and_drain(shards, &manifest("train"), &fixture_open(files.clone()))
        .expect_err("a damaged shard set must be refused");
    assert_eq!(files.bytes_read(&shards.join(TOKENS_NAME)), 0, "{err}");
    assert!(
        err.to_string().contains(expect),
        "expected {expect:?} in: {err}"
    );
}

#[test]
fn a_hand_edited_header_fails_its_shard_hash() {
    let shards = tampered_copy("edited-header", |d| {
        let text = std::fs::read_to_string(d.join("header.json")).expect("read");
        std::fs::write(
            d.join("header.json"),
            text.replacen("\"vocab_size\": 256", "\"vocab_size\": 257", 1),
        )
        .expect("write");
    });
    refused_without_token_bytes(&shards, "shard_hash");
}

#[test]
fn a_truncated_token_file_is_refused_at_open() {
    let shards = tampered_copy("truncated", |d| {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(d.join(TOKENS_NAME))
            .expect("open");
        f.set_len(f.metadata().expect("meta").len() - 4)
            .expect("truncate");
    });
    refused_without_token_bytes(&shards, "truncated");
}

#[test]
fn a_deflated_supervision_archive_is_refused_loudly() {
    let shards = tampered_copy("deflated", |d| {
        std::fs::copy(
            door_fixture("supervision-deflated.npz"),
            d.join("supervision.npz"),
        )
        .expect("copy");
    });
    refused_without_token_bytes(&shards, "STORED");
}

#[test]
fn an_edited_sequence_index_fails_its_pin() {
    let shards = tampered_copy("seqidx", |d| {
        let text = std::fs::read_to_string(d.join("sequence_index.json")).expect("read");
        std::fs::write(
            d.join("sequence_index.json"),
            text.replacen("\"rows_in\"", "\"rows_in\" ", 1),
        )
        .expect("write");
    });
    refused_without_token_bytes(&shards, "sequence_index");
}

#[test]
fn a_remap_that_is_not_the_headers_is_refused() {
    let shards = tampered_copy("remap", |d| {
        let text = std::fs::read_to_string(d.join("remap.json")).expect("read");
        let mut v: serde_json::Value = serde_json::from_str(&text).expect("json");
        v["tokenizer_hash"] = serde_json::Value::from("another-tokenizer");
        std::fs::write(d.join("remap.json"), v.to_string()).expect("write");
    });
    refused_without_token_bytes(&shards, "remap_hash");
}

#[test]
fn a_missing_remap_reads_as_not_run_not_as_matching() {
    let shards = tampered_copy("no-remap", |d| {
        std::fs::remove_file(d.join("remap.npz")).expect("rm");
    });
    let reader = ShardReader::open(
        &shards,
        &manifest("train"),
        &fixture_open(Arc::new(OsFiles)),
    )
    .expect("open");
    assert!(matches!(
        reader.check("shard_remap_matches_header"),
        Some(TriState::NotRun { .. })
    ));
}

#[test]
fn a_missing_supervision_archive_is_refused_not_defaulted_to_cpt() {
    let shards = tampered_copy("no-supervision", |d| {
        std::fs::remove_file(d.join("supervision.npz")).expect("rm");
    });
    refused_without_token_bytes(&shards, "supervision");
}
