//! The release `qd-export` writes, opened by the runtime's own reader (`qd_runtime::release`).
//!
//! The writer and the reader are separate code in separate crates; this file is where they meet.
//! Each test exports the standard fixture with the real exporter and then opens, tampers with, or
//! binds a backend to the result.

mod common;

use qd_export::{ExportSummary, export};
use qd_runtime::backend::BackendIdentity;
use qd_runtime::release::{CONFIG_FILE, MANIFEST_FILE, Release, ReleaseRefusalKind};
use serde_json::Value;

fn exported() -> (common::Fixture, ExportSummary) {
    let fx = common::standard();
    let summary = export(&fx.request()).expect("the standard fixture exports");
    (fx, summary)
}

fn identity(weight_hash: &str, tokenizer_hash: &str) -> BackendIdentity {
    BackendIdentity {
        name: "binding-under-test".to_string(),
        tokenizer_hash: tokenizer_hash.to_string(),
        weight_hash: weight_hash.to_string(),
        head_hash: "h".repeat(64),
        label_set_hash: "l".repeat(64),
        calibration_hash: "c".repeat(64),
        is_model: true,
        state_host_visible: true,
    }
}

fn rewrite_manifest(release: &std::path::Path, edit: impl FnOnce(&mut Value)) {
    let path = release.join(MANIFEST_FILE);
    let mut doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    edit(&mut doc);
    std::fs::write(&path, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();
}

#[test]
fn the_manifest_binds_config_json_to_the_tower_and_the_reader_accepts_the_release() {
    let (fx, summary) = exported();
    let release = Release::open(&fx.out).expect("an untouched release opens");
    let config_sha = summary.files[CONFIG_FILE].clone();
    assert_eq!(release.config_sha256(), config_sha);
    assert_eq!(release.weight_hash(), summary.weight_hash);
    assert_eq!(release.tokenizer_hash(), summary.tokenizer_hash);

    let manifest: Value =
        serde_json::from_slice(&std::fs::read(fx.out.join(MANIFEST_FILE)).unwrap()).unwrap();
    let bound = &manifest["expected_identity"];
    assert_eq!(bound["config_sha256"], Value::from(config_sha));
    assert_eq!(
        bound["weight_hash"],
        Value::from(summary.weight_hash.clone())
    );
}

#[test]
fn a_same_shaped_config_from_another_revision_is_refused_at_load() {
    let (fx, _) = exported();
    let mut other = common::tiny_config();
    other["text_config"]["rope_parameters"]["rope_theta"] = 1_000_000.into();
    let path = fx.out.join(CONFIG_FILE);
    std::fs::write(&path, serde_json::to_string_pretty(&other).unwrap()).unwrap();

    // The hole this closes: the serving loader alone reads the swapped config without complaint,
    // because every shape it checks is unchanged.
    let loaded = qd_metal::config::ModelConfig::load(&fx.out)
        .expect("qd-metal's own config check accepts a same-shaped config");
    assert_eq!(loaded.rope_theta, 1_000_000.0);

    let err = Release::open(&fx.out).expect_err("a config the tower was not exported with");
    assert_eq!(err.kind, ReleaseRefusalKind::ConfigMismatch, "{err}");
    assert!(err.detail.contains("binds the tower"), "{err}");
}

#[test]
fn a_manifest_that_does_not_bind_the_config_or_disagrees_with_itself_is_refused() {
    // A manifest written before the binding existed.
    let (fx, _) = exported();
    rewrite_manifest(&fx.out, |m| {
        m["expected_identity"]
            .as_object_mut()
            .unwrap()
            .remove("config_sha256");
    });
    let err = Release::open(&fx.out).expect_err("no binding, no load");
    assert_eq!(err.kind, ReleaseRefusalKind::Manifest, "{err}");
    assert!(err.detail.contains("predates the binding"), "{err}");

    // The binding moved to another config while `files` still records the exported one.
    let (fx, _) = exported();
    rewrite_manifest(&fx.out, |m| {
        m["expected_identity"]["config_sha256"] = Value::from("ab".repeat(32));
    });
    let err = Release::open(&fx.out).expect_err("the manifest disagrees with itself");
    assert_eq!(err.kind, ReleaseRefusalKind::Manifest, "{err}");
    assert!(err.detail.contains("disagrees with itself"), "{err}");

    // Another format is not interpreted.
    let (fx, _) = exported();
    rewrite_manifest(&fx.out, |m| m["format"] = Value::from("qd-release.v0"));
    let err = Release::open(&fx.out).expect_err("unknown format");
    assert_eq!(err.kind, ReleaseRefusalKind::Manifest, "{err}");

    // No manifest at all.
    let (fx, _) = exported();
    std::fs::remove_file(fx.out.join(MANIFEST_FILE)).unwrap();
    let err = Release::open(&fx.out).expect_err("no manifest");
    assert_eq!(err.kind, ReleaseRefusalKind::Manifest, "{err}");
}

#[test]
fn a_backend_that_loaded_another_tower_or_tokenizer_is_refused() {
    let (fx, summary) = exported();
    let release = Release::open(&fx.out).unwrap();
    release
        .check_backend(&identity(&summary.weight_hash, &summary.tokenizer_hash))
        .expect("the bound tower and tokenizer");

    let err = release
        .check_backend(&identity(&"0".repeat(64), &summary.tokenizer_hash))
        .expect_err("another tower");
    assert_eq!(err.kind, ReleaseRefusalKind::IdentityMismatch, "{err}");
    assert!(err.detail.contains("weight_hash"), "{err}");

    let err = release
        .check_backend(&identity(&summary.weight_hash, &"1".repeat(64)))
        .expect_err("another tokenizer");
    assert_eq!(err.kind, ReleaseRefusalKind::IdentityMismatch, "{err}");
    assert!(err.detail.contains("tokenizer_hash"), "{err}");
}
