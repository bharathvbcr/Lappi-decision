//! The release `qd-export` writes, opened by the runtime's own reader (`qd_runtime::release`).
//!
//! The writer and the reader are separate code in separate crates; this file is where they meet.
//! Each test exports the standard fixture with the real exporter and then opens, tampers with, or
//! binds a backend to the result.

mod common;

use std::path::Path;
use std::sync::Arc;

use qd_export::{ExportSummary, export};
use qd_runtime::backend::{
    BackendIdentity, DecisionBackend, DecodeMode, Logits, PrefillHandle, SlotQuery, StateSnapshot,
};
use qd_runtime::calibration::{CalibrationEntry, CalibrationTable};
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::refusal::BackendError;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::release::{
    CALIBRATION_FILE, CONFIG_FILE, MANIFEST_FILE, Release, ReleaseRefusalKind,
};
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{Response, SlotKind};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::{Value, json};

/// Not the reference table, so a test can tell which one is serving.
fn fitted_table() -> CalibrationTable {
    CalibrationTable::new("release-under-test")
        .with_letters(
            SlotKind::Choice,
            5,
            CalibrationEntry {
                temperature: 1.37,
                conformal_quantile: 0.11,
                noul_margin: 0.03,
            },
        )
        .with_span(CalibrationEntry {
            temperature: 1.0,
            conformal_quantile: 0.0,
            noul_margin: 0.0,
        })
}

/// The prompt format a v5 checkpoint's source manifest states (`<avg>.manifest.json`
/// `prompt_format`), and the one this runtime serves. Written as the wire number rather than
/// `render::PROMPT_FORMAT`, so this file pins the contract and not the constant.
const V5_PROMPT_FORMAT: u64 = 2;

/// The standard fixture, its source manifest stating `prompt_format` (nothing when `None`).
fn fixture_of_format(prompt_format: Option<Value>) -> common::Fixture {
    common::build(&common::standard_tensors(), &common::tiny_config(), |m| {
        if let Some(format) = prompt_format {
            m["prompt_format"] = format;
        }
    })
}

fn release_manifest(release: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(release.join(MANIFEST_FILE)).unwrap()).unwrap()
}

/// Export the standard fixture of a format-2 checkpoint, with `table` passed through as
/// pretty-printed JSON (so the file's sha256 is not the table's hash, as it need not be). Every
/// release the tests below open is checked here to be stamped with the source's format.
fn export_with(table: Option<&CalibrationTable>) -> (common::Fixture, ExportSummary) {
    let fx = fixture_of_format(Some(json!(V5_PROMPT_FORMAT)));
    let mut req = fx.request();
    if let Some(table) = table {
        let path = fx.dir.0.join("table.json");
        std::fs::write(&path, serde_json::to_vec_pretty(table).unwrap()).unwrap();
        req.calibration = Some(path);
    }
    let summary = export(&req).expect("the standard fixture exports");
    let manifest = release_manifest(&fx.out);
    assert_eq!(manifest["format"], json!("qd-release.v2"));
    assert_eq!(
        manifest["expected_identity"]["prompt_format"],
        json!(V5_PROMPT_FORMAT)
    );
    (fx, summary)
}

fn exported() -> (common::Fixture, ExportSummary) {
    export_with(Some(&fitted_table()))
}

fn identity(weight_hash: &str, tokenizer_hash: &str, calibration_hash: &str) -> BackendIdentity {
    BackendIdentity {
        name: "binding-under-test".to_string(),
        tokenizer_hash: tokenizer_hash.to_string(),
        weight_hash: weight_hash.to_string(),
        head_hash: "h".repeat(64),
        label_set_hash: "l".repeat(64),
        calibration_hash: calibration_hash.to_string(),
        is_model: true,
        state_host_visible: true,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    qd_runtime::hex(&qd_runtime::sha256(bytes))
}

fn rewrite_manifest(release: &Path, edit: impl FnOnce(&mut Value)) {
    let path = release.join(MANIFEST_FILE);
    let mut doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    edit(&mut doc);
    std::fs::write(&path, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();
}

/// Replace `calibration.json` and re-record its file sha256 in both manifest fields, leaving the
/// table hash the manifest binds alone: the file is now "the exported file", but not the table.
fn replace_calibration_and_its_file_hashes(release: &Path, bytes: &[u8]) {
    std::fs::write(release.join(CALIBRATION_FILE), bytes).unwrap();
    let sha = sha256_hex(bytes);
    rewrite_manifest(release, |m| {
        m["files"][CALIBRATION_FILE]["sha256"] = Value::from(sha.clone());
        m["calibration"]["file_sha256"] = Value::from(sha);
    });
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

// -- the prompt format ---------------------------------------------------------------------------

/// A checkpoint averaged before the format existed states none, so it is stamped 1 -- what it was
/// trained on -- and this runtime refuses it. Refused, not mislabelled: stamping the runtime's own
/// format here would serve format-2 prompts to a format-1 tower and every check would pass.
#[test]
fn a_checkpoint_that_states_no_prompt_format_is_stamped_1_and_refused_not_mislabelled() {
    let fx = fixture_of_format(None);
    let mut req = fx.request();
    let path = fx.dir.0.join("table.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&fitted_table()).unwrap()).unwrap();
    req.calibration = Some(path);
    export(&req).expect("a v4 checkpoint still exports");
    let manifest = release_manifest(&fx.out);
    assert_eq!(manifest["format"], json!("qd-release.v2"));
    assert_eq!(manifest["expected_identity"]["prompt_format"], json!(1));
    assert_eq!(manifest["source"]["prompt_format"], json!(1));

    let err = Release::open(&fx.out).expect_err("a format-1 tower is not served format-2 prompts");
    assert_eq!(err.kind.as_str(), "prompt_format", "{err}");
    assert!(err.detail.contains("prompt_format 1"), "{err}");
}

/// The stamp is read from the source manifest, never from the exporter's own constant: a source
/// stating 7 is stamped 7 (and then refused by this runtime, which serves format 2).
#[test]
fn the_stamp_is_the_source_manifests_never_the_runtimes_constant() {
    let fx = fixture_of_format(Some(json!(7)));
    export(&fx.request()).expect("exports");
    let manifest = release_manifest(&fx.out);
    assert_eq!(manifest["expected_identity"]["prompt_format"], json!(7));
    assert_eq!(manifest["source"]["prompt_format"], json!(7));
    let err = Release::open(&fx.out).expect_err("format 7 is not this runtime's");
    assert_eq!(err.kind.as_str(), "prompt_format", "{err}");
    assert!(err.detail.contains("prompt_format 7"), "{err}");
}

#[test]
fn a_source_prompt_format_that_is_not_a_positive_integer_is_refused_before_anything_is_written() {
    for bad in [json!("2"), json!(2.0), json!(true), json!(null), json!(0), json!(-1), json!([2])] {
        let fx = fixture_of_format(Some(bad.clone()));
        match export(&fx.request()) {
            Err(err) => {
                assert_eq!(err.kind, qd_export::RefusalKind::SourceManifest, "{bad}: {err}");
                assert!(err.detail.contains("prompt_format"), "{bad}: {err}");
            }
            Ok(_) => panic!("a source manifest stating prompt_format {bad} was exported"),
        }
        assert!(!fx.out.exists(), "{bad}: a refused export left {} behind", fx.out.display());
        assert!(fx.leftovers().is_empty(), "{bad}: staging left behind");
    }
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
    let table = fitted_table().hash();
    release
        .check_backend(&identity(
            &summary.weight_hash,
            &summary.tokenizer_hash,
            &table,
        ))
        .expect("the bound tower, tokenizer and table");

    let err = release
        .check_backend(&identity(&"0".repeat(64), &summary.tokenizer_hash, &table))
        .expect_err("another tower");
    assert_eq!(err.kind, ReleaseRefusalKind::IdentityMismatch, "{err}");
    assert!(err.detail.contains("weight_hash"), "{err}");

    let err = release
        .check_backend(&identity(&summary.weight_hash, &"1".repeat(64), &table))
        .expect_err("another tokenizer");
    assert_eq!(err.kind, ReleaseRefusalKind::IdentityMismatch, "{err}");
    assert!(err.detail.contains("tokenizer_hash"), "{err}");
}

// -- the calibration table (G9(c)) ---------------------------------------------------------------

#[test]
fn the_loaded_table_is_the_exported_table_under_the_hash_the_manifest_binds() {
    let (fx, summary) = exported();
    let release = Release::open(&fx.out).unwrap();
    let table = fitted_table();
    assert_eq!(release.calibration(), &table);
    assert_eq!(summary.calibration_hash, Some(table.hash()));
    // Pretty-printed on purpose: the file's bytes and the table's hash are two quantities.
    assert_ne!(summary.files[CALIBRATION_FILE], table.hash());
}

#[test]
fn a_release_exported_without_a_calibration_table_is_refused_at_load() {
    let (fx, summary) = export_with(None);
    assert_eq!(summary.calibration_hash, None);
    let err = Release::open(&fx.out).expect_err("no table, no load");
    assert_eq!(err.kind, ReleaseRefusalKind::CalibrationMissing, "{err}");
    assert!(err.detail.contains("reference table"), "{err}");

    // The manifest records a table, but the file is gone.
    let (fx, _) = exported();
    std::fs::remove_file(fx.out.join(CALIBRATION_FILE)).unwrap();
    let err = Release::open(&fx.out).expect_err("the recorded file is absent");
    assert_eq!(err.kind, ReleaseRefusalKind::CalibrationMissing, "{err}");
}

#[test]
fn a_calibration_file_that_is_not_the_exported_one_is_refused() {
    let (fx, _) = exported();
    let other = serde_json::to_vec(&CalibrationTable::reference()).unwrap();
    std::fs::write(fx.out.join(CALIBRATION_FILE), &other).unwrap();
    let err = Release::open(&fx.out).expect_err("another table's bytes");
    assert_eq!(err.kind, ReleaseRefusalKind::CalibrationMismatch, "{err}");
    assert!(
        err.detail.contains("not the table that was exported"),
        "{err}"
    );
}

#[test]
fn a_table_whose_hash_is_not_the_bound_one_is_refused_even_when_its_file_hash_was_rerecorded() {
    let (fx, _) = exported();
    let other = serde_json::to_vec(&CalibrationTable::reference()).unwrap();
    replace_calibration_and_its_file_hashes(&fx.out, &other);
    let err = Release::open(&fx.out).expect_err("the reference table is not the bound one");
    assert_eq!(err.kind, ReleaseRefusalKind::CalibrationMismatch, "{err}");
    assert!(
        err.detail.contains(&CalibrationTable::reference().hash()),
        "{err}"
    );
    assert!(err.detail.contains(&fitted_table().hash()), "{err}");
}

#[test]
fn a_calibration_file_that_is_not_a_usable_table_is_refused() {
    let (fx, _) = exported();
    replace_calibration_and_its_file_hashes(&fx.out, br#"{"name": 3}"#);
    let err = Release::open(&fx.out).expect_err("not a CalibrationTable");
    assert_eq!(err.kind, ReleaseRefusalKind::CalibrationInvalid, "{err}");

    let (fx, _) = exported();
    let mut invalid = serde_json::to_value(fitted_table()).unwrap();
    invalid["span"]["temperature"] = json!(0.0);
    replace_calibration_and_its_file_hashes(&fx.out, invalid.to_string().as_bytes());
    let err = Release::open(&fx.out).expect_err("a temperature of zero");
    assert_eq!(err.kind, ReleaseRefusalKind::CalibrationInvalid, "{err}");
    assert!(err.detail.contains("temperature"), "{err}");
}

#[test]
fn a_manifest_whose_calibration_hashes_disagree_with_each_other_is_refused() {
    let (fx, _) = exported();
    rewrite_manifest(&fx.out, |m| {
        m["expected_identity"]["calibration_hash"] = Value::from("cd".repeat(32));
    });
    let err = Release::open(&fx.out).expect_err("identity vs calibration block");
    assert_eq!(err.kind, ReleaseRefusalKind::Manifest, "{err}");
    assert!(err.detail.contains("disagrees with itself"), "{err}");

    let (fx, _) = exported();
    rewrite_manifest(&fx.out, |m| {
        m["calibration"]["file_sha256"] = Value::from("ef".repeat(32));
    });
    let err = Release::open(&fx.out).expect_err("files vs calibration block");
    assert_eq!(err.kind, ReleaseRefusalKind::Manifest, "{err}");
    assert!(err.detail.contains("disagrees with itself"), "{err}");
}

/// The reference backend's behaviour under the identity a backend loaded from the release would
/// report. Only `identity` differs; the name is the inner backend's, so its handles stay its own.
struct LoadedFrom {
    inner: ReferenceBackend,
    identity: BackendIdentity,
}

impl LoadedFrom {
    fn release(summary: &ExportSummary, calibration_hash: &str) -> Self {
        let inner = ReferenceBackend::new(true);
        let mut identity = inner.identity().clone();
        identity.weight_hash = summary.weight_hash.clone();
        identity.tokenizer_hash = summary.tokenizer_hash.clone();
        identity.calibration_hash = calibration_hash.to_string();
        Self { inner, identity }
    }
}

impl DecisionBackend for LoadedFrom {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        self.inner.prefill(prefix)
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        self.inner.snapshot(handle)
    }

    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        self.inner.decode_slot(snapshot, query, mode)
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.inner.pooled_features(snapshot)
    }
}

fn four_option_request(expect_calibration: &str) -> qd_runtime::schema::DecisionRequest {
    let context = b"fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";
    let value = json!({
        "schema_version": 1, "task": "devcouncil.verdict",
        "context_b64": qd_runtime::b64::encode(context), "context_len": context.len(),
        "question": "Does this diff implement what the commit message claims?",
        "slots": [{"name": "verdict", "type": "choice",
                   "options": ["stub", "logic", "cosmetic", "clean"]}],
        "route": "generic",
        "expect": {"calibration_hash": expect_calibration},
    });
    match parse_line(&serde_json::to_vec(&value).unwrap()) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("the test request was not accepted: {other:?}"),
    }
}

#[test]
fn a_runtime_built_from_the_release_calibrates_with_the_release_table_and_pins_it() {
    let (fx, summary) = exported();
    let release = Release::open(&fx.out).unwrap();
    let table = fitted_table();
    let runtime = Runtime::from_release(
        &release,
        Arc::new(LoadedFrom::release(&summary, &table.hash())),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("a backend reporting the release's identity");
    assert_eq!(runtime.calibration(), &table);
    match runtime.answer(&four_option_request(&table.hash()), None) {
        Response::Ok(_) => {}
        other => panic!("the release's own table, pinned, must be answered: {other:?}"),
    }
    match runtime.answer(
        &four_option_request(&CalibrationTable::reference().hash()),
        None,
    ) {
        Response::Refused(envelope) => assert_eq!(envelope.refusal.kind(), "hash_mismatch"),
        other => panic!("a pin of a table that is not serving must be refused: {other:?}"),
    }
}

#[test]
fn a_backend_declaring_another_calibration_table_is_not_built_into_a_runtime() {
    let (fx, summary) = exported();
    let release = Release::open(&fx.out).unwrap();
    let declared = CalibrationTable::reference().hash();

    let err = release
        .check_backend(&identity(
            &summary.weight_hash,
            &summary.tokenizer_hash,
            &declared,
        ))
        .expect_err("the backend declares the reference table");
    assert_eq!(err.kind, ReleaseRefusalKind::IdentityMismatch, "{err}");
    assert!(err.detail.contains("calibration_hash"), "{err}");

    match Runtime::from_release(
        &release,
        Arc::new(LoadedFrom::release(&summary, &declared)),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    ) {
        Err(BackendError::Unavailable { detail }) => {
            assert!(detail.contains("identity_mismatch"), "{detail}");
            assert!(detail.contains("calibration_hash"), "{detail}");
        }
        Err(other) => panic!("expected unavailable, got {other:?}"),
        Ok(_) => panic!("a runtime was built around a backend declaring another table"),
    }
}
