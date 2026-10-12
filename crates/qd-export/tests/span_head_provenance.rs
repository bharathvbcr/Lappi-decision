//! The release says where its span head came from, so the serving loader can refuse a blend.
//!
//! `tools/ckpt_average.py` writes `span_head_source` into its manifest (`mode` `"seed"` with the
//! seed, input and payload digest, or `"averaged"`). The export copies it into the release
//! manifest's `span_head` block, records the averaged-head pre-registration by digest when one is
//! given (`campaign/v6-averaging-preregistered.json`), and states the serving contract the head was
//! trained under. A manifest that states a source in any other shape is refused.

mod common;

use common::{build, standard_tensors, tiny_config, Fixture};
use qd_export::export::SPAN_CONTRACT_VERSION;
use qd_export::{export, RefusalKind};
use serde_json::{json, Value};

const DIGEST: &str = "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f";

fn with_source(source: Option<Value>) -> Fixture {
    build(&standard_tensors(), &tiny_config(), |m| {
        if let Some(s) = source {
            m["span_head_source"] = s;
        }
    })
}

fn release_manifest(fx: &Fixture) -> Value {
    serde_json::from_slice(&std::fs::read(fx.out.join("release_manifest.json")).unwrap()).unwrap()
}

fn prereg(fx: &Fixture, body: Value) -> std::path::PathBuf {
    let p = fx.dir.0.join("prereg.json");
    std::fs::write(&p, serde_json::to_string_pretty(&body).unwrap()).unwrap();
    p
}

fn averaged() -> Value {
    json!({"mode": "averaged", "seed": null, "input": null, "payload_digest": null})
}

fn the_registration() -> Value {
    json!({"name": "v6-averaging", "rows": {"primary": {
        "manifest_must_say": "span_head_source.mode == 'averaged'"}}})
}

#[test]
fn a_seed_heads_source_is_recorded_with_the_serving_contract_and_the_heads_digest() {
    let src = json!({"mode": "seed", "seed": 0, "input": "seed0.ckpt", "payload_digest": DIGEST});
    let fx = with_source(Some(src.clone()));
    let summary = export(&fx.request()).unwrap();
    let m = release_manifest(&fx);
    let block = &m["span_head"];
    assert_eq!(block["source"], src);
    assert!(block["averaged_head_preregistration"].is_null());
    assert_eq!(block["file"], "span_head.safetensors");
    assert_eq!(block["sha256"].as_str(), summary.files.get("span_head.safetensors").map(String::as_str));
    assert_eq!(block["hidden_size"], common::HIDDEN);
    assert_eq!(block["serving_contract"]["version"], SPAN_CONTRACT_VERSION);
    assert!(block["serving_contract"]["hidden_state"].as_str().unwrap().contains("final norm"));
    assert!(block["serving_contract"]["abstain_row"].as_str().unwrap().starts_with("last"));
}

#[test]
fn a_source_that_states_none_is_recorded_as_none_not_guessed() {
    let fx = with_source(None);
    export(&fx.request()).unwrap();
    assert!(release_manifest(&fx)["span_head"]["source"].is_null(), "unknown provenance stays unknown");
}

/// The committed campaign file, which is the only pre-registration the export accepts.
fn committed_prereg() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../campaign/v6-averaging-preregistered.json")
}

/// An ft ledger with one completed ft row per fixture seed, edited by `edit(seed, row)`.
fn ledger(fx: &Fixture, edit: impl Fn(usize, &mut Value)) -> std::path::PathBuf {
    let lines: Vec<String> = common::FT_ROW_IDS
        .iter()
        .enumerate()
        .map(|(seed, id)| {
            let mut row = json!({
                "row_id": id, "run_kind": "ft", "status": "completed", "protocol": {"seed": seed},
                "recipe": {"span_head_init": {"sha256": "a".repeat(64), "content_digest": "c0".repeat(32), "shared": true}},
            });
            edit(seed, &mut row);
            row.to_string()
        })
        .collect();
    let p = fx.dir.0.join("ft-ledger.jsonl");
    std::fs::write(&p, lines.join("\n") + "\n").unwrap();
    p
}

fn averaged_request(fx: &Fixture, ledger_edit: impl Fn(usize, &mut Value)) -> qd_export::ExportRequest {
    let mut req = fx.request();
    req.averaged_head_preregistration = Some(committed_prereg());
    req.ft_ledger = Some(ledger(fx, ledger_edit));
    req
}

fn averaged_with_inputs() -> Fixture {
    build(&standard_tensors(), &tiny_config(), |m| {
        m["span_head_source"] = averaged();
        m["inputs"] = json!([{"seed": 0}, {"seed": 1}, {"seed": 2}]);
    })
}

#[test]
fn the_pinned_preregistration_digest_is_the_committed_campaign_files() {
    let bytes = std::fs::read(committed_prereg()).unwrap();
    assert_eq!(
        qd_runtime::hex(&qd_runtime::sha256(&bytes)),
        qd_export::export::KNOWN_AVERAGED_PREREGISTRATIONS[0].1,
        "campaign/v6-averaging-preregistered.json changed: the pin (and every release citing it) is stale"
    );
}

#[test]
fn the_averaged_head_carries_its_preregistration_and_the_shared_init_verified_from_the_ledger() {
    let fx = with_source(Some(averaged()));
    export(&fx.request()).unwrap();
    let m = release_manifest(&fx);
    assert!(m["span_head"]["averaged_head_preregistration"].is_null());
    assert!(m["span_head"]["shared_init"].is_null());

    let fx = averaged_with_inputs();
    export(&averaged_request(&fx, |_, _| {})).unwrap();
    let block = &release_manifest(&fx)["span_head"];
    let rec = &block["averaged_head_preregistration"];
    assert_eq!((rec["name"].as_str(), rec["row"].as_str()), (Some("v6-averaging"), Some("primary")));
    assert_eq!(rec["sha256"].as_str(), Some(qd_export::export::KNOWN_AVERAGED_PREREGISTRATIONS[0].1));
    let init = &block["shared_init"];
    assert_eq!(init["shared"], true);
    assert_eq!(init["seeds"], json!([0, 1, 2]));
    assert_eq!(init["ft_row_ids"], json!(common::FT_ROW_IDS));
    assert_eq!(init["content_digest"], "c0".repeat(32));
}

#[test]
fn an_averaged_head_is_refused_without_ledger_evidence_that_every_seed_shared_one_init() {
    type LedgerEdit = Box<dyn Fn(usize, &mut Value)>;
    let cases: Vec<(&str, LedgerEdit, &str)> = vec![
        ("not shared", Box::new(|s, r| if s == 1 { r["recipe"]["span_head_init"]["shared"] = false.into() }), "not the shared one"),
        ("different init", Box::new(|s, r| if s == 2 { r["recipe"]["span_head_init"]["content_digest"] = "d1".repeat(32).into() }), "different span-head init"),
        ("no init", Box::new(|s, r| if s == 0 { r["recipe"].as_object_mut().unwrap().remove("span_head_init"); }), "no recipe.span_head_init"),
        ("not completed", Box::new(|s, r| if s == 0 { r["status"] = "failed".into() }), "not a completed ft row"),
        ("repeated seed", Box::new(|s, r| if s == 2 { r["protocol"]["seed"] = 1.into() }), "not distinct"),
        ("another seed", Box::new(|s, r| if s == 2 { r["protocol"]["seed"] = 9.into() }), "not the source manifest's inputs"),
    ];
    for (what, edit, needle) in cases {
        let fx = averaged_with_inputs();
        let err = export(&averaged_request(&fx, &*edit)).expect_err(what);
        assert_eq!(err.kind, RefusalKind::SourceManifest, "{what}: {err}");
        assert!(err.detail.contains(needle), "{what}: expected {needle:?} in {err}");
        assert!(!fx.out.exists(), "{what}");
    }
    // No ledger given, and a ledger with a row missing.
    let fx = averaged_with_inputs();
    let mut req = averaged_request(&fx, |_, _| {});
    req.ft_ledger = None;
    assert!(export(&req).unwrap_err().detail.contains("--ft-ledger"));
    let fx = build(&standard_tensors(), &tiny_config(), |m| {
        m["span_head_source"] = averaged();
        m["ft_row_ids"] = json!(["row-seed0", "row-seed1", "row-missing"]);
    });
    assert!(export(&averaged_request(&fx, |_, _| {})).unwrap_err().detail.contains("0 rows have row_id"));
}

#[test]
fn a_file_that_only_says_the_right_sentence_is_not_the_committed_preregistration() {
    let fx = averaged_with_inputs();
    let mut req = averaged_request(&fx, |_, _| {});
    req.averaged_head_preregistration = Some(prereg(&fx, the_registration()));
    let err = export(&req).unwrap_err();
    assert_eq!(err.kind, RefusalKind::SourceManifest);
    assert!(err.detail.contains("not a committed averaged-head pre-registration"), "{err}");
}

#[test]
fn a_preregistration_cannot_cover_a_head_that_is_not_the_averaged_one_or_does_not_say_so() {
    // A seed's head with a pre-registration of the averaged row: it would call a seed an average.
    let fx = with_source(Some(json!({"mode": "seed", "seed": 1, "input": "s1", "payload_digest": DIGEST})));
    let mut req = fx.request();
    req.averaged_head_preregistration = Some(committed_prereg());
    let err = export(&req).unwrap_err();
    assert_eq!(err.kind, RefusalKind::SourceManifest, "{err}");
    assert!(err.detail.contains("cannot cover another head"), "{err}");
    assert!(!fx.out.exists());

    // No source at all: the same.
    let fx = with_source(None);
    let mut req = fx.request();
    req.averaged_head_preregistration = Some(prereg(&fx, the_registration()));
    assert!(export(&req).unwrap_err().detail.contains("cannot cover another head"));

    // A file that does not pre-register the averaged primary row.
    for body in [
        json!({"name": "x", "rows": {"primary": {"manifest_must_say": "span_head_source.mode == 'seed'"}}}),
        json!({"name": "x", "rows": {}}),
        json!({"name": " ", "rows": {"primary": {"manifest_must_say": "span_head_source.mode == 'averaged'"}}}),
        json!([1, 2]),
    ] {
        let fx = with_source(Some(averaged()));
        let mut req = fx.request();
        req.averaged_head_preregistration = Some(prereg(&fx, body.clone()));
        let err = export(&req).unwrap_err();
        assert_eq!(err.kind, RefusalKind::SourceManifest, "{body}: {err}");
        assert!(fx.leftovers().is_empty());
    }
}

#[test]
fn a_span_head_source_in_the_wrong_shape_is_refused() {
    let bad = [
        json!("averaged"),
        json!({"mode": "blend"}),
        json!({"mode": "averaged", "seed": 0, "input": null, "payload_digest": null}),
        json!({"mode": "seed", "seed": 0, "input": "a", "payload_digest": "short"}),
        json!({"mode": "seed", "seed": -1, "input": "a", "payload_digest": DIGEST}),
        json!({"mode": "seed", "seed": 0, "input": " ", "payload_digest": DIGEST}),
        json!({"mode": "seed", "input": "a", "payload_digest": DIGEST}),
    ];
    for source in bad {
        let fx = with_source(Some(source.clone()));
        let err = export(&fx.request()).unwrap_err();
        assert_eq!(err.kind, RefusalKind::SourceManifest, "{source}: {err}");
        assert!(err.detail.contains("span_head_source"), "{source}: {err}");
        assert!(!fx.out.exists(), "{source}: nothing published");
    }
}
