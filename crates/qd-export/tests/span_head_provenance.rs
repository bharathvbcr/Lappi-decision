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

#[test]
fn the_averaged_head_carries_its_preregistration_by_digest_or_none() {
    let fx = with_source(Some(averaged()));
    export(&fx.request()).unwrap();
    assert!(release_manifest(&fx)["span_head"]["averaged_head_preregistration"].is_null());

    let fx = with_source(Some(averaged()));
    let mut req = fx.request();
    let path = prereg(&fx, the_registration());
    req.averaged_head_preregistration = Some(path.clone());
    export(&req).unwrap();
    let rec = &release_manifest(&fx)["span_head"]["averaged_head_preregistration"];
    assert_eq!(rec["name"], "v6-averaging");
    assert_eq!(rec["row"], "primary");
    assert_eq!(rec["file"], "prereg.json");
    assert_eq!(rec["sha256"].as_str().unwrap(), qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(path).unwrap())));
}

#[test]
fn a_preregistration_cannot_cover_a_head_that_is_not_the_averaged_one_or_does_not_say_so() {
    // A seed's head with a pre-registration of the averaged row: it would call a seed an average.
    let fx = with_source(Some(json!({"mode": "seed", "seed": 1, "input": "s1", "payload_digest": DIGEST})));
    let mut req = fx.request();
    req.averaged_head_preregistration = Some(prereg(&fx, the_registration()));
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
