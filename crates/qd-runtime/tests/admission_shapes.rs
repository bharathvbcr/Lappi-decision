//! A release admits a family only in the slot shapes it trained (lane X1;
//! GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17;
//! `docs/caller-contract.md` section 4, `AUDIT/x1-admission-shape-2026-10-22/enumeration.md`).
//!
//! `release_manifest.json` records `trained_slot_shapes` (`{family: [[slot key, ...], ...]}`,
//! written by `qd-export`): per family, the sorted slot keys (`choice:N`, `score:N`, `span`, N
//! counting the reserved noul row) of every request shape its train manifest held. Admission
//! refuses a trained family's request whose own shape is not one of them, as
//! `slot_shape_not_trained`, before the model is asked. One test per refused shape class, plus the
//! controls that say an exactly-trained shape (in any slot order) is still answered.
//!
//! Every release here is synthetic and bound to the reference backend, whose calibration table
//! has an entry for every legal row count: so before this check existed, each refused shape below
//! was answered, and a test that failed for any other reason would not name `slot_shape_not_trained`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use qd_runtime::backend::DecisionBackend;
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::release::{MANIFEST_FILE, MANIFEST_FORMAT, Release};
use qd_runtime::render::{PROMPT_FORMAT, RenderCaps};
use qd_runtime::runtime::Runtime;
use qd_runtime::refusal::BackendError;
use qd_runtime::schema::{DecisionRequest, Response};
use qd_runtime::service::{RuntimeFactory, Service, ServiceConfig};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::{Value, json};

const DEFECT: &str = "code.defect_class";
const COMMIT: &str = "gitpulse.commit_type";
const DEFECT_CONTEXT: &str = "file: src/add.rs\n\n@@ -1 +1 @@\n-    a + b\n+    a - b\n";

static N: AtomicUsize = AtomicUsize::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "qd-shapes-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("a stale scratch directory is removable");
        }
        std::fs::create_dir_all(&dir).expect("the scratch directory is creatable");
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sha(bytes: &[u8]) -> String {
    qd_runtime::hex(&qd_runtime::sha256(bytes))
}

/// The shapes `code.defect_class` and `gitpulse.commit_type` trained, as the contract pins them.
fn trained_shapes() -> Value {
    json!({
        DEFECT: [["choice:5", "span"]],
        COMMIT: [["choice:12"]],
    })
}

/// A release bound to the reference backend. `families` and `shapes` are written as the
/// manifest's `trained_families` and `trained_slot_shapes`; `None` leaves a key out.
fn release_dir(tag: &str, families: Option<Value>, shapes: Option<Value>) -> Scratch {
    let dir = Scratch::new(tag);
    let reference = ReferenceBackend::new(true);
    let id = reference.identity();
    let config = br#"{"architectures":["Qwen3_5ForConditionalGeneration"]}"#;
    let table = CalibrationTable::reference();
    let calibration = serde_json::to_vec(&table).expect("the reference table serializes");
    let weights = b"not read by the reference backend";
    std::fs::write(dir.0.join("config.json"), config).unwrap();
    std::fs::write(dir.0.join("calibration.json"), &calibration).unwrap();
    std::fs::write(dir.0.join("model.safetensors"), weights).unwrap();
    let mut manifest = json!({
        "format": MANIFEST_FORMAT,
        "expected_identity": {
            "weight_hash": id.weight_hash,
            "tokenizer_hash": id.tokenizer_hash,
            "config_sha256": sha(config),
            "calibration_hash": table.hash(),
            "prompt_format": PROMPT_FORMAT,
        },
        "files": {
            "config.json": {"sha256": sha(config)},
            "model.safetensors": {"sha256": sha(weights)},
            "calibration.json": {"sha256": sha(&calibration)},
        },
        "calibration": {
            "file": "calibration.json",
            "file_sha256": sha(&calibration),
            "table_hash": table.hash(),
        },
    });
    if let Some(families) = families {
        manifest["trained_families"] = families;
    }
    if let Some(shapes) = shapes {
        manifest["trained_slot_shapes"] = shapes;
    }
    std::fs::write(
        dir.0.join(MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    dir
}

fn runtime(dir: &Path) -> Runtime {
    let release = Release::open(dir).expect("the synthetic release opens");
    Runtime::from_release(
        &release,
        Arc::new(ReferenceBackend::new(true)) as Arc<dyn DecisionBackend>,
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("the reference backend is the release's tower")
}

/// A release that trained both families in the contract's shapes.
fn contract_release(tag: &str) -> Scratch {
    release_dir(tag, Some(json!([DEFECT, COMMIT])), Some(trained_shapes()))
}

fn options(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("option{i}")).collect()
}

fn choice(name: &str, n: usize) -> Value {
    json!({"name": name, "type": "choice", "options": options(n)})
}

fn score(name: &str, bins: u32) -> Value {
    json!({"name": name, "type": "score", "bins": bins})
}

fn span(name: &str) -> Value {
    json!({"name": name, "type": "span"})
}

fn request(task: &str, context: &str, slots: Vec<Value>) -> DecisionRequest {
    let value = json!({
        "schema_version": 1, "task": task,
        "context_b64": qd_runtime::b64::encode(context.as_bytes()),
        "context_len": context.len(),
        "question": "What kind of change is this?",
        "slots": slots,
        "route": "generic",
    });
    match parse_line(&serde_json::to_vec(&value).expect("serializes")) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("the test request was not accepted on the wire: {other:?}"),
    }
}

#[track_caller]
fn refused(response: Response, kind: &str) -> Value {
    match response {
        Response::Refused(envelope) => {
            assert_eq!(envelope.refusal.kind(), kind, "{:?}", envelope.refusal);
            serde_json::to_value(&envelope.refusal).expect("serializes")
        }
        other => panic!("expected a `{kind}` refusal, got {other:?}"),
    }
}

#[track_caller]
fn answered(response: Response) {
    match response {
        Response::Ok(_) => {}
        other => panic!("expected an answer, got {other:?}"),
    }
}

#[test]
fn the_exactly_trained_shape_is_answered_in_any_slot_order() {
    let dir = contract_release("exact");
    let runtime = runtime(&dir.0);
    answered(runtime.answer(
        &request(
            DEFECT,
            DEFECT_CONTEXT,
            vec![choice("defect_class", 4), span("defect_span")],
        ),
        None,
    ));
    // The recorded shape is the sorted key multiset, so the order a caller lists its slots in is
    // not part of it.
    answered(runtime.answer(
        &request(
            DEFECT,
            DEFECT_CONTEXT,
            vec![span("defect_span"), choice("defect_class", 4)],
        ),
        None,
    ));
    answered(runtime.answer(&request(COMMIT, "diff", vec![choice("commit_type", 11)]), None));
}

#[test]
fn class_1_a_wrong_row_count_is_refused() {
    let dir = contract_release("rows");
    let refusal = refused(
        runtime(&dir.0).answer(&request(COMMIT, "diff", vec![choice("commit_type", 12)]), None),
        "slot_shape_not_trained",
    );
    // The refusal names the family, the shape sent, what was trained, and the contract section.
    assert_eq!(refusal["task"], json!(COMMIT));
    assert_eq!(refusal["shape"], json!(["choice:13"]));
    assert_eq!(refusal["trained"], json!([["choice:12"]]));
    let section = refusal["contract_section"].as_str().expect("a string");
    assert!(
        section.contains("docs/caller-contract.md") && section.contains('4'),
        "{section}"
    );
}

#[test]
fn class_2_a_wrong_slot_kind_is_refused() {
    let dir = contract_release("kind");
    let runtime = runtime(&dir.0);
    // `score:12` where `choice:12` was trained: the same row count, a different head.
    refused(
        runtime.answer(&request(COMMIT, "diff", vec![score("commit_type", 11)]), None),
        "slot_shape_not_trained",
    );
    // A span where the family trained a choice.
    refused(
        runtime.answer(&request(COMMIT, "diff", vec![span("commit_type")]), None),
        "slot_shape_not_trained",
    );
    // A choice where the family trained a span (the second slot).
    refused(
        runtime.answer(
            &request(
                DEFECT,
                DEFECT_CONTEXT,
                vec![choice("defect_class", 4), choice("defect_span", 4)],
            ),
            None,
        ),
        "slot_shape_not_trained",
    );
}

#[test]
fn class_3_a_missing_slot_is_refused() {
    let dir = contract_release("missing");
    let refusal = refused(
        runtime(&dir.0).answer(
            &request(DEFECT, DEFECT_CONTEXT, vec![choice("defect_class", 4)]),
            None,
        ),
        "slot_shape_not_trained",
    );
    assert_eq!(refusal["shape"], json!(["choice:5"]));
    assert_eq!(refusal["trained"], json!([["choice:5", "span"]]));
}

#[test]
fn class_4_an_extra_slot_is_refused() {
    let dir = contract_release("extra");
    let runtime = runtime(&dir.0);
    refused(
        runtime.answer(
            &request(
                DEFECT,
                DEFECT_CONTEXT,
                vec![choice("defect_class", 4), span("defect_span"), span("again")],
            ),
            None,
        ),
        "slot_shape_not_trained",
    );
    refused(
        runtime.answer(
            &request(
                COMMIT,
                "diff",
                vec![choice("commit_type", 11), choice("extra", 2)],
            ),
            None,
        ),
        "slot_shape_not_trained",
    );
}

#[test]
fn class_5_another_familys_shape_is_refused() {
    let dir = contract_release("cross");
    let runtime = runtime(&dir.0);
    // `gitpulse.commit_type`'s shape sent as `code.defect_class`, and the reverse.
    refused(
        runtime.answer(
            &request(DEFECT, DEFECT_CONTEXT, vec![choice("commit_type", 11)]),
            None,
        ),
        "slot_shape_not_trained",
    );
    refused(
        runtime.answer(
            &request(
                COMMIT,
                "diff",
                vec![choice("defect_class", 4), span("defect_span")],
            ),
            None,
        ),
        "slot_shape_not_trained",
    );
}

#[test]
fn an_untrained_task_is_still_refused_as_an_untrained_task_not_as_a_shape() {
    let dir = contract_release("task-first");
    let refusal = refused(
        runtime(&dir.0).answer(
            &request("devtype.route", "query", vec![choice("route", 11)]),
            None,
        ),
        "task_not_trained",
    );
    assert_eq!(refusal["task"], json!("devtype.route"));
}

#[test]
fn the_shape_is_refused_before_the_defect_context_is_read() {
    let dir = contract_release("shape-before-context");
    refused(
        runtime(&dir.0).answer(
            &request(DEFECT, "not a diff at all", vec![choice("defect_class", 3)]),
            None,
        ),
        "slot_shape_not_trained",
    );
    // With the shape trained, the same context reaches the context check.
    refused(
        runtime(&dir.0).answer(
            &request(
                DEFECT,
                "not a diff at all",
                vec![choice("defect_class", 4), span("defect_span")],
            ),
            None,
        ),
        "context_not_unified_diff",
    );
}

#[test]
fn a_family_may_train_several_shapes_and_each_is_admitted() {
    let dir = release_dir(
        "several",
        Some(json!([COMMIT])),
        Some(json!({COMMIT: [["choice:12"], ["choice:5"]]})),
    );
    let runtime = runtime(&dir.0);
    answered(runtime.answer(&request(COMMIT, "d", vec![choice("t", 11)]), None));
    answered(runtime.answer(&request(COMMIT, "d", vec![choice("t", 4)]), None));
    refused(
        runtime.answer(&request(COMMIT, "d", vec![choice("t", 6)]), None),
        "slot_shape_not_trained",
    );
}

#[test]
fn a_release_that_records_no_shapes_does_not_run_the_shape_check() {
    // `qd-export` before 2026-10-17 wrote no `trained_slot_shapes`. Such a release cannot say what
    // it trained, so the check does not run and says so (`Runtime::trained_slot_shapes`);
    // see `tests/admission_shapes_state.rs`. Its answers are unchanged.
    let dir = release_dir("unrecorded", Some(json!([COMMIT])), None);
    answered(runtime(&dir.0).answer(&request(COMMIT, "d", vec![choice("t", 3)]), None));
}

#[test]
fn a_malformed_or_incoherent_trained_slot_shapes_refuses_the_release_at_open() {
    let families = || Some(json!([DEFECT, COMMIT]));
    let cases: [(&str, Option<Value>, Value, &str); 11] = [
        ("not-object", families(), json!([["choice:12"]]), "not an object"),
        (
            "empty-family",
            families(),
            json!({COMMIT: [], DEFECT: [["choice:5", "span"]]}),
            "non-empty",
        ),
        (
            "empty-shape",
            families(),
            json!({COMMIT: [[]], DEFECT: [["choice:5", "span"]]}),
            "non-empty",
        ),
        (
            "bad-key",
            families(),
            json!({COMMIT: [["choice:05x"]], DEFECT: [["choice:5", "span"]]}),
            "choice:05x",
        ),
        (
            "non-canonical-row-count",
            families(),
            json!({COMMIT: [["choice:012"]], DEFECT: [["choice:5", "span"]]}),
            "choice:012",
        ),
        (
            "key-not-string",
            families(),
            json!({COMMIT: [[12]], DEFECT: [["choice:5", "span"]]}),
            "not a string",
        ),
        (
            "unsorted-key-set",
            families(),
            json!({COMMIT: [["choice:12"]], DEFECT: [["span", "choice:5"]]}),
            "sorted",
        ),
        (
            "family-not-trained",
            Some(json!([COMMIT])),
            json!({COMMIT: [["choice:12"]], DEFECT: [["choice:5", "span"]]}),
            DEFECT,
        ),
        (
            "trained-family-without-shapes",
            families(),
            json!({COMMIT: [["choice:12"]]}),
            DEFECT,
        ),
        (
            "shapes-without-families",
            None,
            json!({COMMIT: [["choice:12"]]}),
            "trained_families",
        ),
        (
            "duplicate-shape",
            families(),
            json!({COMMIT: [["choice:12"], ["choice:12"]], DEFECT: [["choice:5", "span"]]}),
            "twice",
        ),
    ];
    for (tag, families, shapes, needle) in cases {
        let dir = release_dir(tag, families, Some(shapes));
        let err = Release::open(&dir.0)
            .err()
            .unwrap_or_else(|| panic!("{tag}: a malformed trained_slot_shapes opened"));
        assert!(
            err.to_string().contains(needle),
            "{tag}: the refusal should name `{needle}`: {err}"
        );
    }
}

/// A service over `dir`'s release, as `qd-metal-serve` builds one (`Runtime::from_release`).
fn service_over(dir: &Path) -> Service {
    let dir = dir.to_path_buf();
    let factory: RuntimeFactory = Arc::new(move || {
        let release = Release::open(&dir).map_err(|e| BackendError::Unavailable {
            detail: e.to_string(),
        })?;
        Runtime::from_release(
            &release,
            Arc::new(ReferenceBackend::new(true)) as Arc<dyn DecisionBackend>,
            HeadRegistry::new(),
            RenderCaps::DEFAULT,
        )
    });
    Service::with_factory(ServiceConfig::default(), factory)
}

fn status_line(service: &Service) -> Value {
    let reply = service.handle_line(br#"{"op":"status"}"#);
    serde_json::from_slice(&reply).expect("the status reply is JSON")
}

#[test]
fn the_status_op_says_whether_the_shape_check_runs() {
    // Recorded: `checked`. Legacy (no `trained_slot_shapes`): `not_recorded`, never `checked`.
    let recorded = contract_release("status-recorded");
    let legacy = release_dir("status-legacy", Some(json!([COMMIT])), None);
    for (dir, want) in [(&recorded, "checked"), (&legacy, "not_recorded")] {
        let service = service_over(&dir.0);
        // Nothing is warm yet: no claim either way, not a default of either word.
        let cold = status_line(&service);
        assert_eq!(cold["status"], json!("status"), "{cold}");
        assert_eq!(cold["slot_shape_check"], Value::Null, "{want}: {cold}");
        let reply = service.handle_request(&request(COMMIT, "d", vec![choice("t", 11)]));
        answered(reply);
        let warm = status_line(&service);
        assert_eq!(warm["slot_shape_check"], json!(want), "{warm}");
    }
}

#[test]
fn a_legacy_release_serves_a_shape_the_recorded_release_refuses_and_status_tells_them_apart() {
    let recorded = contract_release("pair-recorded");
    let legacy = release_dir("pair-legacy", Some(json!([COMMIT])), None);
    let wrong = || request(COMMIT, "d", vec![choice("t", 3)]);
    let on_recorded = service_over(&recorded.0);
    let on_legacy = service_over(&legacy.0);
    refused(on_recorded.handle_request(&wrong()), "slot_shape_not_trained");
    answered(on_legacy.handle_request(&wrong()));
    assert_ne!(
        status_line(&on_recorded)["slot_shape_check"],
        status_line(&on_legacy)["slot_shape_check"]
    );
}

#[test]
fn a_runtime_with_no_release_reports_no_release_not_checked() {
    let runtime = Runtime::with_backend(
        Arc::new(ReferenceBackend::new(true)),
        CalibrationTable::reference(),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("assembles");
    assert_eq!(runtime.trained_slot_shapes().check_state(), "no_release");
    assert!(!runtime.trained_slot_shapes().is_checked());
}
