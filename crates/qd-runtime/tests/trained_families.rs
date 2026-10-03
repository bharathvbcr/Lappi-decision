//! A release admits only the task families it was trained on (Fable's pipeline ruling, item 4,
//! `AUDIT/finalize-2026-10-03/fable-pipeline-ruling.md`; GAP-RUNTIME-ADMITS-TASKS-NO-RELEASE-
//! FAMILY-TRAINS-2026-10-03).
//!
//! A request's `task` is its family id: training builds every request with `task=family_id`
//! (`python/qd_data/mixture.py::_request`). A task id the release never trained reaches the model
//! as a prompt shape it never saw, and the answer it gets back looks like any other. So
//! `release_manifest.json` records `trained_families` (`qd-export --train-manifest` writes it,
//! sorted and unique, from the train split's manifest), and admission refuses any other task
//! with `task_not_trained`, naming what was trained. A manifest that does not record the field
//! cannot say what it trained, so it admits nothing: `available` is then empty, which a
//! well-formed manifest can never make it, because the reader refuses an empty list at open.
//!
//! Every release here is synthetic and bound to the reference backend's identity, read by the
//! runtime's own `Release::open` and served through `Runtime::from_release`, the path
//! `qd-metal-serve` takes (`crates/qd-metal/src/serve.rs`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use qd_runtime::backend::DecisionBackend;
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::release::{MANIFEST_FILE, Release, ReleaseRefusalKind};
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{DecisionRequest, Response};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::{Value, json};

/// The family training builds `code.defect_class` requests under (`DEFECT_FAMILY_ID`).
const DEFECT: &str = "code.defect_class";
/// The task the schema doc's example sent until 2026-10-03; no release trains it.
const UNTRAINED: &str = "devcouncil.verdict";
/// A one-line Rust diff in the trained `code.defect_class` shape.
const DEFECT_CONTEXT: &str = "file: src/add.rs\n\n@@ -1 +1 @@\n-    a + b\n+    a - b\n";
/// Not a diff: what admission refuses for `code.defect_class` once the task is admitted.
const PROSE: &str = "fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";

static N: AtomicUsize = AtomicUsize::new(0);

/// A directory removed when the test ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "qd-trained-{tag}-{}-{}",
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

/// A release bound to the reference backend. `trained` is written as the manifest's
/// `trained_families` value; `None` leaves the key out.
fn release_dir(tag: &str, trained: Option<Value>) -> Scratch {
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
        "format": "qd-release.v1",
        "expected_identity": {
            "weight_hash": id.weight_hash,
            "tokenizer_hash": id.tokenizer_hash,
            "config_sha256": sha(config),
            "calibration_hash": table.hash(),
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
    if let Some(trained) = trained {
        manifest["trained_families"] = trained;
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

fn request(task: &str, context: &str) -> DecisionRequest {
    let value = json!({
        "schema_version": 1, "task": task,
        "context_b64": qd_runtime::b64::encode(context.as_bytes()),
        "context_len": context.len(),
        "question": "What kind of change is this diff?",
        "slots": [{"name": "defect_class", "type": "choice",
                   "options": ["stub", "logic", "cosmetic", "clean"]}],
        "route": "generic",
    });
    match parse_line(&serde_json::to_vec(&value).expect("serializes")) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("the test request was not accepted on the wire: {other:?}"),
    }
}

/// The refusal's serialized form, so the check names its kind and carries its fields.
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
fn a_task_the_release_did_not_train_is_refused_naming_what_it_trained() {
    let dir = release_dir("untrained", Some(json!([DEFECT])));
    let refusal = refused(
        runtime(&dir.0).answer(&request(UNTRAINED, PROSE), None),
        "task_not_trained",
    );
    assert_eq!(refusal["task"], json!(UNTRAINED));
    assert_eq!(refusal["available"], json!([DEFECT]));
}

#[test]
fn a_task_the_release_trained_is_admitted_and_answered() {
    let dir = release_dir("trained", Some(json!([DEFECT, UNTRAINED])));
    let runtime = runtime(&dir.0);
    answered(runtime.answer(&request(DEFECT, DEFECT_CONTEXT), None));
    // Any trained family, not only the one whose context shape admission also checks.
    answered(runtime.answer(&request(UNTRAINED, PROSE), None));
}

#[test]
fn the_task_is_refused_before_its_context_is_read() {
    // `code.defect_class` holds its context to the trained diff shape. A release that did not
    // train the family refuses the task, not the context: the context's shape is a question
    // about a task the release can answer.
    let dir = release_dir("order", Some(json!([UNTRAINED])));
    refused(
        runtime(&dir.0).answer(&request(DEFECT, PROSE), None),
        "task_not_trained",
    );
    // With the family trained, the same request reaches the context check.
    let dir = release_dir("order-trained", Some(json!([DEFECT])));
    refused(
        runtime(&dir.0).answer(&request(DEFECT, PROSE), None),
        "context_not_unified_diff",
    );
}

#[test]
fn a_release_that_does_not_record_its_trained_families_admits_no_task() {
    for (tag, trained) in [("absent", None), ("null", Some(Value::Null))] {
        let dir = release_dir(tag, trained);
        let runtime = runtime(&dir.0);
        for (task, context) in [(DEFECT, DEFECT_CONTEXT), (UNTRAINED, PROSE)] {
            let refusal = refused(
                runtime.answer(&request(task, context), None),
                "task_not_trained",
            );
            assert_eq!(refusal["task"], json!(task), "{tag}");
            assert_eq!(
                refusal["available"],
                json!([]),
                "{tag}: a release that cannot say what it trained names nothing"
            );
        }
    }
}

#[test]
fn a_malformed_trained_families_refuses_the_release_at_open() {
    let long = "x".repeat(RenderCaps::DEFAULT.max_task_bytes + 1);
    let many: Vec<String> = (0..10_000).map(|i| format!("family.{i:05}")).collect();
    let cases: [(&str, Value, &str); 9] = [
        ("empty", json!([]), "empty"),
        ("not-a-list", json!(DEFECT), "not an array"),
        ("not-a-string", json!([DEFECT, 7]), "not a string"),
        ("blank", json!([" "]), "blank"),
        ("unsorted", json!([DEFECT, "code.change_scope"]), "ascending"),
        ("duplicate", json!([DEFECT, DEFECT]), "ascending"),
        ("untrimmed", json!([" code.defect_class"]), "whitespace"),
        ("too-long", json!([long]), "bytes"),
        ("too-many", json!(many), "over the bound"),
    ];
    for (tag, trained, needle) in cases {
        let dir = release_dir(tag, Some(trained));
        match Release::open(&dir.0) {
            Err(err) => {
                assert_eq!(err.kind, ReleaseRefusalKind::Manifest, "{tag}: {err}");
                assert!(err.detail.contains("trained_families"), "{tag}: {err}");
                assert!(err.detail.contains(needle), "{tag}: {needle:?} not in {err}");
            }
            Ok(_) => panic!("{tag}: a release with a malformed trained_families opened"),
        }
    }
}
