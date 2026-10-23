//! Adversarial tests for lane X1 (slot-shape admission), written by the reviewer.
//! Every test states the expectation a safe implementation must meet; a failing test is a break.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use qd_runtime::backend::DecisionBackend;
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::refusal::BackendError;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::release::{
    ENSEMBLE_DECODE, ENSEMBLE_FORMAT, ENSEMBLE_MANIFEST_FILE, Ensemble, MANIFEST_FILE,
    MANIFEST_FORMAT, Release, ReleaseRefusalKind,
};
use qd_runtime::render::{PROMPT_FORMAT, RenderCaps};
use qd_runtime::runtime::Runtime;
use qd_runtime::service::{RuntimeFactory, Service, ServiceConfig};
use serde_json::{Value, json};

const DEFECT: &str = "code.defect_class";
const COMMIT: &str = "gitpulse.commit_type";
const ROUTE: &str = "devtype.route";

static N: AtomicUsize = AtomicUsize::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "qd-advx1-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
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

const CONFIG: &[u8] = br#"{"architectures":["Qwen3_5ForConditionalGeneration"]}"#;

/// A member/release directory. `weight_tag` makes the recorded weight_hash distinct.
fn write_release(
    dir: &Path,
    weight_tag: Option<&str>,
    families: Option<Value>,
    shapes: Option<Value>,
) {
    let reference = ReferenceBackend::new(true);
    let id = reference.identity();
    let table = CalibrationTable::reference();
    let calibration = serde_json::to_vec(&table).unwrap();
    let weights = format!("weights {}", weight_tag.unwrap_or("x")).into_bytes();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    std::fs::write(dir.join("calibration.json"), &calibration).unwrap();
    std::fs::write(dir.join("model.safetensors"), &weights).unwrap();
    let weight_hash = match weight_tag {
        Some(tag) => sha(tag.as_bytes()),
        None => id.weight_hash.clone(),
    };
    let mut manifest = json!({
        "format": MANIFEST_FORMAT,
        "expected_identity": {
            "weight_hash": weight_hash,
            "tokenizer_hash": id.tokenizer_hash,
            "config_sha256": sha(CONFIG),
            "calibration_hash": table.hash(),
            "prompt_format": PROMPT_FORMAT,
        },
        "files": {
            "config.json": {"sha256": sha(CONFIG)},
            "model.safetensors": {"sha256": sha(&weights)},
            "calibration.json": {"sha256": sha(&calibration)},
        },
        "calibration": {
            "file": "calibration.json",
            "file_sha256": sha(&calibration),
            "table_hash": table.hash(),
        },
    });
    if let Some(f) = families {
        manifest["trained_families"] = f;
    }
    if let Some(s) = shapes {
        manifest["trained_slot_shapes"] = s;
    }
    std::fs::write(dir.join(MANIFEST_FILE), serde_json::to_vec_pretty(&manifest).unwrap())
        .unwrap();
}

fn release_dir(tag: &str, families: Option<Value>, shapes: Option<Value>) -> Scratch {
    let dir = Scratch::new(tag);
    write_release(&dir.0, None, families, shapes);
    dir
}

fn contract_release(tag: &str) -> Scratch {
    release_dir(
        tag,
        Some(json!([DEFECT, ROUTE, COMMIT])),
        Some(json!({
            DEFECT: [["choice:5", "span"]],
            COMMIT: [["choice:12"]],
            ROUTE: [["choice:5"]],
        })),
    )
}

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

fn reply(service: &Service, line: &[u8]) -> Value {
    serde_json::from_slice(&service.handle_line(line)).expect("a JSON reply")
}

fn status_of(service: &Service) -> Value {
    reply(service, br#"{"op":"status"}"#)
}

// -- target 1: live callers' exact request bytes -----------------------------------------------

/// DevCouncil `lappi/request.go:57-62,70-90` as encoded by encoding/json (struct field order,
/// `omitempty` on a span's options, `metadata:{}`).
fn devcouncil_line(context: &str) -> Vec<u8> {
    format!(
        concat!(
            r#"{{"schema_version":1,"task":"code.defect_class","context_b64":"{b64}","context_len":{len},"#,
            r#""question":"What kind of change is this diff, and which lines does it touch?","#,
            r#""slots":[{{"name":"defect_class","type":"choice","options":["stub","logic","cosmetic","clean"]}},"#,
            r#"{{"name":"defect_span","type":"span"}}],"route":"generic","#,
            r#""example_id":"devcouncil:code.defect_class:abc","metadata":{{}}}}"#
        ),
        b64 = qd_runtime::b64::encode(context.as_bytes()),
        len = context.len()
    )
    .into_bytes()
}

/// GitPulse `src-tauri/src/lappi/request.rs:55-64` (its own pinned test line, with KNOWN_TYPES).
fn gitpulse_line() -> Vec<u8> {
    concat!(
        r#"{"schema_version":1,"task":"gitpulse.commit_type","#,
        r#""context_b64":"ZGlmZiAtLWdpdCBhL3ggYi94Cg==","context_len":19,"#,
        r#""question":"Which conventional commit type describes this staged change?","#,
        r#""slots":[{"name":"commit_type","type":"choice","options":["feat","fix","refactor","docs","test","chore","perf","build","ci","style","revert"]}],"#,
        r#""route":"generic","example_id":"gitpulse:gitpulse.commit_type:0123456789abcdef0123456789abcdef","metadata":{}}"#
    )
    .as_bytes()
    .to_vec()
}

/// DevType `LappiRouteRequest.swift:10-32`, sortedKeys, withoutEscapingSlashes.
fn devtype_line() -> Vec<u8> {
    format!(
        concat!(
            r#"{{"context_b64":"{b64}","context_len":5,"example_id":"devtype:devtype.route:r1","#,
            r#""metadata":{{}},"question":"Which DevType tool should answer this palette query?","#,
            r#""route":"generic","schema_version":1,"#,
            r#""slots":[{{"name":"route","options":["resolveDate","runMacro","findSnippet","none"],"type":"choice"}}],"#,
            r#""task":"devtype.route"}}"#
        ),
        b64 = qd_runtime::b64::encode(b"hello")
    )
    .into_bytes()
}

#[test]
fn live_callers_exact_requests_are_admitted_on_a_release_that_recorded_the_contract_shapes() {
    let dir = contract_release("live");
    let service = service_over(&dir.0);
    let ctx = "file: src/add.rs\n\n@@ -1 +1 @@\n-    a + b\n+    a - b\n";
    for (who, line) in [
        ("devcouncil", devcouncil_line(ctx)),
        ("gitpulse", gitpulse_line()),
        ("devtype", devtype_line()),
    ] {
        let r = reply(&service, &line);
        assert_eq!(r["status"], json!("ok"), "{who} was not admitted: {r}");
    }
    assert_eq!(status_of(&service)["slot_shape_check"], json!("checked"));
}

// -- target 1: canonicalisation ------------------------------------------------------------------

#[test]
fn an_option_count_that_differs_by_one_or_a_padded_task_is_refused_not_normalised() {
    let dir = contract_release("canon");
    let service = service_over(&dir.0);
    // 5 options (choice:6) for devtype.route, trained choice:5.
    let five = String::from_utf8(devtype_line())
        .unwrap()
        .replace(r#""none"]"#, r#""none","extra"]"#);
    let r = reply(&service, five.as_bytes());
    assert_eq!(r["refusal"]["kind"], json!("slot_shape_not_trained"), "{r}");
    assert_eq!(r["refusal"]["shape"], json!(["choice:6"]), "{r}");
    // A whitespace-padded task is a different task, never trimmed into admission.
    let padded = String::from_utf8(devtype_line())
        .unwrap()
        .replace("\"devtype.route\"}", "\" devtype.route\"}");
    let r = reply(&service, padded.as_bytes());
    assert_ne!(r["status"], json!("ok"), "{r}");
}

// -- target 2: release parsing -------------------------------------------------------------------

#[test]
fn a_null_trained_slot_shapes_is_not_silently_the_legacy_unchecked_release() {
    // Absent = legacy by design. An explicit `null` is a writer that wrote the key and lost the
    // value: it must fail closed at open, not turn the shape check off.
    let dir = Scratch::new("null-shapes");
    write_release(&dir.0, None, Some(json!([COMMIT])), Some(Value::Null));
    assert!(
        Release::open(&dir.0).is_err(),
        "trained_slot_shapes: null opened as a legacy release; every shape is admitted unchecked"
    );
}

#[test]
fn a_duplicate_family_key_in_trained_slot_shapes_is_refused_not_last_wins() {
    let dir = release_dir(
        "dup-family-key",
        Some(json!([COMMIT])),
        Some(json!({COMMIT: [["choice:12"]]})),
    );
    let path = dir.0.join(MANIFEST_FILE);
    let text = std::fs::read_to_string(&path).unwrap();
    let widened = text.replace(
        "\"trained_slot_shapes\": {",
        "\"trained_slot_shapes\": {\n    \"gitpulse.commit_type\": [[\"choice:2\"]],",
    );
    assert_ne!(text, widened, "the manifest text was not edited");
    std::fs::write(&path, widened).unwrap();
    assert!(
        Release::open(&dir.0).is_err(),
        "two entries for one family opened; the later one silently replaced the first"
    );
}

#[test]
fn empty_object_and_empty_family_list_never_open_as_a_checked_release() {
    for (tag, families, shapes) in [
        ("empty-both", json!([]), json!({})),
        ("families-with-empty-shapes", json!([COMMIT]), json!({})),
    ] {
        let dir = Scratch::new(tag);
        write_release(&dir.0, None, Some(families), Some(shapes));
        assert!(Release::open(&dir.0).is_err(), "{tag} opened");
    }
}

#[test]
fn shapes_with_the_wrong_json_types_or_spellings_are_refused_at_open() {
    for (tag, shapes) in [
        ("string", json!("choice:12")),
        ("number", json!(0)),
        ("bool", json!(false)),
        ("family-string", json!({COMMIT: "choice:12"})),
        ("family-null", json!({COMMIT: null})),
        ("shape-string", json!({COMMIT: ["choice:12"]})),
        ("key-null", json!({COMMIT: [[null]]})),
        ("fullwidth-digit", json!({COMMIT: [["choice:\u{ff11}\u{ff12}"]]})),
        ("trailing-space", json!({COMMIT: [["choice:12 "]]})),
        ("uppercase", json!({COMMIT: [["Choice:12"]]})),
    ] {
        let dir = Scratch::new(tag);
        write_release(&dir.0, None, Some(json!([COMMIT])), Some(shapes));
        assert!(Release::open(&dir.0).is_err(), "{tag} opened");
    }
}

#[test]
fn a_family_name_that_differs_in_case_is_not_the_trained_family() {
    let dir = release_dir(
        "family-case",
        Some(json!([COMMIT])),
        Some(json!({COMMIT: [["choice:12"]]})),
    );
    let service = service_over(&dir.0);
    let line = String::from_utf8(gitpulse_line())
        .unwrap()
        .replace("\"gitpulse.commit_type\"", "\"GitPulse.commit_type\"");
    let r = reply(&service, line.as_bytes());
    assert_eq!(r["refusal"]["kind"], json!("task_not_trained"), "{r}");
}

// -- target 3: ensembles that disagree on trained_slot_shapes alone ------------------------------

fn ensemble_dir(shapes_a: Option<Value>, shapes_b: Option<Value>) -> Scratch {
    let root = Scratch::new("ensemble");
    let reference = ReferenceBackend::new(true);
    let id = reference.identity();
    let mut members = Vec::new();
    for (name, tag, shapes) in [("m0", "seed-0", shapes_a), ("m1", "seed-1", shapes_b)] {
        let dir = root.0.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        write_release(&dir, Some(tag), Some(json!([COMMIT])), shapes);
        let manifest_bytes = std::fs::read(dir.join(MANIFEST_FILE)).unwrap();
        let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
        members.push(json!({
            "dir": name,
            "release_manifest_sha256": sha(&manifest_bytes),
            "weights_sha256": sha(&weights),
            "weight_hash": sha(tag.as_bytes()),
            "config_sha256": sha(CONFIG),
            "tokenizer_hash": id.tokenizer_hash,
            "trained_width": 2048,
        }));
    }
    let table = CalibrationTable::reference();
    let calibration = serde_json::to_vec(&table).unwrap();
    std::fs::write(root.0.join("calibration.json"), &calibration).unwrap();
    let member_hashes = [sha(b"seed-0"), sha(b"seed-1")];
    let refs: Vec<&str> = member_hashes.iter().map(String::as_str).collect();
    let manifest = json!({
        "format": ENSEMBLE_FORMAT,
        "decode": ENSEMBLE_DECODE,
        "members": members,
        "expected_identity": {
            "weight_hash": qd_runtime::ensemble::ensemble_weight_hash(&refs),
            "tokenizer_hash": id.tokenizer_hash,
            "config_sha256": sha(CONFIG),
            "calibration_hash": table.hash(),
            "prompt_format": PROMPT_FORMAT,
            "trained_width": 2048,
        },
        "files": {"calibration.json": {"sha256": sha(&calibration)}},
        "calibration": {
            "file": "calibration.json",
            "file_sha256": sha(&calibration),
            "table_hash": table.hash(),
        },
    });
    std::fs::write(
        root.0.join(ENSEMBLE_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    root
}

#[test]
fn ensemble_members_that_differ_only_in_trained_slot_shapes_are_refused() {
    // Control: identical shapes open, so the fixture is valid and a refusal below is the shapes'.
    let same = ensemble_dir(
        Some(json!({COMMIT: [["choice:12"]]})),
        Some(json!({COMMIT: [["choice:12"]]})),
    );
    let opened = Ensemble::open(&same.0);
    assert!(opened.is_ok(), "control ensemble did not open: {:?}", opened.err());
    assert!(opened.unwrap().trained_slot_shapes().is_some());

    for (tag, a, b) in [
        (
            "different-shapes",
            Some(json!({COMMIT: [["choice:12"]]})),
            Some(json!({COMMIT: [["choice:3"]]})),
        ),
        ("recorded-vs-legacy", Some(json!({COMMIT: [["choice:12"]]})), None),
        ("legacy-vs-recorded", None, Some(json!({COMMIT: [["choice:12"]]}))),
        (
            "superset",
            Some(json!({COMMIT: [["choice:12"]]})),
            Some(json!({COMMIT: [["choice:12"], ["choice:3"]]})),
        ),
    ] {
        let split = ensemble_dir(a, b);
        let err = Ensemble::open(&split.0)
            .err()
            .unwrap_or_else(|| panic!("{tag}: members disagree on shapes only, and it opened"));
        assert_eq!(err.kind, ReleaseRefusalKind::MembersDisagree, "{tag}: {err}");
        assert!(err.to_string().contains("trained_slot_shapes"), "{tag}: {err}");
    }
}
