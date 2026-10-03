//! `qd-metal-serve`'s wiring on the CPU: release -> backend -> `Runtime::from_release` ->
//! `Service::with_factory` -> `Server::bind`, with qd-runtime's reference backend standing in for
//! `MetalBackend` (the only part that needs the GPU). Nothing here touches the GPU.
//!
//! The release directory is synthetic and bound to the reference backend's identity: the same
//! manifest format `qd-export` writes (`qd-release.v2`, binding the prompt format the tower was
//! trained on), read by the same `Release::open`.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use qd_metal::serve::{open_release, release_factory, BackendStarter};
use qd_runtime::backend::DecisionBackend;
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::reference::{ReferenceBackend, REFERENCE_BACKEND_NAME};
use qd_runtime::registry::HeadRegistry;
use qd_runtime::release::ReleaseRefusalKind;
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::RuntimeConfig;
use qd_runtime::serve::{ServeOptions, Server};
use qd_runtime::service::{Service, ServiceConfig};
use serde_json::{json, Value};

static N: AtomicUsize = AtomicUsize::new(0);

/// A fresh directory under the system temp dir (short, so a socket path inside it fits
/// `sockaddr_un`).
fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "qdms-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn sha(bytes: &[u8]) -> String {
    qd_runtime::hex(&qd_runtime::sha256(bytes))
}

/// A release directory whose manifest binds `weight_hash`, the reference tokenizer hash and
/// prompt format 2, and records `devcouncil.verdict` (the task [`request_line`] asks) as trained.
fn release_dir(weight_hash: &str) -> PathBuf {
    release_dir_with(weight_hash, |_| {})
}

/// [`release_dir`], its manifest then changed by `edit`.
fn release_dir_with(weight_hash: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = scratch("release");
    let reference = ReferenceBackend::new(true);
    let id = reference.identity();
    let config = br#"{"architectures":["Qwen3_5ForConditionalGeneration"]}"#;
    let table = CalibrationTable::reference();
    let calibration = serde_json::to_vec(&table).unwrap();
    let weights = b"not read by the reference backend";
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("calibration.json"), &calibration).unwrap();
    std::fs::write(dir.join("model.safetensors"), weights).unwrap();
    let mut manifest = json!({
        "format": "qd-release.v2",
        "expected_identity": {
            "weight_hash": weight_hash,
            "tokenizer_hash": id.tokenizer_hash,
            "config_sha256": sha(config),
            "calibration_hash": table.hash(),
            "prompt_format": 2,
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
        "trained_families": ["devcouncil.verdict"],
    });
    edit(&mut manifest);
    std::fs::write(dir.join("release_manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    dir
}

/// The reference backend, counting how often the factory starts one.
fn reference_starter(starts: Arc<AtomicUsize>) -> BackendStarter {
    Arc::new(move |_release| {
        starts.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(ReferenceBackend::new(true)) as Arc<dyn DecisionBackend>)
    })
}

fn service_over(dir: &Path, starts: Arc<AtomicUsize>) -> Arc<Service> {
    let release = open_release(dir).expect("the synthetic release opens");
    let caps = RenderCaps::DEFAULT;
    let factory = release_factory(release, HeadRegistry::new(), caps, reference_starter(starts));
    let cfg = ServiceConfig {
        runtime: RuntimeConfig {
            caps,
            enable_reference_backend: false,
        },
        ..ServiceConfig::default()
    };
    Arc::new(Service::with_factory(cfg, factory))
}

fn request_line() -> Vec<u8> {
    let context = b"fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";
    serde_json::to_vec(&json!({
        "schema_version": 1, "task": "devcouncil.verdict",
        "context_b64": qd_runtime::b64::encode(context), "context_len": context.len(),
        "question": "Does this diff implement what the commit message claims?",
        "slots": [{"name": "verdict", "type": "choice", "options": ["stub", "logic", "cosmetic", "clean"]}],
        "route": "generic",
    }))
    .unwrap()
}

fn reply(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("the reply is JSON")
}

#[test]
fn a_release_bound_to_its_backend_answers_through_the_service() {
    let dir = release_dir(&ReferenceBackend::new(true).identity().weight_hash);
    let starts = Arc::new(AtomicUsize::new(0));
    let service = service_over(&dir, Arc::clone(&starts));
    assert_eq!(starts.load(Ordering::SeqCst), 0, "the backend starts on the first request, not before");
    let r = reply(&service.handle_line(&request_line()));
    assert_eq!(r["status"], "ok", "{r}");
    assert_eq!(r["backend"], REFERENCE_BACKEND_NAME);
    assert_eq!(r["degraded"], true, "a non-model backend is degraded in every answer");
    // The calibration table serving is the release's, and the pin of it is answered.
    let mut pinned: Value = serde_json::from_slice(&request_line()).unwrap();
    pinned["expect"] = json!({"calibration_hash": CalibrationTable::reference().hash()});
    let r2 = reply(&service.handle_line(&serde_json::to_vec(&pinned).unwrap()));
    assert_eq!(r2["status"], "ok", "{r2}");
    assert_eq!(starts.load(Ordering::SeqCst), 1, "a warm runtime is reused, not rebuilt");
    // Idle eviction drops the backend; the next request starts a new one from the same release.
    assert!(service.evict());
    assert_eq!(reply(&service.handle_line(&request_line()))["status"], "ok");
    assert_eq!(starts.load(Ordering::SeqCst), 2);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Fable's pipeline ruling, item 4: the product path answers only the task families the release
/// says it trained. `request_line` asks `devcouncil.verdict`.
#[test]
fn the_product_path_refuses_a_task_the_release_did_not_train() {
    let dir = release_dir_with(&ReferenceBackend::new(true).identity().weight_hash, |m| {
        m["trained_families"] = json!(["code.defect_class"]);
    });
    let service = service_over(&dir, Arc::new(AtomicUsize::new(0)));
    let r = reply(&service.handle_line(&request_line()));
    assert_eq!(r["status"], "refused", "{r}");
    assert_eq!(r["refusal"]["kind"], "task_not_trained", "{r}");
    assert_eq!(r["refusal"]["task"], "devcouncil.verdict", "{r}");
    assert_eq!(r["refusal"]["available"], json!(["code.defect_class"]), "{r}");
    std::fs::remove_dir_all(&dir).unwrap();

    // A release that does not record what it trained (every v4 export) admits nothing.
    let dir = release_dir_with(&ReferenceBackend::new(true).identity().weight_hash, |m| {
        m.as_object_mut().unwrap().remove("trained_families");
    });
    let service = service_over(&dir, Arc::new(AtomicUsize::new(0)));
    let r = reply(&service.handle_line(&request_line()));
    assert_eq!(r["refusal"]["kind"], "task_not_trained", "{r}");
    assert_eq!(r["refusal"]["available"], json!([]), "{r}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_backend_that_is_not_the_releases_tower_is_unavailable_never_an_answer() {
    let dir = release_dir(&"ab".repeat(32));
    let starts = Arc::new(AtomicUsize::new(0));
    let service = service_over(&dir, starts);
    let r = reply(&service.handle_line(&request_line()));
    assert_eq!(r["status"], "error", "{r}");
    assert_eq!(r["error"]["kind"], "unavailable", "{r}");
    let detail = r["error"]["detail"].as_str().unwrap();
    assert!(detail.contains("identity_mismatch") && detail.contains("weight_hash"), "{detail}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_backend_that_fails_to_start_is_unavailable() {
    let dir = release_dir(&ReferenceBackend::new(true).identity().weight_hash);
    let release = open_release(&dir).unwrap();
    let failing: BackendStarter = Arc::new(|release| {
        Err(qd_runtime::BackendError::Unavailable {
            detail: format!("no GPU in this test for {}", release.dir().display()),
        })
    });
    let factory = release_factory(release, HeadRegistry::new(), RenderCaps::DEFAULT, failing);
    let service = Service::with_factory(ServiceConfig::default(), factory);
    let r = reply(&service.handle_line(&request_line()));
    assert_eq!(r["error"]["kind"], "unavailable", "{r}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn what_is_not_a_release_is_refused_before_anything_serves() {
    let empty = scratch("empty");
    assert_eq!(open_release(&empty).unwrap_err().kind, ReleaseRefusalKind::Manifest);
    // A release whose calibration file is not the bound one.
    let dir = release_dir(&ReferenceBackend::new(true).identity().weight_hash);
    std::fs::write(dir.join("calibration.json"), b"{}").unwrap();
    assert_eq!(open_release(&dir).unwrap_err().kind, ReleaseRefusalKind::CalibrationMismatch);
    std::fs::remove_dir_all(&empty).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A v4 release -- `qd-release.v1`, no prompt format, exactly what `qd-export` wrote for F -- and a
/// v2 release that states no format or another one are each refused before anything serves: a
/// format-2 runtime never feeds its prompts to a tower trained on format 1.
#[test]
fn a_release_of_another_prompt_format_is_refused_before_anything_serves() {
    let weight_hash = ReferenceBackend::new(true).identity().weight_hash.clone();
    let v4 = release_dir_with(&weight_hash, |m| {
        m["format"] = json!("qd-release.v1");
        m["expected_identity"].as_object_mut().unwrap().remove("prompt_format");
    });
    let err = open_release(&v4).unwrap_err();
    assert_eq!(err.kind, ReleaseRefusalKind::Manifest, "{err}");
    assert!(err.detail.contains("qd-release.v1"), "{err}");

    let unstated = release_dir_with(&weight_hash, |m| {
        m["expected_identity"].as_object_mut().unwrap().remove("prompt_format");
    });
    let err = open_release(&unstated).unwrap_err();
    assert_eq!(err.kind.as_str(), "prompt_format", "{err}");

    let format_1 = release_dir_with(&weight_hash, |m| m["expected_identity"]["prompt_format"] = json!(1));
    let err = open_release(&format_1).unwrap_err();
    assert_eq!(err.kind.as_str(), "prompt_format", "{err}");
    assert!(err.detail.contains("prompt_format 1"), "{err}");

    for dir in [v4, unstated, format_1] {
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn the_socket_answers_what_the_service_answers() {
    let dir = release_dir(&ReferenceBackend::new(true).identity().weight_hash);
    let service = service_over(&dir, Arc::new(AtomicUsize::new(0)));
    let socket = dir.join("s.sock");
    let mut options = ServeOptions::new(&socket);
    options.read_timeout = Duration::from_secs(10);
    let server = Server::bind(Arc::clone(&service), options).expect("binds");
    let stop = server.shutdown_handle();
    let handle = std::thread::spawn(move || server.run());

    let mut stream = UnixStream::connect(&socket).expect("connects");
    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut line = request_line();
    line.push(b'\n');
    stream.write_all(&line).unwrap();
    let mut got = String::new();
    BufReader::new(stream.try_clone().unwrap()).read_line(&mut got).unwrap();
    let r = reply(got.trim_end().as_bytes());
    assert_eq!(r["status"], "ok", "{r}");
    // The socket's bytes are the in-process path's bytes for the same request.
    let direct = service.handle_line(&request_line());
    assert_eq!(got.trim_end().as_bytes(), &direct[..]);
    drop(stream);

    stop.stop();
    handle.join().expect("the server thread").expect("the server ran");
    assert!(!socket.exists(), "the socket is removed on shutdown");
    std::fs::remove_dir_all(&dir).unwrap();
}
