//! Shared fixtures for the integration tests.
//!
//! Every test builds its request as **wire bytes** and pushes them through the public entry points,
//! because that is the only way a caller can reach this crate. A test that constructed a
//! `DecisionRequest` directly would be testing a path no caller has.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use qd_runtime::backend::{
    BackendIdentity, DecisionBackend, DecodeMode, Logits, PrefillHandle, SlotQuery, StateSnapshot,
};
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::refusal::BackendError;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::{Runtime, RuntimeConfig};
use qd_runtime::schema::DecisionRequest;
use qd_runtime::service::{Service, ServiceConfig};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::{Value, json};

/// The context used by most tests: a stubbed function, three lines.
pub const SAMPLE_CONTEXT: &str = "fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";

pub const SAMPLE_QUESTION: &str = "Does this diff implement what the commit message claims?";

/// A well-formed wire request with one slot of each kind.
pub fn sample_request() -> Value {
    request_with_slots(
        SAMPLE_CONTEXT.as_bytes(),
        json!([
            {"name": "verdict", "type": "choice", "options": ["stub", "logic", "cosmetic", "clean"]},
            {"name": "severity", "type": "score", "bins": 5},
            {"name": "evidence", "type": "span"}
        ]),
    )
}

/// `docs/schema-api.md`: the context crosses the wire as `context_b64` plus a required
/// `context_len`, and there is no second accepted form. Every fixture goes through here, so a
/// change to the encoding is one edit rather than a sweep.
pub fn request_with_slots(context: &[u8], slots: Value) -> Value {
    json!({
        "schema_version": 1,
        "task": "devcouncil.verdict",
        "context_b64": qd_runtime::b64::encode(context),
        "context_len": context.len(),
        "question": SAMPLE_QUESTION,
        "slots": slots,
        "route": "generic"
    })
}

pub fn line(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).expect("test fixture serializes")
}

/// Parse a wire request the way the service does, and fail the test if it was refused.
pub fn validated(value: &Value) -> DecisionRequest {
    match parse_line(&line(value)) {
        Ok(Incoming::Request(request)) => *request,
        Ok(Incoming::Op(op)) => panic!("expected a request, got the control op {op:?}"),
        Err(refusal) => panic!("fixture was refused: {refusal}"),
    }
}

/// A service with the deterministic reference backend switched on.
pub fn reference_service() -> Service {
    Service::new(ServiceConfig {
        runtime: RuntimeConfig {
            caps: RenderCaps::DEFAULT,
            enable_reference_backend: true,
        },
        ..ServiceConfig::default()
    })
}

/// A service with **no** backend, which is the shipping default.
pub fn backendless_service() -> Service {
    Service::new(ServiceConfig::default())
}

pub fn reference_runtime() -> Runtime {
    Runtime::with_backend(
        Arc::new(ReferenceBackend::new(true)),
        CalibrationTable::reference(),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("the reference runtime assembles")
}

/// Read a reply's `status` tag.
pub fn status_of(reply: &[u8]) -> String {
    serde_json::from_slice::<Value>(reply)
        .ok()
        .and_then(|v| v.get("status").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| format!("<unparseable reply: {}>", String::from_utf8_lossy(reply)))
}

/// Read a refusal reply's `refusal.kind`.
pub fn refusal_kind(reply: &[u8]) -> String {
    serde_json::from_slice::<Value>(reply)
        .ok()
        .and_then(|v| {
            v.get("refusal")
                .and_then(|r| r.get("kind"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| format!("<no refusal.kind in: {}>", String::from_utf8_lossy(reply)))
}

// -- test backends ------------------------------------------------------------------------------

/// Wraps the reference backend and changes exactly one behaviour, so a test that fails names the
/// behaviour it was about.
pub struct Wrapped {
    inner: ReferenceBackend,
    identity: BackendIdentity,
    behaviour: Behaviour,
}

pub enum Behaviour {
    /// Writes to the snapshot even when asked for a read-only decode. Must be caught.
    LeaksUnderReadonly,
    /// Declares itself a model with host-visible state. The **control** for `OpaqueState`: with
    /// this, `degraded` must be false, so a `degraded` seen under `OpaqueState` is attributable to
    /// the unrunnable check and to nothing else.
    ClaimsToBeAModel,
    /// Declares itself a model whose state is **not** host-visible, so the slot-isolation check
    /// cannot run.
    OpaqueState,
    /// Poisons on prefill.
    PoisonsOnPrefill,
    /// Sleeps in prefill when the prompt names `slow-task`.
    SlowForTask { delay: Duration },
    /// Returns one logit row too few.
    WrongLogitShape,
    /// Returns a NaN in row 0.
    NonFiniteLogit,
}

impl Wrapped {
    pub fn new(behaviour: Behaviour) -> Self {
        let inner = ReferenceBackend::new(true);
        let mut identity = inner.identity().clone();
        match behaviour {
            Behaviour::OpaqueState => {
                identity.name = "test-opaque-state".to_string();
                identity.is_model = true;
                identity.state_host_visible = false;
            }
            Behaviour::ClaimsToBeAModel => {
                identity.name = "test-claims-to-be-a-model".to_string();
                identity.is_model = true;
            }
            _ => {}
        }
        Self {
            inner,
            identity,
            behaviour,
        }
    }

    pub fn runtime(behaviour: Behaviour) -> Runtime {
        Runtime::with_backend(
            Arc::new(Self::new(behaviour)),
            CalibrationTable::reference(),
            HeadRegistry::new(),
            RenderCaps::DEFAULT,
        )
        .expect("the wrapped runtime assembles")
    }
}

impl DecisionBackend for Wrapped {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        match &self.behaviour {
            Behaviour::PoisonsOnPrefill => Err(BackendError::Poisoned {
                reason: "the test backend poisons on every prefill".to_string(),
            }),
            Behaviour::SlowForTask { delay } if prefix.contains("slow-task") => {
                std::thread::sleep(*delay);
                self.inner.prefill(prefix)
            }
            _ => self.inner.prefill(prefix),
        }
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
        match &self.behaviour {
            Behaviour::LeaksUnderReadonly => {
                // Ask the inner backend for a real write-back whatever the caller said. This is the
                // bug the runtime must catch, written deliberately.
                self.inner
                    .decode_slot(snapshot, query, DecodeMode::WriteBack)
            }
            Behaviour::WrongLogitShape => {
                let mut logits = self.inner.decode_slot(snapshot, query, mode)?;
                logits.values.pop();
                Ok(logits)
            }
            Behaviour::NonFiniteLogit => {
                let mut logits = self.inner.decode_slot(snapshot, query, mode)?;
                if let Some(first) = logits.values.first_mut() {
                    *first = f32::NAN;
                }
                Ok(logits)
            }
            _ => self.inner.decode_slot(snapshot, query, mode),
        }
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.inner.pooled_features(snapshot)
    }
}

/// A socket path inside the per-user temp directory, short enough for `sun_path`.
pub fn temp_socket_path(tag: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("qd-{tag}-{}-{unique}.sock", std::process::id()))
}
