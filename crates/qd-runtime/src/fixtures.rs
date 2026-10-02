//! The golden wire corpus: every envelope variant this runtime can emit, on disk, as JSON.
//!
//! # Why this module exists
//!
//! `docs/schema-api.md` specifies the **request** side well and the **answer** side with one
//! example object. That asymmetry has already cost this repo twice — `GAP-RT-WIRE-CONTEXT-ENCODING`
//! and `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS` were both *one name meaning two things, with both
//! lanes' suites green*, and neither was caught by either lane testing itself. The cure that worked
//! on the training side (`python/qd_train/artifacts.py`, `docs/training-contract.md`) was to make
//! the seam **executable** rather than prose.
//!
//! This is that cure for the answer side. `fixtures/wire/` holds one file per envelope variant,
//! generated from the real types by the real code paths. The Python lane's parser reads the same
//! files. When a field is added, removed or renamed here, the corpus changes, and the other side
//! **breaks loudly** instead of silently accepting an envelope it no longer understands.
//!
//! # What is in the corpus
//!
//! * **answers** — one per slot kind, the contract's own three-slot example, the registered route,
//!   an all-abstained envelope, and a non-degraded one (which the reference backend can never
//!   produce, because it is not a model);
//! * **refusals** — every one of the [`Refusal`] variants, by `kind()`;
//! * **errors** — every one of the [`BackendError`] variants, by `kind()`.
//!
//! [`corpus`] is checked for exhaustiveness against both enums in
//! `tests/wire_fixtures.rs::the_corpus_covers_every_refusal_and_every_backend_error`, so a new
//! variant fails a test rather than quietly going unexported.
//!
//! # Determinism
//!
//! Every answer here is produced by [`crate::reference::ReferenceBackend`], whose scores are
//! counter-mode SHA-256 over the prompt bytes. The same input yields the same file on any machine,
//! which is what lets the corpus be compared rather than merely regenerated.
//!
//! # Regenerating
//!
//! ```text
//! QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures
//! ```
//!
//! Without the variable the same test **compares** and fails on any difference. That is the
//! direction that matters: the corpus is a gate, and a gate you can silently rewrite is not one.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::backend::DecisionBackend;
use crate::calibration::CalibrationTable;
use crate::reference::{ReferenceBackend, REFERENCE_FEATURE_DIM};
use crate::refusal::{BackendError, HashKind, Refusal};
use crate::registry::{HeadMatrix, HeadRegistry, RegisteredHead};
use crate::render::RenderCaps;
use crate::runtime::Runtime;
use crate::schema::{
    AnswerEnvelope, ConformalSet, Response, SlotAnswer, SlotValue, SpanValue,
    SUPPORTED_SCHEMA_VERSIONS,
};
use crate::wire::{self};

/// Directory, relative to the repository root, the corpus is written to.
pub const CORPUS_DIR: &str = "fixtures/wire";

/// Name of the manifest inside [`CORPUS_DIR`].
pub const MANIFEST: &str = "index.json";

/// One entry of the corpus: a file name, the reply it holds, and what it is for.
#[derive(Debug, Clone)]
pub struct WireFixture {
    /// File name inside [`CORPUS_DIR`], including the `.json`.
    pub file: String,
    /// `ok` | `refused` | `error` — the reply's own `status` discriminant.
    pub status: &'static str,
    /// One sentence saying what a parser should learn from this file.
    pub note: String,
    /// The wire request that produced this reply, where one did.
    pub request: Option<Value>,
    /// The reply exactly as a caller receives it.
    pub response: Response,
}

impl WireFixture {
    /// The bytes written to `file`: pretty JSON with a trailing newline, so a diff is readable.
    pub fn bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut out = serde_json::to_vec_pretty(&self.response)?;
        out.push(b'\n');
        Ok(out)
    }
}

/// The manifest, so a consumer can enumerate the corpus without globbing a directory and can
/// assert it read all of it. A parser that skipped a file silently is a parser that agrees with
/// nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The schema version every reply in the corpus carries.
    pub schema_version: u32,
    /// Total files, excluding the manifest itself.
    pub count: usize,
    /// Counts by `status`, so a consumer can assert it saw all three variants.
    pub by_status: BTreeMap<String, usize>,
    pub entries: Vec<ManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestEntry {
    pub file: String,
    pub status: String,
    /// `refusal.kind` or `error.kind` where the reply has one; absent for an answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub note: String,
    /// The wire request that produced this reply, where one did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Value>,
}

/// What a comparison found. Tri-state by construction: a corpus that could not be read is not a
/// corpus that matched.
#[derive(Debug, Clone, PartialEq)]
pub enum CorpusCheck {
    /// Every expected file was present and byte-identical, and nothing extra was there.
    Matches { files: usize },
    /// Names the differences, one per line.
    Differs { problems: Vec<String> },
    /// The directory could not be read at all.
    NotRead { reason: String },
}

// -- the answers ----------------------------------------------------------------------------------

/// A reference runtime with an optional registered head, for generating answers.
fn reference_runtime(registry: HeadRegistry) -> Result<Runtime, BackendError> {
    Runtime::with_backend(
        Arc::new(ReferenceBackend::new(true)),
        CalibrationTable::reference(),
        registry,
        RenderCaps::DEFAULT,
    )
}

/// The wire form of a request, built the way a caller builds one.
fn request(context: &[u8], slots: Value, route: &str) -> Value {
    json!({
        "schema_version": SUPPORTED_SCHEMA_VERSIONS[0],
        "task": "devcouncil.verdict",
        "context_b64": crate::b64::encode(context),
        "context_len": context.len(),
        "question": "Does this diff implement what the commit message claims?",
        "slots": slots,
        "route": route,
    })
}

const SAMPLE_CONTEXT: &[u8] = b"fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";

/// [`SAMPLE_CONTEXT`] with ` // 6` on its first line, for the fixtures that show answered slots.
///
/// The reference backend's answers are a function of the rendered prompt's bytes
/// (`reference.rs`'s prefill seeds from the prefix digest), so prompt format 2 re-rolled every
/// answer fixture, and over [`SAMPLE_CONTEXT`] all of them abstained. `6` is the first suffix that
/// shows the format-1 corpus's shapes again -- verdict and severity answered, evidence abstained
/// in the three-slot example, severity answered alone -- found by the deterministic search in
/// `AUDIT/v5-fmt-2026-10-02/fixture_search.py` (its output beside it). The question is unchanged:
/// `docs/schema-api.md` quotes it. `tests::every_answer_fixture_shows_its_intended_shape` is what
/// fails if a later render change re-rolls these again.
const ANSWERED_SAMPLE_CONTEXT: &[u8] = b"fn add(a: i32, b: i32) -> i32 { // 6\n    todo!()\n}\n";

/// The span fixture's four-line context, ` 2` on its first line: the first suffix under which the
/// `span` slot answers under prompt format 2 (same search, same record).
const SPAN_CONTEXT: &[u8] = b"alpha 2\nbeta\ngamma\ndelta\n";

/// A head bound to the reference backend's backbone, with rows for `verdict`.
fn registered_registry() -> HeadRegistry {
    let backbone = ReferenceBackend::new(true).identity().weight_hash.clone();
    let mut slots = BTreeMap::new();
    slots.insert(
        "verdict".to_string(),
        // Four named options plus the reserved noul row: the 17-row rule at k = 4.
        HeadMatrix {
            rows: vec![vec![0.01f32; REFERENCE_FEATURE_DIM]; 5],
        },
    );
    let mut registry = HeadRegistry::new();
    registry.insert(RegisteredHead {
        task: "devcouncil.verdict".to_string(),
        head_hash: "qd-fixture-head-v1".to_string(),
        backbone_hash: backbone,
        feature_dim: REFERENCE_FEATURE_DIM,
        slots,
    });
    registry
}

/// Answer one request end to end, so the fixture is what the runtime emits and not what this
/// module thinks it emits.
fn answered(
    file: &str,
    note: &str,
    wire_request: Value,
    registry: HeadRegistry,
) -> Result<WireFixture, String> {
    let runtime = reference_runtime(registry).map_err(|e| format!("{file}: runtime: {e}"))?;
    let bytes = serde_json::to_vec(&wire_request).map_err(|e| format!("{file}: {e}"))?;
    let parsed = wire::parse_line(&bytes).map_err(|e| format!("{file}: request refused: {e}"))?;
    let wire::Incoming::Request(decision) = parsed else {
        return Err(format!("{file}: the fixture request parsed as a control op"));
    };
    let response = runtime.answer(&decision, None);
    if !matches!(response, Response::Ok(_)) {
        return Err(format!(
            "{file}: expected an answer, got {}. A corpus entry that is not the variant it claims \
             would teach the other lane the wrong shape",
            response.status()
        ));
    }
    Ok(WireFixture {
        file: format!("{file}.json"),
        status: "ok",
        note: note.to_string(),
        request: Some(wire_request),
        response,
    })
}

/// The answer fixtures.
fn answer_fixtures() -> Result<Vec<WireFixture>, String> {
    let mut out = Vec::new();

    out.push(answered(
        "answer-example-three-slots",
        "The contract's own example request, answered: one slot of every kind in one envelope. \
         Note the five envelope keys — status, schema_version, backend, degraded, slots — where \
         docs/schema-api.md's example showed only two.",
        request(
            ANSWERED_SAMPLE_CONTEXT,
            json!([
                {"name": "verdict", "type": "choice",
                 "options": ["stub", "logic", "cosmetic", "clean"]},
                {"name": "severity", "type": "score", "bins": 5},
                {"name": "evidence", "type": "span"}
            ]),
            "generic",
        ),
        HeadRegistry::new(),
    )?);

    out.push(answered(
        "answer-choice-only",
        "A single `choice` slot. Its value is one of the request's option strings, never a letter \
         and never the string `noul`; abstention is the `noul` flag.",
        request(
            SAMPLE_CONTEXT,
            json!([{"name": "verdict", "type": "choice",
                    "options": ["stub", "logic", "cosmetic", "clean"]}]),
            "generic",
        ),
        HeadRegistry::new(),
    )?);

    out.push(answered(
        "answer-score-only",
        "A single `score` slot. Its value is a 1-based ordinal bin as a JSON number, and its \
         conformal set is a list of bins.",
        request(
            ANSWERED_SAMPLE_CONTEXT,
            json!([{"name": "severity", "type": "score", "bins": 5}]),
            "generic",
        ),
        HeadRegistry::new(),
    )?);

    out.push(answered(
        "answer-span-only",
        "A single `span` slot over a four-line context. Its value is {start_line, end_line}, \
         1-based and inclusive, and its conformal_set is null — a span has no conformal set.",
        request(
            SPAN_CONTEXT,
            json!([{"name": "evidence", "type": "span"}]),
            "generic",
        ),
        HeadRegistry::new(),
    )?);

    out.push(answered(
        "answer-max-options",
        "A `choice` slot at the documented maximum of 16 named options, which reads a 17-row head \
         slice: 16 option rows plus the reserved noul row last.",
        request(
            SAMPLE_CONTEXT,
            json!([{"name": "verdict", "type": "choice", "options": [
                "o01", "o02", "o03", "o04", "o05", "o06", "o07", "o08",
                "o09", "o10", "o11", "o12", "o13", "o14", "o15", "o16"
            ]}]),
            "generic",
        ),
        HeadRegistry::new(),
    )?);

    out.push(answered(
        "answer-registered-route",
        "The registered route: one GEMV on pooled features, no option text in the prompt and no \
         permuted second pass. The envelope shape is identical to the generic route's — `route` \
         does not appear in an answer.",
        request(
            SAMPLE_CONTEXT,
            json!([{"name": "verdict", "type": "choice",
                    "options": ["stub", "logic", "cosmetic", "clean"]}]),
            "registered",
        ),
        registered_registry(),
    )?);

    // The two shapes the reference backend cannot be made to produce on demand. Both are built
    // through the same public constructors `crate::answer` uses, so they are the real types
    // rather than hand-written JSON: `SlotAnswer::abstained` and `SlotAnswer::answered` are the
    // only two ways a `SlotAnswer` is made anywhere in this crate.
    let mut abstained = BTreeMap::new();
    abstained.insert("verdict".to_string(), SlotAnswer::abstained(0.004, true));
    abstained.insert("severity".to_string(), SlotAnswer::abstained(0.011, true));
    abstained.insert("evidence".to_string(), SlotAnswer::abstained(0.0, true));
    out.push(WireFixture {
        file: "answer-all-slots-abstained.json".to_string(),
        status: "ok",
        note: "Every slot abstained. This is `status: ok` with `noul: true` everywhere — it is \
               NOT a refusal and NOT an error. A caller that treats it as a failure has collapsed \
               model humility into machine failure, which is the confusion docs/schema-api.md \
               exists to prevent. `value` is null in every slot, because value is absent if and \
               only if noul."
            .to_string(),
        request: None,
        response: Response::Ok(AnswerEnvelope {
            schema_version: SUPPORTED_SCHEMA_VERSIONS[0],
            backend: "reference-deterministic-v1".to_string(),
            degraded: true,
            slots: abstained,
        }),
    });

    let mut clean = BTreeMap::new();
    clean.insert(
        "verdict".to_string(),
        SlotAnswer::answered(
            SlotValue::Choice("stub".to_string()),
            Some(ConformalSet::Choices(vec![
                "stub".to_string(),
                "logic".to_string(),
            ])),
            0.71,
        ),
    );
    clean.insert(
        "severity".to_string(),
        SlotAnswer::answered(
            SlotValue::Score(3),
            Some(ConformalSet::Scores(vec![2, 3, 4])),
            0.55,
        ),
    );
    clean.insert(
        "evidence".to_string(),
        SlotAnswer::answered(
            SlotValue::Span(SpanValue {
                start_line: 41,
                end_line: 47,
            }),
            None,
            0.62,
        ),
    );
    out.push(WireFixture {
        file: "answer-not-degraded.json".to_string(),
        status: "ok",
        note: "`degraded: false` everywhere — the shape a real model backend produces. No backend \
               in this build can emit it: the reference backend declares `is_model: false`, so \
               every answer it produces is degraded. The values are those of the example in \
               docs/schema-api.md, so the document and the corpus can be read against each other."
            .to_string(),
        request: None,
        response: Response::Ok(AnswerEnvelope {
            schema_version: SUPPORTED_SCHEMA_VERSIONS[0],
            backend: "a-model-backend".to_string(),
            degraded: false,
            slots: clean,
        }),
    });

    Ok(out)
}

// -- one of every refusal and every backend error ---------------------------------------------------

/// One [`Refusal`] of every variant, with both compared values filled in.
///
/// The canonical list. `tests/refusal_is_not_noul.rs` reads it from here rather than keeping a
/// second copy, and that file's exhaustive `kind_of` match is what fails to compile when a variant
/// is added without a fixture.
pub fn all_refusals() -> Vec<Refusal> {
    vec![
        Refusal::HashMismatch {
            which: HashKind::Tokenizer,
            expected: "aaaa".into(),
            actual: "bbbb".into(),
        },
        Refusal::HashMismatch {
            which: HashKind::HeadBackboneBinding,
            expected: "aaaa".into(),
            actual: "bbbb".into(),
        },
        Refusal::TooManyOptions {
            slot: "verdict".into(),
            limit: 16,
            actual: 17,
        },
        Refusal::TooFewOptions {
            slot: "verdict".into(),
            min: 2,
            actual: 1,
        },
        Refusal::DuplicateOption {
            slot: "verdict".into(),
            option: "stub".into(),
            first_index: 0,
            duplicate_index: 2,
        },
        Refusal::ReservedOptionName {
            slot: "verdict".into(),
            option_index: 1,
            reserved: "noul".into(),
        },
        Refusal::EmptySlots,
        Refusal::DuplicateSlotName {
            name: "verdict".into(),
            first_index: 0,
            duplicate_index: 1,
        },
        Refusal::EmptySlotName { index: 0 },
        Refusal::SlotNameOverCap {
            index: 0,
            cap: crate::schema::MAX_SLOT_NAME_BYTES,
            actual: crate::schema::MAX_SLOT_NAME_BYTES + 1,
        },
        Refusal::TooManySlots { cap: 32, actual: 33 },
        Refusal::BinsOutOfRange {
            slot: "severity".into(),
            min: 2,
            max: 16,
            actual: 1,
        },
        Refusal::UnknownSchemaVersion {
            supported: vec![1],
            actual: 2,
        },
        Refusal::ContextOverCap {
            cap: 131_072,
            actual: 200_000,
        },
        Refusal::ContextLengthMismatch {
            declared: 10,
            actual: 11,
        },
        Refusal::ContextLenMissing {
            found: "absent".into(),
        },
        Refusal::ContextNotBytes {
            got: "a `context` field".into(),
        },
        Refusal::ContextNotBase64 {
            offset: 4,
            expected: "a character from the standard base64 alphabet".into(),
            found: "`*`".into(),
        },
        Refusal::ContextEmpty {
            len: 4,
            non_whitespace: 0,
        },
        Refusal::ContextNotUtf8 {
            offset: 2,
            invalid_len: 1,
        },
        Refusal::ContextNotUnifiedDiff {
            line: 1,
            expected: "a `file: <path>` header line".into(),
            found: "\"What is the capital of France?\"".into(),
        },
        Refusal::ContextLanguageNotInPool {
            path: "src/main.c".into(),
            language: "unrecognised".into(),
            pool: qd_lang::DEFECT_CLASS_POOL_LANGUAGES
                .iter()
                .map(|lang| lang.as_str().to_string())
                .collect(),
        },
        Refusal::RenderedPromptOverCap {
            cap: 802_816,
            actual: 900_000,
        },
        Refusal::EmptyOption {
            slot: "verdict".into(),
            option_index: 1,
        },
        Refusal::OptionTextOverCap {
            slot: "verdict".into(),
            option_index: 0,
            cap: 512,
            actual: 513,
        },
        Refusal::QuestionOverCap {
            cap: 4096,
            actual: 5000,
        },
        Refusal::TaskOverCap {
            cap: 256,
            actual: 300,
        },
        Refusal::EmptyTask,
        Refusal::UnknownRoute {
            supported: vec!["generic".into(), "registered".into()],
            actual: "turbo".into(),
        },
        Refusal::UnknownSlotType {
            slot: "verdict".into(),
            supported: vec!["choice".into()],
            actual: "vibe".into(),
        },
        Refusal::SlotFieldNotAllowed {
            slot: "evidence".into(),
            slot_type: "span".into(),
            field: "bins".into(),
        },
        Refusal::SlotFieldMissing {
            slot: "severity".into(),
            slot_type: "score".into(),
            field: "bins".into(),
        },
        Refusal::RegisteredHeadMissing {
            task: "devtype.route".into(),
            available: vec![],
        },
        Refusal::RegisteredHeadSlotMissing {
            task: "devtype.route".into(),
            slot: "verdict".into(),
            available: vec!["severity".into()],
        },
        Refusal::RegisteredRouteSpanUnsupported {
            slot: "evidence".into(),
            rows: 4,
        },
        Refusal::CalibrationEntryMissing {
            slot: "verdict".into(),
            slot_type: "choice".into(),
            rows: 5,
        },
        Refusal::PayloadOverCap {
            cap: wire::MAX_PAYLOAD_BYTES,
            actual: 2_000_000,
        },
        Refusal::MalformedRequest {
            detail: "expected value at line 1".into(),
        },
        Refusal::AmbiguousEnvelope,
        Refusal::UnknownOp {
            supported: vec!["status".into()],
            actual: "restart".into(),
        },
    ]
}

/// One [`BackendError`] of every variant. The canonical list; see [`all_refusals`].
pub fn all_backend_errors() -> Vec<BackendError> {
    vec![
        BackendError::ReferenceBackendNotEnabled {
            name: "reference-deterministic-v1".into(),
        },
        BackendError::Unavailable {
            detail: "no model backend".into(),
        },
        BackendError::Poisoned {
            reason: "command buffer error".into(),
        },
        BackendError::RebuildFailed {
            detail: "device lost".into(),
        },
        BackendError::ReadonlyViolated {
            slot_index: 1,
            before: "aa".into(),
            after: "bb".into(),
        },
        BackendError::LogitShapeMismatch {
            slot: "verdict".into(),
            expected: 5,
            actual: 4,
        },
        BackendError::NonFiniteLogit {
            slot: "verdict".into(),
            row: 0,
        },
        BackendError::LogitKindMismatch {
            slot: "evidence".into(),
            expected: "pointer_start".into(),
        },
        BackendError::PrefillFailed {
            detail: "oom".into(),
        },
        BackendError::DecodeFailed {
            detail: "oom".into(),
        },
        BackendError::DeadlineExceeded { limit_ms: 30_000 },
        BackendError::Overloaded { limit: 8 },
    ]
}

/// What tells two samples of one `kind()` apart, where anything does.
///
/// `hash_mismatch` is the only refusal with more than one sample, because which hash failed is the
/// whole content of the refusal and a caller pinning `expect.head_hash` needs to see the
/// `head_backbone_binding` spelling as well as the `tokenizer` one.
fn discriminator(refusal: &Refusal) -> Option<&'static str> {
    match refusal {
        Refusal::HashMismatch { which, .. } => Some(which.as_str()),
        _ => None,
    }
}

/// Make a file name unique when two samples share a `kind()`, without renaming either.
///
/// The numeric suffix is a fallback, not the plan: a corpus entry called `-2` tells a reader
/// nothing, so a variant with several samples should carry a [`discriminator`]. The counter stays
/// so that adding one without a discriminator produces a clashing-but-distinct file rather than
/// one sample silently overwriting the other.
fn unique(prefix: &str, kind: &str, disc: Option<&str>, seen: &mut BTreeMap<String, usize>) -> String {
    let base = match disc {
        Some(disc) => format!("{prefix}-{kind}-{disc}"),
        None => format!("{prefix}-{kind}"),
    };
    let count = seen.entry(base.clone()).or_insert(0);
    *count += 1;
    if *count == 1 {
        format!("{base}.json")
    } else {
        format!("{base}-{count}.json")
    }
}

// -- the corpus -------------------------------------------------------------------------------------

/// Every fixture, in a stable order.
pub fn corpus() -> Result<Vec<WireFixture>, String> {
    let mut out = answer_fixtures()?;
    let mut seen = BTreeMap::new();

    for refusal in all_refusals() {
        let kind = refusal.kind();
        let note = format!(
            "Refusal `{kind}`: the request was not answerable as posed, so the model was never \
             asked. A refusal carries no `slots`, no `value`, no `score` and no `noul` at any \
             depth — reading one as an abstention is the confusion docs/schema-api.md forbids. \
             Message: {refusal}"
        );
        let file = unique("refusal", kind, discriminator(&refusal), &mut seen);
        out.push(WireFixture {
            file,
            status: "refused",
            note,
            request: None,
            response: Response::refused(refusal),
        });
    }

    for error in all_backend_errors() {
        let kind = error.kind();
        let note = format!(
            "Backend error `{kind}`: the backend could not serve, so the model was never asked. \
             Distinct from a refusal (a bug in the request) and from an abstention (the model \
             declined). Message: {error}"
        );
        out.push(WireFixture {
            file: unique("error", kind, None, &mut seen),
            status: "error",
            note,
            request: None,
            response: Response::failed(error),
        });
    }

    Ok(out)
}

/// The manifest for a corpus.
pub fn manifest(fixtures: &[WireFixture]) -> Manifest {
    let mut by_status: BTreeMap<String, usize> = BTreeMap::new();
    let entries = fixtures
        .iter()
        .map(|f| {
            *by_status.entry(f.status.to_string()).or_insert(0) += 1;
            ManifestEntry {
                file: f.file.clone(),
                status: f.status.to_string(),
                kind: match &f.response {
                    Response::Ok(_) => None,
                    Response::Refused(e) => Some(e.refusal.kind().to_string()),
                    Response::Error(e) => Some(e.error.kind().to_string()),
                },
                note: f.note.clone(),
                request: f.request.clone(),
            }
        })
        .collect();
    Manifest {
        schema_version: SUPPORTED_SCHEMA_VERSIONS[0],
        count: fixtures.len(),
        by_status,
        entries,
    }
}

/// Every file the corpus consists of, as `(relative name, bytes)`, manifest included.
pub fn files() -> Result<Vec<(String, Vec<u8>)>, String> {
    let fixtures = corpus()?;
    let mut out = Vec::with_capacity(fixtures.len() + 1);
    for fixture in &fixtures {
        let bytes = fixture
            .bytes()
            .map_err(|e| format!("{}: {e}", fixture.file))?;
        out.push((fixture.file.clone(), bytes));
    }
    let mut manifest_bytes =
        serde_json::to_vec_pretty(&manifest(&fixtures)).map_err(|e| format!("{MANIFEST}: {e}"))?;
    manifest_bytes.push(b'\n');
    out.push((MANIFEST.to_string(), manifest_bytes));
    Ok(out)
}

/// Write the corpus into `dir`, creating it if needed, and removing any `.json` no longer in it.
///
/// Removing the strays is deliberate: a corpus with a leftover file from a renamed variant teaches
/// the other lane a shape this build no longer emits.
pub fn write_corpus(dir: &Path) -> Result<usize, String> {
    let files = files()?;
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    let expected: BTreeSet<&str> = files.iter().map(|(name, _)| name.as_str()).collect();
    for stray in strays(dir, &expected)? {
        std::fs::remove_file(&stray).map_err(|e| format!("removing {}: {e}", stray.display()))?;
    }
    for (name, bytes) in &files {
        std::fs::write(dir.join(name), bytes)
            .map_err(|e| format!("writing {}: {e}", dir.join(name).display()))?;
    }
    Ok(files.len())
}

/// `.json` files in `dir` that the corpus does not contain.
fn strays(dir: &Path, expected: &BTreeSet<&str>) -> Result<Vec<PathBuf>, String> {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("reading {}: {e}", dir.display())),
    };
    let mut out = Vec::new();
    for entry in read {
        let entry = entry.map_err(|e| format!("reading {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !expected.contains(name) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Compare `dir` against the corpus this build would write.
///
/// Never returns "matches" for a directory it could not read: [`CorpusCheck::NotRead`] and
/// [`CorpusCheck::Matches`] are different answers, because a check that could not run must not
/// report the same result as a check that ran and passed.
pub fn check_corpus(dir: &Path) -> CorpusCheck {
    let files = match files() {
        Ok(files) => files,
        Err(reason) => return CorpusCheck::NotRead { reason },
    };
    if !dir.is_dir() {
        return CorpusCheck::NotRead {
            reason: format!(
                "{} is not a directory. Generate the corpus with \
                 `QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures`",
                dir.display()
            ),
        };
    }

    let mut problems = Vec::new();
    for (name, want) in &files {
        let path = dir.join(name);
        match std::fs::read(&path) {
            Ok(got) if &got == want => {}
            Ok(got) => problems.push(format!(
                "{name}: on disk differs from this build ({} bytes on disk, {} bytes here)",
                got.len(),
                want.len()
            )),
            Err(e) => problems.push(format!("{name}: {e}")),
        }
    }
    let expected: BTreeSet<&str> = files.iter().map(|(name, _)| name.as_str()).collect();
    match strays(dir, &expected) {
        Ok(strays) => {
            for stray in strays {
                problems.push(format!(
                    "{}: on disk but not in this build's corpus",
                    stray.display()
                ));
            }
        }
        Err(reason) => return CorpusCheck::NotRead { reason },
    }

    if problems.is_empty() {
        CorpusCheck::Matches { files: files.len() }
    } else {
        CorpusCheck::Differs { problems }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What one slot of an answer fixture shows.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Shape {
        Answered,
        Abstained,
    }
    use Shape::{Abstained, Answered};

    /// One row of [`INTENDED`]: the answer file, whether its envelope is degraded, and each
    /// slot's shape by name.
    type Intended = (&'static str, bool, &'static [(&'static str, Shape)]);

    /// Each answer fixture's intended shape, by file: whether the envelope is degraded, and each
    /// slot's shape. This is the corpus's matrix at 92e1c74 (prompt format 1).
    ///
    /// # Why it is pinned
    ///
    /// The reference backend's answers are a function of the rendered prompt's bytes, so a change
    /// to the renderer re-rolls every answer fixture. Prompt format 2 re-rolled them all to
    /// abstentions: the corpus kept its keys and its envelopes and lost every degraded-but-answered
    /// slot, with `check_corpus` and every structural test still green. This is what says so.
    /// A render change that re-rolls a fixture to another shape fails here, and the fix is a new
    /// fixture input, not a new row in this table.
    const INTENDED: &[Intended] = &[
        (
            "answer-example-three-slots.json",
            true,
            &[
                ("evidence", Abstained),
                ("severity", Answered),
                ("verdict", Answered),
            ],
        ),
        ("answer-choice-only.json", true, &[("verdict", Abstained)]),
        ("answer-score-only.json", true, &[("severity", Answered)]),
        ("answer-span-only.json", true, &[("evidence", Answered)]),
        ("answer-max-options.json", true, &[("verdict", Abstained)]),
        ("answer-registered-route.json", true, &[("verdict", Abstained)]),
        (
            "answer-all-slots-abstained.json",
            true,
            &[
                ("evidence", Abstained),
                ("severity", Abstained),
                ("verdict", Abstained),
            ],
        ),
        (
            "answer-not-degraded.json",
            false,
            &[
                ("evidence", Answered),
                ("severity", Answered),
                ("verdict", Answered),
            ],
        ),
    ];

    /// Every answer fixture names its slots `verdict` (choice), `severity` (score) and `evidence`
    /// (span); an answered slot's value must be that kind's.
    fn value_matches_slot(name: &str, value: &SlotValue) -> bool {
        matches!(
            (name, value),
            ("verdict", SlotValue::Choice(_))
                | ("severity", SlotValue::Score(_))
                | ("evidence", SlotValue::Span(_))
        )
    }

    #[test]
    fn every_answer_fixture_shows_its_intended_shape() {
        let corpus = corpus().expect("the corpus generates");
        let answers: Vec<&WireFixture> = corpus.iter().filter(|f| f.status == "ok").collect();
        let named: BTreeSet<&str> = INTENDED.iter().map(|(file, _, _)| *file).collect();
        let produced: BTreeSet<&str> = answers.iter().map(|f| f.file.as_str()).collect();
        assert_eq!(
            produced, named,
            "every answer fixture declares its intended shape here, and every declared one exists"
        );
        for fixture in answers {
            let Some((_, degraded, slots)) = INTENDED.iter().find(|(f, _, _)| *f == fixture.file)
            else {
                panic!("{} declares no intended shape", fixture.file);
            };
            let Response::Ok(envelope) = &fixture.response else {
                panic!("{} is not an answer", fixture.file);
            };
            assert_eq!(envelope.degraded, *degraded, "{}: envelope degraded", fixture.file);
            let got: Vec<(&str, Shape)> = envelope
                .slots
                .iter()
                .map(|(name, answer)| {
                    assert_eq!(answer.degraded, *degraded, "{}: {name} degraded", fixture.file);
                    match &answer.value {
                        Some(value) => {
                            assert!(!answer.noul, "{}: {name} has a value and noul", fixture.file);
                            assert!(
                                value_matches_slot(name, value),
                                "{}: {name} answered with {value:?}",
                                fixture.file
                            );
                            (name.as_str(), Answered)
                        }
                        None => {
                            assert!(answer.noul, "{}: {name} has no value and no noul", fixture.file);
                            (name.as_str(), Abstained)
                        }
                    }
                })
                .collect();
            assert_eq!(got, *slots, "{}: the slot shapes moved", fixture.file);
        }
    }

    /// The table itself keeps the coverage: every slot kind answered and abstained under a
    /// degraded backend, and answered under a model one.
    #[test]
    fn the_intended_shapes_cover_every_slot_kind_answered_and_abstained() {
        for name in ["verdict", "severity", "evidence"] {
            for (degraded, shape) in [(true, Answered), (true, Abstained), (false, Answered)] {
                assert!(
                    INTENDED.iter().any(|(_, d, slots)| *d == degraded
                        && slots.iter().any(|(n, s)| *n == name && *s == shape)),
                    "no answer fixture shows {name} {shape:?} with degraded={degraded}"
                );
            }
        }
    }
}
