//! The refusal layer.
//!
//! `docs/schema-api.md`: "A refusal is a typed error carrying which check failed and both values
//! compared. It is never an empty answer, and never `noul` — `noul` means *the model abstained*, a
//! refusal means *the request was not answerable as posed*. Collapsing the two would let a hash
//! mismatch read as model humility."
//!
//! Three structural properties keep those two apart, and each is asserted by a test in
//! `tests/refusal_is_not_noul.rs`:
//!
//! 1. A [`Refusal`] is not an [`crate::schema::AnswerEnvelope`] and cannot be converted into one.
//!    There is no `impl From<Refusal> for AnswerEnvelope`, no `AnswerEnvelope::refused()`, and
//!    `AnswerEnvelope` has no failure variant — the only way to hold both is
//!    [`crate::schema::Response`], whose variants a caller must match exhaustively.
//! 2. Serialized, a refusal carries `"status":"refused"` and **no** `slots`, `value`, `score`,
//!    `conformal_set` or `noul` key anywhere in its bytes. Deserializing a refusal envelope as an
//!    answer envelope fails.
//! 3. Every refusal names the check (`kind`) and carries both compared values.

use serde::{Deserialize, Serialize};

/// Which hash failed to match the build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HashKind {
    Tokenizer,
    Weights,
    Head,
    LabelSet,
    Calibration,
    /// The registered route's head file must be hash-bound to the backbone it was fitted on.
    HeadBackboneBinding,
}

impl HashKind {
    pub fn as_str(self) -> &'static str {
        match self {
            HashKind::Tokenizer => "tokenizer",
            HashKind::Weights => "weights",
            HashKind::Head => "head",
            HashKind::LabelSet => "label_set",
            HashKind::Calibration => "calibration",
            HashKind::HeadBackboneBinding => "head_backbone_binding",
        }
    }
}

impl std::fmt::Display for HashKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A request that was not answerable as posed.
///
/// Every variant names the check and carries both values compared. `thiserror` renders the same
/// facts in the `Display` string so a log line is as specific as the typed value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Refusal {
    #[error(
        "{which} hash does not match the build: build has {expected}, request/backend has {actual}"
    )]
    HashMismatch {
        which: HashKind,
        expected: String,
        actual: String,
    },

    #[error(
        "slot `{slot}`: {actual} options exceeds the limit of {limit}; the letter-token head slice \
         cannot express the extra option and dropping one would change the question"
    )]
    TooManyOptions {
        slot: String,
        limit: usize,
        actual: usize,
    },

    #[error(
        "slot `{slot}`: {actual} option(s) is below the minimum of {min}; a single-option choice is \
         not a question"
    )]
    TooFewOptions {
        slot: String,
        min: usize,
        actual: usize,
    },

    #[error(
        "slot `{slot}`: option {duplicate_index} duplicates option {first_index} (`{option}`); the \
         question is malformed"
    )]
    DuplicateOption {
        slot: String,
        option: String,
        first_index: usize,
        duplicate_index: usize,
    },

    #[error(
        "slot `{slot}`: option {option_index} is the reserved abstain label `{reserved}`; `noul` is \
         a reserved row in every option set and may not also be a named option"
    )]
    ReservedOptionName {
        slot: String,
        option_index: usize,
        reserved: String,
    },

    #[error("`slots` is empty: there is nothing to answer, so the answer map would be ambiguous")]
    EmptySlots,

    #[error(
        "slot name `{name}` is used at index {duplicate_index} and index {first_index}; the answer \
         map would be ambiguous"
    )]
    DuplicateSlotName {
        name: String,
        first_index: usize,
        duplicate_index: usize,
    },

    #[error("slot at index {index} has an empty name")]
    EmptySlotName { index: usize },

    #[error(
        "slot at index {index} has a name of {actual} bytes, over the cap of {cap}; a slot name is \
         an identifier, and it is refused rather than truncated because the answer map is keyed by \
         it and a truncated key answers a question nobody asked"
    )]
    SlotNameOverCap {
        index: usize,
        cap: usize,
        actual: usize,
    },

    #[error(
        "too many slots: {actual} exceeds the configured cap of {cap}; every fan-out in this \
         runtime is bounded"
    )]
    TooManySlots { cap: usize, actual: usize },

    #[error("slot `{slot}`: bins = {actual} is outside the allowed range [{min}, {max}]")]
    BinsOutOfRange {
        slot: String,
        min: u32,
        max: u32,
        actual: u32,
    },

    #[error(
        "unknown schema_version {actual}; this build supports {supported:?}. Forward-compat \
         guessing is how a field changes meaning silently"
    )]
    UnknownSchemaVersion { supported: Vec<u32>, actual: u32 },

    #[error(
        "context is {actual} bytes, over the cap of {cap}; truncating would move the answer out of \
         the window and line spans would point at the wrong lines"
    )]
    ContextOverCap { cap: usize, actual: usize },

    #[error(
        "declared context_len {declared} does not match the {actual} bytes `context_b64` decoded \
         to; context crosses the boundary as bytes *and* a length, and a disagreement between the \
         two is the signature of a truncated payload that still base64-decodes"
    )]
    ContextLengthMismatch { declared: usize, actual: usize },

    #[error(
        "`context_len` must be a non-negative integer alongside `context_b64`, but it is {found}. \
         The context crosses the boundary as bytes *and* a length, and a payload whose length \
         nobody declared is a payload whose truncation nobody would notice. There is no default"
    )]
    ContextLenMissing { found: String },

    #[error(
        "`context_b64` is a JSON {got}, not a base64 string; the context crosses the wire as \
         `context_b64` plus `context_len` and there is no second accepted form — in particular a \
         JSON array of byte values is no longer read"
    )]
    ContextNotBytes { got: String },

    #[error(
        "`context_b64` is not canonical base64: at offset {offset}, expected {expected}, found \
         {found}. A context is refused rather than best-effort decoded, because a decoder that \
         guesses turns a truncated payload into a shorter context nobody notices"
    )]
    ContextNotBase64 {
        offset: usize,
        expected: String,
        found: String,
    },

    #[error(
        "context has {non_whitespace} non-whitespace characters in {len} bytes; an empty or \
         all-whitespace context has no answer to point at, and would yield a confident letter from \
         nothing"
    )]
    ContextEmpty { len: usize, non_whitespace: usize },

    #[error(
        "context is not valid UTF-8: the first invalid sequence is at byte offset {offset} and is \
         {invalid_len} byte(s) long. The renderer refuses rather than substituting U+FFFD, because \
         a replacement character is exactly the silent rewrite that carrying `context` as bytes \
         exists to prevent"
    )]
    ContextNotUtf8 { offset: usize, invalid_len: usize },

    #[error(
        "context is not a unified diff in the shape this task was trained on: at line {line}, \
         expected {expected}, found {found}. The model was never shown such a context, so its \
         answer would be read from a shape it cannot read"
    )]
    ContextNotUnifiedDiff {
        line: usize,
        expected: String,
        found: String,
    },

    #[error(
        "context's file `{path}` is {language}, a language the pool this task was trained on did \
         not hold (it held {pool:?}); the model was never shown one"
    )]
    ContextLanguageNotInPool {
        path: String,
        language: String,
        pool: Vec<String>,
    },

    /// The shape of [`Refusal::RegisteredHeadMissing`]: the task asked for and the tasks there
    /// are. `available` is empty only for a release that does not record its trained families,
    /// because the release reader refuses a recorded list that is empty.
    #[error(
        "task `{task}` is not one this release was trained on; it trained {available:?}. The \
         model was never shown a request of this task, so its answer would be read from a prompt \
         shape it never saw. A release that does not record its trained families names none and \
         admits no task"
    )]
    TaskNotTrained {
        task: String,
        available: Vec<String>,
    },

    #[error(
        "the rendered prompt is {actual} bytes, over the cap of {cap}; the bound is on the bytes \
         the model sees, not only on those that arrived"
    )]
    RenderedPromptOverCap { cap: usize, actual: usize },

    #[error(
        "slot `{slot}`: option {option_index} is empty or all whitespace; an empty option is an \
         unlabelled letter"
    )]
    EmptyOption { slot: String, option_index: usize },

    #[error(
        "slot `{slot}`: option {option_index} is {actual} bytes, over the cap of {cap}; options are \
         refused, never truncated"
    )]
    OptionTextOverCap {
        slot: String,
        option_index: usize,
        cap: usize,
        actual: usize,
    },

    #[error("question is {actual} bytes, over the cap of {cap}")]
    QuestionOverCap { cap: usize, actual: usize },

    #[error("task identifier is {actual} bytes, over the cap of {cap}")]
    TaskOverCap { cap: usize, actual: usize },

    #[error(
        "`task` is empty or all whitespace; the task identifier is what selects a registered head \
         and what a ledger row attributes an answer to, so there is no default for it"
    )]
    EmptyTask,

    #[error("unknown route `{actual}`; this build supports {supported:?}")]
    UnknownRoute {
        supported: Vec<String>,
        actual: String,
    },

    #[error("unknown slot type `{actual}` on slot `{slot}`; this build supports {supported:?}")]
    UnknownSlotType {
        slot: String,
        supported: Vec<String>,
        actual: String,
    },

    #[error("slot `{slot}` is a `{slot_type}` slot but carries `{field}`")]
    SlotFieldNotAllowed {
        slot: String,
        slot_type: String,
        field: String,
    },

    #[error("slot `{slot}` is a `{slot_type}` slot and requires `{field}`")]
    SlotFieldMissing {
        slot: String,
        slot_type: String,
        field: String,
    },

    #[error(
        "route `registered` requires a head registered for task `{task}`; this build has heads for \
         {available:?}"
    )]
    RegisteredHeadMissing {
        task: String,
        available: Vec<String>,
    },

    #[error(
        "route `registered`: the head for task `{task}` has no rows for slot `{slot}`; it carries \
         rows for {available:?}. Answering the slot on the generic route instead would make \
         `route` advisory and would silently compare the generic route with itself"
    )]
    RegisteredHeadSlotMissing {
        task: String,
        slot: String,
        available: Vec<String>,
    },

    #[error(
        "route `registered`: slot `{slot}` is a `span` slot, whose pointer head ranges over the \
         {rows} line starts of *this* context; a head file has a fixed row count and cannot \
         express a shape that changes per request. Ask for this slot on route `generic`"
    )]
    RegisteredRouteSpanUnsupported { slot: String, rows: usize },

    #[error(
        "no calibration entry for a `{slot_type}` slot with {rows} rows (slot `{slot}`); an \
         uncalibrated answer would carry a `score` and a `conformal_set` that were never fitted"
    )]
    CalibrationEntryMissing {
        slot: String,
        slot_type: String,
        rows: usize,
    },

    #[error("request payload is {actual} bytes, over the cap of {cap}")]
    PayloadOverCap { cap: usize, actual: usize },

    #[error("request is not valid JSON: {detail}")]
    MalformedRequest { detail: String },

    #[error(
        "request carries both `op` and `schema_version`; a line is either a control op or a \
         decision request, never ambiguously both"
    )]
    AmbiguousEnvelope,

    #[error("unknown control op `{actual}`; this build supports {supported:?}")]
    UnknownOp {
        supported: Vec<String>,
        actual: String,
    },
}

impl Refusal {
    /// The machine-readable check name, matching the serialized `kind` tag.
    pub fn kind(&self) -> &'static str {
        match self {
            Refusal::HashMismatch { .. } => "hash_mismatch",
            Refusal::TooManyOptions { .. } => "too_many_options",
            Refusal::TooFewOptions { .. } => "too_few_options",
            Refusal::DuplicateOption { .. } => "duplicate_option",
            Refusal::ReservedOptionName { .. } => "reserved_option_name",
            Refusal::EmptySlots => "empty_slots",
            Refusal::DuplicateSlotName { .. } => "duplicate_slot_name",
            Refusal::EmptySlotName { .. } => "empty_slot_name",
            Refusal::SlotNameOverCap { .. } => "slot_name_over_cap",
            Refusal::TooManySlots { .. } => "too_many_slots",
            Refusal::BinsOutOfRange { .. } => "bins_out_of_range",
            Refusal::UnknownSchemaVersion { .. } => "unknown_schema_version",
            Refusal::ContextOverCap { .. } => "context_over_cap",
            Refusal::ContextLengthMismatch { .. } => "context_length_mismatch",
            Refusal::ContextLenMissing { .. } => "context_len_missing",
            Refusal::ContextNotBytes { .. } => "context_not_bytes",
            Refusal::ContextNotBase64 { .. } => "context_not_base64",
            Refusal::ContextEmpty { .. } => "context_empty",
            Refusal::ContextNotUtf8 { .. } => "context_not_utf8",
            Refusal::ContextNotUnifiedDiff { .. } => "context_not_unified_diff",
            Refusal::ContextLanguageNotInPool { .. } => "context_language_not_in_pool",
            Refusal::TaskNotTrained { .. } => "task_not_trained",
            Refusal::RenderedPromptOverCap { .. } => "rendered_prompt_over_cap",
            Refusal::EmptyOption { .. } => "empty_option",
            Refusal::OptionTextOverCap { .. } => "option_text_over_cap",
            Refusal::QuestionOverCap { .. } => "question_over_cap",
            Refusal::TaskOverCap { .. } => "task_over_cap",
            Refusal::EmptyTask => "empty_task",
            Refusal::UnknownRoute { .. } => "unknown_route",
            Refusal::UnknownSlotType { .. } => "unknown_slot_type",
            Refusal::SlotFieldNotAllowed { .. } => "slot_field_not_allowed",
            Refusal::SlotFieldMissing { .. } => "slot_field_missing",
            Refusal::RegisteredHeadMissing { .. } => "registered_head_missing",
            Refusal::RegisteredHeadSlotMissing { .. } => "registered_head_slot_missing",
            Refusal::RegisteredRouteSpanUnsupported { .. } => "registered_route_span_unsupported",
            Refusal::CalibrationEntryMissing { .. } => "calibration_entry_missing",
            Refusal::PayloadOverCap { .. } => "payload_over_cap",
            Refusal::MalformedRequest { .. } => "malformed_request",
            Refusal::AmbiguousEnvelope => "ambiguous_envelope",
            Refusal::UnknownOp { .. } => "unknown_op",
        }
    }
}

/// Why the backend could not serve, as distinct from why the request was not answerable.
///
/// An absent backend is this, never an answer. There is no code path from a `BackendError` to a
/// letter, a bin, a span or a `noul`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendError {
    #[error(
        "no model backend is configured; `{name}` is a deterministic reference, not a model, and \
         will not serve unless explicitly enabled"
    )]
    ReferenceBackendNotEnabled { name: String },

    #[error("no model backend is available: {detail}")]
    Unavailable { detail: String },

    #[error("backend is poisoned: {reason}")]
    Poisoned { reason: String },

    #[error("backend rebuild failed after poison: {detail}")]
    RebuildFailed { detail: String },

    #[error(
        "slot-isolation violated: the state buffer hash changed across a readonly decode \
         ({before} -> {after}); slot {slot_index}'s answer would depend on the previous slot's"
    )]
    ReadonlyViolated {
        slot_index: usize,
        before: String,
        after: String,
    },

    #[error("backend returned {actual} logit rows for slot `{slot}`, expected {expected}")]
    LogitShapeMismatch {
        slot: String,
        expected: usize,
        actual: usize,
    },

    #[error("backend returned a non-finite logit at row {row} for slot `{slot}`")]
    NonFiniteLogit { slot: String, row: usize },

    #[error("backend returned the wrong logit kind for slot `{slot}`: expected {expected}")]
    LogitKindMismatch { slot: String, expected: String },

    #[error("prefill failed: {detail}")]
    PrefillFailed { detail: String },

    #[error("decode failed: {detail}")]
    DecodeFailed { detail: String },

    #[error("request exceeded the {limit_ms} ms deadline")]
    DeadlineExceeded { limit_ms: u64 },

    #[error("the runtime is at its concurrency limit of {limit} in-flight requests")]
    Overloaded { limit: usize },
}

impl BackendError {
    pub fn kind(&self) -> &'static str {
        match self {
            BackendError::ReferenceBackendNotEnabled { .. } => "reference_backend_not_enabled",
            BackendError::Unavailable { .. } => "unavailable",
            BackendError::Poisoned { .. } => "poisoned",
            BackendError::RebuildFailed { .. } => "rebuild_failed",
            BackendError::ReadonlyViolated { .. } => "readonly_violated",
            BackendError::LogitShapeMismatch { .. } => "logit_shape_mismatch",
            BackendError::NonFiniteLogit { .. } => "non_finite_logit",
            BackendError::LogitKindMismatch { .. } => "logit_kind_mismatch",
            BackendError::PrefillFailed { .. } => "prefill_failed",
            BackendError::DecodeFailed { .. } => "decode_failed",
            BackendError::DeadlineExceeded { .. } => "deadline_exceeded",
            BackendError::Overloaded { .. } => "overloaded",
        }
    }
}

/// Everything that can stop a request short of an answer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QdError {
    #[error("refused: {0}")]
    Refused(#[from] Refusal),
    #[error("backend: {0}")]
    Backend(#[from] BackendError),
}
