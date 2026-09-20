//! The typed request and answer of `docs/schema-api.md`.
//!
//! This module is the contract's Rust form. Wire parsing and validation live in [`crate::wire`];
//! everything here is the shape both the training renderer and the runtime renderer must agree on.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::context::Context;
use crate::refusal::{BackendError, Refusal};

/// Schema versions this build understands. Anything else is
/// [`Refusal::UnknownSchemaVersion`] — never a best-effort read.
pub const SUPPORTED_SCHEMA_VERSIONS: &[u32] = &[1];

/// Named options a `choice` slot may carry, per `docs/schema-api.md`:
/// "one of k <= 16 named options, or `noul`".
pub const MAX_OPTIONS: usize = 16;
/// A `choice` slot is a question only if there is something to choose between.
pub const MIN_OPTIONS: usize = 2;
/// `bins < 2` is "not a question"; `bins > 16` exceeds the letter slice.
pub const MIN_BINS: u32 = 2;
pub const MAX_BINS: u32 = 16;

/// Bytes a slot name may occupy, UTF-8, measured before escaping.
///
/// # Why this exists
///
/// Every other caller-supplied string on this wire has a byte cap — context, question, task and
/// option text all do. A slot name had none: [`crate::wire::parse_slot`] checked it non-empty and
/// stopped there, so a 4096-byte name parsed cleanly and [`crate::wire::MAX_PAYLOAD_BYTES`] was
/// the only thing bounding it. That made [`crate::wire::PayloadBudget::total`] a *lower* bound on
/// the worst-case request rather than the worst case, which is the wrong shape for a number the
/// payload cap decision rests on. `GAP-RT-SLOT-NAME-UNCAPPED`.
///
/// # Why 256, and why it refuses nothing that exists
///
/// A slot name is an **identifier** — a key in the answer map and one line of the rendered prompt
/// — not prose. This repository has already priced an identifier: `RenderCaps::max_task_bytes` is
/// 256, against 512 for option text and 4096 for the question. Reusing that number is a decision
/// already made here rather than a new one invented for this field.
///
/// Measured before choosing it: the longest slot name anywhere in this repository — every fixture,
/// test and producer — is **8 bytes** (`severity`, `evidence`). The cap is 32x that, so no request
/// any lane can build today is refused by it.
///
/// # Both sides state this number
///
/// The cap is enforced on all four readers of a slot name, so no lane refuses what another
/// accepts: [`check_slot_name`] guards the request (called by [`crate::wire::parse_slot`], the sole
/// constructor of [`SlotSpec`]) and the prompt (called by `render::slot_suffix`), and the answer
/// map's keys are checked by [`AnswerEnvelope`]'s `Deserialize` here and by
/// `qd_wire.answer.parse_answer_envelope` on the Python side. Python does not transcribe the
/// number: `qd_wire.contract.MAX_SLOT_NAME_BYTES` is generated from this line by
/// `qd_wire.rust_source`, and `python/tests/test_wire_contract_matches_rust.py` re-derives and
/// compares it on every run, the way `ESCAPE_WORST_CASE_GROWTH` and `HEX_ESCAPED` are pinned.
pub const MAX_SLOT_NAME_BYTES: usize = 256;

/// What is wrong with a slot name, independent of where it was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotNameFault {
    /// Empty, or nothing but whitespace.
    Empty,
    /// Longer than [`MAX_SLOT_NAME_BYTES`].
    OverCap { cap: usize, actual: usize },
}

/// The slot-name rule itself, with no locus: `None` when the name is acceptable.
///
/// A slot name is read in two different places — as `slots[i].name` on a request, and as a key of
/// the `slots` map on an answer — and those two want different error text. The *rule* must still be
/// one thing, because two copies of a threshold is how `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS` and
/// `GAP-RT-WIRE-CONTEXT-ENCODING` each came to exist. So the predicate lives here alone and each
/// locus formats its own message from the fault it returns.
pub fn slot_name_fault(name: &str) -> Option<SlotNameFault> {
    // `crate::is_blank`, not `str::trim`: `python/qd_data/schema.py` asks this with `str.strip()`.
    // See `crate::is_wire_whitespace` for the four codepoints the two notions disagree about.
    if crate::is_blank(name) {
        return Some(SlotNameFault::Empty);
    }
    if name.len() > MAX_SLOT_NAME_BYTES {
        return Some(SlotNameFault::OverCap {
            cap: MAX_SLOT_NAME_BYTES,
            actual: name.len(),
        });
    }
    None
}

/// [`slot_name_fault`] at the request locus: the refusal a caller gets for `slots[index].name`.
///
/// The over-cap refusal carries the slot's **index and byte counts, never the name**. The name is
/// by definition oversized and caller-controlled, and echoing it back into a refusal message would
/// let a request choose the size of its own error — a small amplification, but the kind that is
/// only ever free until it is not.
pub fn check_slot_name(name: &str, index: usize) -> Result<(), Refusal> {
    match slot_name_fault(name) {
        None => Ok(()),
        Some(SlotNameFault::Empty) => Err(Refusal::EmptySlotName { index }),
        Some(SlotNameFault::OverCap { cap, actual }) => {
            Err(Refusal::SlotNameOverCap { index, cap, actual })
        }
    }
}

/// The reserved abstain label. `docs/schema-api.md`: "a reserved letter present in **every** option
/// set". It occupies one head row beyond the named options, so a `choice` slot at the maximum
/// `MAX_OPTIONS` reads `MAX_OPTIONS + 1` rows. See `RESERVED_NOUL_ROWS`.
pub const NOUL_LABEL: &str = "noul";
/// Rows the reserved abstain label adds to every slot's head slice.
pub const RESERVED_NOUL_ROWS: usize = 1;

/// Which of the two routes answers this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// Option text is in the prompt; the answer is one token over the letter slice.
    Generic,
    /// A head file, hash-bound to the backbone, takes a single GEMV over pooled features.
    Registered,
}

impl Route {
    pub fn as_str(self) -> &'static str {
        match self {
            Route::Generic => "generic",
            Route::Registered => "registered",
        }
    }

    pub fn supported() -> Vec<String> {
        vec!["generic".to_string(), "registered".to_string()]
    }
}

/// The three slot kinds, without their payloads. Used where only the kind matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotKind {
    Choice,
    Score,
    Span,
}

impl SlotKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SlotKind::Choice => "choice",
            SlotKind::Score => "score",
            SlotKind::Span => "span",
        }
    }

    pub fn supported() -> Vec<String> {
        vec![
            "choice".to_string(),
            "score".to_string(),
            "span".to_string(),
        ]
    }
}

/// One validated slot of a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotSpec {
    Choice { name: String, options: Vec<String> },
    Score { name: String, bins: u32 },
    Span { name: String },
}

impl SlotSpec {
    pub fn name(&self) -> &str {
        match self {
            SlotSpec::Choice { name, .. } | SlotSpec::Score { name, .. } | SlotSpec::Span { name } => {
                name
            }
        }
    }

    pub fn kind(&self) -> SlotKind {
        match self {
            SlotSpec::Choice { .. } => SlotKind::Choice,
            SlotSpec::Score { .. } => SlotKind::Score,
            SlotSpec::Span { .. } => SlotKind::Span,
        }
    }

    /// Head rows this slot reads, including the reserved `noul` row.
    ///
    /// `span` depends on the context, so it is not answered here; see
    /// [`crate::answer::span_rows`].
    pub fn answer_rows(&self) -> Option<usize> {
        match self {
            SlotSpec::Choice { options, .. } => Some(options.len() + RESERVED_NOUL_ROWS),
            SlotSpec::Score { bins, .. } => Some(*bins as usize + RESERVED_NOUL_ROWS),
            SlotSpec::Span { .. } => None,
        }
    }
}

/// Hashes a caller may pin. Any field present must equal the build's, or the request is refused.
/// A caller that pins nothing still gets the construction-time check the runtime ran on itself.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HashExpectation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_set_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_hash: Option<String>,
}

impl HashExpectation {
    pub fn is_empty(&self) -> bool {
        self.tokenizer_hash.is_none()
            && self.weight_hash.is_none()
            && self.head_hash.is_none()
            && self.label_set_hash.is_none()
            && self.calibration_hash.is_none()
    }
}

/// A validated request. Constructing one is only possible through [`crate::wire::validate`], so
/// holding one is proof that every refusal in `docs/schema-api.md` has already been checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRequest {
    pub schema_version: u32,
    pub task: String,
    pub context: Context,
    pub question: String,
    pub slots: Vec<SlotSpec>,
    pub route: Route,
    pub expect: HashExpectation,
}

impl DecisionRequest {
    /// SHA-256 over every field that can change the answer. Seeds the option permutation, so the
    /// second pass is reproducible for a given request rather than run-to-run random.
    pub fn digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"qd-request-v1\x00");
        h.update(self.schema_version.to_le_bytes());
        h.update(self.task.as_bytes());
        h.update([0u8]);
        h.update(self.question.as_bytes());
        h.update([0u8]);
        h.update(self.context.digest());
        h.update(self.route.as_str().as_bytes());
        for slot in &self.slots {
            h.update([0u8]);
            h.update(slot.name().as_bytes());
            h.update(slot.kind().as_str().as_bytes());
            match slot {
                SlotSpec::Choice { options, .. } => {
                    for o in options {
                        h.update([1u8]);
                        h.update(o.as_bytes());
                    }
                }
                SlotSpec::Score { bins, .. } => h.update(bins.to_le_bytes()),
                SlotSpec::Span { .. } => {}
            }
        }
        h.finalize().into()
    }
}

/// A line span in the context, 1-based and inclusive, matching the example in
/// `docs/schema-api.md`.
///
/// # `1 <= start_line <= end_line` is a term of the format
///
/// It was not, and the difference mattered. The *producer* refused a backwards span —
/// [`crate::answer`] abstains when the end pointer decodes below the start one, on the stated
/// ground that a span running backwards "is not a low-confidence span, it is not a span". But the
/// invariant lived only in that one function: this struct derived `Deserialize` with no ordering
/// check, so both *readers* accepted `{"start_line": 47, "end_line": 41}` without complaint, and
/// the type carried no such guarantee for anything downstream to rely on.
///
/// That is the "one fact, two spellings" shape this crate has now met four times, in its quietest
/// form: a rule that is true of everything the system emits and enforced nowhere it is read. Any
/// consumer computing span overlap or slicing context lines was trusting a producer-side check,
/// and a second producer — a registered head, a different backend, a hand-written fixture — would
/// satisfy the format while breaking that consumer. `GAP-XLANG-SPAN-BOUNDS-UNPINNED`.
///
/// So the invariant now belongs to the type. Both bounds are enforced, on both sides, because both
/// are already true of every span this repository produces and the second is only the first one
/// written down: `start_line: 0` contradicts the "1-based" the doc comment above has always
/// claimed. Measured before tightening: every span in `fixtures/wire/` and in every test is inside
/// both bounds, minimum `start_line` 1, so nothing legal today is refused.
///
/// `qd_wire.answer.SpanValue` refuses the same two shapes, and
/// `python/tests/test_wire_answer_parser.py` pins it from that side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "SpanValueWire")]
pub struct SpanValue {
    pub start_line: usize,
    pub end_line: usize,
}

/// The on-the-wire shape of [`SpanValue`], before its bounds are checked.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpanValueWire {
    start_line: usize,
    end_line: usize,
}

/// Why a `{start_line, end_line}` pair is not a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpanBoundsError {
    #[error(
        "span runs backwards: start_line {start_line} is after end_line {end_line}. A backwards \
         span is not a low-confidence span, it is not a span; it is refused rather than silently \
         reordered into a plausible-looking answer"
    )]
    Backwards { start_line: usize, end_line: usize },
    #[error(
        "span starts at line 0, but a span is 1-based and inclusive, so line 0 does not exist"
    )]
    NotOneBased,
}

impl TryFrom<SpanValueWire> for SpanValue {
    type Error = SpanBoundsError;

    fn try_from(wire: SpanValueWire) -> Result<Self, Self::Error> {
        let SpanValueWire {
            start_line,
            end_line,
        } = wire;
        if start_line == 0 {
            return Err(SpanBoundsError::NotOneBased);
        }
        if start_line > end_line {
            return Err(SpanBoundsError::Backwards {
                start_line,
                end_line,
            });
        }
        Ok(SpanValue {
            start_line,
            end_line,
        })
    }
}

/// What a slot answered with. Never present when the slot abstained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SlotValue {
    Choice(String),
    Score(u32),
    Span(SpanValue),
}

/// The split-conformal set on calibrated scores. `null` for `span`, per the contract's example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConformalSet {
    Choices(Vec<String>),
    Scores(Vec<u32>),
}

/// One slot's answer: `{value, conformal_set, score, noul, degraded}`.
///
/// `noul` is the model abstaining. It is reachable only from this struct — no refusal can produce
/// one, because no refusal can produce a `SlotAnswer` at all.
///
/// # `value` is absent if and only if `noul`
///
/// The two fields are **one fact spelled twice**, and that is the shape this repo has now shipped
/// three times with both suites green: one name, two quantities, nothing asserting agreement
/// (`GAP-RT-WIRE-CONTEXT-ENCODING`, `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`). A caller that
/// branches on `noul` and a caller that branches on `value == null` must never disagree, so the
/// biconditional is enforced rather than described:
///
/// * **producing** — the only constructors are [`SlotAnswer::answered`] and
///   [`SlotAnswer::abstained`], and each sets both fields together. `crate::answer` builds every
///   slot through them, so no code path in this crate can emit a contradictory pair;
/// * **parsing** — the hand-written [`Deserialize`] below refuses `{"value": …, "noul": true}` and
///   `{"value": null, "noul": false}` outright. That is the half the other lane's parser is
///   checked against, via the golden corpus in `fixtures/wire/`.
///
/// `docs/schema-api.md` named this in its own "what this document still does not specify" list —
/// *"no field-presence rules — in particular whether `value` is absent iff `noul`"*. It is now
/// specified, here, in the type.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SlotAnswer {
    /// `None` exactly when `noul` is true. Serialized as an explicit `null`, never omitted: a
    /// missing key and a null key would be two spellings of one state.
    pub value: Option<SlotValue>,
    pub conformal_set: Option<ConformalSet>,
    /// Calibrated confidence — a margin on temperature-scaled scores, never entropy.
    pub score: f64,
    pub noul: bool,
    pub degraded: bool,
}

impl SlotAnswer {
    /// An answered slot. `noul` is false by construction, so the caller cannot pass a value and an
    /// abstention together.
    pub fn answered(value: SlotValue, conformal_set: Option<ConformalSet>, score: f64) -> Self {
        Self {
            value: Some(value),
            conformal_set,
            score,
            noul: false,
            degraded: false,
        }
    }

    /// The abstention. Carries no value and no conformal set; the score is the calibrated score of
    /// the row that won, so a caller can see *how* close the abstention was.
    pub fn abstained(score: f64, degraded: bool) -> Self {
        Self {
            value: None,
            conformal_set: None,
            score,
            noul: true,
            degraded,
        }
    }

    /// Why this answer is not a legal one, or `None`.
    ///
    /// Exported so the fixture generator and the wire parser state the rule once. It is a total
    /// function over the struct rather than an assertion, because a check that panics is a check
    /// that cannot be reported.
    pub fn contradiction(&self) -> Option<String> {
        match (self.value.is_some(), self.noul) {
            (true, true) => Some(
                "carries a `value` and `noul`: true; `noul` means the model declined to answer, \
                 so there is nothing for `value` to hold"
                    .to_string(),
            ),
            (false, false) => Some(
                "carries no `value` and `noul`: false; an answer with neither a value nor an \
                 abstention says nothing, and a caller reading `value` would see the same bytes \
                 as an abstention without the flag that names it"
                    .to_string(),
            ),
            _ => None,
        }
    }
}

/// Hand-written so the `value`/`noul` biconditional is checked at the boundary rather than trusted
/// across it. A derived `Deserialize` accepted both contradictory shapes; see
/// `tests/answer_envelope_contract.rs`, which fails against the derived one.
impl<'de> Deserialize<'de> for SlotAnswer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// The field set, with no interpretation. `deny_unknown_fields` matches the request side's
        /// [`crate::wire`] rule: unknown keys are refused, not dropped. Before this, the two
        /// directions of one protocol had opposite rules and `docs/schema-api.md` said so.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            #[serde(default)]
            value: Option<SlotValue>,
            #[serde(default)]
            conformal_set: Option<ConformalSet>,
            score: f64,
            noul: bool,
            degraded: bool,
        }

        let wire = Wire::deserialize(deserializer)?;
        let answer = SlotAnswer {
            value: wire.value,
            conformal_set: wire.conformal_set,
            score: wire.score,
            noul: wire.noul,
            degraded: wire.degraded,
        };
        match answer.contradiction() {
            Some(why) => Err(serde::de::Error::custom(format!("slot answer {why}"))),
            None => Ok(answer),
        }
    }
}

/// The successful answer envelope.
///
/// Five keys reach a caller, not the two `docs/schema-api.md`'s example showed: the `status`
/// discriminant contributed by [`Response`], plus `schema_version`, `backend`, `degraded` and
/// `slots`. `backend` is the one a caller written from the old example would have missed, and it
/// is the field that says an answer came from `reference-deterministic-v1` rather than from a
/// model. `GAP-RT-SPEC-ANSWER-ENVELOPE-FIELDS`.
///
/// `degraded` here is the **envelope-level** flag, and [`crate::answer`] mirrors it onto every
/// slot: the contract describes `degraded` per slot, but its stated meaning — a rebuilt-after-
/// poison runtime, or a backend that is not a model — is a property of the answer as a whole.
/// The two are therefore always equal, which `tests/answer_envelope_contract.rs` asserts so a
/// caller may rely on it.
///
/// Deserializing refuses unknown keys, matching [`crate::wire`] on the request side. Note that
/// this means the **whole reply** must be read as a [`Response`], not as an `AnswerEnvelope`: the
/// reply carries `status`, which belongs to the enum and not to this struct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerEnvelope {
    pub schema_version: u32,
    /// Which backend answered, by name. A reference backend says so here in every answer.
    pub backend: String,
    /// True if any slot is degraded — a rebuilt-after-poison runtime, or a reference backend.
    pub degraded: bool,
    #[serde(deserialize_with = "deserialize_slot_map")]
    pub slots: BTreeMap<String, SlotAnswer>,
}

/// Read the answer's `slots` map, holding its keys to the same rule the request side holds
/// `slots[i].name` to.
///
/// Without this the two readers of one format disagree: `qd_wire.answer.parse_answer_envelope`
/// refuses an over-cap key, and a derived `BTreeMap<String, _>` here would accept it. A cap that
/// one side enforces and the other does not is not a cap, it is a disagreement — the precise defect
/// `GAP-RT-SLOT-NAME-UNCAPPED` was left open to avoid creating in one direction, so it must not be
/// created in the other.
///
/// The key is named in the error only when it is short enough to be worth naming; an over-cap key
/// is reported by its byte count, for the reason given on [`check_slot_name`].
fn deserialize_slot_map<'de, D>(de: D) -> Result<BTreeMap<String, SlotAnswer>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;

    let map = BTreeMap::<String, SlotAnswer>::deserialize(de)?;
    for name in map.keys() {
        match slot_name_fault(name) {
            None => {}
            Some(SlotNameFault::Empty) => {
                return Err(D::Error::custom(
                    "the answer's `slots` map has an empty key; a slot answer that cannot be keyed \
                     cannot be matched to the slot that was asked",
                ))
            }
            Some(SlotNameFault::OverCap { cap, actual }) => {
                return Err(D::Error::custom(format!(
                    "a key of the answer's `slots` map is {actual} bytes, over the slot-name cap \
                     of {cap}; no request carrying such a name can be accepted, so no answer may \
                     carry one back"
                )))
            }
        }
    }
    Ok(map)
}

/// The refusal envelope. Deliberately shares no field name with [`AnswerEnvelope`] beyond
/// `schema_version`: no `slots`, no `value`, no `score`, no `conformal_set`, no `noul`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefusalEnvelope {
    pub schema_version: u32,
    pub refusal: Refusal,
    /// The `Display` rendering of the refusal, so a human reading a log gets both compared values
    /// without re-deriving them from the typed fields.
    pub message: String,
}

/// The backend-failure envelope. An absent, poisoned or overloaded backend lands here — never in
/// [`AnswerEnvelope`], and never as a `noul`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorEnvelope {
    pub schema_version: u32,
    pub error: BackendError,
    pub message: String,
}

/// What a caller gets back. Three variants, matched exhaustively: an answer, a refusal, or a
/// backend failure. The type system is what stops a refusal from being read as an abstention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Response {
    Ok(AnswerEnvelope),
    Refused(RefusalEnvelope),
    Error(ErrorEnvelope),
}

impl Response {
    pub fn refused(refusal: Refusal) -> Self {
        let message = refusal.to_string();
        Response::Refused(RefusalEnvelope {
            schema_version: SUPPORTED_SCHEMA_VERSIONS[0],
            refusal,
            message,
        })
    }

    pub fn failed(error: BackendError) -> Self {
        let message = error.to_string();
        Response::Error(ErrorEnvelope {
            schema_version: SUPPORTED_SCHEMA_VERSIONS[0],
            error,
            message,
        })
    }

    pub fn status(&self) -> &'static str {
        match self {
            Response::Ok(_) => "ok",
            Response::Refused(_) => "refused",
            Response::Error(_) => "error",
        }
    }
}

/// How a caller may read a response.
///
/// This exists so the "a refusal must not be confusable with `noul`" property is testable as a
/// total function rather than as prose: `tests/refusal_is_not_noul.rs` asserts that no refusal and
/// no backend error maps to [`CallerReading::ModelAbstained`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerReading {
    /// At least one slot carries a value.
    ModelAnswered,
    /// Every slot abstained. The model was asked and declined.
    ModelAbstained,
    /// The request was not answerable as posed. The model was never asked.
    RequestRefused,
    /// The backend could not serve. The model was never asked.
    BackendFailed,
}

impl Response {
    pub fn caller_reading(&self) -> CallerReading {
        match self {
            Response::Ok(env) => {
                if env.slots.values().all(|s| s.noul) {
                    CallerReading::ModelAbstained
                } else {
                    CallerReading::ModelAnswered
                }
            }
            Response::Refused(_) => CallerReading::RequestRefused,
            Response::Error(_) => CallerReading::BackendFailed,
        }
    }
}
