//! The JSON wire form, and the only way to obtain a validated [`DecisionRequest`].
//!
//! Holding a `DecisionRequest` is proof that every refusal in `docs/schema-api.md` has already been
//! checked, which is only true because this module is the sole constructor. It is hand-written
//! against [`serde_json::Value`] rather than derived, because a derived deserializer reports
//! *"missing field `bins`"* where the contract requires a typed refusal that names the check and
//! **carries both values compared**.
//!
//! # `context` on the wire
//!
//! `docs/schema-api.md`, *"The wire encoding of `context` — exactly one form"*: `context_b64`
//! (base64, ASCII) plus a required `context_len` (the decoded byte count). They must agree, or it
//! is a typed refusal. There is no second accepted form, and this module accepts no second form:
//!
//! * `context_b64` absent or not a JSON string -> [`Refusal::ContextNotBytes`];
//! * not canonical base64 -> [`Refusal::ContextNotBase64`];
//! * `context_len` absent -> [`Refusal::ContextLenMissing`];
//! * `context_len` present but disagreeing -> [`Refusal::ContextLengthMismatch`], naming both
//!   numbers.
//!
//! A payload carrying the **old** integer-array `context` field is refused by [`known_keys`] as an
//! unknown field, whose message lists `context_b64` — a typed refusal that names the replacement,
//! not a lenient read. `GAP-RT-WIRE-CONTEXT-ENCODING` closed 2026-09-19; the decoding itself lives
//! in [`crate::context`], which is the single owner of the encoding, and
//! `tests/wire_context_crosslang.rs` drives the real Python encoder into it rather than assuming
//! the two lanes agree.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::context::{self, Context};
use crate::refusal::Refusal;
use crate::schema::{
    check_slot_name, DecisionRequest, HashExpectation, Route, SlotKind, SlotSpec, MAX_BINS,
    MAX_OPTIONS, MAX_SLOT_NAME_BYTES, MIN_BINS, MIN_OPTIONS, NOUL_LABEL,
    SUPPORTED_SCHEMA_VERSIONS,
};

/// Bounded fan-out: a request with an unbounded number of slots is an unbounded number of decode
/// passes over one prefill. Matches `MAX_SLOTS` in `python/qd_data/schema.py`.
pub const MAX_SLOTS: usize = 32;

/// The cap on one JSON line.
///
/// Derived, not guessed. A maximal context is `RenderCaps::DEFAULT.max_context_bytes` = 131072
/// bytes, which as base64 is `ceil(131072 / 3) * 4` = 174764 characters — the whole point of
/// choosing base64 over an integer array, which would have needed up to 524288. The remainder
/// covers the question (4096), the task (256), the slot list, and the envelope. A line over this
/// cap is [`Refusal::PayloadOverCap`]; it is never read into memory whole first.
///
/// **The slot list is what this cap does not cover, and that is now arithmetic rather than a
/// remark.** [`payload_budget`] computes what a request legal at every other documented limit at
/// once would weigh, and `tests/wire_refusals.rs` asserts that it **exceeds** this cap while the
/// context side alone fits inside it. So the answer to "is this cap still derived correctly now
/// that context is base64?" is *no, and it never was*: it is derived from the context term only.
///
/// Such a request is refused by this cap rather than by [`Refusal::OptionTextOverCap`], so the
/// refusal a caller sees names the payload rather than the option that caused it. The number is
/// **left alone deliberately**: raising it is a joint decision with the Python lane's
/// `RenderCaps`, and shrinking it would refuse payloads that are legal today. What has changed is
/// that the shortfall is a failing assertion away from anyone who edits either side, instead of a
/// comment that went stale once already. `GAP-RT-PAYLOAD-CAP-VS-MAXIMAL-SLOTS`.
pub const MAX_PAYLOAD_BYTES: usize = 1_572_864;

/// Worst-case bytes-out per byte-in for **JSON string escaping**: a NUL input byte is emitted as
/// the six characters `\u0000`.
///
/// Deliberately not `render::ESCAPE_WORST_CASE_GROWTH`, which is the same number for a different
/// reason — that one bounds the *prompt* escaper, this one bounds the *JSON* encoder, and they are
/// free to diverge. Sharing a constant between them would be a coincidence maintained by hand.
pub const JSON_ESCAPE_WORST_CASE_GROWTH: usize = 6;

/// Bytes a base64 encoding of `n` bytes occupies: canonical, padded, unwrapped.
const fn base64_len(n: usize) -> usize {
    n.div_ceil(3) * 4
}

/// What a request legal at **every** documented limit at once weighs on the wire, term by term.
///
/// This exists because [`MAX_PAYLOAD_BYTES`]'s derivation lived in a comment and went stale: it
/// was written against the retired integer-array `context` encoding, and when the encoding became
/// base64 the number stayed while its justification quietly stopped describing it. A derivation
/// that is a function cannot go stale without failing a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadBudget {
    /// `context_b64`. Base64's alphabet needs no JSON escaping, so this term is exact.
    pub context_b64: usize,
    /// `question`, at [`crate::render::RenderCaps::max_question_bytes`], worst-case escaped.
    pub question: usize,
    /// `task`, at `max_task_bytes`, worst-case escaped.
    pub task: usize,
    /// [`MAX_SLOTS`] slots, each carrying a name at [`MAX_SLOT_NAME_BYTES`] and [`MAX_OPTIONS`]
    /// options at `max_option_bytes`, all worst-case escaped, plus the JSON punctuation each one
    /// needs.
    pub slot_list: usize,
    /// The envelope's own key names, braces, commas and the `context_len` integer.
    pub envelope: usize,
    /// Fields this budget **could not** count, because nothing caps them.
    ///
    /// Carried rather than omitted: a budget that quietly left a term out would be the same
    /// mistake as the comment it replaces. While this list is non-empty [`PayloadBudget::total`] is
    /// a **lower bound** on the worst case rather than the worst case, and the field exists so that
    /// distinction is a value a test can read instead of a caveat in prose.
    ///
    /// It is currently empty. Its one entry was `slots[].name`, retired when
    /// [`MAX_SLOT_NAME_BYTES`] gave that field a cap on every side that reads it
    /// (`GAP-RT-SLOT-NAME-UNCAPPED`). The field stays, because the next uncapped term should land
    /// here rather than silently not be counted.
    pub uncapped: &'static [&'static str],
}

impl PayloadBudget {
    /// The whole request. A lower bound whenever [`PayloadBudget::uncapped`] is non-empty.
    pub fn total(&self) -> usize {
        self.context_b64 + self.question + self.task + self.slot_list + self.envelope
    }

    /// Everything except the slot list — the terms [`MAX_PAYLOAD_BYTES`]'s comment was derived
    /// from.
    pub fn without_slot_list(&self) -> usize {
        self.total() - self.slot_list
    }

    /// Whether a request at every limit at once is inside [`MAX_PAYLOAD_BYTES`]. It is not.
    pub fn fits(&self) -> bool {
        self.total() <= MAX_PAYLOAD_BYTES
    }
}

/// Compute [`PayloadBudget`] for a set of render caps.
pub fn payload_budget(caps: &crate::render::RenderCaps) -> PayloadBudget {
    // `{"name":"…","type":"choice","options":[…]}` plus the comma joining it to the next slot.
    // The option strings' own quotes and separating commas are three bytes each.
    const SLOT_PUNCTUATION: usize = 48;
    const OPTION_PUNCTUATION: usize = 3;
    // `{"schema_version":1,"task":"","context_b64":"","context_len":0,"question":"","slots":[],
    //  "route":"registered","expect":{…}}` — the key names, the punctuation, and room for the
    // five optional hex digests an `expect` block may pin.
    const ENVELOPE_KEYS_AND_EXPECT: usize = 1024;

    PayloadBudget {
        context_b64: base64_len(caps.max_context_bytes) + 2,
        question: caps.max_question_bytes * JSON_ESCAPE_WORST_CASE_GROWTH + 2,
        task: caps.max_task_bytes * JSON_ESCAPE_WORST_CASE_GROWTH + 2,
        slot_list: MAX_SLOTS
            * (SLOT_PUNCTUATION
                + MAX_SLOT_NAME_BYTES * JSON_ESCAPE_WORST_CASE_GROWTH
                + MAX_OPTIONS
                    * (caps.max_option_bytes * JSON_ESCAPE_WORST_CASE_GROWTH
                        + OPTION_PUNCTUATION)),
        envelope: ENVELOPE_KEYS_AND_EXPECT,
        // Empty, and that is the point: every term of a request is now counted, so `total()` is
        // the worst case rather than a lower bound on it. `slots[].name` was the one entry until
        // `MAX_SLOT_NAME_BYTES` capped it on every side that reads a slot name.
        uncapped: &[],
    }
}

/// The control ops the line protocol accepts alongside decision requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlOp {
    /// Lifecycle and identity, including the measured cold start.
    Status,
    /// Drop the warm runtime now. The next request pays a cold start.
    Evict,
    /// Stop accepting connections and exit once in-flight requests finish.
    Shutdown,
    /// Liveness only. Touches no backend and cannot warm one.
    Ping,
}

impl ControlOp {
    pub fn as_str(self) -> &'static str {
        match self {
            ControlOp::Status => "status",
            ControlOp::Evict => "evict",
            ControlOp::Shutdown => "shutdown",
            ControlOp::Ping => "ping",
        }
    }

    pub fn supported() -> Vec<String> {
        ["status", "evict", "shutdown", "ping"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

/// One line of the protocol: a decision request, or a control op. Never ambiguously both.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Request(Box<DecisionRequest>),
    Op(ControlOp),
}

/// Parse one protocol line.
pub fn parse_line(line: &[u8]) -> Result<Incoming, Refusal> {
    if line.len() > MAX_PAYLOAD_BYTES {
        return Err(Refusal::PayloadOverCap {
            cap: MAX_PAYLOAD_BYTES,
            actual: line.len(),
        });
    }
    let value: Value = serde_json::from_slice(line).map_err(|e| Refusal::MalformedRequest {
        detail: e.to_string(),
    })?;
    let Value::Object(obj) = value else {
        return Err(Refusal::MalformedRequest {
            detail: format!(
                "top level is a JSON {}, not an object",
                json_type_name(&value)
            ),
        });
    };

    let has_op = obj.contains_key("op");
    let has_version = obj.contains_key("schema_version");
    if has_op && has_version {
        return Err(Refusal::AmbiguousEnvelope);
    }
    if has_op {
        return parse_op(&obj).map(Incoming::Op);
    }
    validate(&obj).map(|r| Incoming::Request(Box::new(r)))
}

fn parse_op(obj: &Map<String, Value>) -> Result<ControlOp, Refusal> {
    known_keys(obj, &["op"])?;
    let raw = obj.get("op").and_then(Value::as_str).ok_or_else(|| {
        Refusal::MalformedRequest {
            detail: "`op` must be a string".to_string(),
        }
    })?;
    match raw {
        "status" => Ok(ControlOp::Status),
        "evict" => Ok(ControlOp::Evict),
        "shutdown" => Ok(ControlOp::Shutdown),
        "ping" => Ok(ControlOp::Ping),
        other => Err(Refusal::UnknownOp {
            supported: ControlOp::supported(),
            actual: other.to_string(),
        }),
    }
}

const REQUEST_KEYS: &[&str] = &[
    "schema_version",
    "task",
    // The one accepted context form. A payload carrying the retired integer-array `context` field
    // is refused by `known_keys`, whose message lists these names.
    "context_b64",
    "context_len",
    "question",
    "slots",
    "route",
    "expect",
    // Accepted and ignored: `python/qd_data/schema.py` puts them on the wire, and they steer the
    // *training* option shuffle only. The serving path renders with no shuffle, so they cannot
    // change the prompt. Rejecting them would make a request the training lane emits unservable.
    "example_id",
    "metadata",
];

/// Validate a wire object into a [`DecisionRequest`].
///
/// Every check in `docs/schema-api.md`'s refusal table is here, in an order chosen so the refusal a
/// caller sees names the outermost thing that is wrong: version, then envelope shape, then the
/// context, then the slot list, then each slot.
pub fn validate(obj: &Map<String, Value>) -> Result<DecisionRequest, Refusal> {
    // The retired wire form, named before the generic unknown-field check so a caller still
    // sending it is told what replaced it rather than that a field it has always sent is unknown.
    // Both lanes refuse this under the same identifier; `python/qd_data/schema.py::from_wire`
    // raises `ContextNotBytesRefusal`, whose `check` is this variant's `kind()`.
    if obj.contains_key("context") {
        return Err(Refusal::ContextNotBytes {
            got: format!(
                "a `context` field. That is the retired form — an array of byte values — and it is \
                 no longer read. Send `{}` (base64) and `{}` instead; accepting both is how two \
                 encodings coexist and `which one wins` becomes undefined",
                context::B64_FIELD,
                context::LEN_FIELD
            ),
        });
    }
    known_keys(obj, REQUEST_KEYS)?;

    // -- schema_version --------------------------------------------------------------------
    let schema_version = match obj.get("schema_version") {
        Some(Value::Number(n)) => n.as_u64().and_then(|v| u32::try_from(v).ok()),
        _ => None,
    };
    let Some(schema_version) = schema_version else {
        return Err(Refusal::UnknownSchemaVersion {
            supported: SUPPORTED_SCHEMA_VERSIONS.to_vec(),
            // A non-integer version is as unknown as an integer one this build does not have, and
            // `actual` must carry a number for the typed field. u32::MAX is the sentinel for "not
            // an integer we can name"; the `Display` string is unambiguous either way.
            actual: u32::MAX,
        });
    };
    if !SUPPORTED_SCHEMA_VERSIONS.contains(&schema_version) {
        return Err(Refusal::UnknownSchemaVersion {
            supported: SUPPORTED_SCHEMA_VERSIONS.to_vec(),
            actual: schema_version,
        });
    }

    // -- task ------------------------------------------------------------------------------
    let task = string_field(obj, "task")?;
    // `crate::is_blank`, not `str::trim`: the Python lane asks this with `str.strip()`, which
    // counts four more codepoints as whitespace. See `crate::is_wire_whitespace`.
    if crate::is_blank(&task) {
        return Err(Refusal::EmptyTask);
    }

    // -- context ---------------------------------------------------------------------------
    let context = parse_context(obj)?;

    // -- question --------------------------------------------------------------------------
    let question = string_field(obj, "question")?;

    // -- route -----------------------------------------------------------------------------
    let route_raw = string_field(obj, "route")?;
    let route = match route_raw.as_str() {
        "generic" => Route::Generic,
        "registered" => Route::Registered,
        other => {
            return Err(Refusal::UnknownRoute {
                supported: Route::supported(),
                actual: other.to_string(),
            })
        }
    };

    // -- slots -----------------------------------------------------------------------------
    let Some(Value::Array(raw_slots)) = obj.get("slots") else {
        return Err(Refusal::EmptySlots);
    };
    if raw_slots.is_empty() {
        return Err(Refusal::EmptySlots);
    }
    if raw_slots.len() > MAX_SLOTS {
        return Err(Refusal::TooManySlots {
            cap: MAX_SLOTS,
            actual: raw_slots.len(),
        });
    }

    let mut slots: Vec<SlotSpec> = Vec::with_capacity(raw_slots.len());
    let mut seen_names: BTreeMap<String, usize> = BTreeMap::new();
    for (index, raw) in raw_slots.iter().enumerate() {
        let slot = parse_slot(raw, index)?;
        if let Some(first) = seen_names.get(slot.name()) {
            return Err(Refusal::DuplicateSlotName {
                name: slot.name().to_string(),
                first_index: *first,
                duplicate_index: index,
            });
        }
        seen_names.insert(slot.name().to_string(), index);
        slots.push(slot);
    }

    // -- expect ----------------------------------------------------------------------------
    let expect = match obj.get("expect") {
        None | Some(Value::Null) => HashExpectation::default(),
        Some(v) => serde_json::from_value::<HashExpectation>(v.clone()).map_err(|e| {
            Refusal::MalformedRequest {
                detail: format!("`expect`: {e}"),
            }
        })?,
    };

    Ok(DecisionRequest {
        schema_version,
        task,
        context,
        question,
        slots,
        route,
        expect,
    })
}

/// Pull the two context fields out of the envelope and hand them to the one decoder.
///
/// This function owns the **JSON shape** — is the field there, is it the right JSON type — and
/// [`Context::from_wire_parts`] owns the **meaning**: the base64 alphabet, the padding, and the
/// length agreement. Splitting it that way is what keeps a single decoder: this module and
/// `Context`'s own `Deserialize` both end in the same call, so there is no second reading of the
/// same two fields to drift.
fn parse_context(obj: &Map<String, Value>) -> Result<Context, Refusal> {
    let b64 = match obj.get(context::B64_FIELD) {
        Some(Value::String(s)) => Some(s.as_str()),
        None => None,
        Some(other) => {
            return Err(Refusal::ContextNotBytes {
                got: json_type_name(other).to_string(),
            })
        }
    };

    // Absent and "present but not a non-negative integer" are one condition — no usable declared
    // length arrived — under one identifier, matching `ContextLenMissingRefusal` on the Python
    // side. `found` is what keeps them distinguishable in the message.
    let declared = match obj.get(context::LEN_FIELD) {
        None => None,
        Some(Value::Number(n)) => match n.as_u64() {
            Some(v) => Some(v),
            None => {
                return Err(Refusal::ContextLenMissing {
                    found: format!("{n}, which is not a non-negative integer"),
                })
            }
        },
        Some(other) => {
            return Err(Refusal::ContextLenMissing {
                found: format!("a JSON {}", json_type_name(other)),
            })
        }
    };

    Context::from_wire_parts(b64, declared)
}

fn parse_slot(raw: &Value, index: usize) -> Result<SlotSpec, Refusal> {
    let Value::Object(map) = raw else {
        return Err(Refusal::UnknownSlotType {
            slot: format!("slots[{index}]"),
            supported: SlotKind::supported(),
            actual: json_type_name(raw).to_string(),
        });
    };
    known_keys(map, &["name", "type", "options", "bins"])?;

    let name = match map.get("name") {
        Some(Value::String(s)) => s.clone(),
        _ => return Err(Refusal::EmptySlotName { index }),
    };
    // Both name rules — non-empty and within `MAX_SLOT_NAME_BYTES` — are `check_slot_name`'s, not
    // this function's. `render::slot_suffix` calls the same one, so a `SlotSpec` built by hand
    // cannot reach the prompt with a name this path would have refused.
    check_slot_name(&name, index)?;

    let kind = match map.get("type").and_then(Value::as_str) {
        Some("choice") => SlotKind::Choice,
        Some("score") => SlotKind::Score,
        Some("span") => SlotKind::Span,
        Some(other) => {
            return Err(Refusal::UnknownSlotType {
                slot: name,
                supported: SlotKind::supported(),
                actual: other.to_string(),
            })
        }
        None => {
            return Err(Refusal::UnknownSlotType {
                slot: name,
                supported: SlotKind::supported(),
                actual: "absent".to_string(),
            })
        }
    };

    let has_options = map.contains_key("options");
    let has_bins = map.contains_key("bins");

    match kind {
        SlotKind::Choice => {
            if has_bins {
                return Err(Refusal::SlotFieldNotAllowed {
                    slot: name,
                    slot_type: "choice".to_string(),
                    field: "bins".to_string(),
                });
            }
            let Some(Value::Array(raw_opts)) = map.get("options") else {
                return Err(Refusal::SlotFieldMissing {
                    slot: name,
                    slot_type: "choice".to_string(),
                    field: "options".to_string(),
                });
            };
            if raw_opts.len() > MAX_OPTIONS {
                return Err(Refusal::TooManyOptions {
                    slot: name,
                    limit: MAX_OPTIONS,
                    actual: raw_opts.len(),
                });
            }
            if raw_opts.len() < MIN_OPTIONS {
                return Err(Refusal::TooFewOptions {
                    slot: name,
                    min: MIN_OPTIONS,
                    actual: raw_opts.len(),
                });
            }
            let mut options: Vec<String> = Vec::with_capacity(raw_opts.len());
            for (i, opt) in raw_opts.iter().enumerate() {
                let Value::String(text) = opt else {
                    return Err(Refusal::UnknownSlotType {
                        slot: name,
                        supported: vec!["string option".to_string()],
                        actual: format!("options[{i}] is a JSON {}", json_type_name(opt)),
                    });
                };
                // Both of these are `crate::wire_trim`, not `str::trim`. `ChoiceSlot.__post_init__`
                // in `python/qd_data/schema.py` asks them with `str.strip()`, and the four
                // codepoints the two notions disagree about are exactly the ones that let an
                // option read as blank on one lane and as content on the other — or, worse, let
                // `"\u{1c}noul"` past this check as an ordinary option whose text is the reserved
                // abstain label. See `crate::is_wire_whitespace`.
                if crate::is_blank(text) {
                    return Err(Refusal::EmptyOption {
                        slot: name,
                        option_index: i,
                    });
                }
                if crate::wire_trim(text).eq_ignore_ascii_case(NOUL_LABEL) {
                    return Err(Refusal::ReservedOptionName {
                        slot: name,
                        option_index: i,
                        reserved: NOUL_LABEL.to_string(),
                    });
                }
                if let Some(first) = options.iter().position(|o| o == text) {
                    return Err(Refusal::DuplicateOption {
                        slot: name,
                        option: text.clone(),
                        first_index: first,
                        duplicate_index: i,
                    });
                }
                options.push(text.clone());
            }
            Ok(SlotSpec::Choice { name, options })
        }
        SlotKind::Score => {
            if has_options {
                return Err(Refusal::SlotFieldNotAllowed {
                    slot: name,
                    slot_type: "score".to_string(),
                    field: "options".to_string(),
                });
            }
            let Some(raw_bins) = map.get("bins") else {
                return Err(Refusal::SlotFieldMissing {
                    slot: name,
                    slot_type: "score".to_string(),
                    field: "bins".to_string(),
                });
            };
            let Some(bins) = raw_bins.as_u64().and_then(|v| u32::try_from(v).ok()) else {
                return Err(Refusal::BinsOutOfRange {
                    slot: name,
                    min: MIN_BINS,
                    max: MAX_BINS,
                    // A `bins` that is not a non-negative integer cannot be in range; 0 is the
                    // value the check compared against, and the message names the range.
                    actual: 0,
                });
            };
            if !(MIN_BINS..=MAX_BINS).contains(&bins) {
                return Err(Refusal::BinsOutOfRange {
                    slot: name,
                    min: MIN_BINS,
                    max: MAX_BINS,
                    actual: bins,
                });
            }
            Ok(SlotSpec::Score { name, bins })
        }
        SlotKind::Span => {
            if has_options {
                return Err(Refusal::SlotFieldNotAllowed {
                    slot: name,
                    slot_type: "span".to_string(),
                    field: "options".to_string(),
                });
            }
            if has_bins {
                return Err(Refusal::SlotFieldNotAllowed {
                    slot: name,
                    slot_type: "span".to_string(),
                    field: "bins".to_string(),
                });
            }
            Ok(SlotSpec::Span { name })
        }
    }
}

fn string_field(obj: &Map<String, Value>, key: &str) -> Result<String, Refusal> {
    match obj.get(key) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(Refusal::MalformedRequest {
            detail: format!("`{key}` is a JSON {}, not a string", json_type_name(other)),
        }),
        None => Err(Refusal::MalformedRequest {
            detail: format!("`{key}` is required"),
        }),
    }
}

/// Unknown keys are refused rather than ignored. A caller that misspells `context_len` would
/// otherwise get the `context_len is required` refusal with the field visibly present in its own
/// payload, and a caller on a newer schema would be silently served the older meaning.
fn known_keys(obj: &Map<String, Value>, allowed: &[&str]) -> Result<(), Refusal> {
    for key in obj.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Refusal::MalformedRequest {
                detail: format!("unknown field `{key}`; this build reads {allowed:?}"),
            });
        }
    }
    Ok(())
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
