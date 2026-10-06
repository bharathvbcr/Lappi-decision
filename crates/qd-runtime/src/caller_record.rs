//! Caller records: what an app saw, chose and later observed around one decision point.
//!
//! `docs/caller-contract.md` is the contract; this module is its single checker. Every calling app
//! writes these lines in its own language (Swift, Go, Rust), and none of them is trusted to have
//! written them right: [`validate_line`] is what any later reader runs first, and the
//! `qd-caller-records` binary runs it over a store.
//!
//! Three properties are enforced here rather than hoped for:
//!
//! * **Not admitted.** `admission` has exactly one legal value, `not_admitted`. Real caller data is
//!   not training or eval data (the human's "Synthetic only", 2026-10-06); moving a record into
//!   either needs a new human decision and a new `record_version`, never an app setting.
//! * **Held out by path.** The store lives under a `heldout` path segment
//!   ([`STORE_RELATIVE_TO_HOME`]), so `qd-train`'s rule-3 door refuses it through its existing
//!   marker layer with no new code, and [`check_store_path`] refuses a record file stored anywhere
//!   else.
//! * **Shape, not trust.** Unknown keys are refused, every reading is one of a closed set, and the
//!   `lappi` block must be internally consistent (a refusal names its kind, an answer carries
//!   slots). A secret-shaped string anywhere in the record refuses the whole line: the app's own
//!   redactor is the first line of defence and this is the backstop.

use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value};

/// The `record` discriminant on every line.
pub const RECORD: &str = "lappi.caller_record";
/// The only version this checker reads. A new field or a new admission value is a new version.
pub const RECORD_VERSION: u64 = 1;
/// The only legal `admission` value under version 1.
pub const ADMISSION_NOT_ADMITTED: &str = "not_admitted";
/// One line's cap. A record is structured facts about one decision, not a copy of its inputs.
pub const MAX_RECORD_BYTES: usize = 64 * 1024;
/// The store, relative to `$HOME`. The `heldout` segment is load-bearing: it is what makes
/// `qd-train` refuse the store (`crates/qd-train/tests/caller_records_are_held_out.rs` pins the
/// same literal).
pub const STORE_RELATIVE_TO_HOME: &str = "Library/Application Support/Lappi/heldout/caller-records";
/// Path segments that mark a held-out tree, case-folded. Matches `qd-train`'s default markers.
pub const HELD_OUT_MARKERS: [&str; 3] = ["heldout", "held_out", "held-out"];

/// What a caller read off one exchange with Lappi, or why there was none.
///
/// The first four are [`crate::schema::CallerReading`]'s four, spelled as on the wire; the last
/// two are the caller-side states the runtime cannot see.
pub const READINGS: [&str; 6] = [
    "model_answered",
    "model_abstained",
    "request_refused",
    "backend_failed",
    "unavailable",
    "not_asked",
];

/// High-confidence credential prefixes. A backstop, not a redactor: the app redacts before it
/// writes, and a line that still carries one of these is refused whole.
const SECRET_PREFIXES: [&str; 12] = [
    "sk-ant-",
    "sk-proj-",
    "ghp_",
    "gho_",
    "ghs_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "AKIA",
    "AIza",
    "-----BEGIN",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    Decision,
    Outcome,
}

impl RecordKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RecordKind::Decision => "decision",
            RecordKind::Outcome => "outcome",
        }
    }
}

/// What a valid line says about itself, enough to pair outcomes with decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordSummary {
    pub kind: RecordKind,
    pub record_id: String,
    pub app: String,
    pub decision_point: String,
    /// `None` for an outcome line.
    pub reading: Option<String>,
}

/// Why a line is not a valid caller record. Each names the check and what it saw.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecordError {
    #[error("the line is {got} bytes, over the {cap}-byte record cap")]
    TooLarge { got: usize, cap: usize },
    #[error("the line is not UTF-8")]
    NotUtf8,
    #[error("the line is not a JSON object: {0}")]
    NotJsonObject(String),
    #[error("{path}: unknown key {key:?}; a version-{RECORD_VERSION} record has exactly {expected}")]
    UnknownKey { path: String, key: String, expected: String },
    #[error("{path}: missing key {key:?}")]
    MissingKey { path: String, key: String },
    #[error("{path}: expected {expected}, got {got}")]
    Invalid { path: String, expected: String, got: String },
    #[error("{path}: {why}")]
    Inconsistent { path: String, why: String },
    #[error("{path}: a string carries a {prefix:?}-prefixed credential shape; the line is refused whole")]
    SecretLike { path: String, prefix: String },
    #[error("{path}: a caller record must be stored under a held-out path segment ({markers:?})")]
    NotHeldOutPath { path: String, markers: [&'static str; 3] },
}

/// `$HOME/` + [`STORE_RELATIVE_TO_HOME`], or `None` when `$HOME` is unset (never a relative
/// fallback, which would land the store in whatever directory the app happened to start in).
pub fn default_store_root() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    if home.is_empty() {
        return None;
    }
    Some(PathBuf::from(home).join(STORE_RELATIVE_TO_HOME))
}

/// Refuse a record file that is not under a held-out segment. Lexical, by design: the check is
/// about where the bytes were *put*, and `qd-train`'s door resolves symlinks on its own side.
pub fn check_store_path(path: &Path) -> Result<(), RecordError> {
    let held = path.components().any(|c| match c {
        Component::Normal(seg) => seg
            .to_str()
            .is_some_and(|s| HELD_OUT_MARKERS.iter().any(|m| s.eq_ignore_ascii_case(m))),
        _ => false,
    });
    if held {
        Ok(())
    } else {
        Err(RecordError::NotHeldOutPath {
            path: path.display().to_string(),
            markers: HELD_OUT_MARKERS,
        })
    }
}

/// Validate one line (without its trailing newline).
pub fn validate_line(line: &[u8]) -> Result<RecordSummary, RecordError> {
    if line.len() > MAX_RECORD_BYTES {
        return Err(RecordError::TooLarge { got: line.len(), cap: MAX_RECORD_BYTES });
    }
    let text = std::str::from_utf8(line).map_err(|_| RecordError::NotUtf8)?;
    let value: Value =
        serde_json::from_str(text).map_err(|e| RecordError::NotJsonObject(e.to_string()))?;
    screen_secrets(&value, "$")?;
    let Value::Object(obj) = value else {
        return Err(RecordError::NotJsonObject(kind_of(&value).to_string()));
    };

    let kind = match req_str(&obj, "$", "kind")? {
        "decision" => RecordKind::Decision,
        "outcome" => RecordKind::Outcome,
        other => return Err(invalid("$.kind", "\"decision\" or \"outcome\"", other)),
    };
    let expected: &[&str] = match kind {
        RecordKind::Decision => &[
            "record", "record_version", "kind", "record_id", "app", "app_version",
            "decision_point", "created_at", "admission", "facts", "app_choice", "lappi",
            "redaction",
        ],
        RecordKind::Outcome => &[
            "record", "record_version", "kind", "record_id", "app", "app_version",
            "decision_point", "created_at", "admission", "observed", "redaction",
        ],
    };
    exact_keys(&obj, "$", expected)?;

    let record = req_str(&obj, "$", "record")?;
    if record != RECORD {
        return Err(invalid("$.record", RECORD, record));
    }
    match obj.get("record_version").and_then(Value::as_u64) {
        Some(RECORD_VERSION) => {}
        _ => {
            return Err(invalid(
                "$.record_version",
                &RECORD_VERSION.to_string(),
                &obj["record_version"].to_string(),
            ));
        }
    }
    let record_id = req_str(&obj, "$", "record_id")?;
    if record_id.len() != 32 || !record_id.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(invalid("$.record_id", "32 lowercase hex characters", record_id));
    }
    let app = req_str(&obj, "$", "app")?;
    if !is_ident(app, 32, false) {
        return Err(invalid("$.app", "a lowercase identifier [a-z][a-z0-9-]{0,31}", app));
    }
    let app_version = req_str(&obj, "$", "app_version")?;
    if app_version.is_empty() || app_version.len() > 64 {
        return Err(invalid("$.app_version", "1-64 characters", app_version));
    }
    let decision_point = req_str(&obj, "$", "decision_point")?;
    if !is_ident(decision_point, 128, true) {
        return Err(invalid(
            "$.decision_point",
            "a dotted lowercase identifier of at most 128 bytes",
            decision_point,
        ));
    }
    let created_at = req_str(&obj, "$", "created_at")?;
    if !is_utc_timestamp(created_at) {
        return Err(invalid(
            "$.created_at",
            "an RFC 3339 UTC timestamp YYYY-MM-DDTHH:MM:SS[.fff]Z",
            created_at,
        ));
    }
    let admission = req_str(&obj, "$", "admission")?;
    if admission != ADMISSION_NOT_ADMITTED {
        return Err(only_not_admitted(admission));
    }
    check_redaction(&obj)?;

    let reading = match kind {
        RecordKind::Decision => {
            req_object(&obj, "$", "facts")?;
            req_object(&obj, "$", "app_choice")?;
            Some(check_lappi(req_object(&obj, "$", "lappi")?)?)
        }
        RecordKind::Outcome => {
            req_object(&obj, "$", "observed")?;
            None
        }
    };

    Ok(RecordSummary {
        kind,
        record_id: record_id.to_string(),
        app: app.to_string(),
        decision_point: decision_point.to_string(),
        reading,
    })
}

fn only_not_admitted(got: &str) -> RecordError {
    RecordError::Inconsistent {
        path: "$.admission".into(),
        why: format!(
            "only {ADMISSION_NOT_ADMITTED:?} is legal under version {RECORD_VERSION}, got {got:?}: \
             admitting caller data to training or eval is a human decision, not a record field"
        ),
    }
}

fn check_redaction(obj: &Map<String, Value>) -> Result<(), RecordError> {
    let red = req_object(obj, "$", "redaction")?;
    exact_keys(red, "$.redaction", &["policy", "fields_redacted"])?;
    let policy = req_str(red, "$.redaction", "policy")?;
    if policy.is_empty() || policy.len() > 64 {
        return Err(invalid("$.redaction.policy", "the redactor's name, 1-64 characters", policy));
    }
    if red.get("fields_redacted").and_then(Value::as_u64).is_none() {
        return Err(invalid(
            "$.redaction.fields_redacted",
            "a non-negative integer",
            &red["fields_redacted"].to_string(),
        ));
    }
    Ok(())
}

fn check_lappi(lappi: &Map<String, Value>) -> Result<String, RecordError> {
    let p = "$.lappi";
    exact_keys(lappi, p, &["asked", "task", "reading", "kind", "backend", "slots", "latency_ms"])?;
    let asked = lappi
        .get("asked")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid("$.lappi.asked", "a boolean", &lappi["asked"].to_string()))?;
    let reading = req_str(lappi, p, "reading")?;
    if !READINGS.contains(&reading) {
        return Err(invalid("$.lappi.reading", &format!("one of {READINGS:?}"), reading));
    }
    let task = opt_str(lappi, p, "task")?;
    let kind = opt_str(lappi, p, "kind")?;
    let backend = opt_str(lappi, p, "backend")?;
    let slots = match &lappi["slots"] {
        Value::Null => None,
        Value::Object(m) => Some(m),
        other => return Err(invalid("$.lappi.slots", "an object or null", kind_of(other))),
    };
    match &lappi["latency_ms"] {
        Value::Null => {}
        v if v.as_u64().is_some() => {}
        other => {
            return Err(invalid("$.lappi.latency_ms", "a non-negative integer or null", &other.to_string()));
        }
    }

    let inconsistent = |why: &str| RecordError::Inconsistent { path: p.into(), why: why.into() };
    if asked != (reading != "not_asked") {
        return Err(inconsistent("asked is false exactly when reading is \"not_asked\""));
    }
    if asked && task.is_none() {
        return Err(inconsistent("a request that was sent names its task"));
    }
    match reading {
        "model_answered" | "model_abstained" => {
            if slots.is_none() || backend.is_none() {
                return Err(inconsistent("an answer carries its slots and the backend that gave them"));
            }
            if kind.is_some() {
                return Err(inconsistent("an answer has no refusal or error kind"));
            }
        }
        "request_refused" | "backend_failed" | "unavailable" => {
            if kind.is_none() {
                return Err(inconsistent("a refusal, backend error or unavailability names its kind"));
            }
            if slots.is_some() {
                return Err(inconsistent("only an answer carries slots"));
            }
        }
        _ => {
            if kind.is_some() || slots.is_some() || backend.is_some() {
                return Err(inconsistent("a decision Lappi was not asked carries no reply fields"));
            }
        }
    }
    Ok(reading.to_string())
}

fn screen_secrets(value: &Value, path: &str) -> Result<(), RecordError> {
    match value {
        Value::String(s) => {
            for prefix in SECRET_PREFIXES {
                if let Some(at) = s.find(prefix) {
                    // A prefix inside a word (`TASKAKIA`) is not a credential's start.
                    let boundary = at == 0
                        || !s[..at].chars().next_back().is_some_and(|c| c.is_ascii_alphanumeric());
                    let body = &s[at + prefix.len()..];
                    let long_enough = prefix.starts_with("-----")
                        || body.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').count() >= 12;
                    if boundary && long_enough {
                        return Err(RecordError::SecretLike { path: path.into(), prefix: prefix.into() });
                    }
                }
            }
            Ok(())
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                screen_secrets(item, &format!("{path}[{i}]"))?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for (k, v) in map {
                screen_secrets(&Value::String(k.clone()), &format!("{path}.<key>"))?;
                screen_secrets(v, &format!("{path}.{k}"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn is_ident(s: &str, max: usize, dotted: bool) -> bool {
    if s.is_empty() || s.len() > max {
        return false;
    }
    let parts: Vec<&str> = if dotted { s.split('.').collect() } else { vec![s] };
    parts.iter().all(|part| {
        let mut chars = part.chars();
        matches!(chars.next(), Some('a'..='z'))
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    })
}

fn is_utc_timestamp(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 20 || b[b.len() - 1] != b'Z' {
        return false;
    }
    let digits = |r: std::ops::Range<usize>| b[r].iter().all(u8::is_ascii_digit);
    let shape = digits(0..4)
        && b[4] == b'-'
        && digits(5..7)
        && b[7] == b'-'
        && digits(8..10)
        && b[10] == b'T'
        && digits(11..13)
        && b[13] == b':'
        && digits(14..16)
        && b[16] == b':'
        && digits(17..19);
    if !shape {
        return false;
    }
    let frac = &b[19..b.len() - 1];
    let frac_ok = frac.is_empty()
        || (frac[0] == b'.' && (2..=10).contains(&frac.len()) && frac[1..].iter().all(u8::is_ascii_digit));
    let num = |r: std::ops::Range<usize>| s[r].parse::<u32>().unwrap_or(u32::MAX);
    frac_ok
        && (1..=12).contains(&num(5..7))
        && (1..=31).contains(&num(8..10))
        && num(11..13) <= 23
        && num(14..16) <= 59
        && num(17..19) <= 60
}

fn exact_keys(obj: &Map<String, Value>, path: &str, expected: &[&str]) -> Result<(), RecordError> {
    for key in obj.keys() {
        if !expected.contains(&key.as_str()) {
            return Err(RecordError::UnknownKey {
                path: path.into(),
                key: key.clone(),
                expected: format!("{expected:?}"),
            });
        }
    }
    for key in expected {
        if !obj.contains_key(*key) {
            return Err(RecordError::MissingKey { path: path.into(), key: (*key).into() });
        }
    }
    Ok(())
}

fn req_str<'a>(obj: &'a Map<String, Value>, path: &str, key: &str) -> Result<&'a str, RecordError> {
    match obj.get(key) {
        Some(Value::String(s)) => Ok(s),
        Some(other) => Err(invalid(&format!("{path}.{key}"), "a string", kind_of(other))),
        None => Err(RecordError::MissingKey { path: path.into(), key: key.into() }),
    }
}

fn opt_str<'a>(obj: &'a Map<String, Value>, path: &str, key: &str) -> Result<Option<&'a str>, RecordError> {
    match obj.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.is_empty() && s.len() <= 256 => Ok(Some(s)),
        Some(other) => Err(invalid(
            &format!("{path}.{key}"),
            "a 1-256 character string or null",
            &other.to_string(),
        )),
        None => Err(RecordError::MissingKey { path: path.into(), key: key.into() }),
    }
}

fn req_object<'a>(
    obj: &'a Map<String, Value>,
    path: &str,
    key: &str,
) -> Result<&'a Map<String, Value>, RecordError> {
    match obj.get(key) {
        Some(Value::Object(m)) => Ok(m),
        Some(other) => Err(invalid(&format!("{path}.{key}"), "an object", kind_of(other))),
        None => Err(RecordError::MissingKey { path: path.into(), key: key.into() }),
    }
}

fn invalid(path: &str, expected: &str, got: &str) -> RecordError {
    RecordError::Invalid { path: path.into(), expected: expected.into(), got: got.into() }
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}
