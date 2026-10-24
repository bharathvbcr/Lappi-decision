//! Real Vani teacher readings as training rows: the gates before any real-row pool (v2 plan
//! section 6; owner rulings 1, 9 and 12 of 2026-10-20; `docs/caller-contract.md` section 5, the
//! Vani exception).
//!
//! A row is admitted only when, in this order:
//!
//! 1. **Two teachers agree** ([`agree`]). The input carries exactly two readings of one posting
//!    and slot. They agree when both name the same chosen label. Ruling 9 replaced the margin
//!    rule because Vani's ledger keeps only the chosen label's probability
//!    (`Vani/internal/ledger/annotations.go:27`) and the second teacher carries none, so there is
//!    no margin and none is invented here. One teacher, two readings under the same teacher id
//!    (compared trimmed and ASCII-case-insensitively), a disagreeing pair, or an empty field is
//!    refused. The row carries `source: vani-agreed`, both teacher ids and both readings.
//! 2. **The posting is not near a held-out posting** ([`PostingDoor`]), judged on the text as
//!    Vani exported it, before redaction.
//! 3. **The text is redacted** ([`redact`]) and nothing is left: [`residual`] finds no email,
//!    phone number or signature-block name in the text, the key, the teacher ids, the slot or the
//!    labels; the repository's one secret screen (`secret_like_prefix`) finds no credential; and
//!    at least [`redact::MIN_WORDS_AFTER`] words survive. Otherwise the row is refused, never
//!    passed on partly cleaned.
//!
//! A held-out signature file that cannot be read, or is corrupt, or of another version, refuses
//! the whole pool: [`run`] returns the error before reading a single pair.
//!
//! Where a row may then train is not decided here. A `vani-agreed` row never enters a public
//! adapter or a cloud run: `qd-release-check` (`adapter_visibility`) and `qd-train-data`
//! (`mac_only`) refuse those, reading [`VANI_AGREED_SOURCE`].

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use qd_runtime::caller_record::secret_like_prefix;
use serde_json::{Value, json};

use crate::files;
use crate::heldout::refuse_training_input;
use crate::posting_door::{self, PostingDoor};
use crate::redact::{self, content_words, redact, residual};
use crate::sha256::sha256_hex;

/// The source of a real Vani row. `qd_preflight::pool_policy` reads the same string.
pub const VANI_AGREED_SOURCE: &str = "vani-agreed";
/// Bound on the pairs file.
pub const MAX_PAIRS_BYTES: u64 = 1 << 30;

/// One teacher's reading of one posting's slot: who read it and which label they chose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reading {
    pub teacher: String,
    pub label: String,
}

/// One posting with the readings taken of it. `text` is the description as Vani holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    pub posting_key: String,
    pub slot: String,
    pub text: String,
    pub readings: Vec<Reading>,
}

/// An agreed pair: the label both teachers chose, both teacher ids and both readings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Agreed {
    pub label: String,
    pub teachers: [String; 2],
    pub readings: [String; 2],
}

/// Why a pair is not a row.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// Not exactly two readings: one teacher's reading is not agreement.
    ReadingCount(usize),
    /// A required field is empty.
    Empty(&'static str),
    /// Both readings are from one teacher id.
    SameTeacher(String),
    /// The two teachers chose different labels.
    Disagree { first: String, second: String },
    /// The content door refused the posting.
    Door(posting_door::Refusal),
    /// An email, phone number or name is still in the row after redaction.
    Residual {
        field: &'static str,
        kinds: Vec<&'static str>,
    },
    /// A credential-shaped string is in the row; secrets are refused, not redacted.
    Secret {
        field: &'static str,
        prefix: &'static str,
    },
    /// Redaction left fewer than [`redact::MIN_WORDS_AFTER`] words of the posting.
    EmptiedByRedaction { words_left: usize },
    /// The same posting and slot appeared earlier in the input.
    Duplicate,
}

impl Refusal {
    pub fn kind(&self) -> &'static str {
        match self {
            Refusal::ReadingCount(_) => "reading_count",
            Refusal::Empty(_) => "empty_field",
            Refusal::SameTeacher(_) => "same_teacher",
            Refusal::Disagree { .. } => "disagree",
            Refusal::Door(d) => d.kind(),
            Refusal::Residual { .. } => "residual_pii",
            Refusal::Secret { .. } => "secret",
            Refusal::EmptiedByRedaction { .. } => "emptied_by_redaction",
            Refusal::Duplicate => "duplicate",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::ReadingCount(n) => write!(
                f,
                "{n} readings; agreement needs exactly two readings of the posting"
            ),
            Refusal::Empty(field) => write!(f, "{field} is empty"),
            Refusal::SameTeacher(t) => write!(
                f,
                "both readings are from teacher {t:?}: one teacher is not two teachers"
            ),
            Refusal::Disagree { first, second } => write!(
                f,
                "the teachers chose different labels: {first:?} and {second:?}"
            ),
            Refusal::Door(d) => write!(f, "{d}"),
            Refusal::Residual { field, kinds } => {
                write!(f, "{field} still carries {kinds:?} after redaction")
            }
            Refusal::Secret { field, prefix } => write!(
                f,
                "{field} carries a credential-shaped string (prefix {prefix:?})"
            ),
            Refusal::EmptiedByRedaction { words_left } => write!(
                f,
                "redaction left {words_left} words of the posting; at least {} are needed",
                redact::MIN_WORDS_AFTER
            ),
            Refusal::Duplicate => {
                write!(f, "this posting and slot were read earlier in the input")
            }
        }
    }
}

fn nonempty<'a>(value: &'a str, field: &'static str) -> Result<&'a str, Refusal> {
    let t = value.trim();
    if t.is_empty() {
        Err(Refusal::Empty(field))
    } else {
        Ok(t)
    }
}

/// Two teachers' readings agree when they are from two teachers and name the same chosen label.
pub fn agree(pair: &Pair) -> Result<Agreed, Refusal> {
    nonempty(&pair.posting_key, "posting_key")?;
    nonempty(&pair.slot, "slot")?;
    nonempty(&pair.text, "text")?;
    let [first, second] = pair.readings.as_slice() else {
        return Err(Refusal::ReadingCount(pair.readings.len()));
    };
    let t1 = nonempty(&first.teacher, "teacher")?;
    let t2 = nonempty(&second.teacher, "teacher")?;
    let l1 = nonempty(&first.label, "label")?;
    let l2 = nonempty(&second.label, "label")?;
    if t1.eq_ignore_ascii_case(t2) {
        return Err(Refusal::SameTeacher(t1.to_owned()));
    }
    if l1 != l2 {
        return Err(Refusal::Disagree {
            first: l1.to_owned(),
            second: l2.to_owned(),
        });
    }
    Ok(Agreed {
        label: l1.to_owned(),
        teachers: [t1.to_owned(), t2.to_owned()],
        readings: [l1.to_owned(), l2.to_owned()],
    })
}

/// The row id: slot and a digest of the key, so the row names its posting without echoing the key.
fn row_id(slot: &str, posting_key: &str) -> String {
    format!(
        "{VANI_AGREED_SOURCE}:{slot}:{}",
        &sha256_hex(posting_key.as_bytes())[..16]
    )
}

fn screen(field: &'static str, value: &str) -> Result<(), Refusal> {
    let kinds = residual(value);
    if !kinds.is_empty() {
        return Err(Refusal::Residual { field, kinds });
    }
    match secret_like_prefix(value) {
        Some(prefix) => Err(Refusal::Secret { field, prefix }),
        None => Ok(()),
    }
}

/// Admit one pair as a row, or say why not.
pub fn admit(pair: &Pair, door: &PostingDoor) -> Result<Value, Refusal> {
    let agreed = agree(pair)?;
    door.check(&pair.posting_key, &pair.text)
        .map_err(Refusal::Door)?;
    let redacted = redact(&pair.text);
    let words_left = content_words(&redacted.text);
    if words_left < redact::MIN_WORDS_AFTER {
        return Err(Refusal::EmptiedByRedaction { words_left });
    }
    screen("context", &redacted.text)?;
    screen("posting_key", &pair.posting_key)?;
    screen("slot", &pair.slot)?;
    screen("teacher[0]", &agreed.teachers[0])?;
    screen("teacher[1]", &agreed.teachers[1])?;
    screen("label", &agreed.label)?;
    Ok(json!({
        "id": row_id(&pair.slot, &pair.posting_key),
        "source": VANI_AGREED_SOURCE,
        "source_id": VANI_AGREED_SOURCE,
        "posting_key": pair.posting_key,
        "slot": pair.slot,
        "label": agreed.label,
        "teachers": agreed.teachers,
        "readings": agreed.readings,
        "context": redacted.text,
        "redaction": {
            "emails": redacted.emails, "phones": redacted.phones, "names": redacted.names,
        },
    }))
}

fn parse_pair(line_no: usize, line: &str) -> Result<Pair, String> {
    let at = |m: String| format!("--pairs line {line_no}: {m}");
    let v: Value = serde_json::from_str(line).map_err(|e| at(format!("not JSON: {e}")))?;
    let text_of = |k: &str| {
        v.get(k)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| at(format!("no string {k:?}")))
    };
    let readings = v
        .get("readings")
        .and_then(Value::as_array)
        .ok_or_else(|| at("no readings array".into()))?
        .iter()
        .map(|r| {
            let get = |k: &str| {
                r.get(k)
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| at(format!("a reading has no string {k:?}")))
            };
            Ok(Reading {
                teacher: get("teacher")?,
                label: get("label")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Pair {
        posting_key: text_of("posting_key")?,
        slot: text_of("slot")?,
        text: text_of("text")?,
        readings,
    })
}

/// `qd-prep vani-pool`: `pairs` is JSONL, one `{"posting_key", "slot", "text", "readings":
/// [{"teacher", "label"}, ...]}` per line, taken from Vani's pipeline ledger; `signatures` is
/// the held-out posting export ([`PostingDoor`]). Writes `out_dir/{rows.jsonl, refusals.jsonl,
/// report.json}`. A signature file that cannot be used, or a pairs line that cannot be read,
/// refuses the whole run and writes nothing.
pub fn run(pairs: &Path, signatures: &Path, out_dir: &Path) -> Result<String, String> {
    // The door first: with no usable door no pair is judged, and no row can be admitted.
    let door = PostingDoor::load(signatures)?;
    refuse_training_input(pairs, "--pairs")?;
    let bytes = files::read_bounded(pairs, MAX_PAIRS_BYTES, "--pairs")?;
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("--pairs is not UTF-8: {e}"))?;
    let mut rows: Vec<String> = Vec::new();
    let mut refusals: Vec<String> = Vec::new();
    let mut by_kind: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for (i, line) in text.lines().enumerate() {
        let pair = parse_pair(i + 1, line)?;
        total += 1;
        let verdict = if seen.insert((pair.posting_key.clone(), pair.slot.clone())) {
            admit(&pair, &door)
        } else {
            Err(Refusal::Duplicate)
        };
        match verdict {
            Ok(row) => rows.push(row.to_string()),
            Err(r) => {
                *by_kind.entry(r.kind()).or_default() += 1;
                refusals.push(
                    json!({"line": i + 1, "id": row_id(&pair.slot, &pair.posting_key),
                           "kind": r.kind(), "reason": r.to_string()})
                    .to_string(),
                );
            }
        }
    }
    if total == 0 {
        return Err("--pairs holds no pair: a pool built from nothing is not a pool".into());
    }
    let report = json!({
        "format": "qd-vani-pool.v1",
        "pairs": total,
        "admitted": rows.len(),
        "refused": refusals.len(),
        "refused_by_kind": by_kind,
        "source": VANI_AGREED_SOURCE,
        "door": {
            "held_out_postings": door.len(),
            "ngram_n": posting_door::NGRAM_N,
            "containment_threshold": posting_door::CONTAINMENT_THRESHOLD,
            "jaccard_threshold": posting_door::JACCARD_THRESHOLD,
            "num_perm": posting_door::NUM_PERM,
        },
    });
    let joined = |v: &[String]| {
        let mut s = v.join("\n");
        if !s.is_empty() {
            s.push('\n');
        }
        s
    };
    files::write_new_dir(out_dir, |dir| {
        files::write_synced_new(&dir.join("rows.jsonl"), joined(&rows).as_bytes())?;
        files::write_synced_new(&dir.join("refusals.jsonl"), joined(&refusals).as_bytes())?;
        files::write_synced_new(&dir.join("report.json"), report.to_string().as_bytes())
    })?;
    Ok(format!(
        "qd-prep vani-pool: {total} pairs, {} admitted, {} refused {by_kind:?} -> {}",
        rows.len(),
        refusals.len(),
        out_dir.display()
    ))
}
