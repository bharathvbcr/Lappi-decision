//! The calibration gate: a release is written only if every (trained family, slot shape as
//! trained) has a calibration entry whose recorded check ran and passed.
//!
//! Why it exists. The v0.1 preview shipped a table with no span entry for `code.defect_class`,
//! whose trained request asks for a choice slot and a span slot
//! (GAP-PREVIEW-DOES-NOT-SERVE-CODE-DEFECT-CLASS-2026-10-06), and a `choice:16` entry whose ECE
//! failed (GAP-16-OPTION-CALIBRATION-FAILS-AND-HAS-NO-ID-2026-10-06). `calibration.json` carries
//! no verdict, so the exporter took both on faith.
//!
//! Inputs, each required beside `--calibration`:
//!
//! * the fit's own report (`qd-calib-fit`'s `report.json`): its `tables[].sha256` must be this
//!   table's file hash, its `ece.threshold` must be [`ECE_THRESHOLD`] (the registered constant,
//!   not a number read from the report), and `entries[KEY].scored.ece_after` is the recorded check;
//! * the slot shapes each trained family was trained on, `{family: [["choice:5", "span"], ["choice:7"]]}`:
//!   one array per trained request shape, so choice and span trained together are gated together.
//!   The train manifest owns them (each entry's `slot_shape`, read by `export::read_train_manifest`
//!   under a versioned `slot_shape_digest`); for an ensemble they are the members' own release
//!   manifests'. Nothing is hand-declared: a family with no recorded shape is refused.
//!
//! A check that did not run (`state: "not_run"`, e.g. fewer than 100 examples) is a refusal, never
//! a pass. The only way past a family is [`ExcludedFamily`]: the family leaves
//! `trained_families` (the runtime then refuses its tasks) and the reason is recorded in the
//! manifest.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Value, json};

use qd_runtime::calibration::CalibrationTable;
use qd_runtime::calibration_fit::ECE_THRESHOLD;
use qd_runtime::hex;
use qd_runtime::schema::SlotKind;

use crate::export::{read_small, refuse, sha256};
use crate::refusal::{RefusalKind, Result};

/// A trained family deliberately not admitted, with the reason the manifest records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExcludedFamily {
    pub family: String,
    pub reason: String,
}

impl ExcludedFamily {
    /// `FAMILY=REASON`, both non-empty.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let (family, reason) = s
            .split_once('=')
            .ok_or_else(|| format!("{s:?} is not FAMILY=REASON"))?;
        let (family, reason) = (family.trim(), reason.trim());
        if family.is_empty() || reason.is_empty() {
            return Err(format!(
                "{s:?}: both the family and the reason must be non-empty"
            ));
        }
        Ok(Self {
            family: family.to_string(),
            reason: reason.to_string(),
        })
    }
}

/// What the gate established, for the manifest.
pub(crate) struct GateRecord {
    /// The `calibration.gate` block.
    pub block: Value,
}

/// Per family, the set of request shapes it trained on; each shape is its sorted slot keys.
pub(crate) type FamilyShapes = BTreeMap<String, BTreeSet<Vec<String>>>;

/// A recorded `trained_slot_shapes` object (`{family: [[key, ...], ...]}`), as a release manifest
/// writes it.
pub(crate) fn parse_family_shapes(
    v: &Value,
    what: &str,
    kind: RefusalKind,
) -> Result<FamilyShapes> {
    let obj = v.as_object().ok_or_else(|| {
        refuse(
            kind,
            format!("{what}: trained_slot_shapes is not an object"),
        )
    })?;
    let mut out = FamilyShapes::new();
    for (family, shapes) in obj {
        let arr = shapes.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
            refuse(
                kind,
                format!("{what}: trained_slot_shapes[{family}] is not a non-empty array"),
            )
        })?;
        let mut set = BTreeSet::new();
        for shape in arr {
            let mut keys = Vec::new();
            for key in shape
                .as_array()
                .filter(|s| !s.is_empty())
                .into_iter()
                .flatten()
            {
                let key = key.as_str().filter(|s| is_shape_key(s)).ok_or_else(|| {
                    refuse(
                        kind,
                        format!(
                            "{what}: {family}: shape key {key} is not span, choice:N or score:N"
                        ),
                    )
                })?;
                keys.push(key.to_string());
            }
            if keys.is_empty() {
                return Err(refuse(
                    kind,
                    format!(
                        "{what}: {family}: a shape is a non-empty array of slot keys, got {shape}"
                    ),
                ));
            }
            keys.sort();
            set.insert(keys);
        }
        out.insert(family.clone(), set);
    }
    Ok(out)
}

pub(crate) fn is_shape_key(key: &str) -> bool {
    match key.split_once(':') {
        None => key == "span",
        Some((kind, rows)) => {
            matches!(kind, "choice" | "score")
                && !rows.is_empty()
                && rows.bytes().all(|b| b.is_ascii_digit())
        }
    }
}

/// Whether the runtime would find an entry for the slot shape `key` (`span`, `choice:N`,
/// `score:N`): the key is read into the `(kind, rows)` the runtime's `CalibrationTable::lookup`
/// takes, and that lookup answers. `N` must be the decimal the runtime formats back (no sign, no
/// leading zero, fits `usize`), because the runtime builds its key from the request's row count,
/// never from text a manifest wrote.
fn runtime_lookup(table: &CalibrationTable, key: &str) -> std::result::Result<(), String> {
    let (kind, rows) = match key.split_once(':') {
        None if key == "span" => (SlotKind::Span, 0),
        Some((kind, digits)) => {
            let kind = match kind {
                "choice" => SlotKind::Choice,
                "score" => SlotKind::Score,
                _ => return Err(format!("{key} is not span, choice:N or score:N")),
            };
            let rows: usize = digits
                .parse()
                .map_err(|e| format!("{key}: the row count is not a usize ({e})"))?;
            if rows.to_string() != digits {
                return Err(format!(
                    "{key}: the runtime looks up {}:{rows}, not the spelling {digits:?}",
                    kind.as_str()
                ));
            }
            (kind, rows)
        }
        None => return Err(format!("{key} is not span, choice:N or score:N")),
    };
    table
        .lookup(key, kind, rows)
        .map(|_| ())
        .map_err(|refusal| refusal.to_string())
}

/// The recorded check at `ece`: ran, passed, and consistent with the registered threshold.
fn check_ece(key: &str, ece: Option<&Value>) -> Result<f64> {
    let k = RefusalKind::Calibration;
    let ece = ece.ok_or_else(|| refuse(k, format!("{key}: the fit's report records no ece_after for this entry; an unrecorded check is not a passed one")))?;
    let state = ece.get("state").and_then(Value::as_str).unwrap_or("absent");
    if state != "ran" {
        let why = ece
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("no reason recorded");
        return Err(refuse(
            k,
            format!(
                "{key}: the recorded calibration check is {state}, not ran ({why}); a check that could not run is not a check that passed"
            ),
        ));
    }
    let value = ece
        .get("value")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite());
    let passed = ece.get("passed").and_then(Value::as_bool);
    let Some(value) = value else {
        return Err(refuse(k, format!("{key}: ece_after has no finite value")));
    };
    if passed != Some(true) {
        return Err(refuse(
            k,
            format!(
                "{key}: the fit's recorded calibration check failed (ECE {value:.4}, passed=false, threshold {ECE_THRESHOLD})"
            ),
        ));
    }
    if value > ECE_THRESHOLD {
        return Err(refuse(
            k,
            format!(
                "{key}: the report says passed=true but ECE {value:.4} is over the registered threshold {ECE_THRESHOLD}"
            ),
        ));
    }
    Ok(value)
}

/// Run the gate. `trained` is what the train manifest (or, for an ensemble, its members' release
/// manifests) records each family as trained on; `admitted` is the families the release admits.
pub(crate) fn check(
    table_bytes: &[u8],
    table: &CalibrationTable,
    report_path: &Path,
    trained: &FamilyShapes,
    admitted: &[String],
) -> Result<GateRecord> {
    let k = RefusalKind::Calibration;
    let table_sha = hex(&sha256(table_bytes));

    let report_bytes = read_small(report_path, k)?;
    let report: Value = serde_json::from_slice(&report_bytes)
        .map_err(|e| refuse(k, format!("{}: not JSON: {e}", report_path.display())))?;
    let bound = report
        .get("tables")
        .and_then(Value::as_array)
        .is_some_and(|ts| {
            ts.iter()
                .any(|t| t.get("sha256").and_then(Value::as_str) == Some(table_sha.as_str()))
        });
    if !bound {
        return Err(refuse(
            k,
            format!(
                "{}: no tables[].sha256 equals this calibration file's sha256 {table_sha}; the report is for another table",
                report_path.display()
            ),
        ));
    }
    let threshold = report.pointer("/ece/threshold").and_then(Value::as_f64);
    if threshold != Some(ECE_THRESHOLD) {
        return Err(refuse(
            k,
            format!(
                "{}: the report was fitted under ece.threshold {}, not the registered {ECE_THRESHOLD} (qd_runtime::calibration_fit::ECE_THRESHOLD)",
                report_path.display(),
                threshold.map_or_else(|| "absent".to_string(), |t| t.to_string())
            ),
        ));
    }

    let mut checked: BTreeMap<String, Value> = BTreeMap::new();
    for family in admitted {
        let shapes = trained.get(family).filter(|s| !s.is_empty()).ok_or_else(|| {
            refuse(
                k,
                format!(
                    "{family}: no recorded slot shapes; a trained family whose shapes are unknown cannot be shown calibrated. Exclude the family with --exclude-family"
                ),
            )
        })?;
        for shape in shapes {
            for key in shape {
                let key = key.as_str();
                // The runtime's own lookup decides whether the table has the entry, not a string
                // comparison of keys: a key the runtime would never build (`choice:04`) is no entry.
                runtime_lookup(table, key).map_err(|why| {
                    refuse(
                        k,
                        format!(
                            "{family}: trained slot shape {key} has no calibration entry the runtime would use ({why}); the runtime would refuse it as calibration_entry_missing"
                        ),
                    )
                })?;
                if key == "span" {
                    let fitted =
                        report.pointer("/span/fitted").and_then(Value::as_bool) == Some(true);
                    if !fitted {
                        return Err(refuse(
                            k,
                            format!(
                                "{family}: span has an entry in the table but the report says span.fitted is not true"
                            ),
                        ));
                    }
                    let v = check_ece(
                        &format!("{family}/span"),
                        report.pointer("/span/scored/ece_after"),
                    )?;
                    checked.insert("span".to_string(), json!(v));
                } else {
                    let ece = report.pointer(&format!("/entries/{key}/scored/ece_after"));
                    let v = check_ece(&format!("{family}/{key}"), ece)?;
                    checked.insert(key.to_string(), json!(v));
                }
            }
        }
    }

    let block = json!({
        "report_sha256": hex(&sha256(&report_bytes)),
        "report_path": report_path.display().to_string(),
        "table_sha256": table_sha,
        "ece_threshold": ECE_THRESHOLD,
        "ece_after_by_shape": checked,
        "rule": "every recorded (admitted family, slot shape) has a table entry whose recorded ece_after ran and passed at the registered threshold; a not_run check is a refusal",
    });
    Ok(GateRecord { block })
}
