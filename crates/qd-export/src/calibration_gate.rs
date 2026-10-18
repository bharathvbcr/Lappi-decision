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
//! * the slot shapes each trained family was trained on, `{family: [["choice:5", "span"], ["choice:7"]]}`: one array per trained request shape, so choice and span trained together are declared together. No
//!   tracked registry of these exists (the train manifest records families, not slots), so the
//!   operator declares them; a trained family with no declaration is refused.
//!
//! A check that did not run (`state: "not_run"`, e.g. fewer than 100 examples) is a refusal, never
//! a pass. The only way past a family is [`ExcludedFamily`]: the family leaves
//! `trained_families` (the runtime then refuses its tasks) and the reason is recorded in the
//! manifest.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Map, Value, json};

use qd_runtime::calibration::CalibrationTable;
use qd_runtime::calibration_fit::ECE_THRESHOLD;
use qd_runtime::hex;

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
    /// Declared shapes of the admitted families only.
    pub shapes: BTreeMap<String, Vec<Vec<String>>>,
    /// The `calibration.gate` block.
    pub block: Value,
}

fn is_shape_key(key: &str) -> bool {
    match key.split_once(':') {
        None => key == "span",
        Some((kind, rows)) => {
            matches!(kind, "choice" | "score")
                && !rows.is_empty()
                && rows.bytes().all(|b| b.is_ascii_digit())
        }
    }
}

fn as_object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| {
        refuse(
            RefusalKind::Calibration,
            format!("{what} is not a JSON object"),
        )
    })
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

/// Run the gate. `all_families` is the train manifest's set; `admitted` is that set minus the
/// exclusions.
pub(crate) fn check(
    table_bytes: &[u8],
    table: &CalibrationTable,
    report_path: &Path,
    shapes_path: &Path,
    all_families: &[String],
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

    let shapes_bytes = read_small(shapes_path, k)?;
    let shapes_doc: Value = serde_json::from_slice(&shapes_bytes)
        .map_err(|e| refuse(k, format!("{}: not JSON: {e}", shapes_path.display())))?;
    let declared = as_object(&shapes_doc, &shapes_path.display().to_string())?;
    let known: BTreeSet<&str> = all_families.iter().map(String::as_str).collect();
    for family in declared.keys() {
        if !known.contains(family.as_str()) {
            return Err(refuse(
                k,
                format!(
                    "{}: declares shapes for {family:?}, which the train manifest does not hold",
                    shapes_path.display()
                ),
            ));
        }
    }

    let table_doc = serde_json::to_value(table).map_err(|e| refuse(k, e.to_string()))?;
    let letters: BTreeSet<&str> = table_doc
        .get("letters")
        .and_then(Value::as_object)
        .map(|m| m.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let has_span = table_doc.get("span").is_some_and(|s| !s.is_null());

    let mut shapes: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
    let mut checked: BTreeMap<String, Value> = BTreeMap::new();
    for family in admitted {
        let keys = declared
            .get(family)
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
            .ok_or_else(|| {
                refuse(
                    k,
                    format!(
                        "{family}: no declared slot shapes in {}; a trained family whose shapes are unknown cannot be shown calibrated. Declare them, or exclude the family with --exclude-family",
                        shapes_path.display()
                    ),
                )
            })?;
        let mut mine: Vec<Vec<String>> = Vec::with_capacity(keys.len());
        for shape in keys {
            let slots = shape.as_array().filter(|s| !s.is_empty()).ok_or_else(|| {
                refuse(k, format!("{family}: each declared shape is a non-empty array of slot keys (one trained request), got {shape}"))
            })?;
            let mut this: Vec<String> = Vec::with_capacity(slots.len());
            for key in slots {
                let key = key.as_str().filter(|s| is_shape_key(s)).ok_or_else(|| {
                    refuse(
                        k,
                        format!("{family}: shape {key} is not span, choice:N or score:N"),
                    )
                })?;
                if key == "span" {
                    if !has_span {
                        return Err(refuse(
                            k,
                            format!(
                                "{family}: trained slot shape span has no calibration entry (the table's span is null); the runtime would refuse its span slot as calibration_entry_missing"
                            ),
                        ));
                    }
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
                    if !letters.contains(key) {
                        return Err(refuse(
                            k,
                            format!(
                                "{family}: trained slot shape {key} has no calibration entry; the runtime would refuse it as calibration_entry_missing"
                            ),
                        ));
                    }
                    let ece = report.pointer(&format!("/entries/{key}/scored/ece_after"));
                    let v = check_ece(&format!("{family}/{key}"), ece)?;
                    checked.insert(key.to_string(), json!(v));
                }
                this.push(key.to_string());
            }
            this.sort();
            mine.push(this);
        }
        shapes.insert(family.clone(), mine);
    }

    let block = json!({
        "report_sha256": hex(&sha256(&report_bytes)),
        "report_path": report_path.display().to_string(),
        "table_sha256": table_sha,
        "ece_threshold": ECE_THRESHOLD,
        "ece_after_by_shape": checked,
        "rule": "every declared (admitted family, slot shape) has a table entry whose recorded ece_after ran and passed at the registered threshold; a not_run check is a refusal",
    });
    Ok(GateRecord { shapes, block })
}
