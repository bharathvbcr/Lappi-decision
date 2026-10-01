//! The tri-state a check reports: it ran (and passed or failed), or it did not run.
//!
//! A port of `python/qd_train/tristate.py`'s `Ran` / `NotRun` / `parse_tristate`, because the
//! shard set and the manifest carry their own statuses in that form and the reader must
//! refuse exactly the shapes Python refuses. The rule both encode: **a check that could not
//! run never reads as one that ran and passed**, so `NotRun` has no `passed` field at all and
//! a `ran` record without one is an error rather than a default.

use serde_json::Value;
use thiserror::Error;

/// One check's outcome.
#[derive(Clone, Debug, PartialEq)]
pub enum TriState {
    /// The check executed. `passed` is mandatory and never inferred.
    Ran {
        /// Whether it passed.
        passed: bool,
        /// What it measured, if it states one.
        value: Value,
        /// Items examined, carried with `n_total` or not at all.
        coverage: Option<Coverage>,
        /// Free text.
        detail: String,
    },
    /// The check did not execute. The reason is mandatory and non-empty.
    NotRun {
        /// Why it did not run.
        reason: String,
    },
}

/// `n` of `n_total` items examined. A bare `n` is refused: a sample size without a population
/// reads as full coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coverage {
    /// Examined.
    pub n: u64,
    /// Eligible.
    pub n_total: u64,
}

/// A tri-state record that Python's `parse_tristate` (or `Ran`/`NotRun` validation) refuses.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("{field}: {detail}")]
pub struct TriStateError {
    /// Where the record came from.
    pub field: String,
    /// What is wrong with it.
    pub detail: String,
}

impl TriState {
    /// A passing `Ran` with no measured value -- the shape this crate records for a door check.
    pub fn passed(value: impl Into<Value>, detail: impl Into<String>) -> Self {
        Self::Ran {
            passed: true,
            value: value.into(),
            coverage: None,
            detail: detail.into(),
        }
    }

    /// A `NotRun` with a reason; an empty reason is a programming error caught here.
    pub fn not_run(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        assert!(
            !reason.trim().is_empty(),
            "NotRun requires a non-empty reason"
        );
        Self::NotRun { reason }
    }

    /// `true` only for `Ran { passed: true, .. }`.
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Ran { passed: true, .. })
    }

    /// `true` for `Ran { passed: false, .. }`; `NotRun` is neither a pass nor a failure.
    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Ran { passed: false, .. })
    }
}

/// `qd_train.tristate.parse_tristate`, refusing every shape that could read as a false pass.
pub fn parse_tristate(raw: &Value, field: &str) -> Result<TriState, TriStateError> {
    let err = |detail: String| TriStateError {
        field: field.to_owned(),
        detail,
    };
    let Value::Object(map) = raw else {
        return Err(err(format!(
            "tri-state must be an object, got {}",
            json_type(raw)
        )));
    };
    match map.get("state").and_then(Value::as_str) {
        Some("ran") => {
            let passed = match map.get("passed") {
                None => {
                    return Err(err(
                        "state='ran' without a 'passed' field. A check that ran must say \
                         whether it passed; absence is not success."
                            .to_owned(),
                    ));
                }
                Some(Value::Bool(b)) => *b,
                Some(other) => {
                    return Err(err(format!(
                        "'passed' must be bool, got {}",
                        json_type(other)
                    )));
                }
            };
            let n = optional_count(map.get("n")).map_err(|d| err(format!("n: {d}")))?;
            let n_total =
                optional_count(map.get("n_total")).map_err(|d| err(format!("n_total: {d}")))?;
            let coverage = match (n, n_total) {
                (None, None) => None,
                (Some(n), Some(n_total)) if n <= n_total => Some(Coverage { n, n_total }),
                (Some(n), Some(n_total)) => {
                    return Err(err(format!("examined {n} of {n_total}: n exceeds n_total")));
                }
                (n, n_total) => {
                    return Err(err(format!(
                        "n and n_total are carried together or not at all: got n={n:?}, \
                         n_total={n_total:?}. A sample size without a population reads as \
                         full coverage."
                    )));
                }
            };
            let detail = match map.get("detail") {
                None => String::new(),
                Some(Value::String(s)) => s.clone(),
                Some(other) => {
                    return Err(err(format!(
                        "'detail' must be str, got {}",
                        json_type(other)
                    )));
                }
            };
            Ok(TriState::Ran {
                passed,
                value: map.get("value").cloned().unwrap_or(Value::Null),
                coverage,
                detail,
            })
        }
        Some("not_run") => {
            let reason = match map.get("reason") {
                None => String::new(),
                Some(Value::String(s)) => s.clone(),
                Some(other) => {
                    return Err(err(format!(
                        "'reason' must be str, got {}",
                        json_type(other)
                    )));
                }
            };
            if map.contains_key("passed") {
                return Err(err(
                    "state='not_run' carries a 'passed' field. A check that did not run has \
                     no pass/fail result; carrying one is how a skipped suite becomes a green \
                     one."
                        .to_owned(),
                ));
            }
            if reason.trim().is_empty() {
                return Err(err(
                    "NotRun requires a non-empty reason. A check recorded as not-run without \
                     saying why cannot be distinguished from one that was forgotten."
                        .to_owned(),
                ));
            }
            Ok(TriState::NotRun { reason })
        }
        _ => Err(err(format!(
            "tri-state 'state' must be 'ran' or 'not_run', got {}. There is no third state \
             and no default.",
            map.get("state")
                .map_or_else(|| "None".to_owned(), Value::to_string)
        ))),
    }
}

fn optional_count(raw: Option<&Value>) -> Result<Option<u64>, String> {
    match raw {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => match n.as_u64() {
            Some(v) => Ok(Some(v)),
            None => Err(format!("must be a non-negative integer, got {n}")),
        },
        Some(other) => Err(format!("must be an integer, got {}", json_type(other))),
    }
}

fn json_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ran_without_passed_is_refused_not_defaulted() {
        assert!(parse_tristate(&json!({"state": "ran"}), "f").is_err());
        assert!(parse_tristate(&json!({"state": "ran", "passed": 1}), "f").is_err());
    }

    #[test]
    fn not_run_with_passed_or_without_reason_is_refused() {
        assert!(
            parse_tristate(
                &json!({"state": "not_run", "reason": "x", "passed": true}),
                "f"
            )
            .is_err()
        );
        assert!(parse_tristate(&json!({"state": "not_run", "reason": "  "}), "f").is_err());
        assert!(parse_tristate(&json!({"state": "not_run"}), "f").is_err());
    }

    #[test]
    fn coverage_travels_as_a_pair() {
        assert!(parse_tristate(&json!({"state": "ran", "passed": true, "n": 3}), "f").is_err());
        assert!(
            parse_tristate(
                &json!({"state": "ran", "passed": true, "n": 4, "n_total": 3}),
                "f"
            )
            .is_err()
        );
        let ok = parse_tristate(
            &json!({"state": "ran", "passed": false, "n": 3, "n_total": 4}),
            "f",
        )
        .unwrap();
        assert!(ok.is_fail());
    }

    #[test]
    fn there_is_no_third_state() {
        assert!(parse_tristate(&json!({"state": "skipped"}), "f").is_err());
        assert!(parse_tristate(&json!([]), "f").is_err());
    }
}
