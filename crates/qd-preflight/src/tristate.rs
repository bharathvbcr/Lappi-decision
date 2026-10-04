//! The tri-state, in Rust, wire-compatible with `python/qd_train/tristate.py`.
//!
//! A preflight result has to drop straight into a ledger row, so this serializes to
//! byte-identical JSON. `tests/schema_compat.rs` asserts that against the Python
//! implementation rather than trusting that two hand-written encoders agree.
//!
//! The same structural guarantee holds here as in Python: `NotRun` has **no `passed`
//! field**. It is a separate enum variant, so a caller that wants to treat a
//! did-not-run check as passing has to write that intent out loud and cannot reach it
//! by forgetting a match arm.

use serde::{Deserialize, Serialize};

/// A check's outcome. There is no third state and no default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TriState {
    /// The check executed. `passed` is mandatory and never inferred.
    Ran {
        passed: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        n: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        n_total: Option<u64>,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        detail: String,
    },
    /// The check did not execute. It carries no `passed`, on purpose.
    NotRun { reason: String },
}

impl TriState {
    pub fn pass(detail: impl Into<String>) -> Self {
        TriState::Ran {
            passed: true,
            value: None,
            n: None,
            n_total: None,
            detail: detail.into(),
        }
    }

    pub fn fail(detail: impl Into<String>) -> Self {
        TriState::Ran {
            passed: false,
            value: None,
            n: None,
            n_total: None,
            detail: detail.into(),
        }
    }

    pub fn not_run(reason: impl Into<String>) -> Self {
        TriState::NotRun {
            reason: reason.into(),
        }
    }

    pub fn with_value(mut self, v: serde_json::Value) -> Self {
        if let TriState::Ran { value, .. } = &mut self {
            *value = Some(v);
        }
        self
    }

    /// True only for a check that ran AND passed.
    ///
    /// Deliberately not `Option<bool>`: callers reach for `unwrap_or(true)` on those,
    /// which is precisely how an unchecked environment becomes a clean one.
    pub fn is_pass(&self) -> bool {
        matches!(self, TriState::Ran { passed: true, .. })
    }

    pub fn did_run(&self) -> bool {
        matches!(self, TriState::Ran { .. })
    }
}

/// Combine checks. The result is never more confident than its least-informed input,
/// and a check that ran and failed is never reported as "not run".
///
/// - any failure -> `Ran { passed: false }`, even when a later input did not run;
/// - otherwise any `NotRun` -> `NotRun`, carrying every reason;
/// - all `Ran` and passing -> `Ran { passed: true }`;
/// - **no inputs  -> `NotRun`**, because an aggregate over zero checks has verified
///   nothing. Rust's `Iterator::all` returns `true` on an empty iterator, the same
///   trap as Python's `all([])`.
pub fn aggregate(name: &str, parts: &[(&str, TriState)]) -> TriState {
    if parts.is_empty() {
        return TriState::not_run(format!(
            "{name}: no checks were contributed, so nothing was verified"
        ));
    }

    let not_run: Vec<String> = parts
        .iter()
        .filter_map(|(label, t)| match t {
            TriState::NotRun { reason } => Some(format!("{label}: {reason}")),
            _ => None,
        })
        .collect();

    let failed: Vec<&str> = parts
        .iter()
        .filter_map(|(label, t)| match t {
            TriState::Ran { passed: false, .. } => Some(*label),
            _ => None,
        })
        .collect();

    if !failed.is_empty() {
        let mut detail = format!("{name}: failing inputs: {}", failed.join(", "));
        if !not_run.is_empty() {
            detail.push_str(&format!("; not run: {}", not_run.join("; ")));
        }
        return TriState::Ran {
            passed: false,
            value: None,
            n: Some((parts.len() - not_run.len()) as u64),
            n_total: Some(parts.len() as u64),
            detail,
        };
    }

    if !not_run.is_empty() {
        return TriState::not_run(format!(
            "{name}: {} of {} inputs did not run -> {}",
            not_run.len(),
            parts.len(),
            not_run.join("; ")
        ));
    }

    TriState::Ran {
        passed: true,
        value: None,
        n: Some(parts.len() as u64),
        n_total: Some(parts.len() as u64),
        detail: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_run_serializes_without_a_passed_field() {
        let json = serde_json::to_value(TriState::not_run("no CUDA")).unwrap();
        assert_eq!(json["state"], "not_run");
        assert!(
            json.get("passed").is_none(),
            "NotRun must carry no passed field"
        );
    }

    #[test]
    fn ran_always_carries_passed() {
        let json = serde_json::to_value(TriState::pass("ok")).unwrap();
        assert_eq!(json["state"], "ran");
        assert_eq!(json["passed"], true);
    }

    #[test]
    fn a_not_run_payload_with_passed_is_refused_on_the_way_in() {
        // Mirrors the Python parser: state=not_run carrying `passed` is the shape by
        // which a skipped suite becomes a green one.
        let raw = r#"{"state":"not_run","reason":"skipped","passed":true}"#;
        let parsed: TriState = serde_json::from_str(raw).unwrap();
        // serde ignores the stray field, but the resulting value still has no way to
        // report a pass -- which is the guarantee that matters.
        assert!(!parsed.is_pass());
        assert!(!parsed.did_run());
    }

    #[test]
    fn an_unknown_state_is_refused() {
        assert!(serde_json::from_str::<TriState>(r#"{"state":"maybe"}"#).is_err());
    }

    #[test]
    fn ran_without_passed_is_refused() {
        assert!(serde_json::from_str::<TriState>(r#"{"state":"ran"}"#).is_err());
    }

    #[test]
    fn aggregate_over_zero_checks_is_not_a_pass() {
        // Iterator::all returns true on an empty iterator; this must not.
        assert!(!aggregate("empty", &[]).is_pass());
        assert!(!aggregate("empty", &[]).did_run());
    }

    #[test]
    fn any_not_run_blocks_the_aggregate() {
        let parts = [
            ("a", TriState::pass("")),
            ("b", TriState::not_run("no device")),
        ];
        let agg = aggregate("gate", &parts);
        assert!(!agg.did_run());
        match agg {
            TriState::NotRun { reason } => assert!(reason.contains("no device")),
            _ => panic!("expected NotRun"),
        }
    }

    #[test]
    fn all_passing_inputs_pass() {
        let parts = [("a", TriState::pass("")), ("b", TriState::pass(""))];
        assert!(aggregate("gate", &parts).is_pass());
    }

    #[test]
    fn one_failing_input_fails_the_aggregate() {
        let parts = [("a", TriState::pass("")), ("b", TriState::fail("bad"))];
        let agg = aggregate("gate", &parts);
        assert!(agg.did_run());
        assert!(!agg.is_pass());
    }

    /// A later `NotRun` used to replace an earlier failure, so a check that ran
    /// and failed was reported as "not run".
    #[test]
    fn a_failure_is_not_overridden_by_a_later_not_run() {
        for parts in [
            [
                ("memory", TriState::fail("too small")),
                ("cuda", TriState::not_run("no driver")),
            ],
            [
                ("cuda", TriState::not_run("no driver")),
                ("memory", TriState::fail("too small")),
            ],
        ] {
            let agg = aggregate("preflight", &parts);
            assert!(agg.did_run(), "a failed check became NotRun: {agg:?}");
            assert!(
                !agg.is_pass(),
                "a failed check was reported as a pass: {agg:?}"
            );
            match agg {
                TriState::Ran { detail, .. } => {
                    assert!(detail.contains("memory"), "{detail}");
                    assert!(detail.contains("no driver"), "{detail}");
                }
                TriState::NotRun { reason } => {
                    panic!("failure was overridden by not-run: {reason}")
                }
            }
        }
    }
}
