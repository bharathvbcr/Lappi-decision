//! The parity gate's result as a `smoke` row in `ledger/mac-qd-metal-*.jsonl`.
//!
//! `qd-metal-parity --ledger` (main 276a475) appends its own JSON object
//! (`{"kind": "qd-metal-parity", "row_id": "qdm-parity-…", …}`), which is not a ledger-schema row:
//! no `run_kind`, no `prev_row_hash`, no `protocol`, no tri-states. Written into `ledger/`, it
//! breaks `python/tests/test_ledger_provenance.py` (it reads `run_kind` from every row) and
//! `test_ledger.py`'s `verify_chain` over every file. The fix belongs in `bin/parity.rs`, which
//! another session holds (`GAP-QDM-PARITY-LEDGER-ROW-NOT-SCHEMA-2026-10-02`), so
//! `qd-metal-record-parity` runs the gate with `--ledger` pointed at a scratch file and this
//! module turns that object into a schema row. The thresholds are read from the object, never
//! restated here (rule 2).
//!
//! # A pass from main's parity.rs is not taken on its word
//!
//! Main's `parity.rs` folds its distances with `f64::max` from 0, which drops NaN: an all-NaN GPU
//! output passes every absolute threshold. serde_json writes a NaN measurement as `null`, so
//! every `null` where the object should hold a number is counted here, and so is any `NaN` /
//! `inf` in the gate's stdout. One is enough to fail the row's `parity.gate`, whatever the object
//! says.

use std::collections::BTreeMap;

use qd_train::ledger::{Environment, Protocol, Row, Status, WallClockSource};
use qd_train::tristate::TriState;
use serde_json::{json, Value};

use crate::error::{MetalError, Result};
use crate::ledger::{self, Provenance};

/// What one run of `qd-metal-parity` left behind.
#[derive(Debug, Clone, PartialEq)]
pub struct ParityRun {
    /// The one object it appended to its scratch `--ledger` file; `None` when it wrote none
    /// (exit 2: the gate could not run).
    pub raw_line: Option<String>,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub wall_clock_s: f64,
    /// The arguments it was run with, after the binary.
    pub argv: Vec<String>,
}

/// Where the row says the model came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFacts {
    pub snapshot_name: String,
    pub vocab: usize,
}

fn err(m: impl Into<String>) -> MetalError {
    MetalError::Ledger(m.into())
}

/// Numbers the raw object should hold that are `null` (a NaN or an infinity serialized), and
/// `NaN` / `inf` tokens in the gate's stdout.
pub fn non_finite_count(raw: &Value, stdout: &str) -> usize {
    let numeric = [
        "rel_l2_bf16in",
        "rel_l2_fp32",
        "max_abs_fp32",
        "ref_max_abs",
        "letter_logprob_maxabs_bf16in",
        "letter_logprob_maxabs_fp32",
        "ref_margin_fp32",
        "snapshot_path_maxabs",
        "argmax_agreement",
    ];
    fn walk(v: &Value, numeric: &[&str], n: &mut usize) {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    if numeric.contains(&k.as_str()) && !x.is_number() {
                        *n += 1;
                    }
                    walk(x, numeric, n);
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, numeric, n)),
            _ => {}
        }
    }
    let mut n = 0;
    walk(raw, &numeric, &mut n);
    n + stdout
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| matches!(w.to_ascii_lowercase().as_str(), "nan" | "inf" | "infinity"))
        .count()
}

fn num(v: &Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| err(format!("parity object: {key} is not a number")))
}

/// The worst value of `key` over `items`, or `None` when any is missing or not finite (which
/// [`non_finite_count`] has already counted).
fn worst(items: &[Value], key: &str) -> Option<f64> {
    let mut w = 0f64;
    for it in items {
        let x = it.get(key).and_then(Value::as_f64)?;
        if !x.is_finite() {
            return None;
        }
        w = w.max(x);
    }
    Some(w)
}

fn bounded(name: &str, items: &[Value], key: &str, limit: f64, what: &str) -> Result<(String, TriState)> {
    let t = match worst(items, key) {
        Some(w) => TriState::ran(w <= limit, ledger::float(w)?).with_detail(format!(
            "{what}: worst over {} item(s); the gate's bound is {limit} (read from the gate's own object)",
            items.len()
        )),
        None => TriState::ran(false, Value::Null)
            .with_detail(format!("{what}: a value is missing or not finite; a NaN is a failure, never a 0")),
    };
    Ok((name.to_string(), t))
}

/// The row for one parity run.
pub fn build_row(run: &ParityRun, prov: &Provenance, model: &ModelFacts) -> Result<Row> {
    let mut m: BTreeMap<String, TriState> = prov.metrics();
    m.insert(
        "parity.exit_code".into(),
        TriState::ran(run.exit_code == 0, run.exit_code)
            .with_detail("qd-metal-parity: 0 every threshold held, 1 a threshold failed, 2 the gate could not run"),
    );
    let raw: Option<Value> = match &run.raw_line {
        Some(line) => Some(serde_json::from_str(line).map_err(|e| err(format!("parity object: {e}")))?),
        None => None,
    };
    let (status, recipe_extra, protocol_data, tokenizer_hash) = match &raw {
        None => {
            let tail: String = run.stderr.lines().rev().take(5).collect::<Vec<_>>().join(" | ");
            m.insert(
                "parity.gate".into(),
                TriState::not_run(format!(
                    "qd-metal-parity wrote no result (exit {}): {tail}",
                    run.exit_code
                )),
            );
            (Status::Failed, json!({}), "none: the gate wrote no result".to_string(), "unknown: the gate wrote no result".to_string())
        }
        Some(raw) => {
            if raw.get("kind").and_then(Value::as_str) != Some("qd-metal-parity") {
                return Err(err("the scratch object is not a qd-metal-parity result"));
            }
            let thresholds = raw.get("thresholds").cloned().ok_or_else(|| err("parity object: no thresholds"))?;
            let th = |k: &str| num(&thresholds, k);
            let prompts = raw.get("prompts").and_then(Value::as_array).cloned().ok_or_else(|| err("parity object: no prompts"))?;
            let layers = raw.get("per_layer_worst").and_then(Value::as_array).cloned().ok_or_else(|| err("parity object: no per_layer_worst"))?;
            let n_prompts = prompts.len() as u64;
            let non_finite = non_finite_count(raw, &run.stdout);
            let failures: Vec<String> = raw
                .get("failures")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|f| f.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            let said_pass = raw.get("gate").and_then(Value::as_str) == Some("pass");
            let gate_pass = said_pass && run.exit_code == 0 && non_finite == 0;
            m.insert(
                "parity.gate".into(),
                TriState::ran(gate_pass, if gate_pass { "pass" } else { "fail" }).with_detail(format!(
                    "qd-metal-parity said {:?} with {} failure(s){}; {non_finite} non-finite value(s). \
                     Fails on any non-finite value whatever the gate said: main's parity.rs drops NaN \
                     in its max-folds",
                    raw.get("gate").and_then(Value::as_str).unwrap_or("?"),
                    failures.len(),
                    if failures.is_empty() { String::new() } else { format!(" (first: {})", failures[0]) },
                )),
            );
            m.insert(
                "parity.non_finite_values".into(),
                TriState::ran(non_finite == 0, non_finite as u64)
                    .with_detail("nulls where the gate's object holds a number, plus NaN/inf tokens in its stdout"),
            );
            let agree = prompts
                .iter()
                .filter(|p| p.get("argmax_metal").is_some() && p.get("argmax_metal") == p.get("argmax_fp32"))
                .count() as u64;
            let agreement = num(raw, "argmax_agreement")?;
            let min_agree = th("argmax_min_agreement")?;
            m.insert(
                "parity.argmax_agreement".into(),
                TriState::ran(agreement >= min_agree, ledger::float(agreement)?)
                    .with_coverage(agree, n_prompts)
                    .with_detail(format!("17-row argmax equal to torch fp32's; the gate's bound is {min_agree}")),
            );
            for (k, v) in [
                bounded("parity.worst_layer_rel_l2_vs_bf16in", &layers, "rel_l2_bf16in", th("tight_layer_rel_l2_vs_bf16in")?, "per-layer residual rel L2 vs the bf16-input torch reference")?,
                bounded("parity.worst_layer_rel_l2_vs_fp32", &layers, "rel_l2_fp32", th("loose_layer_rel_l2_vs_fp32")?, "per-layer residual rel L2 vs torch fp32")?,
                bounded("parity.worst_letter_logprob_vs_bf16in", &prompts, "letter_logprob_maxabs_bf16in", th("tight_logprob_abs_vs_bf16in")?, "max |d log-softmax| over the 17 letter rows vs bf16-input torch")?,
                bounded("parity.worst_letter_logprob_vs_fp32", &prompts, "letter_logprob_maxabs_fp32", th("loose_logprob_abs_vs_fp32")?, "max |d log-softmax| over the 17 letter rows vs torch fp32")?,
                bounded("parity.worst_snapshot_path", &prompts, "snapshot_path_maxabs", th("snapshot_path_logprob_abs")?, "prefill-the-prefix + continue-the-suffix vs one pass, same backend")?,
            ] {
                m.insert(k, v.with_coverage(n_prompts, n_prompts));
            }
            let fixtures = raw.get("fixtures_manifest_sha256").and_then(Value::as_str).unwrap_or("").to_string();
            m.insert(
                "parity.fixtures".into(),
                TriState::passed(fixtures.as_str(), format!("sha256 of the torch reference's manifest.json; reference {}", raw.get("reference").unwrap_or(&Value::Null))),
            );
            m.insert(
                "parity.raw_object_sha256".into(),
                TriState::passed(
                    qd_runtime::hex(&qd_runtime::sha256(run.raw_line.as_deref().unwrap_or("").as_bytes())),
                    format!("the gate's own object (row_id {}), kept in the scratch file qd-metal-record-parity names", raw.get("row_id").unwrap_or(&Value::Null)),
                ),
            );
            for key in ["weight_hash", "device"] {
                if let Some(v) = raw.get(key).and_then(Value::as_str) {
                    m.insert(key.to_string(), TriState::passed(v, "as qd-metal-parity reported it"));
                }
            }
            let status = Status::Completed;
            let tok = raw.get("tokenizer_hash").and_then(Value::as_str).unwrap_or("unknown").to_string();
            (status, json!({"thresholds": thresholds, "n_prompts": n_prompts}), fixtures, tok)
        }
    };
    let mut recipe = json!({
        "tool": "crates/qd-metal/src/bin/parity.rs, run by crates/qd-metal/src/bin/record_parity.rs",
        "parity_rs": "as compiled into code_that_ran; main 276a475 drops NaN in its max-folds, so non-finite values are counted here",
        "argv": run.argv,
        "backbone_snapshot": model.snapshot_name,
        "backbone_vocab": model.vocab,
    });
    if let (Some(r), Some(extra)) = (recipe.as_object_mut(), recipe_extra.as_object()) {
        for (k, v) in extra {
            r.insert(k.clone(), v.clone());
        }
    }
    let protocol = Protocol {
        data_snapshot_hash: if protocol_data.is_empty() { "unknown: no fixtures digest".into() } else { protocol_data },
        tokenizer_hash,
        backbone_commit: format!("{}:vocab{}", model.snapshot_name, model.vocab),
        recipe_hash: ledger::recipe_hash(&recipe)?,
        seed: 0,
    };
    Ok(Row {
        run_kind: "smoke".into(),
        protocol,
        status,
        quick_reason: "a parity check of the Metal backend on the base weights against a fixed CPU torch \
                       reference: one run, no training; rule 8: quick, excluded from every decision"
            .into(),
        code_commit: prov.code_commit.clone(),
        env: Environment {
            torch: "n/a: the gate is a Rust binary; the reference's torch version is in parity.fixtures".into(),
            transformers_sha: "n/a: Rust binary, no transformers in the process".into(),
            device: "metal".into(),
            host: prov.host.clone(),
        },
        metrics: m,
        noul_rate: TriState::not_run("a parity check decodes no verdicts against a gate"),
        wall_clock_s: run.wall_clock_s,
        wall_clock_source: WallClockSource::Recorder,
        notes: format!(
            "qd-metal-parity on {} (Qwen3.5-2B-Base: the base weights every qd-metal tool resolves; no \
             trained release exists on this Mac), recorded by qd-metal-record-parity.",
            model.snapshot_name
        ),
        recipe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nan_is_counted_wherever_main_parity_would_have_hidden_it() {
        let raw = json!({"per_layer_worst": [{"rel_l2_bf16in": 0.01}, {"rel_l2_bf16in": null}],
                         "prompts": [{"snapshot_path_maxabs": null, "layers": [{"rel_l2_fp32": null}]}],
                         "argmax_agreement": 1.0});
        assert_eq!(non_finite_count(&raw, ""), 3);
        assert_eq!(non_finite_count(&json!({"argmax_agreement": 1.0}), "letters max|d| NaN; inf here"), 2);
        assert_eq!(non_finite_count(&json!({"argmax_agreement": 1.0}), "infer the info; nanoseconds"), 0);
    }

    #[test]
    fn worst_refuses_to_fold_a_missing_value_into_zero() {
        assert_eq!(worst(&[json!({"x": 0.1}), json!({"x": 0.3})], "x"), Some(0.3));
        assert_eq!(worst(&[json!({"x": 0.1}), json!({"x": null})], "x"), None);
        assert_eq!(worst(&[json!({"x": 0.1}), json!({})], "x"), None);
    }
}
