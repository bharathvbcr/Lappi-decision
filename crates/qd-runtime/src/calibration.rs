//! Calibrated scores and split-conformal sets.
//!
//! `docs/schema-api.md`: *"`conformal_set` is a **split-conformal** set on calibrated scores —
//! margin, not entropy. The audits attribute 'entropy as confidence' to the Laya spec as a defect;
//! entropy is not used here."*
//!
//! So: temperature-scale the logits, softmax, and then
//!
//! * `score` is the **margin** `p_top - p_second`. It is not entropy, and there is no code path in
//!   this module that computes one.
//! * `conformal_set` is every row whose calibrated probability is at least the fitted quantile for
//!   that (slot kind, row count), with the top row always present so the set is never empty.
//! * `noul` fires when the reserved abstain row wins, or when the margin is below the fitted
//!   abstain threshold for that entry.
//!
//! # A missing entry is a refusal
//!
//! The thresholds are **per (slot kind, row count)**, because that is the granularity the plan
//! calibrates at. A request for a shape the table was never fitted for is
//! [`Refusal::CalibrationEntryMissing`], not a default: an uncalibrated answer would carry a
//! `score` and a `conformal_set` that were never fitted, and nothing downstream could tell.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::refusal::{BackendError, Refusal};
use crate::schema::{SlotKind, MAX_BINS, MAX_OPTIONS, MIN_BINS, MIN_OPTIONS, RESERVED_NOUL_ROWS};

/// One fitted entry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEntry {
    /// Temperature for the scaling step. `1.0` means the logits were already calibrated.
    pub temperature: f64,
    /// The split-conformal quantile: rows at or above this calibrated probability enter the set.
    pub conformal_quantile: f64,
    /// Below this margin the answer abstains. This is the runtime half of the abstain rule; the
    /// other half is the permuted second pass.
    pub noul_margin: f64,
}

impl CalibrationEntry {
    fn validate(&self, key: &str) -> Result<(), String> {
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(format!("{key}: temperature must be finite and > 0"));
        }
        if !(self.conformal_quantile.is_finite()
            && (0.0..=1.0).contains(&self.conformal_quantile))
        {
            return Err(format!("{key}: conformal_quantile must be in [0, 1]"));
        }
        if !(self.noul_margin.is_finite() && (0.0..=1.0).contains(&self.noul_margin)) {
            return Err(format!("{key}: noul_margin must be in [0, 1]"));
        }
        Ok(())
    }
}

/// The fitted table, keyed by the shape it was fitted for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationTable {
    pub name: String,
    /// `(slot kind, total decode rows including the reserved `noul` row) -> entry`.
    letters: BTreeMap<String, CalibrationEntry>,
    /// A `span` slot's row count is the context's line count, so it cannot be enumerated; one entry
    /// covers the pointer head.
    span: Option<CalibrationEntry>,
}

fn letters_key(kind: SlotKind, rows: usize) -> String {
    format!("{}:{rows}", kind.as_str())
}

impl CalibrationTable {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            letters: BTreeMap::new(),
            span: None,
        }
    }

    pub fn with_letters(mut self, kind: SlotKind, rows: usize, entry: CalibrationEntry) -> Self {
        self.letters.insert(letters_key(kind, rows), entry);
        self
    }

    pub fn with_span(mut self, entry: CalibrationEntry) -> Self {
        self.span = Some(entry);
        self
    }

    /// Refuse a table whose entries are not usable, at construction rather than at decode.
    pub fn validate(&self) -> Result<(), String> {
        for (key, entry) in &self.letters {
            entry.validate(key)?;
        }
        if let Some(entry) = &self.span {
            entry.validate("span")?;
        }
        Ok(())
    }

    pub fn lookup(
        &self,
        slot: &str,
        kind: SlotKind,
        rows: usize,
    ) -> Result<&CalibrationEntry, Refusal> {
        let found = match kind {
            SlotKind::Span => self.span.as_ref(),
            SlotKind::Choice | SlotKind::Score => self.letters.get(&letters_key(kind, rows)),
        };
        found.ok_or_else(|| Refusal::CalibrationEntryMissing {
            slot: slot.to_string(),
            slot_type: kind.as_str().to_string(),
            rows,
        })
    }

    /// SHA-256 over the canonical serialization, so a table can be hash-bound like the weights.
    pub fn hash(&self) -> String {
        // `BTreeMap` serializes in key order and the struct field order is fixed, so this is
        // canonical without a separate canonicalizer.
        match serde_json::to_vec(self) {
            Ok(bytes) => crate::hex(&crate::sha256(&bytes)),
            // Serializing a BTreeMap of plain numbers cannot fail. If it somehow did, a hash that
            // cannot be computed must not be a hash that matches everything.
            Err(e) => crate::hex(&crate::sha256(
                format!("qd-calibration-unserializable:{e}").as_bytes(),
            )),
        }
    }

    /// The table that ships with [`crate::reference::ReferenceBackend`].
    ///
    /// **These numbers were not fitted on anything.** They are placeholders in the only sense the
    /// repo allows one: they belong to a backend that already declares `is_model: false` and whose
    /// every answer is `degraded`, so no number derived from them can reach a caller unflagged. A
    /// real table is loaded from a file and hash-bound; see [`CalibrationTable::validate`].
    pub fn reference() -> Self {
        let mut table = Self::new("reference-uncalibrated-v1");
        // Named-option and ordinal slots: every legal row count, which is the named count plus the
        // reserved `noul` row.
        let lo = MIN_OPTIONS.min(MIN_BINS as usize);
        let hi = MAX_OPTIONS.max(MAX_BINS as usize);
        for named in lo..=hi {
            let rows = named + RESERVED_NOUL_ROWS;
            let entry = CalibrationEntry {
                temperature: 1.0,
                // A set that would otherwise admit every row on a flat distribution is not a set;
                // scaling the quantile with the row count keeps it informative at every k.
                conformal_quantile: 0.5 / rows as f64,
                noul_margin: 0.02,
            };
            table = table
                .with_letters(SlotKind::Choice, rows, entry)
                .with_letters(SlotKind::Score, rows, entry);
        }
        table.with_span(CalibrationEntry {
            temperature: 1.0,
            conformal_quantile: 0.0,
            noul_margin: 0.0,
        })
    }
}

/// The calibrated view of one decode.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibrated {
    pub probs: Vec<f64>,
    /// Row index of the winner.
    pub top: usize,
    /// `p_top - p_second`. The reported `score`. Never entropy.
    pub margin: f64,
    /// Row indices in the split-conformal set, ascending, always containing `top`.
    pub set: Vec<usize>,
}

/// Temperature-scale, softmax, and read off the margin and the conformal set.
pub fn calibrate(
    slot: &str,
    logits: &[f32],
    entry: &CalibrationEntry,
) -> Result<Calibrated, BackendError> {
    if logits.is_empty() {
        return Err(BackendError::LogitShapeMismatch {
            slot: slot.to_string(),
            expected: 1,
            actual: 0,
        });
    }
    let scaled: Vec<f64> = logits
        .iter()
        .map(|v| *v as f64 / entry.temperature)
        .collect();
    // Stabilised softmax. `scaled` is non-empty and every element is finite (the backend's logits
    // were checked and the temperature is > 0), so the fold has a maximum.
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if !(sum.is_finite() && sum > 0.0) {
        return Err(BackendError::DecodeFailed {
            detail: format!(
                "slot `{slot}`: softmax denominator is {sum}, so no calibrated probability exists"
            ),
        });
    }
    let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();

    let mut top = 0usize;
    for (i, p) in probs.iter().enumerate() {
        if *p > probs[top] {
            top = i;
        }
    }
    let second = probs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != top)
        .map(|(_, p)| *p)
        .fold(0.0f64, f64::max);
    let margin = probs[top] - second;

    let mut set: Vec<usize> = probs
        .iter()
        .enumerate()
        .filter(|(_, p)| **p >= entry.conformal_quantile)
        .map(|(i, _)| i)
        .collect();
    if !set.contains(&top) {
        set.push(top);
        set.sort_unstable();
    }

    Ok(Calibrated {
        probs,
        top,
        margin,
        set,
    })
}
