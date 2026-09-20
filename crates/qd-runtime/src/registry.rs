//! The registered route: head files, hash-bound to the backbone.
//!
//! `docs/schema-api.md`: *"An app ships a head file, hash-bound to the backbone, and gets a single
//! GEMV on pooled features instead of option text in the prompt. **Same prefill, same calibration
//! table, same abstain rule.** A registered task is a speed-up, never a different model."*
//!
//! Two checks make "hash-bound" mean something:
//!
//! 1. the task must have a head at all — [`crate::refusal::Refusal::RegisteredHeadMissing`], which
//!    names the tasks this build does have;
//! 2. the head's declared `backbone_hash` must equal the loaded backend's `weight_hash` —
//!    [`crate::refusal::HashKind::HeadBackboneBinding`]. A head fitted on other weights produces a
//!    confident, wrong letter, and nothing downstream would show it.
//!
//! Both are refusals rather than fallbacks to the generic route. Silently answering a `registered`
//! request on the generic route would make the route field advisory, and the agreement test the
//! contract asks for ("generic-route and registered-route answers must agree on the held-out set at
//! a stated rate") would then be comparing the generic route with itself.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::backend::BackendIdentity;
use crate::refusal::{BackendError, HashKind, Refusal};

/// One slot's rows of the head: `rows x feature_dim`, in decode-row order, the reserved `noul` row
/// last.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadMatrix {
    pub rows: Vec<Vec<f32>>,
}

impl HeadMatrix {
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// A head file for one task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisteredHead {
    pub task: String,
    /// SHA-256 of the head file, for the caller's `expect.head_hash`.
    pub head_hash: String,
    /// The backbone this head was fitted on. Must equal the backend's `weight_hash`.
    pub backbone_hash: String,
    pub feature_dim: usize,
    /// Slot name -> the head rows for that slot.
    pub slots: BTreeMap<String, HeadMatrix>,
}

/// Every head this build carries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HeadRegistry {
    heads: BTreeMap<String, RegisteredHead>,
}

impl HeadRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, head: RegisteredHead) {
        self.heads.insert(head.task.clone(), head);
    }

    pub fn tasks(&self) -> Vec<String> {
        self.heads.keys().cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// Resolve a task's head and check its binding to the loaded backbone.
    pub fn resolve(
        &self,
        task: &str,
        identity: &BackendIdentity,
    ) -> Result<&RegisteredHead, Refusal> {
        let head = self
            .heads
            .get(task)
            .ok_or_else(|| Refusal::RegisteredHeadMissing {
                task: task.to_string(),
                available: self.tasks(),
            })?;
        if head.backbone_hash != identity.weight_hash {
            return Err(Refusal::HashMismatch {
                which: HashKind::HeadBackboneBinding,
                expected: identity.weight_hash.clone(),
                actual: head.backbone_hash.clone(),
            });
        }
        Ok(head)
    }
}

/// The registered route's single GEMV: `logits = M * features`.
///
/// A width or row-count disagreement is a [`BackendError::LogitShapeMismatch`], never a truncated
/// dot product. The accumulation is in `f64` so a wide feature vector does not lose the margin the
/// abstain rule is about to read.
pub fn gemv(
    slot: &str,
    matrix: &HeadMatrix,
    features: &[f32],
    expected_rows: usize,
) -> Result<Vec<f32>, BackendError> {
    if matrix.row_count() != expected_rows {
        return Err(BackendError::LogitShapeMismatch {
            slot: slot.to_string(),
            expected: expected_rows,
            actual: matrix.row_count(),
        });
    }
    let mut out = Vec::with_capacity(matrix.rows.len());
    for (row_index, row) in matrix.rows.iter().enumerate() {
        if row.len() != features.len() {
            return Err(BackendError::LogitShapeMismatch {
                slot: format!("{slot}[row {row_index}]"),
                expected: features.len(),
                actual: row.len(),
            });
        }
        let mut acc = 0.0f64;
        for (w, f) in row.iter().zip(features) {
            acc += *w as f64 * *f as f64;
        }
        let value = acc as f32;
        if !value.is_finite() {
            return Err(BackendError::NonFiniteLogit {
                slot: slot.to_string(),
                row: row_index,
            });
        }
        out.push(value);
    }
    Ok(out)
}
