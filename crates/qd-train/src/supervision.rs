//! What one FT batch supervises: a port of `python/qd_train/trainer.py::ft_supervision` and
//! `python/qd_train/heads.py::plan_span_batch`.
//!
//! **The letter channel** supervises exactly `Batch.target_index` -- position `p` of row `r`
//! predicts `tokens[r, p + 1]`, and `mask[r, p]` is set only at `p == target_index[r]` -- read
//! from the batch, never inferred as `lengths - 2`.
//!
//! **Span rows are excluded from the letter mask.** A span slot's only letter is `noul`, so
//! supervising it as a letter would train the span head to abstain always, with a loss curve
//! that looked fine (`GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL`). They travel in
//! [`SpanSupervision`] instead, and [`plan_span_batch`] lays them out in the head's row order:
//! candidates (the row's line starts, ascending), then **one** reserved abstain row last,
//! because `crates/qd-runtime/src/answer.rs` reads `noul_row = plan.rows - RESERVED_NOUL_ROWS`.

use thiserror::Error;

use crate::shards::{Batch, NO_SPAN, SLOT_LM, SLOT_SPAN, SPAN_ABSTAIN};

/// `qd-runtime/src/schema.rs`'s `RESERVED_NOUL_ROWS`: the abstain row a span head adds.
/// Pinned against that source by `tests/parity.rs`, as `test_heads.py` pins Python's copy.
pub const RESERVED_NOUL_ROWS: usize = 1;

/// A batch the supervision port refuses (`TrainerContractViolation` / the head's `ValueError`).
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{0}")]
pub struct SupervisionError(pub String);

/// The span rows of a batch and their gold positions (`trainer.SpanSupervision`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanSupervision {
    /// Which batch rows are `SLOT_SPAN`, ascending.
    pub rows: Vec<usize>,
    /// The query position each span row's pointer head reads from (`target_index`).
    pub query_index: Vec<i64>,
    /// Gold start token, or `SPAN_ABSTAIN`.
    pub start: Vec<i64>,
    /// Gold end token, or `SPAN_ABSTAIN`.
    pub end: Vec<i64>,
    /// `bool[K, width]`: each span row's line-start candidates, carried through verbatim.
    pub line_starts: Vec<bool>,
    /// The batch width `line_starts` is laid out at.
    pub width: usize,
    /// Rows whose gold is the abstain row.
    pub abstaining: Vec<bool>,
}

impl SpanSupervision {
    /// `SpanSupervision.__post_init__`.
    fn validate(&self) -> Result<(), SupervisionError> {
        let k = self.rows.len();
        if [
            self.query_index.len(),
            self.start.len(),
            self.end.len(),
            self.abstaining.len(),
        ]
        .iter()
        .any(|&n| n != k)
        {
            return Err(SupervisionError(
                "span arrays must share one shape".to_owned(),
            ));
        }
        if k == 0 {
            return Err(SupervisionError(
                "SpanSupervision with no rows; a batch with no span rows carries span=None"
                    .to_owned(),
            ));
        }
        if self.line_starts.len() != k * self.width {
            return Err(SupervisionError(format!("line_starts must be [K={k}, L]")));
        }
        for i in 0..k {
            let declared =
                self.start[i] == i64::from(SPAN_ABSTAIN) && self.end[i] == i64::from(SPAN_ABSTAIN);
            if declared != self.abstaining[i] {
                return Err(SupervisionError(
                    "`abstaining` disagrees with the SPAN_ABSTAIN sentinels in start/end"
                        .to_owned(),
                ));
            }
            if !self.abstaining[i] {
                if self.start[i] > self.end[i] {
                    return Err(SupervisionError(
                        "a span's start is after its end".to_owned(),
                    ));
                }
                if self.start[i] < 0 || self.end[i] < 0 {
                    return Err(SupervisionError(
                        "a non-abstaining span row carries a negative position".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Span rows.
    pub fn n_spans(&self) -> usize {
        self.rows.len()
    }

    /// Line-start candidates per span row (`candidate_counts`).
    pub fn candidate_counts(&self) -> Vec<i64> {
        self.line_starts
            .chunks(self.width.max(1))
            .map(|r| r.iter().filter(|&&m| m).count() as i64)
            .collect()
    }
}

/// The letter channel plus the span channel of one batch (`trainer.Supervision`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Supervision {
    /// `width - 1`: the letter channel's columns.
    pub columns: usize,
    /// `int32[B, width - 1]`: `tokens[:, 1:]`.
    pub targets: Vec<i32>,
    /// `bool[B, width - 1]`: supervised letter positions.
    pub mask: Vec<bool>,
    /// `mask.sum()`.
    pub n_supervised: usize,
    /// Present exactly when the batch has `SLOT_SPAN` rows.
    pub span: Option<SpanSupervision>,
}

impl Supervision {
    /// Supervised letter positions plus span rows.
    pub fn n_answers(&self) -> usize {
        self.n_supervised + self.span.as_ref().map_or(0, SpanSupervision::n_spans)
    }

    /// `(row, position)` of every supervised letter, row-major.
    pub fn letter_positions(&self) -> Vec<(usize, usize)> {
        self.mask
            .iter()
            .enumerate()
            .filter(|(_, m)| **m)
            .map(|(i, _)| (i / self.columns, i % self.columns))
            .collect()
    }
}

/// `trainer.ft_supervision`: supervise each row's declared answer, by the kind it is.
pub fn ft_supervision(batch: &Batch) -> Result<Supervision, SupervisionError> {
    let b = batch.rows();
    let width = batch.width;
    if width < 2 {
        return Err(SupervisionError(format!(
            "a batch padded to width {width} has no next-token pair"
        )));
    }
    if batch.slot_kind.len() != b || batch.target_index.len() != b {
        return Err(SupervisionError(
            "FT needs slot_kind and target_index for every row".to_owned(),
        ));
    }
    if let Some(r) = batch.slot_kind.iter().position(|&k| k == SLOT_LM) {
        return Err(SupervisionError(format!(
            "row {r} is SLOT_LM inside an FT batch; it carries no gold letter, so its target_index names nothing"
        )));
    }
    let columns = width - 1;
    let is_span: Vec<bool> = batch.slot_kind.iter().map(|&k| k == SLOT_SPAN).collect();
    let mut targets = Vec::with_capacity(b * columns);
    let mut mask = vec![false; b * columns];
    for r in 0..b {
        for p in 0..columns {
            targets.push(batch.token(r, p + 1));
        }
        let t = batch.target_index[r];
        if !is_span[r] && t >= 0 && (t as usize) < columns {
            mask[r * columns + t as usize] = true;
        }
    }
    // `_refuse_unsupervised_rows`: a letter row whose mask is empty contributes nothing.
    if let Some(r) =
        (0..b).find(|&r| !is_span[r] && !mask[r * columns..(r + 1) * columns].iter().any(|&m| m))
    {
        return Err(SupervisionError(format!(
            "FT (the letter channel supervises exactly Batch.target_index): row {r} has no supervised position, with length {}",
            batch.lengths[r]
        )));
    }
    let span = if is_span.iter().any(|&s| s) {
        let spans = batch.span_target.as_ref().ok_or_else(|| {
            SupervisionError("a SLOT_SPAN row without span_target reached the trainer".to_owned())
        })?;
        let lines = batch.line_starts.as_ref().ok_or_else(|| {
            SupervisionError("a SLOT_SPAN row without line_starts reached the trainer".to_owned())
        })?;
        let rows: Vec<usize> = (0..b).filter(|&r| is_span[r]).collect();
        let start: Vec<i64> = rows.iter().map(|&r| i64::from(spans[r][0])).collect();
        let end: Vec<i64> = rows.iter().map(|&r| i64::from(spans[r][1])).collect();
        let abstaining: Vec<bool> = start
            .iter()
            .zip(&end)
            .map(|(&s, &e)| s == i64::from(SPAN_ABSTAIN) && e == i64::from(SPAN_ABSTAIN))
            .collect();
        let span = SpanSupervision {
            query_index: rows
                .iter()
                .map(|&r| i64::from(batch.target_index[r]))
                .collect(),
            line_starts: rows
                .iter()
                .flat_map(|&r| lines[r * width..(r + 1) * width].iter().copied())
                .collect(),
            width,
            rows,
            start,
            end,
            abstaining,
        };
        span.validate()?;
        if span.start.contains(&i64::from(NO_SPAN)) {
            return Err(SupervisionError(
                "a SLOT_SPAN row carries NO_SPAN as its gold span".to_owned(),
            ));
        }
        Some(span)
    } else {
        None
    };
    let n_supervised = mask.iter().filter(|&&m| m).count();
    if n_supervised == 0 && span.is_none() {
        return Err(SupervisionError(
            "no position in this batch is supervised, by letter or by span".to_owned(),
        ));
    }
    Ok(Supervision {
        columns,
        targets,
        mask,
        n_supervised,
        span,
    })
}

/// `heads.span_head_rows`: one row per line start, plus the abstention.
pub fn span_head_rows(n_candidates: usize) -> Result<usize, SupervisionError> {
    if n_candidates < 1 {
        return Err(SupervisionError(format!(
            "n_candidates must be at least 1, got {n_candidates}: an empty candidate set is a mapping failure, not an empty answer"
        )));
    }
    Ok(n_candidates + RESERVED_NOUL_ROWS)
}

/// One batch's span rows in the head's row layout (`heads.SpanPlan`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanPlan {
    /// The widest row's candidate count; the padded column count of `candidate_pos`.
    pub max_candidates: usize,
    /// `int64[K, max_candidates]`: row `k`'s line-start token positions, ascending, zero-padded.
    pub candidate_pos: Vec<i64>,
    /// `bool[K, max_candidates]`: which columns are real.
    pub candidate_valid: Vec<bool>,
    /// Real candidates per row.
    pub n_candidates: Vec<i64>,
    /// Query position per row.
    pub query_index: Vec<i64>,
    /// Head row of the gold start: a candidate ordinal, or `n_candidates[k]` when abstaining.
    pub gold_start: Vec<i64>,
    /// Head row of the gold end, likewise.
    pub gold_end: Vec<i64>,
    /// Rows whose gold is the abstain row.
    pub abstaining: Vec<bool>,
}

impl SpanPlan {
    /// Span rows.
    pub fn n_spans(&self) -> usize {
        self.n_candidates.len()
    }

    /// Columns of the padded score matrix: the widest row plus the abstention.
    pub fn max_rows(&self) -> usize {
        self.max_candidates + RESERVED_NOUL_ROWS
    }

    /// Per row, what `qd-runtime` demands of a decode: `line_count + RESERVED_NOUL_ROWS`.
    pub fn runtime_rows(&self) -> Vec<i64> {
        self.n_candidates
            .iter()
            .map(|&n| n + RESERVED_NOUL_ROWS as i64)
            .collect()
    }
}

/// `heads.plan_span_batch`: the trainer's span channel in the head's row layout. The only
/// place the abstain row's position is decided: last, at index `n_candidates[k]`.
pub fn plan_span_batch(span: &SpanSupervision) -> Result<SpanPlan, SupervisionError> {
    let counts = span.candidate_counts();
    if counts.iter().any(|&c| c < 1) {
        return Err(SupervisionError(
            "a span row with no line-start candidate reached the head".to_owned(),
        ));
    }
    let k = counts.len();
    let max_candidates = counts.iter().copied().max().unwrap_or(0) as usize;
    let mut candidate_pos = vec![0i64; k * max_candidates];
    let mut candidate_valid = vec![false; k * max_candidates];
    let mut gold_start = vec![0i64; k];
    let mut gold_end = vec![0i64; k];
    for row in 0..k {
        let positions: Vec<i64> = span.line_starts[row * span.width..(row + 1) * span.width]
            .iter()
            .enumerate()
            .filter(|(_, m)| **m)
            .map(|(p, _)| p as i64)
            .collect();
        for (j, &p) in positions.iter().enumerate() {
            candidate_pos[row * max_candidates + j] = p;
            candidate_valid[row * max_candidates + j] = true;
        }
        if span.abstaining[row] {
            gold_start[row] = positions.len() as i64;
            gold_end[row] = positions.len() as i64;
            continue;
        }
        for (name, gold, out) in [
            ("start", span.start[row], &mut gold_start),
            ("end", span.end[row], &mut gold_end),
        ] {
            let hits: Vec<usize> = positions
                .iter()
                .enumerate()
                .filter(|(_, p)| **p == gold)
                .map(|(j, _)| j)
                .collect();
            if hits.len() != 1 {
                return Err(SupervisionError(format!(
                    "span row {row}: {name} gold at token {gold} is not one of this row's {} line-start candidates",
                    positions.len()
                )));
            }
            out[row] = hits[0] as i64;
        }
    }
    Ok(SpanPlan {
        max_candidates,
        candidate_pos,
        candidate_valid,
        n_candidates: counts,
        query_index: span.query_index.clone(),
        gold_start,
        gold_end,
        abstaining: span.abstaining.clone(),
    })
}
