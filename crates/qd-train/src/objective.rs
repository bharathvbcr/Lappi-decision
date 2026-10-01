//! Lappi's FT objective: the letter channel and the span channel of
//! `QwenDecisionStep.accumulate` / `accumulate_span` (`python/qd_train/backbone.py:1124-1185`).
//!
//! **The two channels and their batch means.**
//!
//! * Letter: a full-vocabulary cross-entropy at each non-span row's `target_index`, predicting
//!   the token after it, **mean over the `N` letter rows of the batch**
//!   (`fused_linear_cross_entropy(..., reduction="mean")`, `fused_ce.py:323`). Through the
//!   provider that is `RowTargets { positions: [target_index], targets: [gold], scale: 1/N }` in
//!   each row's sequence, and the reported letter loss is the sum of the provider's row sums
//!   over `N`.
//! * Span: the pointer head's start and end cross-entropies, summed over the `K` span rows and
//!   divided by `2K` (`heads.py:385-406`), weighted by `span_weight` in the total. The head
//!   reads the final hidden states at its positions; its gradient there, and its own
//!   parameters', carry `span_weight / (2K)`.
//! * Total: `letter + span_weight * span`, or `span_weight * span` when the batch has no letter
//!   row, or `letter` when it has no span row. Span rows are excluded from the letter channel
//!   (`ft_supervision`, `GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL`), and a row in neither channel
//!   is refused (`_refuse_unsupervised_rows`).
//!
//! The batch and the head are reached through two small traits of this crate's own,
//! [`FtBatch`] and [`SpanHead`], so this file does not depend on the shard reader's or the
//! span head's types (L-data's `batching.rs`/`supervision.rs` and L-head's `span_head.rs`
//! implement them when they merge).

use std::marker::PhantomData;

use crate::run_control::{ConsumedPrefix, RunControlError};
use crate::trainer::{BatchCounts, ConsumedBatch, HostParams, MicroBatchPlan, MicroLoss, Objective, SequenceWork, TrainError};

/// One row's letter answer: the hidden state at `position` predicts `target`
/// (`target_index` and the token after it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LetterTarget {
    pub position: u32,
    pub target: u32,
}

/// An FT batch as the objective reads it.
pub trait FtBatch: ConsumedBatch {
    /// A span row's supervision, opaque to this objective and handed to the head.
    type Span: Clone;
    fn n_rows(&self) -> usize;
    /// The padded width every row was bucketed to.
    fn padded_width(&self) -> usize;
    /// Row `row`'s real tokens (its first `lengths[row]` columns).
    fn row_tokens(&self, row: usize) -> &[u32];
    /// Row `row`'s letter answer, `None` for a span row.
    fn letter(&self, row: usize) -> Option<LetterTarget>;
    /// Row `row`'s span supervision, `None` for a letter row.
    fn span(&self, row: usize) -> Option<&Self::Span>;
}

/// The pointer head, as the objective drives it.
pub trait SpanHead: HostParams {
    type Span: Clone;

    /// The distinct positions whose final hidden state the head reads for this row. The
    /// provider requires them distinct; a head that reads a position twice lists it once here
    /// and pre-sums its gradient rows in [`SpanHead::loss_and_grad`].
    fn positions(&self, span: &Self::Span) -> Result<Vec<u32>, String>;

    /// One span row's forward and backward. `hidden` is `[positions.len(), width]` at the
    /// positions [`SpanHead::positions`] returned. Returns the row's UNSCALED loss (its start
    /// cross-entropy plus its end cross-entropy) and the gradient of `scale` times that loss at
    /// the same positions; adds `scale` times the loss's gradient to the head's own gradients.
    fn loss_and_grad(
        &mut self,
        span: &Self::Span,
        positions: &[u32],
        hidden: &[f32],
        width: usize,
        scale: f32,
    ) -> Result<(f64, Vec<f32>), String>;
}

/// What the last planned batch left for `hidden_grad` and `close`.
struct Planned<S> {
    /// Per planned sequence: its span supervision and the head positions, or `None`.
    spans: Vec<Option<(S, Vec<u32>)>>,
    n_letter: usize,
    n_span: usize,
    span_sum: f64,
    span_scale: f32,
}

/// The letter + span objective over an [`FtBatch`] and a [`SpanHead`].
pub struct LetterSpanObjective<B, H: SpanHead> {
    head: H,
    span_weight: f64,
    planned: Option<Planned<H::Span>>,
    _batch: PhantomData<fn(&B)>,
}

impl<B, H> LetterSpanObjective<B, H>
where
    B: FtBatch<Span = H::Span>,
    H: SpanHead,
{
    /// `span_weight` must be positive: zero would train the head on nothing while its loss
    /// still appeared in the log (`QwenDecisionStep.__init__`). The shuffled-label control's
    /// `span_channel_off` is not ported.
    pub fn new(head: H, span_weight: f64) -> Result<Self, TrainError> {
        if !(span_weight.is_finite() && span_weight > 0.0) {
            return Err(TrainError::Refused(format!(
                "span_weight must be positive, got {span_weight}; zero would train the span head on \
                 nothing while its loss still appeared in the log"
            )));
        }
        Ok(Self {
            head,
            span_weight,
            planned: None,
            _batch: PhantomData,
        })
    }

    pub fn head(&self) -> &H {
        &self.head
    }

    pub fn head_mut(&mut self) -> &mut H {
        &mut self.head
    }

    pub fn span_weight(&self) -> f64 {
        self.span_weight
    }
}

impl<B, H> Objective for LetterSpanObjective<B, H>
where
    B: FtBatch<Span = H::Span>,
    H: SpanHead,
{
    type Batch = B;

    fn run_kind(&self) -> &'static str {
        "ft"
    }

    fn plan(&mut self, batch: &B) -> Result<MicroBatchPlan, TrainError> {
        let index = batch.index();
        let rows = batch.n_rows();
        let width = batch.padded_width();
        if rows == 0 {
            return Err(TrainError::Refused(format!("batch {index} has no rows")));
        }
        let mut sequences = Vec::with_capacity(rows);
        let mut spans = Vec::with_capacity(rows);
        let (mut n_letter, mut n_span, mut real) = (0usize, 0usize, 0u64);
        for r in 0..rows {
            let tokens = batch.row_tokens(r);
            if tokens.len() > width {
                return Err(TrainError::Refused(format!(
                    "batch {index} row {r} has {} tokens in a batch {width} wide",
                    tokens.len()
                )));
            }
            real += tokens.len() as u64;
            match (batch.letter(r), batch.span(r)) {
                (Some(_), Some(_)) => {
                    return Err(TrainError::Refused(format!(
                        "batch {index} row {r} is in both channels; a span row's only letter is noul, \
                         and supervising it as a letter trains the head to abstain always"
                    )));
                }
                (None, None) => {
                    return Err(TrainError::Refused(format!(
                        "batch {index} row {r} has no supervised position, by letter or by span; a row \
                         that contributes nothing was still paid for"
                    )));
                }
                (Some(l), None) => {
                    let next = l.position as usize + 1;
                    if next >= tokens.len() {
                        return Err(TrainError::Refused(format!(
                            "batch {index} row {r}: target_index {} has no next real token in {} tokens",
                            l.position,
                            tokens.len()
                        )));
                    }
                    if tokens[next] != l.target {
                        return Err(TrainError::Refused(format!(
                            "batch {index} row {r}: the letter target {} is not the token after \
                             target_index ({}); position p predicts token p + 1",
                            l.target, tokens[next]
                        )));
                    }
                    n_letter += 1;
                    sequences.push(SequenceWork {
                        tokens: tokens.to_vec(),
                        positions: vec![l.position],
                        targets: vec![l.target],
                        hidden_at: Vec::new(),
                    });
                    spans.push(None);
                }
                (None, Some(s)) => {
                    let positions = self
                        .head
                        .positions(s)
                        .map_err(|e| TrainError::Refused(format!("batch {index} row {r}: span head: {e}")))?;
                    if positions.is_empty() {
                        return Err(TrainError::Refused(format!(
                            "batch {index} row {r}: the span head reads no position"
                        )));
                    }
                    n_span += 1;
                    sequences.push(SequenceWork {
                        tokens: tokens.to_vec(),
                        positions: Vec::new(),
                        targets: Vec::new(),
                        hidden_at: positions.clone(),
                    });
                    spans.push(Some((s.clone(), positions)));
                }
            }
        }
        let row_scale = if n_letter > 0 { (1.0 / n_letter as f64) as f32 } else { 0.0 };
        let span_scale = if n_span > 0 {
            (self.span_weight / (2.0 * n_span as f64)) as f32
        } else {
            0.0
        };
        self.planned = Some(Planned {
            spans,
            n_letter,
            n_span,
            span_sum: 0.0,
            span_scale,
        });
        let total = (rows * width) as u64;
        Ok(MicroBatchPlan {
            sequences,
            row_scale,
            counts: BatchCounts {
                supervised_tokens: n_letter as u64,
                span_rows: n_span as u64,
                padded_positions: total - real,
                total_positions: total,
            },
        })
    }

    fn hidden_grad(&mut self, seq: usize, positions: &[u32], hidden: &[f32], width: usize) -> Result<Vec<f32>, TrainError> {
        let planned = self
            .planned
            .as_mut()
            .ok_or_else(|| TrainError::Refused("hidden_grad before any batch was planned".into()))?;
        let Some(Some((span, want))) = planned.spans.get(seq) else {
            return Err(TrainError::Refused(format!("sequence {seq} has no span row to score")));
        };
        if positions != want.as_slice() {
            return Err(TrainError::Refused(format!(
                "sequence {seq}: the provider handed back positions {positions:?}, the head asked for {want:?}"
            )));
        }
        let (loss, dh) = self
            .head
            .loss_and_grad(span, positions, hidden, width, planned.span_scale)
            .map_err(|e| TrainError::Refused(format!("sequence {seq}: span head: {e}")))?;
        if dh.len() != hidden.len() {
            return Err(TrainError::Refused(format!(
                "sequence {seq}: the span head returned {} gradient values for {} hidden values",
                dh.len(),
                hidden.len()
            )));
        }
        planned.span_sum += loss;
        Ok(dh)
    }

    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError> {
        let p = self
            .planned
            .take()
            .ok_or_else(|| TrainError::Refused("close before any batch was planned".into()))?;
        if row_loss_sums.len() != p.spans.len() {
            return Err(TrainError::Refused(format!(
                "{} row-loss sums for {} planned sequences",
                row_loss_sums.len(),
                p.spans.len()
            )));
        }
        let letter = (p.n_letter > 0).then(|| row_loss_sums.iter().sum::<f64>() / p.n_letter as f64);
        let span = (p.n_span > 0).then(|| p.span_sum / (2.0 * p.n_span as f64));
        let total = match (letter, span) {
            (Some(l), Some(s)) => l + self.span_weight * s,
            (Some(l), None) => l,
            (None, Some(s)) => self.span_weight * s,
            (None, None) => return Err(TrainError::Refused("a batch with neither channel was closed".into())),
        };
        Ok(MicroLoss {
            total,
            channels: vec![
                ("letter".to_string(), letter.unwrap_or(0.0)),
                ("span".to_string(), span.unwrap_or(0.0)),
            ],
        })
    }

    fn host(&mut self) -> Option<&mut dyn HostParams> {
        Some(&mut self.head)
    }

    fn host_ref(&self) -> Option<&dyn HostParams> {
        Some(&self.head)
    }
}

/// Fold `trainer._fold`'s parts for a batch whose arrays the caller already holds as their
/// numpy buffers' bytes: a helper for [`ConsumedBatch::fold_into`] implementations. `optional`
/// is `[slot_kind, target_index, span_target, line_starts]`, `None` for an absent array.
pub fn fold_batch_parts(
    prefix: &mut ConsumedPrefix,
    index: u64,
    bucket: u64,
    tokens: &[u8],
    lengths: &[u8],
    optional: [Option<&[u8]>; 4],
) -> Result<(), RunControlError> {
    let idx = index.to_be_bytes();
    let bkt = bucket.to_be_bytes();
    let mut parts: Vec<&[u8]> = vec![&idx, &bkt, tokens, lengths];
    for a in optional {
        parts.push(a.unwrap_or(&[]));
    }
    prefix.fold(&parts)
}
