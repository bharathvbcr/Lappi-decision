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
//! **D17 as a second readout** ([`LetterSpanObjective::with_d17`]). With a [`D17`] the letter
//! channel is trained by the 17-way cross-entropy over `E[answer ids] + D17` at the letter
//! position instead of the full-vocabulary one: each letter row reads its final hidden state
//! through `hidden_at` (the provider hands it to [`Objective::hidden_grad`]), and the
//! full-vocabulary cross-entropy stays in the plan with `row_scale` 0, so the provider reports its
//! value and back-propagates nothing (`RowTargets.scale` 0). The total is
//! `d17 + span_weight * span`; the `letter` channel is the report only. Where the D17's position
//! and the span head's coincide in one sequence, [`merge_hidden_grads`] sums the two gradients
//! once. Today's batches put a row in exactly one channel, so that merge sees one part per
//! sequence; it is the path a row in both would take.
//!
//! The batch and the head are reached through two small traits of this crate's own,
//! [`FtBatch`] and [`SpanHead`], so this file does not depend on the shard reader's or the
//! span head's types (L-data's `batching.rs`/`supervision.rs` and L-head's `span_head.rs`
//! implement them when they merge).

use std::marker::PhantomData;

use crate::d17::{D17, D17Error, HiddenPart, merge_hidden_grads};
use crate::step::ParamSpec;
use crate::trainer::{
    BatchCounts, ConsumedBatch, HostParams, MicroBatchPlan, MicroLoss, Objective, SequenceWork,
    TrainError,
};

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
    /// Per planned sequence: its letter position and target when D17 trains it, else `None`.
    d17_rows: Vec<Option<(u32, u32)>>,
    n_letter: usize,
    n_span: usize,
    span_sum: f64,
    span_scale: f32,
    d17_sum: f64,
    d17_scale: f32,
    d17_seen: usize,
}

/// The host entries: the span head's, then D17's `d17.delta` when D17 trains the letter channel.
/// Without D17 it is the head's entries unchanged.
struct HostSet<H> {
    head: H,
    d17: Option<D17>,
}

impl<H: HostParams> HostParams for HostSet<H> {
    fn entries(&self) -> Vec<ParamSpec> {
        let mut e = self.head.entries();
        if let Some(d) = &self.d17 {
            e.extend(d.entries());
        }
        e
    }

    fn zero_grads(&mut self) {
        self.head.zero_grads();
        if let Some(d) = &mut self.d17 {
            d.zero_grads();
        }
    }

    fn grads(&self) -> Vec<Option<&[f32]>> {
        let mut g = self.head.grads();
        if let Some(d) = &self.d17 {
            g.extend(d.grads());
        }
        g
    }

    fn values(&self) -> Vec<&[f32]> {
        let mut v = self.head.values();
        if let Some(d) = &self.d17 {
            v.extend(d.values());
        }
        v
    }

    fn values_and_grads(&mut self) -> Vec<(&mut [f32], Option<&[f32]>)> {
        let mut v = self.head.values_and_grads();
        if let Some(d) = &mut self.d17 {
            v.extend(d.values_and_grads());
        }
        v
    }
}

fn d17_refusal(row: impl std::fmt::Display, e: &D17Error) -> TrainError {
    TrainError::Refused(format!("{row}: D17: {e}"))
}

/// The letter + span objective over an [`FtBatch`] and a [`SpanHead`].
pub struct LetterSpanObjective<B, H: SpanHead> {
    host: HostSet<H>,
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
            host: HostSet { head, d17: None },
            span_weight,
            planned: None,
            _batch: PhantomData,
        })
    }

    /// Train the letter channel through D17 (see the module docs). The `letter` channel of the
    /// loss becomes the report-only full-vocabulary value.
    pub fn with_d17(mut self, d17: D17) -> Self {
        self.host.d17 = Some(d17);
        self
    }

    pub fn d17(&self) -> Option<&D17> {
        self.host.d17.as_ref()
    }

    pub fn d17_mut(&mut self) -> Option<&mut D17> {
        self.host.d17.as_mut()
    }

    pub fn head(&self) -> &H {
        &self.host.head
    }

    pub fn head_mut(&mut self) -> &mut H {
        &mut self.host.head
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
        let mut d17_rows = Vec::with_capacity(rows);
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
                    if let Some(d) = &self.host.d17
                        && d.row_of(l.target).is_none()
                    {
                        return Err(d17_refusal(
                            format!("batch {index} row {r}"),
                            &D17Error::TargetNotAnswerRow {
                                target: l.target,
                                index: r,
                            },
                        ));
                    }
                    n_letter += 1;
                    // With D17 the full-vocabulary CE stays in the plan (report-only, scale 0)
                    // and D17 reads the same position's hidden state.
                    let hidden_at = if self.host.d17.is_some() {
                        vec![l.position]
                    } else {
                        Vec::new()
                    };
                    sequences.push(SequenceWork {
                        tokens: tokens.to_vec(),
                        positions: vec![l.position],
                        targets: vec![l.target],
                        hidden_at,
                    });
                    spans.push(None);
                    d17_rows.push(self.host.d17.as_ref().map(|_| (l.position, l.target)));
                }
                (None, Some(s)) => {
                    d17_rows.push(None);
                    let positions = self.host.head.positions(s).map_err(|e| {
                        TrainError::Refused(format!("batch {index} row {r}: span head: {e}"))
                    })?;
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
        let letter_mean = if n_letter > 0 {
            (1.0 / n_letter as f64) as f32
        } else {
            0.0
        };
        // D17 trains the letter channel; the provider's full-vocabulary CE is then value-only.
        let (row_scale, d17_scale) = if self.host.d17.is_some() {
            (0.0, letter_mean)
        } else {
            (letter_mean, 0.0)
        };
        let span_scale = if n_span > 0 {
            (self.span_weight / (2.0 * n_span as f64)) as f32
        } else {
            0.0
        };
        self.planned = Some(Planned {
            spans,
            d17_rows,
            n_letter,
            n_span,
            span_sum: 0.0,
            span_scale,
            d17_sum: 0.0,
            d17_scale,
            d17_seen: 0,
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

    fn hidden_grad(
        &mut self,
        seq: usize,
        positions: &[u32],
        hidden: &[f32],
        width: usize,
    ) -> Result<Vec<f32>, TrainError> {
        let planned = self.planned.as_mut().ok_or_else(|| {
            TrainError::Refused("hidden_grad before any batch was planned".into())
        })?;
        let span_row = planned.spans.get(seq).and_then(Option::as_ref);
        let d17_row = planned.d17_rows.get(seq).copied().flatten();
        if span_row.is_none() && d17_row.is_none() {
            return Err(TrainError::Refused(format!(
                "sequence {seq} has no span row to score"
            )));
        }
        // The positions the provider was asked for: the span head's, then D17's when it is not
        // already one of them.
        let mut union: Vec<u32> = span_row.map(|(_, w)| w.clone()).unwrap_or_default();
        if let Some((pos, _)) = d17_row
            && !union.contains(&pos)
        {
            union.push(pos);
        }
        if positions != union.as_slice() {
            return Err(TrainError::Refused(format!(
                "sequence {seq}: the provider handed back positions {positions:?}, the objective asked for {union:?}"
            )));
        }
        if hidden.len() != union.len() * width {
            return Err(TrainError::Refused(format!(
                "sequence {seq}: {} hidden values for {} positions of width {width}",
                hidden.len(),
                union.len()
            )));
        }

        let mut span_part: Option<(&[u32], Vec<f32>)> = None;
        if let Some((span, want)) = span_row {
            let (loss, dh) = self
                .host
                .head
                .loss_and_grad(
                    span,
                    want,
                    &hidden[..want.len() * width],
                    width,
                    planned.span_scale,
                )
                .map_err(|e| TrainError::Refused(format!("sequence {seq}: span head: {e}")))?;
            if dh.len() != want.len() * width {
                return Err(TrainError::Refused(format!(
                    "sequence {seq}: the span head returned {} gradient values for {} hidden values",
                    dh.len(),
                    want.len() * width
                )));
            }
            planned.span_sum += loss;
            span_part = Some((want.as_slice(), dh));
        }
        let mut d17_part: Option<([u32; 1], Vec<f32>)> = None;
        if let Some((pos, target)) = d17_row {
            let d17 = self
                .host
                .d17
                .as_mut()
                .ok_or_else(|| TrainError::Refused("a D17 row was planned without a D17".into()))?;
            if width != d17.hidden() {
                return Err(TrainError::Refused(format!(
                    "sequence {seq}: the provider's hidden width is {width}, D17 is {} wide",
                    d17.hidden()
                )));
            }
            let at = union.iter().position(|&u| u == pos).unwrap_or(0);
            let (loss, dh) = d17
                .loss_and_grad(&hidden[at * width..(at + 1) * width], &[target], planned.d17_scale)
                .map_err(|e| d17_refusal(format!("sequence {seq}"), &e))?;
            planned.d17_sum += loss;
            planned.d17_seen += 1;
            d17_part = Some(([pos], dh));
        }
        match (span_part, d17_part) {
            (Some((_, dh)), None) => Ok(dh),
            (None, Some((_, dh))) => Ok(dh),
            (Some((sp, sg)), Some((dp, dg))) => merge_hidden_grads(
                &union,
                &[
                    HiddenPart { positions: sp, grad: &sg },
                    HiddenPart { positions: &dp, grad: &dg },
                ],
                width,
            )
            .map_err(|e| d17_refusal(format!("sequence {seq}"), &e)),
            (None, None) => Err(TrainError::Refused(format!(
                "sequence {seq} has no span row to score"
            ))),
        }
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
        let trains_d17 = self.host.d17.is_some();
        if trains_d17 && p.d17_seen != p.n_letter {
            return Err(TrainError::Refused(format!(
                "D17 scored {} of {} letter rows; the batch mean would be over the wrong count",
                p.d17_seen, p.n_letter
            )));
        }
        // The letter channel's value is the provider's full-vocabulary CE; with D17 it is a report.
        let letter =
            (p.n_letter > 0).then(|| row_loss_sums.iter().sum::<f64>() / p.n_letter as f64);
        let d17 = (trains_d17 && p.n_letter > 0).then(|| p.d17_sum / p.n_letter as f64);
        let trained_letter = if trains_d17 { d17 } else { letter };
        let span = (p.n_span > 0).then(|| p.span_sum / (2.0 * p.n_span as f64));
        let total = match (trained_letter, span) {
            (Some(l), Some(s)) => l + self.span_weight * s,
            (Some(l), None) => l,
            (None, Some(s)) => self.span_weight * s,
            (None, None) => {
                return Err(TrainError::Refused(
                    "a batch with neither channel was closed".into(),
                ));
            }
        };
        let mut channels = vec![("letter".to_string(), letter.unwrap_or(0.0))];
        if trains_d17 {
            channels.push(("d17".to_string(), d17.unwrap_or(0.0)));
        }
        channels.push(("span".to_string(), span.unwrap_or(0.0)));
        Ok(MicroLoss { total, channels })
    }

    fn host(&mut self) -> Option<&mut dyn HostParams> {
        Some(&mut self.host)
    }

    fn host_ref(&self) -> Option<&dyn HostParams> {
        Some(&self.host)
    }
}
