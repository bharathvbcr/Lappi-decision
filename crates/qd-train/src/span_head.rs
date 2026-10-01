//! The span pointer head, forward and backward, in f32 on the host.
//!
//! The Rust twin of `python/qd_train/heads.py`'s `SpanPointerHead` (Fable's ojas advice, Q1
//! row 8): two bilinear pointers -- start and end -- over a context's line-start tokens plus
//! one abstain row, each scored against the query position's projected hidden state, trained
//! by cross-entropy against the gold rows. It exists so the span loss can sit outside tessl:
//! tessl hands over the final hidden state at chosen positions (`PendingStep::hidden`), this
//! module computes the loss and its gradient, and the gradient goes back through
//! `Qwen35Model::train_backward_into(dh = (positions, [n, H]))`
//! (`tessl/src/qwen35_train.rs:306-330, 697-716`).
//!
//! # Row layout (pinned in `tests/span_head.rs`)
//!
//! A row's scores are `n_candidates + RESERVED_NOUL_ROWS` long: row `j < n` is the `j`-th line
//! start **in ascending token order**, which `qd-runtime/src/answer.rs` serves as line `j + 1`,
//! and row `n` is the abstention, which it reads as `noul_row = plan.rows -
//! RESERVED_NOUL_ROWS`. That is `heads.py`'s layout with the training padding stripped --
//! exactly `serving_scores` -- so there is no `-inf` column to leak and no padded candidate to
//! select. [`SpanRowPlan::new`] is the one place that decides where the abstention goes.
//!
//! # Parameters
//!
//! [`SpanHead`] holds torch's four tensors under torch's names and layouts, so a state dict
//! (or the export lane's `span_head.*` safetensors) maps onto it one to one:
//! `start_proj.weight` and `end_proj.weight` are `nn.Linear` weights, `[out, in]` row-major,
//! applied as `q = W h`; `abstain_start` and `abstain_end` are `[H]`. A gradient bank is a
//! [`SpanHead`] of the same width, as torch's `.grad` has its parameter's shape.
//!
//! # Loss and gradient
//!
//! [`SpanHead::loss_and_backward`] returns the loss `heads.py`'s `loss(reduction="mean")`
//! returns -- the start and end cross-entropies summed over the batch's `K` rows and divided
//! by the `2K` pointer decisions, which is what `QwenDecisionStep.span_log` records -- and the
//! gradient of `span_weight * loss`, which is what `accumulate_span` backpropagates. Weight
//! gradients are **added** into the bank, as torch accumulates `.grad` across micro-batches.
//!
//! # The hidden-state gradient, and why it is pre-summed here
//!
//! A row reads its query position and its candidates' positions, and the query can itself be
//! a line start. tessl's `train_backward_into` takes each position **once**
//! (`qwen35_bwd.rs::check_scatter_rows` refuses a repeat), so the head reads and writes the
//! **distinct** positions, ascending ([`SpanRowPlan::positions`]): a position that is both the
//! query and a candidate is one row of the input and one row of the gradient, and that row's
//! gradient is the **sum** of the query path and the candidate path. torch gets the same sum
//! from `gather`'s backward, a scatter-add.
//!
//! # Numerics
//!
//! Storage is f32 everywhere torch's head materialises an f32 tensor: the projected query, the
//! scores, and every returned gradient. Every reduction -- a dot product over `H`, a
//! log-sum-exp over the candidates, a sum over rows -- accumulates in f64 and is rounded once.
//! The head's compute is a few `H x H` matrix-vector products per row, so this costs nothing
//! that matters, and it makes the result independent of summation order: at `H = 2048` the
//! f32 oracle's own rounding is of the order of the parity tolerance, and this head's is not.

use thiserror::Error;

/// The rows a slot reserves for abstention. `qd-runtime/src/schema.rs` owns the value;
/// `python/qd_train/schema_mirror.py` and this constant restate it, and
/// `tests/span_head.rs` fails if this one moves away from the runtime's.
pub const RESERVED_NOUL_ROWS: usize = 1;

// The scoring below appends exactly one abstention score per pointer; a runtime that reserved
// more rows would need a head that scores more, not a constant that says so.
const _: () = assert!(RESERVED_NOUL_ROWS == 1);

/// torch's names for the head's parameters, in its registration order (`state_dict` keys of
/// `SpanPointerHead`).
pub const PARAMETER_NAMES: [&str; 4] = [
    "start_proj.weight",
    "end_proj.weight",
    "abstain_start",
    "abstain_end",
];

/// Why the head refused its input. Every check runs before any gradient is written.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SpanHeadError {
    #[error(
        "a span row needs at least one line-start candidate: a sequence has at least one line, \
         so an empty candidate set is a mapping failure, not an empty answer"
    )]
    NoCandidates,
    #[error(
        "candidates must be strictly ascending token positions (row j is served as line j + 1 by \
         qd-runtime's answer.rs): position {next} follows {prev} at index {index}"
    )]
    CandidatesNotAscending { index: usize, prev: u32, next: u32 },
    #[error(
        "gold {which} row {row} is not one of this row's {n_candidates} line starts; an \
         abstaining gold is SpanGold::Abstain, never an ordinal"
    )]
    GoldOutOfRange {
        which: &'static str,
        row: usize,
        n_candidates: usize,
    },
    #[error("gold start row {start} is after gold end row {end}")]
    GoldReversed { start: usize, end: usize },
    #[error("hidden_size must be positive")]
    ZeroWidth,
    #[error("{name} has {actual} values; a head of width {hidden_size} needs {expected}")]
    ParameterShape {
        name: &'static str,
        hidden_size: usize,
        expected: usize,
        actual: usize,
    },
    #[error("the gradient bank is {actual} wide and this head is {expected}")]
    BankWidth { expected: usize, actual: usize },
    #[error(
        "row {row}: hidden has {actual} values, but its plan reads {positions} positions of \
         width {hidden_size} ({expected} values)"
    )]
    HiddenShape {
        row: usize,
        positions: usize,
        hidden_size: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "row {row}: {pointer} score {column} is {value}; qd-runtime's validate_logits refuses \
         that as non_finite_logit, and a softmax over it trains on nothing"
    )]
    NonFiniteScore {
        row: usize,
        pointer: &'static str,
        column: usize,
        value: f32,
    },
    #[error("a span batch with no rows: the mean over its 2K pointer decisions is 0/0")]
    EmptyBatch,
    #[error("span_weight must be finite and non-negative, got {0}")]
    SpanWeight(f32),
}

/// How many rows a span slot's head ranges over: one per line start, plus the abstention.
/// The Rust counterpart of `qd_runtime::answer::span_rows` and `heads.py`'s `span_head_rows`.
pub fn span_head_rows(n_candidates: usize) -> Result<usize, SpanHeadError> {
    if n_candidates == 0 {
        return Err(SpanHeadError::NoCandidates);
    }
    Ok(n_candidates + RESERVED_NOUL_ROWS)
}

/// A row's gold, in the head's row space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanGold {
    /// No evidence: both pointers' gold is the abstention row.
    Abstain,
    /// Both pointers name a line: ordinals into the row's ascending candidates, `start <= end`
    /// (the trainer's `SpanSupervision` refuses a span whose start follows its end).
    Lines { start: usize, end: usize },
}

/// One span row, laid out in the head's row order. The validated, Rust form of one row of
/// `heads.py`'s `SpanPlan`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanRowPlan {
    query_index: u32,
    candidates: Vec<u32>,
    gold_start: usize,
    gold_end: usize,
    /// The distinct positions this row reads, ascending: the candidates and the query.
    positions: Vec<u32>,
    /// Where the query's row is in `positions`.
    query_slot: usize,
    /// Where candidate `j`'s row is in `positions`.
    candidate_slots: Vec<usize>,
}

impl SpanRowPlan {
    /// A row whose query reads from token `query_index`, whose candidates are its line-start
    /// tokens (strictly ascending), and whose gold is `gold`.
    pub fn new(
        query_index: u32,
        candidates: Vec<u32>,
        gold: SpanGold,
    ) -> Result<Self, SpanHeadError> {
        let n = candidates.len();
        if n == 0 {
            return Err(SpanHeadError::NoCandidates);
        }
        if let Some(i) = candidates.windows(2).position(|w| w[0] >= w[1]) {
            return Err(SpanHeadError::CandidatesNotAscending {
                index: i + 1,
                prev: candidates[i],
                next: candidates[i + 1],
            });
        }
        let (gold_start, gold_end) = match gold {
            // The abstention is the last row -- `noul_row = plan.rows - RESERVED_NOUL_ROWS` in
            // qd-runtime/src/answer.rs, and column `n_candidates` in heads.py.
            SpanGold::Abstain => (n, n),
            SpanGold::Lines { start, end } => {
                for (which, row) in [("start", start), ("end", end)] {
                    if row >= n {
                        return Err(SpanHeadError::GoldOutOfRange {
                            which,
                            row,
                            n_candidates: n,
                        });
                    }
                }
                if start > end {
                    return Err(SpanHeadError::GoldReversed { start, end });
                }
                (start, end)
            }
        };
        let (positions, query_slot, candidate_slots) = match candidates.binary_search(&query_index)
        {
            // The query is itself a line start: one row serves both, and its gradient is the
            // sum of both paths (the module docs' pre-sum).
            Ok(slot) => (candidates.clone(), slot, (0..n).collect()),
            Err(slot) => {
                let mut positions = candidates.clone();
                positions.insert(slot, query_index);
                let slots = (0..n).map(|j| if j < slot { j } else { j + 1 }).collect();
                (positions, slot, slots)
            }
        };
        Ok(Self {
            query_index,
            candidates,
            gold_start,
            gold_end,
            positions,
            query_slot,
            candidate_slots,
        })
    }

    /// The token the pointers read from (`Batch.target_index` of a span row).
    pub fn query_index(&self) -> u32 {
        self.query_index
    }

    /// The line-start tokens, ascending: candidate `j` is row `j`, line `j + 1`.
    pub fn candidates(&self) -> &[u32] {
        &self.candidates
    }

    pub fn n_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// The rows `qd-runtime` demands of this row's decode: `line_count + RESERVED_NOUL_ROWS`.
    pub fn runtime_rows(&self) -> usize {
        self.candidates.len() + RESERVED_NOUL_ROWS
    }

    /// The abstention's row, as `answer.rs` computes it.
    pub fn noul_row(&self) -> usize {
        self.runtime_rows() - RESERVED_NOUL_ROWS
    }

    pub fn gold_start(&self) -> usize {
        self.gold_start
    }

    pub fn gold_end(&self) -> usize {
        self.gold_end
    }

    pub fn is_abstaining(&self) -> bool {
        self.gold_start == self.noul_row()
    }

    /// The distinct token positions this row reads and returns a gradient for, ascending.
    /// Pass them to `PendingStep::hidden` for the input rows and to `train_backward_into`
    /// with [`HiddenGrad::rows`]; tessl refuses a repeated position, and this has none.
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }
}

/// One row of a batch: its plan, and the hidden states at `plan.positions()`, dense f32
/// `[plan.positions().len(), H]` -- what `PendingStep::hidden(plan.positions())` writes.
#[derive(Debug, Clone, Copy)]
pub struct SpanRowInput<'a> {
    pub plan: &'a SpanRowPlan,
    pub hidden: &'a [f32],
}

/// One row's scores, `runtime_rows()` long each, the abstention last.
#[derive(Debug, Clone, PartialEq)]
pub struct PointerScores {
    pub start: Vec<f32>,
    pub end: Vec<f32>,
}

/// A row's gradient with respect to the hidden states it read: dense f32
/// `[positions.len(), H]`, positions distinct and ascending -- `dh` for tessl's
/// `train_backward_into` as is.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenGrad {
    pub positions: Vec<u32>,
    pub rows: Vec<f32>,
}

/// What [`SpanHead::loss_and_backward`] returns besides the weight gradients it adds.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanStep {
    /// The unweighted mean over the batch's `2K` pointer decisions (`span_log`'s number).
    pub loss: f32,
    /// Per row, in batch order.
    pub scores: Vec<PointerScores>,
    /// Per row, in batch order: the gradient of `span_weight * loss`.
    pub d_hidden: Vec<HiddenGrad>,
}

/// The span pointer head's parameters (or, as a gradient bank, their gradients).
#[derive(Debug, Clone, PartialEq)]
pub struct SpanHead {
    hidden_size: usize,
    start_proj_weight: Vec<f32>,
    end_proj_weight: Vec<f32>,
    abstain_start: Vec<f32>,
    abstain_end: Vec<f32>,
}

/// One pointer's half of a row's forward.
struct PointerForward {
    /// The projected query, `W h_query`, rounded to f32 as torch's `proj(h_query)` is.
    query: Vec<f32>,
    scores: Vec<f32>,
}

struct RowForward {
    start: PointerForward,
    end: PointerForward,
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// `W x` for an `[out, in]` row-major `W`: `nn.Linear(x)` without a bias.
fn project(weight: &[f32], x: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(x.len())
        .map(|row| dot(row, x) as f32)
        .collect()
}

/// `acc += W^T g` for an `[out, in]` row-major `W`: the input gradient of `nn.Linear`.
fn add_project_transposed(weight: &[f32], g: &[f64], acc: &mut [f64]) {
    for (row, &gi) in weight.chunks_exact(acc.len()).zip(g) {
        for (a, &w) in acc.iter_mut().zip(row) {
            *a += f64::from(w) * gi;
        }
    }
}

/// `log sum_j exp(s_j)` and the softmax, in f64.
fn log_softmax_parts(scores: &[f32]) -> (f64, Vec<f64>) {
    let max = scores
        .iter()
        .map(|&s| f64::from(s))
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = scores.iter().map(|&s| (f64::from(s) - max).exp()).sum();
    let lse = max + sum.ln();
    let probs = scores.iter().map(|&s| (f64::from(s) - lse).exp()).collect();
    (lse, probs)
}

impl SpanHead {
    /// A head from torch's four tensors (see the module docs for their layout).
    pub fn new(
        hidden_size: usize,
        start_proj_weight: Vec<f32>,
        end_proj_weight: Vec<f32>,
        abstain_start: Vec<f32>,
        abstain_end: Vec<f32>,
    ) -> Result<Self, SpanHeadError> {
        if hidden_size == 0 {
            return Err(SpanHeadError::ZeroWidth);
        }
        let square = hidden_size * hidden_size;
        for (name, expected, actual) in [
            (PARAMETER_NAMES[0], square, start_proj_weight.len()),
            (PARAMETER_NAMES[1], square, end_proj_weight.len()),
            (PARAMETER_NAMES[2], hidden_size, abstain_start.len()),
            (PARAMETER_NAMES[3], hidden_size, abstain_end.len()),
        ] {
            if actual != expected {
                return Err(SpanHeadError::ParameterShape {
                    name,
                    hidden_size,
                    expected,
                    actual,
                });
            }
        }
        Ok(Self {
            hidden_size,
            start_proj_weight,
            end_proj_weight,
            abstain_start,
            abstain_end,
        })
    }

    /// All zeros: an empty gradient bank (torch's head initialises its projections randomly,
    /// so a zero head is a bank, not an untrained head).
    pub fn zeros(hidden_size: usize) -> Result<Self, SpanHeadError> {
        let square = hidden_size * hidden_size;
        Self::new(
            hidden_size,
            vec![0.0; square],
            vec![0.0; square],
            vec![0.0; hidden_size],
            vec![0.0; hidden_size],
        )
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// The four tensors under torch's names, in [`PARAMETER_NAMES`] order.
    pub fn named(&self) -> [(&'static str, &[f32]); 4] {
        [
            (PARAMETER_NAMES[0], &self.start_proj_weight),
            (PARAMETER_NAMES[1], &self.end_proj_weight),
            (PARAMETER_NAMES[2], &self.abstain_start),
            (PARAMETER_NAMES[3], &self.abstain_end),
        ]
    }

    /// The four tensors, writable in place (an optimizer step, zeroing a bank). Slices, so a
    /// tensor's length -- the head's shape -- cannot change.
    pub fn named_mut(&mut self) -> [(&'static str, &mut [f32]); 4] {
        [
            (PARAMETER_NAMES[0], &mut self.start_proj_weight),
            (PARAMETER_NAMES[1], &mut self.end_proj_weight),
            (PARAMETER_NAMES[2], &mut self.abstain_start),
            (PARAMETER_NAMES[3], &mut self.abstain_end),
        ]
    }

    /// One row's start and end scores, in the serving layout (the abstention last).
    pub fn scores(
        &self,
        plan: &SpanRowPlan,
        hidden: &[f32],
    ) -> Result<PointerScores, SpanHeadError> {
        let forward = self.forward_row(0, plan, hidden)?;
        Ok(PointerScores {
            start: forward.start.scores,
            end: forward.end.scores,
        })
    }

    fn forward_row(
        &self,
        row: usize,
        plan: &SpanRowPlan,
        hidden: &[f32],
    ) -> Result<RowForward, SpanHeadError> {
        let h = self.hidden_size;
        let expected = plan.positions.len() * h;
        if hidden.len() != expected {
            return Err(SpanHeadError::HiddenShape {
                row,
                positions: plan.positions.len(),
                hidden_size: h,
                expected,
                actual: hidden.len(),
            });
        }
        let at = |slot: usize| &hidden[slot * h..(slot + 1) * h];
        let pointer = |name: &'static str, weight: &[f32], abstain: &[f32]| {
            let query = project(weight, at(plan.query_slot));
            let mut scores: Vec<f32> = plan
                .candidate_slots
                .iter()
                .map(|&slot| dot(&query, at(slot)) as f32)
                .collect();
            // The abstention, last: row `n_candidates`, `noul_row` in answer.rs.
            scores.push(dot(&query, abstain) as f32);
            if let Some((column, &value)) = scores.iter().enumerate().find(|(_, s)| !s.is_finite())
            {
                return Err(SpanHeadError::NonFiniteScore {
                    row,
                    pointer: name,
                    column,
                    value,
                });
            }
            Ok(PointerForward { query, scores })
        };
        Ok(RowForward {
            start: pointer("start", &self.start_proj_weight, &self.abstain_start)?,
            end: pointer("end", &self.end_proj_weight, &self.abstain_end)?,
        })
    }

    /// The batch's mean span loss, and the gradient of `span_weight * loss`: returned for the
    /// hidden states (per row, at its distinct positions), **added** into `grads` for the
    /// parameters. Nothing is written to `grads` unless every row is valid.
    pub fn loss_and_backward(
        &self,
        batch: &[SpanRowInput<'_>],
        span_weight: f32,
        grads: &mut SpanHead,
    ) -> Result<SpanStep, SpanHeadError> {
        if batch.is_empty() {
            return Err(SpanHeadError::EmptyBatch);
        }
        if !(span_weight.is_finite() && span_weight >= 0.0) {
            return Err(SpanHeadError::SpanWeight(span_weight));
        }
        if grads.hidden_size != self.hidden_size {
            return Err(SpanHeadError::BankWidth {
                expected: self.hidden_size,
                actual: grads.hidden_size,
            });
        }
        let forwards = batch
            .iter()
            .enumerate()
            .map(|(row, input)| self.forward_row(row, input.plan, input.hidden))
            .collect::<Result<Vec<_>, _>>()?;

        let h = self.hidden_size;
        // `heads.py`: `total / (2 * plan.n_spans)` -- the mean over the 2K pointer decisions.
        let decisions = 2 * batch.len();
        let scale = f64::from(span_weight) / decisions as f64;

        let mut total = 0f64;
        let mut d_hidden = Vec::with_capacity(batch.len());
        // Per row and pointer: d loss / d query (f64), for the projection's weight gradient.
        let mut d_query: Vec<[Vec<f64>; 2]> = Vec::with_capacity(batch.len());
        let mut d_abstain = [vec![0f64; h], vec![0f64; h]];

        for (input, forward) in batch.iter().zip(&forwards) {
            let plan = input.plan;
            let at = |slot: usize| &input.hidden[slot * h..(slot + 1) * h];
            let mut slots = vec![0f64; plan.positions.len() * h];
            let mut d_query_input = vec![0f64; h];
            let mut row_d_query: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            for (p, (pointer, gold, weight, abstain)) in [
                (
                    &forward.start,
                    plan.gold_start,
                    &self.start_proj_weight,
                    &self.abstain_start,
                ),
                (
                    &forward.end,
                    plan.gold_end,
                    &self.end_proj_weight,
                    &self.abstain_end,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                let (lse, probs) = log_softmax_parts(&pointer.scores);
                total += lse - f64::from(pointer.scores[gold]);
                // d (scale * CE) / d score_j = scale * (softmax_j - [j == gold]).
                let d_scores: Vec<f64> = probs
                    .iter()
                    .enumerate()
                    .map(|(j, &pj)| scale * (pj - if j == gold { 1.0 } else { 0.0 }))
                    .collect();
                let n = plan.n_candidates();
                let mut dq = vec![0f64; h];
                for (j, &slot) in plan.candidate_slots.iter().enumerate() {
                    let ds = d_scores[j];
                    for ((acc, &c), (dst, &q)) in dq.iter_mut().zip(at(slot)).zip(
                        slots[slot * h..(slot + 1) * h]
                            .iter_mut()
                            .zip(&pointer.query),
                    ) {
                        *acc += ds * f64::from(c);
                        *dst += ds * f64::from(q);
                    }
                }
                let ds_abstain = d_scores[n];
                for ((acc, &a), (dv, &q)) in dq
                    .iter_mut()
                    .zip(abstain.iter())
                    .zip(d_abstain[p].iter_mut().zip(&pointer.query))
                {
                    *acc += ds_abstain * f64::from(a);
                    *dv += ds_abstain * f64::from(q);
                }
                add_project_transposed(weight, &dq, &mut d_query_input);
                row_d_query[p] = dq;
            }
            // The query path joins its row: added, because that row is also a candidate's
            // when the query is a line start, and tessl takes the position once.
            for (dst, &g) in slots[plan.query_slot * h..(plan.query_slot + 1) * h]
                .iter_mut()
                .zip(&d_query_input)
            {
                *dst += g;
            }
            d_hidden.push(HiddenGrad {
                positions: plan.positions.clone(),
                rows: slots.iter().map(|&g| g as f32).collect(),
            });
            d_query.push(row_d_query);
        }

        // dW = sum over rows of dq (outer) h_query, per pointer; one rounding per element.
        let banks = [&mut grads.start_proj_weight, &mut grads.end_proj_weight];
        for (p, bank) in banks.into_iter().enumerate() {
            for (i, bank_row) in bank.chunks_exact_mut(h).enumerate() {
                for (j, dst) in bank_row.iter_mut().enumerate() {
                    let sum: f64 = batch
                        .iter()
                        .zip(&d_query)
                        .map(|(input, dq)| {
                            let query = input.plan.query_slot * h + j;
                            dq[p][i] * f64::from(input.hidden[query])
                        })
                        .sum();
                    *dst += sum as f32;
                }
            }
        }
        for (bank, d) in [&mut grads.abstain_start, &mut grads.abstain_end]
            .into_iter()
            .zip(&d_abstain)
        {
            for (dst, &g) in bank.iter_mut().zip(d) {
                *dst += g as f32;
            }
        }

        Ok(SpanStep {
            loss: (total / decisions as f64) as f32,
            scores: forwards
                .into_iter()
                .map(|f| PointerScores {
                    start: f.start.scores,
                    end: f.end.scores,
                })
                .collect(),
            d_hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(query: u32, candidates: &[u32], gold: SpanGold) -> SpanRowPlan {
        SpanRowPlan::new(query, candidates.to_vec(), gold).expect("a valid plan")
    }

    /// A small head with distinct, non-symmetric values everywhere.
    fn head(h: usize) -> SpanHead {
        let w = |seed: f32| -> Vec<f32> {
            (0..h * h)
                .map(|i| ((i as f32 + seed) * 0.37).sin() * 0.5)
                .collect()
        };
        let v =
            |seed: f32| -> Vec<f32> { (0..h).map(|i| ((i as f32 + seed) * 0.71).cos()).collect() };
        SpanHead::new(h, w(1.0), w(2.0), v(3.0), v(4.0)).expect("a valid head")
    }

    fn hidden_for(plan: &SpanRowPlan, h: usize, seed: f32) -> Vec<f32> {
        (0..plan.positions().len() * h)
            .map(|i| ((i as f32 + seed) * 0.53).sin())
            .collect()
    }

    #[test]
    fn a_plan_refuses_what_plan_span_batch_never_emits() {
        assert_eq!(
            SpanRowPlan::new(0, vec![], SpanGold::Abstain),
            Err(SpanHeadError::NoCandidates)
        );
        assert_eq!(
            SpanRowPlan::new(9, vec![1, 4, 4], SpanGold::Abstain),
            Err(SpanHeadError::CandidatesNotAscending {
                index: 2,
                prev: 4,
                next: 4
            })
        );
        assert_eq!(
            SpanRowPlan::new(9, vec![5, 3], SpanGold::Abstain),
            Err(SpanHeadError::CandidatesNotAscending {
                index: 1,
                prev: 5,
                next: 3
            })
        );
        // The abstention is not an ordinal: row n is reachable only as SpanGold::Abstain.
        assert_eq!(
            SpanRowPlan::new(9, vec![1, 4], SpanGold::Lines { start: 0, end: 2 }),
            Err(SpanHeadError::GoldOutOfRange {
                which: "end",
                row: 2,
                n_candidates: 2
            })
        );
        assert_eq!(
            SpanRowPlan::new(9, vec![1, 4], SpanGold::Lines { start: 2, end: 2 }),
            Err(SpanHeadError::GoldOutOfRange {
                which: "start",
                row: 2,
                n_candidates: 2
            })
        );
        assert_eq!(
            SpanRowPlan::new(9, vec![1, 4], SpanGold::Lines { start: 1, end: 0 }),
            Err(SpanHeadError::GoldReversed { start: 1, end: 0 })
        );
    }

    #[test]
    fn positions_are_distinct_ascending_and_name_every_row_the_head_reads() {
        let p = plan(5, &[1, 4, 9], SpanGold::Lines { start: 0, end: 2 });
        assert_eq!(p.positions(), [1, 4, 5, 9]);
        assert_eq!(
            (p.query_slot, p.candidate_slots.as_slice()),
            (2, &[0, 1, 3][..])
        );
        let first = plan(0, &[1, 4], SpanGold::Abstain);
        assert_eq!(first.positions(), [0, 1, 4]);
        assert_eq!(
            (first.query_slot, first.candidate_slots.as_slice()),
            (0, &[1, 2][..])
        );
        // The query is itself a line start: one position, shared.
        let shared = plan(4, &[1, 4, 9], SpanGold::Abstain);
        assert_eq!(shared.positions(), [1, 4, 9]);
        assert_eq!(
            (shared.query_slot, shared.candidate_slots.as_slice()),
            (1, &[0, 1, 2][..])
        );
        assert!(shared.is_abstaining());
        assert_eq!((shared.gold_start(), shared.gold_end()), (3, 3));
    }

    #[test]
    fn the_head_refuses_shapes_it_was_not_built_for() {
        assert_eq!(SpanHead::zeros(0), Err(SpanHeadError::ZeroWidth));
        assert!(matches!(
            SpanHead::new(2, vec![0.0; 4], vec![0.0; 3], vec![0.0; 2], vec![0.0; 2]),
            Err(SpanHeadError::ParameterShape {
                name: "end_proj.weight",
                ..
            })
        ));
        let head = head(3);
        let p = plan(7, &[1, 4], SpanGold::Abstain);
        assert!(matches!(
            head.scores(&p, &[0.0; 6]),
            Err(SpanHeadError::HiddenShape {
                expected: 9,
                actual: 6,
                ..
            })
        ));
        let hidden = hidden_for(&p, 3, 0.0);
        let input = [SpanRowInput {
            plan: &p,
            hidden: &hidden,
        }];
        let mut narrow = SpanHead::zeros(2).expect("bank");
        assert_eq!(
            head.loss_and_backward(&input, 1.0, &mut narrow),
            Err(SpanHeadError::BankWidth {
                expected: 3,
                actual: 2
            })
        );
        let mut bank = SpanHead::zeros(3).expect("bank");
        assert_eq!(
            head.loss_and_backward(&[], 1.0, &mut bank),
            Err(SpanHeadError::EmptyBatch)
        );
        for bad in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                head.loss_and_backward(&input, bad, &mut bank),
                Err(SpanHeadError::SpanWeight(_))
            ));
        }
        assert_eq!(
            bank,
            SpanHead::zeros(3).expect("bank"),
            "a refusal wrote a gradient"
        );
    }

    #[test]
    fn a_non_finite_score_is_refused_before_any_gradient_is_written() {
        let head = head(3);
        let good = plan(7, &[1, 4], SpanGold::Abstain);
        let bad = plan(2, &[0, 3], SpanGold::Lines { start: 0, end: 1 });
        let good_hidden = hidden_for(&good, 3, 0.0);
        let mut bad_hidden = hidden_for(&bad, 3, 1.0);
        bad_hidden[0] = f32::NAN;
        let mut bank = SpanHead::zeros(3).expect("bank");
        let err = head
            .loss_and_backward(
                &[
                    SpanRowInput {
                        plan: &good,
                        hidden: &good_hidden,
                    },
                    SpanRowInput {
                        plan: &bad,
                        hidden: &bad_hidden,
                    },
                ],
                1.0,
                &mut bank,
            )
            .expect_err("a NaN hidden state must not train");
        assert!(
            matches!(err, SpanHeadError::NonFiniteScore { row: 1, .. }),
            "{err:?}"
        );
        assert_eq!(bank, SpanHead::zeros(3).expect("bank"));
    }

    #[test]
    fn weight_gradients_accumulate_and_a_zero_weight_trains_nothing() {
        let head = head(4);
        let p = plan(6, &[0, 2, 5], SpanGold::Lines { start: 1, end: 2 });
        let hidden = hidden_for(&p, 4, 0.5);
        let input = [SpanRowInput {
            plan: &p,
            hidden: &hidden,
        }];

        let mut once = SpanHead::zeros(4).expect("bank");
        let first = head
            .loss_and_backward(&input, 1.0, &mut once)
            .expect("step");
        let mut twice = SpanHead::zeros(4).expect("bank");
        head.loss_and_backward(&input, 1.0, &mut twice)
            .expect("step");
        head.loss_and_backward(&input, 1.0, &mut twice)
            .expect("step");
        for ((name, a), (_, b)) in once.named().into_iter().zip(twice.named()) {
            assert!(a.iter().any(|&x| x != 0.0), "{name}: no gradient at all");
            for (&x, &y) in a.iter().zip(b) {
                assert_eq!(2.0 * x, y, "{name}: the bank does not accumulate");
            }
        }

        let mut off = SpanHead::zeros(4).expect("bank");
        let zero = head.loss_and_backward(&input, 0.0, &mut off).expect("step");
        assert_eq!(zero.loss, first.loss, "the logged loss is unweighted");
        assert_eq!(off, SpanHead::zeros(4).expect("bank"));
        assert!(zero.d_hidden[0].rows.iter().all(|&g| g == 0.0));
    }
}
