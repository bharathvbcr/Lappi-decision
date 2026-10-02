//! The real FT batch and the real span head behind the objective's two small traits: L-data's
//! [`Batch`] as an [`FtBatch`], L-head's [`span_head::SpanHead`] as an [`objective::SpanHead`].
//!
//! **No supervision is decided here.** Which rows are letters and where, which are spans and
//! what their candidates and gold are, is L-data's port of Python's `ft_supervision` and
//! `plan_span_batch` ([`ft_supervision`], [`plan_span_batch`], pinned to Python by
//! `tests/parity.rs`); this module only re-shapes their output per row, and
//! `tests/ft_data.rs` checks the re-shaped rows against the same Python dump. The consumed
//! digest is L-data's `fold_batch`.
//!
//! **What the letter channel is.** `ft_supervision` sets exactly one letter position per
//! non-span row (`choice` and `score` alike, at `target_index`), and none on a span row; a row
//! whose letter or span positions run past its real tokens is refused here, because the
//! provider runs each row on its real tokens only (a right-padded causal row's real positions
//! do not see its padding).
//!
//! **The span head through the row-level trait.** [`objective::SpanHead::loss_and_grad`] scores
//! one row at a time with the batch's scale `span_weight / (2K)` already applied; L-head's
//! [`span_head::SpanHead::loss_and_backward`] takes a batch and divides by its own `2K`. One row
//! at weight `2 * scale` is the same gradient: L-head forms `f64(2 * scale) / 2`, and doubling an
//! f32 is exact. The row's unscaled loss is recomputed in f64 from the f32 scores L-head returns
//! (log-sum-exp minus the gold score, start plus end), not read back from its f32 mean.

use std::path::Path;

use qd_export::safetensors::{Dtype, SafeTensorsFile};

use crate::objective::{self, FtBatch, LetterTarget};
use crate::run_control::hex;
use crate::shards::{Batch, ConsumedPrefix};
use crate::span_head::{self, SpanGold, SpanRowInput, SpanRowPlan, PARAMETER_NAMES};
use crate::step::ParamSpec;
use crate::supervision::{ft_supervision, plan_span_batch};
use crate::trainer::{ConsumedBatch, HostParams, TrainError};

fn refuse<T>(m: impl Into<String>) -> Result<T, TrainError> {
    Err(TrainError::Refused(m.into()))
}

fn token(x: i32, what: &str) -> Result<u32, TrainError> {
    u32::try_from(x).map_err(|_| TrainError::Refused(format!("{what} is {x}, not a token id or position")))
}

/// One shard [`Batch`], with each row's real tokens and its one supervised answer.
pub struct RealBatch {
    batch: Batch,
    tokens: Vec<Vec<u32>>,
    letters: Vec<Option<LetterTarget>>,
    spans: Vec<Option<SpanRowPlan>>,
}

impl RealBatch {
    /// Re-shape `batch` per row through L-data's supervision port. Refuses what that port
    /// refuses, and a row whose supervised positions are not inside its real tokens.
    pub fn new(batch: Batch) -> Result<Self, TrainError> {
        let at = format!("batch {}", batch.index);
        batch.validate().map_err(|e| TrainError::Refused(format!("{at}: {e}")))?;
        let sup = ft_supervision(&batch).map_err(|e| TrainError::Refused(format!("{at}: {e}")))?;
        let rows = batch.rows();
        let mut tokens = Vec::with_capacity(rows);
        for r in 0..rows {
            let len = usize::try_from(batch.lengths[r])
                .ok()
                .filter(|&l| l >= 1 && l <= batch.width)
                .ok_or_else(|| {
                    TrainError::Refused(format!("{at} row {r}: length {} in a batch {} wide", batch.lengths[r], batch.width))
                })?;
            tokens.push(
                (0..len)
                    .map(|p| token(batch.token(r, p), &format!("{at} row {r} token {p}")))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        let mut letters: Vec<Option<LetterTarget>> = vec![None; rows];
        for (r, c) in sup.letter_positions() {
            if letters[r].is_some() {
                return refuse(format!("{at} row {r}: more than one supervised letter position"));
            }
            let target = token(sup.targets[r * sup.columns + c], &format!("{at} row {r} letter target"))?;
            let position = u32::try_from(c).map_err(|e| TrainError::Refused(e.to_string()))?;
            if c + 1 >= tokens[r].len() {
                return refuse(format!(
                    "{at} row {r}: the letter at {c} predicts position {} of a {}-token row",
                    c + 1,
                    tokens[r].len()
                ));
            }
            letters[r] = Some(LetterTarget { position, target });
        }
        let mut spans: Vec<Option<SpanRowPlan>> = vec![None; rows];
        if let Some(span) = &sup.span {
            let plan = plan_span_batch(span).map_err(|e| TrainError::Refused(format!("{at}: {e}")))?;
            for (k, &r) in span.rows.iter().enumerate() {
                let n = usize::try_from(plan.n_candidates[k]).map_err(|e| TrainError::Refused(e.to_string()))?;
                let candidates = plan.candidate_pos[k * plan.max_candidates..k * plan.max_candidates + n]
                    .iter()
                    .map(|&p| u32::try_from(p).map_err(|_| TrainError::Refused(format!("{at} row {r}: candidate {p}"))))
                    .collect::<Result<Vec<_>, _>>()?;
                let gold = if plan.abstaining[k] {
                    SpanGold::Abstain
                } else {
                    let ord = |g: i64| usize::try_from(g).map_err(|_| TrainError::Refused(format!("{at} row {r}: gold {g}")));
                    SpanGold::Lines {
                        start: ord(plan.gold_start[k])?,
                        end: ord(plan.gold_end[k])?,
                    }
                };
                let query = u32::try_from(plan.query_index[k])
                    .map_err(|_| TrainError::Refused(format!("{at} row {r}: query index {}", plan.query_index[k])))?;
                let row = SpanRowPlan::new(query, candidates, gold)
                    .map_err(|e| TrainError::Refused(format!("{at} row {r}: {e}")))?;
                if let Some(&p) = row.positions().last()
                    && p as usize >= tokens[r].len()
                {
                    return refuse(format!(
                        "{at} row {r}: the span head reads position {p} of a {}-token row",
                        tokens[r].len()
                    ));
                }
                if letters[r].is_some() {
                    return refuse(format!("{at} row {r} is both a letter row and a span row"));
                }
                spans[r] = Some(row);
            }
        }
        Ok(Self {
            batch,
            tokens,
            letters,
            spans,
        })
    }

    pub fn batch(&self) -> &Batch {
        &self.batch
    }
}

impl ConsumedBatch for RealBatch {
    fn index(&self) -> u64 {
        self.batch.index
    }

    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError> {
        prefix.fold_batch(&self.batch);
        Ok(())
    }
}

impl FtBatch for RealBatch {
    type Span = SpanRowPlan;

    fn n_rows(&self) -> usize {
        self.tokens.len()
    }

    fn padded_width(&self) -> usize {
        self.batch.width
    }

    fn row_tokens(&self, row: usize) -> &[u32] {
        &self.tokens[row]
    }

    fn letter(&self, row: usize) -> Option<LetterTarget> {
        self.letters[row]
    }

    fn span(&self, row: usize) -> Option<&SpanRowPlan> {
        self.spans[row].as_ref()
    }
}

/// L-head's span pointer head and its gradient bank, trained on the host.
pub struct HostSpanHead {
    params: span_head::SpanHead,
    bank: span_head::SpanHead,
}

impl HostSpanHead {
    pub fn new(params: span_head::SpanHead) -> Result<Self, TrainError> {
        let bank = span_head::SpanHead::zeros(params.hidden_size()).map_err(|e| TrainError::Refused(e.to_string()))?;
        Ok(Self { params, bank })
    }

    pub fn params(&self) -> &span_head::SpanHead {
        &self.params
    }
}

/// `(lse(scores) - scores[gold])` in f64 over f32 scores: one pointer's cross-entropy.
fn pointer_ce(scores: &[f32], gold: usize) -> Result<f64, String> {
    let g = *scores.get(gold).ok_or_else(|| format!("gold row {gold} of {} scores", scores.len()))?;
    let max = scores.iter().map(|&s| f64::from(s)).fold(f64::NEG_INFINITY, f64::max);
    let lse = max + scores.iter().map(|&s| (f64::from(s) - max).exp()).sum::<f64>().ln();
    Ok(lse - f64::from(g))
}

impl HostParams for HostSpanHead {
    fn entries(&self) -> Vec<ParamSpec> {
        let h = self.params.hidden_size();
        vec![
            ParamSpec::new(PARAMETER_NAMES[0], &[h, h]),
            ParamSpec::new(PARAMETER_NAMES[1], &[h, h]),
            ParamSpec::new(PARAMETER_NAMES[2], &[h]),
            ParamSpec::new(PARAMETER_NAMES[3], &[h]),
        ]
    }

    fn zero_grads(&mut self) {
        for (_, g) in self.bank.named_mut() {
            g.fill(0.0);
        }
    }

    fn grads(&self) -> Vec<&[f32]> {
        self.bank.named().into_iter().map(|(_, g)| g).collect()
    }

    fn values(&self) -> Vec<&[f32]> {
        self.params.named().into_iter().map(|(_, v)| v).collect()
    }

    fn values_and_grads(&mut self) -> Vec<(&mut [f32], &[f32])> {
        self.params
            .named_mut()
            .into_iter()
            .zip(self.bank.named())
            .map(|((_, v), (_, g))| (v, g))
            .collect()
    }
}

impl objective::SpanHead for HostSpanHead {
    type Span = SpanRowPlan;

    fn positions(&self, span: &SpanRowPlan) -> Result<Vec<u32>, String> {
        Ok(span.positions().to_vec())
    }

    fn loss_and_grad(
        &mut self,
        span: &SpanRowPlan,
        positions: &[u32],
        hidden: &[f32],
        width: usize,
        scale: f32,
    ) -> Result<(f64, Vec<f32>), String> {
        if width != self.params.hidden_size() {
            return Err(format!("hidden width {width}, the head is {} wide", self.params.hidden_size()));
        }
        if positions != span.positions() {
            return Err(format!("positions {positions:?} are not the row's {:?}", span.positions()));
        }
        let weight = 2.0 * scale;
        let step = self
            .params
            .loss_and_backward(&[SpanRowInput { plan: span, hidden }], weight, &mut self.bank)
            .map_err(|e| e.to_string())?;
        let (Some(scores), Some(dh)) = (step.scores.first(), step.d_hidden.into_iter().next()) else {
            return Err("the head returned no row for a one-row batch".into());
        };
        if dh.positions != positions {
            return Err(format!("the head's gradient is at {:?}, not {positions:?}", dh.positions));
        }
        let loss = pointer_ce(&scores.start, span.gold_start())? + pointer_ce(&scores.end, span.gold_end())?;
        Ok((loss, dh.rows))
    }
}

/// The span head's initial weights from a safetensors file, and the sha256 of its bytes (the
/// digest rung (d)'s arms compare first). The file holds exactly the four f32 tensors of
/// [`PARAMETER_NAMES`] at their shapes, all bare or all under `span_head.` (the export's
/// prefix); anything else -- another dtype, a missing, extra or misshapen tensor, a mixed
/// prefix, a non-finite value -- is refused.
pub fn load_span_head(path: &Path, hidden_size: usize) -> Result<(span_head::SpanHead, String), TrainError> {
    let at = path.display().to_string();
    let file = SafeTensorsFile::open(path).map_err(|e| TrainError::Refused(format!("{at}: {e}")))?;
    let sha = hex(&file.sha256().map_err(|e| TrainError::Refused(format!("{at}: {e}")))?);
    let names: Vec<&String> = file.tensors().keys().collect();
    let prefix = if names.iter().all(|n| n.starts_with("span_head.")) { "span_head." } else { "" };
    let want: Vec<String> = PARAMETER_NAMES.iter().map(|n| format!("{prefix}{n}")).collect();
    let mut have: Vec<String> = names.iter().map(|n| (*n).clone()).collect();
    have.sort();
    let mut sorted_want = want.clone();
    sorted_want.sort();
    if have != sorted_want {
        return refuse(format!(
            "{at} holds {have:?}; a span-head init holds exactly {:?}, all bare or all under 'span_head.'",
            PARAMETER_NAMES
        ));
    }
    let h = hidden_size;
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(4);
    for (name, shape) in want.iter().zip([vec![h, h], vec![h, h], vec![h], vec![h]]) {
        let info = &file.tensors()[name];
        if info.dtype != Dtype::F32 || info.shape != shape {
            return refuse(format!("{at}: {name} is {:?} {:?}, not f32 {shape:?}", info.dtype, info.shape));
        }
        let mut bytes = vec![0u8; shape.iter().product::<usize>() * 4];
        file.read_tensor_range(info, 0, &mut bytes)
            .map_err(|e| TrainError::Refused(format!("{at}: {e}")))?;
        let v: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return refuse(format!("{at}: {name}[{i}] = {} is not finite", v[i]));
        }
        out.push(v);
    }
    let mut it = out.into_iter();
    let (Some(s), Some(e), Some(a_s), Some(a_e)) = (it.next(), it.next(), it.next(), it.next()) else {
        return refuse(format!("{at}: four tensors were checked but not all were read"));
    };
    let head = span_head::SpanHead::new(h, s, e, a_s, a_e).map_err(|e| TrainError::Refused(format!("{at}: {e}")))?;
    Ok((head, sha))
}
