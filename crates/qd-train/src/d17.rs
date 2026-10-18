//! D17: the trainable `[17, H]` delta over the frozen tied embedding's answer rows, as a host
//! entry of the FT objective (`python/qd_train/lora.py` `AnswerRowDelta`, Lappi's served
//! readout; the export carries it as `d17.delta`, `qd-export`'s `layout.rs`).
//!
//! **The readout.** `R = E[answer ids] + D17` in f32, `[17, H]`; the logits at a supervised
//! position with final hidden state `h` are `h R^T`; the loss is the 17-way cross-entropy at
//! the answer row of that position's target, **mean over the batch's `N` supervised positions**
//! (`AnswerRowDelta.loss`). With `dz = (softmax - onehot) * scale` and `scale = 1/N`:
//!
//! ```text
//! dh   = dz R          [positions, H]   returned to the provider through the hidden callback
//! dD17 = dz^T h        [17, H]          accumulated into this entry's gradient
//! ```
//!
//! `E` is frozen and tied, so no gradient reaches the table; only `D17` trains. The 17 ids are
//! in D17's row order, the 16 option letters then noul (`qd_export::export::d17_row_letters`),
//! the order `qd-metal`'s `Model::set_answer_delta` serves. Row `k` of `R` is the embedding row
//! of `ids[k]`, so training and serving agree on which answer is row `k`.
//!
//! **Numerics.** `R` is formed in f32 (as `AnswerRowDelta.rows` does); the logits, the softmax
//! and the two products are accumulated in f64 and rounded to f32 once. That keeps the result
//! within about 6e-8 of an f64 evaluation of the same rule, so the parity gates measure the rule
//! and not the summation order.
//!
//! **Fail closed.** A target that is not one of the 17 answer ids is refused, never skipped
//! (a skip would shrink `N` and report a lower loss, as `AnswerRowDelta.loss` refuses); so are
//! duplicate or out-of-vocabulary answer ids, a shape that is not `[17, H]`, and any non-finite
//! embedding, delta, hidden state or scale. A refused call leaves the gradient untouched.

use std::path::Path;

use qd_export::bf16::{bf16_bits_to_f32, f16_bits_to_f32};
use qd_export::safetensors::{Dtype, SafeTensorsFile};

use crate::step::ParamSpec;
use crate::trainer::HostParams;

/// The host entry's tensor name: `d17.delta`, as in `tools/ckpt_average.py`'s `d17` tree,
/// `qd-export` and `tools/gen_span_fixture.py`.
pub const D17_ENTRY: &str = "d17.delta";

/// The number of answer rows: the 16 option letters and noul.
pub const D17_ROWS: usize = qd_export::layout::D17_ROWS;

/// The most elements of an embedding table this reader will read rows from; a bound, not a size
/// the model has (Qwen3.5-2B: 248,320 x 2,048).
const MAX_TABLE_ELEMENTS: u64 = 1 << 31;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum D17Error {
    #[error("shape: {0}")]
    Shape(String),
    #[error("non-finite: {0}")]
    NonFinite(String),
    #[error("answer id {id} appears at rows {first} and {second}; the 17 ids must be distinct")]
    DuplicateId { id: u32, first: usize, second: usize },
    #[error("answer id {id} is outside the vocabulary ({vocab})")]
    IdOutOfVocab { id: u32, vocab: usize },
    #[error(
        "target {target} at position {index} is not one of the 17 answer rows; a skipped target \
         would shrink the denominator and report a lower loss"
    )]
    TargetNotAnswerRow { target: u32, index: usize },
    #[error("scale: {0}")]
    Scale(String),
    #[error("positions: {0}")]
    Positions(String),
    #[error("checkpoint: {0}")]
    Checkpoint(String),
}

/// The delta, the frozen answer rows it rides on, and its gradient.
#[derive(Debug, Clone)]
pub struct D17 {
    hidden: usize,
    ids: [u32; D17_ROWS],
    /// `E[ids]`, `[17, H]` row-major, f32 (bf16 upcast exactly).
    answer_rows: Vec<f32>,
    delta: Vec<f32>,
    grad: Option<Vec<f32>>,
}

fn check_ids(ids: &[u32; D17_ROWS], vocab: usize) -> Result<(), D17Error> {
    for (k, &id) in ids.iter().enumerate() {
        if id as usize >= vocab {
            return Err(D17Error::IdOutOfVocab { id, vocab });
        }
        if let Some(first) = ids[..k].iter().position(|&o| o == id) {
            return Err(D17Error::DuplicateId {
                id,
                first,
                second: k,
            });
        }
    }
    Ok(())
}

fn check_finite(what: &str, values: &[f32]) -> Result<(), D17Error> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(D17Error::NonFinite(format!("{what}[{i}] is {}", values[i]))),
        None => Ok(()),
    }
}

impl D17 {
    /// From the 17 answer rows themselves (`[17, H]`, the embedding rows of `ids` in row order).
    /// The delta starts at zero, so an untrained D17 reproduces the base's own letter rows.
    pub fn from_rows(
        hidden: usize,
        ids: [u32; D17_ROWS],
        answer_rows: Vec<f32>,
    ) -> Result<Self, D17Error> {
        if hidden == 0 {
            return Err(D17Error::Shape("hidden width 0".into()));
        }
        let want = D17_ROWS
            .checked_mul(hidden)
            .ok_or_else(|| D17Error::Shape(format!("[17, {hidden}] overflows")))?;
        if answer_rows.len() != want {
            return Err(D17Error::Shape(format!(
                "{} answer-row values for [17, {hidden}] = {want}",
                answer_rows.len()
            )));
        }
        check_finite("answer rows", &answer_rows)?;
        Ok(Self {
            hidden,
            ids,
            answer_rows,
            delta: vec![0.0; want],
            grad: None,
        })
    }

    /// From a full `[vocab, hidden]` embedding table already in f32.
    pub fn from_embedding(
        table: &[f32],
        vocab: usize,
        hidden: usize,
        ids: [u32; D17_ROWS],
    ) -> Result<Self, D17Error> {
        if hidden == 0 || vocab == 0 {
            return Err(D17Error::Shape(format!("table [{vocab}, {hidden}]")));
        }
        if vocab.checked_mul(hidden) != Some(table.len()) {
            return Err(D17Error::Shape(format!(
                "{} table values for [{vocab}, {hidden}]",
                table.len()
            )));
        }
        check_ids(&ids, vocab)?;
        let mut rows = Vec::with_capacity(D17_ROWS * hidden);
        for &id in &ids {
            let at = id as usize * hidden;
            rows.extend_from_slice(&table[at..at + hidden]);
        }
        Self::from_rows(hidden, ids, rows)
    }

    /// The 17 rows read by tensor name from a safetensors checkpoint: only those rows' bytes are
    /// read, never the table. `embed_name` is the checkpoint's tied embedding (`BF16`, `F16` or
    /// `F32`, shape `[vocab, hidden]`); a name that is absent is refused, not guessed.
    pub fn from_safetensors(
        path: &Path,
        embed_name: &str,
        ids: [u32; D17_ROWS],
    ) -> Result<Self, D17Error> {
        let ck = |e: &dyn std::fmt::Display| D17Error::Checkpoint(format!("{}: {e}", path.display()));
        let file = SafeTensorsFile::open(path).map_err(|e| ck(&e))?;
        let info = file.tensors().get(embed_name).ok_or_else(|| {
            D17Error::Checkpoint(format!(
                "{}: no tensor named {embed_name:?}",
                path.display()
            ))
        })?;
        let [vocab, hidden] = info.shape[..] else {
            return Err(D17Error::Shape(format!(
                "{embed_name} has shape {:?}, not [vocab, hidden]",
                info.shape
            )));
        };
        if (vocab as u64).saturating_mul(hidden as u64) > MAX_TABLE_ELEMENTS {
            return Err(D17Error::Shape(format!(
                "{embed_name} [{vocab}, {hidden}] exceeds {MAX_TABLE_ELEMENTS} elements"
            )));
        }
        check_ids(&ids, vocab)?;
        let size = info.dtype.size();
        let mut rows = Vec::with_capacity(D17_ROWS * hidden);
        let mut raw = vec![0u8; hidden * size];
        for &id in &ids {
            file.read_tensor_range(info, (id as usize * hidden * size) as u64, &mut raw)
                .map_err(|e| ck(&e))?;
            match info.dtype {
                Dtype::F32 => rows.extend(
                    raw.as_chunks::<4>().0.iter()
                        .map(|b| f32::from_le_bytes(*b)),
                ),
                Dtype::Bf16 => rows.extend(
                    raw.as_chunks::<2>().0.iter()
                        .map(|b| bf16_bits_to_f32(u16::from_le_bytes(*b))),
                ),
                Dtype::F16 => rows.extend(
                    raw.as_chunks::<2>().0.iter()
                        .map(|b| f16_bits_to_f32(u16::from_le_bytes(*b))),
                ),
                other => {
                    return Err(D17Error::Checkpoint(format!(
                        "{embed_name} is {}, not a float embedding",
                        other.as_str()
                    )));
                }
            }
        }
        Self::from_rows(hidden, ids, rows)
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn ids(&self) -> &[u32; D17_ROWS] {
        &self.ids
    }

    /// `E[ids]`, `[17, H]`.
    pub fn answer_rows(&self) -> &[f32] {
        &self.answer_rows
    }

    /// The delta, `[17, H]`.
    pub fn delta(&self) -> &[f32] {
        &self.delta
    }

    /// Replace the delta (a resume, or a loaded release). Must be `[17, H]` and finite.
    pub fn set_delta(&mut self, delta: &[f32]) -> Result<(), D17Error> {
        if delta.len() != self.delta.len() {
            return Err(D17Error::Shape(format!(
                "{} delta values for [17, {}] = {}",
                delta.len(),
                self.hidden,
                self.delta.len()
            )));
        }
        check_finite("delta", delta)?;
        self.delta.copy_from_slice(delta);
        Ok(())
    }

    /// `R = E + D17`, `[17, H]`, in f32.
    fn composed(&self) -> Result<Vec<f32>, D17Error> {
        check_finite("delta", &self.delta)?;
        Ok(self
            .answer_rows
            .iter()
            .zip(&self.delta)
            .map(|(e, d)| e + d)
            .collect())
    }

    fn positions_of(&self, hidden: &[f32]) -> Result<usize, D17Error> {
        if hidden.is_empty() || !hidden.len().is_multiple_of(self.hidden) {
            return Err(D17Error::Shape(format!(
                "{} hidden values is not a non-empty multiple of the width {}",
                hidden.len(),
                self.hidden
            )));
        }
        check_finite("hidden", hidden)?;
        Ok(hidden.len() / self.hidden)
    }

    fn logits_f64(r: &[f32], h: &[f32], width: usize, n: usize) -> Vec<f64> {
        let mut z = vec![0.0f64; n * D17_ROWS];
        for p in 0..n {
            let hp = &h[p * width..(p + 1) * width];
            for k in 0..D17_ROWS {
                let rk = &r[k * width..(k + 1) * width];
                z[p * D17_ROWS + k] = hp.iter().zip(rk).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
            }
        }
        z
    }

    /// The `[N, 17]` f32 logits `h R^T` from `hidden` `[N, H]`.
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>, D17Error> {
        let n = self.positions_of(hidden)?;
        let r = self.composed()?;
        Ok(Self::logits_f64(&r, hidden, self.hidden, n)
            .into_iter()
            .map(|v| v as f32)
            .collect())
    }

    /// The row of `id` among the answer rows.
    pub fn row_of(&self, id: u32) -> Option<usize> {
        self.ids.iter().position(|&a| a == id)
    }

    /// One call's forward and backward. `hidden` is `[N, H]`, `targets[i]` the vocabulary id
    /// the position `i` predicts. Returns the UNSCALED sum of the `N` 17-way cross-entropies and
    /// `scale * dL/dh` (`[N, H]`); adds `scale * dL/dD17` to this entry's gradient. The batch
    /// mean's `scale` is `1/N` over every supervised position of the batch.
    pub fn loss_and_grad(
        &mut self,
        hidden: &[f32],
        targets: &[u32],
        scale: f32,
    ) -> Result<(f64, Vec<f32>), D17Error> {
        if !(scale.is_finite() && scale >= 0.0) {
            return Err(D17Error::Scale(format!("{scale} must be finite and >= 0")));
        }
        let n = self.positions_of(hidden)?;
        if targets.len() != n {
            return Err(D17Error::Shape(format!(
                "{} targets for {n} hidden positions",
                targets.len()
            )));
        }
        let mut rows = Vec::with_capacity(n);
        for (index, &target) in targets.iter().enumerate() {
            rows.push(
                self.row_of(target)
                    .ok_or(D17Error::TargetNotAnswerRow { target, index })?,
            );
        }
        let w = self.hidden;
        let r = self.composed()?;
        let z = Self::logits_f64(&r, hidden, w, n);
        let scale = f64::from(scale);
        let mut loss = 0.0f64;
        let mut dh = vec![0.0f64; n * w];
        let mut dd = vec![0.0f64; D17_ROWS * w];
        for p in 0..n {
            let zp = &z[p * D17_ROWS..(p + 1) * D17_ROWS];
            let m = zp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = zp.iter().map(|v| (v - m).exp()).sum();
            loss += m + sum.ln() - zp[rows[p]];
            let hp = &hidden[p * w..(p + 1) * w];
            for k in 0..D17_ROWS {
                let dz = ((zp[k] - m).exp() / sum - if k == rows[p] { 1.0 } else { 0.0 }) * scale;
                let rk = &r[k * w..(k + 1) * w];
                for j in 0..w {
                    dh[p * w + j] += dz * f64::from(rk[j]);
                    dd[k * w + j] += dz * f64::from(hp[j]);
                }
            }
        }
        if !loss.is_finite() {
            return Err(D17Error::NonFinite(format!("the 17-way loss is {loss}")));
        }
        let grad = self.grad.get_or_insert_with(|| vec![0.0; D17_ROWS * w]);
        for (g, d) in grad.iter_mut().zip(&dd) {
            *g += *d as f32;
        }
        Ok((loss, dh.into_iter().map(|v| v as f32).collect()))
    }
}

impl HostParams for D17 {
    fn entries(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::new(D17_ENTRY, &[D17_ROWS, self.hidden])]
    }

    fn zero_grads(&mut self) {
        self.grad = None;
    }

    fn grads(&self) -> Vec<Option<&[f32]>> {
        vec![self.grad.as_deref()]
    }

    fn values(&self) -> Vec<&[f32]> {
        vec![&self.delta]
    }

    fn values_and_grads(&mut self) -> Vec<(&mut [f32], Option<&[f32]>)> {
        vec![(&mut self.delta[..], self.grad.as_deref())]
    }
}

/// One loss's gradient at the positions it read.
#[derive(Debug, Clone, Copy)]
pub struct HiddenPart<'a> {
    pub positions: &'a [u32],
    /// `[positions.len(), width]`, row-major.
    pub grad: &'a [f32],
}

/// The hidden gradient the provider takes, for `union` (distinct positions, the provider's
/// `hidden_at`), from several losses that read subsets of it: where two parts read the same
/// position their rows are summed once, so the provider back-propagates one gradient per
/// position. Parts are added in order, f32.
pub fn merge_hidden_grads(
    union: &[u32],
    parts: &[HiddenPart<'_>],
    width: usize,
) -> Result<Vec<f32>, D17Error> {
    if width == 0 {
        return Err(D17Error::Shape("hidden width 0".into()));
    }
    for (i, p) in union.iter().enumerate() {
        if union[..i].contains(p) {
            return Err(D17Error::Positions(format!(
                "position {p} appears twice in the provider's hidden_at"
            )));
        }
    }
    let mut out = vec![0.0f32; union.len() * width];
    for part in parts {
        if part.grad.len() != part.positions.len() * width {
            return Err(D17Error::Shape(format!(
                "{} gradient values for {} positions of width {width}",
                part.grad.len(),
                part.positions.len()
            )));
        }
        for (i, pos) in part.positions.iter().enumerate() {
            let at = union.iter().position(|u| u == pos).ok_or_else(|| {
                D17Error::Positions(format!("position {pos} is not among {union:?}"))
            })?;
            for j in 0..width {
                out[at * width + j] += part.grad[i * width + j];
            }
        }
    }
    Ok(out)
}
