//! The Qwen tokenizer from the snapshot's `tokenizer.json`, and the answer-letter ids.
//!
//! Training scores the letter that follows `<|qd_answer|>` as the **last token of the row**
//! (`qd_train.shards.training_texts`: `target_index = lengths - 2`), so the letter id is the id
//! of the letter on its own, and the logits are read at the last prompt position. That holds only
//! if the tokenizer never merges the letter into the prompt's final token. [`QwenTokenizer::load`]
//! requires every letter to be exactly one token; `tools/dump_torch_reference.py` checks the
//! boundary itself (`ids(prompt) + [id(L)] == ids(prompt + L)` for all 17) on every parity prompt.

use std::path::Path;

use qd_runtime::render::{NOUL_LETTER, OPTION_LETTERS};

use crate::error::{MetalError, Result};

/// Rows of the letter slice: 16 option letters + the reserved noul row.
pub const LETTER_ROWS: usize = 17;

pub struct QwenTokenizer {
    inner: tokenizers::Tokenizer,
    /// `A`..`P` then `Z`, in `qd_runtime::render::SlotRender::letter_for_row` order.
    letter_ids: [u32; LETTER_ROWS],
    /// SHA-256 of `tokenizer.json`, hex.
    hash: String,
}

impl QwenTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| MetalError::Tokenizer(format!("{}: {e}", path.display())))?;
        let inner = tokenizers::Tokenizer::from_bytes(&bytes)
            .map_err(|e| MetalError::Tokenizer(format!("{}: {e}", path.display())))?;
        let mut t = Self {
            inner,
            letter_ids: [0; LETTER_ROWS],
            hash: qd_runtime::hex(&qd_runtime::sha256(&bytes)),
        };
        let letters = OPTION_LETTERS.iter().copied().chain(std::iter::once(NOUL_LETTER));
        for (row, letter) in letters.enumerate() {
            let text = char::from(letter).to_string();
            let ids = t.encode(&text)?;
            let [id] = ids.as_slice() else {
                return Err(MetalError::Tokenizer(format!(
                    "letter {text:?} encodes to {ids:?}, not one token; the answer row would not \
                     be one logit"
                )));
            };
            if t.letter_ids[..row].contains(id) {
                return Err(MetalError::Tokenizer(format!(
                    "letter {text:?} shares id {id} with an earlier letter"
                )));
            }
            t.letter_ids[row] = *id;
        }
        Ok(t)
    }

    /// Token ids, no special tokens added (the Qwen tokenizer adds none by default; the Python
    /// reference encodes the same way).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| MetalError::Tokenizer(format!("encode: {e}")))
    }

    pub fn hash(&self) -> &str {
        &self.hash
    }

    /// All 17 letter ids: `A`..`P`, then `Z` (noul).
    pub fn letter_ids(&self) -> &[u32; LETTER_ROWS] {
        &self.letter_ids
    }

    /// The ids a slot of `rows` decode rows reads: the first `rows - 1` option letters, then noul.
    /// This is `SlotRender::letter_for_row` for every row.
    pub fn answer_ids(&self, rows: usize) -> Result<Vec<u32>> {
        answer_ids(&self.letter_ids, rows)
    }
}

pub fn answer_ids(letters: &[u32; LETTER_ROWS], rows: usize) -> Result<Vec<u32>> {
    if !(1..=LETTER_ROWS).contains(&rows) {
        return Err(MetalError::Input(format!(
            "a letter query reads 1..={LETTER_ROWS} rows (options + noul), not {rows}"
        )));
    }
    let mut ids = letters[..rows - 1].to_vec();
    ids.push(letters[LETTER_ROWS - 1]);
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_rows_end_on_noul() {
        let letters: [u32; LETTER_ROWS] = std::array::from_fn(|i| 100 + i as u32);
        assert_eq!(answer_ids(&letters, 5).unwrap(), [100, 101, 102, 103, 116]);
        assert_eq!(answer_ids(&letters, 1).unwrap(), [116]);
        assert_eq!(answer_ids(&letters, 17).unwrap().len(), 17);
        assert!(answer_ids(&letters, 0).is_err());
        assert!(answer_ids(&letters, 18).is_err());
    }
}
