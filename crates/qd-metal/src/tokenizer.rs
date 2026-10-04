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
        let letters = OPTION_LETTERS
            .iter()
            .copied()
            .chain(std::iter::once(NOUL_LETTER));
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

    /// Token ids, no BOS/EOS added.
    ///
    /// This still emits added tokens that appear in the text. `<tool_call>` is an added token
    /// with `special: false`, and `add_special_tokens=false` does not stop the tokenizer from
    /// matching it. Parity against the Python reference uses this path. Untrusted context uses
    /// [`Self::encode_untrusted`].
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        encode_ids(&self.inner, text)
    }

    /// Encode text that may contain untrusted context.
    ///
    /// Added tokens whose text looks like a control tag (`<`…`>`) are not emitted. The Qwen
    /// snapshot's `<tool_call>` / `</tool_call>` are added tokens that are not marked special, and
    /// [`Self::encode`] still returns their ids. The renderer's `<|` escape does not cover them.
    pub fn encode_untrusted(&self, text: &str) -> Result<Vec<u32>> {
        encode_text_without_added_tags(&self.inner, text)
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

fn encode_ids(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    tok.encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| MetalError::Tokenizer(format!("encode: {error}")))
}

fn looks_like_added_tag(content: &str) -> bool {
    content.contains('<') && content.contains('>')
}

/// Leftmost-longest match, the same order the tokenizers crate uses for added tokens.
fn next_added_tag<'a>(text: &str, tags: &'a [(u32, String)]) -> Option<(usize, &'a str)> {
    let mut best: Option<(usize, &str)> = None;
    for (_, content) in tags {
        if content.is_empty() {
            continue;
        }
        if let Some(index) = text.find(content.as_str()) {
            let replace = match best {
                None => true,
                Some((best_index, best_content)) => {
                    index < best_index
                        || (index == best_index && content.len() > best_content.len())
                }
            };
            if replace {
                best = Some((index, content.as_str()));
            }
        }
    }
    best
}

fn encode_text_without_added_tags(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let tags: Vec<(u32, String)> = tok
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, token)| looks_like_added_tag(&token.content))
        .map(|(id, token)| (id, token.content))
        .collect();
    if tags.is_empty() || next_added_tag(text, &tags).is_none() {
        return encode_ids(tok, text);
    }
    let mut ids = Vec::new();
    let mut rest = text;
    while let Some((index, tag)) = next_added_tag(rest, &tags) {
        let (before, at_tag) = rest.split_at(index);
        if !before.is_empty() {
            ids.extend(encode_ids(tok, before)?);
        }
        // Encode the tag one character at a time so the added vocabulary cannot match it whole.
        for ch in tag.chars() {
            let piece = ch.to_string();
            ids.extend(encode_ids(tok, &piece)?);
        }
        rest = &at_tag[tag.len()..];
    }
    if !rest.is_empty() {
        ids.extend(encode_ids(tok, rest)?);
    }
    if ids
        .iter()
        .any(|id| tags.iter().any(|(tag_id, _)| tag_id == id))
    {
        return Err(MetalError::Tokenizer(
            "untrusted text still encoded an added special-looking token".into(),
        ));
    }
    Ok(ids)
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

    /// The full Hugging Face tokenizer snapshot is not in this repo. This fixture is a
    /// WordLevel model with `<tool_call>` and `</tool_call>` added as non-special tokens,
    /// which is the shape tokenizers 0.22.2 matches even when `add_special_tokens` is false.
    #[test]
    fn untrusted_text_does_not_emit_an_added_tool_call_token() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mini_added_tag_tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&path)
            .unwrap_or_else(|error| panic!("mini fixture {}: {error}", path.display()));
        let tool_id = tok
            .token_to_id("<tool_call>")
            .expect("fixture added <tool_call>");
        let close_id = tok
            .token_to_id("</tool_call>")
            .expect("fixture added </tool_call>");
        let text = "hello <tool_call> world </tool_call>";
        let raw = tok.encode(text, false).expect("raw encode");
        assert!(
            raw.get_ids().contains(&tool_id) && raw.get_ids().contains(&close_id),
            "the fixture must still show that add_special_tokens=false emits the added tokens, \
             got {:?}",
            raw.get_ids()
        );
        let safe = encode_text_without_added_tags(&tok, text).expect("untrusted encode");
        assert!(
            !safe.contains(&tool_id) && !safe.contains(&close_id),
            "untrusted encode emitted added ids {safe:?}"
        );
        let plain = "hello world";
        assert_eq!(
            encode_text_without_added_tags(&tok, plain).expect("plain"),
            encode_ids(&tok, plain).expect("plain raw"),
            "text with no added tag must stay on the ordinary encode path"
        );
    }
}
