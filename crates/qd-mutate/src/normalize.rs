//! Normalization, and the record that it happened.
//!
//! A BOM shifts every byte offset by three; CRLF makes byte offsets and line numbers diverge
//! because tree-sitter counts bytes and `\r` is one. The spec's answer is to normalize **first**,
//! mutate the normalized text, and record that normalization happened — not to pretend the input
//! was already clean.
//!
//! Everything downstream (parse, edits, both span derivations, the emitted `before` text) works on
//! the normalized string. The example carries the normalization record so a consumer knows the
//! `before` it is looking at is not byte-identical to the file on disk.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Normalization {
    /// A UTF-8 BOM was stripped from the front.
    pub bom: bool,
    /// At least one CRLF was collapsed to LF.
    pub crlf: bool,
    /// At least one bare CR (old-Mac ending) became LF.
    pub lone_cr: bool,
    /// The file used more than one line ending. Recorded separately because a uniformly-CRLF file
    /// is merely converted, while a mixed file is the case where a naive implementation's byte and
    /// line offsets diverge part-way down the file.
    pub mixed_endings: bool,
}

impl Normalization {
    pub fn any(&self) -> bool {
        self.bom || self.crlf || self.lone_cr || self.mixed_endings
    }
}

/// Strip a UTF-8 BOM and convert every line ending to LF.
pub fn normalize(raw: &str) -> (String, Normalization) {
    let mut record = Normalization::default();

    let body = match raw.strip_prefix('\u{feff}') {
        Some(rest) => {
            record.bom = true;
            rest
        }
        None => raw,
    };

    let mut out = String::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0usize;
    let mut saw_lf_alone = false;
    while i < bytes.len() {
        match bytes[i] {
            b'\r' => {
                if bytes.get(i + 1) == Some(&b'\n') {
                    record.crlf = true;
                    i += 2;
                } else {
                    record.lone_cr = true;
                    i += 1;
                }
                out.push('\n');
            }
            b'\n' => {
                saw_lf_alone = true;
                out.push('\n');
                i += 1;
            }
            _ => {
                // Copy the whole character, not the byte: `out` must stay valid UTF-8 and the
                // multi-byte identifier case walks straight through here.
                let ch_len = utf8_len(bytes[i]);
                let end = (i + ch_len).min(bytes.len());
                out.push_str(&body[i..end]);
                i = end;
            }
        }
    }

    let converted = record.crlf || record.lone_cr;
    record.mixed_endings = (converted && saw_lf_alone) || (record.crlf && record.lone_cr);

    (out, record)
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        // A continuation byte cannot start a character in valid UTF-8, which `&str` guarantees.
        // Advancing by one keeps the loop terminating rather than looping forever on an input that
        // cannot exist.
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bom_is_stripped_and_recorded() {
        let (text, rec) = normalize("\u{feff}fn f() {}\n");
        assert_eq!(text, "fn f() {}\n");
        assert!(rec.bom);
        assert!(!rec.mixed_endings);
    }

    #[test]
    fn crlf_becomes_lf_and_is_recorded() {
        let (text, rec) = normalize("a\r\nb\r\n");
        assert_eq!(text, "a\nb\n");
        assert!(rec.crlf);
        assert!(!rec.mixed_endings, "uniformly CRLF is not mixed");
    }

    #[test]
    fn mixed_endings_are_flagged() {
        let (text, rec) = normalize("a\r\nb\nc\r\n");
        assert_eq!(text, "a\nb\nc\n");
        assert!(rec.crlf);
        assert!(rec.mixed_endings);
    }

    #[test]
    fn a_lone_cr_becomes_a_line_break() {
        let (text, rec) = normalize("a\rb\rc");
        assert_eq!(text, "a\nb\nc");
        assert!(rec.lone_cr);
    }

    #[test]
    fn multibyte_text_survives_byte_for_byte() {
        let src = "// 日本語 🎌\nfn f() {}\n";
        let (text, rec) = normalize(src);
        assert_eq!(text, src);
        assert!(!rec.any());
    }

    #[test]
    fn a_bom_with_crlf_records_both() {
        let (text, rec) = normalize("\u{feff}a\r\n");
        assert_eq!(text, "a\n");
        assert!(rec.bom && rec.crlf);
    }
}
