//! One unified-diff hunk in the exact shape the `code.defect_class` corpus uses.
//!
//! Measured over all 50,177 rows of `commitpackft-corpus-v2` on 2026-09-30: every diff is ONE
//! hunk, its header is `@@ -a,old +a,new @@` with both counts present, the same start on both
//! sides and no section heading, and 83% of bodies carry all three markers (` `, `-`, `+`).
//! A noul row that differed from that shape would let the model answer `noul` from the shape
//! instead of the content, and the OOD suite's code cases are multi-hunk with headings -- a
//! shortcut learned here would be graded there. So the header is built from the body, never
//! written by hand, and the change shapes follow the corpus's marker mix.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    Context,
    Removed,
    Added,
}

impl Mark {
    fn prefix(self) -> char {
        match self {
            Mark::Context => ' ',
            Mark::Removed => '-',
            Mark::Added => '+',
        }
    }

    pub fn from_prefix(c: char) -> Option<Mark> {
        match c {
            ' ' => Some(Mark::Context),
            '-' => Some(Mark::Removed),
            '+' => Some(Mark::Added),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    pub mark: Mark,
    pub text: String,
}

/// `@@ -start,old +start,new @@`, then every line with its marker, each newline-terminated.
/// `old` counts context and removed lines, `new` context and added lines, so the header cannot
/// disagree with the body it heads.
pub fn render(start: u32, lines: &[Line]) -> String {
    let old = lines.iter().filter(|l| l.mark != Mark::Added).count();
    let new = lines.iter().filter(|l| l.mark != Mark::Removed).count();
    let mut out = format!("@@ -{start},{old} +{start},{new} @@\n");
    for line in lines {
        out.push(line.mark.prefix());
        out.push_str(&line.text);
        out.push('\n');
    }
    out
}

/// How a changed block reads: lines added, lines removed, or the first half removed and the rest
/// added. Weighted to the corpus's marker mix (all three markers 83%, ` +` 7%, ` -` 2%).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Add,
    Remove,
    Replace,
}

pub fn draw_shape(rng: &mut ChaCha20Rng) -> Shape {
    match rng.random_range(0..100u32) {
        0..80 => Shape::Replace,
        80..94 => Shape::Add,
        _ => Shape::Remove,
    }
}

/// `texts` as context, except `texts[start..end]`, which is marked as `shape`. A `Replace` of
/// one line has no first half to remove, so it reads as an addition.
pub fn mark_block(texts: Vec<String>, start: usize, end: usize, shape: Shape) -> Vec<Line> {
    let k = end.saturating_sub(start);
    let removed_until = match shape {
        Shape::Add => start,
        Shape::Remove => end,
        Shape::Replace if k < 2 => start,
        Shape::Replace => start + k.div_ceil(2),
    };
    texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| {
            let mark = if i < start || i >= end {
                Mark::Context
            } else if i < removed_until {
                Mark::Removed
            } else {
                Mark::Added
            };
            Line { mark, text }
        })
        .collect()
}

fn digest(seed: u64, source: &str, key: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(source.as_bytes());
    hasher.update([0u8]);
    hasher.update(key.as_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    out
}

/// One item's own RNG: per item, not per run, so what one item draws does not depend on how many
/// items were read before it (the same derivation `qd-mutate generate` uses per record).
pub fn item_rng(seed: u64, source: &str, key: &str) -> ChaCha20Rng {
    ChaCha20Rng::from_seed(digest(seed, source, key))
}

/// A sort key that orders items by a keyed hash rather than by where they sat in a file, so a
/// sample is spread over the input instead of being its first `n` lines.
pub fn order_key(seed: u64, source: &str, key: &str) -> [u8; 32] {
    digest(seed.wrapping_add(1), source, key)
}

/// The first 16 hex digits of `sha256(parts joined by NUL)`. Row ids are this, so an id names
/// its content and two rows can share one only by being the same row.
pub fn short_hash(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            hasher.update([0u8]);
        }
        hasher.update(part.as_bytes());
    }
    hasher.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Whether `text` holds a codepoint in one of `ranges` (inclusive). The ranges are
/// `qd_data.render.INVISIBLE_FORMAT_RANGES`, carried in by the allowlist rather than restated
/// here: the mixture refuses such a row at render, so one reaching the corpus would only be
/// counted out again, unevenly by source.
pub fn has_char_in(text: &str, ranges: &[(u32, u32)]) -> bool {
    text.chars().any(|c| {
        let cp = u32::from(c);
        ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("line {i}")).collect()
    }

    #[test]
    fn the_header_counts_are_read_off_the_body() {
        let lines = mark_block(texts(6), 2, 5, Shape::Replace);
        let marks: String = lines.iter().map(|l| l.mark.prefix()).collect();
        assert_eq!(marks, "  --+ ");
        let hunk = render(17, &lines);
        assert!(hunk.starts_with("@@ -17,5 +17,4 @@\n"), "{hunk}");
        assert_eq!(hunk.lines().count(), 7);
        assert!(hunk.ends_with("\n"));
    }

    #[test]
    fn every_shape_leaves_the_lines_outside_the_block_as_context() {
        for shape in [Shape::Add, Shape::Remove, Shape::Replace] {
            let lines = mark_block(texts(5), 1, 3, shape);
            assert_eq!(lines[0].mark, Mark::Context);
            assert_eq!(lines[3].mark, Mark::Context);
            assert_eq!(lines[4].mark, Mark::Context);
            let header = render(1, &lines);
            let (old, new) = match shape {
                Shape::Add => (3, 5),
                Shape::Remove => (5, 3),
                Shape::Replace => (4, 4),
            };
            assert!(header.starts_with(&format!("@@ -1,{old} +1,{new} @@")), "{header}");
        }
        // One line cannot be half removed: a one-line Replace reads as an addition.
        let one = mark_block(texts(3), 1, 2, Shape::Replace);
        assert_eq!(one[1].mark, Mark::Added);
    }

    #[test]
    fn the_shape_mix_follows_the_corpus_and_is_seeded() {
        let mut rng = item_rng(0, "t", "k");
        let mut counts = [0usize; 3];
        for _ in 0..10_000 {
            counts[draw_shape(&mut rng) as usize] += 1;
        }
        // Add, Remove, Replace at 14 / 6 / 80 percent, within sampling noise.
        assert!((1_200..1_600).contains(&counts[0]), "{counts:?}");
        assert!((450..750).contains(&counts[1]), "{counts:?}");
        assert!((7_700..8_300).contains(&counts[2]), "{counts:?}");
        let a: Vec<u32> = (0..8).map(|_| item_rng(3, "s", "x").random()).collect();
        let b: Vec<u32> = (0..8).map(|_| item_rng(3, "s", "x").random()).collect();
        assert_eq!(a, b);
        assert_ne!(order_key(3, "s", "x"), order_key(3, "s", "y"));
    }

    #[test]
    fn a_format_character_is_found_and_ordinary_unicode_is_not() {
        let ranges = [(0x200B, 0x200F), (0xFEFF, 0xFEFF)];
        assert!(has_char_in("a\u{200B}b", &ranges));
        assert!(has_char_in("\u{FEFF}x", &ranges));
        assert!(!has_char_in("Beyoncé — “quoted” 1981", &ranges));
        assert_eq!(short_hash(&["a", "b"]).len(), 16);
        assert_ne!(short_hash(&["a", "b"]), short_hash(&["ab"]));
    }
}
