//! Admission: a request whose context is not the shape its task was trained on is refused before
//! the model is asked.
//!
//! # What is checked, for which task
//!
//! Only [`DEFECT_CLASS_TASK`]. Its rows were built in one shape (`python/qd_data/mixture.py`,
//! the `code.defect_class` rewriter, `header = f"file: {raw.path}\n\n"` then the diff): a
//! `file: <path>` line, an empty line, then one or more unified-diff hunks. The check refuses,
//! each with a typed [`Refusal`] carrying the line, what was expected and what was found:
//!
//! * no `file: ` header, a path that is not UTF-8 or is blank, or no empty line after it;
//! * the path's language is not one the pool held ([`qd_lang::DEFECT_CLASS_POOL_LANGUAGES`],
//!   read through the pool's own [`qd_lang::language_from_path`]) — Swift included, which the
//!   pool reader maps but the built pool holds none of;
//! * anything before the first hunk header (a `---`/`+++` preamble too: the trained contexts
//!   carry the path in `file:` and never show one), or no hunk at all;
//! * a hunk header that is not `@@ -a[,b] +c[,d] @@` with an optional ` section` after it;
//! * a hunk line whose first byte is not ` `, `-`, `+` or `\` (an empty line included: the
//!   trained diffs mark an empty context line with a single space);
//! * a hunk whose body has more or fewer old (` ` and `-`) or new (` ` and `+`) lines than its
//!   header declares.
//!
//! A single trailing `\n` terminates the last line and is accepted: the trained contexts drop it,
//! a caller's diff usually has it, and it opens no line.
//!
//! What this does **not** do: judge content. A diff that parses is admitted whatever is in it; a
//! scrambled diff whose hunks still parse reaches the model, which is the model's call. The OOD
//! gate suite's scoring path (`python/qd_train/ood.py`, `tools/real_ft_run.py --ood`) does not go
//! through this runtime and is unchanged.
//!
//! Admission never answers and never emits `allow`: it only refuses (CLAUDE.md rule 7).

use qd_lang::{DEFECT_CLASS_POOL_LANGUAGES, language_from_path};

use crate::context::Context;
use crate::refusal::Refusal;
use crate::schema::DecisionRequest;

/// The task whose contexts are held to the trained diff shape. The family id of
/// `python/qd_data/defect_class.py::DEFECT_FAMILY_ID`; a request's `task` is its family id
/// (`python/qd_data/mixture.py::_request`, `task=family_id`).
pub const DEFECT_CLASS_TASK: &str = "code.defect_class";

/// The first line's prefix.
const FILE_HEADER: &[u8] = b"file: ";

/// Bytes of caller content a refusal echoes back. The context can be megabytes; a refusal that
/// carried a whole line of it would be a second copy of the payload.
const EXCERPT_BYTES: usize = 80;
/// Bytes of the path a refusal echoes back.
const PATH_EXCERPT_BYTES: usize = 256;

/// Refuse `request` if its task holds its context to a trained shape and the context is not in it.
pub fn admit(request: &DecisionRequest) -> Result<(), Refusal> {
    if request.task == DEFECT_CLASS_TASK {
        admit_defect_context(&request.context)
    } else {
        Ok(())
    }
}

/// `bytes` as a short, escaped, lossless-where-possible excerpt for a refusal.
fn excerpt(bytes: &[u8], cap: usize) -> String {
    let (head, cut) = if bytes.len() > cap {
        (&bytes[..cap], true)
    } else {
        (bytes, false)
    };
    let text = String::from_utf8_lossy(head);
    if cut {
        format!("{text:?} (first {cap} of {} bytes)", bytes.len())
    } else {
        format!("{text:?}")
    }
}

fn not_diff(line: usize, expected: impl Into<String>, found: &[u8]) -> Refusal {
    Refusal::ContextNotUnifiedDiff {
        line,
        expected: expected.into(),
        found: excerpt(found, EXCERPT_BYTES),
    }
}

fn not_diff_end(line: usize, expected: impl Into<String>) -> Refusal {
    Refusal::ContextNotUnifiedDiff {
        line,
        expected: expected.into(),
        found: "the end of the context".to_string(),
    }
}

/// One open hunk: where its header was, what it declared, and how much of that is still owed.
struct Hunk {
    header_line: usize,
    old_declared: u64,
    new_declared: u64,
    old_left: u64,
    new_left: u64,
}

impl Hunk {
    /// Refuse a hunk whose body fell short of its header, naming the header's line.
    fn close(&self) -> Result<(), Refusal> {
        if self.old_left == 0 && self.new_left == 0 {
            return Ok(());
        }
        Err(Refusal::ContextNotUnifiedDiff {
            line: self.header_line,
            expected: format!(
                "a hunk body of {} old and {} new lines, as its header declares",
                self.old_declared, self.new_declared
            ),
            found: format!(
                "{} old and {} new lines",
                self.old_declared - self.old_left,
                self.new_declared - self.new_left
            ),
        })
    }
}

/// `digits[,digits]` at the front of `s`: the start and the count (absent = 1). Returns the rest.
fn range(s: &[u8]) -> Option<(u64, &[u8])> {
    fn number(s: &[u8]) -> Option<(u64, &[u8])> {
        let end = s
            .iter()
            .position(|b| !b.is_ascii_digit())
            .unwrap_or(s.len());
        if end == 0 {
            return None;
        }
        let digits = std::str::from_utf8(&s[..end]).ok()?;
        Some((digits.parse::<u64>().ok()?, &s[end..]))
    }
    let (_start, rest) = number(s)?;
    match rest.strip_prefix(b",") {
        Some(after) => number(after),
        None => Some((1, rest)),
    }
}

/// `@@ -a[,b] +c[,d] @@[ section]` -> `(b, d)`.
fn hunk_counts(line: &[u8]) -> Option<(u64, u64)> {
    let rest = line.strip_prefix(b"@@ -")?;
    let (old, rest) = range(rest)?;
    let rest = rest.strip_prefix(b" +")?;
    let (new, rest) = range(rest)?;
    let rest = rest.strip_prefix(b" @@")?;
    if rest.is_empty() || rest.first() == Some(&b' ') {
        Some((old, new))
    } else {
        None
    }
}

/// The trained `code.defect_class` context shape. See the module docs for every rule.
pub fn admit_defect_context(context: &Context) -> Result<(), Refusal> {
    let bytes = context.as_bytes();
    let body = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    let mut lines = body.split(|b| *b == b'\n').zip(1usize..);

    // Line 1: `file: <path>`.
    let first = lines.next().map_or(&[][..], |(line, _)| line);
    let path = first
        .strip_prefix(FILE_HEADER)
        .ok_or_else(|| not_diff(1, "a `file: <path>` header line", first))?;
    let path =
        std::str::from_utf8(path).map_err(|_| not_diff(1, "a UTF-8 path after `file: `", first))?;
    if crate::is_blank(path) {
        return Err(not_diff(1, "a path after `file: `", first));
    }

    // Line 2: empty.
    match lines.next() {
        Some((b"", _)) => {}
        Some((line, n)) => {
            return Err(not_diff(n, "an empty line after the `file:` header", line));
        }
        None => return Err(not_diff_end(2, "an empty line after the `file:` header")),
    }

    // The path's language, read through the pool's own map.
    let language = language_from_path(path);
    if !language.is_some_and(|lang| DEFECT_CLASS_POOL_LANGUAGES.contains(&lang)) {
        return Err(Refusal::ContextLanguageNotInPool {
            path: if path.len() > PATH_EXCERPT_BYTES {
                excerpt(path.as_bytes(), PATH_EXCERPT_BYTES)
            } else {
                path.to_string()
            },
            language: language
                .map_or("unrecognised", |lang| lang.as_str())
                .to_string(),
            pool: DEFECT_CLASS_POOL_LANGUAGES
                .iter()
                .map(|lang| lang.as_str().to_string())
                .collect(),
        });
    }

    // Line 3 on: hunks.
    let mut open: Option<Hunk> = None;
    let mut last = 2usize;
    for (line, n) in lines {
        last = n;
        if line.starts_with(b"@@ ") {
            if let Some(hunk) = open.take() {
                hunk.close()?;
            }
            let (old, new) = hunk_counts(line)
                .ok_or_else(|| not_diff(n, "a hunk header `@@ -a[,b] +c[,d] @@`", line))?;
            open = Some(Hunk {
                header_line: n,
                old_declared: old,
                new_declared: new,
                old_left: old,
                new_left: new,
            });
            continue;
        }
        let Some(hunk) = open.as_mut() else {
            return Err(not_diff(n, "a hunk header `@@ -a[,b] +c[,d] @@`", line));
        };
        let (old, new) = match line.first() {
            Some(b' ') => (1, 1),
            Some(b'-') => (1, 0),
            Some(b'+') => (0, 1),
            // `\ No newline at end of file`: about the line before it, counted by neither side.
            Some(b'\\') => (0, 0),
            _ => {
                return Err(not_diff(
                    n,
                    "a hunk line starting with ` `, `-`, `+` or `\\`",
                    line,
                ));
            }
        };
        if old > hunk.old_left || new > hunk.new_left {
            return Err(not_diff(
                n,
                format!(
                    "a hunk header or the end: the hunk at line {} declared {} old and {} new \
                     lines and has them all",
                    hunk.header_line, hunk.old_declared, hunk.new_declared
                ),
                line,
            ));
        }
        hunk.old_left -= old;
        hunk.new_left -= new;
    }
    match open {
        Some(hunk) => hunk.close(),
        None => Err(not_diff_end(
            last + 1,
            "a hunk header `@@ -a[,b] +c[,d] @@`",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admit_bytes(bytes: &[u8]) -> Result<(), Refusal> {
        admit_defect_context(&Context::from_bytes(bytes.to_vec()))
    }

    #[test]
    fn hunk_headers_parse_with_and_without_counts_and_with_a_section() {
        assert_eq!(hunk_counts(b"@@ -1,4 +1,4 @@"), Some((4, 4)));
        assert_eq!(hunk_counts(b"@@ -9 +9 @@"), Some((1, 1)));
        assert_eq!(hunk_counts(b"@@ -0,0 +1,3 @@"), Some((0, 3)));
        assert_eq!(hunk_counts(b"@@ -3,2 +3,2 @@ func Sum() {"), Some((2, 2)));
        for bad in [
            &b"@@ -1,4 +1,4"[..],
            b"@@ -1,4 +1,4 @@x",
            b"@@ -,4 +1,4 @@",
            b"@@ -1, +1,4 @@",
            b"@@ +1,4 -1,4 @@",
            b"@@ -1,4 +1,4 @@\r",
            b"@@ -99999999999999999999,1 +1,1 @@",
        ] {
            assert_eq!(hunk_counts(bad), None, "{}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn a_non_utf8_path_is_refused_and_a_non_utf8_body_line_is_left_to_the_renderer() {
        let refusal = admit_bytes(b"file: a\xff.rs\n\n@@ -1 +1 @@\n-x\n+y").unwrap_err();
        assert_eq!(refusal.kind(), "context_not_unified_diff");
        // The body's bytes are the renderer's to judge (`context_not_utf8`); admission reads
        // only the markers.
        admit_bytes(b"file: a.rs\n\n@@ -1 +1 @@\n-\xff\n+y").expect("markers are fine");
    }

    #[test]
    fn a_refusal_echoes_a_bounded_excerpt_not_the_payload() {
        let mut long = b"file: a.rs\n\n".to_vec();
        long.extend(std::iter::repeat_n(b'x', 10_000));
        match admit_bytes(&long).unwrap_err() {
            Refusal::ContextNotUnifiedDiff { line, found, .. } => {
                assert_eq!(line, 3);
                assert!(found.len() < 200, "{} bytes echoed", found.len());
                assert!(found.contains("of 10000 bytes"), "{found}");
            }
            other => panic!("expected context_not_unified_diff, got {other:?}"),
        }
        let path = "d/".repeat(1_000) + "x.c";
        match admit_bytes(format!("file: {path}\n\n@@ -1 +1 @@\n-x\n+y").as_bytes()).unwrap_err() {
            Refusal::ContextLanguageNotInPool { path: echoed, .. } => {
                assert!(echoed.len() < 400, "{} bytes echoed", echoed.len());
            }
            other => panic!("expected context_language_not_in_pool, got {other:?}"),
        }
    }

    #[test]
    fn the_empty_context_is_refused_at_line_one() {
        match admit_bytes(b"").unwrap_err() {
            Refusal::ContextNotUnifiedDiff { line, .. } => assert_eq!(line, 1),
            other => panic!("expected context_not_unified_diff, got {other:?}"),
        }
    }
}
