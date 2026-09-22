//! The input pool, and the function identity that keeps splits honest.
//!
//! A pool record is one agent-authored file: the post-image text, plus the line ranges the agent's
//! diff touched. The mutation lands **inside** one of those hunks, so the model learns to read
//! hunks and not file headers.
//!
//! `hunks` is optional and its absence is **carried, not defaulted**. A record with no hunks is a
//! whole file with no diff attached; every site in it is fair game, and every example produced from
//! it is stamped `hunk_constrained: false` and counted separately in the manifest. That is the
//! difference between a constrained sample and an unconstrained one, and presenting the second as
//! the first is exactly the "capped sample reported as complete coverage" failure.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::lang::LangId;
use crate::span::LineSpan;

/// Largest single pool record. A record is one source file plus metadata, and the parser refuses
/// anything over `parse::MAX_SOURCE_BYTES` anyway; this stops a malformed line from being buffered
/// before that refusal can happen.
pub const MAX_RECORD_BYTES: usize = 12 * 1024 * 1024;
/// Records one `read_jsonl` will take. Bounded like every other fan-out here; the CLI's `--limit`
/// is the knob a caller turns, and this is the ceiling it cannot exceed.
pub const MAX_RECORDS: usize = 5_000_000;

/// A line range the agent's diff touched: 1-based, inclusive on both ends, in `source`
/// coordinates — the same convention as [`LineSpan`], because a second convention for the same
/// thing is how an off-by-one gets in.
pub type Hunk = LineSpan;

/// `(repo, path, symbol_name, arity)`.
///
/// **Not** the diff hash. The same function reformatted has a different diff hash and would walk
/// straight through a hash-based split check; this identity does not move when the layout does.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FunctionIdentity {
    pub repo: String,
    pub path: String,
    pub symbol: String,
    pub arity: usize,
}

impl std::fmt::Display for FunctionIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}/{}", self.repo, self.path, self.symbol, self.arity)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolRecord {
    pub id: String,
    pub repo: String,
    pub path: String,
    /// Overrides the extension when the pool knows better. Absent means "infer from `path`".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<LangId>,
    /// The post-image file: what the agent's diff produced.
    pub source: String,
    /// The line ranges the agent touched. `None` is "no diff attached", which is carried through to
    /// every example rather than silently treated as "the whole file is a hunk".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hunks: Option<Vec<Hunk>>,
    /// The **pre**-image: the file as it stood before the agent's own diff.
    ///
    /// `None` for a pool walked off local sources — a file on disk has no prior version attached,
    /// which is the same reason `hunks` is optional there. Carried rather than defaulted, for the
    /// same reason as `hunks`: "there was no before-image" and "the before-image was identical"
    /// are different facts and only one of them is a clean example.
    ///
    /// It exists because `clean` needed it. `generate.rs` calls a clean example "the original
    /// agent diff, unmodified", and could not produce one: with only the post-image it emitted
    /// `before == after` and an empty diff. On the commitpackft corpus that made `diff == ""`
    /// hold for exactly the 8,450 clean rows and no others, so a model reading diffs could answer
    /// `clean` from the LENGTH of its context (`AUDIT/after-vs-diff-leak.md`). The before-image is
    /// what makes a clean example a real change to read rather than an absence to detect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_source: Option<String>,
}

impl PoolRecord {
    /// The record's language, from the explicit field or the path extension.
    pub fn language(&self) -> Option<LangId> {
        self.language.or_else(|| language_from_path(&self.path))
    }

    pub fn identity(&self, symbol: &str, arity: usize) -> FunctionIdentity {
        FunctionIdentity {
            repo: self.repo.clone(),
            path: self.path.clone(),
            symbol: symbol.to_string(),
            arity,
        }
    }
}

/// The language a path names, or `None`.
///
/// `.d.ts` is excluded on purpose: a declaration file has no function bodies, so it would parse,
/// yield nothing, and inflate the "files seen, zero examples" column with files that never could
/// have produced one.
pub fn language_from_path(path: &str) -> Option<LangId> {
    if path.ends_with(".d.ts") {
        return None;
    }
    let ext = Path::new(path).extension()?.to_str()?;
    match ext {
        "rs" => Some(LangId::Rust),
        "go" => Some(LangId::Go),
        "py" | "pyi" => Some(LangId::Python),
        "ts" | "tsx" | "mts" | "cts" => Some(LangId::TypeScript),
        "swift" => Some(LangId::Swift),
        _ => None,
    }
}

#[derive(Debug)]
pub enum PoolError {
    Io { detail: String },
    /// A malformed line stops the read and names itself. A pool this crate cannot parse in full is
    /// a pool whose record count in the manifest would be a fiction.
    Malformed { line: u64, detail: String },
    LineTooLong { line: u64, bytes: usize },
    TooManyRecords { cap: usize },
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Io { detail } => write!(f, "reading the pool: {detail}"),
            PoolError::Malformed { line, detail } => {
                write!(f, "pool line {line} is not a record: {detail}")
            }
            PoolError::LineTooLong { line, bytes } => write!(
                f,
                "pool line {line} is {bytes} bytes, over the {MAX_RECORD_BYTES}-byte cap"
            ),
            PoolError::TooManyRecords { cap } => write!(f, "pool exceeds {cap} records"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Read a JSONL pool. Blank lines are skipped; anything else that is not a record is an error.
pub fn read_jsonl(reader: impl Read) -> Result<Vec<PoolRecord>, PoolError> {
    let mut buffered = BufReader::new(reader);
    let mut out = Vec::new();
    let mut line_number = 0u64;
    let mut buf = Vec::new();
    loop {
        buf.clear();
        // `read_until` rather than `lines()`: a line over the cap must be refused by length, not
        // discovered as an allocation failure.
        let read = buffered
            .read_until(b'\n', &mut buf)
            .map_err(|e| PoolError::Io {
                detail: e.to_string(),
            })?;
        if read == 0 {
            break;
        }
        line_number += 1;
        if buf.len() > MAX_RECORD_BYTES {
            return Err(PoolError::LineTooLong {
                line: line_number,
                bytes: buf.len(),
            });
        }
        let line = String::from_utf8_lossy(&buf);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if out.len() >= MAX_RECORDS {
            return Err(PoolError::TooManyRecords { cap: MAX_RECORDS });
        }
        let record: PoolRecord = serde_json::from_str(line).map_err(|e| PoolError::Malformed {
            line: line_number,
            detail: e.to_string(),
        })?;
        out.push(record);
    }
    Ok(out)
}

/// Count records by language, for the manifest's "what was in the pool" column.
pub fn language_census(records: &[PoolRecord]) -> BTreeMap<String, u64> {
    let mut out: BTreeMap<String, u64> = BTreeMap::new();
    for record in records {
        let key = record
            .language()
            .map(|l| l.as_str().to_string())
            .unwrap_or_else(|| "unrecognised".to_string());
        *out.entry(key).or_insert(0) += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_names_its_language_and_a_declaration_file_names_none() {
        assert_eq!(language_from_path("a/b.rs"), Some(LangId::Rust));
        assert_eq!(language_from_path("a/b.go"), Some(LangId::Go));
        assert_eq!(language_from_path("a/b.py"), Some(LangId::Python));
        assert_eq!(language_from_path("a/b.tsx"), Some(LangId::TypeScript));
        assert_eq!(language_from_path("a/b.swift"), Some(LangId::Swift));
        assert_eq!(language_from_path("a/b.d.ts"), None, "no bodies to mutate");
        assert_eq!(language_from_path("Makefile"), None);
    }

    #[test]
    fn an_absent_hunk_list_stays_absent_rather_than_becoming_the_whole_file() {
        let json = r#"{"id":"r1","repo":"o/r","path":"a.rs","source":"fn f() {}\n"}"#;
        let records = read_jsonl(json.as_bytes()).expect("parses");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].hunks, None);
    }

    #[test]
    fn hunks_round_trip_in_the_span_convention() {
        // One physical line: the reader is line-delimited, so a pretty-printed record is two
        // truncated ones.
        let json = r#"{"id":"r1","repo":"o/r","path":"a.rs","source":"fn f() {}\n","hunks":[{"start_line":2,"end_line":4}]}"#;
        let records = read_jsonl(json.as_bytes()).expect("parses");
        assert_eq!(records[0].hunks.as_deref(), Some(&[LineSpan::new(2, 4)][..]));
    }

    #[test]
    fn a_malformed_line_names_its_line_number_rather_than_being_skipped() {
        let jsonl = "{\"id\":\"a\",\"repo\":\"o/r\",\"path\":\"a.rs\",\"source\":\"x\"}\nnot json\n";
        match read_jsonl(jsonl.as_bytes()) {
            Err(PoolError::Malformed { line, .. }) => assert_eq!(line, 2),
            other => panic!("expected a malformed-line error, got {other:?}"),
        }
    }

    #[test]
    fn blank_lines_are_skipped_and_do_not_shift_the_line_number() {
        let jsonl = "\n\n{\"id\":\"a\",\"repo\":\"o/r\",\"path\":\"a.rs\",\"source\":\"x\"}\n\nbad\n";
        match read_jsonl(jsonl.as_bytes()) {
            Err(PoolError::Malformed { line, .. }) => assert_eq!(line, 5),
            other => panic!("expected a malformed-line error, got {other:?}"),
        }
    }

    #[test]
    fn an_explicit_language_overrides_the_extension() {
        let json = r#"{"id":"r1","repo":"o/r","path":"a.txt","language":"rust","source":"fn f(){}"}"#;
        let records = read_jsonl(json.as_bytes()).expect("parses");
        assert_eq!(records[0].language(), Some(LangId::Rust));
    }

    #[test]
    fn identity_is_not_the_diff_hash() {
        let record = PoolRecord {
            id: "r".to_string(),
            repo: "o/r".to_string(),
            path: "a.rs".to_string(),
            language: None,
            source: String::new(),
            hunks: None,
            prior_source: None,
        };
        assert_eq!(record.identity("f", 2), record.identity("f", 2));
        assert_ne!(record.identity("f", 2), record.identity("f", 3));
    }
}
