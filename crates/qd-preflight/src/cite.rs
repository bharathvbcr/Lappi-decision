//! `qd-cite`: a measured number in a new handoff or audit must say where it was measured.
//!
//! The v0.1 notes quoted headline figures that no row backed (`GAP-PREVIEW-HEADLINE-NUMBERS-
//! INCLUDE-UNSERVED-DEFECT-ROWS-2026-10-06`). From 2026-10-18 every line of a `HANDOFF/*.md` or
//! `AUDIT/**/*.md` file that contains a measured number must cite, in the same paragraph, list
//! item or table row, one of:
//!
//! * a ledger row id: 8 or more hex digits (or a UUID) that is a prefix of a `row_id` in
//!   `ledger/*.jsonl`;
//! * a gap id (`GAP-...`) that is a record in `gaps.jsonl` (same resolution rule as
//!   `python/tests/test_gaps_ledger.py::citation_resolves`);
//! * an evidence tag `[V: path]` naming a path that exists under the repository root.
//!
//! An id that does not resolve fails wherever it appears in an in-scope file: a citation that
//! points at nothing is worse than none. A file's date is the last `YYYY-MM-DD` in its file name,
//! else in its nearest dated parent directory; undated files and files before the cutoff are not
//! examined (historical files are not flagged).
//!
//! A measured number is a decimal (`0.857`), a percentage (`4.6%`) or an `n/m` count
//! (`14779/17254`). Not measured numbers: dates and times, versions (`1.2.3`, `v0.1`), numbers
//! fused to an identifier (`Qwen3.5`, `H100`, `choice-16`), line references (`file.rs:12`,
//! `line 5`), the inside of a link target, a fenced code block, and the `[V: ...]` tag itself.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

pub const DEFAULT_SINCE: &str = "2026-10-18";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reason {
    /// A measured number with no citation in its unit.
    Uncited(String),
    /// A citation that does not resolve.
    Unresolvable(String),
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub file: PathBuf,
    pub line: usize,
    pub reason: Reason,
    pub text: String,
}

/// What the ids and tags resolve against.
pub struct Known {
    pub ledger_rows: Vec<String>,
    pub gap_ids: BTreeSet<String>,
    pub root: PathBuf,
}

impl Known {
    pub fn load(root: &Path) -> Result<Self> {
        let mut rows = Vec::new();
        let dir = root.join("ledger");
        for e in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let p = e?.path();
            if p.extension().is_some_and(|x| x == "jsonl") {
                let text = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
                for line in text.lines() {
                    if let Some(id) = json_str_field(line, "row_id") {
                        rows.push(id);
                    }
                }
            }
        }
        if rows.is_empty() {
            bail!("{} holds no row_id: no ledger row id could resolve", dir.display());
        }
        let gp = root.join("gaps.jsonl");
        let mut gaps = BTreeSet::new();
        for line in std::fs::read_to_string(&gp).with_context(|| format!("reading {}", gp.display()))?.lines() {
            if let Some(id) = json_str_field(line, "id") {
                gaps.insert(id);
            }
        }
        if gaps.is_empty() {
            bail!("{} holds no gap id", gp.display());
        }
        Ok(Self { ledger_rows: rows, gap_ids: gaps, root: root.to_path_buf() })
    }

    fn row_resolves(&self, token: &str) -> bool {
        self.ledger_rows.iter().any(|r| r.starts_with(token))
    }

    fn gap_resolves(&self, token: &str) -> bool {
        self.gap_ids.contains(token) || (token.ends_with('-') && self.gap_ids.iter().any(|k| k.starts_with(token)))
    }
}

/// The string value of a top-level key in one JSON line, without a full parse of rows that can be
/// megabytes (a ledger row carries its whole protocol).
fn json_str_field(line: &str, key: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    v.get(key)?.as_str().map(str::to_string)
}

// ---------------------------------------------------------------------------------------------
// Dates

fn is_date_at(b: &[u8], i: usize) -> bool {
    b.len() >= i + 10
        && b[i..i + 4].iter().all(u8::is_ascii_digit)
        && b[i + 4] == b'-'
        && b[i + 5..i + 7].iter().all(u8::is_ascii_digit)
        && b[i + 7] == b'-'
        && b[i + 8..i + 10].iter().all(u8::is_ascii_digit)
}

fn last_date(s: &str) -> Option<String> {
    let b = s.as_bytes();
    (0..b.len()).rev().find(|&i| is_date_at(b, i)).map(|i| s[i..i + 10].to_string())
}

/// The date a file is held to: its own name's, else its nearest dated parent directory's.
pub fn file_date(path: &Path) -> Option<String> {
    if let Some(d) = path.file_name().and_then(|n| n.to_str()).and_then(last_date) {
        return Some(d);
    }
    path.parent()?.ancestors().find_map(|a| a.file_name().and_then(|n| n.to_str()).and_then(last_date))
}

// ---------------------------------------------------------------------------------------------
// Tokens

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Does `line` contain a measured number? Returns the first one, for the report.
pub fn measured_number(line: &str) -> Option<String> {
    let line = strip_non_prose(line);
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // The whole run of digits, dots, slashes, colons, hyphens and percent signs.
        let start = i;
        let mut j = i;
        while j < b.len() && (b[j].is_ascii_digit() || matches!(b[j], b'.' | b'/' | b':' | b'-' | b'%' | b',')) {
            j += 1;
        }
        // A trailing separator is punctuation, not part of the number.
        let mut end = j;
        while end > start && !(b[end - 1].is_ascii_digit() || b[end - 1] == b'%') {
            end -= 1;
        }
        let run = &line[start..end];
        let before = if start == 0 { b' ' } else { b[start - 1] };
        let after = b.get(end).copied().unwrap_or(b' ');
        i = j.max(i + 1);
        // Fused to an identifier: Qwen3.5, H100, choice-16, v0.1, file_2.5.
        if is_ident(before) && !before.is_ascii_digit() || before == b'-' && start >= 2 && is_ident(b[start - 2]) && !b[start - 2].is_ascii_digit() || before == b'/' {
            continue;
        }
        if is_ident(after) && !after.is_ascii_digit() {
            continue;
        }
        if after == b':' && b.get(end + 1).is_some_and(u8::is_ascii_digit) {
            continue; // 02:25
        }
        if before == b':' || before == b'#' {
            continue; // file.rs:12, #123
        }
        if is_date_or_time(run) || is_reference(&line[..start], &line[end..], before, run) {
            continue;
        }
        // A range such as 0.0056-0.0379 is two numbers, each judged alone.
        if let Some(m) = run.split('-').filter(|p| !is_version(p)).find_map(classify) {
            return Some(m);
        }
    }
    None
}

/// A number that names something rather than measures it: a section or audit item (`section 3.1`,
/// a heading's own `### 4.4`), a paper id (`arXiv 2305.13282`), a price (`$2.29`).
fn is_reference(prefix: &str, suffix: &str, before: u8, run: &str) -> bool {
    let one_digit_pair = run.len() == 3 && run.as_bytes()[1] == b'.' && run.bytes().filter(u8::is_ascii_digit).count() == 2;
    if one_digit_pair && ((prefix.ends_with('(') && suffix.starts_with(')')) || prefix.ends_with(" in ") || prefix.to_ascii_lowercase().ends_with("see ")) {
        return true;
    }
    if prefix.trim_end().ends_with(['~', '≈', '≥', '≤', '±']) {
        return true; // an estimate or a bound, not a measurement
    }
    if before == b'$' {
        return true;
    }
    let lead = prefix.trim_start_matches('#').trim();
    if prefix.trim_start().starts_with('#') && lead.is_empty() {
        return true;
    }
    let lower = prefix.trim_end().to_ascii_lowercase();
    const NAMES: [&str; 17] = [
        "section", "§", "audit", "item", "step", "phase", "part", "arxiv", "doi:", "figure", "appendix", "table",
        "threshold", "target", "target-precision", "at least", "at most",
    ];
    if NAMES.iter().any(|w| lower.ends_with(w)) {
        return true;
    }
    matches!(run.split_once('.'), Some((a, b)) if a.len() == 4 && (4..=5).contains(&b.len()) && !a.contains(','))
        && run.bytes().all(|c| c.is_ascii_digit() || c == b'.')
}

fn is_date_or_time(run: &str) -> bool {
    run.contains(':') || (run.contains('-') && run.bytes().all(|c| c.is_ascii_digit() || c == b'-'))
}

fn is_version(run: &str) -> bool {
    run.bytes().filter(|c| *c == b'.').count() >= 2 && !run.contains('%') && !run.contains('/')
}

/// A decimal, percentage or n/m count inside `run`.
fn classify(run: &str) -> Option<String> {
    let run = run.trim_matches(|c: char| !c.is_ascii_digit() && c != '%');
    if run.is_empty() {
        return None;
    }
    if run.ends_with('%') && run[..run.len() - 1].bytes().all(|c| c.is_ascii_digit() || c == b'.') {
        return Some(run.to_string());
    }
    if let Some((n, m)) = run.split_once('/') {
        let ok = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit() || c == b',');
        if ok(n) && ok(m) && !m.contains('/') {
            return Some(run.to_string());
        }
        return None;
    }
    let mut parts = run.split('.');
    let (a, b) = (parts.next()?, parts.next()?);
    if parts.next().is_none() && !a.is_empty() && !b.is_empty() && a.bytes().all(|c| c.is_ascii_digit() || c == b',') && b.bytes().all(|c| c.is_ascii_digit()) {
        return Some(run.to_string());
    }
    None
}

/// Remove what is not a claim: link targets, `[V: ...]` tags, and ids that may hold digits.
fn strip_non_prose(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        // [V: ...]
        if line[i..].starts_with("[V:")
            && let Some(end) = line[i..].find(']')
        {
            out.push(' ');
            i += end + 1;
            continue;
        }
        // ](target)
        if line[i..].starts_with("](")
            && let Some(end) = line[i..].find(')')
        {
            out.push_str("] ");
            i += end + 1;
            continue;
        }
        // GAP-... ids
        if line[i..].starts_with("GAP-") {
            let mut j = i + 4;
            while j < b.len() && (b[j].is_ascii_uppercase() || b[j].is_ascii_digit() || b[j] == b'-') {
                j += 1;
            }
            out.push(' ');
            i = j;
            continue;
        }
        let ch = line[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// 8+ hex digits as a token (a UUID's hyphens kept), never 32+ undashed (a hash, not a row id).
fn hex_tokens(line: &str) -> Vec<String> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_hexdigit() || (i > 0 && is_ident(b[i - 1])) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < b.len() && (b[j].is_ascii_hexdigit() || b[j] == b'-') {
            j += 1;
        }
        // Digits-and-hyphens runs that end in a hyphen lose it.
        let mut end = j;
        while end > i && b[end - 1] == b'-' {
            end -= 1;
        }
        let tok = &line[i..end];
        let hex_only: String = tok.chars().filter(|c| *c != '-').collect();
        let next = b.get(end).copied().unwrap_or(b' ');
        let lower = tok.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c) || c == b'-');
        let has_letter = tok.bytes().any(|c| (b'a'..=b'f').contains(&c));
        let has_digit = tok.bytes().any(|c| c.is_ascii_digit());
        let uuid_like = tok.matches('-').count() == 4 && tok.len() == 36;
        let plain = !tok.contains('-');
        if lower
            && !is_ident(next)
            && has_letter
            && has_digit
            && hex_only.len() >= 8
            && ((uuid_like) || (plain && tok.len() < 32))
        {
            out.push(tok.to_string());
        }
        i = j.max(i + 1);
    }
    out
}

/// An unresolved hex token is a claim about the ledger only when the words just before it say row or
/// ledger and the token is not introduced as a hash, commit or snapshot: source-control shas, HF snapshot ids and
/// file digests are hex too, and flagging them is a false positive.
fn is_ledger_claim(unit: &str, tok: &str) -> bool {
    let lower = unit.to_ascii_lowercase();
    let Some(at) = unit.find(tok) else { return false };
    // `6cf99185…`: a truncated digest, not an id anyone could look up.
    let after = &unit[at + tok.len()..];
    if after.starts_with('…') || after.starts_with("...") {
        return false;
    }
    let mut from = at.saturating_sub(40);
    while !lower.is_char_boundary(from) {
        from += 1;
    }
    let before = &lower[from..at];
    let hashy = ["sha", "commit", "snapshot", "hash", "git", "blob", "digest", "@"];
    !hashy.iter().any(|w| before.contains(w)) && (before.contains("row") || before.contains("ledger"))
}

fn gap_tokens(line: &str) -> Vec<String> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(p) = line[i..].find("GAP-") {
        let s = i + p;
        let mut j = s + 4;
        while j < b.len() && (b[j].is_ascii_uppercase() || b[j].is_ascii_digit() || b[j] == b'-') {
            j += 1;
        }
        let tok = &line[s..j];
        // GAP-<X><X>... needs the grammar's 4 trailing characters.
        if tok.len() >= 8 {
            out.push(tok.to_string());
        }
        i = j.max(s + 4);
    }
    out
}

fn v_tags(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(p) = rest.find("[V:") {
        let after = &rest[p + 3..];
        let Some(end) = after.find(']') else { break };
        for path in after[..end].split([',', ';']) {
            let path = path.trim().trim_matches('`');
            if !path.is_empty() {
                out.push(path.to_string());
            }
        }
        rest = &after[end + 1..];
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Units

/// A unit is the span a citation covers: a table row, a list item with its continuation lines, or
/// a paragraph. Returns (first line number, lines).
fn units(text: &str) -> Vec<(usize, Vec<&str>)> {
    let mut out: Vec<(usize, Vec<&str>)> = Vec::new();
    let mut cur: Option<(usize, Vec<&str>)> = None;
    let mut fenced = false;
    fn flush<'a>(cur: &mut Option<(usize, Vec<&'a str>)>, out: &mut Vec<(usize, Vec<&'a str>)>) {
        if let Some(u) = cur.take() {
            out.push(u);
        }
    }
    for (i, line) in text.lines().enumerate() {
        let n = i + 1;
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            flush(&mut cur, &mut out);
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        if t.is_empty() {
            flush(&mut cur, &mut out);
            continue;
        }
        let new_unit = t.starts_with('|')
            || t.starts_with('#')
            || t.starts_with("- ")
            || t.starts_with("* ")
            || t.starts_with("+ ")
            || t.split_once(". ").is_some_and(|(a, _)| !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()));
        if new_unit {
            flush(&mut cur, &mut out);
            cur = Some((n, vec![line]));
            if t.starts_with('|') || t.starts_with('#') {
                flush(&mut cur, &mut out);
            }
        } else if let Some(u) = cur.as_mut() {
            u.1.push(line);
        } else {
            cur = Some((n, vec![line]));
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Check one file's text.
pub fn check_text(file: &Path, text: &str, known: &Known) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (first, lines) in units(text) {
        let joined = lines.join(" ");
        let mut resolved = 0usize;
        let mut bad: Vec<String> = Vec::new();
        for tok in hex_tokens(&joined) {
            if known.row_resolves(&tok) {
                resolved += 1;
            } else if is_ledger_claim(&joined, &tok) {
                bad.push(format!("row id {tok} resolves to no ledger row"));
            }
        }
        for tok in gap_tokens(&joined) {
            if known.gap_resolves(&tok) {
                resolved += 1;
            } else {
                bad.push(format!("gap id {tok} is not in gaps.jsonl"));
            }
        }
        for path in v_tags(&joined) {
            let p = Path::new(&path);
            let full = if p.is_absolute() { p.to_path_buf() } else { known.root.join(p) };
            if full.exists() {
                resolved += 1;
            } else {
                bad.push(format!("[V: {path}] does not exist"));
            }
        }
        for why in bad {
            findings.push(Finding { file: file.to_path_buf(), line: first, reason: Reason::Unresolvable(why), text: lines[0].trim().chars().take(160).collect() });
        }
        if resolved == 0 {
            for (k, line) in lines.iter().enumerate() {
                if let Some(num) = measured_number(line) {
                    findings.push(Finding {
                        file: file.to_path_buf(),
                        line: first + k,
                        reason: Reason::Uncited(num),
                        text: line.trim().chars().take(160).collect(),
                    });
                }
            }
        }
    }
    findings
}

/// Every `HANDOFF/*.md` and `AUDIT/**/*.md` under `root`.
pub fn scope_files(root: &Path) -> Result<Vec<PathBuf>> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        let rd = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
        };
        for e in rd {
            let p = e?.path();
            if p.is_dir() {
                walk(&p, out)?;
            } else if p.extension().is_some_and(|x| x == "md") {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(&root.join("HANDOFF"), &mut out)?;
    walk(&root.join("AUDIT"), &mut out)?;
    out.sort();
    Ok(out)
}

#[derive(Debug)]
pub struct Report {
    pub in_scope: usize,
    pub skipped_undated: usize,
    pub skipped_before_cutoff: usize,
    pub findings: Vec<Finding>,
}

pub fn run(root: &Path, files: &[PathBuf], since: &str) -> Result<Report> {
    let known = Known::load(root)?;
    let mut report = Report { in_scope: 0, skipped_undated: 0, skipped_before_cutoff: 0, findings: Vec::new() };
    for f in files {
        match file_date(f) {
            None => report.skipped_undated += 1,
            Some(d) if d.as_str() < since => report.skipped_before_cutoff += 1,
            Some(_) => {
                let text = std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?;
                report.in_scope += 1;
                report.findings.extend(check_text(f, &text, &known));
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(s: &str) -> Option<String> {
        measured_number(s)
    }

    #[test]
    fn measured_numbers_are_decimals_percentages_and_counts() {
        assert_eq!(num("val choice top-1 **0.857** here").as_deref(), Some("0.857"));
        assert_eq!(num("abstention is 4.6% overall").as_deref(), Some("4.6%"));
        assert_eq!(num("149/180 correct").as_deref(), Some("149/180"));
        assert_eq!(num("14,779/17,254 rows").as_deref(), Some("14,779/17,254"));
    }

    #[test]
    fn dates_versions_shas_lines_and_identifiers_are_not() {
        for s in [
            "written 2026-10-06 at 02:25Z",
            "torch 2.12.1 and cargo 1.98.0",
            "Qwen3.5-2B-Base snapshot b1485b2f",
            "see src/main.rs:123 and lines 5-8",
            "an H100 box, 8xH100, choice-16, v0.1 preview",
            concat!("the [docs](docs/schema-api.md) and GAP", "-FOO-BAR-2026-10-06"),
            "[V: AUDIT/x-2026-10-18/0.5/notes.md] only",
            "23 families, 15 entries",
        ] {
            assert_eq!(num(s), None, "{s}");
        }
    }

    #[test]
    fn file_dates_come_from_the_name_then_the_directory() {
        assert_eq!(file_date(Path::new("HANDOFF/x-2026-10-18.md")).as_deref(), Some("2026-10-18"));
        assert_eq!(file_date(Path::new("AUDIT/harden-2026-10-19/x.md")).as_deref(), Some("2026-10-19"));
        assert_eq!(file_date(Path::new("AUDIT/x.md")), None);
    }

    #[test]
    fn uuid_and_prefix_tokens_are_found_and_hashes_are_not() {
        assert_eq!(hex_tokens("row 6af73bef-aa56-49b3-8cdb-e0b0b56e9181 done"), vec!["6af73bef-aa56-49b3-8cdb-e0b0b56e9181"]);
        assert_eq!(hex_tokens("rows 9a00cfd0, b0d9454e"), vec!["9a00cfd0", "b0d9454e"]);
        assert!(hex_tokens(&format!("hash {}", "ab12".repeat(16))).is_empty());
        assert!(hex_tokens("the word deadbeef alone and 12345678").is_empty());
    }
}
