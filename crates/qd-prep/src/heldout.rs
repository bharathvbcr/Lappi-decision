//! Rule 3's path marker, with one owner in this crate. A held-out set lives under a path segment
//! named for it (`crates/qd-train/src/held_out.rs::DEFAULT_HELD_OUT_PATH_MARKERS`, the trainer's
//! door). An evaluation set's output must carry one ([`check_held_out_path`]), and a producer of
//! training rows refuses to read an input that carries one ([`refuse_training_input`]).
//!
//! Decontamination targets are not training inputs. A producer scans its rows against held-out
//! sets so that none of them leaks into training, so `--target` files may sit under a marker.
//!
//! A path is not the only guard. A copy of an app's caller-record file anywhere else carries no
//! marker, so [`refuse_caller_records`] reads the content: any line whose parsed top-level
//! `"record"` starts with `lappi.caller_record` refuses the whole file, wherever it lives;
//! compressed or UTF-16 bytes, a bare `\r` or any other `str.splitlines()` separator a reader could
//! honour (see [`BreakScan`]), a non-regular file and a file over the byte budget
//! are refused as unscannable, never passed. Text that merely quotes the format is clean. The rule is
//! shared with `crates/qd-train/src/held_out.rs::assert_no_caller_records` and
//! `python/qd_data/caller_records.py`: all three run the oracle
//! `fixtures/caller/guard-cases.json` (this crate cannot depend on `qd-train`).
//!
//! Segments match ASCII case-insensitively, as `natural-bugs` always has. The trainer's door also
//! folds the few non-ASCII code points whose case folding is ASCII; this crate does not depend on
//! `qd-train` (it would pull the serving stack into a data tool), so that door stays the last
//! check before any training read.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Component, Path};

/// `crates/qd-train/src/held_out.rs::DEFAULT_HELD_OUT_PATH_MARKERS`.
pub const HELD_OUT_PATH_MARKERS: [&str; 3] = ["heldout", "held_out", "held-out"];

/// The first segment of `path` that equals a held-out marker, ASCII case-insensitively.
fn marker_segment(path: &Path) -> Option<String> {
    path.components().find_map(|c| match c {
        Component::Normal(seg) => {
            let s = seg.to_string_lossy();
            HELD_OUT_PATH_MARKERS
                .iter()
                .any(|m| s.eq_ignore_ascii_case(m))
                .then(|| s.into_owned())
        }
        _ => None,
    })
}

/// Refuse an output path that `qd-train`'s path check would not refuse: no `..`, and some
/// segment equal (ASCII case-insensitively) to a held-out marker.
pub fn check_held_out_path(path: &Path) -> Result<(), String> {
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!(
            "{} has a '..' segment; a held-out marker before it would not mark where the \
             files land",
            path.display()
        ));
    }
    if marker_segment(path).is_some() {
        Ok(())
    } else {
        Err(format!(
            "{} has no path segment in {HELD_OUT_PATH_MARKERS:?}. This set must never be \
             trained on, and a path without a marker is one qd-train's path check would let \
             a training process open (CLAUDE.md rule 3).",
            path.display()
        ))
    }
}

/// `crates/qd-train/src/held_out.rs::CALLER_RECORD_PREFIX`.
pub const CALLER_RECORD_PREFIX: &str = "lappi.caller_record";
/// A longer line is skipped unparsed: the contract writes no record over 64 KiB.
pub const MAX_GUARD_LINE_BYTES: usize = 1 << 20;
/// A file longer than this is refused as unscanned, never passed as clean.
pub const MAX_GUARD_BYTES: u64 = 8 << 30;

/// The end (exclusive) of the JSON string token that starts at `bytes[start] == b'"'`. The text
/// is valid JSON, so an unescaped quote ends the token.
fn string_token_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    bytes.len()
}

/// The JSON string token `token` (quotes included) as text.
fn decode_string(token: &[u8]) -> Option<String> {
    serde_json::from_slice::<String>(token).ok()
}

/// Whether `json`, which is valid JSON, has a TOP-LEVEL `"record"` key (any copy of a
/// duplicated key) holding a string that starts with the prefix: in a top-level object, or in an
/// object that is a direct element of a top-level array. Nothing nested deeper is read, and a
/// string value that quotes the format is never a key. A parse into a map would keep only the
/// last copy of a duplicated key, hence the token walk.
fn top_level_record(json: &[u8]) -> bool {
    let mut stack: Vec<u8> = Vec::new();
    let mut expect_key = false;
    let mut want_value = false;
    let mut i = 0;
    while i < json.len() {
        let c = json[i];
        if matches!(c, b' ' | b'\t' | b'\r' | b'\n') {
            i += 1;
            continue;
        }
        let is_colon = c == b':';
        let mut end = i + 1;
        match c {
            b'{' => {
                stack.push(b'{');
                expect_key = true;
            }
            b'[' => {
                stack.push(b'[');
                expect_key = false;
            }
            b'}' | b']' => {
                stack.pop();
                expect_key = false;
            }
            b',' => expect_key = stack.last() == Some(&b'{'),
            b':' => expect_key = false,
            b'"' => {
                end = string_token_end(json, i);
                let token = &json[i..end];
                let at_target = matches!(stack.as_slice(), [b'{'] | [b'[', b'{']);
                if want_value {
                    if decode_string(token).is_some_and(|r| r.starts_with(CALLER_RECORD_PREFIX)) {
                        return true;
                    }
                } else if expect_key
                    && at_target
                    && decode_string(token).as_deref() == Some("record")
                {
                    want_value = true;
                    expect_key = false;
                    i = end;
                    continue;
                }
                expect_key = false;
            }
            _ => {
                // A number or literal: scan to its end.
                while end < json.len()
                    && !matches!(json[end], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
                {
                    end += 1;
                }
                expect_key = false;
            }
        }
        if want_value && !is_colon {
            want_value = false;
        }
        i = end;
    }
    false
}

/// A line that does not parse (a BOM, damage, invalid UTF-8) is a record only when it starts,
/// in column 0 after an optional UTF-8 BOM, with `{` or `[` and carries the key and the prefix
/// side by side: `"record"`, optional JSON whitespace, `:`, optional whitespace, `"lappi.caller_record`.
/// A line that merely contains both strings (source code, prose) is not one.
fn damaged_record_line(line: &[u8]) -> bool {
    const BOM: &[u8] = &[0xef, 0xbb, 0xbf];
    let head = line.strip_prefix(BOM).unwrap_or(line);
    if !matches!(head.first(), Some(b'{' | b'[')) {
        return false;
    }
    let ws = |b: &u8| matches!(b, b' ' | b'\t' | b'\r' | b'\n');
    let key = b"\"record\"";
    let prefix = CALLER_RECORD_PREFIX.as_bytes();
    line.windows(key.len()).enumerate().any(|(i, w)| {
        if w != key {
            return false;
        }
        let rest = &line[i + key.len()..];
        let rest = &rest[rest.iter().take_while(|b| ws(b)).count()..];
        let Some(rest) = rest.strip_prefix(b":") else {
            return false;
        };
        let rest = &rest[rest.iter().take_while(|b| ws(b)).count()..];
        rest.strip_prefix(b"\"")
            .is_some_and(|r| r.starts_with(prefix))
    })
}

/// One line (terminator already removed) is a caller record when it is JSONL whose top-level
/// `"record"` key (any copy of a duplicated key) starts with the prefix, or a JSON array holding
/// such an object; a line that parses and is not that is clean, however much text it quotes. A
/// line that does not parse is read by [`damaged_record_line`]. The budget is on the content, so
/// a record of exactly `line_bytes` is read.
fn is_caller_record_line(line: &[u8], line_bytes: usize) -> bool {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if line.len() > line_bytes {
        return false;
    }
    match serde_json::from_slice::<serde_json::Value>(line) {
        Ok(_) => top_level_record(line),
        Err(_) => damaged_record_line(line),
    }
}

/// Whether `chunk` holds a `\r` not followed by `\n`, and whether it ends in one. `prev_cr` is a
/// `\r` that ended the previous chunk of the same line.
fn bare_cr(prev_cr: bool, chunk: &[u8]) -> (bool, bool) {
    if prev_cr && chunk.first() != Some(&b'\n') {
        return (true, false);
    }
    let inner = chunk.windows(2).any(|w| w[0] == b'\r' && w[1] != b'\n');
    (inner, chunk.last() == Some(&b'\r'))
}

/// Finds a `str.splitlines()` separator the guard does not split on, in one line. Fed the line in
/// chunks (an over-long line is read in pieces); the JSON string state and the last two bytes
/// carry across them. A raw control character (`\x0b`, `\x0c`, `\x1c`..`\x1e`) is refused
/// anywhere: it is not legal inside a JSON string either. U+0085, U+2028 and U+2029 are refused
/// outside a string only: inside a string value they are legal JSON and legitimate data, and the
/// readers split on `\n` alone. A backslash escapes the byte after it, in or out of a string.
#[derive(Default)]
struct BreakScan {
    in_string: bool,
    escaped: bool,
    tail: [u8; 2],
    seen: usize,
}

impl BreakScan {
    /// The separator `chunk` holds that the guard must refuse, if any.
    fn feed(&mut self, chunk: &[u8]) -> Option<&'static str> {
        for &b in chunk {
            if matches!(b, 0x0b | 0x0c | 0x1c..=0x1e) {
                return Some("a raw control character (U+000B, U+000C or U+001C..U+001E)");
            }
            if !self.in_string {
                let [p2, p1] = self.tail;
                if b == 0x85 && self.seen >= 1 && p1 == 0xC2 {
                    return Some("U+0085");
                }
                if matches!(b, 0xA8 | 0xA9) && self.seen >= 2 && p2 == 0xE2 && p1 == 0x80 {
                    return Some(if b == 0xA8 { "U+2028" } else { "U+2029" });
                }
            }
            if self.escaped {
                self.escaped = false;
            } else if b == b'\\' {
                self.escaped = true;
            } else if b == b'"' {
                self.in_string = !self.in_string;
            }
            self.tail = [self.tail[1], b];
            self.seen = self.seen.saturating_add(1);
        }
        None
    }
}

fn foreign_break_verdict(line_no: usize, what: &str) -> Verdict {
    Verdict::Unscannable(format!(
        "line {line_no} holds {what}, which str.splitlines() splits on and the guard does not; \
         a reader that did could read a record behind it"
    ))
}

/// Compressed or UTF-16 bytes hide a record from a line scan.
fn unscannable_magic(head: &[u8]) -> Option<&'static str> {
    const MAGIC: [(&[u8], &str); 4] = [
        (&[0x1f, 0x8b], "gzip"),
        (&[0x28, 0xb5, 0x2f, 0xfd], "zstd"),
        (&[0xff, 0xfe], "UTF-16 (little-endian BOM)"),
        (&[0xfe, 0xff], "UTF-16 (big-endian BOM)"),
    ];
    MAGIC
        .iter()
        .find(|(m, _)| head.starts_with(m))
        .map(|(_, name)| *name)
}

/// The guard's three outcomes besides clean, the oracle's words (`fixtures/caller/guard-cases.json`).
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Line `n` (1-based) is a caller record.
    Record(usize),
    /// The bytes could not be scanned: why.
    Unscannable(String),
}

fn bare_cr_verdict(line_no: usize) -> Verdict {
    Verdict::Unscannable(format!(
        "line {line_no} holds a bare carriage return; the readers split lines differently \
         (text mode on '\\r', '\\n' only), so a record could sit behind it"
    ))
}

/// The content guard over a reader: the first caller-record line is `Record`; compressed or
/// UTF-16 bytes and a stream longer than `max_bytes` are `Unscannable`. Memory is bounded by
/// `line_bytes`. There is no sniff of the first byte: garbage, a comment or an array first line
/// is scanned like any other.
fn scan_caller_records(
    mut reader: impl BufRead,
    line_bytes: usize,
    max_bytes: u64,
) -> Result<(), Verdict> {
    let io_err = |e: io::Error| Verdict::Unscannable(e.to_string());
    if let Some(name) = unscannable_magic(reader.fill_buf().map_err(io_err)?) {
        return Err(Verdict::Unscannable(format!("{name} bytes")));
    }
    let over_budget = || {
        Verdict::Unscannable(format!(
            "longer than the {max_bytes}-byte budget, so it was not fully scanned"
        ))
    };
    // Room for the content, a '\r' and a '\n': a record of exactly `line_bytes` is scanned.
    let read_cap = line_bytes as u64 + 2;
    let mut total: u64 = 0;
    let mut line_no = 0usize;
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        let n = (&mut reader)
            .take(read_cap)
            .read_until(b'\n', &mut buf)
            .map_err(io_err)?;
        if n == 0 {
            return Ok(());
        }
        line_no += 1;
        total += n as u64;
        let overlong = buf.last() != Some(&b'\n') && buf.len() as u64 == read_cap;
        let (bare, mut carry) = bare_cr(false, &buf);
        if bare {
            return Err(bare_cr_verdict(line_no));
        }
        let mut breaks = BreakScan::default();
        if let Some(what) = breaks.feed(&buf) {
            return Err(foreign_break_verdict(line_no, what));
        }
        if overlong {
            // Over-long: drain the rest of the line unparsed, still counting every byte.
            loop {
                let chunk = reader.fill_buf().map_err(io_err)?;
                if chunk.is_empty() {
                    if carry {
                        return Err(bare_cr_verdict(line_no));
                    }
                    break;
                }
                let (used, done) = match chunk.iter().position(|b| *b == b'\n') {
                    Some(i) => (i + 1, true),
                    None => (chunk.len(), false),
                };
                total += used as u64;
                let (bare, c) = bare_cr(carry, &chunk[..used]);
                carry = c;
                let foreign = breaks.feed(&chunk[..used]);
                reader.consume(used);
                if bare {
                    return Err(bare_cr_verdict(line_no));
                }
                if let Some(what) = foreign {
                    return Err(foreign_break_verdict(line_no, what));
                }
                if total > max_bytes || done {
                    break;
                }
            }
        } else if carry {
            // A lone '\r' ended the last line of the stream.
            return Err(bare_cr_verdict(line_no));
        } else {
            let content = buf.strip_suffix(b"\n").unwrap_or(&buf);
            if is_caller_record_line(content, line_bytes) {
                return Err(Verdict::Record(line_no));
            }
        }
        if total > max_bytes {
            return Err(over_budget());
        }
    }
}

/// Refuse an input that holds a caller record by content, or that cannot be scanned (layer 5 of
/// the trainer's door, `qd-train`'s `assert_no_caller_records`; the same rule as
/// `qd_data.caller_records`, one oracle in `fixtures/caller/guard-cases.json`). Every regular
/// file is scanned, whatever its first byte; compressed and UTF-16 bytes, a bare `\r`, a
/// non-regular file (a FIFO, device) and a file over 8 GiB are refused as unscannable, never
/// passed. A missing path or a directory has no bytes to scan and is left to the read that
/// follows, which names it.
pub fn refuse_caller_records(path: &Path, what: &str) -> Result<(), String> {
    scan_path(path, what, MAX_GUARD_LINE_BYTES, MAX_GUARD_BYTES)
}

fn scan_path(path: &Path, what: &str, line_bytes: usize, max_bytes: u64) -> Result<(), String> {
    let label = format!("{what} {}", path.display());
    // Decide by stat, never by open: opening a FIFO for reading blocks with no writer.
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("{label}: {e}")),
    };
    if meta.is_dir() {
        return Ok(());
    }
    if !meta.is_file() {
        return Err(format!(
            "{label}: not a regular file (a FIFO, device or socket has no stable bytes to scan); \
             refused rather than passed as clean (CLAUDE.md rule 3: a check that could not run \
             is not a check that passed)."
        ));
    }
    let file = File::open(path).map_err(|e| format!("{label}: {e}"))?;
    scan_label(BufReader::new(file), &label, line_bytes, max_bytes)
}

fn scan_label(
    reader: impl BufRead,
    label: &str,
    line_bytes: usize,
    max_bytes: u64,
) -> Result<(), String> {
    match scan_caller_records(reader, line_bytes, max_bytes) {
        Ok(()) => Ok(()),
        Err(Verdict::Record(n)) => Err(format!(
            "{label}: line {n} is a caller record (`{CALLER_RECORD_PREFIX}`). Caller records are \
             never training data, wherever the file lives; the whole file is refused (CLAUDE.md \
             rule 3)."
        )),
        Err(Verdict::Unscannable(why)) => Err(format!(
            "{label}: {why}; refused rather than passed as clean (CLAUDE.md rule 3: a check \
             that could not run is not a check that passed)."
        )),
    }
}

/// [`refuse_caller_records`] over bytes a bounded read already holds: the same rule and line
/// bound, without a second open of the file (so nothing can be swapped in between the scan and
/// the use). The bytes are already in memory under the caller's own bound (a MinHash request may
/// exceed the 8 GiB file budget), so every one is scanned and the byte budget cannot truncate
/// the check. `label` names the input in the error.
pub fn refuse_caller_record_bytes(bytes: &[u8], label: &str) -> Result<(), String> {
    scan_label(bytes, label, MAX_GUARD_LINE_BYTES, bytes.len() as u64)
}

/// Refuse to read `path` as an input to training rows when it, or the file it resolves to,
/// sits under a held-out marker. `what` names the input in the error (`--config`, `catalog`).
///
/// Both spellings are checked: the path as given, and its canonical form when it exists, so a
/// symlink from an unmarked directory into a held-out one is refused too. A path that does not
/// exist is refused by the read that follows, which names it.
pub fn refuse_training_input(path: &Path, what: &str) -> Result<(), String> {
    let refuse = |seen: &Path, seg: String| {
        Err(format!(
            "{what} {} is held-out data (path segment {seg:?} at {}): a producer of training \
             rows never reads it (CLAUDE.md rule 3). Held-out sets enter only as `--target` \
             decontamination sets.",
            path.display(),
            seen.display()
        ))
    };
    if let Some(seg) = marker_segment(path) {
        return refuse(path, seg);
    }
    match std::fs::canonicalize(path) {
        Ok(real) => match marker_segment(&real) {
            Some(seg) => refuse(&real, seg),
            None => refuse_caller_records(path, what),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("{what} {}: {e}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "qd-prep-heldout-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_output_path_needs_a_held_out_segment_and_no_dot_dot() {
        let ok = |p: &str| check_held_out_path(Path::new(p)).is_ok();
        assert!(ok("/x/v6/heldout/natural-bugs"));
        assert!(ok("/x/HeldOut/natural-bugs"));
        assert!(ok("data/held-out/nb"));
        assert!(ok("data/held_out/nb"));
        assert!(!ok("/x/v6/natural-bugs"));
        assert!(!ok("/x/heldoutset/nb"), "a marker is a whole segment");
        assert!(!ok("/x/heldout/../train/nb"), "'..' escapes the marker");
    }

    #[test]
    fn a_training_input_under_a_marker_is_refused_by_any_spelling() {
        for p in [
            "/d/heldout/tssb.zip",
            "/d/HELDOUT/x.jsonl",
            "rel/held-out/x.json",
            "/d/Held_Out/x",
        ] {
            let err = refuse_training_input(Path::new(p), "--config").unwrap_err();
            assert!(
                err.contains("held-out data") && err.contains("--config"),
                "{p}: {err}"
            );
        }
        assert!(refuse_training_input(Path::new("/d/heldoutset/x"), "c").is_ok());
        assert!(refuse_training_input(Path::new("/d/missing/x.json"), "c").is_ok());
    }

    const REC: &str = r#"{"record":"lappi.caller_record","record_version":1,"kind":"decision"}"#;

    fn oracle_bytes(case: &serde_json::Value) -> Vec<u8> {
        if let Some(t) = case["text"].as_str() {
            return t.as_bytes().to_vec();
        }
        if let Some(h) = case["hex"].as_str() {
            return (0..h.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
                .collect();
        }
        let g = &case["gen"];
        let eol = g["eol"].as_str().unwrap_or("\n");
        let core = if let Some(n) = g["record_padded_to"].as_u64() {
            let head = r#"{"record":"lappi.caller_record","pad":""#;
            format!("{head}{}\"}}{eol}", "x".repeat(n as usize - head.len() - 2))
        } else if let Some(n) = g["filler_line"].as_u64() {
            format!("{{\"pad\":\"{}\"}}{eol}", "x".repeat(n as usize - 10))
        } else {
            g["repeat"]
                .as_str()
                .unwrap()
                .repeat(g["times"].as_u64().unwrap() as usize)
        };
        let first = g["first"].as_str().unwrap_or("");
        let then = g["then"].as_str().unwrap_or("");
        format!("{first}{core}{then}").into_bytes()
    }

    fn oracle() -> Vec<serde_json::Value> {
        let doc: serde_json::Value =
            serde_json::from_str(include_str!("../../../fixtures/caller/guard-cases.json"))
                .unwrap();
        doc["cases"].as_array().unwrap().clone()
    }

    /// One oracle, three implementations (python/tests/test_caller_records_guard.py, qd-train's
    /// tests/caller_records_are_held_out.rs): every case has the same verdict here.
    #[test]
    fn the_shared_oracle_cases_have_the_same_verdicts_here() {
        let cases = oracle();
        assert!(
            cases.len() >= 30,
            "the oracle shrank to {} cases",
            cases.len()
        );
        for case in &cases {
            let name = case["name"].as_str().unwrap();
            let line_bytes = case["line_bytes"].as_u64().map_or(1 << 20, |n| n as usize);
            let total = case["total_bytes"].as_u64().unwrap_or(8 << 30);
            let bytes = oracle_bytes(case);
            let got = match scan_caller_records(&bytes[..], line_bytes, total) {
                Ok(()) => "clean",
                Err(Verdict::Record(_)) => "record",
                Err(Verdict::Unscannable(_)) => "unscannable",
            };
            assert_eq!(got, case["expect"].as_str().unwrap(), "{name}");
        }
    }

    /// The same cases through the door a producer actually calls, from files, whatever their name.
    #[test]
    fn the_input_door_gives_the_oracle_verdicts_on_files() {
        let dir = scratch("oracle-files");
        for (i, case) in oracle().iter().enumerate() {
            let name = case["name"].as_str().unwrap();
            let path = dir.join(format!("input-{i}.dat"));
            std::fs::write(&path, oracle_bytes(case)).unwrap();
            // the production limits, so the cases with their own limits are scanned directly
            if case["line_bytes"].is_null() && case["total_bytes"].is_null() {
                let got = refuse_training_input(&path, "--config");
                match case["expect"].as_str().unwrap() {
                    "clean" => assert!(got.is_ok(), "{name}: {got:?}"),
                    "record" => assert!(got.unwrap_err().contains("caller record"), "{name}"),
                    _ => assert!(
                        got.unwrap_err().contains("refused rather than passed"),
                        "{name}"
                    ),
                }
            } else {
                let line_bytes = case["line_bytes"].as_u64().map_or(1 << 20, |n| n as usize);
                let total = case["total_bytes"].as_u64().unwrap_or(8 << 30);
                let got = scan_path(&path, "x", line_bytes, total);
                match case["expect"].as_str().unwrap() {
                    "clean" => assert!(got.is_ok(), "{name}: {got:?}"),
                    _ => assert!(got.is_err(), "{name}"),
                }
            }
        }
    }

    #[test]
    fn a_copied_record_file_is_refused_at_the_input_door_wherever_it_lives() {
        let dir = scratch("copy");
        let copied = dir.join("data/pool/anything.txt");
        std::fs::create_dir_all(copied.parent().unwrap()).unwrap();
        std::fs::write(&copied, format!("{{\"a\":1}}\n{REC}\n")).unwrap();
        let err = refuse_training_input(&copied, "--config").unwrap_err();
        assert!(
            err.contains("caller record") && err.contains("--config"),
            "{err}"
        );
        let clean = dir.join("data/pool/clean.jsonl");
        std::fs::write(&clean, "{\"a\":1}\n").unwrap();
        assert!(refuse_training_input(&clean, "--config").is_ok());
        // nothing is skipped on its first byte: a binary-looking file is scanned like any other
        let zip = dir.join("x.zip");
        std::fs::write(&zip, b"PK\x03\x04\n{\"record\":\"lappi.caller_record\"}").unwrap();
        assert!(
            refuse_training_input(&zip, "src")
                .unwrap_err()
                .contains("caller record")
        );
        let gz = dir.join("x.gz");
        std::fs::write(&gz, [0x1f, 0x8b, 8, 0, 0]).unwrap();
        assert!(
            refuse_training_input(&gz, "src")
                .unwrap_err()
                .contains("refused rather than")
        );
        // the repo's own fixture of real record lines, copied under another name and tree
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/caller/valid-records.jsonl");
        let copy = dir.join("data/pool/commitpackft/rows.jsonl");
        std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
        std::fs::copy(&fixture, &copy).unwrap();
        assert!(
            refuse_training_input(&copy, "pool")
                .unwrap_err()
                .contains("caller record")
        );
        // a missing path and a directory have no bytes: left to the read that follows
        assert!(refuse_training_input(&dir.join("missing.jsonl"), "src").is_ok());
        assert!(
            refuse_training_input(&dir.join("data"), "src").is_ok(),
            "a directory"
        );
    }

    /// Opening a FIFO for reading blocks forever with no writer: the guard decides by stat, and
    /// refuses it as unscannable (a check that could not run is not one that passed).
    #[test]
    fn a_fifo_is_refused_as_unscannable_and_never_opened() {
        let dir = scratch("fifo");
        let fifo = dir.join("rows.jsonl");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(made.success(), "mkfifo");
        let err = refuse_caller_records(&fifo, "pool").unwrap_err();
        assert!(
            err.contains("not a regular file") && err.contains("refused rather than"),
            "{err}"
        );
    }

    #[test]
    fn a_cr_separated_file_is_refused_not_read_as_one_clean_line() {
        let dir = scratch("cr");
        let path = dir.join("rows.jsonl");
        let mut rows = vec!["{\"a\":1}"; 600];
        rows.push(REC);
        std::fs::write(&path, format!("{}\n", rows.join("\r"))).unwrap();
        let err = refuse_caller_records(&path, "pool").unwrap_err();
        assert!(err.contains("bare carriage return"), "{err}");
    }

    /// A file re-separated with any other `str.splitlines()` separator is refused at the input
    /// door, not scanned as one over-long clean line.
    #[test]
    fn a_file_re_separated_with_a_splitlines_separator_is_refused() {
        let dir = scratch("splitlines");
        let filler = format!("{{\"pad\":\"{}\"}}", "x".repeat(2000));
        for (i, sep) in [
            '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{85}', '\u{2028}', '\u{2029}',
        ]
        .into_iter()
        .enumerate()
        {
            let path = dir.join(format!("rows-{i}.jsonl"));
            let mut rows = vec![filler.as_str(); 600];
            rows.push(REC);
            std::fs::write(&path, format!("{}\n", rows.join(sep.to_string().as_str()))).unwrap();
            let err = refuse_caller_records(&path, "pool").unwrap_err();
            assert!(err.contains("str.splitlines()"), "{sep:?}: {err}");
        }
    }

    /// U+0085, U+2028 and U+2029 are legal raw inside a JSON string: data, not refused.
    #[test]
    fn a_separator_inside_a_string_value_is_data() {
        for sep in ['\u{85}', '\u{2028}', '\u{2029}'] {
            let row = serde_json::json!({"text": format!("a{sep}b")}).to_string();
            assert!(row.contains(sep));
            assert_eq!(
                refuse_caller_record_bytes(format!("{row}\n{row}\n").as_bytes(), "x"),
                Ok(())
            );
            let outside = format!("{row}{sep}{row}\n");
            assert!(
                refuse_caller_record_bytes(outside.as_bytes(), "x").is_err(),
                "{sep:?}"
            );
        }
    }

    #[test]
    fn bytes_already_read_are_scanned_whole_whatever_the_file_budget() {
        let mut big = vec![b'x'; 4096];
        big.push(b'\n');
        big.extend_from_slice(REC.as_bytes());
        assert!(
            refuse_caller_record_bytes(&big, "in")
                .unwrap_err()
                .contains("caller record")
        );
        assert!(refuse_caller_record_bytes(b"{\"a\":1}\n", "in").is_ok());
        assert!(refuse_caller_record_bytes(b"", "in").is_ok());
        assert!(
            refuse_caller_record_bytes(&[0x1f, 0x8b, 8], "in")
                .unwrap_err()
                .contains("refused rather")
        );
    }

    #[test]
    fn a_symlink_into_a_held_out_directory_is_refused() {
        let dir = scratch("symlink");
        let held = dir.join("heldout");
        std::fs::create_dir_all(&held).unwrap();
        std::fs::write(held.join("rows.jsonl"), b"{}\n").unwrap();
        let link = dir.join("innocent.jsonl");
        std::os::unix::fs::symlink(held.join("rows.jsonl"), &link).unwrap();
        let err = refuse_training_input(&link, "catalog").unwrap_err();
        assert!(
            err.contains("innocent.jsonl") && err.contains("catalog"),
            "{err}"
        );
        let plain = dir.join("plain.jsonl");
        std::fs::write(&plain, b"{}\n").unwrap();
        assert!(refuse_training_input(&plain, "catalog").is_ok());
    }
}
