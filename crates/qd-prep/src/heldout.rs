//! Rule 3's path marker, with one owner in this crate. A held-out set lives under a path segment
//! named for it (`crates/qd-train/src/held_out.rs::DEFAULT_HELD_OUT_PATH_MARKERS`, the trainer's
//! door). An evaluation set's output must carry one ([`check_held_out_path`]), and a producer of
//! training rows refuses to read an input that carries one ([`refuse_training_input`]).
//!
//! Decontamination targets are not training inputs. A producer scans its rows against held-out
//! sets so that none of them leaks into training, so `--target` files may sit under a marker.
//!
//! A path is not the only guard. A copy of an app's caller-record file anywhere else carries no
//! marker, so [`refuse_caller_records`] reads the content: any line whose parsed `"record"` starts
//! with `lappi.caller_record` refuses the whole file, wherever it lives; compressed or UTF-16
//! bytes and a file over the byte budget are refused as unscannable, never passed. The rule is
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

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// One line (terminator already removed) is a caller record when it parses to an object whose
/// `"record"` starts with the prefix, or carries both the `"record"` key text and the prefix (a
/// damaged line, a BOM, a duplicated key, a JSON array holding one). Errs toward refusing: this
/// only ever narrows admission. The budget is on the content, so a record of exactly
/// `line_bytes` is read.
fn is_caller_record_line(line: &[u8], line_bytes: usize) -> bool {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if line.len() > line_bytes {
        return false;
    }
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_slice::<serde_json::Value>(line)
        && map
            .get("record")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|r| r.starts_with(CALLER_RECORD_PREFIX))
    {
        return true;
    }
    contains(line, b"\"record\"") && contains(line, CALLER_RECORD_PREFIX.as_bytes())
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
        if buf.last() != Some(&b'\n') && buf.len() as u64 == read_cap {
            // Over-long: drain the rest of the line unparsed, still counting every byte.
            loop {
                let chunk = reader.fill_buf().map_err(io_err)?;
                if chunk.is_empty() {
                    break;
                }
                let (used, done) = match chunk.iter().position(|b| *b == b'\n') {
                    Some(i) => (i + 1, true),
                    None => (chunk.len(), false),
                };
                total += used as u64;
                reader.consume(used);
                if total > max_bytes || done {
                    break;
                }
            }
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
/// file is scanned, whatever its first byte; compressed and UTF-16 bytes and a file over 8 GiB
/// are refused as unscannable, never passed. A missing path or a directory has no bytes to scan
/// and is left to the read that follows, which names it.
pub fn refuse_caller_records(path: &Path, what: &str) -> Result<(), String> {
    scan_path(path, what, MAX_GUARD_LINE_BYTES, MAX_GUARD_BYTES)
}

fn scan_path(path: &Path, what: &str, line_bytes: usize, max_bytes: u64) -> Result<(), String> {
    let label = format!("{what} {}", path.display());
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("{label}: {e}")),
    };
    if !file
        .metadata()
        .map_err(|e| format!("{label}: {e}"))?
        .is_file()
    {
        return Ok(());
    }
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
        std::fs::write(&zip, b"PK\x03\x04 {\"record\":\"lappi.caller_record\"}").unwrap();
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
