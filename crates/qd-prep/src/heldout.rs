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
//! with `lappi.caller_record` refuses the whole file, wherever it lives. It mirrors
//! `crates/qd-train/src/held_out.rs::assert_no_caller_records` (same prefix, same line budget).
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
/// A JSON-lines file longer than this is refused as unscanned, never passed as clean.
pub const MAX_GUARD_BYTES: u64 = 8 << 30;

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// One line is a caller record when it parses to an object whose `"record"` starts with the
/// prefix, or carries both the `"record"` key text and the prefix (a damaged line, a BOM, a
/// duplicated key). Errs toward refusing: this only ever narrows admission.
fn is_caller_record_line(line: &[u8], line_bytes: usize) -> bool {
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

/// The content guard over a reader: refuse on the first caller-record line, and refuse as
/// unscanned a stream longer than `max_bytes`. Memory is bounded by `line_bytes`.
fn scan_caller_records(
    mut reader: impl BufRead,
    what: &str,
    line_bytes: usize,
    max_bytes: u64,
) -> Result<(), String> {
    let refuse_record = |n: usize| {
        Err(format!(
            "{what}: line {n} is a caller record (`{CALLER_RECORD_PREFIX}`). Caller records are never \
             training data, wherever the file lives; the whole file is refused (CLAUDE.md rule 3)."
        ))
    };
    let mut total: u64 = 0;
    let mut line_no = 0usize;
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        let n = (&mut reader)
            .take(line_bytes as u64 + 1)
            .read_until(b'\n', &mut buf)
            .map_err(|e| format!("{what}: {e}"))?;
        if n == 0 {
            return Ok(());
        }
        line_no += 1;
        total += n as u64;
        let mut oversize = buf.len() > line_bytes && buf.last() != Some(&b'\n');
        if oversize {
            // Drain the rest of the line unparsed, still counting every byte against the budget.
            loop {
                let chunk = reader.fill_buf().map_err(|e| format!("{what}: {e}"))?;
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
            oversize = true;
        }
        if !oversize && is_caller_record_line(&buf, line_bytes) {
            return refuse_record(line_no);
        }
        if total > max_bytes {
            return Err(format!(
                "{what}: longer than the {max_bytes}-byte budget of the caller-record guard, so \
                 it was not fully scanned; refused rather than passed as clean."
            ));
        }
    }
}

/// Refuse a JSON-lines input that holds a caller record, by content (layer 5 of the trainer's
/// door, `qd-train`'s `assert_no_caller_records`). A file whose first non-blank byte is not `{`
/// (a zip, parquet, a JSON array) is not JSON lines and is not scanned. A directory or a missing
/// path is the read that follows's to refuse.
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
    let mut reader = BufReader::new(file);
    loop {
        let chunk = reader.fill_buf().map_err(|e| format!("{label}: {e}"))?;
        match chunk.iter().position(|b| !b.is_ascii_whitespace()) {
            Some(i) if chunk[i] == b'{' => break,
            Some(i) if chunk[i..].starts_with(&[0xef, 0xbb, 0xbf]) => break,
            Some(_) => return Ok(()),
            None if chunk.is_empty() => return Ok(()),
            None => {
                let n = chunk.len();
                reader.consume(n);
            }
        }
    }
    scan_caller_records(reader, &label, line_bytes, max_bytes)
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

    fn scan(bytes: &[u8], line_bytes: usize, max: u64) -> Result<(), String> {
        scan_caller_records(bytes, "x", line_bytes, max)
    }

    #[test]
    fn a_caller_record_line_refuses_the_file_by_content() {
        let big = MAX_GUARD_BYTES;
        for (name, body) in [
            ("plain", format!("{{\"a\":1}}\n{REC}\n{{\"b\":2}}\n")),
            ("no final newline", REC.to_string()),
            (
                "later version",
                r#"{"record":"lappi.caller_record.v2"}"#.to_string(),
            ),
            ("crlf", format!("{{\"a\":1}}\r\n{REC}\r\n")),
            (
                "escaped",
                r#"{"rec\u006frd":"lappi.caller_re\u0063ord"}"#.to_string(),
            ),
            ("unparseable beside", format!("garbage\n{REC}\n")),
            (
                "damaged",
                "{\"record\":\"lappi.caller_record\",\"kind\":\n".to_string(),
            ),
            ("bom", format!("{}{REC}\n", '\u{feff}')),
            (
                "duplicate key",
                r#"{"record":"lappi.caller_record","record":"x"}"#.to_string(),
            ),
        ] {
            let err = scan(body.as_bytes(), MAX_GUARD_LINE_BYTES, big).unwrap_err();
            assert!(err.contains("caller record"), "{name}: {err}");
        }
        for ok in [
            "",
            "{\"record\":\"decisions.json\"}\n",
            "{\"record\":7}\n",
            "{\"text\":\"lappi.caller_record\"}\n",
            "[\"lappi.caller_record\"]\n",
            "garbage\n\u{ff}\n{\"a\":1}\n",
        ] {
            assert!(
                scan(ok.as_bytes(), MAX_GUARD_LINE_BYTES, big).is_ok(),
                "{ok:?}"
            );
        }
    }

    #[test]
    fn the_guard_is_bounded_and_unfinished_is_a_refusal() {
        let many = "{\"a\":1}\n".repeat(100);
        let err = scan(many.as_bytes(), 64, 256).unwrap_err();
        assert!(err.contains("not fully scanned"), "{err}");
        let pad = "x".repeat(200);
        let long_then_record = format!("{{\"pad\":\"{pad}\"}}\n{REC}\n");
        let err = scan(long_then_record.as_bytes(), 150, 4096).unwrap_err();
        assert!(err.contains("caller record"), "{err}");
        let only_long = format!("{{\"pad\":\"{pad}\"}}\n");
        assert!(scan(only_long.as_bytes(), 150, 4096).is_ok());
        // an over-long line still counts against the byte budget
        let err = scan(only_long.as_bytes(), 150, 100).unwrap_err();
        assert!(err.contains("not fully scanned"), "{err}");
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
        // not JSON lines: not scanned, even with the prefix inside
        let zip = dir.join("x.zip");
        std::fs::write(&zip, b"PK\x03\x04 {\"record\":\"lappi.caller_record\"}").unwrap();
        assert!(refuse_training_input(&zip, "src").is_ok());
        assert!(
            refuse_training_input(&dir.join("data"), "src").is_ok(),
            "a directory"
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
