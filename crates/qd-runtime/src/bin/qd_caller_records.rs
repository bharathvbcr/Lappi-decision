//! `qd-caller-records <file.jsonl>...` — check caller-record files against
//! `docs/caller-contract.md`, through [`qd_runtime::caller_record::validate_line`].
//!
//! It reads and reports; it moves, admits or deletes nothing. Admitting caller data to training
//! or eval is a human decision (the human's "Synthetic only", 2026-10-06), so this tool has no
//! flag that does it.
//!
//! The report is one JSON object on stdout:
//! `files`, `lines`, `decisions`, `outcomes`, `invalid` (with the first 20 reasons, file and line,
//! and the total), `duplicate_record_ids`, `outcomes_without_decision`, `decisions_without_outcome`
//! and `readings` (count per reading).
//!
//! Exit codes: 0 every line valid; 2 at least one line or file invalid; 1 an I/O failure or no
//! file given. An unreadable file is never reported as an empty valid one.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use qd_runtime::caller_record::{
    check_store_path, validate_line, RecordKind, MAX_RECORD_BYTES,
};
use serde_json::json;

const SHOWN_INVALID: usize = 20;

fn main() -> ExitCode {
    let files: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if files.is_empty() {
        eprintln!("usage: qd-caller-records <file.jsonl>...");
        return ExitCode::from(1);
    }

    let mut lines = 0u64;
    let mut decisions: HashSet<String> = HashSet::new();
    let mut outcome_ids: Vec<String> = Vec::new();
    let mut seen: HashMap<(String, &'static str), u64> = HashMap::new();
    let mut readings: BTreeMap<String, u64> = BTreeMap::new();
    let mut invalid_total = 0u64;
    let mut invalid_shown = Vec::new();
    let mut note = |file: &str, line: u64, why: String, total: &mut u64| {
        *total += 1;
        if invalid_shown.len() < SHOWN_INVALID {
            invalid_shown.push(json!({"file": file, "line": line, "reason": why}));
        }
    };

    for path in &files {
        let shown = path.display().to_string();
        if let Err(e) = check_store_path(path) {
            note(&shown, 0, e.to_string(), &mut invalid_total);
            continue;
        }
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("qd-caller-records: {shown}: {e}");
                return ExitCode::from(1);
            }
        };
        let mut reader = BufReader::new(file);
        let mut n = 0u64;
        loop {
            n += 1;
            let mut buf = Vec::new();
            // Bounded: one byte past the cap is enough to know a line is over it.
            let read = match (&mut reader)
                .take(MAX_RECORD_BYTES as u64 + 2)
                .read_until(b'\n', &mut buf)
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("qd-caller-records: {shown}:{n}: {e}");
                    return ExitCode::from(1);
                }
            };
            if read == 0 {
                break;
            }
            let complete = buf.last() == Some(&b'\n');
            if complete {
                buf.pop();
            } else if buf.len() > MAX_RECORD_BYTES {
                // Over the cap: report it, then skip the rest of this line without holding it.
                note(&shown, n, format!("line over the {MAX_RECORD_BYTES}-byte record cap"), &mut invalid_total);
                let mut sink = Vec::new();
                loop {
                    sink.clear();
                    match (&mut reader).take(1 << 16).read_until(b'\n', &mut sink) {
                        Ok(0) => break,
                        Ok(_) if sink.last() == Some(&b'\n') => break,
                        Ok(_) => continue,
                        Err(e) => {
                            eprintln!("qd-caller-records: {shown}:{n}: {e}");
                            return ExitCode::from(1);
                        }
                    }
                }
                lines += 1;
                continue;
            } else {
                // A final line with no newline is a write that was cut off, not a record.
                note(&shown, n, "a final line with no newline (a torn write)".into(), &mut invalid_total);
                lines += 1;
                break;
            }
            lines += 1;
            match validate_line(&buf) {
                Ok(summary) => {
                    let key = (summary.record_id.clone(), summary.kind.as_str());
                    *seen.entry(key).or_insert(0) += 1;
                    match summary.kind {
                        RecordKind::Decision => {
                            decisions.insert(summary.record_id);
                            if let Some(r) = summary.reading {
                                *readings.entry(r).or_insert(0) += 1;
                            }
                        }
                        RecordKind::Outcome => outcome_ids.push(summary.record_id),
                    }
                }
                Err(e) => note(&shown, n, e.to_string(), &mut invalid_total),
            }
        }
    }

    let duplicates: u64 = seen.values().filter(|&&c| c > 1).map(|c| c - 1).sum();
    let outcome_set: HashSet<&String> = outcome_ids.iter().collect();
    let orphans = outcome_ids.iter().filter(|id| !decisions.contains(*id)).count();
    let unresolved = decisions.iter().filter(|id| !outcome_set.contains(id)).count();
    let report = json!({
        "files": files.len(),
        "lines": lines,
        "decisions": decisions.len(),
        "outcomes": outcome_ids.len(),
        "invalid": {"total": invalid_total, "shown": invalid_shown},
        "duplicate_record_ids": duplicates,
        "outcomes_without_decision": orphans,
        "decisions_without_outcome": unresolved,
        "readings": readings,
    });
    println!("{report}");
    if invalid_total > 0 || duplicates > 0 {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}
