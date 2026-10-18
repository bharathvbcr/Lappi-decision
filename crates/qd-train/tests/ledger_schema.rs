//! The Rust ledger mirror against a ledger Python wrote (W2-1):
//!
//! * `Status::Started` and a reader that skips `started` rows by default, as `Ledger.rows()`
//!   does (GAP-STARTED-STATUS-UNKNOWN-TO-RUST-LEDGER-MIRROR-AND-CAMPAIGN-DRIVER-2026-10-17);
//! * the optional `val_without_train` (Fable Q14 P2) and `train_manifest_sha256` row fields,
//!   read and written, in the bytes Python writes.
//!
//! `tests/fixtures/ledger-schema/ledger.jsonl` is written by `qd_train.ledger` itself
//! (`tools/gen_ledger_schema_fixture.py`): started(A), completed(A), started(B, SIGKILLed),
//! killed(B, reconciled), a row recording both new fields, the same row recording neither.
//! `expected.json` is what Python's `Ledger.rows()` and `Ledger.rows(include_started=True)`
//! returned for it, plus the sha256 of every raw line.

use std::collections::BTreeMap;
use std::path::PathBuf;

use qd_train::ledger::{
    Environment, LedgerEntry, Protocol, Row, RowExtras, Stamp, Status, WallClockSource,
    read_entries,
};
use qd_train::tristate::TriState;
use serde_json::{Value, json};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ledger-schema")
}

fn expected() -> Value {
    let p = dir().join("expected.json");
    serde_json::from_str(
        &std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())),
    )
    .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn lines() -> Vec<String> {
    let p = dir().join("ledger.jsonl");
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .lines()
        .map(str::to_owned)
        .collect()
}

fn pairs(v: &Value) -> Vec<(String, String)> {
    v.as_array()
        .expect("an array of pairs")
        .iter()
        .map(|p| {
            (
                p[0].as_str().expect("row_id").to_string(),
                p[1].as_str().expect("status").to_string(),
            )
        })
        .collect()
}

fn as_pairs(entries: &[LedgerEntry]) -> Vec<(String, String)> {
    entries
        .iter()
        .map(|e| {
            let s = match e.status {
                Status::Started => "started",
                Status::Completed => "completed",
                Status::Killed => "killed",
                Status::Failed => "failed",
            };
            (e.row_id.clone(), s.to_string())
        })
        .collect()
}

#[test]
fn the_default_read_skips_started_rows_exactly_as_python_rows_does() {
    let want = pairs(&expected()["rows"]);
    assert_eq!(want.len(), 4, "the fixture holds four terminal rows");
    let got =
        read_entries(&dir().join("ledger.jsonl"), false).expect("a Python-written ledger parses");
    assert_eq!(as_pairs(&got), want);
    assert!(got.iter().all(|e| e.status != Status::Started));
}

#[test]
fn include_started_returns_every_line_in_file_order() {
    let want = pairs(&expected()["rows_include_started"]);
    let got = read_entries(&dir().join("ledger.jsonl"), true).expect("parses");
    assert_eq!(as_pairs(&got), want);
    let statuses: Vec<&str> = want.iter().map(|(_, s)| s.as_str()).collect();
    assert_eq!(
        statuses,
        [
            "started",
            "completed",
            "started",
            "killed",
            "completed",
            "completed"
        ]
    );
}

#[test]
fn the_chain_hash_is_over_every_raw_line_started_rows_included() {
    let want: Vec<String> = expected()["raw_line_sha256"]
        .as_array()
        .expect("hashes")
        .iter()
        .map(|h| h.as_str().expect("hex").to_string())
        .collect();
    let got = read_entries(&dir().join("ledger.jsonl"), true).expect("parses");
    assert_eq!(
        got.iter()
            .map(|e| e.line_sha256.clone())
            .collect::<Vec<_>>(),
        want
    );
    // the terminal-only read reports the same hashes for the rows it keeps
    let terminal = read_entries(&dir().join("ledger.jsonl"), false).expect("parses");
    for (kept, idx) in terminal.iter().zip([1, 3, 4, 5]) {
        assert_eq!(kept.line_sha256, want[idx]);
    }
    // and the file's own prev_row_hash links agree with those hashes
    let prevs: Vec<Option<String>> = lines()
        .iter()
        .map(|l| {
            let v: Value = serde_json::from_str(l).expect("json");
            v["prev_row_hash"].as_str().map(str::to_string)
        })
        .collect();
    assert_eq!(prevs[0], None);
    for i in 1..prevs.len() {
        assert_eq!(
            prevs[i].as_deref(),
            Some(want[i - 1].as_str()),
            "line {}",
            i + 1
        );
    }
}

#[test]
fn a_missing_file_is_an_empty_ledger_and_an_unknown_status_is_refused() {
    let d = std::env::temp_dir().join(format!("qd-ledger-schema-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("tmp dir");
    assert!(
        read_entries(&d.join("absent.jsonl"), false)
            .expect("absent is empty")
            .is_empty()
    );
    let bad = d.join("bad.jsonl");
    std::fs::write(&bad, "{\"row_id\":\"x\",\"status\":\"paused\"}\n").expect("write");
    let err = read_entries(&bad, false).expect_err("an unknown status is refused");
    assert!(err.to_string().contains("paused"), "{err}");
    std::fs::remove_dir_all(&d).expect("cleanup");
}

#[test]
fn status_parse_names_started_and_is_case_exact() {
    assert_eq!(Status::parse("started").expect("known"), Status::Started);
    assert!(Status::parse("Started").is_err());
}

#[test]
fn the_new_fields_are_read_off_python_rows_and_absent_means_not_recorded() {
    let want = expected();
    let got = read_entries(&dir().join("ledger.jsonl"), true).expect("parses");
    assert_eq!(
        got[4].extras.val_without_train,
        Some(want["val_without_train"].clone())
    );
    assert_eq!(
        got[4].extras.train_manifest_sha256.as_deref(),
        want["train_manifest_sha256"].as_str()
    );
    for e in [&got[0], &got[1], &got[2], &got[3], &got[5]] {
        assert_eq!(e.extras, RowExtras::default(), "row {}", e.row_id);
    }
}

#[test]
fn a_malformed_new_field_is_refused_by_the_reader() {
    let d = std::env::temp_dir().join(format!("qd-ledger-schema-bad-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("tmp dir");
    for (name, tail) in [
        ("sha", r#""train_manifest_sha256":"ABC""#),
        ("sha-type", r#""train_manifest_sha256":5"#),
        ("block", r#""val_without_train":["f"]"#),
        ("families", r#""val_without_train":{"families":["f"]}"#),
    ] {
        let p = d.join(format!("{name}.jsonl"));
        std::fs::write(
            &p,
            format!("{{\"row_id\":\"x\",\"status\":\"completed\",{tail}}}\n"),
        )
        .expect("write");
        let err = read_entries(&p, false).expect_err(name);
        assert!(err.to_string().contains("line 1"), "{name}: {err}");
    }
    // null is the same claim as absent
    let p = d.join("null.jsonl");
    std::fs::write(
        &p,
        "{\"row_id\":\"x\",\"status\":\"completed\",\"val_without_train\":null,\"train_manifest_sha256\":null}\n",
    )
    .expect("write");
    assert_eq!(
        read_entries(&p, false).expect("null parses")[0].extras,
        RowExtras::default()
    );
    std::fs::remove_dir_all(&d).expect("cleanup");
}

/// The content `tools/gen_ledger_schema_fixture.py::_rust_shaped_row` writes, as a Rust row.
fn fixture_row(line: &Value, status: Status) -> Row {
    Row {
        run_kind: "throughput".into(),
        protocol: Protocol {
            data_snapshot_hash: "d".repeat(64),
            tokenizer_hash: "t".repeat(64),
            backbone_commit: "b".repeat(40),
            recipe_hash: line["protocol"]["recipe_hash"]
                .as_str()
                .expect("recipe_hash")
                .into(),
            seed: 0,
        },
        status,
        quick_reason: "ledger-schema fixture".into(),
        code_commit: "c".repeat(40),
        env: Environment {
            torch: "n/a: Rust trainer, no torch in the process".into(),
            transformers_sha: "n/a: Rust trainer, no transformers in the process".into(),
            device: "metal".into(),
            host: "fixture-host".into(),
        },
        metrics: BTreeMap::from([("fixture.n".to_owned(), TriState::ran(true, 7))]),
        noul_rate: TriState::not_run("noul rate not computed by this run"),
        wall_clock_s: 1.5,
        wall_clock_source: WallClockSource::Recorder,
        notes: "ledger-schema fixture row".into(),
        recipe: json!({"tool": "tools/gen_ledger_schema_fixture.py", "kind": "throughput"}),
    }
}

fn stamp_of(line: &Value) -> Stamp {
    Stamp {
        row_id: line["row_id"].as_str().expect("row_id").into(),
        written_at: line["written_at"].as_str().expect("written_at").into(),
        prev_row_hash: line["prev_row_hash"].as_str().map(str::to_owned),
    }
}

#[test]
fn rust_writes_pythons_line_for_a_row_with_and_without_the_new_fields() {
    let all = lines();
    let want = expected();
    let with: Value = serde_json::from_str(&all[4]).expect("json");
    let without: Value = serde_json::from_str(&all[5]).expect("json");

    let extras = RowExtras {
        val_without_train: Some(want["val_without_train"].clone()),
        train_manifest_sha256: want["train_manifest_sha256"].as_str().map(str::to_owned),
    };
    let got = fixture_row(&with, Status::Completed)
        .line_with(&stamp_of(&with), &extras)
        .expect("a valid row");
    assert_eq!(
        got, all[4],
        "the row recording both fields is Python's line"
    );

    let got = fixture_row(&without, Status::Completed)
        .line(&stamp_of(&without))
        .expect("a valid row");
    assert_eq!(
        got, all[5],
        "a row recording neither omits both keys, as Python does"
    );
    assert!(!got.contains("val_without_train") && !got.contains("train_manifest_sha256"));
}

#[test]
fn the_writer_refuses_a_malformed_field_and_a_started_row() {
    let all = lines();
    let line: Value = serde_json::from_str(&all[5]).expect("json");
    let row = fixture_row(&line, Status::Completed);
    let stamp = stamp_of(&line);
    for extras in [
        RowExtras {
            train_manifest_sha256: Some("ABC".into()),
            ..RowExtras::default()
        },
        RowExtras {
            val_without_train: Some(json!(["f"])),
            ..RowExtras::default()
        },
        RowExtras {
            val_without_train: Some(json!({"families": ["f"]})),
            ..RowExtras::default()
        },
    ] {
        assert!(row.line_with(&stamp, &extras).is_err(), "{extras:?}");
    }
    let err = fixture_row(&line, Status::Started)
        .line(&stamp)
        .expect_err("this crate writes no started row");
    assert!(err.to_string().contains("started"), "{err}");
}
