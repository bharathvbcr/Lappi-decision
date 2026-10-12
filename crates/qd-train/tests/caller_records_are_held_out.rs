//! **Caller records cannot reach a training process** (CLAUDE.md rule 3, and the human's
//! "Synthetic only", 2026-10-06).
//!
//! Apps write what they saw and chose around a decision to
//! `$HOME/Library/Application Support/Lappi/heldout/caller-records/<app>/<day>.jsonl`
//! (`docs/caller-contract.md`; `qd_runtime::caller_record::STORE_RELATIVE_TO_HOME` holds the same
//! literal). The `heldout` segment is the whole mechanism: this test proves the door's existing
//! marker layer refuses that store with no caller-specific code, under the default config and with
//! no configured root that happens to cover it.

use std::path::{Path, PathBuf};

use qd_train::files::OsFiles;
use qd_train::held_out::{
    assert_jsonl_file_not_caller_records, assert_no_caller_records, assert_path_not_held_out,
    DataConfig, DoorError, GuardLimits, HeldOutLayer, DEFAULT_HELD_OUT_FAMILIES,
    DEFAULT_HELD_OUT_PATH_MARKERS, DEFAULT_SEED,
};

/// Pinned twice on purpose: the runtime's constant and this literal must move together.
const STORE_RELATIVE_TO_HOME: &str = "Library/Application Support/Lappi/heldout/caller-records";

fn default_config() -> DataConfig {
    DataConfig::new(
        DEFAULT_HELD_OUT_FAMILIES.iter().map(|s| s.to_string()).collect(),
        // A root nowhere near the store, so only the marker layer can refuse it.
        vec![PathBuf::from("data/heldout")],
        DEFAULT_HELD_OUT_PATH_MARKERS.iter().map(|s| s.to_string()).collect(),
        DEFAULT_SEED,
    )
    .expect("the default config is valid")
}

#[test]
fn every_app_s_record_file_is_refused_by_the_marker_layer() {
    let config = default_config();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for app in ["devcouncil", "devtype", "gitpulse", "any-future-app"] {
        let path = PathBuf::from("/Users/someone")
            .join(STORE_RELATIVE_TO_HOME)
            .join(app)
            .join("2026-10-06.jsonl");
        let refused = assert_path_not_held_out(&path, &config, &repo_root);
        assert!(refused.is_err(), "{} must be refused by the training door", path.display());
        let message = refused.unwrap_err().to_string().to_ascii_lowercase();
        assert!(message.contains("rule 3") && message.contains("marker"), "{message}");
    }
}

/// The marker is blind to a copy: the path guard passes it. The content guard is what refuses it.
#[test]
fn a_record_file_copied_into_the_training_tree_is_refused_by_the_content_guard() {
    let config = default_config();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = repo_root.join("fixtures/caller/valid-records.jsonl");
    let dir = scratch("copy");
    for rel in ["data/pool/caller-copy/2026-10-06.jsonl", "anywhere/else/rows.txt"] {
        let copied = dir.join(rel);
        std::fs::create_dir_all(copied.parent().unwrap()).unwrap();
        std::fs::copy(&fixture, &copied).unwrap();
        assert!(
            assert_path_not_held_out(&copied, &config, &dir).is_ok(),
            "the path layer cannot see this copy; that is why the content guard exists"
        );
        let err = assert_jsonl_file_not_caller_records(&copied, &OsFiles, GuardLimits::default())
            .expect_err("a copy of a caller-record file must be refused wherever it lives");
        assert!(
            matches!(&err, DoorError::HeldOut(v) if v.layer == HeldOutLayer::CallerRecord),
            "{err}"
        );
    }
}

fn scratch(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("qd-train-cr-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn layer_of(bytes: &[u8], limits: GuardLimits) -> Option<HeldOutLayer> {
    assert_no_caller_records(bytes, "x.jsonl", limits).err().map(|v| v.layer)
}

const REC: &str = r#"{"record":"lappi.caller_record","record_version":1,"kind":"decision"}"#;

#[test]
fn a_record_line_anywhere_in_the_file_refuses_the_whole_file() {
    let l = GuardLimits::default();
    let rows = |a: &str| format!("{{\"text\":\"a\"}}\n{a}\n{{\"text\":\"b\"}}\n");
    assert_eq!(layer_of(rows(REC).as_bytes(), l), Some(HeldOutLayer::CallerRecord));
    // a later version of the format, a leading-space key spelling, CRLF line ends
    let v2 = r#"{"record":"lappi.caller_record.v2"}"#;
    assert_eq!(layer_of(rows(v2).as_bytes(), l), Some(HeldOutLayer::CallerRecord));
    let crlf = format!("{{\"a\":1}}\r\n{REC}\r\n");
    assert_eq!(layer_of(crlf.as_bytes(), l), Some(HeldOutLayer::CallerRecord));
    // JSON escapes spell the same string; the parsed value is what is compared
    let escaped = r#"{"rec\u006frd":"lappi.caller_re\u0063ord"}"#;
    assert_eq!(layer_of(escaped.as_bytes(), l), Some(HeldOutLayer::CallerRecord));
    // the last line with no trailing newline
    assert_eq!(layer_of(REC.as_bytes(), l), Some(HeldOutLayer::CallerRecord));
}

#[test]
fn an_unparseable_line_beside_a_record_line_refuses_the_file_and_one_alone_does_not() {
    let l = GuardLimits::default();
    let beside = format!("not json at all\n{REC}\n");
    assert_eq!(layer_of(beside.as_bytes(), l), Some(HeldOutLayer::CallerRecord));
    // a record line damaged so it no longer parses is still a record line
    let damaged = "{\"record\":\"lappi.caller_record\",\"kind\":\n";
    assert_eq!(layer_of(damaged.as_bytes(), l), Some(HeldOutLayer::CallerRecord));
    // a byte-order mark in front of a record line breaks the parse and not the refusal
    let bom = format!("{}{REC}\n", '\u{feff}');
    assert_eq!(layer_of(bom.as_bytes(), l), Some(HeldOutLayer::CallerRecord));
    // garbage with no record line is not this guard's business
    assert_eq!(layer_of(b"not json at all\n\xff\xfe\n{\"a\":1}\n", l), None);
}

#[test]
fn other_records_and_other_shapes_are_not_caller_records() {
    let l = GuardLimits::default();
    for ok in [
        "",
        "\n\n",
        r#"{"record":"decisions.json"}"#,
        r#"{"record":7}"#,
        r#"{"text":"lappi.caller_record"}"#,
        r#"["lappi.caller_record"]"#,
    ] {
        assert_eq!(layer_of(ok.as_bytes(), l), None, "{ok}");
    }
}

#[test]
fn a_duplicated_record_key_cannot_hide_a_record_line() {
    // serde_json keeps the last duplicate; the text test still sees the prefix beside the key
    let dup = r#"{"record":"lappi.caller_record","record":"x"}"#;
    assert_eq!(layer_of(dup.as_bytes(), GuardLimits::default()), Some(HeldOutLayer::CallerRecord));
}

#[test]
fn the_budget_is_bounded_and_a_check_that_could_not_finish_is_a_refusal() {
    let tight = GuardLimits { line_bytes: 64, total_bytes: 256 };
    // over the total budget: refused as unexamined, never passed as clean
    let big = format!("{}\n", "{\"a\":1}\n".repeat(100));
    assert_eq!(layer_of(big.as_bytes(), tight), Some(HeldOutLayer::GuardIncomplete));
    // an over-long line is skipped unparsed, but a record line beside it still refuses
    let long = format!("{{\"pad\":\"{}\"}}\n{REC}\n", "x".repeat(200));
    assert_eq!(
        layer_of(long.as_bytes(), GuardLimits { line_bytes: 150, total_bytes: 4096 }),
        Some(HeldOutLayer::CallerRecord)
    );
    let only_long = format!("{{\"pad\":\"{}\"}}\n", "x".repeat(200));
    assert_eq!(
        layer_of(only_long.as_bytes(), GuardLimits { line_bytes: 150, total_bytes: 4096 }),
        None
    );
}

#[test]
fn a_file_over_the_byte_budget_is_refused_by_the_file_door() {
    let dir = scratch("budget");
    let path = dir.join("rows.jsonl");
    std::fs::write(&path, "{\"a\":1}\n".repeat(100)).unwrap();
    let err = assert_jsonl_file_not_caller_records(
        &path,
        &OsFiles,
        GuardLimits { line_bytes: 64, total_bytes: 128 },
    )
    .unwrap_err();
    assert!(
        matches!(&err, DoorError::HeldOut(v) if v.layer == HeldOutLayer::GuardIncomplete),
        "{err}"
    );
}

#[test]
fn the_manifest_door_runs_the_content_guard_before_it_parses() {
    let config = default_config();
    let dir = scratch("manifest");
    let manifest = dir.join("data/pool/manifest.json");
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::copy(repo_root.join("fixtures/caller/valid-records.jsonl"), &manifest).unwrap();
    let err = qd_train::held_out::open_training_manifest(&manifest, &config, &dir, false, &OsFiles)
        .expect_err("a record file offered as a manifest is refused");
    assert!(
        matches!(&err, DoorError::HeldOut(v) if v.layer == HeldOutLayer::CallerRecord),
        "refused for being a caller record, not for failing to parse: {err}"
    );
}

/// One oracle, three implementations: `fixtures/caller/guard-cases.json` is also run by
/// `python/qd_data/caller_records.py` (python/tests/test_caller_records_guard.py) and by
/// `crates/qd-prep/src/heldout.rs`.
#[test]
fn the_shared_oracle_cases_have_the_same_verdicts_here() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let doc: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repo_root.join("fixtures/caller/guard-cases.json")).unwrap(),
    )
    .unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 30, "the oracle shrank to {} cases", cases.len());
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let bytes = oracle_bytes(case);
        let limits = GuardLimits {
            line_bytes: case["line_bytes"].as_u64().map_or(1 << 20, |n| n as usize),
            total_bytes: case["total_bytes"].as_u64().unwrap_or(8 << 30),
        };
        let got = match assert_no_caller_records(&bytes, name, limits) {
            Ok(()) => "clean",
            Err(v) if v.layer == HeldOutLayer::CallerRecord => "record",
            Err(v) if v.layer == HeldOutLayer::GuardIncomplete => "unscannable",
            Err(v) => panic!("{name}: unexpected layer {:?}", v.layer),
        };
        assert_eq!(got, case["expect"].as_str().unwrap(), "{name}");
    }
}

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
        g["repeat"].as_str().unwrap().repeat(g["times"].as_u64().unwrap() as usize)
    };
    format!("{}{core}{}", g["first"].as_str().unwrap_or(""), g["then"].as_str().unwrap_or(""))
        .into_bytes()
}

/// Opening a FIFO for reading blocks with no writer, and it has no stable bytes: the file door
/// decides by stat and refuses it as unscannable rather than passing it unscanned.
#[test]
fn a_fifo_is_refused_as_unscannable_and_never_opened() {
    let dir = scratch("fifo");
    let fifo = dir.join("rows.jsonl");
    let made = std::process::Command::new("mkfifo").arg(&fifo).status().unwrap();
    assert!(made.success(), "mkfifo");
    let err = assert_jsonl_file_not_caller_records(&fifo, &OsFiles, GuardLimits::default())
        .expect_err("a FIFO has nothing to scan");
    assert!(
        matches!(&err, DoorError::HeldOut(v) if v.layer == HeldOutLayer::GuardIncomplete),
        "{err}"
    );
}

/// The text-mode readers split rows on a bare `\r`; a scan that splits on `\n` alone saw one
/// clean line and the readers then returned the record as a row.
#[test]
fn a_cr_separated_file_is_refused_not_scanned_as_one_clean_line() {
    let mut rows = vec![r#"{"a":1}"#; 600];
    rows.push(REC);
    let file = format!("{}\n", rows.join("\r"));
    assert_eq!(
        layer_of(file.as_bytes(), GuardLimits::default()),
        Some(HeldOutLayer::GuardIncomplete)
    );
}

/// Every character `str.splitlines()` splits on beyond `\n`. The readers split on `\n` alone; a
/// file re-separated with one of these must not be scanned as one clean line.
const SPLITLINES_SEPARATORS: [char; 9] = [
    '\r', '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{85}', '\u{2028}', '\u{2029}',
];

#[test]
fn a_record_file_re_separated_with_any_splitlines_separator_is_refused() {
    let filler = format!(r#"{{"pad":"{}"}}"#, "x".repeat(2000));
    for sep in SPLITLINES_SEPARATORS {
        let mut rows = vec![filler.as_str(); 600];
        rows.push(REC);
        let file = format!("{}\n", rows.join(sep.to_string().as_str()));
        assert_eq!(
            layer_of(file.as_bytes(), GuardLimits::default()),
            Some(HeldOutLayer::GuardIncomplete),
            "{sep:?}"
        );
    }
}

/// U+0085, U+2028 and U+2029 are legal raw inside a JSON string and are data; they are refused
/// only outside one.
#[test]
fn a_separator_inside_a_string_value_is_data_and_is_not_refused() {
    for sep in ['\u{85}', '\u{2028}', '\u{2029}'] {
        let row = serde_json::json!({"text": format!("a{sep}b")}).to_string();
        assert!(row.contains(sep), "serde_json writes {sep:?} raw");
        assert_eq!(layer_of(format!("{row}\n{row}\n").as_bytes(), GuardLimits::default()), None);
        let outside = format!("{row}{sep}{row}\n");
        assert_eq!(
            layer_of(outside.as_bytes(), GuardLimits::default()),
            Some(HeldOutLayer::GuardIncomplete)
        );
    }
}

/// A document that only quotes the record format is not a caller record.
#[test]
fn text_that_quotes_the_format_is_clean() {
    let l = GuardLimits::default();
    let quoting = format!(
        "const P: &str = \"lappi.caller_record\"; // \"record\"\nlet rec = r#\"{REC}\"#;\n{}\n",
        serde_json::json!({"text": format!("see {REC}")}),
    );
    assert_eq!(layer_of(quoting.as_bytes(), l), None);
    let nested = r#"{"meta":{"record":"lappi.caller_record"}}"#;
    assert_eq!(layer_of(nested.as_bytes(), l), None);
}
