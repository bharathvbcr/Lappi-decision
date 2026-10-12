//! Both Rust safetensors readers, held to one list of adversarial files.
//!
//! The list is `tests/fixtures/safetensors-adversarial.json`; the Python reader
//! (`qd_train.memory.safetensors_header`) is held to the same file by `python/tests/test_memory.py`.
//!
//! Schema (version 1):
//!
//! ```text
//! { "schema_version": 1,
//!   "readme": "<prose>",
//!   "cases": [ {
//!       "name":               string, unique
//!       "why":                string
//!       "expect":             "ok" | "refuse"
//!       "header":             string   the header text, UTF-8 (default "")
//!       "header_encoding":    "utf-16-le"?  write the header text in that encoding instead of UTF-8
//!       "pad_header_to":      u64?     append spaces until the header is this many bytes
//!       "header_len":         u64?     the length prefix to write instead of the header's real length
//!       "data_len":           u64?     zero bytes after the header (default 0)
//!       "truncate_to":        u64?     cut the finished file to this many bytes
//!       "rust_error_contains": string? (refuse only) a substring both Rust readers' error holds
//!   } ] }
//! ```
//!
//! A file is: the u64-LE length prefix, the header bytes, `data_len` zero bytes. Python matches on
//! refuse/ok alone; the substring is the Rust readers' wording and is not part of the contract.
//!
//! `qd_metal::SafeTensors` is an adapter over `qd_export::SafeTensorsFile`, so besides refusing the
//! same files the two must give the same words: the adapter's error is the owner's `detail`.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use qd_export::safetensors::SafeTensorsFile;
use qd_metal::error::MetalError;
use qd_metal::safetensors::SafeTensors;
use serde_json::Value;

struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        // A leftover temp file is harmless and the test already has its answer.
        let _ignored = std::fs::remove_file(&self.0);
    }
}

fn u64_field(case: &Value, key: &str) -> Option<u64> {
    case.get(key).filter(|v| !v.is_null()).map(|v| v.as_u64().unwrap_or_else(|| panic!("{key} is not a u64")))
}

fn build(case: &Value, tag: &str) -> TempFile {
    let mut header = case.get("header").and_then(Value::as_str).unwrap_or("").to_owned();
    let utf16 = match case.get("header_encoding").and_then(Value::as_str) {
        None => false,
        Some("utf-16-le") => true,
        Some(other) => panic!("header_encoding {other:?} is not one this test writes"),
    };
    if let Some(n) = u64_field(case, "pad_header_to") {
        let n = usize::try_from(n).unwrap();
        assert!(header.len() <= n, "pad_header_to is shorter than the header");
        header.push_str(&" ".repeat(n - header.len()));
    }
    let header_bytes: Vec<u8> = if utf16 {
        header.encode_utf16().flat_map(u16::to_le_bytes).collect()
    } else {
        header.into_bytes()
    };
    let prefix = u64_field(case, "header_len").unwrap_or(header_bytes.len() as u64);
    let data_len = usize::try_from(u64_field(case, "data_len").unwrap_or(0)).unwrap();
    let path = std::env::temp_dir().join(format!("qd-st-adv-{}-{tag}.safetensors", std::process::id()));
    let mut f = File::create(&path).unwrap();
    f.write_all(&prefix.to_le_bytes()).unwrap();
    f.write_all(&header_bytes).unwrap();
    f.write_all(&vec![0u8; data_len]).unwrap();
    if let Some(n) = u64_field(case, "truncate_to") {
        f.set_len(n).unwrap();
    }
    TempFile(path)
}

fn cases() -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/safetensors-adversarial.json");
    let doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(doc["schema_version"], 1, "this test reads schema version 1");
    doc["cases"].as_array().unwrap().clone()
}

#[test]
fn both_rust_readers_refuse_and_accept_the_same_files_in_the_same_words() {
    let cases = cases();
    assert!(cases.len() >= 20, "the list shrank to {}", cases.len());
    let mut names = std::collections::BTreeSet::new();
    let (mut refused, mut accepted) = (0, 0);
    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        assert!(names.insert(name.to_owned()), "duplicate case {name}");
        let f = build(case, name);
        let owner = SafeTensorsFile::open(&f.0);
        let adapter = SafeTensors::open(&f.0);
        match case["expect"].as_str().unwrap() {
            "ok" => {
                accepted += 1;
                if let Err(e) = &owner {
                    failures.push(format!("{name}: qd-export refused: {e}"));
                }
                if let Err(e) = &adapter {
                    failures.push(format!("{name}: qd-metal refused: {e}"));
                }
            }
            "refuse" => {
                refused += 1;
                let (Err(owner_err), Err(adapter_err)) = (owner, adapter) else {
                    failures.push(format!("{name}: accepted by at least one reader"));
                    continue;
                };
                let MetalError::Weights(detail) = &adapter_err else {
                    failures.push(format!("{name}: qd-metal's refusal is {adapter_err:?}, not Weights"));
                    continue;
                };
                if detail != &owner_err.detail {
                    failures.push(format!("{name}: words differ: {detail:?} vs {:?}", owner_err.detail));
                }
                if let Some(needle) = case.get("rust_error_contains").and_then(Value::as_str)
                    && !detail.contains(needle)
                {
                    failures.push(format!("{name}: {needle:?} not in {detail:?}"));
                }
            }
            other => failures.push(format!("{name}: expect {other:?}")),
        }
    }
    assert!(failures.is_empty(), "{} of {} cases failed:\n{}", failures.len(), cases.len(), failures.join("\n"));
    assert!(refused >= 15 && accepted >= 3, "refused {refused}, accepted {accepted}");
}

/// qd-metal's three-dtype admission is its own: a file the owner reads, with a dtype the serving
/// loader does not load, is refused at open by qd-metal alone, with the loader's own words.
#[test]
fn qd_metal_alone_refuses_a_valid_dtype_it_does_not_load() {
    let case = serde_json::json!({
        "header": r#"{"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#,
        "data_len": 1,
    });
    let f = build(&case, "i8");
    SafeTensorsFile::open(&f.0).expect("the owner knows I8");
    let err = SafeTensors::open(&f.0).err().expect("qd-metal refuses I8").to_string();
    assert!(err.contains("dtype I8 is not one this reader loads (BF16, F16, F32)"), "{err}");
}
