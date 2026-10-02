//! Every binary fixture qd-train reads is verified, whole, against the sha256 pins its set
//! carries. A fixture changed in place fails here, naming the file, as well as at the repository
//! gate.
//!
//! The human's decision of 2026-10-02 ("allow pinned fixtures only") lets a tracked binary
//! through `crates/qd-runtime/tests/tracked_source_is_text.rs` only if its sha256 is on record in
//! a manifest under its fixture root. The gate proves the digest is recorded *somewhere* there;
//! it cannot prove that the file at a path is the bytes recorded *for that path*. That is the
//! consumer's question, answered by `common::pins`. The same check runs again before each set is
//! first read: `common::fixture()` (`shards-tiny`), `span_head.rs`'s `fixture_root()`,
//! `adamw_decay_sensitive.rs`'s `pinned_manifest()` and `head_init.rs`'s `tiny_published()`.
//!
//! Pin coverage, by set. `SHA256SUMS` sets list every tracked file in the set, text as well as
//! binary, so a hand edit to one of their JSON files has to be re-pinned too:
//!
//! | set | manifest | written by |
//! | --- | --- | --- |
//! | `span-head/` | `SHA256SUMS` | `shasum -a 256` (lane L-fixture-pins, 2026-10-02) |
//! | `shards-tiny/` | `SHA256SUMS` | `shasum -a 256` (lane L-fixture-pins, 2026-10-02) |
//! | `tiny-published/` | `manifest.json` `files` | `tools/qd_train_oracle_tiny.py` |
//! | `adamw-decay-sensitive/` | `manifest.json` `files` | `tools/qd_train_oracle_adamw.py` |
//! | `span-head-init-seed0-h64.safetensors` | its `.manifest.json` `file` | `tools/qd_train_oracle_span_head_init.py` |

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use common::pins::{self, Form, Pin};
use serde_json::Value;

fn fixtures() -> PathBuf {
    common::crate_dir().join("tests").join("fixtures")
}

/// The fixture sets under `tests/fixtures`, each with the manifest form it carries.
const SETS: [(&str, Form); 4] = [
    ("span-head", Form::Sha256Sums),
    ("shards-tiny", Form::Sha256Sums),
    ("tiny-published", Form::JsonFiles),
    ("adamw-decay-sensitive", Form::JsonFiles),
];

/// The one binary fixture outside a set, and the manifest whose `file` entry pins it.
/// `head_init.rs` loads it through the trainer's own pinned loader (`load_span_head_pinned`).
const SINGLE: (&str, &str) = (
    "span-head-init-seed0-h64.safetensors",
    "span-head-init-seed0-h64.manifest.json",
);

/// The extensions the repository gate allows a binary fixture to have.
const FIXTURE_EXTENSIONS: [&str; 4] = ["npy", "npz", "safetensors", "u32"];

fn json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn every_fixture_set_matches_its_pins() {
    let mut failures = Vec::new();
    for (name, form) in SETS {
        match pins::check(&fixtures().join(name), form) {
            Ok(n) => println!(
                "{name}: {n} files verified against {}",
                form.manifest_name()
            ),
            Err(errors) => failures.push(format!("{name}:\n    {}", errors.join("\n    "))),
        }
    }
    assert!(
        failures.is_empty(),
        "fixture sets that do not match their pins:\n{}",
        failures.join("\n")
    );
}

#[test]
fn the_pinned_h64_init_matches_its_manifest() {
    let m = json(&fixtures().join(SINGLE.1));
    assert_eq!(
        m["file"]["path"].as_str(),
        Some(format!("crates/qd-train/tests/fixtures/{}", SINGLE.0).as_str()),
        "the manifest names this file"
    );
    let path = fixtures().join(SINGLE.0);
    let len = std::fs::metadata(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .len();
    assert_eq!(
        Some(len),
        m["file"]["bytes"].as_u64(),
        "{}: byte count",
        SINGLE.0
    );
    assert_eq!(
        Some(pins::sha256_file_hex(&path).as_str()),
        m["file"]["sha256"].as_str(),
        "{}: sha256",
        SINGLE.0
    );
}

/// The consumer side of the gate's rule. Every file under `tests/fixtures` with a fixture
/// extension lies in a set the tests above verify, or is the one singly-pinned init. A binary
/// added beside the sets, with no consumer-side check, fails here.
#[test]
fn every_binary_fixture_is_in_a_set_a_consumer_verifies() {
    let all = pins::files_under(&fixtures()).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let mut per_set: BTreeMap<&str, usize> = BTreeMap::new();
    let mut uncovered = Vec::new();
    for rel in &all {
        let ext = rel.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !FIXTURE_EXTENSIONS.contains(&ext) {
            continue;
        }
        let set = SETS
            .iter()
            .map(|(name, _)| *name)
            .find(|name| rel.starts_with(name))
            .or_else(|| (rel == Path::new(SINGLE.0)).then_some(SINGLE.0));
        match set {
            Some(name) => *per_set.entry(name).or_insert(0) += 1,
            None => uncovered.push(rel.display().to_string()),
        }
    }
    println!(
        "binary fixtures by set: {per_set:?} ({} files scanned under tests/fixtures)",
        all.len()
    );
    assert!(
        uncovered.is_empty(),
        "binary fixtures in no verified set; add them to a set's manifest (and to SETS here):\n  {}",
        uncovered.join("\n  ")
    );
    assert!(
        per_set.values().sum::<usize>() > 0,
        "no binary fixture was found at all; the walk is wrong"
    );
}

// -------------------------------------------------------------------------------------------------
// The check fails, and names the file, on each way a set can stop matching. Run on scratch
// copies: the tracked fixtures are never touched.
// -------------------------------------------------------------------------------------------------

/// A scratch copy of the fixture set `name`.
fn copy_of(name: &str, tag: &str) -> PathBuf {
    let from = fixtures().join(name);
    let to = common::scratch(tag).join(name);
    for rel in pins::files_under(&from).unwrap_or_else(|e| panic!("{}", e.join("\n"))) {
        let dst = to.join(&rel);
        std::fs::create_dir_all(dst.parent().expect("a file has a parent")).expect("mkdir");
        std::fs::copy(from.join(&rel), &dst).expect("copy");
    }
    assert_eq!(
        pins::check(&to, form_of(name)),
        pins::check(&from, form_of(name)),
        "a copy verifies as the original"
    );
    to
}

fn form_of(name: &str) -> Form {
    SETS.iter()
        .find(|(n, _)| *n == name)
        .map(|(_, f)| *f)
        .expect("a known set")
}

fn append_byte(path: &Path) {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .expect("open for append");
    f.write_all(b"\x00").expect("append");
}

fn errors(dir: &Path, form: Form) -> Vec<String> {
    pins::check(dir, form).expect_err("the tampered copy must not verify")
}

#[test]
fn one_byte_appended_to_a_sha256sums_pinned_fixture_fails_naming_it_and_both_digests() {
    for (set, rel) in [
        ("shards-tiny", "shards/train/tokens.u32"),
        ("span-head", "h2048-point/d_hidden.npy"),
    ] {
        let copy = copy_of(set, &format!("pins-append-{set}"));
        let pinned = pins::sha256_file_hex(&copy.join(rel));
        append_byte(&copy.join(rel));
        let now = pins::sha256_file_hex(&copy.join(rel));
        assert_eq!(
            errors(&copy, Form::Sha256Sums),
            vec![format!("{rel}: sha256 is {now}, SHA256SUMS pins {pinned}")],
            "{set}"
        );
        std::fs::remove_dir_all(copy.parent().expect("scratch")).expect("clean scratch");
    }
}

#[test]
fn one_byte_appended_to_a_json_pinned_fixture_fails_on_length_and_digest() {
    let copy = copy_of("tiny-published", "pins-append-json");
    let rel = "batches/tokens.u32";
    let before = std::fs::metadata(copy.join(rel)).expect("meta").len();
    append_byte(&copy.join(rel));
    let errs = errors(&copy, Form::JsonFiles);
    assert_eq!(errs.len(), 2, "{errs:?}");
    assert_eq!(
        errs[0],
        format!("{rel}: {} bytes, manifest.json pins {before}", before + 1)
    );
    assert!(
        errs[1].starts_with(&format!("{rel}: sha256 is ")),
        "{errs:?}"
    );
    std::fs::remove_dir_all(copy.parent().expect("scratch")).expect("clean scratch");
}

#[test]
fn a_file_the_manifest_does_not_list_fails_naming_it() {
    let copy = copy_of("shards-tiny", "pins-extra");
    std::fs::write(copy.join("door/extra.npy"), b"\x93NUMPY\x00").expect("write");
    let errs = errors(&copy, Form::Sha256Sums);
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].starts_with("door/extra.npy: under the set but not in SHA256SUMS"),
        "{errs:?}"
    );
    std::fs::remove_dir_all(copy.parent().expect("scratch")).expect("clean scratch");
}

#[test]
fn a_listed_file_that_is_gone_fails_naming_it() {
    let copy = copy_of("shards-tiny", "pins-gone");
    std::fs::remove_file(copy.join("door/supervision-deflated.npz")).expect("rm");
    assert_eq!(
        errors(&copy, Form::Sha256Sums),
        vec![
            "door/supervision-deflated.npz: listed in SHA256SUMS but not a file under the set"
                .to_owned()
        ]
    );
    std::fs::remove_dir_all(copy.parent().expect("scratch")).expect("clean scratch");
}

#[test]
fn a_set_without_its_manifest_fails() {
    let copy = copy_of("span-head", "pins-no-manifest");
    std::fs::remove_file(copy.join(pins::SHA256SUMS)).expect("rm");
    let errs = errors(&copy, Form::Sha256Sums);
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("cannot read SHA256SUMS"), "{errs:?}");
    std::fs::remove_dir_all(copy.parent().expect("scratch")).expect("clean scratch");
}

/// `verified` is what each consumer calls before reading: on a mismatch it panics with the
/// reason, and nothing is returned to read from.
#[test]
fn the_before_use_check_panics_with_the_reason() {
    let copy = copy_of("shards-tiny", "pins-before-use");
    append_byte(&copy.join("shards/val-report-only/offsets.npy"));
    let panic = std::panic::catch_unwind(|| pins::verified(&copy, Form::Sha256Sums))
        .expect_err("a tampered set must not be handed out");
    let message = panic
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| panic!("the panic carries a String message"));
    assert!(
        message.contains("shards/val-report-only/offsets.npy: sha256 is"),
        "{message}"
    );
    std::fs::remove_dir_all(copy.parent().expect("scratch")).expect("clean scratch");
}

#[cfg(unix)]
#[test]
fn a_symlink_in_a_set_is_refused() {
    let copy = copy_of("shards-tiny", "pins-symlink");
    std::os::unix::fs::symlink(
        copy.join("door/train-failed.json"),
        copy.join("door/link.json"),
    )
    .expect("symlink");
    let errs = errors(&copy, Form::Sha256Sums);
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].starts_with("door/link.json: not a regular file"),
        "{errs:?}"
    );
    std::fs::remove_dir_all(copy.parent().expect("scratch")).expect("clean scratch");
}

#[test]
fn a_sha256sums_outside_the_grammar_is_refused() {
    let d = pins::sha256_hex(b"x");
    for (why, body) in [
        ("one space", format!("{d} a.npy\n")),
        ("tab", format!("{d}\ta.npy\n")),
        ("upper case", format!("{}  a.npy\n", d.to_uppercase())),
        ("short digest", format!("{}  a.npy\n", &d[..63])),
        ("no path", format!("{d}  \n")),
        ("parent path", format!("{d}  ../a.npy\n")),
        ("dot path", format!("{d}  ./a.npy\n")),
        ("absolute path", format!("{d}  /a.npy\n")),
        ("escaped name", format!("\\{d}  a\\nb.npy\n")),
        ("crlf", format!("{d}  a.npy\r\n")),
        ("blank line", format!("{d}  a.npy\n\n{d}  b.npy\n")),
        ("empty", String::new()),
        ("only a newline", "\n".to_owned()),
        ("a bsd-style line", format!("SHA256 (a.npy) = {d}\n")),
        ("a path listed twice", format!("{d}  a.npy\n{d} *a.npy\n")),
    ] {
        assert!(
            pins::parse_sha256sums(body.as_bytes()).is_err(),
            "{why}: {body:?} must be refused"
        );
    }
    let accepted = pins::parse_sha256sums(format!("{d}  a.npy\n{d} *sub/b.npy").as_bytes())
        .expect("the accepted forms");
    let pin = Pin {
        sha256: d.clone(),
        bytes: None,
    };
    assert_eq!(
        accepted,
        BTreeMap::from([
            (PathBuf::from("a.npy"), pin.clone()),
            (PathBuf::from("sub/b.npy"), pin)
        ])
    );
}

#[test]
fn a_files_map_without_both_pins_is_refused() {
    let d = pins::sha256_hex(b"x");
    for (why, manifest) in [
        ("no files", serde_json::json!({})),
        ("files not an object", serde_json::json!({"files": []})),
        ("files empty", serde_json::json!({"files": {}})),
        (
            "no bytes",
            serde_json::json!({"files": {"a.npy": {"sha256": d}}}),
        ),
        (
            "no sha256",
            serde_json::json!({"files": {"a.npy": {"bytes": 1}}}),
        ),
        (
            "upper case",
            serde_json::json!({"files": {"a.npy": {"sha256": d.to_uppercase(), "bytes": 1}}}),
        ),
        (
            "negative bytes",
            serde_json::json!({"files": {"a.npy": {"sha256": d, "bytes": -1}}}),
        ),
        (
            "parent path",
            serde_json::json!({"files": {"../a.npy": {"sha256": d, "bytes": 1}}}),
        ),
    ] {
        assert!(
            pins::parse_files_map(&manifest).is_err(),
            "{why}: {manifest} must be refused"
        );
    }
    let ok =
        pins::parse_files_map(&serde_json::json!({"files": {"a.npy": {"sha256": d, "bytes": 1}}}))
            .expect("the accepted form");
    assert_eq!(
        ok,
        BTreeMap::from([(
            PathBuf::from("a.npy"),
            Pin {
                sha256: d,
                bytes: Some(1)
            }
        )])
    );
}
