//! A caller record placed in any request file `qd-prep` reads is refused by content, through
//! the CLI, wherever the file lives and whatever it is called (CLAUDE.md rule 3). The rule is
//! `qd_prep::heldout`'s, one oracle in `fixtures/caller/guard-cases.json`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("qd-prep-record-door-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn valid_records() -> Vec<u8> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/caller/valid-records.jsonl");
    std::fs::read(&fixture).expect("the repo's fixture of real record lines")
}

fn qd_prep(args: &[&str], paths: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qd-prep"))
        .args(args)
        .args(paths)
        .output()
        .expect("qd-prep runs")
}

fn refused_as_record(out: &Output, what: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "{what}: admitted a caller record: {stderr}"
    );
    assert!(
        stderr.contains("caller record"),
        "{what}: refused for another reason: {stderr}"
    );
}

/// Every request-reading subcommand (`read_input`), given a copy of the record fixture.
#[test]
fn every_request_subcommand_refuses_a_copied_record_file() {
    let dir = scratch("request");
    let input = dir.join("request.bin");
    std::fs::write(&input, valid_records()).unwrap();
    for command in ["minhash", "ngrams", "linfit", "lsh", "dedupe", "spancheck"] {
        let output = dir.join(format!("{command}.out"));
        let out = qd_prep(
            &[command, "--input"],
            &[&input, Path::new("--output"), &output],
        );
        refused_as_record(&out, command);
        assert!(
            !output.exists(),
            "{command} wrote a reply for a caller record"
        );
    }
    let out = Command::new(env!("CARGO_BIN_EXE_qd-prep"))
        .args(["containment", "--input"])
        .arg(&input)
        .arg("--out-dir")
        .arg(dir.join("containment-out"))
        .output()
        .expect("qd-prep runs");
    refused_as_record(&out, "containment");
    assert!(!dir.join("containment-out").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_clean_request_is_not_refused_as_a_record() {
    let dir = scratch("clean");
    let input = dir.join("request.bin");
    std::fs::write(&input, b"{\"not\":\"a record\"}\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_qd-prep"))
        .args(["spancheck", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(dir.join("o.bin"))
        .output()
        .expect("qd-prep runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("caller record"), "{stderr}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `decision-caps` reads three files; a record in any one of them is refused, and nothing is
/// written.
#[test]
fn decision_caps_refuses_a_record_in_any_of_its_three_inputs() {
    let dir = scratch("caps");
    let (rec, other) = (dir.join("rec.json"), dir.join("other.json"));
    std::fs::write(&rec, valid_records()).unwrap();
    std::fs::write(&other, b"{}\n").unwrap();
    for (i, (config, before, after)) in [
        (&rec, &other, &other),
        (&other, &rec, &other),
        (&other, &other, &rec),
    ]
    .into_iter()
    .enumerate()
    {
        let output = dir.join(format!("caps-{i}.json"));
        let out = Command::new(env!("CARGO_BIN_EXE_qd-prep"))
            .args(["decision-caps", "--config"])
            .arg(config)
            .arg("--before")
            .arg(before)
            .arg("--after")
            .arg(after)
            .arg("--output")
            .arg(&output)
            .output()
            .expect("qd-prep runs");
        refused_as_record(&out, &format!("decision-caps input {i}"));
        assert!(!output.exists());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
