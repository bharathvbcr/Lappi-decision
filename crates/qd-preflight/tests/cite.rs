//! `qd-cite`, driven as a binary over a throwaway repository root.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const BIN: &str = env!("CARGO_BIN_EXE_qd-cite");
// Built from parts: a gap-id-shaped literal in a tracked file is a citation to the repo-wide scanner.
const REAL: &str = concat!("GAP", "-REAL-ONE-2026-10-18");
const ABSENT: &str = concat!("GAP", "-NO-SUCH-THING-2026-10-18");
static N: AtomicU64 = AtomicU64::new(0);

fn root() -> PathBuf {
    let p = std::env::temp_dir().join(format!("qd-cite-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(p.join("ledger")).unwrap();
    std::fs::create_dir_all(p.join("HANDOFF")).unwrap();
    std::fs::create_dir_all(p.join("AUDIT/sub")).unwrap();
    std::fs::create_dir_all(p.join("evidence")).unwrap();
    std::fs::write(p.join("ledger/a.jsonl"), "{\"row_id\":\"6af73bef-aa56-49b3-8cdb-e0b0b56e9181\"}\n{\"row_id\":\"9a00cfd0-3490-44b3-8cc9-453a875f4965\"}\n").unwrap();
    std::fs::write(p.join("gaps.jsonl"), format!("{{\"id\":\"{REAL}\"}}\n")).unwrap();
    std::fs::write(p.join("evidence/run.log"), "x").unwrap();
    p
}

fn doc(root: &Path, rel: &str, text: &str) -> PathBuf {
    let p = root.join(rel);
    std::fs::write(&p, text).unwrap();
    p
}

fn run(root: &Path, files: &[&Path]) -> (Option<i32>, String) {
    let o = Command::new(BIN).arg("--root").arg(root).args(files).output().unwrap();
    (o.status.code(), String::from_utf8_lossy(&o.stdout).into_owned())
}

#[test]
fn an_uncited_measured_number_in_a_new_handoff_fails() {
    let r = root();
    let f = doc(&r, "HANDOFF/x-2026-10-18.md", "# T\n\nThe model scored 0.857 on val.\n");
    let (code, out) = run(&r, &[&f]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("x-2026-10-18.md:3") && out.contains("0.857"), "{out}");
    assert_eq!(out.trim_end().lines().last(), Some("RESULT: FAIL"));
}

#[test]
fn each_kind_of_citation_in_the_same_paragraph_passes() {
    let r = root();
    let f = doc(
        &r,
        "HANDOFF/y-2026-10-19.md",
        &format!("The score is 0.857 (14779/17254), row 6af73bef.\n\nPrecision 87.0% per {REAL}.\n\nECE 0.079 [V: evidence/run.log].\n\n| row | value |\n| --- | --- |\n| a | 0.5 [V: evidence/run.log] |\n"),
    );
    let (code, out) = run(&r, &[&f]);
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(out.trim_end().lines().last(), Some("RESULT: PASS"));
}

#[test]
fn a_table_row_needs_its_own_citation() {
    let r = root();
    let f = doc(&r, "HANDOFF/z-2026-10-19.md", "| a | b |\n| --- | --- |\n| x | 0.5 [V: evidence/run.log] |\n| y | 0.6 |\n");
    let (code, out) = run(&r, &[&f]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("z-2026-10-19.md:4") && !out.contains("z-2026-10-19.md:3"), "{out}");
}

#[test]
fn an_id_that_resolves_to_nothing_fails_even_beside_a_number() {
    let r = root();
    let f = doc(
        &r,
        "AUDIT/sub/w-2026-10-20.md",
        &format!("Score row 6af73bef gives 0.857 and row deadbeef1 is cited.\n\nSee {ABSENT} and [V: evidence/missing.log].\n"),
    );
    let (code, out) = run(&r, &[&f]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains(&format!("{ABSENT} is not in gaps.jsonl")), "{out}");
    assert!(out.contains("[V: evidence/missing.log] does not exist"), "{out}");
}

#[test]
fn dates_versions_shas_lines_and_identifiers_are_not_measured_numbers() {
    let r = root();
    let f = doc(
        &r,
        "HANDOFF/v-2026-10-18.md",
        "Built 2026-10-17 at 02:25Z with cargo 1.98.0 on Qwen3.5-2B-Base, commit f7cb935, see src/main.rs:123 and lines 5-8, 8xH100, choice-16, 23 families.\n",
    );
    let (code, out) = run(&r, &[&f]);
    assert_eq!(code, Some(0), "{out}");
}

#[test]
fn historical_files_are_not_flagged_and_the_count_says_so() {
    let r = root();
    let old = doc(&r, "HANDOFF/old-2026-10-06.md", "Scored 0.857 with nothing cited.\n");
    let undated = doc(&r, "HANDOFF/undated.md", "Scored 0.857 with nothing cited.\n");
    let (code, out) = run(&r, &[&old, &undated]);
    assert_eq!(code, Some(0), "{out}");
    assert!(out.contains("files in scope (dated >= 2026-10-18): 0; before the cutoff: 1; undated: 1"), "{out}");
}

#[test]
fn a_fenced_code_block_is_not_prose() {
    let r = root();
    let f = doc(&r, "HANDOFF/c-2026-10-18.md", "Run it:\n\n```\nqd --threshold 0.05 --n 14/17\n```\n");
    let (code, out) = run(&r, &[&f]);
    assert_eq!(code, Some(0), "{out}");
}

#[test]
fn a_root_with_no_ledger_is_not_run_never_a_pass() {
    let r = root();
    std::fs::remove_file(r.join("ledger/a.jsonl")).unwrap();
    let f = doc(&r, "HANDOFF/n-2026-10-18.md", "Scored 0.857.\n");
    let (code, out) = run(&r, &[&f]);
    assert_eq!(code, Some(3), "{out}");
    assert_eq!(out.trim_end().lines().last(), Some("RESULT: NOT_RUN"));
}
