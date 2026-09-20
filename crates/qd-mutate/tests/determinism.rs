//! Determinism, asserted end to end through the binary.
//!
//! `docs/mutation-operators.md`: *"Seeded with `rand_chacha` from an explicit seed recorded in the
//! manifest. The same seed and the same input pool produce byte-identical output. A test runs the
//! generator twice and compares hashes — a generator that is not reproducible cannot have its data
//! snapshot hashed, and the protocol hash in the ledger would be a fiction."*
//!
//! `src/generate.rs` has a unit test for the library path. This file runs the **binary**, twice,
//! writing real files, because the thing the ledger hashes is a file on disk and the steps between
//! the library and that file — JSONL serialisation, the manifest, the atomic write — are exactly
//! where a timestamp or a hash-map iteration order would get in.

use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_qd-mutate");

const RUST_FILE: &str = "\
use std::collections::HashMap;

pub fn total(a: usize, b: usize) -> Result<usize, String> {
    // a comment
    let mut acc0 = 0;
    for i in 0..10 {
        if a < b {
            acc0 += i;
        } else {
            acc0 -= i;
        }
    }
    let seen: HashMap<usize, usize> = HashMap::new();
    println!(\"{acc0} {seen:?}\");
    Ok(acc0)
}
";

const GO_FILE: &str = "\
package main

import (
\t\"fmt\"
\t\"os\"
)

func Total(a int, b int) ([]byte, error) {
\tacc0 := 0
\tfor i := 0; i < 10; i++ {
\t\tif a < b {
\t\t\tacc0 += i
\t\t}
\t}
\tv, err := readIt(a, b)
\tif err != nil {
\t\treturn nil, err
\t}
\tfmt.Println(acc0, os.Args)
\treturn v, nil
}
";

const PY_FILE: &str = "\
import os
import sys


def total(a: int, b: int) -> int:
    \"\"\"Add things up.\"\"\"
    acc0 = 0
    for i in range(10):
        if a < b:
            acc0 += i
        else:
            acc0 -= i
    print(os.getpid(), sys.argv)
    return acc0
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qd-mutate-determinism-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn write_pool(dir: &Path) -> PathBuf {
    let pool = dir.join("pool.jsonl");
    let mut lines = String::new();
    for (index, (source, extension)) in [
        (RUST_FILE, "rs"),
        (GO_FILE, "go"),
        (PY_FILE, "py"),
        (RUST_FILE, "rs"),
        (PY_FILE, "py"),
        (GO_FILE, "go"),
    ]
    .into_iter()
    .enumerate()
    {
        let record = serde_json::json!({
            "id": format!("rec{index}"),
            "repo": "o/r",
            "path": format!("pkg{index}/file.{extension}"),
            "source": source,
        });
        lines.push_str(&serde_json::to_string(&record).expect("serialises"));
        lines.push('\n');
    }
    std::fs::write(&pool, lines).expect("writing the pool");
    pool
}

/// Run `qd-mutate generate` and return `(examples jsonl, manifest json, the digest it printed)`.
fn generate(dir: &Path, pool: &Path, tag: &str, seed: &str) -> (String, String, String) {
    let out = dir.join(format!("{tag}.jsonl"));
    let manifest = dir.join(format!("{tag}.manifest.json"));
    let output = Command::new(BIN)
        .args([
            "generate",
            "--pool",
            &pool.display().to_string(),
            "--out",
            &out.display().to_string(),
            "--manifest",
            &manifest.display().to_string(),
            "--seed",
            seed,
            "--clean-permille",
            "200",
        ])
        .output()
        .expect("the binary runs");
    assert!(
        output.status.success(),
        "generate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf-8");
    let printed_digest = stdout
        .split_whitespace()
        .last()
        .expect("the digest is the last token of stdout")
        .to_string();
    (
        std::fs::read_to_string(&out).expect("examples file"),
        std::fs::read_to_string(&manifest).expect("manifest file"),
        printed_digest,
    )
}

#[test]
fn two_runs_of_one_seed_produce_byte_identical_output_and_the_same_hash() {
    let dir = scratch("same-seed");
    let pool = write_pool(&dir);

    let (jsonl_a, manifest_a, digest_a) = generate(&dir, &pool, "a", "20260919");
    let (jsonl_b, manifest_b, digest_b) = generate(&dir, &pool, "b", "20260919");

    assert!(!jsonl_a.trim().is_empty(), "the run produced no examples at all");
    assert_eq!(digest_a, digest_b, "the two runs printed different digests");
    assert_eq!(jsonl_a, jsonl_b, "the two runs wrote different examples");
    assert_eq!(
        manifest_a, manifest_b,
        "the manifests differ — a wall clock or a hash-map order has got in"
    );

    // The digest the run printed is the digest of the file it wrote, not of something else.
    let recomputed = qd_mutate::manifest::sha256_hex(jsonl_a.as_bytes());
    assert_eq!(
        digest_a, recomputed,
        "the printed digest does not describe the emitted file"
    );

    // And the manifest names it, so a ledger row can cite the manifest alone.
    let manifest: serde_json::Value = serde_json::from_str(&manifest_a).expect("manifest parses");
    assert_eq!(
        manifest["examples_sha256"].as_str(),
        Some(digest_a.as_str())
    );
    assert_eq!(manifest["seed"].as_u64(), Some(20260919));
}

#[test]
fn a_different_seed_produces_different_output() {
    let dir = scratch("other-seed");
    let pool = write_pool(&dir);
    let (jsonl_a, _, digest_a) = generate(&dir, &pool, "a", "1");
    let (_, _, digest_b) = generate(&dir, &pool, "b", "2");
    assert!(!jsonl_a.trim().is_empty());
    assert_ne!(
        digest_a, digest_b,
        "the seed is recorded but is not reaching the choices"
    );
}

#[test]
fn the_manifest_carries_no_wall_clock_and_names_the_pool_it_read() {
    let dir = scratch("manifest-shape");
    let pool = write_pool(&dir);
    let (_, manifest_json, _) = generate(&dir, &pool, "a", "7");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_json).expect("parses");

    let pool_bytes = std::fs::read(&pool).expect("pool");
    assert_eq!(
        manifest["pool"]["sha256"].as_str(),
        Some(qd_mutate::manifest::sha256_hex(&pool_bytes).as_str()),
        "the manifest must name the exact pool bytes it read"
    );
    assert_eq!(manifest["pool"]["records"].as_u64(), Some(6));

    for forbidden in ["timestamp", "generated_at", "created_at"] {
        assert!(
            !manifest_json.contains(forbidden),
            "`{forbidden}` would make two runs of one seed differ"
        );
    }
}

#[test]
fn the_span_derivations_never_disagreed_over_the_whole_pool() {
    // The number that must stay at zero. Reported here rather than assumed, because a rising one
    // means the mutator's bookkeeping and the textual differ have drifted apart — and that is
    // invisible in every class metric.
    let dir = scratch("span-agreement");
    let pool = write_pool(&dir);
    let (_, manifest_json, _) = generate(&dir, &pool, "a", "31337");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_json).expect("parses");
    assert_eq!(
        manifest["totals"]["span_disagreements"].as_u64(),
        Some(0),
        "examples were dropped on span disagreement"
    );
    assert_eq!(
        manifest["totals"]["mutation_did_not_parse"].as_u64(),
        Some(0),
        "an operator produced output that does not parse"
    );
    assert!(
        manifest["totals"]["examples"].as_u64().unwrap_or(0) > 0,
        "a run that produced nothing proves nothing about agreement"
    );
}

#[test]
fn a_limit_is_honoured_and_an_unreadable_pool_is_a_failure_not_an_empty_run() {
    let dir = scratch("cli-contract");
    let pool = write_pool(&dir);
    let out = dir.join("limited.jsonl");
    let manifest = dir.join("limited.manifest.json");
    let status = Command::new(BIN)
        .args([
            "generate",
            "--pool",
            &pool.display().to_string(),
            "--out",
            &out.display().to_string(),
            "--manifest",
            &manifest.display().to_string(),
            "--seed",
            "5",
            "--limit",
            "2",
        ])
        .output()
        .expect("runs");
    assert!(status.status.success());
    let lines = std::fs::read_to_string(&out).expect("examples").lines().count();
    assert!(lines <= 2, "--limit 2 produced {lines} lines");

    // A pool that is not there must fail loudly rather than write an empty snapshot.
    let missing = Command::new(BIN)
        .args([
            "generate",
            "--pool",
            &dir.join("not-here.jsonl").display().to_string(),
            "--out",
            &dir.join("x.jsonl").display().to_string(),
            "--manifest",
            &dir.join("x.json").display().to_string(),
            "--seed",
            "1",
        ])
        .output()
        .expect("runs");
    assert!(
        !missing.status.success(),
        "a missing pool must not exit zero with an empty output file"
    );
    assert!(!dir.join("x.jsonl").exists(), "an empty snapshot was written anyway");
}
