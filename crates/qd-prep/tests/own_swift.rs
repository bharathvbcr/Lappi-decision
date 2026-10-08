//! `qd-prep own-swift` end to end, on git repositories built in a temp directory: the manifest
//! comes from `qd-prep own-repos` itself, so the split is the real rule's, never a fixture's.
//! Every git call ignores the host's configuration and pins its dates.
//!
//! The round trip through `qd-mutate generate` needs that binary, which this crate cannot link
//! (tree-sitter is C; see Cargo.toml). It is `#[ignore]`d so a plain `cargo test` reports it as
//! ignored -- never as passed -- and it runs with `--ignored` and `QD_MUTATE_BIN` set, failing
//! loudly when the variable is missing.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_qd-prep");

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qd-prep-own-swift-{}-{name}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("clear scratch");
    }
    fs::create_dir_all(&dir).expect("make scratch");
    dir
}

fn git(dir: &Path, args: &[&str]) {
    let date = "2026-03-01T12:00:00+00:00";
    let out = Command::new("git")
        .args([
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            "-c",
            "core.autocrlf=false",
        ])
        .args(args)
        .current_dir(dir)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_AUTHOR_DATE", date)
        .env("GIT_COMMITTER_DATE", date)
        .output()
        .expect("run git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn init(dir: &Path, origin: &str) {
    fs::create_dir_all(dir).expect("mkdir repo");
    git(dir, &["init", "-q"]);
    git(dir, &["remote", "add", "origin", origin]);
}

fn write(dir: &Path, path: &str, text: &str) {
    let p = dir.join(path);
    fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
    fs::write(p, text).expect("write");
}

fn commit(dir: &Path) {
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "-q", "-m", "initial"]);
}

fn run(bin: &str, args: &[&str]) -> Output {
    Command::new(bin).args(args).output().expect("run binary")
}

fn read_json(p: &Path) -> Value {
    serde_json::from_slice(&fs::read(p).expect("read json")).expect("parse json")
}

const SWIFT: &str = "struct Item { let weight: Int }\n\nfunc totalWeight(_ items: [Item]) -> Int {\n    var total = 0\n    for it in items {\n        total += it.weight\n    }\n    if total > 100 {\n        return 100\n    }\n    return total\n}\n";

/// A code root with: a train repo (gusset) holding Swift, a non-Swift file and a symlink; the
/// held-out GitPulse holding Swift; and the train repo ScholarLM whose tree also tracks files
/// under `wisdev-arc/`, which is the held-out repository WisDev nested inside it.
fn fixture(root: &Path) -> (PathBuf, Value) {
    let code = root.join("Code");
    let gusset = code.join("devtools/gusset");
    init(&gusset, "https://github.com/bharathvbcr/gusset.git");
    write(&gusset, "Sources/Weights.swift", SWIFT);
    write(
        &gusset,
        "Sources/b/Other.swift",
        "func one() -> Int {\n    return 1\n}\n",
    );
    write(&gusset, "main.go", "package main\n\nfunc main() {}\n");
    std::os::unix::fs::symlink("Weights.swift", gusset.join("Sources/Link.swift"))
        .expect("symlink");
    commit(&gusset);
    // Uncommitted work never enters the pool: the pinned head is read, not the working tree.
    write(&gusset, "Sources/Uncommitted.swift", SWIFT);

    let gitpulse = code.join("devtools/GitPulse");
    init(&gitpulse, "https://github.com/bharathvbcr/GitPulse.git");
    write(&gitpulse, "Sources/Secret.swift", SWIFT);
    commit(&gitpulse);

    let scholarlm = code.join("scholarlm");
    init(&scholarlm, "https://github.com/bharathvbcr/ScholarLM.git");
    write(&scholarlm, "app/Ok.swift", SWIFT);
    write(&scholarlm, "wisdev-arc/Held.swift", SWIFT);
    commit(&scholarlm);
    init(
        &scholarlm.join("wisdev-arc"),
        "https://github.com/bharathvbcr/WisDev.git",
    );
    write(&scholarlm.join("wisdev-arc"), "x.swift", SWIFT);
    commit(&scholarlm.join("wisdev-arc"));

    let manifest = root.join("own-repos.json");
    let o = run(
        BIN,
        &[
            "own-repos",
            "--code-root",
            code.to_str().unwrap(),
            "--out",
            manifest.to_str().unwrap(),
        ],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let m = read_json(&manifest);
    let split = |repo: &str| {
        m["admitted"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["repo"] == repo)
            .unwrap()["split"]
            .as_str()
            .unwrap()
            .to_string()
    };
    // The rule's verdicts the test depends on, asserted rather than assumed.
    assert_eq!(split("devtools/GitPulse"), "heldout");
    assert_eq!(split("scholarlm/wisdev-arc"), "heldout");
    assert_eq!(split("devtools/gusset"), "train");
    assert_eq!(split("scholarlm"), "train");
    (manifest, m)
}

fn own_swift(manifest: &Path, out: &Path) -> Output {
    run(
        BIN,
        &[
            "own-swift",
            "--manifest",
            manifest.to_str().unwrap(),
            "--out-dir",
            out.to_str().unwrap(),
        ],
    )
}

fn pool_lines(out: &Path) -> Vec<Value> {
    fs::read_to_string(out.join("pool.jsonl"))
        .expect("read pool")
        .lines()
        .map(|l| serde_json::from_str(l).expect("parse record"))
        .collect()
}

#[test]
fn held_out_repositories_are_never_read_and_only_committed_swift_of_train_repos_is_emitted() {
    let s = scratch("heldout");
    let (manifest, _) = fixture(&s);
    let out = s.join("pool");
    let o = own_swift(&manifest, &out);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let recs = pool_lines(&out);
    let ids: Vec<&str> = recs.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(
        ids,
        [
            "own-swift:bharathvbcr/gusset:Sources/Weights.swift",
            "own-swift:bharathvbcr/gusset:Sources/b/Other.swift",
            "own-swift:bharathvbcr/scholarlm:app/Ok.swift",
        ]
    );
    for r in &recs {
        assert_ne!(r["repo"], "bharathvbcr/gitpulse");
        assert!(!r["path"].as_str().unwrap().starts_with("wisdev-arc/"));
        assert_eq!(
            r.as_object().unwrap().len(),
            4,
            "id, repo, path, source and nothing else"
        );
    }
    assert_eq!(recs[0]["source"], SWIFT);
    let m = read_json(&out.join("manifest.json"));
    assert_eq!(m["schema"], "qd-own-swift-pool/v1");
    assert_eq!(
        m["heldout_repos_not_read"],
        serde_json::json!(["devtools/GitPulse", "scholarlm/wisdev-arc"])
    );
    assert_eq!(
        m["per_repo"]["scholarlm"]["refusals"]["inside_nested_repository"],
        1
    );
    assert_eq!(
        m["per_repo"]["devtools/gusset"]["refusals"]["not_a_regular_file"],
        1
    );
    assert!(
        m["per_repo"]["devtools/GitPulse"].is_null(),
        "a held-out repo has no per-repo row"
    );
    assert_eq!(m["totals"]["records"], 3);
    let pool_bytes = fs::read(out.join("pool.jsonl")).unwrap();
    assert_eq!(
        m["pool"]["sha256"],
        qd_prep::sha256::sha256_hex(&pool_bytes)
    );
}

#[test]
fn a_manifest_that_calls_a_held_out_repository_train_is_refused() {
    let s = scratch("tampered");
    let (manifest, mut m) = fixture(&s);
    for a in m["admitted"].as_array_mut().unwrap() {
        if a["repo"] == "devtools/GitPulse" {
            a["split"] = Value::from("train");
        }
    }
    let tampered = s.join("tampered.json");
    fs::write(&tampered, serde_json::to_vec(&m).unwrap()).unwrap();
    let out = s.join("pool");
    let o = own_swift(&tampered, &out);
    assert!(
        !o.status.success(),
        "a held-out repo relabelled train must be refused"
    );
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("GitPulse"), "{err}");
    assert!(!out.exists(), "nothing is written on a refusal");
    let _ = manifest;
}

#[test]
fn the_pool_is_deterministic_and_never_overwritten() {
    let s = scratch("determinism");
    let (manifest, _) = fixture(&s);
    let (a, b) = (s.join("a"), s.join("b"));
    assert!(own_swift(&manifest, &a).status.success());
    assert!(own_swift(&manifest, &b).status.success());
    for f in ["pool.jsonl", "manifest.json"] {
        assert_eq!(
            fs::read(a.join(f)).unwrap(),
            fs::read(b.join(f)).unwrap(),
            "{f} differs"
        );
    }
    let again = own_swift(&manifest, &a);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("refusing to overwrite"));
}

#[test]
#[ignore = "needs the qd-mutate binary: run with --ignored and QD_MUTATE_BIN=<path>"]
fn the_pool_round_trips_through_qd_mutate_generate() {
    let qd_mutate = std::env::var("QD_MUTATE_BIN")
        .expect("QD_MUTATE_BIN must name a built qd-mutate binary for this test");
    let s = scratch("roundtrip");
    let (manifest, _) = fixture(&s);
    let out = s.join("pool");
    assert!(own_swift(&manifest, &out).status.success());
    let pool = out.join("pool.jsonl");
    let (ex, gm) = (s.join("examples.jsonl"), s.join("generate-manifest.json"));
    let o = run(
        &qd_mutate,
        &[
            "generate",
            "--pool",
            pool.to_str().unwrap(),
            "--out",
            ex.to_str().unwrap(),
            "--manifest",
            gm.to_str().unwrap(),
            "--seed",
            "20261008",
            "--language",
            "swift",
            "--clean-permille",
            "0",
            "--report",
            "false",
        ],
    );
    assert!(
        o.status.success(),
        "qd-mutate generate: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let g = read_json(&gm);
    assert_eq!(g["pool"]["records"], 3, "qd-mutate read every record");
    assert_eq!(
        g["pool"]["sha256"],
        read_json(&out.join("manifest.json"))["pool"]["sha256"]
    );
    assert_eq!(g["pool"]["by_language"]["swift"], 3);
    let mutated = g["totals"]["mutated"].as_u64().unwrap();
    assert!(
        mutated > 0,
        "a Swift function body yields mutated examples: {g}"
    );
    assert_eq!(g["diff"]["mutated_without_prior"], mutated);
    for line in fs::read_to_string(&ex).unwrap().lines() {
        let e: Value = serde_json::from_str(line).unwrap();
        assert_eq!(e["language"], "swift");
        assert_eq!(e["hunk_constrained"], false);
        assert!(e["repo"] != "bharathvbcr/gitpulse");
    }
}
