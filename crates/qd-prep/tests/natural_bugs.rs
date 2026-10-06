//! `qd-prep own-repos` and `qd-prep natural-bugs` end to end, on git repositories built in a
//! temp directory by the test. Every git call ignores the host's configuration and pins its
//! dates, so the fixtures are the same on any machine.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_qd-prep");

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "qd-prep-natural-bugs-{}-{name}",
        std::process::id()
    ));
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("clear scratch");
    }
    fs::create_dir_all(&dir).expect("make scratch");
    dir
}

fn git(dir: &Path, args: &[&str], date: &str) {
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
        "git {args:?} in {}: {}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A repository at `root/rel` with the given remotes and no commits.
fn repo(root: &Path, rel: &str, remotes: &[(&str, &str)]) -> PathBuf {
    let dir = root.join(rel);
    fs::create_dir_all(&dir).expect("mkdir repo");
    git(&dir, &["init", "-q"], "2025-01-01T00:00:00+00:00");
    for (name, url) in remotes {
        git(
            &dir,
            &["remote", "add", name, url],
            "2025-01-01T00:00:00+00:00",
        );
    }
    dir
}

fn write(dir: &Path, path: &str, bytes: &[u8]) {
    let p = dir.join(path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(p, bytes).expect("write");
}

/// Stage everything and commit with `message` (subject, then body paragraphs) at `date`.
fn commit(dir: &Path, message: &[&str], date: &str) {
    git(dir, &["add", "-A"], date);
    let mut args = vec!["commit", "-q", "--allow-empty"];
    for m in message {
        args.push("-m");
        args.push(m);
    }
    git(dir, &args, date);
}

fn day(d: u32) -> String {
    format!("2026-03-{d:02}T12:00:00+00:00")
}

fn qd_prep(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().expect("run qd-prep")
}

fn read_json(p: &Path) -> Value {
    serde_json::from_slice(&fs::read(p).expect("read json")).expect("parse json")
}

fn own_repos(root: &Path, out: &Path) -> Value {
    let o = qd_prep(&[
        "own-repos",
        "--code-root",
        root.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    read_json(out)
}

fn excluded_reason(m: &Value, repo: &str) -> String {
    m["excluded"][repo]["reason"]
        .as_str()
        .unwrap_or_else(|| panic!("{repo} is not excluded: {}", m["excluded"]))
        .to_string()
}

#[test]
fn the_enumerator_applies_the_admission_rule_to_every_case() {
    let s = scratch("admission");
    let root = s.join("Code");
    let own = repo(
        &root,
        "apps/Own",
        &[("origin", "https://github.com/bharathvbcr/GitPulse.git")],
    );
    write(&own, "a.py", b"x = 1\n");
    commit(&own, &["initial"], &day(1));
    let train = repo(
        &root,
        "apps/Train",
        &[("origin", "git@github.com:bharathvbcr/gusset.git")],
    );
    write(&train, "a.py", b"x = 1\n");
    commit(&train, &["initial"], &day(1));
    repo(
        &root,
        "devtools/Fork",
        &[
            ("origin", "https://github.com/bharathvbcr/DevPrism.git"),
            ("delibae", "https://github.com/delibae/claude-prism.git"),
        ],
    );
    repo(
        &root,
        "devtools/Up",
        &[
            ("origin", "https://github.com/bharathvbcr/up.git"),
            ("upstream", "https://github.com/bharathvbcr/up-origin.git"),
        ],
    );
    repo(
        &root,
        "devtools/Other",
        &[("origin", "https://github.com/someone/else.git")],
    );
    repo(&root, "devtools/NoRemote", &[]);
    repo(
        &root,
        "devtools/Gitlab",
        &[("origin", "https://gitlab.com/bharathvbcr/x")],
    );
    repo(
        &root,
        "devtools/NoOrigin",
        &[("mine", "https://github.com/bharathvbcr/noorigin.git")],
    );
    repo(
        &root,
        "research/Lappi-decision",
        &[(
            "origin",
            "https://github.com/bharathvbcr/Lappi-decision.git",
        )],
    );
    repo(
        &root,
        "web/Lappi-BDay",
        &[("origin", "https://github.com/bharathvbcr/LAPPI.git")],
    );
    repo(
        &root,
        "apps/Empty",
        &[("origin", "https://github.com/bharathvbcr/empty-x.git")],
    );
    write(
        &root,
        "apps/Linked/.git",
        b"gitdir: /nonexistent/worktrees/linked\n",
    );
    repo(
        &root,
        "apps/node_modules/pkg",
        &[("origin", "https://github.com/bharathvbcr/nm.git")],
    );
    repo(
        &root,
        ".hidden/repo",
        &[("origin", "https://github.com/bharathvbcr/hidden.git")],
    );

    let m = own_repos(&root, &s.join("manifest.json"));

    let admitted: Vec<&str> = m["admitted"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["repo"].as_str().unwrap())
        .collect();
    assert_eq!(admitted, ["apps/Own", "apps/Train"]);
    let split = |r: &str| {
        m["admitted"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["repo"] == r)
            .unwrap()["split"]
            .clone()
    };
    assert_eq!(
        split("apps/Own"),
        "heldout",
        "bharathvbcr/gitpulse hashes to bucket 0"
    );
    assert_eq!(
        split("apps/Train"),
        "train",
        "bharathvbcr/gusset hashes to bucket 3"
    );
    let head = m["admitted"][0]["head"].as_str().unwrap();
    assert_eq!(head.len(), 40, "HEAD is pinned as a full sha");

    assert_eq!(
        excluded_reason(&m, "devtools/Fork"),
        "fork_or_other_owner:delibae"
    );
    assert_eq!(
        excluded_reason(&m, "devtools/Up"),
        "upstream_remote:https://github.com/bharathvbcr/up-origin.git"
    );
    assert_eq!(
        excluded_reason(&m, "devtools/Other"),
        "fork_or_other_owner:someone"
    );
    assert_eq!(excluded_reason(&m, "devtools/NoRemote"), "no_remote");
    assert_eq!(excluded_reason(&m, "devtools/Gitlab"), "non_github_remote");
    assert_eq!(excluded_reason(&m, "devtools/NoOrigin"), "no_origin_remote");
    assert_eq!(
        excluded_reason(&m, "research/Lappi-decision"),
        "excluded_by_the_human_tick"
    );
    assert_eq!(excluded_reason(&m, "web/Lappi-BDay"), "struck_by_the_human");
    assert_eq!(
        excluded_reason(&m, "apps/Linked"),
        "linked_worktree_or_submodule"
    );
    assert!(excluded_reason(&m, "apps/Empty").starts_with("no_head_commit:"));
    assert!(
        m["excluded"].get("apps/node_modules/pkg").is_none()
            && m["excluded"].get(".hidden/repo").is_none(),
        "node_modules and hidden trees are not discovered"
    );
    assert_eq!(m["totals"]["discovered"], 12);

    let diffs: Vec<&str> = m["v5_rule_comparison"]["differences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["repo"].as_str().unwrap())
        .collect();
    assert!(
        diffs.contains(&"devtools/Up"),
        "v5's rule admitted an upstream remote: {diffs:?}"
    );
    let missing = m["totals"]["human_exclusions_not_found"].to_string();
    assert!(missing.contains("web/WhimsicalLove"), "{missing}");

    // Written once: a second run onto the same file is refused.
    let again = qd_prep(&[
        "own-repos",
        "--code-root",
        root.to_str().unwrap(),
        "--out",
        s.join("manifest.json").to_str().unwrap(),
    ]);
    assert!(!again.status.success());
}

/// The library file the fixture's commits rewrite, as lines.
const LIB: [&str; 11] = [
    "def f(a):",
    "    return a - 1",
    "",
    "",
    "def g(b):",
    "    # a",
    "    return b * 2",
    "",
    "",
    "def h(c):",
    "    return c + 0",
];

fn put_lib(dir: &Path, lines: &[String]) {
    write(
        dir,
        "src/lib.py",
        format!("{}\n", lines.join("\n")).as_bytes(),
    );
}

#[test]
fn the_miner_keeps_only_single_statement_fixes_from_held_out_repos() {
    let s = scratch("mine");
    let root = s.join("Code");
    let own = repo(
        &root,
        "apps/Own",
        &[("origin", "https://github.com/bharathvbcr/GitPulse.git")],
    );
    let mut lib: Vec<String> = LIB.iter().map(|l| l.to_string()).collect();
    put_lib(&own, &lib);
    write(&own, "src/two.py", b"t = 2\n");
    write(&own, "src/other.py", b"y = 1\n");
    write(&own, "src/blob.py", b"x = 1\n\0\n");
    write(&own, "notes.py", b"a = 1\n");
    write(&own, "data.txt", b"d = 1\n");
    commit(&own, &["initial"], "2025-11-01T12:00:00+00:00");

    // Before the cutoff: a real single-statement fix, counted and never emitted.
    write(&own, "src/other.py", b"y = 2\n");
    commit(&own, &["fix old bug"], "2025-12-01T12:00:00+00:00");

    lib[1] = "    return a + 1".into();
    put_lib(&own, &lib);
    commit(&own, &["Fix off-by-one in f"], &day(2));

    lib[6] = "    return b * 3".into();
    lib[10] = "    return c + 1".into();
    put_lib(&own, &lib);
    commit(&own, &["fix: two spots"], &day(3));

    lib[6] = "    return b  * 3".into();
    put_lib(&own, &lib);
    commit(&own, &["fix spacing"], &day(4));

    lib[5] = "    # b".into();
    put_lib(&own, &lib);
    commit(&own, &["fix comment"], &day(5));

    git(&own, &["mv", "src/two.py", "src/three.py"], &day(6));
    commit(&own, &["fix: module name"], &day(6));

    write(&own, "src/blob.py", b"x = 2\n\0\n");
    commit(&own, &["fix blob"], &day(7));

    git(&own, &["checkout", "-q", "-b", "side"], &day(8));
    write(&own, "src/other.py", b"y = 3\n");
    commit(&own, &["tweak side"], &day(8));
    git(&own, &["checkout", "-q", "main"], &day(9));
    git(
        &own,
        &["merge", "-q", "--no-ff", "side", "-m", "Merge fix side"],
        &day(9),
    );

    commit(&own, &["fix nothing"], &day(10));

    write(&own, "src/other.py", b"y = 4\n");
    commit(&own, &["tweak other"], &day(11));

    write(&own, "notes.py", b"a = 2\n");
    commit(&own, &["fix notes"], &day(12));

    write(&own, "data.txt", b"d = 2\n");
    commit(&own, &["fix data"], &day(13));

    lib[1] = "    return a + 2".into();
    put_lib(&own, &lib);
    commit(
        &own,
        &[
            "fix sign in f",
            "Co-Authored-By: Claude <noreply@anthropic.com>",
        ],
        &day(14),
    );

    lib[10] = "    return c + 2".into();
    put_lib(&own, &lib);
    write(&own, "src/other.py", b"y = 5\n");
    commit(&own, &["fix both"], &day(15));

    lib.push("    pass".into());
    put_lib(&own, &lib);
    commit(&own, &["fix: add guard"], &day(16));

    // A train repository: never read by the miner, however fix-like its history.
    let train = repo(
        &root,
        "apps/Train",
        &[("origin", "https://github.com/bharathvbcr/gusset.git")],
    );
    write(&train, "a.py", b"x = 1\n");
    commit(&train, &["initial"], &day(1));
    write(&train, "a.py", b"x = 2\n");
    commit(&train, &["fix x"], &day(2));

    // A held-out repository named like scholarlm: its struck directory is never emitted.
    let sch = repo(
        &root,
        "scholarlm",
        &[("origin", "https://github.com/bharathvbcr/WisDev.git")],
    );
    write(&sch, "docs/business/plan.py", b"p = 1\n");
    commit(&sch, &["initial"], &day(1));
    write(&sch, "docs/business/plan.py", b"p = 2\n");
    commit(&sch, &["fix plan"], &day(2));

    let manifest = s.join("manifest.json");
    let m = own_repos(&root, &manifest);
    assert_eq!(
        m["totals"]["heldout_repos"],
        serde_json::json!(["apps/Own", "scholarlm"])
    );

    let v5 = s.join("files.jsonl");
    fs::write(
        &v5,
        b"{\"repo\":\"apps/Own\",\"path\":\"notes.py\",\"sha256\":\"x\",\"words\":1}\n",
    )
    .unwrap();
    let out = s.join("v6").join("heldout").join("natural-bugs");
    let o = qd_prep(&[
        "natural-bugs",
        "--manifest",
        manifest.to_str().unwrap(),
        "--cutoff",
        "2026-02-01",
        "--cutoff-basis",
        "test fixture",
        "--v5-files",
        v5.to_str().unwrap(),
        "--out-dir",
        out.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let report = read_json(&out.join("report.json"));

    let own_r = &report["per_repo"]["apps/Own"];
    assert_eq!(own_r["commits_scanned"], 17);
    assert_eq!(own_r["root_commits"], 1);
    assert_eq!(own_r["merge_commits"], 1);
    assert_eq!(own_r["non_merge_after_cutoff"], 14);
    assert_eq!(own_r["fix_like_all_dates"], 13);
    assert_eq!(own_r["fix_like_after_cutoff"], 12);
    assert_eq!(own_r["fix_like_after_cutoff_with_agent_trailer"], 1);
    assert_eq!(own_r["single_statement_fixes_after_cutoff"], 2);
    assert_eq!(own_r["single_statement_fixes_before_cutoff_not_emitted"], 1);
    assert_eq!(
        own_r["fix_like_after_cutoff_excluded_by_reason"],
        serde_json::json!({
            "binary": 1, "comment_only": 1, "empty_diff": 1, "language_not_admitted": 1,
            "multi_file": 1, "multi_hunk": 1, "not_one_line_each": 1, "rename_or_copy": 1,
            "v5_own_prose_file": 1, "whitespace_only": 1,
        })
    );
    assert_eq!(
        report["per_repo"]["scholarlm"]["fix_like_after_cutoff_excluded_by_reason"],
        serde_json::json!({"struck_path": 1})
    );
    assert!(
        report["per_repo"].get("apps/Train").is_none(),
        "a train repo is never mined"
    );
    assert_eq!(report["selected"]["rows"], 2);
    assert_eq!(report["selected"]["with_agent_coauthor_trailer"], 1);
    assert_eq!(
        report["selected"]["by_language"],
        serde_json::json!({"python": 2})
    );
    assert_eq!(
        report["target"]["met"],
        Value::Bool(false),
        "2 rows is short of 100"
    );
    assert_eq!(report["cutoff"]["basis"], "test fixture");

    let text = fs::read_to_string(out.join("natural-bugs.jsonl")).unwrap();
    let rows: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    let first = &rows[0];
    assert_eq!(first["repo"], "apps/Own");
    assert_eq!(first["path"], "src/lib.py");
    assert_eq!(first["subject"], "Fix off-by-one in f");
    assert_eq!(first["removed_text"], "    return a - 1");
    assert_eq!(first["added_text"], "    return a + 1");
    assert_eq!(first["changed_line_pre"], 2);
    assert!(
        first["mutation_class"].is_null(),
        "the class slot is unknown, not a value"
    );
    let pre = first["pre_content"].as_str().unwrap();
    assert_eq!(pre.split('\n').nth(1), Some("    return a - 1"));
    assert_eq!(rows[1]["agent_coauthor_trailer"], Value::Bool(true));
    assert_eq!(
        rows[1]["agent_trailer_names"],
        serde_json::json!(["anthropic", "claude"])
    );
    assert!(!out.with_extension("partial").exists());
}

#[test]
fn a_missing_cutoff_is_refused_by_the_parser() {
    let s = scratch("no-cutoff");
    let out = s.join("heldout").join("nb");
    let o = qd_prep(&[
        "natural-bugs",
        "--manifest",
        "/nonexistent/manifest.json",
        "--cutoff-basis",
        "x",
        "--v5-files",
        "/nonexistent/files.jsonl",
        "--out-dir",
        out.to_str().unwrap(),
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--cutoff"));
    assert!(!out.exists());
}

fn refused(out: &str, cutoff: &str) -> String {
    let o = qd_prep(&[
        "natural-bugs",
        "--manifest",
        "/nonexistent/manifest.json",
        "--cutoff",
        cutoff,
        "--cutoff-basis",
        "x",
        "--v5-files",
        "/nonexistent/files.jsonl",
        "--out-dir",
        out,
    ]);
    assert!(!o.status.success(), "{out} {cutoff} was accepted");
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn an_output_outside_a_held_out_path_or_a_malformed_cutoff_is_refused_before_any_read() {
    let s = scratch("refusals");
    let plain = s.join("v6").join("natural-bugs");
    let err = refused(plain.to_str().unwrap(), "2026-02-01");
    assert!(err.contains("no path segment"), "{err}");
    assert!(!plain.exists());
    let escape = format!("{}/heldout/../train/nb", s.display());
    assert!(refused(&escape, "2026-02-01").contains("'..'"));
    let ok_path = s.join("heldout").join("nb");
    assert!(refused(ok_path.to_str().unwrap(), "2026-02-30").contains("calendar date"));
    assert!(refused(ok_path.to_str().unwrap(), "Feb 2026").contains("YYYY-MM-DD"));
    assert!(!ok_path.exists());
}
