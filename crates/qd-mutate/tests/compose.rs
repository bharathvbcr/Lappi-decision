//! `qd-mutate compose`, end to end through the binary over a synthetic pool.
//!
//! Each test builds the whole chain on disk -- pool, `generate` (the base corpus), a split map,
//! `compose --aux-pool-out`, `generate --clean-permille 0` (the auxiliary corpus), `compose` --
//! because the things the loader trusts (the composed diff's line numbers, the split of every
//! file, the manifest's digests) are written by that chain, not by any one function.
//!
//! The span check is independent of the walker the composer uses: a span endpoint's diff line,
//! minus its one-character prefix, must be the needle post-image's own line.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use qd_mutate::compose::{
    diff_line_of_after_line, diff_line_span, ComposeManifest, ComposedRow, CorpusRow,
    SpanRefusal, FILE_HEADER_LINES, SPLIT_MAP_SCHEMA,
};
use qd_mutate::manifest::sha256_hex;
use qd_mutate::ops::MutationClass;
use qd_mutate::span::LineSpan;

const BIN: &str = env!("CARGO_BIN_EXE_qd-mutate");
const REPOS: usize = 40;
const RECORDS: usize = 480;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qd-mutate-compose-{name}"));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clearing the scratch directory");
    }
    std::fs::create_dir_all(&dir).expect("creating the scratch directory");
    dir
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().expect("running qd-mutate")
}

fn ok(args: &[&str]) {
    let out = run(args);
    assert!(
        out.status.success(),
        "qd-mutate {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn s(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

fn repo_of(i: usize) -> String {
    format!("org/r{:02}", i % REPOS)
}

/// The split the test map gives a repo: 7 in 10 train, 2 val, 1 held out.
fn split_of(repo: &str) -> &'static str {
    let n: usize = repo.trim_start_matches("org/r").parse().expect("a test repo");
    match n % 10 {
        0..=6 => "train",
        7 | 8 => "val",
        _ => "heldout",
    }
}

/// One Python file before and after a small real-looking change. Every record differs, so no
/// two pristine diffs are equal, and the extra helpers vary the sizes.
fn record(i: usize) -> serde_json::Value {
    let mut prior = format!(
        "import os\n\n\ndef compute_{i}(a, b):\n    total = a + b + {i}\n    if a > b:\n        \
         return total - {i}\n    return total\n"
    );
    let mut source = format!(
        "import os\n\n\ndef compute_{i}(a, b):\n    total = a + b + {i}\n    total += len(os.sep)\n    \
         if a > b:\n        return total - {i}\n    return total\n"
    );
    // Each helper is changed too, so the pristine diffs run from one hunk to six and the
    // sizes spread.
    for k in 0..(i % 6) {
        let head = format!("\n\ndef helper_{i}_{k}(x, y):\n");
        let body = format!("    if x < y:\n        return x * {k}\n    return y - {i}\n");
        prior.push_str(&format!("{head}{body}"));
        source.push_str(&format!("{head}    y = y or {k}\n{body}"));
    }
    serde_json::json!({
        "id": format!("rec{i:04}"),
        "repo": repo_of(i),
        "path": format!("pkg{i}/mod_{i}.py"),
        "source": source,
        "prior_source": prior,
    })
}

struct Fixture {
    dir: PathBuf,
    pool: PathBuf,
    corpus: PathBuf,
    split_map: PathBuf,
    aux_pool: PathBuf,
    aux: PathBuf,
}

fn write_split_map(path: &Path, pool: &Path, repos: &[String]) {
    let bytes = std::fs::read(pool).expect("reading the pool");
    let records = bytes.split(|&b| b == b'\n').filter(|l| !l.is_empty()).count();
    let map: BTreeMap<&str, &str> = repos.iter().map(|r| (r.as_str(), split_of(r))).collect();
    let doc = serde_json::json!({
        "schema": SPLIT_MAP_SCHEMA,
        "split": {"seed": 20260919u64, "train_fraction": 0.9, "val_fraction": 0.05},
        "pool": {"file": "pool.jsonl", "sha256": sha256_hex(&bytes), "records": records},
        "repos": map,
    });
    std::fs::write(path, serde_json::to_vec(&doc).expect("serialising")).expect("writing the map");
}

/// The chain up to, not including, `compose`.
fn fixture(name: &str) -> Fixture {
    let dir = scratch(name);
    let pool = dir.join("pool.jsonl");
    let mut text = String::new();
    for i in 0..RECORDS {
        text.push_str(&serde_json::to_string(&record(i)).expect("serialising"));
        text.push('\n');
    }
    std::fs::write(&pool, text).expect("writing the pool");
    let corpus = dir.join("corpus");
    std::fs::create_dir_all(&corpus).expect("mkdir");
    ok(&[
        "generate", "--pool", s(&pool), "--out", s(&corpus.join("examples.jsonl")),
        "--manifest", s(&corpus.join("manifest.json")), "--seed", "0",
        "--clean-permille", "300", "--report", "false",
    ]);
    let split_map = dir.join("split-map.json");
    let repos: Vec<String> = (0..REPOS).map(repo_of).collect();
    write_split_map(&split_map, &pool, &repos);
    let aux_pool = dir.join("aux-pool.jsonl");
    ok(&[
        "compose", "--corpus", s(&corpus), "--pool", s(&pool), "--split-map", s(&split_map),
        "--aux-pool-out", s(&aux_pool),
    ]);
    let aux = dir.join("aux");
    std::fs::create_dir_all(&aux).expect("mkdir");
    ok(&[
        "generate", "--pool", s(&aux_pool), "--out", s(&aux.join("examples.jsonl")),
        "--manifest", s(&aux.join("manifest.json")), "--seed", "0",
        "--clean-permille", "0", "--report", "false",
    ]);
    Fixture { dir, pool, corpus, split_map, aux_pool, aux }
}

fn compose_args<'a>(f: &'a Fixture, out: &'a Path, aux: &'a Path) -> Vec<&'a str> {
    vec![
        "compose", "--corpus", s(&f.corpus), "--pool", s(&f.pool), "--split-map",
        s(&f.split_map), "--aux", s(aux), "--out", s(out), "--seed", "7",
        "--train-rows", "60", "--val-rows", "60", "--min-tokens", "500", "--max-tokens", "1500",
        "--hard-max-tokens", "1700", "--min-files", "3", "--max-files", "16",
        "--tolerance-permille", "200", "--floor-uses", "2", "--soft-max-uses", "20",
        "--max-filler-uses", "24",
    ]
}

fn compose_ok(f: &Fixture, name: &str) -> (Vec<ComposedRow>, ComposeManifest, Vec<u8>) {
    let out = f.dir.join(name);
    ok(&compose_args(f, &out, &f.aux));
    let bytes = std::fs::read(out.join("examples.jsonl")).expect("reading the rows");
    let rows = String::from_utf8(bytes.clone())
        .expect("UTF-8")
        .lines()
        .map(|l| serde_json::from_str(l).expect("a composed row"))
        .collect();
    let manifest = serde_json::from_slice(&std::fs::read(out.join("manifest.json")).expect("read"))
        .expect("a compose manifest");
    (rows, manifest, bytes)
}

fn corpus_rows(dir: &Path) -> HashMap<String, CorpusRow> {
    std::fs::read_to_string(dir.join("examples.jsonl"))
        .expect("reading a corpus")
        .lines()
        .map(|l| serde_json::from_str::<CorpusRow>(l).expect("a corpus row"))
        .map(|r| (r.id.clone(), r))
        .collect()
}

#[test]
fn every_span_endpoint_is_the_needle_post_image_line_inside_the_needle_block() {
    let f = fixture("span");
    let (rows, _, _) = compose_ok(&f, "out");
    let mut examples = corpus_rows(&f.corpus);
    examples.extend(corpus_rows(&f.aux));
    let mut mutated = 0;
    for row in rows.iter().filter(|r| r.class != MutationClass::Clean) {
        mutated += 1;
        let span = row.diff_span.expect("a mutated row carries diff_span");
        let lines: Vec<&str> = row.diff.split('\n').collect();
        let needle = &row.constituents[row.needle_index.expect("needle_index") as usize];
        assert_eq!(needle.role, "needle", "{}", row.id);
        assert!(
            span.start >= needle.first_line + FILE_HEADER_LINES && span.end <= needle.last_line,
            "{}: span {span} outside the needle body {}..{}",
            row.id,
            needle.first_line,
            needle.last_line
        );
        let example = &examples[needle.example_id.as_deref().expect("the needle names its example")];
        let after_span = example.span.expect("a mutated example has a span");
        assert_eq!(row.span, Some(after_span), "{}", row.id);
        assert_eq!(row.after, example.after, "{}", row.id);
        let after: Vec<&str> = example.after.split('\n').collect();
        for (diff_line, after_line) in [(span.start, after_span.start), (span.end, after_span.end)] {
            let text = lines[(diff_line - 1) as usize];
            assert!(
                text.starts_with('+') || text.starts_with(' '),
                "{}: endpoint {diff_line} is {text:?}",
                row.id
            );
            assert_eq!(
                &text[1..],
                after[(after_line - 1) as usize],
                "{}: diff line {diff_line} is not post-image line {after_line}",
                row.id
            );
        }
        // Each block opens with its three headers, then a hunk header.
        for c in &row.constituents {
            let at = |n: u32| lines[(n - 1) as usize];
            assert_eq!(at(c.first_line), format!("diff --git a/{} b/{}", c.path, c.path));
            assert_eq!(at(c.first_line + 1), format!("--- a/{}", c.path));
            assert_eq!(at(c.first_line + 2), format!("+++ b/{}", c.path));
            assert!(at(c.first_line + 3).starts_with("@@ -"), "{}", row.id);
        }
    }
    assert!(mutated > 40, "only {mutated} mutated rows: the fixture tests too little");
}

#[test]
fn rows_never_mix_splits_and_never_read_a_held_out_repo() {
    let f = fixture("splits");
    let (rows, manifest, _) = compose_ok(&f, "out");
    assert_eq!(manifest.totals.by_split.get("train"), Some(&60));
    assert_eq!(manifest.totals.by_split.get("val"), Some(&60));
    assert_eq!(manifest.mode, "train_and_val");
    for row in &rows {
        assert_eq!(split_of(&row.repo), row.split.as_str(), "{}", row.id);
        for c in &row.constituents {
            assert_ne!(split_of(&c.repo), "heldout", "{}: {}", row.id, c.pool_id);
            assert_eq!(split_of(&c.repo), row.split.as_str(), "{}: {}", row.id, c.pool_id);
        }
    }
    // The aux pool, too, holds no held-out record.
    for line in std::fs::read_to_string(&f.aux_pool).expect("read").lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("json");
        assert_ne!(split_of(v["repo"].as_str().expect("repo")), "heldout");
    }
}

#[test]
fn a_repo_missing_from_the_split_map_refuses_the_run_rather_than_defaulting_it() {
    let f = fixture("missing");
    let repos: Vec<String> = (0..REPOS).map(repo_of).filter(|r| r != "org/r03").collect();
    write_split_map(&f.split_map, &f.pool, &repos);
    let out = run(&compose_args(&f, &f.dir.join("out"), &f.aux));
    assert!(!out.status.success(), "compose accepted a map missing org/r03");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("never defaulted") && err.contains("org/r03"), "{err}");
}

#[test]
fn the_same_seed_composes_byte_identical_rows_and_manifest() {
    let f = fixture("determinism");
    let (_, _, a) = compose_ok(&f, "a");
    let (_, _, b) = compose_ok(&f, "b");
    assert_eq!(sha256_hex(&a), sha256_hex(&b));
    let ma = std::fs::read(f.dir.join("a/manifest.json")).expect("read");
    let mb = std::fs::read(f.dir.join("b/manifest.json")).expect("read");
    assert_eq!(ma, mb);
}

#[test]
fn every_universe_file_is_the_needle_once_and_pristine_in_every_other_appearance() {
    let f = fixture("roles");
    let (rows, manifest, _) = compose_ok(&f, "out");
    let mut examples = corpus_rows(&f.corpus);
    examples.extend(corpus_rows(&f.aux));
    let mutated_diffs: HashSet<&str> = examples
        .values()
        .filter(|e| e.class != MutationClass::Clean)
        .map(|e| e.diff.as_str())
        .collect();
    let mut needles: HashMap<&str, u64> = HashMap::new();
    let mut appearances: HashMap<&str, u64> = HashMap::new();
    for row in &rows {
        let lines: Vec<&str> = row.diff.split('\n').collect();
        let paths: HashSet<&str> = row.constituents.iter().map(|c| c.path.as_str()).collect();
        assert_eq!(paths.len(), row.constituents.len(), "{}: a path twice", row.id);
        for c in &row.constituents {
            *appearances.entry(c.pool_id.as_str()).or_insert(0) += 1;
            if c.role == "needle" {
                *needles.entry(c.pool_id.as_str()).or_insert(0) += 1;
            } else {
                let body = lines[(c.first_line + FILE_HEADER_LINES - 1) as usize..c.last_line as usize]
                    .join("\n")
                    + "\n";
                assert!(
                    !mutated_diffs.contains(body.as_str()),
                    "{}: filler {} is rendered mutated",
                    row.id,
                    c.pool_id
                );
            }
        }
    }
    // Every file that appears is a needle exactly once.
    for (id, n) in &appearances {
        assert_eq!(needles.get(id), Some(&1), "{id} appears {n} times, needle {:?}", needles.get(id));
    }
    for split in ["train", "val"] {
        let r = &manifest.splits[split].needle_rate;
        assert_eq!(r.violations, 0, "{split}");
        assert_eq!(r.files, manifest.splits[split].by_class.iter().filter(|(k, _)| *k != "clean").map(|(_, v)| v).sum::<u64>());
    }
}

#[test]
fn clean_rows_carry_no_span_no_needle_and_only_pristine_files() {
    let f = fixture("clean");
    let (rows, _, _) = compose_ok(&f, "out");
    let clean: Vec<&ComposedRow> = rows.iter().filter(|r| r.class == MutationClass::Clean).collect();
    assert!(!clean.is_empty(), "the fixture composed no clean row");
    for row in clean {
        assert!(row.diff_span.is_none() && row.needle_index.is_none() && row.span.is_none());
        assert!(row.constituents.iter().all(|c| c.role == "filler"), "{}", row.id);
    }
}

#[test]
fn a_clean_example_that_does_not_re_render_refuses_the_run() {
    let f = fixture("parity");
    let path = f.corpus.join("examples.jsonl");
    let text = std::fs::read_to_string(&path).expect("read");
    let mut tampered = String::new();
    let mut done = false;
    for line in text.lines() {
        let mut v: serde_json::Value = serde_json::from_str(line).expect("json");
        if !done && v["class"] == "clean" {
            let diff = v["diff"].as_str().expect("diff").replacen("total", "totl", 1);
            v["diff"] = serde_json::Value::String(diff);
            done = true;
        }
        tampered.push_str(&serde_json::to_string(&v).expect("json"));
        tampered.push('\n');
    }
    assert!(done, "the fixture holds no clean example");
    std::fs::write(&path, &tampered).expect("write");
    let mpath = f.corpus.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&mpath).expect("read")).expect("json");
    m["examples_sha256"] = serde_json::Value::String(sha256_hex(tampered.as_bytes()));
    std::fs::write(&mpath, serde_json::to_vec(&m).expect("json")).expect("write");
    // The aux pool is derived from the corpus, which still mutates the same files.
    let out = run(&compose_args(&f, &f.dir.join("out"), &f.aux));
    assert!(!out.status.success(), "compose accepted a clean diff it cannot re-render");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("do not re-render"), "{err}");
}

#[test]
fn an_auxiliary_corpus_generated_from_other_records_is_refused() {
    let f = fixture("aux");
    let other = f.dir.join("aux-other");
    std::fs::create_dir_all(&other).expect("mkdir");
    ok(&[
        "generate", "--pool", s(&f.pool), "--out", s(&other.join("examples.jsonl")),
        "--manifest", s(&other.join("manifest.json")), "--seed", "0",
        "--clean-permille", "0", "--report", "false",
    ]);
    let out = run(&compose_args(&f, &f.dir.join("out"), &other));
    assert!(!out.status.success(), "compose accepted an aux corpus over the whole pool");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("never-mutated records"), "{err}");
}

#[test]
fn the_walk_matches_the_python_adapter_on_its_cases() {
    let diff = "@@ -1,3 +1,4 @@\n a\n-b\n+B\n+c\n d\n@@ -10,2 +11,2 @@\n x\n+y\n";
    assert_eq!(diff_line_of_after_line(diff, 1), Ok(2));
    assert_eq!(diff_line_of_after_line(diff, 2), Ok(4));
    assert_eq!(diff_line_of_after_line(diff, 3), Ok(5));
    assert_eq!(diff_line_of_after_line(diff, 4), Ok(6));
    assert_eq!(diff_line_of_after_line(diff, 11), Ok(8));
    assert_eq!(diff_line_of_after_line(diff, 12), Ok(9));
    assert_eq!(diff_line_of_after_line(diff, 7), Err(SpanRefusal::OutsideDiff));
    assert_eq!(diff_line_of_after_line("", 1), Err(SpanRefusal::EmptyDiff));
    assert!(matches!(diff_line_of_after_line(diff, 0), Err(SpanRefusal::Malformed(_))));
    for bad in [
        "@@ -1 +x @@\n a\n",
        "@@ -1,1 +1,1\n a\n",
        " a\n@@ -1 +1 @@\n a\n",
        "@@ -1,2 +1,2 @@\n a\n\n b\n",
        "@@ -1 +1 @@\n\\ No newline at end of file\n",
    ] {
        assert!(
            matches!(diff_line_of_after_line(bad, 99), Err(SpanRefusal::Malformed(_))),
            "{bad:?} was not refused"
        );
    }
    // A span over a removed line keeps both ends on the new side.
    assert_eq!(diff_line_span(diff, LineSpan::new(1, 3)), Ok(LineSpan::new(2, 5)));
}

/// `compose_args` for a report-only corpus over the train corpus at `train`.
fn diag_args<'a>(f: &'a Fixture, out: &'a Path, train: &'a Path, diag: &'a str) -> Vec<&'a str> {
    let mut args: Vec<&str> = compose_args(f, out, &f.aux)
        .into_iter()
        .map(|a| if a == "60" { "0" } else { a })
        .collect();
    // --train-rows 0 --val-rows 0 above; val rows back, then the diag rows.
    let val = args.iter().position(|&a| a == "--val-rows").expect("--val-rows") + 1;
    args[val] = "60";
    args.extend(["--diag-rows", diag, "--diag-train-corpus", s(train)]);
    args
}

#[test]
fn diag_rows_probe_train_fillers_from_val_rows_and_never_reach_held_out() {
    let f = fixture("diag");
    let (train_rows, _, _) = compose_ok(&f, "train");
    let train = f.dir.join("train");
    let out = f.dir.join("slice");
    ok(&diag_args(&f, &out, &train, "8"));
    let rows: Vec<ComposedRow> = std::fs::read_to_string(out.join("examples.jsonl"))
        .expect("read")
        .lines()
        .map(|l| serde_json::from_str(l).expect("a composed row"))
        .collect();
    let manifest: ComposeManifest =
        serde_json::from_slice(&std::fs::read(out.join("manifest.json")).expect("read"))
            .expect("manifest");
    assert_eq!(manifest.mode, "report_only");
    let diag = manifest.diag.expect("a diag report");
    assert_eq!((diag.rows, diag.seen_filler, diag.unseen), (8, 4, 4));

    // What the train corpus showed: each file's filler count and its needle's example.
    let mut filler_uses: HashMap<&str, u64> = HashMap::new();
    let mut needle_of: HashMap<&str, &str> = HashMap::new();
    for row in train_rows.iter().filter(|r| r.split.as_str() == "train") {
        for c in &row.constituents {
            if c.role == "needle" {
                needle_of.insert(&c.pool_id, c.example_id.as_deref().expect("needle example"));
            } else {
                *filler_uses.entry(&c.pool_id).or_insert(0) += 1;
            }
        }
    }
    let val_rows: Vec<&ComposedRow> = rows.iter().filter(|r| r.diag_half.is_none()).collect();
    assert_eq!(val_rows.len(), 60);
    let val_files: HashSet<&str> = val_rows
        .iter()
        .flat_map(|r| r.constituents.iter().map(|c| c.pool_id.as_str()))
        .collect();
    let diag_rows: Vec<&ComposedRow> = rows.iter().filter(|r| r.diag_half.is_some()).collect();
    assert_eq!(diag_rows.len(), 8);
    for row in &diag_rows {
        assert!(row.id.starts_with("compose:diag:"), "{}", row.id);
        assert_eq!(row.split.as_str(), "val");
        assert_eq!(split_of(&row.repo), "val", "{}: anchored outside val", row.id);
        let needle = &row.constituents[row.needle_index.expect("needle") as usize];
        for c in row.constituents.iter().filter(|c| c.role == "filler") {
            assert_eq!(split_of(&c.repo), "val", "{}: filler {}", row.id, c.pool_id);
        }
        match row.diag_half.as_deref() {
            Some("seen_filler") => {
                assert_eq!(split_of(&needle.repo), "train", "{}", row.id);
                let uses = filler_uses.get(needle.pool_id.as_str()).copied().unwrap_or(0);
                assert!(uses >= 5, "{}: needle seen as a filler {uses} times", row.id);
                assert_eq!(row.needle_train_appearances, Some(uses));
                assert_ne!(
                    needle_of.get(needle.pool_id.as_str()).copied(),
                    needle.example_id.as_deref(),
                    "{}: the diag needle is the very mutation it trained on",
                    row.id
                );
            }
            Some("unseen") => {
                assert_eq!(split_of(&needle.repo), "val", "{}", row.id);
                assert_eq!(row.needle_train_appearances, Some(0));
                assert!(
                    !val_files.contains(needle.pool_id.as_str()),
                    "{}: the unseen needle is in a val row",
                    row.id
                );
            }
            other => panic!("{}: diag_half {other:?}", row.id),
        }
        for c in &row.constituents {
            assert_ne!(split_of(&c.repo), "heldout", "{}: {}", row.id, c.pool_id);
        }
    }
}

#[test]
fn diag_rows_are_refused_beside_train_rows_and_without_their_train_corpus() {
    let f = fixture("diag-refusals");
    compose_ok(&f, "train");
    let train = f.dir.join("train");
    // Beside train rows: a train needle in a val row of a training corpus.
    let a = f.dir.join("a");
    let mut args = compose_args(&f, &a, &f.aux);
    args.extend(["--diag-rows", "8", "--diag-train-corpus", s(&train)]);
    let out = run(&args);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("never composed into a corpus"));
    // Without the train corpus whose fillers they probe.
    let b = f.dir.join("b");
    let mut args = diag_args(&f, &b, &train, "8");
    args.truncate(args.len() - 2);
    let out = run(&args);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--diag-train-corpus"));
}
