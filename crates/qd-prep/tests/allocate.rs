//! `qd-prep allocate` (Fable's v6 ruling R1/R2, `AUDIT/v6-rulings-2026-10-08/`), through the
//! binary: N `qd-decisions/v1` pools in, one applied pool out.
//!
//! Scratch directories are keyed on the tag, the pid, the clock and a counter, and share no
//! prefix with any other test file's.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use qd_prep::sha256::sha256_hex;
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_qd-prep");

fn scratch(tag: &str) -> PathBuf {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "qd-prep-allocate-{tag}-{}-{nanos}-{}",
        std::process::id(),
        CALLS.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// One pool row in the shape `decisions::example_json` writes.
struct R<'a> {
    id: &'a str,
    family: &'a str,
    stratum: &'a str,
    split: &'a str,
    group: &'a str,
    licence: &'a str,
    context: &'a str,
}

fn row(r: &R<'_>) -> Value {
    json!({
        "id": r.id, "source_id": "test/source", "family_id": r.family, "stratum": r.stratum,
        "split": r.split, "group_key": r.group, "licence": r.licence,
        "context": format!("Which option fits?\n\n{}", r.context), "slot_name": "answer",
        "options": ["alpha", "beta"], "gold_option": "alpha", "gold_noul": false,
        "label_basis": "hard",
    })
}

/// `rows` written as a build pool under `dir/name`: examples.jsonl and a manifest naming its
/// sha256 and count. Returns the pool dir and the examples' sha256.
fn pool(dir: &Path, name: &str, rows: &[Value]) -> (PathBuf, String) {
    let p = dir.join(name);
    std::fs::create_dir_all(&p).expect("pool dir");
    let mut bytes = Vec::new();
    for r in rows {
        serde_json::to_writer(&mut bytes, r).expect("row");
        bytes.push(b'\n');
    }
    let sha = sha256_hex(&bytes);
    std::fs::write(p.join("examples.jsonl"), &bytes).expect("examples");
    let manifest = json!({
        "schema": "qd-decisions/v1", "mode": "build", "examples": rows.len(),
        "examples_sha256": sha,
        "decontamination": {"excluded_total": 0},
    });
    std::fs::write(
        p.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest"),
    )
    .expect("manifest");
    (p, sha)
}

/// A config over `pools` (`name -> (sha, draw)`) and `caps`.
fn config(dir: &Path, pools: &[(&str, &str, &str)], caps: Value, extra: Value) -> PathBuf {
    let mut v = json!({
        "schema": "qd-allocation-config/v1",
        "seed": 7,
        "val_cap_per_family": 1000,
        "val_floor_per_family": 1,
        "val_cap_total": 10000,
        "pools": pools.iter().map(|(n, sha, draw)| {
            ((*n).to_owned(), json!({"examples_sha256": sha, "draw": draw}))
        }).collect::<serde_json::Map<_, _>>(),
        "train_caps": caps,
        "refused_licences": {},
        "stated_zero_train_families": {},
    });
    if let (Value::Object(base), Value::Object(more)) = (&mut v, extra) {
        base.extend(more);
    }
    let path = dir.join("allocation.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&v).expect("config")).expect("config");
    path
}

fn run<P: AsRef<Path>>(config: &Path, pools: &[(&str, P)], out: &Path, threads: usize) -> Output {
    let mut cmd = Command::new(BIN);
    cmd.arg("allocate").arg("--config").arg(config);
    for (n, p) in pools {
        cmd.arg("--pool").arg(format!("{n}={}", p.as_ref().display()));
    }
    cmd.arg("--out")
        .arg(out)
        .arg("--threads")
        .arg(threads.to_string())
        .stdin(Stdio::null())
        .output()
        .expect("qd-prep runs")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A refusal: non-zero (not a panic's 101), nothing written, the reason on stderr.
fn refused(o: &Output, out: &Path) -> String {
    assert!(!o.status.success(), "not refused: {}", stderr(o));
    assert_ne!(o.status.code(), Some(101), "a panic: {}", stderr(o));
    assert!(!out.exists(), "a refusal wrote {}", out.display());
    let mut partial = out.as_os_str().to_owned();
    partial.push(".partial");
    assert!(!Path::new(&partial).exists(), "a refusal left a partial");
    stderr(o)
}

fn ok(o: &Output) {
    assert!(o.status.success(), "refused: {}", stderr(o));
}

fn lines(path: &Path) -> Vec<Vec<u8>> {
    std::fs::read(path)
        .expect("read")
        .split_inclusive(|b| *b == b'\n')
        .map(<[u8]>::to_vec)
        .collect()
}

fn manifest(out: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(out.join("manifest.json")).expect("manifest"))
        .expect("manifest json")
}

fn rows_of(out: &Path) -> Vec<Value> {
    lines(&out.join("examples.jsonl"))
        .iter()
        .map(|l| serde_json::from_slice(l).expect("row"))
        .collect()
}

/// `n` rows of one stratum and split, ids `PREFIX-i`, contexts distinct.
fn many(prefix: &str, family: &str, stratum: &str, split: &str, n: usize) -> Vec<Value> {
    (0..n)
        .map(|i| {
            let id = format!("{prefix}-{i}");
            row(&R {
                id: &id,
                family,
                stratum,
                split,
                group: &id,
                licence: "mit",
                context: &format!("row {id} of {stratum}"),
            })
        })
        .collect()
}

#[test]
fn a_pool_family_missing_from_the_cap_table_is_refused() {
    let d = scratch("missing-family");
    let mut rows = many("a", "fam.one", "fam.one/x", "train", 3);
    rows.extend(many("b", "fam.two", "fam.two/x", "train", 3));
    let (p, sha) = pool(&d, "p", &rows);
    let cfg = config(&d, &[("p", &sha, "capped")], json!({"fam.one": 2}), json!({}));
    let out = d.join("out");
    let err = refused(&run(&cfg, &[("p", &p)], &out, 1), &out);
    assert!(err.contains("fam.two"), "{err}");
}

#[test]
fn a_non_zero_cap_with_no_candidates_is_refused() {
    let d = scratch("empty-cap");
    let (p, sha) = pool(&d, "p", &many("a", "fam.one", "fam.one/x", "train", 3));
    let cfg = config(
        &d,
        &[("p", &sha, "capped")],
        json!({"fam.one": 2, "fam.ghost": 5}),
        json!({}),
    );
    let out = d.join("out");
    let err = refused(&run(&cfg, &[("p", &p)], &out, 1), &out);
    assert!(err.contains("fam.ghost"), "{err}");
}

#[test]
fn a_pool_not_in_the_config_or_in_it_and_not_given_is_refused() {
    let d = scratch("pool-names");
    let (p, sha) = pool(&d, "p", &many("a", "fam.one", "fam.one/x", "train", 3));
    let (q, _) = pool(&d, "q", &many("b", "fam.one", "fam.one/x", "train", 3));
    let caps = json!({"fam.one": 2});
    // Given and not configured.
    let cfg = config(&d, &[("p", &sha, "capped")], caps.clone(), json!({}));
    let out = d.join("out-extra");
    let err = refused(&run(&cfg, &[("p", &p), ("q", &q)], &out, 1), &out);
    assert!(err.contains('q'), "{err}");
    // Configured and not given.
    let cfg = config(
        &d,
        &[("p", &sha, "capped"), ("q", &sha, "capped")],
        caps.clone(),
        json!({}),
    );
    let out = d.join("out-missing");
    let err = refused(&run(&cfg, &[("p", &p)], &out, 1), &out);
    assert!(err.contains('q'), "{err}");
    // Given twice.
    let cfg = config(&d, &[("p", &sha, "capped")], caps, json!({}));
    let out = d.join("out-twice");
    refused(&run(&cfg, &[("p", &p), ("p", &p)], &out, 1), &out);
}

#[test]
fn a_pool_whose_examples_differ_from_the_pinned_sha256_is_refused() {
    let d = scratch("sha");
    let (p, _) = pool(&d, "p", &many("a", "fam.one", "fam.one/x", "train", 3));
    let cfg = config(
        &d,
        &[("p", &"0".repeat(64), "capped")],
        json!({"fam.one": 2}),
        json!({}),
    );
    let out = d.join("out");
    let err = refused(&run(&cfg, &[("p", &p)], &out, 1), &out);
    assert!(err.contains("sha256"), "{err}");
    // And a pool whose own manifest disagrees with its examples.
    let (p2, sha2) = pool(&d, "p2", &many("a", "fam.one", "fam.one/x", "train", 3));
    let mut m: Value =
        serde_json::from_slice(&std::fs::read(p2.join("manifest.json")).unwrap()).unwrap();
    m["examples_sha256"] = json!("1".repeat(64));
    std::fs::write(p2.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
    let cfg = config(&d, &[("p2", &sha2, "capped")], json!({"fam.one": 2}), json!({}));
    let out = d.join("out2");
    let err = refused(&run(&cfg, &[("p2", &p2)], &out, 1), &out);
    assert!(err.contains("manifest"), "{err}");
}

#[test]
fn a_pool_under_a_held_out_path_is_refused() {
    let d = scratch("heldout");
    let (p, sha) = pool(
        &d.join("heldout"),
        "p",
        &many("a", "fam.one", "fam.one/x", "train", 3),
    );
    let cfg = config(&d, &[("p", &sha, "capped")], json!({"fam.one": 2}), json!({}));
    let out = d.join("out");
    let err = refused(&run(&cfg, &[("p", &p)], &out, 1), &out);
    assert!(err.contains("held-out"), "{err}");
}

/// Two pools carry one row (same family, group and text): one is train, the other val. One
/// row comes out, the val one, and the drop is counted.
#[test]
fn two_pools_sharing_an_identity_key_yield_one_row_and_it_is_the_val_one() {
    let d = scratch("dedupe");
    let shared = |id: &str, split: &str| {
        row(&R {
            id,
            family: "fam.one",
            stratum: "fam.one/x",
            split,
            group: "g-shared",
            licence: "mit",
            context: "the one shared text",
        })
    };
    let mut a = many("a", "fam.one", "fam.one/x", "train", 4);
    a.push(shared("a-shared", "train"));
    let mut b = many("b", "fam.one", "fam.one/x", "train", 4);
    b.push(shared("b-shared", "val"));
    let (pa, sa) = pool(&d, "pa", &a);
    let (pb, sb) = pool(&d, "pb", &b);
    let cfg = config(
        &d,
        &[("pa", &sa, "capped"), ("pb", &sb, "capped")],
        json!({"fam.one": 100}),
        json!({}),
    );
    let out = d.join("out");
    ok(&run(&cfg, &[("pa", &pa), ("pb", &pb)], &out, 1));
    let rows = rows_of(&out);
    let shared: Vec<&Value> = rows
        .iter()
        .filter(|r| r["group_key"] == "g-shared")
        .collect();
    assert_eq!(shared.len(), 1, "{shared:?}");
    assert_eq!(shared[0]["split"], "val");
    assert_eq!(shared[0]["id"], "b-shared");
    assert_eq!(rows.len(), 9);
    let m = manifest(&out);
    assert_eq!(m["dedupe"]["dropped_total"], 1, "{}", m["dedupe"]);
    assert_eq!(m["dedupe"]["dropped_cross_pool"], 1, "{}", m["dedupe"]);
}

/// Family cap 25 over strata of 60, 30 and 10 train rows: 15, 7.5 and 2.5 exactly, floors 15,
/// 7 and 2, and the one row left goes to the larger remainder, ties to the earlier stratum
/// (b). Val is capped per family.
#[test]
fn caps_are_per_family_and_drawn_per_stratum_by_largest_remainder() {
    let d = scratch("caps");
    let mut rows = many("a", "fam.one", "fam.one/a", "train", 60);
    rows.extend(many("b", "fam.one", "fam.one/b", "train", 30));
    rows.extend(many("c", "fam.one", "fam.one/c", "train", 10));
    rows.extend(many("va", "fam.one", "fam.one/a", "val", 10));
    rows.extend(many("t", "fam.two", "fam.two/x", "train", 5));
    let (p, sha) = pool(&d, "p", &rows);
    let cfg = config(
        &d,
        &[("p", &sha, "capped")],
        json!({"fam.one": 25, "fam.two": 100}),
        json!({"val_cap_per_family": 4}),
    );
    let out = d.join("out");
    ok(&run(&cfg, &[("p", &p)], &out, 1));
    let rows = rows_of(&out);
    let count = |stratum: &str, split: &str| {
        rows.iter()
            .filter(|r| r["stratum"] == stratum && r["split"] == split)
            .count()
    };
    assert_eq!(count("fam.one/a", "train"), 15);
    assert_eq!(count("fam.one/b", "train"), 8);
    assert_eq!(count("fam.one/c", "train"), 2);
    assert_eq!(count("fam.one/a", "val"), 4);
    assert_eq!(count("fam.two/x", "train"), 5, "a cap over availability takes all");
    let m = manifest(&out);
    assert_eq!(m["allocation"]["state"], "applied");
    assert_eq!(m["rows_by_family_split"]["fam.one"]["train"], 25);
    assert_eq!(m["rows_by_family_split"]["fam.one"]["val"], 4);
    assert_eq!(m["caps_applied"]["fam.one"]["cap"], 25);
    assert_eq!(m["caps_applied"]["fam.one"]["strata"]["fam.one/b"]["taken"], 8);
}

/// An as-built pool's rows come out byte for byte, every one, splits unchanged, while a capped
/// pool beside it is drawn.
#[test]
fn an_as_built_pools_rows_pass_through_byte_identical() {
    let d = scratch("as-built");
    let mut v5 = many("v", "old.fam", "old.fam/x", "train", 7);
    v5.extend(many("vv", "old.fam", "old.fam/x", "val", 3));
    // Non-ASCII and escapes survive.
    v5.push(row(&R {
        id: "v-uni",
        family: "old.fam",
        stratum: "old.fam/y",
        split: "train",
        group: "g\u{e9}",
        licence: "cc-by-sa-4.0",
        context: "caf\u{e9} \"quoted\" \\ tab\there \u{1f600}",
    }));
    let (pv, sv) = pool(&d, "v5", &v5);
    let (pn, sn) = pool(&d, "new", &many("n", "new.fam", "new.fam/x", "train", 20));
    let cfg = config(
        &d,
        &[("v5", &sv, "as_built"), ("new", &sn, "capped")],
        json!({"new.fam": 5}),
        json!({}),
    );
    let out = d.join("out");
    ok(&run(&cfg, &[("v5", &pv), ("new", &pn)], &out, 1));
    let got = lines(&out.join("examples.jsonl"));
    let want = lines(&pv.join("examples.jsonl"));
    for l in &want {
        assert!(got.contains(l), "missing {}", String::from_utf8_lossy(l));
    }
    let as_built = got
        .iter()
        .filter(|l| want.contains(l))
        .count();
    assert_eq!(as_built, want.len());
    assert_eq!(got.len(), want.len() + 5);
    let m = manifest(&out);
    assert_eq!(m["examples"], want.len() + 5);
    assert_eq!(
        m["examples_sha256"],
        sha256_hex(&std::fs::read(out.join("examples.jsonl")).unwrap())
    );
}

/// R2: CSN rows under cc-by-sa-4.0 are refused and counted; the rest of the family is drawn.
#[test]
fn a_refused_licence_rows_are_dropped_and_counted() {
    let d = scratch("licence");
    let mut rows = many("c", "csn.func_match", "csn.func_match/go", "train", 6);
    rows.push(row(&R {
        id: "c-sa",
        family: "csn.func_match",
        stratum: "csn.func_match/go",
        split: "train",
        group: "sa/repo",
        licence: "cc-by-sa-4.0",
        context: "a share-alike function",
    }));
    let (p, sha) = pool(&d, "csn", &rows);
    let cfg = config(
        &d,
        &[("csn", &sha, "capped")],
        json!({"csn.func_match": 100}),
        json!({"refused_licences": {"csn.func_match": ["cc-by-sa-4.0"]}}),
    );
    let out = d.join("out");
    ok(&run(&cfg, &[("csn", &p)], &out, 1));
    let rows = rows_of(&out);
    assert_eq!(rows.len(), 6);
    assert!(rows.iter().all(|r| r["licence"] != "cc-by-sa-4.0"));
    let m = manifest(&out);
    assert_eq!(
        m["refusals"]["licence"]["csn.func_match"]["cc-by-sa-4.0"],
        1,
        "{}",
        m["refusals"]
    );
}

/// knowledge.multiple_choice does not come through a pool; its zero is stated and recorded.
#[test]
fn a_stated_zero_family_is_recorded_and_refused_if_a_pool_carries_it() {
    let d = scratch("stated-zero");
    let (p, sha) = pool(&d, "p", &many("a", "fam.one", "fam.one/x", "train", 3));
    let zero = json!({"stated_zero_train_families":
        {"knowledge.multiple_choice": "R2: MMLU at zero"}});
    let cfg = config(&d, &[("p", &sha, "capped")], json!({"fam.one": 2}), zero);
    let out = d.join("out");
    ok(&run(&cfg, &[("p", &p)], &out, 1));
    let m = manifest(&out);
    assert_eq!(
        m["stated_zero_train_families"]["knowledge.multiple_choice"],
        "R2: MMLU at zero"
    );
    // A pool that does carry it contradicts the statement.
    let (q, shq) = pool(
        &d,
        "q",
        &many("k", "knowledge.multiple_choice", "knowledge.multiple_choice/x", "train", 3),
    );
    let cfg = config(
        &d,
        &[("q", &shq, "as_built")],
        json!({}),
        json!({"stated_zero_train_families": {"knowledge.multiple_choice": "R2"}}),
    );
    let out = d.join("out2");
    let err = refused(&run(&cfg, &[("q", &q)], &out, 1), &out);
    assert!(err.contains("knowledge.multiple_choice"), "{err}");
}

#[test]
fn the_output_is_byte_identical_across_thread_counts() {
    let d = scratch("threads");
    let mut a = many("a", "fam.one", "fam.one/x", "train", 50);
    a.extend(many("av", "fam.one", "fam.one/x", "val", 9));
    a.extend(many("ay", "fam.one", "fam.one/y", "train", 17));
    let (pa, sa) = pool(&d, "pa", &a);
    let (pb, sb) = pool(&d, "pb", &many("b", "fam.two", "fam.two/x", "train", 40));
    let (pc, sc) = pool(&d, "pc", &many("c", "old.fam", "old.fam/x", "train", 11));
    let cfg = config(
        &d,
        &[("pa", &sa, "capped"), ("pb", &sb, "capped"), ("pc", &sc, "as_built")],
        json!({"fam.one": 30, "fam.two": 13}),
        json!({}),
    );
    let pools = [("pa", &pa), ("pb", &pb), ("pc", &pc)];
    let one = d.join("one");
    let four = d.join("four");
    ok(&run(&cfg, &pools, &one, 1));
    ok(&run(&cfg, &pools, &four, 4));
    for f in ["examples.jsonl", "manifest.json"] {
        assert_eq!(
            std::fs::read(one.join(f)).unwrap(),
            std::fs::read(four.join(f)).unwrap(),
            "{f} differs across --threads"
        );
    }
}
