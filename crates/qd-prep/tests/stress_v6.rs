//! The pre-registered data-side stress suite (`AUDIT/data-stress-suite-2026-10-06.md`), items
//! S1-S12, through the crate's public API and the `qd-prep` binary.
//!
//! Every test is named for its item (`s1_...` to `s12_...`), so a runner filters one item with
//! `cargo test -p qd-prep --test stress_v6 s4_`. The scale items (S9) are `#[ignore]`d and run
//! with `--ignored` in their own step, under `/usr/bin/time -l`.
//!
//! The universal criteria are asserted where an item can break them: U1 (a refusal is an error,
//! never a panic: the binary's 101 is checked on every refusal), U2 (a refusal leaves no
//! `out_dir` and no `.partial`), U4 (byte-identical outputs across thread counts) and U5 (an
//! input the code alters is counted under a named reason). Every bound is asserted as it is in
//! the source; none is moved here.
//!
//! Scratch directories are keyed on the test's own tag, the pid, the clock and a counter, never
//! on the pid alone (S7's static criterion), and share no prefix with any other test file's.

use std::collections::BTreeMap;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use qd_prep::decisions::{self, Candidate, Gold, TargetSet};
use qd_prep::sha256::sha256_hex;
use qd_prep::{containment, convert, convert_licence, pool, synth, synth_tools};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_qd-prep");

/// The bound on any one `qd-prep` run that is not a scale item.
const CLI_LIMIT: Duration = Duration::from_secs(1800);

/// The `mac_heavy` RSS cap (U3).
const RSS_CAP_BYTES: u64 = 32 << 30;

/// `synth_tools::MAX_FILE_BYTES` (private there): 16 MiB.
const TOOLS_FILE_BOUND: u64 = 16 << 20;

/// Text no generator writes, as a stand-in decontamination target. The shipped configs pin
/// targets that live outside the repository (`~/qd-campaign/...`), so each test pins its own.
const STAND_IN_TEXT: &str = "a stand-in decontamination target about lighthouse keepers \
                             counting migrating cranes over a frozen northern estuary at dawn";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn next() -> usize {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    CALLS.fetch_add(1, Ordering::SeqCst)
}

/// A fresh directory for one test: its tag, the pid, the clock and a counter.
fn scratch(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "qd-prep-stress-v6-{tag}-{}-{nanos}-{}",
        std::process::id(),
        next()
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn cleanup(dir: &Path) {
    std::fs::remove_dir_all(dir).expect("scratch cleanup");
}

fn s(p: &Path) -> String {
    p.display().to_string()
}

fn partial(out_dir: &Path) -> PathBuf {
    out_dir.with_extension("partial")
}

fn write_json(path: &Path, v: &Value) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("config parent");
    }
    let mut bytes = serde_json::to_vec_pretty(v).expect("json");
    bytes.push(b'\n');
    std::fs::write(path, bytes).expect("write json");
}

// ---------------------------------------------------------------------------------------------
// Running the binary, bounded.

fn spawn(args: &[String]) -> Child {
    Command::new(BIN)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("qd-prep spawns")
}

/// The child's output, or a panic naming the run after killing it at `limit`.
fn wait_bounded(mut child: Child, limit: Duration, what: &str) -> Output {
    let start = Instant::now();
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            return child.wait_with_output().expect("child output");
        }
        if start.elapsed() > limit {
            child.kill().expect("kill a run past its bound");
            let out = child.wait_with_output().expect("child output");
            panic!(
                "{what}: ran past {limit:?} and was killed; stderr: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn cli(args: &[String]) -> Output {
    wait_bounded(spawn(args), CLI_LIMIT, &format!("qd-prep {args:?}"))
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The error of a call that must be refused. `expect_err` would need `Debug` on every `Ok` type.
trait Refusal<E> {
    fn refusal(self, what: &str) -> E;
}

impl<T, E> Refusal<E> for Result<T, E> {
    fn refusal(self, what: &str) -> E {
        match self {
            Ok(_) => panic!("{what}"),
            Err(e) => e,
        }
    }
}

/// U1 and the refusal itself: non-zero, not a panic, naming `needle`.
fn assert_cli_refused(out: &Output, needle: &str, what: &str) {
    let err = stderr_of(out);
    assert_ne!(
        out.status.code(),
        Some(101),
        "{what}: qd-prep panicked (U1): {err}"
    );
    assert!(
        !out.status.success(),
        "{what}: accepted, not refused: {err}"
    );
    assert!(
        err.contains(needle),
        "{what}: refused without naming {needle:?}: {err}"
    );
}

/// U2: a refused run left no pool and no `.partial`.
fn assert_nothing_written(out_dir: &Path, what: &str) {
    assert!(
        !out_dir.exists(),
        "{what}: {} was written by a refused run (U2)",
        out_dir.display()
    );
    assert!(
        !partial(out_dir).exists(),
        "{what}: {} was left by a refused run (U2)",
        partial(out_dir).display()
    );
}

/// A pool a reader may take as complete: its examples hash to what its manifest records.
fn assert_complete_pool(out_dir: &Path) -> Value {
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(out_dir.join("manifest.json")).expect("manifest.json"),
    )
    .expect("manifest is JSON");
    let examples = std::fs::read(out_dir.join("examples.jsonl")).expect("examples.jsonl");
    assert_eq!(
        manifest["examples_sha256"].as_str(),
        Some(sha256_hex(&examples).as_str()),
        "{}: examples.jsonl is not the file its manifest records",
        out_dir.display()
    );
    let lines = examples.iter().filter(|b| **b == b'\n').count() as u64;
    assert_eq!(manifest["examples"].as_u64(), Some(lines));
    manifest
}

// ---------------------------------------------------------------------------------------------
// Synth configs and targets.

/// A decontamination target set written as `{"id", "text"}` JSONL: `(name, path, sha256)`.
fn write_target(dir: &Path, name: &str) -> (String, PathBuf, String) {
    std::fs::create_dir_all(dir).expect("target dir");
    let path = dir.join(format!("{name}.jsonl"));
    let bytes = format!(
        "{}\n",
        json!({"id": format!("{name}-1"), "text": STAND_IN_TEXT})
    );
    std::fs::write(&path, bytes.as_bytes()).expect("target file");
    (name.to_owned(), path, sha256_hex(bytes.as_bytes()))
}

fn pins_of(targets: &[(String, PathBuf, String)]) -> BTreeMap<String, String> {
    targets
        .iter()
        .map(|(n, _, sha)| (n.clone(), sha.clone()))
        .collect()
}

fn given_of(targets: &[(String, PathBuf, String)]) -> Vec<(String, PathBuf)> {
    targets
        .iter()
        .map(|(n, p, _)| (n.clone(), p.clone()))
        .collect()
}

fn target_args(targets: &[(String, PathBuf, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for (n, p, _) in targets {
        out.push("--target".to_owned());
        out.push(format!("{n}={}", s(p)));
    }
    out
}

/// A probe that fits in milliseconds and cannot fail its bound: for tests about something
/// other than the leak probe.
fn fast_probe() -> Value {
    json!({
        "n_min": 3, "n_max": 5, "dim": 16, "max_iter": 1, "tol": 0.0001, "lr": 0.05,
        "l2_grid": [0.1], "max_val_accuracy": 1.0
    })
}

/// An email config with the fast probe.
fn email_config(rows_per_template: u64, pins: &BTreeMap<String, String>) -> Value {
    json!({
        "schema": synth::CONFIG_SCHEMA, "kind": "email", "seed": 20261006u64,
        "rows_per_template": rows_per_template, "val_templates_per_class": 1,
        "injection_rate": 0.1, "ngram_n": 8, "containment_threshold": 0.5,
        "target_sha256": pins, "probe": fast_probe()
    })
}

/// A shipped `data/synth` config, as committed.
fn shipped(name: &str) -> Value {
    let path = repo().join("data/synth").join(name);
    serde_json::from_slice(&std::fs::read(&path).expect("shipped synth config"))
        .expect("shipped config is JSON")
}

/// A shipped config with its target pins replaced by `pins` and, for tools, its data files
/// named by absolute path (they resolve against the config's directory otherwise).
fn shipped_with(name: &str, pins: &BTreeMap<String, String>) -> Value {
    let mut v = shipped(name);
    v["target_sha256"] = json!(pins);
    if v.get("tools").is_some() {
        let data = repo().join("data/synth");
        for key in ["catalog", "phrasings"] {
            let file = v["tools"][key]
                .as_str()
                .expect("tools data file name")
                .to_owned();
            v["tools"][key] = json!(s(&data.join(file)));
        }
    }
    v
}

fn with(v: &Value, path: &[&str], x: Value) -> Value {
    let mut out = v.clone();
    let mut at = &mut out;
    for key in &path[..path.len() - 1] {
        at = &mut at[*key];
    }
    at[path[path.len() - 1]] = x;
    out
}

fn without(v: &Value, path: &[&str]) -> Value {
    let mut out = v.clone();
    let mut at = &mut out;
    for key in &path[..path.len() - 1] {
        at = &mut at[*key];
    }
    let key = path[path.len() - 1];
    let removed = at.as_object_mut().expect("an object").remove(key);
    assert!(
        removed.is_some(),
        "the base config has no {key:?} to remove"
    );
    out
}

/// `synth::run` on `cfg` must refuse naming `needle` and write nothing (U2).
fn synth_refuses(dir: &Path, cfg: &Value, targets: &[(String, PathBuf)], needle: &str, what: &str) {
    let n = next();
    let config = dir.join(format!("config-{n}.json"));
    write_json(&config, cfg);
    let out_dir = dir.join(format!("pool-{n}"));
    let inputs = synth::Inputs {
        config,
        targets: targets.to_vec(),
    };
    match synth::run(&inputs, &out_dir, 2) {
        Ok(line) => panic!("{what}: accepted, not refused: {line}"),
        Err(e) => assert!(
            e.contains(needle),
            "{what}: refused without naming {needle:?}: {e}"
        ),
    }
    assert_nothing_written(&out_dir, what);
}

fn synth_cli_args(
    config: &Path,
    targets: &[(String, PathBuf, String)],
    out_dir: &Path,
) -> Vec<String> {
    let mut args = vec![
        "synth".to_owned(),
        "--config".to_owned(),
        s(config),
        "--out-dir".to_owned(),
        s(out_dir),
    ];
    args.extend(target_args(targets));
    args
}

// ---------------------------------------------------------------------------------------------
// In-process rows.

fn cand(id: &str, context: &str, gold: Gold, options: &[&str]) -> Candidate {
    Candidate {
        id: id.to_owned(),
        source_id: "stress/source",
        family_id: "stress.family".to_owned(),
        stratum: "stress.family/x".to_owned(),
        group_key: id.to_owned(),
        licence: "mit".to_owned(),
        context: context.to_owned(),
        question: "Which?".to_owned(),
        slot_name: "answer".to_owned(),
        options: options.iter().map(|o| (*o).to_owned()).collect(),
        gold,
        label_basis: "hard",
        split: "train",
    }
}

/// Built through `Config::parse`, the boundary every real config crosses, so this source also
/// builds on the pre-fix tree for the red check (which lacks `heldout_templates_per_class`; the
/// key is optional and absent here, so it is 0). `kind` is set after parsing because the parser
/// requires a `tools` object for kind tools, and the tests here exercise tools with none.
fn synth_cfg(kind: synth::Kind) -> synth::Config {
    let doc = json!({
        "schema": synth::CONFIG_SCHEMA,
        "kind": "email",
        "seed": 20261006,
        "rows_per_template": 6,
        "val_templates_per_class": 1,
        "injection_rate": 0.0,
        "ngram_n": 8,
        "containment_threshold": 0.5,
        "target_sha256": {},
        "probe": {
            "n_min": 3,
            "n_max": 5,
            "dim": 1 << 14,
            "max_iter": 300,
            "tol": 1e-4,
            "lr": 0.05,
            "l2_grid": [1e-4, 1e-3],
            "max_val_accuracy": 0.95
        }
    });
    let mut cfg = synth::Config::parse(doc.to_string().as_bytes()).expect("the test config parses");
    cfg.kind = kind;
    cfg
}

fn draft(
    family: &'static str,
    template: &str,
    context: &str,
    options: Vec<String>,
    gold: Gold,
) -> synth::Draft {
    synth::Draft {
        family_id: family,
        stratum: format!("{family}/x"),
        template: template.to_owned(),
        template_class: "alpha".to_owned(),
        context: context.to_owned(),
        question: "Which?".to_owned(),
        slot_name: "answer",
        options,
        gold,
        fixed_options: false,
    }
}

fn abc() -> Vec<String> {
    vec!["alpha".into(), "beta".into(), "gamma".into()]
}

/// Four well-formed templates of one class, so `assemble`'s template floor is met.
fn base_drafts(family: &'static str) -> Vec<synth::Draft> {
    (0..4)
        .map(|i| {
            draft(
                family,
                &format!("t{i}"),
                &format!("an ordinary row number {i}"),
                abc(),
                Gold::Option(0),
            )
        })
        .collect()
}

fn scan(
    n: u32,
    threshold: f64,
    candidates: &[Candidate],
    targets: &[TargetSet],
) -> Result<Value, String> {
    pool::decontaminate(
        n,
        threshold,
        &json!({"tool": "qd-prep stress_v6"}),
        "stress-candidates",
        candidates,
        targets,
        2,
    )
    .map(|s| s.report)
}

fn words(n: usize, salt: &str) -> String {
    (0..n)
        .map(|i| format!("{salt}{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// =============================================================================================
// S1: JSONL readers. `decisions::for_lines` is the one line reader (crate-private); every target
// set reaches it through `pool::read_targets`, which is the door tested here.

/// The rows `pool::read_targets` reads from `bytes`, pinned by their own sha256.
fn read_target_bytes(dir: &Path, bytes: &[u8]) -> Result<usize, String> {
    let path = dir.join(format!("target-{}.jsonl", next()));
    std::fs::write(&path, bytes).expect("target file");
    let pins: BTreeMap<String, String> = [("t".to_owned(), sha256_hex(bytes))].into();
    pool::read_targets(&pins, &[("t".to_owned(), path)]).map(|(sets, _)| sets[0].1.len())
}

/// Refused, naming line `line` the way `for_lines` names one (`path:line:`).
fn assert_refused_at_line(got: Result<usize, String>, line: usize, what: &str) {
    match got {
        Ok(rows) => panic!("{what}: accepted silently ({rows} rows), not refused or counted"),
        Err(e) => assert!(
            e.contains(&format!(":{line}:")) || e.contains(&format!(":{line} ")),
            "{what}: refused without line {line}: {e}"
        ),
    }
}

/// Refused at `line`, or accepted and counted: the inputs map `read_targets` returns names the
/// fact under `line_facts/target/t` (`decisions::record_input`). Anything else is silent.
fn assert_refused_or_counted(dir: &Path, bytes: &[u8], line: usize, fact: &str, what: &str) {
    let path = dir.join(format!("target-{}.jsonl", next()));
    std::fs::write(&path, bytes).expect("target file");
    let pins: BTreeMap<String, String> = [("t".to_owned(), sha256_hex(bytes))].into();
    match pool::read_targets(&pins, &[("t".to_owned(), path)]) {
        Err(e) => assert!(
            e.contains(&format!(":{line}:")) || e.contains(&format!(":{line} ")),
            "{what}: refused without line {line}: {e}"
        ),
        Ok((_, inputs)) => {
            let counted = inputs.get("line_facts/target/t").map(String::as_str);
            assert!(
                counted.is_some_and(|c| c.contains(fact)),
                "{what}: accepted, and the inputs map does not count it ({fact}): {inputs:?}"
            );
        }
    }
}

const ROW_A: &str = r#"{"id":"a","text":"first row"}"#;
const ROW_B: &str = r#"{"id":"b","text":"second row"}"#;

/// One `{"id", "text"}` line of exactly `total` bytes, `\n` included (`for_lines` counts it).
fn line_of(total: usize, id: &str) -> Vec<u8> {
    let prefix = format!(r#"{{"id":"{id}","text":""#);
    let suffix = "\"}\n";
    let mut v = prefix.into_bytes();
    v.resize(total - suffix.len(), b'x');
    v.extend_from_slice(suffix.as_bytes());
    assert_eq!(v.len(), total);
    v
}

#[test]
fn s1_an_empty_target_file_is_refused() {
    let dir = scratch("s1-empty");
    let e = read_target_bytes(&dir, b"").expect_err("an empty target set was accepted");
    assert!(e.contains("empty"), "{e}");
    cleanup(&dir);
}

#[test]
fn s1_a_torn_last_line_is_refused_or_counted_never_accepted_silently() {
    let dir = scratch("s1-torn");
    let bytes = format!("{ROW_A}\n{ROW_B}");
    assert_refused_or_counted(
        &dir,
        bytes.as_bytes(),
        2,
        "torn_last_line=true",
        "torn last line",
    );
    cleanup(&dir);
}

#[test]
fn s1_crlf_line_endings_are_refused_or_counted_never_accepted_silently() {
    let dir = scratch("s1-crlf");
    let bytes = format!("{ROW_A}\r\n{ROW_B}\r\n");
    assert_refused_or_counted(&dir, bytes.as_bytes(), 1, "crlf_lines=2", "CRLF endings");
    cleanup(&dir);
}

#[test]
fn s1_a_utf8_bom_is_refused_or_counted_never_accepted_silently() {
    let dir = scratch("s1-bom");
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend_from_slice(format!("{ROW_A}\n").as_bytes());
    assert_refused_at_line(read_target_bytes(&dir, &bytes), 1, "UTF-8 BOM");
    cleanup(&dir);
}

#[test]
fn s1_non_utf8_bytes_in_a_string_are_refused_with_the_line() {
    let dir = scratch("s1-non-utf8");
    let mut bytes = format!("{ROW_A}\n").into_bytes();
    bytes.extend_from_slice(b"{\"id\":\"b\",\"text\":\"bad \xff\xfe byte\"}\n");
    assert_refused_at_line(read_target_bytes(&dir, &bytes), 2, "non-UTF-8 bytes");
    cleanup(&dir);
}

#[test]
fn s1_a_raw_nul_inside_a_string_is_refused_with_the_line() {
    let dir = scratch("s1-raw-nul");
    let mut bytes = format!("{ROW_A}\n").into_bytes();
    bytes.extend_from_slice(b"{\"id\":\"b\",\"text\":\"nul \x00 inside\"}\n");
    assert_refused_at_line(read_target_bytes(&dir, &bytes), 2, "raw NUL in a string");
    cleanup(&dir);
}

/// S1 scoped by role (AUDIT clarification ~21:30Z): a target row is a decontamination
/// reference, never dropped, so an escaped NUL in one is read and counted.
#[test]
fn s1_an_escaped_nul_in_a_target_row_is_read_and_counted_never_dropped() {
    let dir = scratch("s1-escaped-nul-target");
    let bytes = format!("{ROW_A}\n{}\n", r#"{"id":"b","text":"nul \u0000 inside"}"#);
    let path = dir.join(format!("target-{}.jsonl", next()));
    std::fs::write(&path, &bytes).expect("target file");
    let pins: BTreeMap<String, String> = [("t".to_owned(), sha256_hex(bytes.as_bytes()))].into();
    let (sets, inputs) = pool::read_targets(&pins, &[("t".to_owned(), path)])
        .expect("a target row holding a NUL refused the set");
    assert_eq!(sets[0].1.len(), 2, "a target row was dropped");
    let counted = inputs.get("line_facts/target/t").map(String::as_str);
    assert!(
        counted.is_some_and(|c| c.contains("nul_rows=1 accepted lines [2]")),
        "read, and not counted: {inputs:?}"
    );
    cleanup(&dir);
}

/// S1 scoped by role: a target line past MAX_LINE_BYTES is read, because a target row is never
/// dropped; its own bound is MAX_REFERENCE_LINE_BYTES.
#[test]
fn s1_a_target_line_past_max_line_bytes_is_read_not_dropped() {
    let dir = scratch("s1-line-bound-target");
    let at = read_target_bytes(&dir, &line_of(decisions::MAX_LINE_BYTES, "a"));
    assert_eq!(
        at,
        Ok(1),
        "a target line of exactly MAX_LINE_BYTES was not read"
    );
    let over = read_target_bytes(&dir, &line_of(decisions::MAX_LINE_BYTES + 1, "a"));
    assert_eq!(
        over,
        Ok(1),
        "a target line one past MAX_LINE_BYTES was not read"
    );
    cleanup(&dir);
}

/// S1 scoped by role: a source row holding an escaped NUL, or past MAX_LINE_BYTES, is dropped,
/// counted with its line, and the rest of the file is read. Checked through the view reader
/// every conversion uses; the counts are read from `Lines`' Debug form, so this source also
/// builds on the pre-fix tree, where the reader returned a bare digest.
#[test]
fn s1_a_source_row_with_an_escaped_nul_or_past_the_line_bound_is_dropped_and_counted() {
    let dir = scratch("s1-source-drops");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(format!("{ROW_A}\n").as_bytes());
    bytes.extend_from_slice(b"{\"id\":\"b\",\"text\":\"nul \\u0000 inside\"}\n");
    bytes.extend_from_slice(&line_of(decisions::MAX_LINE_BYTES + 1, "c"));
    bytes.extend_from_slice(&line_of(decisions::MAX_LINE_BYTES, "d"));
    bytes.extend_from_slice(format!("{ROW_B}\n").as_bytes());
    let path = dir.join("source.jsonl");
    std::fs::write(&path, &bytes).expect("source file");
    let v = convert::View {
        kind: "rows".to_owned(),
        path,
        sha256: sha256_hex(&bytes),
        rows: 5,
    };
    let mut seen = Vec::new();
    let got = convert::read_view(&v, |n, r| {
        seen.push((n, r["id"].as_str().unwrap_or("").to_owned()));
        Ok(())
    })
    .map(|lines| format!("{lines:?}"));
    let got = got.expect("a source file with one bad row was refused whole");
    let ids: Vec<(usize, &str)> = seen.iter().map(|(n, id)| (*n, id.as_str())).collect();
    assert_eq!(
        ids,
        [(1, "a"), (4, "d"), (5, "b")],
        "the wrong rows were passed on"
    );
    assert!(
        got.contains("nul_rows: Counted { rows: 1, first_lines: [2] }"),
        "the NUL row was not counted: {got}"
    );
    assert!(
        got.contains("oversized_rows: Counted { rows: 1, first_lines: [3] }"),
        "the oversized row was not counted: {got}"
    );
    cleanup(&dir);
}

#[test]
fn s1_json_nested_200_deep_is_refused_with_the_line() {
    let dir = scratch("s1-nested");
    let deep = format!(
        "{{\"id\":\"b\",\"text\":\"y\",\"deep\":{}{}}}",
        "[".repeat(200),
        "]".repeat(200)
    );
    let bytes = format!("{ROW_A}\n{deep}\n");
    assert_refused_at_line(
        read_target_bytes(&dir, bytes.as_bytes()),
        2,
        "200-deep JSON",
    );
    cleanup(&dir);
}

/// A duplicate id must be refused, with its line, before any pool is written: at the reader
/// or, failing that, at the containment scan the reader feeds.
#[test]
fn s1_a_duplicate_id_is_refused_with_the_line() {
    let dir = scratch("s1-dup-id");
    let dup = r#"{"id":"a","text":"another row with the first row's id"}"#;
    let bytes = format!("{ROW_A}\n{dup}\n");
    let path = dir.join("dup.jsonl");
    std::fs::write(&path, bytes.as_bytes()).expect("target file");
    let pins: BTreeMap<String, String> = [("t".to_owned(), sha256_hex(bytes.as_bytes()))].into();
    let refusal = match pool::read_targets(&pins, &[("t".to_owned(), path)]) {
        Err(e) => e,
        Ok((sets, _)) => {
            let rows = [cand(
                "c1",
                "a candidate row about something else entirely, long enough to scan",
                Gold::Option(0),
                &["yes", "no"],
            )];
            match scan(8, 0.5, &rows, &sets) {
                Ok(report) => panic!("a target set with a duplicate id was scanned: {report}"),
                Err(e) => e,
            }
        }
    };
    assert!(
        refusal.contains("\"a\""),
        "the refusal does not name the id: {refusal}"
    );
    assert!(
        refusal.contains(":2"),
        "the refusal does not name line 2: {refusal}"
    );
    cleanup(&dir);
}

#[test]
fn s1_a_directory_or_a_fifo_given_as_a_target_is_refused_by_type() {
    let dir = scratch("s1-type");
    let pins: BTreeMap<String, String> = [("t".to_owned(), "0".repeat(64))].into();
    let sub = dir.join("a-directory");
    std::fs::create_dir_all(&sub).expect("dir");
    let e = pool::read_targets(&pins, &[("t".to_owned(), sub)])
        .refusal("a directory was read as a target");
    assert!(e.contains("not a regular file"), "{e}");

    let fifo = dir.join("a-fifo");
    let made = Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .expect("mkfifo runs");
    assert!(made.success(), "mkfifo failed");
    let (tx, rx) = std::sync::mpsc::channel();
    let given = vec![("t".to_owned(), fifo)];
    std::thread::spawn(move || {
        let r = pool::read_targets(&pins, &given).map(|_| ());
        tx.send(r)
            .unwrap_or_else(|_| eprintln!("S1 FIFO read returned after the test gave up"));
    });
    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Err(e)) => assert!(e.contains("not a regular file"), "{e}"),
        Ok(Ok(())) => panic!("a FIFO was read as a target"),
        Err(_) => panic!("opening a FIFO target blocked for 10 s instead of being refused"),
    }
    cleanup(&dir);
}

// =============================================================================================
// S2: option and slot bounds. `decisions::tests::structural_refusals_mirror_choice_slot`
// already pins 1 and 17 options, a duplicate, `noul`, an empty option, an option and a context
// one past their bounds; these are the inputs it does not cover.

#[test]
fn s2_structural_bounds_pass_at_the_bound_and_refuse_one_past() {
    let base = cand("x", "c", Gold::Option(0), &["a", "b"]);
    assert_eq!(decisions::structural_refusal(&base), None);
    let with_options = |opts: Vec<String>, gold: Gold| Candidate {
        options: opts,
        gold,
        ..base.clone()
    };
    assert_eq!(
        decisions::structural_refusal(&with_options(Vec::new(), Gold::Noul)),
        Some("too_few_options")
    );
    let sixteen: Vec<String> = (0..decisions::MAX_OPTIONS)
        .map(|i| format!("option {i}"))
        .collect();
    assert_eq!(
        decisions::structural_refusal(&with_options(sixteen, Gold::Option(15))),
        None,
        "MAX_OPTIONS options must pass"
    );
    let at = "o".repeat(decisions::MAX_OPTION_BYTES);
    assert_eq!(
        decisions::structural_refusal(&with_options(vec!["a".into(), at], Gold::Option(1))),
        None,
        "an option of exactly MAX_OPTION_BYTES must pass"
    );
    let q = |len: usize| Candidate {
        question: "q".repeat(len),
        ..base.clone()
    };
    assert_eq!(
        decisions::structural_refusal(&q(decisions::MAX_QUESTION_BYTES)),
        None
    );
    assert_eq!(
        decisions::structural_refusal(&q(decisions::MAX_QUESTION_BYTES + 1)),
        Some("question_too_long")
    );
    // The bound is on the rendered context: question, a blank line, then the record.
    let ctx_at = decisions::MAX_CONTEXT_BYTES - 2 - base.question.len();
    let c = |len: usize| Candidate {
        context: "y".repeat(len),
        ..base.clone()
    };
    assert_eq!(decisions::structural_refusal(&c(ctx_at)), None);
    assert_eq!(
        decisions::structural_refusal(&c(ctx_at + 1)),
        Some("context_too_long")
    );
    assert_eq!(
        decisions::structural_refusal(&with_options(vec!["a".into(), "b".into()], Gold::Option(2))),
        Some("gold_not_in_options")
    );
}

/// `synth::assemble` drops each structurally refused draft and counts it under its reason (U5).
#[test]
fn s2_assemble_counts_every_structural_refusal_by_name() {
    let fam = "stress.family";
    let mut ds = base_drafts(fam);
    let seventeen: Vec<String> = (0..=decisions::MAX_OPTIONS)
        .map(|i| format!("option {i}"))
        .collect();
    let cases: Vec<(&str, synth::Draft)> = vec![
        (
            "too_few_options",
            draft(fam, "bad-none", "zero options", Vec::new(), Gold::Noul),
        ),
        (
            "too_many_options",
            draft(
                fam,
                "bad-17",
                "seventeen options",
                seventeen,
                Gold::Option(0),
            ),
        ),
        (
            "duplicate_options",
            draft(
                fam,
                "bad-dup",
                "two options equal after case folding",
                vec!["Alpha".into(), " alpha ".into(), "beta".into()],
                Gold::Option(2),
            ),
        ),
        (
            "option_too_long",
            draft(
                fam,
                "bad-long-option",
                "an option one byte past the bound",
                vec!["a".into(), "o".repeat(decisions::MAX_OPTION_BYTES + 1)],
                Gold::Option(0),
            ),
        ),
        (
            "context_too_long",
            draft(
                fam,
                "bad-long-context",
                &"y".repeat(decisions::MAX_CONTEXT_BYTES),
                abc(),
                Gold::Option(0),
            ),
        ),
    ];
    for (_, d) in &cases {
        ds.push(d.clone());
    }
    let mut long_question = draft(
        fam,
        "bad-long-question",
        "a long question",
        abc(),
        Gold::Option(0),
    );
    long_question.question = "q".repeat(decisions::MAX_QUESTION_BYTES + 1);
    ds.push(long_question);
    let a = synth::assemble(&synth_cfg(synth::Kind::Email), &ds).expect("assemble");
    for (reason, _) in &cases {
        assert_eq!(
            a.report["refused"][*reason],
            json!(1),
            "{reason} not counted: {}",
            a.report["refused"]
        );
    }
    assert_eq!(a.report["refused"]["question_too_long"], json!(1));
    assert!(
        a.candidates.iter().all(|c| c.group_key.starts_with('t')),
        "a refused draft became a row"
    );
    assert_eq!(a.candidates.len(), 4);
}

#[test]
fn s2_assemble_refuses_a_gold_out_of_range_without_a_panic() {
    let mut ds = base_drafts("stress.family");
    ds.push(draft(
        "stress.family",
        "bad-gold",
        "gold past the options",
        abc(),
        Gold::Option(3),
    ));
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        synth::assemble(&synth_cfg(synth::Kind::Email), &ds).map(|a| a.candidates.len())
    }));
    match got {
        Err(_) => panic!("assemble panicked on a gold out of range (U1)"),
        Ok(Ok(n)) => panic!("assemble admitted a gold out of range ({n} rows)"),
        Ok(Err(e)) => assert!(e.contains("gold"), "{e}"),
    }
}

/// `apps.tool_select` states `noul` as a zero (`synth_tools` module doc). A `noul` gold there is
/// refused or counted, never admitted silently.
#[test]
fn s2_noul_gold_on_a_family_stating_noul_as_zero_is_refused_or_counted() {
    let fam = synth_tools::FAMILY;
    let mut ds = base_drafts(fam);
    ds.push(draft(
        fam,
        "noul-row",
        "a tool row whose gold is noul",
        abc(),
        Gold::Noul,
    ));
    match synth::assemble(&synth_cfg(synth::Kind::Tools), &ds) {
        Err(_) => {}
        Ok(a) => {
            let admitted = a
                .candidates
                .iter()
                .filter(|c| c.family_id == fam && c.gold == Gold::Noul)
                .count();
            assert_eq!(
                admitted, 0,
                "{fam} states noul as zero, yet {admitted} noul row(s) were admitted; refused: {}",
                a.report["refused"]
            );
        }
    }
}

/// `pool::examples` is the seam every v6 producer writes through. A row `structural_refusal`
/// refuses must not reach `examples.jsonl` through it, and must not panic it (U1).
#[test]
fn s2_pool_examples_refuses_what_structural_refusal_refuses() {
    let seventeen: Vec<String> = (0..=decisions::MAX_OPTIONS)
        .map(|i| format!("option {i}"))
        .collect();
    let base = cand("x", "c", Gold::Option(0), &["a", "b"]);
    let bad: Vec<(&str, Candidate)> = vec![
        (
            "gold out of range",
            Candidate {
                gold: Gold::Option(2),
                ..base.clone()
            },
        ),
        (
            "17 options",
            Candidate {
                options: seventeen,
                ..base.clone()
            },
        ),
        (
            "duplicate options",
            Candidate {
                options: vec!["A".into(), "a".into()],
                ..base.clone()
            },
        ),
        (
            "option past its bound",
            Candidate {
                options: vec!["a".into(), "o".repeat(decisions::MAX_OPTION_BYTES + 1)],
                ..base.clone()
            },
        ),
        (
            "context past its bound",
            Candidate {
                context: "y".repeat(decisions::MAX_CONTEXT_BYTES),
                ..base.clone()
            },
        ),
    ];
    for (what, c) in &bad {
        assert!(decisions::structural_refusal(c).is_some(), "{what}");
        let refs = vec![c];
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool::examples(&refs).map(|e| e.rows)
        }));
        match got {
            Err(_) => panic!("pool::examples panicked on a row with {what} (U1)"),
            Ok(Ok(rows)) => panic!("pool::examples wrote a row with {what} ({rows} row(s))"),
            Ok(Err(_)) => {}
        }
    }
}

// =============================================================================================
// S3: containment through `pool::decontaminate`. `pool::tests::nothing_surviving_is_refused`
// and `convert::tests::a_row_with_an_unassigned_code_point_is_refused_and_counted` (candidate
// side) are reused by the runner; these are the other inputs.

fn three_rows() -> Vec<Candidate> {
    (0..3)
        .map(|i| {
            cand(
                &format!("c{i}"),
                &words(20, &format!("candidate{i}w")),
                Gold::Option(0),
                &["yes", "no"],
            )
        })
        .collect()
}

fn one_target() -> Vec<TargetSet> {
    vec![(
        "t".to_owned(),
        vec![("t1".to_owned(), words(20, "targetw"))],
    )]
}

#[test]
fn s3_an_empty_target_set_is_refused() {
    let rows = three_rows();
    let none = scan(8, 0.5, &rows, &[]);
    assert!(
        none.is_err(),
        "a scan against no target set was accepted: {none:?}"
    );
    let empty: Vec<TargetSet> = vec![("t".to_owned(), Vec::new())];
    let got = scan(8, 0.5, &rows, &empty);
    assert!(
        got.is_err(),
        "a scan against a target set with no rows was accepted: {got:?}"
    );
}

#[test]
fn s3_a_synth_config_pinning_no_target_set_is_refused_and_writes_nothing() {
    let dir = scratch("s3-no-target");
    let config = dir.join("email.json");
    write_json(&config, &email_config(6, &BTreeMap::new()));
    let out_dir = dir.join("pool");
    let out = cli(&synth_cli_args(&config, &[], &out_dir));
    assert_cli_refused(&out, "against nothing", "synth with no target set");
    assert_nothing_written(&out_dir, "synth with no target set");
    cleanup(&dir);
}

#[test]
fn s3_threshold_zero_nan_and_out_of_range_are_refused_and_one_passes() {
    let (rows, targets) = (three_rows(), one_target());
    for t in [0.0, -0.0, -0.5, f64::NAN, 1.0 + 1e-9, f64::INFINITY] {
        assert!(
            scan(8, t, &rows, &targets).is_err(),
            "threshold {t} was accepted"
        );
    }
    let ok = scan(8, 1.0, &rows, &targets);
    assert!(ok.is_ok(), "threshold 1 was refused: {ok:?}");
}

#[test]
fn s3_ngram_n_zero_and_past_max_n_are_refused() {
    let (rows, targets) = (three_rows(), one_target());
    for n in [0, containment::MAX_N + 1] {
        assert!(scan(n, 0.5, &rows, &targets).is_err(), "n {n} was accepted");
    }
    let ok = scan(containment::MAX_N, 0.5, &rows, &targets);
    assert!(ok.is_ok(), "n MAX_N was refused: {ok:?}");
}

/// An `ngram_n` longer than every row leaves every row unscanned. That is either refused or
/// counted in full (U5); an unscanned row never reads as clean.
#[test]
fn s3_ngram_n_longer_than_every_row_counts_every_row_as_unscanned() {
    let (rows, targets) = (three_rows(), one_target());
    match scan(containment::MAX_N, 0.5, &rows, &targets) {
        Err(_) => {}
        Ok(report) => {
            let by_set = &report["rows_too_short_by_set"];
            for set in ["stress-candidates", "t"] {
                let rows = by_set[set]["rows"].as_u64();
                assert!(rows.is_some_and(|n| n > 0), "{set}: {by_set}");
                assert_eq!(
                    by_set[set]["too_short"].as_u64(),
                    rows,
                    "{set}: not every unscannable row is counted: {by_set}"
                );
            }
        }
    }
}

/// The recorded gap: a code point Python's tables do not assign is refused loudly, naming the
/// row, never skipped (pinned until the gap is fixed).
#[test]
fn s3_an_unassigned_code_point_in_a_target_is_refused_loudly() {
    let targets: Vec<TargetSet> = vec![(
        "t".to_owned(),
        vec![(
            "t-unassigned".to_owned(),
            format!("{} \u{378} {}", words(6, "a"), words(6, "b")),
        )],
    )];
    match scan(8, 0.5, &three_rows(), &targets) {
        Ok(report) => panic!("an unassigned code point was scanned silently: {report}"),
        Err(e) => assert!(
            e.contains("t-unassigned"),
            "the refusal does not name the row: {e}"
        ),
    }
}

// =============================================================================================
// S4: `qd-synth-config/v1`. Each refusal goes through `synth::run` and must leave nothing
// behind (U2). `pool::tests::a_target_set_not_pinned_or_given_twice_is_refused` covers two of
// the target cases at the function; they are repeated here end to end for U2.

fn s4_base(dir: &Path) -> (Value, Vec<(String, PathBuf)>) {
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    (email_config(6, &pins_of(&t)), given_of(&t))
}

#[test]
fn s4_each_required_field_missing_is_refused() {
    let dir = scratch("s4-missing");
    let (base, targets) = s4_base(&dir);
    for key in [
        "schema",
        "kind",
        "seed",
        "rows_per_template",
        "val_templates_per_class",
        "injection_rate",
        "ngram_n",
        "containment_threshold",
        "target_sha256",
        "probe",
    ] {
        let needle = if key == "probe" {
            "\"probe\"".to_owned()
        } else {
            format!("{key:?}")
        };
        synth_refuses(
            &dir,
            &without(&base, &[key]),
            &targets,
            &needle,
            &format!("no {key}"),
        );
    }
    for key in [
        "n_min",
        "n_max",
        "dim",
        "max_iter",
        "tol",
        "lr",
        "l2_grid",
        "max_val_accuracy",
    ] {
        synth_refuses(
            &dir,
            &without(&base, &["probe", key]),
            &targets,
            &format!("{key:?}"),
            &format!("no probe.{key}"),
        );
    }
    let tools = shipped_with(
        "tools-v6-2026-10-06.json",
        &pins_of(&[write_target(&dir.join("targets"), "stand-in")]),
    );
    synth_refuses(
        &dir,
        &without(&tools, &["tools"]),
        &targets,
        "\"tools\" object",
        "kind tools without tools",
    );
    for key in [
        "catalog",
        "catalog_sha256",
        "phrasings",
        "phrasings_sha256",
        "unable_rate",
        "mask_rate",
        "max_median_overlap",
    ] {
        synth_refuses(
            &dir,
            &without(&tools, &["tools", key]),
            &targets,
            &format!("{key:?}"),
            &format!("no tools.{key}"),
        );
    }
    cleanup(&dir);
}

#[test]
fn s4_wrong_types_are_refused() {
    let dir = scratch("s4-types");
    let (base, targets) = s4_base(&dir);
    // (a dotted path into the config, the value put there, what the refusal must name)
    let cases: Vec<(&str, Value, &str)> = vec![
        ("schema", json!("qd-synth-config/v0"), "schema is not"),
        ("kind", json!(3), "\"kind\""),
        ("kind", json!("mail"), "\"mail\""),
        ("seed", json!("7"), "\"seed\""),
        ("rows_per_template", json!(-1), "\"rows_per_template\""),
        (
            "val_templates_per_class",
            json!(1.5),
            "\"val_templates_per_class\"",
        ),
        ("injection_rate", json!("x"), "\"injection_rate\""),
        ("ngram_n", json!("8"), "\"ngram_n\""),
        (
            "containment_threshold",
            json!(null),
            "\"containment_threshold\"",
        ),
        ("target_sha256", json!([]), "\"target_sha256\""),
        (
            "target_sha256",
            json!({"stand-in": 1}),
            "target_sha256.stand-in",
        ),
        ("probe.l2_grid", json!("x"), "probe.l2_grid"),
        ("probe.l2_grid", json!([-1.0]), "probe.l2_grid"),
        ("probe.dim", json!(1.5), "\"dim\""),
        ("probe.tol", json!("x"), "\"tol\""),
    ];
    for (dotted, x, needle) in cases {
        let path: Vec<&str> = dotted.split('.').collect();
        synth_refuses(
            &dir,
            &with(&base, &path, x.clone()),
            &targets,
            needle,
            &format!("{dotted} = {x}"),
        );
    }
    cleanup(&dir);
}

#[test]
fn s4_rows_per_template_zero_and_past_its_bound_are_refused_and_the_bound_parses() {
    let dir = scratch("s4-rows-per-template");
    let (base, targets) = s4_base(&dir);
    for n in [0, synth::MAX_ROWS_PER_TEMPLATE as u64 + 1] {
        synth_refuses(
            &dir,
            &with(&base, &["rows_per_template"], json!(n)),
            &targets,
            &format!("rows_per_template {n}"),
            &format!("rows_per_template {n}"),
        );
    }
    let at = with(
        &base,
        &["rows_per_template"],
        json!(synth::MAX_ROWS_PER_TEMPLATE),
    );
    let parsed = synth::Config::parse(&serde_json::to_vec(&at).expect("json"));
    assert!(parsed.is_ok(), "{:?}", parsed.err());
    cleanup(&dir);
}

/// 65 email templates at the per-template bound draft ~715k rows, past `MAX_ROWS`. Refused, and
/// nothing written (U2). The refusal happens after drafting (in `assemble`), not before.
#[test]
fn s4_a_total_past_max_rows_is_refused_and_writes_nothing() {
    let dir = scratch("s4-max-rows");
    let (base, targets) = s4_base(&dir);
    synth_refuses(
        &dir,
        &with(
            &base,
            &["rows_per_template"],
            json!(synth::MAX_ROWS_PER_TEMPLATE),
        ),
        &targets,
        &format!("a pool holds at most {}", synth::MAX_ROWS),
        "rows past MAX_ROWS",
    );
    cleanup(&dir);
}

#[test]
fn s4_val_templates_per_class_must_leave_a_template_to_train_on() {
    let dir = scratch("s4-val-templates");
    let (base, targets) = s4_base(&dir);
    let floor = synth::MIN_TEMPLATES_PER_CLASS as u64;
    for n in [0, floor, floor + 1] {
        synth_refuses(
            &dir,
            &with(&base, &["val_templates_per_class"], json!(n)),
            &targets,
            &format!("val_templates_per_class {n}"),
            &format!("val_templates_per_class {n}"),
        );
    }
    let ok = with(&base, &["val_templates_per_class"], json!(floor - 1));
    assert!(synth::Config::parse(&serde_json::to_vec(&ok).expect("json")).is_ok());
    cleanup(&dir);
}

#[test]
fn s4_probe_dim_and_max_iter_past_their_bounds_are_refused_and_the_bounds_parse() {
    let dir = scratch("s4-probe");
    let (base, targets) = s4_base(&dir);
    let cases = [
        ("dim", u64::from(synth::MAX_PROBE_DIM)),
        ("max_iter", u64::from(synth::MAX_PROBE_ITER)),
    ];
    for (key, bound) in cases {
        synth_refuses(
            &dir,
            &with(&base, &["probe", key], json!(bound + 1)),
            &targets,
            &format!("probe.{key}"),
            &format!("probe.{key} past its bound"),
        );
        let at = with(&base, &["probe", key], json!(bound));
        let parsed = synth::Config::parse(&serde_json::to_vec(&at).expect("json"));
        assert!(
            parsed.is_ok(),
            "probe.{key} at its bound: {:?}",
            parsed.err()
        );
    }
    cleanup(&dir);
}

#[test]
fn s4_targets_must_be_exactly_the_pinned_sets_each_once() {
    let dir = scratch("s4-target-pins");
    let (base, targets) = s4_base(&dir);
    synth_refuses(&dir, &base, &[], "not exactly", "pinned, not given");
    synth_refuses(
        &dir,
        &with(&base, &["target_sha256"], json!({})),
        &targets,
        "not exactly",
        "given, not pinned",
    );
    let twice = vec![targets[0].clone(), targets[0].clone()];
    synth_refuses(&dir, &base, &twice, "not exactly", "given twice");
    cleanup(&dir);
}

#[test]
fn s4_a_sha_mismatch_on_a_target_catalog_or_phrasings_is_refused() {
    let dir = scratch("s4-sha");
    let (base, targets) = s4_base(&dir);
    let wrong = "0".repeat(64);
    synth_refuses(
        &dir,
        &with(&base, &["target_sha256", "stand-in"], json!(wrong)),
        &targets,
        "pins",
        "target sha mismatch",
    );
    let tools = shipped_with(
        "tools-v6-2026-10-06.json",
        &pins_of(&[write_target(&dir.join("targets"), "stand-in")]),
    );
    for key in ["catalog_sha256", "phrasings_sha256"] {
        synth_refuses(
            &dir,
            &with(&tools, &["tools", key], json!(wrong)),
            &targets,
            "pins",
            &format!("{key} mismatch"),
        );
    }
    cleanup(&dir);
}

// =============================================================================================
// S5: tools data. Reused unit tests: `synth_tools::tests::a_one_member_group_is_refused`,
// `an_underspecified_phrasing_must_name_a_required_parameter` and
// `a_shared_tool_asks_only_where_its_parameter_is_required` (the shipped case).

fn tool_json(name: &str, desc: &str, required: &[&str]) -> Value {
    json!({
        "name": name, "description": desc,
        "params": required.iter().map(|p| json!({"name": p, "type": "string", "required": true}))
            .collect::<Vec<_>>(),
        "group": null
    })
}

fn catalog_doc(tools: Vec<Value>) -> Value {
    json!({"schema": synth_tools::CATALOG_SCHEMA, "catalogs": [{"app": "demo", "tools": tools}]})
}

fn demo_catalog() -> Value {
    catalog_doc(vec![
        tool_json(
            "graph_search",
            "Find symbols by name in the code graph.",
            &["query"],
        ),
        tool_json("task_next", "The next ready task in the queue.", &[]),
    ])
}

fn demo_phrasings() -> Value {
    json!({
        "schema": synth_tools::PHRASINGS_SCHEMA,
        "tools": [
            {
                "tool": "graph_search",
                "complete": [
                    "find symbols named parse_header",
                    "which symbols match the name RetryPolicy",
                    "search the code graph for SessionCache",
                    "locate symbols called load_conf please"
                ],
                "underspecified": [
                    {"text": "find that symbol again", "missing": "query"},
                    {"text": "where was it declared", "missing": "query"}
                ]
            },
            {
                "tool": "task_next",
                "complete": [
                    "what is the next task for me",
                    "which ready task should I pick up",
                    "pull the next item off the queue",
                    "anything ready to work on now"
                ],
                "underspecified": []
            }
        ],
        "direct": [
            "what is a closure in rust",
            "explain a mutex briefly",
            "how do I write a haiku",
            "what time zone is UTC"
        ]
    })
}

#[test]
fn s5_an_empty_catalog_is_refused() {
    let none = json!({"schema": synth_tools::CATALOG_SCHEMA, "catalogs": []});
    let e = synth_tools::parse_catalogs(&none).refusal("no catalogs accepted");
    assert!(e.contains("no catalogs"), "{e}");
    let e = synth_tools::parse_catalogs(&catalog_doc(Vec::new()))
        .refusal("a catalog with no tools accepted");
    assert!(e.contains("no tools"), "{e}");
}

#[test]
fn s5_a_tool_name_repeated_in_one_app_is_refused() {
    let doc = catalog_doc(vec![
        tool_json("graph_search", "Find symbols by name.", &["query"]),
        tool_json("graph_search", "Find symbols by name, again.", &["query"]),
    ]);
    let e = synth_tools::parse_catalogs(&doc).refusal("a repeated tool name accepted");
    assert!(e.contains("repeated") && e.contains("graph_search"), "{e}");
}

#[test]
fn s5_duplicate_phrasings_including_case_variants_are_refused() {
    let cats = synth_tools::parse_catalogs(&demo_catalog()).expect("demo catalog");
    assert!(synth_tools::parse_phrasings(&demo_phrasings(), &cats).is_ok());
    let mut exact_doc = demo_phrasings();
    exact_doc["direct"][3] = json!("explain a mutex briefly");
    let e = synth_tools::parse_phrasings(&exact_doc, &cats)
        .refusal("an exact duplicate phrasing accepted");
    assert!(e.contains("appears twice"), "{e}");
    let mut case_doc = demo_phrasings();
    case_doc["tools"][1]["complete"][3] = json!("Find Symbols Named PARSE_HEADER");
    let e = synth_tools::parse_phrasings(&case_doc, &cats)
        .refusal("a case-variant duplicate phrasing accepted");
    assert!(e.contains("appears twice"), "{e}");
}

#[test]
fn s5_the_overlap_median_passes_at_the_bound_and_is_refused_just_over() {
    let cats = synth_tools::parse_catalogs(&demo_catalog()).expect("demo catalog");
    let ph = synth_tools::parse_phrasings(&demo_phrasings(), &cats).expect("demo phrasings");
    let (report, _) = synth_tools::overlap_report(&cats, &ph, 1.0);
    let median = report["median"].as_f64().expect("a median");
    assert!(
        median > 0.0,
        "the fixture needs a positive median: {report}"
    );
    let (at, ok_at) = synth_tools::overlap_report(&cats, &ph, median);
    assert!(ok_at, "a median exactly at the bound was refused: {at}");
    let below = f64::from_bits(median.to_bits() - 1);
    let (over, ok_over) = synth_tools::overlap_report(&cats, &ph, below);
    assert!(!ok_over, "a median just over the bound passed: {over}");
}

/// A tools config in `dir` whose catalog is `catalog` (any path), pinned to `catalog_sha`.
fn tools_config_with_catalog(
    dir: &Path,
    catalog: &Path,
    catalog_sha: &str,
) -> (PathBuf, Vec<(String, PathBuf, String)>) {
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    let cfg = with(
        &with(
            &shipped_with("tools-v6-2026-10-06.json", &pins_of(&t)),
            &["tools", "catalog"],
            json!(s(catalog)),
        ),
        &["tools", "catalog_sha256"],
        json!(catalog_sha),
    );
    let path = dir.join(format!("tools-{}.json", next()));
    write_json(&path, &cfg);
    (path, t)
}

#[test]
fn s5_a_catalog_past_the_file_bound_is_refused_by_size_and_the_bound_is_read() {
    let dir = scratch("s5-catalog-size");
    for (size, past) in [(TOOLS_FILE_BOUND + 1, true), (TOOLS_FILE_BOUND, false)] {
        let catalog = dir.join(format!("catalog-{size}.json"));
        let f = std::fs::File::create(&catalog).expect("catalog");
        f.set_len(size).expect("sparse catalog");
        drop(f);
        let (config, t) = tools_config_with_catalog(&dir, &catalog, &"0".repeat(64));
        let out_dir = dir.join(format!("pool-{size}"));
        let out = cli(&synth_cli_args(&config, &t, &out_dir));
        if past {
            assert_cli_refused(
                &out,
                &format!("{size} bytes; the bound is {TOOLS_FILE_BOUND}"),
                "catalog past the bound",
            );
        } else {
            // At the bound it is read, and refused for what it is (its pin), never its size.
            assert_cli_refused(&out, "pins", "catalog at the bound");
            assert!(
                !stderr_of(&out).contains("the bound is"),
                "{}",
                stderr_of(&out)
            );
        }
        assert_nothing_written(&out_dir, "catalog size");
    }
    cleanup(&dir);
}

#[test]
fn s5_a_catalog_that_is_a_directory_or_missing_is_refused() {
    let dir = scratch("s5-catalog-kind");
    let sub = dir.join("catalog-dir");
    std::fs::create_dir_all(&sub).expect("dir");
    let (config, t) = tools_config_with_catalog(&dir, &sub, &"0".repeat(64));
    let out_dir = dir.join("pool-dir");
    let out = cli(&synth_cli_args(&config, &t, &out_dir));
    assert_cli_refused(&out, "not a regular file", "catalog is a directory");
    assert_nothing_written(&out_dir, "catalog is a directory");
    let missing = dir.join("no-such-catalog.json");
    let (config, t) = tools_config_with_catalog(&dir, &missing, &"0".repeat(64));
    let out_dir = dir.join("pool-missing");
    let out = cli(&synth_cli_args(&config, &t, &out_dir));
    assert_cli_refused(&out, "no-such-catalog.json", "catalog is missing");
    assert_nothing_written(&out_dir, "catalog is missing");
    cleanup(&dir);
}

/// The child's resident set in bytes, from `ps` (KiB there), when it can be read.
fn rss_bytes(pid: u32) -> Option<u64> {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kib| kib * 1024)
}

/// The pre-registration says "refused by the size bound"; since `qd_prep::files` the type
/// check refuses it earlier, as "not a regular file", which is stricter. The run is watched:
/// past 512 MiB of RSS it is killed, so a read without end fails the test, not the machine.
#[test]
fn s5_a_catalog_symlinked_to_dev_zero_is_refused_not_read_without_end() {
    let dir = scratch("s5-dev-zero");
    let link = dir.join("zero-catalog.json");
    std::os::unix::fs::symlink("/dev/zero", &link).expect("symlink to /dev/zero");
    let (config, t) = tools_config_with_catalog(&dir, &link, &"0".repeat(64));
    let out_dir = dir.join("pool");
    let mut child = spawn(&synth_cli_args(&config, &t, &out_dir));
    let start = Instant::now();
    let watch = 512u64 << 20;
    let out = loop {
        if child.try_wait().expect("try_wait").is_some() {
            break child.wait_with_output().expect("output");
        }
        let rss = rss_bytes(child.id()).unwrap_or(0);
        if rss > watch || start.elapsed() > Duration::from_secs(60) {
            child.kill().expect("kill");
            child.wait().expect("reap");
            panic!(
                "a catalog symlinked to /dev/zero was read without end: killed at RSS {rss} \
                 bytes after {:?}",
                start.elapsed()
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_cli_refused(&out, "not a regular file", "catalog symlinked to /dev/zero");
    assert_nothing_written(&out_dir, "catalog symlinked to /dev/zero");
    cleanup(&dir);
}

// =============================================================================================
// S6: the output directory.

fn small_built() -> decisions::Built {
    let rows = three_rows();
    let scanned = pool::decontaminate(
        8,
        0.5,
        &json!({"tool": "qd-prep stress_v6"}),
        "stress-candidates",
        &rows,
        &one_target(),
        1,
    )
    .expect("a small scan");
    let ex = pool::examples(&scanned.clean).expect("examples");
    let pool::Scanned { written, .. } = scanned;
    decisions::Built {
        manifest: json!({"examples": ex.rows, "examples_sha256": ex.sha256}),
        examples: ex.bytes,
        texts: None,
        containment: written,
        summary: "stress".to_owned(),
    }
}

#[test]
fn s6_an_existing_out_dir_or_a_file_in_its_place_is_refused_before_any_input_is_read() {
    let dir = scratch("s6-exists");
    let as_dir = dir.join("pool-dir");
    std::fs::create_dir_all(&as_dir).expect("dir");
    let as_file = dir.join("pool-file");
    std::fs::write(&as_file, b"not a pool").expect("file");
    // The inputs do not exist: a refusal naming "exists" proves nothing was read first.
    let missing = dir.join("no-such-input.json");
    for out_dir in [&as_dir, &as_file] {
        let e = synth::run(
            &synth::Inputs {
                config: missing.clone(),
                targets: Vec::new(),
            },
            out_dir,
            1,
        )
        .refusal("synth wrote over an existing path");
        assert!(e.contains("exists"), "synth: {e}");
        let e = convert::run(
            &convert::Inputs {
                config: missing.clone(),
                view_record: missing.clone(),
                pool: "tools".to_owned(),
                licence_cache: None,
            },
            out_dir,
            1,
        )
        .refusal("convert wrote over an existing path");
        assert!(e.contains("exists"), "convert: {e}");
        let e = decisions::write_out(out_dir, &small_built())
            .refusal("write_out wrote over an existing path");
        assert!(e.contains("exists"), "write_out: {e}");
        let out = cli(&[
            "decisions".to_owned(),
            "--config".to_owned(),
            s(&missing),
            "--fetch-record".to_owned(),
            s(&missing),
            "--decider-dir".to_owned(),
            s(&dir),
            "--out-dir".to_owned(),
            s(out_dir),
        ]);
        assert_cli_refused(&out, "exists", "decisions over an existing path");
    }
    assert_eq!(std::fs::read(&as_file).expect("file"), b"not a pool");
    assert_eq!(std::fs::read_dir(&as_dir).expect("dir").count(), 0);
    cleanup(&dir);
}

#[test]
fn s6_a_missing_or_read_only_parent_is_refused_and_writes_nothing() {
    let dir = scratch("s6-parent");
    let missing_parent = dir.join("no-such-parent").join("pool");
    let e = decisions::write_out(&missing_parent, &small_built())
        .refusal("write_out under a missing parent");
    assert!(!e.is_empty());
    assert_nothing_written(&missing_parent, "write_out, missing parent");
    let e = convert::write_dir(&missing_parent, &[("a.txt", b"a")])
        .refusal("write_dir under a missing parent");
    assert!(!e.is_empty());
    assert_nothing_written(&missing_parent, "write_dir, missing parent");

    let ro = dir.join("read-only");
    std::fs::create_dir_all(&ro).expect("dir");
    std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).expect("chmod 555");
    let under_ro = ro.join("pool");
    let built = small_built();
    let a = decisions::write_out(&under_ro, &built);
    let b = convert::write_dir(&under_ro, &[("a.txt", b"a")]);
    std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755)).expect("chmod 755");
    assert!(
        a.is_err(),
        "write_out wrote under a read-only parent (running as root?)"
    );
    assert!(
        b.is_err(),
        "write_dir wrote under a read-only parent (running as root?)"
    );
    assert_nothing_written(&under_ro, "read-only parent");

    // `qd-prep synth` builds the pool before it writes; under a missing parent it must still
    // refuse and leave nothing.
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    let config = dir.join("email.json");
    write_json(&config, &email_config(6, &pins_of(&t)));
    let e = synth::run(
        &synth::Inputs {
            config,
            targets: given_of(&t),
        },
        &missing_parent,
        2,
    )
    .refusal("synth wrote under a missing parent");
    assert!(!e.is_empty());
    assert_nothing_written(&missing_parent, "synth, missing parent");
    cleanup(&dir);
}

/// SIGTERM once `DIR.partial/examples.jsonl` exists. The pool is then absent or complete; a
/// reader never takes a half pool as complete. A run that finished before the signal did not
/// exercise the item: the test fails with `S6-KILL-NOT-EXERCISED`, which the runner records as
/// `not_run`, never as a pass.
#[test]
fn s6_a_kill_during_the_write_leaves_no_pool_a_reader_takes_as_complete() {
    let dir = scratch("s6-kill");
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    let config = dir.join("email.json");
    write_json(&config, &email_config(600, &pins_of(&t)));
    let out_dir = dir.join("pool");
    let mut child = spawn(&synth_cli_args(&config, &t, &out_dir));
    let first = partial(&out_dir).join("examples.jsonl");
    let start = Instant::now();
    let mut signalled = false;
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            break;
        }
        if !signalled && first.exists() {
            let sent = Command::new("kill")
                .args(["-TERM", &child.id().to_string()])
                .status()
                .expect("kill runs");
            assert!(sent.success(), "kill -TERM failed");
            signalled = true;
        }
        if start.elapsed() > CLI_LIMIT {
            child.kill().expect("kill");
            child.wait().expect("reap");
            panic!("the synth run passed {CLI_LIMIT:?}");
        }
        std::thread::sleep(Duration::from_micros(200));
    }
    let out = child.wait_with_output().expect("output");
    if out.status.signal() != Some(15) {
        panic!(
            "S6-KILL-NOT-EXERCISED: the run ended with {:?} before SIGTERM reached it during \
             the write (signalled: {signalled}); stderr: {}",
            out.status,
            stderr_of(&out)
        );
    }
    if out_dir.exists() {
        assert_complete_pool(&out_dir);
    }
    eprintln!(
        "S6 kill: out_dir present {}, {} left {}",
        out_dir.exists(),
        partial(&out_dir).display(),
        partial(&out_dir).exists()
    );
    cleanup(&dir);
}

// =============================================================================================
// S7: two `qd-prep synth` runs to one out_dir at once. The static half (temp paths keyed only
// on the pid) is `python/tests/test_stress_v6.py::test_s7_...`.

#[test]
fn s7_two_concurrent_synth_runs_to_one_out_dir_one_refuses() {
    let dir = scratch("s7-race");
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    let config = dir.join("email.json");
    write_json(&config, &email_config(6, &pins_of(&t)));
    let out_dir = dir.join("pool");
    let args = synth_cli_args(&config, &t, &out_dir);
    let a = spawn(&args);
    let b = spawn(&args);
    let outs = [
        wait_bounded(a, CLI_LIMIT, "synth run A"),
        wait_bounded(b, CLI_LIMIT, "synth run B"),
    ];
    let won = outs.iter().filter(|o| o.status.success()).count();
    for o in &outs {
        assert_ne!(
            o.status.code(),
            Some(101),
            "panicked (U1): {}",
            stderr_of(o)
        );
    }
    assert_eq!(
        won,
        1,
        "exactly one run must win; stderr: {:?}",
        outs.iter().map(stderr_of).collect::<Vec<_>>()
    );
    let lost = outs.iter().find(|o| !o.status.success()).expect("a loser");
    assert!(stderr_of(lost).contains("exists"), "{}", stderr_of(lost));
    assert_complete_pool(&out_dir);
    assert!(
        !partial(&out_dir).exists(),
        "the losing run left {} (U2)",
        partial(&out_dir).display()
    );
    cleanup(&dir);
}

// =============================================================================================
// S8: determinism, each synth kind at --threads 1 and twice at --threads 8 (U4). The decisions
// pool needs its pinned upstream sources and is recorded `not_run` by the runner.

fn s8_build_three_times(tag: &str, config_name: &str) {
    let dir = scratch(tag);
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    let config = dir.join("config.json");
    write_json(&config, &shipped_with(config_name, &pins_of(&t)));
    let mut built = Vec::new();
    for (i, threads) in [1, 8, 8].into_iter().enumerate() {
        let out_dir = dir.join(format!("pool-{i}-t{threads}"));
        let mut args = synth_cli_args(&config, &t, &out_dir);
        args.push("--threads".to_owned());
        args.push(threads.to_string());
        let out = cli(&args);
        assert!(
            out.status.success(),
            "{config_name} at {threads} thread(s): {}",
            stderr_of(&out)
        );
        assert_complete_pool(&out_dir);
        built.push(out_dir);
    }
    for name in ["examples.jsonl", "manifest.json"] {
        let first = std::fs::read(built[0].join(name)).expect("first");
        for other in &built[1..] {
            let b = std::fs::read(other.join(name)).expect("other");
            assert!(
                first == b,
                "{config_name}: {name} differs between {} and {} (U4)",
                built[0].display(),
                other.display()
            );
        }
    }
    cleanup(&dir);
}

#[test]
fn s8_synth_email_is_byte_identical_at_one_and_eight_threads() {
    s8_build_three_times("s8-email", "email-v6-2026-10-06.json");
}

#[test]
fn s8_synth_jarvis_is_byte_identical_at_one_and_eight_threads() {
    s8_build_three_times("s8-jarvis", "jarvis-v6-2026-10-06.json");
}

#[test]
fn s8_synth_tools_is_byte_identical_at_one_and_eight_threads() {
    s8_build_three_times("s8-tools", "tools-v6-2026-10-06.json");
}

// =============================================================================================
// S9: scale, bounded. `#[ignore]`d: run with `--ignored --nocapture`. Each run is wrapped in
// `/usr/bin/time -l` (macOS), and its wall clock and maximum RSS are asserted and printed.

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `qd-prep args` under `/usr/bin/time -l`: refused past `limit` of wall clock or the RSS cap.
fn timed(name: &str, args: &[String], limit: Duration) {
    let start = Instant::now();
    let child = Command::new("/usr/bin/time")
        .arg("-l")
        .arg(BIN)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("/usr/bin/time runs (S9 needs macOS's time -l)");
    let out = wait_bounded(child, limit + Duration::from_secs(60), name);
    let wall = start.elapsed();
    let err = stderr_of(&out);
    let rss = err
        .lines()
        .find(|l| l.contains("maximum resident set size"))
        .and_then(|l| l.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("{name}: no maximum resident set size in: {err}"));
    println!(
        "S9 {name}: wall {:.1} s (bound {} s), max RSS {rss} bytes (cap {RSS_CAP_BYTES})",
        wall.as_secs_f64(),
        limit.as_secs()
    );
    assert!(out.status.success(), "{name}: {err}");
    assert!(wall <= limit, "{name}: {wall:?} past its bound {limit:?}");
    assert!(rss < RSS_CAP_BYTES, "{name}: RSS {rss} past the cap (U3)");
}

#[test]
#[ignore = "S9 scale: run in its own step with --ignored"]
fn s9_synth_email_at_100k_rows_within_ten_minutes() {
    let dir = scratch("s9-email");
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    let config = dir.join("email.json");
    // 65 templates x 1,400 rows, plus a 0.1 injection share: about 100k rows.
    let cfg = with(
        &shipped_with("email-v6-2026-10-06.json", &pins_of(&t)),
        &["rows_per_template"],
        json!(1400),
    );
    write_json(&config, &cfg);
    let out_dir = dir.join("pool");
    timed(
        "synth email 100k",
        &synth_cli_args(&config, &t, &out_dir),
        Duration::from_secs(600),
    );
    let manifest = assert_complete_pool(&out_dir);
    println!("S9 synth email 100k: {} rows", manifest["examples"]);
    cleanup(&dir);
}

fn seeded_text(i: u64, salt: u64, n_words: u64) -> String {
    (0..n_words)
        .map(|w| format!("w{}", splitmix64(salt ^ (i << 8) ^ w) % 20_000))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "S9 scale: run in its own step with --ignored"]
fn s9_containment_200k_candidates_by_50k_targets_within_twenty_minutes() {
    let dir = scratch("s9-containment");
    let targets_rows: Vec<(String, String)> = (0..50_000u64)
        .map(|i| (format!("t{i}"), seeded_text(i, 0x7a7a, 40)))
        .collect();
    let candidates: Vec<Candidate> = (0..200_000u64)
        .map(|i| {
            // Every 1,000th candidate carries a target's text, so the scan has hits to find.
            let text = if i % 1000 == 0 {
                targets_rows[(i / 1000) as usize].1.clone()
            } else {
                seeded_text(i, 0xc0c0, 40)
            };
            cand(&format!("c{i}"), &text, Gold::Option(0), &["yes", "no"])
        })
        .collect();
    let targets: Vec<TargetSet> = vec![("s9-targets".to_owned(), targets_rows)];
    let request = decisions::containment_request_with(
        8,
        0.5,
        &json!({"tool": "qd-prep stress_v6 S9"}),
        "s9-candidates",
        &candidates,
        &targets,
    );
    let input = dir.join("request.bin");
    std::fs::write(&input, &request).expect("request");
    drop(request);
    let out_dir = dir.join("out");
    timed(
        "containment 200k x 50k",
        &[
            "containment".to_owned(),
            "--input".to_owned(),
            s(&input),
            "--out-dir".to_owned(),
            s(&out_dir),
        ],
        Duration::from_secs(1200),
    );
    let exclusions =
        std::fs::read_to_string(out_dir.join(containment::EXCLUSIONS_NAME)).expect("exclusions");
    assert!(
        exclusions.lines().filter(|l| l.starts_with('c')).count() >= 200,
        "the planted hits were not all found"
    );
    cleanup(&dir);
}

#[test]
#[ignore = "S9 scale: run in its own step with --ignored"]
fn s9_lsh_at_one_million_keys_within_ten_minutes() {
    let dir = scratch("s9-lsh");
    let (bands, rows, keys) = (16u32, 8u32, 1_000_000u64);
    let width = u64::from(bands * rows);
    let input = dir.join("request.bin");
    {
        let f = std::fs::File::create(&input).expect("request");
        let mut w = std::io::BufWriter::new(f);
        w.write_all(qd_prep::lsh::INPUT_MAGIC).expect("magic");
        w.write_all(&bands.to_le_bytes()).expect("bands");
        w.write_all(&rows.to_le_bytes()).expect("rows");
        w.write_all(&5_000_000u64.to_le_bytes()).expect("max_pairs");
        w.write_all(&keys.to_le_bytes()).expect("n_keys");
        for _ in 0..keys {
            w.write_all(&8u32.to_le_bytes()).expect("length");
        }
        for i in 0..keys {
            w.write_all(format!("k{i:07}").as_bytes()).expect("key");
        }
        for i in 0..keys {
            // Every 10,000th key repeats its predecessor's signature: a planted pair.
            let src = if i % 10_000 == 1 { i - 1 } else { i };
            for j in 0..width {
                w.write_all(&splitmix64(src * width + j).to_le_bytes())
                    .expect("signature");
            }
        }
        w.flush().expect("flush");
    }
    timed(
        "lsh 1M keys",
        &[
            "lsh".to_owned(),
            "--input".to_owned(),
            s(&input),
            "--output".to_owned(),
            s(&dir.join("reply.bin")),
        ],
        Duration::from_secs(600),
    );
    cleanup(&dir);
}

// =============================================================================================
// S11: held-out inputs (rule 3). A held-out path offered as a training input is refused by every
// producer; `--target` files may sit under one (decontaminating against a held-out set is
// legitimate). The qd-train half is `crates/qd-train/tests/{door,caller_records_are_held_out}.rs`
// and `python/tests/test_stress_v6.py`.

/// Held-out paths in each spelling the marker matches, as the pre-registration names them.
const HELD: [&str; 4] = [
    "heldout/natural-bugs",
    "HeldOut/tssb-3m",
    "held_out/caller-records",
    "held-out/natural-bugs",
];

#[test]
fn s11_synth_refuses_a_held_out_config() {
    let dir = scratch("s11-synth-config");
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    for held in HELD {
        let config = dir.join(held).join("email.json");
        write_json(&config, &email_config(6, &pins_of(&t)));
        let out_dir = dir.join(format!("pool-{}", next()));
        let out = cli(&synth_cli_args(&config, &t, &out_dir));
        assert_cli_refused(
            &out,
            "held-out data",
            &format!("synth --config under {held}"),
        );
        assert_nothing_written(&out_dir, held);
    }
    cleanup(&dir);
}

#[test]
fn s11_synth_refuses_a_held_out_catalog_or_phrasings_file_or_a_symlink_into_one() {
    let dir = scratch("s11-synth-tools-data");
    let data = repo().join("data/synth");
    let t = vec![write_target(&dir.join("targets"), "stand-in")];
    for (key, file) in [
        ("catalog", "tool-catalog-v6-2026-10-06.json"),
        ("phrasings", "tool-phrasings-v6-2026-10-06.json"),
    ] {
        let held = dir.join("heldout/natural-bugs").join(file);
        std::fs::create_dir_all(held.parent().expect("parent")).expect("held dir");
        std::fs::copy(data.join(file), &held).expect("copy the shipped file");
        let link = dir.join("clean").join(format!("link-{file}"));
        std::fs::create_dir_all(link.parent().expect("parent")).expect("clean dir");
        std::os::unix::fs::symlink(&held, &link).expect("symlink");
        for offered in [&held, &link] {
            let cfg = with(
                &shipped_with("tools-v6-2026-10-06.json", &pins_of(&t)),
                &["tools", key],
                json!(s(offered)),
            );
            let config = dir.join("clean").join(format!("tools-{}.json", next()));
            write_json(&config, &cfg);
            let out_dir = dir.join(format!("pool-{}", next()));
            let out = cli(&synth_cli_args(&config, &t, &out_dir));
            assert_cli_refused(
                &out,
                "held-out data",
                &format!("tools.{key} at {}", offered.display()),
            );
            assert_nothing_written(&out_dir, key);
        }
    }
    cleanup(&dir);
}

#[test]
fn s11_synth_accepts_a_held_out_target() {
    let dir = scratch("s11-synth-target");
    let t = vec![write_target(&dir.join("heldout/tssb-3m"), "tssb-target")];
    let config = dir.join("clean").join("email.json");
    write_json(&config, &email_config(6, &pins_of(&t)));
    let out_dir = dir.join("pool");
    let out = cli(&synth_cli_args(&config, &t, &out_dir));
    assert!(
        out.status.success(),
        "a --target under a held-out path was refused: {}",
        stderr_of(&out)
    );
    assert_complete_pool(&out_dir);
    cleanup(&dir);
}

#[test]
fn s11_convert_refuses_a_held_out_config_view_record_or_licence_cache() {
    let dir = scratch("s11-convert-flags");
    let clean = dir.join("clean").join("missing.json");
    for held in HELD {
        let h = dir.join(held).join("input.json");
        for (flag, config, view_record, cache) in [
            ("--config", &h, &clean, None),
            ("--view-record", &clean, &h, None),
            ("--licence-cache", &clean, &clean, Some(&h)),
        ] {
            let out_dir = dir.join(format!("pool-{}", next()));
            let mut args = vec![
                "convert".to_owned(),
                "--config".to_owned(),
                s(config),
                "--view-record".to_owned(),
                s(view_record),
                "--pool".to_owned(),
                "csn".to_owned(),
                "--out-dir".to_owned(),
                s(&out_dir),
            ];
            if let Some(c) = cache {
                args.push("--licence-cache".to_owned());
                args.push(s(c));
            }
            let out = cli(&args);
            assert_cli_refused(
                &out,
                "held-out data",
                &format!("convert {flag} under {held}"),
            );
            assert_nothing_written(&out_dir, flag);
        }
    }
    cleanup(&dir);
}

/// A view-record entry with use `train` whose source path or rows view sits under a held-out
/// segment is refused by `Views::train_rows`, the training reader's only door.
#[test]
fn s11_convert_train_rows_refuses_a_train_entry_under_a_held_out_path() {
    let dir = scratch("s11-convert-train-rows");
    let clean_src = dir.join("clean").join("src.parquet");
    let clean_rows = dir.join("clean").join("rows.jsonl");
    let held_src = dir.join("heldout/tssb-3m").join("src.parquet");
    let held_rows = dir.join("held-out/natural-bugs").join("rows.jsonl");
    let cases = [
        ("held source", &held_src, &clean_rows, true),
        ("held rows view", &clean_src, &held_rows, true),
        ("clean", &clean_src, &clean_rows, false),
    ];
    for (what, src, rows, refused) in cases {
        let fr = json!([{
            "dataset": "d/s", "file": "f.parquet", "sha256": "ab", "use": "train",
            "file_path": s(src)
        }]);
        let fr_path = dir.join(format!("fetch-{}.json", next()));
        write_json(&fr_path, &fr);
        let fr_sha = sha256_hex(&std::fs::read(&fr_path).expect("fetch record"));
        let vr = json!({
            "schema": convert::VIEW_RECORD_SCHEMA, "fetch_record_sha256": fr_sha,
            "fetch_record": s(&fr_path),
            "entries": [{
                "dataset": "d/s", "file": "f.parquet", "source_sha256": "ab", "use": "train",
                "source_path": s(src),
                "views": [{"kind": "rows", "path": s(rows), "sha256": "cd", "rows": 1}]
            }]
        });
        let vr_path = dir.join(format!("view-{}.json", next()));
        write_json(&vr_path, &vr);
        let views = convert::Views::load(&vr_path, &fr_sha).expect("the view record loads");
        let got = views.train_rows("d/s", "f.parquet").map(|v| v.path.clone());
        if refused {
            let e = got.refusal(&format!("{what}: a held-out train entry was read"));
            assert!(e.contains("held-out data"), "{what}: {e}");
        } else {
            assert_eq!(got, Ok(rows.clone()), "{what}");
        }
    }
    cleanup(&dir);
}

/// `decisions::FETCHED` (private): the upstream files `qd-prep decisions` reads, in order.
const DECISIONS_FETCHED: [(&str, &str); 10] = [
    (
        decisions::OPEN_JEV,
        "raw/community-hard-mix-v2-redistributable/train.jsonl.gz",
    ),
    (decisions::PROCEDURAL, "all/train-00000-of-00002.parquet"),
    (decisions::PROCEDURAL, "all/train-00001-of-00002.parquet"),
    (decisions::TYPED, "all/train-00000-of-00001.parquet"),
    (decisions::SYNTH, "data/train.jsonl"),
    (decisions::HELPSTEER, "train.jsonl.gz"),
    (decisions::BOOLQ, "data/train-00000-of-00001.parquet"),
    (decisions::ARC, "ARC-Challenge/train-00000-of-00001.parquet"),
    (decisions::ARC, "ARC-Easy/train-00000-of-00001.parquet"),
    (decisions::VITAMINC, "train.jsonl"),
];

fn decisions_config(fetch_record_sha256: &str) -> Value {
    let empty = sha256_hex(b"");
    json!({
        "schema": decisions::CONFIG_SCHEMA, "seed": 7, "val_fraction": 0.05,
        "mode_threshold": 0.6, "ngram_n": 8, "containment_threshold": 0.5,
        "fetch_record_sha256": fetch_record_sha256,
        "decider_sha256": decisions::DECIDER_FILES.iter().map(|f| ((*f).to_owned(), json!(empty)))
            .collect::<serde_json::Map<_, _>>(),
        "target_sha256": {}, "val_cap_per_family": 1000, "val_floor_per_family": 100,
        "val_cap_total": 6000, "train_caps": {}
    })
}

/// A fetch record naming an empty JSONL file for every fetched source; the first under
/// `first_dir` (held out or not), the rest under `dir/clean`.
fn decisions_fetch_record(dir: &Path, first_dir: &Path) -> PathBuf {
    let empty = sha256_hex(b"");
    let mut entries = Vec::new();
    for (i, (dataset, file)) in DECISIONS_FETCHED.iter().enumerate() {
        let parent = if i == 0 {
            first_dir.to_path_buf()
        } else {
            dir.join("clean")
        };
        std::fs::create_dir_all(&parent).expect("source dir");
        let path = parent.join(format!("source-{i}.jsonl"));
        std::fs::write(&path, b"").expect("source");
        entries.push(json!({
            "dataset": dataset, "file": file, "jsonl": s(&path), "jsonl_sha256": empty
        }));
    }
    let path = dir.join("clean").join(format!("fetch-{}.json", next()));
    write_json(&path, &Value::Array(entries));
    path
}

fn decisions_args(config: &Path, fetch: &Path, decider: &Path, out_dir: &Path) -> Vec<String> {
    vec![
        "decisions".to_owned(),
        "--config".to_owned(),
        s(config),
        "--fetch-record".to_owned(),
        s(fetch),
        "--decider-dir".to_owned(),
        s(decider),
        "--out-dir".to_owned(),
        s(out_dir),
    ]
}

#[test]
fn s11_decisions_refuses_a_held_out_config_fetch_record_source_or_decider_file() {
    let dir = scratch("s11-decisions");
    let clean_decider = dir.join("clean").join("decider");
    let held_decider = dir.join("heldout/natural-bugs").join("decider");
    for d in [&clean_decider, &held_decider] {
        std::fs::create_dir_all(d).expect("decider dir");
        for f in decisions::DECIDER_FILES {
            std::fs::write(d.join(f), b"").expect("decider file");
        }
    }
    let clean_fetch = decisions_fetch_record(&dir, &dir.join("clean"));
    let held_source_fetch = decisions_fetch_record(&dir, &dir.join("HeldOut/tssb-3m"));
    let held_fetch = dir.join("held_out/caller-records").join("fetch.json");
    std::fs::create_dir_all(held_fetch.parent().expect("parent")).expect("dir");
    std::fs::copy(&clean_fetch, &held_fetch).expect("copy");
    let pin = |p: &Path| sha256_hex(&std::fs::read(p).expect("fetch record"));

    let held_config = dir.join("heldout/tssb-3m").join("config.json");
    write_json(&held_config, &decisions_config(&pin(&clean_fetch)));
    let config_for = |fetch: &Path| {
        let p = dir.join("clean").join(format!("config-{}.json", next()));
        write_json(&p, &decisions_config(&pin(fetch)));
        p
    };
    let cases: Vec<(&str, PathBuf, PathBuf, &PathBuf)> = vec![
        (
            "--config",
            held_config.clone(),
            clean_fetch.clone(),
            &clean_decider,
        ),
        (
            "--fetch-record",
            config_for(&held_fetch),
            held_fetch.clone(),
            &clean_decider,
        ),
        (
            "a fetched source",
            config_for(&held_source_fetch),
            held_source_fetch.clone(),
            &clean_decider,
        ),
        (
            "--decider-dir",
            config_for(&clean_fetch),
            clean_fetch.clone(),
            &held_decider,
        ),
    ];
    for (what, config, fetch, decider) in cases {
        let out_dir = dir.join(format!("pool-{}", next()));
        let out = cli(&decisions_args(&config, &fetch, decider, &out_dir));
        assert_cli_refused(&out, "held-out data", &format!("decisions {what} held out"));
        assert_nothing_written(&out_dir, what);
    }
    cleanup(&dir);
}

/// `dedupe --input` and `own-repos --code-root` are not producers of training rows (ruling
/// recorded under S11 in AUDIT/data-stress-suite-2026-10-06.md, 2026-10-06, before the run):
/// `dedupe` is a kernel over a request Python builds, and deduplicating a held-out set is
/// legitimate; `own-repos` reads repositories under a scan root and makes the held-out split.
/// So a held-out path given to either is not refused as held-out data; whatever else they
/// refuse it for is their own check.
#[test]
fn s11_dedupe_and_own_repos_are_not_producers_and_do_not_refuse_a_held_out_path() {
    let dir = scratch("s11-dedupe-own-repos");
    let input = dir.join("heldout/natural-bugs").join("request.bin");
    std::fs::create_dir_all(input.parent().expect("parent")).expect("dir");
    std::fs::write(&input, b"QDPDDIN1 not a request").expect("request");
    let output = dir.join("reply.bin");
    let out = cli(&[
        "dedupe".to_owned(),
        "--input".to_owned(),
        s(&input),
        "--output".to_owned(),
        s(&output),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("held-out data"),
        "dedupe refused a held-out --input as held-out data: {stderr}"
    );
    // The request itself is malformed, so the run still fails, for that reason.
    assert!(!out.status.success() && !output.exists(), "{stderr}");

    let code_root = dir.join("heldout/tssb-3m").join("code");
    std::fs::create_dir_all(&code_root).expect("code root");
    let manifest = dir.join("own-repos.json");
    let out = cli(&[
        "own-repos".to_owned(),
        "--code-root".to_owned(),
        s(&code_root),
        "--out".to_owned(),
        s(&manifest),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("held-out data"),
        "own-repos refused a held-out --code-root as held-out data: {stderr}"
    );
    cleanup(&dir);
}

// =============================================================================================
// S12: licence default-deny in the Rust mirror. `convert_licence::tests` and
// `tests/convert_licence_parity.rs` (Rust against Python on 74 strings) are reused; these add
// NOASSERTION, the empty and whitespace-only strings, and the count each refusal leaves.

fn licence_row(i: usize, licence: Option<&str>) -> convert::Row {
    convert::Row {
        id: format!("r{i}"),
        source_id: "stress/licence",
        family_id: "stress.family",
        stratum: "stress.family/x".to_owned(),
        group_key: format!("g{i}"),
        licence: licence.map(str::to_owned),
        context: format!("an ordinary context for row {i}"),
        question: "Which?".to_owned(),
        slot_name: "answer",
        options: vec!["yes".into(), "no".into()],
        gold: Gold::Option(0),
        label_basis: "hard",
    }
}

#[test]
fn s12_unknown_noassertion_and_empty_licences_are_denied_and_counted() {
    let cfg = convert::Config {
        seed: 7,
        val_fraction: 0.2,
        ngram_n: 8,
        containment_threshold: 0.5,
        fetch_record_sha256: String::new(),
        pools: BTreeMap::new(),
    };
    let mut acc = convert::Acc::new(&cfg);
    let offered: [Option<&str>; 9] = [
        Some("Unknown"),
        Some("NOASSERTION"),
        Some(""),
        Some("   "),
        None,
        Some("Made-Up-Licence-9"),
        Some("  MIT  "),
        Some("mit   LICENSE"),
        Some("MIT\u{1f}License"),
    ];
    for (i, l) in offered.iter().enumerate() {
        acc.offer("stress/licence", "scope", Ok(licence_row(i, *l)))
            .expect("offer");
    }
    let t = &acc.tallies["stress/licence"];
    assert_eq!(t.offered, 9);
    let want: BTreeMap<String, usize> = [
        ("licence_unstated".to_owned(), 3),
        ("licence_needs_human_call:unknown".to_owned(), 1),
        ("licence_needs_human_call:noassertion".to_owned(), 1),
        ("licence_needs_human_call:made-up-licence-9".to_owned(), 1),
    ]
    .into();
    assert_eq!(t.refused, want, "every denial is counted under its reason");
    assert_eq!(t.kept, 3);
    assert_eq!(t.licence_admitted.get("mit"), Some(&3));
    assert!(acc.rows.iter().all(|c| c.licence == "mit"));
}

/// Every table id and alias, in other cases and with other whitespace, resolves to the same
/// id and tier, or is denied; nothing passes unclassified, and an admitted row carries the
/// table's id, never the raw string.
#[test]
fn s12_case_and_whitespace_variants_resolve_or_deny_never_pass_unclassified() {
    let spell = |raw: &str| -> Vec<String> {
        vec![
            raw.to_uppercase(),
            format!("  {raw}\t"),
            raw.replace(' ', "  \u{1f} "),
            raw.chars()
                .enumerate()
                .map(|(i, c)| {
                    if i % 2 == 0 {
                        c.to_ascii_uppercase()
                    } else {
                        c
                    }
                })
                .collect(),
        ]
    };
    let policy: BTreeMap<&str, convert_licence::Tier> =
        convert_licence::POLICY.iter().copied().collect();
    let mut bases: Vec<(&str, &str)> = convert_licence::POLICY
        .iter()
        .map(|(id, _)| (*id, *id))
        .collect();
    bases.extend(convert_licence::ALIASES.iter().copied());
    for (raw, id) in bases {
        let tier = policy[id];
        for v in spell(raw) {
            assert_eq!(
                convert_licence::classify(&v),
                (id.to_owned(), tier),
                "{v:?} did not resolve to {id}"
            );
            match convert_licence::gate(Some(&v)) {
                Ok(convert_licence::Verdict::Admitted(got)) => {
                    assert_eq!(got, id, "{v:?}");
                    assert_eq!(tier, convert_licence::Tier::Allow, "{v:?}");
                }
                Ok(convert_licence::Verdict::Pending(got)) => {
                    panic!("{v:?} is a table id yet gated as pending {got}")
                }
                Err(reason) => {
                    assert_ne!(tier, convert_licence::Tier::Allow, "{v:?}: {reason}");
                    assert!(reason.starts_with("licence_"), "{v:?}: {reason}");
                }
            }
        }
    }
    for raw in [
        "NOASSERTION",
        "noassertion",
        "No Assertion",
        "UNKNOWN",
        "Other",
    ] {
        let got = convert_licence::gate(Some(raw));
        assert!(got.is_err(), "{raw:?} passed the gate: {got:?}");
    }
}
