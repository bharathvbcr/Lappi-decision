//! `qd-noul-rows`, end to end through the binary: determinism, counts, and that nothing the
//! allowlist leaves out reaches a row.
//!
//! The allowlist here is written by hand, so this file does not depend on Python. The one the
//! real corpus is generated from comes from the canonical split (`tools/noul_allowlist.py`);
//! `python/tests/test_defect_noul.py` runs that path, and the loader's rule-3 re-check.
//!
//! `--forms v2` (defect-noul-v2, the v3.1 re-form) is tested here too: the exact share per
//! form, each form's shape, and that the default (`v1`) output stays pinned byte for byte to
//! what the generator wrote before v2 existed.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use sha2::{Digest, Sha256};

const BIN: &str = env!("CARGO_BIN_EXE_qd-noul-rows");
const PER_SOURCE: usize = 4;

/// A word that only appears in content the allowlist excludes.
const MARK: &str = "zqexcludedmark";

const PARA: &str = "The survey of the northern valley recorded that the river moved two \
    kilometres east over a century, cutting new channels through farmland and leaving lakes \
    where the old meanders once ran, and later work compared the accounts with photographs.";

const PY: &str = "import os\nimport sys\n\n\ndef NAME(path, defaults=None):\n    defaults = \
    defaults or {}\n    with open(path) as handle:\n        raw = handle.read()\n    values = \
    dict(defaults)\n    for line in raw.splitlines():\n        key, _, value = \
    line.partition('=')\n        values[key.strip()] = value.strip()\n    return values\n";
const GO: &str = "package conf\n\nimport \"strings\"\n\nfunc NAME(raw string) map[string]string \
    {\n\tout := map[string]string{}\n\tfor _, line := range strings.Split(raw, \"\\n\") {\n\t\t\
    key, value, ok := strings.Cut(line, \"=\")\n\t\tif ok {\n\t\t\tout[key] = value\n\t\t}\n\t}\n\t\
    return out\n}\n";
const RS: &str = "use std::collections::HashMap;\n\npub fn NAME(raw: &str) -> HashMap<String, \
    String> {\n    let mut out = HashMap::new();\n    for line in raw.lines() {\n        if let \
    Some((k, v)) = line.split_once('=') {\n            out.insert(k.to_string(), \
    v.to_string());\n        }\n    }\n    out\n}\n";
const TS: &str = "export function NAME(raw: string): Record<string, string> {\n  const out: \
    Record<string, string> = {};\n  for (const line of raw.split('\\n')) {\n    const [key, \
    value] = line.split('=');\n    if (key) out[key] = value ?? '';\n  }\n  return out;\n}\n\n\
    export default NAME;\n";

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().expect("run qd-noul-rows")
}

struct World {
    dir: PathBuf,
    squad: PathBuf,
    pool: PathBuf,
    allowlist: PathBuf,
    /// A v3-shaped `code.defect_class` corpus over the pool's files: `examples.jsonl` and the
    /// `manifest.json` that pins it and names the pool it was generated from.
    corpus: PathBuf,
}

/// Question `q` of three a fixture paragraph carries. Long enough for a word 8-gram alone.
fn question(title: &str, k: usize, q: usize) -> String {
    format!("Which new channels did the river of {title} section {k} cut through farmland in \
             account {q}?")
}

/// A one-hunk diff over every line of `source`, line 4 replaced: the corpus's own shape.
fn one_hunk(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let n = lines.len();
    let mut out = format!("@@ -1,{n} +1,{n} @@\n");
    for (j, l) in lines.iter().enumerate() {
        if j == 3 {
            out.push_str(&format!("-{l}\n+{l} edited\n"));
        } else {
            out.push_str(&format!(" {l}\n"));
        }
    }
    out
}

/// Two hunks over `source`: lines 1..=6 and 8..=13, one replaced line in each.
fn two_hunks(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::new();
    for (start, change) in [(0usize, 2usize), (7, 9)] {
        out.push_str(&format!("@@ -{},6 +{},6 @@\n", start + 1, start + 1));
        for (j, l) in lines.iter().enumerate().skip(start).take(6) {
            if j == change {
                out.push_str(&format!("-{l}\n+{l} edited\n"));
            } else {
                out.push_str(&format!(" {l}\n"));
            }
        }
    }
    out
}

fn world(name: &str) -> World {
    let dir = std::env::temp_dir().join(format!("qd-noul-rows-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut squad = String::new();
    let mut titles = serde_json::Map::new();
    for t in 0..4 {
        let title = format!("Kept {t}");
        titles.insert(title.clone(), serde_json::json!(format!("squad-title:{title}")));
        for k in 0..2 {
            for q in 0..3 {
                squad.push_str(&serde_json::json!({
                    "title": title, "context": format!("{title} part {k}. {PARA}"),
                    "question": question(&title, k, q),
                }).to_string());
                squad.push('\n');
            }
        }
    }
    squad.push_str(&serde_json::json!({
        "title": "Held", "context": format!("Held {MARK}. {PARA}"),
        "question": format!("Which {MARK} question about the held river is never asked here?"),
    }).to_string());
    squad.push('\n');
    let squad_path = dir.join("train.jsonl");
    std::fs::write(&squad_path, &squad).unwrap();

    // Two files per language, each in its own repo; the last python file is excluded.
    let mut pool = String::new();
    let mut files = serde_json::Map::new();
    let langs = [("py", PY), ("go", GO), ("rs", RS), ("ts", TS)];
    let mut n_records = 0u64;
    let mut corpus = String::new();
    for i in 0..3 {
        for (ext, body) in langs {
            if i == 2 && ext != "py" {
                continue;
            }
            let id = format!("c{i}:pkg/m{i}.{ext}");
            let excluded = i == 2;
            let name = if excluded { format!("{MARK}_{i}") } else { format!("load_{i}") };
            pool.push_str(&serde_json::json!({
                "id": id, "repo": format!("org/{ext}{i}"), "path": format!("pkg/m{i}.{ext}"),
                "source": body.replace("NAME", &name), "hunks": [{"start_line": 5, "end_line": 7}],
            }).to_string());
            pool.push('\n');
            let source = body.replace("NAME", &name);
            let diff = if i == 0 && ext == "py" { two_hunks(&source) } else { one_hunk(&source) };
            let language = match ext {
                "py" => "python",
                "go" => "go",
                "rs" => "rust",
                _ => "typescript",
            };
            corpus.push_str(&serde_json::json!({
                "id": format!("{id}#0"), "pool_id": id, "repo": format!("org/{ext}{i}"),
                "path": format!("pkg/m{i}.{ext}"), "language": language, "class": "logic",
                "diff": diff, "before": "unused", "after": "unused",
            }).to_string());
            corpus.push('\n');
            n_records += 1;
            if !excluded {
                files.insert(id, serde_json::json!("mit"));
            }
        }
    }
    let pool_path = dir.join("pool.jsonl");
    std::fs::write(&pool_path, &pool).unwrap();

    let units: serde_json::Value =
        serde_json::from_slice(&run(&["units"]).stdout).expect("units is JSON");
    let all: Vec<String> = units["units"]
        .as_array()
        .unwrap()
        .iter()
        .map(|u| u.as_str().unwrap().to_string())
        .collect();
    // Every unit but one per language, so the excluded ones are named and checkable.
    let excluded_units: Vec<&String> = all.iter().filter(|u| u.ends_with("/getopts")
        || u.ends_with("/guards") || u.ends_with("/lazy-config") || u.ends_with("/null-guard")
        || u.ends_with("/array-column") || u.ends_with("/sealed-result")).collect();
    let allowed: Vec<&String> = all.iter().filter(|u| !excluded_units.contains(u)).collect();

    let allow = serde_json::json!({
        "schema": "qd-noul-allowlist/v1",
        "split": {"seed": 20260919u64, "train_fraction": 0.9, "val_fraction": 0.05},
        "invisible_format_ranges": [[0x200B, 0x200F], [0xFEFF, 0xFEFF]],
        "squad": {"file": "train.jsonl", "sha256": sha256(squad.as_bytes()),
                  "licence": "cc-by-sa-4.0", "titles": titles, "excluded": {"split:heldout": 1}},
        "pool": {"file": "pool.jsonl", "sha256": sha256(pool.as_bytes()), "records": n_records,
                 "files": files, "excluded": {"split:val": 1}},
        "templates": {"catalogue_sha256": units["catalogue_sha256"], "licence": "apache-2.0",
                      "licence_basis": "authored here", "units": allowed,
                      "excluded": {"split:val": excluded_units.len()}},
    });
    let allowlist = dir.join("allowlist.json");
    std::fs::write(&allowlist, allow.to_string()).unwrap();
    let corpus_dir = dir.join("corpus");
    std::fs::create_dir_all(&corpus_dir).unwrap();
    std::fs::write(corpus_dir.join("examples.jsonl"), &corpus).unwrap();
    std::fs::write(corpus_dir.join("manifest.json"), serde_json::json!({
        "examples_sha256": sha256(corpus.as_bytes()),
        "pool": {"path": "/elsewhere/pool.jsonl", "sha256": sha256(pool.as_bytes()),
                 "records": n_records},
    }).to_string()).unwrap();
    World { dir, squad: squad_path, pool: pool_path, allowlist, corpus: corpus_dir }
}

fn generate(w: &World, out: &Path, extra: &[&str]) -> Output {
    generate_n(w, out, PER_SOURCE, extra)
}

fn generate_n(w: &World, out: &Path, per_source: usize, extra: &[&str]) -> Output {
    let n = per_source.to_string();
    let mut args = vec![
        "generate", "--allowlist", w.allowlist.to_str().unwrap(), "--squad",
        w.squad.to_str().unwrap(), "--pool", w.pool.to_str().unwrap(), "--out",
        out.to_str().unwrap(), "--per-source", &n,
    ];
    args.extend_from_slice(extra);
    run(&args)
}

#[test]
fn the_same_inputs_and_seed_give_byte_identical_files() {
    let w = world("determinism");
    let (a, b) = (w.dir.join("a"), w.dir.join("b"));
    assert!(generate(&w, &a, &[]).status.success());
    assert!(generate(&w, &b, &[]).status.success());
    for f in ["examples.jsonl", "manifest.json"] {
        assert_eq!(std::fs::read(a.join(f)).unwrap(), std::fs::read(b.join(f)).unwrap(), "{f}");
    }
    let c = w.dir.join("c");
    assert!(generate(&w, &c, &["--seed", "9"]).status.success());
    assert_ne!(
        std::fs::read(a.join("examples.jsonl")).unwrap(),
        std::fs::read(c.join("examples.jsonl")).unwrap()
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(a.join("manifest.json")).unwrap()).unwrap();
    let body = std::fs::read(a.join("examples.jsonl")).unwrap();
    assert_eq!(manifest["examples_sha256"], sha256(&body));
}

#[test]
fn every_source_gets_its_share_and_nothing_excluded_reaches_a_row() {
    let w = world("counts");
    let out = w.dir.join("out");
    let done = generate(&w, &out, &[]);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    let text = std::fs::read_to_string(out.join("examples.jsonl")).unwrap();
    assert!(!text.contains(MARK), "excluded content reached a row");
    let rows: Vec<serde_json::Value> =
        text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.len(), 3 * PER_SOURCE);
    for source in ["prose", "scrambled", "unseen-language"] {
        let n = rows.iter().filter(|r| r["noul_source"] == source).count();
        assert_eq!(n, PER_SOURCE, "{source}");
    }
    for r in &rows {
        assert_eq!(r["class"], "noul");
        let unit = r["repo"].as_str().unwrap();
        assert!(!unit.ends_with("/getopts") && !unit.ends_with("/guards"), "{unit}");
        assert!(r["diff"].as_str().unwrap().starts_with("@@ -"));
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["totals"]["examples"], 3 * PER_SOURCE);
    assert_eq!(manifest["pool"]["path"], "pool.jsonl");
    assert_eq!(manifest["split"]["seed"], 20260919u64);
}

#[test]
fn inputs_other_than_the_allowlisted_bytes_are_refused() {
    let w = world("refusals");
    let mut squad = std::fs::read_to_string(&w.squad).unwrap();
    squad.push('\n');
    std::fs::write(&w.squad, squad).unwrap();
    let done = generate(&w, &w.dir.join("x"), &[]);
    assert_eq!(done.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&done.stderr).contains("re-run the allowlist step"));

    // Four allowed titles at two paragraphs each hold eight: a share of forty is refused, and
    // nothing is written.
    let w = world("short");
    let done = generate_n(&w, &w.dir.join("y"), 40, &[]);
    assert_eq!(done.status.code(), Some(2), "a share the inputs cannot fill is refused");
    let stderr = String::from_utf8_lossy(&done.stderr);
    assert!(stderr.contains("eligible paragraphs") && stderr.contains("40 were asked for"),
            "{stderr}");
    assert!(!w.dir.join("y").join("examples.jsonl").exists());
}

// -- v1 pinned, v2 (defect-noul-v2) -------------------------------------------------------------

/// What `qd-noul-rows generate` wrote for this world before `--forms` existed: the binary built
/// at 7fc3af7 (R2's generator, unmodified), default flags. The real corpus is pinned the same
/// way by `python/tests/test_defect_noul.py` against `data/pool/defect-noul-v1/manifest.json`.
const V1_EXAMPLES_SHA256: &str = "98853a4e5e6ffc224aa7c38cd66378122005ada4ad554711cbe91b9b73c1bbef";
const V1_MANIFEST_SHA256: &str = "5b2573791e6a4ba9d83cc27885275e37a88473a12441245e15edc71f793a52dc";
/// What `--forms v2` wrote for this world at a7c6ee5 (defect-noul-v2's generator, merged): v3.1
/// was trained on its real output, so a later change that moved these bytes would leave v2
/// unreproducible. The real corpus is pinned against `data/pool/defect-noul-v2` by
/// `python/tests/test_defect_noul.py`.
const V2_EXAMPLES_SHA256: &str = "2ad7bac72e33ec7f005d704a1c77185f6c8d19de1f5b57b42798b3bfb0d2d853";
const V2_MANIFEST_SHA256: &str = "e930d7103581077a6e837e52c2db65a0d05b8a626c026c0d3709ebc0cfbb67ad";

fn rows_of(out: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(out.join("examples.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn manifest_of(out: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(out.join("manifest.json")).unwrap()).unwrap()
}

fn generate_v2(w: &World, out: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["--forms", "v2", "--corpus", w.corpus.to_str().unwrap()];
    args.extend_from_slice(extra);
    generate(w, out, &args)
}

/// `pool_id -> diff` of the fixture corpus.
fn corpus_diffs(w: &World) -> std::collections::BTreeMap<String, String> {
    std::fs::read_to_string(w.corpus.join("examples.jsonl"))
        .unwrap()
        .lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            (v["pool_id"].as_str().unwrap().to_string(), v["diff"].as_str().unwrap().to_string())
        })
        .collect()
}

/// A diff's hunks: each header with its body lines.
fn hunks(diff: &str) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for line in diff.lines() {
        if line.starts_with("@@ -") {
            out.push((line.to_string(), Vec::new()));
        } else {
            out.last_mut().expect("a body line before any header").1.push(line.to_string());
        }
    }
    out
}

fn sorted<T: Ord + Clone>(v: &[T]) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort();
    v
}

#[test]
fn v1_output_is_pinned_byte_for_byte_and_is_the_default() {
    let w = world("pin");
    let (a, b) = (w.dir.join("a"), w.dir.join("b"));
    let done = generate(&w, &a, &[]);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    assert_eq!(sha256(&std::fs::read(a.join("examples.jsonl")).unwrap()), V1_EXAMPLES_SHA256);
    assert_eq!(sha256(&std::fs::read(a.join("manifest.json")).unwrap()), V1_MANIFEST_SHA256);
    let done = generate(&w, &b, &["--forms", "v1"]);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    for f in ["examples.jsonl", "manifest.json"] {
        assert_eq!(std::fs::read(a.join(f)).unwrap(), std::fs::read(b.join(f)).unwrap(), "{f}");
    }
    assert!(rows_of(&a).iter().all(|r| r.get("noul_form").is_none()));
}

#[test]
fn v2_output_is_pinned_byte_for_byte() {
    let w = world("pin2");
    let out = w.dir.join("a");
    let done = generate_v2(&w, &out, &[]);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    assert_eq!(sha256(&std::fs::read(out.join("examples.jsonl")).unwrap()), V2_EXAMPLES_SHA256);
    assert_eq!(sha256(&std::fs::read(out.join("manifest.json")).unwrap()), V2_MANIFEST_SHA256);
}

#[test]
fn v2_at_twice_the_share_doubles_every_form_and_repeats_no_source_but_templates() {
    // noul-v3 is `--forms v2` at twice v2's per-source count (Fable I-2). Every form doubles in
    // exact halves; paragraphs, question groups and scrambled repos stay one row each, and only
    // template units carry several rows -- the instantiations the manifest's units count shows.
    let w = world("double");
    let out = w.dir.join("a");
    let n = 2 * PER_SOURCE;
    let done = generate_n(&w, &out, n, &["--forms", "v2", "--corpus", w.corpus.to_str().unwrap()]);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    let rows = rows_of(&out);
    assert_eq!(rows.len(), 3 * n);
    let mut by_form: std::collections::BTreeMap<String, usize> = Default::default();
    let mut sources: std::collections::BTreeSet<String> = Default::default();
    for r in &rows {
        let form = r["noul_form"].as_str().unwrap().to_string();
        *by_form.entry(form.clone()).or_insert(0) += 1;
        let diff = r["diff"].as_str().unwrap();
        let text: Vec<&str> = diff.lines().skip(1).flat_map(|l| l[1..].split_whitespace()).collect();
        let source = match form.as_str() {
            // A paragraph row's source is the paragraph itself.
            "paragraph" => format!("paragraph:{}", text.join(" ")),
            // A question row's source is the paragraph whose questions it holds: the fixture's
            // questions name it as "<title> section <k>".
            "question" => {
                let joined = text.join(" ");
                let at = joined.find(" section ").expect("a fixture question names its section");
                let k = &joined[at + 9..at + 10];
                format!("question:{}:{k}", r["squad_title"].as_str().unwrap())
            }
            "lines" | "tokens" => format!("scrambled:{}", r["repo"].as_str().unwrap()),
            _ => continue,
        };
        assert!(sources.insert(source.clone()), "a source repeated: {source}");
    }
    let want: std::collections::BTreeMap<String, usize> = [
        ("paragraph", n / 2), ("question", n - n / 2), ("lines", n / 2), ("tokens", n - n / 2),
        ("template", n),
    ].into_iter().map(|(f, k)| (f.to_string(), k)).collect();
    assert_eq!(by_form, want);
    let manifest = manifest_of(&out);
    assert_eq!(manifest["per_source"], n);
    assert_eq!(manifest["totals"]["units_by_source"]["scrambled"], n, "one repo per row");
}

#[test]
fn v2_fills_every_form_exactly_and_is_deterministic() {
    let w = world("v2counts");
    let (a, b, c) = (w.dir.join("a"), w.dir.join("b"), w.dir.join("c"));
    let done = generate_v2(&w, &a, &[]);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    let rows = rows_of(&a);
    assert_eq!(rows.len(), 3 * PER_SOURCE);
    let mut by: std::collections::BTreeMap<(String, String), usize> = Default::default();
    for r in &rows {
        let key = (r["noul_source"].as_str().unwrap().to_string(),
                   r["noul_form"].as_str().expect("a v2 row names its form").to_string());
        *by.entry(key).or_insert(0) += 1;
        assert_eq!(r["class"], "noul");
        assert!(r["diff"].as_str().unwrap().starts_with("@@ -"), "{r}");
    }
    let half = PER_SOURCE / 2;
    let want: std::collections::BTreeMap<(String, String), usize> = [
        (("prose", "paragraph"), half), (("prose", "question"), PER_SOURCE - half),
        (("scrambled", "lines"), half), (("scrambled", "tokens"), PER_SOURCE - half),
        (("unseen-language", "template"), PER_SOURCE),
    ].into_iter().map(|((s, f), n)| ((s.to_string(), f.to_string()), n)).collect();
    assert_eq!(by, want);
    let text = std::fs::read_to_string(a.join("examples.jsonl")).unwrap();
    assert!(!text.contains(MARK), "excluded content reached a row");
    // Scrambled rows: allowlisted pool files only, one per repo, and their licence.
    let scrambled: Vec<&serde_json::Value> =
        rows.iter().filter(|r| r["noul_source"] == "scrambled").collect();
    let repos: std::collections::BTreeSet<&str> =
        scrambled.iter().map(|r| r["repo"].as_str().unwrap()).collect();
    assert_eq!(repos.len(), scrambled.len());
    for r in &scrambled {
        assert!(!r["pool_id"].as_str().unwrap().starts_with("c2:"), "{r}");
        assert_eq!(r["licence"], "mit");
    }

    let manifest = manifest_of(&a);
    assert_eq!(manifest["forms"], "v2");
    assert_eq!(manifest["totals"]["examples"], 3 * PER_SOURCE);
    assert_eq!(manifest["totals"]["by_form"]["question"], PER_SOURCE - half);
    assert_eq!(manifest["totals"]["by_form"]["lines"], half);
    let corpus_body = std::fs::read(w.corpus.join("examples.jsonl")).unwrap();
    assert_eq!(manifest["corpus"]["examples_sha256"], sha256(&corpus_body));
    assert_eq!(manifest["examples_sha256"], sha256(text.as_bytes()));

    assert!(generate_v2(&w, &b, &[]).status.success());
    for f in ["examples.jsonl", "manifest.json"] {
        assert_eq!(std::fs::read(a.join(f)).unwrap(), std::fs::read(b.join(f)).unwrap(), "{f}");
    }
    assert!(generate_v2(&w, &c, &["--seed", "9"]).status.success());
    assert_ne!(std::fs::read(a.join("examples.jsonl")).unwrap(),
               std::fs::read(c.join("examples.jsonl")).unwrap());
}

#[test]
fn v2_scrambled_rows_are_shuffles_of_their_own_corpus_diff() {
    let w = world("v2shape");
    let out = w.dir.join("out");
    let done = generate_v2(&w, &out, &[]);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    let originals = corpus_diffs(&w);
    let mut seen = 0;
    for r in rows_of(&out).iter().filter(|r| r["noul_source"] == "scrambled") {
        let diff = r["diff"].as_str().unwrap();
        let original = &originals[r["pool_id"].as_str().unwrap()];
        assert_ne!(diff, original.as_str(), "an unscrambled row");
        let (got, want) = (hunks(diff), hunks(original));
        assert_eq!(got.len(), want.len());
        // Every hunk keeps its header and its own lines; only the order moved.
        let got_heads: Vec<&String> = got.iter().map(|h| &h.0).collect();
        let want_heads: Vec<&String> = want.iter().map(|h| &h.0).collect();
        assert_eq!(sorted(&got_heads), sorted(&want_heads));
        for (head, body) in &got {
            let (_, orig_body) = want.iter().find(|h| &h.0 == head).unwrap();
            match r["noul_form"].as_str().unwrap() {
                "lines" => {
                    assert_eq!(sorted(body), sorted(orig_body), "{head}");
                }
                "tokens" => {
                    let key = |l: &String| {
                        let (mark, rest) = l.split_at(1);
                        let mut toks: Vec<&str> = rest.split_whitespace().collect();
                        toks.sort();
                        format!("{mark}{}", toks.join(" "))
                    };
                    let a: Vec<String> = body.iter().map(key).collect();
                    let b: Vec<String> = orig_body.iter().map(key).collect();
                    assert_eq!(sorted(&a), sorted(&b), "{head}");
                    // The tokens moved inside lines, not only the lines.
                    let rewritten = body.iter().filter(|l| !orig_body.contains(l)).count();
                    assert!(rewritten >= 1, "{head}: no line had its tokens shuffled");
                }
                other => panic!("scrambled form {other}"),
            }
        }
        seen += 1;
    }
    assert_eq!(seen, PER_SOURCE);
}

#[test]
fn v2_question_rows_are_short_questions_of_one_allowlisted_paragraph() {
    let w = world("v2questions");
    let out = w.dir.join("out");
    let done = generate_v2(&w, &out, &[]);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    let mut seen = 0;
    for r in rows_of(&out).iter().filter(|r| r["noul_form"] == "question") {
        let title = r["squad_title"].as_str().unwrap();
        assert!(title.starts_with("Kept "), "{title}");
        assert_eq!(r["repo"], format!("squad-title:{title}"));
        let diff = r["diff"].as_str().unwrap();
        assert_eq!(diff.matches("@@ -").count(), 1, "{diff}");
        let body: Vec<&str> = diff.lines().skip(1).collect();
        assert!(body.len() >= 2, "{diff}");
        let words: Vec<&str> = body.iter().flat_map(|l| l[1..].split_whitespace()).collect();
        let text = words.join(" ");
        // One to three of one paragraph's questions, whole, in some order.
        let found = (0..2).any(|k| {
            let qs: Vec<String> = (0..3).map(|q| question(title, k, q)).collect();
            let mut rest = text.as_str();
            let mut used = 0;
            while !rest.is_empty() {
                let Some(q) = qs.iter().find(|q| rest.starts_with(q.as_str())) else {
                    return false;
                };
                rest = rest[q.len()..].trim_start();
                used += 1;
            }
            (1..=3).contains(&used)
        });
        assert!(found, "{text}");
        seen += 1;
    }
    assert_eq!(seen, PER_SOURCE - PER_SOURCE / 2);
}

#[test]
fn v2_refuses_a_corpus_it_cannot_trust() {
    let w = world("v2refusals");
    let done = generate(&w, &w.dir.join("a"), &["--forms", "v2"]);
    assert_eq!(done.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&done.stderr).contains("--corpus"));
    let corpus = w.corpus.to_str().unwrap();
    let done = generate(&w, &w.dir.join("b"), &["--corpus", corpus]);
    assert_eq!(done.status.code(), Some(2), "v1 does not read a corpus");

    // Bytes the corpus's own manifest does not pin.
    let examples = w.corpus.join("examples.jsonl");
    let original = std::fs::read(&examples).unwrap();
    let mut changed = original.clone();
    changed.extend_from_slice(b"\n");
    std::fs::write(&examples, &changed).unwrap();
    let done = generate_v2(&w, &w.dir.join("c"), &[]);
    assert_eq!(done.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&done.stderr).contains("examples_sha256"),
            "{}", String::from_utf8_lossy(&done.stderr));
    std::fs::write(&examples, &original).unwrap();

    // A corpus generated from another pool than the allowlist's.
    let manifest = w.corpus.join("manifest.json");
    let mut m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    m["pool"]["sha256"] = serde_json::json!("0".repeat(64));
    std::fs::write(&manifest, m.to_string()).unwrap();
    let done = generate_v2(&w, &w.dir.join("d"), &[]);
    assert_eq!(done.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&done.stderr).contains("pool"),
            "{}", String::from_utf8_lossy(&done.stderr));
    assert!(!w.dir.join("d").join("examples.jsonl").exists());
}
