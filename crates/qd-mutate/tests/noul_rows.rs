//! `qd-noul-rows`, end to end through the binary: determinism, counts, and that nothing the
//! allowlist leaves out reaches a row.
//!
//! The allowlist here is written by hand, so this file does not depend on Python. The one the
//! real corpus is generated from comes from the canonical split (`tools/noul_allowlist.py`);
//! `python/tests/test_defect_noul.py` runs that path, and the loader's rule-3 re-check.

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
            squad.push_str(&serde_json::json!({
                "title": title, "context": format!("{title} part {k}. {PARA}"), "question": "?"
            }).to_string());
            squad.push('\n');
        }
    }
    squad.push_str(&serde_json::json!({
        "title": "Held", "context": format!("Held {MARK}. {PARA}"), "question": "?"
    }).to_string());
    squad.push('\n');
    let squad_path = dir.join("train.jsonl");
    std::fs::write(&squad_path, &squad).unwrap();

    // Two files per language, each in its own repo; the last python file is excluded.
    let mut pool = String::new();
    let mut files = serde_json::Map::new();
    let langs = [("py", PY), ("go", GO), ("rs", RS), ("ts", TS)];
    let mut n_records = 0u64;
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
    World { dir, squad: squad_path, pool: pool_path, allowlist }
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
