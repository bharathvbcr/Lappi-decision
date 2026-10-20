//! The builder (`qd-prep containment`) and the gate (`qd-train-data`) share ONE exact-short
//! matcher (`qd_prep::exact_short`, owner ruling 2026-10-19). This pins it: over one case table,
//! the rows the builder excludes are exactly the rows the gate calls hits, and both equal the
//! ruling's expectation (a row's whole stripped text, normalised, equals a short target).

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use qd_preflight::train_data::{DEFAULT_N, DEFAULT_THRESHOLD, Inputs, check};

const TARGETS: [&str; 3] = [
    "Thank you!",
    "Set a timer",
    "the quick brown fox jumps over the lazy dog and then runs far away into the forest",
];

/// (row id, source text, a short target is the whole text of it).
const CASES: [(&str, &str, bool); 8] = [
    ("c1", "THANK  you", true),
    ("c2", "thank you.", true),
    ("c3", "please thank you", false),
    ("c4", "a longer row\nthank you\nmore text", false),
    ("c5", "set a timer", true),
    ("c6", "set a timer now", false),
    ("c7", "thanks", false),
    ("c8", "unrelated words only", false),
];

fn s(out: &mut Vec<u8>, v: &str) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    out.extend_from_slice(v.as_bytes());
}

fn builder_request() -> Vec<u8> {
    let mut out = qd_prep::containment::INPUT_MAGIC.to_vec();
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&0.5f64.to_le_bytes());
    s(&mut out, "{\"rev\": \"x\"}");
    s(&mut out, "{}");
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    s(&mut out, "train");
    s(&mut out, "bench");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.push(1);
    out.extend_from_slice(&((CASES.len() + TARGETS.len()) as u64).to_le_bytes());
    for (id, text, _) in CASES {
        out.extend_from_slice(&0u32.to_le_bytes());
        s(&mut out, id);
        s(&mut out, &format!("identity-{id}"));
        s(&mut out, "fam.a");
        s(&mut out, text);
    }
    for (i, text) in TARGETS.iter().enumerate() {
        out.extend_from_slice(&1u32.to_le_bytes());
        s(&mut out, &format!("t{i}"));
        s(&mut out, &format!("tid-{i}"));
        s(&mut out, "fam.a");
        s(&mut out, text);
    }
    out
}

#[test]
fn builder_and_gate_give_the_same_verdict_on_a_shared_case_table() {
    let dir = std::env::temp_dir().join(format!("qd-exact-short-parity-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("dir");

    // Builder.
    let written = qd_prep::containment::run(&builder_request(), 2).expect("the scan runs");
    let excluded: BTreeSet<String> = String::from_utf8(written.exclusions)
        .expect("utf8")
        .lines()
        .map(|k| k.trim_start_matches("identity-").to_owned())
        .collect();
    let attestation = String::from_utf8(written.attestation).expect("utf8");
    assert!(attestation.contains("\"exact_short\": {"), "{attestation}");

    // Gate, over the same texts and targets.
    let examples: Vec<String> = CASES
        .iter()
        .map(|(id, text, _)| {
            serde_json::json!({"id": id, "source_id": "x/y", "family_id": "fam.a",
                               "context": text, "options": []})
            .to_string()
        })
        .collect();
    let examples_path: PathBuf = dir.join("examples.jsonl");
    fs::write(&examples_path, examples.join("\n") + "\n").expect("examples");
    let targets: Vec<String> = TARGETS
        .iter()
        .enumerate()
        .map(|(i, t)| serde_json::json!({"id": format!("t{i}"), "text": t}).to_string())
        .collect();
    let targets_path = dir.join("targets.jsonl");
    fs::write(&targets_path, targets.join("\n") + "\n").expect("targets");
    let verdict = check(&Inputs {
        train: examples_path.clone(),
        examples: None,
        manifest: None,
        targets: vec![("bench".into(), targets_path)],
        forbid_families: Vec::new(),
        n: DEFAULT_N,
        threshold: DEFAULT_THRESHOLD,
        threads: 2,
    });
    let gate: BTreeSet<String> = verdict
        .tiers
        .iter()
        .find(|t| t.name == "exact_short")
        .expect("tier")
        .sample_ids
        .iter()
        .cloned()
        .collect();

    let expected: BTreeSet<String> = CASES
        .iter()
        .filter(|(_, _, hit)| *hit)
        .map(|(id, _, _)| (*id).to_owned())
        .collect();
    assert_eq!(excluded, expected, "the builder's exclusions");
    assert_eq!(gate, expected, "the gate's hits");
    assert_eq!(excluded, gate, "one matcher, one verdict");
}
