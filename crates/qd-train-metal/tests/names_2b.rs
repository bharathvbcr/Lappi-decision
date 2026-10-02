//! Amendment 2 (ii) test 1 runs qd-train's optimizer table over the 2B's names as Python's
//! model lists them (`crates/qd-train/tests/fixtures/optimizer-table-2b.json`). This checks that
//! those are the names and shapes the provider hands the loop: ojas-qwen35's name map for the
//! same `config.json` (the snapshot's, copied byte for byte by the oracle) gives exactly the 320
//! tower entries with the same shapes. CPU only: the config is parsed and nothing is opened.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use ojas_qwen35::{tower_tensors, Qwen35TextConfig};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../qd-train/tests/fixtures")
}

#[test]
fn ojas_qwen35s_2b_name_map_is_the_table_tests_names() {
    let table: Value = serde_json::from_str(&std::fs::read_to_string(fixtures().join("optimizer-table-2b.json")).unwrap()).unwrap();
    let config = fixtures().join(table["config"]["file"].as_str().unwrap());
    let cfg = Qwen35TextConfig::from_file(&config).unwrap();
    let ours: Vec<(String, Vec<usize>)> = tower_tensors(&cfg).into_iter().map(|t| (t.name, t.shape)).collect();
    let theirs: Vec<(String, Vec<usize>)> = table["tower_names"]
        .as_array()
        .unwrap()
        .iter()
        .zip(table["tower_shapes"].as_array().unwrap())
        .map(|(n, s)| {
            (
                n.as_str().unwrap().to_owned(),
                s.as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect(),
            )
        })
        .collect();
    // Name by name, as test 1 compares: the two lists order their entries differently (tessl's
    // parameter table against torch's `named_parameters`), and the loop keys every per-entry
    // number by the provider's own order.
    let ours_by_name: BTreeMap<String, Vec<usize>> = ours.iter().cloned().collect();
    let theirs_by_name: BTreeMap<String, Vec<usize>> = theirs.iter().cloned().collect();
    assert_eq!((ours.len(), ours_by_name.len()), (320, 320), "320 distinct entries");
    assert_eq!(theirs_by_name.len(), 320);
    assert_eq!(ours_by_name, theirs_by_name, "ojas-qwen35's 2B entries are not the names and shapes the table was validated on");
}
