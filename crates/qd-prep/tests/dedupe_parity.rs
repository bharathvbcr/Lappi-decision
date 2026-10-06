//! `qd-prep dedupe` pinned to the Python reference's own answers.
//!
//! `tests/fixtures/dedupe-parity.json` is written by
//! `python/tests/test_qd_prep_dedupe_parity.py::build_fixture` from `qd_data.dedupe.dedupe`, and
//! that module's `test_the_committed_fixture_is_the_references_answer` fails if the file and the
//! reference ever disagree. So every expectation here is the oracle's, not a transcription: v5's
//! lexical keep, v6's split-priority keep under the agreement prefilter, and a truncated search.

use qd_prep::dedupe;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/dedupe-parity.json");

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_u64().expect("a u64"))
        .collect()
}

fn le_u32(n: usize) -> [u8; 4] {
    u32::try_from(n).expect("fits u32").to_le_bytes()
}

/// The `QDPDDIN1` request for `case` over the fixture's units, as the Python test encodes it.
fn request(fixture: &Value, case: &Value) -> Vec<u8> {
    let units = fixture["units"].as_array().expect("units");
    let family = &fixture["family"];
    let key_hex = family["key_hex"].as_str().expect("key_hex");
    let key: Vec<u8> = (0..key_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&key_hex[i..i + 2], 16).expect("hex"))
        .collect();
    let (a, b) = (u64s(&family["a"]), u64s(&family["b"]));
    let field = |name: &str| case[name].as_u64().expect(name);
    let mut out = dedupe::INPUT_MAGIC.to_vec();
    out.extend_from_slice(
        &case["threshold"]
            .as_f64()
            .expect("threshold")
            .to_bits()
            .to_le_bytes(),
    );
    for name in ["bands", "rows", "min_agreement_permille", "keep_rule"] {
        out.extend_from_slice(&u32::try_from(field(name)).expect("u32").to_le_bytes());
    }
    out.extend_from_slice(&field("max_pairs").to_le_bytes());
    out.extend_from_slice(&(units.len() as u64).to_le_bytes());
    for name in ["key", "repo"] {
        let strings: Vec<&str> = units
            .iter()
            .map(|u| u[name].as_str().expect(name))
            .collect();
        for s in &strings {
            out.extend_from_slice(&le_u32(s.len()));
        }
        for s in &strings {
            out.extend_from_slice(s.as_bytes());
        }
    }
    for u in units {
        out.push(u8::try_from(u["split_rank"].as_u64().expect("rank")).expect("u8"));
    }
    out.extend_from_slice(b"QDPMHIN1");
    out.extend_from_slice(&le_u32(a.len()));
    out.extend_from_slice(&le_u32(key.len()));
    out.extend_from_slice(&key);
    for v in a.iter().chain(&b) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(units.len() as u64).to_le_bytes());
    for u in units {
        let shingles: Vec<&str> = u["shingles"]
            .as_array()
            .expect("shingles")
            .iter()
            .map(|s| s.as_str().expect("a shingle"))
            .collect();
        out.extend_from_slice(&le_u32(shingles.len()));
        for s in &shingles {
            out.extend_from_slice(&le_u32(s.len()));
        }
        for s in &shingles {
            out.extend_from_slice(s.as_bytes());
        }
    }
    out
}

#[test]
fn every_fixture_case_is_the_references_answer() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("the fixture parses");
    let keys: Vec<&str> = fixture["units"]
        .as_array()
        .expect("units")
        .iter()
        .map(|u| u["key"].as_str().expect("key"))
        .collect();
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 3);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let buf = request(&fixture, case);
        for threads in [1, 3] {
            let got = dedupe::parse(&buf)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .run(threads)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let want = &case["expect"];
            assert_eq!(
                got.truncated,
                want["truncated"].as_bool().expect("flag"),
                "{name}"
            );
            for (field, value) in [
                ("n_candidate_pairs", got.n_candidate_pairs),
                ("n_cross_repo_pairs", got.n_cross_repo_pairs),
                ("n_within_repo_pairs", got.n_within_repo_pairs),
            ] {
                assert_eq!(value, want[field].as_u64().expect(field), "{name}: {field}");
            }
            let clusters = want["clusters"].as_array().expect("clusters");
            assert_eq!(got.clusters.len(), clusters.len(), "{name}");
            for (g, w) in got.clusters.iter().zip(clusters) {
                assert_eq!(
                    keys[g.kept as usize],
                    w["kept"].as_str().expect("kept"),
                    "{name}"
                );
                let dropped: Vec<&str> = g.dropped.iter().map(|d| keys[*d as usize]).collect();
                let want_dropped: Vec<&str> = w["dropped"]
                    .as_array()
                    .expect("dropped")
                    .iter()
                    .map(|d| d.as_str().expect("key"))
                    .collect();
                assert_eq!(dropped, want_dropped, "{name}");
                assert_eq!(
                    g.min_edge_jaccard.to_bits(),
                    w["min_edge_jaccard_bits"].as_u64().expect("bits"),
                    "{name}"
                );
            }
        }
    }
}

#[test]
fn the_keep_rule_alone_moves_the_survivor_to_the_eval_copy() {
    // The v5 and v6 cases differ in the prefilter too; this isolates the keep rule: the v6 case
    // re-run under rule 0 keeps exactly what v5 kept.
    let fixture: Value = serde_json::from_str(FIXTURE).expect("the fixture parses");
    let cases = fixture["cases"].as_array().expect("cases");
    let mut v6 = cases[1].clone();
    v6["keep_rule"] = Value::from(dedupe::KEEP_LEXICAL);
    let got = dedupe::parse(&request(&fixture, &v6))
        .expect("parses")
        .run(2)
        .expect("runs");
    let keys: Vec<&str> = fixture["units"]
        .as_array()
        .expect("units")
        .iter()
        .map(|u| u["key"].as_str().expect("key"))
        .collect();
    let kept: Vec<&str> = got.clusters.iter().map(|c| keys[c.kept as usize]).collect();
    let v5_kept: Vec<&str> = cases[0]["expect"]["clusters"]
        .as_array()
        .expect("clusters")
        .iter()
        .map(|c| c["kept"].as_str().expect("kept"))
        .collect();
    assert_eq!(kept, v5_kept);
}

#[test]
fn malformed_requests_are_refused() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("the fixture parses");
    let good = request(&fixture, &fixture["cases"][1]);
    assert!(dedupe::parse(&good).is_ok());
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(
        dedupe::parse(&magic)
            .unwrap_err()
            .contains("does not start")
    );
    let mut trailing = good.clone();
    trailing.push(0);
    assert!(
        dedupe::parse(&trailing)
            .unwrap_err()
            .contains("follow the last")
    );
    // threshold at offset 8, bands 16, rows 20, permille 24, keep_rule 28.
    let mut nan = good.clone();
    nan[8..16].copy_from_slice(&f64::NAN.to_bits().to_le_bytes());
    assert!(dedupe::parse(&nan).unwrap_err().contains("threshold"));
    let mut permille = good.clone();
    permille[24..28].copy_from_slice(&1001u32.to_le_bytes());
    assert!(
        dedupe::parse(&permille)
            .unwrap_err()
            .contains("min_agreement_permille")
    );
    let mut rule = good.clone();
    rule[28..32].copy_from_slice(&2u32.to_le_bytes());
    assert!(dedupe::parse(&rule).unwrap_err().contains("keep_rule"));
    let mut wide = good.clone();
    wide[16..20].copy_from_slice(&64u32.to_le_bytes());
    assert!(
        dedupe::parse(&wide)
            .unwrap_err()
            .contains("permutations signed")
    );
    let mut dup = fixture.clone();
    let first = dup["units"][0]["key"].clone();
    dup["units"][1]["key"] = first;
    let dup_buf = request(&dup, &dup["cases"][1]);
    assert!(
        dedupe::parse(&dup_buf)
            .unwrap_err()
            .contains("appears twice")
    );
}
