//! `cluster_key`: the function identity every row carries, from `docs/hardening.md` §1
//! (the "mutants of one function landed in different splits" survey, M4).
//!
//! The key is read off the serialized row, which is what a consumer sees. Every test builds its own
//! sources, and the assertions are on equality and inequality of keys: what goes into the key is
//! documented on `qd_mutate::cluster`.

use std::collections::{BTreeMap, BTreeSet};

use qd_mutate::generate::{DiffShape, Generator, Options};
use qd_mutate::manifest::PoolReport;
use qd_mutate::pool::PoolRecord;

fn options(clean_permille: u32) -> Options {
    Options {
        seed: 20261017,
        clean_permille,
        limit: None,
        languages: Vec::new(),
        max_examples_per_file: 64,
        max_per_function: usize::MAX,
        diff_shape: DiffShape::default(),
    }
}

/// One emitted row, as a consumer of the JSONL sees it.
struct Row {
    symbol: String,
    class: String,
    key: Option<String>,
}

/// Every row a source yields, over `copies` records that share one `(repo, path)` -- so each
/// function is offered `copies` times and its mutants vary.
fn rows(source: &str, repo: &str, path: &str, clean_permille: u32, copies: usize) -> Vec<Row> {
    let records: Vec<PoolRecord> = (0..copies)
        .map(|i| PoolRecord {
            id: format!("rec{i}"),
            repo: repo.to_string(),
            path: path.to_string(),
            language: None,
            source: source.to_string(),
            hunks: None,
            prior_source: None,
        })
        .collect();
    let run = Generator::new(options(clean_permille)).run(
        &records,
        PoolReport {
            path: "inline".to_string(),
            sha256: String::new(),
            records: 0,
            by_language: BTreeMap::new(),
        },
    );
    run.examples
        .iter()
        .map(|e| {
            let json = serde_json::to_value(e).expect("an example serialises");
            Row {
                symbol: e.function.symbol.clone(),
                class: e.class.as_str().to_string(),
                key: json
                    .get("cluster_key")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            }
        })
        .collect()
}

/// Symbol -> the set of keys its rows carry. Panics on a row with no key: that is the point.
fn keys_by_symbol(rows: &[Row]) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for row in rows {
        let key = row.key.as_ref().unwrap_or_else(|| {
            panic!(
                "a {} row of `{}` carries no cluster_key",
                row.class, row.symbol
            )
        });
        out.entry(row.symbol.clone())
            .or_default()
            .insert(key.clone());
    }
    out
}

fn one_key(map: &BTreeMap<String, BTreeSet<String>>, symbol: &str) -> String {
    let keys = map
        .get(symbol)
        .unwrap_or_else(|| panic!("no row for `{symbol}`: {:?}", map.keys()));
    assert_eq!(
        keys.len(),
        1,
        "`{symbol}` must have exactly one key across all its rows, got {keys:?}"
    );
    keys.iter().next().unwrap().clone()
}

const RUST_TWO_FNS: &str = "\
pub fn alpha(a: i64, b: i64) -> i64 {
    if a > b {
        return a - b;
    }
    let total = a + b;
    total * 2
}

pub fn beta(x: i64) -> i64 {
    let y = x + 1;
    if y == 4 {
        return 0;
    }
    y
}
";

const RUST_TWO_FNS_REFORMATTED: &str = "\
pub fn alpha( a : i64 ,
              b : i64 ) -> i64
{
        if a > b { return a - b; }
        // an added comment
        let total = a + b ;
        total * 2
}
pub fn beta(x: i64) -> i64 { let y = x + 1;
    if y == 4 { return 0; } y }
";

#[test]
fn every_row_carries_a_key_and_a_function_has_one_across_mutated_and_clean_rows() {
    let mutated = rows(RUST_TWO_FNS, "o/r", "src/lib.rs", 0, 12);
    let clean = rows(RUST_TWO_FNS, "o/r", "src/lib.rs", 1000, 1);
    assert!(
        mutated.iter().any(|r| r.class != "clean"),
        "no mutants were produced"
    );
    assert!(clean.iter().all(|r| r.class == "clean"));
    let mut all = mutated;
    all.extend(clean);
    let keys = keys_by_symbol(&all);
    // Mutated, cosmetic and clean rows of one function: one key.
    let alpha = one_key(&keys, "alpha");
    let beta = one_key(&keys, "beta");
    assert_ne!(alpha, beta);
}

#[test]
fn the_same_function_reformatted_has_the_same_key() {
    let plain = keys_by_symbol(&rows(RUST_TWO_FNS, "o/r", "src/lib.rs", 0, 12));
    let ugly = keys_by_symbol(&rows(RUST_TWO_FNS_REFORMATTED, "o/r", "src/lib.rs", 0, 12));
    assert_eq!(one_key(&plain, "alpha"), one_key(&ugly, "alpha"));
    assert_eq!(one_key(&plain, "beta"), one_key(&ugly, "beta"));
}

#[test]
fn identical_bodies_in_different_files_have_different_keys() {
    let a = keys_by_symbol(&rows(RUST_TWO_FNS, "o/r", "src/a.rs", 0, 12));
    let b = keys_by_symbol(&rows(RUST_TWO_FNS, "o/r", "src/b.rs", 0, 12));
    assert_ne!(one_key(&a, "alpha"), one_key(&b, "alpha"));
    // And across repositories, for the same path and the same text.
    let c = keys_by_symbol(&rows(RUST_TWO_FNS, "other/repo", "src/a.rs", 0, 12));
    assert_ne!(one_key(&a, "alpha"), one_key(&c, "alpha"));
}

const RUST_NESTED: &str = "\
fn outer(n: i64) -> i64 {
    fn helper(m: i64) -> i64 {
        let k = m + 1;
        if k > 3 { return k - 1; }
        k
    }
    let c = |q: i64| { if q > 2 { q - 1 } else { q + 1 } };
    let r = helper(n) + c(n);
    if r > 10 { return r - 10; }
    r
}

fn other(n: i64) -> i64 {
    fn helper(m: i64) -> i64 {
        let k = m + 1;
        if k > 3 { return k - 1; }
        k
    }
    let r = helper(n) + 2;
    if r > 10 { return r - 10; }
    r
}
";

#[test]
fn a_nested_function_has_its_own_key_distinct_from_its_parent_and_from_a_namesake() {
    let keys = keys_by_symbol(&rows(RUST_NESTED, "o/r", "src/n.rs", 0, 24));
    // `helper` is a nested function of two different parents: two functions, so two keys under
    // one symbol, and neither is a parent's key.
    assert_eq!(
        keys["helper"].len(),
        2,
        "outer::helper and other::helper differ: {keys:?}"
    );
    let outer = one_key(&keys, "outer");
    let other = one_key(&keys, "other");
    assert!(!keys["helper"].contains(&outer) && !keys["helper"].contains(&other));
    assert_ne!(outer, other);
    // The closure is not a body (`function_bodies` takes named declarations), so no row names it.
    assert_eq!(
        keys.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        BTreeSet::from(["helper", "other", "outer"])
    );
}

#[test]
fn crlf_and_a_bom_do_not_move_the_key() {
    let lf = keys_by_symbol(&rows(RUST_TWO_FNS, "o/r", "src/lib.rs", 0, 12));
    let crlf_bom = format!("\u{feff}{}", RUST_TWO_FNS.replace('\n', "\r\n"));
    let twin = keys_by_symbol(&rows(&crlf_bom, "o/r", "src/lib.rs", 0, 12));
    assert_eq!(one_key(&lf, "alpha"), one_key(&twin, "alpha"));
    assert_eq!(one_key(&lf, "beta"), one_key(&twin, "beta"));
}

#[test]
fn a_renamed_function_is_a_different_function_with_a_different_key() {
    // Right, not a gap: the key is a function's identity, and an identity that followed renames
    // would need a similarity judgement. `FunctionIdentity` already treats the symbol as part of
    // identity. A rename that crosses a split is the near-duplicate check's job (MinHash over the
    // body), not this key's.
    let before = keys_by_symbol(&rows(RUST_TWO_FNS, "o/r", "src/lib.rs", 0, 12));
    let renamed = RUST_TWO_FNS.replace("fn alpha", "fn alpha_renamed");
    let after = keys_by_symbol(&rows(&renamed, "o/r", "src/lib.rs", 0, 12));
    assert_ne!(one_key(&before, "alpha"), one_key(&after, "alpha_renamed"));
    assert_eq!(one_key(&before, "beta"), one_key(&after, "beta"));
}

const GO_METHODS: &str = "\
package p

type A struct{}
type B struct{}

func (a *A) Get(x int) int {
	if x > 1 {
		return x - 1
	}
	return x + 1
}

func (b *B) Get(x int) int {
	if x > 1 {
		return x - 1
	}
	return x + 1
}
";

#[test]
fn methods_of_the_same_name_on_different_receivers_have_different_keys() {
    let keys = keys_by_symbol(&rows(GO_METHODS, "o/r", "p.go", 0, 24));
    assert_eq!(
        keys["Get"].len(),
        2,
        "A.Get and B.Get are different functions: {keys:?}"
    );
    // The receiver's variable name is layout-level, not identity.
    let renamed_receiver = GO_METHODS.replace("(a *A)", "(self *A)");
    let again = keys_by_symbol(&rows(&renamed_receiver, "o/r", "p.go", 0, 24));
    assert_eq!(keys["Get"], again["Get"]);
}
