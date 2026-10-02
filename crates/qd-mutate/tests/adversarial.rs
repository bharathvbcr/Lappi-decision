//! The adversarial corpus from `docs/hardening.md` §1, worked through row by row.
//!
//! Every attack in that table gets a test here, named after the row. The table is a checklist, not
//! inspiration: a row with no test is a row nobody checked.
//!
//! Two invariants are asserted on **every** example any test in this file produces, through
//! [`check_every_example`]:
//!
//! * the recorded span is the one a fresh textual diff of `before` against `after` implies, and it
//!   is inside the post-mutation file;
//! * the mutated text still parses as the language it claims to be.
//!
//! The second is what turns "only where removal still parses" from a promise each operator makes
//! into one the pipeline enforces.

use std::collections::BTreeSet;

use qd_mutate::diffspan::span_from_text_diff;
use qd_mutate::generate::{DiffShape, Generator, Options, Run};
use qd_mutate::lang::LangId;
use qd_mutate::manifest::PoolReport;
use qd_mutate::ops::MutationClass;
use qd_mutate::parse::parse_bounded;
use qd_mutate::pool::PoolRecord;
use qd_mutate::span::{total_lines, LineSpan};

/// Run one source through the generator many times over, under one seed.
///
/// Many *records* rather than many seeds: each record gets its own RNG stream keyed on its id, so
/// this explores the same choice space, and it builds a single [`Generator`] — which probes the
/// formatters once instead of once per seed. On a machine where `swift-format` resolves through
/// `xcrun`, per-seed probing dominates the runtime of this file.
fn run_over(source: &str, extension: &str, copies: usize) -> Run {
    let records: Vec<PoolRecord> = (0..copies)
        .map(|i| PoolRecord {
            id: format!("rec{i}"),
            repo: "o/r".to_string(),
            path: format!("dir{i}/file.{extension}"),
            language: None,
            source: source.to_string(),
            hunks: None,
            prior_source: None,
        })
        .collect();
    let run = Generator::new(Options {
        seed: 20260919,
        clean_permille: 0,
        limit: None,
        languages: Vec::new(),
        max_examples_per_file: 8,
        diff_shape: DiffShape::default(),
    })
    .run(&records, pool_report());
    check_every_example(&run);
    run
}

fn pool_report() -> PoolReport {
    PoolReport {
        path: "inline".to_string(),
        sha256: String::new(),
        records: 0,
        by_language: std::collections::BTreeMap::new(),
    }
}

/// The two invariants, on every example, every time.
fn check_every_example(run: &Run) {
    for example in &run.examples {
        if example.class == MutationClass::Clean {
            assert!(example.span.is_none(), "{}: clean points at nothing", example.id);
            continue;
        }
        let span = example.span.expect("a mutated example carries a span");
        assert_eq!(
            span_from_text_diff(&example.before, &example.after),
            Some(span),
            "{} ({}): the recorded span is not the one the texts imply",
            example.id,
            example.detail
        );
        assert!(
            span.is_well_formed(total_lines(&example.after)),
            "{}: span {span} runs past a {}-line file",
            example.id,
            total_lines(&example.after)
        );
        let language = qd_mutate::lang::for_id(example.language);
        assert!(
            parse_bounded(&language.grammar(), &example.after).is_ok(),
            "{} ({}): the mutation broke the syntax",
            example.id,
            example.detail
        );
    }
    assert_eq!(
        run.manifest.totals.span_disagreements, 0,
        "an example was dropped on span disagreement; the derivations have drifted"
    );
    assert_eq!(
        run.manifest.totals.mutation_did_not_parse, 0,
        "an operator produced output that does not parse"
    );
}

fn operators_used(run: &Run) -> BTreeSet<String> {
    run.examples
        .iter()
        .filter_map(|e| e.operator.clone())
        .collect()
}

// -------------------------------------------------------------------------------------------
// A well-formed baseline, so a later "it produced nothing" means something.
// -------------------------------------------------------------------------------------------

const RUST_BASE: &str = "\
use std::collections::HashMap;
use std::io::Read;

/// Doc comment.
pub async fn total<T>(a: usize, b: usize) -> Result<Vec<u8>, String>
where
    T: Clone,
{
    // a comment
    let mut acc0 = 0;
    for i in 0..10 {
        if a < b {
            acc0 += i;
        } else {
            acc0 -= i;
        }
    }
    let v = read_it(a, b)?;
    other(1, 2).await;
    println!(\"{acc0} {v:?}\");
    Ok(vec![])
}
";

#[test]
fn the_baseline_produces_examples_so_a_later_silence_means_something() {
    let run = run_over(RUST_BASE, "rs", 24);
    assert!(!run.examples.is_empty(), "the baseline produced nothing");
    assert!(
        operators_used(&run).len() >= 3,
        "only {:?} fired on the baseline",
        operators_used(&run)
    );
}

// -------------------------------------------------------------------------------------------
// "CRLF line endings, and a file with mixed CRLF/LF"
// "UTF-8 BOM at file start"
// -------------------------------------------------------------------------------------------

#[test]
fn crlf_line_endings_are_normalized_and_recorded() {
    let crlf = RUST_BASE.replace('\n', "\r\n");
    let run = run_over(&crlf, "rs", 16);
    assert!(!run.examples.is_empty(), "a CRLF file produced nothing");
    for example in &run.examples {
        assert!(example.normalization.crlf, "{}: not recorded", example.id);
        assert!(
            !example.normalization.mixed_endings,
            "a uniformly CRLF file is converted, not mixed"
        );
        assert!(!example.before.contains('\r'), "a CR survived into the example");
        assert!(!example.after.contains('\r'));
    }
}

#[test]
fn a_file_with_mixed_endings_is_flagged_not_merely_converted() {
    // Half the file CRLF, half LF: this is where byte and line offsets diverge part-way down.
    let head = RUST_BASE[..RUST_BASE.len() / 2].replace('\n', "\r\n");
    let mixed = format!("{head}{}", &RUST_BASE[RUST_BASE.len() / 2..]);
    let run = run_over(&mixed, "rs", 16);
    assert!(!run.examples.is_empty(), "a mixed-ending file produced nothing");
    for example in &run.examples {
        assert!(
            example.normalization.mixed_endings,
            "{}: mixed endings were converted without being flagged",
            example.id
        );
        assert!(!example.before.contains('\r'));
    }
}

#[test]
fn a_bom_is_stripped_and_recorded_and_does_not_shift_the_first_line() {
    let with_bom = format!("\u{feff}{RUST_BASE}");
    let run = run_over(&with_bom, "rs", 16);
    assert!(!run.examples.is_empty(), "a BOM'd file produced nothing");
    for example in &run.examples {
        assert!(example.normalization.bom, "{}: the BOM was not recorded", example.id);
        assert!(
            !example.before.starts_with('\u{feff}'),
            "the BOM survived into the example and shifts every byte offset by three"
        );
        assert!(example.before.starts_with("use std::collections"));
    }
}

// -------------------------------------------------------------------------------------------
// "Multi-byte identifiers, emoji in comments and strings"
// -------------------------------------------------------------------------------------------

const RUST_MULTIBYTE: &str = "\
/// 日本語のドキュメント 🎌
pub fn 合計(最初: usize, 最後: usize) -> Result<usize, String> {
    // コメント 🚀 with emoji
    let mut 合計値 = 0;
    for i in 0..10 {
        if 最初 < 最後 {
            合計値 += i;
        }
    }
    let 名前 = \"絵文字 🎌 in a string\";
    println!(\"{名前} {合計値}\");
    Ok(合計値)
}
";

#[test]
fn multibyte_identifiers_and_emoji_do_not_move_a_line_number() {
    let run = run_over(RUST_MULTIBYTE, "rs", 24);
    assert!(!run.examples.is_empty(), "a multi-byte file produced nothing");
    for example in &run.examples {
        // Byte offset != char offset != column; the span is in *lines*, and lines are what must
        // survive. `check_every_example` already cross-checked it; this pins the content too.
        let span = example.span.expect("mutated");
        let lines = qd_mutate::span::lines_in_span(&example.after, span)
            .expect("the span is inside the file");
        assert!(!lines.is_empty());
        assert!(
            example.after.is_char_boundary(0),
            "{}: the output is not valid UTF-8 text",
            example.id
        );
    }
}

// -------------------------------------------------------------------------------------------
// "A string literal containing `todo!()` / `pass` / `NotImplementedError`"
// "A comment containing what looks like the mutated construct"
//
// This is the substring gate the model is meant to beat. The generator must not depend on it.
// -------------------------------------------------------------------------------------------

const RUST_STRING_TRAP: &str = "\
pub fn work(a: usize) -> Result<usize, String> {
    // this comment says todo!() and unimplemented!() on purpose
    let marker = \"todo!()\";
    let other = \"raise NotImplementedError\";
    let mut acc = 0;
    for i in 0..a {
        acc += i;
    }
    println!(\"{marker} {other}\");
    Ok(acc)
}
";

#[test]
fn a_string_literal_saying_todo_does_not_make_a_working_body_read_as_a_stub() {
    let run = run_over(RUST_STRING_TRAP, "rs", 32);
    let used = operators_used(&run);
    assert!(
        used.iter().any(|o| o.starts_with("stub.")),
        "the body does real work, so the stub operators must still fire; got {used:?}"
    );
    // And no stub operator was refused for "already a stub".
    let rust = run
        .manifest
        .languages
        .iter()
        .find(|r| r.language == LangId::Rust)
        .expect("rust report");
    let already_stub_refusals: u64 = ["stub.panic", "stub.default_return", "stub.early_return"]
        .iter()
        .filter_map(|op| rust.operators.get(*op))
        .map(|r| r.refused.get("precondition_unmet").copied().unwrap_or(0))
        .sum();
    assert_eq!(
        already_stub_refusals, 0,
        "a substring in a literal or a comment was mistaken for a stub body"
    );
}

const PYTHON_STRING_TRAP: &str = "\
def work(a: int) -> int:
    # raise NotImplementedError, says the comment
    marker = \"pass\"
    other = \"raise NotImplementedError\"
    acc = 0
    for i in range(a):
        acc += i
    print(marker, other)
    return acc
";

#[test]
fn a_python_string_saying_pass_does_not_make_a_working_body_read_as_a_stub() {
    let run = run_over(PYTHON_STRING_TRAP, "py", 32);
    let used = operators_used(&run);
    assert!(
        used.iter().any(|o| o.starts_with("stub.")),
        "got {used:?}"
    );
}

// -------------------------------------------------------------------------------------------
// "A function whose body is already `todo!()`"
// -------------------------------------------------------------------------------------------

const RUST_ALREADY_STUB: &str = "\
pub fn not_done(a: usize) -> Result<usize, String> {
    todo!()
}
";

#[test]
fn a_body_that_is_already_a_stub_is_refused_by_every_stub_operator() {
    let run = run_over(RUST_ALREADY_STUB, "rs", 16);
    let used = operators_used(&run);
    assert!(
        !used.iter().any(|o| o.starts_with("stub.")),
        "mutating a stub into a stub labels a clean-looking diff as stub; got {used:?}"
    );
    let rust = run
        .manifest
        .languages
        .iter()
        .find(|r| r.language == LangId::Rust)
        .expect("rust report");
    assert!(
        rust.refusals.get("precondition_unmet").copied().unwrap_or(0) > 0,
        "the refusal was not recorded: {:?}",
        rust.refusals
    );
}

// -------------------------------------------------------------------------------------------
// "Nested functions, closures, lambdas, trait default bodies"
// -------------------------------------------------------------------------------------------

const RUST_NESTED: &str = "\
pub fn outer(a: usize) -> Result<usize, String> {
    fn inner(x: usize) -> Result<usize, String> {
        let doubled = x * 2;
        Ok(doubled)
    }
    let closure = |y: usize| -> usize {
        let tripled = y * 3;
        tripled
    };
    let first = inner(a)?;
    let second = closure(a);
    Ok(first + second)
}

trait Thing {
    fn required(&self) -> u32;
    fn defaulted(&self) -> u32 {
        let base = 7;
        base
    }
}
";

#[test]
fn a_nested_function_is_its_own_body_and_says_so() {
    let run = run_over(RUST_NESTED, "rs", 32);
    assert!(!run.examples.is_empty());
    let names: BTreeSet<&str> = run
        .examples
        .iter()
        .map(|e| e.function.symbol.as_str())
        .collect();
    assert!(
        names.contains("inner"),
        "the nested fn was never offered as its own body: {names:?}"
    );
    for example in run.examples.iter().filter(|e| e.function.symbol == "inner") {
        assert!(
            example.is_nested,
            "the mutator must say which node it took, and `inner` is an inner one"
        );
        assert_eq!(example.node_kind, "function_item");
    }
    for example in run.examples.iter().filter(|e| e.function.symbol == "outer") {
        assert!(!example.is_nested);
    }
}

#[test]
fn a_trait_default_body_is_mutable_and_a_bare_signature_is_not() {
    let run = run_over(RUST_NESTED, "rs", 32);
    let names: BTreeSet<&str> = run
        .examples
        .iter()
        .map(|e| e.function.symbol.as_str())
        .collect();
    assert!(
        names.contains("defaulted"),
        "a trait default body has a body and must be mutable: {names:?}"
    );
    assert!(
        !names.contains("required"),
        "a signature with no body has nothing to stub; an empty span would be a poisoned label"
    );
}

#[test]
fn a_closures_sites_are_not_attributed_to_the_enclosing_body() {
    // `closure` binds `tripled` inside a `closure_expression`, which is a function boundary. A
    // *site-based* mutation attributed to `outer` must never be picked from inside it — that is the
    // "which node did the mutator take" ambiguity.
    //
    // Whole-body `stub.*` operators are excluded, and that is not a loophole: replacing `outer`'s
    // entire body necessarily spans the closure's lines, because the closure is part of that body.
    // The ambiguity is about *where a site was found*, not about how much a stub covers.
    let run = run_over(RUST_NESTED, "rs", 32);
    let closure_start = RUST_NESTED.find("|y: usize|").expect("fixture");
    let closure_end = RUST_NESTED.find("let first").expect("fixture");
    let interior = LineSpan::new(
        qd_mutate::span::line_of(RUST_NESTED, closure_start) + 1,
        qd_mutate::span::line_of(RUST_NESTED, closure_end) - 1,
    );
    // Single-site operators only. `stub.*` replaces the whole body, and `cosmetic.rename_local`
    // rewrites every occurrence of one name, so both report a hull that legitimately spans the
    // closure without any edit landing inside it. `source_span` is that hull, not the site list.
    let single_site: Vec<_> = run
        .examples
        .iter()
        .filter(|e| {
            e.function.symbol == "outer"
                && e.class == MutationClass::Logic
                && e.operator.as_deref() != Some("cosmetic.rename_local")
        })
        .collect();
    for example in &single_site {
        let source_span = example.source_span.expect("mutated");
        assert!(
            !interior.intersects(&source_span),
            "{} ({}) reached into the closure body at {source_span}",
            example.id,
            example.detail
        );
    }
    // And the closure's own binding is never offered as one of `outer`'s locals.
    assert!(
        !run.examples
            .iter()
            .any(|e| e.function.symbol == "outer" && e.detail.contains("`tripled`")),
        "a binding inside the closure was renamed as if it belonged to the outer body"
    );
}

const RUST_SHADOWING_CLOSURE: &str = "\
pub fn shadower(seed: usize) -> usize {
    let v = seed + 1;
    let bump = |v: usize| -> usize {
        let doubled = v * 2;
        doubled
    };
    let a = bump(v);
    let b = bump(v + 1);
    a + b
}
";

#[test]
fn a_local_shadowed_by_a_closure_parameter_is_never_renamed() {
    // `v` is bound once by `let`, and every occurrence of it is inside `shadower`'s body — so the
    // "bound once, never escapes" proof passes. But the closure takes its own `v`, and renaming the
    // outer one would rewrite the parameter and its uses too. That diff changes behaviour while
    // wearing a `cosmetic` label, which is the poisoned label the spec names by hand.
    //
    // `Language::local_bindings` stops at a function boundary, so the closure's parameter is not
    // counted as a binding anywhere and the occurrence check alone cannot see the shadow.
    let run = run_over(RUST_SHADOWING_CLOSURE, "rs", 40);
    for example in &run.examples {
        assert!(
            !example.detail.starts_with("local `v` renamed"),
            "{}: {} — the closure parameter shadows this local",
            example.id,
            example.detail
        );
    }
    // The fixture must still be productive, or the test would pass by producing nothing.
    assert!(!run.examples.is_empty(), "the fixture produced nothing");
}

// -------------------------------------------------------------------------------------------
// "`async fn`, generics with `where` clauses, macros with braces"
// -------------------------------------------------------------------------------------------

const RUST_HARD_EXTENTS: &str = "\
use std::collections::HashMap;

macro_rules! shout {
    ($x:expr) => {{
        let loud = $x;
        loud
    }};
}

pub async fn tricky<T, U>(a: usize, b: usize) -> Result<Vec<T>, String>
where
    T: Clone + Default,
    U: Into<T>,
{
    let mapped: HashMap<usize, usize> = HashMap::new();
    let value = shout!({
        let inner = a + b;
        inner
    });
    if a < b {
        return Ok(Vec::new());
    }
    println!(\"{value} {mapped:?}\");
    Ok(Vec::new())
}
";

#[test]
fn async_generics_where_clauses_and_brace_macros_do_not_confuse_the_body_extent() {
    let run = run_over(RUST_HARD_EXTENTS, "rs", 32);
    assert!(!run.examples.is_empty(), "produced nothing");
    let body_start = qd_mutate::span::line_of(RUST_HARD_EXTENTS, RUST_HARD_EXTENTS.find("let mapped").expect("fixture"));
    for example in &run.examples {
        assert_eq!(example.function.symbol, "tricky");
        let source_span = example.source_span.expect("mutated");
        assert!(
            source_span.start >= body_start,
            "{} ({}) landed at {source_span}, above the body's first line {body_start} — a \
             brace-counting heuristic got the extent wrong",
            example.id,
            example.detail
        );
    }
}

// -------------------------------------------------------------------------------------------
// "An empty body / a declaration with no body (trait sig, interface)"
// "A single-line function"
// -------------------------------------------------------------------------------------------

const RUST_EDGE_BODIES: &str = "\
pub fn empty() {}

pub fn one_line() -> u32 { 3 }

trait Only {
    fn sig(&self) -> u32;
}
";

#[test]
fn an_empty_body_is_skipped_rather_than_emitted_with_an_empty_span() {
    let run = run_over(RUST_EDGE_BODIES, "rs", 16);
    for example in &run.examples {
        assert_ne!(
            example.function.symbol, "empty",
            "an empty body has nothing to mutate"
        );
        assert_ne!(example.function.symbol, "sig", "a signature has no body at all");
    }
}

#[test]
fn a_single_line_function_gets_a_span_whose_ends_are_equal() {
    let run = run_over(RUST_EDGE_BODIES, "rs", 16);
    // In-place operators only. `cosmetic.reformat` is the formatter expanding
    // `fn one_line() -> u32 { 3 }` into three lines, which is a three-line span and correct;
    // `check_every_example` has already cross-checked it against a fresh diff. The row this test
    // covers is the one where an off-by-one hides: a mutation *within* a one-line body.
    let single: Vec<_> = run
        .examples
        .iter()
        .filter(|e| {
            e.function.symbol == "one_line"
                && matches!(e.class, MutationClass::Stub | MutationClass::Logic)
        })
        .collect();
    assert!(!single.is_empty(), "the single-line function produced nothing");
    for example in single {
        let span = example.span.expect("mutated");
        assert_eq!(
            span.start, span.end,
            "{} ({}): a single-line body's span must not straddle two lines — this is where an \
             off-by-one hides",
            example.id, example.detail
        );
        assert_eq!(span.line_count(), 1);
    }
}

// -------------------------------------------------------------------------------------------
// "Last line of file with no trailing newline"
// -------------------------------------------------------------------------------------------

#[test]
fn a_file_with_no_trailing_newline_keeps_its_span_inside_the_file() {
    let no_newline = RUST_BASE.trim_end_matches('\n');
    assert!(!no_newline.ends_with('\n'));
    let run = run_over(no_newline, "rs", 24);
    assert!(!run.examples.is_empty(), "produced nothing");
    for example in &run.examples {
        let span = example.span.expect("mutated");
        assert!(
            span.end <= total_lines(&example.after),
            "{}: span {span} runs past EOF in a {}-line file",
            example.id,
            total_lines(&example.after)
        );
    }
}

// -------------------------------------------------------------------------------------------
// "A file that is 1 byte, 0 bytes, or 10 MB"
// "Deeply nested blocks (1000 levels)"
// "A file tree-sitter parses with ERROR nodes"
// -------------------------------------------------------------------------------------------

/// Generate without the invariant checks, for inputs expected to produce nothing at all.
fn run_raw(source: &str, extension: &str) -> Run {
    let record = PoolRecord {
        id: "solo".to_string(),
        repo: "o/r".to_string(),
        path: format!("a.{extension}"),
        language: None,
        source: source.to_string(),
        hunks: None,
            prior_source: None,
    };
    Generator::new(Options {
        seed: 1,
        clean_permille: 0,
        limit: None,
        languages: vec![LangId::Rust],
        max_examples_per_file: 8,
        diff_shape: DiffShape::default(),
    })
    .run(std::slice::from_ref(&record), pool_report())
}

#[test]
fn a_zero_byte_and_a_one_byte_file_are_refused_by_name() {
    let empty = run_raw("", "rs");
    assert!(empty.examples.is_empty());
    assert_eq!(empty.manifest.refusals.get("empty_source"), Some(&1));

    let one_byte = run_raw("x", "rs");
    assert!(one_byte.examples.is_empty());
    assert!(
        one_byte.manifest.refusals.contains_key("parse_errors")
            || one_byte.manifest.refusals.contains_key("no_function_body"),
        "a one-byte file must be refused by name, not crash: {:?}",
        one_byte.manifest.refusals
    );
}

#[test]
fn a_ten_megabyte_file_is_refused_by_size_not_discovered_as_a_timeout() {
    let mut big = String::with_capacity(11 * 1024 * 1024);
    while big.len() < 10 * 1024 * 1024 + 16 {
        big.push_str("pub fn f() -> u32 { 1 }\n");
    }
    let run = run_raw(&big, "rs");
    assert!(run.examples.is_empty());
    assert_eq!(run.manifest.refusals.get("too_large"), Some(&1));
}

#[test]
fn a_thousand_levels_of_nesting_is_a_bounded_refusal_not_a_stack_overflow() {
    let nested = format!(
        "pub fn deep() {{\n{}\n{}\n}}\n",
        "{ ".repeat(1000),
        "}".repeat(1000)
    );
    let run = run_raw(&nested, "rs");
    assert!(run.examples.is_empty());
    let refused: u64 = ["too_deep", "parse_errors", "parse_budget", "too_many_nodes"]
        .iter()
        .filter_map(|k| run.manifest.refusals.get(*k))
        .sum();
    assert_eq!(
        refused, 1,
        "expected exactly one bounded refusal, got {:?}",
        run.manifest.refusals
    );
}

#[test]
fn a_file_with_error_nodes_is_refused_rather_than_mutated_into_nonsense() {
    let broken = "pub fn ok() -> u32 { 1 }\npub fn @@@ not rust {\n";
    let run = run_raw(broken, "rs");
    assert!(
        run.examples.is_empty(),
        "mutating an unparsed file produces nonsense with a confident label"
    );
    assert_eq!(run.manifest.refusals.get("parse_errors"), Some(&1));
}

// -------------------------------------------------------------------------------------------
// The same corpus, across the other four languages.
// -------------------------------------------------------------------------------------------

const GO_BASE: &str = "\
package main

import (
\t\"fmt\"
\t\"os\"
)

func Total(a int, b int) ([]byte, error) {
\tacc0 := 0
\tfor i := 0; i < 10; i++ {
\t\tif a < b {
\t\t\tacc0 += i
\t\t} else {
\t\t\tacc0 -= i
\t\t}
\t}
\tv, err := readIt(a, b)
\tif err != nil {
\t\treturn nil, err
\t}
\tfmt.Println(acc0, os.Args)
\treturn v, nil
}
";

const PYTHON_BASE: &str = "\
import os
import sys


def total(a: int, b: int) -> int:
    \"\"\"Sum things.\"\"\"
    # a comment
    acc0 = 0
    for i in range(10):
        if a < b:
            acc0 += i
        else:
            acc0 -= i
    print(os.getpid(), sys.argv)
    return acc0
";

const TS_BASE: &str = "\
import { one } from \"./one\";
import { two } from \"./two\";

export async function total(a: number, b: number): Promise<number[]> {
  // a comment
  let acc0 = 0;
  for (let i = 0; i < 10; i++) {
    if (a < b) {
      acc0 += i;
    } else {
      acc0 -= i;
    }
  }
  try {
    await readIt(a, b);
  } catch (e) {
    throw e;
  }
  console.log(acc0, one, two);
  return [];
}
";

const SWIFT_BASE: &str = "\
import Foundation
import os

func total(a: Int, b: Int) throws -> [UInt8] {
    // a comment
    var acc0 = 0
    for i in 0..<10 {
        if a < b {
            acc0 += i
        } else {
            acc0 -= i
        }
    }
    let v = try readIt(a, b)
    print(acc0, v)
    return []
}
";

fn language_fixture(id: LangId) -> (&'static str, &'static str) {
    match id {
        LangId::Rust => (RUST_BASE, "rs"),
        LangId::Go => (GO_BASE, "go"),
        LangId::Python => (PYTHON_BASE, "py"),
        LangId::TypeScript => (TS_BASE, "ts"),
        LangId::Swift => (SWIFT_BASE, "swift"),
    }
}

#[test]
fn every_language_survives_the_byte_offset_attacks() {
    for id in LangId::ALL {
        let (base, extension) = language_fixture(id);
        // CRLF, a BOM, and no trailing newline, all at once.
        let hostile = format!("\u{feff}{}", base.replace('\n', "\r\n"));
        let hostile = hostile.trim_end_matches("\r\n").to_string();
        let run = run_over(&hostile, extension, 12);
        assert!(
            !run.examples.is_empty(),
            "{id}: the hostile variant produced nothing at all"
        );
        for example in &run.examples {
            assert!(example.normalization.bom, "{id}: BOM not recorded");
            assert!(example.normalization.crlf, "{id}: CRLF not recorded");
            assert!(!example.before.contains('\r'), "{id}: a CR survived");
        }
    }
}

#[test]
fn every_language_produces_at_least_one_example_from_a_plain_file() {
    for id in LangId::ALL {
        let (base, extension) = language_fixture(id);
        let run = run_over(base, extension, 16);
        assert!(
            !run.examples.is_empty(),
            "{id}: no example from the plain fixture — the facade may be finding nothing"
        );
        assert!(
            run.examples.iter().all(|e| e.language == id),
            "{id}: an example was attributed to another language"
        );
    }
}

#[test]
fn every_language_refuses_a_file_with_error_nodes() {
    let broken: [(LangId, &str, &str); 5] = [
        (LangId::Rust, "rs", "pub fn ok() -> u32 { 1 }\nfn @@@ ! {\n"),
        (LangId::Go, "go", "package p\nfunc f() {\n\tif ( {\n"),
        (LangId::Python, "py", "def f():\n    return (((\n"),
        (LangId::TypeScript, "ts", "function f() {\n  const = ;\n"),
        (LangId::Swift, "swift", "func f() {\n    let = = =\n"),
    ];
    for (id, extension, source) in broken {
        let record = PoolRecord {
            id: "solo".to_string(),
            repo: "o/r".to_string(),
            path: format!("a.{extension}"),
            language: Some(id),
            source: source.to_string(),
            hunks: None,
            prior_source: None,
        };
        let run = Generator::new(Options {
            seed: 1,
            clean_permille: 0,
            limit: None,
            languages: vec![id],
            max_examples_per_file: 8,
            diff_shape: DiffShape::default(),
        })
        .run(std::slice::from_ref(&record), pool_report());
        assert!(
            run.examples.is_empty(),
            "{id}: an unparsed file was mutated anyway"
        );
        assert!(
            run.manifest.refusals.contains_key("parse_errors"),
            "{id}: the refusal was not recorded as parse_errors: {:?}",
            run.manifest.refusals
        );
    }
}

#[test]
fn a_declaration_without_a_body_is_skipped_in_every_language_that_has_one() {
    // Trait signature, interface member, abstract method, protocol requirement — the same shape
    // under four names. None of them has a body, and emitting an empty span for one would be a
    // poisoned label.
    let cases: [(LangId, &str, &str, &str); 4] = [
        (
            LangId::Rust,
            "rs",
            "trait T {\n    fn sig(&self) -> u32;\n    fn body(&self) -> u32 {\n        let a = 1;\n        a\n    }\n}\n",
            "sig",
        ),
        (
            LangId::Go,
            "go",
            "package p\ntype I interface {\n\tSig() int\n}\nfunc Body() int {\n\tacc := 1\n\treturn acc\n}\n",
            "Sig",
        ),
        (
            LangId::TypeScript,
            "ts",
            "interface I { sig(): number }\nabstract class C {\n  abstract sig2(): void;\n  body(): number {\n    const a = 1;\n    return a;\n  }\n}\n",
            "sig",
        ),
        (
            LangId::Swift,
            "swift",
            "protocol P {\n    func sig() -> Int\n}\nextension P {\n    func body() -> Int {\n        let a = 1\n        return a\n    }\n}\n",
            "sig",
        ),
    ];
    for (id, extension, source, bodyless) in cases {
        let run = run_over(source, extension, 12);
        assert!(
            run.examples.iter().all(|e| e.function.symbol != bodyless),
            "{id}: `{bodyless}` has no body and must not be mutated"
        );
        assert!(
            !run.examples.is_empty(),
            "{id}: the fixture's real body produced nothing, so the test proves nothing"
        );
    }
}
