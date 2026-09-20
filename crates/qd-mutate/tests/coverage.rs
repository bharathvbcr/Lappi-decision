//! Per-language operator coverage, pinned as a fact.
//!
//! `docs/mutation-operators.md`: *"A language whose facade returns an empty list for a construct
//! simply yields no mutations of that kind. That is correct and must be **recorded per language**,
//! so a coverage report shows which operators actually fired where — rather than a silent zero
//! reading as 'no opportunities'."*
//!
//! Recording it in the manifest is half the job. The other half is this file, which pins the
//! current answer: for each language, every operator either finds sites in a fixture built to
//! contain its construct, or appears in that language's explicitly listed exceptions with a reason.
//! An operator that quietly stops finding anything — a grammar upgrade renaming a node kind is the
//! obvious way — turns this red instead of turning the mixture thin in silence.
//!
//! The fixtures are deliberately maximal: each one contains every construct the seventeen operators
//! look for, so a zero here is a statement about the facade and not about the fixture.

use std::collections::BTreeSet;

use qd_mutate::generate::{Generator, Options, Run};
use qd_mutate::lang::LangId;
use qd_mutate::manifest::{LanguageReport, PoolReport};
use qd_mutate::ops::OpId;
use qd_mutate::pool::PoolRecord;

const RUST: &str = "\
use std::collections::HashMap;
use std::io::Read;

pub async fn total<T>(first: usize, second: usize) -> Result<Vec<u8>, String>
where
    T: Clone,
{
    // a comment inside the body
    let mut acc0 = 0;
    // The extra spaces are the `cosmetic.reformat` site, and they are deliberately nowhere near the
    // over-long line below: rustfmt wraps that one *and adds a trailing comma*, which is a
    // non-whitespace change, and if the two deviations were adjacent the line alignment would merge
    // them into a single hunk that the operator then — correctly — declines.
    let spaced   =   1;
    for i in 0..10 {
        if first < second {
            acc0 += 7;
        } else {
            acc0 -= 3;
        }
    }
    let guard = mutex.lock().unwrap();
    let fetched = read_it(first, second)?;
    remote(first, second).await;
    let label = \"a plain literal\";
    let long_call = compute_something(first_part_of_it, second_part_of_it, third_part_of_it, fourth_part);
    println!(\"{acc0} {guard:?} {fetched:?} {label} {long_call:?} {spaced}\");
    Ok(Vec::new())
}

pub fn pick(first: usize, second: usize) -> usize {
    let chosen = first + 7;
    chosen + second
}
";

const GO: &str = "\
package main

import (
\t\"fmt\"
\t\"os\"
)

func Total(first int, second int) ([]byte, error) {
\t// a comment inside the body
\tacc0 := 0
\tfor i := 0; i < 10; i++ {
\t\tif first < second {
\t\t\tacc0 += 7
\t\t} else {
\t\t\tacc0 -= 3
\t\t}
\t}
\tmu.Lock()
\tv, err := readIt(first, second)
\tif err != nil {
\t\treturn nil, err
\t}
\tlabel := \"a plain literal\"
\tspaced   :=   1
\tfmt.Println(acc0, label, os.Args, v, spaced)
\treturn v, nil
}

func Pick(first int, second int) int {
\tchosen := first + 7
\treturn chosen + second
}
";

const PYTHON: &str = "\
import os
import sys


async def total(first: int, second: int) -> int:
    \"\"\"Add things up.\"\"\"
    # a comment inside the body
    acc0 = 0
    for i in range(10):
        if first < second:
            acc0 += 7
        else:
            acc0 -= 3
    try:
        await remote(first, second)
    except OSError:
        raise
    label = \"a plain literal\"
    spaced   =   1
    print(acc0, label, os.getpid(), sys.argv, spaced)
    return acc0


def pick(first: int, second: int) -> int:
    chosen = first + 7
    return chosen + second
";

const TYPESCRIPT: &str = "\
import { one } from \"./one\";
import { two } from \"./two\";

export async function total(first: number, second: number): Promise<number[]> {
  // a comment inside the body
  let acc0 = 0;
  for (let i = 0; i < 10; i++) {
    if (first < second) {
      acc0 += 7;
    } else {
      acc0 -= 3;
    }
  }
  try {
    await remote(first, second);
  } catch (e) {
    throw e;
  }
  const label = \"a plain literal\";
  console.log(acc0, label, one, two, pick(first, second));
  return [];
}

export function pick(first: number, second: number): number {
  const chosen = first + 7;
  return chosen + second;
}
";

const SWIFT: &str = "\
import Foundation
import os

func total(first: Int, second: Int) async throws -> [UInt8] {
    // a comment inside the body
    var acc0 = 0
    for i in 0..<10 {
        if first < second {
            acc0 += 7
        } else {
            acc0 -= 3
        }
    }
    lock.lock()
    let fetched = try readIt(first, second)
    await remote(first, second)
    let label = \"a plain literal\"
    let spaced   =   1
    print(acc0, label, fetched, spaced)
    return []
}

func pick(first: Int, second: Int) -> Int {
    let chosen = first + 7
    return chosen + second
}
";

fn fixture(id: LangId) -> (&'static str, &'static str) {
    match id {
        LangId::Rust => (RUST, "rs"),
        LangId::Go => (GO, "go"),
        LangId::Python => (PYTHON, "py"),
        LangId::TypeScript => (TYPESCRIPT, "ts"),
        LangId::Swift => (SWIFT, "swift"),
    }
}

/// Operators that find nothing in the maximal fixture, with the reason each one is expected to.
///
/// Every entry is a language-level fact, not a fixture gap. Anything **not** listed here must find
/// at least one site, or the facade has stopped seeing a construct it used to see.
fn expected_silent(id: LangId) -> Vec<(OpId, &'static str)> {
    match id {
        LangId::Rust => vec![],
        LangId::Go => vec![(
            OpId::CosmeticWrapLine,
            "Go inserts semicolons at line ends; the operator is refused for the language",
        )],
        LangId::Python => vec![
            (
                OpId::CosmeticWrapLine,
                "Python is whitespace-sensitive; the operator is refused for the language",
            ),
            (
                OpId::CosmeticReorderImports,
                "Python imports execute in order; the facade reports the group as not reorderable",
            ),
        ],
        LangId::TypeScript => vec![
            (
                OpId::CosmeticWrapLine,
                "automatic semicolon insertion makes a wrap unsafe; refused for the language",
            ),
            (
                OpId::CosmeticReorderImports,
                "an ES module is evaluated when it is imported, in source order; refused for the \
                 language, not for the missing formatter",
            ),
        ],
        LangId::Swift => vec![(
            OpId::CosmeticWrapLine,
            "a newline terminates a statement in Swift; refused for the language",
        )],
    }
}

fn run_language(id: LangId) -> Run {
    let (source, extension) = fixture(id);
    let records: Vec<PoolRecord> = (0..8)
        .map(|i| PoolRecord {
            id: format!("rec{i}"),
            repo: "o/r".to_string(),
            path: format!("pkg{i}/file.{extension}"),
            language: Some(id),
            source: source.to_string(),
            hunks: None,
        })
        .collect();
    Generator::new(Options {
        seed: 20260919,
        clean_permille: 0,
        languages: vec![id],
        limit: None,
        max_examples_per_file: 8,
    })
    .run(
        &records,
        PoolReport {
            path: "inline".to_string(),
            sha256: String::new(),
            records: 0,
            by_language: std::collections::BTreeMap::new(),
        },
    )
}

fn report_for(run: &Run, id: LangId) -> LanguageReport {
    run.manifest
        .languages
        .iter()
        .find(|r| r.language == id)
        .cloned()
        .expect("a report per language")
}

#[test]
fn every_operator_either_finds_sites_or_is_an_explicitly_listed_exception() {
    let mut failures: Vec<String> = Vec::new();
    for id in LangId::ALL {
        let run = run_language(id);
        let report = report_for(&run, id);
        let exceptions: BTreeSet<OpId> = expected_silent(id).into_iter().map(|(op, _)| op).collect();

        for op in OpId::ALL {
            let sites = report
                .operators
                .get(op.as_str())
                .map(|r| r.sites_found)
                .unwrap_or(0);
            let listed = exceptions.contains(&op);
            // `cosmetic.reformat` is the operator that *runs* the formatter, so with no usable
            // one there is nothing for it to find. That is a fact about this machine, not about
            // the language, and `expected_silent` says of itself that every entry there is a
            // language-level fact — so it is derived from the measured availability instead of
            // listed. It used to be listed, for TypeScript, with the reason "prettier does not
            // resolve on this machine"; that entry broke the table's own contract and went
            // stale in both directions, since installing prettier would have tripped the
            // `sites > 0 && listed` arm below. Python reached the same state today when black
            // was found to be announcing that its equivalence check does not run here.
            let formatter_silent = op == OpId::CosmeticReformat && !report.formatter.is_usable();
            if sites == 0 && !listed && !formatter_silent {
                failures.push(format!(
                    "{id}: {op} found no sites and is not a listed exception — the facade has \
                     stopped seeing its construct, or the fixture no longer contains one"
                ));
            }
            if sites > 0 && listed {
                let why = expected_silent(id)
                    .into_iter()
                    .find(|(candidate, _)| *candidate == op)
                    .map(|(_, why)| why)
                    .unwrap_or("");
                failures.push(format!(
                    "{id}: {op} found {sites} sites but is listed as silent ({why}); the list is \
                     now wrong and would hide a real regression"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn every_language_reports_its_formatter_as_a_measured_fact() {
    for id in LangId::ALL {
        let run = run_language(id);
        let report = report_for(&run, id);
        let declared = qd_mutate::lang::for_id(id).formatter().is_some();
        assert_eq!(
            report.formatter.key() == "not_declared",
            !declared,
            "{id}: the manifest's formatter state contradicts what the facade declares"
        );
        if !report.formatter.is_usable() {
            for op in OpId::ALL.into_iter().filter(|o| o.requires_formatter()) {
                assert!(
                    report.restricted_operators.contains_key(op.as_str()),
                    "{id}: {op} needs a formatter that is {} here, and the restriction is not \
                     recorded — an unverified cosmetic label would read as a verified one",
                    report.formatter.key()
                );
            }
        }
    }
}

#[test]
fn the_silent_stub_operators_are_available_in_every_language() {
    // These are the class the exercise is for. If a language cannot produce them, its slice of the
    // mixture teaches nothing a substring gate could not already catch.
    for id in LangId::ALL {
        let run = run_language(id);
        let report = report_for(&run, id);
        let silent_sites: u64 = OpId::ALL
            .into_iter()
            .filter(|o| o.is_silent())
            .filter_map(|o| report.operators.get(o.as_str()))
            .map(|r| r.sites_found)
            .sum();
        assert!(
            silent_sites > 0,
            "{id}: no silent stub operator found a site; this language would contribute only the \
             visible stub, which a substring gate already catches"
        );
    }
}

#[test]
fn a_coverage_table_names_every_operator_for_every_language() {
    // The table is what a human reads instead of the JSON. It must list all seventeen for all five,
    // including the zeros, because a missing row is what a silent zero looks like.
    let run = run_language(LangId::Rust);
    let table = run.manifest.coverage_table();
    for op in OpId::ALL {
        assert!(table.contains(op.as_str()), "the table omits {op}");
    }
    for id in LangId::ALL {
        assert!(
            table.contains(&format!("[{id}]")),
            "the table omits the {id} section"
        );
    }
    assert!(table.contains("span-disagreements"), "the alarm is not in the table");
}

#[test]
fn the_span_derivations_agree_across_every_language() {
    for id in LangId::ALL {
        let run = run_language(id);
        assert!(
            !run.examples.is_empty(),
            "{id}: the maximal fixture produced no examples at all"
        );
        assert_eq!(
            run.manifest.totals.span_disagreements, 0,
            "{id}: examples were dropped on span disagreement"
        );
        assert_eq!(
            run.manifest.totals.mutation_did_not_parse, 0,
            "{id}: an operator produced output that does not parse"
        );
    }
}
