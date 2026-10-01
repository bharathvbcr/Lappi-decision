//! Bounded parsing, and the refusals that come out of it.
//!
//! Every bound here exists because an unbounded version of it has a real failure mode:
//!
//! * `MAX_SOURCE_BYTES` — a 10 MB file is in the adversarial corpus; parsing it is not useful and
//!   the byte arithmetic downstream is `usize` but the line count is `u32`.
//! * `PARSE_BUDGET` — a pathological parse is stopped by tree-sitter's progress callback, which is
//!   the only bound that works from outside a grammar this crate does not control.
//! * `MAX_TREE_DEPTH` — 1000-level nesting is in the corpus. Every walk in this crate uses an
//!   explicit stack, so depth cannot overflow *our* stack; the bound is here so that a tree deep
//!   enough to worry a consumer is refused loudly instead of mutated.
//! * `MAX_NODE_VISITS` — the walk itself is bounded, so a tree with a pathological branching factor
//!   cannot turn one file into an unbounded amount of work.
//!
//! A NUL byte is refused at the boundary. That is not fastidiousness: `devmap-extract` measured a
//! vendored grammar spinning for over three minutes on six bytes containing NULs, with the progress
//! callback never reached because the scanner never yielded. A file containing a NUL is not source.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser, Tree};

/// 10 MiB. The adversarial corpus includes a 10 MB file; this refuses it rather than discovering
/// the limit as a timeout.
pub const MAX_SOURCE_BYTES: usize = 10 * 1024 * 1024;
/// Wall clock for one parse.
pub const PARSE_BUDGET: Duration = Duration::from_secs(5);
/// Deeper than any hand-written source and shallower than anything that worries a consumer.
pub const MAX_TREE_DEPTH: usize = 400;
/// Nodes one validation walk may visit.
pub const MAX_NODE_VISITS: usize = 4_000_000;

/// Why a file, or a site in it, was not mutated.
///
/// The `key` is what lands in the manifest's refusal histogram. Refusal counts are part of the
/// manifest because a language with a high refusal rate has a thinner mixture than its headline
/// count suggests, and the mixture weights have to know that.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Refusal {
    /// The file contains a NUL byte and is therefore not source text.
    NulByte,
    /// Over `MAX_SOURCE_BYTES`.
    TooLarge { bytes: usize },
    /// Empty, or nothing but whitespace: there is no function to mutate.
    EmptySource,
    /// tree-sitter returned no tree at all.
    NoTree,
    /// The parse hit `PARSE_BUDGET`. A cancelled parse can still hand back a partial tree; taking
    /// it would publish a prefix of the file as the whole of it.
    ParseBudget { millis: u64 },
    /// The tree contains `ERROR` or `MISSING` nodes. Mutating an unparsed file produces nonsense
    /// carrying a confident label, which is worse than producing nothing.
    ParseErrors { count: usize, first_line: u32 },
    /// The **input** parsed cleanly and the **output** did not: this operator's edit broke the
    /// syntax. Kept distinct from `ParseErrors` on purpose — "the pool gave us junk" and "we made
    /// junk" are different findings, and a single counter covering both would hide a regression in
    /// an operator behind a noisy pool.
    MutationDidNotParse {
        operator: String,
        count: usize,
        first_line: u32,
    },
    /// Nesting past `MAX_TREE_DEPTH`.
    TooDeep { depth: usize },
    /// The validation walk hit `MAX_NODE_VISITS`.
    TooManyNodes { visited: usize },
    /// No function in the file has a body to mutate: a header of declarations, a trait with only
    /// signatures, an interface. Emitting an empty span here would be a poisoned label.
    NoFunctionBody,
    /// The operator's preconditions were not met at any site.
    PreconditionUnmet { operator: String, detail: String },
    /// Every candidate site fell outside the hunks the agent touched. A site outside every hunk is
    /// skipped, never relocated.
    OutsideHunk { operator: String },
    /// The two span derivations disagreed. The example is dropped; this is the count that must stay
    /// visible, because a rising one means the mutator and the differ have drifted apart.
    SpanDisagreement {
        operator: String,
        from_bytes: String,
        from_diff: String,
    },
    /// The edit applied cleanly and produced text identical to the input.
    ///
    /// Kept out of `SpanDisagreement` deliberately. A no-op is an operator picking a candidate that
    /// rewrites something to itself — `stub.hardcoded` on `fn f() -> u32 { 3 }`, whose only literal
    /// is the body. That is a routine, harmless decline. `SpanDisagreement` means the mutator's
    /// bookkeeping and the textual differ have drifted apart, which is the single most important
    /// alarm in the manifest. Counting the first as the second buries the alarm in noise.
    NoTextualChange { operator: String },
    /// A cosmetic mutation failed its behaviour-preservation check.
    CosmeticNotPreserving { operator: String, detail: String },
    /// The language declared no formatter, so the formatter-verified cosmetic operators are not
    /// available here. Recorded rather than assumed harmless.
    NoFormatter { language: String, operator: String },
    /// An edit set that could not be applied. A bug, surfaced rather than swallowed.
    EditFailed { detail: String },
    /// The same function was about to be emitted both mutated and clean. Split safety says it never
    /// is; the disposition is keyed on `(repo, path)` so this cannot happen, and the counter exists
    /// to prove that rather than to assume it.
    SplitConflict { identity: String },
    /// A `clean` example whose pre-image and post-image are identical after normalization, so
    /// the agent's own diff is empty and there is no change to judge.
    ///
    /// Distinct from `NoTextualChange`, which is an *operator* declining a site. This is the
    /// pool handing over a record whose commit changed nothing in this file once normalized --
    /// a BOM added and then stripped, a line-ending flip. Emitting it would put an EMPTY
    /// context on a `clean` label, and on a corpus where only clean rows are empty that lets a
    /// model answer from the length of its input (`AUDIT/after-vs-diff-leak.md`). Dropped and
    /// counted, because "the pool had no pre-image at all" and "the pre-image was identical"
    /// are different facts and only the second is a record that promised a change.
    CleanDiffEmpty,
    /// The multi-hunk renderer declined the record's two texts: over its line cap or its
    /// edit-distance cap (`diffspan::MAX_DIFF_LINES`, `diffspan::MAX_EDIT_DISTANCE`). Refused, not
    /// truncated -- a diff of part of a file is a diff of a different file.
    DiffRefused { detail: String },
    /// Taken from the pre-image, the diff shows no change at the injected edit's span: no added
    /// line inside it and no deletion at it.
    ///
    /// The placement rule puts every mutation inside a hunk the commit touched, so an operator can
    /// undo the commit's own edit there -- flip `<=` back to `<`, drop the line the commit added.
    /// The post-image-to-mutated diff shows that change; the pre-image-to-mutated diff does not,
    /// because the mutated line is the pre-image's line again. Emitting it would put a defect
    /// label on text the reader is told did not change.
    NeedleNotInDiff { operator: String, detail: String },
    /// The injected edit's span is not inside one hunk of the diff: it reaches past every hunk,
    /// or two hunks each hold part of it. "Which hunk is the defect in" then has no single answer,
    /// and a span end the diff does not carry is one `qd_train.mutate_adapter` cannot rebase.
    NeedleSplitAcrossHunks { operator: String, detail: String },
}

impl Refusal {
    /// The histogram key. Stable, and deliberately coarser than the payload.
    pub fn key(&self) -> &'static str {
        match self {
            Refusal::NulByte => "nul_byte",
            Refusal::TooLarge { .. } => "too_large",
            Refusal::EmptySource => "empty_source",
            Refusal::NoTree => "no_tree",
            Refusal::ParseBudget { .. } => "parse_budget",
            Refusal::ParseErrors { .. } => "parse_errors",
            Refusal::MutationDidNotParse { .. } => "mutation_did_not_parse",
            Refusal::TooDeep { .. } => "too_deep",
            Refusal::TooManyNodes { .. } => "too_many_nodes",
            Refusal::NoFunctionBody => "no_function_body",
            Refusal::PreconditionUnmet { .. } => "precondition_unmet",
            Refusal::OutsideHunk { .. } => "outside_hunk",
            Refusal::SpanDisagreement { .. } => "span_disagreement",
            Refusal::NoTextualChange { .. } => "no_textual_change",
            Refusal::CosmeticNotPreserving { .. } => "cosmetic_not_preserving",
            Refusal::NoFormatter { .. } => "no_formatter",
            Refusal::EditFailed { .. } => "edit_failed",
            Refusal::SplitConflict { .. } => "split_conflict",
            Refusal::CleanDiffEmpty => "clean_diff_empty",
            Refusal::DiffRefused { .. } => "diff_refused",
            Refusal::NeedleNotInDiff { .. } => "needle_not_in_diff",
            Refusal::NeedleSplitAcrossHunks { .. } => "needle_split_across_hunks",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key())
    }
}

/// A parsed file, with the tree kept alongside the text it describes.
#[derive(Debug)]
pub struct Parsed {
    pub tree: Tree,
    /// Maximum node depth, measured during validation. Reported so a caller can see how close the
    /// input came to the bound rather than only whether it crossed it.
    pub depth: usize,
}

/// Parse `src` with `grammar`, refusing anything the mutator must not touch.
pub fn parse_bounded(grammar: &tree_sitter::Language, src: &str) -> Result<Parsed, Refusal> {
    if src.as_bytes().contains(&0) {
        return Err(Refusal::NulByte);
    }
    if src.len() > MAX_SOURCE_BYTES {
        return Err(Refusal::TooLarge { bytes: src.len() });
    }
    if src.trim().is_empty() {
        return Err(Refusal::EmptySource);
    }

    let mut parser = Parser::new();
    if parser.set_language(grammar).is_err() {
        return Err(Refusal::NoTree);
    }

    let started = Instant::now();
    let deadline = started + PARSE_BUDGET;
    let mut over_budget = false;
    let tree = {
        // Returning `true` from the progress callback cancels the parse. This is the only way to
        // bound a parse that has already entered a pathological state, and it is polled from
        // inside tree-sitter's own loop.
        let mut cancel = |_: &tree_sitter::ParseState| -> bool {
            if Instant::now() >= deadline {
                over_budget = true;
                return true;
            }
            false
        };
        let options = tree_sitter::ParseOptions::new().progress_callback(&mut cancel);
        let bytes = src.as_bytes();
        parser.parse_with_options(
            &mut |offset: usize, _| {
                if offset < bytes.len() {
                    &bytes[offset..]
                } else {
                    &[][..]
                }
            },
            None,
            Some(options),
        )
    };

    let tree = match tree {
        Some(t) if !over_budget => t,
        // A cancelled parse still hands back a tree describing a prefix of the file. Accepting it
        // would mutate a truncated view and label the result exactly.
        Some(_) | None if over_budget => {
            let elapsed = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
            return Err(Refusal::ParseBudget { millis: elapsed });
        }
        _ => return Err(Refusal::NoTree),
    };

    let depth = validate(&tree, src)?;
    Ok(Parsed { tree, depth })
}

/// One iterative pass: find `ERROR`/`MISSING` nodes and measure depth.
///
/// Iterative on purpose. A recursive walk over the 1000-level-nesting case in the adversarial
/// corpus is a stack overflow, which aborts the process — a generator that crashes on one pool
/// entry loses the whole run, and an abort cannot be recorded as a refusal.
fn validate(tree: &Tree, src: &str) -> Result<usize, Refusal> {
    let mut stack: Vec<(Node<'_>, usize)> = vec![(tree.root_node(), 1)];
    let mut visited = 0usize;
    let mut error_count = 0usize;
    let mut first_error_line: Option<u32> = None;

    while let Some((node, depth)) = stack.pop() {
        visited += 1;
        if visited > MAX_NODE_VISITS {
            return Err(Refusal::TooManyNodes { visited });
        }
        if depth > MAX_TREE_DEPTH {
            return Err(Refusal::TooDeep { depth });
        }
        if node.is_error() || node.is_missing() {
            error_count += 1;
            if first_error_line.is_none() {
                first_error_line = Some(crate::span::line_of(src, node.start_byte()));
            }
        }
        // `has_error` is false for a whole clean subtree, so skipping it is not a shortcut that
        // can miss an error — it is the grammar's own answer.
        if !node.has_error() {
            continue;
        }
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                stack.push((cursor.node(), depth + 1));
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }

    if error_count > 0 {
        return Err(Refusal::ParseErrors {
            count: error_count,
            first_line: first_error_line.unwrap_or(1),
        });
    }

    // `has_error` short-circuits the depth walk on a clean tree, so depth was not actually
    // measured above. Measure it on its own pass — bounded identically.
    measure_depth(tree)
}

fn measure_depth(tree: &Tree) -> Result<usize, Refusal> {
    let mut stack: Vec<(Node<'_>, usize)> = vec![(tree.root_node(), 1)];
    let mut max_depth = 1usize;
    let mut visited = 0usize;
    while let Some((node, depth)) = stack.pop() {
        visited += 1;
        if visited > MAX_NODE_VISITS {
            return Err(Refusal::TooManyNodes { visited });
        }
        if depth > max_depth {
            max_depth = depth;
        }
        if depth > MAX_TREE_DEPTH {
            return Err(Refusal::TooDeep { depth });
        }
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                stack.push((cursor.node(), depth + 1));
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }
    Ok(max_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rust() -> tree_sitter::Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    #[test]
    fn a_clean_file_parses() {
        let p = parse_bounded(&rust(), "fn f() -> u32 { 1 }\n").expect("parses");
        assert!(p.depth > 1);
    }

    #[test]
    fn error_nodes_are_refused_with_a_line() {
        let err = parse_bounded(&rust(), "fn f() -> u32 { 1 }\nfn @@@ {\n").expect_err("refused");
        match err {
            Refusal::ParseErrors { count, first_line } => {
                assert!(count >= 1);
                assert!(first_line >= 1);
            }
            other => panic!("expected ParseErrors, got {other:?}"),
        }
    }

    #[test]
    fn a_nul_byte_is_refused_before_any_parser_sees_it() {
        assert_eq!(
            parse_bounded(&rust(), "fn f() {}\0\n").expect_err("refused"),
            Refusal::NulByte
        );
    }

    #[test]
    fn an_empty_file_is_refused() {
        assert_eq!(
            parse_bounded(&rust(), "").expect_err("refused"),
            Refusal::EmptySource
        );
        assert_eq!(
            parse_bounded(&rust(), "   \n\t\n").expect_err("refused"),
            Refusal::EmptySource
        );
    }

    #[test]
    fn an_oversized_file_is_refused_by_size_not_by_timeout() {
        let big = "x".repeat(MAX_SOURCE_BYTES + 1);
        assert!(matches!(
            parse_bounded(&rust(), &big).expect_err("refused"),
            Refusal::TooLarge { .. }
        ));
    }

    #[test]
    fn a_thousand_levels_of_nesting_is_bounded_not_a_stack_overflow() {
        let src = format!("fn f() {}{}\n", "{ ".repeat(1000), "}".repeat(1000));
        // Either bound is an acceptable answer; a crash is not, and neither is success.
        let err = parse_bounded(&rust(), &src).expect_err("refused");
        assert!(
            matches!(
                err,
                Refusal::TooDeep { .. } | Refusal::ParseErrors { .. } | Refusal::ParseBudget { .. }
            ),
            "unexpected refusal {err:?}"
        );
    }
}
