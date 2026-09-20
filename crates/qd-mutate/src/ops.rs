//! The seventeen operators, written **once** against the [`Language`] facade.
//!
//! Every function here is **pure**: it enumerates candidates and returns them in a deterministic
//! order. Nothing in this module draws from the RNG, and nothing here writes. [`crate::generate`]
//! holds the seeded stream and does the choosing, which is what makes "the same seed and the same
//! input pool produce byte-identical output" a property of one small, testable place rather than of
//! seventeen scattered ones.
//!
//! Two rules bind every operator:
//!
//! * **One mutation per diff.** A candidate is one [`EditSet`], and an `EditSet` is one mutation —
//!   `cosmetic.rename_local` is the only one that touches more than one byte range, and every range
//!   it touches is the same identifier.
//! * **A candidate that cannot be proven is not emitted.** Where a precondition cannot be
//!   established — an unresolvable return type, a rename that might be captured, a formatter that is
//!   not installed — the operator produces nothing and the caller records a refusal. It never
//!   guesses, because a guessed mutation carries a confident label that is wrong.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use tree_sitter::Tree;

use crate::edit::{self, Edit, EditSet, MAX_EDITS};
use crate::fmt::{self, Resolved};
use crate::lang::util::{self, text};
use crate::lang::{FunctionBody, Language, LiteralKind};
use crate::parse::Refusal;

/// Candidates one operator may offer for one body. A bound, because every fan-out here has one.
pub const MAX_CANDIDATES_PER_OP: usize = 64;
/// A line longer than this is a `cosmetic.wrap_line` candidate.
pub const WRAP_THRESHOLD: usize = 100;
/// Line count either side of a `cosmetic.reformat` diff may have before the alignment is declined.
/// The alignment is a quadratic DP; this keeps its worst case at a few megabytes and milliseconds.
pub const MAX_REFORMAT_LINES: usize = 1200;

/// The class the example is labelled with. `Clean` is not produced by an operator: it is the
/// original agent diff, unmodified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MutationClass {
    Stub,
    Logic,
    Cosmetic,
    Clean,
}

impl MutationClass {
    pub fn as_str(self) -> &'static str {
        match self {
            MutationClass::Stub => "stub",
            MutationClass::Logic => "logic",
            MutationClass::Cosmetic => "cosmetic",
            MutationClass::Clean => "clean",
        }
    }
}

impl std::fmt::Display for MutationClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The operator ids, spelled exactly as `docs/mutation-operators.md` spells them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpId {
    StubPanic,
    StubDefaultReturn,
    StubHardcoded,
    StubEarlyReturn,
    LogicNegateCondition,
    LogicOffByOne,
    LogicSwapArgs,
    LogicDropElse,
    LogicWidenComparison,
    LogicSwallowError,
    LogicDropAwaitOrLock,
    LogicChangeConstant,
    CosmeticRenameLocal,
    CosmeticReformat,
    CosmeticReorderImports,
    CosmeticEditComment,
    CosmeticWrapLine,
}

impl OpId {
    pub const ALL: [OpId; 17] = [
        OpId::StubPanic,
        OpId::StubDefaultReturn,
        OpId::StubHardcoded,
        OpId::StubEarlyReturn,
        OpId::LogicNegateCondition,
        OpId::LogicOffByOne,
        OpId::LogicSwapArgs,
        OpId::LogicDropElse,
        OpId::LogicWidenComparison,
        OpId::LogicSwallowError,
        OpId::LogicDropAwaitOrLock,
        OpId::LogicChangeConstant,
        OpId::CosmeticRenameLocal,
        OpId::CosmeticReformat,
        OpId::CosmeticReorderImports,
        OpId::CosmeticEditComment,
        OpId::CosmeticWrapLine,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            OpId::StubPanic => "stub.panic",
            OpId::StubDefaultReturn => "stub.default_return",
            OpId::StubHardcoded => "stub.hardcoded",
            OpId::StubEarlyReturn => "stub.early_return",
            OpId::LogicNegateCondition => "logic.negate_condition",
            OpId::LogicOffByOne => "logic.off_by_one",
            OpId::LogicSwapArgs => "logic.swap_args",
            OpId::LogicDropElse => "logic.drop_else",
            OpId::LogicWidenComparison => "logic.widen_comparison",
            OpId::LogicSwallowError => "logic.swallow_error",
            OpId::LogicDropAwaitOrLock => "logic.drop_await_or_lock",
            OpId::LogicChangeConstant => "logic.change_constant",
            OpId::CosmeticRenameLocal => "cosmetic.rename_local",
            OpId::CosmeticReformat => "cosmetic.reformat",
            OpId::CosmeticReorderImports => "cosmetic.reorder_imports",
            OpId::CosmeticEditComment => "cosmetic.edit_comment",
            OpId::CosmeticWrapLine => "cosmetic.wrap_line",
        }
    }

    pub fn parse(s: &str) -> Option<OpId> {
        OpId::ALL.into_iter().find(|o| o.as_str() == s)
    }

    pub fn class(self) -> MutationClass {
        match self {
            OpId::StubPanic
            | OpId::StubDefaultReturn
            | OpId::StubHardcoded
            | OpId::StubEarlyReturn => MutationClass::Stub,
            OpId::CosmeticRenameLocal
            | OpId::CosmeticReformat
            | OpId::CosmeticReorderImports
            | OpId::CosmeticEditComment
            | OpId::CosmeticWrapLine => MutationClass::Cosmetic,
            _ => MutationClass::Logic,
        }
    }

    /// True for a stub a substring gate cannot see.
    ///
    /// `stub.panic` writes `todo!()` / `panic(` / `NotImplementedError`, and dcverify's substring
    /// gate already catches all three. The other three write `Ok(())`, `return nil`, `return []` —
    /// indistinguishable from working code by substring. They are the class that earns the model,
    /// and [`OpId::weight`] says so in numbers.
    pub fn is_silent(self) -> bool {
        matches!(
            self,
            OpId::StubDefaultReturn | OpId::StubHardcoded | OpId::StubEarlyReturn
        )
    }

    /// True where the operator cannot run at all without a resolvable formatter.
    ///
    /// This is the spec's shrink, stated as data: *"where no formatter is available the operator
    /// set shrinks to `edit_comment` and, where provably safe, `rename_local`."* Everything this
    /// answers `true` for is refused with [`Refusal::NoFormatter`] and counted per language, so a
    /// machine without `prettier` produces a visibly thinner TypeScript mixture rather than a
    /// silently unverified one.
    pub fn requires_formatter(self) -> bool {
        matches!(
            self,
            OpId::CosmeticReformat | OpId::CosmeticWrapLine | OpId::CosmeticReorderImports
        )
    }

    /// True where behaviour preservation is proven by **running the formatter on both sides** and
    /// comparing canonical forms.
    ///
    /// Only the layout-only operators qualify. `rename_local`, `edit_comment` and
    /// `reorder_imports` all change the canonical form by design — a rename that left the canonical
    /// form identical would not have renamed anything — so their proof is the structural one their
    /// candidate generator performs, not this one. Asserting canonical equality for them would be a
    /// check that always fails; asserting it for nothing at all would be a check that never runs.
    pub fn verified_by_canonical_form(self) -> bool {
        matches!(self, OpId::CosmeticReformat | OpId::CosmeticWrapLine)
    }

    /// Selection weight. Relative, not a probability; [`crate::generate`] normalises over whichever
    /// operators actually produced a candidate for a given body.
    ///
    /// The silent stubs carry five times the weight of the visible one. That ratio is the mixture
    /// decision the spec asks for — "these are weighted up: they are the class that earns the
    /// model" — made explicit here so it is one number to argue with rather than an emergent
    /// property of how many candidate sites each operator happens to find.
    pub fn weight(self) -> u32 {
        match self {
            OpId::StubDefaultReturn | OpId::StubEarlyReturn => 25,
            OpId::StubHardcoded => 20,
            OpId::StubPanic => 5,
            // The logic operators are the classic mutation-testing set and share the middle band.
            OpId::LogicSwallowError | OpId::LogicNegateCondition => 10,
            OpId::LogicOffByOne
            | OpId::LogicWidenComparison
            | OpId::LogicDropElse
            | OpId::LogicSwapArgs
            | OpId::LogicDropAwaitOrLock
            | OpId::LogicChangeConstant => 8,
            // Cosmetic is the contrast class; enough of it to learn the boundary, no more.
            OpId::CosmeticEditComment => 6,
            OpId::CosmeticRenameLocal => 6,
            OpId::CosmeticReformat => 5,
            OpId::CosmeticWrapLine | OpId::CosmeticReorderImports => 3,
        }
    }
}

impl std::fmt::Display for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One concrete mutation, ready to apply.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub op: OpId,
    pub edits: EditSet,
    /// What the operator did, for the example record and for a human reading a dropped example.
    pub detail: String,
}

/// Everything an operator needs that is not the body itself.
pub struct OpCtx<'t> {
    pub language: &'t dyn Language,
    pub tree: &'t Tree,
    pub src: &'t str,
    /// `None` when the language's formatter could not be resolved here. The formatter-verified
    /// cosmetic operators then produce nothing and the caller records [`Refusal::NoFormatter`].
    pub formatter: Option<&'t Resolved>,
    /// How many times each name is **bound** anywhere in the file (locals and parameters alike).
    /// The rename-safety proof needs "exactly once", and a count taken only inside the body would
    /// miss the shadow that makes the rename wrong.
    binding_counts: HashMap<String, usize>,
    /// Every identifier occurrence in the file: text -> (byte range, kind, parent kind, is the
    /// first named child of its parent).
    occurrences: HashMap<String, Vec<Occurrence>>,
    /// Byte ranges of every node that opens a new function scope — a nested fn, a closure, a
    /// lambda. `cosmetic.rename_local` refuses any name with an occurrence inside one.
    boundary_ranges: Vec<(usize, usize)>,
    /// The formatter's canonical form of `src`, computed at most once per file.
    ///
    /// Without the cache, `cosmetic.reformat` spawns one formatter process per function body in the
    /// file. That is not merely slow: a bounded subprocess is the most expensive thing this crate
    /// does, and a per-body cost turns a large file into a run that looks hung.
    canonical: std::cell::OnceCell<Result<String, String>>,
}

#[derive(Debug, Clone, Copy)]
struct Occurrence {
    start: usize,
    end: usize,
    kind: &'static str,
    parent_kind: &'static str,
    is_first_named_child: bool,
}

/// Node kinds whose *second* named child is a member name rather than a value.
///
/// Renaming a local named `lock` must not rewrite `mutex.lock`. Rust, Go and TypeScript give the
/// member its own node kind, which [`MEMBER_KINDS`] catches; Python's `attribute` and Swift's
/// `navigation_suffix` reuse the plain identifier kind, which this catches instead.
const MEMBER_ACCESS_PARENTS: &[&str] = &[
    "attribute",
    "navigation_suffix",
    "navigation_expression",
    "selector_expression",
    "field_expression",
    "member_expression",
];

/// Identifier kinds that are never a local binding.
const MEMBER_KINDS: &[&str] = &[
    "field_identifier",
    "property_identifier",
    "shorthand_property_identifier",
    "type_identifier",
    "package_identifier",
];

/// Replacement names for `cosmetic.rename_local`, tried in order.
const RENAME_POOL: &[&str] = &[
    "tmp", "acc", "cur", "item", "val", "res", "buf", "idx", "out", "aux",
];

/// Replacement comment bodies for `cosmetic.edit_comment`.
///
/// Deliberately free of `"`, `\`, `*/` and newlines, so the same phrase can be dropped into a `//`
/// comment, a `/* */` block and a Python docstring without any escaping step that could go wrong.
const COMMENT_POOL: &[&str] = &[
    "Kept for clarity; the caller relies on the invariant below.",
    "Rewritten: the original wording predated the current signature.",
    "This branch is the common case; the other one is the fallback.",
    "Explains the step below, which the names alone do not.",
    "Mirrors the behaviour of the sibling function in this module.",
];

impl<'t> OpCtx<'t> {
    pub fn new(
        language: &'t dyn Language,
        tree: &'t Tree,
        src: &'t str,
        formatter: Option<&'t Resolved>,
    ) -> Self {
        let mut binding_counts: HashMap<String, usize> = HashMap::new();
        for body in language.function_bodies(tree, src) {
            for binding in language.local_bindings(&body) {
                *binding_counts.entry(binding.name).or_insert(0) += 1;
            }
            for (name, _) in &body.params {
                *binding_counts.entry(name.clone()).or_insert(0) += 1;
            }
        }

        let mut occurrences: HashMap<String, Vec<Occurrence>> = HashMap::new();
        for node in language.identifiers(tree) {
            let name = text(node, src);
            if name.is_empty() {
                continue;
            }
            let parent = node.parent();
            let parent_kind = parent.map(|p| p.kind()).unwrap_or("");
            let is_first_named_child = parent
                .and_then(|p| p.named_child(0))
                .map(|c| c.id() == node.id())
                .unwrap_or(false);
            occurrences.entry(name.to_string()).or_default().push(Occurrence {
                start: node.start_byte(),
                end: node.end_byte(),
                kind: node.kind(),
                parent_kind,
                is_first_named_child,
            });
        }

        let mut boundary_ranges = Vec::new();
        util::walk_tree(tree, |node| {
            if language.is_function_boundary(node.kind()) {
                boundary_ranges.push((node.start_byte(), node.end_byte()));
            }
        });

        OpCtx {
            language,
            tree,
            src,
            formatter,
            binding_counts,
            occurrences,
            boundary_ranges,
            canonical: std::cell::OnceCell::new(),
        }
    }

    /// The canonical form of this file, or the reason there is not one.
    pub fn canonical_src(&self) -> Result<&str, &str> {
        self.canonical
            .get_or_init(|| match self.formatter {
                Some(resolved) => fmt::canonical(resolved, self.src).map_err(|e| e.to_string()),
                None => Err("no formatter resolved for this language on this machine".to_string()),
            })
            .as_deref()
            .map_err(String::as_str)
    }
}

/// Enumerate every candidate `op` offers for `body`, in a deterministic order.
///
/// `Err` is a **recorded** refusal — a reason this operator produced nothing here. `Ok(vec![])`
/// means the construct simply is not present, which is the "a language whose facade returns an
/// empty list yields no mutations of that kind" case and is counted separately in the manifest so a
/// silent zero never reads as "the operator ran".
pub fn candidates(op: OpId, ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Result<Vec<Candidate>, Refusal> {
    let out = match op {
        OpId::StubPanic => stub_panic(ctx, body)?,
        OpId::StubDefaultReturn => stub_default_return(ctx, body)?,
        OpId::StubHardcoded => stub_hardcoded(ctx, body)?,
        OpId::StubEarlyReturn => stub_early_return(ctx, body)?,
        OpId::LogicNegateCondition => negate_condition(ctx, body),
        OpId::LogicOffByOne => off_by_one(ctx, body),
        OpId::LogicSwapArgs => swap_args(ctx, body),
        OpId::LogicDropElse => drop_else(ctx, body),
        OpId::LogicWidenComparison => widen_comparison(ctx, body),
        OpId::LogicSwallowError => swallow_error(ctx, body),
        OpId::LogicDropAwaitOrLock => drop_await_or_lock(ctx, body),
        OpId::LogicChangeConstant => change_constant(ctx, body),
        OpId::CosmeticRenameLocal => rename_local(ctx, body),
        OpId::CosmeticReformat => reformat(ctx, body)?,
        OpId::CosmeticReorderImports => reorder_imports(ctx, body)?,
        OpId::CosmeticEditComment => edit_comment(ctx, body),
        OpId::CosmeticWrapLine => wrap_line(ctx, body)?,
    };
    Ok(out.into_iter().take(MAX_CANDIDATES_PER_OP).collect())
}

// ---------------------------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------------------------

/// The body interior expressed as **whole lines**, or `None` for a single-line body.
///
/// This is the load-bearing helper for the span guarantee on every `stub` operator. A block body's
/// interior begins immediately after `{`, so an edit starting there reports the *signature* line
/// under derivation A while the text diff — which sees the signature line unchanged — reports the
/// line below. The two would disagree on every stub, and the disagreement would be this crate's
/// fault rather than the mutator's. Starting the edit at the first line the body actually owns
/// makes them agree for the right reason: the edit really does replace whole lines.
pub fn body_line_region(src: &str, interior: (usize, usize)) -> Option<(usize, usize)> {
    let (s, e) = interior;
    if e <= s || e > src.len() {
        return None;
    }
    let bytes = src.as_bytes();
    let at_line_start = s == 0 || bytes.get(s - 1) == Some(&b'\n');
    let start = if at_line_start {
        s
    } else {
        // The first line the body owns begins after the first newline inside the interior.
        s + src.get(s..e).and_then(|t| t.find('\n'))? + 1
    };
    let ends_at_line_start = bytes.get(e - 1) == Some(&b'\n');
    let end = if ends_at_line_start {
        e
    } else {
        start + src.get(start..e).and_then(|t| t.rfind('\n'))? + 1
    };
    (start < end).then_some((start, end))
}

/// Expand `[start, end)` to whole lines **only when the edit already owns its first line**.
///
/// [`edit::snap_to_lines`] widens each end independently. Widening only the end — which is what it
/// does for `} else { ... }` sitting after code — would swallow the trailing newline and join two
/// lines that the mutation never touched. Both ends snap together here, or neither does.
fn snap_if_line_owned(src: &str, start: usize, end: usize) -> (usize, usize) {
    let line_start = src.get(..start).and_then(|t| t.rfind('\n')).map_or(0, |i| i + 1);
    let prefix_is_blank = src
        .get(line_start..start)
        .map(|t| t.chars().all(|c| c == ' ' || c == '\t'))
        .unwrap_or(false);
    if prefix_is_blank {
        edit::snap_to_lines(src, start, end)
    } else {
        (start, end)
    }
}

/// One replacement, wrapped as a candidate. `None` when the edit set is not well-formed, which is
/// a bug in the operator rather than a property of the input — and one that must not be shipped.
fn one(op: OpId, detail: impl Into<String>, edits: Vec<Edit>) -> Option<Candidate> {
    EditSet::new(edits).ok().map(|edits| Candidate {
        op,
        edits,
        detail: detail.into(),
    })
}

/// Refuse when the body already is a stub.
///
/// Works on the **statement nodes** of the body, never on a substring of the file — which is why a
/// body containing the string literal `"todo!()"` is not mistaken for a stub. That substring gate
/// is exactly what the model is being trained to beat, so the generator must not depend on it.
fn already_a_stub(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> bool {
    let statements = ctx.language.statement_texts(body);
    util::body_is_already_stub(&statements, ctx.language.panic_stub())
}

fn stub_precondition(ctx: &OpCtx<'_>, body: &FunctionBody<'_>, op: OpId) -> Result<(), Refusal> {
    if body.interior_text().trim().is_empty() {
        return Err(Refusal::PreconditionUnmet {
            operator: op.as_str().to_string(),
            detail: "the body is empty; there is nothing to stub".to_string(),
        });
    }
    if already_a_stub(ctx, body) {
        return Err(Refusal::PreconditionUnmet {
            operator: op.as_str().to_string(),
            detail: "the body already is a stub; mutating a stub into a stub labels a \
                     clean-looking diff as stub"
                .to_string(),
        });
    }
    Ok(())
}

/// Build the edit that replaces a whole body with `statement`.
///
/// Declines when the replacement is byte-for-byte what is already there. That is not a theoretical
/// case: `stub.hardcoded` derives its value from a literal already in the function, so on
/// `fn one_line() -> u32 { 3 }` — whose only literal *is* the body — it would rewrite `3` to `3`.
/// A no-op reaching the generator would be caught there, but only after being counted, and the
/// place to decline a candidate is where the candidate is built.
fn replace_body(ctx: &OpCtx<'_>, body: &FunctionBody<'_>, op: OpId, statement: &str) -> Option<Candidate> {
    match body_line_region(ctx.src, body.interior) {
        Some((start, end)) => {
            let indent = edit::indent_of_line(ctx.src, start);
            let replacement = format!("{indent}{statement}\n");
            if ctx.src.get(start..end)? == replacement {
                return None;
            }
            one(
                op,
                format!("body replaced with `{statement}`"),
                vec![Edit::replace(start, end, replacement)],
            )
        }
        // A single-line body: replace in place, with the spacing the original had.
        None => {
            let (s, e) = body.interior;
            let raw = ctx.src.get(s..e)?;
            let lead = if raw.starts_with(' ') { " " } else { "" };
            let trail = if raw.ends_with(' ') { " " } else { "" };
            let replacement = format!("{lead}{statement}{trail}");
            if raw == replacement {
                return None;
            }
            one(
                op,
                format!("single-line body replaced with `{statement}`"),
                vec![Edit::replace(s, e, replacement)],
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------
// stub
// ---------------------------------------------------------------------------------------------

fn stub_panic(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Result<Vec<Candidate>, Refusal> {
    stub_precondition(ctx, body, OpId::StubPanic)?;
    Ok(ctx
        .language
        .panic_stub()
        .iter()
        .filter_map(|stub| replace_body(ctx, body, OpId::StubPanic, stub))
        .collect())
}

fn stub_default_return(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Result<Vec<Candidate>, Refusal> {
    stub_precondition(ctx, body, OpId::StubDefaultReturn)?;
    let Some(statement) = ctx.language.default_return(body.return_type.as_deref()) else {
        return Err(Refusal::PreconditionUnmet {
            operator: OpId::StubDefaultReturn.as_str().to_string(),
            detail: format!(
                "no zero value for return type {:?}; declining rather than guessing",
                body.return_type
            ),
        });
    };
    Ok(replace_body(ctx, body, OpId::StubDefaultReturn, &statement)
        .into_iter()
        .collect())
}

fn stub_hardcoded(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Result<Vec<Candidate>, Refusal> {
    stub_precondition(ctx, body, OpId::StubHardcoded)?;
    let language = ctx.language;
    let mut literals: Vec<(String, LiteralKind)> = Vec::new();
    for node in language.integer_literals(body) {
        literals.push((text(node, ctx.src).to_string(), LiteralKind::Integer));
    }
    for node in language.string_literals(body) {
        literals.push((text(node, ctx.src).to_string(), LiteralKind::Str));
    }
    if literals.is_empty() {
        return Err(Refusal::PreconditionUnmet {
            operator: OpId::StubHardcoded.as_str().to_string(),
            detail: "no literal in the function to hardcode; the spec derives the value from one \
                     already present rather than inventing it"
                .to_string(),
        });
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out = Vec::new();
    for (literal, kind) in literals {
        if !seen.insert(literal.clone()) {
            continue;
        }
        let Some(statement) =
            language.hardcoded_return(body.return_type.as_deref(), &literal, kind)
        else {
            continue;
        };
        if let Some(c) = replace_body(ctx, body, OpId::StubHardcoded, &statement) {
            out.push(c);
        }
    }
    if out.is_empty() {
        return Err(Refusal::PreconditionUnmet {
            operator: OpId::StubHardcoded.as_str().to_string(),
            detail: format!(
                "no literal in the function matches return type {:?}",
                body.return_type
            ),
        });
    }
    Ok(out)
}

fn stub_early_return(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Result<Vec<Candidate>, Refusal> {
    stub_precondition(ctx, body, OpId::StubEarlyReturn)?;
    let Some(statement) = ctx.language.early_return(body.return_type.as_deref()) else {
        return Err(Refusal::PreconditionUnmet {
            operator: OpId::StubEarlyReturn.as_str().to_string(),
            detail: format!(
                "no zero value for return type {:?}; declining rather than guessing",
                body.return_type
            ),
        });
    };
    // The whole point of this operator is that the real body stays visible below the inserted
    // return. A single-line body has no "below", so there is nothing here that looks like working
    // code — the mutation would be indistinguishable from `stub.default_return`.
    let Some((start, _)) = body_line_region(ctx.src, body.interior) else {
        return Err(Refusal::PreconditionUnmet {
            operator: OpId::StubEarlyReturn.as_str().to_string(),
            detail: "a single-line body has nothing to leave in place below the early return"
                .to_string(),
        });
    };
    let indent = edit::indent_of_line(ctx.src, start);
    Ok(one(
        OpId::StubEarlyReturn,
        format!("`{statement}` inserted before the real work"),
        vec![Edit::insert(start, format!("{indent}{statement}\n"))],
    )
    .into_iter()
    .collect())
}

// ---------------------------------------------------------------------------------------------
// logic
// ---------------------------------------------------------------------------------------------

fn negate_condition(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    let language = ctx.language;
    language
        .conditions(body)
        .into_iter()
        // A condition already negated at the top level would become a double negative, which is
        // churn rather than a logic change.
        .filter(|node| !language.is_negated(node, ctx.src))
        .filter_map(|node| {
            let original = text(node, ctx.src);
            let negated = language.negate(original);
            one(
                OpId::LogicNegateCondition,
                format!("`{}` negated", truncate(original)),
                vec![Edit::replace(node.start_byte(), node.end_byte(), negated)],
            )
        })
        .collect()
}

fn off_by_one(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    let mut out = Vec::new();
    for node in ctx.language.bound_literals(body) {
        let raw = text(node, ctx.src);
        let Some(value) = decimal_value(raw) else {
            continue;
        };
        for delta in [1i64, -1] {
            let Some(next) = value.checked_add(delta) else {
                continue;
            };
            let replacement = rewrite_integer(raw, next);
            if replacement == raw {
                continue;
            }
            if let Some(c) = one(
                OpId::LogicOffByOne,
                format!("bound `{raw}` -> `{replacement}`"),
                vec![Edit::replace(node.start_byte(), node.end_byte(), replacement)],
            ) {
                out.push(c);
            }
        }
    }
    out
}

fn swap_args(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    // Types are not resolved by this crate, so the spec restricts the swap to arguments that are
    // both plain identifiers naming parameters of the enclosing function **with the same declared
    // type**. Anything looser produces a swap that does not compile, and a diff that does not
    // compile is not the failure this class is meant to teach.
    let declared: HashMap<&str, &str> = body
        .params
        .iter()
        .filter_map(|(n, t)| t.as_deref().map(|t| (n.as_str(), t)))
        .collect();
    if declared.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for site in ctx.language.call_arguments(body) {
        if site.args.len() < 2 {
            continue;
        }
        for i in 0..site.args.len() {
            for j in (i + 1)..site.args.len() {
                let (a, b) = (site.args[i], site.args[j]);
                let (ta, tb) = (text(a, ctx.src), text(b, ctx.src));
                if ta == tb {
                    continue;
                }
                let (Some(type_a), Some(type_b)) = (declared.get(ta), declared.get(tb)) else {
                    continue;
                };
                if type_a != type_b {
                    continue;
                }
                if let Some(c) = one(
                    OpId::LogicSwapArgs,
                    format!("`{ta}` and `{tb}` swapped (both `{type_a}`)"),
                    vec![
                        Edit::replace(a.start_byte(), a.end_byte(), tb),
                        Edit::replace(b.start_byte(), b.end_byte(), ta),
                    ],
                ) {
                    out.push(c);
                }
            }
        }
    }
    out
}

fn drop_else(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    ctx.language
        .else_branches(body)
        .into_iter()
        .filter_map(|site| {
            let (start, end) = snap_if_line_owned(ctx.src, site.range.0, site.range.1);
            one(
                OpId::LogicDropElse,
                format!("else branch removed: `{}`", truncate(ctx.src.get(start..end)?)),
                vec![Edit::delete(start, end)],
            )
        })
        .collect()
}

fn widen_comparison(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    ctx.language
        .comparison_ops(body)
        .into_iter()
        .filter_map(|site| {
            let widened = match site.op.as_str() {
                "<" => "<=",
                ">" => ">=",
                "<=" => "<",
                ">=" => ">",
                _ => return None,
            };
            one(
                OpId::LogicWidenComparison,
                format!("`{}` -> `{widened}`", site.op),
                vec![Edit::replace(
                    site.op_node.start_byte(),
                    site.op_node.end_byte(),
                    widened,
                )],
            )
        })
        .collect()
}

fn swallow_error(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    ctx.language
        .error_propagations(body)
        .into_iter()
        .filter_map(|site| {
            let (start, end) = if site.whole_statement {
                snap_if_line_owned(ctx.src, site.range.0, site.range.1)
            } else {
                site.range
            };
            let detail = if site.replacement.is_empty() {
                format!("error check removed: `{}`", truncate(ctx.src.get(start..end)?))
            } else {
                format!(
                    "`{}` -> `{}`",
                    truncate(ctx.src.get(start..end)?),
                    site.replacement
                )
            };
            one(
                OpId::LogicSwallowError,
                detail,
                vec![Edit::replace(start, end, site.replacement.clone())],
            )
        })
        .collect()
}

fn drop_await_or_lock(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    ctx.language
        .awaits_and_locks(body)
        .into_iter()
        .filter_map(|site| {
            let (start, end) = if site.whole_statement {
                snap_if_line_owned(ctx.src, site.range.0, site.range.1)
            } else {
                site.range
            };
            one(
                OpId::LogicDropAwaitOrLock,
                format!("removed `{}`", truncate(ctx.src.get(start..end)?)),
                vec![Edit::delete(start, end)],
            )
        })
        .collect()
}

fn change_constant(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    let language = ctx.language;
    // A literal in a bound position belongs to `logic.off_by_one`. Emitting it here too would put
    // the same edit under two labels, and a model cannot learn a boundary that does not exist.
    let bounds: BTreeSet<(usize, usize)> = language
        .bound_literals(body)
        .into_iter()
        .map(|n| (n.start_byte(), n.end_byte()))
        .collect();

    let mut out = Vec::new();
    for node in language.integer_literals(body) {
        if bounds.contains(&(node.start_byte(), node.end_byte())) {
            continue;
        }
        let raw = text(node, ctx.src);
        let Some(value) = decimal_value(raw) else {
            continue;
        };
        // The spec excludes a perturbation of 0 or 1 that would be a no-op; every delta here is
        // non-zero, and the equality guard below catches any that still round-trips.
        for delta in [2i64, 10, -2] {
            let Some(next) = value.checked_add(delta) else {
                continue;
            };
            let replacement = rewrite_integer(raw, next);
            if replacement == raw {
                continue;
            }
            if let Some(c) = one(
                OpId::LogicChangeConstant,
                format!("constant `{raw}` -> `{replacement}`"),
                vec![Edit::replace(node.start_byte(), node.end_byte(), replacement)],
            ) {
                out.push(c);
            }
        }
    }

    for node in language.string_literals(body) {
        let raw = text(node, ctx.src);
        // Only a plain double-quoted literal with no escapes: inserting before the closing quote is
        // then provably still one literal. A raw string, an f-string or anything with a backslash
        // could have the insertion land inside an escape sequence.
        if raw.len() < 2 || !raw.starts_with('"') || !raw.ends_with('"') || raw.contains('\\') {
            continue;
        }
        let close = node.end_byte() - 1;
        if let Some(c) = one(
            OpId::LogicChangeConstant,
            format!("string constant `{}` extended", truncate(raw)),
            vec![Edit::insert(close, "_x")],
        ) {
            out.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// cosmetic
// ---------------------------------------------------------------------------------------------

fn rename_local(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    let (body_start, body_end) = (body.body.start_byte(), body.body.end_byte());
    // Scopes nested inside this body: closures, lambdas, inner functions. A name bound *here* whose
    // occurrences reach into one of them is either captured or shadowed, and the two are not
    // distinguishable without resolving the inner scope's own bindings.
    //
    // This is load-bearing rather than tidy. `Language::local_bindings` stops at a function
    // boundary, so a closure's parameters are never counted as bindings — which means
    // `fn f() { let v = 1; let c = |v: usize| v + 1; }` looks like a name bound exactly once, all of
    // whose occurrences are inside the body. Renaming it would rewrite the closure's parameter and
    // its uses, and ship that as a behaviour-preserving `cosmetic` diff. It is not.
    let inner_scopes: Vec<(usize, usize)> = ctx
        .boundary_ranges
        .iter()
        .copied()
        .filter(|(s, e)| *s > body_start && *e <= body_end)
        .collect();
    let mut out = Vec::new();

    for binding in ctx.language.local_bindings(body) {
        let name = binding.name.as_str();
        // 1. Bound exactly once in the whole file. More than once and some occurrence below belongs
        //    to a different binding — the shadowing case.
        if ctx.binding_counts.get(name).copied().unwrap_or(0) != 1 {
            continue;
        }
        let Some(sites) = ctx.occurrences.get(name) else {
            continue;
        };
        // 2. Never referenced outside this body. An occurrence outside is either a capture or a
        //    different symbol; either way the rename is not confined and is not provably safe.
        if sites
            .iter()
            .any(|o| o.start < body_start || o.end > body_end)
        {
            continue;
        }
        // 3. Never a member name. `mutex.lock` must not be rewritten by renaming a local `lock`.
        if sites.iter().any(|o| {
            MEMBER_KINDS.contains(&o.kind)
                || (MEMBER_ACCESS_PARENTS.contains(&o.parent_kind) && !o.is_first_named_child)
        }) {
            continue;
        }
        // 4. Never reaches into a nested scope. Refusing a legitimate capture costs one example;
        //    renaming a shadow costs a poisoned `cosmetic` label, which is worse than having none.
        if sites.iter().any(|o| {
            inner_scopes
                .iter()
                .any(|(s, e)| o.start >= *s && o.end <= *e)
        }) {
            continue;
        }
        if sites.is_empty() || sites.len() > MAX_EDITS {
            continue;
        }

        for candidate_name in RENAME_POOL {
            if *candidate_name == name {
                continue;
            }
            // 5. The new name must not already mean something in this file.
            if ctx.occurrences.contains_key(*candidate_name) {
                continue;
            }
            let edits: Vec<Edit> = sites
                .iter()
                .map(|o| Edit::replace(o.start, o.end, *candidate_name))
                .collect();
            if let Some(c) = one(
                OpId::CosmeticRenameLocal,
                format!("local `{name}` renamed to `{candidate_name}` at {} sites", sites.len()),
                edits,
            ) {
                out.push(c);
            }
        }
    }
    out
}

fn reformat(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Result<Vec<Candidate>, Refusal> {
    let Some(resolved) = ctx.formatter else {
        return Err(Refusal::NoFormatter {
            language: ctx.language.id().to_string(),
            operator: OpId::CosmeticReformat.as_str().to_string(),
        });
    };
    let canonical = ctx
        .canonical_src()
        .map_err(|e| Refusal::CosmeticNotPreserving {
            operator: OpId::CosmeticReformat.as_str().to_string(),
            detail: format!("could not canonicalise the input: {e}"),
        })?
        .to_string();
    if canonical == ctx.src {
        // Already canonical. Not a refusal: there is genuinely nothing to reformat.
        return Ok(Vec::new());
    }

    let before_lines: Vec<&str> = ctx.src.split('\n').collect();
    let after_lines: Vec<&str> = canonical.split('\n').collect();
    if before_lines.len() > MAX_REFORMAT_LINES || after_lines.len() > MAX_REFORMAT_LINES {
        return Err(Refusal::PreconditionUnmet {
            operator: OpId::CosmeticReformat.as_str().to_string(),
            detail: format!(
                "aligning {} against {} lines is over the {MAX_REFORMAT_LINES}-line cap",
                before_lines.len(),
                after_lines.len()
            ),
        });
    }

    let body_start_line = crate::span::line_of(ctx.src, body.body.start_byte()) as usize;
    let body_end_line = crate::span::line_of(ctx.src, body.body.end_byte()) as usize;

    let mut out = Vec::new();
    for hunk in line_hunks(&before_lines, &after_lines) {
        // One hunk only, and only one inside this body: the mutation must be one mutation, and it
        // must land where the model is being taught to look.
        let hunk_first_line = hunk.before_start + 1;
        if hunk_first_line < body_start_line || hunk_first_line > body_end_line {
            continue;
        }
        let Some((start, end)) = line_byte_range(ctx.src, hunk.before_start, hunk.before_end) else {
            continue;
        };
        let mut replacement = String::new();
        for line in after_lines.get(hunk.after_start..hunk.after_end).unwrap_or_default() {
            replacement.push_str(line);
            replacement.push('\n');
        }
        let original = ctx.src.get(start..end).unwrap_or_default();
        if replacement == original {
            continue;
        }
        // **The hunk must be layout-only.** A whole-file reformat can move a token across a line
        // boundary — swift-format pulling a `{` up onto the signature line is the case that found
        // this — and the alignment then splits that single move into two hunks: one that removes
        // the token and one that adds it. Applying either alone unbalances the file, which is a
        // `cosmetic` diff that does not parse.
        //
        // Requiring the hunk's non-whitespace content to be unchanged makes each candidate
        // self-contained by construction: no token crosses its boundary, so no partner hunk is
        // needed for it to be valid. It is also exactly what `reformat` claims to be — a change of
        // layout — so a hunk that fails this is one the operator should not have offered anyway.
        if !same_ignoring_whitespace(original, &replacement) {
            continue;
        }
        if let Some(c) = one(
            OpId::CosmeticReformat,
            format!(
                "lines {}..={} reformatted by {}",
                hunk.before_start + 1,
                hunk.before_end,
                resolved.path
            ),
            vec![Edit::replace(start, end, replacement)],
        ) {
            out.push(c);
        }
    }
    Ok(out)
}

fn reorder_imports(ctx: &OpCtx<'_>, _body: &FunctionBody<'_>) -> Result<Vec<Candidate>, Refusal> {
    // `docs/mutation-operators.md` shrinks the no-formatter set to `edit_comment` and a provably
    // safe `rename_local`. `docs/hardening.md` §1 words the same shrink as "comment text, import
    // order within a group". They disagree about this operator; the specification document wins and
    // the restriction is recorded rather than resolved silently in our favour.
    let Some(_resolved) = ctx.formatter else {
        return Err(Refusal::NoFormatter {
            language: ctx.language.id().to_string(),
            operator: OpId::CosmeticReorderImports.as_str().to_string(),
        });
    };
    let Some(block) = ctx.language.import_block(ctx.tree, ctx.src) else {
        return Ok(Vec::new());
    };
    if !block.reorderable {
        return Err(Refusal::CosmeticNotPreserving {
            operator: OpId::CosmeticReorderImports.as_str().to_string(),
            detail: block
                .why_not
                .unwrap_or("import order is semantic here")
                .to_string(),
        });
    }
    let mut out = Vec::new();
    for pair in block.items.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let (ta, tb) = (text(a, ctx.src), text(b, ctx.src));
        if ta == tb {
            continue;
        }
        if let Some(c) = one(
            OpId::CosmeticReorderImports,
            format!("`{}` and `{}` swapped", truncate(ta), truncate(tb)),
            vec![
                Edit::replace(a.start_byte(), a.end_byte(), tb),
                Edit::replace(b.start_byte(), b.end_byte(), ta),
            ],
        ) {
            out.push(c);
        }
    }
    Ok(out)
}

fn edit_comment(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Vec<Candidate> {
    let mut out = Vec::new();

    for node in ctx.language.comments(body) {
        let raw = text(node, ctx.src);
        for phrase in COMMENT_POOL {
            let Some(rewritten) = rewrite_comment(raw, phrase) else {
                continue;
            };
            if rewritten == raw {
                continue;
            }
            if let Some(c) = one(
                OpId::CosmeticEditComment,
                format!("comment rewritten: `{}`", truncate(raw)),
                vec![Edit::replace(node.start_byte(), node.end_byte(), rewritten)],
            ) {
                out.push(c);
            }
        }
    }

    // A docstring is a comment that the language happens to keep at run time. Its *content* node is
    // handed over without its delimiters, so the phrase goes in directly.
    if let Some(node) = ctx.language.docstring(body) {
        let raw = text(node, ctx.src);
        for phrase in COMMENT_POOL {
            if raw == *phrase {
                continue;
            }
            if let Some(c) = one(
                OpId::CosmeticEditComment,
                format!("docstring rewritten: `{}`", truncate(raw)),
                vec![Edit::replace(node.start_byte(), node.end_byte(), *phrase)],
            ) {
                out.push(c);
            }
        }
    }
    out
}

fn wrap_line(ctx: &OpCtx<'_>, body: &FunctionBody<'_>) -> Result<Vec<Candidate>, Refusal> {
    if !ctx.language.line_wrapping_is_safe() {
        return Err(Refusal::CosmeticNotPreserving {
            operator: OpId::CosmeticWrapLine.as_str().to_string(),
            detail: format!(
                "{} terminates statements at line ends, so a wrap can change the program",
                ctx.language.id()
            ),
        });
    }
    if ctx.formatter.is_none() {
        return Err(Refusal::NoFormatter {
            language: ctx.language.id().to_string(),
            operator: OpId::CosmeticWrapLine.as_str().to_string(),
        });
    }

    let (body_start, body_end) = (body.body.start_byte(), body.body.end_byte());
    let mut out = Vec::new();
    let mut line_start = 0usize;
    for line in ctx.src.split_inclusive('\n') {
        let start = line_start;
        line_start += line.len();
        if start < body_start || start >= body_end {
            continue;
        }
        let trimmed = line.trim_end_matches('\n');
        if trimmed.chars().count() <= WRAP_THRESHOLD {
            continue;
        }
        let Some(at) = wrap_point(trimmed) else {
            continue;
        };
        let indent = edit::indent_of_line(ctx.src, start);
        if let Some(c) = one(
            OpId::CosmeticWrapLine,
            format!("line {} wrapped", crate::span::line_of(ctx.src, start)),
            vec![Edit::insert(start + at, format!("\n{indent}    "))],
        ) {
            out.push(c);
        }
    }
    Ok(out)
}

/// A byte offset just after a top-level `, ` on `line`, outside any string or comment.
///
/// Returns `None` when there is no such point, which refuses the wrap rather than picking one that
/// only looks safe — the operator is allowed to produce nothing.
fn wrap_point(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut depth = 0i32;
    let mut in_string: Option<u8> = None;
    let mut i = 0usize;
    let mut best: Option<usize> = None;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(quote) = in_string {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == quote {
                in_string = None;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' | b'\'' => in_string = Some(b),
            b'/' if bytes.get(i + 1) == Some(&b'/') => break,
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            // Just after the comma and its space, and only past the indentation, so the wrap
            // actually shortens the line.
            b',' if depth > 0
                && bytes.get(i + 1) == Some(&b' ')
                && i + 2 > 8
                && i + 2 < bytes.len() =>
            {
                best = Some(i + 2);
            }
            _ => {}
        }
        i += 1;
    }
    best
}

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------

/// A decimal integer literal's value, or `None` for any other spelling.
///
/// Hex, octal, binary and float literals are declined rather than reformatted into decimal: the
/// diff `0xFF` -> `256` is a base change as well as a value change, and the operator's label claims
/// only the second.
fn decimal_value(raw: &str) -> Option<i64> {
    let digits: String = raw
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '_')
        .filter(|c| *c != '_')
        .collect();
    if digits.is_empty() {
        return None;
    }
    // Anything after the digits must be a type suffix (`10u32`, `10L`), never another base marker.
    let rest = &raw[raw
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit() || *c == '_')
        .count()..];
    if rest.starts_with('x') || rest.starts_with('b') || rest.starts_with('o') || rest.starts_with('.')
    {
        return None;
    }
    digits.parse::<i64>().ok()
}

/// Rewrite a decimal literal to `value`, keeping any type suffix it carried.
fn rewrite_integer(raw: &str, value: i64) -> String {
    let digit_len = raw
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit() || *c == '_')
        .count();
    format!("{value}{}", &raw[digit_len..])
}

/// Rewrite a comment's text, keeping its delimiters exactly as they were.
fn rewrite_comment(raw: &str, phrase: &str) -> Option<String> {
    let trimmed = raw.trim_end();
    if let Some(rest) = trimmed.strip_prefix("/*") {
        if !rest.ends_with("*/") {
            return None;
        }
        return Some(format!("/* {phrase} */"));
    }
    for prefix in ["///", "//!", "//", "#"] {
        if trimmed.starts_with(prefix) {
            return Some(format!("{prefix} {phrase}"));
        }
    }
    None
}

/// True when two texts differ only in whitespace.
///
/// Compared lazily rather than by building two filtered `String`s: a reformat candidate is checked
/// once per hunk per body, and the texts are whole line ranges of a source file.
fn same_ignoring_whitespace(a: &str, b: &str) -> bool {
    let mut left = a.chars().filter(|c| !c.is_whitespace());
    let mut right = b.chars().filter(|c| !c.is_whitespace());
    loop {
        match (left.next(), right.next()) {
            (None, None) => return true,
            (x, y) if x == y => continue,
            _ => return false,
        }
    }
}

/// Shorten a snippet for a one-line `detail` field, on a character boundary.
fn truncate(s: &str) -> String {
    let flat = s.replace('\n', " ");
    let mut out: String = flat.chars().take(48).collect();
    if flat.chars().count() > 48 {
        out.push('…');
    }
    out
}

/// The byte range of 0-based half-open line range `[from, to)`.
fn line_byte_range(src: &str, from: usize, to: usize) -> Option<(usize, usize)> {
    if to <= from {
        return None;
    }
    let mut start = None;
    let mut end = None;
    let mut offset = 0usize;
    for (index, line) in src.split_inclusive('\n').enumerate() {
        if index == from {
            start = Some(offset);
        }
        offset += line.len();
        if index + 1 == to {
            end = Some(offset);
        }
    }
    match (start, end) {
        (Some(s), Some(e)) if e > s => Some((s, e)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Hunk {
    before_start: usize,
    before_end: usize,
    after_start: usize,
    after_end: usize,
}

/// Line-level hunks between two texts, from a longest-common-subsequence alignment.
///
/// [`crate::diffspan`] deliberately uses prefix/suffix trimming and nothing else, because its job
/// is to answer "which lines changed" by a route that shares no code with the mutator. This is a
/// different job — it must *produce an edit* that reformats one region, so it needs a real
/// alignment rather than one hunk covering everything between the first and last difference. The
/// two are not interchangeable and must not be merged.
fn line_hunks(before: &[&str], after: &[&str]) -> Vec<Hunk> {
    let (n, m) = (before.len(), after.len());
    // Directions: 0 = diagonal (equal), 1 = came from above, 2 = came from the left.
    let mut table = vec![0u32; (n + 1) * (m + 1)];
    let idx = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            table[idx(i, j)] = if before[i] == after[j] {
                table[idx(i + 1, j + 1)] + 1
            } else {
                table[idx(i + 1, j)].max(table[idx(i, j + 1)])
            };
        }
    }

    let mut hunks = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    let mut pending: Option<Hunk> = None;
    while i < n || j < m {
        let equal = i < n && j < m && before[i] == after[j];
        if equal {
            if let Some(h) = pending.take() {
                hunks.push(h);
            }
            i += 1;
            j += 1;
            continue;
        }
        let take_before = j >= m || (i < n && table[idx(i + 1, j)] >= table[idx(i, j + 1)]);
        let h = pending.get_or_insert(Hunk {
            before_start: i,
            before_end: i,
            after_start: j,
            after_end: j,
        });
        if take_before {
            i += 1;
            h.before_end = i;
            h.after_end = j;
        } else {
            j += 1;
            h.after_end = j;
            h.before_end = i;
        }
    }
    if let Some(h) = pending.take() {
        hunks.push(h);
    }
    // A hunk with no before-lines is a pure insertion; there is no region to replace, and the
    // operator is a *reformat* of something that exists.
    hunks.retain(|h| h.before_end > h.before_start);
    hunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{for_id, LangId};
    use crate::parse::parse_bounded;

    /// Drive one operator over one file's first matching body, with no RNG in the way.
    ///
    /// Going through [`crate::generate`] would make the test depend on a weighted draw picking this
    /// operator, which is how the first version of the shadowed-rename test passed against code
    /// that had the bug.
    fn candidates_for(
        id: LangId,
        src: &str,
        symbol: &str,
        op: OpId,
    ) -> Result<Vec<Candidate>, Refusal> {
        let language = for_id(id);
        let parsed = parse_bounded(&language.grammar(), src).expect("the fixture parses");
        let bodies = language.function_bodies(&parsed.tree, src);
        let body = bodies
            .iter()
            .find(|b| b.name == symbol)
            .unwrap_or_else(|| panic!("no body named {symbol} in the fixture"));
        let ctx = OpCtx::new(language, &parsed.tree, src, None);
        candidates(op, &ctx, body)
    }

    #[test]
    fn a_local_shadowed_by_a_closure_parameter_is_not_a_rename_candidate() {
        // `v` is bound once by `let`, and all five of its occurrences are inside `shadower`'s body,
        // so "bound once, never escapes" passes. Two of those five are the closure's own parameter
        // and its use. `Language::local_bindings` stops at a function boundary, so the parameter is
        // counted as a binding nowhere and the occurrence check alone cannot see the shadow.
        //
        // Against the code before the nested-scope check, this produced
        // `local `v` renamed to `cur` at 5 sites` — a diff that changes behaviour under a
        // `cosmetic` label.
        let src = "\
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
        let found = candidates_for(LangId::Rust, src, "shadower", OpId::CosmeticRenameLocal)
            .expect("the operator runs");
        for candidate in &found {
            assert!(
                !candidate.detail.starts_with("local `v` renamed"),
                "offered {}",
                candidate.detail
            );
        }
        // The operator must still be productive here, or the assertion above is vacuous: `a`, `b`
        // and `bump` are all renameable, and none of them is shadowed.
        assert!(
            found.iter().any(|c| c.detail.starts_with("local `a` renamed")),
            "the unshadowed locals should still be offered: {:?}",
            found.iter().map(|c| &c.detail).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_local_whose_name_is_also_a_field_is_not_a_rename_candidate() {
        // Renaming a local `lock` must not rewrite `mutex.lock`.
        let src = "\
pub fn guard(mutex: Mutex) -> usize {
    let lock = 1;
    let held = mutex.lock;
    lock + held
}
";
        let found = candidates_for(LangId::Rust, src, "guard", OpId::CosmeticRenameLocal)
            .expect("the operator runs");
        assert!(
            found.iter().all(|c| !c.detail.starts_with("local `lock` renamed")),
            "a field access with the same name was in scope: {:?}",
            found.iter().map(|c| &c.detail).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_no_op_stub_candidate_is_declined_at_generation() {
        // `stub.hardcoded` derives its value from a literal already present, and this function's
        // only literal *is* its whole body. Rewriting `3` to `3` is not a mutation.
        let src = "pub fn one_line() -> u32 { 3 }\n";
        let found = candidates_for(LangId::Rust, src, "one_line", OpId::StubHardcoded);
        match found {
            Ok(list) => assert!(
                list.is_empty(),
                "offered a no-op: {:?}",
                list.iter().map(|c| &c.detail).collect::<Vec<_>>()
            ),
            // Declining for want of a matching literal is also an acceptable answer.
            Err(Refusal::PreconditionUnmet { .. }) => {}
            Err(other) => panic!("unexpected refusal {other:?}"),
        }
    }

    #[test]
    fn a_formatter_less_language_refuses_the_operators_that_need_one() {
        let src = "\
export function f(a: number): number {
  const x = 1;
  return x + a;
}
";
        // `OpCtx` is built with `formatter: None` by `candidates_for`, which is exactly the state a
        // machine without `prettier` is in.
        for op in [OpId::CosmeticReformat, OpId::CosmeticReorderImports] {
            match candidates_for(LangId::TypeScript, src, "f", op) {
                Err(Refusal::NoFormatter { .. }) => {}
                other => panic!("{op} must refuse without a formatter, got {other:?}"),
            }
        }
        // And the two that do not need one still work.
        assert!(
            candidates_for(LangId::TypeScript, src, "f", OpId::CosmeticRenameLocal)
                .expect("runs")
                .iter()
                .any(|c| c.detail.starts_with("local `x` renamed")),
            "rename_local is provably safe without a formatter and must survive the shrink"
        );
    }

    #[test]
    fn dropping_a_go_else_takes_the_keyword_with_the_block() {
        // Go's `alternative` field is the block *after* the `else` token. Removing the node alone
        // leaves a dangling `} else`, which does not parse.
        let src = "\
package p

func f(a int) int {
\tif a < 3 {
\t\treturn 1
\t} else {
\t\treturn 2
\t}
}
";
        let found = candidates_for(LangId::Go, src, "f", OpId::LogicDropElse).expect("runs");
        assert_eq!(found.len(), 1);
        let applied = found[0].edits.apply(src).expect("applies");
        assert!(
            !applied.text.contains("else"),
            "the keyword was left behind:\n{}",
            applied.text
        );
        let language = for_id(LangId::Go);
        parse_bounded(&language.grammar(), &applied.text)
            .expect("the result must still be Go");
    }

    #[test]
    fn the_operator_ids_are_spelled_exactly_as_the_spec_spells_them() {
        let spelled: Vec<&str> = OpId::ALL.iter().map(|o| o.as_str()).collect();
        assert_eq!(
            spelled,
            vec![
                "stub.panic",
                "stub.default_return",
                "stub.hardcoded",
                "stub.early_return",
                "logic.negate_condition",
                "logic.off_by_one",
                "logic.swap_args",
                "logic.drop_else",
                "logic.widen_comparison",
                "logic.swallow_error",
                "logic.drop_await_or_lock",
                "logic.change_constant",
                "cosmetic.rename_local",
                "cosmetic.reformat",
                "cosmetic.reorder_imports",
                "cosmetic.edit_comment",
                "cosmetic.wrap_line",
            ]
        );
        for op in OpId::ALL {
            assert_eq!(OpId::parse(op.as_str()), Some(op));
        }
    }

    #[test]
    fn the_silent_stubs_outweigh_the_visible_one() {
        let visible = OpId::StubPanic.weight();
        for op in OpId::ALL.into_iter().filter(|o| o.is_silent()) {
            assert!(
                op.weight() > visible * 2,
                "{op} at {} does not outweigh stub.panic at {visible}",
                op.weight()
            );
        }
        assert!(!OpId::StubPanic.is_silent());
    }

    #[test]
    fn a_block_body_region_starts_on_the_line_after_the_brace() {
        let src = "fn f() {\n    x();\n}\n";
        let interior = (
            src.find('{').expect("fixture") + 1,
            src.rfind('}').expect("fixture"),
        );
        let region = body_line_region(src, interior).expect("multi-line");
        assert_eq!(&src[region.0..region.1], "    x();\n");
    }

    #[test]
    fn a_single_line_body_has_no_line_region() {
        let src = "fn f() -> u32 { 1 }\n";
        let interior = (
            src.find('{').expect("fixture") + 1,
            src.rfind('}').expect("fixture"),
        );
        assert_eq!(body_line_region(src, interior), None);
    }

    #[test]
    fn a_python_body_region_is_the_whole_indented_block() {
        let src = "def f():\n    x()\n    y()\n";
        let interior = (src.find("    x()").expect("fixture"), src.len());
        let region = body_line_region(src, interior).expect("multi-line");
        assert_eq!(&src[region.0..region.1], "    x()\n    y()\n");
    }

    #[test]
    fn snapping_leaves_an_else_that_shares_its_line_with_code_alone() {
        let src = "if a { x } else { y }\n";
        let start = src.find("else").expect("fixture");
        let end = src.rfind('}').expect("fixture") + 1;
        assert_eq!(
            snap_if_line_owned(src, start, end),
            (start, end),
            "snapping here would swallow the newline and join two lines"
        );
    }

    #[test]
    fn a_braced_else_sharing_a_line_with_the_if_block_is_left_unsnapped() {
        // The `else` keyword has `} ` before it on the line, so the start cannot move — and if the
        // end moved on its own it would swallow the newline and join two untouched lines.
        let src = "if a {\n    x\n} else {\n    y\n}\nrest\n";
        let start = src.find("else").expect("fixture");
        let end = src.find("}\nrest").expect("fixture") + 1;
        assert_eq!(&src[start..end], "else {\n    y\n}");
        assert_eq!(snap_if_line_owned(src, start, end), (start, end));
    }

    #[test]
    fn an_else_that_starts_its_own_line_snaps_out_to_whole_lines() {
        let src = "if a:\n    x\nelse:\n    y\nz\n";
        let start = src.find("else").expect("fixture");
        let end = src.find("\nz\n").expect("fixture");
        assert_eq!(&src[start..end], "else:\n    y");
        let (s, e) = snap_if_line_owned(src, start, end);
        assert_eq!(s, start, "it already begins at a line start");
        assert_eq!(
            &src[s..e],
            "else:\n    y\n",
            "the trailing newline comes too, so whole lines go and none are joined"
        );
    }

    #[test]
    fn decimal_values_parse_and_other_bases_decline() {
        assert_eq!(decimal_value("10"), Some(10));
        assert_eq!(decimal_value("1_000"), Some(1000));
        assert_eq!(decimal_value("10u32"), Some(10));
        assert_eq!(decimal_value("0xFF"), None, "a base change is not this operator");
        assert_eq!(decimal_value("0b1010"), None);
        assert_eq!(decimal_value("1.5"), None);
        assert_eq!(decimal_value("abc"), None);
    }

    #[test]
    fn rewriting_an_integer_keeps_its_type_suffix() {
        assert_eq!(rewrite_integer("10u32", 11), "11u32");
        assert_eq!(rewrite_integer("1_000", 1001), "1001");
        assert_eq!(rewrite_integer("7", 6), "6");
    }

    #[test]
    fn comment_delimiters_survive_a_rewrite() {
        assert_eq!(
            rewrite_comment("// old", "new text").as_deref(),
            Some("// new text")
        );
        assert_eq!(
            rewrite_comment("/// doc", "new text").as_deref(),
            Some("/// new text")
        );
        assert_eq!(
            rewrite_comment("# py", "new text").as_deref(),
            Some("# new text")
        );
        assert_eq!(
            rewrite_comment("/* a\nb */", "new text").as_deref(),
            Some("/* new text */")
        );
        assert_eq!(rewrite_comment("not a comment", "x"), None);
    }

    #[test]
    fn no_comment_phrase_can_break_out_of_any_delimiter() {
        for phrase in COMMENT_POOL {
            assert!(!phrase.contains('\n'), "{phrase}");
            assert!(!phrase.contains('"'), "{phrase}");
            assert!(!phrase.contains('\\'), "{phrase}");
            assert!(!phrase.contains("*/"), "{phrase}");
        }
    }

    #[test]
    fn line_hunks_finds_one_changed_region_not_the_span_between_two() {
        let before = vec!["a", "b", "c", "d", "e"];
        let after = vec!["a", "B", "c", "d", "E"];
        let hunks = line_hunks(&before, &after);
        assert_eq!(hunks.len(), 2, "two separate edits, not one hunk covering both");
        assert_eq!(hunks[0].before_start, 1);
        assert_eq!(hunks[0].before_end, 2);
        assert_eq!(hunks[1].before_start, 4);
        assert_eq!(hunks[1].before_end, 5);
    }

    #[test]
    fn line_byte_range_covers_whole_lines_including_their_terminators() {
        let src = "aa\nbb\ncc\n";
        assert_eq!(line_byte_range(src, 1, 2), Some((3, 6)));
        assert_eq!(&src[3..6], "bb\n");
        assert_eq!(line_byte_range(src, 2, 2), None);
    }

    #[test]
    fn a_wrap_point_is_never_inside_a_string_or_a_comment() {
        // The only comma is inside the literal, so there is no wrap point at all.
        assert_eq!(
            wrap_point("let x = format_it(\"a, b, c\");"),
            None,
            "every comma here is inside a string"
        );
        // Here there is one comma in the string and one real one; only the real one is offered.
        let mixed = "let x = call_it(\"a, b\", second_argument);";
        let at = wrap_point(mixed).expect("the argument separator is a wrap point");
        assert_eq!(
            &mixed[..at],
            "let x = call_it(\"a, b\", ",
            "it picked the separator, not a comma inside the literal"
        );

        let line = "    let result = compute(first_argument, second_argument, third_argument);";
        let at = wrap_point(line).expect("a top-level comma exists");
        assert!(line[..at].ends_with(", "), "wrapped at {at}: {:?}", &line[..at]);

        assert_eq!(wrap_point("    // a, b, c"), None, "the commas are in a comment");
        assert_eq!(wrap_point("let x = 1;"), None);
        assert_eq!(wrap_point("a, b"), None, "a comma outside any bracket is not a wrap point");
    }
}
