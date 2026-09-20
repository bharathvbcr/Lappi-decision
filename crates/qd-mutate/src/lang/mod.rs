//! The `Language` facade.
//!
//! Operators are written **once** against this trait, never five times per language. A `Language`
//! implementation supplies the node kinds; an operator asks the facade for what it needs.
//!
//! A language whose facade returns an empty list for a construct simply yields no mutations of that
//! kind. That is correct, and it is **recorded per language** in the manifest so a silent zero
//! reads as "no opportunities here", not as "the operator ran".

use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Tree};

pub mod go_lang;
pub mod python_lang;
pub mod rust_lang;
pub mod swift_lang;
pub mod ts_lang;
pub mod util;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LangId {
    Rust,
    Go,
    Python,
    TypeScript,
    Swift,
}

impl LangId {
    pub const ALL: [LangId; 5] = [
        LangId::Rust,
        LangId::Go,
        LangId::Python,
        LangId::TypeScript,
        LangId::Swift,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            LangId::Rust => "rust",
            LangId::Go => "go",
            LangId::Python => "python",
            LangId::TypeScript => "typescript",
            LangId::Swift => "swift",
        }
    }

    pub fn parse(s: &str) -> Option<LangId> {
        match s {
            "rust" | "rs" => Some(LangId::Rust),
            "go" => Some(LangId::Go),
            "python" | "py" => Some(LangId::Python),
            "typescript" | "ts" => Some(LangId::TypeScript),
            "swift" => Some(LangId::Swift),
            _ => None,
        }
    }

    /// Extension used when a formatter needs a filename to decide its dialect.
    pub fn extension(self) -> &'static str {
        match self {
            LangId::Rust => "rs",
            LangId::Go => "go",
            LangId::Python => "py",
            LangId::TypeScript => "ts",
            LangId::Swift => "swift",
        }
    }
}

impl std::fmt::Display for LangId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A function or method **with a real body**.
///
/// Declarations without one — trait signatures, interface members, abstract methods — never become
/// a `FunctionBody`. There is nothing to stub, and emitting an empty span would be a poisoned
/// label.
#[derive(Clone)]
pub struct FunctionBody<'t> {
    /// The declaration node.
    pub decl: Node<'t>,
    /// The body node. For Rust/Go/TS a block; for Python a `block`; for Swift the `statements`
    /// inside a `function_body`.
    pub body: Node<'t>,
    /// The byte range of the body's *interior* — what a stub replaces. Excludes the delimiters, so
    /// replacing it leaves `{` and `}` where they were.
    pub interior: (usize, usize),
    pub name: String,
    pub arity: usize,
    /// Rendered return type, when the language resolves one. `None` means the facade could not
    /// resolve it, which makes `stub.default_return` decline rather than guess.
    pub return_type: Option<String>,
    /// Declared parameter types by name, for `logic.swap_args`. Empty where the language does not
    /// declare them.
    pub params: Vec<(String, Option<String>)>,
    /// True when this body is lexically inside another function body. "The function body" is
    /// ambiguous for nested functions and closures; the mutator names which node it took, and this
    /// says whether that node was an inner one.
    pub is_nested: bool,
    /// The grammar's own name for `decl`. Carried into the example record for the same reason.
    pub node_kind: &'static str,
    /// The normalized source these nodes index into.
    pub src: &'t str,
}

impl<'t> FunctionBody<'t> {
    pub fn interior_text(&self) -> &'t str {
        self.src
            .get(self.interior.0..self.interior.1)
            .unwrap_or_default()
    }

    pub fn body_text(&self) -> &'t str {
        self.src
            .get(self.body.start_byte()..self.body.end_byte())
            .unwrap_or_default()
    }

    /// `(repo, path, symbol_name, arity)` is the split-safety identity; this is its local half.
    pub fn identity(&self) -> (String, usize) {
        (self.name.clone(), self.arity)
    }
}

/// A comparison operator token and what it currently says.
pub struct ComparisonSite<'t> {
    pub op_node: Node<'t>,
    pub op: String,
}

/// A call, with its argument nodes in source order.
pub struct CallSite<'t> {
    pub call: Node<'t>,
    pub args: Vec<Node<'t>>,
}

/// A site where an error is propagated, and what swallowing it looks like here.
pub struct ErrorSite<'t> {
    /// The byte range to replace.
    pub range: (usize, usize),
    /// The replacement that swallows the error.
    pub replacement: String,
    /// True when the whole statement goes away, so the edit should snap to whole lines.
    pub whole_statement: bool,
    pub node: Node<'t>,
}

/// An `await` or a lock acquisition, with the range whose removal still parses.
pub struct SyncSite<'t> {
    pub range: (usize, usize),
    pub whole_statement: bool,
    pub node: Node<'t>,
}

/// An `else` branch, with **the exact range whose removal leaves valid syntax**.
///
/// The range is not always the node's own extent, which is why this is a site rather than a bare
/// `Node`. Rust, Python and TypeScript wrap the keyword and the block in one `else_clause`, so the
/// range is that node. Go's `alternative` field points at the *block only*, with the `else` keyword
/// as its preceding sibling — deleting the node alone would leave a dangling `} else`. Swift splits
/// it three ways: a named `else` node, then `{`, `statements`, `}`. Each language states its own
/// range here rather than an operator guessing from a node kind it does not own.
pub struct ElseSite<'t> {
    pub range: (usize, usize),
    pub node: Node<'t>,
}

/// A local binding and the identifier that names it.
pub struct Binding<'t> {
    pub name_node: Node<'t>,
    pub name: String,
}

/// A contiguous group of imports.
pub struct ImportBlock<'t> {
    /// One node per import, in source order.
    pub items: Vec<Node<'t>>,
    /// False when reordering would change behaviour here.
    pub reorderable: bool,
    /// Why not, when `reorderable` is false. Recorded, never assumed.
    pub why_not: Option<&'static str>,
}

/// An external formatter, used to verify that a `cosmetic` mutation really is behaviour-preserving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Formatter {
    /// Program name, or the tool `xcrun` should find.
    pub program: &'static str,
    pub args: &'static [&'static str],
    /// Resolve `program` through `xcrun --find` first (the Xcode toolchain is not on `PATH`).
    pub via_xcrun: bool,
}

/// The facade. Object-safe: operators hold `&dyn Language`.
pub trait Language: Sync + Send {
    fn id(&self) -> LangId;
    fn grammar(&self) -> tree_sitter::Language;

    /// Functions/methods with a real body. Excludes declarations without one.
    fn function_bodies<'t>(&self, tree: &'t Tree, src: &'t str) -> Vec<FunctionBody<'t>>;

    fn conditions<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    fn comparison_ops<'t>(&self, body: &FunctionBody<'t>) -> Vec<ComparisonSite<'t>>;
    fn call_arguments<'t>(&self, body: &FunctionBody<'t>) -> Vec<CallSite<'t>>;
    fn else_branches<'t>(&self, body: &FunctionBody<'t>) -> Vec<ElseSite<'t>>;
    fn error_propagations<'t>(&self, body: &FunctionBody<'t>) -> Vec<ErrorSite<'t>>;
    fn awaits_and_locks<'t>(&self, body: &FunctionBody<'t>) -> Vec<SyncSite<'t>>;
    fn integer_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    /// Integer literals in a **bound** position — a loop or slice bound. `logic.off_by_one` uses
    /// only these; perturbing any integer is `logic.change_constant`.
    fn bound_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    fn string_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    fn local_bindings<'t>(&self, body: &FunctionBody<'t>) -> Vec<Binding<'t>>;
    fn comments<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    /// A docstring node whose *content* may be rewritten, where the language has them.
    fn docstring<'t>(&self, body: &FunctionBody<'t>) -> Option<Node<'t>>;
    fn import_block<'t>(&self, tree: &'t Tree, src: &'t str) -> Option<ImportBlock<'t>>;

    /// Every identifier node in the file, for the rename-safety proof.
    fn identifiers<'t>(&self, tree: &'t Tree) -> Vec<Node<'t>>;
    /// True for a node kind that starts a new function scope — a nested fn, a closure, a lambda.
    fn is_function_boundary(&self, kind: &str) -> bool;

    /// The body's statements as text, comments dropped.
    ///
    /// This is what [`util::body_is_already_stub`] consumes. It works on **statement nodes**, never
    /// on a substring of the file, which is why a string literal containing `todo!()` does not read
    /// as a stub body — that substring gate is precisely what the model is being trained to beat.
    fn statement_texts(&self, body: &FunctionBody<'_>) -> Vec<String>;

    /// Source text for `stub` bodies.
    fn panic_stub(&self) -> &'static [&'static str];
    /// The zero value for the return type, rendered as the **whole replacement body**. `None` means
    /// it could not be resolved, and the operator declines rather than guessing.
    ///
    /// Note the difference from [`Language::early_return`]: Rust's replacement body is the tail
    /// expression `Ok(())`, which is not a statement and cannot be inserted above live code.
    fn default_return(&self, return_type: Option<&str>) -> Option<String>;
    /// The same zero, rendered as a **statement that can sit above the original body** — which is
    /// what `stub.early_return` inserts. Rust needs `return …;` here where `default_return` needs a
    /// bare expression; the other four spell both the same way.
    fn early_return(&self, return_type: Option<&str>) -> Option<String>;
    /// A hardcoded return built from a literal already present in the function. `None` where the
    /// literal does not match the declared return type.
    fn hardcoded_return(&self, return_type: Option<&str>, literal: &str, kind: LiteralKind)
    -> Option<String>;
    /// Negate a condition's rendered text.
    fn negate(&self, condition: &str) -> String;
    /// True when the condition is already negated at the top level.
    fn is_negated(&self, node: &Node<'_>, src: &str) -> bool;
    fn line_comment_prefix(&self) -> &'static str;

    /// Formatter for verifying `cosmetic` really is behaviour-preserving. `None` **restricts** the
    /// cosmetic operator set rather than being assumed harmless.
    fn formatter(&self) -> Option<Formatter>;
    /// False where the language is whitespace-sensitive, which refuses `cosmetic.wrap_line`.
    fn line_wrapping_is_safe(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralKind {
    Integer,
    Str,
}

/// The one place a `LangId` becomes an implementation.
pub fn for_id(id: LangId) -> &'static dyn Language {
    match id {
        LangId::Rust => &rust_lang::Rust,
        LangId::Go => &go_lang::Go,
        LangId::Python => &python_lang::Python,
        LangId::TypeScript => &ts_lang::TypeScript,
        LangId::Swift => &swift_lang::Swift,
    }
}
