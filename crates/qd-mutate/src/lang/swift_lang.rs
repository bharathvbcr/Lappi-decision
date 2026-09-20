//! Swift. Node kinds verified against `tree-sitter-swift` 0.7.3 with `examples/probe.rs`.
//!
//! This grammar overloads the `name` field harder than any of the other four: on a
//! `function_declaration`, the function's own identifier, **each parameter's type**, and **the
//! return type** all arrive under the field name `name`. `child_by_field_name("name")` therefore
//! answers the function's identifier and nothing else useful, and the return type has to be found
//! positionally — it is the named node following the anonymous `->` token. Getting this wrong
//! silently gives `stub.default_return` the wrong type to zero, which is a poisoned label rather
//! than a compile error, so the lookup is written once here and tested.
//!
//! Three more shapes that differ from the other four languages:
//!
//! * `protocol_function_declaration` is a distinct kind with no `body` field, so protocol
//!   requirements are excluded without being named. A default implementation lives in an
//!   `extension` as an ordinary `function_declaration` **with** a body, and is mutated like any
//!   other — which is the trait-default-body case from the adversarial corpus.
//! * An `if_statement`'s else is a bare named `else` node followed by `{ statements }` as loose
//!   siblings, not a single clause node. The removable range runs from that node to the end of the
//!   enclosing `if_statement`.
//! * A call's arguments are not a field: they sit under `call_suffix` -> `value_arguments`.

use tree_sitter::{Node, Tree};

use super::util::{self, all_children, named_children, text, walk_body, walk_tree};
use super::{
    Binding, CallSite, ComparisonSite, ElseSite, ErrorSite, Formatter, FunctionBody, ImportBlock,
    LangId, Language, LiteralKind, SyncSite,
};

pub struct Swift;

const PANIC_STUBS: &[&str] = &[
    "fatalError(\"not implemented\")",
    "preconditionFailure(\"not implemented\")",
];

fn is_boundary(kind: &str) -> bool {
    matches!(
        kind,
        "function_declaration" | "lambda_literal" | "init_declaration" | "deinit_declaration"
    )
}

/// Node kinds that denote a type in this grammar.
fn is_type_kind(kind: &str) -> bool {
    matches!(
        kind,
        "user_type"
            | "array_type"
            | "dictionary_type"
            | "optional_type"
            | "function_type"
            | "tuple_type"
            | "metatype"
            | "protocol_composition_type"
            | "opaque_type"
            | "existential_type"
    )
}

/// The declared return type of a `function_declaration`: the named node that follows the `->`
/// token.
///
/// Absent `->` means the function returns `Void`, which is the language's own answer rather than a
/// failure to resolve — so `Some("Void")`, not `None`.
fn return_type_of(decl: Node<'_>, src: &str) -> Option<String> {
    let children = all_children(decl);
    let arrow = children.iter().position(|c| !c.is_named() && text(*c, src) == "->");
    match arrow {
        Some(i) => children
            .iter()
            .skip(i + 1)
            .find(|c| c.is_named() && is_type_kind(c.kind()))
            .map(|c| text(*c, src).trim().to_string()),
        None => Some("Void".to_string()),
    }
}

/// The `value_argument` nodes of a call, which live under `call_suffix` -> `value_arguments`.
fn call_args<'t>(call: Node<'t>) -> Vec<Node<'t>> {
    for child in all_children(call) {
        if child.kind() != "call_suffix" {
            continue;
        }
        for grand in all_children(child) {
            if grand.kind() == "value_arguments" {
                return named_children(grand)
                    .into_iter()
                    .filter(|n| n.kind() == "value_argument")
                    .collect();
            }
        }
    }
    Vec::new()
}

impl Language for Swift {
    fn id(&self) -> LangId {
        LangId::Swift
    }

    fn grammar(&self) -> tree_sitter::Language {
        tree_sitter_swift::LANGUAGE.into()
    }

    fn function_bodies<'t>(&self, tree: &'t Tree, src: &'t str) -> Vec<FunctionBody<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            if node.kind() != "function_declaration" {
                return;
            }
            let Some(body) = node.child_by_field_name("body") else {
                return;
            };
            if body.kind() != "function_body" {
                return;
            }
            // The first `name`-field child is the function's identifier; the later ones are types.
            let name = node
                .child_by_field_name("name")
                .filter(|n| n.kind() == "simple_identifier")
                .map(|n| text(n, src).to_string())
                .unwrap_or_default();
            if name.is_empty() {
                return;
            }
            let params: Vec<(String, Option<String>)> = all_children(node)
                .into_iter()
                .filter(|c| c.kind() == "parameter")
                .map(|param| {
                    let kids = all_children(param);
                    let pname = kids
                        .iter()
                        .find(|c| c.kind() == "simple_identifier")
                        .map(|c| text(*c, src).to_string())
                        .unwrap_or_default();
                    let ptype = kids
                        .iter()
                        .find(|c| is_type_kind(c.kind()))
                        .map(|c| text(*c, src).trim().to_string());
                    (pname, ptype)
                })
                .collect();
            out.push(FunctionBody {
                decl: node,
                body,
                interior: util::interior_lines(body, src),
                name,
                arity: params.len(),
                return_type: return_type_of(node, src),
                params,
                is_nested: util::has_function_ancestor(node, &is_boundary),
                node_kind: "function_declaration",
                src,
            });
        });
        out.sort_by_key(|f| f.decl.start_byte());
        out
    }

    fn conditions<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // `guard` is deliberately excluded: its condition is the *inverse* of an `if`'s, so
            // negating it reads as the same edit while meaning the opposite, and the two would
            // train the model on contradictory examples of one operator.
            if matches!(node.kind(), "if_statement" | "while_statement")
                && let Some(c) = node.child_by_field_name("condition")
            {
                out.push(c);
            }
        });
        out
    }

    fn comparison_ops<'t>(&self, body: &FunctionBody<'t>) -> Vec<ComparisonSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "comparison_expression" {
                return;
            }
            if let Some(op) = node.child_by_field_name("op") {
                let s = text(op, body.src);
                if matches!(s, "<" | ">" | "<=" | ">=") {
                    out.push(ComparisonSite {
                        op_node: op,
                        op: s.to_string(),
                    });
                }
            }
        });
        out
    }

    fn call_arguments<'t>(&self, body: &FunctionBody<'t>) -> Vec<CallSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "call_expression" {
                return;
            }
            // The *value* of each argument, not the `value_argument` wrapper. Swift argument
            // labels bind to positions, so swapping two labelled wrappers (`f(x: a, y: b)` ->
            // `f(y: b, x: a)`) is the same call — it would emit a `logic.swap_args` diff that
            // changes nothing. Swapping the values is the mutation that actually swaps them.
            let args: Vec<Node<'t>> = call_args(node)
                .into_iter()
                .filter_map(|a| a.child_by_field_name("value").or(Some(a)))
                .collect();
            if !args.is_empty() {
                out.push(CallSite { call: node, args });
            }
        });
        out
    }

    fn else_branches<'t>(&self, body: &FunctionBody<'t>) -> Vec<ElseSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "if_statement" {
                return;
            }
            // The `else` keyword is a named node and the branch that follows it is a loose
            // `{ statements }` sibling, so the removable range is from the keyword to the end of
            // the whole `if_statement` — the else is always last. This also covers `else if`,
            // where the sibling is a nested `if_statement`.
            let Some(kw) = all_children(node).into_iter().find(|c| c.kind() == "else") else {
                return;
            };
            out.push(ElseSite {
                range: (kw.start_byte(), node.end_byte()),
                node: kw,
            });
        });
        out
    }

    fn error_propagations<'t>(&self, body: &FunctionBody<'t>) -> Vec<ErrorSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // `try expr` -> `try? expr`. This is Swift's exact analogue of Rust's `?` becoming a
            // default: the error stops propagating and the value silently becomes `nil`. It is the
            // realistic agent failure here, and unlike deleting the call it still parses.
            if node.kind() != "try_expression" {
                return;
            }
            let Some(op) = all_children(node).into_iter().find(|c| c.kind() == "try_operator")
            else {
                return;
            };
            // `try?` and `try!` are already written; re-marking them changes nothing.
            if text(op, body.src).trim() != "try" {
                return;
            }
            out.push(ErrorSite {
                range: (op.start_byte(), op.end_byte()),
                replacement: "try?".to_string(),
                whole_statement: false,
                node,
            });
        });
        out
    }

    fn awaits_and_locks<'t>(&self, body: &FunctionBody<'t>) -> Vec<SyncSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() == "await_expression"
                && let Some(inner) = node.child_by_field_name("expr")
            {
                // `await expr` — drop the keyword and the space up to the operand.
                out.push(SyncSite {
                    range: (node.start_byte(), inner.start_byte()),
                    whole_statement: false,
                    node,
                });
            }
            // `lock.lock()` as a statement of its own. A lock acquisition takes no arguments; a
            // call with some is something else wearing a similar name.
            if node.kind() == "call_expression"
                && call_args(node).is_empty()
                && let Some(nav) = all_children(node)
                    .into_iter()
                    .find(|c| c.kind() == "navigation_expression")
                && let Some(suffix) = nav.child_by_field_name("suffix")
                && let Some(field) = suffix.child_by_field_name("suffix")
                && matches!(text(field, body.src), "lock" | "wait" | "enter")
                && node.parent().map(|p| p.kind()) == Some("statements")
            {
                out.push(SyncSite {
                    range: (node.start_byte(), node.end_byte()),
                    whole_statement: true,
                    node,
                });
            }
        });
        out
    }

    fn integer_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() == "integer_literal" {
                out.push(node);
            }
        });
        out
    }

    fn bound_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // `a..<n` / `a...n` — the end is a bound.
            if node.kind() == "range_expression"
                && let Some(end) = node.child_by_field_name("end")
                && end.kind() == "integer_literal"
            {
                out.push(end);
            }
            if node.kind() == "comparison_expression"
                && let Some(rhs) = node.child_by_field_name("rhs")
                && rhs.kind() == "integer_literal"
            {
                out.push(rhs);
            }
        });
        out
    }

    fn string_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // Only a single-line literal, and only one with no interpolation: a
            // `interpolated_expression` child is code, and perturbing it is not a constant change.
            if node.kind() == "line_string_literal"
                && !all_children(node)
                    .iter()
                    .any(|c| c.kind() == "interpolated_expression")
            {
                out.push(node);
            }
        });
        out
    }

    fn local_bindings<'t>(&self, body: &FunctionBody<'t>) -> Vec<Binding<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "property_declaration" {
                return;
            }
            let Some(pattern) = node.child_by_field_name("name") else {
                return;
            };
            if let Some(ident) = pattern.child_by_field_name("bound_identifier")
                && ident.kind() == "simple_identifier"
            {
                out.push(Binding {
                    name_node: ident,
                    name: text(ident, body.src).to_string(),
                });
            }
        });
        out
    }

    fn comments<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if matches!(node.kind(), "comment" | "multiline_comment") {
                out.push(node);
            }
        });
        out
    }

    fn docstring<'t>(&self, _body: &FunctionBody<'t>) -> Option<Node<'t>> {
        // Swift documentation is `///` above the declaration, already a `comment` node.
        None
    }

    fn import_block<'t>(&self, tree: &'t Tree, src: &'t str) -> Option<ImportBlock<'t>> {
        let mut items: Vec<Node<'t>> = Vec::new();
        let mut cursor = tree.root_node().walk();
        for child in tree.root_node().named_children(&mut cursor) {
            if child.kind() == "import_declaration" {
                items.push(child);
            } else if !items.is_empty() {
                break; // one contiguous group only
            }
        }
        if items.len() < 2 {
            return None;
        }
        // Swift module imports have no ordered side effect: a module is initialised on first use,
        // not at the import statement. The one shape that is order-sensitive in practice is a
        // conditional import guarded by `#if`, which is not an `import_declaration` at top level
        // and so cannot appear in this list. A submodule-kind import (`import struct Foo.Bar`) is
        // still order-free.
        let has_preprocessor = items
            .iter()
            .any(|n| text(*n, src).contains("#if") || text(*n, src).contains("@_exported"));
        Some(ImportBlock {
            items,
            reorderable: !has_preprocessor,
            why_not: has_preprocessor
                .then_some("a conditional or re-exported import is not order-free"),
        })
    }

    fn identifiers<'t>(&self, tree: &'t Tree) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            if matches!(node.kind(), "simple_identifier" | "type_identifier") {
                out.push(node);
            }
        });
        out
    }

    fn is_function_boundary(&self, kind: &str) -> bool {
        is_boundary(kind)
    }

    fn statement_texts(&self, body: &FunctionBody<'_>) -> Vec<String> {
        statement_texts(body)
    }

    fn panic_stub(&self) -> &'static [&'static str] {
        PANIC_STUBS
    }

    fn default_return(&self, return_type: Option<&str>) -> Option<String> {
        let raw = return_type?.trim();
        zero_value(raw).map(|z| format!("return {z}"))
    }

    fn early_return(&self, return_type: Option<&str>) -> Option<String> {
        // `return x` is already a statement; the two renderings coincide.
        self.default_return(return_type)
    }

    fn hardcoded_return(
        &self,
        return_type: Option<&str>,
        literal: &str,
        kind: LiteralKind,
    ) -> Option<String> {
        let raw = return_type?.trim();
        let ok = match kind {
            LiteralKind::Integer => is_integer_type(raw),
            LiteralKind::Str => raw == "String",
        };
        ok.then(|| format!("return {literal}"))
    }

    fn negate(&self, condition: &str) -> String {
        format!("!({condition})")
    }

    fn is_negated(&self, node: &Node<'_>, src: &str) -> bool {
        node.kind() == "prefix_expression" && text(*node, src).trim_start().starts_with('!')
    }

    fn line_comment_prefix(&self) -> &'static str {
        "//"
    }

    fn formatter(&self) -> Option<Formatter> {
        // The Xcode toolchain is not on `PATH`; `xcrun --find swift-format` locates it, which is
        // what `via_xcrun` asks `crate::fmt` to do. `-` is required for stdin: invoking it without
        // a path is deprecated and warns.
        Some(Formatter {
            program: "swift-format",
            args: &["format", "-"],
            via_xcrun: true,
        })
    }

    fn line_wrapping_is_safe(&self) -> bool {
        // A newline terminates a statement in Swift, so `return\n  x` returns Void. The wrap is
        // safe only inside brackets, and "safe where I happened to put it" is not a proof.
        false
    }
}

fn is_integer_type(ty: &str) -> bool {
    matches!(
        ty,
        "Int" | "Int8"
            | "Int16"
            | "Int32"
            | "Int64"
            | "UInt"
            | "UInt8"
            | "UInt16"
            | "UInt32"
            | "UInt64"
    )
}

/// The zero value for a Swift type, or `None` where it cannot be resolved.
fn zero_value(ty: &str) -> Option<String> {
    let ty = ty.trim();
    if ty.is_empty() {
        return None;
    }
    if ty == "Void" || ty == "()" {
        // Nothing to return. The mutation would be "delete the body", which is `stub.panic`'s shape
        // wearing a different label; decline rather than blur the two classes.
        return None;
    }
    if ty.ends_with('?') || ty.ends_with('!') {
        return Some("nil".to_string());
    }
    if ty.starts_with('[') && ty.ends_with(']') {
        // `[T]` is an array and `[K: V]` a dictionary; both zero to their own empty literal.
        return Some(if ty.contains(':') { "[:]" } else { "[]" }.to_string());
    }
    if ty.starts_with("Optional<") {
        return Some("nil".to_string());
    }
    if ty.starts_with("Array<") {
        return Some("[]".to_string());
    }
    if ty.starts_with("Dictionary<") {
        return Some("[:]".to_string());
    }
    if ty.starts_with("Set<") {
        return Some(format!("{ty}()"));
    }
    if is_integer_type(ty) {
        return Some("0".to_string());
    }
    match ty {
        "Float" | "Double" | "CGFloat" => Some("0".to_string()),
        "String" => Some("\"\"".to_string()),
        "Bool" => Some("false".to_string()),
        "Data" => Some("Data()".to_string()),
        // A named struct, class, enum or protocol existential has no zero this crate can spell.
        _ => None,
    }
}

pub fn statement_texts(body: &FunctionBody<'_>) -> Vec<String> {
    let mut out = Vec::new();
    // `function_body` wraps a single `statements` node; the statements are its children.
    for stmts in all_children(body.body) {
        if stmts.kind() != "statements" {
            continue;
        }
        for child in all_children(stmts) {
            if !child.is_named() || matches!(child.kind(), "comment" | "multiline_comment") {
                continue;
            }
            out.push(text(child, body.src).to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_bounded;

    fn parse(src: &str) -> tree_sitter::Tree {
        parse_bounded(&Swift.grammar(), src).expect("parses").tree
    }

    #[test]
    fn a_protocol_requirement_has_no_body_and_a_default_implementation_does() {
        let src = "protocol P {\n    func sig() -> Int\n}\n\
                   extension P {\n    func sig() -> Int { return 3 }\n}\n";
        let tree = parse(src);
        let bodies = Swift.function_bodies(&tree, src);
        assert_eq!(
            bodies.len(),
            1,
            "the requirement is skipped, the default body is taken"
        );
        assert_eq!(bodies[0].name, "sig");
    }

    #[test]
    fn the_return_type_is_the_node_after_the_arrow_not_the_first_name_field() {
        // Every one of `f`, `Int` and `[UInt8]` arrives under the field name `name`.
        let src = "func f(a: Int) throws -> [UInt8] {\n    return []\n}\n";
        let tree = parse(src);
        let bodies = Swift.function_bodies(&tree, src);
        let b = bodies.first().expect("one body");
        assert_eq!(b.name, "f");
        assert_eq!(b.return_type.as_deref(), Some("[UInt8]"));
        assert_eq!(b.params, vec![("a".to_string(), Some("Int".to_string()))]);
    }

    #[test]
    fn no_arrow_means_void_which_is_an_answer_not_a_failure() {
        let src = "func g() {\n    doIt()\n}\n";
        let tree = parse(src);
        let bodies = Swift.function_bodies(&tree, src);
        assert_eq!(
            bodies.first().and_then(|b| b.return_type.as_deref()),
            Some("Void")
        );
        assert_eq!(Swift.default_return(Some("Void")), None, "Void has no zero");
    }

    #[test]
    fn the_else_range_runs_from_the_keyword_to_the_end_of_the_if() {
        let src = "func f(a: Int) -> Int {\n    if a < 3 { return 1 } else { return 2 }\n}\n";
        let tree = parse(src);
        let bodies = Swift.function_bodies(&tree, src);
        let sites = Swift.else_branches(&bodies[0]);
        assert_eq!(sites.len(), 1);
        let (s, e) = sites[0].range;
        assert_eq!(&src[s..e], "else { return 2 }");
    }

    #[test]
    fn zero_values_resolve_or_decline() {
        assert_eq!(zero_value("Int").as_deref(), Some("0"));
        assert_eq!(zero_value("[UInt8]").as_deref(), Some("[]"));
        assert_eq!(zero_value("[String: Int]").as_deref(), Some("[:]"));
        assert_eq!(zero_value("String?").as_deref(), Some("nil"));
        assert_eq!(zero_value("Void"), None);
        assert_eq!(zero_value("MyWidget"), None, "declines rather than guesses");
    }

    #[test]
    fn a_try_becomes_a_swallowing_try_question_mark() {
        let src = "func f() throws -> Int {\n    let v = try g()\n    return v\n}\n";
        let tree = parse(src);
        let bodies = Swift.function_bodies(&tree, src);
        let sites = Swift.error_propagations(&bodies[0]);
        assert_eq!(sites.len(), 1);
        let (s, e) = sites[0].range;
        assert_eq!(&src[s..e], "try");
        assert_eq!(sites[0].replacement, "try?");
    }
}
