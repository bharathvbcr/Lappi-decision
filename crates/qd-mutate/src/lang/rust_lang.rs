//! Rust. Node kinds verified against `tree-sitter-rust` 0.23.3 with `examples/probe.rs`.
//!
//! `function_item` carries a `body`; `function_signature_item` (a trait signature) does not, which
//! is how trait signatures are excluded without naming them.

use tree_sitter::{Node, Tree};

use super::util::{self, all_children, named_children, text, walk_body, walk_tree};
use super::{
    Binding, CallSite, ComparisonSite, ElseSite, ErrorSite, Formatter, FunctionBody, ImportBlock,
    LangId, Language, LiteralKind, SyncSite,
};

pub struct Rust;

const PANIC_STUBS: &[&str] = &[
    "todo!()",
    "unimplemented!()",
    "panic!(\"not implemented\")",
];

fn is_boundary(kind: &str) -> bool {
    matches!(kind, "function_item" | "closure_expression")
}

impl Language for Rust {
    fn id(&self) -> LangId {
        LangId::Rust
    }

    fn grammar(&self) -> tree_sitter::Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn function_bodies<'t>(&self, tree: &'t Tree, src: &'t str) -> Vec<FunctionBody<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            if node.kind() != "function_item" {
                return;
            }
            let Some(body) = node.child_by_field_name("body") else {
                return;
            };
            let name = node
                .child_by_field_name("name")
                .map(|n| text(n, src).to_string())
                .unwrap_or_default();
            if name.is_empty() {
                return;
            }
            let params = node
                .child_by_field_name("parameters")
                .map(|p| {
                    named_children(p)
                        .into_iter()
                        .filter_map(|param| match param.kind() {
                            "parameter" => {
                                let n = param.child_by_field_name("pattern")?;
                                let t = param.child_by_field_name("type");
                                Some((
                                    text(n, src).to_string(),
                                    t.map(|t| text(t, src).to_string()),
                                ))
                            }
                            "self_parameter" => Some(("self".to_string(), None)),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let return_type = node
                .child_by_field_name("return_type")
                .map(|n| text(n, src).trim().to_string())
                // No `->` in Rust means `()`, which the compiler says, not a guess.
                .or_else(|| Some("()".to_string()));
            out.push(FunctionBody {
                decl: node,
                body,
                interior: util::interior_lines(body, src),
                name,
                arity: params.len(),
                return_type,
                params,
                is_nested: util::has_function_ancestor(node, &is_boundary),
                node_kind: "function_item",
                src,
            });
        });
        out.sort_by_key(|f| f.decl.start_byte());
        out
    }

    fn conditions<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if matches!(node.kind(), "if_expression" | "while_expression")
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
            if node.kind() != "binary_expression" {
                return;
            }
            if let Some(op) = node.child_by_field_name("operator") {
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
            if let Some(args) = node.child_by_field_name("arguments") {
                out.push(CallSite {
                    call: node,
                    args: named_children(args),
                });
            }
        });
        out
    }

    fn else_branches<'t>(&self, body: &FunctionBody<'t>) -> Vec<ElseSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // `else_clause` spans the keyword and the block together, so the node's own range is
            // the removable one.
            if node.kind() == "else_clause" {
                out.push(ElseSite {
                    range: (node.start_byte(), node.end_byte()),
                    node,
                });
            }
        });
        out
    }

    fn error_propagations<'t>(&self, body: &FunctionBody<'t>) -> Vec<ErrorSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // `expr?` — the realistic agent failure is turning it into a default.
            if node.kind() == "try_expression" {
                let end = node.end_byte();
                if end > node.start_byte() && body.src.as_bytes().get(end - 1) == Some(&b'?') {
                    out.push(ErrorSite {
                        range: (end - 1, end),
                        replacement: ".unwrap_or_default()".to_string(),
                        whole_statement: false,
                        node,
                    });
                }
            }
        });
        out
    }

    fn awaits_and_locks<'t>(&self, body: &FunctionBody<'t>) -> Vec<SyncSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() == "await_expression" {
                // `expr.await` — drop the `.await` suffix; the inner expression still parses.
                if let Some(inner) = named_children(node).first() {
                    out.push(SyncSite {
                        range: (inner.end_byte(), node.end_byte()),
                        whole_statement: false,
                        node,
                    });
                }
            }
            // A lock acquisition bound to a guard: `let g = m.lock().unwrap();`. Removing the whole
            // binding is the mutation that drops the lock and still parses, provided the guard is
            // not used afterwards — which the caller checks by rename-style occurrence counting.
            if node.kind() == "let_declaration"
                && let Some(value) = node.child_by_field_name("value")
                && text(value, body.src).contains(".lock()")
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
            // `a..b` / `a..=b`: the end is a bound. So is the right operand of a `<`/`<=` in a
            // `while` condition.
            if node.kind() == "range_expression"
                && let Some(end) = named_children(node).last()
                && end.kind() == "integer_literal"
            {
                out.push(*end);
            }
            if node.kind() == "binary_expression"
                && let Some(op) = node.child_by_field_name("operator")
                && matches!(text(op, body.src), "<" | "<=" | ">" | ">=")
                && let Some(rhs) = node.child_by_field_name("right")
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
            if node.kind() == "string_literal" {
                out.push(node);
            }
        });
        out
    }

    fn local_bindings<'t>(&self, body: &FunctionBody<'t>) -> Vec<Binding<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "let_declaration" {
                return;
            }
            if let Some(pattern) = node.child_by_field_name("pattern")
                && pattern.kind() == "identifier"
            {
                out.push(Binding {
                    name_node: pattern,
                    name: text(pattern, body.src).to_string(),
                });
            }
        });
        out
    }

    fn comments<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if matches!(node.kind(), "line_comment" | "block_comment") {
                out.push(node);
            }
        });
        out
    }

    fn docstring<'t>(&self, _body: &FunctionBody<'t>) -> Option<Node<'t>> {
        // Rust doc comments live outside the body and are already `line_comment` nodes.
        None
    }

    fn import_block<'t>(&self, tree: &'t Tree, src: &'t str) -> Option<ImportBlock<'t>> {
        let mut items: Vec<Node<'t>> = Vec::new();
        let mut cursor = tree.root_node().walk();
        for child in tree.root_node().named_children(&mut cursor) {
            if child.kind() == "use_declaration" {
                items.push(child);
            } else if !items.is_empty() {
                break; // one contiguous group only
            }
        }
        if items.len() < 2 {
            return None;
        }
        // A `use` with a side effect does not exist in Rust; the only hazard is a glob shadowing
        // a later name, which reordering within a group cannot introduce.
        let has_glob = items
            .iter()
            .any(|n| text(*n, src).contains("::*"));
        Some(ImportBlock {
            items,
            reorderable: !has_glob,
            why_not: has_glob.then_some("a glob import makes name resolution order-sensitive"),
        })
    }

    fn identifiers<'t>(&self, tree: &'t Tree) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            if matches!(node.kind(), "identifier" | "field_identifier" | "type_identifier") {
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
        let raw = return_type?.trim().trim_start_matches("->").trim();
        zero_value(raw)
    }

    fn early_return(&self, return_type: Option<&str>) -> Option<String> {
        let raw = return_type?.trim().trim_start_matches("->").trim();
        zero_value(raw).map(|z| format!("return {z};"))
    }

    fn hardcoded_return(
        &self,
        return_type: Option<&str>,
        literal: &str,
        kind: LiteralKind,
    ) -> Option<String> {
        let raw = return_type?.trim().trim_start_matches("->").trim();
        let inner = result_inner(raw).unwrap_or(raw);
        let matches_kind = match kind {
            LiteralKind::Integer => is_integer_type(inner),
            LiteralKind::Str => inner == "String" || inner == "&str",
        };
        if !matches_kind {
            return None;
        }
        let value = match (kind, inner) {
            (LiteralKind::Str, "String") => format!("{literal}.to_string()"),
            _ => literal.to_string(),
        };
        Some(if raw.starts_with("Result<") {
            format!("Ok({value})")
        } else {
            value
        })
    }

    fn negate(&self, condition: &str) -> String {
        format!("!({condition})")
    }

    fn is_negated(&self, node: &Node<'_>, src: &str) -> bool {
        node.kind() == "unary_expression" && text(*node, src).starts_with('!')
    }

    fn line_comment_prefix(&self) -> &'static str {
        "//"
    }

    fn formatter(&self) -> Option<Formatter> {
        Some(Formatter {
            program: "rustfmt",
            args: &["--emit", "stdout", "--edition", "2021"],
            via_xcrun: false,
        })
    }

    fn smoke_source(&self) -> &'static str {
        "fn qd_mutate_smoke() {}\n"
    }

    fn line_wrapping_is_safe(&self) -> bool {
        true
    }

    fn import_order_is_semantic(&self) -> Option<&'static str> {
        // Not for the language: `use` binds a name and runs nothing. The one order-sensitive shape
        // is a glob, which is a per-file fact and `import_block` refuses it.
        None
    }
}

fn result_inner(ty: &str) -> Option<&str> {
    let rest = ty.strip_prefix("Result<")?.strip_suffix('>')?;
    // `Result<T, E>` — take T, respecting one level of nesting so `Result<Vec<u8>, E>` works.
    let mut depth = 0i32;
    for (i, c) in rest.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => return Some(rest[..i].trim()),
            _ => {}
        }
    }
    Some(rest.trim())
}

fn is_integer_type(ty: &str) -> bool {
    matches!(
        ty,
        "u8" | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
    )
}

/// The zero value for a Rust type, or `None` where it cannot be resolved.
///
/// Declining is the point. A guessed zero compiles into a diff whose `stub.default_return` label is
/// a lie about what the code now does.
fn zero_value(ty: &str) -> Option<String> {
    let ty = ty.trim();
    if ty.is_empty() || ty == "()" {
        // A unit return has no interesting zero: the mutation would be "delete the body", which is
        // `stub.panic`'s shape with a different label. Decline rather than blur the two classes.
        return None;
    }
    if let Some(inner) = result_inner(ty) {
        let inner_zero = if inner.is_empty() || inner == "()" {
            "()".to_string()
        } else {
            zero_value(inner)?
        };
        return Some(format!("Ok({inner_zero})"));
    }
    if ty.starts_with("Option<") {
        return Some("None".to_string());
    }
    if ty.starts_with("Vec<") {
        return Some("Vec::new()".to_string());
    }
    if ty.starts_with("HashMap<") || ty.starts_with("BTreeMap<") {
        return Some("Default::default()".to_string());
    }
    if is_integer_type(ty) {
        return Some("0".to_string());
    }
    match ty {
        "bool" => Some("false".to_string()),
        "f32" | "f64" => Some("0.0".to_string()),
        "String" => Some("String::new()".to_string()),
        "&str" => Some("\"\"".to_string()),
        "char" => Some("'\\0'".to_string()),
        _ => None,
    }
}

/// Statement texts of a body, for the already-a-stub check.
pub fn statement_texts(body: &FunctionBody<'_>) -> Vec<String> {
    let mut out = Vec::new();
    for child in all_children(body.body) {
        if !child.is_named() {
            continue;
        }
        if matches!(child.kind(), "line_comment" | "block_comment") {
            continue;
        }
        out.push(text(child, body.src).to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_values_resolve_or_decline() {
        assert_eq!(zero_value("Result<(), E>").as_deref(), Some("Ok(())"));
        assert_eq!(
            zero_value("Result<Vec<u8>, E>").as_deref(),
            Some("Ok(Vec::new())")
        );
        assert_eq!(zero_value("Option<T>").as_deref(), Some("None"));
        assert_eq!(zero_value("usize").as_deref(), Some("0"));
        assert_eq!(zero_value("()"), None, "unit is vacuous, not a zero");
        assert_eq!(zero_value("MyOpaqueThing"), None, "declines rather than guesses");
    }

    #[test]
    fn result_inner_respects_nesting() {
        assert_eq!(result_inner("Result<Vec<u8>, Error>"), Some("Vec<u8>"));
        assert_eq!(result_inner("Result<(), Error>"), Some("()"));
    }
}
