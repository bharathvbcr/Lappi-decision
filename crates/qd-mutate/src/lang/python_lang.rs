//! Python. Node kinds verified against `tree-sitter-python` 0.23.6 with `examples/probe.rs`.
//!
//! Python has no braces, so `interior` is the body block itself rather than what sits between
//! delimiters, and every replacement has to carry the body's indentation. `cosmetic.wrap_line` is
//! refused here on principle: the language is whitespace-sensitive and a wrap that happens to be
//! safe today is not a proof.

use tree_sitter::{Node, Tree};

use super::util::{self, all_children, named_children, text, walk_body, walk_tree};
use super::{
    Binding, CallSite, ComparisonSite, ElseSite, ErrorSite, Formatter, FunctionBody, ImportBlock,
    LangId, Language, LiteralKind, SyncSite,
};

pub struct Python;

const PANIC_STUBS: &[&str] = &["raise NotImplementedError", "pass"];

fn is_boundary(kind: &str) -> bool {
    matches!(kind, "function_definition" | "lambda")
}

impl Language for Python {
    fn id(&self) -> LangId {
        LangId::Python
    }

    fn grammar(&self) -> tree_sitter::Language {
        tree_sitter_python::LANGUAGE.into()
    }

    fn function_bodies<'t>(&self, tree: &'t Tree, src: &'t str) -> Vec<FunctionBody<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            if node.kind() != "function_definition" {
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
                            "identifier" => Some((text(param, src).to_string(), None)),
                            "typed_parameter" | "typed_default_parameter" => {
                                let n = named_children(param).first().copied()?;
                                let t = param.child_by_field_name("type");
                                Some((
                                    text(n, src).to_string(),
                                    t.map(|t| text(t, src).to_string()),
                                ))
                            }
                            "default_parameter" => {
                                let n = param.child_by_field_name("name")?;
                                Some((text(n, src).to_string(), None))
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            // Python is gradually typed: an absent annotation is **unknown**, not `None`. The
            // facade reports `None` and `stub.default_return` declines, which is the spec's
            // "where it cannot be resolved, the operator declines rather than guessing".
            let return_type = node
                .child_by_field_name("return_type")
                .map(|n| text(n, src).trim().to_string());
            out.push(FunctionBody {
                decl: node,
                body,
                interior: util::interior_lines(body, src),
                name,
                arity: params.len(),
                return_type,
                params,
                is_nested: util::has_function_ancestor(node, &is_boundary),
                node_kind: "function_definition",
                src,
            });
        });
        out.sort_by_key(|f| f.decl.start_byte());
        out
    }

    fn conditions<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
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
            if node.kind() != "comparison_operator" {
                return;
            }
            // The operator tokens are anonymous children of the comparison, not a field.
            for child in all_children(node) {
                if child.is_named() {
                    continue;
                }
                let s = text(child, body.src);
                if matches!(s, "<" | ">" | "<=" | ">=") {
                    out.push(ComparisonSite {
                        op_node: child,
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
            if node.kind() != "call" {
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
            // `else_clause` spans `else :` and the block. A `for`/`while`/`try` else is a different
            // construct with different semantics, so only an `if`'s else is offered.
            if node.kind() != "else_clause" {
                return;
            }
            if node.parent().map(|p| p.kind()) != Some("if_statement") {
                return;
            }
            out.push(ElseSite {
                range: (node.start_byte(), node.end_byte()),
                node,
            });
        });
        out
    }

    fn error_propagations<'t>(&self, body: &FunctionBody<'t>) -> Vec<ErrorSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // `except ...: raise` — swallowing means the re-raise goes away.
            if node.kind() != "except_clause" {
                return;
            }
            let Some(block) = named_children(node)
                .into_iter()
                .find(|c| c.kind() == "block")
            else {
                return;
            };
            let stmts = named_children(block);
            if stmts.len() != 1 {
                return;
            }
            let only = stmts[0];
            let is_bare_raise = only.kind() == "raise_statement"
                && text(only, body.src).trim() == "raise";
            if !is_bare_raise {
                return;
            }
            out.push(ErrorSite {
                range: (only.start_byte(), only.end_byte()),
                replacement: "pass".to_string(),
                whole_statement: false,
                node: only,
            });
        });
        out
    }

    fn awaits_and_locks<'t>(&self, body: &FunctionBody<'t>) -> Vec<SyncSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "await" {
                return;
            }
            // `await expr` — remove the keyword and the space after it.
            if let Some(inner) = named_children(node).first() {
                out.push(SyncSite {
                    range: (node.start_byte(), inner.start_byte()),
                    whole_statement: false,
                    node,
                });
            }
        });
        out
    }

    fn integer_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() == "integer" {
                out.push(node);
            }
        });
        out
    }

    fn bound_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // `range(n)` / `range(a, n)` — the last argument is the bound.
            if node.kind() == "call"
                && let Some(func) = node.child_by_field_name("function")
                && text(func, body.src) == "range"
                && let Some(args) = node.child_by_field_name("arguments")
                && let Some(last) = named_children(args).last()
                && last.kind() == "integer"
            {
                out.push(*last);
            }
            if node.kind() == "comparison_operator" {
                let named = named_children(node);
                if named.len() == 2 && named[1].kind() == "integer" {
                    let has_relational = all_children(node).iter().any(|c| {
                        !c.is_named() && matches!(text(*c, body.src), "<" | "<=" | ">" | ">=")
                    });
                    if has_relational {
                        out.push(named[1]);
                    }
                }
            }
        });
        out
    }

    fn string_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() == "string" {
                out.push(node);
            }
        });
        out
    }

    fn local_bindings<'t>(&self, body: &FunctionBody<'t>) -> Vec<Binding<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "assignment" {
                return;
            }
            if let Some(left) = node.child_by_field_name("left")
                && left.kind() == "identifier"
            {
                out.push(Binding {
                    name_node: left,
                    name: text(left, body.src).to_string(),
                });
            }
        });
        out
    }

    fn comments<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() == "comment" {
                out.push(node);
            }
        });
        out
    }

    fn docstring<'t>(&self, body: &FunctionBody<'t>) -> Option<Node<'t>> {
        // The first statement of the body, when it is a bare string.
        let first = named_children(body.body).into_iter().next()?;
        if first.kind() != "expression_statement" {
            return None;
        }
        let s = named_children(first).into_iter().next()?;
        if s.kind() != "string" {
            return None;
        }
        // A doctest is executable. Rewriting it is a behaviour change wearing a `cosmetic` label.
        if text(s, body.src).contains(">>>") {
            return None;
        }
        named_children(s)
            .into_iter()
            .find(|c| c.kind() == "string_content")
    }

    fn import_block<'t>(&self, tree: &'t Tree, _src: &'t str) -> Option<ImportBlock<'t>> {
        let mut items: Vec<Node<'t>> = Vec::new();
        let mut cursor = tree.root_node().walk();
        for child in tree.root_node().named_children(&mut cursor) {
            if matches!(child.kind(), "import_statement" | "import_from_statement") {
                items.push(child);
            } else if !items.is_empty() {
                break;
            }
        }
        if items.len() < 2 {
            return None;
        }
        // Import order in Python is execution order: a module can rely on a prior import having
        // run (the circular-import workaround), and no static check here can rule that out. The
        // operator is refused for the whole language and the restriction is recorded.
        Some(ImportBlock {
            items,
            reorderable: false,
            why_not: Some("Python imports execute in order; a circular-import workaround depends on it"),
        })
    }

    fn identifiers<'t>(&self, tree: &'t Tree) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            if node.kind() == "identifier" {
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
        let raw = return_type?.trim().trim_start_matches("->").trim();
        let ok = match kind {
            LiteralKind::Integer => raw == "int",
            LiteralKind::Str => raw == "str",
        };
        ok.then(|| format!("return {literal}"))
    }

    fn negate(&self, condition: &str) -> String {
        format!("not ({condition})")
    }

    fn is_negated(&self, node: &Node<'_>, src: &str) -> bool {
        node.kind() == "not_operator" || text(*node, src).trim_start().starts_with("not ")
    }

    fn line_comment_prefix(&self) -> &'static str {
        "#"
    }

    fn formatter(&self) -> Option<Formatter> {
        Some(Formatter {
            program: "black",
            args: &["-q", "-"],
            via_xcrun: false,
        })
    }

    fn line_wrapping_is_safe(&self) -> bool {
        // Whitespace-sensitive. The spec refuses `cosmetic.wrap_line` here by name.
        false
    }
}

fn zero_value(ty: &str) -> Option<String> {
    let ty = ty.trim();
    if ty.starts_with("List[") || ty.starts_with("list[") || ty == "list" {
        return Some("[]".to_string());
    }
    if ty.starts_with("Dict[") || ty.starts_with("dict[") || ty == "dict" {
        return Some("{}".to_string());
    }
    if ty.starts_with("Optional[") {
        return Some("None".to_string());
    }
    match ty {
        "None" => Some("None".to_string()),
        "int" => Some("0".to_string()),
        "float" => Some("0.0".to_string()),
        "str" => Some("\"\"".to_string()),
        "bool" => Some("False".to_string()),
        "bytes" => Some("b\"\"".to_string()),
        _ => None,
    }
}

pub fn statement_texts(body: &FunctionBody<'_>) -> Vec<String> {
    let mut out = Vec::new();
    for child in named_children(body.body) {
        if child.kind() == "comment" {
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
    fn an_unannotated_return_type_is_unknown_not_none() {
        assert_eq!(Python.default_return(None), None);
        assert_eq!(Python.default_return(Some("int")).as_deref(), Some("return 0"));
        assert_eq!(
            Python.default_return(Some("List[int]")).as_deref(),
            Some("return []")
        );
        assert_eq!(Python.default_return(Some("Widget")), None);
    }
}
