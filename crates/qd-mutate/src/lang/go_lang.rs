//! Go. Node kinds verified against `tree-sitter-go` 0.23.4 with `examples/probe.rs`.
//!
//! `function_declaration`/`method_declaration` carry a `body`; `method_elem` inside an
//! `interface_type` does not, which excludes interface members without naming them.

use tree_sitter::{Node, Tree};

use super::util::{self, all_children, named_children, text, walk_body, walk_tree};
use super::{
    Binding, CallSite, ComparisonSite, ElseSite, ErrorSite, Formatter, FunctionBody, ImportBlock,
    LangId, Language, LiteralKind, SyncSite,
};

pub struct Go;

const PANIC_STUBS: &[&str] = &["panic(\"not implemented\")", "panic(\"TODO\")"];

fn is_boundary(kind: &str) -> bool {
    matches!(
        kind,
        "function_declaration" | "method_declaration" | "func_literal"
    )
}

impl Language for Go {
    fn id(&self) -> LangId {
        LangId::Go
    }

    fn grammar(&self) -> tree_sitter::Language {
        tree_sitter_go::LANGUAGE.into()
    }

    fn function_bodies<'t>(&self, tree: &'t Tree, src: &'t str) -> Vec<FunctionBody<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            let kind = node.kind();
            if !matches!(kind, "function_declaration" | "method_declaration") {
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
                        .filter(|c| c.kind() == "parameter_declaration")
                        .filter_map(|param| {
                            let n = param.child_by_field_name("name")?;
                            let t = param.child_by_field_name("type");
                            Some((
                                text(n, src).to_string(),
                                t.map(|t| text(t, src).to_string()),
                            ))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            // Go's `result` is absent for a function returning nothing, which is the language's own
            // answer rather than a failure to resolve.
            let return_type = Some(
                node.child_by_field_name("result")
                    .map(|n| text(n, src).trim().to_string())
                    .unwrap_or_default(),
            );
            let node_kind: &'static str = if kind == "method_declaration" {
                "method_declaration"
            } else {
                "function_declaration"
            };
            out.push(FunctionBody {
                decl: node,
                body,
                interior: util::interior_lines(body, src),
                name,
                arity: params.len(),
                return_type,
                params,
                is_nested: util::has_function_ancestor(node, &is_boundary),
                node_kind,
                src,
            });
        });
        out.sort_by_key(|f| f.decl.start_byte());
        out
    }

    fn conditions<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() == "if_statement"
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
            if node.kind() != "if_statement" {
                return;
            }
            let Some(alt) = node.child_by_field_name("alternative") else {
                return;
            };
            // Go's `alternative` is the block **after** the `else` keyword, which is a separate
            // preceding token. Removing only the block leaves a dangling `} else` — a mutation
            // that does not parse, carrying a confident `logic.drop_else` label. Verified against
            // tree-sitter-go 0.23.4: `if_statement` children are `if`, condition, consequence,
            // `else`, alternative.
            let start = match alt.prev_sibling() {
                Some(kw) if kw.kind() == "else" => kw.start_byte(),
                // No `else` token in front of the alternative is a grammar shape this code has not
                // seen. Skipping is the answer that cannot emit a broken diff.
                _ => return,
            };
            out.push(ElseSite {
                range: (start, alt.end_byte()),
                node: alt,
            });
        });
        out
    }

    fn error_propagations<'t>(&self, body: &FunctionBody<'t>) -> Vec<ErrorSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // `if err != nil { return ..., err }` — deleting it is the realistic Go agent failure.
            if node.kind() != "if_statement" {
                return;
            }
            let Some(cond) = node.child_by_field_name("condition") else {
                return;
            };
            if cond.kind() != "binary_expression" {
                return;
            }
            let op_is_ne = cond
                .child_by_field_name("operator")
                .map(|o| text(o, body.src) == "!=")
                .unwrap_or(false);
            let rhs_is_nil = cond
                .child_by_field_name("right")
                .map(|r| text(r, body.src) == "nil")
                .unwrap_or(false);
            if !(op_is_ne && rhs_is_nil) {
                return;
            }
            // Only where the consequence returns: an `if err != nil` that logs and continues is
            // not an error propagation, and deleting it changes behaviour in a way `swallow_error`
            // does not describe.
            let Some(cons) = node.child_by_field_name("consequence") else {
                return;
            };
            let mut returns = false;
            walk_body(cons, &is_boundary, |n| {
                if n.kind() == "return_statement" {
                    returns = true;
                }
            });
            if !returns {
                return;
            }
            // An `else` attached to the check would be orphaned by the deletion.
            if node.child_by_field_name("alternative").is_some() {
                return;
            }
            out.push(ErrorSite {
                range: (node.start_byte(), node.end_byte()),
                replacement: String::new(),
                whole_statement: true,
                node,
            });
        });
        out
    }

    fn awaits_and_locks<'t>(&self, body: &FunctionBody<'t>) -> Vec<SyncSite<'t>> {
        // Go has no `await`. The lock acquisition is the whole of this operator here.
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "expression_statement" {
                return;
            }
            let Some(call) = named_children(node).first().copied() else {
                return;
            };
            if call.kind() != "call_expression" {
                return;
            }
            let Some(func) = call.child_by_field_name("function") else {
                return;
            };
            if func.kind() != "selector_expression" {
                return;
            }
            let field = func
                .child_by_field_name("field")
                .map(|f| text(f, body.src))
                .unwrap_or("");
            if matches!(field, "Lock" | "RLock") {
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
            if node.kind() == "int_literal" {
                out.push(node);
            }
        });
        out
    }

    fn bound_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() == "binary_expression"
                && let Some(op) = node.child_by_field_name("operator")
                && matches!(text(op, body.src), "<" | "<=" | ">" | ">=")
                && let Some(rhs) = node.child_by_field_name("right")
                && rhs.kind() == "int_literal"
            {
                out.push(rhs);
            }
        });
        out
    }

    fn string_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if matches!(node.kind(), "interpreted_string_literal" | "raw_string_literal") {
                out.push(node);
            }
        });
        out
    }

    fn local_bindings<'t>(&self, body: &FunctionBody<'t>) -> Vec<Binding<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "short_var_declaration" {
                return;
            }
            let Some(left) = node.child_by_field_name("left") else {
                return;
            };
            let names = named_children(left);
            // Only a single-name binding: renaming one of `v, err := f()` is safe too, but a
            // multi-name form makes the "bound exactly once in this body" proof harder to state,
            // and a rename that is not provably safe is a poisoned cosmetic label.
            if names.len() == 1 && names[0].kind() == "identifier" {
                out.push(Binding {
                    name_node: names[0],
                    name: text(names[0], body.src).to_string(),
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

    fn docstring<'t>(&self, _body: &FunctionBody<'t>) -> Option<Node<'t>> {
        None
    }

    fn import_block<'t>(&self, tree: &'t Tree, src: &'t str) -> Option<ImportBlock<'t>> {
        let mut spec_list: Option<Node<'t>> = None;
        walk_tree(tree, |node| {
            if node.kind() == "import_spec_list" && spec_list.is_none() {
                spec_list = Some(node);
            }
        });
        let list = spec_list?;
        let items: Vec<Node<'t>> = named_children(list)
            .into_iter()
            .filter(|n| n.kind() == "import_spec")
            .collect();
        if items.len() < 2 {
            return None;
        }
        // A blank import (`_ "net/http/pprof"`) is there for its side effect, and side effects run
        // in import order. `gofmt` would re-sort the block anyway, but the refusal is stated here
        // rather than delegated to a tool that may not be installed.
        let has_blank = items.iter().any(|n| {
            named_children(*n)
                .iter()
                .any(|c| c.kind() == "blank_identifier" || text(*c, src) == "_")
        });
        Some(ImportBlock {
            items,
            reorderable: !has_blank,
            why_not: has_blank.then_some("a blank import runs for its side effect, and side effects are ordered"),
        })
    }

    fn identifiers<'t>(&self, tree: &'t Tree) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            if matches!(
                node.kind(),
                "identifier" | "field_identifier" | "type_identifier" | "package_identifier"
            ) {
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
        if raw.is_empty() {
            // No results: `return` alone is a legitimate stub, but it is indistinguishable from
            // deleting the body. Decline, as Rust does for `()`.
            return None;
        }
        let zeros = result_zeros(raw)?;
        Some(format!("return {}", zeros.join(", ")))
    }

    fn early_return(&self, return_type: Option<&str>) -> Option<String> {
        // Go's `return a, b` is already a statement; the two renderings coincide.
        self.default_return(return_type)
    }

    fn hardcoded_return(
        &self,
        return_type: Option<&str>,
        literal: &str,
        kind: LiteralKind,
    ) -> Option<String> {
        let raw = return_type?.trim();
        let types = split_results(raw);
        let first = types.first()?.trim();
        let ok = match kind {
            LiteralKind::Integer => is_integer_type(first),
            LiteralKind::Str => first == "string",
        };
        if !ok {
            return None;
        }
        let mut values = vec![literal.to_string()];
        for t in types.iter().skip(1) {
            values.push(zero_value(t.trim())?);
        }
        Some(format!("return {}", values.join(", ")))
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
            program: "gofmt",
            args: &[],
            via_xcrun: false,
        })
    }

    fn smoke_source(&self) -> &'static str {
        "package p\n"
    }

    fn line_wrapping_is_safe(&self) -> bool {
        // Go inserts semicolons at line ends, so a wrap in the wrong place changes the program.
        // The operator is restricted here rather than trusted to pick a safe point.
        false
    }

    fn import_order_is_semantic(&self) -> Option<&'static str> {
        // Not for the language. The Go spec orders the initialization of an imported package
        // before its importer, but leaves the order *among* independent imports unspecified, so no
        // correct program may depend on it. The one shape that is deliberately ordered — a blank
        // import taken for its side effect — is a per-file fact and `import_block` refuses it by
        // name.
        None
    }
}

/// Split a Go result list into its component types.
fn split_results(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    let inner = trimmed
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(trimmed);
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in inner.chars() {
        match c {
            '[' | '(' | '{' => {
                depth += 1;
                current.push(c);
            }
            ']' | ')' | '}' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                out.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

fn result_zeros(raw: &str) -> Option<Vec<String>> {
    let types = split_results(raw);
    if types.is_empty() {
        return None;
    }
    types.iter().map(|t| zero_value(t.trim())).collect()
}

fn is_integer_type(ty: &str) -> bool {
    matches!(
        ty,
        "int" | "int8"
            | "int16"
            | "int32"
            | "int64"
            | "uint"
            | "uint8"
            | "uint16"
            | "uint32"
            | "uint64"
            | "uintptr"
            | "byte"
            | "rune"
    )
}

fn zero_value(ty: &str) -> Option<String> {
    let ty = ty.trim();
    // A named result (`n int`) carries the name first; take the type.
    let ty = ty.rsplit(' ').next().unwrap_or(ty).trim();
    if ty.starts_with("[]")
        || ty.starts_with("map[")
        || ty.starts_with('*')
        || ty.starts_with("chan ")
        || ty.starts_with("func(")
    {
        return Some("nil".to_string());
    }
    if is_integer_type(ty) {
        return Some("0".to_string());
    }
    match ty {
        "error" | "any" | "interface{}" => Some("nil".to_string()),
        "string" => Some("\"\"".to_string()),
        "bool" => Some("false".to_string()),
        "float32" | "float64" => Some("0".to_string()),
        // A named struct type's zero is a composite literal, but writing `T{}` for a type this
        // crate cannot confirm is a struct would not compile. Decline.
        _ => None,
    }
}

pub fn statement_texts(body: &FunctionBody<'_>) -> Vec<String> {
    let mut out = Vec::new();
    for child in all_children(body.body) {
        if !child.is_named() || child.kind() == "comment" {
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
    fn a_result_list_splits_on_top_level_commas_only() {
        assert_eq!(
            split_results("([]byte, error)"),
            vec!["[]byte".to_string(), "error".to_string()]
        );
        assert_eq!(
            split_results("(map[string]int, error)"),
            vec!["map[string]int".to_string(), "error".to_string()]
        );
        assert_eq!(split_results("int"), vec!["int".to_string()]);
    }

    #[test]
    fn zero_values_resolve_or_decline() {
        assert_eq!(zero_value("[]byte").as_deref(), Some("nil"));
        assert_eq!(zero_value("error").as_deref(), Some("nil"));
        assert_eq!(zero_value("string").as_deref(), Some("\"\""));
        assert_eq!(zero_value("MyStruct"), None, "declines rather than guesses");
    }
}
