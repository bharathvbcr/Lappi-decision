//! TypeScript. Node kinds verified against `tree-sitter-typescript` 0.23.2 with `examples/probe.rs`.
//!
//! `function_declaration` and `method_definition` carry a `body`; `method_signature` (an interface
//! member) and `abstract_method_signature` (an abstract class member) do not, which is how both are
//! excluded without naming them.
//!
//! Two shapes this grammar gets right and a brace-counting heuristic gets wrong, both in the
//! adversarial corpus: an `arrow_function` whose body is an expression rather than a
//! `statement_block` has no block to stub, and a `type_annotation` node's text carries its own
//! leading `:` — reading it as a bare type name would produce `: number` where `number` was meant.
//!
//! **No formatter is resolvable on this machine.** `prettier` is declared because it is the
//! language's formatter, and [`crate::fmt`] resolves it at run time: when it is absent the
//! formatter-verified cosmetic operators are refused with `Refusal::NoFormatter` and the
//! restriction is recorded per language in the manifest. It is never assumed harmless.

use tree_sitter::{Node, Tree};

use super::util::{self, all_children, named_children, text, walk_body, walk_tree};
use super::{
    Binding, CallSite, ComparisonSite, ElseSite, ErrorSite, Formatter, FunctionBody, ImportBlock,
    LangId, Language, LiteralKind, SyncSite,
};

pub struct TypeScript;

const PANIC_STUBS: &[&str] = &[
    "throw new Error(\"not implemented\")",
    "throw new Error(\"TODO\")",
];

fn is_boundary(kind: &str) -> bool {
    matches!(
        kind,
        "function_declaration"
            | "generator_function_declaration"
            | "function_expression"
            | "generator_function"
            | "arrow_function"
            | "method_definition"
    )
}

/// The declaration kinds that can carry a `body`, in the order the grammar names them.
const DECL_KINDS: &[&str] = &[
    "function_declaration",
    "generator_function_declaration",
    "method_definition",
];

/// `type_annotation` text is `": T"`. Strip the colon; `None` when nothing is left, because an
/// empty type is not a resolved type.
fn annotation_type(node: Node<'_>, src: &str) -> Option<String> {
    let raw = text(node, src).trim();
    let stripped = raw.strip_prefix(':').unwrap_or(raw).trim();
    (!stripped.is_empty()).then(|| stripped.to_string())
}

impl Language for TypeScript {
    fn id(&self) -> LangId {
        LangId::TypeScript
    }

    fn grammar(&self) -> tree_sitter::Language {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    }

    fn function_bodies<'t>(&self, tree: &'t Tree, src: &'t str) -> Vec<FunctionBody<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            let Some(&node_kind) = DECL_KINDS.iter().find(|k| **k == node.kind()) else {
                return;
            };
            let Some(body) = node.child_by_field_name("body") else {
                return;
            };
            // An arrow function or a method can be handed an expression body. There is no block to
            // replace, so there is nothing to stub and no interior to name.
            if body.kind() != "statement_block" {
                return;
            }
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
                            "required_parameter" | "optional_parameter" => {
                                let n = param.child_by_field_name("pattern")?;
                                let t = param
                                    .child_by_field_name("type")
                                    .and_then(|t| annotation_type(t, src));
                                Some((text(n, src).to_string(), t))
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            // TypeScript infers a missing return type; an absent annotation is **unknown** here,
            // not `void`. `stub.default_return` then declines rather than guessing, which is the
            // spec's rule for an unresolvable return type.
            let return_type = node
                .child_by_field_name("return_type")
                .and_then(|n| annotation_type(n, src));
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
            if matches!(node.kind(), "if_statement" | "while_statement")
                && let Some(c) = node.child_by_field_name("condition")
            {
                // The condition is a `parenthesized_expression`; the interesting node is inside it,
                // and negating the inner one keeps the parentheses the grammar requires.
                let inner = named_children(c).into_iter().next();
                out.push(inner.unwrap_or(c));
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
            if let Some(args) = node.child_by_field_name("arguments")
                && args.kind() == "arguments"
            {
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
            // `else_clause` spans the keyword and the block together.
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
            // `catch (e) { throw e; }` — swallowing means the re-throw goes away. This is the
            // `try/catch -> swallowed catch` form the spec names.
            if node.kind() != "catch_clause" {
                return;
            }
            let Some(block) = node.child_by_field_name("body") else {
                return;
            };
            let stmts = named_children(block);
            if stmts.len() != 1 || stmts[0].kind() != "throw_statement" {
                return;
            }
            out.push(ErrorSite {
                range: (stmts[0].start_byte(), stmts[0].end_byte()),
                replacement: String::new(),
                whole_statement: true,
                node: stmts[0],
            });
        });
        out
    }

    fn awaits_and_locks<'t>(&self, body: &FunctionBody<'t>) -> Vec<SyncSite<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "await_expression" {
                return;
            }
            // `await expr` — drop the keyword and the whitespace up to the operand. JavaScript has
            // no lock construct, so `await` is the whole of this operator here.
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
            // The grammar has one `number` kind for integers and floats alike; only the integral
            // ones are offered, so `change_constant` does not turn 1.5 into 2.
            if node.kind() == "number" && text(node, body.src).parse::<i64>().is_ok() {
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
                && rhs.kind() == "number"
                && text(rhs, body.src).parse::<i64>().is_ok()
            {
                out.push(rhs);
            }
            // `xs.slice(0, n)` / `xs.splice(i, n)` — the last argument is a bound.
            if node.kind() == "call_expression"
                && let Some(func) = node.child_by_field_name("function")
                && func.kind() == "member_expression"
                && let Some(prop) = func.child_by_field_name("property")
                && matches!(text(prop, body.src), "slice" | "splice" | "substring")
                && let Some(args) = node.child_by_field_name("arguments")
                && let Some(last) = named_children(args).last()
                && last.kind() == "number"
                && text(*last, body.src).parse::<i64>().is_ok()
            {
                out.push(*last);
            }
        });
        out
    }

    fn string_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            // A `template_string` may carry `${}` substitutions, which are code. Only the plain
            // `string` kind is offered as a literal.
            if node.kind() == "string" {
                out.push(node);
            }
        });
        out
    }

    fn local_bindings<'t>(&self, body: &FunctionBody<'t>) -> Vec<Binding<'t>> {
        let mut out = Vec::new();
        walk_body(body.body, &is_boundary, |node| {
            if node.kind() != "variable_declarator" {
                return;
            }
            // Only a plain identifier: a destructuring pattern binds several names at once and the
            // "bound exactly once in this body" proof gets harder to state, which makes the rename
            // no longer provably safe.
            if let Some(name) = node.child_by_field_name("name")
                && name.kind() == "identifier"
            {
                out.push(Binding {
                    name_node: name,
                    name: text(name, body.src).to_string(),
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
        // A JSDoc block lives outside the body and is already a `comment` node.
        None
    }

    fn import_block<'t>(&self, tree: &'t Tree, _src: &'t str) -> Option<ImportBlock<'t>> {
        let mut items: Vec<Node<'t>> = Vec::new();
        let mut cursor = tree.root_node().walk();
        for child in tree.root_node().named_children(&mut cursor) {
            if child.kind() == "import_statement" {
                items.push(child);
            } else if !items.is_empty() {
                break; // one contiguous group only
            }
        }
        if items.len() < 2 {
            return None;
        }
        // Not reorderable, and the clause-less `import "./polyfill";` shape is only the obvious
        // half of why.
        //
        // An ES module is *evaluated* when it is imported, and the evaluation order is specified:
        // the importer's `[[RequestedModules]]` are walked in source order. So `import { a } from
        // './a'` runs `./a`'s module body just as surely as `import './a'` does — it merely also
        // binds a name. Swapping two such statements swaps the order two module bodies run in, and
        // no check confined to this one file can establish that those bodies do not interact.
        //
        // The earlier check looked only for the clause-less shape and declared everything else
        // safe. That is a `cosmetic` label — a claim of behaviour preservation — resting on a
        // property nothing verified, which is the poisoned label this crate exists to prevent. The
        // operator is refused for the language, as it already is for Python and for the same
        // reason: import order is execution order here.
        Some(ImportBlock {
            items,
            reorderable: false,
            why_not: self.import_order_is_semantic(),
        })
    }

    fn identifiers<'t>(&self, tree: &'t Tree) -> Vec<Node<'t>> {
        let mut out = Vec::new();
        walk_tree(tree, |node| {
            if matches!(
                node.kind(),
                "identifier" | "property_identifier" | "type_identifier" | "shorthand_property_identifier"
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
        zero_value(raw).map(|z| {
            if z.is_empty() {
                "return".to_string()
            } else {
                format!("return {z}")
            }
        })
    }

    fn early_return(&self, return_type: Option<&str>) -> Option<String> {
        // Explicitly terminated: automatic semicolon insertion makes a bare `return` above live
        // code depend on what the next line starts with.
        self.default_return(return_type).map(|s| format!("{s};"))
    }

    fn hardcoded_return(
        &self,
        return_type: Option<&str>,
        literal: &str,
        kind: LiteralKind,
    ) -> Option<String> {
        let raw = return_type?.trim();
        let inner = promise_inner(raw).unwrap_or(raw);
        let ok = match kind {
            LiteralKind::Integer => inner == "number",
            LiteralKind::Str => inner == "string",
        };
        ok.then(|| format!("return {literal}"))
    }

    fn negate(&self, condition: &str) -> String {
        format!("!({condition})")
    }

    fn is_negated(&self, node: &Node<'_>, src: &str) -> bool {
        node.kind() == "unary_expression" && text(*node, src).trim_start().starts_with('!')
    }

    fn line_comment_prefix(&self) -> &'static str {
        "//"
    }

    fn formatter(&self) -> Option<Formatter> {
        // `prettier` is the language's formatter. Declaring it is not a claim that it is installed:
        // `crate::fmt::resolve` answers that at run time, and an unresolved formatter shrinks the
        // cosmetic operator set and says so in the manifest.
        Some(Formatter {
            program: "prettier",
            args: &["--stdin-filepath", "qd-mutate-input.ts"],
            via_xcrun: false,
        })
    }

    fn smoke_source(&self) -> &'static str {
        "const x = 1;\n"
    }

    fn line_wrapping_is_safe(&self) -> bool {
        // Automatic semicolon insertion makes a newline in the wrong place change the program —
        // `return\n  x` returns undefined. The operator is restricted rather than trusted to pick a
        // safe point.
        false
    }

    fn import_order_is_semantic(&self) -> Option<&'static str> {
        Some(
            "an ES module is evaluated when it is imported, in source order; a check confined to \
             one file cannot rule out an interaction between two module bodies",
        )
    }
}

/// `Promise<T>` unwrapped to `T`, respecting one level of nesting.
fn promise_inner(ty: &str) -> Option<&str> {
    let rest = ty.strip_prefix("Promise<")?.strip_suffix('>')?;
    Some(rest.trim())
}

/// The zero value for a TypeScript type, or `None` where it cannot be resolved.
///
/// Declining is the point: `stub.default_return`'s label claims the body now returns the type's
/// zero, and a guessed zero makes that claim false.
fn zero_value(ty: &str) -> Option<String> {
    let ty = ty.trim();
    if ty.is_empty() {
        return None;
    }
    if let Some(inner) = promise_inner(ty) {
        if inner == "void" {
            // `async` with no result: `return` alone. Rendered as an empty zero so the caller adds
            // the bare keyword.
            return Some(String::new());
        }
        return zero_value(inner);
    }
    if ty.ends_with("[]") || ty.starts_with("Array<") || ty.starts_with("ReadonlyArray<") {
        return Some("[]".to_string());
    }
    if ty.starts_with("Record<") || ty.starts_with("Map<") || ty.starts_with("Set<") {
        // A `Map`/`Set` zero is a constructor call and a `Record` zero is an object literal; the
        // three do not share a spelling, so each is given its own rather than one blurred guess.
        return Some(match ty.split('<').next().unwrap_or("") {
            "Map" => "new Map()".to_string(),
            "Set" => "new Set()".to_string(),
            _ => "{}".to_string(),
        });
    }
    match ty {
        "void" | "undefined" => Some("undefined".to_string()),
        "null" => Some("null".to_string()),
        "number" => Some("0".to_string()),
        "bigint" => Some("0n".to_string()),
        "string" => Some("\"\"".to_string()),
        "boolean" => Some("false".to_string()),
        // `any` and `unknown` have no zero that says anything, and a named interface's zero is not
        // spellable without knowing its members.
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
    use crate::parse::parse_bounded;

    fn parse(src: &str) -> tree_sitter::Tree {
        parse_bounded(&TypeScript.grammar(), src).expect("parses").tree
    }

    #[test]
    fn interface_and_abstract_members_have_no_body_and_are_skipped() {
        let src = "interface I { sig(): number }\n\
                   abstract class C { abstract m(): void; n(): number { return 1; } }\n";
        let tree = parse(src);
        let bodies = TypeScript.function_bodies(&tree, src);
        let names: Vec<&str> = bodies.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["n"], "only the member with a body: {names:?}");
    }

    #[test]
    fn an_expression_bodied_arrow_offers_nothing_to_stub() {
        let src = "const f = (x: number): number => x * 2;\n";
        let tree = parse(src);
        assert!(TypeScript.function_bodies(&tree, src).is_empty());
    }

    #[test]
    fn a_return_annotation_loses_its_colon() {
        let src = "function f(a: number): Promise<number[]> { return []; }\n";
        let tree = parse(src);
        let bodies = TypeScript.function_bodies(&tree, src);
        assert_eq!(
            bodies.first().and_then(|b| b.return_type.as_deref()),
            Some("Promise<number[]>")
        );
    }

    #[test]
    fn zero_values_resolve_or_decline() {
        assert_eq!(zero_value("number").as_deref(), Some("0"));
        assert_eq!(zero_value("number[]").as_deref(), Some("[]"));
        assert_eq!(zero_value("Promise<string>").as_deref(), Some("\"\""));
        assert_eq!(zero_value("Promise<void>").as_deref(), Some(""));
        assert_eq!(zero_value("any"), None, "declines rather than guesses");
        assert_eq!(zero_value("MyWidget"), None);
    }

    #[test]
    fn an_unannotated_return_type_is_unknown_not_void() {
        let src = "function f(a: number) { return 1; }\n";
        let tree = parse(src);
        let bodies = TypeScript.function_bodies(&tree, src);
        assert_eq!(bodies.first().and_then(|b| b.return_type.clone()), None);
        assert_eq!(TypeScript.default_return(None), None);
    }
}
