//! Traversal helpers shared by every `Language` implementation.
//!
//! Every walk here uses an explicit stack. The adversarial corpus contains 1000-level nesting; a
//! recursive walk over it is a stack overflow, and an overflow aborts the process rather than
//! producing a refusal that can be counted.

use tree_sitter::{Node, Tree};

/// Nodes one operator's site search may visit inside a single body. A bound, because every search
/// in this crate has one.
pub const MAX_WALK_VISITS: usize = 200_000;

/// Visit `root` and its descendants, **not** descending past a function boundary.
///
/// Nested functions and closures are their own `FunctionBody` candidates. Attributing an inner
/// function's sites to the outer body is the "which node did the mutator take" ambiguity the
/// hardening doc names, so the walk stops at each boundary and the inner body is offered
/// separately.
///
/// Returns false when the visit cap was hit, so a caller can record a truncated search rather than
/// read it as "no sites".
pub fn walk_body<'t, F>(root: Node<'t>, is_boundary: &dyn Fn(&str) -> bool, mut visit: F) -> bool
where
    F: FnMut(Node<'t>),
{
    let mut stack: Vec<Node<'t>> = vec![root];
    let mut visited = 0usize;
    let mut complete = true;
    while let Some(node) = stack.pop() {
        visited += 1;
        if visited > MAX_WALK_VISITS {
            complete = false;
            break;
        }
        visit(node);
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                // The root itself may be a boundary kind (it is the body we were handed);
                // its children are not skipped for that reason.
                if !is_boundary(child.kind()) {
                    stack.push(child);
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }
    complete
}

/// Visit every node in the tree. Iterative and bounded, same reasoning.
pub fn walk_tree<'t, F>(tree: &'t Tree, mut visit: F) -> bool
where
    F: FnMut(Node<'t>),
{
    let mut stack: Vec<Node<'t>> = vec![tree.root_node()];
    let mut visited = 0usize;
    while let Some(node) = stack.pop() {
        visited += 1;
        if visited > MAX_WALK_VISITS {
            return false;
        }
        visit(node);
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                stack.push(cursor.node());
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }
    true
}

/// Named children of `node`, in source order.
pub fn named_children<'t>(node: Node<'t>) -> Vec<Node<'t>> {
    let mut out = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        out.push(child);
    }
    out
}

/// All children of `node` including anonymous tokens, in source order.
pub fn all_children<'t>(node: Node<'t>) -> Vec<Node<'t>> {
    let mut out = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        out.push(child);
    }
    out
}

/// The text of `node`, or `""` when the range is not inside `src`.
pub fn text<'t>(node: Node<'_>, src: &'t str) -> &'t str {
    src.get(node.start_byte()..node.end_byte()).unwrap_or("")
}

/// True when `node` is lexically inside a function body other than itself.
///
/// Walks parents rather than recursing, and bounded by the tree depth the parser already accepted.
pub fn has_function_ancestor(node: Node<'_>, is_boundary: &dyn Fn(&str) -> bool) -> bool {
    let mut current = node.parent();
    let mut hops = 0usize;
    while let Some(parent) = current {
        hops += 1;
        if hops > crate::parse::MAX_TREE_DEPTH {
            // Deeper than the parser accepted: treat as nested, which is the conservative answer.
            return true;
        }
        if is_boundary(parent.kind()) {
            return true;
        }
        current = parent.parent();
    }
    false
}

/// The interior byte range of a braced block: everything strictly between the first `{` and the
/// last `}`. Falls back to the node's own range where the delimiters are not there (Python, and
/// Swift's `statements`).
pub fn braced_interior(body: Node<'_>, src: &str) -> (usize, usize) {
    let bytes = src.as_bytes();
    let (s, e) = (body.start_byte(), body.end_byte());
    if e <= s || e > bytes.len() {
        return (s, s);
    }
    if bytes[s] == b'{' && bytes[e - 1] == b'}' {
        (s + 1, e - 1)
    } else {
        (s, e)
    }
}

/// The interior range snapped to whole lines where that is possible.
///
/// `stub` operators replace the interior. Snapping makes the replacement cover whole lines, which
/// is what keeps the byte-derived span and the diff-derived span agreeing: an unsnapped edit
/// beginning right after `{` reports the signature line, while the text diff reports the line
/// below.
pub fn interior_lines(body: Node<'_>, src: &str) -> (usize, usize) {
    let (s, e) = braced_interior(body, src);
    crate::edit::snap_to_lines(src, s, e)
}

/// Does `haystack`'s trimmed, comment-stripped body already consist of nothing but one of
/// `stubs`?
///
/// This is how "a function whose body is already `todo!()`" is refused. It works on the *text of
/// statement nodes*, never on a substring of the whole file, so a string literal containing
/// `todo!()` does not trip it — which is exactly the gate the model is meant to beat.
pub fn body_is_already_stub(statements: &[String], stubs: &[&str]) -> bool {
    let meaningful: Vec<&String> = statements.iter().filter(|s| !s.trim().is_empty()).collect();
    if meaningful.len() != 1 {
        return false;
    }
    let only = meaningful[0].trim().trim_end_matches(';').trim();
    stubs
        .iter()
        .any(|stub| only == stub.trim().trim_end_matches(';').trim())
}
