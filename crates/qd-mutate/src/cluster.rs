//! `cluster_key`: the identity of the function a row was made from.
//!
//! Every row `qd-mutate` writes carries one, so a consumer can keep every row of one function in
//! one split and cap how many rows one function contributes (survey M4: mutants of one function
//! landed in different splits, and `identity_disjoint` only knew `repo + path + symbol`).
//!
//! # What goes into the key
//!
//! A SHA-256 over these fields, in this order, each framed by a NUL:
//!
//! 1. the tool's key version (`qd-mutate/cluster_key/v1`);
//! 2. `repo`, as the pool record states it;
//! 3. `path`, as the pool record states it (no separator normalisation);
//! 4. the language id;
//! 5. the **scope chain** of the declaration node: for every ancestor of it, outermost first, the
//!    grammar's node kind and the text of its `name` field (a Rust `impl_item` has no name, so its
//!    trait and type are used), joined by U+001F; and, for a Go method, the receiver's *type*;
//! 6. the declaration's own node kind and name;
//! 7. the declaration's arity.
//!
//! Every piece of text is stripped of all whitespace first, so `Foo < T >` and `Foo<T>` agree.
//!
//! The key is output as `fk1:` and the first 32 hex digits (128 bits).
//!
//! # What does not go into it
//!
//! * the body, the diff, the mutation, the operator, the span: the key is computed from the
//!   **unmutated** parse, so a mutant, its cosmetic twin and the clean row share one key;
//! * line numbers and byte offsets, comments, parameter names and types, the return type, the
//!   receiver's variable name: layout and spelling inside the declaration do not move it;
//! * line endings and a BOM: the generator parses the normalized source, and nothing here reads
//!   the raw bytes.
//!
//! # What it distinguishes
//!
//! Two functions with identical bodies in different files or repos differ (path and repo are in the
//! key). Nested functions are separate bodies and differ from their parent and from a namesake
//! nested elsewhere, because the scope chain is in the key. Methods of one name on different
//! receivers or in different classes differ for the same reason. A closure is never a body here (the
//! languages' `function_bodies` offer named declarations only), so no row names one.
//!
//! A **renamed** function is a different function: the key is an identity, and an identity that
//! followed renames would have to judge similarity. `FunctionIdentity` already treats the symbol
//! as part of identity, and a rename that crosses a split is the near-duplicate check's to catch.
//! Moving a function to another file likewise changes its key.

use sha2::{Digest, Sha256};
use tree_sitter::Node;

use crate::lang::util::text;
use crate::lang::{FunctionBody, LangId};
use crate::parse::MAX_TREE_DEPTH;

/// Version stamped into the hashed material, so a change to this recipe cannot reuse old keys.
const KEY_VERSION: &str = "qd-mutate/cluster_key/v1";

/// The key of `body`, a function declared in `path` of `repo`.
pub fn cluster_key(repo: &str, path: &str, language: LangId, body: &FunctionBody<'_>) -> String {
    let mut scope: Vec<String> = Vec::new();
    let mut current = body.decl.parent();
    let mut hops = 0usize;
    while let Some(node) = current {
        hops += 1;
        if hops > MAX_TREE_DEPTH + 1 {
            break;
        }
        if let Some(segment) = scope_segment(node, body.src) {
            scope.push(segment);
        }
        current = node.parent();
    }
    scope.reverse();
    if let Some(receiver) = receiver_type(body.decl, body.src) {
        scope.push(format!("receiver:{receiver}"));
    }
    scope.push(format!("{}:{}", body.node_kind, squash(&body.name)));

    let mut hasher = Sha256::new();
    for field in [
        KEY_VERSION,
        repo,
        path,
        language.as_str(),
        &scope.join("\u{1f}"),
        &body.arity.to_string(),
    ] {
        hasher.update(field.as_bytes());
        hasher.update([0u8]);
    }
    let digest = hasher.finalize();
    let hex: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
    format!("fk1:{hex}")
}

/// One ancestor's contribution to the scope chain, `None` when it names nothing.
fn scope_segment(node: Node<'_>, src: &str) -> Option<String> {
    if node.kind() == "impl_item" {
        let of = |field: &str| {
            node.child_by_field_name(field)
                .map(|n| squash(text(n, src)))
                .unwrap_or_default()
        };
        return Some(format!("impl:{}|{}", of("trait"), of("type")));
    }
    let name = node.child_by_field_name("name")?;
    Some(format!("{}:{}", node.kind(), squash(text(name, src))))
}

/// The receiver's type for a Go method (`func (s *Server) Get`), not its variable name.
fn receiver_type(decl: Node<'_>, src: &str) -> Option<String> {
    if decl.kind() != "method_declaration" {
        return None;
    }
    let receiver = decl.child_by_field_name("receiver")?;
    let mut cursor = receiver.walk();
    let param = receiver
        .named_children(&mut cursor)
        .find(|n| n.kind() == "parameter_declaration")?;
    let ty = param.child_by_field_name("type")?;
    Some(squash(text(ty, src)))
}

fn squash(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}
