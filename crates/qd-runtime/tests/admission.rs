//! Admission: a `code.defect_class` request whose context is not the shape the task was trained
//! on is refused before the model is asked (Fable round G, G6).
//!
//! Trained shape (`python/qd_data/mixture.py`, the `code.defect_class` rewriter): `file: <path>`,
//! a blank line, then a unified diff's hunks. The OOD suite's three categories are what this
//! keeps away from the model: prose (no `file:` header, no hunk), languages the pool never held,
//! and scrambled diffs whose hunks no longer parse.

use std::sync::Arc;

use qd_runtime::calibration::CalibrationTable;
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{DecisionRequest, Response};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::{Value, json};

/// Three old lines (two context, one removed) and three new (two context, one added).
const RUST_DIFF: &str = "file: src/lib.rs\n\n@@ -1,3 +1,3 @@\n fn add(a: i32, b: i32) -> i32 {\n-    a + b\n+    todo!()\n }";

fn runtime() -> Runtime {
    Runtime::with_backend(
        Arc::new(ReferenceBackend::new(true)),
        CalibrationTable::reference(),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("assembles")
}

fn request(task: &str, context: &str) -> DecisionRequest {
    let value = json!({
        "schema_version": 1, "task": task,
        "context_b64": qd_runtime::b64::encode(context.as_bytes()),
        "context_len": context.len(),
        "question": "What kind of change is this diff?",
        "slots": [{"name": "class", "type": "choice",
                   "options": ["stub", "logic", "cosmetic", "clean"]}],
        "route": "generic",
    });
    match parse_line(&serde_json::to_vec(&value).expect("serializes")) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("the test request was not accepted on the wire: {other:?}"),
    }
}

fn defect(context: &str) -> Response {
    runtime().answer(&request("code.defect_class", context), None)
}

/// The refusal's serialized form, so the check names its kind and carries its fields.
#[track_caller]
fn refused(response: Response, kind: &str) -> Value {
    match response {
        Response::Refused(envelope) => {
            assert_eq!(envelope.refusal.kind(), kind, "{:?}", envelope.refusal);
            serde_json::to_value(&envelope.refusal).expect("serializes")
        }
        other => panic!("expected a `{kind}` refusal, got {other:?}"),
    }
}

#[track_caller]
fn answered(response: Response) {
    match response {
        Response::Ok(_) => {}
        other => panic!("expected an answer, got {other:?}"),
    }
}

#[test]
fn a_diff_in_the_trained_shape_and_a_pool_language_is_answered() {
    answered(defect(RUST_DIFF));
    // A trailing newline terminates the last line; it does not open an empty one.
    answered(defect(&format!("{RUST_DIFF}\n")));
    // Two hunks, a section heading, an omitted count (= 1), and the no-newline marker.
    answered(defect(
        "file: pkg/sum.go\n\n@@ -3,2 +3,2 @@ func Sum(xs []int) int {\n total := 0\n-for _, x := range xs {\n+for _, x := range xs[1:] {\n@@ -9 +9 @@\n-return total\n+return total + 1\n\\ No newline at end of file",
    ));
    // Every extension the pool reader maps to a held language.
    for path in [
        "a.py", "a.pyi", "a.ts", "a.tsx", "a.mts", "a.cts", "a.go", "a.rs",
    ] {
        answered(defect(&RUST_DIFF.replace("src/lib.rs", path)));
    }
}

#[test]
fn prose_is_not_a_unified_diff() {
    let refusal = refused(
        defect("What is the capital of France? Paris, which sits on the Seine."),
        "context_not_unified_diff",
    );
    assert_eq!(refusal["line"], json!(1));
    assert!(
        refusal["expected"].as_str().unwrap().contains("file:"),
        "{refusal}"
    );
}

#[test]
fn a_header_with_no_hunk_or_a_body_before_the_first_hunk_is_refused() {
    let refusal = refused(
        defect("file: src/lib.rs\n\nfn add(a: i32, b: i32) -> i32 { a + b }"),
        "context_not_unified_diff",
    );
    assert_eq!(refusal["line"], json!(3));
    // `---`/`+++` file headers are not in the trained shape: the path is in `file:`.
    refused(
        defect(&RUST_DIFF.replace("\n\n@@", "\n\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@")),
        "context_not_unified_diff",
    );
    // The header and nothing after it.
    refused(defect("file: src/lib.rs\n\n"), "context_not_unified_diff");
    // No blank line between the header and the diff.
    refused(
        defect(&RUST_DIFF.replace("\n\n@@", "\n@@")),
        "context_not_unified_diff",
    );
}

#[test]
fn a_hunk_whose_body_disagrees_with_its_header_is_refused() {
    // Declares 4 old and 4 new lines; the body has 3 of each.
    let refusal = refused(
        defect(&RUST_DIFF.replace("@@ -1,3 +1,3 @@", "@@ -1,4 +1,4 @@")),
        "context_not_unified_diff",
    );
    assert_eq!(
        refusal["line"],
        json!(3),
        "the line of the hunk that came up short"
    );
    // Declares 2 and 2; the body runs over at its last line.
    let refusal = refused(
        defect(&RUST_DIFF.replace("@@ -1,3 +1,3 @@", "@@ -1,2 +1,2 @@")),
        "context_not_unified_diff",
    );
    assert_eq!(refusal["line"], json!(7));
    // A malformed header.
    refused(
        defect(&RUST_DIFF.replace("@@ -1,3 +1,3 @@", "@@ -1,x +1,3 @@")),
        "context_not_unified_diff",
    );
    refused(
        defect(&RUST_DIFF.replace("@@ -1,3 +1,3 @@", "@@ -1,3 +1,3")),
        "context_not_unified_diff",
    );
}

#[test]
fn a_line_without_a_diff_marker_is_refused() {
    // Line-shuffling a diff's lines out of their markers, as the scrambled OOD rows do.
    let refusal = refused(
        defect(&RUST_DIFF.replace("\n }", "\n}")),
        "context_not_unified_diff",
    );
    assert_eq!(refusal["line"], json!(7));
    // An empty line inside a hunk: the trained diffs mark an empty context line with a space.
    refused(
        defect(&RUST_DIFF.replace("\n-    a + b", "\n\n-    a + b")),
        "context_not_unified_diff",
    );
}

#[test]
fn a_language_the_pool_never_held_is_refused() {
    for (path, language) in [
        ("src/main.c", "unrecognised"),
        ("Main.java", "unrecognised"),
        ("lib/thing.rb", "unrecognised"),
        ("types.d.ts", "unrecognised"),
        ("Makefile", "unrecognised"),
        // The pool reader maps Swift, but the pool that was built holds no Swift row.
        ("App/View.swift", "swift"),
    ] {
        let refusal = refused(
            defect(&RUST_DIFF.replace("src/lib.rs", path)),
            "context_language_not_in_pool",
        );
        assert_eq!(refusal["path"], json!(path));
        assert_eq!(refusal["language"], json!(language), "{path}");
        assert_eq!(
            refusal["pool"],
            json!(["go", "python", "rust", "typescript"]),
            "{path}"
        );
    }
}

#[test]
fn other_tasks_are_not_held_to_the_defect_class_shape() {
    let prose = "What is the capital of France? Paris, which sits on the Seine.";
    answered(runtime().answer(&request("qa.answerability", prose), None));
    answered(runtime().answer(&request("devcouncil.verdict", prose), None));
}
