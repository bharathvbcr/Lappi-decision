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
    // A bare `---`/`+++` preamble is in neither trained shape: single-file contexts carry the
    // path in `file:`, and composed blocks open with `diff --git` before their `---`/`+++`.
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

/// A composed row as `rewrite_defect_class` renders it: `file:` names the repository (the
/// composed corpora's `path` is the repo on every row), then file blocks under the three headers
/// `_composed_diff_span` requires, in several pool languages, the last line whitespace-only as
/// 15,199 of composed-v2's 25,000 rows end.
const COMPOSED: &str = concat!(
    "file: hoop33/limo\n",
    "\n",
    "diff --git a/chef/tests/__init__.py b/chef/tests/__init__.py\n",
    "--- a/chef/tests/__init__.py\n",
    "+++ b/chef/tests/__init__.py\n",
    "@@ -18,2 +18,3 @@ class ChefTestCase(unittest.TestCase):\n",
    "         self.api.set_default()\n",
    "+        self.objects = []\n",
    " \n",
    "diff --git a/endpoints/downloads/retry.go b/endpoints/downloads/retry.go\n",
    "--- a/endpoints/downloads/retry.go\n",
    "+++ b/endpoints/downloads/retry.go\n",
    "@@ -4 +4 @@\n",
    "-\treturn 3\n",
    "+\treturn 4\n",
    "@@ -9,0 +10 @@ func Retry() {\n",
    "+\tlog()\n",
    "diff --git a/tests/custom.spec.ts b/tests/custom.spec.ts\n",
    "--- a/tests/custom.spec.ts\n",
    "+++ b/tests/custom.spec.ts\n",
    "@@ -1,3 +1,3 @@\n",
    " import { x } from './x';\n",
    "-export const y = x;\n",
    "+export const y = x + 1;\n",
    " ",
);

#[test]
fn a_composed_context_in_the_trained_shape_is_answered() {
    answered(defect(COMPOSED));
    answered(defect(&format!("{COMPOSED}\n")));
    // One block is the composed shape too: three or more is a generation option, not a shape.
    answered(defect(
        "file: src/lib.rs\n\ndiff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-a\n+b",
    ));
    // A path that itself holds ` b/` reads as one path, because both copies are equal.
    answered(defect(
        "file: o/r\n\ndiff --git a/x b/y.rs b/x b/y.rs\n--- a/x b/y.rs\n+++ b/x b/y.rs\n@@ -1 +1 @@\n-a\n+b",
    ));
}

#[test]
fn a_composed_block_out_of_the_trained_shape_is_refused() {
    let go_block = "diff --git a/endpoints/downloads/retry.go b/endpoints/downloads/retry.go\n";
    // (the edit, the line the refusal names)
    let cases: [(String, u64); 8] = [
        // The two paths of `diff --git` differ: a rename, which no trained block shows.
        (
            COMPOSED.replace(
                "b/endpoints/downloads/retry.go\n---",
                "b/endpoints/downloads/moved.go\n---",
            ),
            10,
        ),
        // `git diff`'s `index` line: no trained block has one.
        (
            COMPOSED.replace(
                go_block,
                &format!("{go_block}index 3b18e51..a9c4f02 100644\n"),
            ),
            11,
        ),
        // A new file's `/dev/null` side.
        (
            COMPOSED.replace("--- a/endpoints/downloads/retry.go", "--- /dev/null"),
            11,
        ),
        // The `+++` line names another file.
        (
            COMPOSED.replace("+++ b/tests/custom.spec.ts", "+++ b/tests/other.spec.ts"),
            20,
        ),
        // A block with no hunk: the next block's header stands where a hunk header must.
        (
            COMPOSED.replace(
                "@@ -4 +4 @@\n-\treturn 3\n+\treturn 4\n@@ -9,0 +10 @@ func Retry() {\n+\tlog()\n",
                "",
            ),
            13,
        ),
        // A hunk cut short by the next block: its header's line is named.
        (COMPOSED.replace("+        self.objects = []\n", ""), 6),
        // Empty paths.
        (
            "file: o/r\n\ndiff --git a/ b/\n--- a/\n+++ b/\n@@ -1 +1 @@\n-a\n+b".to_string(),
            3,
        ),
        // Header lines then the end.
        (
            "file: o/r\n\ndiff --git a/x.rs b/x.rs\n--- a/x.rs".to_string(),
            5,
        ),
    ];
    for (context, line) in cases {
        let refusal = refused(defect(&context), "context_not_unified_diff");
        assert_eq!(refusal["line"], json!(line), "{context}\n{refusal}");
    }
    // Each block's own path is held to the pool, not the `file:` line's.
    let refusal = refused(
        defect(&COMPOSED.replace("tests/custom.spec.ts", "App/View.swift")),
        "context_language_not_in_pool",
    );
    assert_eq!(refusal["path"], json!("App/View.swift"));
    // A file block inside a single-file context is no shape either.
    refused(
        defect(&format!(
            "{RUST_DIFF}\ndiff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-a\n+b"
        )),
        "context_not_unified_diff",
    );
}

/// The request under `## Request` in `docs/schema-api.md`, the one a caller copies.
fn documented_request() -> Value {
    let doc = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/schema-api.md"),
    )
    .expect("docs/schema-api.md reads");
    let section = doc
        .split_once("## Request")
        .expect("docs/schema-api.md has a `## Request` section")
        .1;
    let block = section
        .split_once("```json\n")
        .and_then(|(_, rest)| rest.split_once("\n```"))
        .expect("a json block under `## Request`")
        .0;
    serde_json::from_str(block).expect("the documented request is JSON")
}

#[test]
fn the_documented_request_is_admitted_and_answered() {
    let value = documented_request();
    assert_eq!(value["task"], json!("code.defect_class"));
    let request = match parse_line(&serde_json::to_vec(&value).expect("serializes")) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("docs/schema-api.md's request is not accepted on the wire: {other:?}"),
    };
    qd_runtime::admission::admit_defect_context(&request.context)
        .expect("docs/schema-api.md's example context is a trained shape");
    answered(runtime().answer(&request, None));
}

/// Every row of real defect corpora, rendered as `rewrite_defect_class` renders it
/// (`file: {path}\n\n` + the diff without its final newline), is admitted. Opt-in: the corpora
/// are campaign data outside the repo, so this is `#[ignore]`d and reports *ignored*, never
/// *passed*, unless run with `QD_DEFECT_EXAMPLES=<examples.jsonl>[:<examples.jsonl>...]` and
/// `--ignored`. A row the loader refuses (an empty diff) is counted, not rendered.
#[test]
#[ignore = "needs QD_DEFECT_EXAMPLES: corpora outside the repo"]
fn every_trained_defect_context_in_the_named_corpora_is_admitted() {
    let paths = std::env::var("QD_DEFECT_EXAMPLES")
        .expect("QD_DEFECT_EXAMPLES names the examples.jsonl files to check");
    let (mut admitted, mut no_diff) = (0usize, 0usize);
    let mut refused: Vec<String> = Vec::new();
    for path in paths.split(':') {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        for line in text.lines() {
            let row: Value = serde_json::from_str(line).expect("a JSON row");
            let (Some(file), Some(diff)) = (row["path"].as_str(), row["diff"].as_str()) else {
                panic!("{path}: a row without path and diff: {}", row["id"]);
            };
            if diff.trim().is_empty() {
                no_diff += 1;
                continue;
            }
            let body = diff.strip_suffix('\n').unwrap_or(diff);
            let context = format!("file: {file}\n\n{body}");
            match qd_runtime::admission::admit_defect_context(
                &qd_runtime::context::Context::from_bytes(context.into_bytes()),
            ) {
                Ok(()) => admitted += 1,
                Err(refusal) => refused.push(format!("{}: {refusal:?}", row["id"])),
            }
        }
    }
    eprintln!(
        "admitted {admitted}, refused {}, skipped (empty diff) {no_diff}",
        refused.len()
    );
    assert!(
        refused.is_empty(),
        "{} trained contexts refused; first: {:?}",
        refused.len(),
        &refused[..refused.len().min(5)]
    );
}

#[test]
fn other_tasks_are_not_held_to_the_defect_class_shape() {
    let prose = "What is the capital of France? Paris, which sits on the Seine.";
    answered(runtime().answer(&request("qa.answerability", prose), None));
    answered(runtime().answer(&request("devcouncil.verdict", prose), None));
}
