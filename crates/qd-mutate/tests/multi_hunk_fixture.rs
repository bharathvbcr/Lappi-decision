//! The committed multi-hunk fixture that `python/tests/test_multi_hunk_rebase.py` reads.
//!
//! The Python test checks that `qd_train.mutate_adapter` rebases a span over these diffs onto the
//! injected hunk's lines. That is only evidence about this crate while the fixture is still what
//! this crate renders, so this test re-renders every row's diff from its own `before` and `after`
//! and fails on any byte of drift — the renderer is the part the Python walker consumes.
//!
//! It does not regenerate the rows from the pool: which operator a body gets depends on which
//! formatters resolve on the machine running the test, and a byte comparison of operator choices
//! would pass or fail by machine. The regeneration command is in the Python test's docstring.

use std::path::PathBuf;

use qd_mutate::diffspan;
use qd_mutate::generate::{Example, DIFF_CONTEXT};
use qd_mutate::manifest::{sha256_hex, Manifest};
use qd_mutate::ops::MutationClass;
use qd_mutate::pool;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/tests/data/qd_mutate_multi_hunk")
}

fn read_fixture() -> (Vec<Example>, Manifest, Vec<u8>) {
    let dir = fixture_dir();
    let bytes = std::fs::read(dir.join("examples.jsonl")).expect("the fixture is committed");
    let examples: Vec<Example> = String::from_utf8(bytes.clone())
        .expect("utf-8")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("an Example row"))
        .collect();
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(dir.join("manifest.json")).expect("the manifest is committed"),
    )
    .expect("a Manifest");
    (examples, manifest, bytes)
}

#[test]
fn the_fixture_is_pinned_by_its_own_manifest() {
    let (examples, manifest, bytes) = read_fixture();
    assert_eq!(sha256_hex(&bytes), manifest.examples_sha256);
    assert_eq!(examples.len() as u64, manifest.totals.examples);
    assert_eq!(manifest.diff.renderer, "multi_hunk");
}

/// Pool id -> the commit's post-image, the text each mutation was applied to.
fn post_images() -> std::collections::HashMap<String, String> {
    let bytes = std::fs::read(fixture_dir().join("pool.jsonl")).expect("the pool is committed");
    pool::read_jsonl(bytes.as_slice())
        .expect("a pool")
        .into_iter()
        .map(|r| (r.id, r.source))
        .collect()
}

#[test]
fn every_diff_in_the_fixture_is_what_the_renderer_produces_today() {
    let (examples, _, _) = read_fixture();
    let sources = post_images();
    assert!(!examples.is_empty());
    for example in &examples {
        let rendered = diffspan::unified_multi_detailed(&example.before, &example.after, DIFF_CONTEXT)
            .expect("renders");
        assert_eq!(
            rendered.text, example.diff,
            "{}: the renderer no longer produces the committed diff; regenerate the fixture \
             (command in python/tests/test_multi_hunk_rebase.py) and re-run the Python test",
            example.id
        );
        if example.class == MutationClass::Clean {
            assert_eq!(example.span, None);
            continue;
        }
        let span = example.span.expect("a mutated row has a span");
        let source = sources.get(&example.pool_id).expect("every row's pool record is committed");
        assert!(
            rendered.shows_edit(&example.before, source, &example.after),
            "{}",
            example.id
        );
        assert!(rendered.hunk_containing(span).is_some(), "{}", example.id);
    }
}
