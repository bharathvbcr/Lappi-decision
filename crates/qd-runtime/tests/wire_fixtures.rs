//! The golden wire corpus in `fixtures/wire/`, checked and regenerated.
//!
//! # What this file is for
//!
//! `docs/schema-api.md` specified the request side well and the answer side with a single example
//! object, and a concurrent lane is writing a Python parser for that answer side. Two lanes with
//! two readings of one format and no shared artefact is precisely how
//! `GAP-RT-WIRE-CONTEXT-ENCODING` and `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS` both survived with
//! both suites green. `fixtures/wire/` is the shared artefact: **when a field changes here, the
//! corpus changes, and the other side breaks loudly.**
//!
//! # Running it
//!
//! ```text
//! cargo test -p qd-runtime --test wire_fixtures              # compare; fails on any difference
//! QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures   # regenerate
//! ```
//!
//! The comparison is the default on purpose. A corpus that regenerates itself on every run records
//! whatever the code currently does and can never disagree with it, which is not a gate.

use std::collections::BTreeSet;
use std::path::PathBuf;

use qd_runtime::fixtures::{
    self, CorpusCheck, Manifest, CORPUS_DIR, MANIFEST,
};
use qd_runtime::schema::{Response, SlotAnswer};

/// `crates/qd-runtime/tests/..` -> the repository root.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .expect("the crate lives two directories below the repository root")
}

fn corpus_dir() -> PathBuf {
    repo_root().join(CORPUS_DIR)
}

/// True when this run was asked to rewrite the corpus rather than check it.
fn write_mode() -> bool {
    matches!(
        std::env::var("QD_WRITE_FIXTURES").as_deref(),
        Ok("1") | Ok("true")
    )
}

#[test]
fn the_corpus_on_disk_is_what_this_build_emits() {
    let dir = corpus_dir();
    if write_mode() {
        let written = fixtures::write_corpus(&dir).expect("the corpus generates and writes");
        println!("wrote {written} files to {}", dir.display());
        // Writing is not checking. Fall through to the comparison so a run that regenerates still
        // proves the bytes it wrote are the bytes it would check.
    }

    match fixtures::check_corpus(&dir) {
        CorpusCheck::Matches { files } => {
            assert!(files > 40, "a corpus this small has lost entries: {files}");
        }
        CorpusCheck::Differs { problems } => panic!(
            "fixtures/wire/ disagrees with this build in {} place(s):\n  {}\n\nIf the change to \
             the envelope was intended, regenerate with\n  QD_WRITE_FIXTURES=1 cargo test -p \
             qd-runtime --test wire_fixtures\nand say so in docs/schema-api.md in the same \
             change: the Python lane reads these files.",
            problems.len(),
            problems.join("\n  ")
        ),
        CorpusCheck::NotRead { reason } => panic!(
            "the corpus could not be read, which is not the same answer as `it matched`: {reason}"
        ),
    }
}

#[test]
fn the_corpus_covers_every_refusal_and_every_backend_error() {
    // A corpus missing a variant teaches the other lane that the variant does not exist. The
    // counts are pinned rather than merely compared to each other, so adding a variant without a
    // fixture fails here as well as in `refusal_is_not_noul.rs`.
    let entries = fixtures::corpus().expect("the corpus generates");
    let refusal_kinds: BTreeSet<&str> = entries
        .iter()
        .filter_map(|f| match &f.response {
            Response::Refused(e) => Some(e.refusal.kind()),
            _ => None,
        })
        .collect();
    assert_eq!(
        refusal_kinds.len(),
        36,
        "every refusal kind needs a fixture; missing one hides a whole failure mode from the \
         other lane"
    );

    let error_kinds: BTreeSet<&str> = entries
        .iter()
        .filter_map(|f| match &f.response {
            Response::Error(e) => Some(e.error.kind()),
            _ => None,
        })
        .collect();
    assert_eq!(error_kinds.len(), 12, "every backend-error kind needs a fixture");

    let answers = entries
        .iter()
        .filter(|f| matches!(f.response, Response::Ok(_)))
        .count();
    assert_eq!(
        answers, 8,
        "the answer fixtures are: the contract's three-slot example, one per slot kind, a maximal \
         option set, the registered route, an all-abstained envelope, and a non-degraded one"
    );
}

#[test]
fn every_answer_in_the_corpus_covers_a_slot_kind_and_the_three_together_cover_all_of_them() {
    let entries = fixtures::corpus().expect("the corpus generates");
    let mut seen_choice = false;
    let mut seen_score = false;
    let mut seen_span = false;
    let mut seen_abstention = false;
    let mut seen_not_degraded = false;

    for fixture in &entries {
        let Response::Ok(envelope) = &fixture.response else {
            continue;
        };
        if !envelope.degraded {
            seen_not_degraded = true;
        }
        for slot in envelope.slots.values() {
            match (&slot.value, slot.noul) {
                (Some(qd_runtime::schema::SlotValue::Choice(_)), false) => seen_choice = true,
                (Some(qd_runtime::schema::SlotValue::Score(_)), false) => seen_score = true,
                (Some(qd_runtime::schema::SlotValue::Span(_)), false) => seen_span = true,
                (None, true) => seen_abstention = true,
                (value, noul) => panic!(
                    "{}: a slot answer with value {value:?} and noul {noul} violates the \
                     biconditional the corpus is supposed to demonstrate",
                    fixture.file
                ),
            }
        }
    }
    assert!(seen_choice, "no answered `choice` value in the corpus");
    assert!(seen_score, "no answered `score` value in the corpus");
    assert!(seen_span, "no answered `span` value in the corpus");
    assert!(seen_abstention, "no `noul` slot in the corpus");
    assert!(
        seen_not_degraded,
        "every fixture is degraded, so a parser could not tell the flag was a variable"
    );
}

#[test]
fn every_fixture_round_trips_through_this_builds_own_parser() {
    // The corpus is only a contract if the bytes in it parse back to the values they came from.
    // A file that serializes but does not deserialize would be a shape the other lane is told to
    // accept and this lane cannot read.
    for fixture in fixtures::corpus().expect("the corpus generates") {
        let bytes = fixture.bytes().expect("serializes");
        let parsed: Response = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            panic!(
                "{} does not parse back: {e}\n{}",
                fixture.file,
                String::from_utf8_lossy(&bytes)
            )
        });
        assert_eq!(
            parsed, fixture.response,
            "{} did not round-trip to an equal value",
            fixture.file
        );
        assert_eq!(parsed.status(), fixture.status, "{}", fixture.file);
    }
}

#[test]
fn the_manifest_accounts_for_every_file_and_for_all_three_statuses() {
    let files = fixtures::files().expect("the corpus generates");
    let manifest_bytes = files
        .iter()
        .find(|(name, _)| name == MANIFEST)
        .map(|(_, bytes)| bytes.clone())
        .expect("the corpus carries a manifest");
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).expect("the manifest parses");

    assert_eq!(
        manifest.count,
        files.len() - 1,
        "the manifest counts every file except itself"
    );
    let named: BTreeSet<&str> = manifest.entries.iter().map(|e| e.file.as_str()).collect();
    let on_disk: BTreeSet<&str> = files
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| *name != MANIFEST)
        .collect();
    assert_eq!(
        named, on_disk,
        "a file the manifest does not name is a file a consumer enumerating the manifest will \
         silently skip"
    );
    for status in ["ok", "refused", "error"] {
        assert!(
            manifest.by_status.get(status).copied().unwrap_or(0) > 0,
            "the manifest must show that `{status}` replies exist; a parser that never sees one \
             has not been told the variant is possible"
        );
    }
}

#[test]
fn a_refusal_fixture_is_not_mistakable_for_an_abstention() {
    // The corpus's central lesson for whoever parses it, asserted on the corpus itself rather
    // than only on the types: `refused` and an all-`noul` `ok` are different replies.
    for fixture in fixtures::corpus().expect("the corpus generates") {
        let bytes = fixture.bytes().expect("serializes");
        let text = String::from_utf8(bytes).expect("UTF-8");
        let value: serde_json::Value = serde_json::from_str(&text).expect("JSON");
        match fixture.status {
            "refused" | "error" => {
                assert!(
                    value.get("slots").is_none(),
                    "{}: a failure reply must carry no `slots`",
                    fixture.file
                );
                assert!(
                    serde_json::from_str::<SlotAnswer>(&text).is_err(),
                    "{}: a failure reply must not parse as a slot answer",
                    fixture.file
                );
            }
            "ok" => {
                assert!(
                    value.get("refusal").is_none() && value.get("error").is_none(),
                    "{}: an answer must carry neither `refusal` nor `error`",
                    fixture.file
                );
            }
            other => panic!("{}: unexpected status `{other}`", fixture.file),
        }
    }
}
