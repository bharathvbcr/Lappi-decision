//! GPU gates for the span head, on the tiny fixture tower (`tools/gen_span_fixture.py`): the tower
//! and a trained-shaped span head are exported by `qd_export` and served by `MetalBackend`, and
//! every number is compared with the Python that trains them.
//!
//! **Every test here is `#[ignore]`d**: it runs on the GPU. `cargo test` never runs them and
//! reports them as ignored, not passed.
//!
//! ```text
//! cargo test --release -p qd-metal --test gpu_span -- --ignored --test-threads=1
//! ```
//!
//! The reference is a bf16 tower on CPU in torch; tessl runs bf16 operand GEMMs on the GPU, so
//! the two agree to bf16 noise, not bit for bit. The tolerances below are fractions of the
//! largest score, and the convention test shows they separate the final-norm hidden state from the
//! pre-norm residual stream by a wide margin.

mod common;

use common::{decode, edit_release_manifest, rel_diff, start, tiny_release, Reference, HIDDEN, SCORE_REL};
use qd_metal::model::Model;
use qd_metal::span::SpanServing;
use qd_metal::tokenizer::QwenTokenizer;
use qd_runtime::backend::{DecisionBackend, DecodeMode, QueryKind, SlotQuery};
use serde_json::json;

#[test]
#[ignore = "GPU"]
fn gpu_a_span_decode_on_the_fixture_request_is_answered_and_equals_pythons_head() {
    let refc = Reference::load();
    let (_keep, release) = tiny_release("gpuspan", |_| {});
    let backend = start(&release);
    let rows = refc.usize("line_count") + 1;
    for (kind, key) in [(QueryKind::PointerStart, "start_scores"), (QueryKind::PointerEnd, "end_scores")] {
        let got = decode(&backend, &refc, "span_suffix", kind, rows).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(got.kind, kind);
        assert_eq!(got.values.len(), rows, "line_count + 1 rows, the abstain row last");
        let d = rel_diff(&got.values, &refc.floats(key));
        println!("{kind:?}: max |rust - python| / max |python| = {d:.5}");
        assert!(d <= SCORE_REL, "{kind:?} differs from SpanPointerHead by {d}");
        let best = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
        assert_eq!(best(&got.values), best(&refc.floats(key)), "{kind:?}: the chosen row is Python's");
    }
    // The runtime's row count is the only one answered.
    for wrong in [rows - 1, rows + 1, 1] {
        let err = decode(&backend, &refc, "span_suffix", QueryKind::PointerStart, wrong).unwrap_err().to_string();
        assert!(err.contains("line_count + 1"), "{wrong} rows: {err}");
    }
}

#[test]
#[ignore = "GPU"]
fn gpu_the_head_reads_the_final_norm_hidden_state_not_the_residual_stream() {
    let refc = Reference::load();
    let (_keep, release) = tiny_release("gpuconv", |_| {});
    let rt = tessl::GpuRuntime::new().expect("runtime");
    let model = Model::load(&rt, &release).expect("load");
    let tok = QwenTokenizer::load(&release.join("tokenizer.json")).unwrap();
    let text = format!("{}{}", refc.str("prefix"), refc.str("span_suffix"));
    let (ids, offsets) = tok.encode_untrusted_with_offsets(&text).unwrap();
    assert_eq!(ids, refc.rows("token_ids"), "the same tokens as Python's tokenizer");
    let want_offsets: Vec<(usize, usize)> = refc.0["token_offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (usize::try_from(p[0].as_u64().unwrap()).unwrap(), usize::try_from(p[1].as_u64().unwrap()).unwrap()))
        .collect();
    assert_eq!(offsets, want_offsets, "and the same offsets (ASCII: bytes are characters)");
    let tokens = qd_metal::span::plan_candidates(&text, &offsets).unwrap();
    let want_tokens: Vec<usize> = refc.rows("candidate_tokens").into_iter().map(|t| t as usize).collect();
    assert_eq!(tokens, want_tokens, "the line -> token rule agrees with shards._token_indices_for_chars");

    let seq = u32::try_from(ids.len()).unwrap();
    let out = model.run(&ids, 1, seq, None, false, None).unwrap();
    let mut rows: Vec<u32> = tokens.iter().map(|t| u32::try_from(*t).unwrap()).collect();
    rows.push(seq - 1);
    let post = model.final_hidden(&out, &rows).unwrap();
    let pre = model.residual_rows(&out, &rows).unwrap();

    // The hidden states themselves.
    let mut want_hidden = refc.floats("hidden_candidates");
    want_hidden.extend(refc.floats("hidden_query"));
    let hidden_diff = rel_diff(&post, &want_hidden);
    println!("final-norm hidden state vs Python last_hidden_state: {hidden_diff:.5}");
    assert!(hidden_diff <= 0.08, "final-norm hidden state differs from Python's by {hidden_diff}");

    // The scores they give.
    let head = match qd_metal::span::open(&release, HIDDEN) {
        SpanServing::Ready(h) => h,
        other => panic!("the fixture's head must be served: {other:?}"),
    };
    let n = tokens.len();
    let score = |h: &[f32]| head.scores(&h[..n * HIDDEN], n, &h[n * HIDDEN..]).unwrap();
    let (post_s, pre_s) = (score(&post), score(&pre));
    let want_start = refc.floats("start_scores");
    let want_end = refc.floats("end_scores");
    let post_d = rel_diff(&post_s.start, &want_start).max(rel_diff(&post_s.end, &want_end));
    let pre_d = rel_diff(&pre_s.start, &want_start).max(rel_diff(&pre_s.end, &want_end));
    println!("start/end vs Python: final-norm {post_d:.5}, pre-norm residual {pre_d:.5}");
    assert!(post_d <= SCORE_REL, "final-norm convention misses Python by {post_d}");
    assert!(
        pre_d > 20.0 * SCORE_REL,
        "the pre-norm residual stream must be far from Python's scores ({pre_d}); otherwise this fixture \
         cannot tell the two conventions apart"
    );
}

#[test]
#[ignore = "GPU"]
fn gpu_a_head_the_loader_refuses_answers_span_queries_with_the_reason_and_leaves_choice_alone() {
    let refc = Reference::load();
    let rows = refc.usize("line_count") + 1;

    // A blend that was not pre-registered.
    let (_keep, blend) = tiny_release("gpublend", |m| {
        m["span_head_source"] = json!({"mode": "averaged", "seed": null, "input": null, "payload_digest": null});
    });
    let backend = start(&blend);
    let err = decode(&backend, &refc, "span_suffix", QueryKind::PointerStart, rows).unwrap_err().to_string();
    assert!(err.contains("weight average") && err.contains("pre-registered"), "{err}");
    decode(&backend, &refc, "choice_suffix", QueryKind::Letters, 5).expect("choice is unaffected");

    // The same head once the manifest says it was pre-registered as the primary row: answered.
    edit_release_manifest(&blend, |m| {
        m["span_head"]["averaged_head_preregistration"] =
            json!({"file": "v6-averaging-preregistered.json", "sha256": "ab".repeat(32), "name": "v6-averaging", "row": "primary"});
    });
    let backend = start(&blend);
    assert_eq!(decode(&backend, &refc, "span_suffix", QueryKind::PointerEnd, rows).unwrap().values.len(), rows);

    // No head in the release at all: the old refusal, still.
    let (_keep, none) = tiny_release("gpunone", |_| {});
    std::fs::remove_file(none.join("span_head.safetensors")).unwrap();
    let backend = start(&none);
    let err = decode(&backend, &refc, "span_suffix", QueryKind::PointerStart, rows).unwrap_err().to_string();
    assert!(err.contains("base weights have no pointer head"), "{err}");
}

#[test]
#[ignore = "GPU"]
fn gpu_a_context_whose_lines_cannot_be_mapped_refuses_its_span_slot_and_still_answers_choice() {
    let refc = Reference::load();
    let (_keep, release) = tiny_release("gpumap", |_| {});
    let backend = start(&release);
    // A blank line: with this tokenizer no token holds the byte a blank line starts at.
    let prefix = refc.str("prefix").replace("total = a + b\n", "total = a + b\n\n");
    assert_ne!(prefix, refc.str("prefix"));
    let rows = refc.usize("line_count") + 2;
    let handle = backend.prefill(&prefix).expect("a prefix whose lines cannot be mapped still prefills");
    let mut snapshot = backend.snapshot(&handle).unwrap();
    let ask = |snapshot: &mut qd_runtime::backend::StateSnapshot, key: &str, kind: QueryKind, rows: usize| {
        let query = SlotQuery { slot_name: "slot", suffix: refc.str(key), rows, kind };
        backend.decode_slot(snapshot, &query, DecodeMode::ReadOnly)
    };
    let err = ask(&mut snapshot, "span_suffix", QueryKind::PointerStart, rows).unwrap_err().to_string();
    assert!(err.contains("lies in no token"), "{err}");
    assert!(ask(&mut snapshot, "choice_suffix", QueryKind::Letters, 5).is_ok(), "choice is unaffected");
}
