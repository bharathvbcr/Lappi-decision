//! GPU gates for D17, the 17-row readout delta: `embed[answer ids] + D17`, on the tiny fixture
//! tower (`tools/gen_span_fixture.py`) and on the real 2B snapshot.
//!
//! **Every test here is `#[ignore]`d**: it runs on the GPU. `cargo test` never runs them and
//! reports them as ignored, not passed.
//!
//! ```text
//! cargo test --release -p qd-metal --test gpu_d17 -- --ignored --test-threads=1
//! ```
//!
//! The reference is a bf16 tower on CPU in torch; tessl runs bf16 operand GEMMs on the GPU, so the
//! two agree to bf16 noise (`SCORE_REL` of the largest logit), while D17 = 0 against today's
//! readout on the same GPU is compared bit for bit.

mod common;

use common::{
    HIDDEN, Reference, SCORE_REL, decode, edit_release_manifest, rel_diff, start, tiny_release,
};
use qd_metal::model::Model;
use qd_metal::tokenizer::QwenTokenizer;
use qd_metal::weights::AnswerDelta;
use qd_runtime::backend::{DecisionBackend, QueryKind};
use serde_json::Value;

#[test]
#[ignore = "GPU"]
fn gpu_d17_zero_reproduces_todays_scores_bit_for_bit_and_a_real_d17_equals_pythons_rows() {
    let refc = Reference::load();
    let (_keep, release) = tiny_release("gpud17", |_| {});
    let rt = tessl::GpuRuntime::new().expect("runtime");
    let mut model = Model::load(&rt, &release).expect("load");
    assert!(
        !model.has_answer_delta(),
        "a plain Model::load carries no D17: today's readout"
    );
    let tok = QwenTokenizer::load(&release.join("tokenizer.json")).unwrap();
    let text = format!("{}{}", refc.str("prefix"), refc.str("choice_suffix"));
    let ids = tok.encode_untrusted(&text).unwrap();
    let seq = u32::try_from(ids.len()).unwrap();
    let out = model.run(&ids, 1, seq, None, false, None).unwrap();
    let answers = tok.answer_ids(5).unwrap();
    let today = model.score(&out, &[seq - 1], &answers).unwrap();

    // D17 = 0.
    let zero = AnswerDelta::from_values(HIDDEN, vec![0.0; 17 * HIDDEN]).unwrap();
    model.set_answer_delta(zero, *tok.letter_ids()).unwrap();
    let with_zero = model.score(&out, &[seq - 1], &answers).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&with_zero.logits),
        bits(&today.logits),
        "logits: D17 = 0 is the tied embedding, bit for bit"
    );
    assert_eq!(
        bits(&with_zero.logprobs),
        bits(&today.logprobs),
        "log-softmax likewise"
    );
    // A second score with the same answers reuses the composed head and still agrees.
    assert_eq!(
        bits(&model.score(&out, &[seq - 1], &answers).unwrap().logits),
        bits(&today.logits)
    );

    // Today's scores against Python's tied-embedding rows.
    let base_d = rel_diff(&today.logits, &refc.floats("choice_logits_base"));
    println!("tied-embedding letter logits vs Python: {base_d:.5}");
    assert!(base_d <= SCORE_REL, "{base_d}");

    // The real D17 of the fixture.
    let delta = AnswerDelta::load(&release.join("d17.safetensors"), HIDDEN).unwrap();
    model.set_answer_delta(delta, *tok.letter_ids()).unwrap();
    let with_d17 = model.score(&out, &[seq - 1], &answers).unwrap();
    let d17_d = rel_diff(&with_d17.logits, &refc.floats("choice_logits_d17"));
    println!("embed + D17 letter logits vs Python AnswerRowDelta: {d17_d:.5}");
    assert!(d17_d <= SCORE_REL, "{d17_d}");
    assert_ne!(
        bits(&with_d17.logits),
        bits(&today.logits),
        "the fixture's D17 must move the scores"
    );

    // A different answer set recomposes: the 3-row set (A, B, noul) scores the same logits those
    // letters had in the 5-row set, which the cache for the 5-row set must not leak into.
    let three = model
        .score(&out, &[seq - 1], &tok.answer_ids(3).unwrap())
        .unwrap();
    assert_eq!(
        bits(&three.logits),
        bits(&[with_d17.logits[0], with_d17.logits[1], with_d17.logits[4]]),
        "a different answer set recomposes its rows"
    );

    // An id that is not one of the 17 letters (id 0, [UNK]) has no D17 row.
    let err = model
        .score(&out, &[seq - 1], &[answers[0], 0])
        .unwrap_err()
        .to_string();
    assert!(err.contains("not one of the 17 answer letters"), "{err}");
}

#[test]
#[ignore = "GPU"]
fn gpu_a_release_with_d17_serves_the_letter_rows_through_the_backend() {
    let refc = Reference::load();
    let (_keep, release) = tiny_release("gpubackend", |_| {});
    let backend = start(&release);
    let got = decode(&backend, &refc, "choice_suffix", QueryKind::Letters, 5).unwrap();
    let d = rel_diff(&got.values, &refc.floats("choice_logits_d17"));
    println!("backend letter logits (embed + D17) vs Python: {d:.5}");
    assert!(d <= SCORE_REL, "{d}");
    let id = backend.identity().head_hash.clone();
    // A release with no D17 and no served head reports the head hash it always did, and this one
    // does not.
    let (_k2, plain) = tiny_release("gpuplain", |m| {
        m.as_object_mut().unwrap().remove("span_head_source");
    });
    std::fs::remove_file(plain.join("d17.safetensors")).unwrap();
    edit_release_manifest(&plain, |m| {
        m["d17"] = Value::Null;
        m["files"]
            .as_object_mut()
            .unwrap()
            .remove("d17.safetensors");
    });
    let plain_backend = start(&plain);
    assert_ne!(
        plain_backend.identity().head_hash,
        id,
        "D17 and a served span head are part of the head's identity"
    );
}

/// The same bit-for-bit claim on the real 2B snapshot: 2048-wide rows out of a 248,320-row bf16
/// table, 17 letters, a real prompt.
#[test]
#[ignore = "GPU + model snapshot"]
fn gpu_snapshot_d17_zero_is_the_tied_embedding_on_the_real_2b() {
    let snapshot = qd_metal::config::resolve_snapshot(None).expect("snapshot");
    let rt = tessl::GpuRuntime::new().expect("runtime");
    let mut model = Model::load(&rt, &snapshot).expect("load");
    let tok = QwenTokenizer::load(&snapshot.join("tokenizer.json")).unwrap();
    let text = "<|qd_begin|>\nfile: bowling_test.go\n@@ -3,7 +3,7 @@\n-\tfor i := 0; i < runs; i++ {\n+\tfor i := 0; i <= runs; i++ {\n\
                <|qd_slot|>defect_class\n<|qd_type|>choice\n<|qd_options_begin|>\nA. stub\nB. logic\nC. cosmetic\nD. clean\nZ. noul\n\
                <|qd_options_end|>\n<|qd_answer|>";
    let ids = tok.encode_untrusted(text).unwrap();
    let seq = u32::try_from(ids.len()).unwrap();
    let out = model.run(&ids, 1, seq, None, false, None).unwrap();
    let hidden = model.config().hidden;
    let answers = tok.answer_ids(5).unwrap();
    let today = model.score(&out, &[seq - 1], &answers).unwrap();
    model
        .set_answer_delta(
            AnswerDelta::from_values(hidden, vec![0.0; 17 * hidden]).unwrap(),
            *tok.letter_ids(),
        )
        .unwrap();
    let zero = model.score(&out, &[seq - 1], &answers).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&zero.logits), bits(&today.logits), "logits");
    assert_eq!(bits(&zero.logprobs), bits(&today.logprobs), "log-softmax");
    println!("real 2B, D17 = 0: logits {:?}", today.logits);
}
