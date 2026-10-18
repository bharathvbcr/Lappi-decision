//! D17 as a host entry in the FT objective (`qd_train::d17`, `LetterSpanObjective::with_d17`).
//!
//! The readout is `R = E[answer ids] + D17`, a 17-way cross-entropy at each letter position,
//! `dz = (softmax - onehot) * scale`, `dh = dz R`, `dD17 = dz^T h`. What each test pins:
//!
//! 1. loss, `dh`, `dD17` and the logits equal `qd_train.lora.AnswerRowDelta` (the oracle,
//!    `tools/gen_d17_objective_fixture.py`) on five cases: N = 1, all 17 targets, repeated
//!    targets, noul only, and H = 64. Gates are below.
//! 2. `D17 = 0` gives exactly the base letter-row logits.
//! 3. a target outside the answer rows, duplicate or out-of-vocabulary ids, a wrong delta
//!    shape, a NaN input and a NaN delta are refused, never skipped.
//! 4. the report-only full-vocabulary CE (`RowTargets` scale 0) has a value and contributes
//!    exactly zero gradient.
//! 5. where a D17 position and a span-head position coincide, the summed hidden gradient is the
//!    sum of the separate gradients, and the model's bank is the sum of the two runs.
//! 6. random shapes `(H, N)` match a direct f64 reference computed in the test.
//!
//! # The gates (written before the first run of the implementation)
//!
//! * Against the oracle's float64 arm (`ref64_*`), for every tensor of every case:
//!   `max |rust - ref64| <= 1e-6 * max |ref64|`; for the loss `|rust - ref64| <= 1e-6 * |ref64|`.
//!   The implementation accumulates in f64 and rounds the outputs to f32 once, so the expected
//!   error is about 6e-8; 1e-6 is the same gate the span head carries (`span_head.rs`).
//! * Against torch float32 (the arm that trains), `<= 2e-5 * max |torch|`: torch's f32 matmul
//!   and softmax round at every step, about `H * 6e-8` at H <= 64, so 2e-5 is 5x that bound. It
//!   is a sanity gate on the Python rule; the f64 gate is the one that measures this code. An
//!   exact bit match with torch f32 is not claimed: summation order differs.

mod common;

use std::path::PathBuf;

use common::pins::{self, Form};
use common::{ToyBatch, ToyRow, ToySpan, ToySpanHead};
use qd_export::layout::{D17_PREFIX, D17_TENSOR};
use qd_export::safetensors::{Dtype, PlannedTensor, Writer};
use qd_train::d17::{D17, D17_ENTRY, D17_ROWS, D17Error, HiddenPart, merge_hidden_grads};
use qd_train::mock::ToyProvider;
use qd_train::objective::{LetterSpanObjective, LetterTarget, SpanHead};
use qd_train::step::{BankMode, HiddenGrad, RowTargets, SequenceJob, StepProvider};
use qd_train::trainer::{HostParams, Objective, TrainError};
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    pins::verified(
        &common::crate_dir()
            .join("tests")
            .join("fixtures")
            .join("d17-objective"),
        Form::Sha256Sums,
    )
}

fn doc() -> Value {
    let path = fixture_dir().join("cases.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

fn f32s(v: &Value) -> Vec<f32> {
    u64s(v).iter().map(|&b| f32::from_bits(b as u32)).collect()
}

fn f64s(v: &Value) -> Vec<f64> {
    u64s(v).iter().map(|&b| f64::from_bits(b)).collect()
}

fn bf16s(v: &Value) -> Vec<f32> {
    u64s(v)
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}

fn ids17(v: &Value) -> [u32; D17_ROWS] {
    let ids: Vec<u32> = u64s(v).iter().map(|&x| x as u32).collect();
    ids.try_into().expect("17 ids")
}

/// `max |a - b|` over `max |b|`, with `b` the reference.
fn rel_to_max(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let diff = a
        .iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((f64::from(*x) - y).abs()));
    diff / scale.max(f64::MIN_POSITIVE)
}

struct Case {
    d17: D17,
    h: Vec<f32>,
    targets: Vec<u32>,
    n: usize,
    v: Value,
}

fn load_case(v: &Value) -> Case {
    let hidden = v["hidden"].as_u64().unwrap() as usize;
    let vocab = v["vocab"].as_u64().unwrap() as usize;
    let mut d17 = D17::from_embedding(&bf16s(&v["embed"]), vocab, hidden, ids17(&v["ids"]))
        .expect("the fixture's embedding and ids are valid");
    d17.set_delta(&f32s(&v["delta"]))
        .expect("the fixture's delta fits");
    Case {
        d17,
        h: f32s(&v["h"]),
        targets: u64s(&v["targets"]).iter().map(|&x| x as u32).collect(),
        n: v["n"].as_u64().unwrap() as usize,
        v: v.clone(),
    }
}

// ---- 1. the oracle -------------------------------------------------------------------------

#[test]
fn loss_dh_dd17_and_logits_match_answer_row_delta_on_every_case() {
    let doc = doc();
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        5,
        "the generator writes five cases; none is dropped"
    );
    for raw in cases {
        let mut c = load_case(raw);
        let name = raw["name"].as_str().unwrap();
        let logits = c.d17.logits(&c.h).unwrap();
        c.d17.zero_grads();
        let scale = (1.0 / c.n as f64) as f32;
        let (loss_sum, dh) = c.d17.loss_and_grad(&c.h, &c.targets, scale).unwrap();
        let loss = loss_sum / c.n as f64;
        let d_delta: Vec<f32> = c.d17.grads()[0].expect("D17 reached").to_vec();

        let ref_loss = f64::from_bits(u64s(&c.v["ref64_loss"])[0]);
        assert!(
            (loss - ref_loss).abs() <= 1e-6 * ref_loss.abs(),
            "{name}: loss {loss} vs float64 {ref_loss}"
        );
        for (what, got, want) in [
            ("logits", &logits, f64s(&c.v["ref64_logits"])),
            ("dh", &dh, f64s(&c.v["ref64_dh"])),
            ("dD17", &d_delta, f64s(&c.v["ref64_d_delta"])),
        ] {
            let rel = rel_to_max(got, &want);
            println!("{name} {what}: rust vs float64 {rel:.3e}");
            assert!(
                rel <= 1e-6,
                "{name} {what}: {rel:.3e} of max vs float64 exceeds 1e-6"
            );
        }

        let torch_loss = f64::from(f32::from_bits(u64s(&c.v["loss"])[0] as u32));
        assert!(
            (loss - torch_loss).abs() <= 2e-5 * torch_loss.abs(),
            "{name}: loss {loss} vs torch f32 {torch_loss}"
        );
        for (what, got, key) in [
            ("logits", &logits, "logits"),
            ("dh", &dh, "dh"),
            ("dD17", &d_delta, "d_delta"),
        ] {
            let want: Vec<f64> = f32s(&c.v[key]).iter().map(|&x| f64::from(x)).collect();
            let rel = rel_to_max(got, &want);
            println!("{name} {what}: rust vs torch f32 {rel:.3e}");
            assert!(
                rel <= 2e-5,
                "{name} {what}: {rel:.3e} of max vs torch f32 exceeds 2e-5"
            );
        }
    }
}

#[test]
fn the_oracle_covers_n1_all_17_targets_and_repeated_targets() {
    let doc = doc();
    let by_name = |n: &str| -> Vec<u64> {
        let c = doc["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == n)
            .unwrap();
        u64s(&c["target_rows"])
    };
    assert_eq!(by_name("n1").len(), 1);
    let mut all = by_name("all17");
    all.sort_unstable();
    assert_eq!(all, (0..17).collect::<Vec<u64>>());
    let rep = by_name("repeated");
    let mut distinct = rep.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(distinct.len() < rep.len(), "a target repeats");
}

#[test]
fn the_entry_is_named_d17_delta_as_the_export_and_the_averager_name_it() {
    assert_eq!(D17_ENTRY, format!("{D17_PREFIX}{D17_TENSOR}"));
    assert_eq!(D17_ROWS, qd_export::layout::D17_ROWS);
    let c = load_case(&doc()["cases"][0]);
    let entries = c.d17.entries();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "d17.delta");
    assert_eq!(entries[0].shape, vec![17, 16]);
}

// ---- 2. D17 = 0 ----------------------------------------------------------------------------

#[test]
fn a_zero_d17_gives_exactly_the_base_letter_row_logits() {
    for raw in doc()["cases"].as_array().unwrap() {
        let mut c = load_case(raw);
        let hidden = c.d17.hidden();
        c.d17.set_delta(&vec![0.0; D17_ROWS * hidden]).unwrap();
        let embed = bf16s(&raw["embed"]);
        let ids = ids17(&raw["ids"]);
        let got = c.d17.logits(&c.h).unwrap();
        for r in 0..c.n {
            for (k, id) in ids.iter().enumerate() {
                let row = &embed[*id as usize * hidden..(*id as usize + 1) * hidden];
                let want: f64 = c.h[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(row)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                assert_eq!(
                    got[r * D17_ROWS + k].to_bits(),
                    (want as f32).to_bits(),
                    "{} position {r} letter {k}",
                    raw["name"]
                );
            }
        }
    }
}

// ---- 3. refusals ---------------------------------------------------------------------------

fn small() -> (Vec<f32>, [u32; D17_ROWS]) {
    let table: Vec<f32> = (0..40 * 4)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let ids: [u32; D17_ROWS] = std::array::from_fn(|k| (k * 2) as u32);
    (table, ids)
}

#[test]
fn a_target_outside_the_answer_rows_is_refused_not_skipped() {
    let (table, ids) = small();
    let mut d = D17::from_embedding(&table, 40, 4, ids).unwrap();
    let h = vec![0.5f32; 8];
    // 1 is a vocabulary id but not an answer id; a skip would shrink N and lower the loss.
    let err = d.loss_and_grad(&h, &[2, 1], 0.5).unwrap_err();
    assert!(
        matches!(err, D17Error::TargetNotAnswerRow { target: 1, .. }),
        "{err}"
    );
    assert!(
        d.grads()[0].is_none(),
        "a refused call leaves no gradient behind"
    );
}

#[test]
fn duplicate_and_out_of_vocabulary_answer_ids_are_refused() {
    let (table, mut ids) = small();
    ids[5] = ids[2];
    assert!(matches!(
        D17::from_embedding(&table, 40, 4, ids),
        Err(D17Error::DuplicateId { .. })
    ));
    let (table, mut ids) = small();
    ids[16] = 40;
    assert!(matches!(
        D17::from_embedding(&table, 40, 4, ids),
        Err(D17Error::IdOutOfVocab { id: 40, vocab: 40 })
    ));
}

#[test]
fn a_wrong_shape_is_refused_everywhere_it_can_enter() {
    let (table, ids) = small();
    assert!(matches!(
        D17::from_embedding(&table[..table.len() - 1], 40, 4, ids),
        Err(D17Error::Shape(_))
    ));
    assert!(D17::from_embedding(&table, 40, 0, ids).is_err());
    let mut d = D17::from_embedding(&table, 40, 4, ids).unwrap();
    assert!(matches!(
        d.set_delta(&vec![0.0; 17 * 4 - 1]),
        Err(D17Error::Shape(_))
    ));
    assert!(matches!(
        d.set_delta(&vec![0.0; 17 * 5]),
        Err(D17Error::Shape(_))
    ));
    assert!(matches!(d.logits(&[0.0; 5]), Err(D17Error::Shape(_))));
    assert!(matches!(
        d.loss_and_grad(&[0.0; 8], &[0], 1.0),
        Err(D17Error::Shape(_))
    ));
    assert!(matches!(
        d.loss_and_grad(&[], &[], 1.0),
        Err(D17Error::Shape(_))
    ));
}

#[test]
fn non_finite_inputs_are_refused() {
    let (mut table, ids) = small();
    let mut d = D17::from_embedding(&table, 40, 4, ids).unwrap();
    let mut h = vec![0.5f32; 4];
    h[2] = f32::NAN;
    assert!(matches!(
        d.loss_and_grad(&h, &[0], 1.0),
        Err(D17Error::NonFinite(_))
    ));
    h[2] = f32::INFINITY;
    assert!(matches!(d.logits(&h), Err(D17Error::NonFinite(_))));
    let mut delta = vec![0.0f32; 17 * 4];
    delta[3] = f32::NAN;
    assert!(matches!(d.set_delta(&delta), Err(D17Error::NonFinite(_))));
    // An optimizer that wrote a NaN through values_and_grads is caught at the next use.
    d.values_and_grads()[0].0[0] = f32::NAN;
    assert!(matches!(
        d.loss_and_grad(&[0.5; 4], &[0], 1.0),
        Err(D17Error::NonFinite(_))
    ));
    // A scale that is negative or NaN is refused.
    let mut ok = D17::from_embedding(&table, 40, 4, ids).unwrap();
    assert!(matches!(
        ok.loss_and_grad(&[0.5; 4], &[0], -1.0),
        Err(D17Error::Scale(_))
    ));
    assert!(matches!(
        ok.loss_and_grad(&[0.5; 4], &[0], f32::NAN),
        Err(D17Error::Scale(_))
    ));
    // A non-finite embedding row is refused at construction.
    table[ids[3] as usize * 4 + 1] = f32::NEG_INFINITY;
    assert!(matches!(
        D17::from_embedding(&table, 40, 4, ids),
        Err(D17Error::NonFinite(_))
    ));
}

#[test]
fn the_answer_rows_are_read_by_tensor_name_from_a_checkpoint() {
    let (table, ids) = small();
    let dir = common::scratch("d17-ckpt");
    let path = dir.join("model.safetensors");
    let plan = vec![
        PlannedTensor {
            name: "model.embed_tokens.weight".into(),
            dtype: Dtype::F32,
            shape: vec![40, 4],
        },
        PlannedTensor {
            name: "decoy".into(),
            dtype: Dtype::F32,
            shape: vec![2, 4],
        },
    ];
    let mut w = Writer::create(&path, plan).unwrap();
    for (name, values) in [
        ("decoy", vec![9.0f32; 8]),
        ("model.embed_tokens.weight", table.clone()),
    ] {
        w.begin(name).unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write(&bytes).unwrap();
        w.end().unwrap();
    }
    w.finish().unwrap();

    let d = D17::from_safetensors(&path, "model.embed_tokens.weight", ids).unwrap();
    let want = D17::from_embedding(&table, 40, 4, ids).unwrap();
    assert_eq!(d.answer_rows(), want.answer_rows());
    assert!(
        matches!(
            D17::from_safetensors(&path, "embed_tokens.weight", ids),
            Err(D17Error::Checkpoint(_))
        ),
        "a missing name is refused, not guessed"
    );
    assert!(matches!(
        D17::from_safetensors(&path, "decoy", ids),
        Err(D17Error::IdOutOfVocab { .. })
    ));
    std::fs::remove_dir_all(&dir).unwrap();
}

// ---- 4. the report-only full-vocabulary CE -------------------------------------------------

const V: usize = 40;
const H: usize = 8;

fn letter_row(tokens: Vec<u32>, position: u32) -> ToyRow {
    let target = tokens[position as usize + 1];
    ToyRow {
        tokens,
        letter: Some(LetterTarget { position, target }),
        span: None,
    }
}

/// Rows whose letter targets are answer ids `ids[k]`, with the answer row `k` chosen per row.
fn letter_batch(ids: &[u32; D17_ROWS], picks: &[usize]) -> ToyBatch {
    let mut rows = Vec::new();
    for (r, &k) in picks.iter().enumerate() {
        let tokens: Vec<u32> = vec![
            (r as u32 * 3 + 1) % V as u32,
            (r as u32 * 5 + 2) % V as u32,
            (r as u32 * 7 + 3) % V as u32,
            ids[k],
        ];
        rows.push(letter_row(tokens, 2));
    }
    ToyBatch {
        index: 0,
        bucket: 4,
        width: 4,
        rows,
    }
}

fn objective(d17: D17) -> LetterSpanObjective<ToyBatch, ToySpanHead> {
    LetterSpanObjective::new(ToySpanHead::new(H), 0.5)
        .unwrap()
        .with_d17(d17)
}

/// Drive one planned batch through the provider exactly as the loop does.
fn drive(
    p: &mut ToyProvider,
    o: &mut LetterSpanObjective<ToyBatch, ToySpanHead>,
    b: &ToyBatch,
) -> qd_train::trainer::MicroLoss {
    let plan = o.plan(b).unwrap();
    let mut sums = Vec::new();
    for (i, s) in plan.sequences.iter().enumerate() {
        let job = SequenceJob {
            tokens: &s.tokens,
            rows: RowTargets {
                positions: &s.positions,
                targets: &s.targets,
                scale: plan.row_scale,
            },
            hidden_at: &s.hidden_at,
        };
        let mode = if i == 0 {
            BankMode::Overwrite
        } else {
            BankMode::Add
        };
        let mut cb = |pos: &[u32], h: &[f32]| -> Result<Vec<f32>, qd_train::step::StepError> {
            o.hidden_grad(i, pos, h, H)
                .map_err(|e| qd_train::step::StepError::Contract(e.to_string()))
        };
        let ext: Option<&mut HiddenGrad<'_>> = if s.hidden_at.is_empty() {
            None
        } else {
            Some(&mut cb)
        };
        sums.push(p.accumulate(&job, mode, ext).unwrap());
    }
    o.close(&sums).unwrap()
}

fn d17_from_provider(p: &ToyProvider, ids: [u32; D17_ROWS]) -> D17 {
    D17::from_embedding(&p.values()[0], V, H, ids).unwrap()
}

#[test]
fn the_full_vocabulary_ce_is_report_only_a_value_and_exactly_zero_gradient() {
    let ids: [u32; D17_ROWS] = std::array::from_fn(|k| (k * 2 + 1) as u32);
    let mut p = ToyProvider::new(V, H, 1, 11).unwrap();
    let b = letter_batch(&ids, &[0, 5, 16]);
    let mut o = objective(d17_from_provider(&p, ids));
    let plan = o.plan(&b).unwrap();
    assert_eq!(
        plan.row_scale, 0.0,
        "the letter channel is report-only once D17 trains it"
    );
    for s in &plan.sequences {
        assert_eq!(
            s.positions.len(),
            1,
            "the full-vocabulary CE is still asked for"
        );
        assert_eq!(s.hidden_at, s.positions, "and D17 reads the same position");
    }

    // The provider with scale 0 and no outside loss: a value, zero gradient.
    let s = &plan.sequences[0];
    let job = SequenceJob {
        tokens: &s.tokens,
        rows: RowTargets {
            positions: &s.positions,
            targets: &s.targets,
            scale: 0.0,
        },
        hidden_at: &[],
    };
    let value = p.accumulate(&job, BankMode::Overwrite, None).unwrap();
    assert!(value > 0.0 && value.is_finite(), "a report value: {value}");
    assert_eq!(
        p.grad_sq_norm().unwrap(),
        0.0,
        "scale 0 contributes no gradient"
    );
    assert!(
        p.grads()
            .iter()
            .flatten()
            .all(|g| g.to_bits() == 0.0f32.to_bits())
    );

    // Through the objective: the bank equals the bank of the same sequence with no CE rows.
    let mut p_a = ToyProvider::new(V, H, 1, 11).unwrap();
    let mut o_a = objective(d17_from_provider(&p_a, ids));
    let loss = drive(&mut p_a, &mut o_a, &b);
    let mut p_b = ToyProvider::new(V, H, 1, 11).unwrap();
    let mut o_b = objective(d17_from_provider(&p_b, ids));
    let plan_b = o_b.plan(&b).unwrap();
    for (i, s) in plan_b.sequences.iter().enumerate() {
        let job = SequenceJob {
            tokens: &s.tokens,
            rows: RowTargets {
                positions: &[],
                targets: &[],
                scale: 0.0,
            },
            hidden_at: &s.hidden_at,
        };
        let mode = if i == 0 {
            BankMode::Overwrite
        } else {
            BankMode::Add
        };
        let mut cb = |pos: &[u32], h: &[f32]| -> Result<Vec<f32>, qd_train::step::StepError> {
            o_b.hidden_grad(i, pos, h, H)
                .map_err(|e| qd_train::step::StepError::Contract(e.to_string()))
        };
        p_b.accumulate(&job, mode, Some(&mut cb)).unwrap();
    }
    for (a, b) in p_a.grads().iter().zip(p_b.grads()) {
        let (a, b): (Vec<u32>, Vec<u32>) = (
            a.iter().map(|x| x.to_bits()).collect(),
            b.iter().map(|x| x.to_bits()).collect(),
        );
        assert_eq!(a, b, "the scale-0 rows changed a gradient bit");
    }

    let channel = |name: &str| loss.channels.iter().find(|(n, _)| n == name).unwrap().1;
    assert!(
        channel("letter") > 0.0,
        "the full-vocabulary CE is reported"
    );
    assert!(channel("d17") > 0.0);
    assert_eq!(
        loss.total,
        channel("d17"),
        "the total trains D17 only; the report is not in it"
    );
    assert!(
        o_a.host_ref().unwrap().grads().last().unwrap().is_some(),
        "D17 got a gradient"
    );
}

#[test]
fn the_objective_d17_loss_is_the_mean_17_way_ce_and_its_entry_trails_the_head() {
    let ids: [u32; D17_ROWS] = std::array::from_fn(|k| (k * 2 + 1) as u32);
    let mut p = ToyProvider::new(V, H, 1, 3).unwrap();
    let b = letter_batch(&ids, &[2, 2, 9, 16]);
    let mut o = objective(d17_from_provider(&p, ids));
    let names: Vec<String> = o
        .host_ref()
        .unwrap()
        .entries()
        .iter()
        .map(|e| e.name.clone())
        .collect();
    assert_eq!(names, ["pointer.weight", "abstain", "d17.delta"]);
    let loss = drive(&mut p, &mut o, &b);

    // Reference: the mean 17-way CE of the final hidden state at each letter position.
    let probe = d17_from_provider(&p, ids);
    let mut want = 0.0f64;
    for (r, row) in b.rows.iter().enumerate() {
        let x = &p.hidden_states(&row.tokens)[2];
        let k = [2usize, 2, 9, 16][r];
        let mut one = probe.clone();
        let (l, _) = one.loss_and_grad(x, &[ids[k]], 1.0).unwrap();
        want += l;
    }
    want /= 4.0;
    let got = loss.channels.iter().find(|(n, _)| n == "d17").unwrap().1;
    assert!((got - want).abs() <= 1e-6 * want.abs(), "{got} vs {want}");
}

#[test]
fn a_letter_target_that_is_not_an_answer_id_refuses_the_batch() {
    let ids: [u32; D17_ROWS] = std::array::from_fn(|k| (k * 2 + 1) as u32);
    let p = ToyProvider::new(V, H, 1, 3).unwrap();
    let mut b = letter_batch(&ids, &[0, 1]);
    b.rows[1].tokens[3] = 4; // an even id: not an answer row
    b.rows[1].letter = Some(LetterTarget {
        position: 2,
        target: 4,
    });
    let mut o = objective(d17_from_provider(&p, ids));
    // Refused at plan time, before any sequence runs, naming the row; never skipped.
    let err = o.plan(&b).unwrap_err();
    assert!(
        matches!(&err, TrainError::Refused(m) if m.contains("row 1") && m.contains("answer row")),
        "{err}"
    );
    // Without D17 the same batch plans: the full-vocabulary channel has no such restriction.
    let mut plain =
        LetterSpanObjective::<ToyBatch, ToySpanHead>::new(ToySpanHead::new(H), 0.5).unwrap();
    assert!(plain.plan(&b).is_ok());
    // And a provider that hands the objective the wrong hidden width is refused.
    let good = letter_batch(&ids, &[0, 1]);
    let mut o2 = objective(d17_from_provider(&p, ids));
    o2.plan(&good).unwrap();
    let wide = vec![0.1f32; H + 1];
    assert!(matches!(
        o2.hidden_grad(0, &[2], &wide, H + 1),
        Err(TrainError::Refused(_))
    ));
}

// ---- 5. coinciding positions ---------------------------------------------------------------

#[test]
fn a_coinciding_span_and_d17_position_sums_the_hidden_gradient_once() {
    let ids: [u32; D17_ROWS] = std::array::from_fn(|k| (k * 2 + 1) as u32);
    let mut p = ToyProvider::new(V, H, 1, 21).unwrap();
    let tokens: Vec<u32> = vec![3, 5, 7, ids[4], 9, 11];
    // The D17 reads position 2 (target ids[4]); the span head reads positions 1, 2, 4 (2 coincides).
    let d17_pos = [2u32];
    let span_pos = [1u32, 2, 4];
    let union = [1u32, 2, 4];
    let x: Vec<Vec<f32>> = p.hidden_states(&tokens);
    let block =
        |pos: &[u32]| -> Vec<f32> { pos.iter().flat_map(|&q| x[q as usize].clone()).collect() };

    let mut d17 = d17_from_provider(&p, ids);
    let (_, dh_d17) = d17
        .loss_and_grad(&block(&d17_pos), &[ids[4]], 0.25)
        .unwrap();
    let mut head = ToySpanHead::new(H);
    let span = ToySpan {
        candidates: span_pos.to_vec(),
        start: 0,
        end: 1,
    };
    let (_, dh_span) = head
        .loss_and_grad(&span, &span_pos, &block(&span_pos), H, 0.25)
        .unwrap();

    let merged = merge_hidden_grads(
        &union,
        &[
            HiddenPart {
                positions: &d17_pos,
                grad: &dh_d17,
            },
            HiddenPart {
                positions: &span_pos,
                grad: &dh_span,
            },
        ],
        H,
    )
    .unwrap();
    assert_eq!(merged.len(), union.len() * H);
    for (u, &pos) in union.iter().enumerate() {
        for i in 0..H {
            let mut want = 0.0f32;
            if let Some(k) = span_pos.iter().position(|&q| q == pos) {
                want += dh_span[k * H + i];
            }
            if let Some(k) = d17_pos.iter().position(|&q| q == pos) {
                want += dh_d17[k * H + i];
            }
            assert_eq!(
                merged[u * H + i].to_bits(),
                want.to_bits(),
                "position {pos} element {i}"
            );
        }
    }
    // Position 2's merged gradient is genuinely a sum, not either part alone.
    assert_ne!(merged[H].to_bits(), dh_span[H].to_bits());

    // In the model: one run with the merged callback equals the two runs' banks added.
    let run = |p: &mut ToyProvider, hidden_at: &[u32], grad: Vec<f32>| {
        let job = SequenceJob {
            tokens: &tokens,
            rows: RowTargets {
                positions: &[],
                targets: &[],
                scale: 0.0,
            },
            hidden_at,
        };
        let mut cb = |_: &[u32], _: &[f32]| -> Result<Vec<f32>, qd_train::step::StepError> {
            Ok(grad.clone())
        };
        p.accumulate(&job, BankMode::Overwrite, Some(&mut cb))
            .unwrap();
        p.grads().to_vec()
    };
    let both = run(&mut p, &union, merged);
    let mut zero_span = vec![0.0f32; union.len() * H];
    zero_span[H..2 * H].copy_from_slice(&dh_d17); // d17's gradient at position 2, zeros elsewhere
    let only_d17 = run(&mut p, &union, zero_span);
    let only_span = run(&mut p, &union, dh_span.clone());
    for ((b, d), s) in both.iter().zip(&only_d17).zip(&only_span) {
        for ((b, d), s) in b.iter().zip(d).zip(s) {
            assert!(
                (b - (d + s)).abs() <= 1e-5 * (1.0 + b.abs()),
                "bank: {b} vs {d} + {s}"
            );
        }
    }
}

#[test]
fn merging_refuses_a_part_outside_the_union_a_repeated_position_and_a_bad_length() {
    let g = vec![1.0f32; 2 * 4];
    assert!(matches!(
        merge_hidden_grads(
            &[0, 1],
            &[HiddenPart {
                positions: &[0, 9],
                grad: &g
            }],
            4
        ),
        Err(D17Error::Positions(_))
    ));
    assert!(matches!(
        merge_hidden_grads(
            &[0, 0],
            &[HiddenPart {
                positions: &[0, 1],
                grad: &g
            }],
            4
        ),
        Err(D17Error::Positions(_))
    ));
    assert!(matches!(
        merge_hidden_grads(
            &[0, 1],
            &[HiddenPart {
                positions: &[0, 1],
                grad: &g[..7]
            }],
            4
        ),
        Err(D17Error::Shape(_))
    ));
}

// ---- 6. random shapes against a direct f64 reference ---------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f64 {
        (self.next() % 2_000_001) as f64 / 1_000_000.0 - 1.0
    }
}

#[test]
fn random_shapes_match_a_direct_f64_reference() {
    let mut rng = Lcg(0x05EE_DD17);
    for trial in 0..200 {
        let hidden = 1 + (rng.next() % 40) as usize;
        let n = 1 + (rng.next() % 20) as usize;
        let vocab = 17 + (rng.next() % 30) as usize;
        let mut pool: Vec<u32> = (0..vocab as u32).collect();
        for i in (1..pool.len()).rev() {
            pool.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let ids: [u32; D17_ROWS] = std::array::from_fn(|k| pool[k]);
        let table: Vec<f32> = (0..vocab * hidden)
            .map(|_| (rng.unit() * 0.8) as f32)
            .collect();
        let delta: Vec<f32> = (0..D17_ROWS * hidden)
            .map(|_| (rng.unit() * 0.3) as f32)
            .collect();
        let h: Vec<f32> = (0..n * hidden).map(|_| (rng.unit() * 2.0) as f32).collect();
        let pick: Vec<usize> = (0..n).map(|_| (rng.next() % 17) as usize).collect();
        let targets: Vec<u32> = pick.iter().map(|&k| ids[k]).collect();
        let scale = (1.0 / n as f64) as f32;

        let mut d = D17::from_embedding(&table, vocab, hidden, ids).unwrap();
        d.set_delta(&delta).unwrap();
        d.zero_grads();
        let (loss_sum, dh) = d.loss_and_grad(&h, &targets, scale).unwrap();
        let dd = d.grads()[0].unwrap().to_vec();

        // The reference: R = E + D in f32 (as AnswerRowDelta.rows), everything else in f64.
        let r: Vec<f64> = (0..D17_ROWS * hidden)
            .map(|i| f64::from(table[ids[i / hidden] as usize * hidden + i % hidden] + delta[i]))
            .collect();
        let (mut ref_loss, mut ref_dh, mut ref_dd) = (
            0.0f64,
            vec![0.0f64; n * hidden],
            vec![0.0f64; D17_ROWS * hidden],
        );
        for row in 0..n {
            let hr = &h[row * hidden..(row + 1) * hidden];
            let z: Vec<f64> = (0..D17_ROWS)
                .map(|k| {
                    (0..hidden)
                        .map(|j| f64::from(hr[j]) * r[k * hidden + j])
                        .sum()
                })
                .collect();
            let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let se: f64 = z.iter().map(|v| (v - m).exp()).sum();
            ref_loss += m + se.ln() - z[pick[row]];
            for k in 0..D17_ROWS {
                let dz = ((z[k] - m).exp() / se - if k == pick[row] { 1.0 } else { 0.0 })
                    * f64::from(scale);
                for j in 0..hidden {
                    ref_dh[row * hidden + j] += dz * r[k * hidden + j];
                    ref_dd[k * hidden + j] += dz * f64::from(hr[j]);
                }
            }
        }
        assert!(
            (loss_sum - ref_loss).abs() <= 1e-9 * ref_loss.abs().max(1.0),
            "trial {trial} (H={hidden}, N={n}): loss {loss_sum} vs {ref_loss}"
        );
        for (what, got, want) in [("dh", &dh, &ref_dh), ("dD17", &dd, &ref_dd)] {
            let rel = rel_to_max(got, want);
            assert!(
                rel <= 1e-6,
                "trial {trial} (H={hidden}, N={n}) {what}: {rel:.3e}"
            );
        }
        // A second call adds to the gradient; zero_grads clears it.
        d.loss_and_grad(&h, &targets, scale).unwrap();
        let twice = d.grads()[0].unwrap();
        let rel = rel_to_max(twice, &ref_dd.iter().map(|v| 2.0 * v).collect::<Vec<_>>());
        assert!(rel <= 1e-6, "trial {trial}: accumulation {rel:.3e}");
        d.zero_grads();
        assert!(d.grads()[0].is_none());
    }
}
