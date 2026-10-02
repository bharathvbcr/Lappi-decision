//! Rung (b) on the GPU: the Rust trainer over ojas-qwen35/tessl against L-oracle's torch
//! reference on tessl's tiny Qwen3.5 (`crates/qd-train/tests/fixtures/tiny-published`), with
//! F's recipe at the tiny scale: lr 1e-5, layers 0-1 at 0.1x, span weight 1, clip 1.0,
//! `real_ft(1e-5, 20)`, one batch per step, the same 20 batches in the same order.
//!
//! **Bars, written before any run** (`AUDIT/ojas-training-2026-10-01/fable-advice.md` Q3 rung
//! (b), as Amendment 2 (i) clarifies the weights measure, main 931c320):
//! * ExactF32 vs torch fp32: per-step loss <= 1e-5 relative on steps 0-5, <= 1e-4 to step 20;
//!   final weights max|rust - ref| <= 1e-5 x the denominator, where the denominator is the
//!   tensor's own max|ref| for a tower tensor and the max over all `span_head.*` of max|ref|
//!   for a span-head tensor. Amendment 2 loosens the effective tolerance on `abstain_start`
//!   and `abstain_end` from about 4.8e-10 to about 1.25e-6 absolute, recorded as a gate change
//!   made before any Rust run. Every tensor is printed over its own max and over its
//!   displacement from the init;
//! * two Rust ExactF32 runs are bit-identical;
//! * Bf16 operands vs torch fp32: loss <= 2^-8 relative per step; vs torch master-bf16:
//!   report-only (printed).
//! * `grads_step0`: report-only at this rung, and not measured by this file (the loop does not
//!   expose the bank before the update).
//!
//! **Not runnable yet, and failing rather than passing when tried:** the reference trains
//! layers 0-1 at 0.1x, and tessl has no per-entry learning rate. The loop refuses the recipe
//! before the clock starts (`supports_lr_scale` is false), so the two parity tests fail with
//! that refusal until tessl's `lr_scale` lands. They never pass vacuously. The single-group
//! arm runs today as a **smoke test, not a parity test**: it uses one learning rate, so it is
//! a different recipe from the reference. It checks only that the run completes with finite
//! losses and repeats bit for bit, and it prints its distance from the reference for
//! information.
//!
//! The `max_grad_norm = 150` arm Amendment 2 (iii) adds waits on L-oracle's fixture for it.
//!
//! GPU only: `#[ignore]`, named `gpu_*`, run serialized in an agreed window:
//!
//! ```text
//! cargo test -p qd-train-metal --release --test gpu_rung_b -- --ignored --test-threads=1 --nocapture
//! ```

#![cfg(target_os = "macos")]

mod tiny_published;

use std::collections::BTreeMap;

use ojas_qwen35::Snapshot;
use qd_train::ft_data::{HostSpanHead, RealBatch};
use qd_train::objective::LetterSpanObjective;
use qd_train::recipe::{self, LowerLayers, OptimizerRecipe};
use qd_train::run_control::{MonotonicClock, WallClockCap};
use qd_train::schedule::LrSchedule;
use qd_train::step::StepProvider;
use qd_train::trainer::{train, HostParams, Hooks, Termination, TrainConfig, TrainError, TrainResult};
use qd_train_metal::args::Operands;
use qd_train_metal::provider::Qwen35Provider;
use serde_json::Value;
use tiny_published::{dir, init_head, json, tensors, Sequences};

const TOWER_PREFIX: &str = "model.language_model.";

struct Run {
    result: TrainResult,
    /// Final weights under the reference's names.
    weights: BTreeMap<String, Vec<f32>>,
}

/// The reference's recipe, read from its manifest and checked against qd-train's constants.
fn recipe_lower() -> LowerLayers {
    let r = &json("manifest.json")["recipe"];
    assert_eq!(r["lr"].as_f64(), Some(1e-5));
    assert_eq!(r["span_weight"].as_f64(), Some(1.0));
    assert_eq!(r["max_grad_norm"].as_f64(), Some(recipe::MAX_GRAD_NORM));
    assert_eq!(r["train_attention_mask"], "none", "tessl's step is unpadded");
    let s = &r["schedule"];
    let ours = LrSchedule::real_ft(1e-5, 20).unwrap();
    assert_eq!(s["total_steps"].as_u64(), Some(ours.total_steps()));
    assert_eq!(s["warmup_steps"].as_u64(), Some(ours.warmup_steps()));
    assert_eq!(s["min_lr"].as_f64().map(f64::to_bits), Some(ours.min_lr().to_bits()));
    assert_eq!(s["grad_accum"].as_u64(), Some(1));
    LowerLayers {
        n: r["lower_layers_n"].as_u64().unwrap() as usize,
        lr_scale: r["lower_layers_lr_scale"].as_f64().unwrap(),
    }
}

fn run(operands: Operands, lower: Option<LowerLayers>) -> Result<Run, TrainError> {
    let snap = Snapshot::from_files(&dir().join("config.json"), &dir().join("init.safetensors")).unwrap();
    let mut p = Qwen35Provider::open(&snap, operands).unwrap();
    let mut o: LetterSpanObjective<RealBatch, HostSpanHead> =
        LetterSpanObjective::new(HostSpanHead::new(init_head()).unwrap(), 1.0).unwrap();
    let cfg = TrainConfig {
        schedule: LrSchedule::real_ft(1e-5, 20).unwrap(),
        cap: WallClockCap::new(3600.0).unwrap(),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: 1e-5,
            beta2: recipe::DEFAULT_BETA2,
            lower_layers: lower,
        },
        max_grad_norm: recipe::MAX_GRAD_NORM,
        epoch: 0,
        seed: 0,
        checkpoint: None,
        eta: None,
    };
    let source = Sequences::load().batches().into_iter().map(RealBatch::new);
    let result = train(&mut p, &mut o, source, &cfg, &MonotonicClock::new(), None, &mut Hooks::default())?;
    assert_eq!(result.termination, Termination::StepsExhausted);
    assert_eq!(result.optimizer_steps, 20);
    let mut weights = BTreeMap::new();
    for (spec, v) in p.parameters().to_vec().iter().zip(p.read_parameters()?) {
        weights.insert(format!("{TOWER_PREFIX}{}", spec.name), v);
    }
    let head = o.head();
    for (spec, v) in head.entries().iter().zip(head.values()) {
        weights.insert(format!("span_head.{}", spec.name), v.to_vec());
    }
    Ok(Run { result, weights })
}

fn trajectory(arm: &str) -> Vec<Value> {
    json(&format!("{arm}/trajectory.json")).as_array().unwrap().clone()
}

fn max_abs(v: &[f32]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(f64::from(x.abs())))
}

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((f64::from(*x) - f64::from(*y)).abs()))
}

/// Per-step loss against `arm`'s trajectory: prints every step and returns the worst relative
/// difference on steps 0-5 and on all steps.
fn losses_vs(run: &Run, arm: &str) -> (f64, f64) {
    let t = trajectory(arm);
    assert_eq!(t.len(), run.result.steps.len());
    let (mut early, mut all) = (0.0f64, 0.0f64);
    for (s, (want, got)) in t.iter().zip(&run.result.steps).enumerate() {
        let w = want["loss_total"].as_f64().unwrap();
        let rel = (got.loss - w).abs() / w.abs();
        println!("  step {s:2}: rust {:.9} {arm} {w:.9} rel {rel:.3e}", got.loss);
        if s <= 5 {
            early = early.max(rel);
        }
        all = all.max(rel);
    }
    (early, all)
}

/// Amendment 2 (i)'s final-weights measure against `fp32/final.safetensors`. Prints every
/// tensor over its own max and over its displacement from the init; returns the worst gated
/// ratio and the tensor it is on.
fn final_weights(run: &Run) -> (f64, String) {
    let reference = tensors(&dir().join("fp32/final.safetensors"));
    let init = tensors(&dir().join("init.safetensors"));
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        run.weights.keys().collect::<Vec<_>>(),
        "every reference tensor is compared, and nothing else"
    );
    let head_max = reference
        .iter()
        .filter(|(n, _)| n.starts_with("span_head."))
        .fold(0.0f64, |m, (_, (_, v))| m.max(max_abs(v)));
    let mut worst = (0.0f64, String::new());
    for (name, (_, want)) in &reference {
        let got = &run.weights[name];
        let d = max_diff(got, want);
        let own = max_abs(want);
        let disp = max_diff(want, &init[name].1);
        let denom = if name.starts_with("span_head.") { head_max } else { own };
        let gated = d / denom;
        println!(
            "  {name}: |rust-ref| {d:.3e}  /own max {:.3e}  /displacement {:.3e}  gated {gated:.3e}",
            d / own,
            if disp > 0.0 { d / disp } else { f64::INFINITY }
        );
        if gated > worst.0 {
            worst = (gated, name.clone());
        }
    }
    worst
}

fn bits(run: &Run) -> BTreeMap<&String, Vec<u32>> {
    run.weights.iter().map(|(n, v)| (n, v.iter().map(|x| x.to_bits()).collect())).collect()
}

/// CPU only, and run by `cargo test`: what the GPU tests read is checked before the GPU
/// window. The reference recipe is the one this file trains, and the span head in the init is
/// the tower's width.
#[test]
fn rung_b_recipe_and_head_are_the_ones_this_file_trains() {
    let lower = recipe_lower();
    assert_eq!((lower.n, lower.lr_scale), (2, 0.1));
    let cfg = json("config.json");
    assert_eq!(init_head().hidden_size() as u64, cfg["text_config"]["hidden_size"].as_u64().unwrap());
}

/// CPU only. Whether ojas-qwen35 can open the fixture, checked in whichever state the fixture is
/// in, so it never passes vacuously.
/// - As L-oracle first wrote it, the top-level `tie_word_embeddings` is `false` though
///   `text_config` says `true`. The torch reference ties the head by construction (`backbone.py`
///   logits against the embedding), but ojas-qwen35 refuses any `false`. Every GPU test in this
///   file would then fail at `Snapshot::from_files` before training
///   (`GAP-LTRAINER-RUNG-B-FIXTURE-UNTIED-TOP-LEVEL-2026-10-01`), so that refusal is asserted.
/// - Once the fixture says `true`, the snapshot must open and match the head's width.
#[test]
fn rung_b_fixture_config_opens_in_ojas_or_is_refused_for_its_untied_top_level_flag() {
    let cfg = json("config.json");
    assert_eq!(cfg["text_config"]["tie_word_embeddings"], true);
    let opened = Snapshot::from_files(&dir().join("config.json"), &dir().join("init.safetensors"));
    match (cfg["tie_word_embeddings"].as_bool(), opened) {
        (Some(false), Err(e)) => assert!(e.to_string().contains("tie_word_embeddings is false"), "{e}"),
        (Some(false), Ok(_)) => panic!("ojas-qwen35 accepted an untied flag it is documented to refuse"),
        (_, Ok(snap)) => assert_eq!(snap.config.hidden as usize, init_head().hidden_size()),
        (_, Err(e)) => panic!("the fixture's config does not open in ojas-qwen35: {e}"),
    }
}

#[test]
#[ignore = "GPU; and F's two-group recipe is refused until tessl's per-entry lr_scale lands"]
fn gpu_rung_b_exact_f32_trains_as_torch_fp32_on_fs_recipe() {
    let lower = Some(recipe_lower());
    let a = run(Operands::ExactF32, lower).expect("rung (b) needs tessl's per-entry lr_scale (ruling 6)");
    println!("rung (b) ExactF32 vs torch fp32, losses:");
    let (early, all) = losses_vs(&a, "fp32");
    println!("rung (b) ExactF32 vs torch fp32, final weights (Amendment 2 measure):");
    let (worst, on) = final_weights(&a);
    println!("grads_step0: report-only at this rung; not measured by this test");
    let b = run(Operands::ExactF32, lower).unwrap();
    assert_eq!(bits(&a), bits(&b), "two Rust runs are bit-identical");
    assert_eq!(a.result.loss_log_digest().unwrap(), b.result.loss_log_digest().unwrap());
    assert!(early <= 1e-5, "loss steps 0-5: {early:.3e} > 1e-5");
    assert!(all <= 1e-4, "loss steps 0-19: {all:.3e} > 1e-4");
    assert!(worst <= 1e-5, "final weights: {worst:.3e} > 1e-5 on {on}");
}

#[test]
#[ignore = "GPU; and F's two-group recipe is refused until tessl's per-entry lr_scale lands"]
fn gpu_rung_b_bf16_operands_loss_stays_within_2e_8_of_torch_fp32() {
    let a = run(Operands::Bf16, Some(recipe_lower())).expect("rung (b) needs tessl's per-entry lr_scale (ruling 6)");
    println!("rung (b) Bf16 operands vs torch fp32 (gated):");
    let (_, all) = losses_vs(&a, "fp32");
    println!("rung (b) Bf16 operands vs torch master-bf16 (report-only):");
    losses_vs(&a, "master_bf16");
    assert!(all <= 2f64.powi(-8), "Bf16 loss vs torch fp32: {all:.3e} > 2^-8");
}

/// SMOKE, not parity: one learning rate for every tensor, which is not the reference's recipe.
#[test]
#[ignore = "GPU"]
fn gpu_rung_b_single_group_smoke_completes_finite_and_repeats_bit_for_bit() {
    let a = run(Operands::ExactF32, None).unwrap();
    assert!(a.result.steps.iter().all(|s| s.loss.is_finite() && s.grad_norm.is_finite()));
    let b = run(Operands::ExactF32, None).unwrap();
    assert_eq!(bits(&a), bits(&b), "two Rust runs are bit-identical");
    assert_eq!(a.result.loss_log_digest().unwrap(), b.result.loss_log_digest().unwrap());
    println!("single-group smoke vs the two-group torch fp32 reference (information only; not a parity measure):");
    losses_vs(&a, "fp32");
    let (worst, on) = final_weights(&a);
    println!("  worst gated ratio {worst:.3e} on {on} (a different recipe; no bar applies)");
}
