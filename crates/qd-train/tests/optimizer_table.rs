//! Amendment 2 (ii) and (iii) (`AUDIT/ojas-training-2026-10-01/fable-rung-b-bars.md`): the
//! optimizer table and the clip, at rung (a), on the host.
//!
//! * **Test 1, exact table equality.** The trainer's per-entry `(lr_scale, weight_decay, eps,
//!   beta1, beta2)` on the Qwen3.5-2B's 320 tower names and the span head's 4 equals, name by
//!   name and bit for bit, what Lappi's `layerwise_param_groups` + `build_optimizer` produce
//!   with F's flags. The expected table is read off a built torch optimizer by
//!   `tools/qd_train_oracle_optimizer_table.py` (`fixtures/optimizer-table-2b.json`), whose
//!   names are checked there against the snapshot's `model.language_model.*` header.
//! * **Test 3, the wiring.** A recording provider in front of the loop sees, at every
//!   `adamw_step`, exactly test 1's table for its entries.
//! * **The clip.** `min(1, max_norm / (norm + 1e-6))`, exact, below, at and above the clamp,
//!   at zero, and refused when not finite.

mod common;

use std::collections::BTreeMap;
use std::path::Path;

use common::{ToyBatch, ToyRow, ToySpan, ToySpanHead};
use qd_train::ft_data::HostSpanHead;
use qd_train::mock::{SplitMix, ToyProvider};
use qd_train::objective::{LetterSpanObjective, LetterTarget};
use qd_train::pyjson::float_fromhex;
use qd_train::recipe::{self, clip_coefficient, optimizer_table, EntryHyper, LowerLayers, OptimizerRecipe};
use qd_train::run_control::{MonotonicClock, WallClockCap};
use qd_train::schedule::LrSchedule;
use qd_train::span_head::SpanHead;
use qd_train::step::{AdamWHyper, BankMode, HiddenGrad, ParamSpec, SequenceJob, StepError, StepProvider};
use qd_train::trainer::{train, HostParams, Hooks, TrainConfig, TrainError};
use serde_json::Value;

fn fixture() -> Value {
    let path = common::crate_dir().join("tests/fixtures/optimizer-table-2b.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

/// F's optimizer flags (`campaign/f-v4-preregistered.json`: `--optimizer master --lr 1e-5
/// --lower-layers-n 8 --lower-layers-lr-scale 0.1`, default beta2), checked against what the
/// fixture says it was built with.
fn f_recipe(o: &Value) -> OptimizerRecipe {
    let r = OptimizerRecipe {
        lr: 1e-5,
        beta2: recipe::DEFAULT_BETA2,
        lower_layers: Some(LowerLayers { n: 8, lr_scale: 0.1 }),
    };
    assert_eq!(common::fhex(&o["recipe"]["lr"]), r.lr);
    assert_eq!(common::fhex(&o["recipe"]["beta2"]), r.beta2);
    assert_eq!(o["recipe"]["lower_layers_n"].as_u64(), Some(8));
    assert_eq!(common::fhex(&o["recipe"]["lower_lr_scale"]), 0.1);
    assert_eq!(o["recipe"]["optimizer_recipe"], "master");
    r
}

/// The 2B's tower entries, in the order the oracle's model lists them.
fn tower(o: &Value) -> Vec<ParamSpec> {
    let names = o["tower_names"].as_array().unwrap();
    let shapes = o["tower_shapes"].as_array().unwrap();
    assert_eq!(names.len(), 320, "the 2B text tower has 320 tensors");
    names
        .iter()
        .zip(shapes)
        .map(|(n, s)| {
            let shape: Vec<usize> = s.as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            ParamSpec::new(n.as_str().unwrap(), &shape)
        })
        .collect()
}

/// The span head's entries as the trainer holds them: the real head type's own names.
fn head(o: &Value) -> Vec<ParamSpec> {
    let h = o["hidden_size"].as_u64().unwrap() as usize;
    HostSpanHead::new(SpanHead::zeros(h).unwrap()).unwrap().entries()
}

/// The oracle's row for each name, as an [`EntryHyper`].
fn expected(o: &Value) -> BTreeMap<String, EntryHyper> {
    o["table"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, row)| {
            let f = |k: &str| float_fromhex(row[k].as_str().unwrap()).unwrap();
            (
                name.clone(),
                EntryHyper {
                    name: name.clone(),
                    lr_scale: f("lr_scale"),
                    weight_decay: f("weight_decay"),
                    eps: f("eps"),
                    beta1: f("beta1"),
                    beta2: f("beta2"),
                },
            )
        })
        .collect()
}

fn bits(e: &EntryHyper) -> [u64; 5] {
    [e.lr_scale, e.weight_decay, e.eps, e.beta1, e.beta2].map(f64::to_bits)
}

/// The trainer's table keyed as the oracle keys it: tower names as they are, the head's
/// under `span_head.`.
fn keyed(table: &[EntryHyper], n_tower: usize) -> BTreeMap<String, EntryHyper> {
    table
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let name = if i < n_tower { e.name.clone() } else { format!("span_head.{}", e.name) };
            (name.clone(), EntryHyper { name, ..e.clone() })
        })
        .collect()
}

fn assert_tables_equal(got: &BTreeMap<String, EntryHyper>, want: &BTreeMap<String, EntryHyper>) {
    let missing: Vec<&String> = want.keys().filter(|k| !got.contains_key(*k)).collect();
    let extra: Vec<&String> = got.keys().filter(|k| !want.contains_key(*k)).collect();
    assert!(missing.is_empty() && extra.is_empty(), "missing {missing:?}, extra {extra:?}");
    let wrong: Vec<(&String, &EntryHyper, &EntryHyper)> =
        want.iter().filter(|(k, w)| bits(&got[*k]) != bits(w)).map(|(k, w)| (k, &got[k], w)).collect();
    assert!(wrong.is_empty(), "{} of {} entries differ, first: {:?}", wrong.len(), want.len(), wrong.first());
}

#[test]
fn the_trainers_table_is_fs_builder_table_on_the_2b_names_exactly() {
    let o = fixture();
    let tower = tower(&o);
    let head = head(&o);
    let table = optimizer_table(&tower, &head, &f_recipe(&o)).unwrap();
    assert_eq!(table.len(), 324);
    let want = expected(&o);
    assert_eq!(want.len(), 324);
    assert_tables_equal(&keyed(&table, tower.len()), &want);
    // What the equality covers, said once in numbers: layers 0-7 at 0.1x, everything else
    // (embedding, final norm, layers 8-23, the head) at 1.0x, and weight decay 0.01 on every
    // entry, norms and dt_bias included (tessl's default exclusions would fail this test).
    let lower = want.values().filter(|e| e.lr_scale == 0.1).count();
    assert_eq!(lower, 106);
    for name in ["layers.0.linear_attn.dt_bias", "layers.0.input_layernorm.weight", "norm.weight", "span_head.abstain_start"] {
        assert_eq!(want[name].weight_decay, 0.01, "{name}");
    }
}

/// A step provider that records the table each `adamw_step` is handed, in front of a real toy
/// model; its entries are the 2B's names, so the loop builds F's table over them.
struct Recording {
    inner: ToyProvider,
    entries: Vec<ParamSpec>,
    calls: Vec<(AdamWHyper, u64, Vec<f64>, Vec<f64>)>,
}

impl StepProvider for Recording {
    fn parameters(&self) -> &[ParamSpec] {
        &self.entries
    }

    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn accumulate(&mut self, job: &SequenceJob<'_>, mode: BankMode, external: Option<&mut HiddenGrad<'_>>) -> Result<f64, StepError> {
        self.inner.accumulate(job, mode, external)
    }

    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        self.inner.grad_sq_norm()
    }

    fn supports_lr_scale(&self) -> bool {
        true
    }

    fn adamw_step(&mut self, hyper: &AdamWHyper, step: u64, lr_scale: &[f64], weight_decay: &[f64]) -> Result<(), StepError> {
        self.calls.push((*hyper, step, lr_scale.to_vec(), weight_decay.to_vec()));
        let n = self.inner.parameters().len();
        self.inner.adamw_step(hyper, step, &vec![1.0; n], &vec![recipe::WEIGHT_DECAY; n])
    }

    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        Err(StepError::Unsupported("a recording provider holds no 2B values".into()))
    }

    fn describe(&self) -> String {
        "recording provider over ToyProvider".into()
    }

    fn save_state(&mut self, _dir: &Path) -> Result<Vec<std::path::PathBuf>, StepError> {
        Err(StepError::Unsupported("a recording provider is not checkpointed".into()))
    }

    fn load_state(&mut self, _dir: &Path) -> Result<(), StepError> {
        Err(StepError::Unsupported("a recording provider is not checkpointed".into()))
    }
}

fn toy_batch(index: u64) -> ToyBatch {
    let mut rng = SplitMix::new(31 + index);
    let tok = |n: usize, rng: &mut SplitMix| -> Vec<u32> { (0..n).map(|_| (rng.next_u64() % 13) as u32).collect() };
    let a = tok(6, &mut rng);
    let b = tok(7, &mut rng);
    ToyBatch {
        index,
        bucket: 1,
        width: 7,
        rows: vec![
            ToyRow {
                letter: Some(LetterTarget { position: 4, target: a[5] }),
                tokens: a,
                span: None,
            },
            ToyRow {
                tokens: b,
                letter: None,
                span: Some(ToySpan {
                    candidates: vec![0, 2, 3, 6],
                    start: 1,
                    end: 2,
                }),
            },
        ],
    }
}

#[test]
fn the_loop_hands_adamw_step_exactly_the_validated_table() {
    let o = fixture();
    let tower = tower(&o);
    let mut p = Recording {
        inner: ToyProvider::new(13, 4, 2, 5).unwrap(),
        entries: tower.clone(),
        calls: Vec::new(),
    };
    let mut obj: LetterSpanObjective<ToyBatch, ToySpanHead> = LetterSpanObjective::new(ToySpanHead::new(4), 1.0).unwrap();
    let cfg = TrainConfig {
        schedule: LrSchedule::real_ft(1e-5, 3).unwrap(),
        cap: WallClockCap::new(3600.0).unwrap(),
        grad_accum: 1,
        optimizer: f_recipe(&o),
        max_grad_norm: recipe::MAX_GRAD_NORM,
        epoch: 0,
        seed: 0,
        checkpoint: None,
        eta: None,
    };
    let batches: Vec<Result<ToyBatch, TrainError>> = (0..3).map(|i| Ok(toy_batch(i))).collect();
    let r = train(&mut p, &mut obj, batches, &cfg, &MonotonicClock::new(), None, &mut Hooks::default()).unwrap();
    assert_eq!(r.optimizer_steps, 3);
    assert_eq!(p.calls.len(), 3);
    let want = expected(&o);
    for (i, (hyper, step, lr_scale, weight_decay)) in p.calls.iter().enumerate() {
        assert_eq!(*step, i as u64 + 1);
        assert_eq!(lr_scale.len(), tower.len());
        assert_eq!(weight_decay.len(), tower.len());
        for (j, e) in tower.iter().enumerate() {
            let w = &want[&e.name];
            let handed = [lr_scale[j], weight_decay[j], hyper.eps, hyper.beta1, hyper.beta2].map(f64::to_bits);
            assert_eq!(handed, bits(w), "step {step}: {} was handed another row than test 1's", e.name);
        }
    }
}

#[test]
fn the_clip_coefficient_is_the_formula_exactly_and_refuses_a_non_finite_norm() {
    let hex = |s: &str| float_fromhex(s).unwrap();
    // norm below max_norm: the clip is off.
    assert_eq!(clip_coefficient(0.25, 1.0).unwrap(), 1.0);
    // norm = 1 - 1e-6: norm + 1e-6 rounds to exactly 1.0, so the coefficient is exactly 1.
    let n: f64 = 1.0 - 1e-6;
    assert_eq!((n * n).sqrt(), n, "the squared norm round-trips");
    assert_eq!(clip_coefficient(n * n, 1.0).unwrap().to_bits(), hex("0x1.0000000000000p+0").to_bits());
    // norm = 1: just past the clamp.
    assert_eq!(clip_coefficient(1.0, 1.0).unwrap().to_bits(), hex("0x1.ffffde7212f19p-1").to_bits());
    // norm above: 1 / (4 + 1e-6), and the max_norm=150 arm at norm 200 and 62.
    assert_eq!(clip_coefficient(16.0, 1.0).unwrap().to_bits(), hex("0x1.fffff79c8452dp-3").to_bits());
    assert_eq!(clip_coefficient(40_000.0, 150.0).unwrap().to_bits(), hex("0x1.7fffffdfc9a9bp-1").to_bits());
    assert_eq!(clip_coefficient(3_844.0, 150.0).unwrap(), 1.0);
    // zero: 1 / 1e-6, clamped.
    assert_eq!(clip_coefficient(0.0, 1.0).unwrap(), 1.0);
    // not finite, or negative: refused, never a NaN coefficient.
    for bad in [f64::INFINITY, f64::NAN, -1.0] {
        assert!(matches!(clip_coefficient(bad, 1.0), Err(StepError::Contract(_))), "{bad}");
    }
}
