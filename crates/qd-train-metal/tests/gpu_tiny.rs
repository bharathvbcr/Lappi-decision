//! The adapter on the GPU, on tessl's committed tiny Qwen3.5 fixture
//! (`tessl/tests/fixtures/qwen35_train`: 2 layers, hidden 64, vocabulary 64).
//!
//! Every test here opens a Metal runtime, so every test is `#[ignore]` and named `gpu_*`:
//! `cargo test` never runs one. Run them only in an agreed GPU window, serialized:
//!
//! ```text
//! cargo test -p qd-train-metal --release --test gpu_tiny -- --ignored --test-threads=1
//! ```
//!
//! What they claim: the adapter computes exactly what `Qwen35Step` driven directly computes
//! (same losses, same gradient bank, bit for bit), its bank modes and refusals hold, and
//! qd-train's loop over it resumes from a checkpoint to the same bits as an uninterrupted run.
//! The step's own parity with transformers is ojas-qwen35's and tessl's (their `gpu_tiny_*`
//! suites), not re-tested here.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_core::{Budget, Tensor};
use ojas_qwen35::{ExternalGrad, Numerics, Qwen35Step, Sequence, Snapshot, Which};
use qd_train::ft_data::HostSpanHead;
use qd_train::mock::SplitMix;
use qd_train::npy::NpyArray;
use qd_train::objective::{FtBatch, LetterSpanObjective, LetterTarget};
use qd_train::recipe::{self, OptimizerRecipe};
use qd_train::run_control::{MonotonicClock, WallClockCap};
use qd_train::schedule::LrSchedule;
use qd_train::shards::ConsumedPrefix;
use qd_train::span_head::{SpanGold, SpanHead, SpanRowPlan};
use qd_train::step::{BankMode, HiddenGrad, RowTargets, SequenceJob, StepError, StepProvider};
use qd_train::trainer::{
    train, CheckpointPolicy, ConsumedBatch, HostParams, Hooks, ResumeState, Termination, TrainConfig, TrainError,
};
use qd_train_metal::args::Operands;
use qd_train_metal::provider::{Qwen35Provider, TESSL_DIR};

fn tiny() -> PathBuf {
    Path::new(TESSL_DIR).join("tests/fixtures/qwen35_train")
}

fn ids() -> Vec<u32> {
    let bytes = std::fs::read(tiny().join("ids.npy")).expect("ids.npy");
    NpyArray::parse(&bytes).unwrap().cast::<u32>("ids").unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn dh_of(h: &[f32]) -> Vec<f32> {
    h.iter().enumerate().map(|(i, x)| 1e-3 * x + 1e-4 * (i % 7) as f32).collect()
}

#[test]
#[ignore = "GPU: opens a Metal runtime; run only in an agreed GPU window"]
fn gpu_tiny_adapter_accumulates_exactly_what_qwen35step_accumulates() {
    let snap = Snapshot::from_dir(&tiny()).unwrap();
    let mut adapter = Qwen35Provider::open(&snap, Operands::ExactF32).unwrap();
    let mut direct = Qwen35Step::open(&snap, Numerics::ExactF32, Budget::new(1 << 30)).unwrap();
    let hidden = adapter.hidden_size();
    let ids = ids();
    let n = ids.len() - 1;
    let rows: Vec<u32> = (0..n as u32).collect();
    let scale = 1.0 / n as f32;

    // A sequence with letter rows only, over the bank.
    let job = SequenceJob {
        tokens: &ids,
        rows: RowTargets {
            positions: &rows,
            targets: &ids[1..],
            scale,
        },
        hidden_at: &[],
    };
    let a1 = adapter.accumulate(&job, BankMode::Overwrite, None).unwrap();
    let p = direct
        .forward(&Sequence {
            ids: &ids,
            letter_rows: &rows,
            letter_targets: &ids[1..],
            letter_scale: scale,
            span_positions: &[],
        })
        .unwrap();
    let d1 = p.letter_ce_sum();
    direct.backward(p, None).unwrap();
    assert_eq!(a1.to_bits(), d1.to_bits());

    // A second sequence added to it, with an outside gradient at three hidden rows.
    let ids2 = &ids[..40];
    let rows2 = [10u32, 38];
    let targets2 = [ids[11], ids[39]];
    let at = [3u32, 17, 39];
    let job2 = SequenceJob {
        tokens: ids2,
        rows: RowTargets {
            positions: &rows2,
            targets: &targets2,
            scale: 0.5,
        },
        hidden_at: &at,
    };
    let mut seen = Vec::new();
    let cb: &mut HiddenGrad<'_> = &mut |pos: &[u32], h: &[f32]| -> Result<Vec<f32>, StepError> {
        seen.push((pos.to_vec(), h.len()));
        Ok(dh_of(h))
    };
    let a2 = adapter.accumulate(&job2, BankMode::Add, Some(cb)).unwrap();
    assert_eq!(seen, vec![(at.to_vec(), at.len() * hidden)], "the callback is called once, at the job's positions");
    let p = direct
        .forward(&Sequence {
            ids: ids2,
            letter_rows: &rows2,
            letter_targets: &targets2,
            letter_scale: 0.5,
            span_positions: &at,
        })
        .unwrap();
    let d2 = p.letter_ce_sum();
    let dh = Tensor::from_f32(&dh_of(&p.hidden().to_f32_vec().unwrap()), &[at.len(), hidden], &Budget::new(1 << 20)).unwrap();
    direct
        .backward(
            p,
            Some(ExternalGrad {
                positions: &at,
                dh: &dh,
            }),
        )
        .unwrap();
    assert_eq!(a2.to_bits(), d2.to_bits());
    assert_eq!(adapter.grad_sq_norm().unwrap().to_bits(), direct.grad_sq_norm().unwrap().to_bits());
    let ga = adapter.qwen35().read_table(Which::Gradients).unwrap();
    let gd = direct.read_table(Which::Gradients).unwrap();
    assert_eq!(ga.len(), gd.len());
    for (a, d) in ga.iter().zip(&gd) {
        assert_eq!(a.name, d.name);
        assert_eq!(bits(&a.tensor.to_f32_vec().unwrap()), bits(&d.tensor.to_f32_vec().unwrap()), "{}", a.name);
    }

    // Overwrite on a holding bank starts the bank again: one sequence's gradient, as direct's.
    adapter.accumulate(&job, BankMode::Overwrite, None).unwrap();
    direct.discard_gradients();
    let p = direct
        .forward(&Sequence {
            ids: &ids,
            letter_rows: &rows,
            letter_targets: &ids[1..],
            letter_scale: scale,
            span_positions: &[],
        })
        .unwrap();
    direct.backward(p, None).unwrap();
    assert_eq!(adapter.grad_sq_norm().unwrap().to_bits(), direct.grad_sq_norm().unwrap().to_bits());

    // Refusals, on a fresh adapter whose bank is empty.
    let mut fresh = Qwen35Provider::open(&snap, Operands::ExactF32).unwrap();
    let err = fresh.accumulate(&job, BankMode::Add, None).unwrap_err();
    assert!(matches!(&err, StepError::Contract(m) if m.contains("empty bank")), "{err}");
    let err = fresh.accumulate(&job2, BankMode::Overwrite, None).unwrap_err();
    assert!(matches!(&err, StepError::Contract(m) if m.contains("no gradient callback")), "{err}");
    let ones = vec![1.0; fresh.parameters().len()];
    let mut scaled = ones.clone();
    scaled[0] = 0.1;
    let wd = vec![0.01; ones.len()];
    fresh.accumulate(&job, BankMode::Overwrite, None).unwrap();
    let hyper = qd_train::step::AdamWHyper {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    let err = fresh.adamw_step(&hyper, 1, &scaled, &wd).unwrap_err();
    assert!(matches!(&err, StepError::Unsupported(m) if m.contains("entry 0 asks for 0.1x")), "{err}");
    let err = fresh.adamw_step(&hyper, 2, &ones, &wd).unwrap_err();
    assert!(matches!(&err, StepError::Contract(m) if m.contains("has taken 0")), "{err}");
    assert_eq!(fresh.step_count(), 0, "a refused step moves nothing");
}

/// One row of a tiny batch: real tokens, and either a letter answer or a span row.
#[derive(Clone)]
struct TinyRow {
    tokens: Vec<u32>,
    letter: Option<LetterTarget>,
    span: Option<SpanRowPlan>,
}

#[derive(Clone)]
struct TinyBatch {
    index: u64,
    width: usize,
    rows: Vec<TinyRow>,
}

impl ConsumedBatch for TinyBatch {
    fn index(&self) -> u64 {
        self.index
    }

    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError> {
        let mut tokens = Vec::new();
        for r in &self.rows {
            for t in &r.tokens {
                tokens.extend_from_slice(&t.to_le_bytes());
            }
        }
        prefix.fold(&[&self.index.to_be_bytes(), &tokens]);
        Ok(())
    }
}

impl FtBatch for TinyBatch {
    type Span = SpanRowPlan;

    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn padded_width(&self) -> usize {
        self.width
    }

    fn row_tokens(&self, row: usize) -> &[u32] {
        &self.rows[row].tokens
    }

    fn letter(&self, row: usize) -> Option<LetterTarget> {
        self.rows[row].letter
    }

    fn span(&self, row: usize) -> Option<&SpanRowPlan> {
        self.rows[row].span.as_ref()
    }
}

/// Batch `i`: a letter row and a span row cut from the fixture's ids at an offset that moves
/// with `i`, so every step sees different tokens.
fn batch(ids: &[u32], i: u64) -> TinyBatch {
    let off = (i as usize * 7) % 20;
    let a = ids[off..off + 30].to_vec();
    let b = ids[off + 5..off + 41].to_vec();
    let lb = b.len() as u32;
    let gold = if i.is_multiple_of(2) { SpanGold::Lines { start: 1, end: 2 } } else { SpanGold::Abstain };
    TinyBatch {
        index: i,
        width: 48,
        rows: vec![
            TinyRow {
                letter: Some(LetterTarget {
                    position: a.len() as u32 - 2,
                    target: a[a.len() - 1],
                }),
                tokens: a,
                span: None,
            },
            TinyRow {
                tokens: b,
                letter: None,
                span: Some(SpanRowPlan::new(lb - 1, vec![2, 9, 15, 22], gold).unwrap()),
            },
        ],
    }
}

fn head(hidden: usize) -> SpanHead {
    let mut rng = SplitMix::new(7);
    let mut v = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.uniform(0.05)).collect() };
    SpanHead::new(hidden, v(hidden * hidden), v(hidden * hidden), v(hidden), v(hidden)).unwrap()
}

fn config(steps: u64, checkpoint: Option<CheckpointPolicy>) -> TrainConfig {
    TrainConfig {
        schedule: LrSchedule::real_ft(1e-3, steps).unwrap(),
        cap: WallClockCap::new(3600.0).unwrap(),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: 1e-3,
            beta2: recipe::DEFAULT_BETA2,
            lower_layers: None,
        },
        max_grad_norm: recipe::MAX_GRAD_NORM,
        epoch: 0,
        seed: 0,
        checkpoint,
        eta: None,
    }
}

type Objective = LetterSpanObjective<TinyBatch, HostSpanHead>;

fn fresh(snap: &Snapshot) -> (Qwen35Provider, Objective) {
    let p = Qwen35Provider::open(snap, Operands::ExactF32).unwrap();
    let o = LetterSpanObjective::new(HostSpanHead::new(head(p.hidden_size())).unwrap(), 1.0).unwrap();
    (p, o)
}

fn source(ids: &[u32], n: u64) -> Vec<Result<TinyBatch, TrainError>> {
    (0..n).map(|i| Ok(batch(ids, i))).collect()
}

#[test]
#[ignore = "GPU: opens a Metal runtime; run only in an agreed GPU window"]
fn gpu_tiny_loop_resumed_from_a_checkpoint_is_the_uninterrupted_run_bit_for_bit() {
    let snap = Snapshot::from_dir(&tiny()).unwrap();
    let ids = ids();
    let clock = MonotonicClock::new();

    let (mut p, mut o) = fresh(&snap);
    let full = train(&mut p, &mut o, source(&ids, 4), &config(4, None), &clock, None, &mut Hooks::default()).unwrap();
    assert_eq!(full.termination, Termination::StepsExhausted);
    assert_eq!(full.optimizer_steps, 4);
    let full_tower = p.read_parameters().unwrap();
    let full_head: Vec<Vec<f32>> = o.head().values().into_iter().map(<[f32]>::to_vec).collect();
    drop(p);

    // Under the workspace's target directory, not the system temp dir: on macOS that is under
    // /var, a symbolic link, and ojas-io's state writer refuses symbolic links on the way to
    // the state directory.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("qd-train-metal-gpu-ckpt-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    let policy = CheckpointPolicy {
        dir: dir.clone(),
        every: 2,
    };
    let (mut p, mut o) = fresh(&snap);
    let first = train(&mut p, &mut o, source(&ids, 2), &config(4, Some(policy)), &clock, None, &mut Hooks::default()).unwrap();
    assert_eq!(first.termination, Termination::DataExhausted);
    let ckpt = first.final_checkpoint.clone().unwrap();
    assert!(ckpt.join("provider").join(qd_train_metal::provider::STATE_DIR).is_dir());
    drop(p);

    let (mut p, mut o) = fresh(&snap);
    let state = ResumeState::read(&ckpt).unwrap();
    let second = train(&mut p, &mut o, source(&ids, 4), &config(4, None), &clock, Some(state), &mut Hooks::default()).unwrap();
    assert_eq!(second.optimizer_steps, 4);
    assert_eq!(second.loss_log_digest().unwrap(), full.loss_log_digest().unwrap());
    assert_eq!(second.consumed_digest, full.consumed_digest);
    let tower = p.read_parameters().unwrap();
    assert_eq!(tower.len(), full_tower.len());
    for ((spec, a), b) in p.parameters().iter().zip(&tower).zip(&full_tower) {
        assert_eq!(bits(a), bits(b), "{}", spec.name);
    }
    for (a, b) in o.head().values().into_iter().zip(&full_head) {
        assert_eq!(bits(a), bits(b));
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
