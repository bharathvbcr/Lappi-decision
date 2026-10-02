//! The loop end to end on the CPU: `ToyProvider` (a real tiny model) under Lappi's letter +
//! span objective with a toy pointer head. What each test pins:
//!
//! * the provider's and the head's gradients are the loss's (central differences), so the
//!   loop tests below are not checking a tautology;
//! * one step of `train` is exactly the composed step: sequences into the bank with the
//!   batch-mean scales, clip over tower and head, AdamW per entry with Lappi's weight decay;
//! * two runs are the same bits; a resume of 6 + 6 steps is the uninterrupted 12;
//! * a non-finite loss or gradient stops the run before any moment moves;
//! * the wall-clock cap stops only between groups; a partial group, a backwards index, a
//!   layer-wise rate on a provider without one, a different source on resume are refused.

mod common;

use std::cell::Cell;
use std::path::{Path, PathBuf};

use common::{ToyBatch, ToyRow, ToySpan, ToySpanHead};
use qd_train::adamw::{adamw_update, Moments};
use qd_train::mock::{SplitMix, ToyProvider};
use qd_train::objective::{LetterSpanObjective, LetterTarget};
use qd_train::recipe::{self, LowerLayers, OptimizerRecipe};
use qd_train::run_control::{Clock, MonotonicClock, WallClockCap};
use qd_train::schedule::LrSchedule;
use qd_train::step::{AdamWHyper, BankMode, HiddenGrad, RowTargets, SequenceJob, StepError, StepProvider};
use qd_train::trainer::{
    parameters_digest, train, CheckpointPolicy, EtaRule, Hooks, MicroLoss, Objective, ResumeState, Termination,
    TrainConfig, TrainError, TrainResult,
};

const V: usize = 13;
const H: usize = 4;
const L: usize = 2;
const LR: f64 = 3e-2;

type Obj = LetterSpanObjective<ToyBatch, ToySpanHead>;

fn batch(index: u64, seed: u64) -> ToyBatch {
    let mut rng = SplitMix::new(seed.wrapping_mul(7919).wrapping_add(index));
    let mut rows = Vec::new();
    let tok = |n: usize, rng: &mut SplitMix| -> Vec<u32> { (0..n).map(|_| (rng.next_u64() % V as u64) as u32).collect() };
    for _ in 0..2 {
        let n = 4 + (rng.next_u64() % 4) as usize;
        let tokens = tok(n, &mut rng);
        let letter = LetterTarget {
            position: (n - 2) as u32,
            target: tokens[n - 1],
        };
        rows.push(ToyRow {
            tokens,
            letter: Some(letter),
            span: None,
        });
    }
    let n = 6 + (rng.next_u64() % 3) as usize;
    let tokens = tok(n, &mut rng);
    let candidates = vec![0u32, 2, 3, (n - 1) as u32];
    let start = (rng.next_u64() % 5) as usize;
    let end = start.max((rng.next_u64() % 5) as usize);
    rows.push(ToyRow {
        tokens,
        letter: None,
        span: Some(ToySpan { candidates, start, end }),
    });
    let width = rows.iter().map(|r| r.tokens.len()).max().unwrap();
    ToyBatch {
        index,
        bucket: 1,
        width,
        rows,
    }
}

fn batches(n: u64, seed: u64) -> Vec<Result<ToyBatch, TrainError>> {
    (0..n).map(|i| Ok(batch(i, seed))).collect()
}

fn config(total: u64) -> TrainConfig {
    TrainConfig {
        schedule: LrSchedule::real_ft(LR, total).unwrap(),
        cap: WallClockCap::new(3600.0).unwrap(),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: LR,
            beta2: recipe::DEFAULT_BETA2,
            lower_layers: None,
        },
        max_grad_norm: recipe::MAX_GRAD_NORM,
        epoch: 0,
        seed: 7,
        checkpoint: None,
        eta: None,
    }
}

fn fresh() -> (ToyProvider, Obj) {
    (
        ToyProvider::new(V, H, L, 11).unwrap(),
        LetterSpanObjective::new(ToySpanHead::new(H), 0.75).unwrap(),
    )
}

fn run(p: &mut ToyProvider, o: &mut Obj, data: Vec<Result<ToyBatch, TrainError>>, cfg: &TrainConfig) -> Result<TrainResult, TrainError> {
    train(p, o, data, cfg, &MonotonicClock::new(), None, &mut Hooks::default())
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qd-train-{tag}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    d
}

// ---- the gradients are the loss's ----------------------------------------------------------

#[test]
fn toy_provider_gradients_match_central_differences() {
    let mut p = ToyProvider::new(V, H, L, 3).unwrap();
    let tokens = [1u32, 5, 7, 2, 9, 4];
    let positions = [1u32, 4];
    let targets = [7u32, 4];
    let hidden_at = [0u32, 3];
    let c: Vec<f32> = (0..hidden_at.len() * H).map(|i| 0.1 * (i as f32) - 0.3).collect();
    let scale = 0.5f32;
    let job = SequenceJob {
        tokens: &tokens,
        rows: RowTargets {
            positions: &positions,
            targets: &targets,
            scale,
        },
        hidden_at: &hidden_at,
    };
    // total = scale * rows + sum_k c_k . x[hidden_at[k]]
    let total = |p: &ToyProvider| -> f64 {
        let x = p.hidden_states(&tokens);
        let ext: f64 = hidden_at
            .iter()
            .enumerate()
            .map(|(k, &pos)| (0..H).map(|i| f64::from(c[k * H + i] * x[pos as usize][i])).sum::<f64>())
            .sum();
        f64::from(scale) * p.row_loss(&job).unwrap() + ext
    };
    let cc = c.clone();
    let mut ext = move |_: &[u32], _: &[f32]| -> Result<Vec<f32>, StepError> { Ok(cc.clone()) };
    let ext_ref: &mut HiddenGrad<'_> = &mut ext;
    p.accumulate(&job, BankMode::Overwrite, Some(ext_ref)).unwrap();
    let grads = p.grads().to_vec();
    let mut checked = 0;
    for (e, ge) in grads.iter().enumerate() {
        for i in (0..ge.len()).step_by(3) {
            let g = f64::from(ge[i]);
            if g.abs() < 1e-3 {
                continue;
            }
            let eps = 1e-2f32;
            let x0 = p.values()[e][i];
            p.values_mut()[e][i] = x0 + eps;
            let up = total(&p);
            p.values_mut()[e][i] = x0 - eps;
            let down = total(&p);
            p.values_mut()[e][i] = x0;
            let fd = (up - down) / (2.0 * f64::from(eps));
            assert!((fd - g).abs() <= 2e-2 * g.abs().max(1e-2), "entry {e}[{i}]: analytic {g} vs central {fd}");
            checked += 1;
        }
    }
    assert!(checked > 20, "only {checked} gradients checked");
}

#[test]
fn the_toy_heads_gradients_match_central_differences() {
    let mut head = ToySpanHead::new(H);
    let span = ToySpan {
        candidates: vec![0, 2, 5],
        start: 1,
        end: 3,
    };
    let hidden: Vec<f32> = (0..3 * H).map(|i| (i as f32 * 0.37).sin()).collect();
    use qd_train::objective::SpanHead;
    let (loss, dh) = head.loss_and_grad(&span, &span.candidates, &hidden, H, 1.0).unwrap();
    assert!((loss - head.row_loss(&span, &hidden, H)).abs() < 1e-9);
    for i in 0..hidden.len() {
        let mut up = hidden.clone();
        up[i] += 1e-3;
        let mut dn = hidden.clone();
        dn[i] -= 1e-3;
        let fd = (head.row_loss(&span, &up, H) - head.row_loss(&span, &dn, H)) / 2e-3;
        assert!((fd - f64::from(dh[i])).abs() < 1e-3, "dh[{i}]: {} vs {fd}", dh[i]);
    }
    let gu = head.gu.clone();
    for (i, g) in gu.iter().enumerate() {
        let u0 = head.u[i];
        head.u[i] = u0 + 1e-3;
        let up = head.row_loss(&span, &hidden, H);
        head.u[i] = u0 - 1e-3;
        let dn = head.row_loss(&span, &hidden, H);
        head.u[i] = u0;
        assert!(((up - dn) / 2e-3 - f64::from(*g)).abs() < 1e-3);
    }
}

// ---- one step of the loop is the composed step ---------------------------------------------

/// One batch's gradients by hand, through the same public pieces the loop uses: plan, bank,
/// close, clip. The gradients are left in `p` and `o`. Returns the micro-batch loss, the clip
/// coefficient and the plan's row scale.
fn accumulate_by_hand(p: &mut ToyProvider, o: &mut Obj, b: &ToyBatch) -> (MicroLoss, f64, f32) {
    let plan = o.plan(b).unwrap();
    o.host().unwrap().zero_grads();
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
        let mode = if i == 0 { BankMode::Overwrite } else { BankMode::Add };
        let sum = if s.hidden_at.is_empty() {
            p.accumulate(&job, mode, None).unwrap()
        } else {
            let oo = &mut *o;
            let mut ext = |pos: &[u32], h: &[f32]| oo.hidden_grad(i, pos, h, H).map_err(|e| StepError::Contract(e.to_string()));
            p.accumulate(&job, mode, Some(&mut ext)).unwrap()
        };
        sums.push(sum);
    }
    let loss = o.close(&sums).unwrap();
    let tower_sq = p.grad_sq_norm().unwrap();
    let host_sq: f64 = o.host_ref().unwrap().grads().into_iter().flatten().flatten().map(|&x| f64::from(x) * f64::from(x)).sum();
    let coef = recipe::clip_coefficient(tower_sq + host_sq, 1.0).unwrap();
    (loss, coef, plan.row_scale)
}

#[test]
fn one_loop_step_is_exactly_the_composed_step() {
    let b = batch(0, 5);
    // The composed step, by hand, through the same public pieces.
    let (mut p, mut o) = fresh();
    let (loss, coef, row_scale) = accumulate_by_hand(&mut p, &mut o, &b);
    assert_eq!(row_scale, 0.5, "two letter rows: 1/N");
    let lr = config(4).schedule.lr_at(0).unwrap();
    let hyper = AdamWHyper {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: coef,
    };
    let n = p.parameters().len();
    p.adamw_step(&hyper, 1, &vec![1.0; n], &vec![0.01; n]).unwrap();
    let mut head_vals: Vec<Vec<f32>> = o.host_ref().unwrap().values().iter().map(|v| v.to_vec()).collect();
    let head_grads: Vec<Vec<f32>> = o.host_ref().unwrap().grads().into_iter().map(|g| g.expect("a span row reached the head").to_vec()).collect();
    for (v, g) in head_vals.iter_mut().zip(&head_grads) {
        let mut m = Moments::zeros(v.len());
        adamw_update(v, g, &mut m, &hyper, lr, 0.01, 1).unwrap();
    }

    // The loop.
    let (mut lp, mut lo) = fresh();
    let r = run(&mut lp, &mut lo, vec![Ok(b)], &config(4)).unwrap();
    assert_eq!(r.termination, Termination::DataExhausted);
    assert_eq!(r.optimizer_steps, 1);
    assert_eq!(r.loss_log.points()[0].loss.to_bits(), loss.total.to_bits());
    assert_eq!(r.steps[0].clip_coefficient.to_bits(), coef.to_bits());
    for (a, b) in lp.values().iter().zip(p.values()) {
        assert_eq!(a, b, "tower parameters after one step");
    }
    let looped: Vec<Vec<f32>> = lo.host_ref().unwrap().values().iter().map(|v| v.to_vec()).collect();
    assert_eq!(looped, head_vals, "head parameters after one step");
    // letter + 0.75 * span, and both channels logged.
    let ch = |name: &str| r.channel_log.iter().find(|(n, _)| n == name).unwrap().1[0];
    assert!((loss.total - (ch("letter") + 0.75 * ch("span"))).abs() < 1e-12);
    assert!(ch("letter") > 0.0 && ch("span") > 0.0);
    assert_eq!(r.counts.supervised_tokens, 2);
    assert_eq!(r.counts.span_rows, 1);
}

#[test]
fn weight_decay_reaches_every_entry_norms_included() {
    // With a zero gradient only the decay moves a weight, so after one step every entry --
    // norm.weight included -- has shrunk by exactly (1 - lr * 0.01): Lappi decays everything.
    let mut p = ToyProvider::new(V, H, L, 3).unwrap();
    let before = p.values().to_vec();
    let n = p.parameters().len();
    let tokens = [1u32, 2];
    let job = SequenceJob {
        tokens: &tokens,
        rows: RowTargets {
            positions: &[],
            targets: &[],
            scale: 0.0,
        },
        hidden_at: &[],
    };
    p.accumulate(&job, BankMode::Overwrite, None).unwrap();
    let hyper = AdamWHyper {
        lr: 0.5,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    p.adamw_step(&hyper, 1, &vec![1.0; n], &recipe::weight_decays(p.parameters())).unwrap();
    let f = (1.0 - 0.5f64 * 0.01) as f32;
    for (e, (a, b)) in p.values().iter().zip(&before).enumerate() {
        for (x, y) in a.iter().zip(b) {
            assert_eq!(*x, y * f, "entry {} ({})", e, p.parameters()[e].name);
        }
    }
}

// ---- a host entry with no gradient is skipped, as torch skips a `None` grad -----------------
//
// torch zeroes with `zero_grad(set_to_none=True)`, and a letter-only batch never reaches the
// span head, so the head's grads stay `None` and AdamW skips it: no decay, no moment update, no
// step count. Its bias corrections then use its own count, not the model's. The rung (b)
// reference shows it: `tiny-published/manifest.json` `per_parameter` has `span_head.*` at 13
// AdamW steps and every tower entry at 20, over the 7 letter-only batches of 20.

/// `batch` without its span row: a letter-only batch.
fn letter_only(index: u64, seed: u64) -> ToyBatch {
    let mut b = batch(index, seed);
    b.rows.retain(|r| r.span.is_none());
    b.width = b.rows.iter().map(|r| r.tokens.len()).max().unwrap();
    b
}

fn head_values(o: &Obj) -> Vec<Vec<f32>> {
    o.host_ref().unwrap().values().iter().map(|v| v.to_vec()).collect()
}

#[test]
fn a_letter_only_step_leaves_the_span_head_bit_for_bit_where_it_was() {
    let init = head_values(&fresh().1);
    let (mut p, mut o) = fresh();
    let r = run(&mut p, &mut o, vec![Ok(letter_only(0, 5))], &config(4)).unwrap();
    assert_eq!(r.optimizer_steps, 1);
    assert_eq!(r.counts.span_rows, 0);
    assert_ne!(p.values(), fresh().0.values(), "the tower stepped");
    assert_eq!(head_values(&o), init, "the head moved on a step that never reached it");
}

#[test]
fn under_grad_accum_a_group_steps_the_head_only_when_one_of_its_micro_batches_reached_it() {
    let mut cfg = config(2);
    cfg.grad_accum = 2;
    // Group 0 is letter-only then span: the head has a gradient and steps. Group 1 is two
    // letter-only batches: the head is skipped.
    let data = || vec![Ok(letter_only(0, 5)), Ok(batch(1, 5)), Ok(letter_only(2, 5)), Ok(letter_only(3, 5))];
    let (mut p, mut o) = fresh();
    assert_eq!(run(&mut p, &mut o, data(), &cfg).unwrap().optimizer_steps, 2);
    let (mut p1, mut o1) = fresh();
    assert_eq!(run(&mut p1, &mut o1, data().into_iter().take(2).collect(), &cfg).unwrap().optimizer_steps, 1);
    assert_ne!(head_values(&o1), head_values(&fresh().1), "the group with a span row stepped the head");
    assert_eq!(head_values(&o), head_values(&o1), "the all-letter group left the head where it was");
}

#[test]
fn the_heads_first_step_after_a_letter_only_step_bias_corrects_with_its_own_count_of_one() {
    let (l0, s1) = (letter_only(0, 5), batch(1, 5));
    // By hand: the loop takes the letter-only step, then the span step is composed by hand. The
    // tower is at its second step; the head takes its first, with fresh moments.
    let (mut p, mut o) = fresh();
    assert_eq!(run(&mut p, &mut o, vec![Ok(l0.clone())], &config(4)).unwrap().optimizer_steps, 1);
    let (_, coef, _) = accumulate_by_hand(&mut p, &mut o, &s1);
    let lr = config(4).schedule.lr_at(1).unwrap();
    let hyper = AdamWHyper {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: coef,
    };
    let n = p.parameters().len();
    p.adamw_step(&hyper, 2, &vec![1.0; n], &vec![0.01; n]).unwrap();
    let mut want_head = head_values(&o);
    let head_grads: Vec<Vec<f32>> = o.host_ref().unwrap().grads().into_iter().map(|g| g.expect("the span row reached the head").to_vec()).collect();
    for (v, g) in want_head.iter_mut().zip(&head_grads) {
        let mut fresh_moments = Moments::zeros(v.len());
        adamw_update(v, g, &mut fresh_moments, &hyper, lr, 0.01, 1).unwrap();
    }

    let (mut lp, mut lo) = fresh();
    assert_eq!(run(&mut lp, &mut lo, vec![Ok(l0), Ok(s1)], &config(4)).unwrap().optimizer_steps, 2);
    assert_eq!(lp.values(), p.values(), "tower parameters after two steps");
    assert_eq!(head_values(&lo), want_head, "the head's one step, bias-corrected at t = 1");
}

/// Twelve batches, every third one letter-only, so at the checkpoint (step 6) the head has
/// taken 4 steps to the tower's 6.
fn mixed_batches(n: u64, seed: u64) -> Vec<Result<ToyBatch, TrainError>> {
    (0..n).map(|i| Ok(if i % 3 == 1 { letter_only(i, seed) } else { batch(i, seed) })).collect()
}

#[test]
fn a_resume_carries_each_host_entrys_own_step_count() {
    let dir = temp_dir("resume-host-steps");
    let full = {
        let (mut p, mut o) = fresh();
        let r = run(&mut p, &mut o, mixed_batches(12, 4), &config(12)).unwrap();
        assert_eq!(r.optimizer_steps, 12);
        parameters_digest(&mut p, o.host_ref()).unwrap()
    };
    let mut cfg = config(12);
    cfg.checkpoint = Some(CheckpointPolicy {
        dir: dir.clone(),
        every: 6,
    });
    let ckpt = {
        let (mut p, mut o) = fresh();
        run(&mut p, &mut o, mixed_batches(6, 4), &cfg).unwrap().final_checkpoint.unwrap()
    };
    let state = ResumeState::read(&ckpt).unwrap();
    assert_eq!(state.optimizer_step, 6);
    assert_eq!(state.host_state.iter().map(|s| s.steps).collect::<Vec<_>>(), vec![4, 4], "batches 1 and 4 were letter-only");
    let (mut p, mut o) = fresh();
    let mut cfg2 = config(12);
    cfg2.checkpoint = None;
    train(&mut p, &mut o, mixed_batches(12, 4), &cfg2, &MonotonicClock::new(), Some(state), &mut Hooks::default()).unwrap();
    assert_eq!(parameters_digest(&mut p, o.host_ref()).unwrap(), full);

    // A host count past the checkpoint's optimizer step cannot be a real run's, and is refused.
    let mut bad = ResumeState::read(&ckpt).unwrap();
    bad.host_state[0].steps = 7;
    let (mut p, mut o) = fresh();
    let err = train(&mut p, &mut o, mixed_batches(12, 4), &cfg2, &MonotonicClock::new(), Some(bad), &mut Hooks::default()).unwrap_err();
    assert!(matches!(&err, TrainError::Refused(m) if m.contains("7 AdamW steps")), "{err}");

    // A v1 checkpoint carries no host counts, so it is refused rather than resumed with a guess.
    let body = ckpt.join("trainer.json");
    let text = std::fs::read_to_string(&body).unwrap();
    assert!(text.contains(qd_train::trainer::CHECKPOINT_FORMAT));
    std::fs::write(&body, text.replace(qd_train::trainer::CHECKPOINT_FORMAT, "qd-train-checkpoint-v1")).unwrap();
    let err = ResumeState::read(&ckpt).unwrap_err();
    assert!(matches!(&err, TrainError::Refused(m) if m.contains("not a qd-train-checkpoint-v2")), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

// ---- determinism and resume ---------------------------------------------------------------

#[test]
fn two_runs_are_the_same_bits() {
    let digest = || {
        let (mut p, mut o) = fresh();
        let r = run(&mut p, &mut o, batches(9, 1), &config(9)).unwrap();
        assert_eq!(r.termination, Termination::StepsExhausted);
        (r.loss_log_digest().unwrap(), r.consumed_digest.clone(), parameters_digest(&mut p, o.host_ref()).unwrap())
    };
    let a = digest();
    assert_eq!(a, digest());
    // And the data matters: another seed's batches are another run.
    let (mut p, mut o) = fresh();
    let other = run(&mut p, &mut o, batches(9, 2), &config(9)).unwrap();
    assert_ne!(other.consumed_digest, a.1);
}

#[test]
fn a_resume_of_six_and_six_is_the_uninterrupted_twelve() {
    let dir = temp_dir("resume");
    let full = {
        let (mut p, mut o) = fresh();
        let r = run(&mut p, &mut o, batches(12, 3), &config(12)).unwrap();
        (r, parameters_digest(&mut p, o.host_ref()).unwrap())
    };
    assert_eq!(full.0.optimizer_steps, 12);

    // First half: the source ends after six batches; the final checkpoint is ckpt-00000006.
    let mut cfg = config(12);
    cfg.checkpoint = Some(CheckpointPolicy {
        dir: dir.clone(),
        every: 4,
    });
    let first = {
        let (mut p, mut o) = fresh();
        run(&mut p, &mut o, batches(6, 3), &cfg).unwrap()
    };
    assert_eq!(first.termination, Termination::DataExhausted);
    let ckpt = first.final_checkpoint.clone().unwrap();
    assert!(ckpt.ends_with("ckpt-00000006"));
    assert!(!dir.join("ckpt-00000004").exists(), "one resume point is kept");

    // Second half: a fresh model, restored, fed the whole epoch from the start.
    let state = ResumeState::read(&ckpt).unwrap();
    let (mut p, mut o) = fresh();
    let mut cfg2 = config(12);
    cfg2.checkpoint = None;
    let second = train(&mut p, &mut o, batches(12, 3), &cfg2, &MonotonicClock::new(), Some(state), &mut Hooks::default()).unwrap();
    assert_eq!(second.optimizer_steps, 12);
    assert_eq!(second.loss_log_digest().unwrap(), full.0.loss_log_digest().unwrap());
    assert_eq!(second.consumed_digest, full.0.consumed_digest);
    assert_eq!(parameters_digest(&mut p, o.host_ref()).unwrap(), full.1);

    // A different source before the resume point is refused, though the indices line up.
    let state = ResumeState::read(&ckpt).unwrap();
    let (mut p, mut o) = fresh();
    let mut other = batches(12, 3);
    other[2] = Ok(batch(2, 99));
    let err = train(&mut p, &mut o, other, &cfg2, &MonotonicClock::new(), Some(state), &mut Hooks::default()).unwrap_err();
    assert!(matches!(&err, TrainError::Refused(m) if m.contains("different batches")), "{err}");
    // A different seed is refused.
    let state = ResumeState::read(&ckpt).unwrap();
    let (mut p, mut o) = fresh();
    let mut cfg3 = cfg2.clone();
    cfg3.seed = 8;
    assert!(train(&mut p, &mut o, batches(12, 3), &cfg3, &MonotonicClock::new(), Some(state), &mut Hooks::default()).is_err());
    // A tampered checkpoint is refused on read.
    let host = ckpt.join("host.f32");
    let mut bytes = std::fs::read(&host).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&host, &bytes).unwrap();
    assert!(ResumeState::read(&ckpt).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

// ---- the non-finite kill --------------------------------------------------------------------

/// A provider that turns its `nth` accumulate call's loss into NaN, or poisons its gradient.
struct Poison {
    inner: ToyProvider,
    calls: u64,
    nth: u64,
    gradient: bool,
}

impl StepProvider for Poison {
    fn parameters(&self) -> &[qd_train::step::ParamSpec] {
        self.inner.parameters()
    }
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn accumulate(&mut self, job: &SequenceJob<'_>, mode: BankMode, external: Option<&mut HiddenGrad<'_>>) -> Result<f64, StepError> {
        let hit = self.calls == self.nth;
        self.calls += 1;
        let loss = self.inner.accumulate(job, mode, external)?;
        if hit && self.gradient {
            // A finite loss over a gradient that is not: what a broken kernel looks like.
            self.inner.grads_mut()[0][0] = f32::NAN;
            return Ok(loss);
        }
        Ok(if hit { f64::NAN } else { loss })
    }
    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        self.inner.grad_sq_norm()
    }
    fn supports_lr_scale(&self) -> bool {
        self.inner.supports_lr_scale()
    }
    fn adamw_step(&mut self, h: &AdamWHyper, s: u64, l: &[f64], w: &[f64]) -> Result<(), StepError> {
        self.inner.adamw_step(h, s, l, w)
    }
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }
    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        self.inner.read_parameters()
    }
    fn describe(&self) -> String {
        "poisoned toy".into()
    }
    fn save_state(&mut self, d: &Path) -> Result<Vec<PathBuf>, StepError> {
        self.inner.save_state(d)
    }
    fn load_state(&mut self, d: &Path) -> Result<(), StepError> {
        self.inner.load_state(d)
    }
}

fn after_one_step() -> (Vec<Vec<f32>>, Vec<Moments>) {
    let (mut p, mut o) = fresh();
    let r = run(&mut p, &mut o, batches(1, 4), &config(4)).unwrap();
    assert_eq!(r.optimizer_steps, 1);
    (p.values().to_vec(), p.moments().to_vec())
}

#[test]
fn a_non_finite_loss_stops_the_run_before_any_moment_moves() {
    let (want_values, want_moments) = after_one_step();
    // Step 2's first sequence (call 3: each batch has three sequences) reports NaN.
    let (inner, mut o) = fresh();
    let mut p = Poison {
        inner,
        calls: 0,
        nth: 3,
        gradient: false,
    };
    let err = train(&mut p, &mut o, batches(4, 4), &config(4), &MonotonicClock::new(), None, &mut Hooks::default()).unwrap_err();
    assert!(matches!(err, TrainError::NonFinite(_)), "{err}");
    assert_eq!(p.inner.step_count(), 1);
    assert_eq!(p.inner.values(), want_values.as_slice());
    assert_eq!(p.inner.moments(), want_moments.as_slice());
}

#[test]
fn a_non_finite_gradient_stops_the_run_before_any_moment_moves() {
    let (want_values, want_moments) = after_one_step();
    let (inner, mut o) = fresh();
    let mut p = Poison {
        inner,
        calls: 0,
        nth: 3,
        gradient: true,
    };
    let err = train(&mut p, &mut o, batches(4, 4), &config(4), &MonotonicClock::new(), None, &mut Hooks::default()).unwrap_err();
    assert!(matches!(&err, TrainError::NonFinite(m) if m.contains("squared norm")), "{err}");
    assert_eq!(p.inner.step_count(), 1);
    assert_eq!(p.inner.values(), want_values.as_slice());
    assert_eq!(p.inner.moments(), want_moments.as_slice());
}

// ---- the cap, groups, indices ---------------------------------------------------------------

struct Ticking(Cell<f64>);

impl Clock for Ticking {
    fn now_s(&self) -> f64 {
        let t = self.0.get();
        self.0.set(t + 1.0);
        t
    }
}

#[test]
fn the_cap_stops_the_loop_between_groups_only() {
    let mut cfg = config(20);
    cfg.cap = WallClockCap::new(9.0).unwrap();
    cfg.grad_accum = 2;
    let (mut p, mut o) = fresh();
    let clock = Ticking(Cell::new(0.0));
    let r = train(&mut p, &mut o, batches(40, 6), &cfg, &clock, None, &mut Hooks::default()).unwrap();
    assert_eq!(r.termination, Termination::WallClockCap);
    assert!(r.optimizer_steps > 0 && r.optimizer_steps < 20, "{} steps", r.optimizer_steps);
    assert_eq!(r.micro_batches, 2 * r.optimizer_steps, "no group was cut in half");
}

/// A clock that moves only when the test moves it.
struct Manual(Cell<f64>);

impl Clock for Manual {
    fn now_s(&self) -> f64 {
        self.0.get()
    }
}

/// Rung (d)'s rule on rung (d)'s numbers: 200 steps, cap 21,600 s, evaluated after step 10,
/// stop when `elapsed / 10 * 200 > 21600 - 900`. Each optimizer step costs `per_step` seconds.
fn eta_run(per_step: f64) -> (TrainResult, ToyProvider) {
    let mut cfg = config(200);
    cfg.cap = WallClockCap::new(21_600.0).unwrap();
    cfg.eta = Some(EtaRule { at_step: 10, margin_s: 900.0 });
    let (mut p, mut o) = fresh();
    let clock = Manual(Cell::new(1_000.0));
    let mut advance = |_: &qd_train::trainer::Progress| clock.0.set(clock.0.get() + per_step);
    let mut hooks = Hooks {
        on_progress: Some(&mut advance),
    };
    let r = train(&mut p, &mut o, batches(200, 3), &cfg, &clock, None, &mut hooks).unwrap();
    (r, p)
}

#[test]
fn the_eta_rule_stops_a_run_projected_past_the_cap_at_its_step_and_not_otherwise() {
    // 10 steps at 103.625 s (exact in binary) project to 20,725 s > 20,700 s: stop after step 10.
    let (r, p) = eta_run(103.625);
    assert_eq!(r.termination, Termination::EtaRule);
    assert_eq!(r.termination.as_str(), "eta_rule");
    assert!(!r.termination.finished());
    assert_eq!((r.optimizer_steps, p.step_count()), (10, 10), "it stops after the step it projects from");
    assert_eq!(r.eta_projected_s, Some(20_725.0));
    // Exactly at the line is not past it (`>`, not `>=`).
    let (r, _) = eta_run(103.5);
    assert_eq!(r.eta_projected_s, Some(20_700.0));
    assert_eq!((r.termination, r.optimizer_steps), (Termination::StepsExhausted, 200));
    // Comfortably inside: the run goes to its end, and the projection is recorded either way.
    let (r, _) = eta_run(60.0);
    assert_eq!((r.termination, r.optimizer_steps), (Termination::StepsExhausted, 200));
    assert_eq!(r.eta_projected_s, Some(12_000.0));
}

#[test]
fn an_eta_rule_that_could_not_mean_anything_is_refused_before_step_0() {
    for (at_step, margin_s) in [(0, 900.0), (200, 900.0), (500, 900.0), (10, 21_600.0), (10, f64::NAN), (10, -1.0)] {
        let mut cfg = config(200);
        cfg.cap = WallClockCap::new(21_600.0).unwrap();
        cfg.eta = Some(EtaRule { at_step, margin_s });
        let (mut p, mut o) = fresh();
        let err = run(&mut p, &mut o, batches(200, 3), &cfg).unwrap_err();
        assert!(matches!(&err, TrainError::Refused(m) if m.contains("ETA rule")), "({at_step}, {margin_s}): {err}");
        assert_eq!(p.step_count(), 0);
    }
}

#[test]
fn a_partial_group_at_the_end_of_the_data_is_refused() {
    let mut cfg = config(4);
    cfg.grad_accum = 2;
    let (mut p, mut o) = fresh();
    let err = run(&mut p, &mut o, batches(3, 1), &cfg).unwrap_err();
    assert!(matches!(&err, TrainError::Refused(m) if m.contains("accumulation group")), "{err}");
    assert_eq!(p.step_count(), 1, "the one full group was applied, the half one was not");
}

#[test]
fn grad_accum_sums_gradients_and_means_the_loss() {
    let mut cfg = config(2);
    cfg.grad_accum = 2;
    let (mut p, mut o) = fresh();
    let r = run(&mut p, &mut o, batches(4, 1), &cfg).unwrap();
    assert_eq!((r.optimizer_steps, r.micro_batches), (2, 4));
    let letters = &r.channel_log.iter().find(|(n, _)| n == "letter").unwrap().1;
    assert_eq!(letters.len(), 4, "every micro-batch's channels are logged");
}

#[test]
fn indices_must_strictly_increase() {
    let (mut p, mut o) = fresh();
    let data = vec![Ok(batch(0, 1)), Ok(batch(2, 1)), Ok(batch(1, 1))];
    let err = run(&mut p, &mut o, data, &config(3)).unwrap_err();
    assert!(matches!(&err, TrainError::Refused(m) if m.contains("strictly increasing")), "{err}");
}

#[test]
fn a_read_error_from_the_source_fails_the_run() {
    let (mut p, mut o) = fresh();
    let data = vec![Ok(batch(0, 1)), Err(TrainError::Io("shard read failed".into()))];
    assert!(matches!(run(&mut p, &mut o, data, &config(3)), Err(TrainError::Io(_))));
}

// ---- the layer-wise split ---------------------------------------------------------------------

#[test]
fn a_layer_wise_rate_is_refused_before_the_first_step_on_a_provider_without_one() {
    let mut cfg = config(3);
    cfg.optimizer.lower_layers = Some(LowerLayers { n: 1, lr_scale: 0.1 });
    let mut p = ToyProvider::new(V, H, L, 11).unwrap().without_lr_scale();
    let mut o = LetterSpanObjective::new(ToySpanHead::new(H), 0.75).unwrap();
    let before = p.values().to_vec();
    let err = run(&mut p, &mut o, batches(3, 1), &cfg).unwrap_err();
    assert!(matches!(&err, TrainError::Refused(m) if m.contains("per-entry learning rate")), "{err}");
    assert_eq!(p.step_count(), 0);
    assert_eq!(p.values(), before.as_slice());
    // The same recipe on a provider that has one runs: after one step, layer 0 has moved by
    // AdamW at lr * 0.1 and layer 1 at lr, from the same bank and clip coefficient.
    let (mut q, mut o2) = fresh();
    let init = q.values().to_vec();
    let r = run(&mut q, &mut o2, batches(1, 1), &cfg).unwrap();
    assert_eq!(r.optimizer_steps, 1);
    let s = r.steps[0];
    let hyper = AdamWHyper {
        lr: s.lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: s.clip_coefficient,
    };
    for (entry, rate) in [(1usize, s.lr * 0.1), (2, s.lr), (0, s.lr)] {
        let mut want = init[entry].clone();
        let mut m = Moments::zeros(want.len());
        adamw_update(&mut want, &q.grads()[entry], &mut m, &hyper, rate, 0.01, 1).unwrap();
        assert_eq!(q.values()[entry], want, "{} at {rate:e}", q.parameters()[entry].name);
    }
}

#[test]
fn the_objective_refuses_rows_in_neither_or_both_channels_and_a_wrong_letter() {
    assert!(LetterSpanObjective::<ToyBatch, ToySpanHead>::new(ToySpanHead::new(H), 0.0).is_err());
    let (mut p, mut o) = fresh();
    let mut b = batch(0, 1);
    b.rows[0].letter = None;
    assert!(matches!(run(&mut p, &mut o, vec![Ok(b)], &config(2)), Err(TrainError::Refused(_))));
    let (mut p, mut o) = fresh();
    let mut b = batch(0, 1);
    b.rows[0].letter.as_mut().unwrap().target = (b.rows[0].letter.unwrap().target + 1) % V as u32;
    assert!(matches!(run(&mut p, &mut o, vec![Ok(b)], &config(2)), Err(TrainError::Refused(_))));
    let (mut p, mut o) = fresh();
    let mut b = batch(0, 1);
    b.rows[2].letter = Some(LetterTarget { position: 1, target: b.rows[2].tokens[2] });
    assert!(matches!(run(&mut p, &mut o, vec![Ok(b)], &config(2)), Err(TrainError::Refused(_))));
}

#[test]
fn a_fresh_run_refuses_a_provider_that_has_already_stepped_and_a_mismatched_lr() {
    let (mut p, mut o) = fresh();
    run(&mut p, &mut o, batches(1, 1), &config(3)).unwrap();
    let (_, mut o2) = fresh();
    assert!(matches!(run(&mut p, &mut o2, batches(3, 1), &config(3)), Err(TrainError::Refused(_))));
    let (mut p, mut o) = fresh();
    let mut cfg = config(3);
    cfg.optimizer.lr = LR * 2.0;
    assert!(matches!(run(&mut p, &mut o, batches(3, 1), &cfg), Err(TrainError::Refused(_))));
}
