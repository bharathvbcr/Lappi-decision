//! The real batch and the real span head behind the objective's traits:
//!
//! * every row of every fixture batch re-shapes to what Python's `ft_supervision` and
//!   `plan_span_batch` said (`oracle/supervision.json`, the dump `tests/parity.rs` pins L-data's
//!   port to);
//! * the row-level head adapter gives L-head's batch-level gradients and loss;
//! * the span-head init loader refuses every file that is not exactly the four tensors;
//! * the loop runs end to end on CPU over real fixture batches -- a toy tower, the real head, the
//!   real supervision -- and folds exactly the batches it trained on.

mod common;

use std::cell::Cell;

use qd_export::safetensors::{Dtype, PlannedTensor, Writer};
use qd_train::ft_data::{load_span_head, HostSpanHead, RealBatch};
use qd_train::mock::{SplitMix, ToyProvider};
use qd_train::objective::{FtBatch, LetterSpanObjective, SpanHead as _};
use qd_train::recipe::{self, OptimizerRecipe};
use qd_train::run_control::{Clock, WallClockCap};
use qd_train::schedule::LrSchedule;
use qd_train::shards::{Batch, ConsumedPrefix};
use qd_train::span_head::{SpanHead, SpanRowInput, PARAMETER_NAMES};
use qd_train::trainer::{train, ConsumedBatch, Hooks, HostParams, Termination, TrainConfig, TrainError};

fn i64s(v: &serde_json::Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

fn batches(seed: u64, epoch: u64, bt: u64) -> Vec<Batch> {
    common::open_train()
        .batches(bt, seed, epoch)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

#[test]
fn every_fixture_row_is_what_pythons_supervision_said() {
    let dumps = common::oracle("supervision.json");
    let (mut letters, mut spans, mut abstains) = (0, 0, 0);
    for cfg in dumps.as_array().unwrap() {
        let (seed, epoch, bt) = (
            cfg["seed"].as_u64().unwrap(),
            cfg["epoch"].as_u64().unwrap(),
            cfg["batch_tokens"].as_u64().unwrap(),
        );
        let got = batches(seed, epoch, bt);
        let want = cfg["batches"].as_array().unwrap();
        assert_eq!(got.len(), want.len());
        for (batch, w) in got.into_iter().zip(want) {
            let rb = RealBatch::new(batch).unwrap();
            let what = format!("bt={bt} batch {}", rb.index());
            // Letters: (row, position) and the target, row-major, exactly Python's.
            let mut lp = Vec::new();
            let mut lt = Vec::new();
            for r in 0..rb.n_rows() {
                if let Some(l) = rb.letter(r) {
                    lp.push(vec![r as i64, i64::from(l.position)]);
                    lt.push(i64::from(l.target));
                    assert_eq!(rb.row_tokens(r)[l.position as usize + 1], l.target, "{what} row {r}");
                }
            }
            let want_lp: Vec<Vec<i64>> = w["letter_positions"].as_array().unwrap().iter().map(i64s).collect();
            assert_eq!(lp, want_lp, "{what}: letter positions");
            assert_eq!(lt, i64s(&w["letter_targets"]), "{what}: letter targets");
            letters += lp.len();
            // Spans: per span row, in row order, Python's query, candidates and gold rows.
            let span_rows: Vec<usize> = (0..rb.n_rows()).filter(|&r| rb.span(r).is_some()).collect();
            if w["span"].is_null() {
                assert!(span_rows.is_empty(), "{what}: Python has no span row");
                continue;
            }
            let plan = &w["span"]["plan"];
            assert_eq!(span_rows.len() as u64, plan["n_spans"].as_u64().unwrap(), "{what}");
            for (k, &r) in span_rows.iter().enumerate() {
                let s = rb.span(r).unwrap();
                let n = plan["n_candidates"][k].as_u64().unwrap() as usize;
                let cands: Vec<i64> = i64s(&plan["candidate_pos"][k])[..n].to_vec();
                assert_eq!(s.candidates().iter().map(|&c| i64::from(c)).collect::<Vec<_>>(), cands, "{what} span {k}");
                assert_eq!(i64::from(s.query_index()), plan["query_index"][k].as_i64().unwrap(), "{what} span {k}");
                assert_eq!(s.gold_start() as i64, plan["gold_start"][k].as_i64().unwrap(), "{what} span {k}");
                assert_eq!(s.gold_end() as i64, plan["gold_end"][k].as_i64().unwrap(), "{what} span {k}");
                assert_eq!(s.is_abstaining(), plan["abstaining"][k].as_bool().unwrap(), "{what} span {k}");
                abstains += usize::from(s.is_abstaining());
                assert!(rb.letter(r).is_none(), "{what}: a span row is never a letter row");
            }
            spans += span_rows.len();
        }
    }
    assert!(letters > 100 && spans > 10 && abstains > 0, "{letters} letters, {spans} spans, {abstains} abstaining");
}

#[test]
fn a_row_whose_letter_runs_past_its_tokens_is_refused() {
    let mut b = batches(20260919, 0, 3192).remove(0);
    // Shorten row 0 to end at its own target_index: the letter would predict a padded position.
    // L-data's `Batch::validate` (`_check_supervision`) is the layer that refuses it.
    b.lengths[0] = i64::from(b.target_index[0]) + 1;
    let err = RealBatch::new(b).err().expect("refused");
    assert!(matches!(&err, TrainError::Refused(m) if m.contains("target_index")), "{err}");
}

fn random_head(h: usize, seed: u64) -> SpanHead {
    let mut rng = SplitMix::new(seed);
    let mut v = |n: usize| -> Vec<f32> { (0..n).map(|_| ((rng.next_u64() % 2001) as f32 / 1000.0 - 1.0) * 0.3).collect() };
    SpanHead::new(h, v(h * h), v(h * h), v(h), v(h)).unwrap()
}

#[test]
fn the_row_level_adapter_gives_l_heads_batch_level_gradients_and_loss() {
    let h = 8;
    // The fixture's first span batch, with deterministic stand-in hidden states.
    let rb = batches(20260919, 0, 6385)
        .into_iter()
        .map(|b| RealBatch::new(b).unwrap())
        .find(|rb| (0..rb.n_rows()).any(|r| rb.span(r).is_some()))
        .expect("a span batch");
    let rows: Vec<_> = (0..rb.n_rows()).filter_map(|r| rb.span(r).cloned()).collect();
    let k = rows.len();
    assert!(k >= 2 && rows.iter().any(|s| s.is_abstaining()), "rows of both kinds");
    let hidden: Vec<Vec<f32>> = rows
        .iter()
        .enumerate()
        .map(|(i, s)| (0..s.positions().len() * h).map(|j| ((i * 131 + j) as f32 * 0.37).sin()).collect())
        .collect();
    let span_weight = 1.0f32;

    // L-head, batch level.
    let head = random_head(h, 5);
    let mut bank = SpanHead::zeros(h).unwrap();
    let inputs: Vec<SpanRowInput<'_>> = rows.iter().zip(&hidden).map(|(p, x)| SpanRowInput { plan: p, hidden: x }).collect();
    let whole = head.loss_and_backward(&inputs, span_weight, &mut bank).unwrap();

    // The adapter, row by row at the objective's scale.
    let mut adapter = HostSpanHead::new(head.clone()).unwrap();
    adapter.zero_grads();
    assert!(adapter.grads().iter().all(Option::is_none), "no span row yet: torch's grads are None");
    let scale = (f64::from(span_weight) / (2.0 * k as f64)) as f32;
    let mut loss_sum = 0.0;
    for ((s, x), want) in rows.iter().zip(&hidden).zip(&whole.d_hidden) {
        let (loss, dh) = adapter.loss_and_grad(s, s.positions(), x, h, scale).unwrap();
        loss_sum += loss;
        for (a, b) in dh.iter().zip(&want.rows) {
            assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-3), "dh {a} vs batch-level {b}");
        }
    }
    let mean = loss_sum / (2.0 * k as f64);
    assert!((mean - f64::from(whole.loss)).abs() <= 1e-6 * f64::from(whole.loss).abs(), "{mean} vs {}", whole.loss);
    let grads = adapter.grads();
    assert_eq!(grads.len(), 4);
    for (g, (name, w)) in grads.into_iter().zip(bank.named()) {
        let g = g.unwrap_or_else(|| panic!("{name}: a span row reads all four tensors, so each has a gradient"));
        for (a, b) in g.iter().zip(w) {
            assert!((a - b).abs() <= 1e-5 * b.abs().max(1e-3), "weight grad {a} vs batch-level {b}");
        }
    }
    // A width or position mismatch is refused, not scored.
    let s = &rows[0];
    assert!(adapter.loss_and_grad(s, s.positions(), &hidden[0], h + 1, scale).is_err());
    assert!(adapter.loss_and_grad(s, &s.positions()[1..], &hidden[0], h, scale).is_err());
}

fn write_head(path: &std::path::Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<f32>)]) {
    let plan = tensors
        .iter()
        .map(|(n, d, s, _)| PlannedTensor { name: n.to_string(), dtype: *d, shape: s.clone() })
        .collect();
    let mut w = Writer::create(path, plan).unwrap();
    for p in w.order() {
        let (_, d, _, v) = tensors.iter().find(|t| t.0 == p.name).unwrap();
        w.begin(&p.name).unwrap();
        let bytes: Vec<u8> = match d {
            Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            _ => v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect(),
        };
        w.write(&bytes).unwrap();
        w.end().unwrap();
    }
    w.finish().unwrap();
}

#[test]
fn the_span_head_init_is_exactly_four_f32_tensors_and_its_digest_is_the_files() {
    let h = 3;
    let dir = common::scratch("head-init");
    let head = random_head(h, 9);
    let full = |prefix: &str| -> Vec<(String, Dtype, Vec<usize>, Vec<f32>)> {
        head.named()
            .iter()
            .zip([vec![h, h], vec![h, h], vec![h], vec![h]])
            .map(|((n, v), s)| (format!("{prefix}{n}"), Dtype::F32, s, v.to_vec()))
            .collect()
    };
    let as_refs = |v: &[(String, Dtype, Vec<usize>, Vec<f32>)]| -> Vec<(String, Dtype, Vec<usize>, Vec<f32>)> { v.to_vec() };
    let write = |name: &str, t: Vec<(String, Dtype, Vec<usize>, Vec<f32>)>| {
        let p = dir.join(name);
        let refs: Vec<(&str, Dtype, Vec<usize>, Vec<f32>)> = t.iter().map(|(n, d, s, v)| (n.as_str(), *d, s.clone(), v.clone())).collect();
        write_head(&p, &refs);
        p
    };
    for prefix in ["", "span_head."] {
        let p = write(&format!("ok{}.safetensors", prefix.len()), as_refs(&full(prefix)));
        let (loaded, sha) = load_span_head(&p, h).unwrap();
        assert_eq!(loaded, head, "prefix {prefix:?}");
        use sha2::Digest;
        let want: String = sha2::Sha256::digest(std::fs::read(&p).unwrap()).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(sha, want);
    }
    let mut missing = full("");
    missing.pop();
    let mut extra = full("");
    extra.push(("lm_head.weight".into(), Dtype::F32, vec![1], vec![0.0]));
    let mut mixed = full("");
    mixed[0].0 = format!("span_head.{}", PARAMETER_NAMES[0]);
    let mut bent = full("");
    bent[2].2 = vec![1, h];
    let mut bf16 = full("");
    bf16[3].1 = Dtype::Bf16;
    let mut nan = full("");
    nan[0].3[1] = f32::NAN;
    for (what, t) in [("missing", missing), ("extra", extra), ("mixed", mixed), ("bent", bent), ("bf16", bf16), ("nan", nan)] {
        let p = write(&format!("{what}.safetensors"), t);
        assert!(load_span_head(&p, h).is_err(), "{what} was admitted");
    }
    assert!(load_span_head(&dir.join("ok0.safetensors"), h + 1).is_err(), "a head of another width");
    std::fs::remove_dir_all(&dir).unwrap();
}

struct Fixed(Cell<f64>);

impl Clock for Fixed {
    fn now_s(&self) -> f64 {
        self.0.get()
    }
}

#[test]
fn the_loop_trains_on_real_fixture_batches_and_folds_exactly_those() {
    let h = 4;
    let all: Vec<Batch> = batches(20260919, 0, 3192);
    // Through the first span batch.
    let first_span = all
        .iter()
        .position(|b| b.slot_kind.contains(&qd_train::shards::SLOT_SPAN))
        .expect("a span batch in the fixture");
    let taken: Vec<Batch> = all[..=first_span].to_vec();
    let n = taken.len() as u64;
    let lr = 1e-3;
    let cfg = TrainConfig {
        schedule: LrSchedule::real_ft(lr, n).unwrap(),
        cap: WallClockCap::new(3600.0).unwrap(),
        grad_accum: 1,
        optimizer: OptimizerRecipe { lr, beta2: recipe::DEFAULT_BETA2, lower_layers: None },
        max_grad_norm: recipe::MAX_GRAD_NORM,
        epoch: 0,
        seed: 20260919,
        checkpoint: None,
        eta: None,
    };
    let vocab = common::open_train().header().vocab_size as usize;
    let mut provider = ToyProvider::new(vocab, h, 2, 3).unwrap();
    let mut objective: LetterSpanObjective<RealBatch, HostSpanHead> =
        LetterSpanObjective::new(HostSpanHead::new(random_head(h, 4)).unwrap(), 1.0).unwrap();
    let source = taken.iter().cloned().map(RealBatch::new);
    let before = objective.host_ref().unwrap().values().concat();
    let r = train(&mut provider, &mut objective, source, &cfg, &Fixed(Cell::new(0.0)), None, &mut Hooks::default()).unwrap();
    assert_eq!(r.termination, Termination::StepsExhausted);
    assert_eq!(r.optimizer_steps, n);
    assert!(r.loss_log.points().iter().all(|p| p.loss.is_finite() && p.loss > 0.0));
    let span = &r.channel_log.iter().find(|(c, _)| c == "span").unwrap().1;
    assert!(span[first_span] > 0.0, "the span batch trained the span channel");
    assert!(span[..first_span].iter().all(|&s| s == 0.0), "letter-only batches have no span loss");
    assert_ne!(objective.host_ref().unwrap().values().concat(), before, "the head moved");
    // The digest is L-data's fold over exactly the batches trained on, in order.
    let mut want = ConsumedPrefix::new();
    for b in &taken {
        want.fold_batch(b);
    }
    assert_eq!(r.consumed_digest, want.hexdigest());
    assert_eq!(r.consumed_n, n);
}

use qd_train::trainer::Objective as _;
