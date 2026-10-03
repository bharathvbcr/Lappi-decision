//! GPU gates on the real 2B weights. **Every test here is `#[ignore]`d** because it runs on the
//! GPU and loads 4.5 GB: `cargo test` never runs them, and reports them as ignored, not passed.
//!
//! ```text
//! cargo test --release -p qd-metal --test gpu -- --ignored --test-threads=1
//! ```

use std::time::{Duration, Instant};

use qd_metal::backend::{MetalBackend, MetalConfig, StateRecord};
use qd_metal::model::{Model, PrefixState, RECURRENT_MAX_SEQ};
use qd_metal::tokenizer::QwenTokenizer;
use qd_runtime::backend::{DecisionBackend, DecodeMode, QueryKind, SlotQuery};

/// Continuation vs one pass over the whole text, same backend: max |Δ log-softmax|. The same
/// bound as the parity gate's snapshot-path check.
const PATH_LOGPROB_ABS: f64 = 0.05;

const PREFIX: &str = "<|qd_begin|>\nfile: bowling_test.go\n\n@@ -3,7 +3,7 @@\n import \"testing\"\n \n \
func (g *Game) RollMany(runs int, pins int) {\n-\tfor i := 0; i < runs; i++ {\n+\tfor i := 0; i <= runs; i++ {\n \
\t\tg.Roll(pins)\n \t}\n }\n<|qd_ctx_end|>\n";
const SUFFIX: &str = "<|qd_slot|>defect_class\n<|qd_type|>choice\n<|qd_options_begin|>\nA. stub\nB. logic\n\
C. cosmetic\nD. clean\nZ. noul\n<|qd_options_end|>\n<|qd_answer|>";

fn setup() -> (Model, QwenTokenizer) {
    let snapshot = qd_metal::config::resolve_snapshot(None).expect("snapshot");
    let rt = tessl::GpuRuntime::new().expect("runtime");
    rt.set_async_encode(true).expect("async encode");
    let model = Model::load(&rt, &snapshot).expect("load");
    let tok = QwenTokenizer::load(&snapshot.join("tokenizer.json")).expect("tokenizer");
    (model, tok)
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from((x - y).abs())).fold(0.0, f64::max)
}

fn whole_logprobs(model: &Model, ids: &[u32], letters: &[u32]) -> Vec<f32> {
    let t = u32::try_from(ids.len()).unwrap();
    let out = model.run(ids, 1, t, None, false, None).expect("whole pass");
    let s = model.score(&out, &[t - 1], letters).expect("score");
    assert!(s.logits.iter().all(|x| x.is_finite()), "non-finite logits {:?}", s.logits);
    s.logprobs
}

fn continue_logprobs(model: &Model, state: &PrefixState, cont: &[u32], letters: &[u32], keep: bool) -> (Vec<f32>, Option<PrefixState>) {
    let s = u32::try_from(cont.len()).unwrap();
    let mut out = model.run(cont, 1, s, Some(state), keep, None).expect("continuation");
    let sc = model.score(&out, &[s - 1], letters).expect("score");
    (sc.logprobs, out.state.take())
}

#[test]
#[ignore = "GPU + model snapshot"]
fn gpu_continuations_read_the_snapshot_and_never_write_it() {
    let (model, tok) = setup();
    let letters = tok.letter_ids().to_vec();
    let prefix = tok.encode(PREFIX).unwrap();
    let full = tok.encode(&format!("{PREFIX}{SUFFIX}")).unwrap();
    assert_eq!(&full[..prefix.len()], &prefix[..], "prefix/suffix token boundary");
    let suffix = &full[prefix.len()..];
    assert!(suffix.len() as u32 > RECURRENT_MAX_SEQ, "the long suffix must take the chunked path");

    let (_, state) = model.prefill(&prefix).unwrap();
    let d0 = state.digest().unwrap();

    // Three continuation shapes, one per kernel path:
    //  - 1 token: gdn_recurrent + attn_prefix_decode
    //  - 5 tokens: gdn_recurrent + attn_prefix_rows
    //  - the whole suffix (> RECURRENT_MAX_SEQ): gdn_chunk_forward + attn_prefix_rows
    for n in [1usize, 5, suffix.len()] {
        let cont = &suffix[..n];
        let (got, none) = continue_logprobs(&model, &state, cont, &letters, false);
        assert!(none.is_none());
        let mut whole_ids = prefix.clone();
        whole_ids.extend_from_slice(cont);
        let want = whole_logprobs(&model, &whole_ids, &letters);
        let d = max_abs(&got, &want);
        println!("continuation of {n} tokens vs whole pass: max |d logprob| = {d:.5}");
        assert!(d <= PATH_LOGPROB_ABS, "{n}-token continuation differs from the whole pass by {d}");
        assert_eq!(state.digest().unwrap(), d0, "a read-only {n}-token continuation changed the snapshot");
    }

    // A batch of 3 *different* continuations from the one snapshot (prefix K/V, GDN and conv
    // state all read at batch stride 0): each row must agree with its own batch-1 answer, which
    // a row reading another row's suffix cache would not.
    let len = suffix.len() - 2;
    let conts: Vec<&[u32]> = (0..3).map(|r| &suffix[r..r + len]).collect();
    let s = u32::try_from(len).unwrap();
    let ids3: Vec<u32> = conts.iter().flat_map(|c| c.iter().copied()).collect();
    let out3 = model.run(&ids3, 3, s, Some(&state), false, None).unwrap();
    let sc3 = model.score(&out3, &[s - 1, 2 * s - 1, 3 * s - 1], &letters).unwrap();
    for (r, cont) in conts.iter().enumerate() {
        let (one, _) = continue_logprobs(&model, &state, cont, &letters, false);
        let row = &sc3.logprobs[r * letters.len()..(r + 1) * letters.len()];
        let d = max_abs(row, &one);
        println!("batch row {r} vs its batch-1 continuation: max |d logprob| = {d:.2e}");
        assert!(d <= 1e-4, "batch row {r} differs from its batch-1 continuation by {d}");
    }
    assert_eq!(state.digest().unwrap(), d0, "a batched read-only continuation changed the snapshot");
}

#[test]
#[ignore = "GPU + model snapshot"]
fn gpu_write_back_moves_only_the_new_state() {
    let (model, tok) = setup();
    let letters = tok.letter_ids().to_vec();
    let prefix = tok.encode(PREFIX).unwrap();
    let full = tok.encode(&format!("{PREFIX}{SUFFIX}")).unwrap();
    let suffix = &full[prefix.len()..];
    let (a, b) = suffix.split_at(suffix.len() / 2);

    let (_, state) = model.prefill(&prefix).unwrap();
    let d0 = state.digest().unwrap();
    let (_, extended) = continue_logprobs(&model, &state, a, &letters, true);
    let extended = extended.expect("write-back keeps a state");
    assert_eq!(extended.tokens() as usize, prefix.len() + a.len());
    assert_ne!(extended.digest().unwrap(), d0, "write-back produced the same state bytes");
    assert_eq!(state.digest().unwrap(), d0, "write-back modified the state it read from");

    // Continuing the extended state == one pass over prefix + a + b.
    let (got, _) = continue_logprobs(&model, &extended, b, &letters, false);
    let want = whole_logprobs(&model, &full, &letters);
    let d = max_abs(&got, &want);
    println!("prefix -> write-back {} -> continue {}: max |d logprob| vs whole = {d:.5}", a.len(), b.len());
    assert!(d <= PATH_LOGPROB_ABS, "chained continuation differs from the whole pass by {d}");
}

#[test]
#[ignore = "GPU + model snapshot"]
fn gpu_backend_readonly_decode_leaves_the_runtime_hash_unchanged() {
    let backend = MetalBackend::start(MetalConfig {
        snapshot: None,
        calibration_hash: qd_runtime::calibration::CalibrationTable::reference().hash(),
        queue_capacity: 4,
        max_entries: 8,
        max_tokens: 16_384,
        job_timeout: Duration::from_secs(300),
    })
    .expect("backend");
    let handle = backend.prefill(PREFIX).unwrap();
    let mut snap = backend.snapshot(&handle).unwrap();
    let before = snap.state_digest();
    let before_rec = StateRecord::parse(snap.state.as_bytes()).unwrap();
    let query = SlotQuery {
        slot_name: "defect_class",
        suffix: SUFFIX,
        rows: 5,
        kind: QueryKind::Letters,
    };
    let first = backend.decode_slot(&mut snap, &query, DecodeMode::ReadOnly).unwrap();
    qd_runtime::backend::validate_logits(&first, &query).unwrap();
    assert_eq!(snap.state_digest(), before, "read-only decode moved the runtime's state hash");
    let second = backend.decode_slot(&mut snap, &query, DecodeMode::ReadOnly).unwrap();
    // Same inputs, same kernels: equal to within float noise (a read-only decode that had
    // written the state would move these by far more).
    let d = max_abs(&first.values, &second.values);
    assert!(d <= 1e-6, "the same read-only decode twice differs by {d}");

    // The same prefix again is served from the prefill cache: same entry, same state record.
    let again = backend.prefill(PREFIX).unwrap();
    assert_eq!(again.state, handle.state, "the second prefill of one prefix was not reused");

    // Write-back must move the hash, or the read-only check above would pass vacuously.
    backend.decode_slot(&mut snap, &query, DecodeMode::WriteBack).unwrap();
    assert_ne!(snap.state_digest(), before, "write-back left the runtime's state hash unchanged");
    let rec = StateRecord::parse(snap.state.as_bytes()).unwrap();
    assert!(rec.tokens > handle.token_count as u64);
    // The record also carries `tokens`, which a write-back always grows, so the hash moving above
    // does not by itself show the device state moved; the record's device digest must (audit
    // 2026-10-03: the assertion above alone is satisfied by the token count).
    assert_ne!(rec.digest, before_rec.digest, "write-back left the device state digest unchanged");

    // A fresh snapshot of the same prefill is untouched by that write-back. Every snapshot is its
    // own entry (`Worker::snapshot` inserts one), so its record, and the runtime's hash of that
    // record, names a different entry: comparing `state_digest()` here failed on the first GPU run
    // (2026-10-02) with the device state unchanged. The device state is the record's `digest`.
    let fresh = backend.snapshot(&handle).unwrap();
    let fresh_rec = StateRecord::parse(fresh.state.as_bytes()).unwrap();
    assert_ne!(fresh_rec.entry, before_rec.entry, "a second snapshot reused the first one's entry");
    assert_eq!(
        (fresh_rec.tokens, fresh_rec.digest),
        (before_rec.tokens, before_rec.digest),
        "the write-back changed the device state of the cached prefill"
    );

    // Pointer heads are refused, typed, never answered with letters.
    let span = SlotQuery { kind: QueryKind::PointerStart, ..query };
    let mut fresh = fresh;
    assert!(backend.decode_slot(&mut fresh, &span, DecodeMode::ReadOnly).is_err());
}

/// The tower the v1 pins below were captured on: Qwen3.5-2B-Base snapshot `b1485b2f`, the one
/// [`setup`] resolves. A pin is a property of (weights, prompt, digest code); on other weights the
/// test refuses rather than reporting a mismatch it cannot interpret.
const PIN_WEIGHT_HASH: &str = "92f6bd1c32837882d87e8c2446ba23783563eb67811dbed71de493ef71fba2e1";

/// tessl's HEAD when the pins were reproduced by the 1a GPU slot (2026-10-03 03:42Z).
const PIN_TESSL_HEAD: &str = "cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9";

/// v1 prefix-state digests (`PrefixState::digest`) of three fixed states, captured from the digest
/// code as it stood **before** the parallel digest: Lappi `3d68484`, qd-metal `model.rs` sha256
/// `b95db97c…` (one SHA-256 per buffer on one thread), `Cargo.lock` sha256 `3c536ef1…`, captured
/// 2026-10-03 03:14Z and identical on a second process run. Characterization, not a fail-first
/// test: it pins the bytes the runtime's slot-isolation check compares
/// (`qd_runtime::answer::readonly_decode`), so any change to how the digest is computed must leave
/// these exactly as they are.
///
/// The state bytes are kernel output, so a pin is a property of (weights, tessl's kernels, digest
/// code). Reproduced 2026-10-03 03:42Z on tessl [`PIN_TESSL_HEAD`] with the dirty tree ledger row
/// `28505f4c` records in `tessl_dirty_files` (30 paths, kernels among them): a tessl change can move
/// these without the digest being at fault, which the test's failure message says.
///
/// `prefill_8192` (221 MB, twelve ~17 MB K/V buffers: the size at which the parallel digest runs
/// more than one round of 8 threads) was not in the pre-parallel capture. Its value was computed by
/// BOTH the one-thread digest and the parallel one, equal, in three GPU slots before the one-thread
/// digest was deleted: qdmgpu1 and qdmm4 (sha2 0.10) and qdm1b (sha2 0.11), 2026-10-03.
const PINNED_V1: &[(&str, &str)] = &[
    ("prefill_short", "3ba7b21a3ec8e15a40d6f6e539dcd56007c28a6ec2ee20499ce1ea51719b6f6b"),
    ("write_back", "85b7dab4bd04ca714af1481b2c89f60dc8a59eea0abb0ae0758f0fc99e36121e"),
    ("prefill_2000", "408872926bc09797c3473be4c87a9746d46baf82c7f04693aa70f6e83e683749"),
    ("prefill_8192", "41faece110e132f163652a9b03eea57a0b7b298aeb6d19b6e2db9d639a672b1c"),
];

/// The four pinned states: the short prefill, a write-back extension of it (different K/V
/// lengths), and prefills of 2,000 and 8,192 tokens, whose K/V buffers dominate the state.
fn pinned_states(model: &Model, tok: &QwenTokenizer) -> Vec<(&'static str, PrefixState)> {
    let letters = tok.letter_ids().to_vec();
    let prefix = tok.encode(PREFIX).unwrap();
    let full = tok.encode(&format!("{PREFIX}{SUFFIX}")).unwrap();
    let suffix = &full[prefix.len()..];
    let (a, _) = suffix.split_at(suffix.len() / 2);
    let (_, short) = model.prefill(&prefix).unwrap();
    let (_, extended) = continue_logprobs(model, &short, a, &letters, true);
    let repeated = |n: usize| -> PrefixState {
        let mut ids = Vec::new();
        while ids.len() < n {
            ids.extend_from_slice(&prefix);
        }
        ids.truncate(n);
        model.prefill(&ids).unwrap().1
    };
    vec![
        ("prefill_short", short),
        ("write_back", extended.expect("write-back keeps a state")),
        ("prefill_2000", repeated(2000)),
        ("prefill_8192", repeated(8192)),
    ]
}

#[test]
#[ignore = "GPU + model snapshot"]
fn gpu_prefix_state_digest_v1_is_pinned() {
    let (model, tok) = setup();
    assert_eq!(
        model.weight_hash(),
        PIN_WEIGHT_HASH,
        "the pins were captured on Qwen3.5-2B-Base b1485b2f; refusing to compare on other weights"
    );
    let mut got: Vec<(&str, String)> = Vec::new();
    for (name, s) in &pinned_states(&model, &tok) {
        // The GPU-free size formula the digest scratch's retain cap is computed from.
        let formula = qd_metal::model::prefix_state_nbytes(model.config(), s.tokens()).unwrap();
        assert_eq!(
            s.nbytes(),
            formula,
            "{name}: device bytes vs prefix_state_nbytes"
        );
        let d = s.digest().unwrap();
        // Twice more: the scratch is reused across digests on this thread.
        assert_eq!(s.digest().unwrap(), d, "{name}: a repeat digest differs");
        assert_eq!(s.digest().unwrap(), d, "{name}: a third digest differs");
        println!(
            "v1 digest {name} ({} MB): {}",
            s.nbytes() / 1_000_000,
            qd_runtime::hex(&d)
        );
        got.push((name, qd_runtime::hex(&d)));
    }
    let want: Vec<(&str, String)> = PINNED_V1.iter().map(|(n, h)| (*n, h.to_string())).collect();
    if got != want {
        let tessl = qd_metal::ledger::Provenance::of(None).map_or_else(
            |e| format!("tessl's tree could not be read: {e}"),
            |p| format!("tessl is now {} with {} dirty paths", p.tessl.head, p.tessl.dirty.len()),
        );
        panic!(
            "the v1 prefix-state digests changed:\n  got    {got:?}\n  pinned {want:?}\nBlame the \
             digest only after its CPU guards fail (qd-metal \
             the_v1_record_matches_an_independent_oracle, qd-runtime sha256_parallel_tests): if \
             they pass, tessl's kernels moved the state bytes ({tessl}; the pins were reproduced \
             on {PIN_TESSL_HEAD} plus the dirty tree in ledger row 28505f4c)."
        );
    }
}

/// Two row counts that used to be multiplied unchecked, each refused as the caller's input.
///
/// `score`: `RunOutput`'s fields are public, and `batch * seq` was a plain `u32` multiply. Built
/// in release, 65,537 x 65,536 wrapped to 65,536 rows and the call reached tessl, which refused
/// the 1-row residual as a `Gpu` error; in debug it panicked. Fails against the pre-fix code.
///
/// `run`: the MLP's `rows * intermediate` element count is converted to `u32` in the layer loop,
/// after the activations are allocated and earlier layers encoded. The guard moves that refusal
/// ahead of any allocation. Not run against the pre-fix code on purpose: there, 699,051 rows
/// allocate the activations first, tens of GB on this Mac, before the layer loop refuses them.
#[test]
#[ignore = "GPU + model snapshot"]
fn gpu_row_counts_past_u32_are_refused_as_input() {
    use qd_metal::error::MetalError;

    let (model, tok) = setup();
    let ids = tok.encode(PREFIX).unwrap();
    let t = u32::try_from(ids.len()).unwrap();
    let mut out = model.run(&ids, 1, t, None, false, None).expect("run");
    let letters = [ids[0]];
    out.batch = 65_537;
    out.seq = 65_536;
    match model.score(&out, &[0], &letters) {
        Err(MetalError::Input(m)) => assert!(m.contains("overflow"), "{m}"),
        other => panic!("a wrapping batch x seq was not refused as input: {other:?}"),
    }

    let inter = model.config().intermediate;
    let rows = u32::try_from(u32::MAX as usize / inter + 1).unwrap();
    let huge = vec![0u32; rows as usize];
    let started = std::time::Instant::now();
    match model.run(&huge, 1, rows, None, false, None) {
        Err(MetalError::Input(m)) => assert!(m.contains("mlp elements"), "{m}"),
        other => panic!("{rows} rows x {inter} were not refused as input: {:?}", other.map(|o| o.seq)),
    }
    assert!(started.elapsed() < Duration::from_secs(5), "the refusal took {:?}", started.elapsed());
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// One product-path decision through [`MetalBackend`]: prefill, snapshot, and the two read-only
/// `decode_slot` calls, each timed from the caller's side (job hop included). Also returns the
/// letter logits of both passes and the prefill's entry id.
fn product_decision(
    backend: &MetalBackend,
    prefix: &str,
    suffixes: &[String; 2],
    rows: usize,
) -> (qd_metal::decision::ProductSample, Vec<f32>, u64) {
    let t0 = Instant::now();
    let handle = backend.prefill(prefix).expect("prefill");
    let prefill_ms = ms_since(t0);
    let ts = Instant::now();
    let mut snap = backend.snapshot(&handle).expect("snapshot");
    let snapshot_ms = ms_since(ts);
    let mut decode_ms = [0.0; 2];
    let mut logits = Vec::new();
    for (i, suffix) in suffixes.iter().enumerate() {
        let query = SlotQuery {
            slot_name: "defect_class",
            suffix,
            rows,
            kind: QueryKind::Letters,
        };
        let td = Instant::now();
        let l = backend
            .decode_slot(&mut snap, &query, DecodeMode::ReadOnly)
            .expect("decode");
        decode_ms[i] = ms_since(td);
        logits.extend_from_slice(&l.values);
    }
    let total_ms = ms_since(t0);
    let entry = StateRecord::parse(handle.state.as_bytes())
        .expect("handle record")
        .entry;
    let sample = qd_metal::decision::ProductSample {
        total_ms,
        prefill_ms,
        snapshot_ms,
        decode_ms,
    };
    (sample, logits, entry)
}

/// Item B (Fable ruling 2, step 6a): the product path's cost over the Model-driven decision
/// bench. At T = 131, 409, 8192 on the frozen bench context it times `MetalBackend` (the prefix
/// tokenized once and prefix + suffix re-tokenized per pass, the prompt sha256, the job-channel
/// hops, the entry bookkeeping) against `decision::run_decision` on the same ids. Phases run
/// Model, backend, Model in one process, one model loaded at a time; `WARMUP` then `ITERS` per
/// phase; the two Model phases bound the drift. The equal-work check: both paths' letter
/// logits are bit-identical. `max_entries: 1` makes every prefill a cache miss (the snapshot
/// evicts it), as every prefill in the Model bench is. On this branch it times the committed
/// backend.rs/tokenizer.rs (`QwenTokenizer::encode`), not main's dirty `encode_untrusted`, so it
/// does not measure H1.
///
/// It writes one `throughput` row (`decision::build_product_row`) to `QDM_PRODUCT_LEDGER`, or
/// none when that is `none` (a development run; its numbers are not citable). The row is written
/// before the checks are asserted, so a run whose checks fail is a failed row, not a missing one.
#[test]
#[ignore = "GPU + model snapshot"]
fn gpu_product_path_host_cost_vs_model() {
    use qd_metal::decision::{self, ProductPathT, RunContext};
    use qd_metal::ledger::{self, Provenance, TreeState};

    const WARMUP: usize = 2;
    const ITERS: usize = 5;
    const K: usize = 4;
    const TS: [usize; 3] = [131, 409, 8192];
    let target = std::env::var("QDM_PRODUCT_LEDGER").expect(
        "say where the row goes: QDM_PRODUCT_LEDGER=ledger/mac-qd-metal-<date>.jsonl, or none",
    );
    let ledger_path = (target != "none").then(|| std::path::PathBuf::from(&target));
    if let Some(p) = &ledger_path {
        ledger::check_ledger_path(p).expect("a qd-metal ledger path");
    }
    let started = Instant::now();
    let provenance = Provenance::of(None).expect("provenance");
    let snapshot = qd_metal::config::resolve_snapshot(None).expect("snapshot");
    let tok = QwenTokenizer::load(&snapshot.join("tokenizer.json")).expect("tokenizer");
    let answers = tok.answer_ids(K + 1).unwrap();
    let prompts: Vec<_> = TS
        .iter()
        .map(|&t| {
            let p = decision::build_prompt(&tok, t, K).unwrap();
            let text = decision::prompt_text(&p, K).unwrap();
            (p, text)
        })
        .collect();

    // The host tokenization the backend adds, timed alone: the prefix once, prefix + suffix twice.
    let encode_ms: Vec<Vec<f64>> = prompts
        .iter()
        .map(|(p, (prefix, suffixes))| {
            (0..ITERS)
                .map(|_| {
                    let t0 = Instant::now();
                    assert_eq!(tok.encode(prefix).unwrap(), p.prefix);
                    for s in suffixes {
                        tok.encode(&format!("{prefix}{s}")).unwrap();
                    }
                    ms_since(t0)
                })
                .collect()
        })
        .collect();

    // Per T: every total ms, and the first sample's logits. Also the facts the row records.
    struct Phase {
        per_t: Vec<(Vec<f64>, Vec<f32>)>,
        device: String,
        weight_hash: String,
        vocab: usize,
        load_s: f64,
    }
    let model_phase = || -> Phase {
        let rt = decision::RuntimeKind::Timestamps.open().expect("runtime");
        let t0 = Instant::now();
        let model = Model::load(&rt, &snapshot).expect("load");
        let load_s = t0.elapsed().as_secs_f64();
        let mut per_t = Vec::new();
        for (p, _) in &prompts {
            for _ in 0..WARMUP {
                decision::run_decision(&model, p, &answers).unwrap();
            }
            let s: Vec<_> = (0..ITERS)
                .map(|_| decision::run_decision(&model, p, &answers).unwrap())
                .collect();
            per_t.push((s.iter().map(|x| x.total_ms).collect(), s[0].logits.clone()));
        }
        Phase {
            per_t,
            device: rt.device_name(),
            weight_hash: model.weight_hash().to_string(),
            vocab: model.config().vocab,
            load_s,
        }
    };
    let before = model_phase();

    let backend = MetalBackend::start(MetalConfig {
        snapshot: Some(snapshot.clone()),
        calibration_hash: qd_runtime::calibration::CalibrationTable::reference().hash(),
        queue_capacity: 4,
        max_entries: 1,
        max_tokens: 16_384,
        job_timeout: Duration::from_secs(300),
    })
    .expect("backend");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let mut last_entry = 0u64;
    let mut product = Vec::new();
    for (j, (_, (prefix, suffixes))) in prompts.iter().enumerate() {
        let (mut samples, mut logits_same, mut fresh) = (Vec::new(), 0, 0);
        for i in 0..WARMUP + ITERS {
            let (s, logits, entry) = product_decision(&backend, prefix, suffixes, K + 1);
            let is_fresh = entry > last_entry;
            last_entry = entry;
            if i >= WARMUP {
                logits_same += usize::from(bits(&logits) == bits(&before.per_t[j].1));
                fresh += usize::from(is_fresh);
                samples.push(s);
            }
        }
        product.push((samples, logits_same, fresh));
    }
    drop(backend);
    let after = model_phase();
    let tessl_after = TreeState::read(&provenance.tessl.dir).expect("tessl state");

    let results: Vec<ProductPathT> = prompts
        .iter()
        .zip(product)
        .enumerate()
        .map(
            |(j, ((p, _), (samples, logits_same, fresh)))| ProductPathT {
                prompt: p.clone(),
                model_before_ms: before.per_t[j].0.clone(),
                model_after_ms: after.per_t[j].0.clone(),
                product: samples,
                encode_ms: encode_ms[j].clone(),
                logits_same,
                model_phases_same: bits(&before.per_t[j].1) == bits(&after.per_t[j].1),
                fresh_prefills: fresh,
            },
        )
        .collect();
    let ctx = RunContext {
        device: before.device.clone(),
        snapshot: snapshot.clone(),
        vocab: before.vocab,
        weight_hash: before.weight_hash.clone(),
        tokenizer_hash: tok.hash().to_string(),
        load_s: before.load_s,
        wall_clock_s: started.elapsed().as_secs_f64(),
        provenance,
        tessl_after,
    };
    let row = decision::build_product_row(&TS, K, WARMUP, &results, &ctx).expect("the row builds");
    let value = |key: String| match &row.metrics[&key] {
        qd_train::tristate::TriState::Ran { value, .. } => value.to_string(),
        qd_train::tristate::TriState::NotRun { reason } => format!("not run: {reason}"),
    };
    for t in &results {
        let k = format!("product_path.t{}", t.prompt.target);
        println!("T={} ({} tok)", t.prompt.target, t.prompt.prefix.len());
        for m in [
            "model_ms",
            "product_ms",
            "delta_ms",
            "host_encode_ms",
            "logits_bit_identical",
            "fresh_prefills",
        ] {
            println!("  {m}: {}", value(format!("{k}.{m}")));
        }
    }
    match &ledger_path {
        Some(p) => {
            let stamp = ledger::write_row(p, &row).expect("the row is written");
            println!("ledger row {} appended to {}", stamp.row_id, p.display());
        }
        None => println!("QDM_PRODUCT_LEDGER=none: NO ROW WRITTEN; these numbers are not citable"),
    }
    println!("status {:?}", row.status);

    for t in &results {
        let n = t.product.len();
        assert_eq!(
            t.logits_same, n,
            "T={}: product vs Model logits",
            t.prompt.target
        );
        assert!(
            t.model_phases_same,
            "T={}: the Model phases disagree",
            t.prompt.target
        );
        assert_eq!(
            t.fresh_prefills, n,
            "T={}: a timed prefill hit the cache",
            t.prompt.target
        );
    }
    assert_eq!(
        row.status,
        qd_train::ledger::Status::Completed,
        "{}",
        row.notes
    );
}
