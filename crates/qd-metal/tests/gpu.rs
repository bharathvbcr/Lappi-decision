//! GPU gates on the real 2B weights. **Every test here is `#[ignore]`d** because it runs on the
//! GPU and loads 4.5 GB: `cargo test` never runs them, and reports them as ignored, not passed.
//!
//! ```text
//! cargo test --release -p qd-metal --test gpu -- --ignored --test-threads=1
//! ```

use std::time::Duration;

use qd_metal::backend::{MetalBackend, MetalConfig, StateRecord};
use qd_metal::model::{EmbedPath, Model, PrefixState, RECURRENT_MAX_SEQ};
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

/// `EmbedPath::Device` (`tessl::qwen35::embed_rows`) against the host gather. The widening is a
/// 16-bit shift on both sides, so the claim is bit identity, not a tolerance: of the gathered
/// rows, of a whole pass's logits, and of the prefix state the runtime hashes.
#[test]
#[ignore = "GPU + model snapshot"]
fn gpu_embed_paths_are_bit_identical() {
    let (mut model, tok) = setup();
    let letters = tok.letter_ids().to_vec();
    let full = tok.encode(&format!("{PREFIX}{SUFFIX}")).unwrap();
    let prefix = tok.encode(PREFIX).unwrap();
    let suffix = &full[prefix.len()..];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();

    // The gather itself, including the first and last vocabulary rows.
    let vocab = u32::try_from(model.config().vocab).unwrap();
    let mut ids = full.clone();
    ids.extend_from_slice(&[0, vocab - 1]);
    let mut per_path = Vec::new();
    for path in [EmbedPath::Host, EmbedPath::Device] {
        model.set_embed_path(path);
        let rows = model.gather_embeddings(&ids).unwrap();
        let t = u32::try_from(full.len()).unwrap();
        let (out, state) = model.prefill(&prefix).unwrap();
        drop(out);
        let digest = state.digest().unwrap();
        let (cont, _) = continue_logprobs(&model, &state, suffix, &letters, false);
        let whole = model.run(&full, 1, t, None, false, None).unwrap();
        let logits = model.score(&whole, &[t - 1], &letters).unwrap().logits;
        per_path.push((bits(&rows), digest, bits(&cont), bits(&logits)));
    }
    let (h, d) = (&per_path[0], &per_path[1]);
    assert_eq!(h.0, d.0, "the gathered embedding rows differ between the host and the device path");
    assert_eq!(h.1, d.1, "the prefix state differs between the paths");
    assert_eq!(h.2, d.2, "a continuation's letter log-probs differ between the paths");
    assert_eq!(h.3, d.3, "a whole pass's letter logits differ between the paths");
    println!("embed paths bit-identical: {} gathered rows, prefix state, continuation and whole-pass letters", ids.len());
}
