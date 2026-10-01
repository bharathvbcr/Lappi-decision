//! Byte-level parity with the Python pipeline, on the shard set the oracle built through it
//! (`tools/qd_train_oracle_shards.py`). Every assertion compares against a value Python
//! computed; none against a value this crate computed earlier.
//!
//! The RNG intermediates are checked before any batch order, so a mismatch is located to
//! `SeedSequence`, `PCG64` or the shuffle rather than reported as "the order differs".

mod common;

use std::collections::HashMap;

use common::{fixture, open_train, oracle, repo_root};
use qd_train::held_out::DataConfig;
use qd_train::np_random::{Pcg64, SeedSequence, default_rng};
use qd_train::pyjson::{self, CANONICAL, SORTED_DEFAULT, float_repr};
use qd_train::shards::{
    Batch, CONSUMED_PREFIX_DOMAIN, ConsumedPrefix, MAX_PADDING_WASTE, MAX_POSITIONS_PER_BATCH,
    MAX_ROWS_PER_BATCH, NO_SPAN, PAD_ID, REMAP_FORMAT, SEQUENCE_INDEX_FORMAT, SHARD_FORMAT,
    SLOT_CHOICE, SLOT_LM, SLOT_SCORE, SLOT_SPAN, SPAN_ABSTAIN,
};
use qd_train::supervision::{
    RESERVED_NOUL_ROWS, SpanSupervision, ft_supervision, plan_span_batch, span_head_rows,
};
use qd_train::tristate::TriState;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("u64"))
        .collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_i64().expect("i64"))
        .collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_bool().expect("bool"))
        .collect()
}

fn flat_i64(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("2-D array")
        .iter()
        .flat_map(i64s)
        .collect()
}

fn flat_bool(v: &Value) -> Vec<bool> {
    v.as_array()
        .expect("2-D array")
        .iter()
        .flat_map(bools)
        .collect()
}

#[test]
fn pyjson_renders_floats_strings_and_objects_as_python_does() {
    let cases = oracle("pyjson.json");
    for case in cases["floats"].as_array().expect("floats") {
        let value = case["value"].as_f64().expect("float");
        assert_eq!(
            float_repr(value),
            case["repr"].as_str().expect("repr"),
            "repr({value:e})"
        );
        assert_eq!(
            pyjson::dumps(&case["value"], CANONICAL).expect("dumps"),
            case["canonical"].as_str().expect("canonical"),
            "canonical_json({value:e})"
        );
    }
    for case in cases["strings"].as_array().expect("strings") {
        let v = &case["value"];
        assert_eq!(
            pyjson::dumps(v, CANONICAL).expect("dumps"),
            case["canonical"].as_str().expect("c")
        );
        assert_eq!(
            pyjson::dumps(v, SORTED_DEFAULT).expect("dumps"),
            case["sorted_default"].as_str().expect("s")
        );
    }
    let obj = &cases["object"];
    assert_eq!(
        pyjson::dumps(&obj["value"], CANONICAL).expect("dumps"),
        obj["canonical"].as_str().expect("c")
    );
    assert_eq!(
        pyjson::dumps(&obj["value"], SORTED_DEFAULT).expect("dumps"),
        obj["sorted_default"].as_str().expect("s")
    );
}

#[test]
fn data_config_defaults_and_contract_constants_are_pythons() {
    let meta = oracle("meta.json");
    let dc = &meta["data_config"];
    let config = DataConfig::default();
    let strings = |v: &Value| -> Vec<String> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|s| s.as_str().expect("str").to_owned())
            .collect()
    };
    assert_eq!(
        config.held_out_families(),
        strings(&dc["held_out_families"]).as_slice()
    );
    assert_eq!(
        config.held_out_path_markers(),
        strings(&dc["held_out_path_markers"]).as_slice()
    );
    let roots: Vec<String> = config
        .held_out_roots()
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    assert_eq!(roots, strings(&dc["held_out_roots"]));
    assert_eq!(config.seed(), dc["seed"].as_u64().expect("seed"));

    let c = &meta["constants"];
    let int = |k: &str| c[k].as_i64().unwrap_or_else(|| panic!("{k}"));
    assert_eq!(i64::from(PAD_ID), int("PAD_ID"));
    assert_eq!(MAX_ROWS_PER_BATCH as i64, int("MAX_ROWS_PER_BATCH"));
    assert_eq!(
        MAX_POSITIONS_PER_BATCH as i64,
        int("MAX_POSITIONS_PER_BATCH")
    );
    assert_eq!(
        MAX_PADDING_WASTE,
        c["MAX_PADDING_WASTE"].as_f64().expect("f64")
    );
    assert_eq!(i64::from(NO_SPAN), int("NO_SPAN"));
    assert_eq!(i64::from(SPAN_ABSTAIN), int("SPAN_ABSTAIN"));
    assert_eq!(i64::from(SLOT_LM), int("SLOT_LM"));
    assert_eq!(i64::from(SLOT_CHOICE), int("SLOT_CHOICE"));
    assert_eq!(i64::from(SLOT_SCORE), int("SLOT_SCORE"));
    assert_eq!(i64::from(SLOT_SPAN), int("SLOT_SPAN"));
    assert_eq!(RESERVED_NOUL_ROWS as i64, int("RESERVED_NOUL_ROWS"));
    assert_eq!(
        CONSUMED_PREFIX_DOMAIN,
        c["CONSUMED_PREFIX_DOMAIN"].as_str().expect("d").as_bytes()
    );
    assert_eq!(SHARD_FORMAT, c["SHARD_FORMAT"].as_str().expect("f"));
    assert_eq!(REMAP_FORMAT, c["REMAP_FORMAT"].as_str().expect("f"));
    assert_eq!(
        SEQUENCE_INDEX_FORMAT,
        c["SEQUENCE_INDEX_FORMAT"].as_str().expect("f")
    );
}

#[test]
fn reserved_noul_rows_is_the_runtimes_and_the_abstention_is_last() {
    // Pinned against qd-runtime's source, as python/tests/test_heads.py pins Python's copy.
    let src = repo_root().join("crates/qd-runtime/src");
    let schema = std::fs::read_to_string(src.join("schema.rs")).expect("schema.rs");
    let line = schema
        .lines()
        .find(|l| {
            l.trim_start()
                .starts_with("pub const RESERVED_NOUL_ROWS: usize =")
        })
        .expect("RESERVED_NOUL_ROWS in qd-runtime/src/schema.rs");
    let value: usize = line
        .rsplit('=')
        .next()
        .and_then(|v| v.trim().trim_end_matches(';').trim().parse().ok())
        .expect("a usize literal");
    assert_eq!(value, RESERVED_NOUL_ROWS, "qd-runtime says {value}");
    let answer = std::fs::read_to_string(src.join("answer.rs")).expect("answer.rs");
    assert!(
        answer.contains("context.line_count() + RESERVED_NOUL_ROWS"),
        "span_rows changed shape"
    );
    assert!(
        answer.contains("plan.rows - RESERVED_NOUL_ROWS"),
        "the abstention is no longer last"
    );
    assert_eq!(span_head_rows(7).expect("rows"), 7 + RESERVED_NOUL_ROWS);
    assert!(span_head_rows(0).is_err());
}

#[test]
fn seed_sequence_pcg64_and_permutation_match_numpy() {
    let rng = oracle("rng.json");
    for stream in rng["streams"].as_array().expect("streams") {
        let entropy = u64s(&stream["entropy"]);
        let seq = SeedSequence::new(&entropy);
        let pool: Vec<u64> = seq.pool().iter().map(|&w| u64::from(w)).collect();
        assert_eq!(
            pool,
            u64s(&stream["pool"]),
            "SeedSequence({entropy:?}).pool"
        );
        assert_eq!(
            seq.generate_state_u64(4),
            u64s(&stream["generate_state_u64"]),
            "generate_state {entropy:?}"
        );
        let mut pcg = Pcg64::new(&seq);
        let raw: Vec<u64> = (0..8).map(|_| pcg.next_u64()).collect();
        assert_eq!(
            raw,
            u64s(&stream["random_raw"]),
            "PCG64 random_raw {entropy:?}"
        );
        let perm = default_rng(&entropy).permutation(&(0..16u64).collect::<Vec<_>>());
        assert_eq!(
            perm,
            u64s(&stream["permutation_16"]),
            "permutation(16) {entropy:?}"
        );
    }
}

#[test]
fn the_reader_sees_every_sequence_and_check_python_sees() {
    let facts = oracle("reader.json");
    let reader = open_train();
    let header = reader.header();
    assert_eq!(
        header.shard_hash().expect("hash"),
        facts["shard_hash"].as_str().expect("hash")
    );
    assert_eq!(
        reader.len() as u64,
        facts["n_sequences"].as_u64().expect("n")
    );
    assert_eq!(
        header.total_tokens,
        facts["total_tokens"].as_u64().expect("t")
    );
    assert_eq!(
        header.max_seq_len,
        facts["max_seq_len"].as_u64().expect("m")
    );
    assert_eq!(header.buckets, u64s(&facts["buckets"]));
    assert_eq!(reader.lengths(), u64s(&facts["lengths"]).as_slice());
    let digests = facts["sequence_sha256"].as_array().expect("digests");
    let candidates = facts["candidates"].as_array().expect("candidates");
    let spans = facts["span_target"].as_array().expect("spans");
    for i in 0..reader.len() {
        let seq = reader.sequence(i).expect("sequence");
        let bytes: Vec<u8> = seq.iter().flat_map(|t| t.to_le_bytes()).collect();
        assert_eq!(
            hex(&Sha256::digest(&bytes)),
            digests[i].as_str().expect("d"),
            "sequence {i}"
        );
        assert_eq!(
            reader
                .candidates(i)
                .expect("c")
                .iter()
                .map(|&p| i64::from(p))
                .collect::<Vec<_>>(),
            i64s(&candidates[i])
        );
        assert_eq!(
            i64::from(reader.slot_kind(i)),
            facts["slot_kind"][i].as_i64().expect("k")
        );
        assert_eq!(
            i64::from(reader.target_index(i)),
            facts["target_index"][i].as_i64().expect("t")
        );
        assert_eq!(
            reader.span_target(i).map(i64::from).to_vec(),
            i64s(&spans[i])
        );
    }

    // Python's recorded checks, by name and state. `shard_code_current` is compared for
    // presence only: the fixture is frozen and `qd_data` moves, so its verdict is expected
    // to change over time (it is recorded either way; door.rs tests the refusal).
    let py = &facts["to_json"];
    for (name, check) in py["checks"].as_object().expect("checks") {
        let ours = reader
            .check(name)
            .unwrap_or_else(|| panic!("Python records {name}; Rust does not"));
        if name == "shard_code_current" {
            continue;
        }
        assert_same_state(name, ours, check);
    }
    assert_same_state("coverage", reader.coverage(), &py["coverage"]);
    assert_same_state(
        "slot_coverage",
        reader.slot_coverage(),
        &py["slot_coverage"],
    );
    assert_same_state("span_check", reader.span_check(), &py["span_check"]);
    if let TriState::Ran {
        coverage: Some(c), ..
    } = reader.slot_coverage()
    {
        assert_eq!(c.n, py["slot_coverage"]["n"].as_u64().expect("n"));
        assert_eq!(
            c.n_total,
            py["slot_coverage"]["n_total"].as_u64().expect("n_total")
        );
    } else {
        panic!("slot coverage must be a Ran with coverage on a set that pins its index");
    }
    let waste = reader.padding_waste().expect("waste");
    let detail = py["padding_waste"]["detail"].as_str().expect("detail");
    let words: Vec<&str> = detail.split_whitespace().collect();
    assert_eq!(
        words[0].parse::<u64>().expect("padding"),
        waste.padded - waste.real,
        "{detail}"
    );
    assert_eq!(
        words[3].parse::<u64>().expect("positions"),
        waste.padded,
        "{detail}"
    );
    assert_eq!(
        waste.tristate().is_pass(),
        py["padding_waste"]["passed"].as_bool().expect("passed")
    );
}

fn assert_same_state(name: &str, ours: &TriState, python: &Value) {
    match (ours, python["state"].as_str()) {
        (TriState::Ran { passed, .. }, Some("ran")) => {
            assert_eq!(
                Some(*passed),
                python["passed"].as_bool(),
                "{name}: passed differs"
            );
        }
        (TriState::NotRun { .. }, Some("not_run")) => {}
        (ours, theirs) => panic!("{name}: Rust {ours:?}, Python {theirs:?}"),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn batch_order_and_consumed_digest_match_python_on_every_config() {
    let reader = open_train();
    let configs = oracle("batches.json");
    let configs = configs.as_array().expect("configs");
    assert_eq!(configs.len(), 8, "2 seeds x 2 epochs x 2 batch_tokens");
    let mut digests: HashMap<String, (u64, u64, u64)> = HashMap::new();
    for cfg in configs {
        let (seed, epoch, bt) = (
            cfg["seed"].as_u64().expect("seed"),
            cfg["epoch"].as_u64().expect("epoch"),
            cfg["batch_tokens"].as_u64().expect("bt"),
        );
        let what = format!("seed={seed} epoch={epoch} batch_tokens={bt}");
        let plans = reader.plan(bt, seed, epoch).expect("plan");
        let expected = cfg["batches"].as_array().expect("batches");
        assert_eq!(plans.len(), expected.len(), "{what}: batch count");
        let mut prefix = ConsumedPrefix::new();
        for (index, (plan, want)) in plans.iter().zip(expected).enumerate() {
            assert_eq!(index as u64, want["index"].as_u64().expect("index"));
            assert_eq!(
                plan.bucket as u64,
                want["bucket"].as_u64().expect("bucket"),
                "{what} batch {index}"
            );
            assert_eq!(
                plan.width as u64,
                want["width"].as_u64().expect("width"),
                "{what} batch {index}"
            );
            assert_eq!(
                plan.rows.iter().map(|&r| r as u64).collect::<Vec<_>>(),
                u64s(&want["rows"]),
                "{what} batch {index}: membership and row order"
            );
            let batch = reader.batch(plan, index as u64).expect("batch");
            assert_eq!(
                batch.lengths,
                i64s(&want["lengths"]),
                "{what} batch {index}: lengths"
            );
            let mut one = ConsumedPrefix::new();
            one.fold_batch(&batch);
            assert_eq!(
                one.hexdigest(),
                want["fold_digest"].as_str().expect("fd"),
                "{what} batch {index}: _fold bytes"
            );
            prefix.fold_batch(&batch);
        }
        assert_eq!(
            prefix.n_folded(),
            cfg["n_folded"].as_u64().expect("n_folded")
        );
        assert_eq!(
            prefix.hexdigest(),
            cfg["consumed_digest"].as_str().expect("digest"),
            "{what}: consumed digest"
        );
        let via_iter: Vec<Batch> = reader
            .batches(bt, seed, epoch)
            .expect("batches")
            .collect::<Result<_, _>>()
            .expect("ok");
        let mut again = ConsumedPrefix::new();
        via_iter.iter().for_each(|b| again.fold_batch(b));
        assert_eq!(
            again.hexdigest(),
            prefix.hexdigest(),
            "{what}: batches() and plan()+batch() agree"
        );
        assert!(
            digests
                .insert(prefix.hexdigest(), (seed, epoch, bt))
                .is_none(),
            "{what}: two configs, one order"
        );
    }
}

#[test]
fn ft_supervision_and_span_plans_match_python_on_every_batch() {
    let reader = open_train();
    let dumps = oracle("supervision.json");
    let mut span_batches = 0;
    for cfg in dumps.as_array().expect("configs") {
        let (seed, epoch, bt) = (
            cfg["seed"].as_u64().expect("seed"),
            cfg["epoch"].as_u64().expect("epoch"),
            cfg["batch_tokens"].as_u64().expect("bt"),
        );
        let batches: Vec<Batch> = reader
            .batches(bt, seed, epoch)
            .expect("batches")
            .collect::<Result<_, _>>()
            .expect("ok");
        let expected = cfg["batches"].as_array().expect("batches");
        assert_eq!(batches.len(), expected.len());
        for (batch, want) in batches.iter().zip(expected) {
            let what = format!("bt={bt} batch {}", batch.index);
            let sup = ft_supervision(batch).expect("supervision");
            assert_eq!(
                vec![batch.rows() as u64, sup.columns as u64],
                u64s(&want["mask_shape"]),
                "{what}"
            );
            let positions: Vec<Vec<i64>> = sup
                .letter_positions()
                .iter()
                .map(|&(r, c)| vec![r as i64, c as i64])
                .collect();
            let want_positions: Vec<Vec<i64>> = want["letter_positions"]
                .as_array()
                .expect("lp")
                .iter()
                .map(i64s)
                .collect();
            assert_eq!(positions, want_positions, "{what}: letter mask");
            let targets: Vec<i64> = sup
                .letter_positions()
                .iter()
                .map(|&(r, c)| i64::from(sup.targets[r * sup.columns + c]))
                .collect();
            assert_eq!(
                targets,
                i64s(&want["letter_targets"]),
                "{what}: letter targets"
            );
            assert_eq!(
                sup.n_supervised as u64,
                want["n_supervised"].as_u64().expect("n"),
                "{what}"
            );
            assert_eq!(
                sup.n_answers() as u64,
                want["n_answers"].as_u64().expect("n"),
                "{what}"
            );
            match (&sup.span, want["span"].is_null()) {
                (None, true) => {}
                (Some(span), false) => {
                    span_batches += 1;
                    let w = &want["span"];
                    assert_eq!(
                        span.rows.iter().map(|&r| r as i64).collect::<Vec<_>>(),
                        i64s(&w["rows"]),
                        "{what}"
                    );
                    for r in &span.rows {
                        assert!(
                            !sup.mask[r * sup.columns..(r + 1) * sup.columns]
                                .iter()
                                .any(|&m| m),
                            "{what}: a span row is in the letter mask"
                        );
                    }
                    assert_eq!(span.query_index, i64s(&w["query_index"]), "{what}");
                    assert_eq!(span.start, i64s(&w["start"]), "{what}");
                    assert_eq!(span.end, i64s(&w["end"]), "{what}");
                    assert_eq!(span.abstaining, bools(&w["abstaining"]), "{what}");
                    assert_eq!(
                        span.candidate_counts(),
                        i64s(&w["candidate_counts"]),
                        "{what}"
                    );
                    let plan = plan_span_batch(span).expect("plan");
                    let p = &w["plan"];
                    assert_eq!(
                        plan.candidate_pos,
                        flat_i64(&p["candidate_pos"]),
                        "{what}: candidate_pos"
                    );
                    assert_eq!(
                        plan.candidate_valid,
                        flat_bool(&p["candidate_valid"]),
                        "{what}: candidate_valid"
                    );
                    assert_eq!(plan.n_candidates, i64s(&p["n_candidates"]), "{what}");
                    assert_eq!(plan.query_index, i64s(&p["query_index"]), "{what}");
                    assert_eq!(
                        plan.gold_start,
                        i64s(&p["gold_start"]),
                        "{what}: gold_start"
                    );
                    assert_eq!(plan.gold_end, i64s(&p["gold_end"]), "{what}: gold_end");
                    assert_eq!(plan.abstaining, bools(&p["abstaining"]), "{what}");
                    assert_eq!(
                        plan.runtime_rows(),
                        i64s(&p["runtime_rows"]),
                        "{what}: runtime_rows"
                    );
                    assert_eq!(
                        plan.max_rows() as u64,
                        p["max_rows"].as_u64().expect("max_rows"),
                        "{what}"
                    );
                    assert_eq!(
                        plan.n_spans() as u64,
                        p["n_spans"].as_u64().expect("n_spans"),
                        "{what}"
                    );
                    for (k, &abstains) in plan.abstaining.iter().enumerate() {
                        if abstains {
                            assert_eq!(
                                plan.gold_start[k], plan.n_candidates[k],
                                "{what}: the abstain row is last"
                            );
                        }
                    }
                }
                (ours, _) => panic!(
                    "{what}: span channel presence differs: Rust {}",
                    ours.is_some()
                ),
            }
        }
    }
    assert!(
        span_batches >= 2,
        "the fixture must exercise the span channel ({span_batches} batches did)"
    );
    let abstaining = dumps
        .as_array()
        .expect("c")
        .iter()
        .flat_map(|c| c["batches"].as_array().expect("b").iter())
        .filter_map(|b| b["span"]["abstaining"].as_array())
        .flatten()
        .filter(|a| a.as_bool() == Some(true))
        .count();
    assert!(
        abstaining >= 1,
        "the fixture must exercise an abstaining span row"
    );
}

// --- refusals the oracle's valid batches never reach ------------------------------------------

fn small_batch(
    kinds: Vec<u8>,
    spans: Option<Vec<[i32; 2]>>,
    line_starts: Option<Vec<bool>>,
) -> Batch {
    let rows = kinds.len();
    Batch {
        index: 0,
        bucket: 0,
        width: 4,
        sequences: (0..rows).collect(),
        tokens: vec![7; rows * 4],
        lengths: vec![4; rows],
        slot_kind: kinds,
        target_index: vec![2; rows],
        span_target: spans,
        line_starts,
    }
}

#[test]
fn ft_supervision_refuses_what_the_trainer_refuses() {
    assert!(
        ft_supervision(&small_batch(vec![SLOT_LM], None, None)).is_err(),
        "SLOT_LM inside an FT batch"
    );
    assert!(
        ft_supervision(&small_batch(
            vec![SLOT_SPAN],
            None,
            Some(vec![true, false, false, false])
        ))
        .is_err(),
        "span without span_target"
    );
    assert!(
        ft_supervision(&small_batch(vec![SLOT_SPAN], Some(vec![[0, 0]]), None)).is_err(),
        "span without line_starts"
    );
    assert!(
        ft_supervision(&small_batch(
            vec![SLOT_SPAN],
            Some(vec![[SPAN_ABSTAIN, 0]]),
            Some(vec![true, false, false, false])
        ))
        .is_err(),
        "half-abstaining span row"
    );
    let ok = ft_supervision(&small_batch(
        vec![SLOT_CHOICE, SLOT_SPAN],
        Some(vec![[NO_SPAN, NO_SPAN], [0, 0]]),
        Some(
            vec![false; 4]
                .into_iter()
                .chain([true, false, true, false])
                .collect(),
        ),
    ))
    .expect("a valid mixed batch");
    assert_eq!(
        ok.letter_positions(),
        vec![(0, 2)],
        "only the choice row is a letter; the span row is excluded"
    );
    assert_eq!(ok.n_answers(), 2);
}

#[test]
fn plan_span_batch_refuses_a_gold_off_the_candidates_and_an_empty_candidate_set() {
    let span = |start: i64, mask: Vec<bool>| SpanSupervision {
        rows: vec![0],
        query_index: vec![1],
        start: vec![start],
        end: vec![start],
        line_starts: mask,
        width: 4,
        abstaining: vec![false],
    };
    assert!(
        plan_span_batch(&span(1, vec![true, false, true, false])).is_err(),
        "gold not a candidate"
    );
    assert!(
        plan_span_batch(&span(0, vec![false; 4])).is_err(),
        "no candidates"
    );
    let plan = plan_span_batch(&span(2, vec![true, false, true, false])).expect("valid");
    assert_eq!((plan.gold_start[0], plan.runtime_rows()[0]), (1, 3));
}

#[test]
fn the_fixture_is_the_one_the_oracle_wrote() {
    // A regenerated fixture changes every digest above; this names the generator so a
    // reader of a failure knows what to re-run.
    let meta = oracle("meta.json");
    assert_eq!(meta["corpus_n"], 24);
    assert!(
        fixture().join("shards/train/tokens.u32").is_file(),
        "run tools/qd_train_oracle_shards.py --out {}",
        fixture().display()
    );
}
