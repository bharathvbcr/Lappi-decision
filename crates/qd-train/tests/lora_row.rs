//! The `lora` ledger row: `recipe.optimizer_recipe` "lora", quick with its reason (rule 8), and
//! the seven digests `campaign/lora-2b-preregistered.json` `init.record_in_every_row` names. The
//! LoRA-A digest is recomputed by `attach_lora`'s rule and a mismatch refuses the row. Rows are
//! built from the oracle fixture and written only to a temp `ledger/mac-ojas-*.jsonl`.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use qd_train::export::METAL_TRAINER;
use qd_train::ledger::{
    self, Environment, FtRecipe, FtRow, LORA_OPTIMIZER_RECIPE, LoraInit, LoraQuickReason, Protocol,
    RunFacts, Stamp, Status, WallClockSource, lora_ft_row,
};
use qd_train::lora_init::{d17_zero_init_digest, lora_a_digest, recompute_lora_a};
use qd_train::run_control::LossLog;
use qd_train::step::ParamSpec;
use qd_train::trainer::{BatchCounts, Termination, TrainResult};
use serde_json::Value;

const SHARD_HASH: &str = "d773b87666e1b042279271ab0f891246b7268d4ce0cad2c3e677bb415c147e1a";
const SNAPSHOT: &str = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c";
const SAFETENSORS_SHA: &str = "928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d";
const DTYPE_MAP_SHA: &str = "f03e32c6115fe57e701de501b202232e53b0ec28339cc25aa75d7b6c85c5641a";
const HEAD_DIGEST: &str = "1c30346c9ea8c1fe12cbc41d556587187ad5a27fe165ac7bc9c69d6bcd7def68";

fn oracle() -> Value {
    let path = common::crate_dir().join("tests/fixtures/lora-plan-oracle.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn specs(v: &Value) -> Vec<ParamSpec> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let shape: Vec<usize> = p["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            ParamSpec::new(p["name"].as_str().unwrap(), &shape)
        })
        .collect()
}

struct Fixture {
    block: Value,
    adapters: Vec<ParamSpec>,
    host: Vec<ParamSpec>,
    modules: Vec<(String, usize)>,
    rank: usize,
}

fn fixture(seed_index: usize) -> Fixture {
    let o = oracle();
    let block = o["seeds"][seed_index].clone();
    let adapters = specs(&block["adapter_parameters"]);
    let mut host = specs(&Value::Array(vec![serde_json::json!({
        "name": block["d17"]["name"], "shape": block["d17"]["shape"]
    })]));
    host.extend(specs(&block["span_head"]));
    let modules = block["modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["module"].as_str().unwrap().to_owned(),
                m["in_features"].as_u64().unwrap() as usize,
            )
        })
        .collect();
    Fixture {
        block,
        adapters,
        host,
        modules,
        rank: o["constants"]["LORA_RANK"].as_u64().unwrap() as usize,
    }
}

fn lora_recipe() -> FtRecipe {
    FtRecipe {
        trainer: METAL_TRAINER.into(),
        tag: "epoch".into(),
        device: "metal".into(),
        lr: 2e-4,
        passes: 1,
        batches: 60,
        width: 9638,
        span_weight: 1.0,
        shard_hash: SHARD_HASH.into(),
        backbone_snapshot: SNAPSHOT.into(),
        backbone_vocab: 248_320,
        backbone_params: 1_881_825_088,
        attn_implementation: "sdpa".into(),
        optimizer_recipe: LORA_OPTIMIZER_RECIPE.into(),
        wall_clock_cap_s: Some(21_600.0),
        batch_tokens: Some(35_403),
        no_memorise: true,
        beta2: None,
        lower_layers: None,
        provider: "ojas-qwen35 over tessl".into(),
        operands: "bf16".into(),
        optimizer_groups: "three".into(),
        train_attention_mask: "padding".into(),
        deterministic: false,
        extra: BTreeMap::new(),
    }
}

fn base_row(recipe: FtRecipe, seed: u64) -> FtRow {
    let protocol = Protocol::for_recipe(
        &recipe,
        "d".repeat(64).as_str(),
        "t".repeat(64).as_str(),
        seed,
    )
    .unwrap();
    let result = TrainResult {
        termination: Termination::StepsExhausted,
        optimizer_steps: 60,
        micro_batches: 60,
        counts: BatchCounts {
            supervised_tokens: 7,
            span_rows: 2,
            padded_positions: 10,
            total_positions: 100,
        },
        loss_log: LossLog::new(),
        steps: Vec::new(),
        channel_log: Vec::new(),
        consumed_digest: "ab".repeat(32),
        consumed_n: 60,
        next_index: 60,
        wall_clock_s: 100.0,
        final_checkpoint: None,
        eta_projected_s: None,
    };
    let facts = RunFacts {
        cap_s: 21_600.0,
        provider: "ojas-qwen35 over tessl",
        head_init: None,
        eta: None,
    };
    FtRow {
        protocol,
        recipe,
        status: Status::Completed,
        quick_reason: String::new(),
        code_commit: "2733015".to_string() + &"0".repeat(33),
        env: Environment::rust_metal("fixture-host".into()),
        metrics: ledger::ft_metrics(&result, &facts).unwrap(),
        wall_clock_s: 100.0,
        wall_clock_source: WallClockSource::Recorder,
        notes: "lora row built from the oracle fixture; no run happened".into(),
    }
}

fn init<'a>(f: &'a Fixture, a_digest: &'a str) -> LoraInit<'a> {
    let _ = f;
    LoraInit {
        span_head_content_digest: HEAD_DIGEST,
        lora_a_digest: a_digest,
        base_safetensors_sha256: SAFETENSORS_SHA,
        dtype_map_sha256: DTYPE_MAP_SHA,
        plan_seed: 0,
    }
}

fn stamp() -> Stamp {
    Stamp {
        row_id: "00000000-0000-4000-8000-0000000000a1".into(),
        written_at: "2026-10-17T10:00:00+00:00".into(),
        prev_row_hash: None,
    }
}

fn good_row(f: &Fixture, seed: u64) -> FtRow {
    let a = f.block["a_digest"].as_str().unwrap();
    lora_ft_row(
        base_row(lora_recipe(), seed),
        &init(f, a),
        &f.adapters,
        &f.host,
        &[
            LoraQuickReason::Smoke,
            LoraQuickReason::FewerThanThreeSeeds { seeds: 1 },
        ],
    )
    .unwrap()
}

#[test]
fn the_lora_row_carries_the_label_quick_with_its_reason_and_all_seven_digests() {
    let f = fixture(0);
    let line = good_row(&f, 0).line(&stamp()).unwrap();
    let v: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["recipe"]["optimizer_recipe"], "lora");
    assert_eq!(v["run_kind"], "ft");
    assert_eq!(v["quick"], true);
    let reason = v["quick_reason"].as_str().unwrap();
    assert!(
        reason.contains("smoke") && reason.contains("fewer than 3 seeds"),
        "{reason}"
    );
    let m = &v["metrics"];
    let val = |k: &str| {
        m[k]["value"]
            .as_str()
            .unwrap_or_else(|| panic!("{k}: {}", m[k]))
    };
    // 1. span-head init content digest, 2. D17 init digest, 3. LoRA-A digest.
    assert_eq!(val("train.span_head_init_digest"), HEAD_DIGEST);
    assert_eq!(
        val("lora.d17_init_digest"),
        d17_zero_init_digest(17, 6).unwrap(),
        "recomputed from the host entry's shape"
    );
    assert_eq!(
        val("lora.a_init_digest"),
        f.block["a_digest"].as_str().unwrap()
    );
    assert!(
        m["lora.a_init_digest"]["detail"]
            .as_str()
            .unwrap()
            .contains("recomputed")
    );
    // 4. base snapshot and safetensors sha256, 5. dtype map, 6. plan seed, 7. train shard hash.
    assert_eq!(v["recipe"]["backbone_snapshot"], SNAPSHOT);
    assert_eq!(val("lora.base_safetensors_sha256"), SAFETENSORS_SHA);
    assert_eq!(val("lora.dtype_map_sha256"), DTYPE_MAP_SHA);
    assert_eq!(m["corpus.plan_seed"]["value"], 0);
    assert_eq!(v["recipe"]["shard_hash"], SHARD_HASH);
    for k in [
        "train.span_head_init_digest",
        "lora.d17_init_digest",
        "lora.a_init_digest",
        "lora.base_safetensors_sha256",
        "lora.dtype_map_sha256",
        "corpus.plan_seed",
    ] {
        assert_eq!(m[k]["passed"], true, "{k}");
    }
}

#[test]
fn a_lora_a_digest_the_rule_does_not_give_voids_the_row() {
    let f = fixture(0);
    let wrong_seed = fixture(1).block["a_digest"].as_str().unwrap().to_owned();
    let err = lora_ft_row(
        base_row(lora_recipe(), 0),
        &init(&f, &wrong_seed),
        &f.adapters,
        &f.host,
        &[LoraQuickReason::Smoke],
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("LoRA-A digest") && err.contains("voids the row"),
        "{err}"
    );
    // The digest is right for seed 0 but the row's protocol says seed 7: recomputed at 7, refused.
    let a0 = f.block["a_digest"].as_str().unwrap();
    assert!(
        lora_ft_row(
            base_row(lora_recipe(), 7),
            &init(&f, a0),
            &f.adapters,
            &f.host,
            &[LoraQuickReason::Smoke],
        )
        .is_err()
    );
    // A checkpoint A that differs by one ulp from the rule's draw hashes apart and is refused.
    let mut drawn = recompute_lora_a(0, f.rank, &f.modules).unwrap();
    drawn[0].1[0] = f32::from_bits(drawn[0].1[0].to_bits() ^ 1);
    let tampered = lora_a_digest(f.rank, &drawn).unwrap();
    assert!(
        lora_ft_row(
            base_row(lora_recipe(), 0),
            &init(&f, &tampered),
            &f.adapters,
            &f.host,
            &[LoraQuickReason::Smoke],
        )
        .is_err()
    );
}

#[test]
fn a_row_that_is_not_quick_for_a_stated_reason_or_not_lora_is_refused() {
    let f = fixture(0);
    let a = f.block["a_digest"].as_str().unwrap();
    let go = |row: FtRow, reasons: &[LoraQuickReason]| {
        lora_ft_row(row, &init(&f, a), &f.adapters, &f.host, reasons)
    };
    assert!(
        go(base_row(lora_recipe(), 0), &[]).is_err(),
        "no quick reason"
    );
    assert!(
        go(
            base_row(lora_recipe(), 0),
            &[LoraQuickReason::FewerThanThreeSeeds { seeds: 3 }]
        )
        .is_err(),
        "three seeds is not fewer than three"
    );
    let f32_recipe = FtRecipe {
        optimizer_recipe: ledger::METAL_FULL_FT_OPTIMIZER_RECIPE.into(),
        ..lora_recipe()
    };
    assert!(go(base_row(f32_recipe, 0), &[LoraQuickReason::Smoke]).is_err());
    let split = FtRecipe {
        lower_layers: Some((8, 0.1)),
        ..lora_recipe()
    };
    assert!(go(base_row(split, 0), &[LoraQuickReason::Smoke]).is_err());
    // A malformed digest is not hex; none of them is copied through unchecked.
    for bad in ["", "xyz", &"A".repeat(64)] {
        let mut i = init(&f, a);
        i.dtype_map_sha256 = bad;
        assert!(
            lora_ft_row(
                base_row(lora_recipe(), 0),
                &i,
                &f.adapters,
                &f.host,
                &[LoraQuickReason::Smoke]
            )
            .is_err(),
            "{bad:?}"
        );
    }
    // The plan that would be refused cannot be written as a row either.
    assert!(
        lora_ft_row(
            base_row(lora_recipe(), 0),
            &init(&f, a),
            &[],
            &f.host,
            &[LoraQuickReason::Smoke]
        )
        .is_err()
    );
}

#[test]
fn a_full_ft_row_still_says_f32() {
    let r = FtRecipe {
        optimizer_recipe: ledger::METAL_FULL_FT_OPTIMIZER_RECIPE.into(),
        ..lora_recipe()
    };
    let mut row = base_row(r, 0);
    row.quick_reason = "full-FT fixture row".into();
    let line = row.line(&stamp()).unwrap();
    let v: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["recipe"]["optimizer_recipe"], "f32");
    assert!(v["metrics"].get("lora.a_init_digest").is_none());
    assert_eq!(ledger::METAL_FULL_FT_OPTIMIZER_RECIPE, "f32");
    assert_eq!(LORA_OPTIMIZER_RECIPE, "lora");
}

fn ml_python() -> PathBuf {
    PathBuf::from("/Users/bharath/.venvs/ml/bin/python")
}

/// Python's own verifier over a ledger the Rust writer produced: `Ledger.verify_chain` re-reads
/// every row through `LedgerRow.from_json`. Needs the ml venv, so it is `ignore`d by default and
/// run by name; a host without the venv reports it as not run, never as passed.
#[test]
#[ignore = "needs /Users/bharath/.venvs/ml (Python's qd_train.ledger)"]
fn pythons_verifier_accepts_a_lora_row() {
    let f = fixture(0);
    let dir = std::env::temp_dir().join(format!("qd-train-lora-row-{}", std::process::id()));
    let path = dir
        .join("ledger")
        .join("mac-ojas-lora-test-2026-10-17.jsonl");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    let row = good_row(&f, 0);
    ledger::append(
        &path,
        &row,
        ledger::uuid4().unwrap(),
        "2026-10-17T10:00:00+00:00".into(),
    )
    .unwrap();
    ledger::append(
        &path,
        &row,
        ledger::uuid4().unwrap(),
        "2026-10-17T10:00:01+00:00".into(),
    )
    .unwrap();
    let out = Command::new(ml_python())
        .args(["-m", "qd_train.ledger", "verify", "--ledger"])
        .arg(&path)
        .env("PYTHONPATH", common::repo_root().join("python"))
        .output()
        .expect("spawn the ml venv's python");
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "python refused the lora row: {said}");
    assert!(said.contains("chain verifies: 2 row(s)"), "{said}");
    std::fs::remove_dir_all(&dir).unwrap();
}
