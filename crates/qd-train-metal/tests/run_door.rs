//! `run()` up to the GPU, on the tiny tower's vocabulary. CPU only: every check here is one
//! `run()` makes (or refuses at) before it opens a Metal runtime -- the shard door with no escape
//! hatch, the identity remap over the snapshot's 64 ids, the head init's pins, and `--resume`'s
//! recipe check (`run.rs`, the `resume` match). The fixture is `shards-tiny-v64`; see
//! `tests/run_v64/mod.rs`. What `run()` does after the GPU opens is `tests/gpu_tiny.rs`.

#![cfg(target_os = "macos")]

mod run_v64;

use std::sync::Arc;

use qd_train::files::OsFiles;
use qd_train::held_out::DataConfig;
use qd_train::shards::{ShardOpen, ShardReader};
use qd_train_metal::remap::{REMAP_CHECK, require_identity_remap};
use qd_train_metal::run::{RECIPE_FILE, RunError, run};
use run_v64::{BATCH_TOKENS, args, fixture, repo, scratch};

/// The fixture through the door as `run()` opens it: no `allow_stale_code`.
fn reader() -> ShardReader {
    let a = args(&scratch("reader"), 1);
    let mut opts = ShardOpen::new(
        DataConfig::default(),
        a.data_root.clone(),
        a.qd_data.clone(),
    );
    opts.expect_rev = a.expect_rev.clone();
    opts.files = Arc::new(OsFiles);
    ShardReader::open(&a.shards, &a.manifest, &opts)
        .expect("the v64 fixture opens through the door, qd_data frozen beside it")
}

#[test]
fn the_v64_fixture_passes_the_door_and_is_the_identity_over_the_snapshots_64_ids() {
    let reader = reader();
    assert_eq!(reader.header().split, "train");
    assert!(
        reader.check(REMAP_CHECK).unwrap().is_pass(),
        "the door verified the remap first"
    );
    assert_eq!(require_identity_remap(&reader, &OsFiles, 64), Ok(64));
    let plans = reader.plan(BATCH_TOKENS, 0, 0).unwrap();
    println!(
        "the v64 epoch has {} batches at batch_tokens {BATCH_TOKENS}",
        plans.len()
    );
    assert!(
        plans.len() >= 8,
        "the epoch has {} batches at batch_tokens {BATCH_TOKENS}",
        plans.len()
    );
    assert!(
        fixture().join("qd_data").join("render.py").is_file(),
        "the frozen qd_data sits beside the set"
    );
}

/// A current-rule shard set (`code_fingerprint_version` 3, fingerprint over the live `python/qd_data`'s
/// `SHARD_CONTENT_MODULES`) opens through the same door `run()` uses, which is `qd-train`'s
/// reader: no `allow_stale_code`, and the live package carries unlisted modules the v64
/// fixture's frozen copy never had. Before `qd-train` read the version it hashed every `*.py` and
/// refused this set as "added since", so no v6 set could reach a LoRA lane.
#[test]
fn a_current_rule_shard_set_opens_through_the_door_run_uses() {
    let reader = door_reader_under_rule("v3-door", 3);
    assert!(reader.check("shard_code_current").unwrap().is_pass());
    assert_eq!(reader.header().code_fingerprint_version, 3);
}

/// The fingerprint metric a Rust ft row carries (`run::shard_coverage_metrics`, the one place
/// `run()` takes it from): a v3 set records nothing uncovered, a v2 set records the two modules
/// rule 3 added, so `Ledger.promotion_verdict` can refuse its row. Without this a row written on
/// a v2 set carried no metric and read as covered.
#[test]
fn a_rust_ft_row_on_a_v2_set_is_marked_not_covered_and_a_v3_set_is_not() {
    use qd_train::tristate::TriState;
    use qd_train_metal::run::{SHARD_NOT_COVERED_METRIC, shard_coverage_metrics};
    let value_of = |reader: &ShardReader| match shard_coverage_metrics(reader).as_slice() {
        [(name, TriState::Ran { value, passed: true, .. })] => {
            assert_eq!(name, SHARD_NOT_COVERED_METRIC);
            assert_eq!(SHARD_NOT_COVERED_METRIC, "shard.code_not_covered");
            value.clone()
        }
        other => panic!("expected one passing Ran metric, got {other:?}"),
    };
    assert_eq!(
        value_of(&door_reader_under_rule("v2-row", 2)),
        serde_json::json!("manifest.py,val_requests.py")
    );
    assert_eq!(
        value_of(&door_reader_under_rule("v3-row", 3)),
        serde_json::json!("")
    );
}

/// A door whose `qd_data` cannot say which modules a header covers (the v64 fixture's frozen copy
/// states no module list) records a `not_run` metric, never nothing and never a clean pass.
#[test]
fn a_rust_ft_row_whose_qd_data_cannot_decide_coverage_records_not_run() {
    use qd_train::tristate::TriState;
    use qd_train_metal::run::shard_coverage_metrics;
    match shard_coverage_metrics(&reader()).as_slice() {
        [(_, TriState::NotRun { reason })] => {
            assert!(reason.contains("could not be decided"), "{reason}");
        }
        other => panic!("expected one not_run metric, got {other:?}"),
    }
}

/// The v64 fixture's train set re-stamped to fingerprint rule `rule` over the live
/// `python/qd_data`, opened the way `run()` opens it (no `allow_stale_code`).
fn door_reader_under_rule(tag: &str, rule: u64) -> ShardReader {
    use qd_train::shards::{ShardHeader, shard_code_fingerprint};
    let a = args(&scratch(tag), 1);
    let live_qd_data = repo().join("python").join("qd_data");
    let shards = scratch(&format!("{tag}-set")).join("train");
    std::fs::create_dir_all(&shards).unwrap();
    for entry in std::fs::read_dir(&a.shards).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), shards.join(entry.file_name())).unwrap();
    }
    let path = shards.join("header.json");
    let mut raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut header = ShardHeader::from_json(&raw).expect("the v64 fixture header");
    let fingerprint =
        shard_code_fingerprint(&live_qd_data, rule).expect("the live qd_data fingerprint");
    header.code_fingerprint = fingerprint.clone();
    header.code_fingerprint_version = rule;
    raw["code_fingerprint"] = serde_json::to_value(&fingerprint).unwrap();
    raw["code_fingerprint_version"] = serde_json::json!(rule);
    raw["shard_hash"] = serde_json::json!(header.shard_hash().unwrap());
    std::fs::write(&path, serde_json::to_vec_pretty(&raw).unwrap()).unwrap();

    let mut opts = ShardOpen::new(DataConfig::default(), a.data_root.clone(), live_qd_data);
    opts.expect_rev = a.expect_rev.clone();
    opts.files = Arc::new(OsFiles);
    ShardReader::open(&shards, &a.manifest, &opts)
        .unwrap_or_else(|e| panic!("a rule {rule} set is refused at the door: {e}"))
}

/// A checkpoint root as `run()` leaves one: `<root>/recipe.json` beside `<root>/ckpt-<step>/`.
fn checkpoint_root(
    tag: &str,
    recipe: &serde_json::Value,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let root = scratch(tag);
    let ckpts = root.join("checkpoints");
    let ckpt = ckpts.join("ckpt-00000002");
    std::fs::create_dir_all(&ckpt).unwrap();
    std::fs::write(
        ckpts.join(RECIPE_FILE),
        serde_json::to_vec_pretty(recipe).unwrap(),
    )
    .unwrap();
    (root, ckpt)
}

fn refusal(r: Result<qd_train_metal::run::RunSummary, RunError>) -> String {
    match r {
        Err(RunError::Refused(m)) => m,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_resume_under_a_changed_recipe_is_refused_through_run_before_the_gpu() {
    let mut earlier = args(&scratch("earlier"), 4);
    earlier.lr = 5e-4;
    let (_, ckpt) = checkpoint_root("changed", &earlier.checkpoint_recipe());
    let mut now = args(&scratch("now"), 4);
    // The ojas pin is checked first and depends on where the sibling checkout sits, which is the
    // host's business (tests/pin.rs and make pin own it); this test is about the resume recipe.
    now.unpinned_ojas = true;
    now.resume = Some(ckpt.clone());
    let m = refusal(run(&now));
    assert!(
        m.contains("--resume") && m.contains("another recipe") && m.contains("\"lr\""),
        "{m}"
    );
    assert!(!now.ledger.exists(), "a refused run writes no row");
}

#[test]
fn a_resume_whose_recipe_matches_gets_past_the_recipe_check_and_stops_at_the_checkpoint() {
    let now = args(&scratch("same"), 4);
    let (_, ckpt) = checkpoint_root("matching", &now.checkpoint_recipe());
    let mut now = now;
    // The ojas pin is checked first and depends on where the sibling checkout sits, which is the
    // host's business (tests/pin.rs and make pin own it); this test is about the resume recipe.
    now.unpinned_ojas = true;
    now.resume = Some(ckpt);
    let m = refusal(run(&now));
    assert!(
        m.contains("--resume") && !m.contains("another recipe"),
        "{m}"
    );
    assert!(!now.ledger.exists(), "a refused run writes no row");
}

#[test]
fn a_resume_from_a_checkpoint_root_without_its_recipe_is_refused() {
    let root = scratch("no-recipe");
    let ckpt = root.join("checkpoints").join("ckpt-00000002");
    std::fs::create_dir_all(&ckpt).unwrap();
    let mut now = args(&scratch("no-recipe-now"), 4);
    // The ojas pin is checked first and depends on where the sibling checkout sits, which is the
    // host's business (tests/pin.rs and make pin own it); this test is about the resume recipe.
    now.unpinned_ojas = true;
    now.resume = Some(ckpt);
    let m = refusal(run(&now));
    assert!(m.contains("another recipe") && m.contains("missing"), "{m}");
}

#[test]
fn a_resume_of_another_epoch_count_is_refused_through_run_before_the_gpu() {
    // The checkpoints were written by a one-epoch run; this command line says two epochs. The
    // ojas pin is the host's business, not this test's, so it runs unpinned.
    let earlier = args(&scratch("epochs-earlier"), 4);
    let (_, ckpt) = checkpoint_root("epochs-one", &earlier.checkpoint_recipe());
    let mut now = args(&scratch("epochs-now"), 4);
    now.epochs = 2;
    now.unpinned_ojas = true;
    now.resume = Some(ckpt);
    let m = refusal(run(&now));
    assert!(
        m.contains("--resume") && m.contains("another recipe") && m.contains("\"epochs\""),
        "{m}"
    );
    assert!(!now.ledger.exists(), "a refused run writes no row");
}
