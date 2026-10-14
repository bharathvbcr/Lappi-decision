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
use run_v64::{BATCH_TOKENS, args, fixture, scratch};

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
