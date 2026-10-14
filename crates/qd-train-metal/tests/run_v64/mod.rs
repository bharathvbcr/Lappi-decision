//! What a `run()` of `qd-train-metal` needs at the tiny tower's vocabulary, shared by
//! `tests/run_door.rs` (CPU: every refusal that comes before the GPU) and `tests/gpu_tiny.rs`
//! (`run()` end to end on Metal).
//!
//! * the snapshot: tessl's `tests/fixtures/qwen35_train` (2 layers, hidden 64, vocabulary 64);
//! * the shard set: `crates/qd-train/tests/fixtures/shards-tiny-v64`, written by
//!   `tools/qd_train_oracle_shards.py --vocab 64`, whose remap is the identity over 64 ids and
//!   which carries a frozen copy of `qd_data` (`run()` has no `allow_stale_code`, and the live
//!   package drifts from the header's fingerprint);
//! * the span head init at hidden 64, with the digests its tracked manifest records.
//!
//! The door is opened exactly as the binary opens it: no escape hatch, the fixture's own root
//! anchoring the held-out roots, the revision it was written at.

#![allow(dead_code)]
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use qd_train::ft_data::HeadInitPin;
use qd_train_metal::args::{Operands, TrainArgs};
use qd_train_metal::provider::TESSL_DIR;

/// `crates/qd-train-metal/tests/../..`: the repository root.
pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn fixture() -> PathBuf {
    repo().join("crates/qd-train/tests/fixtures/shards-tiny-v64")
}

/// tessl's tiny snapshot, which `run()` cannot read as it stands: its `config.json` omits
/// `attn_output_gate`, and qd-export's `Layout` requires the key to be `true` (the loader's
/// `q_proj` carries the output gate; the fixture's `q_proj` is 1024 x 64 = 2 heads x 256 x 2, so
/// the weights do have it). This copies the snapshot's two files into `<root>/qwen35_train/` and
/// adds that one key. tessl's tree is read, never written.
pub fn snapshot(root: &Path) -> PathBuf {
    let from = Path::new(TESSL_DIR).join("tests/fixtures/qwen35_train");
    let to = root.join("qwen35_train");
    std::fs::create_dir_all(&to).unwrap();
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(from.join("config.json")).unwrap()).unwrap();
    assert!(
        config.get("attn_output_gate").is_none(),
        "tessl's fixture now carries the key; drop this patch"
    );
    config["attn_output_gate"] = serde_json::Value::Bool(true);
    std::fs::write(
        to.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    std::fs::copy(from.join("model.safetensors"), to.join("model.safetensors")).unwrap();
    to
}

pub fn head_init() -> PathBuf {
    repo().join("crates/qd-train/tests/fixtures/span-head-init-seed0-h64.safetensors")
}

/// The pins in `span-head-init-seed0-h64.manifest.json` (`file.sha256`, `content_digest.value`);
/// `qd-train`'s `fixture_pins` test holds the file to the first.
pub fn head_pin() -> HeadInitPin {
    HeadInitPin {
        sha256: "3c1cdbb7e181386605fa4756f676351a5682a59ddc4fc4dcbfbbffb5689555b0".to_owned(),
        content_digest: "81bd953e1f27e8e2b703264df663f15f239327c9523d3e85973687009db471d6"
            .to_owned(),
    }
}

/// The widest bucket of the v64 train set (`shards/train/header.json`): one row of it fits a
/// batch, and the rest pack by length.
pub const BATCH_TOKENS: u64 = 1618;

/// A fresh scratch directory under the workspace's `target`, not the system temp dir: on macOS
/// that is under `/var`, a symbolic link, and ojas-io's state writer refuses symbolic links on
/// the way to the state directory.
pub fn scratch(tag: &str) -> PathBuf {
    let dir = repo()
        .join("target")
        .join(format!("qd-run-v64-{tag}-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A run over the v64 fixture writing `<root>/out` and `<root>/ledger/mac-ojas-w32.jsonl`, with
/// a checkpoint every step. `out` must not exist yet: `run()` refuses a non-empty one.
pub fn args(root: &Path, steps: u64) -> TrainArgs {
    let fixture = fixture();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("oracle/meta.json")).unwrap())
            .unwrap();
    TrainArgs {
        snapshot: snapshot(root),
        data_root: fixture.clone(),
        shards: fixture.join("shards/train"),
        manifest: fixture.join("data/pool/train.json"),
        qd_data: fixture.join("qd_data"),
        expect_rev: Some(meta["corpus_rev"].as_str().unwrap().to_owned()),
        head_init: head_init(),
        head_init_pin: head_pin(),
        out: root.join("out"),
        ledger: root.join("ledger").join("mac-ojas-w32.jsonl"),
        repo: repo(),
        seed: 0,
        lr: 1e-3,
        steps,
        batch_tokens: BATCH_TOKENS,
        span_weight: 1.0,
        cap_s: 3600.0,
        eta: None,
        operands: Operands::ExactF32,
        checkpoint_every: Some(1),
        min_lr: None,
        resume: None,
        lower_layers: None,
    }
}

/// The ledger's lines, parsed; empty when the file does not exist.
pub fn rows(ledger: &Path) -> Vec<serde_json::Value> {
    match std::fs::read_to_string(ledger) {
        Ok(text) => text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => panic!("{}: {e}", ledger.display()),
    }
}
