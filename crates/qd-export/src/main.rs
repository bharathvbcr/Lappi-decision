//! `qd-export` — write a release directory from an averaged checkpoint.
//!
//! ```text
//! qd-export --source AVG.safetensors --base-snapshot SNAP --tokenizer-sha256 HEX \
//!           --expect-vocab-size 248320 --train-manifest TRAIN.json --out RELEASE \
//!           [--calibration TABLE.json] [--source-manifest AVG.safetensors.manifest.json] \
//!           [--allow-extra NAME=REASON]...
//! ```
//!
//! Exit codes: 0 a release was written; 2 the export was refused (nothing is left behind);
//! 1 the filesystem failed.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use qd_export::{export, AllowedExtra, ExportRequest, RefusalKind};

#[derive(Parser, Debug)]
#[command(
    name = "qd-export",
    version,
    about = "Export an averaged tower.*/span_head.* checkpoint to the release directory qd-metal loads"
)]
struct Cli {
    /// The averaged checkpoint written by tools/ckpt_average.py.
    #[arg(long, value_name = "PATH")]
    source: PathBuf,

    /// Its manifest. Defaults to <source>.manifest.json, where ckpt_average writes it.
    #[arg(long, value_name = "PATH")]
    source_manifest: Option<PathBuf>,

    /// The Qwen3.5-2B-Base snapshot: config.json and the four tokenizer files are copied from it.
    #[arg(long, value_name = "DIR")]
    base_snapshot: PathBuf,

    /// SHA-256 the snapshot's tokenizer.json must have (hex). It is the tokenizer_hash the
    /// serving backend reports.
    #[arg(long, value_name = "HEX")]
    tokenizer_sha256: String,

    /// The vocabulary the release must have: config.json, the embedding and the source manifest
    /// must all say it. 248320 for the Qwen3.5-2B checkpoint.
    #[arg(long, value_name = "ROWS")]
    expect_vocab_size: usize,

    /// A calibration table (qd_runtime CalibrationTable JSON) to pass through, hash-recorded.
    #[arg(long, value_name = "PATH")]
    calibration: Option<PathBuf>,

    /// The train split's data manifest the tower was trained from (python/qd_data/manifest.py,
    /// split "train"; data/pool/train.json for v4). Its family set is recorded as
    /// trained_families, and the runtime refuses every other task. Required: a release that
    /// cannot say what it trained admits nothing.
    #[arg(long, value_name = "PATH")]
    train_manifest: PathBuf,

    /// Drop a source tensor the release would not carry, with the reason, e.g.
    /// `tower.rotary_emb.inv_freq=recomputed from config`. Repeatable. Every other extra tensor
    /// is refused.
    #[arg(long, value_name = "NAME=REASON", value_parser = AllowedExtra::parse)]
    allow_extra: Vec<AllowedExtra>,

    /// The release directory to create. It must not exist.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let req = ExportRequest {
        source: cli.source,
        source_manifest: cli.source_manifest,
        base_snapshot: cli.base_snapshot,
        tokenizer_sha256: cli.tokenizer_sha256,
        expect_vocab_size: cli.expect_vocab_size,
        calibration: cli.calibration,
        train_manifest: cli.train_manifest,
        allow_extra: cli.allow_extra,
        out: cli.out,
    };
    match export(&req) {
        Ok(s) => {
            println!("release: {}", s.out.display());
            println!(
                "tower: {} tensors rounded to bf16 (round to nearest, ties to even), {} copied",
                s.rounded, s.copied
            );
            for d in &s.dropped {
                println!("dropped: {} ({})", d.name, d.reason);
            }
            println!("weight_hash: {}", s.weight_hash);
            println!("tokenizer_hash: {}", s.tokenizer_hash);
            match &s.calibration_hash {
                Some(h) => println!("calibration_hash: {h}"),
                None => println!("calibration_hash: none (no table passed)"),
            }
            println!("trained_families: {}", s.trained_families.join(", "));
            for (name, sha) in &s.files {
                println!("{sha}  {name}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("qd-export: refused: {e}");
            if e.kind == RefusalKind::Io {
                ExitCode::from(1)
            } else {
                ExitCode::from(2)
            }
        }
    }
}
