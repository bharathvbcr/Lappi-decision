//! The command line refuses before anything is read or opened. CPU only: nothing here opens a
//! shard set, a snapshot or a GPU (the one test that runs the binary feeds it paths that do not
//! exist, and the refusal it checks comes first).

use std::path::PathBuf;
use std::process::Command;

use qd_train_metal::args::{parse, ArgsError, Operands, LR_SCALE_REFUSAL};

fn base() -> Vec<String> {
    [
        "--snapshot", "/nonexistent/snap/b1485b2f",
        "--data-root", "/nonexistent/out",
        "--shards", "/nonexistent/out/shards/train",
        "--manifest", "/nonexistent/out/data/pool/train.json",
        "--qd-data", "/nonexistent/python/qd_data",
        "--head-init", "/nonexistent/span_head_init-seed0.safetensors",
        "--out", "/nonexistent/run",
        "--ledger", "/nonexistent/ledger/mac-ojas-rung-d-2026-10-01.jsonl",
        "--repo", "/nonexistent",
        "--seed", "0",
        "--lr", "1e-5",
        "--steps", "200",
        "--batch-tokens", "35403",
        "--span-weight", "1.0",
        "--cap-s", "21600",
        "--operands", "bf16",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn with(extra: &[&str]) -> Vec<String> {
    let mut v = base();
    v.extend(extra.iter().map(|s| s.to_string()));
    v
}

fn without(flag: &str) -> Vec<String> {
    let v = base();
    let i = v.iter().position(|a| a == flag).expect("flag in base");
    v.into_iter().enumerate().filter(|(j, _)| *j != i && *j != i + 1).map(|(_, a)| a).collect()
}

fn refused(args: &[String]) -> String {
    match parse(args) {
        Err(ArgsError::Refused(m)) => m,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

fn usage(args: &[String]) -> String {
    match parse(args) {
        Err(ArgsError::Usage(m)) => m,
        other => panic!("expected a usage error, got {other:?}"),
    }
}

#[test]
fn a_complete_command_line_is_the_recipe_it_states() {
    let a = parse(&with(&["--eta-at-step", "10", "--eta-margin-s", "900", "--expect-rev", "abc"])).unwrap();
    assert_eq!(a.snapshot, PathBuf::from("/nonexistent/snap/b1485b2f"));
    assert_eq!(a.seed, 0);
    assert_eq!(a.lr, 1e-5);
    assert_eq!(a.steps, 200);
    assert_eq!(a.batch_tokens, 35_403);
    assert_eq!(a.span_weight, 1.0);
    assert_eq!(a.cap_s, 21_600.0);
    assert_eq!(a.operands, Operands::Bf16);
    assert_eq!(a.expect_rev.as_deref(), Some("abc"));
    let eta = a.eta.unwrap();
    assert_eq!((eta.at_step, eta.margin_s), (10, 900.0));
    assert_eq!(a.checkpoint_every, None);
    // `--flag=value` is the same flag.
    let mut eq = base();
    let i = eq.iter().position(|s| s == "--operands").unwrap();
    eq.remove(i + 1);
    eq[i] = "--operands=exact_f32".into();
    assert_eq!(parse(&eq).unwrap().operands, Operands::ExactF32);
}

/// The brief's refusal: a layer-wise learning rate cannot run on tessl's single-rate AdamW, so
/// it is refused at the command line, by name, whatever form the flag takes.
#[test]
fn a_layer_wise_learning_rate_is_refused_at_the_command_line() {
    for extra in [
        &["--lower-layers-n", "8"][..],
        &["--lower-layers-lr-scale", "0.1"][..],
        &["--lower-layers-n=8", "--lower-layers-lr-scale=0.1"][..],
    ] {
        let m = refused(&with(extra));
        assert!(m.contains("lr_scale") && m.contains(LR_SCALE_REFUSAL), "{extra:?}: {m}");
    }
}

#[test]
fn real_fts_source_transforms_and_the_doors_escape_hatches_are_refused_by_name() {
    for flag in ["--passes", "--max-width", "--option-permutation-seed", "--replay-shards"] {
        let m = refused(&with(&[flag, "2"]));
        assert!(m.starts_with(flag) && m.contains("source transforms are not ported"), "{m}");
    }
    for flag in ["--allow-stale-code", "--allow-rev-mismatch", "--allow-not-run-snapshot"] {
        let m = refused(&with(&[flag]));
        assert!(m.starts_with(flag) && m.contains("escape hatches"), "{m}");
    }
}

#[test]
fn malformed_command_lines_are_usage_errors() {
    for required in [
        "--snapshot", "--data-root", "--shards", "--manifest", "--qd-data", "--head-init", "--out",
        "--ledger", "--repo", "--seed", "--lr", "--steps", "--batch-tokens", "--span-weight", "--cap-s",
        "--operands",
    ] {
        assert!(usage(&without(required)).contains(required), "{required} must be required");
    }
    assert!(usage(&with(&["--seed", "1"])).contains("given twice"));
    assert!(usage(&with(&["--beta2", "0.99"])).contains("unknown argument"));
    assert!(usage(&with(&["--checkpoint-every"])).contains("needs a value"));
    assert!(usage(&with(&["--checkpoint-every", "0"])).contains("at least 1"));
    assert!(usage(&with(&["--eta-at-step", "10"])).contains("come together"));
    // The ETA rule is checked against this run's steps and cap before anything is read.
    assert!(usage(&with(&["--eta-at-step", "200", "--eta-margin-s", "900"])).contains("ETA rule"));
    assert!(usage(&with(&["--eta-at-step", "10", "--eta-margin-s", "21600"])).contains("margin"));
    let bad = |flag: &str, value: &str| {
        let mut v = without(flag);
        v.extend([flag.to_string(), value.to_string()]);
        usage(&v)
    };
    assert!(bad("--lr", "0").contains("positive"));
    assert!(bad("--lr", "nan").contains("positive"));
    assert!(bad("--lr", "fast").contains("not a valid number"));
    assert!(bad("--steps", "0").contains("at least 1"));
    assert!(bad("--batch-tokens", "0").contains("at least 1"));
    assert!(bad("--span-weight", "-1").contains("positive"));
    assert!(bad("--cap-s", "200000").contains("MAX_CAP_S"), "the program's 40 h cap is read-only");
    assert!(bad("--operands", "fp16").contains("exact_f32 or bf16"));
    assert!(bad("--seed", "-1").contains("not a valid number"));
}

#[test]
fn the_row_can_only_go_to_a_mac_ojas_ledger() {
    let mut v = without("--ledger");
    v.extend(["--ledger".to_string(), "/r/ledger/gh200-p4-v4-2026-10-01.jsonl".to_string()]);
    assert!(refused(&v).contains("mac-ojas"));
}

/// The binary itself refuses `--lower-layers-n` before it reads anything: exit code 2 and the
/// lr_scale refusal, though every path it was given is absent. Were the refusal gone, the
/// binary would get past the parse and fail at the shard door instead (exit 1), still before
/// any GPU work.
#[test]
fn the_binary_refuses_a_layer_wise_learning_rate_before_reading_anything() {
    let out = Command::new(env!("CARGO_BIN_EXE_qd-train-metal"))
        .args(with(&["--lower-layers-n", "8", "--lower-layers-lr-scale", "0.1"]))
        .output()
        .expect("run qd-train-metal");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("refused: --lower-layers-n") && stderr.contains("lr_scale"), "{stderr}");
    assert!(out.stdout.is_empty());
}
