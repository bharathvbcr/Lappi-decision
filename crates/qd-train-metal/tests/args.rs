//! The command line refuses before anything is read or opened. CPU only: nothing here opens a
//! shard set, a snapshot or a GPU (the one test that runs the binary feeds it paths that do not
//! exist, and the refusal it checks comes first).

use std::path::PathBuf;
use std::process::Command;

use qd_train::recipe::LowerLayers;
use qd_train::shards::plan_epoch;
use qd_train_metal::args::{parse, ArgsError, Operands};

fn base() -> Vec<String> {
    [
        "--snapshot", "/nonexistent/snap/b1485b2f",
        "--data-root", "/nonexistent/out",
        "--shards", "/nonexistent/out/shards/train",
        "--manifest", "/nonexistent/out/data/pool/train.json",
        "--qd-data", "/nonexistent/python/qd_data",
        "--head-init", "/nonexistent/span_head_init-seed0.safetensors",
        "--head-init-sha256", "8f10b58bb45af911e5e8524157f1898604c3bca7fb97ff66437614cda5d48230",
        "--head-init-content-digest", "81bd953e1f27e8e2b703264df663f15f239327c9523d3e85973687009db471d6",
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

/// Fable's board-sweep ruling 2h retires the lead's ruling 6: ojas-qwen35's plan carries a
/// per-entry learning-rate scale, so F's layer-wise split is accepted, as a pair, validated
/// before anything is read. The one-flag and the degenerate forms stay usage errors.
#[test]
fn a_layer_wise_learning_rate_is_accepted_as_a_validated_pair() {
    let a = parse(&with(&["--lower-layers-n", "8", "--lower-layers-lr-scale", "0.1"])).unwrap();
    assert_eq!(a.lower_layers, Some(LowerLayers { n: 8, lr_scale: 0.1 }));
    let eq = parse(&with(&["--lower-layers-n=8", "--lower-layers-lr-scale=0.1"])).unwrap();
    assert_eq!(eq.lower_layers, a.lower_layers);
    assert_eq!(parse(&base()).unwrap().lower_layers, None);
    for alone in [&["--lower-layers-n", "8"][..], &["--lower-layers-lr-scale", "0.1"][..]] {
        assert!(usage(&with(alone)).contains("come together"), "{alone:?}");
    }
    for (n, scale, why) in [
        ("0", "0.1", "at least 1"),
        ("8", "0", "finite and positive"),
        ("8", "-0.1", "finite and positive"),
        ("8", "nan", "finite and positive"),
        ("8", "inf", "finite and positive"),
    ] {
        let m = usage(&with(&["--lower-layers-n", n, "--lower-layers-lr-scale", scale]));
        assert!(m.contains(why), "{n} {scale}: {m}");
    }
}

/// `--seed` is the data order: two seeds plan two orders, one seed plans one, and the plan is
/// not the `DataConfig().seed` order every seed used before.
#[test]
fn the_seed_flag_is_the_seed_the_epoch_is_planned_at() {
    let lengths: Vec<u64> = (0..64).map(|i| 128 + (i % 4) * 128).collect();
    let buckets = [128u64, 256, 384, 512];
    let order = |seed: &str| {
        let mut v = without("--seed");
        v.extend(["--seed".to_string(), seed.to_string()]);
        let a = parse(&v).unwrap();
        assert_eq!(a.data_seed(), a.seed);
        plan_epoch(&lengths, &buckets, 1024, a.data_seed(), 0).unwrap()
    };
    let (zero, one) = (order("0"), order("1"));
    assert_ne!(zero, one, "two seeds, one batch order");
    assert_eq!(order("1"), one, "one seed, two batch orders");
    let default_seed = qd_train::held_out::DataConfig::default().seed();
    assert_ne!(
        plan_epoch(&lengths, &buckets, 1024, default_seed, 0).unwrap(),
        zero,
        "seed 0 planned at the DataConfig seed"
    );
}

#[test]
fn min_lr_and_resume_are_flags_with_bounds() {
    let a = parse(&base()).unwrap();
    assert_eq!((a.min_lr, a.resume.clone()), (None, None));
    let a = parse(&with(&["--min-lr", "0", "--resume", "/nonexistent/run/checkpoints/ckpt-00000100"])).unwrap();
    assert_eq!(a.min_lr, Some(0.0));
    assert_eq!(a.resume, Some(PathBuf::from("/nonexistent/run/checkpoints/ckpt-00000100")));
    assert_eq!(parse(&with(&["--min-lr", "1e-5"])).unwrap().min_lr, Some(1e-5), "the floor may equal the peak");
    for bad in ["-1", "nan", "inf", "2e-5", "low"] {
        assert!(!usage(&with(&["--min-lr", bad])).is_empty(), "--min-lr {bad}");
    }
    assert!(usage(&with(&["--min-lr", "0", "--min-lr", "0"])).contains("given twice"));
    assert!(usage(&with(&["--resume"])).contains("needs a value"));
}

#[test]
fn the_recipe_records_the_batch_order_and_the_floor_it_was_given() {
    let none = parse(&base()).unwrap().recipe_extra(1);
    assert_eq!(none.get("batch_order"), Some(&serde_json::json!("seed")));
    assert!(!none.contains_key("min_lr"), "no floor given, no key: the lr / 10 recipe's hash is unchanged by it");
    let floor = parse(&with(&["--min-lr", "0"])).unwrap().recipe_extra(2);
    assert_eq!(floor.get("min_lr"), Some(&serde_json::json!(0.0)));
    assert_eq!(floor.get("prompt_format"), Some(&serde_json::json!(2)));
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
        "--snapshot", "--data-root", "--shards", "--manifest", "--qd-data", "--head-init",
        "--head-init-sha256", "--head-init-content-digest", "--out",
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
    assert!(bad("--head-init-sha256", "abc").contains("64 lowercase hex"));
    assert!(bad("--head-init-content-digest", &"A".repeat(64)).contains("64 lowercase hex"));
}

#[test]
fn the_row_can_only_go_to_a_mac_ojas_ledger() {
    let mut v = without("--ledger");
    v.extend(["--ledger".to_string(), "/r/ledger/gh200-p4-v4-2026-10-01.jsonl".to_string()]);
    assert!(refused(&v).contains("mac-ojas"));
}

/// The binary refuses a half-given split before it reads anything: exit code 2, though every
/// path it was given is absent.
#[test]
fn the_binary_refuses_half_a_layer_wise_split_before_reading_anything() {
    let out = Command::new(env!("CARGO_BIN_EXE_qd-train-metal"))
        .args(with(&["--lower-layers-n", "8"]))
        .output()
        .expect("run qd-train-metal");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("come together"), "{stderr}");
    assert!(out.stdout.is_empty());
}

/// D3 (verify-lane3): a checkpoint stores no optimizer recipe, so a resume under another one
/// would be accepted and continue under a different optimizer. The run writes the recipe beside
/// its checkpoints and a resume refuses any difference in it.
#[test]
fn a_resume_under_another_recipe_is_refused() {
    let base_args = parse(&base()).unwrap();
    let saved = base_args.checkpoint_recipe();
    assert!(base_args.check_resume_recipe(&saved).is_ok(), "the same recipe resumes");
    for (what, extra) in [
        ("lower layers", &["--lower-layers-n", "8", "--lower-layers-lr-scale", "0.1"][..]),
        ("min-lr", &["--min-lr", "0"][..]),
        ("operands", &[][..]),
    ] {
        let mut v = base();
        if what == "operands" {
            let i = v.iter().position(|s| s == "--operands").unwrap();
            v[i + 1] = "exact_f32".into();
        } else {
            v.extend(extra.iter().map(|s| s.to_string()));
        }
        let other = parse(&v).unwrap();
        let err = other.check_resume_recipe(&saved).unwrap_err();
        assert!(err.contains("recipe"), "{what}: {err}");
        assert!(base_args.check_resume_recipe(&other.checkpoint_recipe()).is_err(), "{what}, reversed");
    }
    for (flag, value) in [("--seed", "9"), ("--lr", "2e-5"), ("--span-weight", "2"), ("--batch-tokens", "9000")] {
        let mut v = without(flag);
        v.extend([flag.to_string(), value.to_string()]);
        assert!(parse(&v).unwrap().check_resume_recipe(&saved).is_err(), "{flag}");
    }
    // A checkpoint root with no recipe file at all is not a checkpoint this binary wrote.
    assert!(base_args.check_resume_recipe(&serde_json::Value::Null).is_err());
}

/// The `--resume` command line itself, as `run()` checks it: `--resume` and `--checkpoint-every`
/// are not part of the recipe (a resumed command line is the original's recipe), and a changed
/// flag is refused naming the key that differs.
#[test]
fn the_resume_command_line_carries_the_original_recipe_and_names_what_changed() {
    let original = parse(&with(&["--checkpoint-every", "1"])).unwrap();
    let saved = original.checkpoint_recipe();
    let resumed = parse(&with(&["--checkpoint-every", "2", "--resume", "/nonexistent/run/checkpoints/ckpt-00000002"])).unwrap();
    assert_eq!(resumed.checkpoint_recipe(), saved);
    assert!(resumed.check_resume_recipe(&saved).is_ok());
    let mut v = without("--steps");
    v.extend(["--steps".to_string(), "199".to_string(), "--resume".to_string(), "/nonexistent/ckpt-00000002".to_string()]);
    let err = parse(&v).unwrap().check_resume_recipe(&saved).unwrap_err();
    assert!(err.contains("\"steps\"") && !err.contains("\"lr\""), "{err}");
}
