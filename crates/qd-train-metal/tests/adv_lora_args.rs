//! Adversarial tests for `--optimizer-recipe` (ft-57fc7484 L5, cc60581). CPU only: parse never
//! opens a shard set, snapshot or GPU.

use qd_train_metal::args::{ArgsError, parse};

fn base() -> Vec<String> {
    [
        "--snapshot", "/nonexistent/snap/b1485b2f",
        "--data-root", "/nonexistent/out",
        "--shards", "/nonexistent/out/shards/train",
        "--manifest", "/nonexistent/out/data/pool/train.json",
        "--qd-data", "/nonexistent/python/qd_data",
        "--head-init", "/nonexistent/span_head_init-seed0.safetensors",
        "--head-init-sha256",
        "8f10b58bb45af911e5e8524157f1898604c3bca7fb97ff66437614cda5d48230",
        "--head-init-content-digest",
        "81bd953e1f27e8e2b703264df663f15f239327c9523d3e85973687009db471d6",
        "--out", "/nonexistent/run",
        "--ledger", "/nonexistent/ledger/mac-ojas-rung-d-2026-10-01.jsonl",
        "--repo", "/nonexistent",
        "--seed", "0", "--lr", "1e-5", "--steps", "200", "--batch-tokens", "35403",
        "--span-weight", "1.0", "--cap-s", "21600", "--operands", "bf16",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

fn with(extra: &[&str]) -> Vec<String> {
    let mut v = base();
    v.extend(extra.iter().map(|s| (*s).to_owned()));
    v
}

fn is_refused_or_usage(r: &Result<qd_train_metal::args::TrainArgs, ArgsError>) -> bool {
    matches!(r, Err(ArgsError::Refused(_)) | Err(ArgsError::Usage(_)))
}

#[test]
fn adv_repeated_flag_never_parses_either_order() {
    for order in [["f32", "lora"], ["lora", "f32"]] {
        let r = parse(&with(&[
            "--optimizer-recipe", order[0], "--optimizer-recipe", order[1],
        ]));
        assert!(is_refused_or_usage(&r), "{order:?}: {r:?}");
    }
}

#[test]
fn adv_empty_case_unicode_and_padded_values_are_not_f32_runs() {
    for v in ["", "LORA", "Lora", "l\u{43e}ra", "lora ", " lora", "lora\n", "F32", "f32 ", "f\u{ff13}2", "lora\0"] {
        let r = parse(&with(&["--optimizer-recipe", v]));
        assert!(r.is_err(), "{v:?} parsed as a run: {r:?}");
        let r = parse(&with(&[&format!("--optimizer-recipe={v}")]));
        assert!(r.is_err(), "={v:?} parsed as a run: {r:?}");
    }
}

#[test]
fn adv_flag_after_unpinned_ojas_and_equals_form() {
    let r = parse(&with(&["--unpinned-ojas", "--optimizer-recipe", "lora"]));
    assert!(matches!(r, Err(ArgsError::Refused(_))), "{r:?}");
    let r = parse(&with(&["--unpinned-ojas", "--optimizer-recipe=lora"]));
    assert!(matches!(r, Err(ArgsError::Refused(_))), "{r:?}");
    let r = parse(&with(&["--optimizer-recipe", "f32", "--unpinned-ojas"]));
    assert!(r.is_ok(), "{r:?}");
}

/// The refusal must come before any other validation that could mask or reorder it: a lora
/// recipe next to an otherwise-invalid argument set is still reported, not silently Usage.
#[test]
fn adv_lora_refusal_wins_over_a_later_usage_error() {
    let mut v = base();
    v.retain(|a| a != "--lr" && a != "1e-5");
    v.extend(["--optimizer-recipe".to_owned(), "lora".to_owned()]);
    let r = parse(&v);
    assert!(
        matches!(r, Err(ArgsError::Refused(_))),
        "a lora request with --lr missing is reported as {r:?}, so the lora refusal is hidden"
    );
}

/// An equals form whose value contains `=` must not be read as the f32 label.
#[test]
fn adv_equals_inside_value() {
    let r = parse(&with(&["--optimizer-recipe=f32=lora"]));
    assert!(r.is_err(), "{r:?}");
}
