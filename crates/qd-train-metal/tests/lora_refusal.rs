//! A LoRA recipe given to qd-train-metal is refused on the host, before the shard door, the
//! snapshot or the GPU, and never becomes an `f32`-labelled row (ft-57fc7484 item b5).
//!
//! The decision this pins (outcome b): at the pinned ojas `4123dfb` the only LoRA provider is
//! `ojas_model::qwen35::TapeStepProvider::open_lora`, which implements ojas-train's copy of the
//! `StepProvider` trait and sits behind a dependency this crate does not have. `Qwen35Provider`
//! here is full fine-tuning, so there is no producer for `qd_train::ledger::lora_ft_row` to wire
//! to. CPU only: nothing here opens a shard set, a snapshot or a GPU.

use serde_json::json;

use qd_train_metal::args::{ArgsError, parse};

fn base() -> Vec<String> {
    [
        "--snapshot",
        "/nonexistent/snap/b1485b2f",
        "--data-root",
        "/nonexistent/out",
        "--shards",
        "/nonexistent/out/shards/train",
        "--manifest",
        "/nonexistent/out/data/pool/train.json",
        "--qd-data",
        "/nonexistent/python/qd_data",
        "--head-init",
        "/nonexistent/span_head_init-seed0.safetensors",
        "--head-init-sha256",
        "8f10b58bb45af911e5e8524157f1898604c3bca7fb97ff66437614cda5d48230",
        "--head-init-content-digest",
        "81bd953e1f27e8e2b703264df663f15f239327c9523d3e85973687009db471d6",
        "--out",
        "/nonexistent/run",
        "--ledger",
        "/nonexistent/ledger/mac-ojas-rung-d-2026-10-01.jsonl",
        "--repo",
        "/nonexistent",
        "--seed",
        "0",
        "--lr",
        "1e-5",
        "--steps",
        "200",
        "--batch-tokens",
        "35403",
        "--span-weight",
        "1.0",
        "--cap-s",
        "21600",
        "--operands",
        "bf16",
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

#[test]
fn a_lora_recipe_is_refused_naming_the_missing_provider() {
    for form in [
        with(&["--optimizer-recipe", "lora"]),
        with(&["--optimizer-recipe=lora"]),
    ] {
        match parse(&form) {
            Err(ArgsError::Refused(m)) => {
                assert!(
                    m.contains("TapeStepProvider")
                        && m.contains("no LoRA provider")
                        && m.contains("lora_ft_row"),
                    "the refusal must name the missing provider and the unwired row writer: {m}"
                );
            }
            other => panic!("a lora recipe must be Refused, got {other:?}"),
        }
    }
}

#[test]
fn only_the_two_known_recipe_labels_parse_and_f32_is_todays_run() {
    let plain = parse(&base()).unwrap();
    let named = parse(&with(&["--optimizer-recipe", "f32"])).unwrap();
    assert_eq!(
        plain, named,
        "naming the only label this binary trains changes nothing"
    );
    for bad in ["master", "LoRA", "lora-2b", "adamw"] {
        assert!(
            matches!(
                parse(&with(&["--optimizer-recipe", bad])),
                Err(ArgsError::Usage(_))
            ),
            "{bad}"
        );
    }
    assert!(matches!(
        parse(&with(&[
            "--optimizer-recipe",
            "f32",
            "--optimizer-recipe",
            "lora"
        ])),
        Err(ArgsError::Usage(_))
    ));
}

#[test]
fn a_resume_across_the_lora_boundary_is_refused_both_ways() {
    let args = parse(&base()).unwrap();
    // A checkpoint root whose recipe.json says it was trained as LoRA, resumed as full FT.
    let mut saved = args.checkpoint_recipe();
    saved["optimizer_recipe"] = json!("lora");
    saved["precision"] = json!("lora");
    let err = args.check_resume_recipe(&saved).unwrap_err();
    assert!(err.contains("optimizer_recipe"), "{err}");
    // The reverse: a full-FT checkpoint resumed under a LoRA label cannot be requested, because
    // the LoRA label is refused at parse; and this binary's own recipe.json never names
    // optimizer_recipe, so an existing checkpoint still resumes.
    assert!(args.checkpoint_recipe().get("optimizer_recipe").is_none());
    assert!(args.check_resume_recipe(&args.checkpoint_recipe()).is_ok());
    assert!(matches!(
        parse(&with(&[
            "--optimizer-recipe",
            "lora",
            "--resume",
            "/nonexistent/ckpt-5"
        ])),
        Err(ArgsError::Refused(_))
    ));
}
