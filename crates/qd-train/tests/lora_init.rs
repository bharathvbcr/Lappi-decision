//! The LoRA-A init rule reproduces `attach_lora` without torch, and a digest that is not what
//! the rule gives is refused. The oracle is `tests/fixtures/lora-plan-oracle.json`, written by
//! `tools/qd_train_oracle_lora_plan.py` running `qd_train.lora.attach_lora` itself.

mod common;

use qd_train::lora_init::{
    LoraInitError, d17_zero_init_digest, lora_a_digest, recompute_lora_a, verify_lora_a_digest,
};
use serde_json::Value;

fn oracle() -> Value {
    let path = common::crate_dir().join("tests/fixtures/lora-plan-oracle.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn rank(o: &Value) -> usize {
    o["constants"]["LORA_RANK"].as_u64().unwrap() as usize
}

fn modules(block: &Value) -> Vec<(String, usize)> {
    block["modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["module"].as_str().unwrap().to_owned(),
                m["in_features"].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

#[test]
fn every_a_matrix_is_the_value_attach_lora_drew_bit_for_bit() {
    let o = oracle();
    for block in o["seeds"].as_array().unwrap() {
        let seed = block["seed"].as_u64().unwrap();
        let drawn = recompute_lora_a(seed, rank(&o), &modules(block)).unwrap();
        for (got, want) in drawn.iter().zip(block["modules"].as_array().unwrap()) {
            let (first, last) = (&want["first_values"], &want["last_values"]);
            for (i, v) in got.1.iter().take(6).enumerate() {
                assert_eq!(
                    v.to_bits(),
                    (common::fhex(&first[i]) as f32).to_bits(),
                    "seed {seed} {} value {i}",
                    got.0
                );
            }
            let n = got.1.len();
            for (i, v) in got.1[n - 6..].iter().enumerate() {
                assert_eq!(
                    v.to_bits(),
                    (common::fhex(&last[i]) as f32).to_bits(),
                    "seed {seed} {} tail {i}",
                    got.0
                );
            }
        }
    }
}

#[test]
fn the_a_digest_is_the_one_python_computes_at_that_seed() {
    let o = oracle();
    for block in o["seeds"].as_array().unwrap() {
        let seed = block["seed"].as_u64().unwrap();
        let want = block["a_digest"].as_str().unwrap();
        let got = verify_lora_a_digest(seed, rank(&o), &modules(block), want).unwrap();
        assert_eq!(got, want);
    }
    let digests: std::collections::BTreeSet<_> = o["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["a_digest"].as_str().unwrap())
        .collect();
    assert_eq!(digests.len(), 3, "three seeds, three different A inits");
}

#[test]
fn a_digest_that_is_not_the_rules_is_refused() {
    let o = oracle();
    let (zero, other) = (&o["seeds"][0], &o["seeds"][1]);
    let mods = modules(zero);
    // Another seed's digest: a checkpoint's A copied from the wrong run.
    let err =
        verify_lora_a_digest(0, rank(&o), &mods, other["a_digest"].as_str().unwrap()).unwrap_err();
    assert!(
        matches!(err, LoraInitError::Mismatch { seed: 0, .. }),
        "{err}"
    );
    // The right digest, the wrong seed.
    assert!(matches!(
        verify_lora_a_digest(1, rank(&o), &mods, zero["a_digest"].as_str().unwrap()),
        Err(LoraInitError::Mismatch { .. })
    ));
    // The right seed, a module visited in another order.
    let mut swapped = mods.clone();
    swapped.swap(0, 1);
    assert!(matches!(
        verify_lora_a_digest(0, rank(&o), &swapped, zero["a_digest"].as_str().unwrap()),
        Err(LoraInitError::Mismatch { .. })
    ));
    // The right seed and order, one tensor perturbed by one ulp.
    let mut drawn = recompute_lora_a(0, rank(&o), &mods).unwrap();
    drawn[3].1[5] = f32::from_bits(drawn[3].1[5].to_bits() ^ 1);
    assert_ne!(
        lora_a_digest(rank(&o), &drawn).unwrap(),
        zero["a_digest"].as_str().unwrap()
    );
}

#[test]
fn degenerate_inputs_are_refused_not_drawn() {
    assert_eq!(
        recompute_lora_a(0, 16, &[]).unwrap_err(),
        LoraInitError::NoModules
    );
    assert!(matches!(
        recompute_lora_a(u64::from(u32::MAX) + 1, 16, &[("m".into(), 4)]),
        Err(LoraInitError::SeedTooLarge(_))
    ));
    assert!(matches!(
        recompute_lora_a(0, 16, &[("m".into(), 0)]),
        Err(LoraInitError::BadShape { .. })
    ));
    assert!(matches!(
        recompute_lora_a(0, 0, &[("m".into(), 4)]),
        Err(LoraInitError::BadShape { .. })
    ));
    assert!(matches!(
        recompute_lora_a(0, 16, &[("m".into(), 4), ("m".into(), 4)]),
        Err(LoraInitError::DuplicateModule(_))
    ));
}

#[test]
fn the_d17_init_digest_is_the_preregistered_zero_tensor_digest() {
    let path = common::repo_root().join("campaign/lora-2b-preregistered.json");
    let prereg: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let d17 = &prereg["init"]["d17"];
    assert_eq!(
        d17_zero_init_digest(17, 2048).unwrap(),
        d17["tensor_ref_digest_h2048"].as_str().unwrap()
    );
    assert_eq!(
        d17_zero_init_digest(17, 1024).unwrap(),
        d17["tensor_ref_digest_h1024"].as_str().unwrap()
    );
}
