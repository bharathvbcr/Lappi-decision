//! The LoRA optimizer plan is `qd_train.lora.lora_param_groups`, group for group: adapters at
//! lr_scale 1.0, D17 and the span head at 0.5, weight decay 0.01 on all, no 0.1x split. The
//! oracle is `tests/fixtures/lora-plan-oracle.json` (`tools/qd_train_oracle_lora_plan.py`). The
//! plan refuses what would train a different model than the pre-registration names.

mod common;

use qd_train::recipe::{
    self, D17_ENTRY_NAME, EntryHyper, HEAD_LR_SCALE, LORA_TARGET_SUFFIXES,
    LORA_UNADAPTED_LINEAR_SUFFIXES, LORA_WEIGHT_DECAY, LowerLayers, OptimizerRecipe,
    lora_optimizer_table, lora_param_groups, optimizer_table,
};
use qd_train::step::ParamSpec;
use serde_json::Value;

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

/// `(adapters, host)` as the loop would hold them: the provider's trainable entries, then D17
/// and the span head as host entries.
fn tables(block: &Value) -> (Vec<ParamSpec>, Vec<ParamSpec>) {
    let adapters = specs(&block["adapter_parameters"]);
    let d17 = &block["d17"];
    let mut host = specs(&Value::Array(vec![serde_json::json!({
        "name": d17["name"], "shape": d17["shape"]
    })]));
    host.extend(specs(&block["span_head"]));
    (adapters, host)
}

fn recipe_lora() -> OptimizerRecipe {
    OptimizerRecipe {
        lr: 2e-4,
        beta2: recipe::DEFAULT_BETA2,
        lower_layers: None,
    }
}

#[test]
fn the_constants_are_the_torch_references_and_the_preregistrations() {
    let o = oracle();
    let c = &o["constants"];
    assert_eq!(HEAD_LR_SCALE, common::fhex(&c["HEAD_LR_SCALE"]));
    assert_eq!(HEAD_LR_SCALE, 0.5);
    assert_eq!(LORA_WEIGHT_DECAY, common::fhex(&c["LORA_WEIGHT_DECAY"]));
    assert_eq!(D17_ENTRY_NAME, "d17.delta");
    let strs = |v: &Value| -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect()
    };
    assert_eq!(
        LORA_TARGET_SUFFIXES.map(String::from).to_vec(),
        strs(&c["LORA_TARGET_SUFFIXES"])
    );
    assert_eq!(
        LORA_UNADAPTED_LINEAR_SUFFIXES.map(String::from).to_vec(),
        strs(&c["LORA_UNADAPTED_LINEAR_SUFFIXES"])
    );
    let path = common::repo_root().join("campaign/lora-2b-preregistered.json");
    let prereg: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let opt = &prereg["recipe"]["optimizer"];
    assert_eq!(opt["weight_decay"].as_f64(), Some(LORA_WEIGHT_DECAY));
    assert_eq!(
        opt["lr"]["d17"].as_f64().unwrap() / opt["lr"]["lora"].as_f64().unwrap(),
        HEAD_LR_SCALE,
        "the pre-registered lr_scale form is 1e-4 / 2e-4"
    );
}

#[test]
fn the_plans_groups_are_lora_param_groups_name_scale_decay_and_members() {
    let o = oracle();
    for block in o["seeds"].as_array().unwrap() {
        let (adapters, host) = tables(block);
        let got = lora_param_groups(&adapters, &host).unwrap();
        let want = block["groups"].as_array().unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.name, w["name"].as_str().unwrap());
            assert_eq!(g.lr_scale.to_bits(), common::fhex(&w["lr_scale"]).to_bits());
            assert_eq!(
                g.weight_decay.to_bits(),
                common::fhex(&w["weight_decay"]).to_bits()
            );
            let names: Vec<&str> = w["params"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_str().unwrap())
                .collect();
            assert_eq!(g.params, names, "group {}", g.name);
        }
    }
}

#[test]
fn the_per_entry_table_matches_the_groups_and_has_no_lower_layer_split() {
    let o = oracle();
    let (adapters, host) = tables(&o["seeds"][0]);
    let table = lora_optimizer_table(&adapters, &host, &recipe_lora()).unwrap();
    assert_eq!(table.len(), adapters.len() + host.len());
    let by_name = |n: &str| -> &EntryHyper { table.iter().find(|e| e.name == n).unwrap() };
    for e in &table {
        assert_eq!(e.weight_decay, 0.01, "{}", e.name);
        assert_eq!((e.eps, e.beta1, e.beta2), (1e-8, 0.9, 0.999), "{}", e.name);
    }
    for a in &adapters {
        assert_eq!(by_name(&a.name).lr_scale, 1.0, "{}", a.name);
    }
    for h in &host {
        assert_eq!(by_name(&h.name).lr_scale, 0.5, "{}", h.name);
    }
    // A layer index in an adapter's name is not a split: layer 0 trains at 1.0 as layer 2 does.
    assert!(
        adapters.iter().any(|a| a.name.contains("layers.0.")),
        "the fixture has layer-0 adapters"
    );
}

fn refused(adapters: &[ParamSpec], host: &[ParamSpec], needle: &str) {
    let err = lora_param_groups(adapters, host).expect_err(needle).to_string();
    assert!(err.contains(needle), "{err:?} lacks {needle:?}");
    let err = lora_optimizer_table(adapters, host, &recipe_lora())
        .expect_err(needle)
        .to_string();
    assert!(err.contains(needle), "{err:?} lacks {needle:?}");
}

#[test]
fn a_lora_run_with_no_adapter_parameters_is_refused() {
    let o = oracle();
    let (_, host) = tables(&o["seeds"][0]);
    refused(&[], &host, "no adapter");
}

#[test]
fn an_adapter_on_a_module_outside_the_target_table_is_refused() {
    let o = oracle();
    let (mut adapters, host) = tables(&o["seeds"][0]);
    // The GDN gates are excluded on purpose (D6): an adapter on one is a different model.
    for leaf in ["in_proj_a", "in_proj_b"] {
        let mut bad = adapters.clone();
        bad.push(ParamSpec::new(
            format!("model.layers.0.linear_attn.{leaf}.lora_A.weight"),
            &[16, 12],
        ));
        bad.push(ParamSpec::new(
            format!("model.layers.0.linear_attn.{leaf}.lora_B.weight"),
            &[2, 16],
        ));
        refused(&bad, &host, leaf);
    }
    adapters.push(ParamSpec::new("model.layers.0.input_layernorm.lora_A.weight", &[16, 12]));
    adapters.push(ParamSpec::new("model.layers.0.input_layernorm.lora_B.weight", &[12, 16]));
    refused(&adapters, &host, "input_layernorm");
}

#[test]
fn a_zero_point_one_split_under_the_lora_recipe_is_refused() {
    let o = oracle();
    let (adapters, host) = tables(&o["seeds"][0]);
    let split = OptimizerRecipe {
        lower_layers: Some(LowerLayers {
            n: 1,
            lr_scale: 0.1,
        }),
        ..recipe_lora()
    };
    let err = lora_optimizer_table(&adapters, &host, &split)
        .unwrap_err()
        .to_string();
    assert!(err.contains("lower-layer split"), "{err}");
    // The same request on the full-FT table is still honoured: that path is unchanged.
    let full = vec![
        ParamSpec::new("layers.0.w", &[1]),
        ParamSpec::new("layers.3.w", &[1]),
    ];
    let t = optimizer_table(&full, &[], &split).unwrap();
    assert_eq!(t[0].lr_scale, 0.1);
    assert_eq!(t[1].lr_scale, 1.0);
}

#[test]
fn a_base_parameter_marked_trainable_is_refused() {
    let o = oracle();
    let (adapters, host) = tables(&o["seeds"][0]);
    for base in [
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
    ] {
        let mut bad = adapters.clone();
        bad.push(ParamSpec::new(base, &[4, 4]));
        refused(&bad, &host, base);
    }
    // A host entry that is neither D17 nor a span-head tensor is a base parameter too.
    let mut bad_host = host.clone();
    bad_host.push(ParamSpec::new("model.norm.weight", &[12]));
    refused(&adapters, &bad_host, "model.norm.weight");
}

#[test]
fn half_an_adapter_and_repeats_are_refused() {
    let o = oracle();
    let (adapters, host) = tables(&o["seeds"][0]);
    let only_a: Vec<ParamSpec> = adapters
        .iter()
        .filter(|a| !a.name.ends_with("model.layers.0.mlp.up_proj.lora_B.weight"))
        .cloned()
        .collect();
    refused(&only_a, &host, "up_proj");
    let mut twice = adapters.clone();
    twice.push(adapters[0].clone());
    refused(&twice, &host, "twice");
    let mut dup_host = host.clone();
    dup_host.push(host[0].clone());
    refused(&adapters, &dup_host, "twice");
}

#[test]
fn the_full_ft_table_is_unchanged() {
    // The f32 path's table: no adapter logic, host at 1.0, split honoured exactly as before.
    let entries = vec![
        ParamSpec::new("embed_tokens.weight", &[2]),
        ParamSpec::new("layers.0.w", &[2]),
        ParamSpec::new("layers.5.w", &[2]),
    ];
    let host = vec![ParamSpec::new("start_proj.weight", &[2, 2])];
    let r = OptimizerRecipe {
        lr: 1e-5,
        beta2: recipe::DEFAULT_BETA2,
        lower_layers: Some(LowerLayers {
            n: 2,
            lr_scale: 0.1,
        }),
    };
    let t = optimizer_table(&entries, &host, &r).unwrap();
    let scales: Vec<f64> = t.iter().map(|e| e.lr_scale).collect();
    assert_eq!(scales, vec![1.0, 0.1, 1.0, 1.0]);
    assert!(t.iter().all(|e| e.weight_decay == 0.01));
}
