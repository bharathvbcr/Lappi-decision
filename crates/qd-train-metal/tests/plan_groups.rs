//! The adapter's optimizer plan is the loop's per-entry table, entry by entry, with no GPU:
//! `build_plan` over the tiny fixture's parameter table (header and config only). The layer-wise
//! split reaches ojas-qwen35's `OptimizerPlan` as the caller's own `lr_scale` vector; it is no
//! longer refused (board-sweep ruling 2h).

#![cfg(target_os = "macos")]

use std::path::Path;

use ojas_qwen35::{tower_tensors, Snapshot};
use qd_train::recipe::{layer_index, lr_scales, weight_decays, LowerLayers};
use qd_train::step::{ParamSpec, StepError};
use qd_train_metal::provider::{build_plan, TESSL_DIR};

fn table() -> Vec<ojas_qwen35::TensorSpec> {
    let snap = Snapshot::from_dir(&Path::new(TESSL_DIR).join("tests/fixtures/qwen35_train")).unwrap();
    tower_tensors(&snap.config)
}

fn specs(table: &[ojas_qwen35::TensorSpec]) -> Vec<ParamSpec> {
    table.iter().map(|t| ParamSpec::new(t.name.clone(), &t.shape)).collect()
}

#[test]
fn the_layer_wise_split_is_the_loops_lr_scale_vector_bit_for_bit() {
    let table = table();
    let entries = specs(&table);
    let wd = weight_decays(&entries);
    let scales = lr_scales(&entries, Some(LowerLayers { n: 1, lr_scale: 0.1 }), 1).unwrap();
    assert!(scales.contains(&0.1) && scales.contains(&1.0));
    let plan = build_plan(&table, &scales, &wd).unwrap();
    assert!(plan.lr_scale().iter().map(|s| s.to_bits()).eq(scales.iter().map(|s| s.to_bits())));
    assert!(plan.weight_decay().iter().map(|w| w.to_bits()).eq(wd.iter().map(|w| (*w as f32).to_bits())));
    for (t, s) in table.iter().zip(plan.lr_scale()) {
        let lower = layer_index(&t.name).is_some_and(|i| i < 1);
        assert_eq!(*s == 0.1, lower, "{}", t.name);
    }
    let lr_groups: std::collections::BTreeSet<_> = plan.groups().iter().map(|g| g.lr_label.clone()).collect();
    assert_eq!(lr_groups.len(), 2, "two learning-rate rules, not one per tensor");
}

#[test]
fn without_a_split_every_entry_is_one_group() {
    let table = table();
    let entries = specs(&table);
    let plan = build_plan(&table, &lr_scales(&entries, None, 1).unwrap(), &weight_decays(&entries)).unwrap();
    assert!(plan.lr_scale().iter().all(|s| *s == 1.0));
    assert_eq!(plan.groups().len(), 1);
}

#[test]
fn a_scale_ojas_would_refuse_or_a_misaligned_vector_is_refused_here_first() {
    let table = table();
    let n = table.len();
    let wd = vec![0.01; n];
    for bad in [-0.1, f64::NAN, f64::INFINITY] {
        let mut s = vec![1.0; n];
        s[0] = bad;
        assert!(matches!(build_plan(&table, &s, &wd), Err(StepError::Contract(_))), "{bad}");
    }
    assert!(build_plan(&table, &vec![1.0; n - 1], &wd).is_err(), "short lr_scale");
    assert!(build_plan(&table, &vec![1.0; n], &wd[..n - 1]).is_err(), "short weight decay");
    // qd-train's per-entry contract (`check_per_entry`) wants a scale above zero: a zero would
    // freeze an entry while it still paid for moments.
    let mut s = vec![1.0; n];
    s[0] = 0.0;
    assert!(matches!(build_plan(&table, &s, &wd), Err(StepError::Contract(_))));
}
