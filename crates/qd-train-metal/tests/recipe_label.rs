//! A Metal full fine-tune row is labelled with the Metal label, not torch's `master`
//! (Fable's board-sweep ruling 2h). `run.rs` builds the row only on macOS with a GPU, so the
//! check here reads its source: the label must come from the one constant, and no literal
//! `"master"` may be written beside `optimizer_recipe`.

#[test]
fn run_rs_labels_the_recipe_from_the_one_constant() {
    let src = include_str!("../src/run.rs");
    assert!(
        src.contains("optimizer_recipe: METAL_FULL_FT_OPTIMIZER_RECIPE.into()"),
        "run.rs must label the recipe with qd_train::ledger::METAL_FULL_FT_OPTIMIZER_RECIPE"
    );
    for line in src.lines().filter(|l| l.contains("optimizer_recipe")) {
        assert!(!line.contains("\"master\""), "a Metal row labelled master: {line}");
    }
}
