# L1: qd-train onto ojas-train (card ft-d066187f), 2026-10-20

Branch `lane/ft-d066187f` from main d09222d. Lane L1 of the GitPulse backlog run, lead session lappi-decision-0f.

## Result: STOPPED before any deletion. The switch is blocked by one behaviour ojas-train lacks.

No Cargo or source file changed. The only edits on the branch are `gaps.jsonl` (one record) and this file.

## What was measured

Method: `diff` of each Lappi module against `/Users/bharath/Code/research/ojas/ojas-train/src/lappi/<same>.rs`
(ojas 4123dfb), and `git diff fecfd5d d09222d` on the eight moved files (fecfd5d is the import base).
No ledger rows: nothing was run.

| Moved file | Post-fecfd5d Lappi change | In ojas 4123dfb? |
| --- | --- | --- |
| step, trainer, schedule, adamw, mock, pyjson | none | n/a (ojas differs by its rewiring only) |
| run_control | ffee022 rustfmt of one test, formatting only | n/a |
| **recipe** | **200a2d2: `lora_adapter_layout` returns entries in canonical draw order** | **NO** |

The delta, exactly. Lappi `crates/qd-train/src/recipe.rs` (after the `b_of` check, about line 347):

```rust
let names: Vec<&str> = a_of.iter().map(|(m, _, _)| m.as_str()).collect();
let order = crate::lora_init::canonical_draw_order(&names)
    .map_err(|e| StepError::Contract(e.to_string()))?;
Ok((rank, order.into_iter().map(|i| (a_of[i].0.clone(), a_of[i].2)).collect()))
```

ojas `ojas-train/src/lappi/recipe.rs` still ends `Ok((rank, a_of.into_iter().map(|(m, _, i)| (m, i)).collect()))`
(provider listing order), with the old doc comment, and has no ordering hook. Consequences if Lappi's
recipe.rs were deleted: `ledger.rs:1173` (the ledger verifier recomputing A) and
`tests/lora_plan.rs:273-298` (`the_adapter_layout_is_canonical_whatever_order_the_provider_lists_entries_in`
and its refusals) lose the canonical order, which the owner approved on 2026-10-19 (see below).
`lora_init::canonical_draw_order` stays in Lappi, so ojas cannot call it; ojas needs the function itself or a
parameter.

Not a partial switch: ojas's trainer consumes ojas's `recipe` types (`LoraGroup`), so keeping recipe.rs in
Lappi while moving the other seven would leave two type families.

Also found: `crates/qd-metal/src/ledger.rs:392,399` use `qd_train::pyjson::{dumps, CANONICAL_ASCII, float}`,
a moved symbol, reached directly (not through `ledger`). qd-metal needs repointing too when the switch lands.
`crates/qd-train-metal` has 28 references to moved modules.

## Owner decision, 2026-10-19, in their words (from `AUDIT/lora-2b-w2-2-2026-10-18/owner-decision-2026-10-19.md`)

Item 2a, asked after ojas main reached 4123dfb: "Which pin target?" Answer: "Pin 4123dfb (phase A+B) (Recommended)".
Effect recorded there: pin target becomes `4123dfbe570794d9b498c2bb0614990dee285ec0`; delta report in
`AUDIT/ojas-pin-4123dfb-2026-10-19/`; "Lappi build breakage is fixed in the same lane and verified under the
lock before it merges." Item 1 (the same file): "Should I commit the amendment ... and merge W3-1 once its
tests pass?" Answer: "Yes, amend and merge (Recommended)".
The ojas work order (`tools/a6/LAPPI-WORK-ORDER.md` L0) records the pin bump as: Owner, 2026-10-19: "choose the best".

## What is open

- GAP-L1-OJAS-TRAIN-RECIPE-LACKS-CANONICAL-LORA-A-DRAW-ORDER-2026-10-20 (gaps.jsonl): ojas-train's
  `lora_adapter_layout` needs the canonical draw order (layer index, then D6 suffix table, refusing a module with no
  layer index or two at one position), with a parity test equal to `tests/lora_plan.rs:273-298`.
- AC1 `make pin` PASS is owner-run (not run here).
- Baseline test counts on d09222d, and every after-count: NOT RUN (nothing changed, and the heavy lock was not taken).

## First command for the next lane

After ojas carries the canonical order and the pin moves to that ojas sha (owner yes; do not edit
`upstream.lock.json` before that):

`git switch -c lane/ft-d066187f-2 main` then re-run the module diff:
`diff crates/qd-train/src/recipe.rs /Users/bharath/Code/research/ojas/ojas-train/src/lappi/recipe.rs`
and proceed with the card's steps 2-5 only if the only differences are the `crate::`/`super::` paths and `LoraHostNames`.
