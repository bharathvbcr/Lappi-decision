# csqa-control-fix — 2026-10-01

Lane: why the FT linear control on `commonsense.multiple_choice` was nonsense. Branch
`csqa-control-fix`, off main `7fc3af7`. Local only: not pushed, not merged. Everything ran on
the Mac's CPU. Nothing ran on MPS or the GPU, and nothing was sent to the box.

Labels: **verified** means I ran it and the output is quoted below. **Inferred** means it
follows from verified facts but was not run. **Unverified** means it was not checked.

**No number here is a ledger row.** These are diagnostics from scratch runs on the Mac, and
they promote nothing.

## The cause

`qd_train.baseline.request_texts` labelled every control doc by its gold **value**. The line is
`value=slot.letter_to_value[letter]` (`python/qd_train/baseline.py:200` at `7fc3af7`).
`tools/ft_linear_control.fit_task` fitted on those values (`ft_linear_control.py:396`
`classes`, `:411` `train_labels`) and scored val by `p == d.value` (`:457`).

A value is a class only when every row offers the same option set. CommonsenseQA and MMLU
answer with the text of one of the question's own options
(`qd_data/general.py` `rewrite_csqa`/`rewrite_mmlu`: `gold_text = options[...]`).
CLINC's `repo_key` is the intent (`general.py` `rewrite_clinc_two_stage`,
`mixture.py` `rewrite_clinc`), so the split is by intent.

Measured on J1's docs (**verified**). The rebuild used `ft_split_rows(... general_record=
fetch-record-2026-09-29.json, replay_partition=True)` on this Mac, then `request_texts`.

| task | train docs | k by value | val golds that are train classes | k after fix |
| --- | --- | --- | --- | --- |
| commonsense.multiple_choice/answer | 8,179 | **4,414** | 757/1,197 (no replay: 9,619 rows, k=4,956) | 5 (letter) |
| knowledge.multiple_choice/answer | 12,072 | **10,323** | 276/1,485 (no replay) | 4 (letter) |
| intent.classification/intent | 21,143 | 133 | **0/1,500** | 17 (letter) |
| intent.within_domain/intent | 21,141 | 133 | **0/1,499** | 16 (letter) |
| intent.domain/domain | 21,142 | 11 | 1,500/1,500 | 11 (value, unchanged) |
| intent.in_scope/in_scope | 21,143 | 2 | 1,500/1,500 | 2 (value, unchanged) |

These are J1's docs, not just similar ones (**verified**). My CSQA docs hash to `8179 rows x
65536 cols, 8014545 nonzeros, 1197 eval rows`, the same as J1's box log line. Every rebuilt
general val doc pairs 1:1 by `(row_id, slot_name)` with J1's verdicts
(`/Users/bharath/qd-campaign/p4-base-2026-10-01/verdicts.jsonl`): 1,197 + 1,485 + 1,500 +
1,499 + 1,500 + 1,500 paired, 0 unpaired. So **pairing is not the defect**. The label space
is the defect, and so is k.

**It is not native-only** (**verified**). With value labels, the Python oracle fails exactly
as qd-prep does:
- **The CSQA-shaped fixture**, pre-fix, through `fit_task` (native): grid `0/24` at every L2,
  val 0/50.
- **The first 60 / 30 of J1's real CSQA train / val docs**, value labels (k=60, one class per
  row), production settings, sparse oracle against native:
  ```
  native: grid [l2=0.0001 0/12 UNCONVERGED in 6000; l2=0.001 0/12 UNCONVERGED in 6000; l2=0.01 0/12 converged in 671; l2=0.1 0/12 converged in 755]; refit l2=0.0001 converged in 249 iterations
  native:    converged=True iterations=249 l2=0.0001 grad=9.944192939234302e-05 val top-1 1/30
  reference: converged=True iterations=249 l2=0.0001 grad=9.944192939234302e-05 val top-1 1/30
  PARITY: converged True, iterations True, l2 True, grad bits True, W bit-identical True, predictions 30/30 equal
  ```
  Only native printed the per-grid accuracies. The reference picked the same L2 by the
  same first-best rule, and its refit is bit-identical.

**J1's `intent.classification` 253/4228 is a different thing**, but the same bug sits
underneath it. In value space every selection-slice gold *was* reachable (k=133). The fit did
not converge at 6,000 iterations. 253 is exactly the number of `noul`-gold rows in that
4,228-row selection slice (**verified**: same permutation, counted). So J1's intent control
scored the same as always answering the majority class (**inferred**). On val it could not
have answered any row: 0/1,500 val golds were classes.

## Which families it hits

- **Hit** (verified on J1's docs): `commonsense.multiple_choice`,
  `knowledge.multiple_choice`, `intent.classification`, `intent.within_domain`.
- **Not hit** (verified: `control_label_space` returns `value` and every label equals the
  pre-fix label): `intent.domain`, `intent.in_scope`, and `code.defect_class`.
  - For `code.defect_class` I checked two rebuilds:
    - the phase-3 split, corpus-v2 alone: 45,405 train / 2,332 val, one option set
      `[clean, cosmetic, logic, noul, stub]`;
    - corpus-v2 plus defect-noul-v1: 47,907 / 2,332.
  - J4's v3 corpus was **not** rebuilt here (unverified). Its slot is the same
    `ChoiceSlot(options=DEFECT_CLASSES)` (`mixture.py:703`), so I infer the same result.
- `qa.answer_span` is a span slot and was never fitted. That is unchanged.

## The fix (commit `bc51bd7`)

- **`RequestDoc` gains `letter` and `offered`.** `letter` is the gold letter in this rendering.
  `offered` is the rendering's `(letter, value)` pairs, noul included.
- **`control_label_space(train, val)`** sits in `baseline.py` beside `RequestDoc`, so the
  oracle and the tool share one rule. It returns `value` when every row offers one option
  set, and `letter` otherwise. It reads the option sets the prompt offers, never the golds.
  It raises in three cases:
  - train offers one set and val offers another;
  - a doc carries no rendering;
  - the docs span more than one task.
- **`fit_task` uses it for both arms.** A refusal becomes that task's `not_run`, which makes
  the pooled gate `not_run`.
- **`linear_control_top1` detail** now ends `labelled by value|letter`.
- **Unchanged:** features, objective, grid, `max_iter`, `tol`, `lr`, L2 selection, the Rust
  engine, and `control_cache`.
- **Cost: every cached control is orphaned.** `control_cache._digest_source()` hashes
  `baseline.py`, so all cache keys change. For the four affected tasks the key would have
  changed anyway, because `train_labels` changed. For `code.defect_class` it means one
  refit, about 2 minutes native.
- Docstrings that said "value, not letter" now state the rule:
  - `RequestDoc`;
  - the `ft_linear_control` module docstring;
  - the `linear_control_native` module docstring;
  - the docstring of `test_a_native_fit_is_cached_and_read_back_under_the_unchanged_key`.

## Tests

`python/tests/test_control_label_space.py` has 9 tests. I ran a copy without the new imports
against `7fc3af7` (**verified**): **8 failed, 1 passed.** The one that passes,
`test_a_fixed_option_set_task_hands_the_engine_its_values_unchanged`, is a characterization
test: fixed-set labels equal values before and after the fix, and passing on both sides is
its point. Two of the eight fail on assertions:

- `test_a_csqa_shaped_control_beats_chance`:
  - **pre-fix** `Ran(value=0.0, n=0, n_total=50)` with grid `0/24`;
  - **post-fix** 39/50 = 0.78, against chance 0.2.
- `test_no_task_hands_the_engine_more_classes_than_a_row_can_offer`:
  - **pre-fix** `('commonsense.multiple_choice/answer', 40)`, `assert 40 <= 17`.

The other six fail on `NameError`/`AttributeError`/`TypeError`, because the API they test
does not exist pre-fix. They cover:
- the rule: one set → value; per-row → letter; and the letter names the gold value on its
  own row;
- the refusals (split mismatch, bare doc, two tasks), including the task going `not_run`;
- letter-space parity, native against oracle.

Score slots stay in value space. `change_scope`'s `bins` is the constant
`len(CHANGE_SCOPE_BIN_EDGES) + 1` (`mixture.py:421`), so every row offers one set. I read
this from the code; no real score task reached `control_label_space` in a test. Even if it
did not, an ordinal slot's letters map to bins in order and are never shuffled, so the two
spaces would be isomorphic.

The existing helpers in `test_ft_linear_control.py` and `test_length_control.py` now build
docs with `letter`/`offered`. The parity test there now uses the shared rule.

Post-fix runs (**verified**):
- `test_control_label_space`, `test_ft_linear_control`, `test_length_control`,
  `test_qd_prep_linear_parity`, `test_calib_fit_row`, `test_margin_probe_row`,
  `test_tool_call_sites`: 111 passed, 4 skipped.
- The 2 real-corpus skips in that run were rerun with the corpus symlinked from main's
  checkout: `test_qd_prep_linear_parity` 35 passed, 2 skipped (the skips are the
  `QD_PREP_BENCH` benchmark).
- `test_gaps_ledger` passes.
- Ruff is clean on all six files.
- **The full CPU suite** (`python/tests -k "not mps"`, at `bc51bd7`, with the corpus
  symlinks below in place): **2,893 passed, 27 skipped, 2 failed.** Both failures come from
  running in a worktree, not from this change:
  - `test_lint_gate::test_ruff_is_installed_not_merely_declared`: the worktree has no
    `.venv/bin/ruff`.
  - `test_gaps_writer::test_the_real_ledger_is_not_touched_by_any_of_this`: it asserts the
    gaps file's parent directory is named `Lappi-decision`, and here it is
    `agent-a55f38d6486706fee`.

  I read both assertions; I did not rerun them on `7fc3af7`. The phase-4 handoff reports the
  same kind of worktree failures.
- To run the real-corpus tests, I symlinked main's ignored `data/pool` files into the
  worktree for those runs, then removed the links. No symlink remains.

## Oracle against native, on real CSQA docs (verified)

This is J1's full CSQA task: 8,179 train / 1,197 val, letter labels, production settings
(seed 0, `max_iter` 6000, `CharNGramHasher(3,5,2**16)`). The oracle is `LinearBaseline` on its
sparse operand (`dense_budget_bytes=0`), the operand qd-prep reproduces.

```
native: qd-prep linfit: 8179 rows x 65536 cols, 8014545 nonzeros, 1197 eval rows; grid [l2=0.0001 343/1635 converged in 256; l2=0.001 334/1635 converged in 138; l2=0.01 340/1635 converged in 159; l2=0.1 340/1635 converged in 203]; refit l2=0.0001 converged in 227 iterations (grad norm 9.973588e-5)
native:    converged=True iterations=227 l2=0.0001 grad=9.973587647897201e-05 val top-1 222/1197
reference: converged=True iterations=227 l2=0.0001 grad=9.973587647897201e-05 val top-1 222/1197
PARITY: converged True, iterations True, l2 True, grad bits True, W bit-identical True, predictions 1197/1197 equal
```

The oracle took 728 s and native took 25 s on 4 threads. Each is one run.

## What the fixed control does on J1's other affected tasks (native only, verified)

Same docs. Selection is on the 20% train carve; "val" is val top-1.

| task | iterations to converge | selected L2: selection | val |
| --- | --- | --- | --- |
| commonsense (k=5) | 227 | 343/1,635 | 222/1,197 (18.5%; chance 20%) |
| knowledge (k=4) | 449 | 628/2,414 | 368/1,485 (24.8%; chance 25%) |
| intent.classification (k=17) | 157 | 315/4,228 | 69/1,500 (4.6%; 16 options + noul) |
| intent.within_domain (k=16) | 301 | 363/4,228 | 80/1,499 (5.3%) |

**All four now converge**, so none stays NotRun. **All four are at chance.** The control is
honest: same input as the model, same output space, labels that mean the same on every row.
But a bag of char n-grams can tie a letter to an option's content only through
letter-prefixed n-grams (`"B. sh"`).

- **J1 at its own CSQA accuracy would read as a +0.5 margin.** The model's 71.9% against
  18.5% is about +0.53 (**inferred**: not computed as a paired margin, and not a ledger row).
  That says little.
- **A stronger opponent changes the control's definition.** A candidate-scoring control
  scores each option with one weight vector over question-plus-option features. That changes
  the objective and the features, which rule 2 reserves, so I did not build it.
- **This is Fable's/the human's call.** `GAP-GENERAL-LETTER-CONTROL-CANNOT-READ-PER-ROW-OPTIONS`.
  Until it is answered, **no general-family `paired_margin_vs_linear` should be reported as a
  finding, even after the rerun**.

## Rows this invalidates

No committed ledger row carries a general-family control metric (**verified**: `rg` over
`ledger/` and `/Users/bharath/qd-campaign/**/*.jsonl` finds `linear_control_top1.` only for
`code.defect_class`). The invalid rows are the ones the box writes, or wrote, with
`tools/ft_linear_control.py` before this commit, over a split with the general families:

- **J1**: the control row for eval `09ff303f` (the native run behind the 4/1635 log).
- **J4**: the three per-seed control rows on v3 `89b619d9`.

In each of those rows:
- **Invalid:**
  - `linear_control_{convergence,top1}` and `length_control_{convergence,top1}` for the four
    hit tasks (the length arm used the same labels);
  - `paired_margin_vs_linear` and `.choice`;
  - `paired_margin_vs_length_control`.
- **Still valid:** the per-task metrics for `code.defect_class`, `intent.domain` and
  `intent.in_scope`, whose labels are identical.
- **Pre-fix, J1's pooled gate was probably `not_run`:** `intent.classification` was
  UNCONVERGED. After the fix all tasks converge, so a rerun will write a `Ran` pooled gate
  that includes near-chance letter controls (**inferred**). See the gap above before reading
  it.

Unaffected, all defect-only with labels identical (**verified** on corpus-v2):
- phase 3 (`eeda5db4`, `2b08a357`, `635c19d9`);
- the Mac rows `bba89379` / `43a366d0`;
- `lc-parity-2026-10-01`;
- the `mac-phase3-*` smokes.

## Open

- A full value-label native refit of J1's CSQA, meant to reproduce `4/1635` on the Mac, was
  **stopped** after about 35 minutes on 12 threads (J1 took 1,423 s on 64), to free the CPU
  for the full test suite. So `4/1635` itself was not reproduced here (**unverified**). The
  nnz identity above is the evidence that the inputs are J1's.
- J4's v3 defect rows: label space inferred, not rebuilt (above).
- The control's cache key covers train labels but not val golds (pre-existing). The label
  space is a function of the offered sets, which are in the val text, so a cached verdict
  cannot be read back across spaces (**inferred**).
- `GAP-GATES-POOL-THE-GENERAL-FAMILIES-INTO-DEFECT-CONTRACTS` (existing) still applies: the
  pooled gate mixes defect and general letter rows.

Gaps recorded: `GAP-CSQA-CONTROL-GITPULSE-UNTRUSTED` (every GitPulse facet failed,
REPOSITORY_TRUST_REQUIRED, so collisions were not checked),
`GAP-CSQA-CONTROL-LISTAGENTS-UNAVAILABLE`, `GAP-CSQA-CONTROL-DEVMAP-INDEXES-MAIN-NOT-THE-WORKTREE`,
`GAP-GENERAL-LETTER-CONTROL-CANNOT-READ-PER-ROW-OPTIONS`.

## First command for the next lane

First, review and merge `csqa-control-fix` into main. Only Python changed, so the box's
cross-built `qd-prep` is still the engine.

Then, on the box, from a checkout at the merge commit:
- run `/home/ubuntu/box_j4_controls_native.sh` (`HANDOFF/gh200-phase4-2026-10-01.md:185`
  names it the native control runner of record from J4 on);
- run J1's equivalent native control script for eval row `09ff303f`;
- point `--control-cache` at a fresh directory in both.

I could not read those scripts from the Mac, so their argv is not restated here. Their split
flags have to be the ones the eval rows were built with.

Do not report their general-family margins until the gap above is answered.
