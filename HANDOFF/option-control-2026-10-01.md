# option-control — 2026-10-01

Lane: Fable H, the per-option linear control for the general families. The lead's box finding
(per-family paired margins) was added here as well. Branch `worktree-agent-ac16dc1e5177b64f8`,
off main `88373b8`. It is local only: not pushed and not merged. Everything ran on the Mac's
CPU. Nothing ran on MPS, the GPU or the box.

Labels: **verified** means I ran it and the output is quoted or summarised from the run.
**Inferred** means it follows from verified facts but was not run. **Unverified** means it was
not checked.

**No number here is a ledger row.** These are Mac diagnostics. They promote nothing, and no row
was written to `ledger/`.

## What changed (commits)

- `885d5fe` **qd-prep linfit: features 3 (option pairs) and best-option selection.**
  - Each example is two documents, the pair text and the option text. They are hashed natively
    and stacked into `2*dim` columns.
  - The request carries a row id per training example.
  - The grid counts whole carve rows, each by its first option of greatest `z[1]-z[0]`
    (`linfit::Selection::BestOption`).
  - These are refused before training: a row split into two runs, a row on both sides of the
    carve, a validation row without exactly one positive, and a request with `k != 2`.
  - Features 1 and 2 are byte-identical. A test pins the BLAKE2b-256 of the replies that the
    `88373b8` build wrote for a documents request and a rows request.
- `8857557` **The per-option control.**
  - `python/qd_train/option_control.py` is the oracle. It sits outside `baseline.py`, so
    `control_cache` keys are unchanged.
  - `tools/linear_control_native.fit_options` is the native binding.
  - `tools/ft_linear_control.py --option-control` writes the control's own supplement row.
- `c394f6d` **Per-family paired margins for the letter and length controls** (the lead's box
  finding, J4 rows `d597ee7d`/`7921ae18`).
  - New keys: `paired_margin_vs_linear.{kind}.{family}` and
    `paired_margin_vs_length_control.{kind}.{family}`.
  - These are metrics only. The pooled gate and its semantics are unchanged.
- This handoff, plus four gap records (below).

In total: 8 files, +2,066 / −101 lines (`git diff --numstat 88373b8 c394f6d`).

## The per-option control, as built

- **Which tasks.** A task is scored only when `control_label_space(train, val) == "letter"`,
  the same per-row-options test the label uses.
  - On J1 that is CSQA, MMLU, `intent.classification` and `intent.within_domain`.
  - `intent.domain`, `intent.in_scope` and `code.defect_class` get
    `linear_option_control.top1.<task>` = `not_run`, with the reason "every row offers one
    option set".
- **Reading a row.** The options are `RequestDoc.offered`, the same source as
  `control_label_space`.
  - They are read only when the prompt's option block equals, byte for byte, the renderer's
    lines for them.
  - The question is the prompt's context region.
  - A row that fails either check is `not_run` with the reason: it is never fitted and never
    scored.
  - A task with no readable val rows is `not_run`.
  - This is my reading of Fable's "train and val option sets mismatch → not_run", applied per
    row, as the lead suggested.
- **Interpretation taken** (it is recorded on the row as `recipe.option_question_text`).
  Fable wrote "(question ⊕ option) pair text". I used the context region (the CSQA/MMLU
  question, the CLINC utterance), not the whole prompt minus options.
  - The remainder of the prompt is constant within a task, so a linear scorer gains nothing
    from it.
  - On the intent tasks it would have multiplied the nonzeros about fivefold.
  - This is reversible in one function, `read_option_row`.
- **Examples and features.**
  - Every shown option is one example, noul included.
  - Its features are the n-grams of `question + "\n" + option` in columns `[0, dim)` and of the
    option alone in `[dim, 2*dim)`.
  - Each block is L2-normalised exactly as `CharNGramHasher` normalises.
- **Fit and selection.** One binary scorer, fitted by `LinearBaseline._train_once` with two
  classes. This is the letter control's own Adam loop.
  - Shared with the letter control: grid `(1e-4, 1e-3, 1e-2, 1e-1)`, `tol`, `lr`,
    `max_iter` 6000 and the first-strictly-best rule.
  - The 20% carve is of ROWS, drawn by `default_rng(seed).permutation`.
  - The carve accuracy is the best-option top-1 per row, mirroring the letter control's
    selection on its own scored decision.
- **The row.**
  - It is a separate supplement with `eval_row_id`, `recipe.control = linear_option_control`,
    `report_only = true` and `comparator_decision = "the human's, under G1"`.
  - It has metrics only. Every key starts with `linear_option_control.` or
    `paired_margin_vs_linear_option.`, plus the recorder's own `code_that_ran`.
  - It writes no gate. The recorder's `_fill_unreported` still writes every required gate as
    `not_run`, as on every row, and `_joined` ignores those when a measured value exists.
  - A test asserts two things with and without the option row: every required gate and control
    joins to the same state, and the promotion reasons are identical apart from the unit's
    label.
  - It inherits the eval row's `quick` flag, as the letter row does. **A failed option-control
    run blocks the family's promotion until it is rerun**, like any failed row of the unit.
    That is fail-closed by design.
- **Refused under `--option-control`:** `--hold-out-operator` and `--control-cache`. The control
  is not cached.

## Oracle against native parity (verified)

This is J1's full CSQA task: 8,179 train rows / 1,197 val rows, 49,074 / 7,182 option
examples. Settings: seed 0, `max_iter` 6000, the oracle on its sparse operand
(`dense_budget_bytes=0`), native on 4 threads.

```
native:    converged=True iterations=201 l2=0.1 grad=9.406667180198243e-05 grid=[(0.0001, 270, 1635, 429), (0.001, 311, 1635, 122), (0.01, 321, 1635, 157), (0.1, 326, 1635, 201)]
reference: converged=True iterations=201 l2=0.1 grad=9.406667180198243e-05 grid=[(0.0001, 270, 1635, 429), (0.001, 311, 1635, 122), (0.01, 321, 1635, 157), (0.1, 326, 1635, 201)]
PARITY: {'converged': True, 'iterations': True, 'l2': True, 'selected': True, 'grid': True, 'grad_bits': True, 'loss_history': True, 'W_bits': True, 'b_bits': True, 'eval_logits_bits': True}; predictions 1197/1197 equal; val top-1 native 287/1197, reference 287/1197
```

The oracle took 293 s; native took 23.5 s on 4 threads. Each figure is from one run. The same
bit-for-bit comparison is a committed test on a CSQA-shaped fixture at production settings
(`test_the_native_option_fit_is_the_oracle_bit_for_bit`).

## J1, per task (verified, Mac diagnostics)

- **Docs.** J1's general-family docs, rebuilt by the csqa-control-fix lane with
  `ft_split_rows(... general_record=fetch-record-2026-09-29.json, replay_partition=True)`.
- **Pairing.** Re-verified here: every general val doc pairs 1:1 with
  `/Users/bharath/qd-campaign/p4-base-2026-10-01/verdicts.jsonl` by `(row_id, slot_name)`.
  The counts were 1,197 / 1,485 / 1,500 / 1,499 / 1,500 / 1,500.
- **How the margins were computed.** They ran through `score_against_option_control`, with the
  verdicts restricted to those 8,681 general rows.
- **Chance.** Uniform over the shown options, noul included. Noul is never gold on CSQA/MMLU,
  so excluding it gives 1/5 and 1/4.

| task | option control | letter control | chance | model (J1) | margin vs option | margin vs letter |
| --- | --- | --- | --- | --- | --- | --- |
| commonsense.multiple_choice | **287/1,197 (24.0%)** | 222/1,197 (18.5%) | 16.7% (20% excl. noul) | 860/1,197 | +0.479 [+0.444, +0.512] | +0.533 |
| knowledge.multiple_choice | **431/1,485 (29.0%)** | 368/1,485 (24.8%) | 20.0% (25% excl. noul) | 783/1,485 | +0.237 [+0.206, +0.269] | +0.280 |
| intent.classification | **0/1,500 (0.0%)** | 69/1,500 (4.6%) | 5.9% | 1,091/1,500 | +0.727 [+0.705, +0.749] | +0.681 |
| intent.within_domain | **0/1,499 (0.0%)** | 80/1,499 (5.3%) | 6.25% | 786/1,499 | +0.524 [+0.499, +0.549] | +0.471 |

- **Convergence.** All four option fits converged: 201 / 306 / 466 / 465 refit iterations.
  CSQA selected L2 = 0.1, the edge of the grid. The grid is mirrored, not tuned.
- **Pooled.** `paired_margin_vs_linear_option.choice` = +0.493, 95% CI [+0.478, +0.508], over
  5,681 rows.
- **Native time** on 18 threads: CSQA 7.6 s, MMLU 28.6 s, `intent.classification` 51.7 s
  (359,431 examples), `intent.within_domain` 48.9 s.
- **The letter column reproduces the csqa-control-fix handoff exactly.**
- **Why the intent tasks score 0** (verified on `intent.classification`):
  - 0/1,500 val gold intents were ever a training gold, because the CLINC split is by intent.
  - Every val row also shows at least one training-gold intent.
  - The control picks a training-gold option on 1,500/1,500 rows.
  - So its option block learns that val-only intents are never gold, and on these two tasks
    the margin measures the split, not the model.
  - For `within_domain` I infer the same mechanism, from the same split and the prior lane's
    0/1,499. → `GAP-OPTION-CONTROL-CLINC-SPLIT-DEFEATS-AN-OPTION-PRIOR` (owner: human, G1).
- **Seen in passing on the letter control** (value-space tasks, unchanged code):
  - `intent.domain` did not converge on the Mac either: 6,000 iterations, grad norm 1.746e-2.
    That matches the box's J4 finding.
  - `intent.in_scope`'s control scored 1,500/1,500. J1's margin against it is −0.245,
    95% CI [−0.267, −0.223].
    - Verified: every val gold is `yes` (1,500/1,500), while train holds 1,350 `no`. Out of
      scope is one split unit, so its rows all fall on one side, and a control that answers
      `yes` cannot miss.
    - **The per-family margins will surface this on the box** as a measured negative
      `paired_margin_vs_linear.choice.intent.in_scope`. It reflects the split.

## Per-family margins (the lead's finding)

- `score_against_control` now also writes `paired_margin_vs_linear.{kind}.{family}` and
  `paired_margin_vs_length_control.{kind}.{family}`.
  - Each covers that family's rows against that family's own controls, fitted by `fit_task` as
    before; `code.defect_class`'s is the by-value one, unchanged.
  - A family is `not_run` when any of its controls did not run.
  - If the verdicts are not the rebuilt split's population, every family margin is `not_run`.
    A model row the split lacks belongs to no family and would otherwise vanish silently.
- The family is `task.rsplit("/", 1)[0]`, because `RequestDoc` carries no family field and
  adding one would orphan the cache.
- **Not run on J4's real rows here** (unverified). The Mac does not hold J4's rebuilt split.

## Tests

- **Post-change** (verified, with `QD_PREP_BIN` set to the release build):
  - `cargo test -p qd-prep`: 36 unit + 2 integration tests passed. `clippy -D warnings` and
    `fmt --check` are clean.
  - These modules: `test_option_control`, `test_ft_linear_control`, `test_control_label_space`,
    `test_length_control`, `test_qd_prep_linear_parity`, `test_calib_fit_row`,
    `test_margin_probe_row`, `test_tool_call_sites`, `test_control_cache_key_agreement` and
    `test_gaps_ledger`. Result: 153 passed, 4 skipped.
  - The real-corpus skips in `test_qd_prep_linear_parity` were rerun with main's ignored
    `data/pool` files symlinked in: 35 passed, 2 skipped (the opt-in benchmark). The links were
    removed afterwards.
  - Ruff is clean on all changed Python.
  - **The full CPU suite** (`python/tests -k "not mps"`, at `c394f6d`, with main's `data/pool`
    files linked in and removed afterwards): **2,924 passed, 31 skipped, 2 failed**, in 581 s.
    - Both failures are the worktree-only ones the csqa-control-fix lane also reported:
      - `test_gaps_writer::test_the_real_ledger_is_not_touched_by_any_of_this` asserts the gaps
        file's parent directory is named `Lappi-decision`.
      - `test_lint_gate::test_ruff_is_installed_not_merely_declared` finds no `.venv/bin/ruff`
        in the worktree.
    - Neither touches a file this lane changed. No pre-existing test failed on the new
      per-family keys.
    - The 31 skips were not itemised (`-rs` was not passed).
  - Re-run after the last gap record and this file existed: `test_gaps_ledger`,
    `test_wire_gap_pins` and `test_gaps_writer` gave 42 passed and the same 1 worktree-only
    failure. So the five gap ids cited here resolve.
- **Pre-change** (verified):
  - **New tests on the `88373b8` tree:**
    - `test_ft_linear_control`: 2 failed, 39 passed. The two failures are the new family tests,
      on `KeyError: 'paired_margin_vs_linear.choice.code.defect_class'` and `….choice.fam`.
    - `test_option_control`: a collection error, `ModuleNotFoundError: qd_train.option_control`.
      This fails by API absence.
  - **New Python against the `88373b8` binary:** 7 of 18 `test_option_control` tests fail. They
    are every test that reaches the engine, refused with `features tag 3 is neither 1
    (documents) nor 2 (rows)`. The other 11 are oracle and refusal tests that never call the
    engine.
  - **Rust:** the new tests do not compile against the old crate (`Selection` is absent). The
    golden-reply test is a characterization test: it passes on both sides by design.

## Box (the lead deploys; I did not ssh)

- **New aarch64 binary,** cross-built from `885d5fe`:
  - path: `/Users/bharath/qd-campaign/target-aarch64-linux-optioncontrol/aarch64-unknown-linux-gnu/release/qd-prep`
  - sha256 `c7626dfd4bac7a2250c8835821a856b171aedca44b87f251041a4217289287d8`
  - It was **not run on aarch64** (unverified there).
  - It does not include the prep-perf lane's `lsh` subcommand. If both branches merge first,
    rebuild once from the merge commit.
- **Which binary for which run:**
  - `qd-prep-lc2` refuses `--option-control` loudly.
  - The letter control needs only Python, so `qd-prep-lc2` stays its engine.
- **The commands.** They are written out literally, all eight invocations, in
  `box_controls_optctl.sh`, which sits beside `box_controls_fixed.sh` in the lead's scratchpad.
  - It carries `box_controls_fixed.sh`'s guards: a clean `qd-lane5`, and no running control.
  - It also refuses unless `qd-lane5` contains `c394f6d`, and unless `qd-prep-oc1` has the
    sha256 above.
  - Block 1 is the letter control with per-family margins, on `qd-prep-lc2`, for J4 seeds 0–2
    and then J1. It supersedes `d597ee7d`/`7921ae18` and J1's fixed-control row.
  - Block 2 is the same four argvs plus `--option-control`, on `qd-prep-oc1`.
- **Do not cap these runs with `--max-fit-minutes` below about 20 h.** The option control's fit
  projection is the Python-calibrated worst case:

  | task | projection | native, Mac, 18 threads |
  | --- | --- | --- |
  | CSQA | 3.1 h | 7.6 s |
  | MMLU | 14.9 h | 28.6 s |
  | `intent.classification` | 16.4 h | 51.7 s |
  | `intent.within_domain` | 15.4 h | 48.9 s |

  A smaller cap refuses those tasks before fitting. This is the same overshoot shape as the
  letter control's documented 17.6×. The script passes no cap.

## Unverified

- The box binary on aarch64, and the option control's bit parity there. The Mac parity rests
  on the Mac's libm, as the letter control's does.
- The per-family margins on J4's real rows.
- The option control on J4's v3 split, and its wall clock on the box.
- `intent.within_domain`'s zero mechanism. It is inferred, not measured.

## Gaps

- `GAP-OPTION-CONTROL-GITPULSE-UNTRUSTED`
- `GAP-OPTION-CONTROL-LISTAGENTS-UNAVAILABLE`
- `GAP-OPTION-CONTROL-DEVMAP-INDEXES-MAIN-NOT-THE-WORKTREE`
- `GAP-OPTION-CONTROL-CLINC-SPLIT-DEFEATS-AN-OPTION-PRIOR`
- `GAP-GENERAL-LETTER-CONTROL-CANNOT-READ-PER-ROW-OPTIONS` (existing): partly answered for CSQA
  and MMLU, by a control above chance; not answered for the intent tasks.

## First command for the next lane

1. Review and merge this branch with a real merge (the script checks for `c394f6d`).
2. Deploy the `c7626dfd…` binary to the box as `/home/ubuntu/bin/qd-prep-oc1`.
3. Run `bash box_controls_optctl.sh` from the lead's scratchpad, as `box_controls_fixed.sh` was
   run.
4. Report the general-family option margins only with
   `GAP-OPTION-CONTROL-CLINC-SPLIT-DEFEATS-AN-OPTION-PRIOR` beside them.
