# v5 promotion verdict re-run under the 2026-10-08 re-specs (task ft-f8a5a19e51e32bf90f07405259d19aae, criterion 2)

Lane L1, branch `wave1-l1-promotion-verdict` (cut from main 94b9d0d, which contains e99c643 and the v5 ledgers from 72119a6).
Full outputs, the pre-e99c643 decisions record and `SHA256SUMS` are in `AUDIT/promotion-verdict-respec-2026-10-13/`.
No ledger, gate, threshold or `docs/promotion-decisions.json` byte was changed. The `verdict` subcommand only prints.

## Navigation gaps

- DevMap: no index in this worktree (`gitpulse_insights` codeintel.available=false, no `.devmap/codeintel/devmap.sqlite`). Nothing here is graph-confirmed; the CLI flags were read from `python/qd_train/ledger.py:2961-3040` (`_build_arg_parser`, `_cmd_verdict`) [verified, direct read].
- GitPulse insights: worktrees, agents, collisions facets all ok; 0 overlapping files.

## CLI (verified from source)

`python -m qd_train.ledger verdict (--seed-family H | --row-id ID) [--kind seeds|avg] [--ledger P] [--input-ledger P ...] [--decisions P]`.
`--decisions` exists, so the pre-re-spec record can be passed without editing code. Exit 0 = promoted, 1 = refused, 2 = usage/ledger error.
`--row-id` needs the full UUID (the short `6af73bef` matches 0 rows; that first attempt was discarded and re-run with the full id).
Interpreter: `/Users/bharath/.venvs/ml/bin/python3` with `PYTHONPATH=python`.

## Commands, exit codes

Records: after = `docs/promotion-decisions.json` (sha256 af21801e61de...); before = `git show e99c643^:docs/promotion-decisions.json` (sha256 e49fd446...), saved as `promotion-decisions.pre-e99c643.json`.

| # | Command (all `PYTHONPATH=python`, ml venv) | rc | Output |
|---|---|---|---|
| 1 | `verdict --ledger ledger/h100x2-v5-2026-10-03.jsonl --row-id 166f7ebd-695c-4294-8e8a-75c07aaba068` | 1 | `seeds.after.txt` |
| 2 | same + `--decisions AUDIT/promotion-verdict-respec-2026-10-13/promotion-decisions.pre-e99c643.json` | 1 | `seeds.before.txt` |
| 3 | `verdict --kind avg --ledger ledger/h100x2-v5avg-2026-10-05.jsonl --row-id 6af73bef-aa56-49b3-8cdb-e0b0b56e9181 --input-ledger ledger/h100x2-v5-2026-10-03.jsonl` | 1 | `avg.after.txt` |
| 4 | same + `--decisions ...pre-e99c643.json` | 1 | `avg.before.txt` |

The seed family is the one `--row-id 166f7ebd` (seed-0 val-score eval row) resolves to: 19 rows, seeds 0-4 (seed 4 is a 3-row unit, the others 4-row units). stderr empty in all four.
sha256 of each full output: see `SHA256SUMS`.

## Before / after delta

| | before (pre-e99c643 record) | after (current record) |
|---|---|---|
| seeds verdict | REFUSED. `promotion_population: UNKNOWN: the promotion decisions record could not be read` ... `no entry for ['paired_margin_population', 'shuffled_label_seeds']` (the "refuses every record" state). Every gate is read as built, and nothing is judged on a population. | REFUSED, but on judged gates: `promotion_population: code.defect_class`, record af21801e61de, and every gate is read under it. |
| avg verdict | REFUSED, same unreadable-record reason, 6 rows. | REFUSED, 6 rows, read under the record. |

So the re-specs change the reason for refusal, not the outcome. v5 does not promote either way.

## Gate table, after (current record), seeds 0-4 (values from `seeds.after.txt`)

| Gate / control | as built | under record | where it names the re-spec |
|---|---|---|---|
| `paired_margin_vs_linear` | not_run on every seed | PASS on all five seeds (0.3915, 0.3937, 0.3898, 0.3911, 0.3919; [2304/2304] each) | `paired_margin_population = per family over the population` (judged on code.defect_class) |
| `permutation_consistency` | FAIL on every seed (0.9288 to 0.9493) | PASS on every seed (0.9922 to 0.9952; 2286-2293 of 2304) | `promotion_population = code.defect_class` |
| `ood_abstain` | FAIL on every seed (152, 156, 158, 162, 155 of 180) | still FAIL, same counts | `promotion_population` |
| `needle_hunk_recall` | FAIL [300/300] (two units) | not re-read (not population-judged) | none |
| `ece` | not_run (7 of 20 inputs: n<100 per shape, no `lang` metadata) | PASS [1/1] reported under `ece_population` in the avg and seed output | `ece_population` (separate earlier decision) |
| `degenerate_head` | FAIL [15/15] | PASS [15/15] | `degenerate_head_floor = class share only` |
| `shuffled_label` | required on every seed | `shuffled_label_seeds = at least 3 seeds: ran and passed at complete coverage on 4 seed(s) [0, 1, 2, 3]; not counted: seed 4` | `shuffled_label_seeds` |
| `privileged_hunk`, `transfer_gate` | not_run | not required (decided retired) | retired decisions |

Refusal reasons that remain after the re-specs: `ood_abstain` FAIL on all seeds, and `needle_hunk_recall` FAIL on the two units that carry it (listed as `-` lines in `seeds.after.txt`).
Caveat on seed 4: its eval unit is 3 rows, not 4, and the shuffled-label control passes on seeds 0-3 only; seed 4 is "not counted". [verified, verbatim from output]
Seed 4 has 3 rows because it has no `--shuffled-label` row. Its unit holds only the score-val, `paired_margin_vs_linear` and option-control rows, so the control was never run on seed 4; it did not run and fail. The record's "at least 3 seeds" is met by seeds 0-3. [verified by the adversarial re-check, 2026-10-13]

## Averaged row 6af73bef (`avg.after.txt`)

- `paired_margin_vs_linear`: not_run as built and under record. The reason is `paired_margin_vs_linear.choice.code.defect_class is absent from rows 6af73bef...`. This is reported as not_run, not as pass, and it blocks promotion.
- `ood_abstain`: FAIL 0.8278 [149/180] under record.
- `permutation_consistency`: PASS 0.9961 [2295/2304] under record (as built 0.9571 PASS).
- `shuffled_label`: did not run ("never evaluated by this run").
- `degenerate_head_floor`: PASS under record (as built FAIL).
- `ece`: PASS [1/1] under `ece_population` (as built not_run). The same output still lists 7 of 20 inputs as not run and the per-language half as unmeasured. `ece_population` covers only the per-shape half, so per-language ECE stays unmeasured under the record.
- `average_may_promote` is decided false by the human (2026-10-03 ruling), which refuses independently.

## Claims

- Verified (ran, output saved): the exit codes and outputs above; `--decisions` is accepted; the pre-record state refuses on the unreadable record naming exactly `paired_margin_population` and `shuffled_label_seeds`; the current record is read, and both new decisions appear by name in the output.
- Verified: `python/tests/test_promotion_respec.py` = 21 passed (`/Users/bharath/Code/research/Lappi-decision/.venv/bin/python3 -m pytest python/tests/test_promotion_respec.py -q`; the ml venv has no pytest). The `-rA` re-run counted 21 PASSED.
- Inferred: the "before" run is the state the 2026-10-12 sweep ruling described (verdict refusing every record), simulated by feeding the pre-e99c643 record to current code. It is not a replay of a historical run on historical code.
- Inferred: that `ece` PASS [1/1] relies on `ece_population` ("per shape over the population"), which is a separate prior decision and not one of the two 2026-10-08 re-specs.
- Not done: no ledger row written (the verdict CLI does not write one); no GPU or Mac eval run; DevMap not consulted (no index).
