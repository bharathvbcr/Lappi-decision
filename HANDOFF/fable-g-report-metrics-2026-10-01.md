# Fable round G: report metrics for the open gate questions (2026-10-01)

Lane: the advisor decision "Fable round G" (G1, G2, G3, G7, G8). Branch
`worktree-agent-afa44428978bc5521`, fast-forwarded to main `73bcf12` before any change; not
merged, not pushed. **Everything here is report-only:** no gate's pass/fail, threshold or
population changed, and nothing became promotable (rule 2).

## What was measured (ledger rows)

- **Cross-fit calibration rows, J4 seeds 0/1/2:** `385dc91e`, `df9468ae`, `8feb5a8f` in
  `ledger/mac-calib-crossfit-2026-10-01.jsonl` (quick, Mac CPU, from `d8a0047`, split key
  `j7-calib`). Every reported number equals J7's two-fold rows `585f3ce2` / `986f5b95` /
  `dc946a82`: bit for bit on seeds 0 and 1 (72 numbers each), within 8.1e-9 on seed 2 (J7 fitted
  on the box's glibc libm, these on macOS). Out-of-fold ECE after calibration, k2 / k4:
  0.0230 / 0.0169, 0.0084 / 0.0171, 0.0105 / 0.0146.
- **Gate report rows** in `ledger/mac-gate-report-2026-10-01.jsonl` (quick, `run_kind` eval,
  Mac CPU, from `0603f7f`, written by `tools/gate_report_row.py`):

  | Report row | Re-reads eval row | Verdicts sha256 |
  | --- | --- | --- |
  | `676e6498` | `2f5fe57a` (J4 s0) | `783e38be…` |
  | `a8ddf12c` | `f3612f73` (J4 s1) | `0f956a83…` |
  | `1563b03c` | `02cf5ff4` (J4 s2) | `f4923587…` |
  | `c308c802` | `1d93b3ee` (the average) | `fe544685…` |

  On each row, `gate_report.cross_check` is 52/65: 52 numbers the eval row records were
  recomputed from its verdicts and are equal, and 13 are absent or not_run there. Each row has
  165 metrics, 103 ran and 62 not_run, and none carries a failed verdict.
  - Rows 2–4 carry a `-dirty` `code_commit`. Their only dirt is this ledger file, which row 1
    created untracked; the same happened in `mac-calib-crossfit`.
  - The tables below cite these rows. Per seed, s0 / s1 / s2:

  | Report-only number | defect_class | MMLU | CSQA | Pooled (the gate's population) |
  | --- | --- | --- | --- | --- |
  | permutation agreement | 0.986 / 0.983 / 0.991 | 0.664 / 0.624 / 0.700 | 0.868 / 0.865 / 0.897 | 0.914 / 0.897 / 0.923, FAILS 0.95 |
  | in-distribution abstention | 0.014 / 0.017 / 0.009 | 0.336 / 0.376 / 0.300 | 0.132 / 0.135 / 0.103 | 0.113 / 0.144 / 0.105, cap 0.05 |
  | ECE (own slot shape) | 0.020 / 0.010 / 0.015 | 0.040 / 0.027 / 0.030 | 0.037 / 0.027 / 0.039 | gate not_run |
  | mean entropy, nats | 0.090 / 0.096 / 0.092 | 1.066 / 1.083 / 1.091 | 0.803 / 0.831 / 0.852 | k4 0.472 / 0.483 / 0.483 |

  The intents: permutation 0.86–0.99, abstention 0.010–0.219, ECE 0.033–0.315 (the per-shape
  table is in the report).
  - **G2:** ECE over all 10,985 letter rows, pooled across shapes, is 0.080 / 0.087 / 0.069. The
    `none` language bucket (8,681 rows) is 0.096 / 0.108 / 0.084; python 0.016 / 0.012 / 0.015.
  - **G3, k2** (only `intent.in_scope`): entropy 0.065 / 0.107 / 0.144, under the 0.15 floor on
    every seed. Accuracy is 0.924 / 0.893 / 0.912, and the predicted marginal is at most 2.0σ
    from a 49/51 gold. That is a confidently right head, not a collapsed one.
  - **G3, k4, defect_class:** entropy 0.090–0.096, also under the floor, with marginals within
    1σ. The pooled k4 control passes only because MMLU lifts the mean.
  - **G3, k4, MMLU:** over-predicts option 3 by +11.6σ / +16.3σ / +7.9σ, a last-position bias.
- **The average `1d93b3ee`** (report row `c308c802`). The report's per-family numbers for the
  average:

  | Report-only number | defect_class | MMLU | CSQA | within_domain |
  | --- | --- | --- | --- | --- |
  | permutation agreement | 0.990 | 0.669 | 0.899 | 0.919 |
  | in-distribution abstention | 0.010 | 0.331 | 0.101 | 0.107 |
  | ECE | 0.009 | 0.041 | 0.041 | 0.191 |

  - Mean entropy: defect_class k4 is 0.111 and in_scope k2 is 0.171.
  - G2: pooled ECE is 0.0304, and the `none` bucket is 0.0373.
  - G3: MMLU over-predicts option 3 by +14.5σ; across all k4 rows the excess is +9.8σ.
  - The OOD suite has no val family: per-family `ood_suite` is not_run, with that reason.
- **`verdict --kind avg` on the average**, with `--input-ledger ledger/gh200-p4-v3-2026-10-01.jsonl`,
  prints `REFUSED [avg]` over 4 rows: the average and its three J4 ft inputs.
  - The input checks raised nothing. The inputs are non-quick and ran to `steps_exhausted`; they
    share one recipe, snapshot, tokenizer and backbone; there are 3 distinct seeds, matching
    `recipe.averaged`. On real data that is no structural finding.
  - It is refused by its own gates:

    | Gate | Result |
    | --- | --- |
    | `ood_abstain` | FAILED, 0/180 |
    | `needle_hunk_recall` | FAILED, worst bucket 0.2787 |
    | `permutation_consistency` | FAILED, 10127/10985 |
    | `paired_margin_vs_linear` | not run |
    | `ece` | not run: 8,681 rows carry no language |

  - Also by its controls: `shuffled_label`, `privileged_hunk` and `transfer_gate` did not run.
  - And by all six open human decisions.
  - J7n's needle-control row `0b86fae3` is quick and has no `eval_row_id`, so the promotion join
    does not read it.

## What changed (commits)

| Commit | Change |
| --- | --- |
| `caac799` | `docs/promotion-decisions.json`: the open promotion questions as data a human edits. `PromotionVerdict.promotion_population` is read from it with its sha256. `Ledger.promotion_verdict_avg` and `verdict --kind avg`. Documented in `docs/ledger-schema.md`, section Promotion |
| `92c7a07` | `qd-gate-report`, a Rust bin in qd-runtime with no new crate, covering G1–G3 and the population. `python/tests/test_gate_report_parity.py` |
| `d8a0047` | `calib_fit_row.py --population cross-fit` (G7) |
| `a05490b` | the three cross-fit rows above |
| `0fa94d2` | on the lead's yes, the eval row also carries the report metrics: `real_ft_run.calibration_states` adds `degenerate_head.family.{family}.{shape}` (G1) and `ece.report.pooled` / `ece.report.lang.{lang\|none}` (G2), as metrics only, never in `eces` or `degenerate`. `qd-gate-report` cross-checks them where a row records them |
| `0603f7f` | `tools/gate_report_row.py`: a `qd-gate-report` run as its own quick ledger row, on the `calib_fit_row.py` pattern |
| `229efc4` | the four gate report rows above |

**G7, what was missing.** Each population was its own row. The `all` rows recorded in-sample
numbers, and no row named the shipped table or bound the two populations together.
`--population cross-fit` fixes that on one row:
- every number on the row is two-fold;
- `calib_fit.shipped_table_sha256` is the all-val table;
- the fold tables and both reports are bound by sha256.

`all` and `two-fold` are unchanged.

**G8.** An average is judged on four things:
- its own full gate row, with its supplements;
- non-quick inputs that ran their schedule to `steps_exhausted`, all of one recipe, snapshot,
  tokenizer and backbone;
- at least 3 distinct seeds, matching `recipe.averaged`;
- every open human question in the record, each of which refuses.

`test_an_average_is_not_promotable_under_the_current_decisions` passes an all-green average with
clean inputs and gets REFUSED, by the human items alone. Its positive control promotes the same
fixture once a test record decides every question.

**In-process metrics (`0fa94d2`), with the gates unchanged.** Since the lead's yes,
`calibration_states` returns the report metrics too. `qd-gate-report` stays the cross-check: it
recomputes them from the verdicts and refuses a 1e-9 difference.
- `test_real_ft_report_metrics.py` pins the sha256 of every gate, control and earlier metric. The
  golden was computed from the unmodified function at `90e7586`.
- On real data the same digest is identical before and after, on J4 s0 and on the average.

**The GDN path is not recorded here.** These report rows run no model. The training-step perf
lane owns the record that GDN ran through fla's `chunk_gated_delta_rule` with `causal_conv1d`,
and nothing here duplicates it.

## Tests

- Before the fix, `test_promotion_decisions.py` failed at import (the API did not exist) and the
  two cross-fit tests failed on argparse.
- After the fix:

  | Suite | Result |
  | --- | --- |
  | `test_promotion_decisions.py` | 30 passed |
  | `test_ledger.py`, `test_gaps_ledger.py` | 112 passed |
  | `test_gate_report_parity.py` | 11 passed (2 assertions fail against the pre-`0fa94d2` binary) |
  | `test_real_ft_report_metrics.py` | 5 passed (gate sha256 pinned at `90e7586`) |
  | `test_gate_report_row.py` | 5 passed (before `0603f7f`: `ModuleNotFoundError: gate_report_row`) |
  | `test_calib_fit_row.py` | 7 passed |
  | `test_calib_fit_parity.py` | 18 passed, 1 opt-in benchmark skipped |
  | `cargo test -p qd-runtime --bin qd-gate-report` | 5 passed |

- **Full CPU suite at `229efc4`:** 1 failed, 2945 passed, 48 skipped, 8 deselected, in 13 min.
  - Deselected by name: the `mps` tests and the two environmental tests below.
  - The one failure was `test_tool_call_sites.py::test_every_tool_that_hashes_a_recipe_is_found_by_this_check`.
    The new tool lacked its recipe-hash spelling pin.
  - The pin was added in the next commit, with the tight side's count 5 → 6. That file now
    passes 19 of 19.
- clippy `-D warnings` and rustfmt are clean.
- Mutation checks: a last-max argmax, and dropping the permuted half of the in-distribution rule,
  each fail the parity tests.
- Environmental, not this lane: `test_gaps_writer.py::test_the_real_ledger_is_not_touched_by_any_of_this`
  asserts the checkout directory is named `Lappi-decision`, which a worktree's is not.
- Cargo re-resolved the out-of-repo `tessl` path dependency into `Cargo.lock`, adding
  `dispatch2` and `block2`. That change was reverted and is in no commit.

## What is open (gap ids)

- New, human-owned: `GAP-AVERAGE-PROMOTION-IS-THE-HUMANS-DECISION`.
- Navigation: `GAP-FABLE-G-REPORT-METRICS-NAVIGATION-2026-10-01`. This worktree has no DevMap
  store, GitPulse needs trust, and there is no ListAgents.
- Unchanged and human-owned. Each is now a `decisions` entry in `docs/promotion-decisions.json`,
  and each still refuses any average:
  - `GAP-GATES-POOL-THE-GENERAL-FAMILIES-INTO-DEFECT-CONTRACTS`
  - `GAP-ECE-GATE-IS-NOT-RUN-ON-THE-FULL-MIXTURE`
  - `GAP-DEGENERATE-HEAD-FAILS-ON-BINARY-SLOTS-TOO`
  - `GAP-PRIVILEGED-HUNK-IS-NEARLY-VACUOUS-ON-COMMITPACKFT`
  - `GAP-TRANSFER-GATE-IS-NAMED-NOT-SPECIFIED-AND-HAS-NO-DATA`
- The average was re-reported (`c308c802`), so nothing for it is still to copy.
- `verdict --row-id` takes the full row id, not a prefix. `1d93b3ee` alone reports "0 rows have
  id"; `tools/*_row.py` resolve unique prefixes and the CLI does not. This is an observation
  only; nothing was changed.

## First command for the next lane

For any new eval row whose `--verdicts-out` file is on the Mac, write its report row (this works
the same for an average's verdicts). Replace `<…>` with that row's files:

    cargo build --release -p qd-runtime --bin qd-gate-report \
      --target-dir /Users/bharath/qd-campaign/target-reportmetrics
    git checkout -- Cargo.lock   # the tessl path dep re-resolves; never commit it
    /Users/bharath/.venvs/ml/bin/python tools/gate_report_row.py \
      --bin /Users/bharath/qd-campaign/target-reportmetrics/release/qd-gate-report \
      --verdicts <verdicts.jsonl> --suite-verdicts <suite-verdicts.jsonl> \
      --eval-ledger ledger/<eval ledger>.jsonl \
      --ledger ledger/mac-gate-report-<date>.jsonl --out-json <new report.json>

Then the promotion verdict on an average, with the full row id. For `1d93b3ee` it must print
`REFUSED [avg]`:

    PYTHONPATH=python python -m qd_train.ledger verdict --kind avg \
      --ledger ledger/gh200-p6-j7-avg-2026-10-01.jsonl \
      --row-id 1d93b3ee-28fc-4345-a02c-f757bcdb8466 \
      --input-ledger ledger/gh200-p4-v3-2026-10-01.jsonl
