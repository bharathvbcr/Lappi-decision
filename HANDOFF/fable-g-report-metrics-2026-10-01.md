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
- **The gate report** (`qd-gate-report` at `92c7a07`) re-read J4's eval rows `2f5fe57a`,
  `f3612f73` and `02cf5ff4` from their local verdict files (sha256 `783e38be…`, `0f956a83…`,
  `f4923587…`) and J4's suite verdicts. It writes no row of its own: its numbers are those rows'
  verdicts, and on each row 52 recorded numbers were recomputed and equal before anything was
  reported. Per seed, s0 / s1 / s2:

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

## What changed (commits)

| Commit | Change |
| --- | --- |
| `caac799` | `docs/promotion-decisions.json`: the open promotion questions as data a human edits. `PromotionVerdict.promotion_population` is read from it with its sha256. `Ledger.promotion_verdict_avg` and `verdict --kind avg`. Documented in `docs/ledger-schema.md`, section Promotion |
| `92c7a07` | `qd-gate-report`, a Rust bin in qd-runtime with no new crate, covering G1–G3 and the population. `python/tests/test_gate_report_parity.py` |
| `d8a0047` | `calib_fit_row.py --population cross-fit` (G7) |
| `a05490b` | the three cross-fit rows above |

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

**Not done, by choice.** No in-process metric was added to `real_ft_run.py`. The new numbers
have one owner, the offline reporter, and the box writes `--verdicts-out` on every run, so any
future eval row can be re-reported. Mirroring them in-process would be a second implementation.

## Tests

- Before the fix, `test_promotion_decisions.py` failed at import (the API did not exist) and the
  two cross-fit tests failed on argparse.
- After the fix:

  | Suite | Result |
  | --- | --- |
  | `test_promotion_decisions.py` | 30 passed |
  | `test_ledger.py`, `test_gaps_ledger.py` | 112 passed |
  | `test_gate_report_parity.py` | 10 passed |
  | `test_calib_fit_row.py` | 7 passed |
  | `test_calib_fit_parity.py` | 18 passed, 1 opt-in benchmark skipped |
  | `cargo test -p qd-runtime --bin qd-gate-report` | 5 passed |

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
- **The average is not re-reported.** Its files are not on the Mac. Copy these from the box;
  the paths are J7's planned ones, so confirm them first with
  `grep -n -- '--verdicts-out\|--suite-verdicts-out\|--ledger' /home/ubuntu/box_q_j7g.sh /home/ubuntu/box_j7.sh`:
  - `/home/ubuntu/ledger/gh200-p6-j7-avg-2026-10-01.jsonl`, the average's eval row and any
    supplement;
  - `/home/ubuntu/j7/avg-verdicts.jsonl`;
  - `/home/ubuntu/j7/avg-suite-verdicts.jsonl`.

## First command for the next lane

Once the three files are in `/Users/bharath/qd-campaign/p4-v3-2026-10-01/`:

    cargo build --release -p qd-runtime --bin qd-gate-report
    target/release/qd-gate-report \
      --verdicts /Users/bharath/qd-campaign/p4-v3-2026-10-01/avg-verdicts.jsonl \
      --suite-verdicts /Users/bharath/qd-campaign/p4-v3-2026-10-01/avg-suite-verdicts.jsonl \
      --eval-ledger /Users/bharath/qd-campaign/p4-v3-2026-10-01/gh200-p6-j7-avg-2026-10-01.jsonl \
      --decisions docs/promotion-decisions.json

Then the promotion verdict on the average. It must print `REFUSED [avg]`:

    PYTHONPATH=python python -m qd_train.ledger verdict --kind avg \
      --ledger /Users/bharath/qd-campaign/p4-v3-2026-10-01/gh200-p6-j7-avg-2026-10-01.jsonl \
      --row-id <the avg-score-val row> --input-ledger ledger/gh200-p4-v3-2026-10-01.jsonl
