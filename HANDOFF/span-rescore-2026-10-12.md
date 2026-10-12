# HANDOFF span re-score, 2026-10-12 (lane w3-rescore)

State: **A not run, B not run.** No heavy job launched, no ledger row written, no number measured by this lane.
Labels: [V] read or run this session.

## Why
- [V] Head selection undetermined: of the pre-registered arms only (a) ran (5,596/7,238 gate pass, `build/needle-span-mac-2026-10-06/results/span-refit-score.json`, a `--gate-only` run with no ledger row). Arms b-e were never scored; avg-train extraction is at 11,896/12,000 rows (MPS OOMs at 8192 batch tokens); no seed-1 val cache.
- [V] A blocked: main's `qd_data` differs from the val shard's code fingerprint (11 modules + `caller_records.py`), so it is refused and `tools/real_ft_run.py` has no allow_stale_code. The pin tree reads the shard but lacks the always-written logit/gold fields (a1b1799). GAP-RESCORE-VAL-SHARD-STALE-VS-LOGIT-WRITING-CODE-2026-10-12.
- [V] B: `span_refit_score.py` only drafts ledger rows; no committed recorder exists, so arm numbers would not cite a row (rule 5). GAP-SPAN-REFIT-ARMS-B-E-UNSCORED-NO-LEDGER-RECORDER-2026-10-12. A refit head also has no export/scoring mode (GAP-QDM-REFIT-HEAD-HAS-NO-EXPORT-MODE-2026-10-12).

## Where the span entry would go
Not produced. When it is: `qd-calib-fit --population two-fold` via `tools/calib_fit_row.py`; the entry goes into the release table only by the human.

## Open decisions for the lead
1. allow_stale_code ruling or regenerated val shard (or a pin tree with a1b1799).
2. A committed ledger recorder for the arm rows.

## First command for the next lane
Resume the pre-registered chain (memory choice: `--batch-tokens 4096 --max-rows 16`):
`SPAN_CODE_REPO=/Users/bharath/Code/research/Lappi-decision/build/span-pin-8a531e8 bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh span-extract-avg-train-r4 /Users/bharath/.venvs/ml/bin/python /Users/bharath/Code/research/Lappi-decision/build/needle-span-mac-2026-10-06/span_refit_extract.py --tower avg --split train --device mps --batch-tokens 4096 --max-rows 16`
Full sequence: `scratchpad/lanes/w3-rescore-plan.md`.
