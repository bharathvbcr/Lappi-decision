# HANDOFF span re-score, 2026-10-12 (lane w3-rescore)

State: **A2 refused by the pin, A3 not run, B1-B5 not run (extraction wrapper SIGTERMed twice).** No ledger row was written by this lane's heavy jobs, so there is no row id and no measured number to cite. Labels: [V] read or run.

## Done (committed)
- A0 [V]: branch `span-pin-8a531e8-verdicts` in `build/span-pin-8a531e8`, one commit (a1b1799's `tools/real_ft_run.py` hunk; `git apply --check` exit 0). Local only.
- A1 [V]: `ledger/h100x2-v5-2026-10-03.jsonl` (sha256 8e651fae...) and `ledger/h100x2-v5avg-2026-10-05.jsonl` (2248a388...), byte-identical to the campaign copies, commit 72119a6.
- B0 [V]: `tools/span_refit_rows.py` + `python/tests/test_span_refit_rows.py` (red first: module absent; then 6 passed), commits 3fa3785, e723043. The four scripts the chain pins are tracked as `tools/span_refit_{common,extract,fit,score}.py` (origin build/, original sha256s in e723043's message; score gained `--arms`).
- qd-prep for the label rebuild [V]: built from a clean 142a67c worktree (`build/qd-prep-v5-142a67c`), sha256 `729d40a208db6820c99dd5f578c0b9fe2f86d9d519955bfedfb17c427a0a6f05`, `cargo build --release --offline` because `--locked` refused (142a67c's Cargo.lock needs 97 added lines). Canaries equal in the run.

## A2: refused, not worked around
Run via `build/w3-rescore/a2.sh` (the section-6 command). Got through the split rebuild, `pair_labels` and the val set (24,492 sequences, split cache stored `build/split-cache-ft/d3fcfacf...rows.pkl`), then the pin said: `--batch-tokens 4096 is outside [9638, 1048576]: the widest bucket (9638) must fit in one batch`. GAP-RESCORE-A2-REFUSED-BATCH-TOKENS-BELOW-WIDEST-BUCKET-2026-10-12. The ruling's 4096 cannot be used; a value >= 9638 against the 52 GiB footprint cap (prior MPS OOMs at 8192 in extraction) needs a ruling. Earlier attempts (logs `build/w3-rescore/a2-attempt{1..5}*.log`) failed on absent data, now provisioned as untracked symlinks (details in the gap): composed-v2 `examples.jsonl` was not on main (sha 6cad5948... checked by the loader against the tracked manifest).

## B: not run
`build/w3-rescore/b.sh` (extract avg-train remainder, seed-1 val, fit c, fit e). The first step's `mac_heavy` wrapper exited 143 (its TERM trap, signalled from outside, not the footprint guard) about 30 s in, twice. GAP-RESCORE-B1-EXTRACT-WRAPPER-SIGTERMED-TWICE-2026-10-12. avg-train cache still 11,896/12,000 rows. Arms b-e, the arm rows and the e-vs-c reading do not exist.

## Open
Both gaps above, plus GAP-RESCORE-VAL-SHARD-STALE-VS-LOGIT-WRITING-CODE-2026-10-12 (mechanism built, no row) and GAP-SPAN-REFIT-ARMS-B-E-UNSCORED-NO-LEDGER-RECORDER-2026-10-12 (recorder built, no rows). Span entry: not produced; release table untouched.

## First command for the next lane
Rule on A2's batch tokens (>= 9638), edit `--batch-tokens` in `build/w3-rescore/a2.sh`, then `bash /Users/bharath/Code/research/Lappi-decision/build/w3-rescore/a2.sh`; the split cache is warm. B: `bash /Users/bharath/Code/research/Lappi-decision/build/w3-rescore/b.sh` once nothing is signalling the wrapper.
