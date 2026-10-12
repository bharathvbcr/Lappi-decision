# HANDOFF span re-score, 2026-10-12 (lane w3-rescore)

State: **A2 refused (pairing), A3 not run, B partly run: avg-train extraction complete, seed-1 val extraction 4,811 of 7,238 rows, fits and scoring not run.** No ledger row came from this lane's heavy jobs: there is no row id and no measured number to cite. Labels: [V] read or run.

## Committed
- A0: branch `span-pin-8a531e8-verdicts` in `build/span-pin-8a531e8`, one commit (a1b1799's `tools/real_ft_run.py` hunk; `git apply --check` exit 0), local only.
- A1: `ledger/h100x2-v5-2026-10-03.jsonl` (sha256 8e651fae...) and `ledger/h100x2-v5avg-2026-10-05.jsonl` (2248a388...), byte-identical, commit 72119a6.
- B0: `tools/span_refit_rows.py` + `python/tests/test_span_refit_rows.py` (red first, then 6 passed): 3fa3785, e723043. The four pinned scripts are tracked as `tools/span_refit_{common,extract,fit,score}.py` (origins in e723043's message).
- qd-prep for the label rebuild: built from a clean 142a67c worktree (`build/qd-prep-v5-142a67c`), sha256 `729d40a208db6820c99dd5f578c0b9fe2f86d9d519955bfedfb17c427a0a6f05`, `cargo build --release --offline` (`--locked` refused: 142a67c's Cargo.lock needs 97 added lines).

## A2: refused, not worked around
Command: `build/w3-rescore/a2.sh` (section 6; `--batch-tokens 9638` per the lead's ruling, the tool's floor; real_ft_run has no `--max-rows`). Rationale: scoring builds no optimizer (a76a9f8), the 52 GiB footprint guard stops an overrun, and the span tolerance [5,581, 5,611] checks that batch composition did not change what is scored. The batch choice is recorded here because the run wrote no row. At 9638 the split rebuild, `span_check` (177,240/177,240) and the remap passed; then the pin refused, verbatim:

> --score-checkpoint v5-avg-s0-4.safetensors: input /home/ubuntu/ckpt/v5/epoch-seed0-cuda.json (seed 0) is not on disk at /home/ubuntu/ckpt/v5/epoch-seed0-cuda.json, so its payload_digest cannot be checked against the manifest's. Refusing: copy that checkpoint's JSON body there (its sidecar is never read), or score where it is.

GAP-RESCORE-A2-AVERAGE-INPUT-JSONS-ABSENT-PAIRING-REFUSAL-2026-10-12. (Earlier: GAP-RESCORE-A2-REFUSED-BATCH-TOKENS-BELOW-WIDEST-BUCKET-2026-10-12, resolved by the 9638 ruling.) Logs: `build/w3-rescore/a2-attempt*.log`.

## B: partly run
`build/w3-rescore/b.sh` (steps: seed-1 val extraction, fit c, fit e; the avg-train remainder finished: 12,000 rows, exit 0 at 14:57Z). The seed-1 extraction was killed by wrapper exit 143 three times (one at batch 321/652, 23 min in) and once more with a detached (setsid) launch, 28 s in; no footprint-guard line. Each death falls within about 6 s of another lane's `mac_heavy` start, sender not identified [U]. GAP-RESCORE-B1-EXTRACT-WRAPPER-SIGTERMED-TWICE-2026-10-12. The seed-1 cache resumes (4,811 rows). No arm row, no (e)-vs-(c) reading.

## Pool symlinks (untracked, provisioning only)
Listed with target sha256 in `build/w3-rescore/symlinks.txt`. Main's five (`noul-v5-allowlist.json` a0f1c220..., `commitpackft-composed-v2/examples.jsonl` 6cad5948..., `defect-noul-v3b/examples.jsonl` b194f14c..., `commitpackft-g6-v1/examples.jsonl` ff277d02..., `own-prose-v1/units.jsonl` 1e616fd1..., all targets under `/Users/bharath/qd-campaign/v5-data-2026-10-02/data/pool/`) are **removed** from main's checkout. The pin worktree's 21 links (gitignored there) remain; they point at main's data/pool and at the same v5-data files, and are removed with the worktree.

## Open
The four gaps above plus GAP-RESCORE-VAL-SHARD-STALE-VS-LOGIT-WRITING-CODE-2026-10-12 and GAP-SPAN-REFIT-ARMS-B-E-UNSCORED-NO-LEDGER-RECORDER-2026-10-12. Span entry not produced; release table untouched.

## First command for the next lane
B, when no other lane starts a mac_heavy job: `/usr/bin/python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' bash /Users/bharath/Code/research/Lappi-decision/build/w3-rescore/b.sh > /Users/bharath/Code/research/Lappi-decision/build/w3-rescore/b.log 2>&1 &`; then score: `python tools/span_refit_rows.py --eval-ledger ledger/h100x2-v5avg-2026-10-05.jsonl --results-dir RESULTS --head-c build/needle-span-mac-2026-10-06/heads/arm-c --head-e .../heads/arm-e --seed1-val-cache .../cache/seed1-val --val-cache .../cache/avg-val` under `mac_heavy.sh`. A2: needs the pairing decision above.
