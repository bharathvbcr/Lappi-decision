# Rung 3 readiness — handoff, 2026-09-22 (written 22:40Z)

Lane: the real Qwen3.5-2B FT path (`tools/real_ft_run.py`), fed from the plan's commitpackft
download, measured against rows it never trained on. Written while two GPU jobs were still
running; §1 says how their results come home and what to do with them.

## 1. Running now — and the exact first command for the next lane

| where | what | ETA (UTC) | writes |
| --- | --- | --- | --- |
| GH200 `192.222.58.240` | run 3, the controlled operator-holdout re-run: 18 arms, 7 exited at 22:36Z, ~17.5 min/arm | ~01:50 | `/home/ubuntu/qwen-decision/ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl` |
| same box, chained | `/home/ubuntu/chain_ft_score_after_run3.sh` (pid 210214): once run 3 is done, the FT from the clean clone `/home/ubuntu/qd-48f994a`. Real tower, `--optimizer master`, 3 seeds, memorise + epoch arms, `--score-val` on `/home/ubuntu/shards-cpft-2000-val`. Caps: 1800 s/run, worst case $4.47 | ~02:35 | `/home/ubuntu/qd-48f994a/ledger/gh200-ft-commitpackft-2026-09-22.jsonl` |
| this Mac | `copy_down_when_done.sh` (session scratchpad): waits for both, copies both ledgers into `ledger/`, checksum-verifies, prints `SAFE TO DELETE` or `NOT SAFE` | after the box | `ledger/` (untracked) |

The chain was pre-checked on the box from that clone (CPU stand-in, scratch ledger outside the
clone): all four rows carry `code_commit` 48f994a clean, the val set opened (184 sequences, 145
batches, the train set's remap `1b5e9101…`), eval row `ff5de04b` written.

**Only those two ledgers exist nowhere but the box.** All 253 rows of the box's `runs.jsonl` are
on this Mac by row id (97 of them in `gh200-2026-09-21.jsonl`). The box is safe to delete once
both ledgers are here and verified. If the copy-down did not run (app closed), fetch them:

    scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240:/home/ubuntu/qwen-decision/ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl ubuntu@192.222.58.240:/home/ubuntu/qd-48f994a/ledger/gh200-ft-commitpackft-2026-09-22.jsonl ledger/

**First command for the next lane**, once the run-3 ledger is here:

    /Users/bharath/.venvs/ml/bin/python tools/operator_holdout_report.py ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl

Pass it **alone**. `cells_of` keys on (operator, condition) and not on the data snapshot
(GAP-OPERATOR-HOLDOUT-CELLS-OMIT-THE-DATA-SNAPSHOT), so first check that every row carries one
`data_snapshot_hash`. Then: every cell should now have `paired_margin_vs_linear` ran. Report the
stub.panic holdout margin as measured, whatever its sign. Update
`AUDIT/operator-holdout-model.md` ("That re-run is in flight") and commit the ledger.

Then the FT ledger. Each epoch seed wrote an `eval` row (`epoch-score-val`) with
`val_top1.{choice,score,span}` against input-blind baselines, `ece.*` and
`degenerate_head.*` per slot shape, gate `ece`, and control `degenerate_head`. These rows are
`quick`: the span rows are this repository's Markdown standing in for SQuAD. Rule 8: they
promote nothing. Commit the ledger with a summary that cites its row ids.

## 2. What was measured (committed ledger rows)

- `ledger/mac-remap-parity-2026-09-22.jsonl` `af64957d` — the remap parity gate on the real tower.
- `ledger/mac-shards-commitpackft-2026-09-22.jsonl`:
  - `0c3fd775`, `686d6c21`: shard builds from the download.
  - `d9b461ca`, `1a1f7d67`: remap coverage and byte-fallback cost. The train-only remap encodes
    5 of 184 val rows and 124 of 2,109 held-out rows.
  - `74dfe7b1`: the first set with a val split. Built with `--val-shards`: 184 of 184 val rows
    written, a 28,239-id remap. Its `remap_vocabulary` detail undercounts; dd2e958 fixes that
    for later rows.

Stand-in end-to-end checks of the scorer and the new gates are in **scratch** ledgers (Mac
eval rows `0e5fe554`, `43cfd5f7`; box `ff5de04b`). They verify plumbing, not a model, and are
not committed.

## 3. What changed (commits, this lane)

`a4bec9c` `1b3a435` `cbcbff6` `5d678b6` `3a2dce6` `dd2e958` — the shard pipeline reads the
download, counts what the remap cannot encode, prices a byte fallback, and writes val under a
remap that covers it. `48f994a` — `--score-val`. `b5b73fb` — `ece` / `degenerate_head` on the
FT eval row. Ledger and gap records: `b57f15b` `c52370d` `2eefea6` `3411dbd` `64052f4` `e7b4ee8`.
`8cd766b` swept another lane's staged files into its commit (shared index); commit
path-limited.

## 4. What is open

For a promotable final model, the FT eval row needs five gates `ran` and four controls
`ran`+`passed` (`qd_train.ledger.REQUIRED_GATES` / `REQUIRED_CONTROLS`). Where each stands on
the FT path:

| gate / control | state | blocked on |
| --- | --- | --- |
| `degenerate_head` | **ran** (b5b73fb) | — |
| `ece` | per-k ran; gate `not_run` | GAP-FT-ECE-HAS-NO-LANGUAGE-TO-SPLIT-BY (agent) |
| `permutation_consistency` | not built for FT | a render hook for a Sattolo second pass on top of each example's training shuffle. That is a qd_data edit, which invalidates every shard set through the fingerprint, so do it in the same change as the language metadata (agent) |
| `shuffled_label` | not built for FT | agent. Plan: permute gold-letter tokens (`tokens[r, target_index+1]`) within (kind, letter set) on the epoch plan only, skip arm 2, put the shuffle in the recipe, judge val with `eval_harness.shuffled_label_control` |
| `paired_margin_vs_linear` | not built for FT | a linear control over FT rows (agent, larger) |
| `privileged_hunk` | not built | a hunk-only rendering (agent, larger) |
| `transfer_gate` | not built | scoring the held-out families needs the remap decision (human) |
| `needle_hunk_recall` | built, uncallable | span→hunk mapping (human; GAP-NEEDLE-HUNK-RECALL-IS-BUILT-AND-UNCALLABLE) |
| `ood_abstain` | unspecified | human |

Human decisions that block the final train itself:

- GAP-REMAP-CANNOT-ENCODE-THE-ROWS-IT-WAS-NOT-BUILT-FROM. Options (a), (b) and (c); the byte
  fallback costs +4.31% tokens on held-out (row 74dfe7b1).
- GAP-FINAL-TRAIN-HAS-NO-MULTI-GPU-PATH: data-parallel or seed-parallel.
- GAP-TRAIN-CPT-HAS-NO-DRIVER: the CPT corpus. Note that `ShardReader` has no CPT format at all.
- GAP-MEMORY-REMAP-SAVING-RESTS-ON-SURGERY-VERIFIED-ONLY-AGAINST-STUBS.
- GAP-RT-CALIBRATION-NOT-FITTED: `row_logits` from b5b73fb are the fitter's input. The runtime
  has no model backend to serve a table yet.
- Downloading SQuAD / CLINC150: needs the user's permission. Every FT row stays `quick` until
  the span rows are real.
- Rule 4's yes for any 8×H100 hour.
- Whether to clear the `quick` flag on the 72 rows another lane asked about. Not decided;
  rule 2 makes it a human call.

## 5. Tooling notes

- This session's `devmap mcp` server holds deleted WAL/SHM files and errors
  (`database disk image is malformed`). The store is healthy (generation 2434 via the GitPulse
  plugin's `devmap_status`). Use `mcp__plugin_gitpulse_gitpulse__devmap_*`, or restart the
  session's server. See GAP-DEVMAP-NODES-FTS-CORRUPT-WHILE-STATUS-SAYS-ONLY-STALE.
- Do not `git -C` or `cd` in this harness, and do not redirect into the scratchpad. Box kills
  are by PID only (memory: pgrep/pkill match their own command line).
- `/home/ubuntu/qwen-decision` on the box is run 3's checkout (52c8fe9 closure). Do not sync it
  while run 3 runs.
