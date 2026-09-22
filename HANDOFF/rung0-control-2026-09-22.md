# rung0-control — 2026-09-22

Lane: the char-n-gram control and the commitpackft pool. This session answered the question
`AUDIT/operator-holdout.md` left open, and found a defect in the tooling meant to finish it.

## What was measured

### Operator holdout, against the model (the headline)

`ledger/gh200-operator-holdout-model-2026-09-22.jsonl` — 72 `ft` rows, 9 arms × 8 seeds, all
`status: completed`, all `quick: true`. Written on 1×GH200, $2.39, 5,781.4s summed.
Launch rev `ecbd370a4d40a6ddc3d308b389cc1112a5e3cd30`.

Row ids by arm (first 8 hex of each seed's `row_id`):

| operator | arm | row ids |
| --- | --- | --- |
| stub.panic | reference | `0eef4b10 3bfad18d 4e53ef89 5678c451 7bb662e7 8e6c21f4 a8734210 a9138687` |
| stub.panic | holdout | `01a57707 4183ef1d 559d6f1f 8ade4f76 8dd9e815 95e08162 a7048802 e641cb44` |
| stub.panic | sizematch | `1770bae5 1c34ad11 2f9e702b 3cfd289c 6627fdf0 98dfb048 a73a647f e92cf255` |
| logic.change_constant | reference | `0c2e2d94 1a2ff175 52f2f230 67a9a2b7 77d245ac 86c42c89 be97a812 f527fe48` |
| logic.change_constant | holdout | `00f976c1 89f3a98f 9671ab40 9d404076 a6777226 b035ae27 b50a918e ebc9fc6f` |
| logic.change_constant | sizematch | `239e4c16 3fc696f4 7508e8b1 81ef92b0 bc84806d ce9353da e9e414d5 f6d62a85` |
| cosmetic.rename_local | reference | `186b543d 2b15c9ce 46140277 4d5e3141 b4a8b6cd dda80fa5 e05bc378 fbe3d266` |
| cosmetic.rename_local | holdout | `3bf953ee 4bd84fbf 5ba37e5e 6805b9b4 74faf2fd a838c3a3 ca37f32b f478ad14` |
| cosmetic.rename_local | sizematch | `160fa607 6ad3f1c6 7a1070d6 7ecca3c0 8cccabc3 948f86ef c2344ea0 f440e6cd` |

**Finding: the model classifies by recognising the mutation operator, not by reading the
change.** All three operators agree. Held out, an operator's own validation rows fall
58–77pp while its same-class siblings hold or rise, and dropping the same number of rows at
random moves the score by at most 1.59pp. Full tables, the three controls that rule out
volume / class-prior / language, and the provenance are in
[`AUDIT/operator-holdout-model.md`](../AUDIT/operator-holdout-model.md).

The reference arms reproduce the capacity grid's best cell across days and code versions:
−7.94%, −7.91%, −8.12% paired margin, **0 of 8 seeds positive in every arm**. The margin is
real and stable; what this session establishes is that it is not about architecture.

### Span-in-diff

`ledger/gh200-span-in-diff-2026-09-22.jsonl` — **in flight at handoff time**, 2 arms × 8
seeds, launched at rev `084b7f1d69c9aeeb8ef2a351dcf8fb9a03f06171`. Driver log
`/home/ubuntu/spandiff-driver.log` on the box; done marker `SPAN IN DIFF DONE`. Do not cite
any number from it until the marker is present and the ledger is copied down.

## What changed

15 commits earlier in this session, `a48ba3c`…`084b7f1`, now on `main` (another session
fast-forwarded `main` to that tip and deleted the build branch; the tree did not change).
This session's remaining work is on `build/operator-holdout-model-and-control-fit-cap`.

The substantive change of the last stretch: **`tools/fit_operator_holdout_controls.sh` was
unrunnable as written and is now fixed.**

- It wrapped each fit in `timeout 3600` while the fit projects to **101.4 minutes**. All
  nine fits were doomed before the first byte was read. The script's header asserted the
  opposite ("378.9s", "nine fits … under an hour of CPU"); both claims were false and are
  now replaced with measured numbers.
- The fit is **not** single-threaded, whatever `qd_train/control_cache.py`'s docstring says.
  One fit drove load average to **64.8 on a 64-core box** — it is a dense BLAS GEMM. So the
  fits cannot be overlapped to buy wall clock, and they are not free of the GPU either: a
  span arm's seed went from ~92s to **140.5s** while one fit ran beside it.
- `tools/fit_linear_control.py` gained `--max-fit-minutes` and `fit_budget_refusal()`. It
  computed the projection and printed it, then started anyway; now it refuses up front.
  `FIT_CAP_MIN` in the shell script feeds both `timeout` and `--max-fit-minutes`, so the two
  cannot drift.
- Commit `5fd0ea8` ("The linear control could not finish inside the cap it was launched
  under") had already recorded this failure once. It recurred because nothing enforced the
  precondition. It is now enforced, with `python/tests/test_fit_budget_refusal.py` — 9 tests,
  all 9 verified failing against the pre-fix files and passing after.

## What is open

### 1. The 48 `not_run` margins — the one result that could still overturn the headline

Every holdout and size-matched arm carries `paired_margin_vs_linear: not_run`. **That is not
a loss.** No control was fitted on those reduced training sets, so the model had no opponent.

Under holdout the control collapses to 3.08% on `stub.panic` while the model holds 15.44%.
Those come from different tools on different splits and cannot be subtracted. Fitted
properly, this would be the first configuration in this project where rung 0 wins.

**Two things make this expensive, and both are human calls:**

- **The cache is cold.** Six fits sit in `/home/ubuntu/control-cache` and none can be
  served. `control_key` folds in a sha256 of `python/qd_train/baseline.py`, deliberately, so
  an edit can never serve a verdict from code that no longer exists. Commit `20cc7f5` moved
  `context_texts` into that file. The move was a relocation; the digest cannot see intent,
  sees different bytes, and fails closed. **That is what the digest is for — do not weaken
  it to reclaim the six entries.** The cost of rebuilding is the price of the guarantee.
- **Warming the cache does not backfill the 72 rows.** `rung0_real_run.py` reads the cache
  once, before its seed loop, and records `not_run` on a miss; it never fits inline. The 72
  rows are final. Getting margins means fitting the controls **and then re-running the arms**.

Measured cost: 9 fits × ~101 min ≈ **15 hours of the box (~$22)**, then ~2.4 GPU-hours
(~$3.60) to re-run the 9 arms. The fit script now refuses to start without
`QD_CONTROL_FIT_ACK=1` because that spend crosses the $20 line rule 4 draws.

### 2. Clearing `quick` on the 72 rows

Rule 8: they promote nothing as recorded. The recorded reason says no rule-8 condition
actually stands — 8 seeds, full schedule, no subsample, real labels — and that clearing the
flag is a promotion decision rule 2 reserves for a human. The finding is readable as a
diagnostic either way; only promotion needs the flag cleared.

### 3. Gap ids

- `GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD` — the span arms in flight record
  `val_span_pointing_added_line_chance` beside the span gates; read the arm against the
  larger of the two nulls (the added-line null is ~90.8% on v1, 85.6% on the v2 sample).
- `GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN` — `code_commit` on these rows is a dirty sha
  from an older commit. `recipe.rev` pins the launch; `what_ran_state` pins the bytes.
- `GAP-A-LEDGER-ROW-CITED-A-COMMIT-NO-REF-COULD-REACH` — opened by another lane, carried
  here. Of 1,343 ledger rows with `code_commit`, one cited a commit reachable from no ref:
  `ledger/runs.jsonl:173` → `4e99e1a`, **`quick: false`**, so a row that can promote. It is
  an orphan of the deleted `build/s2-s4-trainer-wire-contract` lane. Verified in this
  session: the object exists, and `git for-each-ref --contains` returns exactly one ref,
  `refs/tags/provenance/runs-173`. **Do not delete that tag** — it is the only thing
  keeping the object out of `git gc --prune`.

### 4. Human-owned, explicitly not agent work

`needle_hunk_recall` (needs a contract decision on span→hunk mapping, and must **not** become
a gate wired to a guessed mapping), `privileged_hunk` (near-vacuous on this corpus),
`transfer_gate` (no data), `ood_abstain` (unspecified).

## The exact first command for the next lane

Wait for the span arms, then copy the ledger down and read it against the added-line null:

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 'grep -c "SPAN IN DIFF DONE" /home/ubuntu/spandiff-driver.log'
```

When that prints `1`:

```bash
scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240:/home/ubuntu/qwen-decision/ledger/gh200-span-in-diff-2026-09-22.jsonl ledger/
```

Only if the human accepts the ~15-hour / ~$22 spend in item 1, and **only when no training
is live on the box**:

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 'QD_CONTROL_FIT_ACK=1 nohup bash /home/ubuntu/qwen-decision/tools/fit_operator_holdout_controls.sh > /home/ubuntu/opctl-driver.log 2>&1 &'
```

and after `OPERATOR HOLDOUT CONTROLS DONE`, re-run the arms to pick the fits up:

```bash
bash tools/launch_operator_holdout_model.sh <40-char-sha-of-the-synced-HEAD>
```
