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

`ledger/gh200-span-in-diff-2026-09-22.jsonl` — 16 rows, 2 arms × 8 seeds, $0.69, 1,665.0s,
rev `084b7f1d69c9aeeb8ef2a351dcf8fb9a03f06171`, both arms exited 0. All `quick: true`.

**`--span-in-diff` is free for the choice head and buys nothing.** Choice: `nospan` 80.46%
± 1.61 vs `span` 79.83% ± 1.11 — `ledger_arms.py` puts the +0.63pp difference *inside* the
1.94/2.08pp floor at n=8, so no cost is demonstrated. Span: start 63.61% ± 6.72 against an
**added-line null of 89.91%** over 10,635 pointing rows. The head is 26.3 points worse than
pointing at a random `+` line without reading any code.

The two span gates read `passed: True` on all 8 seeds, because they score against the
uniform-over-candidate-lines rate of 8.44% and rule 2 makes a gate's baseline read-only.
`val_span_pointing_added_line_chance` sits beside them on every row saying to read against
the larger null. **Quote the 89.91%.** Full writeup:
[`AUDIT/span-in-diff.md`](../AUDIT/span-in-diff.md).

## What changed

15 commits earlier in this session, `a48ba3c`…`084b7f1`, now on `main` (another session
fast-forwarded `main` to that tip and deleted the build branch; the tree did not change).
This session's remaining work is on `build/operator-holdout-model-and-control-fit-cap`.

The substantive change of the last stretch: **`tools/fit_operator_holdout_controls.sh` had
a `timeout` that could not be reconciled with its own projection, and now refuses instead of
being killed.**

- **Correction, and it is against my own earlier claim in this file and in commit
  `ab61781`.** I wrote that the script wrapped a 101.4-minute fit in `timeout 3600` so "all
  nine fits were doomed before the first byte was read." **That was wrong.** The fits take
  204–352s. `timeout 3600` was never going to kill one. I reached that conclusion from
  `projected_fit_seconds` — the same worst-case bound that later produced the ~$22 estimate
  — and then killed a running fit on the strength of it, about six minutes in, which is
  roughly when it would have finished. The original script's cost header ("378.9s", "nine
  fits under an hour") was much closer to right than my correction to it.
- What survives is narrower and still worth having: a `timeout` and a tool that knew its own
  projection had no relationship to each other, so the 5fd0ea8 failure ("could not finish
  inside the cap it was launched under") remained possible on any corpus where the
  projection is not a 17× overestimate. That is now a precondition rather than a printed
  number. It was not, on this corpus, a live bug.
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

### 1. The 48 `not_run` margins — FITS DONE, RE-RUN STILL OWED

> **Status 2026-09-22 18:01 UTC:** the user approved the spend ("proceed") and the fits are
> **complete** — `OPERATOR HOLDOUT CONTROLS DONE`. Seven fits (not nine; nine is the arm
> count), 204.2s–351.7s each, **2,190.4s total = 36.5 minutes, ~$0.91**. All converged.
>
> The ~15h/$22 figure this file and four other places carried was wrong by 17.6×. It came
> from `projected_fit_seconds`, which prices `max_iter` iterations; the fits converge at
> `tol` in ~440. **Never quote that projection as a cost.** The `QD_CONTROL_FIT_ACK` gate
> built around it has been removed — it was friction guarding a cost that did not exist.
>
> **The fits do not finish the job.** The 72 rows are final; `rung0_real_run.py` reads the
> cache once before its seed loop and never fits inline. The nine arms must be **re-run**
> to pick the fits up — see the last command in this file. Until then nothing about the
> margin has changed.
>
> Control accuracies now cached (overall, same 12,792 val rows): full 88.5%; holdout
> stub.panic **50.0%**, logic.change_constant 80.2%, cosmetic.rename_local 82.9%; all three
> sizematch arms 88.5–88.6%. Against the model's 51.74% / 71.48% / 76.70%, the `stub.panic`
> holdout arm is the first where the model's number is larger — **but that is a difference
> of means across two tools, not a margin**, and it sits inside the ±2.07 seed spread.

The rest of this item is the reasoning, kept because the re-run depends on it.

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

Measured cost, now that it has been paid: **7 fits, 2,190.4s = 36.5 minutes, ~$0.91**, then
~2.4 GPU-hours (~$3.60) to re-run the 9 arms. Well under the $20 line rule 4 draws, so the
re-run needs no further yes.

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

State as of 2026-09-22 18:05 UTC. Check the box before starting anything:

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 'tail -3 /home/ubuntu/opctl-driver.log; tail -3 /home/ubuntu/ophold2-driver.log; ps -eo pid,etime,cmd | grep -E "fit_linear[_]control|rung0_real[_]run" | grep -v grep'
```

- **CPU — DONE.** The seven control fits finished in 36.5 minutes; all seven verdicts are
  cached in `/home/ubuntu/control-cache`.
- **GPU — live, ~2h left.** Three *minor*-operator holdout arms (`logic.negate_condition`,
  `cosmetic.edit_comment`, `stub.default_return`) at rev
  `728562b3da609a3aa3ad35efaa597c870e7ba1fe`, marker `OPERATOR HOLDOUT MODEL DONE`, driver
  log `ophold2-driver.log`. They append to the **same** ledger as the first nine, so one
  report renders all six operators. ~$2.40, single-GPU, under rule 4's line.

Why minor operators: the first three were the dominant operator in each class (42.8%, 14.2%,
8.9%), which moves the class prior hardest and makes the sibling metric work hardest to rule
it out. These three each leave their class's dominant operator in training, so the prior
barely moves and a collapse is harder to explain any way but the generator.

While the fits were running the two jobs contended — the fits own all 64 cores, and arm
seeds ran at 215s against 93s uncontended. The fits are done, so the remaining arms run at
full speed.

**When `OPERATOR HOLDOUT MODEL DONE` appears** (the GPU job, first to finish), pull and read:

```bash
scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240:/home/ubuntu/qwen-decision/ledger/gh200-operator-holdout-model-2026-09-22.jsonl ledger/
```

```bash
python tools/operator_holdout_report.py ledger/gh200-operator-holdout-model-2026-09-22.jsonl
```

**When `OPERATOR HOLDOUT CONTROLS DONE` appears**, the fits are cached but no row has gained
a margin yet — `rung0_real_run.py` reads the cache once before its seed loop and never fits
inline. Sync, then re-run the arms so they pick the fits up. **`sync_box.sh` refuses while
training is live, and that refusal must not be bypassed**; wait for the GPU job first.

```bash
bash tools/sync_box.sh
```

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 'setsid nohup bash /home/ubuntu/qwen-decision/tools/launch_operator_holdout_model.sh <40-char-sha-of-the-synced-HEAD> > /home/ubuntu/ophold3-driver.log 2>&1 < /dev/null &'
```

Only that last run produces `paired_margin_vs_linear` on holdout and sizematch arms. It is
the first configuration in this project where rung 0 could beat the control.

One footgun: on this box `pgrep -f` / `pkill -f` match the ssh command line that carries the
pattern, so a kill can hit your own session and leave the target orphaned. Resolve PIDs with
`ps` and `kill` by number.
