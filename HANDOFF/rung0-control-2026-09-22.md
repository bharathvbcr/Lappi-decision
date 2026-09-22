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

### Operator holdout, minor operators

Rows 73–144 of the same ledger — 72 `ft` rows, 9 arms × 8 seeds, all `completed`, all
`quick: true`, $2.96, 7,158.1s summed. Launch rev `728562b3da609a3aa3ad35efaa597c870e7ba1fe`,
`code_that_ran` `7318cbb3…` (rows 1–72 ran `5bad419d…`; the report never mixes arms across
the two).

| operator | arm | row ids |
| --- | --- | --- |
| logic.negate_condition | reference | `271d5ca7 2c5544a0 460fcf95 8a40a0eb 9cb1e66d c43966d6 ed6263d5 ff539384` |
| logic.negate_condition | holdout | `4b2720e6 5ef0473a 75f6482f 892f41a2 cd91107f ec8498e2 edf9a24a ff730b2e` |
| logic.negate_condition | sizematch | `035fbd4e 0cc2e16c 2843c911 2e909b2f 4b62c8f9 b633aeea c75c18db d8ff2089` |
| cosmetic.edit_comment | reference | `0a4c1034 1318f5c2 257708b6 3584d6d9 53898012 58479d98 a1acec18 c7f82e77` |
| cosmetic.edit_comment | holdout | `048ced6e 06ea1994 454d66eb 48d7b280 72e09628 8d0945d6 bd2080ca d5f27c57` |
| cosmetic.edit_comment | sizematch | `2fcdcc2f 31d4cace 5e0661d2 6957f6ba 908e8f85 9c113297 c138b145 c5a149a4` |
| stub.default_return | reference | `2caba155 3ec41c8d 4cabd961 5393dd62 8f647e5e 9b90b51f a085be56 f404f3f1` |
| stub.default_return | holdout | `20b3723e 52a6ca7b 5704a4aa 7021faa5 767d6da5 9ecbb04e d4395045 fbfd06ef` |
| stub.default_return | sizematch | `0e7168a5 283e77cf 3551e4e7 384c87b5 51a580d9 aef7e896 c55e1af9 ecb2f1c6` |

**Two of three collapse, one transfers.** Holding out a minor operator barely moves the class
prior, so these are the sharper test. `logic.negate_condition` 88.91% → 45.56% with siblings
moving 0.40pp; `cosmetic.edit_comment` 78.56% → **1.00%** with siblings *rising* 4.73pp — an
operator under 5% of training, and the largest gap of all six. `stub.default_return` is the
exception: 81.69% → 76.71%, only 2.91pp worse than its siblings (5,448 of which are
`stub.panic`, still in training), inside the seed spread. Whatever the model learned from
`stub.panic` transfers to it. That bounds the headline — the shortcut is "whatever the
training generators share", not "one string per operator" — without rescuing the claim that
the model reads changes. Tables in the AUDIT.

The two reference arms that ran after the full-corpus control was cached,
`cosmetic.edit_comment` and `stub.default_return`, report **−7.91% and −8.09%, 0 of 8
positive** — and are the first end-to-end proof that `fit_linear_control.py` and the arm
compute the same cache key (both logs: `CACHE HIT 0a1b4a64b890460c`). The third,
`logic.negate_condition`, started at 17:25:25, four minutes before that fit landed, and has
no margin.

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
Everything since is on `main` too: that build branch was folded in and deleted at `728562b`,
and every later commit landed on `main` directly.

The last stretch, `52c8fe9`…:

- **`52c8fe9`** — the fit script kept a hardcoded operator list after the launcher learned
  `QD_HOLDOUT_OPERATORS`, so the minor operators could run as arms but could not be given
  controls. Both now source `tools/operator_holdout_operators.sh`. Executing that file in a
  test (rather than grepping it) found that `${VAR:-defaults}` sends a set-but-EMPTY override
  to the defaults, silently running the dominant three; it is `${VAR-defaults}` now. The
  size-matched fit reads N from its own holdout fit's log, not the arm's, so a control no
  longer waits for a GPU arm. `LEDGER` takes `QD_HOLDOUT_LEDGER`, so a re-run gets its own
  ledger. The "never fits inline" claim in both headers is corrected: the arm fits inline
  when the projection is under the 900s budget, and on this corpus no arm's was.
- **`692c154`** — `tools/operator_holdout_report.py` refuses to pool a repeated seed in one
  `(operator, condition)` cell (a re-run beside the run it repeats, or one ledger named
  twice), naming both rows; it prints each not_run margin's recorded reason instead of
  assuming one; its docstring no longer says `fit_linear_control.py` "has no holdout flags".
- **`b8b71f7`** — `GAP-LEDGER-ARMS-POOLS-A-REPEATED-SEED-AS-A-NEW-ONE`, the same defect in
  `tools/ledger_arms.py`, recorded and not ported (13 committed ledgers repeat seeds, some by
  design).

Before those, the substantive change was: **`tools/fit_operator_holdout_controls.sh` had
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

### 1. The 104 `not_run` margins — RE-RUN IN FLIGHT (run 3)

> **Status 2026-09-22 20:25 UTC:** run 3 launched — all 18 arms, six operators (dominant
> three first), at `52c8fe9fc39acbd5e3aa2b859c38637d2696a586`, `code_that_ran` `7318cbb3…`
> (sync verified both sides), writing to its **own** ledger
> `ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl`. Driver log
> `/home/ubuntu/ophold3-driver.log`, marker `OPERATOR HOLDOUT MODEL DONE`. ~18 arms × 8 seeds
> × ~90s ≈ 3.6 GPU-hours, ≈$5.40 at $1.49/h, single-GPU and under rule 4's line; the user
> approved closing these margins. Run-1/2 logs were copied to `/home/ubuntu/ophold-logs-run1/`
> first, because the re-run overwrites `ophold-<arm>.log`.
>
> Alongside it, on the CPU, the six minor-operator controls:
> `QD_HOLDOUT_OPERATORS="logic.negate_condition cosmetic.edit_comment stub.default_return"
> bash tools/fit_operator_holdout_controls.sh` → `/home/ubuntu/opctl2-driver.log`, marker
> `OPERATOR HOLDOUT CONTROLS DONE`. They must land before run 3 reaches those operators
> (~9 arms in, ~2h); at ~6 minutes a fit they will, with a wide margin.
>
> The first arm, `ref-stub.panic`, logged `CACHE HIT 0a1b4a64b890460c` and a control at
> 88.5%.
>
> Earlier: seven fits (not nine; nine is the arm count), 204.2s–351.7s each, **2,190.4s total
> = 36.5 minutes, ~$0.91**. All converged. The ~15h/$22 figure this file once carried was
> wrong by 17.6×; it came from `projected_fit_seconds`, which prices `max_iter` iterations.
> **Never quote that projection as a cost.**
>
> **The fits alone do not finish the job.** Rows 1–144 are final. `rung0_real_run.py` reads
> the cache once, before its seed loop; on a miss it fits inline only when the fit projects
> under its 900s budget, and on this corpus every arm projected 14.7–25.7 hours. So the arms
> must be re-run to pick the fits up, which is what run 3 is.
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
- **Warming the cache does not backfill written rows.** `rung0_real_run.py` reads the cache
  once, before its seed loop; on this corpus every miss recorded `not_run` (the inline-fit
  fallback only runs under a 900s projection). Rows are final. Getting margins means fitting
  the controls **and then re-running the arms**.

Measured cost of the first seven fits: **2,190.4s = 36.5 minutes, ~$0.91**. The re-run is
larger than first priced here (~2.4 GPU-hours for 9 arms) because it now covers all 18 arms:
~3.6 GPU-hours, ~$5.40. Still under rule 4's line.

### 2. Clearing `quick` on the 144 + 16 rows

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
- `GAP-LEDGER-ARMS-POOLS-A-REPEATED-SEED-AS-A-NEW-ONE` — `tools/ledger_arms.py` counts a
  repeated seed as a new one; 13 of 27 committed ledgers contain such repeats. Whether any
  reported number pooled them is open. Fixed for the operator-holdout report only.

### 4. Human-owned, explicitly not agent work

`needle_hunk_recall` (needs a contract decision on span→hunk mapping, and must **not** become
a gate wired to a guessed mapping), `privileged_hunk` (near-vacuous on this corpus),
`transfer_gate` (no data), `ood_abstain` (unspecified).

## The exact first command for the next lane

State as of 2026-09-22 20:30 UTC. Two jobs are live on the box. Check them before starting
anything:

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 'cat /home/ubuntu/ophold3-driver.log; tail -4 /home/ubuntu/opctl2-driver.log; ps -eo pid,etime,cmd | grep -E "fit_linear[_]control|rung0_real[_]run" | cut -c1-160'
```

- **GPU — run 3, ~3.6h from 20:25 UTC.** 18 arms at `52c8fe9`, marker
  `OPERATOR HOLDOUT MODEL DONE` in `ophold3-driver.log`, own ledger
  `gh200-operator-holdout-controlled-2026-09-22.jsonl`.
- **CPU — six minor-operator control fits**, marker `OPERATOR HOLDOUT CONTROLS DONE` in
  `opctl2-driver.log`. They slow the GPU arms while they run; that is expected.

**Before run 3 reaches `logic.negate_condition`** (its tenth arm), the six fits must be
cached. If they are not, stop the GPU driver by PID rather than let those arms train
without an opponent. `sync_box.sh` must not be run while either job is live; it refuses
anyway.

**Every arm of run 3 should log `CACHE HIT`.** A `linear control NOT RUN` from any of them
means a key disagreement or a fit that was not ready, and every arm after it wastes GPU time:

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 'grep -H -E "CACHE HIT|linear control NOT RUN" $(find /home/ubuntu -maxdepth 1 -name "ophold-*.log" -newermt "2026-09-22 20:25:00") | cut -c1-120'
```

**When `OPERATOR HOLDOUT MODEL DONE` appears**, pull run 3's ledger (never mid-append) and
read it **on its own** — the report refuses to pool it with the first ledger, because they
hold two runs of the same seeds:

```bash
scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240:/home/ubuntu/qwen-decision/ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl ledger/
```

```bash
python tools/operator_holdout_report.py ledger/gh200-operator-holdout-controlled-2026-09-22.jsonl
```

What to read, in order:

1. **Every cell should show `paired margin vs the control` and no `NO margin` line.** Any
   `NO margin` line carries the reason its run recorded.
2. **The holdout margins**, above all `stub.panic`: the control is at 50.0% overall there,
   and the model was at 51.74% in run 1. That is the first place rung 0 could win. Report
   the paired margin and how many of 8 seeds are positive, whichever way it goes.
3. **Run 3's model numbers against rows 1–144.** For the minor operators the code is
   byte-identical (`7318cbb3`), so the differences are run-to-run noise
   (`recipe.deterministic` is false). For the dominant three the code differs (`5bad419d` →
   `7318cbb3`), so it is a replicate across a code change; say which.

Then extend `AUDIT/operator-holdout-model.md` ("The comparison that is still missing") and
this file with the margins and row ids, and commit the ledger.

One footgun: on this box `pgrep -f` / `pkill -f` match the ssh command line that carries the
pattern, so a kill can hit your own session and leave the target orphaned. Resolve PIDs with
`ps` and `kill` by number.
