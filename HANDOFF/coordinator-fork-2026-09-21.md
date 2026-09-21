# HANDOFF — coordinator-fork — 2026-09-21

A second lane in the same worktree as `HANDOFF/coordinator-2026-09-21.md`. That one ran the
audit and the engineering hardening; this one held the GH200 and ran the experiments. Where
the two overlap the other lane's record is the one to read for the fixes, and this one for
what was measured and what it cost.

**One sentence:** the capacity lever is not visible at n=8 and the corpus lever is —
partially — and every number here comes from a ledger row rather than from a log, which is
new today and is most of what changed.

---

## 1. The capacity question is answered, and the answer is "not this"

Six arms on the 4096 window, 8 seeds each, all 48 rows in
`ledger/gh200-rung0-capacity-4096-{e10,e30}-2026-09-21.jsonl` (chains verified, committed at
`7f191bd` — they existed only on a rented box until then).

**At 30 epochs every seed fit and every arm generalised worse than a constant predictor.**

| backbone | n | val mean | vs baseline | sd | collapsed |
| --- | --- | --- | --- | --- | --- |
| `128x4:2layer:ctx4096` | 8 | 38.36% | **−10.04pp** | 4.41pp | 0 |
| `256x8:4layer:ctx4096` | 8 | 36.08% | **−12.32pp** | 5.58pp | 0 |
| `512x8:6layer:ctx4096` | 8 | 40.57% | **−7.83pp** | 3.89pp | 0 |

Baseline 48.4%. Each arm's own distance from it clears that arm's one-sample floor
(3.85–5.53pp), so *below a constant predictor* is resolved three times over. **No difference
between arms is:**

| comparison | difference | pooled sd | floor at n=8 | verdict |
| --- | --- | --- | --- | --- |
| 256 − 128 | −2.28pp | 5.03pp | 7.05pp | inside |
| 512 − 128 | +2.20pp | 4.16pp | 5.82pp | inside |
| 512 − 256 | +4.48pp | 4.81pp | 6.74pp | inside |

At 10 epochs the picture is different for a different reason: `256x4` collapsed on 6 of 8
seeds and `512x6` on 7 of 8. **A collapsed arm's held-out mean sits *at* the baseline,
because a constant predictor scores the baseline**, and its sd is tight (1.11pp, 0.67pp
against 4.41–5.58pp at e30) because a constant predictor has almost nothing to vary. Read
without the collapse counts those arms look like near-misses with excellent precision. They
are neither. I reported "capacity helps +2.59pp" from them this morning, from a log that
printed the warning beside the number.

---

## 2. The learning curve rises, and only its top point clears its own floor

`ledger/gh200-rung0-learning-curve-2026-09-21.jsonl`, 24 rows, chain verified, committed at
`afa7c1f` with `AUDIT/rung0-learning-curve-2026-09-21.json`.

The validation side is never subsampled: every point is scored on the same **288 held-out
decisions from 32 files** against the same **49.0%** majority-class baseline, which makes
each point's comparison one against a *fixed reference* rather than against another noisy
arm. Training files are ordered by hash, so the 25% set is a strict subset of the 50% set.

| `train_subsample` | recipe | n | val mean | vs baseline | realised sd | floor | resolved? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | `918c73733b` | 8 | 49.48% | +0.48pp | 0.96pp | 0.96pp | no — **3 of 8 collapsed** |
| 0.25 *fitted* | | 5 | 49.79% | +0.79pp | 1.14pp | 1.43pp | no |
| 0.5 | `64b61aea47` | 8 | 50.56% | +1.56pp | 2.39pp | 2.37pp | no |
| 1.0 | `2b8564f63c` | 8 | 52.78% | **+3.78pp** | 2.89pp | 2.86pp | **yes** |

**What is not established.** No *adjacent* pair is separated — 0.5 − 0.25 is +0.77pp against
a 3.32pp floor, 1.0 − 0.5 is +2.21pp against 3.71pp, 1.0 − 0.25 is +2.99pp against 3.89pp,
all with pooled sd and the √2 two-arm floor. The **shape** is monotone in the direction
more-data-helps and **no single step of it is individually resolved at n=8**. No trend test
was pre-registered and none is applied; reaching for one after reading three ordered points
is how a shape becomes a finding it has not earned.

**Collapse is itself a size effect, with a confound.** Three of eight seeds at 25% never
cleared the training majority, against zero at both larger sizes. But that quarter also has
the most skewed training labels of the three (51.4% stub, against 43.0% and 46.9%), so its
bar is the highest and a constant predictor is most attractive there. The subsets are nested
and hashed, so this is a property of that particular quarter rather than a design choice.

### 2a. What this supersedes, and what it does not

`AUDIT/rung0-learning-curve-2026-09-20.json` ran this experiment yesterday and found it
flat: *"4x the training files changes the held-out number by 0.0% … so 'more data' is not
the next step and this measurement is what rules it out."*

That conclusion is right and narrowly scoped, and its own numbers say why: **every point of
it had collapsed** — measured train accuracy equal to the training majority share to three
decimals at both ends, 0.442 against 0.442 and 0.480 against 0.480 — because it ran at
`span_weight 1.0`, before the objective was fixed. It did not answer the data question; it
established that the question could not be asked yet. A constant predictor is perfectly
insensitive to how much data it ignores, so that flatness was real and meant something else.

**The premise is gone, not the arithmetic.** Today's run has zero collapsed seeds at 0.5 and
1.0. It is the same experiment on a model that fits.

That file is also the cleanest instance in this repository of the trap in §1: three points,
all flat, all at the majority share, flatness genuine, and any one-line summary of it
("learning curve: flat") carries the opposite of its meaning. It is honest and says so
itself; the summary would not be.

### 2b. Pre-registration, and what it was worth

`--prior-sd 0.0197 --target-difference 0.05`, fixed in the launch script before any of these
numbers existed. The sd is the seed spread of the full-data arm at this configuration,
computed from its **eight ledger rows** (`gh200-rung0-capacity-sw005-nondet-2026-09-21.jsonl`,
recipe `97d0f8e2b6`, mean 0.5399, sd 0.0197) — an arm that *fit*, every seed above the 0.469
training majority, which is what makes its spread a spread. The sds available from the
collapsed 4096 e10 arms (0.0111, 0.0067) would have claimed roughly three times this
design's sensitivity.

Pre-committed rule: **any point whose realised sd exceeded 0.0505 is reported as unable to
see its own target rather than as a null.** Realised: 0.0096, 0.0239, 0.0289. None triggered
it. Nothing moved, and both numbers are carried.

**Every floor in this document is a lower bound.** `qd_train.power.resolvable_difference`
assumes `sd` is *known*; it is estimated from 8 runs, and that module's own docstring says
the honest quantile is Student's t — about 3.20 against 2.80 at n=5 — reported rather than
corrected because an inverse-t needs SciPy and SciPy is not a dependency here. The one
resolved result clears its bound by 32% and the correction at n=8 is smaller than the ~14%
quoted at n=5, so it plausibly survives. **It was not computed.**

---

## 3. What it cost, and why that sentence is possible for the first time

**24 rows, 1420.2 GPU-seconds, $0.5878** on `lambda-1xGH200` at $1.49/h — summed from the
rows themselves. Single GPU, far under the $20 that needs a human yes under rule 4.

Every rung 0 row written before today reads `wall_clock_s ≈ 1e-4` and `cost_usd 0.0`: the
recorder timing its own lifetime, on a machine recorded as a local Mac. All 24 of these
carry `wall_clock_source="caller"`, a real duration and a real price.

They also carry **one** `code_that_ran` digest: `173fa35737ed…`, 29 sources. It was verified
equal on this Mac and on the box before launch, and re-checked equal on the box afterwards.
This Mac now reports `9cae19bef139…` because `f93b22c` landed mid-run — so **the rows pin the
code that ran, not the code at HEAD**, which is the case that metric was added for. Note the
source *count* is 29 on both; only the digest distinguishes them. `code_commit` reads
`8b9df39-dirty` with 85 files dirty and distinguishes nothing.

---

## 4. What changed (commits by this lane)

| commit | what |
| --- | --- |
| `ea6522a` | cost out of the run dict — it was a `TypeError` waiting for `json.dumps` |
| `8003d09` | end-to-end tests: a cost and a clock reach a row **from a runner that ran** |
| `03ba0a9` | `make gates` recorded with the suite sizes it actually covered |
| `7f191bd` | 61 capacity rows brought off the rented box, chains verified |
| `77aaf2b` | `tools/ledger_arms.py` — read a sweep back from its rows |
| `bc3901e` | name both arms on a line that compares two arms; + the curve-legibility gap |
| `afa7c1f` | the curve: ledger + `AUDIT/rung0-learning-curve-2026-09-21.json` |

`tools/ledger_arms.py` is the durable piece. Nothing in `tools/` read rows back before it;
every runner writes them and the reading was done by parsing stdout. It groups on
**`(recipe_hash, backbone_commit)`** — a capacity sweep's arms share a recipe and differ in
the backbone, a curve's points share a backbone and differ in the recipe — takes the collapse
verdict from the gate the runner already recorded rather than from its English `detail`, and
asks `qd_train.power` for the floors instead of carrying its own copy of 2.80.

---

## 5. What is open

| gap | state | who |
| --- | --- | --- |
| `GAP-A-KILLED-RUNG0-RUN-LEAVES-NO-ROW-AND-NO-COST` | **resolved** `f93b22c` | opened here, fixed by the other lane |
| `GAP-A-ROW-CANNOT-SAY-WHICH-CURVE-POINT-IT-IS` | **open** | opened here, other lane's file |
| `GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN` | resolved-with-residual | — |
| `GAP-LEDGER-NO-STORY-FOR-A-CHAIN-FORKED-ACROSS-TWO-MACHINES` | open since 2026-09-20 | — |

The second one is why the table in §2 labels its points by recipe hash. `--train-subsample`
goes into `recipe_hash`'s *input* (`0e897b3`) and is stored nowhere readable on the row, so
the three points are correctly **distinguished** and cannot be **identified**. I recovered
`918c73733b=0.25, 64b61aea47=0.5, 2b8564f63c=1.0` by recomputing the hash over the candidate
values with the other seven recipe fields held at what the launch used — which needs the
launch command, so it recovers the label from the log by a longer route. Legibility, not
correctness.

**Not run, and reported as such:** the GPU suites that cannot run on this host; the t-corrected
floors (§2b); any 8×H100 work.

---

## 6. The exact first command for the next lane

**Nine seeds would have resolved the full span of the curve. This ran eight.**

Inverting `resolvable_difference` against the observed differences and pooled sds:

| comparison | observed | pooled sd | seeds/point needed |
| --- | --- | --- | --- |
| 1.0 vs 0.25 | +2.99pp | 2.20pp | **9** |
| 1.0 vs 0.5 | +2.21pp | 2.65pp | **23** |
| 0.5 vs 0.25 | +0.77pp | 1.87pp | 93 |

These are sized from the *observed* effects, so they are optimistic twice over — the known-sd
bound again, and an observed difference is itself noisy and biased upward when it is the one
you chose to power against. Treat 9 as "8 was one short" and not as a design.

At $0.5878 for 24 seeds, **24 seeds per point costs about $1.80** and resolves both of the top
two comparisons with margin. Sync first — the box is a clone pinned at `8b9df39` with files
copied over it, so `git pull` is not how it updates:

```bash
bash tools/sync_box.sh
```

`tools/sync_box.sh` refuses three ways, and each refusal is there because the thing happened:

* **while a training process is live** — rsync replaces a tool file under a running
  interpreter, and `what_ran_state` hashes the file at row-write time, so the remaining seeds
  would record the digest of code they did not run. Verified in both directions via
  `tools/check_live_run.sh`; its first version counted its own `pgrep` and could never have
  permitted a sync at all.
* **when the sync set has uncommitted files** — it syncs the working tree, not HEAD. Its first
  run pushed the other lane's half-finished edit to the box. Harmless by luck. Override with
  `QD_SYNC_DIRTY=1` once you have checked whose the files are.
* **when the two digests disagree** after the copy. Do not remove this check: it is the only
  thing that made §3's provenance claim sayable.

Host and key come from `QD_BOX` / `QD_BOX_KEY`, defaulting to today's box.

Then, from `/home/ubuntu/qwen-decision` on the box:

```bash
timeout --signal=TERM --kill-after=60 9000 /home/ubuntu/qd-venv/bin/python tools/rung0_real_run.py --out /home/ubuntu/rung0-curve --rev 0632f693d3b765b726499e7b4bf19c67959b75cb --examples /home/ubuntu/rung0-corpus/examples.jsonl --manifest-in /home/ubuntu/rung0-corpus/manifest.json --device cuda --instance lambda-1xGH200 --usd-per-hour 1.49 --seeds 24 --epochs 10 --context-bytes 8192 --width 128 --layers 2 --heads 4 --batch-size 16 --span-weight 0.05 --train-subsample 1.0 --prior-sd 0.0289 --target-difference 0.0299 --ledger /home/ubuntu/qwen-decision/ledger/gh200-rung0-curve-n24-2026-09-22.jsonl
```

and the same with `--train-subsample 0.25`. **`--prior-sd 0.0289` is today's realised
full-data sd** — a previous measurement of a fitted arm, which is what that flag requires;
do not prime it with 0.0197, which came from a different corpus split. Read the result with:

```bash
python tools/ledger_arms.py ledger/gh200-rung0-curve-n24-2026-09-22.jsonl
```

**Before either:** `ListAgents`. Two sessions shared this worktree all afternoon and it worked
only because we split by file and said so. Three tests were failing in the tree as this was
written — `_recipe_of_a_row` undefined in `test_rung0_real_run.py` — from the other lane's
in-flight edit on the gap in §5, not from anything here. `afa7c1f` stages two untracked files
and nothing of theirs.
