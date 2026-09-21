# HANDOFF — coordinator — 2026-09-21

Continues `HANDOFF/coordinator-2026-09-20.md`. Two threads, in the order they were worked:
**an objective whose two channels are not the same size**, and then **what a full train
needs that this repository did not have.**

---

# Part 1 — the objective

## 1. Rung 0's constraint is its objective, not its capacity, context or corpus

`Rung0Step` has always taken a `span_weight`; `tools/rung0_real_run.py` never passed one,
so every rung 0 run before yesterday trained at the library default of 1.0 — an unweighted
sum of channels opening at 2.026 and 12.520.

| `span_weight` | train acc per seed | held out vs baseline |
| --- | --- | --- |
| 0.05 | 73.4 68.0 58.8 54.3 58.2 | **+5.0%**, 5 of 5 seeds above |
| 0.1 | 77.3 68.4 56.2 54.8 60.0 | +1.2%, 4 of 5 |
| 0.2 | 57.8 59.5 65.4 51.8 48.6 | −0.1%, 1 of 5 |
| 0.5 | 48.0 48.0 48.0 47.8 48.0 | −0.1%, 0 of 5 |
| 1.0 (the default) | 48.0 48.0 48.0 | +0.0%, 0 of 3 |

The **train** column is the proof. 48.0% is the training set's own majority share, to the
row, on every seed — a constant predictor. Below 0.2 it breaks free. Held out at 0.05 all
five seeds beat the 52.1% baseline, mean +5.0%, which *matches* the linear control's +5.3%
rather than exceeding it. `AUDIT/rung0-span-weight-2026-09-20.json`.

## 2. The same objective on the FT path, seeded

`QwenDecisionStep` — what `train_ft` drives — sums the same way and defaults `span_weight`
to 1.0, and **no caller anywhere had ever passed one.** Three seeds per arm, real 1.4B
tower, identical plan and 128-step budget, both channel floors measured at 0.0 so every gap
is optimisation rather than the corpus:

| `span_weight` | balance at open | letter gap per seed | letter gate | span gap per seed |
| --- | --- | --- | --- | --- |
| **1.0** (default) | 254:1 | 0.0437 0.0069 **0.1192** | **2 of 3** | 0.0000 0.0998 0.0601 |
| 0.2 | 72:1 | 0.0037 0.0394 0.0154 | 3 of 3 | 0.1527 0.5663 0.9779 |
| 0.05 | 12:1 | 0.0004 0.0174 0.0010 | 3 of 3 | 0.0001 0.0045 0.5746 |

The letter channel improves monotonically and on every seed — mean gap 0.0566 → 0.0195 →
0.0062, a 9x reduction, and the distributions separate rather than overlap. **At the
shipped default one seed in three fails the letter floor gate outright.**

**The span channel does not show a clean opposite trend**, and three seeds cannot separate
that from noise. `AUDIT/ft-span-weight-seeded-2026-09-21.json`.

**The default stays at 1.0.** This is a much stronger recommendation to change it than the
one-seed run was and still not a measurement that changes it: what would is seeds enough to
price what 0.05 costs the span channel, and a corpus with a nonzero floor.

**This is not rung 0's collapse.** Rung 0's head sat *exactly* at the majority share. Here
the letter channel falls from ~2.07 to within 0.0997 of zero even at 1.0. The FT path
degrades where rung 0 stopped; likeliest cause is capacity, not isolated.

---

# Part 2 — what a full train needed

## 3. A schedule could outlive its own second moment, and nothing checked

`optim.py` could measure that a bf16 `exp_avg_sq` stops moving and settles at half the
value it is chasing, say so across forty lines of its own docstring, and still hand back
the optimizer that does it. Nothing knew how long the run would be.

    bfloat16  stops at step    384   settling 0.500000   50.00% low
    float16   stops at step  1,204   settling 0.732422   26.76% low
    float32   stops at step 10,301   settling 0.999970    0.00% low

Measured by simulating the EMA in the real dtype, not derived: the closed form gets the
condition right and the answer wrong, because bf16 lands exactly on a binade boundary.
**fp16 is the new datapoint** — three times bf16's life, still 27% low.

`build_optimizer` now **requires** `total_steps`. Not defaulted: an opt-in guard against an
invisible failure is a guard that is off, and every caller already computed the number.
`allow_frozen_moments=False` is the named exception, after
`write_shards(allow_contradictions=...)`. A 383-step run is still admitted.

## 4. The FT checkpoint carried weights and not the optimizer

**Every resume test in this repository drove `TinyStep`, a test double.**
`QwenDecisionStep.state()` returned the tower, the span head, `micro_batches`,
`span_weight`, `vocab_size` — and not the optimizer. A resume restored the parameters
exactly and reset AdamW's moments to zero.

Measured through the real loop: 12 FT steps as one run against the same 12 as 6+6 across a
checkpoint written to and read from a real file — **5 of 12 losses differ**, step 7 giving
3.431338 against 3.440382, every divergence in the post-resume leg. The weights restore
bit-exactly; only what the next step does with them differs, which is why nothing caught it.

`run_control.py`'s own header already warned of exactly this, and `MAX_SIDECAR_BYTES` was
derived from "weights **and optimizer state** in fp32". The design was sized for it; the
step never put it there. `MasterWeightAdamW`'s fp32 masters were missing too — a resume
would have rebuilt them from the bf16 parameters, the rounded copy that class exists to
avoid.

## 5. Writing a checkpoint is somebody's job now

`HANDOFF/resume-2026-09-20.md`: *"No driver in `tools/` writes a checkpoint, so nothing on
disk survives a kill … which is still nobody's job. That last piece is the first command
for the next lane."*

`--checkpoint-dir` / `--checkpoint-every` / `--resume-from` on `tools/real_ft_run.py`.

**Proven end to end on the real 1.4B tower, not on a tiny one:**

1. run wrote checkpoints at steps 20, 40, 60 — 7.93 GiB in ~10.4 s each;
2. `SIGKILL`, no cleanup, no result reported;
3. resumed from the file: `at optimizer step 60, batch index 60, seed 0`;
4. continued through 80, 100 and completed.

**Each cell of the run gets its own file.** The first spelling was
`f"{tag}-seed{seed}.json"`, which collides across devices — and the collision is worse than
an overwrite, because the survivor passes every check the trainer has. As the peer session
that caught it put it: the two colliding runs share a seed *and* a schedule, so all three
things `Checkpoint` can be refused on match, and `consumed_digest` matches too because the
batch order really is identical — *"the checkpoint is valid in every way except that it
came off other silicon."* `_checkpoint_name` / `_resume_arm` are inverses and refuse
anything this tool did not write.

**The interval is a cost decision and the driver prices it.** At 7.93 GiB and ~10.4 s a
write against ~1.5 s/step, a 20-step interval spends a third of its wall clock writing. A
full train wants a far larger interval; the number is now on screen beside the step.

---

## 6. Two open gaps that were not open

Re-verified while auditing, both **stale and now resolved**:

* `GAP-TRAINER-TORCH-SUITES-NEVER-RUN-IN-THE-GATE` — `LEDGER_RECORD_CMD` has carried
  `--suite pytest_torch_python_tests=…` all along, and `ledger.py` maps a NotRun suite to
  exit 3 rather than 0. Confirmed by running the gate: it reports the torch suite's
  1612/1614 by name.
* `GAP-TOOLS-REAL-FT-DEFAULTS-ARE-THE-STAND-INS-DEFAULTS` — `REAL_BACKBONE_LR` is separate
  from `STANDIN_LR`, and `--hidden`/`--heads` are refused under `--real-backbone` rather
  than accepted and ignored.

An open gap naming the test gate as blind is worse than it sounds: it invites the next lane
to build a second one beside it.

---

## State at handoff

| Gate | Result |
| --- | --- |
| `make gates` | **PASS** — every gate ran and passed, on the tree merged with the peer's `d13cfae` |
| ruff | clean over `python/` and `tools/` |
| pytest, torch venv | 1640 passed, 2 skipped |
| pytest, repo venv | 1415 passed, 20 skipped |
| cargo, workspace | **362 passed, 0 failed** across 23 binaries |
| ledger chains | 3 files, each verifying: 171 / 253 / 174 rows |
| ledger forks | **1**, the known pre-existing divergence at row 157 |

Every row written today is `quick=True` and promotes nothing (rule 8).

Commits: `391fba2`, `1cc7647`, `1567949`, `8e019d7`, `84487d2`, `79d1ee1`, `109ff95`,
`3f06c41`.

**Two counting cautions.** The cargo number was mis-reported twice in this repository by
piping `cargo test` through `tail`, which truncates the per-binary lines the total is summed
from; 362 is from the untruncated stream. And `pkill -f real_ft_run.py` matches the
monitoring shell that contains the pattern — kill by anchored binary path
(`pkill -f "^/home/ubuntu/qd-venv/bin/python"`) instead.

**DevMap caveat:** `devmap_clones` answered truncated at 25 of 102 groups. The 25 shown are
all pre-existing and none involve functions added today, which is **not** the same as "no
duplication was introduced."

---

## Open, in the order the evidence suggests

1. **A driver-level test that a resume onto a different corpus order is refused.**
   `trainer.py` rehashes the consumed prefix and `test_trainer.py` covers it — with
   `TinyStep` only. The end-to-end proof above covers the happy path.
2. **A nonzero-floor FT run.** Every FT measurement so far has both floors at 0.0. The
   2026-09-20 shard set had a nonzero span floor through contradictory twins; the current
   one does not. The regime with less headroom is untested.
3. **Seeds enough to price the span channel** at low `span_weight`, which is what stands
   between the seeded sweep and moving the default.
4. **Two rung-0 candidates** from `GAP-RUNG0-A-LINEAR-CONTROL-BEATS-THE-MODEL`: mean-pooling
   an 8192-byte context into 128 dims, and `lr=3e-3` chosen for a memorisable toy batch.

**Claimed by a peer session** (fork `0c8dc0`, same worktree):
`GAP-SHARD-SET-GOES-STALE-AGAINST-THE-CORPUS-CODE-THAT-REPRODUCES-ITS-LABELS`, holding
`python/qd_data/` and its tests. Its reasoning is worth keeping: today's refusal fired only
because the drift moved the *counts*, and code drift that changes labels without changing
counts produces no refusal at all. `data_snapshot_hash` hashes the corpus; nothing hashes
the code that turns the corpus into rows.

**User-owned, unchanged:** the ledger fork at row 157; `clean` at 1.7% of examples against
`--clean-permille 200`.

## A precondition for the full train, from the peer's landed work

`d13cfae` gives `ShardHeader` a `code_fingerprint` — a `{module: sha256}` map over every
`.py` in `qd_data`, checked by `assert_shard_trainable` in `ShardReader.__init__`, so the
answer arrives at `open()` rather than at the first optimizer step.

**Every shard set that exists today predates the field and therefore reports `NotRun`,
including `~/shardset-v2` on the box.** It does not raise — they stay readable and the
ledger rows naming them keep their evidence — but `shard_code_current` will read NotRun in
every row until the set is regenerated. **Regenerate `shardset-v2` before the full train,
not after**, or the run's own row records "nobody could tell" on exactly the question the
field exists to answer.

Second consequence: the fingerprint is over source *bytes*, so a docstring-only edit to any
`qd_data` module invalidates every shard set written before it. That is the safe direction
to be wrong in, and it couples `qd_data` edits to regeneration — if either is touched during
run prep, regenerate.

## First command for the next lane

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/bharath/.local/bin/uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis python -m pytest /Users/bharath/Code/research/qwen-decision/python/tests/test_backbone.py -o addopts= -q -k resume
```

Those are the tests that prove the real step resumes. Read them before touching
`QwenDecisionStep.state()`: the optimizer half is what makes a resume a resume, and it is
invisible in the weights.

---

# Part 3 — the night of the 21st

Written after Parts 1 and 2, and it revises Part 1.

## 7. The third provenance pin: which revision the corpus was read at

`data_snapshot_hash` hashes the rows that came out. `code_fingerprint` hashes the code that
turned them into rows. Neither covers **which revision was read**, so a set built from the
wrong one is self-consistent in both and passes everything — the rev for the 2026-09-21
incident had to be recovered from the free-text NOTES of ledger row `46e63ff3`.

`ShardHeader.corpus_rev` records it and `shard_hash` covers it. `_shard_rev_check` is
tri-state and the middle answer is the point:

| state | when |
| --- | --- |
| `NotRun` | the header carries no rev — every set written before today |
| `NotRun` | the **caller** named no rev. A set describing itself is not a set that was verified |
| `Ran(passed=False)` + refusal | the two disagree |

An absent rev contributes nothing to `shard_hash`, so no set on disk was invalidated.

It matters because `tools/real_ft_run.py` does not read labels out of the shard set: it
**reconstructs** them from the repo at `--rev` and pairs them with the set's sequences. The
321-against-341 refusal fired only because the drift moved the row *count*. A revision that
changes which rows exist without changing how many lands every label on the wrong sequence
with no other symptom.

**Wiring it exposed a second defect.** `real_tokenizer_pipeline.py` takes `--rev`
**defaulting to `HEAD`** and resolves it internally into `resolved`, which is what every
corpus read uses. Pinning the argument would have written `"HEAD"` into the header — worse
than writing nothing, because an absent rev reads `NotRun` and `"HEAD"` compares equal to
`"HEAD"` tomorrow and reads as *verified*. `tools/repo_git.py` gained `resolve_rev`; both
tools go through it, and two tests read the tools' source so the next caller cannot quietly
pin an unresolved name.

`~/shardset-v4` on the box is the first set whose `shard_rev_matches` reads
`Ran(passed=True)`. Identical `data_snapshot_hash` to v3 (`101168a90055`), 321 sequences —
the rows did not move, only the pin was added.

## 8. The seed named the batch order and not the model

**This revises §2 of Part 1.**

A single-seed smoke against v4 did not reproduce §2's seed-0 numbers. Four runs at
`--seeds 0 0 1 1` found two runs at *one* seed differing **at step one**:

    opening total loss, six runs of ONE configuration
    253.48   279.09   298.44   334.62   382.58   489.91

`QwenDecisionStep` builds a randomly-initialised `SpanPointerHead`, and `backbone.py`
contained no `torch.manual_seed`, no `torch.Generator` and no seed argument at all.
`tools/real_ft_run.py`'s **other** branch — the stand-in `RealFtStep` — calls
`torch.manual_seed(seed)` on the first line of `__init__`.

So the tool was reproducible on the backbone nobody measures and not on the one every GH200
row used. One name, two behaviours, split across the two branches of one tool, and the
branch that got it right is the one whose numbers do not matter.

**What it cost.** The letter head is pretrained and the span head is random, so the draw
landed almost entirely on one channel:

| across those six runs | range |
| --- | --- |
| final letter loss | 1.07e-4 … 8.81e-4 |
| final span loss | 0.000000 … 1.823686 |

That is exactly the shape §2 reported as *"the span channel does not show a clean opposite
trend … three seeds cannot separate that from noise."* The spread was not seed noise. The
seeds were not controlling the quantity that moved, and more seeds of the same kind would
not have resolved it.

**What survives §2:** the letter column. Monotone across the three arms — 0.0566, 0.0195,
0.0062 — on a head that was never the random one. **What does not:** the span column, every
per-seed reading of it, and the `2 of 3` gate counts (a repeat of seed 0 at `span_weight`
1.0 passed the gate the first run failed). `AUDIT/ft-span-weight-{seeded,paired}` and
`AUDIT/ft-channel-balance` each carry a `correction` key saying so; their numbers are not
rewritten, because they were measured.

**Rung 0 is unaffected** — `tools/rung0_real_run.py:525` seeds. Part 1 §1 stands.

`seed` is now a **required** keyword on `QwenDecisionStep`, like `total_steps` before it.
The fix is in the class, not the driver: a driver that seeds before each construction gives
the same guarantee only until the next caller forgets, or until anything between the
seeding and the construction draws a number.
`test_the_seed_survives_an_arbitrarily_advanced_global_rng` advances the ambient stream by a
different amount before each build and requires the two steps to agree.

**Confirmed on the box:** two runs at seed 3 now both open at exactly `369.5100`.

## 9. The overnight run, and why it is not the sweep it was going to be

The plan was a 20-seed `span_weight` sweep — §2's open question. Running 100 arms first
would have measured the same mistake more precisely, so phase 1 is now the residual.

| phase | what | arms |
| --- | --- | --- |
| 1 | the residual once the seed is honoured | one config, **8 repeats at seed 0** |
| 2 | the sweep, seeds that now control the init | 5 × `span_weight` × **12 seeds** |
| 3 | optimizer control: `master` at the **same** 128 steps | 2 × 5 seeds |
| 4 | the long schedule: **512 steps** under `master` | 2 × 5 seeds |

Phase 3 exists so phase 4's difference is attributable to the schedule rather than to the
optimizer that makes the schedule legal — 512 steps is past the 384-step bf16 second-moment
bound, so `build_optimizer` refuses bf16 there and `master` is the recipe that makes the
length meaningful rather than the flag that silences the guard.

Driver `/home/ubuntu/overnight.sh`, 9-hour cap checked **between** arms (killing a run
mid-flight leaves a ledger row claiming a schedule it did not finish). Logs in
`/home/ubuntu/overnight/`, ledger `ledger/gh200-overnight-2026-09-21.jsonl`, shard set
`~/shardset-v4`. Every row is `quick=True` and promotes nothing (rule 8).

**`MAX_SEEDS` is 8**, which the first launch discovered by being refused; phase 2's 12 seeds
are two invocations of 6 rather than a sweep shrunk to fit a per-invocation bound.

## State at the end of Part 3

| Gate | Result |
| --- | --- |
| `make gates` | **PASS** — lint, clippy, ledger-record, ledger-verify |
| cargo | 362 / 362 |
| pytest, torch venv | 1661 passed, 2 skipped |
| pytest, repo venv | 1431 passed, 20 skipped |
| ledger chain, `runs.jsonl` | 177 rows |

Commits: `33e20a1` (the rev pin, 16 tests), `fee0f9a` (the seeding, 5 tests). Every new test
verified to fail against the pre-fix tree by reverting the sources and re-running.

## Open, revised

1. **Phases 1–4 above** are the current measurements. Phase 1 decides how the rest are read.
2. **Whether `span_weight`'s default moves** now depends on phase 2, not on §2.
3. **A nonzero-floor FT run.** Unchanged and still untested; every FT measurement so far has
   both floors at 0.0.
4. **A driver-level test that a resume onto a different corpus order is refused.** Unchanged.
5. **Two rung-0 candidates** from `GAP-RUNG0-A-LINEAR-CONTROL-BEATS-THE-MODEL`. Unchanged.

**Carried forward, unchanged:** the ledger fork at row 157 and `clean` at 1.7% against
`--clean-permille 200` are user-owned.

## First command for the next lane

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 'cat /home/ubuntu/overnight/driver.log; grep -h "^cuda seed=" /home/ubuntu/overnight/p1-residual-seed0-x8.log'
```

Read phase 1 before reading anything else. Eight runs of one configuration at one seed: the
spread between them is the error bar every later arm has to be read against, and if it is
not small then phase 2's per-seed numbers are still a mixture — of the arm and of whatever
CUDA does with atomics — and must be reported as one.

---

# Part 4 — what the night measured

Part 3 said "read phase 1 before reading anything else". Phase 1 said the span channel was
not measurable. Phase 4 said why, and the answer retires the question rather than answering
it.

## 10. Two days of span-weight work were measuring an unconverged budget

88 runs, 15 arms, on shardset-v4 under the seeded step. Every row `quick=True`,
`ledger/gh200-overnight-2026-09-21.jsonl`, chain verifying at 176 rows.
`AUDIT/ft-overnight-2026-09-21.json` carries every run.

**Phase 1 — the residual.** Eight repeats of one configuration at one seed. All eight
opened at an identical **518.5886**, so the seed now fixes the start. The final span loss
still ranged **0.000000 … 1.505752**, 3 of 8 over the bar — the whole range the 3×3 had
attributed to its three arms, from one arm at one seed. The letter channel stayed inside
1.0e-5 … 1.3e-4, ~380× below its bar.

**Phase 2 — the sweep, 12 seeds per arm.** Neither of the 3×3's headlines survives.

| `span_weight` | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| mean letter gap | 0.00036 | 0.00050 | 0.00010 | **0.01360** | 0.00045 |
| letter gate | 12/12 | 12/12 | 12/12 | **11/12** | 12/12 |
| span gate | 10/12 | 11/12 | 10/12 | 10/12 | 9/12 |

No trend. The shipped default fails nothing; the only arm that fails at all is 0.5. "Improves
monotonically on every seed" and "one seed in three fails the letter floor gate at the
default" were three draws each.

**Phases 3 and 4 — the schedule, isolated from the recipe.**

| | letter gate | span gate | mean letter gap | max span gap |
| --- | --- | --- | --- | --- |
| bf16 @ 128, sw 1.0 | 12/12 | 9/12 | 0.00045 | 2.93503 |
| master @ 128, sw 1.0 | **2/5** | 4/5 | 0.38609 | 0.68058 |
| master @ 512, sw 1.0 | **5/5** | **5/5** | **0.0000004** | **0.00012** |
| master @ 512, sw 0.05 | **5/5** | **5/5** | 0.00001 | 0.00000 |

Phase 3 is what makes phase 4 attributable: at equal length the master recipe is *worse*, so
the length is what closes the gap, not the recipe. At 512 steps both arms reach both floors
on every seed and the span channel's spread collapses four orders of magnitude.

There is no bf16 arm at 512 — `build_optimizer` refuses one past step 384. That is the guard
working, not a hole.

**The conclusion.** `span_weight` **stays at 1.0**, now for a positive reason rather than for
absence of evidence: at a converged budget it reaches both floors on every seed, and at an
unconverged one the arms are not separable from the residual. **Any claim in this repository
resting on a 128-step FT number is measuring an unconverged budget.**

Not settled: both floors are 0.0 on this corpus. The regime where an objective's weighting
could still matter — a nonzero floor — remains untested, and is now the single most
informative experiment left.

## 11. `--deterministic`, and a rationale its own night corrected

`tools/real_ft_run.py` gained the flag. `CUBLAS_WORKSPACE_CONFIG` is set from `sys.argv`
**before torch is imported** — cuBLAS reads it at its first matmul, so setting it from parsed
arguments would be a setting that looks applied and is not. Only when asked: a 32 MB
workspace on a run that did not ask for one is a cost paid for nothing.

Every row now carries a `deterministic_kernels` TriState. A *completed* deterministic run is
a checkable claim, because torch **raises** where an op has no deterministic implementation.
A run without it established nothing and now says so, rather than being silent — which reads
as the former.

**The flag's help and the NotRun reason were written from phase 1 while phase 4 was still
running, and phase 4 contradicted them.** Both now carry both numbers and a test requires it.
Arguing for determinism on the 128-step spread alone argues from a premise the same night
measured to be mostly wrong. The flag is still worth having; the case for paying its price is
weaker than it looked at 03:00.

## 12. A root fork is not a divergence

Pulling the night's ledger in took `find_forks` from 1 to 2. The new one is a **root** fork —
the detector saying "these files are unrelated", which is what a ledger opened for one run
*is*, and is the direction `GAP-LEDGER-NO-STORY-FOR-A-CHAIN-FORKED-ACROSS-TWO-MACHINES`
names as the fix. Counting it beside the row-157 **divergence** made the recommended fix
register as the disease.

`test_the_repositorys_own_ledgers_are_checked_not_assumed` now counts divergences and allows
at most one root group. Teeth unchanged: a new divergence is non-root and still takes it past
one. `verify_no_fork` is **not** changed — whether it should ignore root forks is part of
that gap's open human decision.

A concurrent lane recorded this failure in `fad5315` as *"transient … did not reproduce"*. It
was neither; it was this file existing. They have been told.

## State at the end of Part 4

| Gate | Result |
| --- | --- |
| `make gates` | **PASS** — lint, clippy, ledger-record, ledger-verify |
| cargo | 362 / 362 |
| pytest, torch venv | 1669 passed, 2 skipped |
| pytest, repo venv | 1435 passed, 20 skipped |
| `ledger/runs.jsonl` | 178 rows |
| `ledger/gh200-overnight-2026-09-21.jsonl` | 176 rows, 88 runs |
| `find_forks` | 1 divergence (row 157, user-owned) + 1 root group |

Commits: `33e20a1`, `fee0f9a`, `ad2c219`, and the overnight commit. A concurrent lane landed
`fad5315` between them, which **edits `CLAUDE.md`** — flagged to the user rather than
reverted.

## Open, revised again

1. **A nonzero-floor FT run at 512 steps.** Now the most informative experiment left: it is
   the only regime in which `span_weight` could still matter, and everything else about the
   objective is settled at a converged budget.
2. **How much of the 128-step spread is kernels.** `--deterministic` exists to ask and has
   not been run. Lower value than it looked before phase 4.
3. **A driver-level test that a resume onto a different corpus order is refused.** Unchanged.
4. **Two rung-0 candidates** from `GAP-RUNG0-A-LINEAR-CONTROL-BEATS-THE-MODEL`. Unchanged.

**User-owned, unchanged:** the ledger divergence at row 157; `clean` at 1.7% against
`--clean-permille 200`.

## First command for the next lane

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 '/home/ubuntu/qd-venv/bin/python -c "import json; d=json.load(open(\"/home/ubuntu/overnight/summary.json\")); [print(a[\"arm\"], a[\"n\"], a[\"letter_gate_passed\"], a[\"span_gate_passed\"], round(a[\"span_gap_max\"],5)) for a in d[\"arms\"]]"'
```

Then build a corpus with a nonzero span floor and run phase 4's two arms against it. Every FT
number in this repository before today was taken at 128 steps, and 128 steps is not where
this model's answer lives.

---

# Part 5 — the correction, and two findings that are not mine

## 13. The 128-step spread was the kernels, and Part 4 said otherwise

Part 4 read phase 4 as saying *"most of phase 1's residual is an unconverged budget rather
than the kernels."* That was an inference from "removing the amplifier removes the symptom",
and it is wrong. Measured directly:

**Eight repeats of phase 1's exact configuration with `--deterministic` came back
bit-identical.**

    total 518.5886 -> 0.000078   letter 1.9029 -> 0.000078   span 518.5886 -> 0.000000

Every run, every digit. Without the flag the same eight gave final span losses of
0.000000 … 1.505752 with 3 over the bar.

The reading that needs both measurements:

* **kernel nondeterminism is the SOURCE** — remove it and the spread is gone at 128 steps;
* **non-convergence is the AMPLIFIER** — at 512 steps the spread collapses to
  [0.00000, 0.00012] *without* determinism, because a converged run stops amplifying the
  perturbation.

Remove either and the symptom goes, which is precisely why the cause could not be inferred
from phase 4 alone. **A long schedule hides this rather than fixing it.**

The run also *completed* rather than raising, which establishes something narrower and
useful: every op this model uses has a deterministic implementation.

**Price on the FT path: 131.3 s per run against 111.4 s — 18% wall clock.** Cheap enough
that a full train should pay it.

**Do not carry that 18% to another path.** The concurrent lane measured the same flag on
`tools/rung0_real_run.py`, same box: **239.0 s against 103.0 s, 132%**. Different kernels, a
much smaller model, 460 steps of a byte transformer rather than 128 of a 1.4B tower. On rung
0 determinism is a budget decision rather than a free win, and the reason for the 7×
difference in the ratio is not established. That is the opposite of what the flag's own help string said when it was
written from phase 1, and the string, the ledger metric's `NotRun` reason, the audit and the
gap record all carried the reversed claim and have all been corrected.

**Part 4's span_weight conclusions are unaffected** and stand: at 512 steps both arms reach
both floors on 5/5 seeds, at 128 the arms are not separable from the residual, and the
default stays at 1.0. Only the attribution of the residual changed — and with it the
recommendation.

`ledger/gh200-det-2026-09-21.jsonl`. Its rows are also the first in this project to say
which transformers produced them: `transformers==5.17.0`.

## 14. Two library-chosen quantities that decide a number

Raised by the concurrent lane while closing `MEMORY-GRADIENT-CHECKPOINTING-IS-NOT-WIRED`,
verified here rather than taken on their word — and checking it found a second hole beside
it.

`load_text_tower` built `Qwen3_5TextModel(text_config)` and the attention kernel was
whatever transformers resolved. `Environment.detect` auto-detected torch's version while
taking transformers' as a parameter **defaulting to `"unknown"`** — only
`real_tokenizer_pipeline.py` ever passed one, so every ft row, which is every training
number this project has, said `"unknown"`.

    this Mac    transformers 5.12.1   flash_attn absent   -> sdpa
    the GH200   transformers 5.17.0   flash_attn absent   -> sdpa

They agree today, are five minor versions apart, and nothing in any row said so. Attention is
where the activation memory and most of the arithmetic is: an upgrade on one host moves the
kernel without moving `protocol_hash`, `recipe_hash`, `data_snapshot_hash` or `corpus_rev`.

`attn_implementation` is now required with no default on `load_text_tower`, the third
argument there to be so after `gradient_checkpointing` and `optimizer`. `TextTower` reports
what the model **resolved**; `remap_text_tower` carries it through, because the real path
always remaps. It is in `backbone_keys`, so it enters `recipe_hash`. Default `sdpa`, so no
existing number moves.

## 15. Two things the concurrent lane found that change what this repository believes

Both verified here before being written down.

**Every rung-0 sweep before 2026-09-20 23:50 ran inside the collapsed regime.**
`--span-weight` did not exist until `391fba2` (23:50:32), so the context sweep (19:33) and
the capacity sweep (21:22) both ran at the unweighted default — where, by
`AUDIT/rung0-span-weight-2026-09-20.json`'s own words, train accuracy is *"pinned at 48.0%,
the training set's own majority share, on every seed."* A constant predictor's accuracy
cannot respond to capacity or context. So `GAP-RUNG0-CONTEXT-IS-NOT-THE-BINDING-CONSTRAINT`
and `GAP-RUNG0-CAPACITY-IS-NOT-THE-BINDING-CONSTRAINT-EITHER` are **unestablished, not
established**, and the corpus conclusion they feed is premature.

**Rule 8's three-seed floor is admissibility, not sufficiency.** From the five seeds of my
own span-weight sweep at 0.05 — 0.587, 0.545, 0.571, 0.617, 0.538 — sd = 0.0322, so n=3
resolves ~7.4 pp, n=5 ~5.7 pp, n=8 ~4.5 pp, n=12 ~3.7 pp, against a demonstrated extractable
signal of **5.3 pp**, the linear control over baseline. The capacity sweep's −1.4 / 0.0 /
−1.2 / −0.3 pp were all inside its own noise floor. A null from three seeds means *"no effect
larger than about 7 pp"* and was written down as *"no effect"* — the same defect as
presenting a capped sample as complete coverage, moved from sampling to statistical power.

**That sd is not seed variation, and the figures above describe a NONDETERMINISTIC design.**
Rung 0 turned out not to be reproducible at a fixed seed either: two runs at seed 0, same
corpus by sha256, same 727/288 decisions, gave 50.3% and 53.1% — 2.8 pp apart, on a corpus
whose entire demonstrated signal is 5.3 pp. So 0.0322 mixes seed with kernel and is an
**upper bound** on the seed-only sd.

Which way that cuts depends on the design, and both directions matter:

* for a sweep run **as all of them were** — nondeterministic — 0.0322 *is* the right
  statistic and ~7.4 pp at n=3 is the correct floor. The criticism of those sweeps stands
  exactly as written.
* for a **deterministic** sweep the relevant sd is seed-only and smaller, so the floor
  improves — but by how much is unmeasured. A crude estimate from that single pair puts
  sd_seed somewhere in 0.020–0.029, and one pair is a poor variance estimator whichever
  correction factor is used.

**Determinism therefore buys power as well as reproducibility — but not for free.** At rung
0's 2.3× cost, spending the same GPU-hours on more nondeterministic seeds is roughly
break-even on resolution, and which side of break-even it lands on depends on exactly the
sd the single pair cannot pin down. The concurrent lane's phase 0 — five deterministic seeds
at the cheap control point — measures it directly, and is the right thing to size a
four-hour sweep from.

**This applies to my own Part 4 table.** Phase 2 ran 12 seeds and I reported per-arm means
without stating what 12 resolves. It happens not to bite — the FT letter channel's spread is
~380× below its bar — but I checked that *after* publishing the table, not before. The fix
is to pre-register the resolvable difference with the seed count.

## State at the end of Part 5

| Gate | Result |
| --- | --- |
| `make gates` | **PASS** at the commit before this part; re-run below |
| cargo | 362 / 362 |
| pytest, torch venv | 1676 passed, 2 skipped |
| pytest, repo venv | 1435 passed, 20 skipped |
| `ledger/gh200-det-2026-09-21.jsonl` | 18 rows |

## Open, revised

1. **A nonzero-floor FT run at 512 steps**, with `--deterministic`. Still the most
   informative experiment left, and the only regime where `span_weight` could still matter.
2. **The rung-0 capacity sweep at a working `span_weight`** — held by the concurrent lane,
   5 seeds, pre-registered floor, the linear control at 57.4% as the bar.
3. **A driver-level test that a resume onto a different corpus order is refused.** Unchanged.
4. **Pre-register the resolvable difference** in every sweep runner's header, beside the
   seed count. Not done anywhere yet.

## First command for the next lane

```bash
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.58.240 'grep -h "^cuda seed=" /home/ubuntu/det-residual.log'
```

Eight lines, identical to the digit. That is what a reproducible run looks like on this
model, it costs 18%, and nothing before today in this repository had it.

# Part 6 — one defect shape, found three times in one afternoon

Part 5 named two "library-chosen quantities that decide a number". This part is about what
happened when the same shape was looked for deliberately, and about the one time the search
itself caused the damage.

The shape, stated once: **a complete list is written down, and then one member of it is
fixed.** Not enumeration failing. Enumeration succeeding, and the result not being used.

## 16. Four tools, one literal, one of them fixed

`6b18a52` landed `CostEstimate.for_device` and wired `tools/real_ft_run.py`. Its docstring
opens: *"Four tools reached this class through the same literal."* Three of them still had
it four hours later.

```python
cost=CostEstimate(cap=cap, usd_per_hour=0.0, n_gpus=0, instance=f"local-{device}")
```

with a comment reading *"A Mac that is already bought costs nothing per hour."* `--device`
is a free string defaulting to `mps`. Nothing enforced that comment's premise, and
`tools/rung0_real_run.py` is the tool that ran on the rented GH200 with `--device cuda`.

So those rows do not under-report a cost. They assert `instance = "local-cuda"` on
`n_gpus=0` at `$0.00/h` — a local machine, with no GPUs in it, that does not exist. And at
those two values **both** disjuncts of `requires_human_approval` are False for any cap, an
8×H100 job included, while the per-GPU column check is gated on `n_gpus > 1` and never runs.

`c74f4ba` routes all four through `for_device`; `rg 'usd_per_hour=0.0, n_gpus=0' tools/`
returns nothing. `rung0_real_run.py` gained `--instance`, `--usd-per-hour`,
`--usd-per-gpu-hour`, `--approved-by`, and an argv-time refusal reached before every other
argv check.

`n_gpus` is **counted**, not passed: the rate is a fact about a contract and the caller
states it; the count is a fact about the machine and the machine states it. Counting needs
torch and `run_control.py` is torch-free by contract, so it could not sit beside
`LOCAL_DEVICES` — hence `tools/run_cost.py`, on `repo_git.py`'s precedent.

Seven tests, six verified failing pre-fix. The seventh is the control that a **local** run
is unchanged, which passes both ways by design and is the point: `for_device` had to leave
the honest case byte-for-byte alone.

One test-design note worth carrying: `--epochs 0` is the instrument in the argv tests, not
the subject. It is refused a few lines further down, so a tool that reaches *that* refusal
first had not yet decided anything about the price. It makes the pre-fix failure immediate
instead of a corpus build on a rented box.

## 17. The same shape, one layer over — and the concurrent lane found it

`d39f60b` added `_what_ran` / `_what_ran_state` to `real_ft_run.py`, closing
`GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN` — a row's `code_commit` reads `"<sha>-dirty"`
for any uncommitted change, and on the GH200 that one bit stood for 6894 insertions across
72 paths.

The concurrent lane's criticism was exact: two private functions in one tool, answering a
question every runner's rows raise, for one of four runners — while `rung0_real_run.py`, the
tool that wrote the GH200 rows, had **no** fingerprint at all and its provenance had to be
reconstructed by hand with `sha256sum`.

`b185dcb` lifts it to `qd_train/ledger.py` as public `what_ran` / `what_ran_state`. All four
runners record a `code_that_ran` metric — two row kinds each for `real_ft_run` and
`ft_toy_run`. `real_ft_run`'s private copy is deleted in the same change; a lift landing
beside the original is the accumulate-instead-of-replace failure that produced this.

Both arguments required, no defaults. A default package answers for the wrong one when a
fifth runner appears; an unnamed tool lets two runners produce identical fingerprints from
different code, which is exactly `code_commit`'s defect. `qd_data` stays excluded and the
reasoning moved with the function: the shard header already carries it and
`ShardReader.open()` checks it, so a second recorder would be free to disagree — and the
disagreement would surface as a shard-contract failure on a shard set that is fine.

Seven tests, **all seven** verified failing pre-lift.

## 18. The search caused the damage: a stash, and a rule that never arrived

Mid-afternoon the concurrent lane reported two of its gap records missing from
`gaps.jsonl` — present on disk after an `O_APPEND` + `fsync` write it had verified by
re-reading, and gone later. It wrote `GAP-GAPS-JSONL-APPENDS-LOST-WHILE-TWO-LANES-WROTE-IT`,
restored both records by hand, and changed its workflow.

Nothing was lost. Both records were in `stash@{0}`, which **this lane created**, running
`git stash push -- gaps.jsonl` three times to get a clean tree for a gate run. `stash push
-- <path>` reverts that path to HEAD, which is why the other lane found the file *clean at
440 lines* rather than dirty — the one detail it correctly said a lost concurrent write
cannot produce. It had diagnosed the mechanism ("a checkout/stash/restore of a path whose
second writer was not known about") and attributed it to an unknown actor, because it could
not see another session's stash.

Recovery took only this lane's own line and left both of theirs as rewritten, then dropped
the stash — popping would have duplicated records for an id that already supersedes itself.
`f9c3ca1` retracts the loss record as `closed-no-defect`.

**Then the more useful finding.** The other lane wrote that CLAUDE.md already forbade this
and that both lanes had read the rule too narrowly. Checked rather than accepted: that rule
was never in this session's context at all.

`git log -S` puts it in `fad5315`, 07:02:34 — about 2.5 hours before the stash. The current
file's item 3 (*"`ListAgents` before changing, too"*) ends *"never `git stash` or `git add
-A` in a shared worktree — both move the other lane's work"*, and item 4 gained *"Append
with `O_APPEND` + `fsync` … never read-modify-write"*. This session's in-context copy has
none of the three; what it calls item 3 is the current item 4.

The first record blamed compaction. The other lane then checked its own copy and killed
that: **it was compacted too, and its copy is current** — it had received a harness notice
that instruction files were re-read. Two sessions, one repository, both compacted on the
same day, on different versions of the rules.

So there is no signal a lane can read off its own history. *"I was compacted, so I may be
stale"* is unusable when the other compacted lane is fine; *"I was refreshed once, so I am
current"* is unusable because that notice is a past event and the file can move after it.
Reading the file on disk is the only thing that establishes currency, it is cheap, and
neither lane did it until an incident forced it.

`1f04153` records it; `e1191d7` supersedes it with the corrected premise.

Two consequences that outlive the incident:

* **A rule that does not reach a lane is not a weak rule — it is not a rule for that lane.**
  Any "I followed CLAUDE.md" covers the rules that lane was given, not the rules that exist.
* **Quote a rule, do not refer to it.** This surfaced only because a quoted rule sounded
  unfamiliar and was checked against the file in one command. A vaguer message would have
  been agreed with, and the stale-context defect would still be live.

The asymmetry is worth stating: either lane can read the repository and see what the other
*has*, and neither can see the other's *context*. When a peer looks careless, "does that
lane actually have this rule" is the cheap first question.

## State at the end of Part 6

| Gate | Result |
| --- | --- |
| `make gates` | **PASS** at `b185dcb` |
| cargo | 362 / 362 |
| pytest, torch venv | 1727 passed, 2 skipped |
| pytest, repo venv | 1466 passed, 20 skipped |
| `ledger/runs.jsonl` | chain verifies, 188 rows |

Pre-fix verification from Part 6 onward is done by **copying files, not `git stash`** —
restoring in a `finally` and asserting the restore byte-for-byte. The worktree has a live
second writer.

## Open, revised

1. **A nonzero-floor FT run at 512 steps**, with `--deterministic`. Unchanged, and still the
   most informative experiment left — the only regime where `span_weight` could matter.
2. **The row-level half of the cost fix.** `for_device` makes the *estimate* honest; the
   *row* is still zero. `ledger.py` defaults `cost_usd_per_hour` to 0.0 and nothing but
   `real_ft_run.py` passes it, while `trainer.py:870` already computes
   `control.cost.cost_for(wall)` into `TrainResult` and never into the row. Held by the
   concurrent lane, which raised it.
3. **The 4096 capacity arms are confounded** and should not be read as a capacity result.
   They move context *and* training-set size together — 382 train decisions against 727 —
   while the linear control finds +6.3pp on the same split and the models sit below the
   prior (128×2: −2.97pp, 0/8 seeds above; 256×4: −0.39pp, 0/8). An 8192 run subsampled to
   ~382 train decisions, **val untouched**, is the ~7 minutes that turns a confounded null
   into an attributable one. Agreed with the concurrent lane; not yet run.
4. **A driver-level test that a resume onto a different corpus order is refused.** Unchanged.
5. **Pre-register the resolvable difference** in every sweep runner's header, beside the
   seed count. Still not done anywhere.
6. **Whether `code_commit` should refuse `-dirty` for a run that will cost money.** A
   human's call: it changes what a ledger row means and belongs with whoever owns the launch
   procedure.
7. **Whether lanes should re-read CLAUDE.md on a schedule rather than on incident.** Also a
   human's call — nothing in this repository can detect the divergence, because it is a
   property of the harness.

## First command for the next lane

```bash
rg -n 'usd_per_hour=0\.0, n_gpus=0|"code_that_ran"' /Users/bharath/Code/research/qwen-decision/tools
```

Four `code_that_ran` sites across four runners and no zero-price literal anywhere. Both
halves of this afternoon in one screen — and both were one-of-four fixes that read as
complete until somebody counted.

# Part 7 — saying what a run could not have seen

Part 6 named a shape: a complete list written down, then one member of it fixed. This part
closes the open items with that shape in mind, and hits it once more — in my own work,
minutes after committing the message about it.

## 19. A sweep that could not have seen the effect it reports absent

`python/qd_train/power.py`. `resolvable_difference` returns the smallest difference a sweep
could detect; `resolution_state` turns it into a `TriState` against a target the sweep
declares **before** it runs.

`agreement.py` already made this argument for Cohen's kappa at n=50: *"Reporting 0.62 as a
pass, with no interval, is how an underpowered gate becomes a confident decision."* The same
sentence is true of a seed sweep, and it is the repository's own rule about a check that
could not run, moved up one level from a check to a comparison.

Three things in it are load-bearing:

* **The constant is computed, not remembered.** `z[0.975] + z[0.80]` = 2.801585, from
  `statistics.NormalDist`. So `alpha` and `power` are real parameters rather than a folklore
  2.8, and the assumption is visible to whoever reads it.
* **`against_known_reference` is required, with no default.** An arm against the
  majority-class baseline has one noisy side; against another arm it has two. The answers
  differ by 1.41×, and the flattering one is always the wrong one. Both cases occur here.
* **Missing `sd` or `target` gives `NotRun`, not silence.** A sweep that never stated what it
  was looking for must not be indistinguishable from one that looked and found nothing.

**What it does not claim.** The bound assumes `sd` is *known*. At 3, 5 or 8 seeds it is
estimated from those same runs, where the honest quantile is Student's t — about 14% wider
at n=5 (3.195 against 2.802, checked against scipy, which is *not* a dependency and so is
not imported). Every number it returns is therefore optimistic by construction, and both the
adequate and underpowered details say so.

It is also only the **second** question a sweep must answer. The concurrent lane found the
first: an arm that never fit produces no evidence about generalisation, so the resolution
question does not arise for it at all. That is in the test docstring and in `--prior-sd`'s
help, because that flag is exactly where someone would reach for a collapsed arm's spread.

**And then I shipped it for one of two runners.** `25fb639` wired `rung0_real_run.py` and
left `real_ft_run.py` without it — the fourth instance of the Part 6 shape in one afternoon,
committed minutes after writing that message. `89a03b2` is the correction, and the class
test now iterates both runners rather than asserting about one.

## 20. The enumeration tool reported a scope as a list

Both lanes adopted `test_every_tool_call_into_this_repository_binds_against_its_callee` as
the way to enumerate call sites before a signature change, instead of listing them by hand.
Its first real use turned every `make gates` run red.

The concurrent lane made `RunRecorder`'s `cost` required, took the list, updated all six
sites in `tools/`, and missed a seventh: `record_build_run`, **inside `ledger.py`**, which
every gate run goes through. The tool was right. Its scope — by its own docstring, calls in
`tools/` to names imported from this repository's packages — was narrower than the question,
and nothing in its output said so. `checked` and `unchecked` were both reported and both
were about `tools/`.

The non-obvious part, for whoever reaches for the obvious fix: **pointing the existing scan
at `python/` would not have caught it.** `record_build_run` constructs a class defined in its
own module, so there is no import for an import-driven scan to follow at any glob width.
`a1c7d54` adds a definition-resolving scan — a different mechanism, not a wider one. 570
in-package sites bind today, 199 unchecked and printed.

It is conservative on purpose: a name rebound anywhere in the file at any depth is skipped,
because a local shadowing a module-level function would report a failure that is not real,
and a test that cries wolf gets muted. Muted is worse than narrow.

I also nearly shipped a green run as evidence. The new test passed first time — but by then
the other lane had already fixed `ledger.py`, so the run proved nothing about the mechanism.
Injecting the defect shape is what turned it into evidence.

## 21. The driver half of the resume refusal

`test_backbone.py` holds `train_ft`'s refusal of a resume onto a different corpus order,
through the real step. What it could not hold is whether `real_ft_run.py` ever hands the
checkpoint over. A driver that dropped it would run from scratch while reporting a resume,
and every check the trainer has would pass **vacuously**, because there would be no
checkpoint to compare against.

Not hypothetical in this file: `real_ft_run.py:1566` carries the note *"The hook nobody
passed"* — `on_checkpoint` existed, was correct, and was never wired, so checkpointing was
dead for as long as nobody looked. Same shape, one argument over.

`31da931` adds three AST tests: `_train` passes `resume_from` to `train_ft`; **both** arms
pass it rather than just the first; and each site conditions it on the cell the checkpoint
was cut from, since one checkpoint handed to every cell resumes one and aborts the rest on a
mismatch that is not a defect. Verified by mutation — dropping it from one arm fails two,
dropping it from `_train` fails one.

## 22. What the e30 arms settled, and what they did not

Measured from `ledger/gh200-rung0-capacity-4096-e30-2026-09-21.jsonl`, n=8 per arm, against
the 0.484 majority-class prior:

| arm | train | val | sd | vs prior | floor | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `128x4:2layer` | 0.9944 | 0.3836 | 0.0441 | −10.04pp | 4.37pp | resolvable |
| `256x8:4layer` | 0.9532 | 0.3608 | 0.0558 | −12.32pp | 5.53pp | resolvable |
| `128x2` vs `256x4` | | | 0.0503 pooled | +2.28pp | **7.04pp** | **inside the floor** |

**Supported:** at e30 both arms fit, and both generalise resolvably *below* the prior. That
is the first held-out claim from this sweep that survives its own noise floor, and it is a
negative one. The concurrent lane's reading — memorisation at a converged budget, not
undertraining — is now measured rather than inferred, and it retires "the lever only helps at
an unconverged budget" from the other side: the budget converged and the result got worse.

**Not supported:** "the bigger model is worse." 2.28pp against a 7.04pp two-arm floor. It
fits the mechanism, the ordering is right, and it is inside the noise — which is precisely
the claim §19 exists to stop. Separating those arms would need roughly 76 seeds each.

Note the two-arm floor is 7.04pp where the against-prior floors are 4–5pp. That is the
`sqrt(2)` `against_known_reference` exists for, and it is the first live case where using the
wrong form would have moved a number that a reader acts on.

**Still not attributable:** 4096-vs-8192 moves the context window and the training set
together, 382 decisions against 727. The e30 arms remove "it needed longer" as an
explanation; they do not separate context from data. The 8192 run subsampled to ~382
training decisions with the validation set untouched is still the ~7 minutes that does, and
it is now more informative than when it was first proposed, because both ends of it are
regimes we have characterised.

## State at the end of Part 7

| Gate | Result |
| --- | --- |
| `make gates` | **PASS** at `31da931` |
| cargo | 362 / 362 |
| pytest, torch venv | 1754 passed, 2 skipped |
| pytest, repo venv | 1493 passed, 20 skipped |
| `ledger/runs.jsonl` | chain verifies, 193 rows |

Every number in §22 is read from a ledger row on the box, not from a lane's log parse. The
two lanes' figures agreed to 1e-4; that is reassuring about the parse and irrelevant to
rule 5, which is what a citation means.

## Open

1. **A nonzero-floor FT run at 512 steps**, with `--deterministic`. Unchanged, and still the
   most informative experiment left.
2. **The 8192 run subsampled to ~382 train decisions, validation untouched.** ~7 minutes,
   agreed with the concurrent lane, the only thing that attributes the 4096-vs-8192 swing.
3. **Seed the pre-registration flags for the next sweep.** The machinery exists on both
   runners and nothing has used it yet. `--prior-sd` wants a spread from an arm that FIT:
   128x2 e30's 0.0441, or an 8192 arm — not from a collapsed one.
4. **Whether `code_commit` should refuse `-dirty` for a run that will cost money.** A
   human's call.
5. **Whether lanes should re-read CLAUDE.md on a schedule rather than on incident.** A
   human's call; nothing in this repository can detect the divergence.

## First command for the next lane

```bash
rg -n 'prior-sd|sweep_can_resolve' /Users/bharath/Code/research/qwen-decision/tools
```

Both sweep runners can now state what they could have seen, and neither has been asked to.
The flags are the cheapest thing on this list and the one that decides whether the next
null means anything.

# Part 8 — running the thing instead of asserting it

## 23. The pre-registration flags, exercised rather than described

Part 7 §19 added `--prior-sd` and `--target-difference` to both sweep runners and checked
them with **source-level** tests. That is not the same as knowing the metric reaches a row.
It is the gap `real_ft_run.py:1566` sat in — *"the hook nobody passed"*, where
`on_checkpoint` existed, was correct, was never wired, and nothing noticed.

`61130da` closes it with two real rung-0 runs on this Mac: cpu, 2 seeds, 266 train and 89
val decisions from this repository at HEAD, in
`ledger/mac-rung0-prereg-2026-09-21.jsonl`. Both tri-state branches exist as rows:

| flags | recorded |
| --- | --- |
| omitted | `state: not_run` — *"no prior sd and target difference supplied, so what this sweep could resolve at n=2 is unknown"* |
| supplied | `state: ran`, `passed: false`, `value: 0.0874`, `n = n_total = 2`, **UNDERPOWERED** |

2 seeds at sd 0.0441 resolve 8.74pp against a 5pp target. Correct, and the kind of thing a
row should have been saying all along.

**The prior sd could not come from the run itself**, and the reason is the caveat written
into `--prior-sd`'s help an hour earlier: both of that run's seeds collapsed to a constant
predictor, so its own spread is 0.0% and would have claimed *infinite* resolution. The
warning demonstrated itself on its first use. The 0.0441 used instead is from the 4096 e30
128×2 arm, which fit.

`cost_usd` is 0.0 on those rows and that is honest — `for_device` prices cpu at zero because
this machine is already bought. `find_forks` reports 1 root group and 1 known divergence
with the new ledger file present: independent chains group under one root, so a new file
joins it rather than adding a second.

## 24. Five for five — every instance was found by the other lane

Worth recording as its own observation, because it is no longer anecdote. The shape named in
Part 6 — *a complete list written down, then one member of it fixed* — occurred five times
on 2026-09-21:

| # | defect | shipped by | found by |
| --- | --- | --- | --- |
| 1 | the zero-price literal: `for_device` fixed 1 of 4 tools, its own docstring saying "four" | coordinator | coordinator, next day's read |
| 2 | `_what_ran` as two private functions in 1 of 4 runners | coordinator | concurrent lane |
| 3 | the call-site check's scope read as a complete list | — (latent) | coordinator, via the other lane's red gate |
| 4 | `power.py` wired into 1 of 2 sweep runners | coordinator | concurrent lane |
| 5 | an external corpus-thinning script rebuilt while `--train-subsample` already existed | concurrent lane | concurrent lane, on going back to the rule |

**Not one was caught by the lane that shipped it, in the moment of shipping it.**

> **Corrected, same day.** A sixth instance turned up after this table was written, and it
> *was* caught by the lane that shipped it: `code_that_ran` hashed `qd_train` plus the
> invoking tool and not the modules that tool imports — missing `run_cost.py`, which
> decides whether rule 4 can fire, and `ft_toy_run.py`, which owns the floors every FT gate
> is scored against. Both were added the same day the metric was. So the claim above is too
> strong. The refinement that survives: it was not caught by care during the work — care is
> what both lanes were exercising throughout — but by a deliberate second pass hunting for
> this specific shape by name, *after* five instances had been written down. The shape is
> self-catchable, and only on purpose. #4 was
committed minutes after the commit message describing the shape, by the author of that
message. #5 was written by the lane that had been quoting the check-for-the-tool-first rule
all afternoon.

That is an argument for the two-lane arrangement rather than for either lane being more
careful. Care is what both lanes were exercising at the time. What actually worked was a
second reader with different context and no stake in the change — and, in #3 and #5, going
back to a written rule *after* the work rather than before it.

The corollary is uncomfortable and should be stated: a single-lane version of today would
have shipped all five, each under a commit message explaining why it was thorough.

## State at the end of Part 8

| Gate | Result |
| --- | --- |
| `make gates` | **PASS** at `61130da` |
| cargo | 362 / 362 |
| pytest, torch venv | 1758 passed, 2 skipped |
| pytest, repo venv | 1493 passed, 20 skipped |
| `ledger/runs.jsonl` | chain verifies, 194 rows |
| forks across `ledger/` | 1 root group, 1 known divergence |

## Open

1. **A nonzero-floor FT run at 512 steps**, with `--deterministic`. Unchanged, needs the box.
2. **The 8192 run at `--train-subsample 0.5`**, validation untouched, with the linear control
   on the same thinned training set. Held by the concurrent lane, pre-registered. One number
   to check before it runs: `--prior-sd 0.0200` must come from an arm that **fit**. The e10
   4096 sds (0.0212, 0.0111, 0.0067) are collapse fingerprints, and at 0.0200 vs 0.0441 vs
   0.0558 the same sweep reads adequate, marginal, or blind to its own target.
3. **Whether `code_commit` should refuse `-dirty` for a run that will cost money.** Human's
   call.
4. **Whether lanes should re-read CLAUDE.md on a schedule rather than on incident.** Human's
   call; nothing in this repository can detect the divergence, as Part 6 §18 showed.

## First command for the next lane

```bash
/Users/bharath/.venvs/ml/bin/python -c "import json;[print(json.loads(l)['metrics']['sweep_can_resolve']) for l in open('/Users/bharath/Code/research/qwen-decision/ledger/mac-rung0-prereg-2026-09-21.jsonl')]"
```

Four rows: two that say what the sweep could have seen, two that say nobody asked. That
distinction did not exist in this repository twelve hours ago, and it is the one that decides
whether the final train's nulls mean anything.

# Part 9 — a readiness audit, rule by rule

"Ready for the final train" is not a feeling. This is what was checked, how, and what it
returned — so the next lane can disagree with the evidence rather than with the conclusion.

Every row below was run on 2026-09-21 against the tree at `33fc3f0`.

## The rules that bind a full train

| Rule | Enforced where | Checked how | Result |
| --- | --- | --- | --- |
| **2** — gates and kill criteria are read-only | `run_control.py` refuses a cap above `MAX_CAP_S` | `test_a_cap_above_the_programs_own_cap_is_refused_as_read_only`, `test_the_approval_threshold_is_a_closed_boundary_and_one_cent_either_side_of_it` | pinned, both sides of the boundary |
| **3** — held-out data is never read by a training process | `data_access.assert_path_not_held_out`, and it **refuses an empty `held_out_roots`** rather than passing vacuously | refusal present at `data_access.py:81`, mirrored at `artifacts.py:1042` for a shard whose header honestly declares `split="train"` | enforced, fails closed |
| **4** — every 8×H100 job carries a cap, an estimate and a human yes | `CostEstimate.requires_human_approval` + `RunControl`'s refusal | **exercised live at full-train parameters**, below | now enforced; was dead this morning |
| **5** — nothing is green that was not run | `TriState` throughout; `NotRun` carries a reason | the day's work is the evidence: §23, §19, `_floor_state` | discipline held, and extended |
| **8** — `quick` runs promote nothing | `ledger.py:955` **raises** on a quick row in a promotion | `f"{r.row_id}: marked quick ({r.quick_reason}); quick runs cannot promote"` | enforced by refusal, not convention |

## Rule 4, at the parameters that matter

Rule 4 exists for one event: the full train. So it was run at that shape rather than at a
test's shape — 8×H100, 24h cap, priced from a real list:

```
lambda-8xH100: 8 GPU(s) at $23.92/h per instance ($2.99/GPU/h x 8),
capped at 24.00 h -> $574.08 at the cap -- NEEDS A HUMAN YES
```

and `RunControl` **refuses to construct** without `approved_by`:

> rule 4: this run needs a human yes before launch and has none.

The same run, through the literal every tool carried this morning:

```
local-cuda: 0 GPU(s) at $0.00/h per instance, capped at 24.00 h
-> $0.00 at the cap -- no approval required
```

`requires_human_approval` was `False`. **For a $574 8×H100 job, at any cap.** That is what
Part 6 §16 fixed, stated at the size where it would have mattered.

## What this audit does not cover

Named, because a capped check reported as complete coverage is the failure this repository
spent the day finding:

* **The experiments.** Engineering readiness is not scientific readiness. The two open runs
  (Part 8) decide whether the *conclusions* a full train would rest on are attributable;
  nothing above speaks to that.
* **The GPU suites.** They cannot run in this lane's environment and are reported as **not
  run**, not as passing — rule 5, applied to this audit.
* **`qd_wire`, `qd_label`, `crates/`.** Not on the train path, so not audited here. That is
  a scope statement, not a clean bill.
* **Rules 1, 6, 7, 9, 10.** Process rules about navigation, the nested tessl crate, product
  wiring, parity fixtures and handoffs. Unchanged today and not re-verified.

## State at the end of Part 9

| Gate | Result |
| --- | --- |
| `make gates` | **PASS** at `80cddb8`; see the note for the tree right now |
| cargo | 362 / 362 |
| pytest, repo venv | 1499 passed, 21 skipped |
| pytest, torch venv | 1763 passed, 2 skipped, **2 failed** — both the other lane's |
| `ledger/runs.jsonl` | chain verifies, 196 rows |
| forks across `ledger/` | 1 root group, 1 known divergence |

Those two failures are named rather than rounded away. Both are repo-**wide** scans —
`test_the_repository_has_no_ruff_findings` and
`test_no_docstring_names_a_function_that_does_not_exist` — and both point at one file:
`python/tests/test_runner_cost_end_to_end.py`, untracked and mid-write in the concurrent
lane. `--ignore` does not suppress them, correctly: a scan of the tree should see the tree.
Nothing in this lane's committed work fails, and this lane did not touch that file.

The first line of this table is therefore the weaker claim it looks like: `make gates` last
returned PASS at `80cddb8`, before that file existed. It is recorded that way rather than
re-run and reported green by excluding the thing that makes it red.

## The honest summary

The **engineering** is ready for a full train in the sense that the rules which protect it
now have teeth that were verified rather than assumed, and the rows it writes can say what
produced them, what they cost, and what they could have seen.

The **science** is not ready, and the gap is two runs long: the 512-step nonzero-floor FT
run, and the 8192 curve at `--train-subsample`. Until those land, the capacity question is
confounded and the 4096 arms cannot carry a conclusion.

Those are different claims and should not be reported as one.

# Part 10 — the oldest open item was asking to reproduce a fixed bug

## 25. "A nonzero-floor FT run at 512 steps", withdrawn

It has been item 1 on every open list since 2026-09-20, described as *"the most informative
experiment left, and the only regime where `span_weight` could still matter"*, and carried
all of today as blocked on GPU time. Both halves of that were wrong.

**It was never GPU-blocked.** Floors are conditional entropy over labels — CPU arithmetic.
Read on the box under `CUDA_VISIBLE_DEVICES=""` and `nice -19`, taking nothing from the
other lane's run:

| shard set | batches | letter floor | span floor | `corpus_rev` |
| --- | --- | --- | --- | --- |
| `shardset` | 111 | 0.000000 (109) | 0.000000 (4) | none |
| `shardset-v2` | 108 | 0.000000 (101) | 0.000000 (8) | none |
| `shardset-v3` | 108 | 0.000000 (101) | 0.000000 (8) | none |
| `shardset-v4` | 108 | 0.000000 (101) | 0.000000 (8) | `0632f693` |
| `shards-rerun` | 111 | 0.000000 (109) | 0.000000 (4) | none |

There was nothing on that box to run it against. Freeing the GPU would not have unblocked
it — and the failure that prevents is specific: `shardset-v4` is the newest and the only one
carrying a `corpus_rev`, so it is what anyone would have reached for. Both floors come back
0.0, the gates pass, the row looks exactly like the experiment having been run.

**And the regime it names was a defect.** `tools/real_tokenizer_pipeline.py:271` documents
it in place:

> Until 2026-09-20 both rows of this pair asked `f"Which line of {name} states the rule?"`
> over the same passage, so the prompt was byte-identical and only the gold differed — one a
> real line span, the other `noul`. That is a corpus no model can score above chance on, and
> the REALFT lane measured exactly that: the span loss floor came out at 0.693147 = ln 2,
> the entropy of a fair coin, and span accuracy was capped at 39 of 78 permanently.

Already recorded twice — `GAP-REALFT-CONTRADICTORY-SPAN-SUPERVISION` and
`GAP-DATA-NOTHING-REFUSES-TWO-ROWS-THAT-CONTRADICT` — and guarded by `test_mixture.py:613`
and `test_pipeline_corpus.py:10`. The two questions were made to differ. **The floor is
0.0 now because the bug is gone.**

So the item asks to test a regime whose only known instance was 39 prompt-identical pairs,
removed deliberately and now fenced by tests. Running it as written means re-introducing
them, and the result would be the fair-coin floor again — box time spent measuring a bug
this repository already understands.

Withdrawn as specified. What survives of the intent needs naming, because the two things it
could mean are not the same:

* **genuine ambiguity** — a hard task where the gold is uncertain given the context. Nothing
  in the plan calls for one, and `real_ft_run`'s own docstring scopes the lane to *whether
  `train_ft` executes on a real shard set*, not to evaluating a model.
* **contradictory supervision** — two rows that disagree about the same question. That is
  the defect.

Anyone reviving this should say which. They are a hard corpus and a broken one.

## What this does to the readiness position

Part 9 closed with *"the engineering is ready, the science is two runs short."* One of those
two was not a run. The position is now:

* **Engineering:** ready on the audited terms of Part 9 §1–5, **with one gap found after
  that audit and not covered by it** — see the correction below.
* **Science:** **one** run short — the 8192 curve at `--train-subsample`, held and
  pre-registered by the concurrent lane, which attributes the 4096-vs-8192 swing between
  context window and training-set size.

That is a smaller gap than this morning's, and it closed by reading rather than by running.

## 26. Correction to Part 9: the audit missed where the recorder is entered

Raised by the concurrent lane after Part 9 landed, and it is in this lane's file.

`ledger.py`'s docstring promises *"a run that dies without a row is the one failure this
module is built to make impossible."* That is a property of the recorder's **block**, and
`rung0_real_run.py` enters the block after training is over — `for seed` at :951,
`train_once(...)` at :952, `with RunRecorder(` at :1010, all at one indent. Verified
structurally here; the concurrent lane measured the consequence, a real SIGTERM 8s into
training giving returncode -15 and zero rows.

The second-order cost is the one that matters: spend summed from the ledger is short by
exactly the runs that were killed, which are the ones that ran longest. **A cost ledger
whose error is correlated with the quantity rule 4 exists to control.** Bounded for rung 0 —
the loop is per seed and finished seeds have already written — so a kill costs the seed in
flight, not the sweep.

**And the fix does not cost comparability**, which is what the other lane deferred it for.
`ledger.py:1189` and `:1215` both read `self.wall_clock_s` at `_finish` time, not at
construction. So entering the block before training with `wall_clock_s=None` and assigning
the measured training duration on success gives a successful row that is identical in
meaning to today's — `source="caller"`, the training loop alone — while a killed run
writes a row carrying the recorder's lifetime under `source="recorder"`. The field added
this morning is what keeps those two distinguishable.

Not landed: the file is about to be run by the other lane's curve, and changing it an hour
beforehand is its own risk whatever the semantics say. Theirs to time.

**What this says about Part 9.** The audit named its exclusions — the experiments, the GPU
suites, three packages, five process rules — and recorder placement was not among them. So
it is the one-of-N shape a seventh time, in the artifact written to guard against it: an
enumeration of what was not checked, which was itself incomplete. An audit's exclusion list
is a claim like any other.

## 27. Every capacity number today is on a row with no duration and no price

Raised by the concurrent lane while re-verifying its prior; widened and cause-checked here.
Read from the box directly — all four rung-0 capacity sweeps, 59 rows:

| sweep | rows | `wall_clock_s` | `cost_usd` | `wall_clock_source` | `code_that_ran` |
| --- | --- | --- | --- | --- | --- |
| `4096-e10` | 24 | 7.35e-05 – 1.18e-04 | 0.0 | absent | 0 |
| `4096-e30` | 22 | 7.48e-05 – 1.26e-04 | 0.0 | absent | 0 |
| `sw005` | 5 | 7.38e-05 – 1.12e-04 | 0.0 | absent | 0 |
| `sw005-nondet` | 8 | 7.56e-05 – 8.83e-05 | 0.0 | absent | 0 |

Seventy-three to one hundred and twenty-six **microseconds**, for seeds the logs record in
minutes. So every capacity number this project produced today — including §22's table and
the 0.0197 the curve is pre-registered against — sits on a row that says the run took no
time and cost nothing. **The accuracies are real; the duration, the price and the
provenance are recorded nowhere.**

**The cause is a stale box, not the current code, and the two must not be conflated.**
`rung0_real_run.py:1019` passes `wall_clock_s=float(run["wall_clock_s"])` — the measured
training duration — and has since `a409895` at 08:40 today. The box is a clone pinned at
`8b9df39` (2026-09-20) with files copied over it, and it was last copied before that
commit. Its rows therefore fall back to the recorder's own lifetime, which is the time to
write metrics.

So §26's gap and this are different faults that happen to share a symptom. §26 is a
present-tense defect in the current source: the recorder is entered after training, so a
*killed* run writes nothing. This is a stale deployment: an *uninterrupted* run writes a row
whose duration is the reporting time. Fixing §26 does not fix this, and syncing does not fix
§26.

After the sync, the curve's rows will be the first rung-0 rows in this project carrying a
true duration, a true price and a source fingerprint. That is worth more than a bookkeeping
improvement: it is the first rung-0 evidence that could be re-priced or reproduced from its
own row.

# Part 11 — the kill-row gap, landed; and what was living on one machine

## 28. `f93b22c` — the training is now inside the recorder that survives it

Held since §26 on one caveat, not on the semantics: the concurrent lane was about to run
`tools/rung0_real_run.py` on the box, and changing a file beforehand is its own risk whatever
the argument says. That lane reported the sync done and the curve running at 16:13:51Z, with
the closure digest matching on both sides, so the box has its own copy and landing here
cannot reach it. Landed at `f93b22c`.

**The reordering.** `protocol` is built from `args`, `config`, `corpus_hash` and `seed` and
never from the training result, so it can be computed before the run. It moves above the
recorder; `train_once` moves inside the block. That is the whole structural change.

**What keeps the rows comparable.** `_finish` reads `self.wall_clock_s` at exit
(`ledger.py:1189`, `:1215`), not at construction. So the recorder is entered with
`wall_clock_s=None` and the new `RunRecorder.measured()` states the training duration the
moment `train_once` returns. A successful row carries the same quantity under the same
`wall_clock_source="caller"` as the 53 rows written before this. A run that dies first falls
back to the recorder's lifetime under `"recorder"`.

**Measured both ways**, on the same corpus and the same flags:

| | pre-change | post-change |
| --- | --- | --- |
| SIGTERM 2.0s into a ~12s loop | dead of the signal, **ledger file never created** | row exists, `status="killed"` |
| that row's source | — | `"recorder"` |
| that row's cost at $18/h | — | `> 0`, and equal to the rate applied to its own duration |

**Eleven failing cases against the pre-change tree, from seven new tests** — both numbers,
because they are not the same number. Six tests go in `test_wall_clock.py` beside the
argument they extend, and one of those is parametrised over four non-durations, so six
tests are ten cases; the seventh is the SIGTERM test in `test_runner_cost_end_to_end.py`.
(`f93b22c`'s own message says "eleven tests" where it means eleven cases.) An eighth test
passes before *and* after, on purpose: it pins that a successful row still carries
the training loop rather than the block, which now also spans model construction and three
evaluations. `wall_clock_source == "caller"` alone would not have shown that — it says
`measured()` was reached, not that it was handed the right quantity.

`measured()` also refuses after the row is written, refuses a second statement — a recorder
told twice keeps only the last, which is how a loop's total becomes its final iteration's —
and refuses a figure that is not a duration, through `_is_measured_duration`, which the
constructor now asks too rather than repeating the `nan < 0` reasoning in two places.

**One thing found by measuring that would have made the test pass for the wrong reason.**
Python block-buffers a pipe. A subprocess test that waits for a progress line before
signalling does not wait at all: the line does not leave the child until it exits. Measured
— **14.365s** to see the line on a run that takes 14.4s, against **0.540s** with
`PYTHONUNBUFFERED=1`. Without that variable the signal goes to a corpse and the test passes
against the *unfixed* tool by reading the row a completed run wrote. That is not a detail of
this test; it is the shape of every "wait for the child to get somewhere, then interrupt it"
test anyone writes here.

Gates PASS at `90de7ff`, row `da36b672-66f1-4762-89a5-f5dab8e4cdd7`: cargo 362/362, pytest
1510/1531, pytest-torch 1780/1782. Recorded twice on purpose — `1fb10659-…` covers the same
code at `7f191bd-dirty`, before the change had a commit, and `90de7ff` says so in its message
rather than leaving a reader to infer what `-dirty` meant.

## 29. 37 rows were living on the rented box and nowhere else

The concurrent lane's `7f191bd` brought back 61 capacity rows that existed only on the GH200,
and asked whether anything else did. Answered by **row id**, not by filename — a file can
exist on both sides with different contents, and the box's copy is the one that has been
appended to. Every row in the box's `ledger/` compared against the union of every row id in
this repo's:

| | rows |
| --- | --- |
| only on the box | **37** |
| of those, the concurrent lane's live learning curve | 15 |
| of those, finished and unclaimed | **22** |

The 15 are being appended to right now and are that lane's to bring back. The 22 are here at
`4db2ef6`: `gh200-determinism` (8), `gh200-seedcheck` (4), `gh200-smoke` (4),
`gh200-rung0-repro` (2), `gh200-rung0-reprodet` (2), `shardset-v3` (1), `shardset-v4` (1),
spanning 06:17Z to 12:57Z. Each chain verified with `qd_train.ledger verify` before being
copied in — each on its own, which is the check that applies: these are seven independent
chains, not seven copies of one.

**Two things a reader should not have to infer.**

`find_forks` over the whole directory reports a fork, and reported the same one before this
commit: two fork points, one non-root — `gh200-2026-09-20.jsonl:157` against
`runs.jsonl:157` after `364b7d12…`. Run with and without the seven new files, both numbers
are identical; a new independent chain shows up only as another root, which is what
`ForkPoint.is_root` exists to distinguish. The non-root one is
`GAP-LEDGER-NO-STORY-FOR-A-CHAIN-FORKED-ACROSS-TWO-MACHINES`, opened 2026-09-20, still open.

And **all 22 carry `wall_clock_source: null` and `cost_usd: 0.0`** on a machine billed by the
hour. All 22 are `quick=True` and `completed`. They predate `a409895`, and the runner they
came from is the one `f93b22c` just fixed. They are preserved, not believed: the rows say
what they were written with, and what they were written with priced a GPU hour at zero.

# Part 12 — fixing the shape rather than the site

## 30. `71bdd2f` — the other eight places §28's defect could live

§28 fixed `tools/rung0_real_run.py`. That is one instance of a shape **nine call sites can
take**, and this repository has found "a complete list was written down and then one member
of it was fixed" seven times in a day. So the inventory is pinned rather than the site.

Eleven `wall_clock_s` arguments reach a recorder — two forwarding inside `_recorder`
helpers, nine deciding. Both numbers are asserted, because a helper that stops forwarding
and a decision that disappears are different failures. Every deciding site that hands over a
duration measured *before* the block is now named with the reason its work is safe outside
one, and a new one fails until somebody writes that reason down.

**The exposure, measured from rows rather than reasoned about.** After §28 the only
*billed* work still outside a block is `real_ft_run.py`'s verdict decode:

| | seconds | rows |
| --- | --- | --- |
| ft rows (block wraps `train_ft`) | 40919.1 | 194 |
| verdict rows (decode outside) | **0.1** | 170 |

0.0003% of the wrapped time, and a killed process loses only the addendum — the parent ft
row is already written by then. The other four non-wrapping sites are structurally
unbillable: two pass `cost=None`, which `RunRecorder` accepts *only* on a local device, and
`rung0_toy_run.py` prices through `CostEstimate.for_device` with no rate, which refuses
anything but cpu and mps. A second test checks those claims against the source, so a tool
that gains a rate has to re-argue its entry instead of inheriting it.

Both injections verified: reverting `rung0_real_run.py` fails the first test with that file
named; giving `rung0_linear_control.py` a `--usd-per-hour` fails the second.

**And the second injection found a hole in the test's own first draft.**
`add_argument("--usd-per-hour")` produces `args.usd_per_hour` and puts neither underscore
spelling in the file, so a check for the identifier alone never fires on the way a rate
actually arrives — it passed against the injected flag. That is a check that could not run
reporting the same result as one that ran and passed, written by the lane that spent the day
saying so. Both spellings are checked now, and the fact that only the injection caught it is
the argument for injecting.

## 31. `45291c3` — a row can now name its own arm

`GAP-A-ROW-CANNOT-SAY-WHICH-CURVE-POINT-IT-IS`, raised by the concurrent lane while
labelling a learning curve, and left for this one because it is this lane's file and a
schema question.

`recipe_hash` makes two runs of different settings incomparable, which is its job, and it
makes neither of them readable. Every field a runner varies went into the hash's input and
was stored nowhere. Labelling three curve points took re-hashing four candidate
`train_subsample` values against the other seven pinned at the launch command's values:
correct, and it recovers the label **from the log**, by a longer route, rather than from the
row.

`LedgerRow` gains `recipe`, `RunRecorder` gains `recipe=`, and every runner builds its
recipe once, names it, and hands that one object to both the hash and the row. Seven sites,
not one — `rung0_real_run.py`, `real_ft_run.py`, `rung0_toy_run.py`, `ft_toy_run.py`,
`rung0_linear_control.py`, `real_tokenizer_pipeline.py` and `record_build` — five of them
with the dict written inline *inside* the `hashlib.sha256(...)` call, where nothing else
could reach it. The two toy runners get a `_recipe()` function rather than a second literal:
a recipe built twice can drift, and a row whose stored recipe does not hash to its own
`recipe_hash` is worse than a row with none, because it reads as an answer.

**The identity does not move, and that was measured.** The tool ran before and after on the
same corpus and flags; `recipe_hash`, `backbone_commit`, `data_snapshot_hash` and
`protocol_hash` all came back identical, and the new row's stored recipe re-hashes to its
own `recipe_hash`. This mattered: the canonical `separators=(",", ":")` spelling **would**
have moved it — `b5e6c489…` against `705dc18f…` on the same dict — and the 61 capacity rows
and the curve running on the box at the time were written with the other one.

Which is the next gap, opened rather than closed: **`GAP-SIX-SPELLINGS-OF-ONE-RECIPE-HASH`**.
Six independent implementations of "hash a recipe", two of which disagree on identical
dicts. `_canonical` at `ledger.py:260` is the owner that already exists, and five of the six
do not call it. Nothing is currently wrong — each tool is internally consistent, and
cross-tool comparisons are foreclosed by `backbone_commit` anyway. The damage would come
from a future refactor tidying one spelling toward another: every row afterwards
incomparable with every row before, both files verifying, nothing saying so. Unifying it is
a **declared break**, not a tidy-up, and it is not this lane's to declare mid-experiment.

Three tests, all failing against the pre-change tree. Two are rewrites — they read the text
between `recipe_hash=hashlib.sha256(` and `).hexdigest(),` and broke when the recipe was
given a name, a refactor that changed nothing about what is hashed. They now read the recipe
off a row the tool actually wrote and re-hash it, which asserts the same thing and survives
the next such move. The third pins the eight keys as a **set**, so a field silently dropped
from the recipe fails even though every other test stays green: it would still hash, still
identify, and quietly stop separating the arm it was added to separate. Both failure modes
injected — no recipe on the row, and a recipe drifted from its hash.

`recipe` enters no hash. `null` means unrecorded; `{}` is refused, because a run with a
`recipe_hash` always had settings. `docs/ledger-schema.md` gains the field, the rules, and
the `wall_clock_source` row it had been missing since `a409895`.

## 32. The first row that says what it ran

`b3878ad`, row `4f192818-694f-413b-b557-f6c85b37537c` at `8ffc0cc`, clean tree:

```
commands:  cargo test --workspace
           .venv/bin/python -m pytest python/tests -o addopts=
           uv run --with pytest --with hypothesis python -m pytest python/tests
toolchain: cargo 1.98.0 / CPython 3.13.14 (.venv) / ruff 0.16.8
```

`Protocol.for_build` has always hashed exactly those two things, so *"did the gates run the
suite I think they did"* was answerable only by re-hashing candidates. Gates: cargo 362/362,
pytest 1523/1544, pytest-torch **1794**/1796 — up 14 from 1780 at `90de7ff`. Chain verifies
over 200 rows.

**1 of 200 rows carries a recipe**, which is the right number and worth printing: the field
was added today and enters no hash, so the other 199 read back as `null` — not recording
their settings, rather than having run without any.

## 33. `081889d` — the correction a row quoted was true of one case and printed on all of them

`resolution_state` — added by this lane earlier today — printed **"~14% at n=5"** on every
row it wrote, regardless of `n` and regardless of whether the comparison was against a fixed
reference or another arm. It reached an audit that way.

**How it was found, which is the part worth keeping.** The concurrent lane read the figure
off a one-sample row at n=8, tried to reproduce it, could not, and declined to put any number
derived from it into an `AUDIT/` file. This lane had offered them *"roughly 9% at n=8"* off
the top of its head; they declined that too, and said why: *"putting it in an AUDIT file on
your word would be me laundering inferred into verified, on the one number the whole
resolved result rests on."* They were right on both counts. The 9% was not a case at all.

**All three numbers, reproduced from both ends before anything was changed.** The convention
is the ordinary central t on **both** terms — not a noncentral t, which was the reasonable
guess:

| case | df | constant | vs normal 2.8016 |
| --- | --- | --- | --- |
| two-sample, n=5/arm — *what the module quoted* | 8 | 3.1949 | **14.0%** |
| one-sample, n=8 — *the row it was printed on* | 7 | 3.2607 | **16.4%** |
| one-sample, n=5 | 4 | 3.7174 | **32.7%** |
| two-sample, n=8/arm | 14 | 3.0128 | **7.5%** |

The correction on the row that was read is **larger** than the row claimed, on the side that
says a margin is smaller than it looks. One sentence covered a spread of 7.5% to 32.7%, and
the number printed was the smallest of them.

**It was not a SciPy question.** Both lanes concluded it was one for the human under the
no-new-dependency rule. It is not: the regularised incomplete beta by Lentz's continued
fraction is forty lines of `math`. `t_quantile` is pinned against the standard two-sided 95%
table at sixteen degrees of freedom from 1 to 120 — worst deviation **0.00045** against a
table printed to three decimals — and against `NormalDist` in the limit, because a continued
fraction that is subtly wrong returns plausible numbers and every figure the module now puts
on a row comes out of it. It raises rather than returning its last iterate: a quantile that
did not converge is not a quantile.

Before and after, on the row that was actually read:

```
before: ... The bound is optimistic on top of that, because it assumes sd is known
        rather than estimated from these runs
after:  ... It is worse than it looks: sd is ESTIMATED from these 8 runs, not known, so
        the honest quantile is Student's t on 7 degrees of freedom and this bound is
        16.4% too small: 0.0333, not 0.0286
```

The corrected bound is a **number**, not a percentage for the reader to apply themselves —
one more step is one more place to get the case wrong, which is exactly what happened.

Twelve tests. Beyond the arithmetic they pin that the correction falls with `n`, falls again
with a second sample, never reaches 1.0 at finite `n`, and is **refused at n=1**, where there
are no degrees of freedom to estimate a spread from and any number would be invented for a
row whose sd came from nowhere.

**It does not change the concurrent lane's conclusion.** Their resolved point is +3.78pp
against a 2.86pp known-sd bound; corrected, the floor is 3.33pp, so it clears by 14% rather
than 32%. Still clears.

### The shape all three of today's best findings share

| found | by | what it had only ever been seen doing |
| --- | --- | --- |
| a sync guard matching its own `pgrep` | positive control | refusing |
| a `usd_per_hour` check argparse spells with hyphens | injection | passing |
| a correction constant true of one case | a peer refusing to cite it | printed |

None of the three would have been found by re-reading. Re-reading confirms the check exists,
which was never in doubt.

## 34. `6efd874` — the eighth instance, found inside a test written to catch it

`test_every_runner_records_which_sources_produced_its_row` opens by naming the defect it
exists for: `real_ft_run.py` had the closure digest, `rung0_real_run.py` did not, and the
GH200 rows' provenance had to be reconstructed by hand with `sha256sum`. Its scope was then
**four hardcoded names**, written when four runners were the ones in view.

**Six tools write ledger rows.** The two it omitted recorded no digest at all:

| tool | why its rows are the ones that need it |
| --- | --- |
| `rung0_linear_control.py` | its one row is a control arm whose whole purpose is interpreting another run |
| `real_tokenizer_pipeline.py` | writes the shard sets every FT run downstream trains on |

*(An earlier version of that first cell said this tool "runs on the same rented box, beside
the arm it exists to interpret." It does not — see §36. The gap was real; the reason was
invented.)*

Both green throughout. A hardcoded scope is the defect that test catches, one level up from
the code to the check — the eighth instance today of *a complete list written down and then
one member of it fixed*, and the first found inside a test written for exactly that shape.

**Why it is not bookkeeping.** `code_commit` is `<sha>-dirty` for any uncommitted change at
all, and on the box — synced by *copying* files into a clone pinned at an old commit — that
is its permanent state, not an exception. The concurrent lane's log reads `8b9df39 85
file(s) dirty` on every row it writes. One bit standing for 6894 insertions across 72 paths
distinguishes nothing; the closure digest is the field that says what ran, and two
row-writing tools had none.

**Both scopes are derived now.** `_row_writing_tools()` (constructs a recorder) and
`_pricing_tools()` (constructs a `CostEstimate`) — deliberately two functions, because the
four pricing tools are a proper subset and the two new ones pass `cost=None` on a local
device, where requiring `for_device` would be requiring a price for nothing billed. The
pricing four happen to be exactly the four that were hardcoded, which was measured rather
than assumed: `real_tokenizer_pipeline.py` and `run_cost.py` mention `CostEstimate` without
building one, so a substring scope would have pulled in two tools with no rate to state. It
was right because it was current, not because anything kept it so — and its twin in the same
file was four names that had stopped being all of them.

Both derivations assert their own size. A rename that made either match nothing would
otherwise read exactly like a clean repository.

## 35. `GAP-SIX-SPELLINGS-OF-ONE-RECIPE-HASH`: still open, no longer armed

§31 recorded it as open **by choice** — unifying six spellings renames every future
`recipe_hash` and breaks comparability with the 61 capacity rows and 24 learning-curve rows
of today, so it is a declared break and not a tidy-up. That reasoning stands. What it left
behind was a live landmine: the entire danger is a future refactor tidying one spelling
toward another, after which every row is incomparable with every row before, both files
still verify their chains, and **nothing says so**.

Leaving a gap open is a decision about *when* to fix it. It is not a decision to leave it
able to go off quietly. Those are separable, and only the first one was this lane's to make.

So each of the six tools' `json.dumps` keyword arguments is now read through the AST —
following `recipe_hash=` into the expression, or into the `digest` helper the toy runners
route it through — and pinned to the spelling its existing rows were written under:

| spelling | tools |
| --- | --- |
| `sort_keys=True` | `rung0_real_run`, `rung0_linear_control`, `real_tokenizer_pipeline` |
| `sort_keys=True, separators=(",", ":")` | `real_ft_run`, `rung0_toy_run`, `ft_toy_run` |

Structural, not textual: reformatting the call does not move it, changing what it serialises
does. The split is **asserted rather than described** — both spellings are run over one probe
dict and required to produce exactly two distinct digests, so if they ever agreed the concern
would be shown imaginary rather than assumed real, and the six could be unified for free.

Injected and confirmed: tidying `rung0_real_run.py` toward the other spelling fails with
*"Every recipe_hash it writes from here differs from every one it has written, both verify,
and nothing in the ledger says the family changed. If that is intended, update this entry
and declare the break in a handoff."*

Failing there is not a prohibition. It is the sentence that was missing.

## The three open items, stated precisely

The previous part listed three and called them all not-this-lane's. Two of those were
partly wrong, and the correction is the point:

1. **`GAP-SIX-SPELLINGS-OF-ONE-RECIPE-HASH`** — *unifying* it is a declared break and stays
   the next lane's or the human's. *Guarding* it was never a comparability question and is
   done (§35). "Open by choice" was the right call about the fix and the wrong place to
   stop.
2. **`code_commit` and `-dirty`** — whether to *refuse* a paid run on a dirty tree is a
   policy call with real cost, and stays the human's. But the reason it is hard is that the
   field cannot distinguish "the code under test changed" from "someone's prose was open in
   an editor" — demonstrated on this repo's own gate row at `a8b4986-dirty`, where the dirt
   was the other lane's audit file. The *substitute* signal is the closure digest, and
   making sure every row-writing tool carries one was this lane's (§34). The policy question
   is now the only part left.
3. **Whether lanes should re-read `CLAUDE.md` on a schedule** — this stays the human's, and
   the earlier claim that *"nothing here can detect the divergence"* was too strong. A
   `CLAUDE.md` digest on a row would let a reader tell which rules a lane was operating
   under, exactly as `code_that_ran` does for code. It is not built, because recording which
   rules a lane had does not make a stale lane re-read them, and re-reading is the actual
   ask. Said precisely rather than dismissed.

## 36. A correction, and the half of §34 that a source test could never have caught

### The claim that was invented

§34 justified giving `rung0_linear_control.py` a closure digest by saying it *"runs on the
same rented box, beside the arm it exists to interpret."* That was asserted from the shape
of the tool and never read off a row. The concurrent lane checked it and it is false.

There is **one** linear-control row in this repository: `2988ac80`, written
`03:35:33Z`, `env.device: cpu`, `env.host: Mac.lan`. It ran here, about thirteen hours
before the curve, whose first row is `16:14:29Z`. `learning_curve.sh` invokes
`rung0_real_run.py` and nothing else. Verified in this lane against `ledger/` rather than
taken from the report.

The gap was real — that row carries no digest — and the reason printed beside it was made
up. A true finding reached for the wrong reason is still a reason nobody checked, which is
this repository's own failure mode arriving from the inside. Corrected in the tool's
comment, in §34's table, and in the gap record. `6efd874`'s commit message keeps the wrong
claim: amending it would rewrite two commits that cite it, and the correction is better
placed where it will be read.

### The bigger half: a source test asserts a fact about *this checkout*

The concurrent lane's second point is the one that matters, and it is a limit on the fix
rather than a flaw in it:

> 65 of 89 `gh200-rung0` rows carry no digest — and they came from `rung0_real_run.py`, a
> tool that has recorded one since `80cddb8` and was **never missing from that test's
> scope**. The copy on the box predated the commit.

So there are two halves, and only one is a property of this repository's source:

| half | question | checkable where |
| --- | --- | --- |
| **tool** | does the code record provenance? | source — closed by §34 |
| **deployment** | did the code that *ran* record it? | nowhere in this tree |

§34's test would pass over a repository whose every row was written by sources three
thousand miles away and six commits stale. It asserts a true fact about six files here.

### Measured, and worse than 65 of 89

| | rows | with a digest |
| --- | --- | --- |
| training / smoke | **727** | **28 (3.9%)** |
| build | 205 | 0 — correct; a build row is identified by its commands and toolchain |

And the part that closes off the obvious alternative: **`code_commit` cannot make this
distinction, by measurement.** `8b9df39bf816d149efc108e3581ab4ba8192688f-dirty` sits on
**389 rows without a digest and 24 with one** — the same string, both sides. The box is
synced by *copying* files into a clone pinned at an old commit, so its commit is the
clone's and its `-dirty` bit is permanent. A commit string that appears on both sides of a
distinction cannot be used to make it.

699 of 727 training rows in this repository are pinned by nothing but that string.

### What can still be done about it, and what cannot

The 699 cannot be fixed. The ledger is append-only and those rows say what they were
written with; rewriting them would be inventing provenance, which is worse than lacking it.

The **700th** can be prevented, and the deployment half is visible here after all — not in
any source file, but in the *rows*, at the moment they are brought back.
`python/tests/test_ledger_provenance.py` pins the digest-less inventory per ledger file and
fails when a training row arrives without one, whoever wrote it and wherever it ran. It
needs no ssh and trusts nothing the box says about itself: a row either carries a digest or
it does not.

Injected and confirmed: two digest-less rows appended to a capacity file, as a stale box
would return them, fail with *"26 without a digest, 24 known"* and the instruction to check
the deployment before the tool — because that is what happened.

`tools/sync_box.sh`'s digest comparison is the same check pointed the other way: before a
run rather than after it. Two checks, two moments, neither able to cover the other's.

### Three more numbers that are now printed rather than derivable

`test_the_coverage_is_reported_as_two_numbers` prints `28 of 727 (3.9%)` on every run.
*"28 rows carry provenance"* reads like progress; *"28 of 727"* reads like what it is. The
floor is pinned so it cannot quietly fall, and nothing fails for history being what it is.

## Where this leaves the final train

On the axes this lane owns:

* **A killed run always writes a row, and a priced killed run records its price.** Measured
  both ways; the one remaining unwrapped billed path is 0.0003% of wrapped time.
* **A row names its own arm** without the launch command, and cannot claim a recipe that is
  not the one identifying it.
* **The shape is pinned, not just the sites** — a tenth recorder or a seventh runner fails
  the inventory until its author states the reason.
* **A sweep's own resolution warning states its own case** — the correction is computed per
  row from an inverse-t this project now owns, instead of one constant true of one case.
* **Every tool that writes a row records what produced it**, and both scopes that enforce
  that are derived from the tree rather than listed by hand.
* **A training row cannot arrive without provenance** — pinned in the ledger rather than in
  the source, so it holds for rows written on a machine this checkout has never seen. What
  it cannot do is repair the 699 that already have none (§36).
* **The one open gap cannot go off quietly** — the six recipe-hash spellings are pinned, so
  the tidy-up that would silently rename every future `recipe_hash` fails with the
  consequence spelled out.

What is not this lane's: *unifying* the six spellings, which renames every future
`recipe_hash` and is a comparability break somebody has to declare in a handoff — not a
tidy-up, and not a thing to do mid-experiment. Guarding it was separable and is done.

## Open

1. **`GAP-SIX-SPELLINGS-OF-ONE-RECIPE-HASH` — open, and guarded (§35).** Six ways to hash a
   recipe, two disagreeing on identical dicts. Unifying them renames every future
   `recipe_hash`, so it is declared in a handoff rather than merged. The accidental version
   now fails a test that names the consequence, so what remains is a decision, not a risk.
2. **Whether `code_commit` should refuse `-dirty` for a run that will cost money.** Human's,
   and now the *only* part left of that question. The field cannot distinguish "the code
   under test changed" from "someone's prose was open in an editor" — this repo's own gate
   row at `a8b4986-dirty` was dirtied by the other lane's audit file. The substitute signal
   is the closure digest, and every row-writing tool carries one as of §34.
3. **Whether lanes should re-read CLAUDE.md on a schedule rather than on incident.** Human's.
   Earlier parts said *"nothing here can detect the divergence"*, which was too strong: a
   `CLAUDE.md` digest on a row would say which rules a lane ran under, exactly as
   `code_that_ran` does for code. Not built, because recording which rules a lane had does
   not make a stale lane re-read them, and re-reading is the ask.
4. **`GAP-A-SOURCE-TEST-CANNOT-SEE-WHAT-THE-BOX-ACTUALLY-RAN`.** 699 of 727 training rows
   carry no closure digest, and `code_commit` cannot distinguish the ones that do — the same
   `-dirty` string sits on 389 without and 24 with. The historical rows are not fixable and
   the next one is now refused on arrival (§36). What is open is whether the numbers already
   quoted from those 699 should be re-derived, which is a science call, not an engineering
   one.

## First command for the next lane

```bash
grep -n "_row_writing_tools\|_pricing_tools\|_SPELLINGS\|NOT_WRAPPING" /Users/bharath/Code/research/qwen-decision/python/tests/test_tool_call_sites.py
```

The four things in this file that decide what the other checks look at: two derived scopes,
the pinned recipe-hash spellings, and the inventory of every place a recorder can be handed
a duration it did not time. Three of today's findings were in the scopes rather than in the
code they checked — a list of four that had stopped being all of them, twice, and a check
whose pattern could not match the way its flag is actually spelled. Read these before
trusting anything else here.
