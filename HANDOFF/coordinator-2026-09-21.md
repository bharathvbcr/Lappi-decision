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
