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
