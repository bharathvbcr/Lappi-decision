# Ruling: the LoRA-2B step count N and the rate row (pre-registration amendment 1, pending values)

Fable, 2026-10-17. Read-only: nothing edited, no GPU job run. Inputs: Lappi-decision main 2733015
(`CLAUDE.md` rules 2 and 8; `campaign/lora-2b-preregistered.json` `recipe.steps` and `amendments[1]`;
`AUDIT/lora-2b-audit2-2026-10-17/fable-final-ruling.md` section 2), ojas main 4da214b
(`bench/results/2026-10-17-lora-step/rows.jsonl`, `README.md`, `env.txt`), the v5r train shard
(`~/qd-campaign/phase4-v5r-2026-10-03/shards/train/{header.json,offsets.npy}`) and the planner
(`crates/qd-train/src/shards.rs:1972-2017`, `plan_epoch`).

## 1. Ruling

**Option (a). N = 127, from the 2B T=9,638 row (rows.jsonl line 4, 261.607 tok/s), the slowest measured
LoRA rate.** The parenthetical's wrong premise is corrected in the same pre-result amendment (section 4).

Exact values to write into `recipe.steps.mac_steps_rule`:

| field | value |
| --- | --- |
| `n` | `127` |
| `tokens_per_s_row.row_id` | `"ojas@4da214b:bench/results/2026-10-17-lora-step/rows.jsonl#L4"` |
| `tokens_per_s_row.tokens_per_s` | `261.607027` (the row's own `tokens_per_s` field, verbatim) |
| `tokens_per_s_row.rate_choice` | the text in section 5 |
| `warmup_steps` | `"max(1, N // 20) = 6 (N = 127)"` |
| `status` | `"FIXED by amendments[2] on 2026-10-17 before step 0: n = 127, row ojas@4da214b:bench/results/2026-10-17-lora-step/rows.jsonl#L4, 261.607027 tok/s. A reduced-2B row whose step count differs from 127, or that names no row id, is not a LoRA-2B-registered row."` |

Derived, for the amendment text: one 35,403-token step at 261.607 tok/s is 135.33 s; 127 steps are
17,186 s (4.77 h), inside 17,280 s (80% of 6 h) with 94 s to spare, and 4,414 s short of the 6 h bound.

## 2. The arithmetic (verified from the files)

- Rule: N = max N <= 400 with N x 35,403 / tokens_per_s <= 17,280.
- Line 4 (2B, T=9,638, ckpt on): `mean_step_ms` 36,837.70 over 9,637 positions = 261.607 tok/s
  -> 17,280 x 261.607 / 35,403 = 127.69 -> **N = 127**. Per timed step (36.37 / 36.69 / 37.46 s): 129 / 128 / 125.
- Line 3 (2B, T=2,048, ckpt on): 6,093.80 ms over 2,047 = 335.915 tok/s -> 163.96 -> N = 163.
- Ordering: T=2,048 is the **faster** rate (335.9 > 261.6; ratio 1.284 in per-token wall). The 0.8B pair
  (lines 1-2: 658.2 vs 499.5) shows the same ordering. The parenthetical's premise ("the T=2,048 rate as the
  conservative figure") came from the full-gradient Tape rows in my section 2 (203-246 tok/s at T=2,048 vs
  233-281 at T=9,638), where T=2,048 *was* the slower rate. LoRA reverses it: the per-launch fixed cost that
  dominated the full-gradient step (frozen dW, the [V,d] head gradient) is gone, and the attention term that
  grows with T is what remains.

## 3. Why (a), and why not (b), (c), (d)

**(b) literal T=2,048, N = 163 — refused.** The clause names a row for a reason it states: "as the conservative
figure". The reason was true of the measurements that existed when the clause was written and is false of the
LoRA rows. Reading the row name and dropping the reason inverts the clause: it would fix N from the *fastest*
measured rate. 163 x 135.33 s at the slower measured rate = 22,059 s = 6.13 h, over the 6 h bound outright.

**(c) the run's bucket mix — computed, informative, not binding.** I reconstructed the plan read-only:
558,790 rows, lengths from `offsets.npy` (min 133, max 9,638, mean 691.2), bucketed by `bucket_for` (smallest
width >= length, `header.json` buckets), packed `min(35,403 // width, 4096)` rows per batch (`plan_epoch`).
This gives **12,176 batches per epoch**, exactly v5's ledger figure cited in `recipe.steps.full_recipe`,
so the packing model is right. Mean padded tokens per step: 33,670 (the rule's 35,403 is 5% above it; wide
buckets under-fill: 3 x 9,638 = 28,914). Interpolation stated: per-token wall c(w) linear in w between the two
measured points (c(2,048) = 2.977 ms, c(9,638) = 3.823 ms), clamped at both ends; expected step wall =
sum over buckets of (padded tokens x c(width)) / 12,176.

| model | E[step wall] | implied tok/s | N (rule form, 35,403/step) |
| --- | ---: | ---: | ---: |
| linear-clamped | 107.4 s | 313.6 | 153 |
| all buckets at the T=9,638 rate | 128.7 s | 261.6 | 127 |
| all buckets at the T=2,048 rate | 100.2 s | 335.9 | 163 |
| linear-clamped + 0.1 s per launch | 112.0 s | | 154 (by E[wall]) |
| linear-clamped + 0.5 s per launch | 130.3 s | | 132 (by E[wall]) |

Why it cannot bind: 53.6% of batches (55.8% of padded tokens) are narrower than 2,048, the narrowest measured
width, so the "interpolation" is an extrapolation for the majority of steps; those steps run 18 to 197 launches
each (mean 45.9 launches per step across the epoch) and the per-launch overhead of the LoRA step below T=2,048
is unmeasured (the 150-site adapter gradient collection and bank Add run once per launch). The rule's words are
"the measured LoRA rate of the row named"; a blend is not a row. (c) would need a measured row at a narrow width
or at the mix itself, which is a GPU job I am not to run; if one is ever measured it can replace this value by a
further pre-result amendment, and only before step 0.

**(d) variants considered and not taken.** Using the actual 33,670 mean padded tokens per step (N = 134) changes
a number the rule fixes in words (`tokens_per_step: 35403`); the 5% it leaves on the table is the right size to
cover what the arm does not measure (D17 and span-head host work, data loading, padding masks, dropout 0 vs the
real mask), so I keep 35,403. Using the slowest single timed step (37.455 s -> 257.3 tok/s -> N = 125) is not
"the row's rate": the row's `tokens_per_s` field is the citable figure and the three-step spread (3%) is already
inside the margin. Lowering the cap or the 80% figure is an owner matter I was not asked to reopen.

**The asymmetry that decides it.** `crates/qd-train/src/trainer.rs:697,1130`: a run that reaches its cap ends with
`Termination::WallClockCap` and fewer steps than scheduled; `mac_steps_rule.status` then makes that row "not a
LoRA-2B-registered row". So an N that overshoots the window loses the row; an N that undershoots finishes early
with a valid row and unused window. The slowest measured rate protects the only outcome that matters for a
quick mechanics row. At N = 127 the run has 25% of headroom to the 6 h bound (17,186 s vs 21,600 s) for the
unmeasured adders in (c).

## 4. The parenthetical: corrected in the same amendment, how, and why rule 2 is not engaged

**Yes, in the same pre-result amendment (amendments[2]).** Writing `rate_choice` = "T=9,638" next to a rule text
that says "the T=2,048 rate as the conservative figure" leaves the file contradicting itself, and a later reader
would correctly say the chosen rate violates the rule as written. `amendments_rule` permits it: no LoRA-2B result
exists (`rg -uu` over `ledger/` for `optimizer_recipe.{0,4}lora|lora-2b`: no match, exit 1, 2026-10-17), and the
original text stays in git history (dac3b21).

Replace, in `mac_steps_rule.rule`, exactly the parenthetical
`(at the run's bucket mix, or the T=2,048 rate as the conservative figure)`
with
`(the slowest measured LoRA rate among the 2B rows of that result set, i.e. the row that gives the smallest N; no row measures the run's bucket mix, and the T=2,048 row is the faster of the two, not the conservative one)`.
Everything else in `rule`, `cap`, `floor`, `window`, `tokens_per_step`, `schedule` stays verbatim.

Rule 2 is not engaged: the rate row is an input to a schedule parameter of a `quick`, non-promoting Mac row
(my section 2 already ruled N a schedule parameter, adopted by the owner). No gate, threshold, kill criterion, seed
count, held-out set, val set or width changes; nothing is reclassified; the 0.8B smoke (60 steps) is unchanged.
Rule 8 is unchanged: the row stays `quick` and cannot promote.

Suggested `amendments[2]` fields (same shape as [1]): `n: 2`; `recorded: "2026-10-17, after Lappi dac3b21 and
ojas 4da214b, before any LoRA-2B result exists anywhere (rg -uu over ledger/ for optimizer_recipe lora or
lora-2b: no match)"`; `what`: the five values in section 1 plus the parenthetical replacement; `why`: section 3
in two sentences (the measured LoRA rates reverse the parenthetical's premise; the slowest measured rate is the
one that keeps the row registered, since a WallClockCap stop makes the row unregistered); `decided_by`: "Fable,
2026-10-17, this ruling; owner's standing delegation of open decisions"; `pending_before_step_0: []`. The
reduced-2B ledger row copies `n`, `row_id` and `tokens_per_s`.

## 5. `rate_choice` text (write verbatim)

"T=9,638 row (ojas@4da214b:bench/results/2026-10-17-lora-step/rows.jsonl#L4; tape-lora arm, 2B-Base
b1485b2f, LoRA r16/alpha16 on 150 sites, bf16 base, checkpointing on, 28 GiB budget; tokens/s = 9,637
positions / mean of 3 timed steps after one warmup step, 36.84 s). Chosen as the slowest measured LoRA rate:
the T=2,048 row (#L3, 335.915 tok/s) is the faster one, so the rule's parenthetical as first written named
the wrong row; corrected by amendments[2]. No row measures the run's bucket mix; a read-only reconstruction of
the v5r train plan (12,176 batches, 33,670 mean padded tokens per step, 54% of batches narrower than 2,048)
with per-token wall interpolated linearly between the two rows projects N = 153, falling to 132 with 0.5 s of
unmeasured per-launch overhead; 127 is the measured floor. N = 127 gives 17,186 s at the measured rate against
the 17,280 s budget and 21,600 s bound. The arm scores every position through the fused tied head with the
same ids each step; the real step adds D17, the span head, data loading and padding, none measured here."

## 6. Not a ruling item, but a trap the lane must not walk into

`crates/qd-train/src/run_control.rs:33`: `DEFAULT_CAP_S = 1_800.0`. A reduced-2B launch that does not pass the
cap explicitly ends at WallClockCap after about 13 steps and produces an unregistered row. The run's cap must be
the window's 6 h (21,600 s), passed on the command line (`qd-train-metal` takes `cap_s`, `run.rs:244`).

## 7. What was verified and how; gaps

- verified: all rows.jsonl figures (read the file at ojas 4da214b); the N arithmetic (recomputed); the plan
  reconstruction (12,176 batches = v5's figure); `plan_epoch` packing (read); WallClockCap behaviour (read
  trainer.rs:1012-1016, 697, 1130); no LoRA row in `ledger/` (rg default and `-uu`, both exit 1).
- inferred: the (c) projections (an interpolation, stated); the real-step adders being within 25% headroom.
- unverified: the LoRA step wall below T=2,048 (no measurement exists); D17/span-head cost.
- navigation gap: the planner and cap lookups were by rg, not DevMap (read-only mandate, no `gaps.jsonl` append
  made); the amendment lane should record a GAP line if it relies on this ruling's file:line citations.
