# HANDOFF: train-step perf lane, 2026-10-01

Branch `worktree-agent-a59bae74f0f03844b`. Box writes were confined to `/home/ubuntu/perf/`. Every
GPU session ran under `flock /home/ubuntu/queue/gpu.lock timeout 1200`, and none was left running
(`pgrep -af "[p]erf_parity|[p]erf_step|[p]erf_session"` returned nothing at 17:46 UTC).

## What was measured

**Tier-A parity ledger rows.** These are perf rows marked quick (capped), never decision rows.
Box copy: `/home/ubuntu/perf/ledger-parity-2026-10-01.jsonl`; repo copy:
`AUDIT/perf-train-step-2026-10-01/session2/`. All arms ran `--deterministic` through
`real_ft_run._train`:

| row | arm | code | loss_log_digest |
| --- | --- | --- | --- |
| `a095a6e6` | B-base | lane4 7ed66d7 | 0ec7f81e… |
| `bf409c64` | B-skip6 | overlay | 0ec7f81e… (identical) |
| `2be65177` | A-base | lane4 | de620bee… |
| `f796d3b4` | A-skip6 | overlay | de620bee… (identical) |
| `665f51e5` | A-over0 | overlay, N=0 | de620bee… (identical) |
| `cdf200e0` | A-base2 | lane4, determinism floor | de620bee… (identical) |

- Every per-step letter and span loss is identical, and so is consumed_digest.
- Recipes differ only by the per-arm `tag`, plus `checkpoint_skip_layers: 6` on the skip arms.
- consumed_digest in `perf_parity` is recomputed from the plan. It is equal by construction
  unless `_train` reorders batches.

**Benchmarks and probes.** These are perf_step result rows, not ledger rows, in
`session2/bench.jsonl` and `probe-nomask.jsonl`.

Default kernels at B (4 × ≤8,441, bt 35,403), overlapped timing, 3 interleaved rounds, min-of-3:

| config | pos/s | s/step | peak |
| --- | --- | --- | --- |
| off:0 | 15,280 | 1.934 | 42.16 GiB |
| skip:6 | 16,543 | 1.786 | 60.51 GiB |

Peak at 35,145 tokens (9 × 3,905), one round each: skip 6/7/8 = 60.03 / 64.94 / 69.84 GiB.

No-mask probe, one round each:

| kernels | mask | pos/s |
| --- | --- | --- |
| default | none | 25,703 |
| det | none | 23,208 |
| det (session 1, for comparison) | padding | 9,983 |

Session 1 is in `AUDIT/perf-train-step-2026-10-01/session1*` (commit e1c33e3): profiles of the
unmodified path. The fla GDN path was named on every row.

**F's flag composition.** `AUDIT/perf-train-step-2026-10-01/compose-a502670/run.log`, 5 passed.
This ran on CPU against main a502670's own code.

## What changed

**Merged by the lead:** e728a1d, fc417e6, e19dcf6, f791c34.

**Ready to merge, in order:**
1. **e1c33e3** session-1 artifacts.
2. **076143a, 7c09ad3** session-2 harness.
3. **0b95adb** RECIPE_PIECE_KEYS gains `checkpoint_skip_layers` and `optimizer_fused`, so scored
   rows mirror them. A failing-first test is included.
4. **8b6e858** skip × lower-layers composition test, plus the a502670 evidence.
5. **93b7977** session-2 artifacts.
6. **0d2739e** perf_tierb_fused builds p3fused with 0b95adb.
7. **09407f5, e008379** scoping note and a gap record.
8. **c3be616** no-mask Tier-B design draft.

**On the box:** `/home/ubuntu/perf/p3fused` was rebuilt at f6a0928, which is 884b658 + 1af64f6 +
f6a0928. `box_q_tierb.sh` (the lead's) runs it after `f.done`.

**F** (launched 17:55 by the lead): `--checkpoint-skip-layers 6`, at a502670. At a502670 its
scored rows do not mirror skip 6; that is fixed by 0b95adb, and the lead is recording the gap.

## What is open

- GAP-TRAINSTEP-PERF-GITPULSE-UNTRUSTED-DEVMAP-WALK-INCOMPLETE and
  GAP-TRAINSTEP-PERF-WORKTREE-HAS-NO-DEVMAP-STORE-GITPULSE-UNTRUSTED: tool coverage, owned by the
  human.
- **Tier-B fused run.** Queued; not run. Judge it against the envelope in
  `tools/perf_tierb_fused.sh`.
- **No-mask Tier-B screen.** Design is in `AUDIT/tierb-nomask-flash-screen-design-2026-10-01.md`
  and awaits Fable. The code change it needs is not written.
- **Later Tier-A cleanups:** the host syncs at `fused_ce.py:164`, and det-mode `fill_` (6% of det
  GPU time).
- **Not verified:**
  - skip and lower-layers together on CUDA; F seed 0 is the first run;
  - det no-mask self-repeat;
  - skip 7 and 8 under the parity harness;
  - the default-kernel no-mask profile.

## First command for the next lane

```
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246 'ls /home/ubuntu/queue/tierb.done; tail -60 /home/ubuntu/logs/q-tierb.log; tail -5 /home/ubuntu/perf/ledger-tierb-fused-2026-10-01.jsonl | cut -c1-400'
```

## Addendum: the no-mask Tier-B screen, built CPU-only after Fable's ruling

The ruling is in `campaign/f-j7prime-preregistered.json`, under `no_mask` and `tier_b_changes`,
on main 4fd08cf. **No GPU work was done and nothing was queued.**

**Commits** (on top of main 4fd08cf):

| commit | what |
| --- | --- |
| **1cf8e6d** | `QwenDecisionStep(train_attention_mask="padding"\|"none")` and `--train-attention-mask` |
| **de59c98** | `perf_parity.py --p2`, the screen statistic, plus `--train-attention-mask` and `--min-span-batches` for parity arms |
| **eb3eafa** | the scripts |
| **46da2a5** | the outcome-run gate: `perf_tierb_outcome.sh nomask --run` refuses (exit 5, before the lock) unless both shapes' latest P2 verdict is `pass` (`perf_parity.py --p2-gate`), 8 tests |

1cf8e6d in detail:
- Under `"none"`, only the training forward drops the mask. The training forward and backward run
  with SDPA's flash backend as the only one enabled, so on CUDA a shape flash cannot take raises
  instead of falling back to mem-efficient.
- `hidden()` keeps the mask for every scoring caller.
- The recipe key is `train_attention_mask`, written only when the value is `"none"`, and it is in
  RECIPE_PIECE_KEYS so scored rows mirror it.
- `train.path` records the mode.
- The flag is refused without `--real-backbone`, and refused with `--replay-shards`, because
  replay's KL forward keeps the mask.
- `perf_step`'s no-mask probe now flips this switch instead of monkeypatching `hidden()`.

**Fail-first evidence:** `AUDIT/tierb-nomask-2026-10-01/failfirst-pre-change.log`.
- I ran the final new tests against unchanged 4fd08cf code: 6 in test_backbone, 5 in
  test_real_ft_pieces and 8 in test_perf_parity_p2.
- Result: 19 failed, and every pre-existing test in those files passed.
- The one setup error is `test_ft_split_rows`, which needs the Cargo workspace; the extracted
  tree does not have it.
- The 8 gate tests from 46da2a5, run against de59c98's `perf_parity.py`: 8 failed, and the 8
  existing P2 tests passed (`failfirst-gate-pre-change.log`).

**The P2 rule as first accepted failed a candidate identical to the baseline.** Fable has since
retired it and amended the rule (main 1de0a34); see the last addendum below.

Evidence: v1 of `AUDIT/tierb-nomask-2026-10-01/p2_null_sim.py`, a CPU simulation against the
retired `p2_screen` (v1 is in git history at 7ba9df0; the file and its `.out` are now v2).
- **The setup.** Six exchangeable arms, the same path plus iid noise. Three are labelled baseline,
  three candidate.
- **The result.** D(t) ≤ S(t) held on a mean of **20.5%** of steps (range 11–36%, 200 trials). With
  one candidate the mean is 50.1%. The ≥ 90% rule passed **0 of 200** null candidates.
- **The cause is exact, and my first figure for it was wrong.** I wrote "about 1/4". Fable
  corrected it: per step, P(D ≤ S) = C(3,2)/C(6,2) = **1/5**, for any exchangeable noise.
  - The losses are scalars, so D(t) ≤ S(t) holds exactly when every candidate lies inside the
    baselines' range. That means the lowest and the highest of the six values are both
    baselines.
  - Under exchangeability, the pair {lowest, highest} is uniform over the 15 pairs of arms, and
    3 of those are baseline pairs.
  - With one candidate the same count gives C(3,2)/C(4,2) = 1/2. Both match the simulation
    (20.5% and 50.1%).
  - My 1/4 counted 3 baseline pairs out of 12 compared pairs, as though the largest distance
    were uniform over those 12. It isn't.
- **The permutation alternative fails** (`p2_perm_sim.py`/`.out`, kept as the record of why).
  It passes 97% of null candidates, but it also passes 100% of candidates offset by 4σ, because
  the labelling that swaps baseline and candidate mirrors the true one. Fable rejected this
  direction and the leave-one-out one I proposed alongside it.

**Full suite on the branch:** 3,282 passed, 58 skipped, 2 failed. That run was taken before
46da2a5; the gate's test file passes 16 of 16 since. The two failures are the same worktree-only
ones as before: the repo directory name, and no `.venv/bin/ruff` in this worktree.

**Scripts** (eb3eafa):
- `tools/perf_nomask_p2.sh` takes `--build <sha>`, `T1` or `T2`. T1/T2 each run one GPU session
  under `flock /home/ubuntu/queue/gpu.lock timeout 1200`, then the P2 statistic and the det
  self-repeat compare on CPU.
- `tools/perf_nomask_p2_body.sh` runs, in order:
  1. 3 masked and 3 no-mask arms, interleaved;
  2. the det no-mask pair (two runs at A, plus the pair at B, per Fable's Tier-A goldens rule);
  3. speed at B (T1) or peak memory (T2).
- `tools/perf_tierb_outcome.sh fused|nomask --build|--run|--print` runs the phase-3 seed-0
  outcome run plus its linear control and the fp32 all-gates re-score.
  - `tools/perf_tierb_fused.sh` is now a wrapper at its old path. On the box, `--print` resolves to
    the same three commands as before.
- `tools/perf_mkoverlay.sh` now takes lane, bundle and destination; its defaults are unchanged.

**One deviation from the design draft: both P2 shapes read v4.** v3's code fingerprint is stale
for main at or after 4fd08cf. The shapes are unchanged:
- A = bt 16,384 at widths ≤ 5,383;
- B = bt 35,403 at widths 7,001–7,936.

**Staged on the box** (CPU only, under `/home/ubuntu/perf`):
- `overlay-nomask` at 46da2a5, clean.
  - Dry-run selection: A gives 100 batches, 94 carrying a span row; B gives 50 batches, all 50
    carrying a span row, at widths 7,035, 7,404 and 7,936.
  - v4 reads with no stale-shard override.
- `p3nomask` at 60909b3: 884b658 + trainstep 28eb2bd + mirror d387730 + nomask 60909b3.
  - nomask.patch also applied to a local reconstruction of that tree, and the 6 backbone and 17
    argv/recipe no-mask tests passed there.
- `nomask.bundle` and `nomask.patch`.
- `perf_tierb_fused.sh` replaced by the wrapper, plus `perf_tierb_outcome.sh`.

**Rules, per Fable:**
- Item 8 (P2) may fill a gap after F's J7′ decision rows exist.
- Item 9 (the outcome run) runs only after J5′, and only if P2 passed.
- A P2 fail cancels item 9. A P2 pass admits nothing on its own.
- No Tier-B change enters phase 5/6.

**Not verified:**
- Nothing here has run on a GPU.
- The CUDA fail-closed behaviour (flash raising on an ineligible shape) is untestable on CPU,
  because CPU flash accepts masks. The tests check that flash is the only enabled backend during
  the forward and the checkpoint recompute.

**Commands for the queue builder** (items 8 and 9):

```
bash /home/ubuntu/perf/perf_nomask_p2.sh T1      # then T2; each session holds the lock at most 1,200 s
bash /home/ubuntu/perf/perf_tierb_outcome.sh nomask --run   # item 9, after J5', iff both P2 verdicts pass
```

Results go to:
- `/home/ubuntu/perf/nomask-p2-T{1,2}.jsonl`
- `/home/ubuntu/perf/nomask-p2-verdicts.jsonl`
- `/home/ubuntu/perf/ledger-nomask-p2.jsonl`
- `/home/ubuntu/perf/ledger-tierb-nomask.jsonl`

Done markers are `/home/ubuntu/perf/nomask-p2-T{1,2}.done` and `/home/ubuntu/perf/tierb-nomask.done`.

## Addendum: the amended P2 rule is implemented, and its calibration misses 3 of 78 checks

**T1 is not ready.** The amended rule is Fable's, on main 1de0a34
(`campaign/f-j7prime-preregistered.json`: `no_mask.p2_rule`, `p2_rule_retired` and `p2_timing`).
- I implemented it and ran every calibration target against the implemented function.
- 75 checks are met and **3 are missed**. `p2_null_sim.py` exits 1.
- Per the instruction, nothing was tuned: no threshold, trial count, seed, noise model or grid
  changed after the first run.
- `p2_timing`'s precondition needs every target met with the `.out` committed, and then the box
  overlay rebuilt at that commit. It is **not met**.
- So the overlay was **not** rebuilt. `/home/ubuntu/perf/overlay-nomask` is still at
  `46da2a594e7b`, which carries the **retired** rule; I read its `.git/HEAD` on the box after
  this commit. T1 must not run from it.
- **The marker `/home/ubuntu/queue/nomask-p2-ruled` must stay unset.** Item 8's waiter
  (`box_q_nomaskp2.sh`, main 234dc12) only runs T1/T2 when it exists, and it did not exist when I
  checked. The `perf_nomask_p2.sh T1` command in the queue-builder block above waits on the same
  precondition.

**What changed** (commit c18d245 on `worktree-agent-a59bae74f0f03844b`, from main 1de0a34; CPU
only, nothing queued):

- `tools/perf_parity.py`, `p2_screen`, now follows the amended text:
  - the refusals, including the two new labelling ones: every arm must have
    `termination == "steps_exhausted"`, baselines must carry `train_attention_mask == "padding"`
    and candidates `"none"`;
  - live steps;
  - per (channel, T ∈ {first live step, all live steps}): bbar, L, dev_i, noise_j, dev and noise;
  - `inconclusive` if noise > τ/κ, else `fail` if dev > τ, else `pass`;
  - the worst result over T, then fail > not_run > inconclusive > pass at the shape level.
- τ = 0.02 and κ = 3 are module constants. There is no flag, env var or `--tau`, and a test
  rejects `--tau`. `P2_MIN_FRAC` and `min_frac` are retired.
- The verdict row records:
  - τ and κ;
  - per (channel, T): L, every dev_i, every noise_j, dev, noise and the verdict;
  - each arm's `train_path`;
  - the report-only per-step profile: median and max of D and S, the fraction D ≤ S, the first
    step with D > max S, and the final losses.
- `--p2` exits 0 on pass, 1 on fail and 2 on not_run or inconclusive. The module docstring now
  describes this.
- Unchanged: `p2_gate`, `perf_nomask_p2_body.sh`, the arms and the result-row schema.
- `python/tests/test_perf_parity_p2.py` is rewritten to the amended rule: 32 tests, all of which
  pass now.
- Text: the comment block in `tools/perf_nomask_p2.sh`, the addendum in
  `AUDIT/tierb-nomask-flash-screen-design-2026-10-01.md`, and the 1/5 correction above.

**Fail-first evidence** (`AUDIT/tierb-nomask-2026-10-01/`):
- `failfirst-p2-amended-vs-de59c98.log`: the new tests against de59c98's `perf_parity.py`.
  **27 failed, 5 passed.** The 5 cover behaviour the amendment keeps:
  - the digest refusal;
  - the steps refusal;
  - two "fewer than 3 baselines / no candidate" cases;
  - the CLI exit mapping.
- `failfirst-p2-amended-vs-1de0a34.log`: the same tests against 1de0a34. **18 failed, 14
  passed.** The extra 9 passes are the `p2_gate` tests, which are unchanged by design.
- The test file both logs ran is byte-identical to the one committed in c18d245 (checked with
  `diff`).

**Calibration** (`p2_null_sim.py`/`.out` v2): 1,000 trials per configuration, over n ∈ {100, 50}
× {3v3, 3v1} × {noisy, bit-identical step 0}, and every configuration applied to every target.

| target | limit | result (worst configuration) | |
| --- | --- | --- | --- |
| exchangeable null, v1 noise | pass ≥ 99%, fail 0 | pass ≥ 99.7%, fail 0.0% in all 8 | met |
| null at noise ×3 | fail ≤ 12% | fail ≤ 0.1% (pass 43–92%, the rest inconclusive) | met |
| null at noise ×10 | fail ≤ 12% | fail ≤ 8.0% (n=100 3v3 noisy; 92% inconclusive) | met |
| null at noise ×20 | fail ≤ 12% | **fail 12.1%** at n=100 3v3 noisy (87.9% inconclusive); the other 7 ≤ 11.2% | **missed** |
| null at noise ×100 | fail ≤ 12% | fail ≤ 1.6% (≥ 98.4% inconclusive) | met |
| all candidates ×1.005 | pass ≥ 99% | pass ≥ 99.8% | met |
| all candidates ×1.02 | fail ≥ 95% | 3v3: 99.9–100%; 3v1 bit-identical: 100%; **3v1 noisy: 92.6% (n=100), 91.0% (n=50)** | **missed** |
| all candidates ×1.04 | fail ≥ 99% | 100% in all 8 | met |
| one candidate ×1.04, two null | fail ≥ 99% | 100% in all 4 (n=100/50 × noisy/bit-identical) | met |
| step 1 only, ×1.03, bit-identical | fail ≥ 99% | 100% in all 4 | met |
| step 1 only, ×1.01, bit-identical | pass ≥ 99% | pass ≥ 99.7% | met |
| Gaussian single application, worst P(fail \| null) over σ/τ | ≤ 4.5% | 3v3 **3.94%** at σ/τ 0.70; 3v1 1.79% at σ/τ 0.75 (20,000 trials per point, σ/τ 0.05–3.00) | met |

**Why the three misses happen.** `p2_miss_diag.py`/`.out` replays them with the simulation's own
seeds and reproduces its counts exactly. In each case the cause is the rule text, not the code:
- **δ = 2% sits exactly at τ.**
  - With one candidate, each of the four applications (letter/first, letter/all, span/first,
    span/all) fails in 46–51% of trials.
  - The shape passes only when all four pass: 7.4% and 9.0% of trials, close to 1/16.
  - The 3v3 rows reach 99.9–100% through the maximum over 3 candidates (more chances to land
    above τ), not through margin.
- **The bit-identical δ = 2% rows "meet" the target through float rounding.**
  `(1.02·x − x)/x` evaluates to `0.020000000000000018`, just above τ, for both channel levels.
  So the step-0 application always fails. That is not statistical power: at another loss level,
  those rows would be coin flips too.
- **×20 noise, n=100 3v3: 121 fails in 1,000**, against a limit of 120.
  - Every fail is an `any fail → fail` aggregation over four applications. In 100 of the 121,
    exactly one application failed and the other three read inconclusive.
  - The binomial standard error at 1,000 trials is about 1 point. I did not rerun with more
    trials or other seeds to move it.
- **A question for Fable, not one I can settle.** The rule attaches the configuration list
  "(n=100 and n=50, 3v3 and 3v1, with and without bit-identical baseline forwards)" to the null
  target. The δ targets name no configurations.
  - The simulation applies every configuration to every target, which is the strict reading.
    Under it, δ = 2% misses at 3v1.
  - If the list binds only the null target, 3v1 is out of scope for δ, and only the ×20 miss
    remains.
  - Gates are read-only, so I have not chosen between the readings.

**Another observation, within its target.** At ×3 noise, n=100, the null reads inconclusive in
50–57% of trials. If the box's run-to-run noise is about 3× the v1 model, item 9 would be held for
the human about half the time.

**Step-1 dev** (p2_timing asks for it whatever it is): not measured. It comes from T1/T2, which
have not run.

**Two other things to know:**
- **The harness bug in section 4.** It was found and fixed before the `.out` was committed. As
  first written, section 4 iterated `CONFIGS[::2]` and so ran noisy step 0 four times under
  duplicate labels. It now runs n ∈ {100, 50} × {noisy, bit-identical}, using the same seed
  count, so the other sections' seeds are unchanged. `diff` of the two outputs differs in
  exactly two lines, section 4's bit-identical rows (both 100% fail). The three misses
  reproduce digit for digit.
- **`p2_perm_sim.py` runs only against 7ba9df0's `tools/perf_parity.py`.** It calls the retired
  `min_frac` argument and reads `frac_d_le_s`. Its committed `.out` is the record. It is left
  unchanged, as instructed.

**Open:**
- Fable's call on the three misses: the δ = 2% placement at τ, the 3v1 scope, and the ×20
  aggregation.
- GAP-TRAINSTEP-PERF-WORKTREE-HAS-NO-DEVMAP-STORE-GITPULSE-UNTRUSTED still applies.
  `devmap_impact p2_screen` on the main checkout's index (generation 2965) gave:
  - the CLI `main` and the test file as callers;
  - the two simulations only as unresolved namesakes, confirmed by reading them.

**First command for the next lane** (after Fable rules, and after any amended targets are
committed):

```
/Users/bharath/.venvs/ml/bin/python AUDIT/tierb-nomask-2026-10-01/p2_null_sim.py
```

If, and only if, that exits 0 with its `.out` committed on main:
1. Bundle the commit.
2. `scp` the bundle and `tools/perf_nomask_p2.sh` to `/home/ubuntu/perf/`.
3. Run `bash /home/ubuntu/perf/perf_nomask_p2.sh --build <sha>` on the box. That is CPU work: it
   builds the overlay and runs both dry runs.
