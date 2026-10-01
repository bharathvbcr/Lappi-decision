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

**The P2 rule as accepted fails a candidate identical to the baseline.** Do not wire item 9's
cancel until Fable amends the rule.

Evidence: `AUDIT/tierb-nomask-2026-10-01/p2_null_sim.py` and its `.out`, a CPU simulation
against `p2_screen`.
- **The setup.** Six exchangeable arms, the same path plus iid noise. Three are labelled baseline,
  three candidate.
- **The result.** D(t) ≤ S(t) held on a mean of **20.5%** of steps (range 11–36%, 200 trials). With
  one candidate the mean is 50.1%. The ≥ 90% rule passed **0 of 200** null candidates.
- **The cause is combinatorial.** D is a maximum over 9 cross pairs, S over 3 baseline pairs. For
  exchangeable arms, the largest of those 12 pairs is a baseline pair only about 1/4 of the time.
- `--p2` implements the accepted text unchanged. It is pre-registered, and amending it is
  Fable's call.
- **A calibrated alternative is not ready.** I tried a role-permutation version
  (`p2_perm_sim.py`/`.out`). It passes 97% of null candidates, but it also passes 100% of
  candidates offset by 4σ, because the labelling that swaps baseline and candidate mirrors the
  true one. A sound rule needs design first. Options for Fable:
  - per step, compare each candidate's distance to the baseline mean against the baselines'
    leave-one-out spread;
  - a permutation test that excludes the complementary labelling.

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
