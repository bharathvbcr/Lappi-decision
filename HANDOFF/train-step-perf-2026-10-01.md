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
