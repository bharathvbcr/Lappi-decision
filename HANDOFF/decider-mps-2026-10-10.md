# decider-2b on MPS, after the upstream fix (2026-10-10)

The lane: answer Mapika/decider#21's maintainer reply, confirm decider-ai 1.8.2 on this Mac, then the
human's "now run decider-2b on MPS for the full test-g6" and "post the results on the issue too".
It follows section "decider-2b" of `HANDOFF/merge-and-bench-2026-10-06.md`, where the MPS path could not
be scored (NaN).

## What was measured

JevJudge runs are recorded in their own run directories, not in `ledger/` (the same as the CPU run of
2026-10-06). Every number below comes from these directories or from the logs named.

- **Run `decider-2b-v11-mps-fp16-test-g6`** (`external/jevarena/runs/`).
  - decider-2b v11 (rev `533964da`) through decider-ai 1.8.2, MPS, Decider's default float16, torch 2.12.1,
    transformers 5.19.0.
  - Complete: 1,698 / 1,698 calls, 0 failed, 04:35:38Z–04:42:41Z on 2026-10-11.
  - The smoke (`...-probe5b`) also passed: 10/10, and it matched the CPU smoke's choices on all 10.
- **Against `decider-2b-v11-cpu-test-g6`** (`compare.py`, reference decider-2b-cpu).
  - Attempted accuracy 0.720 vs 0.719. Order consistency 0.748 vs 0.749. Domain macro 0.613 vs 0.612.
  - Paired MPS minus CPU: +0.001 [+0.000, +0.002] (168 groups, 849 pairs).
  - Median latency 190 ms vs 5,679 ms. p90 462 ms vs 8,075 ms.
  - So decider-2b minus Lappi v0.1 is unchanged at +0.110 [+0.071, +0.157] on either device.
- **Per call** (`~/qd-campaign/lappi-bench-2026-10-06/agree.py <cpu run> <mps run>`).
  - 1,697 / 1,698 choices are equal. The one flip is a near-tie: `rewardbench2:65:0:1`, unswapped, CPU
    B 0.4727 vs A 0.4711, MPS A 0.4719 = B 0.4719.
  - Max |Δp| 0.052, median 4e-4, 19 calls above 0.01.
- **The fix itself** (logs in the repo's gitignored `build/decider182-check/`).
  - `scaled_softmax(lg, [1.164])` on MPS: 0/500 wrong on 1.8.2, on transformers 5.12.1 and 5.19.0
    (`run-tf5121.log`, `run-tf519.log`).
  - `system_one` on the warm-up: MPS float32 equals CPU exactly. MPS float16 is within 1e-4.
  - Upstream's `test_scaled_softmax_per_row_temperature_on_mps_matches_cpu` (commit 2d3ef87) passes on
    1.8.2 and fails on the snapshot's bundled pre-fix code: all NaN, on the first of its 200 iterations
    (`upstream-test-182.log`, `upstream-test-prefix.log`).
- **The `patch_mps()` warning cannot fire on transformers 5.12.1.** There `causal_conv1d_fn` is `None`, so
  the `callable` check raises AttributeError before the version check. The `except` clause logs it at
  DEBUG only. This was seen in `run-tf5121.log` and is reported upstream.

## What changed

No commits; nothing in this repo changed except this file. Outside the repo, nothing pushed:
- `~/qd-campaign/lappi-bench-2026-10-06/decider_stdio.py` gained `--code DIR`. It imports `decider` from
  DIR rather than the snapshot. It exits before ready unless `decider` resolves under DIR. The ready line
  names `decider_ai_package`. Without the flag, behaviour is unchanged.
- `~/qd-campaign/lappi-bench-2026-10-06/pkgs/decider-ai-1.8.2/` holds `uv pip install --target --no-deps`.
- `external/jevarena/configs/decider-2b-v11-mps-fp16.local.json`, uncommitted like the other local
  configs.
- `decider_queue.sh`: its run list comment now names this run.
- `agree.py` (new, in the bench dir): per-call choice and probability agreement of two runs.
- Three comments on Mapika/decider#21, posted with the human's yes each time: 6104804538, 6105095007
  and 6105534744.

## What is open

- `docs/outreach/README.md` and `compare_all.sh` still describe decider-2b's MPS path as unusable. Add
  the MPS row where the CPU row is cited, if the outreach copy should say it.
- On transformers 5.19 with mlx 0.31.2, 1.8.2's `fast_invert_unitriangular_64` falls back to PyTorch.
  The cause is `mx.from_dlpack` missing. Results are correct; the MLX path does not run. This is
  reported upstream as an observation only.
- No GAP record: DevMap/GitPulse were not needed. Everything touched was outside the indexed repo or
  was a new file.

## First command for the next lane

```bash
/Users/bharath/.venvs/ml/bin/python /Users/bharath/qd-campaign/lappi-bench-2026-10-06/compare.py /Users/bharath/Code/research/external/jevarena/data/lappi/test-g6.jsonl decider-2b-cpu decider-2b-cpu=/Users/bharath/Code/research/external/jevarena/runs/decider-2b-v11-cpu-test-g6 decider-2b-mps-fp16=/Users/bharath/Code/research/external/jevarena/runs/decider-2b-v11-mps-fp16-test-g6
```
