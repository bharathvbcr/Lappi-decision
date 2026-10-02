# HANDOFF rungd-flags (lane L-rungd-flags), 2026-10-02

Fable's idle-GPU ruling, lane task 2 (`AUDIT/idle-gpu-queue-2026-10-02/fable-idle-queue.md`, Q1
item 2): the additive `tools/real_ft_run.py` flags rung (d)'s GH200 torch reference arms need
(T-bf16, T-fp32; design `AUDIT/ojas-training-2026-10-01/fable-rung-d-resize.md`, Amendment 2 (iv)
under Q3 of `AUDIT/ojas-training-2026-10-01/fable-advice.md`). No gate, threshold, seed rule or
verdict logic changed (rule 2). `python/qd_data` untouched. No dependency added.

Lane: worktree `agent-af2fee24b0ffeacd5`, branch `worktree-agent-af2fee24b0ffeacd5`,
fast-forwarded from `c65d7da` to main `e21066c` before any change.

Labels: **[V]** read at file:line or produced by a command this lane ran; **[R]** read through a
summary I did not check against the raw source (WebFetch's summariser over upstream files);
**[I]** inferred; **[U]** not verified.

## What was measured

**On a GPU or the box: nothing. No ledger row.** Every number below is a pytest run on this
Mac's CPU (torch 2.12.1), with its raw output under `AUDIT/idle-gpu-queue-2026-10-02/`.

- **Fail-first** (`rungd-flags-failfirst-pre-change.log`): the final test file run against
  `tools/real_ft_run.py` at `e21066c`'s bytes (swapped in after `8da3bc8`, then restored with
  `git checkout`), opt-ins set: **40 failed, 5 passed** of its 45. The 5 that pass are
  characterizations, which must pass on both sides: the schedule formula at 200 steps, the master
  path's eps/weight-decay defaults, the old `_check_piece_flags` Namespace still accepted, and the
  two byte-identity arms (identical code on both sides then). The same run included
  `test_tool_call_sites.py`: 18 passed, 1 failed,
  `test_every_tool_that_hashes_a_recipe_is_found_by_this_check` (it finds
  `qd_train_oracle_trainer.py`, an L-oracle tool), so that failure is pre-existing at `e21066c`.
- **After the change** (`rungd-flags-after.log`): **45 passed of 45**, both opt-in tests included
  (`QD_BYTE_IDENTITY_BASE=e21066c`, `QD_SPAN_HEAD_INIT_H2048=<main checkout file>`). The repo
  guards on the final tree, this HANDOFF and these logs included (`rungd-flags-after-repo-guards.log`:
  `test_gaps_ledger`, `test_gaps_writer`, `test_wire_gap_pins`, `test_tool_call_sites`,
  `test_lint_gate`, run with `-rfEs` -- an `-rA` log lists `test_gaps_writer`'s deliberately
  malformed id as a passed parameter, which the citation scan then reads as a dangling gap id):
  70 passed, 3 failed, 4 skipped. The 3:
  `test_tool_call_sites` (pre-existing, above); `test_gaps_writer::test_the_real_ledger_is_not_touched_by_any_of_this`
  (asserts the ledger's directory is named `Lappi-decision`; a worktree's is
  `agent-af2fee24b0ffeacd5`); `test_lint_gate::test_ruff_is_installed_not_merely_declared` (a
  worktree has no `.venv`; the direct ruff run below is the check). The 4 skips are the lint
  gate's, for the same reason.
- **Regression** (`rungd-flags-regression.log`): every `python/tests` file that names
  `real_ft_run` (51, this lane's included, opt-ins unset) plus the gap, provenance, lint-gate and
  wire-pin files: **1405 passed, 4 failed, 34 skipped**, counted from the progress markers (the
  doubled `-q` suppressed pytest's totals line). The 4: the three above, and
  `test_gaps_ledger::test_every_gap_id_cited_in_a_tracked_file_exists_in_the_ledger`, because this
  HANDOFF cited `GAP-RUNGD-DESIGN-EPOCH-ARM-IS-PLAN-ALL-2026-10-02` before the record was
  appended; it passes in the after run. The 34 skips are data not on this host, env-gated
  benchmarks and opt-ins (3 of them this lane's). The cargo-building parity tests rewrote
  `Cargo.lock` (+46 lines); restored, not committed.
- **Lint**: the locked ruff 0.16.8 (`/Users/bharath/Code/research/Lappi-decision/.venv/bin/ruff
  check --config pyproject.toml`) over the three changed Python files: clean. Over the whole
  worktree: 29 errors, all in three files this lane did not touch
  (`tools/qd_train_oracle_trainer.py` 19, `tools/qd_train_oracle_optimizer_table.py` 6,
  `tools/qd_train_oracle_head_digest.py` 4), present at `e21066c`.

### The byte-identity proof

`test_with_every_new_flag_off_the_rows_are_the_base_commits` (opt-in, `QD_BYTE_IDENTITY_BASE`)
loads `git show e21066c:tools/real_ft_run.py` as a second module and runs both `main()`s on one
CPU toy corpus (byte-tokenized code.defect_class, 48 repos) with identical argv, `--epoch
--no-memorise --score-val`, and compares every ledger row. Two arms:

| Arm | Flags | Rows | recipe_hash (ft, eval) |
| --- | --- | --- | --- |
| stand-in | none | ft + eval, identical | `e8d0d5be2a7f800a`, `a079159b81ff1ed4` |
| real tiny tower (hidden 64, 4 layers) | `--optimizer master --lower-layers-n 1 --lower-layers-lr-scale 0.1 --train-attention-mask none --deterministic` | ft + eval, identical | `f492c30a3a9f96b5`, `6e2332fd63c1f9e7` |

Identical means: the recipes are equal, the protocol hashes are equal, the metric key sets are
equal, and the whole rows are equal after dropping `row_id`, `written_at`, `prev_row_hash`,
`wall_clock_s` and the metric `code_that_ran` (the digest of the tool's own source, which is the
change under test), with each ledger's row ids replaced by their position so the eval row's
`ft_run_row_id` compares. Losses, notes, quick reasons, every metric value: equal
[V `rungd-flags-byte-identity-and-h2048.log`]. It is opt-in because a differential against a
fixed base fails by design the next time the tool legitimately changes.

The permanent, non-opt-in pins: `_recipe_pieces` with every new piece off returns `{}`;
`quick_reasons` with `max_steps=None` returns what it did; the digest metric is absent from the
ft row without its flag and the recipe hash does not move with it.

## What changed (commits)

- `8da3bc8`: `tools/real_ft_run.py`, `python/tests/test_real_ft_rungd_flags.py` (new, 45
  tests, 3 of them opt-in: the h2048 head and the two byte-identity arms),
  `python/tests/test_backbone.py` (the tiny-snapshot helpers take `hidden=`, defaulting to the
  old width: every existing caller builds what it did), and this lane's three navigation gaps.
- The next commit: this HANDOFF, the four logs under `AUDIT/idle-gpu-queue-2026-10-02/`
  (`rungd-flags-*.log`) and the two resolution gap records.

### `--max-steps N` (epoch arm only; needs `--epoch`)

- `_train` cuts `plan` to `plan[:N]` before `steps = len(plan) * passes`, so the schedule is
  `LRSchedule(lr, N, max(1, N // 20), lr / 10)` (the recipe's formula, `_control`; at N=200 and
  lr 1e-5 that is warmup 10, min lr 1e-6, per `GAP-LTRAINER-RUNGD-MIN-LR-FORMULA-VS-LITERAL-2026-10-01`),
  and the loop, the loss log, the consumed digest and the row all see N. Tested: the cut run
  consumes exactly the full run's first 3 batches, byte for byte, and the spy sees that schedule.
- **Correction to Fable's design text** (`GAP-RUNGD-DESIGN-EPOCH-ARM-IS-PLAN-ALL-2026-10-02`):
  the epoch arm receives `plan_all`, not `plan_small` [V `a502670:tools/real_ft_run.py:8707`,
  `e21066c:9569`]; `plan_small` (the `--max-width` filter) is the memorisation arm's plan
  (`:8602` / `:9463`). The design's `:9164 _train(plan=plan_small)` citation is that arm. No
  effect on the order: F's epoch order is `reader.batches(batch_tokens=35403, seed=20260919,
  epoch=0)` either way, and `--max-steps 200` takes its first 200 batches.
- **That main plans F's order.** `git diff --stat a502670 e21066c -- python/qd_data
  python/qd_train/shards.py python/qd_train/artifacts.py` [V]: `qd_data` unchanged; `shards.py`
  and `artifacts.py` changed only by the report-only slice (`037c2b8`, `913f79f`): two
  `ShardHeader` fields (`span_collapse_policy`, `report_only`) that hash to nothing at their
  defaults, and `write_shards`. The planner (`ShardReader.batches` / `_plan`) is not in the diff,
  so a main-based lane plans F's batches from the v4 set [I beyond the diff; not run on the v4
  set]. Criterion 0 compares the Rust reader's consumed digest with both torch arms', all at
  main, so it does not rest on this.
- `train.optimizer_steps` = N; `train.termination` = `steps_exhausted`, the existing vocabulary
  (`run_control.TerminationReason` is `data_exhausted | steps_exhausted | wall_clock_cap`): the
  N-step schedule did run out. The truncation is carried by the recipe key `max_steps` and by
  rule 8's reason, `max_steps_reason(N)`: "truncated schedule (--max-steps N): the epoch arm
  trained the first N batches of its order under an LR schedule recomputed over N steps, not the
  epoch". It is on the ft row and on every row the run writes after it.
- **Rule 8 for rows scored later.** A torch `--score-checkpoint` row reads only its ft row's
  termination (`GAP-SCORE-CHECKPOINT-SINGLE-SEED-ROW-DOES-NOT-INHERIT-QUICK-2026-10-01`), which a
  cut schedule leaves at `steps_exhausted`: T-fp32's val row (criterion 6) would have come out
  quick=False. `ScoredModel.max_steps` is read from the ft row's `recipe.max_steps` (a malformed
  value is refused), and `model_reasons` adds the same reason. Rows without the key: unchanged.
  The general gap stays open.
- Refused: N < 2 (the warmup is at least one step and `LRSchedule` refuses a run that ends inside
  it, after the tower loads); N >= the epoch's batches (not a truncation); without `--epoch`;
  with `--score-checkpoint`/`--score-plan`; with `--shuffled-label`.

### `--train-dtype fp32` (needs `--real-backbone`)

- The tower trains in fp32. `optimizer_spec("fp32", recipe)` is `ADAMW_FP32` for either recipe
  (an fp32 parameter is its own master; `MasterWeightAdamW` refuses all-fp32 parameters,
  `optim.py:360-366`), so `build_optimizer` returns plain `torch.optim.AdamW(groups, lr, betas)`
  (`optim.py:604`). `--optimizer master` stays legal, so T-fp32 is F's argv plus one flag and its
  recipe differs from T-bf16's in `train_dtype` (and the mask, below); `train.path.optimizer`
  says `AdamW` [V test].
- Before step 0, `fp32_adamw_problems` reads every group of the optimizer that was BUILT: its
  class is `torch.optim.AdamW`, and its `eps`/`weight_decay` equal `MasterWeightAdamW.__init__`'s
  signature defaults (1e-8, 0.01; read by `inspect`, so the check is "equals F's master path",
  not two literals), `betas == (0.9, beta2)`, not amsgrad, not fused. Otherwise `SystemExit`
  naming the group and value [V test with a builder monkeypatched to eps 1e-6]. This is what
  covers the box's torch 2.10.0, whose defaults this lane did not read
  (`GAP-OJAS-RUNGD-FP32-ADAMW-EPS-WD-UNREAD-2026-10-01` was resolved by reading torch 2.12.1).
- Refused with `--fused-adamw` and, **by name, with `--train-attention-mask none`** (below).

### The fp32 + no-mask finding

- The no-mask training forward and backward run inside `sdpa_kernel([SDPBackend.FLASH_ATTENTION])`
  (`QwenDecisionStep.training_attention`, `backbone.py:1090-1104`) [V].
- The box runs torch `2.10.0+cu128` [V, read-only ssh]. In torch v2.10.0's
  `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, `can_use_flash_attention` runs
  `check_for_attn_mask` and `check_dtypes_low_precision`, which on sm80+ accepts Half and BFloat16
  only [R]. So fp32 + `none` on the GH200 has no kernel; SDPA would raise "No available kernel.
  Aborting execution." at the first forward [I; not run]. CPU's flash kernel does take fp32 (the
  rung (b) oracle trains fp32 + `none` on CPU, `tools/qd_train_oracle_tiny.py` `make_step`) [V
  code], but no arm runs on CPU. **The flag refuses the combination.**
- **T-fp32 therefore trains with the padding mask.** Kernel it would use on the GH200 (sm90),
  fp32, explicit mask, `--deterministic`: flash out (dtype and mask), cuDNN out (dtype; and
  `check_cudnn_deterministic` makes it unavailable under deterministic algorithms), memory-
  efficient in: fp32 allowed on sm80+, alignment 4 for fp32, no mask check, no stated maximum
  head dim (Qwen3.5's full-attention head_dim is 256) [R]. Under deterministic algorithms its
  backward forces `num_splits_key = 1` [R, `attention_backward.cu`]. If its shape checks refused
  anyway, SDPA would fall to the math backend (fp32), not raise [I]. Not run: which backend ran
  is not recorded by the tool, and this lane added no probe.
- **Still "the numerics tessl's unpadded step matches"?** At every supervised position, yes
  [I]: batches are right-padded, full attention is causal and the GDN recurrence is causal, so no
  real position reads a pad with or without the mask (`backbone.py:1064-1068`), and the losses
  read only `supervision.mask` and the span rows. What differs from T-bf16 is the kernel
  (mem-efficient fp32 against flash bf16), on top of the dtype. T-fp32 stays the tight reference;
  T-bf16's gap to it is the yardstick criterion 1 already reports.
- Other fp32 facts for that arm: matmuls run at full fp32, not TF32 (`torch.get_float32_matmul_
  precision()` is `highest` by default and nothing in `tools/real_ft_run.py` or `python/qd_train`
  sets it [V rg; V default on 2.12.1; I for 2.10.0]). `torch.backends.cudnn.allow_tf32` defaults
  True, which reaches only cuDNN convolutions: if the GDN short conv runs on `nn.Conv1d` rather
  than `causal_conv1d_fn` on the box, it runs in TF32 [U]; the row's `train.path`
  `linear_attention_kernels` names which. fla 0.5.2's `chunk_gated_delta_rule` has no dtype
  refusal in its source [R]; it has not been run in fp32 [U]. Memory: unmeasured
  (`GAP-OJAS-RUNGD-FP32-ARM-MEMORY-UNMEASURED-2026-10-01`); `device_budget` refuses before step 0
  if memory.py's estimate exceeds the free GPU memory.

### The span head's init digest and `--span-head-init`

- `span_head_digest(step)`: `run_control._sidecar_digest` over the `qd-tensor-ref-v1` digests of
  `span_head.*`, the manifest's formula. Head-only (no host copy of the tower), pinned equal to
  the digest over `step.state()["span_head"]` on a real step [V test].
- `--span-head-init-digest` records it as the metric `train.span_head_init_digest` on the ft
  row, before step 0. **A flag, not always on**: an always-on metric would add a key to every new
  row and break the byte-identity proof. A metric, not a recipe key, so no hash moves.
- `--span-head-init PATH --span-head-init-manifest PATH` (Amendment 2 (iv)'s fallback): at argv
  time, the file's sha256 and its content digest (`tensor_refs_from_safetensors` then
  `_sidecar_digest`) must equal the manifest's, the names must be the manifest's `span_head.*`,
  all float32, the manifest's `construction.seed` must be every `--seeds` entry, and the file at
  most 1 GiB. After the step and its optimizer exist, the tensors are copied into the head in
  place (shape-checked by name first), and the head's digest is recomputed and must equal the
  file's. In place matters: an fp32 parameter is its own master, so the copy is what the first
  step updates [V test: the master optimizer's masters for the head are the head's parameters
  and equal the file]. In the recipe as `span_head_init = {sha256, content_digest}`; it implies
  the digest metric.
- Digests [V]: the tiny tower's QwenDecisionStep at hidden 64, seed 0 =
  `81bd953e1f27e8e2b703264df663f15f239327c9523d3e85973687009db471d6` (the tracked h64
  manifest's), in-tree and end to end through `main`. Opt-in (`QD_SPAN_HEAD_INIT_H2048`): a
  2-layer stand-in tower at hidden 2048 built through L-oracle's `make_step`, seed 0 =
  `1c30346c9ea8c1fe12cbc41d556587187ad5a27fe165ac7bc9c69d6bcd7def68`, the h2048 manifest's;
  and the main checkout's `data/checkpoints/span-head-init/span_head_init-seed0.safetensors`
  reads and loads under its manifest.
- On the box the head is drawn on the CPU (`backbone.py:999` builds it before `.to(device)`), so
  both arms should record `1c30346c...` [I]; that the box's torch 2.10.0 draws the same stream as
  2.12.1 here is exactly what the metric checks. If it does not, the fallback is
  `--span-head-init` with the h2048 file copied to the box (not done by this lane).

## The two arm commands (documented, NOT run)

Preconditions, none done by this lane: a box lane at a main commit that contains this lane's
merge (`--train-attention-mask` and these flags exist only on main; `python/qd_data` is
unchanged since `a502670` [R Fable], and this lane did not touch it), holding F's
`data/pool/{commitpackft-composed-v1,commitpackft,defect-noul-v3b}` as qd-lane8 does; the queue's
`gpu.lock`; nothing else on the GPU. F's argv is `/home/ubuntu/box_q_f.sh` [V, read-only ssh],
with `/home/ubuntu/queue/f.ckpt-skip` = 6 [V], so F ran `--checkpoint-skip-layers 6`.

```
LANE=<box lane at the main commit holding this lane's merge>
cd $LANE  # clean, at that commit, refuse otherwise (as box_q_f.sh does)
export HF_HUB_OFFLINE=1 QD_PREP_BIN=/home/ubuntu/bin/qd-prep-m1
PY=/home/ubuntu/qd-venv/bin/python
REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
LEDGER=/home/ubuntu/ledger/gh200-rungd-torch-2026-10-02.jsonl
SPLIT=(--out /home/ubuntu/phase4-v4-2026-10-01 --no-repo-history
       --rev 881ab304f15ea13529002391dda8520c2ea47af4
       --defect-class data/pool/commitpackft-composed-v1
       --defect-download data/pool/commitpackft
       --defect-noul data/pool/defect-noul-v3b
       --general-record $REC --general-max-rows 200000
       --real-backbone $BACKBONE)
mkdir -p /home/ubuntu/rungd /home/ubuntu/ckpt/rungd-tbf16 /home/ubuntu/ckpt/rungd-tfp32

# T-bf16: F's flags + --max-steps 200 --train-attention-mask none --deterministic, seed 0.
$PY -u tools/real_ft_run.py "${SPLIT[@]}" --optimizer master --lr 1e-5 --devices cuda --seeds 0 \
  --epoch --no-memorise --batch-tokens 35403 \
  --lower-layers-n 8 --lower-layers-lr-scale 0.1 --checkpoint-skip-layers 6 \
  --max-steps 200 --train-attention-mask none --deterministic --span-head-init-digest \
  --checkpoint-dir /home/ubuntu/ckpt/rungd-tbf16 --checkpoint-every 100000 \
  --wall-clock-cap-s 3600 --instance lambda-1xgh200 --usd-per-hour 2.29 --ledger $LEDGER \
  2>&1 | tee /home/ubuntu/rungd/t-bf16.log

# T-fp32: the same + --train-dtype fp32, minus --train-attention-mask none (refused: no fp32
# kernel on CUDA's flash backend) and minus --checkpoint-skip-layers 6 (Fable: "attempted with
# full gradient checkpointing"; selective checkpointing is numerics-neutral per the lead's
# Tier-A parity [R]).
$PY -u tools/real_ft_run.py "${SPLIT[@]}" --optimizer master --lr 1e-5 --devices cuda --seeds 0 \
  --epoch --no-memorise --batch-tokens 35403 \
  --lower-layers-n 8 --lower-layers-lr-scale 0.1 \
  --max-steps 200 --deterministic --train-dtype fp32 --span-head-init-digest \
  --checkpoint-dir /home/ubuntu/ckpt/rungd-tfp32 --checkpoint-every 100000 \
  --wall-clock-cap-s 10800 --instance lambda-1xgh200 --usd-per-hour 2.29 --ledger $LEDGER \
  2>&1 | tee /home/ubuntu/rungd/t-fp32.log
```

- `--checkpoint-dir/--checkpoint-every 100000` are required, not optional: the ft row carries
  only `train.loss_log_digest`; the 200 loss points criterion 1 compares, and the consumed digest
  criterion 0 compares, are in the final checkpoint body (`epoch-seed0-cuda.json`), written only
  when both flags are set (`on_checkpoint.final`). Each final checkpoint carries the optimizer
  state too; F's seed checkpoint directory is 25 GB and the box's root has 3.6 TB free [V,
  read-only ssh, 2026-10-02].
- Caps and rule 4: 3,600 s x $2.29/h = $2.29 and 10,800 s x $2.29/h = $6.87, both under $20, so
  no `--approved-by` and no auto-terminate. T-bf16 at F's 1.79 s/step is about 6 min plus
  `--deterministic` [R Fable]. T-fp32's cap is sized for fp32 matmuls without TF32, which on
  H100-class tensor cores run several times slower than bf16 [I; Fable's "2-3x" is also [I]]; a
  capped arm ends `wall_clock_cap`, is not a parity result, and is rerun (rule 5).
- No `--score-val`: criteria 4-6 wait for the Mac artifact. Read criteria 0-3 from the ft rows
  and the two checkpoints. Both arms are quick (rule 8, the `--max-steps` reason).
- Each arm's exit status can be 1 with the rows written: the epoch arm's "the letter loss did
  not fall" claim compares the first batch's letter loss with the last's [V tool]; it is a claim
  about the run, not a refusal, and the ft row is what criteria 0-3 read.

## Open (gap ids)

- `GAP-RUNGD-TORCH-ARMS-NEED-MAX-STEPS-AND-TRAIN-DTYPE-2026-10-02`: resolved-with-residual (the
  arms not run; fp32 memory, fla fp32 and the conv kernel unverified).
- `GAP-RUNGD-DESIGN-EPOCH-ARM-IS-PLAN-ALL-2026-10-02`: the design text's `plan_small` /
  `--max-width` sentence; the lead amends it.
- `GAP-RUNGD-FLAGS-GITPULSE-UNTRUSTED-2026-10-02`, `GAP-RUNGD-FLAGS-NO-LISTAGENTS-2026-10-02`,
  `GAP-RUNGD-FLAGS-DEVMAP-WORKTREE-UNINDEXED-2026-10-02`: navigation.
- `GAP-SCORE-CHECKPOINT-SINGLE-SEED-ROW-DOES-NOT-INHERIT-QUICK-2026-10-01`: still open in general;
  `--max-steps` rows are covered.
- `GAP-OJAS-RUNGD-FP32-ARM-MEMORY-UNMEASURED-2026-10-01`: open; T-fp32's `device_budget` row or
  its run answers it.

## First command for the next lane

The lead merges this branch, then, on the box, after `gpu.lock` frees in the idle queue's order
(`rung0.done` -> `cudadev` -> `rungd`): the T-bf16 command above, then T-fp32. Before that, on
the Mac, the contract as a test:

```
PYTHONPATH=python /Users/bharath/.local/bin/uv run --offline --no-project \
  --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \
  python -m pytest -q python/tests/test_real_ft_rungd_flags.py
```
