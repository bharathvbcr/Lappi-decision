# F's optimizer, read: what the Rust trainer over tessl must reproduce (L-oracle, 2026-10-01)

Closes `GAP-OJAS-ADVICE-F-ADAMW-EPS-WD-UNREAD-2026-10-01` (fable-advice.md Q1 row 10, "eps and
weight decay: not read"). Labels: **[V]** read at file:line or measured by a command this lane
ran; **[I]** inferred from read code; **[U]** not verified.

Sources: Lappi at main `6a06bd3` (this worktree, fast-forwarded before reading), with
`tools/real_ft_run.py` line numbers moved to the file after the merge of main `85caea1`
(the only cited source that merge changed; every moved line checked to hold the same
text); torch 2.12.1 at
`/Users/bharath/.venvs/ml/lib/python3.14/site-packages/torch` (cited as `torch/...`); canonical
tessl at `~/Code/research/tessl`, HEAD `cf65d9d` [V `git log -1`]. Whether the cited tessl files
are among tessl's ~20 dirty files is **[U]**: the harness refused `git -C` on that tree and
GitPulse answered `REPOSITORY_TRUST_REQUIRED`.

Measured numbers below come from the rung (b) fixture this lane committed,
`crates/qd-train/tests/fixtures/tiny-published/manifest.json` (and its `trajectory.json` files),
written by `tools/qd_train_oracle_tiny.py`. They are fixture facts, not ledger rows: this lane
was given no ledger file to write (rule 5 caveat).

## 1. What F runs

F's flags: `--optimizer master --lr 1e-5 --epoch --no-memorise --batch-tokens 35403
--lower-layers-n 8 --lower-layers-lr-scale 0.1`, no `--beta2`, no `--fused-adamw`
[V `campaign/f-v4-preregistered.json` `recipe.flags`].

| Fact | Value in F | Evidence |
| --- | --- | --- |
| Optimizer class (bf16 tower, `master`) | `qd_train.optim.MasterWeightAdamW`: fp32 masters of the bf16 tower, `torch.optim.AdamW` over the masters | [V `real_ft_run.py:1994-1998` `optimizer_spec("bf16","master")` -> `keeps_fp32_master=True`; `optim.py:546-547`] |
| Builder call | `MasterWeightAdamW(groups, lr=lr, betas=(0.9, beta2), fused=fused)`: **eps and weight_decay not passed** | [V `optim.py:545-547`] |
| eps | **1e-8** (MasterWeightAdamW's default, passed to the inner AdamW) | [V `optim.py:337`, `optim.py:402-405`] |
| weight decay | **0.01** (MasterWeightAdamW's default) | [V `optim.py:338`, `optim.py:402-405`] |
| weight decay coverage | **every parameter**: no group carries a `weight_decay` key, so all take the constructor's 0.01 -- norms (zero-centred `w`, decayed toward 0 = scale 1), `dt_bias`, `A_log`, `conv1d`, the tied embedding/head, and the span head (both projections and both abstain vectors) | [V groups built at `optim.py:313-316` carry only `params`, `lr_scale`, `name`; `_normalise_groups` keeps keys `optim.py:197-199`; inner groups copy extra keys `optim.py:389-396`]; measured: all 59 entries of the tiny tower + head at 0.01 [V manifest `per_parameter`] |
| The non-master path (fp32 tower, `ADAMW_FP32`; or `bf16` recipe) | `torch.optim.AdamW(groups, lr=lr, betas=betas)`: torch defaults eps 1e-8, weight_decay 1e-2, same coverage | [V `optim.py:604`; `torch/optim/adamw.py:27` `weight_decay: float = 1e-2`; `torch/optim/adam.py:40` `eps: float = 1e-8`] |
| betas | `(0.9, 0.999)`: F passes no `--beta2`, so `DEFAULT_BETA2` | [V `optim.py:63`, `optim.py:545`; `real_ft_run.py:7835-7836`] |
| amsgrad / maximize / capturable | off (torch defaults; not passed) | [V `optim.py:402-405`; manifest `arms.*.optimizer.groups`] |
| fused | off: F has no `--fused-adamw`, so `fused=None` | [V flags above; `optim.py:404`] |
| Update dispatch | CPU: single-tensor path (`_default_to_fused_or_foreach` -> `(False, False)`, measured on this host) [V manifest `torch_dispatch_on_this_host`]. CUDA (F on the GH200): foreach [I from `torch/optim/adam.py:935-945` + `torch/optim/optimizer.py:161-180`; not measured on the box] |
| Decoupled decay | `param.mul_(1 - lr * weight_decay)` with the **group's** `lr`, before the moments | [V `torch/optim/adam.py:414-419` (single), `:688-691` (foreach)] |
| Moments and step | `exp_avg.lerp_(grad, 1-beta1)`; `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1-beta2)`; `step_size = lr / (1 - beta1**t)`; `denom = sqrt(v) / sqrt(1 - beta2**t) + eps`; `param.addcdiv_(m, denom, -step_size)`; `t` per parameter, incremented first | [V `torch/optim/adam.py:414,456,475,528-546`] |
| Parameters without a gradient | **skipped entirely** (no decay, no moment update, no step count) | [V `torch/optim/adam.py:150-151` `if p.grad is not None`]; measured: the span head takes 13 AdamW steps in a 20-step run whose 7 letter-only batches give it no gradient, the tower 20 [V manifest `coverage.span_head_adamw_steps`, `tower_adamw_steps`, `per_parameter.*.adamw_steps_taken`] |

### Parameter groups (`--lower-layers-n 8 --lower-layers-lr-scale 0.1`)

- Two groups: `base` at `lr_scale` 1.0 and `base_lower` at the given scale [V `optim.py:313-316`].
  A parameter goes to `base_lower` iff its name matches `(?:^|\.)layers\.(\d+)\.` with index
  `< lower_layers_n` and does not contain `visual` [V `optim.py:247`, `:285-296`]. Everything
  else -- embedding/tied head, final norm, every layer `>= n`, and the span head (`extra`) -- is
  `base` [V `optim.py:297-298`; `backbone.py:1027-1036`].
- Refusals: `lower_layers_n < 1`, a non-int, an empty lower group, `n` past the deepest layer, an
  empty base group, a non-finite or non-positive scale [V `optim.py:274-312`, `:211-221`].
- `--lower-layers-lr-scale` defaults to RSI's 0.1 when `--lower-layers-n` is given [V
  `real_ft_run.py:324`, `:7821-7830`].
- The group `lr` is `lr_at(step) * lr_scale`, written every step by `apply_lr` [V
  `optim.py:224-242`, called at `backbone.py:1190`; at construction `optim.py:392-395`].
- **Because decay is `1 - group_lr * wd`, the lower layers also decay at 0.1x** [V by the two
  lines above]. A per-entry `lr_scale` in tessl must scale `decay_mul` as well as `step_size`.

### Clipping

- `QwenDecisionStep.apply`: `apply_lr` -> `clip_grad_norm_(self.parameters(), max_grad_norm)` ->
  `optimizer.step()` -> `zero_grad(set_to_none=True)` [V `backbone.py:1187-1195`].
- `self.parameters()` is the tower's parameters then the span head's: **norm over tower + head**
  [V `backbone.py:1052-1053`].
- `max_grad_norm` = 1.0: the default, and `_real_step` does not pass it [V `backbone.py:927`;
  `real_ft_run.py:2087-2103`].
- `clip_grad_norm_` = `get_total_norm` + `clip_grads_with_norm_` [V `torch/nn/utils/clip_grad.py:198-199,230-232`]:
  per-tensor 2-norms grouped by (device, dtype) (`_foreach_norm`, or `vector_norm` per tensor),
  stacked, 2-norm of the stack [V `:95,102,106`]; `coef = max_norm / (total + 1e-6)`, clamped at
  1.0, multiplied **in place into `.grad`** [V `:164,168,173`].
- Under the master recipe the tower's grads are **bf16**: per-tensor norms come back in bf16
  (`torch._foreach_norm` on bf16 returns bf16 [V measured on this host]), the stack is promoted to
  f32 only when the f32 span-head norms are present, and the scale is applied to the bf16 grads
  **before** `MasterWeightAdamW.step` upcasts them [V `optim.py:436-437`]. Measured: on the
  fixture's letter-only steps (no span-head grads) the master-bf16 arm's pre-clip norm is
  bf16-representable -- 105.5, 98.0, 93.5, 118.0, 67.0, 128.0, 62.5 [V
  `master_bf16/trajectory.json` steps 0,4,6,10,12,16,18].

### Schedule

- `LRSchedule(peak_lr=lr, total_steps=steps, warmup_steps=max(1, steps // 20), min_lr=lr / 10)`,
  `grad_accum=1` [V `real_ft_run.py:1832-1838`]. Fable's row 12 wrote `steps//20`; the code is
  `max(1, steps//20)`, which differs for runs under 20 steps (warmup 1, not 0).
- `lr_at(step)`, 0-based: `peak * (step + 1) / warmup` while `step < warmup`; else
  `min + (peak - min) * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))`; raises at
  `step >= total` [V `run_control.py:353-375`].
- The loop calls `lr_at(optimizer_step)` then `apply(lr=...)`, `optimizer_step` from 0 [V
  `trainer.py:749,856-857,866`]. `steps = len(plan) * passes` [V `real_ft_run.py:2145`].
- At F's lr, `min_lr` is `1e-5 / 10 = 1.0000000000000002e-06` in f64 [V manifest
  `recipe.schedule`]; a Rust port must compute `lr / 10.0` in f64, not write `1e-6`.

### What each step's gradient is the gradient of

- One batch per optimizer step (`grad_accum=1`), one `loss.backward()` per batch with no extra
  scaling [V `backbone.py:1153,1182`; `real_ft_run.py:1838`].
- `total = letter + span_weight * span` (letter absent on a span-only batch) [V
  `backbone.py:1176-1181`]; `span_weight` 1.0 [V `real_ft_run.py:8209`]. `letter` = mean
  full-vocabulary CE over the batch's supervised positions (`target_index` of non-span rows) [V
  `fused_ce.py:323`; `trainer.py:424-425`]; `span` = (sum of start CE + end CE) / (2 * span rows)
  [V `heads.py:400-406`]. In tessl's sequence-at-a-time bank this is a per-row scale of
  `1/n_supervised(batch)` on the letter CE and `span_weight/(2K)` on each pointer CE [I].

## 2. tessl's AdamW, read

| Fact | tessl | Evidence |
| --- | --- | --- |
| Hyperparameters | `AdamWHyper { lr, beta1, beta2, eps, grad_scale }`; `Default` = lr 1e-3, 0.9/0.999, eps 1e-8, grad_scale 1 | [V `src/qwen35_adamw.rs:32-56`] |
| Learning rate | **one** `lr` for every entry of a step | [V `qwen35_adamw.rs:364-370`] |
| Weight decay | per parameter-table entry (`weight_decay: &[f32]`) | [V `qwen35_adamw.rs:369,376-382`] |
| `default_weight_decay(wd)` | 0 for `excluded_from_weight_decay` names (`bias`, `layernorm`, `rmsnorm`, a `norm` or `*_norm` segment: every Qwen3.5 norm and `linear_attn.dt_bias`), `wd` elsewhere | [V `qwen35_adamw.rs:103-112,491-497`]; on the tiny tower that is 17 of 55 tower entries [V manifest `per_parameter.*.tessl_default_excludes`] |
| Scalars | `decay_mul = (1 - lr*wd) as f32`, `lerp_w = (1-beta1) as f32`, `step_size = (lr/bc1) as f32`, `bc2_sqrt = (1-beta2^t)^0.5 as f32`, formed in f64 | [V `qwen35_adamw.rs:224-247`] |
| Kernel | decay; `g * grad_scale`; torch's two-form lerp; `v*beta2 + (1-beta2)*g*g`; `denom = sqrt(v)/bc2_sqrt + eps`; `p += -step_size * m/denom` | [V `kernels/qwen35_adamw.metal:58-66`] |
| Step count | one counter for the whole model, incremented once per `adamw_step` | [V `qwen35_adamw.rs:418-430`] |
| Gradient norm | `grad_sq_norm`: sum of squares over the model's parameter table only (rows in f32, total in f64) | [V `qwen35_adamw.rs:433-441` and the body after it] |

## 3. Every difference, and what the fixture can catch

"Caught at lr 1e-5 / 1e-3" is the fixture's own counterfactual measurement: the fp32 arm rerun
with that one detail changed, compared the way rung (b) compares (final weights within 1e-5 of
each parameter's max; loss within 1e-5 relative over steps 0-5 and 1e-4 to step 20). Worst
`max|delta| / max|w|` over all parameters [V manifest `discrimination.by_lr`].

"xN floor" divides that gap by the **independent floor** at the same lr: the gap between the
reference and `IndependentStep`, the same optimizer, clip and schedule with the forward and
loss restated per unpadded sequence and every target read off the stored sequence
[V manifest `independent_check_fp32` and `discrimination.by_lr.*.independent_floor`]. That is
what two correct torch implementations already differ by, sharing the GDN and attention
kernels; the Rust trainer shares neither, so its own floor is at least as likely to be higher
as lower [I]. A difference within a small factor of the floor is not one the bar separates
reliably, whichever side of 1e-5 it lands on.

| Independent floor | final weights, all | final weights, tower | loss, steps 0-5 / all | within rung (b)'s bars |
| --- | --- | --- | --- | --- |
| lr 1e-5 (F) | 9.12e-6 (`span_head.abstain_start`) | 9.66e-7 | 4.9e-7 / 6.0e-7 | yes, with 1.1x headroom on final weights |
| lr 1e-3 | 9.57e-5 (`layers.2.mlp.up_proj.weight`) | 9.57e-5 | 7.1e-6 / 7.1e-6 | **no**: final weights 9.6x over the bar; losses within (1.4x headroom early) |

| # | Difference | F | tessl as it stands | What the Rust trainer must do | Caught at 1e-5 | Caught at 1e-3 |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Weight-decay coverage | 0.01 on **every** entry, norms and `dt_bias` included | `default_weight_decay` zeroes norms and `dt_bias` | pass an all-0.01 vector (the manifest's `per_parameter.*.weight_decay`), never `default_weight_decay` | **no** (4.96e-6 weights = x0.54 floor, tower x1.6; loss 9.3e-7 = x1.6) | weights x1.5 floor (tower x1.25): not separated; **loss** 7.6e-5 = **x10.8** floor |
| D1' | No weight decay at all | -- | -- | -- | yes, barely (1.58e-5 = x1.7 floor; loss x6.2) | weights x7.0 (tower x1.3); loss x34 |
| D2 | Per-entry lr | `base` 1.0x, layers `0..n-1` 0.1x | one `lr` | per-entry `lr_scale` (L-tessl's change), refuse `--lower-layers-n` until it lands | yes (5.28e-3 = x580 floor) | yes (0.51 = x5300) |
| D3 | Decay under a scaled lr | `1 - (lr*scale)*wd` | `1 - lr*wd` | the `lr_scale` vector must enter `decay_mul` too | marginal (1.13e-5 = x1.24 floor, tower x1.8; loss 6.0e-6 = x10) | weights x10 (tower x1.1); loss x67 |
| D4 | Clip norm coverage | tower + span head | tower only (`grad_sq_norm`) | add the head's host-side sum of squares before the sqrt | -- | -- |
| D5 | Clip coefficient | `min(1, 1/(norm + 1e-6))`, applied to `.grad` (bf16 grads under master) | caller-supplied `grad_scale`, applied in f32 in the kernel | form `1/(sqrt(sum_sq) + 1e-6)` clamped at 1 on the host, f32 | dropping the clip: yes (0.24 = x26000 floor) | yes (0.25) |
| D6 | Schedule | warmup `max(1, steps//20)`, cosine to `lr/10`, 0-based | none | port `lr_at` with f64 bit-equality against `trajectory.json`'s `lr` column | a cosine one step late: yes (8.3e-2 = x9000 floor) | yes (7.6e-2) |
| D7 | Step count per parameter | per parameter; the span head skips letter-only batches | one model-wide counter | the span head's optimizer keeps its own `t` and does nothing on a batch with no span row (13 of 20 steps in the fixture) | -- | -- |
| D8 | Update path | single-tensor on CPU (the oracle), foreach on CUDA [I] | mirrors single-tensor | nothing; same arithmetic order, rounding may differ (FMA) | -- | -- |
| D9 | Precision of the run | bf16 live tower and activations, fp32 masters and moments | f32 weights, bf16 GEMM operands only | nothing: Fable row 9, bitwise is not a goal; compare Bf16 against the fp32 arm (2^-8) and report the master-bf16 arm | -- | -- |
| D10 | `wd` scalar | Python double 0.01 | `f32` 0.01 widened to f64 | none needed: `decay_mul` is rounded to f32 on both sides, so the two agree except at a rounding boundary [I] | -- | -- |
| D11 | eps, betas | 1e-8, (0.9, 0.999) | same defaults | none; never use `AdamWHyper::default().lr` (1e-3) | -- | -- |

The clip is active on all 20 fixture steps (coefficient 0.0023-0.0161; pre-clip norms 62-438)
[V `fp32/trajectory.json`], so the `min(1, .)` branch at coefficient 1 is **not** exercised by
this fixture.

**The open questions for the lead** (rule 2: the rung's thresholds are untouched here;
recorded as `GAP-L-ORACLE-RUNG-B-FINAL-WEIGHTS-BAR-AT-FLOOR-2026-10-01` for 1 and
`GAP-L-ORACLE-RUNG-B-BLIND-TO-WD-AT-F-LR-2026-10-01` for 2-3).

1. *The final-weights bar sits at the floor.* At F's learning rate two correct torch
   implementations already differ by 9.12e-6 against a 1e-5 bar, on `span_head.abstain_start`:
   the abstain vectors start at zero (heads.py:311-314), so `max|w|` is their own small trained
   displacement and the relative measure is harshest there. The tower alone has 10x headroom
   (9.66e-7). A correct Rust trainer, with its own GDN, attention and reduction order, can fail
   the all-parameter bar on the abstain vectors without being wrong.
2. *At F's learning rate the weight-decay details are inside the floor.* D1 sits at x0.5-1.6
   of the floor on every measure and D3 at x1.2-1.8 on final weights, so neither a pass nor a
   fail on those rows says anything about weight decay. Only the loss trajectory moves past
   noise for D3 (x10), and it stays under the loss bars.
3. *A 1e-3 sensitivity arm separates weight decay on the loss, not on final weights.* At lr
   1e-3 the independent floor itself fails the 1e-5 final-weights bar (9.57e-5), and the
   weight-decay rows sit at x1.1-1.3 of it on the tower; on the loss trajectory they sit at
   x11 (D1), x34 (D1') and x67 (D3) of a 7.1e-6 floor. An arm at 1e-3 would need its own
   final-weights bar, or be judged on losses.
   `tools/qd_train_oracle_tiny.py --lr 1e-3 --out <dir>` writes that arm's fixture; whether
   rung (b) carries it, and under which bar, is the lead's call.
