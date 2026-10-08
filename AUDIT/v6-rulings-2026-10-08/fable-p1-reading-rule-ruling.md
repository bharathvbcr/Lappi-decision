# Fable's ruling on P1's reading rule (2026-10-08)

**What this is.** Fable's ruling, asked by the v6prep lane, on the four items the P1 reader lane
found could not be computed as `campaign/next-train-first-box-2026-10-06.DRAFT.json` was written:
- GAP-P1-LOSS-GAP-UNDEFINED-ON-SPAN-ONLY-STEPS-2026-10-08
- GAP-P1-VAL-PAIRED-BOOTSTRAP-NEEDS-VERDICTS-OUT-2026-10-08
- GAP-P1-STEP-TIME-AND-PEAK-SHAPE-NOT-NAMED-BY-THE-DRAFT-2026-10-08

The draft was not yet pre-registered, so its definitions may be fixed now. **No threshold moved.**
The lead recorded the ruling as Fable returned it, condensed.

## Facts the ruling rests on

- **The batch plan is a pure function of `(seed, epoch, batch_tokens)` and the shard set** (`python/qd_train/shards.py:1989-1997`, `_plan` `:2376-2433`). So step s is the same batch in every arm. A span-only step is therefore span-only in all three arms at once.
- **When letter loss reads 0.0.** The trainer logs `letter 0.0` exactly when a batch has no supervised letter position (`backbone.py:1339`).
- **Where the log lives.** It is stored as `model_state.channel_log.letter` (float.hex) in the checkpoint (`backbone.py:1411-1412`), not under the name "letter_log".
- **The span-only count in 385-600 cannot be computed on this host,** but it can be computed before launch: the Mac prelude prints `batches_span_only` (`real_ft_run.py:11580-11586`).

## 1. loss_gap: exclude span-only steps, as a pair

- **The step set.** `S = { s in 385..600 : letter_master[s] > 0 }`.
- **Agreement required.** `letter_a[s] > 0 ⇔ letter_master[s] > 0` must hold for every s and every arm a; otherwise the reading is refused.
- **The formula.** `loss_gap(a, master)` = the mean over s in S of `|letter_a[s] - letter_master[s]| / letter_master[s]`.
- **The floor.** `floor = loss_gap(master-repeat, master)`, computed over the same S.
- **Too few steps.** The reading is refused if `|S| < 108`. This is a new refusal condition, not a threshold on the outcome.
- **What is reported:** `|S|` and the excluded steps.
- **Threshold unchanged:** `max(2 × floor, 0.02)`.

## 2. val_gap: `--verdicts-out`, PCG64, a percentile interval

- **The rows.** Each arm's `--verdicts-out` lines of kind `choice`, paired by `row_id`.
  - The two arms must hold identical row sets.
  - Each arm's count must equal its eval row's `val_top1.choice`.
- **The gap.** `d_i = correct_kahan − correct_master`, and `val_gap = mean(d)`.
- **The bootstrap.**
  - Generator: `numpy.random.default_rng(20261006)` (PCG64).
  - 2,000 draws in sequence, each `rng.integers(0, n, size=n)`.
  - Interval: `numpy.percentile(D, [2.5, 97.5])` with linear interpolation.
- **A required test.** The reader's chunked draws must equal the sequential form, and a test must assert it.
- **Threshold unchanged:** the interval lies inside [−0.02, +0.02].

## 3. Step time: `train.optimizer_step_s` decides

- **What is timed.** `apply` alone: `clip_grad_norm_`, `optimizer.step()` and `zero_grad`. It is device-synced, and the figure is the median over steps 101-600, present only when all 500 steps were timed.
- **The ratio.** kahan's value divided by the master arm's.
- **Reported but deciding nothing:** `train.step_time_s`, and master-repeat's value (the noise floor).
- **Why `apply` alone.** Forward and backward are identical across the recipes, and the `slow` remedy (foreach slices) touches only that loop.
- **Threshold unchanged:** 1.5×.

## 4. Peak: the footprint tool's own shape, every width, the maximum ratio decides

- **The draft's "at the arm's recipe" is false as written.** The tool uses its own fixed shape:
  - 1 row;
  - widths [2048, 4096, 8192, 14759, 34522];
  - the proxy loss;
  - full gradient checkpointing, with no skip layers;
  - sdpa, bf16, snapshot b1485b2f;
  - `torch.cuda.max_memory_allocated` over the second optimizer step.
- **Equal shapes required.** The two footprints' `shape` blocks must be equal.
- **The ratio.** `peak_ratio = max_w measured_kahan[w] / measured_master[w]`.
- **Refused:** an out-of-memory error at any width.
- **Threshold unchanged:** 0.80.
- **Headroom.** The predicted ratio on this host is 0.703-0.744.
- **What the pre-registration must say.** This peak is not the training step's peak. It answers only whether the 14 vs 20 B/param arithmetic survives the allocator.

## Departures from v5's recipe, to be stated in the pre-registration

None of these conflicts with kahan.
- **`--wall-clock-cap-s 3600`** instead of 32400. It is a recipe key, so the rows' `recipe_hash` differs from v5's.
- **`--max-steps 600`** recomputes the cosine schedule over 600 steps, identically across the arms. The draft's `not_claimed` already covers this.
- **`--checkpoint-every 100`.** The final checkpoint carries `channel_log`.
- **Dropped:** `--needle`, `--ood` and the controls, which are eval-only.
- **Not used:** `--fused-adamw`, so every master arm and footprint is unfused, like for like.

## The exact edits to the draft

1. **`P1.what`.** Add `--verdicts-out <per-arm path>`, `--checkpoint-every 100` and `--wall-clock-cap-s 3600`, with the recipe_hash note.
2. **`P1.per_arm`.**
   - The loss log is `model_state.channel_log.letter`/`.span` (float.hex; entry s-1 is step s).
   - The timing is `train.optimizer_step_s` and `train.step_time_s`.
   - The peak is from `gh200_footprint.py --optimizer <recipe> --out`, at the tool's own shape.
3. **`loss_gap`.** As in §1.
4. **`val_gap`.** As in §2.
5. **Add `step_ratio`.** As in §3.
6. **Add `peak_ratio`.** As in §4.
7. **The outcome words.** `admissible` uses `peak_ratio ≤ 0.80` and `step_ratio ≤ 1.5`. `no_saving` and `slow` are mirrored.
8. **Add `P1.data_fact`.** The span-only steps in 385-600 of the seed-0 plan at batch_tokens 35403, computed on the Mac prelude before launch.
9. **`status`.** Item (3) becomes "Fable's ruling of 2026-10-08 applied".

**The reader edits these imply:**
- `loss_gap` takes the paired mask, with the three-arm agreement check and the `|S| ≥ 108` floor.
- `CHOICES["zero_letter"]` is reworded.
- A chunked-vs-sequential bootstrap equality test is added.
- The numbers gain `loss_gap_steps_used` and `loss_gap_steps_excluded`.
