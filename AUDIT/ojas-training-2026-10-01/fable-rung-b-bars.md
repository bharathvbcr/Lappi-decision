# Fable on rung (b)'s bars, weight-decay detection, clip coverage and the span-head init (2026-10-01)

Advisor: claude-fable-5-1, through a read-only lane (task a97591eb3fe6976c2), on L-oracle's merged rung (b) fixture (`fc7d783`). Recorded verbatim. The lead ratified Amendment 2 under the user's standing instruction to follow Fable's recommendations, and inserted it under Q3 of `fable-advice.md`.

**Amendment 2 (i) loosens the effective tolerance on two tensors**, `span_head.abstain_start` and `span_head.abstain_end`. It was ratified before any Rust rung (b) run existed: there is no `mac-ojas-rung-b` row. It is listed for the user in the lead's report.

## Verified numbers

| Claim | Value | Source |
|---|---|---|
| Floor at lr 1e-5, all parameters | 9.117e-6, on `span_head.abstain_start` | `independent_check_fp32.final_weights` |
| Floor at lr 1e-5, tower only | 9.655e-7, on `layers.2.mlp.up_proj` | same key, `tower_*` |
| Floor at lr 1e-5, losses | 4.90e-7 (steps 0–5), 5.98e-7 (all steps) | `independent_check_fp32.losses` |
| Gradient floor at step 0 | 6.31e-5 on `layers.0.A_log`, which is 0.63× rung (a)'s 1e-4 gradient bar | `independent_check_fp32.grads_step0` |
| Floor at lr 1e-3 | 9.574e-5 on weights, which fails the 1e-5 bar; 7.09e-6 on losses | `discrimination.by_lr.0.001.independent_floor` |
| D1 (`tessl_default_weight_decay`) at lr 1e-5, × floor | 0.544 weights, 1.61 tower, 1.56 losses; not caught | `by_lr.1e-05.counterfactuals.*.over_independent_floor` |
| D3 (`decay_not_lr_scaled`) at lr 1e-5, × floor | 1.244 weights, 1.80 tower, 10.06 losses; caught on weights only, at 1.13e-5 | same |
| Same at lr 1e-3, losses only | D1 10.8×, no weight decay 33.8×, D3 66.6×. D1's 7.65e-5 still passes the 1e-4 all-steps bar | `by_lr.0.001` |
| Clip | active on all 20 steps; coefficient 0.00228–0.01610; pre-clip norm 62.1–438.4 | `fp32/trajectory.json`; `coverage.clip_inactive_steps == []` |

- **Why `abstain_start` is the worst parameter.** It starts at zero, so its max after training (4.788e-5) is exactly how far it moved. Relative to that distance, the floor is 9.1e-6 for the abstain vectors and about 4.4e-3 for the tower. The current measure therefore holds the two abstain vectors about 500× tighter than every other tensor. I computed this from `init.safetensors` and `fp32/final.safetensors`.
- **No float64 reference is possible here.** Torch's GDN casts its inputs to f32 (`transformers/.../modeling_qwen3_5.py:263`), so the Amendment 1 route of switching to an f64 reference doesn't exist for rung (b).
- **No Rust result exists yet.** There is no rung (b) Rust test and no `mac-ojas-rung-b` ledger row, so any amendment now is made before the first run.
- **The span-head init file has no owner.**
  - It is named in the design at `fable-rung-d-resize.md:13,35` and `HANDOFF/ojas-training-2026-10-01.md:134`.
  - It is missing from L-oracle's lane row (`HANDOFF:59`), and no gap record covers it.
  - The seed is set at `backbone.py:983`; the head is built on the CPU at `:997-1000`. Nothing draws from the random stream between those lines.
  - The comment at `backbone.py:995` says "16.8 MB", which is one projection; both together are 33.55 MB.
- **Where a large file would go, and its digest.** `*.safetensors` and `/data/checkpoints/` are gitignored; the tiny fixtures were force-added. A canonical tensor digest already exists at `run_control.py:999-1003,1140-1158,1308-1322`. It has no Rust port; I searched `crates/` with rg's default settings, not `-uu`.
- **L-cuda-oracle's float64 AdamW test (K11).** The weight-decay vector arrives as an input. eps and weight decay are typed in directly (`gen_goldens.py:437`) rather than taken from F's own optimizer builder. It catches the D3 mutation at 9.0e-8 against a 1e-13 bound.

## Fable's ruling

**Q1 (pass criterion). This is a measure clarification that the lead can ratify before any Rust run, but it must be labelled as loosening two tensors.**
- "Each param's max" was meant as the parameter's scale. A zero-initialised vector has no scale of its own after 20 steps. That inconsistency shows a definition error, not a threshold the human chose.
- No Rust run exists, so nothing is being moved to make a result pass, which is what rule 2 forbids.
- The fix:
  - Every `span_head.*` tensor is measured against the largest value in the whole span head, which is 0.125.
  - Tower tensors keep their own max.
  - The loss bars are unchanged.
- **Don't widen the tower bar** because the Rust trainer's floor is probably higher. That is inferred, not measured; if the tower fails, it is reported as a failure.
- **Don't gate the step-0 gradients at 1e-4.** The torch-vs-torch floor there is already 6.31e-5, so they stay report-only at this rung.
- **Recommended measurement, not a gate.** L-oracle adds a second independent arm that swaps kernels: the recurrent GDN (`torch_recurrent_gated_delta_rule`, `modeling_qwen3_5.py:327`) and eager attention. This gives a floor that doesn't share kernels with the reference, recorded as `independent_check_kernel_diverse`.

**Q2 (D1, D2, D3). Neither of the offered options as stated; three tests at rung (a), none of them a rung (b) bar.**
- D1 is a configuration error and D3 is an arithmetic error, so no training trajectory at any learning rate is the right instrument.
- The three tests:
  1. **Exact table equality, no tolerance.** The trainer's per-parameter table of learning-rate scale, weight decay, eps and betas must equal what Lappi's `layerwise_param_groups` and F's builder produce, name by name, on the real 2B parameter names.
  2. **The existing per-entry AdamW golden at ≤1e-6, made sensitive.** It runs at a pre-registered setting, for example lr 1e-2, weight decay 0.1, scales {1.0, 0.1}, 5 steps, where each planted mutation clears the bound by at least 100×. The ratio is recorded before the run. The golden is built through F's builder and includes norm and `dt_bias` names plus one parameter with no gradient (D7). The reason: at F's settings, D3's per-step effect is about 9e-8 and can't be seen under 1e-6.
  3. **A wiring test.** A recording step-provider checks that the table handed to `adamw_step` is the one test 1 validated.
- Rung (b) already proves D2's learning-rate wiring, which it catches at 580× the floor.
- The lr 1e-3 arm is carried report-only on losses. It can become a gate only once a Rust loss floor at 1e-3 exists.

**Q3 (clip clamp). Leaving it unexercised is not acceptable, and covering it is cheap.**
- A host unit test of the clip coefficient, exact, covering norm below 1, exactly 1 − 1e-6, above 1, zero, and non-finite (refused).
- A second rung (b) arm with `max_grad_norm = 150`, inside the measured 62–438 range, so both branches run in the composed step under the same bars. `run_arm` already takes the parameter (`qd_train_oracle_tiny.py:681`). F's recipe is unchanged.

**Q4 (span-head init).**
- **Who writes it:** L-oracle's tool, as a fixture generator, which is a permitted use of Python. Porting torch's random stream to Rust would create a second owner of the same guarantee, and that kind of copy drifts. The tool already has the pattern at `:705-713`.
- **Where it lives:** the binary goes under the ignored path `data/checkpoints/span-head-init/`. A tracked manifest goes at `crates/qd-train/tests/fixtures/span-head-init-seed0.manifest.json`.
- **What pins it:** the file's sha256 plus the repo's existing digest (`_sidecar_digest` over the `qd-tensor-ref-v1` per-tensor digests), not the oracle's ad hoc `digest_tensors`.
  - Rust ports that formula, with a cross-language test on both the tiny head and the 2048-wide file.
  - The Rust trainer refuses to start on a mismatch.
- **The GH200 torch arms:** they build their own head from seed 0 and record `train.span_head_init_digest` as an additive metric, which changes no row hash. Gate 0 compares the digests. On a mismatch the result is `digest_mismatch`, and the pre-registered fallback is an additive `--span-head-init PATH` flag.
- Also fix the "16.8 MB" comment at `backbone.py:995`.

## Exact amendment text (append under Q3, rung b)

> **Amendment 2 (2026-10-01, Fable; ratified by the lead, pre-run). Rung (b) final-weights measure, weight-decay detection, clip coverage, span-head init.**
> **(i) Measure.** "Final weights ≤1e-5 of each param's max" is clarified: for every `span_head.*` tensor the denominator is max over `span_head.*` of max|w_ref|; every tower tensor keeps its own max; loss bars unchanged. Reason: the abstain vectors start at zero (`heads.py:313-314`), so their max after 20 steps is their displacement (4.788e-5), and the measure held them ~500× tighter than every other tensor; two correct torch implementations already differ by 9.12e-6 there (manifest `independent_check_fp32`), against 9.66e-7 on the tower. **This loosens the effective tolerance on `abstain_start`/`abstain_end` from ~4.8e-10 to ~1.25e-6 absolute**; it is recorded as a gate change made before any Rust run existed (no `mac-ojas-rung-b` row), not as "tolerance unchanged". The test prints rust-vs-ref over own max and over displacement per tensor. `grads_step0` is report-only at this rung.
> **(ii) D1/D2/D3** are not gated by rung (b)'s trajectory at any lr (D1 0.54–1.6×, D3 1.2–1.8× of the floor at F's lr; the lr 1e-3 floor fails the bar). They are gated at rung (a) by: exact equality of the trainer's per-entry `(lr_scale, weight_decay, eps, betas)` table against Lappi's `layerwise_param_groups` + F's builder on the 2B names; the per-entry AdamW golden run at a pre-registered decay-sensitive setting where each pre-registered mutation clears ≤1e-6 by ≥100× (ratio recorded before the run), the golden built through F's builder and including norm/`dt_bias` names and a no-gradient entry; and a recording-provider test that the trainer hands that table to `adamw_step`. The lr 1e-3 arm is report-only on losses.
> **(iii) Clip.** A host unit test of `min(1, max_norm/(norm+1e-6))` (norm <1, =1−1e-6, >1, 0, non-finite → refuse), and a second rung (b) arm at `max_grad_norm=150` (inside the measured 62–438) on the same bars. F's recipe is unchanged.
> **(iv) Span-head init.** L-oracle writes `span_head_init-seed0.safetensors` from `QwenDecisionStep(seed=0)`'s construction path (`backbone.py:983-1000`), stored under `data/checkpoints/span-head-init/` (ignored), with a tracked manifest at `crates/qd-train/tests/fixtures/span-head-init-seed0.manifest.json` recording `sha256` of the file and the content digest by `run_control._sidecar_digest` over `qd-tensor-ref-v1` (names `span_head.*`, dtype `float32`), plus generator provenance. The Rust trainer loads it by path, recomputes both, and refuses on mismatch. Torch arms build their head from seed 0 and record `train.span_head_init_digest`; gate 0 compares; mismatch is `digest_mismatch`, fallback an additive `--span-head-init PATH`.
> **Sequence (rule 2):** the lead records this before any test encodes the measure; the first rung (b) Rust test is written against the amended measure.

## Gap ids

**Append:**
- `GAP-OJAS-RUNGD-SPAN-HEAD-INIT-UNOWNED-2026-10-01`: the file is named in the design but not in L-oracle's lane row, and has no manifest or digest formula.
- `GAP-L-ORACLE-RUNG-B-FLOOR-SHARES-KERNELS-2026-10-01`: the floor was measured with GDN and attention kernels shared with the reference; the Rust floor is unmeasured.
- `GAP-OJAS-K11-GOLDEN-RETYPES-F-HYPER-2026-10-01`: `gen_goldens.py:437` types the hyperparameters in rather than using F's builder.

**Close once the amendment is recorded, citing it:**
- `GAP-L-ORACLE-RUNG-B-FINAL-WEIGHTS-BAR-AT-FLOOR-2026-10-01`
- `GAP-L-ORACLE-RUNG-B-BLIND-TO-WD-AT-F-LR-2026-10-01`, resolved as "not a rung (b) bar; three rung (a) tests"
- `GAP-L-ORACLE-CLIP-CLAMP-BRANCH-UNEXERCISED-2026-10-01`

**What is verified and what isn't:** every number above I checked against the manifest, the trajectory, the safetensors or the code at the cited line. That the Rust trainer's floor will be higher than torch's is inferred, not measured, and it is the reason the amendment leaves the tower bar alone.
