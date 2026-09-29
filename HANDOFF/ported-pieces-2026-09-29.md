# Ported-pieces lane — 2026-09-29

These pieces were ported from RSI-Jev (MIT, @8f34a4f) and Mapika/decider (Apache-2.0, @23579f7)
into `qd_train`, following `docs/train-plan-2026-09-28.md` ("Borrowed code" and "What gets built on
the Mac first", item 3). All work was on the CPU. Nothing is committed.

## What was measured

- **Ledger rows:** none were written to `ledger/`. The e2e wiring runs wrote rows to scratch
  ledgers on purpose, because they are wiring checks and not measurements.
- **Moment settling** (`qd_train.optim.moment_settling`):

  | dtype | beta2 | settled at | low by | stops at step |
  | --- | --- | --- | --- | --- |
  | bf16 | 0.999 | 0.500000 | 50% | 384 |
  | bf16 | 0.99 | 0.980469 | 1.95% | 184 |
  | bf16 | 0.95 | 0.980469 | 1.95% | 64 |
  | fp32 | 0.95 | 0.999999 | about 0 | 279 |

- **Real Qwen tokenizer, token-space option swap:** the swap equals tokenizing the reordered
  prompt. This held on 216 of 216 checks (6 adversarial option sets × 3 contexts × 3 golds × 4
  permutations).
- **Decontamination, 100-pair shard set at rev 7f23355:** the train shards, used as a replay
  candidate, hit heldout on 222 of 234 rows and val on 0 of 234.

## What changed (uncommitted)

**Modified files:** `python/qd_train/{optim,trainer,backbone,byte_train}.py`,
`tools/{real_ft_run,ft_toy_run}.py`, `python/tests/{test_optim,test_trainer,test_backbone}.py`.

**New files:** `python/qd_train/replay.py`, `tools/ckpt_average.py`, `tools/replay_decontam.py`,
`python/tests/{test_replay,test_real_ft_pieces}.py`.

**Lines:** modified files +1867 / −54. New files add 1597 lines.

### The pieces

1. **Layer-wise lr** (`--lower-layers-n N` and `--lower-layers-lr-scale`, default 0.1)
   - `optim.apply_lr` is now the only writer of `group["lr"]`, at all four drivers.
   - `optim.layerwise_param_groups` is RSI's rule.
   - The flag needs `--real-backbone`.
2. **beta2** (`--beta2`, default 0.999, unchanged)
   - The bf16 fidelity check is now asked at the optimizer's own beta2.
3. **prior_kl replay** (`qd_train.replay`: `prior_kl`, `PriorCache`, `PriorKLReplay`)
   - Flags: `--replay-shards`, `--replay-attestation`, `--replay-cache`, `--replay-weight`,
     `--replay-every` (default 6). Epoch arm only.
   - The KL is computed in fp32.
   - The replay cursor is kept in the checkpoint state.
   - `QwenDecisionStep.load_state` now refuses keys it did not write. That closes the case where a
     replay checkpoint was resumed without replay.
4. **Option permutation** (`trainer.ChoicePermutation` and `permute_choice_row`)
   - Flag: `--option-permutation-seed`. Epoch arm only.
   - The permutation is done in token space.
   - A line start is found by the newline-terminated predecessor ids read from `tokenizer.json`.
5. **Checkpoint averaging** (`tools/ckpt_average.py`)
   - Averages the weights only, and marks the result `resumable: false`.
   - Refuses inputs that differ in schedule, step, architecture or scalars, and refuses
     duplicates.
6. **Decontamination** (`qd_train.replay.decontaminate` and `tools/replay_decontam.py`)
   - RSI's 8-gram containment check at 0.5, with a stable blake2b hash.
   - Writes an attestation that `real_ft_run` checks: replay shard hash, val digest, and corpus
     identity. Held-out rows are read only by the decontam tool (rule 3).
   - It lives in `qd_train`, not `qd_data` as the plan table says, because `qd_data` belongs to
     another lane.

### Coordinator items

- **`ft_split_rows(*, commitpackft, max_pairs, rev, config) -> (train, val)`**
  - `main` calls it. `ft_splits` also returns `heldout`, for the decontam tool.
  - Tests: `test_main_trains_on_exactly_what_ft_split_rows_returns` and
    `test_ft_split_rows_is_the_rebuild_main_used_to_inline`.
- **`--verdicts-out`**
  - Valid only with `--score-val`.
  - Refused at argv time and again at write time if the file exists.
  - Written atomically once, after all seeds.
  - Not written when no eval row was recorded.
  - `slot_name` is threaded into both verdict shapes.
  - Tests: `test_verdict_lines_carry_the_fields_a4_reads` and
    `test_the_decoder_threads_slot_name_into_every_verdict`.

- **`--defect-class` / `--defect-download` / `--defect-max-rows`**
  - These flags pass through `ft_splits` and `ft_split_rows` as extra keyword arguments.
  - `check_defect_source` compares them against `data/pool/train.json` `admitted_source_ids`,
    and refuses in both directions.
  - `pair_labels` pairs every label to its sequence by `(row_id, slot_name)`, using the data-fixes
    lane's `sequence_index.json`. It refuses on any disagreement. A `--defect-class` set that has
    no index is refused, and no reconstructed-order fallback exists.
  - Tests: `test_a_defect_class_shard_set_is_relabelled_by_id_from_the_pipelines_own_rows`
    (runs against a real pipeline build) and `test_pairing_is_by_id_and_refuses_every_disagreement`.
  - `GAP-PORTED-SHARD-EXCLUSIONS-ARE-FREE-TEXT` was opened and then closed.
- **The bf16 refusal now depends on beta2.**
  - `build_optimizer` asks `moment_settling(dtype, beta2)` and `survives(total_steps)`.
  - bf16 at 0.95 is still unsafe: it settles 1.95% low after 64 steps. So it is refused past 63
    steps.
  - Test: `test_the_bf16_fidelity_check_is_asked_at_the_optimizers_own_beta2`.
- **Resume is guarded.**
  - `QwenDecisionStep.load_state` refuses keys it did not write (such as `replay`), and refuses
    optimizer groups whose betas, `lr_scale` or name differ.
  - Both checks run before anything is loaded.

### Decisions

- **Recipe keys are recorded only when a piece is on.** This departs on purpose from the
  precedent in 72fc2d0, where `deterministic` was added unconditionally. The effect is that
  default runs hash exactly as before.
- **`tools/replay_decontam.py` imports `tokenizers`.** It is in the ml venv through transformers
  and was approved on 2026-09-28. It is not declared in `pyproject`.

## What is open

- `GAP-PORTED-LISTAGENTS-UNAVAILABLE`
- `GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN`: the family holdout holds out tasks, not
  content. So any replay set drawn from this corpus is refused, and replay cannot run through
  `main` until a human picks a replay source.
- `GAP-PORTED-BF16-BETA2-095-IS-REFUSED-PAST-64-STEPS`: plan arm A2 "bf16 + beta2 0.95" can
  only run with `--optimizer master`. The gate was not moved.
- `GAP-PORTED-DECODE-KEYERROR-ON-A-LETTER-NO-ROW-USES-AS-GOLD`: this bug predates this lane.
- A3's request-level permutation (`qd_data.defect_class.with_permuted_options`) now sits beside
  this lane's token-space one. They answer different questions: an eval derangement versus train
  augmentation. The lead should decide whether one owner should hold both.

### Not run

- Layer-wise lr on the real 2B tower. It was tested only on the 4-layer tiny tower.
- Replay through `main` end to end. Only the refusal path and a direct `_train` wiring run were
  exercised.
- Resume with replay on a real run.
- Any MPS or GPU run.

## First command for the next lane

This is `SMOKE_CMD.sh` with `CODE=$MAIN` and a fresh `OUT`.

- The frozen `qd-head` worktree lacks these uncommitted changes.
- Its shard sets are refused by the stale-code check because other lanes moved `qd_data`.

Step 2 adds these flags:

```
--lower-layers-n 8 --option-permutation-seed 1 --verdicts-out $SCR/verdicts.jsonl \
--optimizer master --beta2 0.95 --seeds 0 1
```

- **`--optimizer master` is memory-heavy.** It costs about +12 B/param, roughly +17 GB on
  1.4B parameters, on a 64 GB Mac. Without it, drop `--beta2 0.95`: bf16 at 0.95 is refused past
  64 steps.

Then run:

```
python tools/ckpt_average.py --out $SCR/avg.safetensors $CK/memorise-seed0-mps.json $CK/memorise-seed1-mps.json
```

To see the replay refusal, run:

```
python tools/replay_decontam.py ...
```

and then `real_ft_run.py --replay-shards ... --replay-attestation ...`.
