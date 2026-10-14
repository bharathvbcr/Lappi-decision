# Fable ruling: how the Arm B 200-row MPS probe is constructed and run

Date: 2026-10-14. Judge: Fable ("choose best"). Resolves `GAP-ARM-B-PROBE-HAS-NO-RUNNABLE-COMMAND-2026-10-14`
(and the `next` lines of `GAP-ARM-B-SHARDS-BUILT-PROBE-NOT-RUN-2026-10-13`, `GAP-ARM-B-SWIFT-V2-REBUILT-E1CCB673-SUPERSEDED-2026-10-13`).
Recorded BEFORE any probe result exists: no candidate row has been scored. Every choice below that the
pre-registration (`campaign/v6-arm-b-mining-preregistered.json`, "the record") does not make is marked
**construction choice**; nothing it registers (scorer, dtype, rule, 200 rows, formula, caps, gates) moves.

Tooling. `devmap_status`: generation 3889, `is_fresh: false`, so no claim here is graph-confirmed; every `file:line` is from
`rg`/`sed`/`Read` on HEAD `e9bf68f`, every count from `python3 -I -c` over the files named. `gitpulse_insights`: every facet `ok`,
5 of 5 worktrees scanned, `data/pool/.build-b/` dirty in both kept worktrees (a3ceb68f, aee488bf), 2 live Claude sessions in the
main worktree (pids 12401, 47940). `ListAgents` is not in this session's tool set. The lane appends the `GAP-` record; I write only this file.

## Evidence (all [V] unless marked)

- Scorer: `/Users/bharath/qd-campaign/weights-2026-10-05/v5-seed1/epoch-seed1-cuda-step12176.259933b9cac3ad67.safetensors`
  **exists** (3,797,264,240 B), `shasum -a 256` = `84edcc91…405e59` = the record's pin. Its loader sidecar `epoch-seed1-cuda-step12176.json`
  is beside it (`optimizer_step` 12176 = `schedule.total_steps`: the final step; `sidecar.digest 259933b9…`). v5 seed 1's ft row is
  `b0d9454e` in `ledger/h100x2-v5-2026-10-03.jsonl` (`HANDOFF/lead-pipeline-2026-10-03.md:238`; `shard_hash 10808254…`,
  `backbone_snapshot b1485b2f…`, cuda, `steps_exhausted`); `9da4c6b1` is the J5′ shuffled-label control at seed 1 (`:695`), not the scorer.
- The pairing check (`_ft_row_mismatches`, `real_ft_run.py:7748-7763`: ft `shard_hash` == the `--out` train reader's) is satisfiable:
  `/Users/bharath/qd-campaign/phase4-v5r-2026-10-03/shards/train` has shard `10808254…`, remap `e7d0890c…`, rev `8e6a0095…`; the
  backbone snapshot `b1485b2f…` is in `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/`.
- Candidate shard headers: ts-composed `958970ba…` rev `922ece55` 2,500 seq; single-file `7fa909d5…` rev `922ece55`; swift-composed
  (a3ceb68f, ledger `cd1b1168`) `2adfef21…` rev `1a946c19`. **All three carry remap `e7d0890c…` = v5 train's**, split `train`,
  `refuse-gold`, no `report_only`; `require_gate_population` (`python/qd_train/artifacts.py:411-420`) refuses refuse-gold.
- `plan/candidates.jsonl` (a3ceb68f; sha256 = `plan.json.candidates_sha256 1c2d0978…`, kept 9,908, `ids_unique` true) is in the record's
  corpus order: rows 1-1250 `arm-b-ts-v1:compose:train:…`, 1251-2565 `arm-b-swift-v2:…`, 2566-9908 `arm-b-sfm-v1:…`. **Rows 1-200 are
  `arm-b-ts-v1:compose:train:000000`–`000199`**: TypeScript, composed, split train; stub 118 / clean 50 / logic 17 / cosmetic 15;
  `est_tokens` 978-9,384, mean 5,180; all 200 in ts-composed's `sequence_index.json` (0 excluded). `plan.json.candidate_ids` is sorted, not corpus order.
- Scoring path: `--score-checkpoint` needs `--score-val`, `--real-backbone`, `--ft-ledger`, `--ft-row-id`, one `--devices`, one `--seeds`
  (`:9836-9890`); `--score-val` opens `<out>/shards/val` through `open_val_reader` + `require_gate_population` (`:5266-5287`), so the
  candidate set cannot be "the val set"; `--score-dtype fp32` is the default (`:10878`). The one path that scores a pinned non-val
  set is the composed-slice pass (`open_composed_slice :8528`, `run_composed_slice :8750`), whose contract is slice-specific: ids
  `compose:val/diag` (`composed_slice.py:187-191`), manifest `mode: report_only` (`real_tokenizer_pipeline.py:1096-1101`), pinned
  `MEASURED_EXCLUSIONS`, and the pipeline refuses a training corpus as a slice. Forcing the probe through it is refused (over-unification).
- Verdict lines (`_verdict_lines :10465-10513`) carry `noul_row`, `perm`, `top_permuted`, `expected_abstain` when the second pass ran;
  `arm_b_mine.abstentions` (`:630-650`) reads exactly those; the second pass is built per set by `second_pass_batches` (`:5878`).
  Rule 8 on MPS is automatic: `quick_reasons` (`:5061-5126`) marks any non-campaign device.
- Memory: `tools/mac_heavy.sh` caps **phys_footprint** (commit `c34dc0b`, ft-0eb828f178f; `git merge-base --is-ancestor c34dc0b HEAD` holds),
  default 52 GiB, exit 70 on breach, 71 when the reader fails. `real_ft_run.py:136-137` sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`/`LOW 0.9`;
  `device_budget` refuses before the step (`:3258-3260`), Metal-side allocations excluded from its estimate by its own comment (`:3254-3257`).
  `BUILD_REPORT.md:159-160` ("fix only in review") is stale. Precedent: three completed fp32 MPS `epoch-score-val` rows over 2,332 val rows,
  1,363-3,330 s (`ledger/mac-score-ckpt*-2026-09-30.jsonl` rows 20cf3845, 784868b3, 314837f2). The probe rows are wider (max 9,384 tokens): footprint **[U]**.

## Rulings

1. **The 200-row cut is forced by the text.** "The first 200 deduped candidate rows, in the corpus order the seed fixed" is one list,
   not one per corpus: the plan's `candidates.jsonl`, which is the deduped candidate set in the record's corpus order (ts, swift,
   sfm; within a composed corpus, compose's seed-20261008 order). So the probe is `arm-b-ts-v1:compose:train:000000`–`000199`
   from plan `1c2d0978…`. Swift's coverage gate (`passed=False`, 46 over-length rows) is moot: no Swift row is in the cut.
   Consequence, stated and not corrected: these are the longest rows in the candidate pool (sfm rows are ≤ 1,095 tokens), so the
   measured rows_per_s is a lower bound for the mixed scan; the record's formula takes it as-is (ruling 4).
2. **How the rows reach `--score-checkpoint` (construction choice).** A 200-row materialised corpus, then the one shard writer, then a
   new candidate-set mode of the scorer; never a cut of an existing shard set and never `shards/val`:
   (a) `tools/arm_b_mine.py probe` (new subcommand; `PROBE_ROWS = 200` as a constant, no flag) takes `--prereg --plan-dir --out`,
   re-checks `candidates_sha256`, and writes `<out>/corpus/commitpackft-composed-ts-v1/` through `materialise`'s own staging
   (same `load_defect_rows` proof, planbase, manifest recompute, `selected_by_plan` + `probe: {rows: 200, first_id, last_id,
   candidates_sha256}`) and `<out>/probe.json` with every sha256. One owner: it calls `materialise`'s functions, it does not copy them.
   (b) `tools/real_tokenizer_pipeline.py` with the argv recorded in `GAP-ARM-B-SHARDS-BUILT-PROBE-NOT-RUN-2026-10-13` (`--no-repo-history
   --defect-class <that corpus> --defect-download data/pool/commitpackft --vocab full --memo-limit 0 --max-seq-len 10240
   --span-collapse-policy refuse-gold`, `--rev` = the lane's committed HEAD, not `-dirty`) into `<out>/shards/` (split `train`),
   ledger row appended to `ledger/arm-b-shards-2026-10-13.jsonl`. Expect remap `e7d0890c…`; a different remap is a refusal.
   (c) `tools/real_ft_run.py --score-checkpoint` gains `--arm-b-candidates SET_OUT --arm-b-candidates-corpus DIR
   --arm-b-candidates-build-row ROW_ID` (ledger default `ledger/arm-b-shards-2026-10-13.jsonl`), a mode like the OOD-only trajectory
   row (`_ood_only :9142`): no `--score-val`, the val set is never opened, the step is sized by a `ValPlan` of the candidate set.
   `open_candidate_set` is modelled on `open_composed_slice` with its own contract: build row names the set's `shard_hash` and
   `rev`; header split `train`, refuse-gold, not report_only, and `require_gate_population` **must refuse it** (asserted);
   remap == the train reader's; labels from `qd_data.defect_class.load_defect_rows` over the probe corpus (manifest sha256 ==
   the build row's), `rewrite_defect_class`, `pair_labels(require_index=True)`; row ids ⊆ `probe.json`. Decode = `_decode` +
   the permuted second pass over the candidate set (`second_pass_batches`, `score_permutation_consistency`), lines via
   `_verdict_lines` to `--verdicts-out`. **Rule 3**: this is train-side mining data, not held-out; what rule 3 requires is that
   every input and output path passes `assert_path_not_held_out` (segments `heldout`/`held_out`/`held-out` and configured
   roots; `python/qd_train/data_access.py:1-60`) and that the loader path check runs, which (a)-(c) already do; so the paths are
   `<worktree>/data/pool/.build-b/probe/{corpus,shards,verdicts-probe.jsonl}` with no marker, and the v5 set's `shards/val` is
   not decoded (the row records `noul_rate NotRun`).
3. **dtype fp32 (registered), device `mps`, batch width = the candidate set's own plan** (`val_plan`: `batch_tokens` = its widest
   bucket, ≈ 9,400 tokens, so one row per batch at the top bucket); no `--batch-tokens`. Ledger: `ledger/arm-b-probe-2026-10-14.jsonl`
   (construction choice; `--ledger`). The row is `run_kind eval`, tag `arm-b-probe`, quick by `quick_reasons` (device mps) plus the
   reason "the Arm B throughput probe: 200 candidate rows; its abstention count gates nothing and is no yield estimate", gate name
   `arm_b.probe.report_only` on every line (no gate is called that; qd-gate-report refuses it, as `composed_slice.report_only`).
4. **What it measures and decides.** Metrics on the row: `arm_b_probe.rows_decoded`, `.choice_rows`, `.decode_s` (both passes,
   model load excluded and recorded separately as `.load_s`), **`.rows_per_s` = choice rows decoded / decode_s**, and `.abstained`
   (count under `choice_rule_abstentions`, recorded, gates nothing). It decides one number: `rows_scanned = min(20000,
   floor(rows_per_s × 36 × 3600))`, computed by `arm_b_mine.py budget --eval-row <id> --ledger <file>` (extend `budget` to read the
   row, so the number cites a ledger row; the bare `--rows-per-s` form stays for tests). No scan parameter, cap, floor, chunk
   length, corpus, dtype or rule may be amended from the probe's result; the abstention count is not read as yield, and
   `arm_b_mine.py select` is **not** run on probe verdicts (a key-presence check of the lines is allowed; the count stays the scorer's).
5. **The driver and its tests** (every test red on `e9bf68f` before its change lands):
   `python/tests/test_arm_b_mine.py`: `test_probe_writes_the_first_200_plan_rows_in_candidates_order_and_pins_them`;
   `test_probe_refuses_a_plan_whose_candidates_changed_or_an_existing_out`; `test_budget_reads_rows_per_s_from_a_probe_row_and_refuses_another_row`.
   `python/tests/test_real_ft_arm_b_probe.py`: `test_candidate_argv_is_refused_before_anything_loads` (needs `--score-checkpoint`;
   refuses `--score-val`, `--ood`, `--needle`, `--score-plan`); `test_the_candidate_set_is_never_a_gate_population_and_never_opened_as_val`;
   `test_the_probe_row_is_quick_records_rows_per_s_and_abstained_and_no_gate` (pattern: `test_the_slice_row_is_quick_holds_the_tables_and_no_gate`);
   `test_probe_lines_carry_every_key_arm_b_mine_abstentions_reads`. CPU-only until the run; `cargo` untouched. Final command (one job, under the lock):
   ```
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh arm-b-probe-200 \
     env HF_HUB_OFFLINE=1 QD_PREP_BIN=/Users/bharath/Code/research/Lappi-decision/target/release/qd-prep \
     /Users/bharath/.venvs/ml/bin/python -u /Users/bharath/Code/research/Lappi-decision/tools/real_ft_run.py \
     --out /Users/bharath/qd-campaign/phase4-v5r-2026-10-03 --no-repo-history --rev 8e6a00952a09e31a09cc3f5270229a935b7235cc \
     <v5's source flags verbatim from AUDIT/board-sweep-2026-10-12/fable-rescore-unblock-ruling.md:205-215> \
     --real-backbone /Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
     --devices mps --seeds 1 --score-dtype fp32 \
     --score-checkpoint /Users/bharath/qd-campaign/weights-2026-10-05/v5-seed1/epoch-seed1-cuda-step12176.json \
     --ft-ledger /Users/bharath/Code/research/Lappi-decision/ledger/h100x2-v5-2026-10-03.jsonl --ft-row-id b0d9454e \
     --arm-b-candidates <W>/data/pool/.build-b/probe --arm-b-candidates-corpus <W>/data/pool/.build-b/probe/corpus/commitpackft-composed-ts-v1 \
     --arm-b-candidates-build-row <probe shard build row id> --wall-clock-cap-s 7200 \
     --ledger /Users/bharath/Code/research/Lappi-decision/ledger/arm-b-probe-2026-10-14.jsonl \
     --verdicts-out <W>/data/pool/.build-b/probe/verdicts-probe.jsonl
   ```
   `<W>` is the lane's worktree. The lane first confirms the `.json`'s sidecar digest pairs the pinned `.safetensors`
   (`Checkpoint.read_weights` checks `payload_digest`) and that `arm_b_mine.check_scorer` still hashes the pin. A refusal from any
   pairing check is the finding, reported with its message; it is not worked around.
6. **Memory (card ft-0eb828f1).** The card's premise is resolved on HEAD: the cap is phys_footprint, which the commit measured as
   including MPS allocations (`c34dc0b`; the +3.06 GiB measurement in the script header is that commit's, not re-run here). So the
   probe runs with the **default** `MAC_HEAVY_MAX_FOOTPRINT_MB` (52 GiB), never raised, plus the scorer's own watermark 1.0/0.9 and
   `device_budget`. Residual risk is activation memory at ~9.4k-token width in fp32, which no row has measured [U]; the guard is
   the cap itself: an exit 70/71 means the result does not count, the lane reports it with the footprint line, and does not
   retry at a higher cap or in bf16 (`GAP-TORCH-MPS-BF16-LETTER-LOGITS-DRIFT`) without a human yes. Wall cap 7,200 s on the
   launch line. Under rule 4 this is a single-GPU job at $0: no human yes is needed to launch, but the lane runs
   `ListAgents`/`gitpulse_insights` first and holds no other heavy job (`GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-2026-10-02`).

Lane order: probe subcommand + tests → pipeline build + ledger row → scorer mode + tests → commit → the command → `budget` →
`HANDOFF/arm-b-probe-<date>.md` with the row ids, the rows_per_s and rows_scanned, and the gaps closed/opened.
