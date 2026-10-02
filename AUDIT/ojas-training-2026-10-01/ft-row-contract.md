# What a Rust-written `ft` row and export must be for `real_ft_run --score-checkpoint` (L-oracle, 2026-10-01)

A test-first spec for L-trainer's ledger writer and exporter (fable-advice.md Q1 row 15, Q3 rung
(d), human ask 7). Each item is a check the scorer makes, with the line that makes it.

**Labels.**

- **[V]** read at file:line on this branch after the merge of main `85caea1` (merge `c631874`).
  That merge includes L-scorer's Metal export route (`2bc29d9`, `da193cc`).
- **[I]** inferred from read code.
- **[U]** not verified.

**Sources and their line numbers.**

- Of the cited sources, only `tools/real_ft_run.py` changed between `6a06bd3` and this merge
  (`git diff --stat`), so every other citation reads the same file as before.
- `real_ft_run.py` line numbers are the merged file's.

**Scope of the read.** A check outside this scope is not claimed absent.

- `tools/real_ft_run.py`:
  - 270-510 (key lists);
  - 1473-1612 (checkpoint names);
  - 1848-1906 (backbone, protocol);
  - 3291-3345 (quick reasons);
  - 5277-5340 and 5440-6070 (the `--score-checkpoint` section and the Metal export route);
  - 6072-6140 (average and ensemble inputs);
  - 6344-6440 (scoring, rule-8 reasons);
  - 7714-7762 (argv checks).
- `tools/ckpt_average.py` 100-150, 760-1068.
- `python/qd_train/run_control.py` 940-1000, 1364-1380, 1383-1897.
- `python/qd_train/ledger.py` 85-170, 470-520, 700-770, 866-1230, 1674-1960.
- `python/qd_train/backbone.py` 978-1002, 1199-1360.
- `python/qd_train/optim.py` 540-604.
- `python/tests/test_real_ft_metal_export.py` (L-scorer's tests).
- The two tests that read every committed ledger file.

**Reconciliation.** This version reconciles the L-oracle side of
`GAP-LSCORER-FT-ROW-CONTRACT-UNRECONCILED-2026-10-01`.

- This document's first version (`75ca2fb`) was written before ask 7 landed. It read the only
  route then open, the `.json` `Checkpoint` (it proposed widening `ARM_DEVICES`).
- L-scorer instead built a separate route: a `.safetensors` export plus manifest, with the
  `.json` route left closed to `metal`.
- This version specifies that route. It agrees with L-scorer's contract
  (`HANDOFF/ojas-scorer-2026-10-01.md`, "The contract the scorer now enforces") item for item,
  and adds what that table leaves out:
  - the tensor names;
  - the ledger-format rules a committed row is held to, which the scorer itself never checks;
  - the open provenance question.
- L-trainer's `ledger.rs` and export do not exist yet (`crates/qd-train/src` holds no such file
  [V]). Reconciling with them is L-trainer's half.

## 0. The route

```
real_ft_run.py --score-checkpoint <dir>/epoch-seed<N>-metal.safetensors --ft-row-id <id> \
  --ft-ledger <ledger> --seeds <N> --devices <scoring device> --score-val --real-backbone <snapshot> ...
```

The routing is decided by the file name [V `real_ft_run.py:5530-5549`, `:5796`].

- One path, suffix `.safetensors`, stem ending `-metal`: a Metal export.
- Any other single `.safetensors`: an average.
- More than one path: an ensemble.

**Closed to a Rust artifact:**

- A `.json` checkpoint named `-metal`: `_resume_arm` accepts only
  `ARM_DEVICES = ("cpu", "mps", "cuda")` [V `:1478,1604-1607`]. L-scorer pins this as a
  characterization [V test `test_a_json_checkpoint_named_for_metal_is_not_a_device_this_tool_trains_on`].
- An ensemble with any `.safetensors` tower [V `:7735-7740`].
- An average (section 4).

## 1. Ordered checks on the Metal export route

| # | Check | Where | Status |
| --- | --- | --- | --- |
| 1 | argv: `--score-val`, `--real-backbone`, `--ft-ledger`, `--devices`, `--ft-row-id` all present; one `--ft-row-id`; exactly one `--devices` and one `--seeds`; no `--epoch`/`--resume-from` | `_check_score_checkpoint_flags`, `:7714-7762` | [V] |
| 2 | name `<tag>-seed<N>-metal.safetensors`: three `-` parts from the right; `seed<digits>`; `tag == "epoch"` | `_metal_export_seed`, `:5879-5892` | [V] |
| 3 | `--seeds == [N]` | `:6011` | [V] |
| 4 | manifest at `<name>.manifest.json` exists and is a UTF-8 JSON object | `_read_metal_manifest`, `:5901-5913`; `manifest_path`, `ckpt_average.py:764-766` | [V] |
| 5 | manifest `from == "qd-train-export"`; an average's `from` (`tower`, `masters`) is refused by name | `:5915-5925`; `SOURCES`, `ckpt_average.py:113` | [V] |
| 6 | manifest fields present with JSON type (bool refused as int): `from` str, `trainer` str, `device` str, `seed` int, `optimizer_step` int, `ft_row_id` str, `vocab_size` int, `span_weight` int or float, `safetensors_sha256` str, `n_tensors` int. Extra keys are allowed | `METAL_MANIFEST_FIELDS` `:5467-5471`; loop `:5927-5931` | [V] |
| 7 | `trainer == "qd-train-metal"`, `device == "metal"`, `safetensors_sha256` is 64 lowercase hex | `:5932-5941` | [V] |
| 8 | the manifest's identity is `sha256(manifest file bytes)`; **no canonical form is required** of the manifest's JSON | `:5947-5949` | [V] |
| 9 | the ft row: `--ft-row-id` is >= 8 characters and prefixes **exactly one** line of `--ft-ledger`; that row has `run_kind == "ft"` and `status == "completed"`. The lines are read as raw JSON (no chain, hash or schema check here) | `_ft_row`, `:5720-5739` | [V] |
| 10 | the row has a mapping `recipe` and an int `protocol.seed` | `:6017-6023` | [V] |
| 11 | `recipe.tag == "epoch"`, `recipe.device == "metal"`, `protocol.seed == N`, `recipe.shard_hash ==` the scoring run's `header.shard_hash()`, `recipe.backbone_snapshot ==` `--real-backbone`'s directory name | `_ft_row_mismatches(..., trained_on="metal")`, `:5826-5840`, called `:6025-6031`; why a name, `_backbone_commit` `:1848-1881` | [V] |
| 12 | `recipe.trainer == "qd-train-metal"`; the recipe names `attn_implementation`, `lr`, `span_weight`, `optimizer_recipe` (required, never defaulted, unlike a `.json` row's `optimizer_recipe`) | `_metal_row_problems`, `:5958-5965`; `METAL_ROW_RECIPE_KEYS` `:5475-5477` | [V] |
| 13 | `quick` is a bool; if true, `quick_reason` is a non-empty string | `:5966-5973` | [V] |
| 14 | `metrics["train.optimizer_steps"]["value"]` is an int (not bool); `metrics["train.termination"]["value"]` is a str | `:5975-5983` | [V] |
| 15 | manifest `ft_row_id ==` the row's **full** `row_id`; manifest `seed == N`; manifest `optimizer_step == train.optimizer_steps`; `float(manifest.span_weight) == float(recipe.span_weight)` | `:6033-6050` | [V] |
| 16 | the `.safetensors` bytes hash to `safetensors_sha256`; every tensor name is `tower.<k>` or `span_head.<k>`; the count equals `n_tensors`; each tensor has a stored dtype, a length that matches its shape and finite values; the payload is little-endian | `read_average`, `ckpt_average.py:1042-1069`; `tensor_refs_from_safetensors`, `run_control.py:1364-1379`; dtypes `:946-960` | [V] |
| 17 | the scoring step is built from `recipe.attn_implementation`, `recipe.lr`, `recipe.span_weight` and `optimizer_spec(--score-dtype, recipe.optimizer_recipe)`, with `total_steps = manifest.optimizer_step` | `_scoring_step`, `:5808-5823`; `_checkpoint_step` `:5796-5804` | [V] |
| 18 | `vocab_size` (from the manifest) `==` the remapped tower's; `span_weight` (from the manifest) `==` the step's | `QwenDecisionStep.load_weights`, `backbone.py:1335-1350`; `read_average` returns both `ckpt_average.py:1068-1069` | [V] |
| 19 | `tower` loads with `load_state_dict(strict=True)` under the step's own keys (below; no `model.language_model.` prefix); `span_head` likewise; each tensor is cast to `--score-dtype` | `backbone.py:1351-1356` | [V] |
| 20 | every row scored from the export carries `recipe.trained_by = {trainer, device, source, manifest_sha256}` and `scored_checkpoint = "<name>:<safetensors_sha256>"`; its single-seed metric is `ft_run_row_id` | `:6060-6068`, `scored_model` `:5692-5708`, `recipe_block` `:5583-5592`, `_score_checkpoint` `:6383-6395` | [V] |
| 21 | rule 8 is inherited: a quick ft row makes every row scored from it quick, with its reason; a `train.termination` other than `steps_exhausted` adds a truncated-schedule reason | `provenance_reasons` `:5624-5633`; `model_reasons` `:6425-6439`; `quick_reasons` `:3324-3326` | [V] |
| 22 | the eval row's recipe copies every `BACKBONE_KEYS` and `RECIPE_PIECE_KEYS` entry present in the ft recipe | `:6390`; lists `:282-305`, `:496-503` | [V] |

L-scorer's tests pin items 2-16, 20 and 21, plus an end-to-end CPU scoring that shows the export
and the `.json` checkpoint of the same weights give equal verdicts [V
`python/tests/test_real_ft_metal_export.py:153-586`]. **Not run by this lane**: I read those tests'
names and bodies; this lane did not run them separately from the full suite.

Nothing on this route recomputes `protocol.recipe_hash`, `protocol_hash` or the chain
[V for the read scope; [U] repo-wide]. Those bind only when the row is committed (section 3).

## 2. The export the Rust writer must produce

- **Name**: `epoch-seed<N>-metal.safetensors`, with `epoch-seed<N>-metal.safetensors.manifest.json`
  beside it (items 2, 4).
- **Tensors**: exactly the step's two `state_dict`s, prefixed. These are transformers' names below
  the text tower, then the head's:
  - `tower.embed_tokens.weight`, `tower.layers.<i>.<...>`, `tower.norm.weight`;
  - `span_head.start_proj.weight`, `span_head.end_proj.weight`, `span_head.abstain_start`,
    `span_head.abstain_end`.

  [V `backbone.py:1351-1356` strict load; names from `state()` `:1223-1236`.]
  - The rung (b) fixture (`crates/qd-train/tests/fixtures/tiny-published`) names the tower
    `model.language_model.<k>`, the checkpoint-snapshot prefix. The export maps that to
    `tower.<k>`.
  - The span-head names are the same in both.
  - The tied head has no separate tensor: `lm_head` is the embedding [V `backbone.py:1106-1122`].
- **Dtypes**: any dtype `TensorRef` stores [V item 16]. F's own checkpoint holds the **live bf16**
  tower, which is RNE of the fp32 masters, and an f32 span head [V `backbone.py:1223-1236`,
  `optim.py:439-441`]. Fable row 15's export (f32 -> bf16 RNE `tower`, f32 `span_head`) matches it
  [I]. The scorer casts to `--score-dtype` either way (item 19).
- **Manifest**: the ten fields of item 6 with the values of items 7 and 15. Free JSON formatting:
  its sha256 is taken over whatever bytes were written (item 8).
  - **Ordering constraint** (L-scorer's, confirmed): the manifest names the row's `row_id`, and
    Python mints a row's id only when it appends it [V `ledger.py:1940-1942`, `uuid4`]. The Rust
    writer mints the uuid first, or writes the manifest after the row.
- **What the Metal route does *not* need**, unlike the `.json` route (appendix A):
  - `payload_digest`, the sidecar set digest or the per-tensor domain digests;
  - `consumed_digest`, `loss_digest` or a `schedule` block;
  - Python float formatting in the export.

  Those return only if a Rust run must be resumed or averaged by the Python tools.

## 3. The `ft` row the Rust writer must append

The scorer reads the row as raw JSON (item 9). If the row lives in a committed `ledger/*.jsonl`,
however, two repository tests apply:

- `Ledger(path).verify_chain()` runs over every file
  [V `python/tests/test_ledger.py:1449-1474`];
- `test_ledger_provenance` requires every new training row there to carry `code_that_ran`
  [V `python/tests/test_ledger_provenance.py:125-150`].

The Python-side source is unchanged since `6a06bd3`, so these citations are as before.

- **Line format.** One line, `json.dumps(row, sort_keys=True, separators=(",", ":"),
  ensure_ascii=False)`. Note **`ensure_ascii=False`** [V `ledger.py:475-477,1220`].
  - `crates/qd-train/src/pyjson.rs` already ports Python's `json.dumps` byte for byte: both
    `ensure_ascii` modes, `separators`, `sort_keys` and float `repr` (`1e-05`, not serde's
    `1e-5`) [V its module doc]. Its fitness for the row is [I]; its tests were not run by this
    lane.
- **Chain.** `prev_row_hash` is the sha256 of the previous line's exact bytes, and `null` on a
  file's first line [V `:1152-1172,1219`].
  - Append under an exclusive `flock`, with `O_APPEND` and `fsync`.
  - Refuse a duplicate `row_id` [V `:1208-1229`].
- **Fields** [V `:1007-1029`]:
  - `row_id`, `written_at`, `prev_row_hash`, `protocol_hash`, `protocol`, `run_kind`, `status`;
  - `quick`, `quick_reason`, `code_commit`, `env`;
  - `metrics` (sorted), `noul_rate`, `controls` (sorted), `gates` (sorted);
  - `wall_clock_s`, `wall_clock_source`, `cost_usd`, `notes`, `recipe`.
- **Validation on read** [V `:913-1001,1031-1080`]:
  - `protocol_hash == sha256(_canonical(protocol))` [V `:514-515,1034-1040`].
  - `run_kind` is in the RunKind list; `status` is one of `completed|killed|failed` [V `:85-96`].
  - `quick=True` needs a non-empty `quick_reason`; `quick=False` forbids one.
  - `wall_clock_source` is one of `caller|recorder|unrecorded`.
  - `wall_clock_s` and `cost_usd` are finite and >= 0.
  - `recipe` is a non-empty string-keyed mapping, or `null`.
  - No protocol component of an `ft` row may be the build marker `n/a:build`.
  - Every tri-state parses.
- **Tri-state JSON** [V `tristate.py:79-111`]: `{"state":"ran","passed":bool[,"value"]
  [,"n","n_total"][,"detail"]}` or `{"state":"not_run","reason":str}`.
- **Protocol**, as `real_ft_run._protocol` builds it [V `real_ft_run.py:1894-1906`]:
  - `data_snapshot_hash` and `tokenizer_hash` from the shard header;
  - `backbone_commit = _backbone_commit(recipe)`, the snapshot revision [V `:1848-1881`];
  - `recipe_hash = sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")))`, with the
    **default** `ensure_ascii` (True);
  - `seed`.
- **Recipe**, for the Metal route:
  - `tag="epoch"`, `device="metal"`, `trainer="qd-train-metal"`;
  - `shard_hash`, `backbone_snapshot`;
  - `attn_implementation` (a torch kernel name such as `"sdpa"`, because the scorer builds its
    torch tower from it: L-scorer's "awkward item", confirmed at item 17);
  - `lr`, `span_weight`, `optimizer_recipe`;
  - plus every `RECIPE_PIECE_KEYS` entry the run used, which the eval row copies (item 22):
    `lower_layers_n`, `lower_lr_scale`, `batch_tokens`, `train_attention_mask`, ...
    [V `:282-305`; which appear when, `_recipe_pieces` `:343-394`].
  - `optimizer_recipe: "master"` is F's recipe and matches Fable row 9's numerics [I]. It also
    keeps a bf16 scoring step clear of `build_optimizer`'s 383-step moment-fidelity refusal at a
    longer schedule [V `optim.py:576-590`].
- **Metrics**:
  - `train.optimizer_steps`: Ran, an int value.
  - `train.termination`: Ran. `"steps_exhausted"` for a complete schedule; anything else makes
    every scored row quick (item 21).
- **`controls`**: Python fills `shuffled_label`, `privileged_hunk`, `degenerate_head` and
  `transfer_gate` with `not_run` when a run did not evaluate them [V `ledger.py:155-160,
  1840-1843`]. A Rust row should carry the same four [I].
- **`code_that_ran`**: a Ran whose value is a closure digest [V `ledger.py:744-770,1883-1893`].
  It is defined over Python sources only, so a Rust trainer has no definition yet [U]. See
  `GAP-L-ORACLE-RUST-ROW-CODE-THAT-RAN-UNDEFINED-2026-10-01` and section 5.
- **Quick**: rung (d)'s row is `quick=True` with Fable's reason (rule 8) [V fable-advice.md Q3].
  The Metal route requires the bool either way (item 13).

## 4. The average and ensemble paths are closed to one Rust artifact

- **Average.** `read_manifest` refuses a manifest naming **fewer than 2 inputs**
  [V `ckpt_average.py:955-956`], and each input must be a `Checkpoint` JSON body whose
  `payload_digest`, `seed` and `optimizer_step` match [V `:1002-1039`].
- **Ensemble.** A tower that is a `.safetensors` is refused [V `real_ft_run.py:7735-7740`; test
  `test_a_metal_export_among_ensemble_towers_is_refused`].
- **Quick rows.** Every ft row of an average or an ensemble must say `quick: false`
  [V `_one_configuration`, `:6072-6085`]. Rung (d)'s quick row can never be averaged or
  ensembled, whatever its format.

## 5. What is still open

Ask 7 has landed, so no scorer change is needed for `metal`.

1. **Provenance for a Rust trainer** (`GAP-L-ORACLE-RUST-ROW-CODE-THAT-RAN-UNDEFINED-2026-10-01`).
   - A committed Rust row fails `test_ledger_provenance` on arrival without a defined
     `code_that_ran`. For example, a digest over the trainer crate's sources and the tessl
     commit.
   - This is a definition, not a scorer change, and it is the lead's call.
2. **Cost of a `metal` row.**
   - `RunRecorder` refuses `cost=None` on a device outside
     `CostEstimate.LOCAL_DEVICES = {"cpu", "mps"}`.
   - `CostEstimate.for_device("metal")` refuses without a rate, an instance and a GPU count
     [V `ledger.py:1740-1754`; `run_control.py:526-585`].
   - A Rust writer bypasses both. A hand-set `cost_usd: 0.0` is exactly the unpriced-device
     pattern those refusals exist for.
   - What a Mac `metal` row records as cost is the human's call (rule 4's machinery). The scorer
     does not read it.
3. **Not on this route:**
   - `GAP-SCORE-CHECKPOINT-SINGLE-SEED-ROW-DOES-NOT-INHERIT-QUICK-2026-10-01` concerns torch
     `.json` rows. The Metal route inherits quick (item 21).
   - The campaign-device reason (`CAMPAIGN_DEVICES = {"cuda"}`, `:320`) is computed from the
     **scoring** device [V `:6425-6436`].

## 6. The tests that pin the writer, written before it

L-scorer's 53 tests pin the scorer's side with Python stand-ins. For L-trainer, each of these
fails until the Rust writer exists and passes once it is right:

1. **Export acceptance.** A Rust-written `epoch-seed0-metal.safetensors` plus manifest from the
   tiny tower:
   - passes `real_ft_run._read_metal_manifest` and `ckpt_average.read_average`;
   - its tensors equal, bit for bit, the Rust trainer's in-memory weights. Over the rung (b)
     fixture, that is `fp32/final.safetensors` renamed `model.language_model.` -> `tower.`.
2. **Pairing.** `real_ft_run._metal_export_weights` accepts the Rust row and export together. It
   refuses each one-field mutation of either: L-scorer's malformed cases
   (`test_a_malformed_metal_export_is_refused_before_the_tower`), applied to Rust-written
   artifacts.
3. **Row parity.**
   - A Rust-appended row passes `Ledger(path).verify_chain()` in a scratch ledger that already
     holds a Python-written row.
   - Its `protocol_hash` and `recipe_hash` equal Python's over the same values. The recipe holds
     `lr = 1e-05`, the float where serde and Python disagree.
4. **End to end.** `real_ft_run.main --score-checkpoint epoch-seed0-metal.safetensors` on a
   **Rust-written** export of the tiny tower gives verdicts equal to the same weights scored from
   their `.json` checkpoint. This is L-scorer's
   `test_a_metal_export_of_a_tiny_tower_is_scored_end_to_end_on_cpu` with the Python stand-in
   export replaced by the Rust one.

## Appendix A. The `.json` `Checkpoint` format (not the Metal route)

Kept because a Rust run that must be **resumed** by Python, or be an **input to an average**,
would have to write it [V `run_control.Checkpoint`, `run_control.py:1383-1897`]. The scorer
reads it at `Checkpoint.read_weights` [V `:1777-1897`].

- **Body.** In order:
  - `position{epoch,index}`, `optimizer_step`, `seed`;
  - `schedule{peak_lr, total_steps, warmup_steps, min_lr}`;
  - `loss_log[{step,epoch,index,loss_hex}]`, `loss_digest`, `consumed_digest`;
  - `sidecar`, `model_state`;
  - then `payload_digest`.

  [V `:1475-1507`, `:645-646`, `:377-383`, LossPoint `:676-683`.]
- **`payload_digest`**: `sha256(json.dumps(body_without_payload_digest, sort_keys=True,
  separators=(",", ":")))`, with Python's default `ensure_ascii=True` [V `:841-845`]. Python's
  float `repr` and `\uXXXX` escapes are required byte for byte [I].
- **`loss_digest`**: the same `json.dumps` over the points, each loss as `float.hex()`
  [V `:756-761`].
- **`consumed_digest`**: sha256 over `b"qd-consumed-prefix-v1"`, then, per batch:
  - `b"\x00"` and `n_parts` (u32 BE);
  - per part, `len` (u64 BE) and the bytes.

  [V `:771-828`.] L-data owns its parity (merged `4a5ebda`).
- **Tensors** are `{"__tensor_ref__": {key, dtype, shape, nbytes, digest}}` under `model_state`.
  - The key is the Python `repr` path, e.g. `model_state['tower']['layers.0.mlp.down_proj.weight']`.
  - The per-tensor digest is `sha256(b"qd-tensor-ref-v1" + dtype-len (u32 BE) + dtype + ndim
    (u32 BE) + axes (u64 BE) + nbytes (u64 BE) + data)`.
  - The sidecar set digest uses the domain `b"qd-checkpoint-sidecar-v1"`.
  - The sidecar file is `<stem>.<digest[:16]>.safetensors`.

  [V `:999-1003,1140-1168,1238-1251,1308-1322,1489-1498,1510-1520`.]
- **`model_state`** (Python's writer): `tower`, `span_head`, `optimizer`, `micro_batches`,
  `channel_log`, `span_weight`, `vocab_size` [V `backbone.py:1238-1261`].
  `ckpt_average --from masters` reads `model_state['optimizer']['masters']`
  [V `ckpt_average.py:114-115`].
