# What a Rust-written `ft` row and checkpoint must be for `real_ft_run --score-checkpoint` (L-oracle, 2026-10-01)

A test-first spec for L-trainer's ledger writer and exporter (fable-advice.md Q1 row 15, Q3 rung
(d), human ask 7). Each item is a check the scorer makes today, with the line that makes it.
Labels: **[V]** read at file:line (Lappi main `6a06bd3`); **[I]** inferred from read code;
**[U]** not verified. Scope of the read: `tools/real_ft_run.py` 1463-1603 (checkpoint names),
1758-1835 (cost, control), 1884-1896 (protocol), 1929-1988 (recorder, optimizer spec),
2097-2335 (how a Python `ft` row is built), 5419-6144 (the whole `--score-checkpoint` section),
7419-7480 (its argv checks); `python/qd_train/run_control.py` 520-590, 626-870, 990-1380,
1383-1897; `python/qd_train/ledger.py` 85-170, 470-520, 700-770, 866-1230, 1674-1960;
`python/qd_train/backbone.py` 1199-1360; `tools/ckpt_average.py` 95-160, 764-1060; the two tests
that read every committed ledger file. A check outside that scope is not claimed absent.

## 0. The one path a single Metal-trained seed can take, and where it stops today

A Rust artifact is one seed's model, so it is scored as one seed's epoch checkpoint:

```
real_ft_run.py --score-checkpoint <dir>/epoch-seed0-<device>.json --ft-row-id <id> \
  --ft-ledger <ledger> --seeds 0 --devices <scoring device> --score-val ... --real-backbone <snapshot>
```

The average and ensemble paths are closed to it (section 4). The single-seed path stops at the
**file name**, before any pairing: `_resume_arm` accepts a device only from
`ARM_DEVICES = ("cpu", "mps", "cuda")`, so `epoch-seed0-metal.json` is refused [V
`real_ft_run.py:1468,1593-1597`].

## 1. Ordered checks on the single-seed path

| # | Check | Where | Status |
| --- | --- | --- | --- |
| 1 | argv: `--score-val`, `--real-backbone`, `--ft-ledger`, `--devices`, `--ft-row-id` present; one `--ft-row-id`; exactly one `--devices` and one `--seeds`; no `--epoch`/`--resume-from` | `real_ft_run.py:7419-7467` | [V] |
| 2 | file name `<tag>-seed<N>-<device>.json`: suffix `.json`; `tag` in `("memorise", "epoch")`; `device` in `("cpu", "mps", "cuda")`; `seed<digits>` | `_resume_arm`, `:1570-1603`; `ARM_TAGS`/`ARM_DEVICES` `:1463,1468` | [V] -- **refuses `metal`** |
| 3 | `tag == "epoch"`; `--seeds == [N]` | `_seed_weights`, `:5751-5755` | [V] |
| 4 | the ft row: `--ft-row-id` is >= 8 characters and prefixes **exactly one** row of `--ft-ledger`; that row has `run_kind == "ft"` and `status == "completed"`. Read as raw JSON lines (no chain or schema check here) | `_ft_row`, `:5622-5641` | [V] |
| 5 | `recipe.tag == "epoch"` | `_ft_row_mismatches`, `:5733` | [V] |
| 6 | `recipe.device ==` the device in the file name | `:5734` | [V] |
| 7 | `protocol.seed ==` the seed in the file name | `:5735` | [V] |
| 8 | `recipe.shard_hash ==` the scoring run's shard set `header.shard_hash()` | `:5736` | [V] |
| 9 | `recipe.backbone_snapshot ==` `--real-backbone`'s directory name (the HF revision, never a path) | `:5737`; why a name: `_backbone_commit` `:1838-1870` | [V] |
| 10 | `metrics["train.optimizer_steps"]["value"]` is present and an int | `:5767` | [V] -- a missing key is a `KeyError`, not a refusal message [I] |
| 11 | checkpoint body: JSON object with a valid `payload_digest` (section 2) | `Checkpoint.read_weights` -> `read_body`, `run_control.py:1804,1763-1774` | [V] |
| 12 | body declares a non-null `sidecar` and has `model_state` as an object with `tower` and `span_head` | `run_control.py:1805-1823` | [V] |
| 13 | sidecar file `<stem>.<sidecar.digest[:16]>.safetensors` beside the JSON | `:1510-1520,1829-1831`; `_SIDECAR_DIGEST_CHARS = 16` `:1027` | [V] |
| 14 | every tensor named under `tower`/`span_head` is in the sidecar, with the dtype, shape, byte count and per-tensor digest the body records | `:1833-1889` via `_join_tensors` `:1258-1300` | [V] |
| 15 | body has `optimizer_step`, `seed`, `schedule` (read into the header unconditionally) | `:1890-1896` | [V] |
| 16 | `body.optimizer_step == train.optimizer_steps` and `body.seed == N` | `:5769-5774` | [V] |
| 17 | the scoring step is built from `recipe.attn_implementation`, `recipe.lr`, `recipe.span_weight`, `recipe.get("optimizer_recipe", "bf16")`, with `total_steps = body.optimizer_step` | `_scoring_step`, `:5707-5722` | [V] |
| 18 | `model_state.span_weight`, `model_state.vocab_size` present; `vocab_size ==` the remapped tower's; `span_weight ==` `recipe.span_weight` | `QwenDecisionStep.load_weights`, `backbone.py:1335-1351` | [V] |
| 19 | `tower` loads with `load_state_dict(strict=True)` under transformers' names below the text tower (`embed_tokens.weight`, `layers.N....`, `norm.weight`; no `model.language_model.` prefix); `span_head` the same with `start_proj.weight`, `end_proj.weight`, `abstain_start`, `abstain_end`; each tensor is cast to `--score-dtype` | `backbone.py:1352-1356`; names from `state()` `:1223-1236` | [V] |
| 20 | eval row: `metrics["train.termination"]["value"]` read with `.get` (absent -> `None`); `row_id`; the sidecar digest names the scored checkpoint | `scored_model`, `:5611-5619` | [V] |
| 21 | the eval row's recipe copies every `BACKBONE_KEYS` and `RECIPE_PIECE_KEYS` entry present in the ft recipe | `:6096`; lists `:272-297,486-493` | [V] |

Nothing in the read scope recomputes `protocol.recipe_hash` from `recipe` on the scoring path
[V for the scope above; [U] repo-wide].

**Optimizer recipe at scoring.** Item 17 builds an optimizer it never steps. With
`--score-dtype bf16` and a recipe that says `"bf16"` (or says nothing), `build_optimizer` refuses
any schedule past bf16's 383-step moment fidelity [V `optim.py:576-590`; the incident is recorded
at `real_ft_run.py:1979-1981`]. A Metal row whose numerics are the fp32-master analogue (Fable row
9) should record `optimizer_recipe: "master"` [I].

## 2. The checkpoint the Rust exporter must write

Format owner: `run_control.Checkpoint` [V `run_control.py:1383-1897`]. Item-by-item:

- **Body.** Keys `position{epoch,index}`, `optimizer_step`, `seed`, `schedule{peak_lr,
  total_steps, warmup_steps, min_lr}`, `loss_log[{step,epoch,index,loss_hex}]`, `loss_digest`,
  `consumed_digest`, `sidecar`, `model_state`, then `payload_digest` [V `:1475-1507`,
  `:645-646`, `:377-383`, LossPoint `:676-683`]. `read_weights` needs only items 11-16;
  `Checkpoint.read`/`from_json` (resume, `ckpt_average`'s `--from` reading) also verify
  `loss_digest` and the sidecar set digest [V `:1532-1587`] -- write all of them.
- **`payload_digest`** = `sha256(json.dumps(body_without_payload_digest, sort_keys=True,
  separators=(",", ":")))` with Python's default `ensure_ascii=True` [V `:841-845`]. A Rust
  writer must reproduce Python's `json.dumps` byte for byte: key order, separators, `\uXXXX`
  escapes, and **float formatting** (`repr`: `1e-05`, `1.0000000000000002e-06`, not serde_json's
  `1e-5`). The schedule's floats make this unavoidable [I].
- **`loss_digest`** = `sha256(json.dumps([point...], sort_keys=True, separators=(",", ":")))`, each
  loss as `float.hex()` [V `:756-761`, `:676-683`].
- **`consumed_digest`**: 64 lowercase hex, sha256 over `b"qd-consumed-prefix-v1"` then per batch
  `b"\x00" + n_parts(u32 BE)` and per part `len(u64 BE) + bytes`; must equal the empty-prefix
  digest iff `position.index == 0` [V `:771-828`, `:1458-1468`]. What `trainer._fold` folds per
  batch is **[U]** here (`trainer.py:589-614`, not read by this lane; L-data owns the
  consumed-digest parity).
- **Tensors** go in `model_state` as `{"__tensor_ref__": {"key", "dtype", "shape", "nbytes",
  "digest"}}` [V `:1008,1161-1168,1238`]. `key` is the tree path in Python `repr` notation:
  `model_state['tower']['layers.0.mlp.down_proj.weight']` (single quotes, from `{key!r}`) [V
  `:1243,1251`]; the sidecar's safetensors names are these keys [V `:1709-1720`].
- **Per-tensor digest**: `sha256(b"qd-tensor-ref-v1" + len(dtype)(u32 BE) + dtype + ndim(u32 BE)
  + each axis(u64 BE) + nbytes(u64 BE) + data)`, dtype spelled as torch names it (`bfloat16`,
  `float32`) [V `:999,1140-1159`]. Non-finite values are refused on read [V `:1364-1369`;
  probe table `:973-990`].
- **Sidecar set digest**: `sha256(b"qd-checkpoint-sidecar-v1" + count(u64 BE) + for each key in
  sorted order: len(u64 BE) + key utf-8 + raw 32-byte tensor digest)`; `sidecar = {digest,
  n_tensors, nbytes, format: "safetensors"}` [V `:1003,1308-1322,1489-1498`].
- **What `model_state` holds** (Python's writer): `tower`, `span_head`, `optimizer`,
  `micro_batches`, `channel_log{letter,span}` (float.hex), `span_weight`, `vocab_size` [V
  `backbone.py:1238-1261`]. Scoring reads `tower`, `span_head` and the scalars; resume and
  `ckpt_average --from masters` read `optimizer` (masters at `model_state['optimizer']
  ['masters']`) [V `ckpt_average.py:114-115`]. Whether a weights-only Rust checkpoint without
  `optimizer` is acceptable is a scope decision: it scores, it cannot resume [I].
- **Dtype of the exported tower.** F's checkpoint stores the **live bf16** tower (the
  `state_dict` of the bf16 model, i.e. RNE of the masters) and the f32 span head [V
  `backbone.py:1223-1236`, `optim.py:439-441`]; Fable row 15's export (f32 -> bf16 RNE `tower`,
  f32 `span_head`) matches that [I].
- Limits: body <= 1 GiB, sidecar <= 32 GiB, <= 2^20 tensors, little-endian [V `:190-211,
  1640-1664`].

## 3. The `ft` row the Rust writer must append

If the row lives in a committed `ledger/*.jsonl`, the repository's own test runs
`Ledger(path).verify_chain()` over every file [V `python/tests/test_ledger.py:1449-1474`], and
`test_ledger_provenance` requires every new training row there to carry `code_that_ran` [V
`python/tests/test_ledger_provenance.py:125-150`].

- **One line**, `json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` --
  note **`ensure_ascii=False`** here, unlike the checkpoint digest [V `ledger.py:475-477,1220`].
- **Chain.** `prev_row_hash` = sha256 of the previous line's exact bytes, `null` on a file's
  first line [V `:1152-1172,1219`]. Append under an exclusive `flock`, `O_APPEND`, `fsync`; refuse
  a duplicate `row_id` [V `:1208-1229`].
- **Fields** [V `:1007-1029`]: `row_id`, `written_at`, `prev_row_hash`, `protocol_hash`,
  `protocol`, `run_kind`, `status`, `quick`, `quick_reason`, `code_commit`, `env`, `metrics`
  (sorted), `noul_rate`, `controls` (sorted), `gates` (sorted), `wall_clock_s`,
  `wall_clock_source`, `cost_usd`, `notes`, `recipe`.
- **Validation on read** [V `:913-1001,1031-1080`]: `protocol_hash ==
  sha256(_canonical(protocol))` [V `:514-515,1034-1040`]; `run_kind` in the RunKind list and
  `status` in `completed|killed|failed` [V `:85-96`]; `quick=True` needs a non-empty
  `quick_reason`, `quick=False` forbids one; `wall_clock_source` in `caller|recorder|unrecorded`;
  `wall_clock_s`, `cost_usd` finite and >= 0; `recipe` a non-empty string-keyed mapping or
  `null`; no protocol component may be the build marker `n/a:build` on an `ft` row; every
  tri-state parses.
- **Tri-state JSON** [V `tristate.py:79-111`]: `{"state":"ran","passed":bool[,"value"]
  [,"n","n_total"][,"detail"]}` or `{"state":"not_run","reason":str}`.
- **Protocol**, as `real_ft_run._protocol` builds it [V `real_ft_run.py:1884-1896`]:
  `data_snapshot_hash` and `tokenizer_hash` from the shard header; `backbone_commit =
  _backbone_commit(recipe)` (the snapshot revision) [V `:1838-1870`]; `recipe_hash =
  sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")))` (default `ensure_ascii`);
  `seed`.
- **Recipe keys** the scorer reads or carries (section 1): `tag` = `"epoch"`, `device`,
  `shard_hash`, `backbone_snapshot`, `attn_implementation`, `lr`, `span_weight`,
  `optimizer_recipe`, plus every `RECIPE_PIECE_KEYS` entry the run used (`lower_layers_n`,
  `lower_lr_scale`, `batch_tokens`, `no_memorise`, `wall_clock_cap_s`, `train_attention_mask`,
  ...) [V `:272-297`; which appear when: `_recipe_pieces` `:333-372`]. Python's `ft` recipe also
  carries `tool`, `passes`, `batches`, `width`, `deterministic`, `gradient_checkpointing`,
  `backbone_params`, `backbone_vocab` [V `:2160-2173,2248-2253`]; the scorer does not read them
  [V for the read scope].
- **Metrics the scorer reads**: `train.optimizer_steps` (Ran, int value), `train.termination`
  (Ran, `"steps_exhausted"` for a complete schedule; anything else makes the row quick by rule 8)
  [V `real_ft_run.py:5615,5767`; `ledger.py:1845-1870`].
- **`controls`**: Python fills `shuffled_label`, `privileged_hunk`, `degenerate_head`,
  `transfer_gate` with `not_run` when a run did not evaluate them [V `ledger.py:155-160,
  1840-1843`]; a Rust row should carry the same four [I].
- **`code_that_ran`**: a Ran whose value is a closure digest [V `ledger.py:744-770,1883-1893`].
  Python defines it over `python/qd_train` plus the invoking tool's sibling-import closure; a
  Rust trainer has no such definition yet [U] -- see section 5, item 5.
- **Quick**: rung (d)'s row is `quick=True` with Fable's reason (rule 8) [V fable-advice.md Q3].
  A quick row still scores on this path: `_ft_row` checks only kind and status [V `:5636-5640`].

## 4. The average/ensemble paths are closed to one Rust artifact

- `read_manifest` refuses a manifest naming **fewer than 2 inputs** [V `ckpt_average.py:955-956`].
  Fable row 15 asked whether a one-input `from="tower"` manifest is accepted: **it is not**.
- Each input must be a `Checkpoint` JSON body on disk whose `payload_digest`, `seed` and
  `optimizer_step` match the manifest [V `ckpt_average.py:1002-1039`], so a manifest cannot stand
  in for the checkpoint format.
- Every ft row of an average or ensemble must be **not quick** [V `real_ft_run.py:5786-5791`]:
  rung (d)'s quick row can never be averaged or ensembled, whatever its format.
- For the record, the manifest fields `write` emits [V `ckpt_average.py:809-833`]: `tool`,
  `method`, `source`, `from` (`tower`|`masters`, `:113`), `accumulator`, `resumable: false`,
  `why_not_resumable`, `n_inputs`, `optimizer_step`, `schedule`, `vocab_size`, `span_weight`,
  `inputs[{path, resolved_path, seed, payload_digest}]` (`:137-150`), `ft_row_ids`,
  `safetensors_sha256`, `n_tensors`, `tensor_sources`, `master_index`, optional
  `norm_preserving`; and what `read_manifest` checks of them [V `:931-1000`].

## 5. What would need an additive scorer change for `recipe.device = "metal"` (human ask 7 -- not made here)

1. **The file name.** `ARM_DEVICES` gates both writing (`_checkpoint_name` `:1552-1567`) and
   reading (`_resume_arm` `:1570-1603`). Additive: a separate scorable-device set that adds
   `metal` for `_resume_arm` on the `--score-checkpoint` path only, leaving training devices
   unchanged. With that, items 5-9 hold unchanged (`recipe.device` is compared with the file
   name, both `metal`) [I].
2. **The row's cost.** `RunRecorder` refuses `cost=None` on a device outside
   `CostEstimate.LOCAL_DEVICES = {"cpu", "mps"}`, and `CostEstimate.for_device("metal")` refuses
   without a rate, an instance and a GPU count [V `ledger.py:1740-1754`;
   `run_control.py:526-585`]. A Rust writer bypasses both, but a `metal` row at `cost_usd 0.0`
   is exactly the unpriced-device pattern these refusals exist for; adding `metal` to
   `LOCAL_DEVICES` (it is this Mac) is the additive change, and it touches rule 4's cost
   machinery, so it is the human's call.
3. **`env.device`.** `Environment.device` is a free string [V `ledger.py:576-595`]; nothing in
   the read scope refuses `metal` there [V for the scope].
4. **Campaign-device quick reason.** `quick_reasons` flags a device outside
   `CAMPAIGN_DEVICES = {"cuda"}` [V `real_ft_run.py:310,3281-3320`]; on the scoring path it is
   given the **scoring** device, not the ft row's [V `:6131-6144`], so `metal` does not reach it.
   The rung (d) row is quick by rule 8 regardless.
5. **Provenance for a Rust trainer.** `code_that_ran` is defined over Python sources [V
   `ledger.py:700-743`]; a committed Rust row needs a defined digest (e.g. over the trainer
   crate's sources and the tessl commit) or `test_ledger_provenance` fails on arrival [V
   `test_ledger_provenance.py:125-150`]. A definition, not a scorer change; the lead's call.
6. **The checkpoint format needs no scorer change** if the Rust exporter writes section 2 exactly
   [I]. The alternative -- teaching the scorer a Rust-native artifact -- is the larger change and
   is not recommended here.

## 6. The tests that pin this, written before the writer

For L-trainer, each failing against today's tree and passing once the Rust writer is right:

1. **Body digest parity**: a Rust-written checkpoint body passes `Checkpoint.read_body` (Python
   oracle in `tools/`), and the Rust-computed `payload_digest`, `loss_digest`, per-tensor and
   sidecar digests equal Python's over the same values -- including a schedule whose `min_lr` is
   `1.0000000000000002e-06`.
2. **Weights round trip**: `Checkpoint.read_weights(path, subtrees=("tower", "span_head"))` then
   `QwenDecisionStep.load_weights` on the tiny tower (`crates/qd-train/tests/fixtures/
   tiny-published`) reproduces the exported tensors bit for bit.
3. **Row parity**: a Rust-appended row passes `Ledger(path).verify_chain()` (chain, protocol hash,
   tri-states, quick rules) in a scratch ledger that already holds a Python-written row.
4. **Pairing**: `_ft_row_mismatches` returns `{}` for the Rust row and its checkpoint's
   `(seed, device)` against the shard set and snapshot it trained on; and, as written today,
   `_resume_arm("epoch-seed0-metal.json")` raises -- the fail-first test for ask 7, which flips
   only when the human approves the change in section 5.
