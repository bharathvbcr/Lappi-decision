# HANDOFF ojas-scorer (lane L-scorer), 2026-10-01

Human ask 7 of `AUDIT/ojas-training-2026-10-01/fable-advice.md` (approved 2026-10-01):
`tools/real_ft_run.py --score-checkpoint` accepts a model the Rust trainer exported. A scoring
input only. No gate, threshold, seed rule or verdict logic changed (rule 2).

## Measured

No ledger row. Nothing ran on a GPU. Every number below is a pytest or diff result on this Mac's CPU.

- **Fail-first.** `python/tests/test_real_ft_metal_export.py` run against unmodified 6a06bd3 (worktree
  clean, HEAD `6a06bd3`): **49 failed, 2 passed**. The 2 that pass are characterizations of refusals
  that did not change (a Metal export among ensemble towers; a `.json` named `-metal`). At 6a06bd3
  every Metal export took the average route and was refused by `ckpt_average.read_manifest` ("not a
  manifest an average can be scored from") or by argv ("--ft-row-id would name one of them").
- **After the change.** That file plus `test_real_ft_score_checkpoint.py`: 91 passed.
- **The torch path is byte-identical.** This was checked by a throwaway differential that was not
  committed. It loaded the 6a06bd3 `real_ft_run.py` and the changed one side by side and scored one
  checkpoint through each `main()`. The checkpoint was a tiny tower with ft row `recipe.device="cuda"`,
  named `epoch-seed0-cuda.json` and scored on CPU, at fp32 and at bf16. In both cases the two eval rows
  were byte-equal, and so were the 16 verdict lines and the `recipe_hash` / `protocol_hash`. The fields
  left out of the comparison differ by construction: `row_id`, `written_at`, `prev_row_hash`,
  `wall_clock_s`, and the `code_that_ran` digest of the entry point.
- **Regression.** I ran every `python/tests` file that names `real_ft_run` (50 files), plus
  `test_lint_gate.py` and `test_wire_gap_pins.py`: **1322 passed, 1 failed, 31 skipped**.
  - The failure is `test_lint_gate.py::test_ruff_is_installed_not_merely_declared`, which is
    environmental: a worktree has no `.venv`.
  - With the main checkout's `.venv` symlinked in for one run (the link was removed afterwards), the
    lint gate and gap pins ran **27 passed, 0 skipped**.
  - The 31 skips are data that is not on this host, env-gated benchmarks, and the lint gate tests
    that skip without a `.venv`.
- **Lint.** The locked ruff 0.16.8 (`/Users/bharath/Code/research/Lappi-decision/.venv/bin/ruff check
  --config pyproject.toml`) was run over the whole worktree: clean. `test_lint_gate.py` itself SKIPS in a
  worktree, because there is no `.venv/bin/ruff` there. That skip is not a pass. The direct ruff run is
  the check.

## Changed (commit `2bc29d9`, on branch `worktree-agent-a04efcb2bc598bc8c` from `6a06bd3`)

- `tools/real_ft_run.py`
  - `_is_average` now excludes `*-metal.safetensors`.
  - New `_is_metal_export`, `_metal_export_seed`, `_read_metal_manifest`, `_metal_row_problems` and
    `_metal_export_weights`. `_checkpoint_step` routes to them.
  - `ScoredModel` gains `trained_by` and `ft_quick_reason`. `recipe_block()` emits `trained_by`.
    `model_reasons` adds `provenance_reasons()`.
  - `SCORED_CHECKPOINT_KEYS` gains `trained_by`. The score row's notes have a Metal branch.
  - The `--score-checkpoint` help text is updated.
- `python/tests/test_real_ft_metal_export.py` (new, 51 tests). `python/tests/test_real_ft_score_checkpoint.py`:
  the registry pin gains `"trained_by"`, with a comment. This is the only edit to an existing test.

## The contract the scorer now enforces

L-oracle's `ft-row-contract.md` was not on main, so this contract was derived from the scorer's own code.

**File name:** `epoch-seed<N>-metal.safetensors`. The manifest sits at `epoch-seed<N>-metal.safetensors.manifest.json`.

**Safetensors:** only `tower.<k>` and `span_head.<k>`. Each `<k>` is exactly the torch `state_dict` key
of `QwenDecisionStep`'s tower model and span head, because `load_state_dict(strict=True)` rejects any
other key. Dtypes are any `TensorRef` dtype: bf16 tower and f32 head are expected.

**Manifest** (JSON object; extra keys are allowed):

| Key | Required value |
| --- | --- |
| `from` | `"qd-train-export"` |
| `trainer` | `"qd-train-metal"` |
| `device` | `"metal"` |
| `seed` | int, equal to N |
| `optimizer_step` | int, equal to the ft row's `train.optimizer_steps` |
| `ft_row_id` | the ft row's full `row_id` |
| `vocab_size` | int |
| `span_weight` | number, equal to `recipe.span_weight` |
| `safetensors_sha256` | lowercase hex sha256 of the file bytes |
| `n_tensors` | int |

**ft row** (`--ft-row-id`): `run_kind="ft"`, `status="completed"`, `protocol.seed` equal to N,
`quick` an explicit bool (with a non-empty `quick_reason` when true), and these metrics:
`train.optimizer_steps.value` (int) and `train.termination.value` (str). The recipe needs:

| Recipe key | Required value |
| --- | --- |
| `tag` | `"epoch"` |
| `device` | `"metal"` |
| `trainer` | `"qd-train-metal"` |
| `shard_hash` | this run's train shard set |
| `backbone_snapshot` | `--real-backbone`'s directory name |
| `attn_implementation` | named by the row (see below) |
| `lr` | named by the row |
| `span_weight` | named by the row |
| `optimizer_recipe` | named by the row |

**Ordering constraint for L-trainer.** The manifest names the row's `row_id`. Python's `RunRecorder`
mints the id only when it writes the row (`ledger.py:1942`). The Rust writer must therefore mint the
uuid up front, or write the manifest after the row.

**Awkward item.** A Rust row must name `attn_implementation`, which is a torch kernel name (e.g.
`"sdpa"`), because `_scoring_step` builds the torch scoring tower from the ft recipe. It is required,
not defaulted.

**Every row scored from a Metal export** carries
`recipe.trained_by = {trainer, device, source, manifest_sha256}` and
`scored_checkpoint = "<name>:<safetensors_sha256>"`. Its single-seed metric is `ft_run_row_id`. When
its ft row is quick, the row's `quick_reason` adds "ft row <id>, which qd-train-metal trained on metal,
is quick (<reason>) ...".

## Open

- `GAP-LSCORER-FT-ROW-CONTRACT-UNRECONCILED-2026-10-01`: the contract above has not been reconciled
  with L-oracle's document or L-trainer's `ledger.rs` / export.
- `GAP-LSCORER-GITPULSE-UNTRUSTED-2026-10-01`, `GAP-LSCORER-NO-LISTAGENTS-2026-10-01` and
  `GAP-LSCORER-DEVMAP-WORKTREE-UNINDEXED-2026-10-01`.
- Seen and not changed, because existing rows must stay byte-identical: a torch single-seed
  `--score-checkpoint` row does not inherit `quick` from its ft row. `model_reasons` reads only the
  termination, the scoring device and the corpus. Whether `Ledger.promotion_verdict` catches such a
  row is **unverified**.
- A Metal export among an ensemble's towers is refused with the average's message ("an average
  (.safetensors) is scored on its own"). The refusal is loud but misleading; it was left as it was.

## First command for the next lane (L-trainer, once `crates/qd-train` writes an export and a row)

```
PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/real_ft_run.py --out <corpus> --rev <sha> \
  --devices mps --seeds 0 --score-val --real-backbone <Qwen3.5-2B-Base snapshot> \
  --score-checkpoint <dir>/epoch-seed0-metal.safetensors --ft-ledger ledger/mac-ojas-rung-d-<date>.jsonl \
  --ft-row-id <row id> --ledger ledger/mac-ojas-rung-d-<date>.jsonl --tokenizer-json <tokenizer.json>
```

Before that, run the contract as a test:
`uv run ... python -m pytest -q python/tests/test_real_ft_metal_export.py`.
