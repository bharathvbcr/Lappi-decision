# Ledger schema (S6)

The ledger is the decision record. **The decision reads the ledger, not the console.**

Two rules from the plan drive every design choice here:

- *"Every run writes to the protocol-hashed ledger before it is allowed to finish. A run whose row is
  missing is rerun, not remembered."*
- *"A run cannot exit 0 until its row is written. A deliberately broken run must produce a row that
  says so."*

Storage: `ledger/runs.jsonl`, append-only, one JSON object per line, UTF-8, newline-terminated.
Append-only is enforced by the writer (open in `O_APPEND`) and by a test that rewriting a byte of an
existing line is detected via the chain hash.

## The tri-state rule (this is the load-bearing invariant)

Every metric, gate and control is a **tri-state**, never a bare number or a bare boolean:

```json
{ "state": "ran", "passed": true, "value": 0.83, "n": 300 }
{ "state": "ran", "passed": false, "value": 0.41, "n": 300 }
{ "state": "not_run", "reason": "no GPU in this environment: MPS only, suite needs CUDA" }
```

`state` is `ran` | `not_run`. There is **no third option and no default**. A consumer that reads a
tri-state must branch on `state` before touching `passed` or `value`; the deserializer refuses a row
where `state == "ran"` and `passed` is absent, and refuses a row where `state == "not_run"` and
`reason` is empty.

This exists because *a check that could not run must never report the same result as a check that ran
and passed*. The common failure is an `Option<bool>` that a report renders as "no failures found".
Here, `not_run` has no `passed` field at all, so a report that prints it as passing has to fabricate
one — which the schema test catches.

**Aggregation rule.** A gate over several tri-states is `ran/passed` only if *every* input is
`ran/passed`. If any input is `not_run`, the aggregate is `not_run` and carries the union of reasons.
An aggregate is never allowed to be more confident than its least-informed input.

**Coverage rule.** Any metric computed over a sample carries both `n` (examined) and `n_total`
(eligible). A report renders `n/n_total`. Presenting a capped sample as complete coverage is a bug,
not a rounding choice.

## Row

| Field | Type | Notes |
| --- | --- | --- |
| `row_id` | string | ULID. Unique; the writer refuses a duplicate |
| `written_at` | RFC3339 string | UTC |
| `prev_row_hash` | string \| null | SHA-256 of the previous line's bytes; `null` on the first row |
| `protocol_hash` | string | SHA-256 over the five components below, canonically serialized |
| `protocol` | object | `{data_snapshot_hash, tokenizer_hash, backbone_commit, recipe_hash, seed}` |
| `run_kind` | enum | `teacher` \| `lr_probe` \| `cpt` \| `prune_heal` \| `ft` \| `ablation` \| `eval` \| `calibration` \| `smoke` \| `throughput` \| `resume` \| `scale` \| `build` (see below) |
| `status` | enum | `completed` \| `killed` \| `failed`. A killed run still writes a row |
| `quick` | bool | true if <3 seeds, truncated schedule, or subsampled. **A `quick` row cannot promote anything** |
| `quick_reason` | string \| null | Required non-empty when `quick` is true |
| `code_commit` | string | git HEAD of this repo at launch; `-dirty` suffix if the tree was dirty |
| `env` | object | `{torch, transformers_sha, fla_present, causal_conv1d_present, device, host}` |
| `metrics` | object | tri-states, keyed by name; see below |
| `noul_rate` | object | tri-state; in-domain abstain rate |
| `controls` | object | `{shuffled_label, privileged_hunk, degenerate_head, transfer_gate}` — all tri-states |
| `gates` | object | `{paired_margin_vs_linear, ood_abstain, needle_hunk_recall, permutation_consistency, ece}` — all tri-states |
| `wall_clock_s` | number | measured, not estimated |
| `cost_usd` | number | measured from instance rate x wall clock |
| `notes` | string | free text; never parsed by a gate |

### `metrics` keys

Broken out **by k and by language**, because the plan's calibration gate is per-k and per-language and
an aggregate hides exactly the failure it is meant to catch:

- `accuracy.k{2,3,4,8,16}` — one tri-state each
- `accuracy.lang.{rust,go,swift,python,typescript}`
- `brier.k{...}`, `ece.k{...}`, `ece.lang.{...}`
- `span_iou`, `span_exact`
- `ordinal_mae`, `ordinal_spearman`

A key that was not computed is present with `state: "not_run"` and a reason, or absent entirely.
It is **never** present with a zero value.

## `fla_present` is not cosmetic

`env.fla_present` records whether `flash-linear-attention` was actually active, established by
grepping the startup log for the fallback warning (S1). Without it the GDN layers run an fp32 torch
chunk loop ~43% slower. A throughput or cost number recorded with `fla_present: false` is not
comparable to one recorded with `true`, and the comparison code refuses to pair them.

`fla_present` is itself a tri-state in the row (`ran/passed` = confirmed present; `ran/failed` =
confirmed fallback; `not_run` = the dry run did not execute, e.g. on the Mac where there is no CUDA).
An unchecked environment must not read as a clean one.

## `run_kind: "build"` — the row a build-and-test lane writes

Every other `run_kind` names a training or evaluation run, and their required protocol components —
`data_snapshot_hash`, `tokenizer_hash`, `backbone_commit` — have no meaning for a lane that compiles
the workspace and runs a test suite. Repo rule 5 says *a number in a report cites a ledger row or is
not in the report*, so before this kind existed a build lane had no honest way to comply: three lanes
in a row printed command output instead and flagged it (`GAP-MUTATE-NO-LEDGER-ROW-KIND`). Inventing a
value for the missing components would have been worse than the gap — a fabricated entry in the
decision record — so the kind carries a marker instead.

**Protocol.** `Protocol.for_build(commands=…, toolchain=…)` builds it:

| Component | Value |
| --- | --- |
| `data_snapshot_hash` | `"n/a:build"` |
| `tokenizer_hash` | `"n/a:build"` |
| `backbone_commit` | `"n/a:build"` |
| `recipe_hash` | **real**: SHA-256 over the canonical `{commands, toolchain}` — two build rows are comparable exactly when they ran the same commands on the same toolchain |
| `seed` | `0` unless the lane has a reason |

`"n/a:build"` is not a hash and not empty, so a reader, a diff and a test can all tell it from a real
value. The marker is checked **in both directions**: a `build` row missing it is refused, and a
non-`build` row carrying it is refused. A training row with an unidentified protocol component makes
every comparison against it meaningless, and a build row claiming a snapshot it never had looks
comparable to training rows it has nothing to do with.

**Metrics.** Suite results go in `metrics` under `suite.<name>`, as ordinary tri-states — there is no
second mechanism:

```json
{ "suite.cargo_test_workspace": { "state": "ran", "passed": true, "value": 344, "n": 344, "n_total": 344 },
  "suite.gpu_throughput":       { "state": "not_run", "reason": "no CUDA on this host: MPS only" } }
```

`value` is the passing count, `n`/`n_total` the coverage pair — tests that ran over tests that were
collected, so a suite whose collection silently shrank is visible. A suite the environment cannot run
is `not_run` with the reason, never absent and never a zero.

**`wall_clock_s` is measured** as for any other row. `cost_usd` is `0.0` for a local build; it is a
measured quantity, and zero is the measurement, not a placeholder.

**A `build` row never promotes.** See below.

## Promotion

A result may promote a decision only if **all** hold:

1. `status == "completed"`
2. `quick == false`
3. Three rows exist sharing a `protocol_hash` that differs only in `seed`
4. Every gate in `gates` is `state: "ran"` — a `not_run` gate blocks promotion, it does not pass it
5. Every control in `controls` is `ran` and `passed`
6. `run_kind` is not in `NON_PROMOTING_RUN_KINDS` — currently `{build}`

Rule 4 is the one that is easy to get wrong and the reason the tri-state exists.

Rule 6 is stated on the run kind rather than left to rule 4 to catch. A build lane's gates are
*vacuous*, not satisfied: it has nothing to say about `paired_margin_vs_linear`, and the only reason
rule 4 would currently block it is that `RunRecorder._fill_unreported` marks unreported gates
`not_run`. That is a safety net, not an invariant — a build lane that set a gate for an unrelated
reason would promote. Measured: with rule 6 removed, three completed non-quick `build` rows with
every gate and control `ran`/`passed` return `promoted=True`
(`python/tests/test_ledger.py::test_a_build_row_cannot_promote_anything`).

## Chain integrity

`prev_row_hash` chains rows. `qd-ledger verify` recomputes the chain and reports the first break.
An edited or reordered history is detectable; it is not cryptographically prevented (a local
append-only research log, not a tamper-proof audit log — stated here so nobody claims more).
