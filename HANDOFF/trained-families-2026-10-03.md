# HANDOFF: L-trained-families, a release admits only the task families it trained, 2026-10-03

Lane: worktree `.claude/worktrees/agent-ab37c1d2b676acdc9`, branch
`worktree-agent-ab37c1d2b676acdc9`. The worktree was created at `c65d7da`, 585 commits behind
main; it was fast-forwarded to main `82fcf72` before any change.

The ruling is Fable's, `AUDIT/finalize-2026-10-03/fable-pipeline-ruling.md` item 4 (that file was
untracked in the main checkout when this lane read it):

> Refuse untrained task ids: yes. Release manifest gains `trained_families` (export writes it
> from the train manifest's family set); `admission::admit` refuses a task absent from it with a
> typed refusal mirroring `RegisteredHeadMissing { task, available }` [...] Absent field ->
> refuse.

It closes GAP-RUNTIME-ADMITS-TASKS-NO-RELEASE-FAMILY-TRAINS-2026-10-03 on this branch, once merged.

No GPU and no box were used, nothing was downloaded, and no ledger row was written: nothing was
trained or scored. Every cargo and pytest run went through `tools/mac_heavy.sh`.

## Task id and family id are the same string

- [V] `python/qd_data/mixture.py:258`: training builds every request with `task=family_id`.
- [V] On `v5-build`, `python/tests/test_schema_api_doc.py` requires the documented request's
  `task == DEFECT_FAMILY_ID` (`code.defect_class`).

No mapping was invented. The documented example on main still sends `devcouncil.verdict`. That
is not a trained family, so once this merges, every release refuses it as `task_not_trained`.
`v5-build`'s `0810487` already fixes the example.

## What changed

Three commits on `worktree-agent-ab37c1d2b676acdc9`, on top of main `82fcf72`:

| Commit | What |
| --- | --- |
| `ce24e56` | Tests only, committed first. They fail against `82fcf72`: 12 fail and 1 control passes (see below). |
| `0f491bc` | The implementation, the fixtures, the wire corpus and the doc row. |
| this handoff's commit | This file, and four `gaps.jsonl` records (appended with `O_APPEND` and `fsync`). |

`82fcf72..0f491bc`: 28 files changed, 1287 insertions and 32 deletions. 733 of the insertions
are in test files and fixtures (`git diff --numstat`, paths matching `tests/`, `test_` or
`fixtures`).

### The release reader (`crates/qd-runtime/src/release.rs`)

- New constants: `TRAINED_FAMILIES_KEY = "trained_families"` (one spelling for the writer and
  the reader) and `MAX_TRAINED_FAMILIES = 1024`.
- `validate_trained_families` holds the rules the writer and the reader share. A recorded list
  must be:
  - non-empty;
  - at most 1024 ids;
  - made of ids that are non-blank, at most `RenderCaps::DEFAULT.max_task_bytes` (256) bytes,
    and without surrounding whitespace;
  - strictly ascending, which means sorted and unique.
- `Tower`, `Release` and `Ensemble` gain `trained_families() -> Option<&[String]>`. An absent
  field or `null` gives `None`, and the release still opens. A malformed list refuses the
  release at open with `ReleaseRefusalKind::Manifest`. So an empty `available` in a refusal
  always means "not recorded".
- `Ensemble::open` refuses members whose lists differ as `MembersDisagree`. One member recording
  a list and another recording none also counts as a disagreement.

### The refusal and admission (`refusal.rs`, `admission.rs`)

- `Refusal::TaskNotTrained { task, available }`, kind `task_not_trained`. It has the
  `RegisteredHeadMissing` shape; no other new shape was added.
- `admission::TrainedFamilies` has three states, never an `Option`:
  - `Recorded(list)`;
  - `Unrecorded`: the release admits no task;
  - `NoRelease`: the check does not run.
- `admit(request, &trained)` checks the task first, before the `code.defect_class` context
  shape. `admit_task` holds the task check.

### The runtime (`runtime.rs`)

- `from_release` and `from_ensemble` take the gate from the manifest.
- `with_backend` and `build` hold `NoRelease`.
- `Runtime::trained_families()` reports which of the three states the runtime holds.

### The exporter (`crates/qd-export`)

- `--train-manifest PATH` is **required** (`ExportRequest.train_manifest`).
- The exporter reads the train split's data manifest (`Manifest.to_json()` shape) and refuses
  it (`RefusalKind::TrainManifest`) unless:
  - it is format 1 and declares `split: "train"`;
  - it carries a 64-hex `data_snapshot_hash` and `held_out_families`;
  - `n_rows` equals its entry count;
  - every entry is a train row with a non-blank `family_id`, outside the held-out families;
  - its family set passes `validate_trained_families`.
- The release manifest gains:
  - `trained_families`: sorted and unique;
  - a `train_manifest` block: path, sha256, split, `n_rows`, `rows_by_family`, and
    `data_snapshot_hash_as_recorded`. That hash is not re-derived: the canonical-JSON port is in
    qd-train, which depends on qd-export.
- `ExportSummary.trained_families`, and a `trained_families:` line on stdout.
- The ensemble writer refuses members whose families differ, as the reader does.

### The product path and docs

- `qd-metal-serve` (`crates/qd-metal/src/bin/serve.rs`) prints the admitted families at
  startup. For a release that records none, it says that every request will be refused.
- `docs/schema-api.md` gains one row in the refusal table.

### Fixtures that changed

- `crates/qd-export/tests/common/mod.rs`:
  - `Fixture.train_manifest` and `write_train_manifest`;
  - `TRAINED_FAMILIES = ["code.defect_class", "devcouncil.verdict"]`. `devcouncil.verdict` is
    the task the existing serving tests ask.
- `crates/qd-export/tests/roundtrip.rs`: the ignored real-size test passes a train manifest.
- `crates/qd-metal/tests/serve_cpu.rs`:
  - the synthetic manifest records `["devcouncil.verdict"]`;
  - `release_dir_with(weight_hash, edit)` has the same name and shape as `v5-build`'s helper,
    so one helper survives the merge.
- `crates/qd-runtime/tests/refusal_is_not_noul.rs`: the exhaustive arm, and the kind count goes
  from 39 to 40.
- `crates/qd-runtime/src/fixtures.rs::all_refusals`: one entry.
- `fixtures/wire/` regenerated with `QD_WRITE_FIXTURES=1`:
  - one new file, `refusal-task_not_trained.json`;
  - `index.json`: count 60 -> 61, refused 40 -> 41;
  - no other fixture moved.
- `crates/qd-runtime/tests/wire_fixtures.rs`: the pinned refusal-kind count goes from 39 to 40.
- `python/qd_wire/contract.py`: the `task_not_trained` entry.
- `python/tests/test_qd_data_wire_agreement.py`: `task_not_trained` added to
  `NO_QD_DATA_COUNTERPART`, with the reason.
- `python/tests/test_export_oracle.py`: writes a train manifest, passes `--train-manifest`, and
  asserts `trained_families`.

## What was measured

**No ledger row was written:** nothing was trained or scored. Every run below went through
`tools/mac_heavy.sh` with cargo at `-j 2`. The logs are in the lane worktree, under
`build/trained-families/` (gitignored, so not committed).

### Fail-first

`failfirst.log`, also kept as `failfirst.keep.log`. The new tests ran against unmodified
sources. The log's `git diff --stat HEAD -- crates/*/src` is empty.

| Target | Passed | Failed | The failures |
| --- | --- | --- | --- |
| `qd-runtime --test trained_families` | 1 | 4 | an untrained task answered `ok`; an unrecorded release answered; the context was refused before the task; a malformed list opened |
| `qd-export --test trained_families` | 0 | 3 | `--train-manifest` unknown to clap; an export without it exited 0 |
| `qd-export --test release_binding` | 12 | 1 | the exported release answered an untrained task |
| `qd-export --test ensemble` | 12 | 3 | the writer and the reader accepted members with different families; the ensemble answered an untrained task |
| `qd-metal --test serve_cpu` | 5 | 1 | the product path answered an untrained task |

- 12 new tests failed, each for the reason it names.
- The one new test that passed is the control
  `a_task_the_release_trained_is_admitted_and_answered`, which characterizes behaviour the fix
  keeps.
- All 29 existing tests in these targets passed.

### Post-fix

`postfix.log`, the final run at 16:32Z, on the tree committed as `0f491bc`. `postfix.run1.log`
is the first run; one pinned count in `wire_fixtures.rs` was still 39 there.

| Command | Result |
| --- | --- |
| `QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures` | 6 passed |
| `cargo test -p qd-runtime` (all targets) | **343 passed, 0 failed**. `tests/trained_families.rs` 6, `tests/lifecycle.rs` 15 (unchanged file) |
| `cargo test -p qd-export` (all targets) | **56 passed, 0 failed, 1 ignored**. `trained_families` 3, `release_binding` 13, `ensemble` 15 |
| `cargo test -p qd-metal --test serve_cpu` | **6 passed** |
| `cargo clippy -p qd-runtime -p qd-export --all-targets -- -D warnings` | clean |
| `cargo clippy -p qd-metal --test serve_cpu -- -D warnings`; `--bin qd-metal-serve` (`extra.log`) | clean |
| pytest, project venv, single process: `test_wire_contract_matches_rust`, `test_qd_data_wire_agreement`, `test_wire_golden_corpus`, `test_wire_refusal_is_not_noul`, `test_wire_answer_parser`, `test_wire_values`, `test_wire_gap_pins`, `test_schema_mirror` (`pytest.log`) | **316 passed, 1 skipped**. The skip is `test_real_ft_composed_slice.py:27`, because torch is absent from this venv |
| pytest `test_export_oracle.py`, ml venv, `uv run --offline`, `QD_EXPORT_BIN` = this branch's binary | **4 passed** |

The ignored test is `roundtrip::snapshot_the_base_weights_export_at_248320_and_load_through_qd_metal`.
It needs the 4.5 GB snapshot and about 9 GB of temp disk.

### Not run

These are not passes:

- That ignored real-size round trip. Its fixture change compiled; the test itself did not run.
- `cargo test --workspace`. The lane ran only the crates it touched, as told.
- The full pytest suite.
- The Metal backend serving a real release. It needs the GPU, and no v5/v4 release with the
  field exists yet.
- The v4 re-export on the box.

## What is open

1. **GAP-RUNTIME-WITH-BACKEND-RUNS-NO-TRAINED-FAMILY-CHECK-2026-10-03.** `Runtime::with_backend`
   and `Runtime::build` (the reference backend) do not run the task check. They hold
   `TrainedFamilies::NoRelease`, and `Runtime::trained_families()` says so.
   - [V] The release path is gated: `qd-metal-serve` goes through `Release::open` and then
     `from_release` (`crates/qd-metal/src/serve.rs:63`). So is the ensemble path.
   - [V] Closing the seam needs `crates/qd-runtime/tests/common/mod.rs` (`:88`, `:172`) and
     `tests/lifecycle.rs` (`:87`, `:214`, `:226`). Both build runtimes through `with_backend` with
     `is_model = true` and ask `devcouncil.verdict`. `Runtime::build` is called from
     `service.rs:97`.
   - All three files carry another owner's uncommitted edits in the main checkout, and this lane
     was told not to touch them.
   - [V] `tests/trained_families.rs::the_release_and_the_runtime_report_what_a_task_is_checked_against`
     characterizes the gap. It asserts that the seam reports `NoRelease` and still answers.
     Closing the gap changes that line on purpose.
2. **The exporter flag is required, which breaks one live queue script once the box binary is
   rebuilt.** [V] `campaign/post-f-queue/box_q_j7p_cpu.sh:59` runs `qd-export` without
   `--train-manifest`, and so does the command in `HANDOFF/j7-export-2026-10-01.md`.
   - A `qd-export` built from this branch exits 2 on them; the clap error names the flag.
   - The script was not edited, because it is a queued campaign script. The box's
     `/home/ubuntu/bin/qd-export` is unchanged until someone rebuilds it.
   - Every future export command needs `--train-manifest <the train split's manifest>`.
3. **v4 releases refuse every task until they are re-exported.**
   - [V] The bench release `p4-v4-avg-masters-v1/release_manifest.json` has no
     `trained_families`.
   - [V] It also has no `expected_identity.config_sha256` and has `calibration: null`, so main's
     `Release::open` already refuses it, on its config binding.
   - The re-export is a box job. The paths come from that manifest's `source` and
     `base_snapshot` blocks. [I] The train manifest's box path is inferred: it is v4's
     `data/pool/train.json`, 152,784,427 bytes, sha256 `7feb6ee6…`
     (`HANDOFF/longctx-compose-2026-10-01.md:319`). Run it with a `qd-export` built from this
     branch:

     ```text
     qd-export --source /home/ubuntu/ckpt/p4-v4/avg/epoch-avg-seed01234-masters.safetensors \
       --base-snapshot /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
       --tokenizer-sha256 fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927 \
       --expect-vocab-size 248320 --train-manifest <box checkout>/data/pool/train.json \
       --calibration <the table the bench serves> --out <a new release directory>
     ```

   - [U] v4's family set was not measured. `data/pool/train.json` is not on the Mac. The
     export prints it (`trained_families: ...`) and records `rows_by_family`.
   - The train manifest is parsed whole, so it costs about its size several times over in
     memory. That is fine on the box; on the Mac it would go through `mac_heavy.sh`.
4. **This branch conflicts with `v5-build`.** [V] `git merge-tree` of `0f491bc` and `v5-build`
   conflicts in four files:
   - `crates/qd-runtime/src/release.rs`, `crates/qd-export/src/export.rs` and
     `crates/qd-metal/tests/serve_cpu.rs`. Both sides add fields to the same structs and blocks:
     `prompt_format` there, `trained_families` here. Keep both.
   - `gaps.jsonl`. This conflict already exists between main and `v5-build`.

   In `serve_cpu.rs`, both sides add the same `release_dir_with(weight_hash, edit)` helper. Keep
   one, with `v5-build`'s `qd-release.v2` and `prompt_format: 2` and this branch's
   `trained_families` line.

   After the merge:
   - `crates/qd-runtime/tests/trained_families.rs`'s synthetic manifest needs
     `expected_identity.prompt_format = render::PROMPT_FORMAT`. It already uses
     `MANIFEST_FORMAT`.
   - Rerun `QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures` if
     `the_corpus_on_disk_is_what_this_build_emits` fails. `index.json` merges textually, but
     both sides changed it.
   - [V] Under `v5-build`'s runtime, every v4 release is already refused by its format, so
     item 3 matters only for a v4-era (main) runtime.
5. **What the service reports.** `qd serve` status and ping do not report the trained families,
   because `service.rs` is held. `qd-metal-serve` prints them at startup.
6. **Navigation gaps:**
   - GAP-TRAINED-FAMILIES-LANE-DEVMAP-UNAVAILABLE-2026-10-03. There is no store in the worktree
     and main's store is malformed. Every call site above came from rg.
   - GAP-TRAINED-FAMILIES-LANE-LISTAGENTS-UNAVAILABLE-2026-10-03. `ListAgents` could not be
     loaded in this subagent session. `gitpulse_insights` showed this worktree clean and
     uncontended.

GAP-RUNTIME-ADMITS-TASKS-NO-RELEASE-FAMILY-TRAINS-2026-10-03 is recorded `fixed-on-branch`. Mark
it resolved when the branch merges.

## First command for the lead

From the main checkout. The branch touches none of the dirty paths in main's checkout as the lane
brief listed them. Main has moved two AUDIT-only commits (`c96290e`, `bf95f56`) since `82fcf72`;
`git merge-tree` of the branch with main is clean.

```text
git merge --no-ff worktree-agent-ab37c1d2b676acdc9
```

Then verify on the merged tree:

```text
bash tools/mac_heavy.sh tf-merge cargo test -p qd-runtime -p qd-export
bash tools/mac_heavy.sh tf-merge-serve cargo test -p qd-metal --test serve_cpu
```
