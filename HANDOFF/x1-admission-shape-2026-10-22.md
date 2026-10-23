# HANDOFF x1-admission-shape, 2026-10-22 (PARTIAL: see "Not run")

Lane X1, branch `lane/x1-admission-shape` (from main a012dac), worktree
`/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-a398b3073bed54ce4`.
Not merged, not pushed. Gap: GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17.

## Commits
- 1cb9cfa AUDIT enumeration (written before any refusal)
- 3c987ae fails-before tests
- 21736cc runtime change, docs/caller-contract.md, docs/schema-api.md, python wire contract rows
- be91658 gaps.jsonl (two records)
- this HANDOFF

## Enumeration summary
`AUDIT/x1-admission-shape-2026-10-22/enumeration.md`. Seven families are pinned by the caller
contract (`code.defect_class` [choice:5, span]; `gitpulse.commit_type` [choice:12]; `devtype.route`
[choice:5]; four `career.jd_*` [choice:6]/[choice:5]). Five shape classes are newly refused: wrong
row count, wrong slot kind, missing slot, extra slot, another family's shape. Callers read, not
edited: DevCouncil `lappi/request.go:57-62`, GitPulse `lappi/request.rs:55-64`, DevType
`LappiRouteRequest.swift:10-32`: each sends exactly its contract shape, none is refused. Vani, Jarvis,
Manvi: no socket client found (rg, default ignores, not -uu). DevMap answered `admit` but reported
walk_incomplete, so call-site lists are rg.

## Contract and security impact
Admission only narrows: a new refusal `slot_shape_not_trained` (task, shape, trained,
contract_section) before the model is asked. It never emits PASS/allow (rule 7), widens no access,
touches no auth, secret or quota path. Contract changes: one new refusal kind on the wire (documented
in schema-api.md and caller-contract.md section 4; callers already treat any refusal as
`request_refused`); one new `status` op field `slot_shape_check` (`checked` | `not_recorded` |
`no_release` | null while cold). Answers and refusals gain no other field. A release with no
`trained_slot_shapes` (every export before 2026-10-17) is admitted on the task check alone (lead's
ruling: refusing them would refuse every live caller); that state is visible through `status`.
A release that records shapes must now name exactly its `trained_families`, with canonical sorted
slot keys, or it is refused at open; ensemble members must agree on the set.

## Runs (command, exit, counts)
- Before (unmodified code): `MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh x1-before cargo test -p qd-runtime --test admission_shapes` exit 101, 8 failed, 3 passed (the controls).
- After: same command, label x1-after, exit 0, 14 passed, 0 failed (19:08:48Z).

## Later runs (all under tools/mac_heavy.sh, MAC_HEAVY_MAX_LOAD=48)
- Wire fixtures: `QD_WRITE_FIXTURES=1 ... x1-fixtures cargo test -p qd-runtime --test wire_fixtures` exit 0, 6 passed (2026-10-22T23:09Z). Diff was only the new kind: `refusal-slot_shape_not_trained.json` added, `index.json` count 61->62, refused 41->42, one entry. Committed 5f6f38f.
- Full `cargo test -p qd-runtime` (x1-rt-full) exit 0 at 23:36Z: 32 binaries, 481 passed, 0 failed, 1 ignored (includes wire_fixtures, refusal_is_not_noul, lifecycle, admission, trained_families, new admission_shapes).
- `cargo test -p qd-export` (x1-export-full) exit 101 on this branch before the fixture fixes: `ensemble` 15 passed 2 failed (`the_reader_refuses_members_trained_on_different_families`, `a_runtime_built_from_the_ensemble_refuses_a_task_its_members_did_not_train`). `--no-fail-fast` (x1-export-nff) exit 101: also `release_binding` 14 passed 2 failed (`a_runtime_built_from_the_release_refuses_a_task_the_release_did_not_train`, `a_runtime_built_from_the_release_calibrates_with_the_release_table_and_pins_it`); every other binary passed.
  Causes (fixture fallout, not runtime bugs): the three families-editing fixtures (`set_trained_families` in ensemble.rs, the `rewrite_manifest` closure in release_binding.rs) changed `trained_families` only, leaving `trained_slot_shapes` naming other families, which the reader now refuses at open; and `four_option_request` asked `devcouncil.verdict` in `[choice:5]` while the exporter recorded `[choice:5, span]` for the release, a real shape refusal. Fixes (lead-granted for ensemble.rs; release_binding.rs extended, reported): helpers rewrite shapes coherently (kept shapes for recorded families, the fixture train manifest's shape for an added family, never invented; shapes removed with the families), and the request gains a span slot. Assertions unchanged; the families-versus-shapes check at open untouched. Committed 4e885e4.
- After the fixes: `cargo test -p qd-export --no-fail-fast` (x1-export-fix) exit 0 at 2026-10-23T03:22Z: 18 binaries, 145 passed, 0 failed, 2 ignored, including the calibration-gate binary (`calibration_gate`), so the export gate still agrees with the runtime's lookup.
- `make torch-pytest TORCH_PYTEST_RUN=<uv run ... pytest python/tests/test_wire_contract_matches_rust.py python/tests/test_qd_data_wire_agreement.py -o addopts=>` (x1-pytest) exit 0 at 05:54Z: 71 passed. The hand-added `python/qd_wire/contract.py` row matches the extractor.

## Adversarial pass (2026-10-23), commits 225c978 (cherry-pick of adv/x1 726de8a), a24dd22, and the follow-up
Predicted breaks fixed in `crates/qd-runtime/src/release.rs` (a24dd22):
1. fail-open: `trained_slot_shapes: null` was read as the legacy unchecked release. Only an ABSENT key is legacy now; `null` is refused at open as a non-object.
2. a repeated object key anywhere in a manifest was last-wins; `read_manifest` now refuses it via `first_duplicate_json_key` (pub; qd-export's safetensors reader now calls it and its own copy is deleted).
3. cross-crate: qd-export's gate accepted a table entry `choice:0` plus shape `choice:0`, exporting a release the runtime refuses at open. Fails-before: `x1-gate-before` (`cargo test -p qd-export --test calibration_gate a_shape_no_request_can_produce`) exit 101, export had succeeded (exit 0). Fix: the gate's `runtime_lookup` also applies `qd_runtime::release::is_slot_key`; `is_shape_key` left alone (excluded families keep unlookable shapes, `w2b_adversarial` pins that). `choice:05` was already stopped by the gate (`calibration_gate.rs` `a_shape_spelled_so_the_runtimes_lookup_misses_it_is_refused`).

Fails-before that RAN for the two runtime fixes (x1-adv, fixes reverted in the working tree only, never committed; restored blob 84ffd4bd matched; `git status` afterwards showed only gaps.jsonl): `cargo test -p qd-runtime --test adv_x1_shapes` exit 101, 4 passed 4 failed, including `a_null_trained_slot_shapes_is_not_silently_the_legacy_unchecked_release` and `a_duplicate_family_key_in_trained_slot_shapes_is_refused_not_last_wins`. The other two failures in that run (`live_callers_...`, `an_option_count_...`) were a fixture bug in the adversarial file, not the fixes: its `contract_release` listed `trained_families` as [COMMIT, DEFECT, ROUTE], and the reader has always required strictly ascending order, so the release was refused at open. One line changed to [DEFECT, ROUTE, COMMIT]; no expectation changed.

After (x1-adv-after, `cargo test -p qd-runtime`) exit 0, 2026-10-23T09:47Z: 33 binaries, 489 passed, 0 failed, 1 ignored; `adv_x1_shapes` 8 passed. `cargo test -p qd-export --no-fail-fast` (x1-export-final) exit 0, 09:48Z: 18 binaries, 146 passed, 0 failed, 2 ignored (the new gate test included).

New gap: GAP-DUPLICATE-JSON-KEY-SCAN-COPY-IN-QD-PREP-2026-10-23 (open): `crates/qd-prep/src/allocate.rs:120` keeps a third copy; qd-prep is lane W2a's, untouched; convergence is a new workspace edge, the owner's call.

## Not run
- Nothing in the requested set. Not covered: an ensemble-level test that members disagreeing on `trained_slot_shapes` alone (same families) are refused `MembersDisagree` (the rule is written in `Ensemble::open`, release.rs, but no test drives it); qd-metal's own tests (`serve_cpu`, manifests with trained_families only, so the shape check is not run there) were not run; python outside the two files was not run.

## Gaps
- GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17: resolved-with-residual (legacy releases unchecked).
- GAP-QD-METAL-SERVE-STARTUP-OMITS-SLOT-SHAPE-CHECK-STATE-2026-10-22: open (qd-metal not edited).

## First command for the next lane
`QD_WRITE_FIXTURES=1 MAC_HEAVY_MAX_LOAD=48 MAC_HEAVY_MAX_WAIT_S=28800 bash tools/mac_heavy.sh x1-fixtures cargo test -p qd-runtime --test wire_fixtures`
then `git diff --stat fixtures/wire` (expect only refusal-slot_shape_not_trained.json added and index.json changed), then the full suites above.
