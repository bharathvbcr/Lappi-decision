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

## Not run
- Nothing in the requested set. Not covered: an ensemble-level test that members disagreeing on `trained_slot_shapes` alone (same families) are refused `MembersDisagree` (the rule is written in `Ensemble::open`, release.rs, but no test drives it); qd-metal's own tests (`serve_cpu`, manifests with trained_families only, so the shape check is not run there) were not run; python outside the two files was not run.

## Gaps
- GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17: resolved-with-residual (legacy releases unchecked).
- GAP-QD-METAL-SERVE-STARTUP-OMITS-SLOT-SHAPE-CHECK-STATE-2026-10-22: open (qd-metal not edited).

## First command for the next lane
`QD_WRITE_FIXTURES=1 MAC_HEAVY_MAX_LOAD=48 MAC_HEAVY_MAX_WAIT_S=28800 bash tools/mac_heavy.sh x1-fixtures cargo test -p qd-runtime --test wire_fixtures`
then `git diff --stat fixtures/wire` (expect only refusal-slot_shape_not_trained.json added and index.json changed), then the full suites above.
