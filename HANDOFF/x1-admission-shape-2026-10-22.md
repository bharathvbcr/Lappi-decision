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

## Not run (NOT RUN, not passed)
Both need the cargo lock, which was held or over load for 2+ hours.
- Regenerating `fixtures/wire/` (`QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures`), job `x1-fixtures`, still queued when this was written. Until it runs, `wire_fixtures` and `refusal_is_not_noul` fail (the corpus lacks the new kind; the pinned counts are already 41). The diff must be shown to be only the new refusal file plus its index.json entry. If the job completes after this commit it writes into the worktree uncommitted.
- Full `cargo test -p qd-runtime` (oneshot/serve/lifecycle/golden request fixtures), `cargo test -p qd-export` (including the calibration gate, to confirm it agrees with `CalibrationTable::lookup`), and pytest `python/tests/test_wire_contract_matches_rust.py python/tests/test_qd_data_wire_agreement.py` (via `make torch-pytest`, narrowed). Unverified: that `python/qd_wire/contract.py` (hand-added row, type `Vec<Vec<String>>`) matches the extractor.

## Gaps
- GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17: resolved-with-residual (legacy releases unchecked).
- GAP-QD-METAL-SERVE-STARTUP-OMITS-SLOT-SHAPE-CHECK-STATE-2026-10-22: open (qd-metal not edited).

## First command for the next lane
`QD_WRITE_FIXTURES=1 MAC_HEAVY_MAX_LOAD=48 MAC_HEAVY_MAX_WAIT_S=28800 bash tools/mac_heavy.sh x1-fixtures cargo test -p qd-runtime --test wire_fixtures`
then `git diff --stat fixtures/wire` (expect only refusal-slot_shape_not_trained.json added and index.json changed), then the full suites above.
