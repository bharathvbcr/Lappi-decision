# X1 enumeration: (family, slot shape) admission, written before any refusal lands

Lane X1, 2026-10-22, branch `lane/x1-admission-shape` from main a012dac. Gap:
GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17 (two records in gaps.jsonl:
line 1691 opened by L-release, line 1902 narrowed by W2b; both read).

Evidence labels: verified = read at the cited line this session; inferred = derived from verified
lines.

## 1. What admission checks today (verified)

`crates/qd-runtime/src/admission.rs:124-131` `admit`: `admit_task` (family against
`TrainedFamilies`) then, only for `code.defect_class`, `admit_defect_context`. No slot is read.
`Runtime::answer_inner` calls it (`crates/qd-runtime/src/runtime.rs:388`) after hash checks.
Calibration is looked up per slot by `(kind, rows)` (`calibration.rs:103-118`), so an admitted
family may send any shape the table has an entry for.

## 2. Shape spelling (verified)

A request's shape is its sorted slot keys: `choice:N` with N = options + 1 (reserved noul row),
`score:N` with N = bins + 1, `span` (`python/qd_data/schema.py:724-745`; qd-preflight
`request_shape`, `release_check.rs:248-264`; qd-export `calibration_gate.rs` `is_shape_key`).
A release records, per admitted family, the set of shapes its train manifest holds as
`trained_slot_shapes` (`export.rs:1869-1879`, bound by `slot_shape_digest`,
`export.rs:975-1006`). The runtime reads only `trained_families` (`release.rs:99`, `:326`); it
never reads `trained_slot_shapes`. That unread record is the gap.

## 3. Families and their allowed shape(s)

The allowed shape set of a family is whatever the release recorded for it; the runtime has no
static registry of families (the head registry `registry.rs` is keyed by task and slot name, and
is the registered route only). The shapes the callers' contract pins (docs/caller-contract.md
section 4), against which the callers below are checked:

| family | pinned slots | shape key set |
| --- | --- | --- |
| code.defect_class | defect_class choice x4, defect_span span | [choice:5, span] |
| gitpulse.commit_type | commit_type choice x11 | [choice:12] |
| devtype.route | route choice x4 | [choice:5] |
| career.jd_role_fit | role_fit choice x5 | [choice:6] |
| career.jd_sponsorship | sponsorship choice x4 | [choice:5] |
| career.jd_work_mode | work_mode choice x4 | [choice:5] |
| career.jd_seniority | seniority choice x5 | [choice:6] |

The in-repo fixtures family `devcouncil.verdict` (`crates/qd-runtime/src/fixtures.rs:151`) is
only ever served by `Runtime::with_backend` (TrainedFamilies::NoRelease).

## 4. Shapes that would be newly refused

With a release that records `trained_slot_shapes`, a request whose family is trained is refused
unless its shape equals one of the family's recorded shapes. Classes (each gets a fails-before
test):

1. wrong row count: same slot kinds, different option or bin count (`choice:5` recorded,
   `choice:6` sent).
2. wrong slot kind: `score:N` or `span` where the family trained `choice:N` (and the reverse).
3. missing slot: a recorded multi-slot shape with a slot absent (`[choice:5]` for
   `[choice:5, span]`).
4. extra slot: a slot the family never trained (`[choice:5, span, span]`,
   `[choice:5, choice:3]`).
5. a shape of the right multiset in another family's set (family A sends family B's shape).

Not refused (stated): slot names, option text and order, and the question are not part of the
key (they are in the label-set hash and the prompt, not in the calibration key). The key is what
the export gate and the calibration lookup already use.

Releases that record no `trained_slot_shapes` (every export before 7129209, 2026-10-17) cannot
say what shapes they trained. The HANDOFF states how that case is carried (reported as not run,
never as passed).

## 5. Callers (read-only; sibling repos not edited)

| caller | path:line | family | shape sent | refused? |
| --- | --- | --- | --- | --- |
| DevCouncil | `/Users/bharath/Code/devtools/DevCouncil/backend/go_orchestrator/devcouncil/lappi/request.go:57-62` (`DefectClassSlots`), built at `asker.go:161`, options `request.go:28` (4 options) | code.defect_class | choice(4) + span = [choice:5, span] | no |
| GitPulse | `/Users/bharath/Code/devtools/GitPulse/src-tauri/src/lappi/request.rs:55-64`; options `KNOWN_TYPES`, `src-tauri/src/ai/commit_brief.rs:47-49` (11) | gitpulse.commit_type | [choice:12] | no |
| DevType | `/Users/bharath/Code/apps/DevType/Sources/ExpanderEngine/Lappi/LappiRouteRequest.swift:10-32` (`options`, 4, line 14) | devtype.route | [choice:5] | no |
| Vani | `/Users/bharath/Code/apps/Vani/internal/decide/jdtriage/fixture.json` (four families, one choice slot each); rg for `lappi`/`qd.sock` under internal/decide and suggest finds definitions only, no socket client | career.jd_* | [choice:6], [choice:5], [choice:5], [choice:6] | no (and not sent) |
| Jarvis, Manvi | rg for `qd.sock`, `LAPPI_SOCKET`, `lappi` over `*.go` found no Lappi client | none | none | n/a |
| in-repo `qd oneshot`, `qd serve`, `qd-metal-serve` | pass requests through; author none | any | the caller's | per release |
| in-repo qd-preflight `build-requests` | builds requests from the release's own `trained_slot_shapes` (`release_check.rs:314-316`) | per release | recorded shapes | no, by construction |

Option counts for DevCouncil, GitPulse and DevType were read from the client constants above
(verified). Vani's counts are from the contract section 4 table and the pinned fixture's slot
headers (inferred; not each option counted here).

Result: no live caller sends a shape this rule would refuse; each sends exactly the contract
shape. The method's stop condition is not met; work proceeds.

Search method: rg and file reads for the sibling repos (their DevMap indexes were not queried).
DevMap in this repo located `admit` (admission.rs:124) but reported `walk_incomplete`, so the
call-site lists here come from rg with default ignores (not `-uu`).
