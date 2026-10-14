# ruff: noqa: E501
"""Wave-3 L4: the ft-a4ef800 and ft-ab55b250 ledger records. Each keeps the record's earlier fields. Dry run unless --apply."""

import sys

from qd_train.gaps import append_gap, current_records

APPLY = "--apply" in sys.argv
LANE = "wave3-l4-gaps-remainder (ft-5c92df4cc91)"
DAY = "2026-10-13"
cur = current_records()


def put(gid: str, **fields: object) -> None:
    out = {**cur[gid], **fields}
    out["lane"] = LANE
    if APPLY:
        append_gap(out)
    else:
        print(gid, out["status"])


put(
    "GAP-NEEDLE-VERDICT-STRING-NAMES-A-CAUSE-THE-BUCKET-CONTRADICTS-2026-10-06",
    status="resolved",
    closed=DAY,
    evidence_commit="39ccdbc",
    closure=(
        "Fixed in 39ccdbc ('needle: the failing verdict names the failing bucket's position, not a "
        "fixed cause', ft-a4ef800cc7a item 1), an ancestor of this branch. "
        "python/qd_train/needle.py:485-490 builds the verdict from _POSITION[worst.label] ('depth "
        "<label> (<position> of the context) is below the <min_recall> threshold') and asserts no "
        "cause; the 'recurrent model losing early context' string appears nowhere under python/ or "
        "tools/ except the docstring of the test that pins its absence. "
        "python/tests/test_needle_verdict_position.py (5 test functions, 13 cases with the bucket "
        "parametrisation) passes: 13 passed in 0.19s, run 2026-10-13 on this branch with "
        "pytest -o addopts=. The pre-fix string is the thing the test's test_no_verdict_asserts_a_"
        "fixed_cause rejects."
    ),
)

put(
    "GAP-SPAN-CANDIDATE-STATE-HAS-NOT-READ-ITS-LINE-2026-10-06",
    status="open",
    update_wave3_l4=(
        "ft-a4ef800cc7a item 4 (arm (e), last-token candidates on frozen features) is BLOCKED on "
        "the MPS feature extraction, not on a decision: HANDOFF/span-rescore-2026-10-12.md:3 "
        "('B partly run: avg-train extraction complete, seed-1 val extraction 4,811 of 7,238 rows, "
        "fits and scoring not run') and :19 (the seed-1 extraction was killed by wrapper exit 143 "
        "three times and once more with a detached launch, each within about 6 s of another lane's "
        "mac_heavy start, sender not identified; 'No arm row, no (e)-vs-(c) reading'). The seed-1 "
        "cache resumes (4,811 rows). No ledger row exists for arm (e). Needs a heavy MPS job and "
        "the human's Mac-risk yes (Fable wave-3 ruling section 5). The cost named in this record "
        "stays [U]."
    ),
    board_card="ft-a4ef800cc7a (item 4)",
)

put(
    "GAP-DEVMAP-RUST-CALL-RESOLUTION-28PCT-2026-10-06",
    status="resolved-with-residual",
    board_card=(
        "DevCouncil ft-5ec7b111af07d336b89f2f31044a5767 (named by ft-ab55b250 criterion 1; "
        "ft-5c92df4c also lists ft-32522da4 for DevMap Rust call resolution)"
    ),
    update_wave3_l4=(
        "Board annotation, nothing re-measured: the Rust call-resolution capability is raised on "
        "DevCouncil ft-5ec7b111af07d336b89f2f31044a5767. The residual in this record (resolution "
        "is a DevMap capability, net 528 permille at gen 3883, no agreed bar) is that card's "
        "to carry; Lappi cannot close it. Appended here rather than in main's checkout because "
        "main's gaps.jsonl had uncommitted lines (ft-be802e07)."
    ),
)
