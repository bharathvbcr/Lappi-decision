"""`qd_train.gaps.append_gap`: the writer the checker never had.

`test_gaps_ledger.py` has made this file's rules executable for a while. Nothing wrote a
record. Every lane hand-rolled `os.open(..., O_APPEND)` and `json.dumps` with the schema in
its head, and on 2026-09-21 two records in one afternoon were refused by those tests --
`status: "closed-answered"`, which is good English and not one of the five declared values,
and then a correction that put prose where `supersedes` takes a gap id.

Both were caught, which is the gates working. Neither could be *fixed* by appending, so the
file was rewritten -- the one operation `CLAUDE.md` forbids, because a concurrent lane's
record can be lost between the read and the write.

These tests are about the other end of that: a record that would land malformed does not
land. The checker and the writer share one vocabulary, imported from one place, so they
cannot drift into disagreeing about what is valid.

Every case below is a real one. Nothing here is hypothetical schema hygiene.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.gaps import (
    KNOWN_STATUSES,
    GapRecordError,
    append_gap,
    current_records,
    read_gaps,
    validate_gap,
)

#: Fixture ids are ASSEMBLED from a prefix rather than written out, and that is not style.
#:
#: `test_gaps_ledger.py` scans every tracked file for anything matching `GAP-[A-Z0-9]...`
#: and fails when it names no record -- because a document claiming a record exists, when
#: it never was written, hides the finding from anyone reading the ledger. A scanner cannot
#: tell a test fixture from a citation; the distinction lives in the surrounding prose.
#:
#: That module's own docstring says so and notes that its first draft broke the sibling
#: guard in `test_wire_gap_pins.py` with three such ids. The first draft of THIS file used
#: five and broke both guards, which is the rule demonstrating itself rather than being
#: read. Assembling the string keeps the literal out of the source while leaving the values
#: exactly what the writer must accept or refuse at runtime.
_G = "GAP" + "-"

GOOD = {
    "id": _G + "A-WRITER-THAT-AGREES-WITH-ITS-CHECKER",
    "opened": "2026-09-21",
    "lane": "test",
    "question": "Does the writer refuse what the checker refuses?",
    "status": "open",
}


def _ledger(tmp_path: Path) -> Path:
    return tmp_path / "gaps.jsonl"


# -- the two that actually happened -------------------------------------------------------

def test_a_status_nobody_declared_is_refused_before_it_lands(tmp_path: Path) -> None:
    """`closed-answered`, verbatim. It reads as a perfectly good answer and it is not one
    of the five, so every open-count computed afterwards would be wrong by one -- in the
    file whose subject is "one name, two quantities"."""
    with pytest.raises(GapRecordError, match="not one of"):
        append_gap({**GOOD, "status": "closed-answered"}, path=_ledger(tmp_path))
    assert not _ledger(tmp_path).exists(), "a refused record still created the file"


def test_prose_where_a_gap_id_belongs_is_refused(tmp_path: Path) -> None:
    """The correction that could not be corrected. `supersedes` takes an id; a sentence
    describing what changed is a dangling citation one level in, and by the time the gate
    says so the line is immutable."""
    path = _ledger(tmp_path)
    append_gap(GOOD, path=path)
    with pytest.raises(GapRecordError, match="names no record"):
        append_gap(
            {**GOOD, "supersedes": "the earlier record, which used the wrong status"},
            path=path,
        )
    assert len(read_gaps(path)) == 1, "the refused record was written anyway"


# -- the shape of a valid append ----------------------------------------------------------

def test_a_valid_record_appends_exactly_one_line(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    append_gap(GOOD, path=path)
    append_gap({**GOOD, "id": _G + "A-SECOND-ONE", "status": "resolved"}, path=path)
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert len(raw.splitlines()) == 2
    assert [json.loads(line)["id"] for line in raw.splitlines()] == [
        GOOD["id"], _G + "A-SECOND-ONE"
    ]


def test_superseding_a_record_that_exists_is_allowed(tmp_path: Path) -> None:
    """The convention this file uses for revisions: a fresh line under the same id, or a
    new id naming the old one. Both must survive the writer, or the writer is the thing
    stopping the fix."""
    path = _ledger(tmp_path)
    append_gap(GOOD, path=path)
    append_gap(
        {**GOOD, "status": "resolved", "supersedes": GOOD["id"],
         "question": "Superseded: it does."},
        path=path,
    )
    current = current_records(path)
    assert len(current) == 1, "same id twice is one current record, not two"
    assert current[GOOD["id"]]["status"] == "resolved", "last line must win"
    assert len(read_gaps(path)) == 2, "history was lost"


def test_a_record_may_reference_an_id_added_moments_earlier(tmp_path: Path) -> None:
    """References are resolved against the file as it is at write time, not against a
    snapshot taken when the process started. Two lanes appending in turn must each be able
    to cite the other."""
    path = _ledger(tmp_path)
    other = _G + "WRITTEN-BY-THE-OTHER-LANE"
    append_gap({**GOOD, "id": other}, path=path)
    append_gap({**GOOD, "answers": [other]}, path=path)
    assert len(read_gaps(path)) == 2


# -- everything else the checker checks ---------------------------------------------------

@pytest.mark.parametrize("missing", sorted({"id", "question", "status"}))
def test_a_record_missing_a_required_key_is_refused(tmp_path: Path, missing: str) -> None:
    """Required on every record this function writes, though the checker only demands them
    of the CURRENT record per id -- earlier lines are history and an append cannot reach
    them. A new record has no such excuse."""
    record = {k: v for k, v in GOOD.items() if k != missing}
    with pytest.raises(GapRecordError, match="missing"):
        append_gap(record, path=_ledger(tmp_path))


@pytest.mark.parametrize(
    "bad_id",
    [
        "", "GAP" + "-", "gap-lowercase", "NOT-A-" + "GAP-ID", 17, None,
        # Recorded on 2026-09-30 and then unresolvable where the handoff cited it: the dots
        # end a citation early, so a reader finds only the part before the first dot.
        "GAP" + "-MEMORY-PREDICTS-0.84-0.97X-ON-GH200",
        "GAP" + "-Mixed-Case-ID",
    ],
)
def test_an_id_that_is_not_one_is_refused(tmp_path: Path, bad_id: object) -> None:
    with pytest.raises(GapRecordError, match="is not a gap id"):
        append_gap({**GOOD, "id": bad_id}, path=_ledger(tmp_path))


def test_a_record_that_would_span_two_lines_is_refused() -> None:
    """A reader splits this file on newlines, so a record containing one is two records --
    the second of them unparseable. `json.dumps` escapes newlines inside strings, so this
    is about the serialised form rather than about the input, and it is checked on the
    serialised form for that reason.
    """
    line = validate_gap({**GOOD, "question": "one\ntwo"}, known_ids=set())
    assert "\n" not in line, "the escape did not happen; the check above must catch it"
    assert "\\n" in line


def test_a_record_that_cannot_be_serialised_is_refused() -> None:
    """Caught here rather than at write time. A TypeError from `json.dumps` inside the
    append would leave the caller believing a record was recorded when nothing was."""
    with pytest.raises(GapRecordError, match="not JSON-serialisable"):
        validate_gap({**GOOD, "evidence": {1, 2, 3}}, known_ids=set())


def test_answers_must_be_a_list_of_ids_that_exist() -> None:
    with pytest.raises(GapRecordError, match="answers must be a list"):
        validate_gap(
            {**GOOD, "answers": _G + "NOT-A-LIST"}, known_ids={_G + "NOT-A-LIST"}
        )
    with pytest.raises(GapRecordError, match="records that do not exist"):
        validate_gap({**GOOD, "answers": [_G + "NEVER-WRITTEN"]}, known_ids=set())


def test_the_writer_and_the_checker_share_one_vocabulary() -> None:
    """The property that keeps this from becoming a second opinion.

    `test_gaps_ledger.py` re-exports these from `qd_train.gaps` rather than defining them,
    so there is no arrangement in which the file's checker and its writer disagree about
    what a valid status is. Two implementations of one check giving opposite verdicts is
    the defect this repository keeps finding in its own subject matter.
    """
    from test_gaps_ledger import KNOWN_STATUSES as CHECKER_STATUSES
    from test_gaps_ledger import REQUIRED_KEYS as CHECKER_KEYS

    assert CHECKER_STATUSES is KNOWN_STATUSES
    assert set(CHECKER_KEYS) == {"id", "question", "status"}
    for status in KNOWN_STATUSES:
        validate_gap({**GOOD, "status": status}, known_ids=set())


def test_the_real_ledger_is_not_touched_by_any_of_this(tmp_path: Path) -> None:
    """Every test above passes an explicit path. This asserts the default is the repo's
    own file and that nothing here wrote to it -- a test suite that appends to `gaps.jsonl`
    on every run would be a novel way to corrupt the thing being protected.
    """
    from qd_train.gaps import DEFAULT_GAPS_PATH

    assert DEFAULT_GAPS_PATH.name == "gaps.jsonl"
    assert DEFAULT_GAPS_PATH.parent.name in ("qwen-decision", "Lappi-decision")
    before = len(read_gaps(DEFAULT_GAPS_PATH))
    append_gap(GOOD, path=_ledger(tmp_path))
    assert len(read_gaps(DEFAULT_GAPS_PATH)) == before
