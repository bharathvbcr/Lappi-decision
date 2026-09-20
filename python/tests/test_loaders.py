"""Bounded readers, and the one network test in the lane.

The offline tests are the substance: bounds are enforced, a malformed row is refused
naming the field, and a gated source raises with its gate named rather than with an
opaque 403.

:func:`test_a_few_hundred_real_commitpackft_rows_parse` is marked ``network``. It is
**reported not run** when the host is unreachable and is never reported as passed --
``pyproject.toml`` registers the marker for exactly this. A single synthetic fixture
proving the parser works is not evidence that the parser matches the *upstream
schema*, which is why the network test exists at all; and a network test that
silently passes offline would be worse than not having one.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from data_fixtures import commitpackft_row
from qd_data.loaders import (
    MAX_FETCH_ROWS,
    MalformedRowRefusal,
    SourceUnavailableRefusal,
    fetch_rows,
    load_agentpack,
    parse_clinc,
    parse_commitpackft,
    parse_squad,
    read_jsonl,
)


def _raw(row) -> dict[str, object]:
    return {
        "commit": row.commit, "repos": row.repos, "old_file": row.old_file,
        "new_file": row.new_file, "old_contents": row.old_contents,
        "new_contents": row.new_contents, "subject": row.subject,
        "message": row.message, "lang": row.lang, "license": row.licence,
    }


def _write_jsonl(path: Path, objects: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(o) + "\n" for o in objects), encoding="utf-8"
    )
    return path


# -- bounds ------------------------------------------------------------------


def test_reading_stops_at_the_row_limit_and_says_it_was_capped(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "pool.jsonl", [_raw(commitpackft_row(i)) for i in range(10)]
    )
    rows, capped = read_jsonl(path, limit=4, max_row_bytes=1 << 20)
    assert len(rows) == 4
    assert capped is True
    rows, capped = read_jsonl(path, limit=100, max_row_bytes=1 << 20)
    assert len(rows) == 10
    assert capped is False


def test_an_oversized_row_is_refused_not_truncated(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "pool.jsonl", [_raw(commitpackft_row(0))])
    with pytest.raises(MalformedRowRefusal) as excinfo:
        read_jsonl(path, limit=10, max_row_bytes=16)
    assert "refused rather than truncated" in str(excinfo.value)


def test_a_malformed_line_is_refused_naming_the_line(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    path.write_text('{"a": 1}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(MalformedRowRefusal) as excinfo:
        read_jsonl(path, limit=10, max_row_bytes=1 << 20)
    assert ":2" in str(excinfo.value)


def test_a_json_array_line_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(MalformedRowRefusal):
        read_jsonl(path, limit=10, max_row_bytes=1 << 20)


# -- schema fidelity ---------------------------------------------------------


def test_commitpackft_parses_every_documented_field() -> None:
    row = commitpackft_row(3, licence="apache-2.0", repo="a/b,c/d")
    parsed = parse_commitpackft(_raw(row), index=0)
    assert parsed.licence == "apache-2.0"
    assert parsed.primary_repo == "a/b", "the first repo is the split unit"


def test_a_missing_licence_field_is_refused_not_defaulted() -> None:
    """The per-row ``license`` is why this source is primary; its absence is a
    refusal, never an empty string."""
    raw = _raw(commitpackft_row(1))
    del raw["license"]
    with pytest.raises(MalformedRowRefusal) as excinfo:
        parse_commitpackft(raw, index=7)
    assert "'license'" in str(excinfo.value)
    assert "row 7" in str(excinfo.value)


def test_a_field_of_the_wrong_type_is_refused_naming_both_types() -> None:
    raw = _raw(commitpackft_row(1))
    raw["lang"] = 42
    with pytest.raises(MalformedRowRefusal) as excinfo:
        parse_commitpackft(raw, index=0)
    assert "str" in str(excinfo.value) and "int" in str(excinfo.value)


def test_clinc_marks_the_out_of_scope_class() -> None:
    assert parse_clinc({"text": "hello", "intent": "oos"}, index=0).is_oos is True
    assert parse_clinc({"text": "hello", "intent": "balance"}, index=0).is_oos is False


def test_squad_derives_unanswerability_from_an_empty_answer_list() -> None:
    """SQuAD 2.0 on the hub encodes unanswerability as an empty answers list."""
    base = {
        "id": "q1", "title": "T", "context": "A passage.", "question": "What?",
        "answers": {"text": [], "answer_start": []},
    }
    assert parse_squad(base, index=0).is_impossible is True
    answered = {**base, "answers": {"text": ["A"], "answer_start": [0]}}
    assert parse_squad(answered, index=0).is_impossible is False


def test_squad_refuses_a_row_that_is_both_unanswerable_and_answered() -> None:
    raw = {
        "id": "q1", "title": "T", "context": "A passage.", "question": "What?",
        "answers": {"text": ["A"], "answer_start": [0]}, "is_impossible": True,
    }
    with pytest.raises(MalformedRowRefusal) as excinfo:
        parse_squad(raw, index=0)
    assert "two labels" in str(excinfo.value)


def test_squad_refuses_mismatched_answers_and_offsets() -> None:
    raw = {
        "id": "q1", "title": "T", "context": "A passage.", "question": "What?",
        "answers": {"text": ["A", "B"], "answer_start": [0]},
    }
    with pytest.raises(MalformedRowRefusal) as excinfo:
        parse_squad(raw, index=0)
    assert "one start offset per answer" in str(excinfo.value)


# -- the gate ----------------------------------------------------------------


def test_agentpack_raises_with_its_gate_named_and_says_what_unblocks_it() -> None:
    """BLOCKING-1. Nothing depends on this; it exists so the slot-in is one edit."""
    with pytest.raises(SourceUnavailableRefusal) as excinfo:
        load_agentpack()
    message = str(excinfo.value)
    assert "gated" in message
    assert "accept the terms" in message
    assert "Reachability.LOADABLE" in message


@pytest.mark.parametrize(
    "source_id", ["nuprl/AgentPack", "PolyAI/banking77", "microsoft/CodeReviewer"]
)
def test_fetching_an_unreachable_source_refuses_before_touching_the_network(
    source_id: str,
) -> None:
    with pytest.raises(SourceUnavailableRefusal) as excinfo:
        fetch_rows(source_id, config_name="default", split_name="train", limit=1)
    assert excinfo.value.check == "source_reachable"


def test_the_fetch_bound_cannot_be_raised_by_a_caller() -> None:
    with pytest.raises(ValueError, match=f"1..{MAX_FETCH_ROWS}"):
        fetch_rows(
            "bigcode/commitpackft", config_name="default", split_name="train",
            limit=MAX_FETCH_ROWS + 1,
        )


# -- the one network test ----------------------------------------------------


def _host_reachable() -> bool:
    try:
        request = urllib.request.Request(
            "https://datasets-server.huggingface.co/valid",
            headers={"User-Agent": "qwen-decision/0.1"},
        )
        with urllib.request.urlopen(request, timeout=5):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@pytest.mark.network
def test_a_few_hundred_real_commitpackft_rows_parse() -> None:
    """The parser against the real upstream schema, bounded to 100 rows.

    Skipped -- and therefore reported NOT RUN, never passed -- when the host is
    unreachable. A synthetic fixture proves the parser is self-consistent; only this
    proves it matches what ``bigcode/commitpackft`` actually serves.
    """
    if not _host_reachable():
        pytest.skip("datasets-server unreachable: reported NOT RUN, never as passed")
    rows = fetch_rows(
        "bigcode/commitpackft", config_name="python", split_name="train", limit=100
    )
    assert rows, "the host answered but returned no rows"
    parsed = [parse_commitpackft(r, index=i) for i, r in enumerate(rows)]
    assert all(p.licence for p in parsed), "every row must carry a per-row licence"
    assert all(p.primary_repo for p in parsed)
    declared = {p.licence for p in parsed}
    from qd_data.licences import COMMITPACKFT_DECLARED_VALUES

    unknown = declared - set(COMMITPACKFT_DECLARED_VALUES)
    assert not unknown, f"upstream serves licence values the policy table lacks: {unknown}"
