"""The span rebase over qd-mutate's multi-hunk diffs, checked on rows the generator wrote.

``data/qd_mutate_multi_hunk/`` is real ``qd-mutate generate`` output, committed: a twelve-record
pool whose commits change ``alpha`` and ``beta`` (records 0-7) or one line inside ``gamma``
(records 8-11), with every mutation pinned inside ``gamma`` by the pool's ``hunks``. Its diffs
run from the pre-image, so a mutated row carries the commit's hunks beside the injected one --
three ``@@`` headers when the commit is far away, one when it is adjacent. Regenerated with::

    target/release/qd-mutate generate \\
      --pool python/tests/data/qd_mutate_multi_hunk/pool.jsonl \\
      --out python/tests/data/qd_mutate_multi_hunk/examples.jsonl \\
      --manifest python/tests/data/qd_mutate_multi_hunk/manifest.json \\
      --seed 0 --clean-permille 250

``crates/qd-mutate/tests/multi_hunk_fixture.rs`` re-renders every diff here and fails if the
renderer drifts from these bytes. What this file pins is the other half: that
``mutate_adapter.diff_offset_of_after_line`` -- through ``defect_class.diff_line_span`` and
``load_defect_rows`` -- walks past the commit's hunks and lands the span on the injected lines,
rather than on a line of an earlier hunk that merely shares a line number.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from qd_data.config import DataConfig
from qd_data.defect_class import diff_line_span, load_defect_rows

FIXTURE = Path(__file__).parent / "data" / "qd_mutate_multi_hunk"


def _rows() -> list[dict[str, Any]]:
    with (FIXTURE / "examples.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _mutated() -> list[dict[str, Any]]:
    return [r for r in _rows() if r["class"] != "clean"]


def _hunk_of(diff_lines: list[str], line_no: int) -> int:
    """1-based index of the hunk holding diff line ``line_no`` (1-based)."""
    return sum(1 for line in diff_lines[:line_no] if line.startswith("@@ "))


def test_the_fixture_exercises_both_shapes() -> None:
    headers = [r["diff"].count("\n@@ ") + r["diff"].startswith("@@ ") for r in _mutated()]
    assert headers.count(3) >= 4, headers
    assert headers.count(1) >= 2, headers
    assert {r["class"] for r in _rows()} >= {"clean", "logic"}


def test_the_rebased_span_lands_on_the_injected_lines_of_the_injected_hunk() -> None:
    for row in _mutated():
        diff = row["diff"]
        span = row["span"]
        start, end = diff_line_span(
            diff.encode(), start_line=span["start_line"], end_line=span["end_line"]
        )
        diff_lines = diff.split("\n")
        after_lines = row["after"].split("\n")
        picked = diff_lines[start - 1 : end]
        # Both ends are lines of `after` (added or unchanged), and no header falls between them.
        # Removed lines may: a span over several lines with changes interleaved covers the `-`
        # lines among them in diff space, exactly as `diff -u` interleaves them.
        assert picked[0][:1] in ("+", " ") and picked[-1][:1] in ("+", " "), (row["id"], picked)
        assert not any(line.startswith("@") for line in picked), (row["id"], picked)
        # The `after` lines the span covers in diff space are THE lines of `after` it names, in
        # order -- not a same-numbered line of an earlier hunk.
        assert [line[1:] for line in picked if line[:1] in ("+", " ")] == after_lines[
            span["start_line"] - 1 : span["end_line"]
        ], row["id"]
        # It is the needle: the change is shown there, as an added line in the span or as the
        # removed lines directly before a pure deletion's join line.
        shown = any(line.startswith("+") for line in picked) or diff_lines[start - 2].startswith(
            "-"
        )
        assert shown, (row["id"], picked)
        # One hunk holds the whole span, and when the commit's two hunks are there too, it is
        # the third -- the walk did not stop in an earlier hunk on a shared line number.
        assert _hunk_of(diff_lines, start) == _hunk_of(diff_lines, end), row["id"]
        n_hunks = sum(1 for line in diff_lines if line.startswith("@@ "))
        if n_hunks == 3:
            assert _hunk_of(diff_lines, start) == 3, (row["id"], start)


def test_load_defect_rows_rebases_every_span_with_no_refusal(tmp_path: Path) -> None:
    pool_dir = tmp_path / "data" / "pool"
    corpus = pool_dir / "corpus"
    download = pool_dir / "commitpackft"
    corpus.mkdir(parents=True)
    download.mkdir(parents=True)
    # Copied byte for byte, so the sha256s the generator's manifest pins still hold.
    shutil.copyfile(FIXTURE / "pool.jsonl", pool_dir / "pool.jsonl")
    shutil.copyfile(FIXTURE / "examples.jsonl", corpus / "examples.jsonl")
    shutil.copyfile(FIXTURE / "manifest.json", corpus / "manifest.json")

    with (FIXTURE / "pool.jsonl").open(encoding="utf-8") as fh:
        pool_ids = [json.loads(line)["id"] for line in fh if line.strip()]
    dl_rows = []
    for pid in pool_ids:
        commit, new_file = pid.split(":", 1)
        dl_rows.append({"commit": commit, "new_file": new_file, "license": "mit"})
    dl_path = download / "python.jsonl"
    dl_path.write_text("".join(json.dumps(r) + "\n" for r in dl_rows), encoding="utf-8")
    (download / "manifest.json").write_text(
        json.dumps(
            {
                "source_id": "bigcode/commitpackft",
                "languages": {
                    "python": {"sha256": hashlib.sha256(dl_path.read_bytes()).hexdigest()}
                },
            }
        ),
        encoding="utf-8",
    )

    load = load_defect_rows(
        corpus, download_root=download, config=DataConfig(), repo_root=tmp_path
    )
    rows = _rows()
    assert load.n_corpus == len(rows)
    assert load.span_refusals == {}
    by_id = {r["id"]: r for r in rows}
    for row in load.rows:
        raw = by_id[row.example_id]
        if raw["class"] == "clean":
            assert row.diff_span is None
            continue
        span = raw["span"]
        assert row.diff_span == diff_line_span(
            raw["diff"].encode(), start_line=span["start_line"], end_line=span["end_line"]
        )
        assert row.diff == raw["diff"]
