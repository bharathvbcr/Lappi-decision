"""``tools/train_examples_export.py``: the text file ``qd-train-data --examples`` reads.

A manifest that disagrees with itself is refused, the written lines are the shape
``qd_prep::pool::read_example_texts`` parses (id, source_id, family_id, context, options), and an
existing output is never overwritten. The rebuild itself (``ft_split_report``) is heavy and is
exercised by the build that uses this tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))


def _manifest(tmp_path: Path, n_rows: int, ids: list[str]) -> Path:
    path = tmp_path / "train.json"
    path.write_text(json.dumps({
        "n_rows": n_rows,
        "entries": [{"row_id": i, "source_id": f"src-{k}"} for k, i in enumerate(ids)],
    }), encoding="utf-8")
    return path


def test_manifest_ids_maps_rows_to_sources(tmp_path: Path) -> None:
    from train_examples_export import _manifest_ids

    assert _manifest_ids(_manifest(tmp_path, 2, ["a", "b"])) == {"a": "src-0", "b": "src-1"}


def test_manifest_with_wrong_row_count_is_refused(tmp_path: Path) -> None:
    from train_examples_export import _manifest_ids

    with pytest.raises(SystemExit, match="n_rows 3 but 2 entries"):
        _manifest_ids(_manifest(tmp_path, 3, ["a", "b"]))


def test_written_lines_carry_the_fields_qd_train_data_reads(tmp_path: Path) -> None:
    from train_examples_export import _write

    path = tmp_path / "x.jsonl"
    digest = _write(path, [("r1#slot", "clinc/clinc_oos", "intent.domain", "what is the time")])
    (line,) = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {
        "id": "r1#slot", "source_id": "clinc/clinc_oos", "family_id": "intent.domain",
        "context": "what is the time", "options": [],
    }
    import hashlib

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    from train_examples_export import _write

    path = tmp_path / "x.jsonl"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _write(path, [("k", "s", "f", "t")])
    assert path.read_text(encoding="utf-8") == "keep"
