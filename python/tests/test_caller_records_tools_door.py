"""Caller-record guard on the tool entry points that open pool or source rows (CLAUDE.md rule 3).

``test_caller_records_guard.py`` covers the library readers and the shard pipeline's. These are
the ``tools/*.py`` scripts that read pool JSONL themselves and write the corpus or map a later
step trains from: each is shown refusing a copy of ``fixtures/caller/valid-records.jsonl`` placed
under an ordinary data directory (never a ``heldout`` segment), and admitting a clean file.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import build_commitpackft_pool  # noqa: E402
import compose_split_map  # noqa: E402

from qd_data.caller_records import CallerRecordRefusal  # noqa: E402
from qd_data.config import DataConfig  # noqa: E402

RECORDS = REPO / "fixtures/caller/valid-records.jsonl"


def _copied(tmp_path: Path, rel: str) -> Path:
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(RECORDS, dest)
    return dest


def test_build_commitpackft_pool_refuses_a_copied_record_file(tmp_path):
    _copied(tmp_path, "data/pool/commitpackft/python.jsonl")
    out = tmp_path / "out/pool.jsonl"
    with pytest.raises(CallerRecordRefusal):
        build_commitpackft_pool.build(tmp_path / "data/pool/commitpackft", out, max_records=None)
    assert not out.exists() or out.read_text(encoding="utf-8") == ""


def test_build_commitpackft_pool_refuses_a_record_file_beside_clean_ones(tmp_path):
    src = tmp_path / "data/pool/commitpackft"
    src.mkdir(parents=True)
    (src / "a.jsonl").write_text('{"not":"a commit"}\n', encoding="utf-8")
    shutil.copy(RECORDS, src / "z.jsonl")
    with pytest.raises(CallerRecordRefusal):
        build_commitpackft_pool.build(src, tmp_path / "out/pool.jsonl", max_records=None)


def test_build_commitpackft_pool_still_reads_a_clean_source(tmp_path):
    src = tmp_path / "data/pool/commitpackft"
    src.mkdir(parents=True)
    (src / "python.jsonl").write_text('{"not":"a commit"}\n', encoding="utf-8")
    manifest = build_commitpackft_pool.build(src, tmp_path / "out/pool.jsonl", max_records=None)
    assert manifest["records"] == 0 and manifest["dropped_total"] == 1


def test_compose_split_map_refuses_a_copied_record_file(tmp_path):
    pool = _copied(tmp_path, "data/pool/commitpackft-pool.jsonl")
    with pytest.raises(CallerRecordRefusal):
        compose_split_map.split_map(pool, config=DataConfig(), repo_root=tmp_path)


def test_compose_split_map_still_reads_a_clean_pool(tmp_path):
    pool = tmp_path / "data/pool/pool.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text(json.dumps({"repo": "example/r", "id": "x"}) + "\n", encoding="utf-8")
    doc, repos = compose_split_map.split_map(pool, config=DataConfig(), repo_root=tmp_path)
    assert list(repos) == ["example/r"] and doc["schema"] == compose_split_map.SCHEMA
