"""``tools/arm_b_concat_corpus.py``: parts joined byte for byte, and a manifest that pins them."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import arm_b_concat_corpus as concat  # noqa: E402


def _part(root: Path, name: str, rows: list[str], *, pool_sha: str = "p" * 64) -> Path:
    d = root / name
    d.mkdir(parents=True)
    body = "".join(r + "\n" for r in rows).encode()
    (d / "examples.jsonl").write_bytes(body)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "seed": 7,
                "pool": {"path": "data/pool/p.jsonl", "sha256": pool_sha, "records": 3},
                "examples_sha256": hashlib.sha256(body).hexdigest(),
                "totals": {"examples": len(rows), "clean": 1, "note": "text"},
            }
        )
    )
    return d


def test_parts_are_joined_byte_for_byte_and_the_manifest_pins_the_result(tmp_path: Path) -> None:
    a = _part(tmp_path, "go", ['{"id":"a"}', '{"id":"b"}'])
    b = _part(tmp_path, "python", ['{"id":"c"}'])
    manifest = concat.concat([a, b], tmp_path / "out")
    body = (tmp_path / "out" / "examples.jsonl").read_bytes()
    assert body == (a / "examples.jsonl").read_bytes() + (b / "examples.jsonl").read_bytes()
    assert manifest["examples_sha256"] == hashlib.sha256(body).hexdigest()
    assert manifest["totals"] == {"examples": 3, "clean": 2, "note": "text"}
    assert [p["part"] for p in manifest["concatenation_of"]] == ["go", "python"]
    assert json.loads((tmp_path / "out" / "manifest.json").read_text()) == manifest


def test_a_part_over_another_pool_or_changed_after_its_manifest_is_refused(
    tmp_path: Path,
) -> None:
    a = _part(tmp_path, "go", ['{"id":"a"}'])
    other = _part(tmp_path, "rust", ['{"id":"b"}'], pool_sha="q" * 64)
    with pytest.raises(concat.ConcatRefusal, match="different pool"):
        concat.concat([a, other], tmp_path / "o1")
    (a / "examples.jsonl").write_bytes(b'{"id":"tampered"}\n')
    with pytest.raises(concat.ConcatRefusal, match="does not hash"):
        concat.concat([a], tmp_path / "o2")
    ok = _part(tmp_path, "ok", ['{"id":"z"}'])
    concat.concat([ok], tmp_path / "o3")
    with pytest.raises(concat.ConcatRefusal, match="exists"):
        concat.concat([ok], tmp_path / "o3")
