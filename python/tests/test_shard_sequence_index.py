"""Slot-scoped refusal and the structured sequence index ``write_shards`` writes beside the set.

Two defects, one mechanism:

* ``GAP-A3-BPE-SPAN-COLLAPSE-DROPS-DEFECT-CLASS-LABELS``. A ``code.defect_class`` row renders
  two sequences, a class choice and a defect span. When the span's line starts collapsed
  under BPE the writer refused the WHOLE row, class label included, and did so unevenly by
  class -- which moved the majority rate the choice head is judged against.
* ``GAP-PORTED-SHARD-EXCLUSIONS-ARE-FREE-TEXT``. Which ``(row, slot)`` each written sequence
  is lived nowhere but in a free-text coverage summary, so a consumer re-deriving labels
  could not pair them to sequences once any slot was dropped.

The tokenizer here is byte-level for every sequence except a span slot's, which it
collapses onto two tokens -- the shape ``_span_token_positions`` refuses.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.defect_class import DEFECT_FAMILY_ID, DEFECT_SOURCE_ID, DefectRow
from qd_data.general import REPLAY_ONLY, REPLAY_ROLE_KEY
from qd_data.manifest import build_manifests
from qd_data.mixture import build_mixture
from qd_data.rows import DataRow
from qd_data.split import HELD_OUT, split
from qd_train.artifacts import (
    SLOT_CHOICE,
    RemapTable,
    ShardContractViolation,
    ShardHeader,
)
from qd_train.shards import (
    COVERAGE_NAME,
    GOLD_ROLE,
    HEADER_NAME,
    SEQUENCE_INDEX_FORMAT,
    SEQUENCE_INDEX_NAME,
    ShardReader,
    UnencodableGold,
    write_shards,
)
from qd_train.tristate import NotRun, Ran

CHOICE = "defect_class"
SPAN = "defect_span"


def _tokenize(text: str) -> list[int]:
    if "<|qd_type|>span" in text:
        return [65, 66]
    return list(text.encode("utf-8"))


def _offsets(text: str) -> list[tuple[int, int]]:
    if "<|qd_type|>span" in text:
        cut = max(1, len(text) - 1)
        return [(0, cut), (cut, len(text))]
    out: list[tuple[int, int]] = []
    for ci, ch in enumerate(text):
        out.extend((ci, ci + 1) for _ in ch.encode("utf-8"))
    return out


def _remap() -> RemapTable:
    old_to_new = np.arange(256, dtype=np.int32)
    return RemapTable(
        old_to_new=old_to_new,
        new_to_old=np.arange(256, dtype=np.int32),
        tokenizer_hash="tokhash-sequence-index",
        special_ids=(),
    )


def _defect_rows(n_repos: int = 30) -> list[DefectRow]:
    out = []
    for i in range(n_repos):
        for j, cls in enumerate(("stub", "logic", "cosmetic", "clean")):
            diff = f"@@ -1,2 +1,2 @@\n def f{i}_{j}(x):\n-    return x\n+    return {i}{j}\n"
            out.append(
                DefectRow(
                    example_id=f"c{i}_{j}:x.py#0", pool_id=f"c{i}_{j}:x.py", repo=f"org/r{i}",
                    path="x.py", symbol="f", arity=1, language="python", mutation_class=cls,
                    operator="clean" if cls == "clean" else f"{cls}.op", diff=diff,
                    diff_span=None if cls == "clean" else (3, 3), span_refusal=None,
                    licence="mit",
                )
            )
    return out


def _train(tmp_path: Path) -> tuple[Path, list[DataRow], DataConfig]:
    config = DataConfig()
    mixture = build_mixture(
        {DEFECT_SOURCE_ID: _defect_rows()}, config=config, families=[DEFECT_FAMILY_ID]
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    paths = {}
    for name, manifest in manifests.items():
        path = tmp_path / "data" / (HELD_OUT if name == HELD_OUT else "pool") / f"{name}.json"
        manifest.write(path)
        paths[name] = path
    rows = list(split_report.rows_by_split["train"])
    assert rows and all(len(r.request.slots) == 2 for r in rows)
    return paths["train"], rows, config


def _write(
    tmp_path: Path, out: Path, **kw: object
) -> tuple[ShardHeader, list[DataRow], DataConfig]:
    manifest, rows, config = _train(tmp_path)
    header = write_shards(
        manifest, rows, out_dir=out, remap=_remap(), tokenize=_tokenize,
        token_offsets=_offsets, config=config, repo_root=tmp_path, buckets=None, **kw,  # type: ignore[arg-type]
    )
    return header, rows, config


def test_a_two_slot_row_whose_span_collapses_keeps_its_choice_sequence(tmp_path: Path) -> None:
    out = tmp_path / "shards"
    header, rows, config = _write(tmp_path, out, allow_unencodable=True)
    # One sequence per row: the choice. Before the fix every row was refused whole, so the
    # write had nothing left and raised.
    assert header.n_sequences == len(rows)
    reader = ShardReader(out, config=config, repo_root=tmp_path)
    assert reader.sequence_index is not None
    ordered = sorted(rows, key=lambda r: r.row_id)
    assert reader.sequence_index.sequences == tuple((r.row_id, CHOICE) for r in ordered)
    assert set(reader.sequence_index.slot_kinds) == {SLOT_CHOICE}
    # No ROW lost everything, so row coverage is complete ...
    assert isinstance(reader.coverage, Ran) and reader.coverage.is_complete_coverage
    # ... and the slot refusal is counted on its own, by (row, slot), with its reason.
    ex = reader.sequence_index.excluded
    assert [(e.row_id, e.slot_name, e.scope) for e in ex] == [
        (r.row_id, SPAN, "slot") for r in ordered
    ]
    assert {e.refusal for e in ex} == {"UnencodableGold"}
    assert all("share one candidate" in e.detail for e in ex)
    slot_cov = reader.slot_coverage
    assert isinstance(slot_cov, Ran) and not slot_cov.passed
    assert (slot_cov.n, slot_cov.n_total) == (len(rows), 2 * len(rows))


def test_the_default_still_refuses_the_whole_write(tmp_path: Path) -> None:
    with pytest.raises(UnencodableGold, match="share one candidate"):
        _write(tmp_path, tmp_path / "shards")


def test_the_index_file_is_the_documented_schema(tmp_path: Path) -> None:
    out = tmp_path / "shards"
    header, rows, _ = _write(tmp_path, out, allow_unencodable=True)
    raw = json.loads((out / SEQUENCE_INDEX_NAME).read_text(encoding="utf-8"))
    assert raw["format"] == SEQUENCE_INDEX_FORMAT
    assert raw["rows_in"] == len(rows)
    assert set(raw["sequences"][0]) == {"row_id", "slot_name", "slot_kind"}
    assert set(raw["excluded"][0]) == {"row_id", "slot_name", "scope", "refusal", "detail"}
    offered = {(r.row_id, s.name) for r in rows for s in r.request.slots}
    listed = [(s["row_id"], s["slot_name"]) for s in raw["sequences"]] + [
        (e["row_id"], e["slot_name"]) for e in raw["excluded"]
    ]
    assert sorted(listed) == sorted(offered), "every offered (row, slot) in exactly one list"
    assert header.sequence_index_hash
    assert not list(out.glob(".*.tmp")), "the atomic write left its temporary behind"


def test_an_edited_index_is_refused_by_the_header_pin(tmp_path: Path) -> None:
    out = tmp_path / "shards"
    _, _, config = _write(tmp_path, out, allow_unencodable=True)
    path = out / SEQUENCE_INDEX_NAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sequences"][0], raw["sequences"][1] = raw["sequences"][1], raw["sequences"][0]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ShardContractViolation, match="hashes to"):
        ShardReader(out, config=config, repo_root=tmp_path)


def test_the_index_hash_is_covered_by_the_shard_hash(tmp_path: Path) -> None:
    out = tmp_path / "shards"
    _write(tmp_path, out, allow_unencodable=True)
    raw = json.loads((out / HEADER_NAME).read_text(encoding="utf-8"))
    raw["sequence_index_hash"] = "0" * 64
    with pytest.raises(ShardContractViolation, match="shard_hash on disk"):
        ShardHeader.from_json(raw)


def test_a_set_without_an_index_reports_slot_coverage_not_run(tmp_path: Path) -> None:
    """A header written before the field existed: no index, and not a pass."""
    out = tmp_path / "shards"
    _, _, config = _write(tmp_path, out, allow_unencodable=True)
    raw = json.loads((out / HEADER_NAME).read_text(encoding="utf-8"))
    raw.pop("sequence_index_hash")
    header = ShardHeader.from_json({k: v for k, v in raw.items() if k != "shard_hash"})
    (out / HEADER_NAME).write_text(json.dumps(header.to_json()), encoding="utf-8")
    (out / SEQUENCE_INDEX_NAME).unlink()
    reader = ShardReader(out, config=config, repo_root=tmp_path)
    assert reader.sequence_index is None
    assert isinstance(reader.slot_coverage, NotRun)
    assert json.loads((out / COVERAGE_NAME).read_text(encoding="utf-8"))["state"] == "ran"


# -- the replay guard ----------------------------------------------------------------------


def _as_replay(rows: list[DataRow]) -> list[DataRow]:
    # metadata is outside row_content_hash, so these still match the manifest.
    return [dataclasses.replace(r, metadata={**r.metadata, REPLAY_ROLE_KEY: REPLAY_ONLY})
            for r in rows]


def _write_rows(tmp_path: Path, rows_of, **kw: object) -> tuple[ShardHeader, DataConfig]:
    manifest, rows, config = _train(tmp_path)
    header = write_shards(
        manifest, rows_of(rows), out_dir=tmp_path / "shards", remap=_remap(),
        tokenize=_tokenize, token_offsets=_offsets, config=config, repo_root=tmp_path,
        allow_unencodable=True, **kw,  # type: ignore[arg-type]
    )
    return header, config


def test_a_gold_shard_set_refuses_a_replay_only_row(tmp_path: Path) -> None:
    """A replay row in a gold set would be cross-entropy-trained to the gold it exists
    not to read."""
    def one_replay(rows: list[DataRow]) -> list[DataRow]:
        return [*_as_replay(rows[:1]), *rows[1:]]

    with pytest.raises(ShardContractViolation, match="replay-only row"):
        _write_rows(tmp_path, one_replay)


def test_a_replay_shard_set_refuses_a_gold_row(tmp_path: Path) -> None:
    def one_gold(rows: list[DataRow]) -> list[DataRow]:
        return [*rows[:1], *_as_replay(rows[1:])]

    with pytest.raises(ShardContractViolation, match=r"gold \(not replay-only\)"):
        _write_rows(tmp_path, one_gold, replay=True)


def test_a_replay_set_of_replay_rows_is_written_and_says_so(tmp_path: Path) -> None:
    _, config = _write_rows(tmp_path, _as_replay, replay=True)
    reader = ShardReader(tmp_path / "shards", config=config, repo_root=tmp_path)
    assert reader.sequence_index is not None
    assert reader.sequence_index.role == REPLAY_ONLY
    _, config = _write_rows(tmp_path / "gold", lambda rows: rows)
    gold = ShardReader(tmp_path / "gold" / "shards", config=config, repo_root=tmp_path / "gold")
    assert gold.sequence_index is not None and gold.sequence_index.role == GOLD_ROLE
