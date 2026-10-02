"""``tools/real_tokenizer_pipeline.py --report-only-slice``: the Fable G5(ii) slice's doors.

The slice is composed long-context val rows plus diagnostic rows whose needle is a train file
v4 trained on as a filler. It is scored for a report and never gated, so what is pinned here
is every way it could leak into training or a gate population:

* the corpus's own marker (manifest ``mode: report_only``) and the pipeline's flag must
  agree: a slice without the marker is not read as one, and a corpus with it is never a
  training corpus (condition 2);
* the one relaxation of ``qd_data.defect_class._composed_violation`` is a ``seen_filler``
  row's needle, which must be train; every other check is the canonical one (fallback 9);
* the slice's split is slice rows in val and its base carrier in train, and none of its rows
  is a row of the gate val set or claims the needle suite's identity;
* the gate set a slice is checked against must itself be a gate population.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402
from test_defect_class import (  # noqa: E402
    _all,
    _base_files,
    _composed_row,
    _example,
    _repos_in,
    _write_composed,
)
from test_shards import _report_only, _snapshot, _write  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID  # noqa: E402
from qd_train.artifacts import ShardContractViolation  # noqa: E402

TRAIN_CORPUS_SHA = "a" * 64


def _diag(
    files: list[dict[str, Any]], *, needle: int, half: str, index: int
) -> dict[str, Any]:
    """A diagnostic row as ``qd-mutate compose --diag-rows`` writes it: anchored at its
    first file, a val filler, whatever its needle is."""
    row = _composed_row(files, needle=needle, index=index, split_name="val")
    anchor = files[0]
    row.update({
        "id": f"compose:diag:{index:06d}", "pool_id": anchor["pool_id"],
        "repo": anchor["repo"], "path": anchor["repo"], "diag_half": half,
        "function": {**row["function"], "repo": anchor["repo"], "path": anchor["repo"]},
    })
    return row


def _files() -> dict[str, list[dict[str, Any]]]:
    """``_base_files`` with a third val file, so a row can hold two val fillers."""
    files = _base_files()
    extra = _repos_in("val", 3)[2]
    files["val"].append(_example(99, repo=extra, cls="clean"))
    return files


def _good(files: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    v, t = files["val"], files["train"]
    return [
        _composed_row([v[0], v[1], v[2]], needle=1, index=0, split_name="val"),
        _diag([v[0], t[0], v[2]], needle=1, half=pipeline.DIAG_SEEN_FILLER, index=1),
        _diag([v[1], v[2], v[0]], needle=1, half=pipeline.DIAG_UNSEEN, index=2),
    ]


def _slice(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    files: dict[str, list[dict[str, Any]]],
    *,
    mode: str | None = pipeline.REPORT_ONLY_MODE,
) -> tuple[Path, Path]:
    corpus, download = _write_composed(tmp_path, _all(files), rows)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    if mode is not None:
        manifest["mode"] = mode
    manifest["diag"] = {"train_corpus": {"examples_sha256": TRAIN_CORPUS_SHA}}
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return corpus, download


def _load(tmp_path: Path, corpus: Path, download: Path) -> pipeline.ReportOnlySlice:
    return pipeline.load_report_only_slice(
        corpus, download_root=download, config=DataConfig(), repo_root=tmp_path,
        base_max_rows=100,
    )


def test_a_slice_reads_its_one_relaxation_and_its_base_carrier_is_train_only(
    tmp_path: Path,
) -> None:
    files = _files()
    corpus, download = _slice(tmp_path, _good(files), files)
    got = _load(tmp_path, corpus, download)
    assert got.by_half == {"composed_val": 1, "seen_filler": 1, "unseen": 1}
    assert [r.example_id for r in got.rows] == [
        "compose:val:000000", "compose:diag:000001", "compose:diag:000002",
    ]
    assert {r.repo for r in got.base} == {e["repo"] for e in files["train"]}
    assert got.base_sampled == len(_all(files))
    assert got.diag_train_corpus_sha256 == TRAIN_CORPUS_SHA
    assert len(got.manifest_sha256) == 64


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        # The relaxation is the seen_filler needle, and it must be train.
        (lambda f: _diag([f["val"][0], f["val"][1], f["val"][2]], needle=1,
                         half="seen_filler", index=5), "seen_filler_needle_not_train"),
        (lambda f: _diag([f["val"][0], f["heldout"][0], f["val"][2]], needle=1,
                         half="seen_filler", index=5), "seen_filler_needle_not_train"),
        # ... and only the needle: a seen_filler row's fillers are the canonical check's.
        (lambda f: _diag([f["val"][0], f["train"][0], f["train"][1]], needle=1,
                         half="seen_filler", index=5), "constituents_span_two_splits"),
        # No other row is relaxed: a plain or unseen row with a train file is refused.
        (lambda f: _composed_row([f["val"][0], f["train"][0], f["val"][2]], needle=0,
                                 index=5, split_name="val"), "constituents_span_two_splits"),
        (lambda f: _diag([f["val"][0], f["train"][0], f["val"][2]], needle=1,
                         half="unseen", index=5), "constituents_span_two_splits"),
        (lambda f: _diag([f["val"][0], f["train"][0], f["val"][2]], needle=1,
                         half="seen", index=5), "unknown_diag_half"),
        # Every slice row is val.
        (lambda f: _composed_row([f["train"][0], f["train"][1], f["train"][2]], needle=1,
                                 index=5), "row_does_not_split_val"),
    ],
)
def test_a_slice_row_outside_the_one_relaxation_refuses_the_slice(
    tmp_path: Path, make: Any, reason: str
) -> None:
    files = _files()
    corpus, download = _slice(tmp_path, [*_good(files), make(files)], files)
    with pytest.raises(SystemExit, match=reason):
        _load(tmp_path, corpus, download)


def test_a_composed_corpus_without_the_marker_is_not_read_as_a_slice(tmp_path: Path) -> None:
    files = _files()
    corpus, download = _slice(tmp_path, _good(files), files, mode=None)
    with pytest.raises(SystemExit, match="must agree"):
        _load(tmp_path, corpus, download)


def test_a_report_only_corpus_is_refused_as_a_training_corpus(tmp_path: Path) -> None:
    files = _files()
    corpus, _ = _slice(tmp_path, _good(files), files)
    with pytest.raises(SystemExit, match="never a training corpus"):
        pipeline.run(
            out=tmp_path / "out", max_pairs=1, blank_line_runs=False, rev="HEAD",
            defect_class=corpus, repo_history=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "needs"),
        ({"gate_set": Path("g"), "defect_class": Path("d")}, "reads no other source"),
        ({"gate_set": Path("g"), "repo_history": True}, "reads no other source"),
    ],
)
def test_the_slice_flag_refuses_any_other_source_and_any_missing_need(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    base: dict[str, Any] = {
        "out": tmp_path / "out", "max_pairs": 1, "blank_line_runs": False, "rev": "HEAD",
        "report_only_slice": tmp_path / "slice", "repo_history": False,
    }
    with pytest.raises(SystemExit, match=match):
        pipeline.run(**{**base, **kwargs})
    with pytest.raises(SystemExit, match="read only by --report-only-slice"):
        pipeline.run(**{**base, "report_only_slice": None, "gate_set": tmp_path})


def _slice_ids(got: pipeline.ReportOnlySlice) -> set[str]:
    return {f"qdm:{DEFECT_FAMILY_ID}:{r.example_id}" for r in got.rows}


def test_a_slice_split_that_is_not_slice_in_val_and_base_in_train_is_refused(
    tmp_path: Path,
) -> None:
    files = _files()
    corpus, download = _slice(tmp_path, _good(files), files)
    got = _load(tmp_path, corpus, download)
    ids = _slice_ids(got)
    ok = {"train": {"qdm:x:base"}, "val": set(ids), "heldout": set()}
    pipeline.check_slice_split(got, by_split=ok, gate_ids=frozenset({"qdm:x:gate"}))
    for bad in (
        {**ok, "val": {*ids, "qdm:x:stray"}},
        {**ok, "heldout": {"qdm:x:held"}},
    ):
        with pytest.raises(SystemExit, match="not slice rows in val"):
            pipeline.check_slice_split(got, by_split=bad, gate_ids=frozenset())
    with pytest.raises(SystemExit, match="rows of the gate val set"):
        pipeline.check_slice_split(got, by_split=ok, gate_ids=frozenset({sorted(ids)[0]}))


def test_the_gate_set_is_read_through_its_reader_and_must_be_a_gate_population(
    tmp_path: Path,
) -> None:
    snap = _snapshot(tmp_path / "gate")
    train = _write(snap, "train", snap.root / "shards" / "train")
    _write(snap, "val", snap.root / "shards" / "val")
    ids, remap_hash, val_hash = pipeline.gate_val_row_ids(snap.root, config=snap.config)
    assert ids == {r.row_id for r in snap.rows["val"]}
    assert remap_hash == train.remap_hash and len(val_hash) == 64

    other = _snapshot(tmp_path / "other")
    _write(other, "train", other.root / "shards" / "train")
    _report_only(other, other.root / "shards" / "val")
    with pytest.raises(ShardContractViolation, match="not a gate population"):
        pipeline.gate_val_row_ids(other.root, config=other.config)
