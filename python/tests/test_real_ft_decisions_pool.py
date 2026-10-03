"""The tools that rebuild a shard set's split, on a set built with ``--decisions-pool``.

``tools/real_tokenizer_pipeline.py --decisions-pool DIR`` reads the general-decision pool
(``qd-prep decisions``' examples and manifest) into the mixture, and names it in
``corpus_identity``. ``tools/real_ft_run.py`` (train and every scoring mode),
``tools/containment_scan.py`` and ``tools/ft_linear_control.py`` all rebuild that split
through ``real_ft_run.ft_split_report``. Before this change none of them could read the
pool. On a pool set, then, the rebuild was a different corpus from the one the shards were
written from: the pool's rows were missing from train and val. ``pair_labels`` would refuse
that only after minutes of rebuilding, and the containment scan would attest a corpus the
shards never held. These tests pin the pool's path through every rebuild:

* ``ft_splits(decisions_pool=...)`` rebuilds exactly the split the pipeline wrote, the pool's
  rows included, and without the pool those rows are absent;
* the corpus identity names the pool exactly when it is given;
* ``corpus_facts`` refuses, before the rebuild, a set a pool source fed when no pool is given,
  and a pool given for a set that no pool source fed;
* ``main`` and ``containment_scan.main`` forward the flag (``ft_linear_control``'s test is in
  ``test_ft_linear_control.py``).

Torch-gated like ``test_real_ft_general_record.py``: the tool raises at import without torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
import real_tokenizer_pipeline as pipeline  # noqa: E402
from repo_git import resolve_rev  # noqa: E402
from test_decisions_pool import ARC, OPEN_JEV, _line, _write_pool  # noqa: E402
from test_real_ft_general_record import (  # noqa: E402
    PINNED,
    _facts,
    _manifest,
    _manifest_ids,
    _ManifestsWritten,
    _StubTokenizer,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_data.sources import DECISION_FAMILIES  # noqa: E402

#: Every source the general-decision pool supplies.
POOL_SOURCES = sorted({source_id for _, source_id, _ in DECISION_FAMILIES})


def _pool(root: Path) -> Path:
    """A pool of Open-Jev policy rows, train and val, in groups of their own."""
    lines = [
        _line(id=f"openjev:g{i}/0/choice", group_key=f"g{i}",
              split="val" if i % 4 == 0 else "train",
              # Distinct vocabulary per row, so no two rows are near-duplicates to dedupe.
              context=" ".join(f"term{i}x{j}" for j in range(16)),
              options=[f"K{i}", f"P{i}", f"Q{i}"], gold_option=f"P{i}")
        for i in range(12)
    ]
    return _write_pool(root, lines)


@pytest.mark.usefixtures("qd_prep")
def test_the_rebuild_is_the_pipelines_own_split_with_the_pool_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path / "pool")
    out = tmp_path / "out"
    rev = resolve_rev(REPO, PINNED)
    monkeypatch.setattr(pipeline.RealTokenizer, "load", lambda **_: _StubTokenizer())

    def stop(*_a: object, **_k: object) -> None:
        raise _ManifestsWritten

    monkeypatch.setattr(pipeline, "census", stop)
    with pytest.raises(_ManifestsWritten):
        pipeline.run(out=out, max_pairs=3, blank_line_runs=False, rev=rev, decisions_pool=pool)

    config = DataConfig()
    splits = rft.ft_splits(commitpackft=None, max_pairs=3, rev=rev, config=config,
                           decisions_pool=pool)
    for name, rel in (("train", "data/pool/train.json"), ("val", "data/pool/val.json"),
                      ("heldout", "data/heldout/heldout.json")):
        assert sorted(r.row_id for r in splits[name]) == _manifest_ids(out / rel), name
    pooled = {name: [r for r in rows if r.source_id == OPEN_JEV] for name, rows in splits.items()}
    assert pooled["train"] and pooled["val"], "the fixture's pool fed neither split"
    assert pooled["heldout"] == [], "a pool row in held-out: the pool has no held-out split"
    # Without the pool the rebuild is a different corpus: every pool row is gone.
    bare = rft.ft_splits(commitpackft=None, max_pairs=3, rev=rev, config=config)
    assert not any(r.source_id == OPEN_JEV for rows in bare.values() for r in rows)


@pytest.mark.usefixtures("qd_prep")
def test_a_pool_with_an_opt_in_source_rebuilds_with_the_pipelines_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARC is an opt-in source: the pipeline admits it for a pool build (``pool_data_config``),
    so the rebuild must too, or ``build_mixture`` refuses ARC's rows at load and the set
    cannot be rebuilt at all."""
    lines = [
        _line(id=f"arc:easy:Mercury_{i}", source_id=ARC, family_id="arc.science",
              stratum="arc.science/easy/choice", group_key=f"Mercury_{i}",
              licence="cc-by-sa-4.0", split="val" if i % 4 == 0 else "train",
              context=" ".join(f"arc{i}x{j}" for j in range(16)),
              options=[f"A{i}", f"B{i}", f"C{i}", f"D{i}"], gold_option=f"B{i}")
        for i in range(12)
    ]
    pool = _write_pool(tmp_path / "pool", lines)
    out = tmp_path / "out"
    rev = resolve_rev(REPO, PINNED)
    monkeypatch.setattr(pipeline.RealTokenizer, "load", lambda **_: _StubTokenizer())

    def stop(*_a: object, **_k: object) -> None:
        raise _ManifestsWritten

    monkeypatch.setattr(pipeline, "census", stop)
    with pytest.raises(_ManifestsWritten):
        pipeline.run(out=out, max_pairs=3, blank_line_runs=False, rev=rev, decisions_pool=pool)

    splits = rft.ft_splits(commitpackft=None, max_pairs=3, rev=rev, config=DataConfig(),
                           decisions_pool=pool)
    for name, rel in (("train", "data/pool/train.json"), ("val", "data/pool/val.json"),
                      ("heldout", "data/heldout/heldout.json")):
        assert sorted(r.row_id for r in splits[name]) == _manifest_ids(out / rel), name
    arc = {name: [r for r in rows if r.source_id == ARC] for name, rows in splits.items()}
    assert arc["train"] and arc["val"] and arc["heldout"] == []


def test_the_corpus_identity_names_the_pool_only_when_given(tmp_path: Path) -> None:
    kw: dict[str, Any] = {"rev": "r", "max_pairs": 400, "commitpackft": None,
                          "defect_class": None, "defect_max_rows": None}
    assert rft.replay_corpus_identity(**kw) == kw
    pool = _pool(tmp_path / "pool")
    sha = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))["examples_sha256"]
    assert rft.replay_corpus_identity(**kw, decisions_pool=pool) == {
        **kw, "decisions_pool_examples_sha256": sha,
    }


def test_a_pool_given_to_a_run_that_already_reads_its_source_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline's clash refusal, on the rebuild: a source read twice is not a corpus."""
    pool = _pool(tmp_path / "pool")

    from types import SimpleNamespace

    monkeypatch.setattr(pipeline, "base_sources", lambda **_: SimpleNamespace(raw={OPEN_JEV: []}))
    with pytest.raises(SystemExit, match="already read by this run"):
        rft.ft_split_report(commitpackft=None, max_pairs=1, rev="a" * 40,
                            config=DataConfig(), decisions_pool=pool)


# --- corpus_facts: a pool set needs its pool, and only a pool set takes one ---------------------


def test_a_pool_set_is_refused_without_its_pool_and_accepted_with_it(tmp_path: Path) -> None:
    _manifest(tmp_path, {OPEN_JEV: 9, "bigcode/commitpackft": 5})
    with pytest.raises(SystemExit, match="built with --decisions-pool"):
        _facts(tmp_path)
    assert _facts(tmp_path, decisions_pool=True).history_rows == {"bigcode/commitpackft": 5}


def test_a_pool_given_for_a_set_no_pool_source_fed_is_refused(tmp_path: Path) -> None:
    _manifest(tmp_path, {"bigcode/commitpackft": 5})
    with pytest.raises(SystemExit, match="built without it"):
        _facts(tmp_path, decisions_pool=True)
    _facts(tmp_path)


def test_every_pool_source_is_one_corpus_facts_checks() -> None:
    assert sorted(rft.DECISION_POOL_SOURCES) == POOL_SOURCES


# --- the three entry points forward the flag ---------------------------------------------------


class _Stop(Exception):
    pass


def test_main_forwards_the_pool_and_pairs_a_pool_set_by_its_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    pool = _pool(tmp_path / "pool")
    calls: list[dict[str, object]] = []
    facts: list[dict[str, object]] = []
    paired: list[bool] = []

    def split_rows(**kw: object) -> tuple[list[object], list[object]]:
        calls.append(kw)
        return [], []

    def pair(reader: object, labels: object, *, require_index: bool) -> None:
        paired.append(require_index)
        raise _Stop

    rev = "a" * 40
    monkeypatch.setattr(rft, "ft_split_rows", split_rows)
    monkeypatch.setattr(rft, "_labels", lambda rows, *, config: ([], []))
    monkeypatch.setattr(rft, "pair_labels", pair)
    monkeypatch.setattr(rft, "resolve_rev", lambda repo, rev: rev)
    monkeypatch.setattr(
        rft, "ShardReader",
        lambda *a, **k: SimpleNamespace(header=SimpleNamespace(
            data_snapshot_hash="d" * 64, exclusions_sha256="",
        )),
    )
    monkeypatch.setattr(rft, "check_defect_source", lambda out, *, defect_class: None)
    monkeypatch.setattr(rft, "corpus_facts", lambda out, **kw: facts.append(kw))
    with pytest.raises(_Stop):
        rft.main(["--out", str(tmp_path), "--max-pairs", "7", "--rev", rev,
                  "--decisions-pool", str(pool)])
    assert [c["decisions_pool"] for c in calls] == [pool]
    assert [f["decisions_pool"] for f in facts] == [True]
    assert paired == [True]


def test_containment_scan_rebuilds_and_names_the_corpus_with_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import containment_scan as cs

    pool = _pool(tmp_path / "pool")
    seen: dict[str, object] = {}

    def report(**kw: object) -> object:
        seen["split"] = kw["decisions_pool"]
        raise _Stop

    monkeypatch.setattr(rft, "ft_split_report", report)
    monkeypatch.setattr(cs, "prep_binary", lambda: tmp_path / "qd-prep")
    monkeypatch.setattr(cs, "resolve_rev", lambda repo, rev: rev)
    with pytest.raises(_Stop):
        cs.main(["--out-dir", str(tmp_path / "scan"), "--rev", "a" * 40,
                 "--decisions-pool", str(pool)])
    assert seen == {"split": pool}
