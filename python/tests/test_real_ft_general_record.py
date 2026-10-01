"""``tools/real_ft_run.py`` relabelling a ``--general-record`` (phase-4) shard set.

GAP-REALFT-CANNOT-RELABEL-A-GENERAL-RECORD-SET: ``ft_splits`` rebuilt labels from the base
sources and ``--defect-class`` only, so a set built by ``tools/real_tokenizer_pipeline.py
--general-record [--replay-shards]`` could not be relabelled, and ``corpus_facts`` refused it
up front. It also ran ``build_mixture`` at the library's 250,000-row consistency bound where
the pipeline runs at ``PIPELINE_MAX_CONSISTENCY_ROWS`` (1,000,000): identical below 250k
rows, different above it.

The central test drives the pipeline's own ``run`` over a tiny general record up to the point
its manifests are written, then checks ``ft_splits`` rebuilds exactly those splits -- the
replay-only rows out of the gold train split included. No network: the record and its caches
are written here.

Torch-gated like ``test_real_ft_pieces.py``: the tool raises at import without torch.
"""

from __future__ import annotations

import hashlib
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

from qd_data.config import DataConfig  # noqa: E402
from qd_data.general import REPLAY_ONLY, REPLAY_ROLE_KEY  # noqa: E402

#: A pinned revision in this repository's history, as the other rebuild tests use.
PINNED = "0632f693d3b765b726499e7b4bf19c67959b75cb"

MMLU_ROWS = [
    {"question": f"What follows {i} in the sequence?", "subject": f"s{i % 7}",
     "choices": [str(i + 1), str(i + 2), str(i + 3), str(i + 4)], "answer": 0}
    for i in range(300)
]
SQUAD_ROWS = [
    {"id": "q1", "title": "T", "context": "One line.\nAnother names X.",
     "question": "Which names X?", "answers": {"text": ["X"], "answer_start": [24]}},
    {"id": "q2", "title": "T", "context": "One line.\nAnother names X.",
     "question": "Which names Y?", "answers": {"text": [], "answer_start": []}},
]


def _file(root: Path, rel: str, rows: list[dict[str, Any]]) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _entry(dataset: str, path: Path, rows: int) -> dict[str, Any]:
    return {"dataset": dataset, "jsonl": str(path), "rows": rows,
            "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _record(root: Path) -> Path:
    """A fetch record over MMLU (the replay family) and SQuAD, caches beside it."""
    mmlu = _file(root, "cais__mmlu/rev/test.jsonl", MMLU_ROWS)
    squad = _file(root, "rajpurkar__squad_v2/rev/train.jsonl", SQUAD_ROWS)
    path = root / "fetch-record.json"
    path.write_text(json.dumps([
        _entry("cais/mmlu", mmlu, len(MMLU_ROWS)),
        _entry("rajpurkar/squad_v2", squad, len(SQUAD_ROWS)),
    ]), encoding="utf-8")
    return path


class _StubTokenizer:
    """Enough of ``RealTokenizer`` for ``run`` to write its manifests; nothing is tokenized."""

    class _Inner:
        vocab_size = 1
        is_fast = True

        def __len__(self) -> int:
            return 1

    tok = _Inner()


class _ManifestsWritten(Exception):
    """Raised in place of stage 3: every manifest ``run`` writes is on disk by then."""


def _manifest_ids(path: Path) -> list[str]:
    return sorted(e["row_id"] for e in json.loads(path.read_text(encoding="utf-8"))["entries"])


@pytest.mark.usefixtures("qd_prep")
def test_the_rebuild_is_the_pipelines_own_split_general_rows_and_replay_slice_included(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path / "cache")
    out = tmp_path / "out"
    rev = resolve_rev(REPO, PINNED)
    monkeypatch.setattr(pipeline.RealTokenizer, "load", lambda **_: _StubTokenizer())

    def stop(*_a: object, **_k: object) -> None:
        raise _ManifestsWritten

    monkeypatch.setattr(pipeline, "census", stop)
    with pytest.raises(_ManifestsWritten):
        pipeline.run(out=out, max_pairs=3, blank_line_runs=False, rev=rev,
                     general_record=record, replay_shards=True)

    config = DataConfig()
    splits = rft.ft_splits(commitpackft=None, max_pairs=3, rev=rev, config=config,
                           general_record=record, replay_partition=True)
    for name, rel in (("train", "data/pool/train.json"), ("val", "data/pool/val.json"),
                      ("heldout", "data/heldout/heldout.json")):
        assert sorted(r.row_id for r in splits[name]) == _manifest_ids(out / rel), name
    replay_ids = _manifest_ids(out / rft.REPLAY_MANIFEST)
    assert replay_ids, "a 15% draw over 300 MMLU rows drew nothing; the fixture is too small"
    train_ids = {r.row_id for r in splits["train"]}
    assert train_ids.isdisjoint(replay_ids)
    assert all(REPLAY_ROLE_KEY not in r.metadata for r in splits["train"])
    # The real SQuAD rows replaced the repository-prose stand-in, on both sides.
    all_rows = [r for rows in splits.values() for r in rows]
    assert any(r.source_id == "rajpurkar/squad_v2" for r in all_rows)
    assert any(r.source_id == "cais/mmlu" for r in all_rows)

    # The partition is load-bearing: without it the replay-only rows are gold-trained.
    unpartitioned = rft.ft_splits(commitpackft=None, max_pairs=3, rev=rev, config=config,
                                  general_record=record)
    assert set(replay_ids) <= {r.row_id for r in unpartitioned["train"]}
    assert unpartitioned["val"] == splits["val"]
    # And without the record the rebuild is a different corpus altogether.
    bare = rft.ft_splits(commitpackft=None, max_pairs=3, rev=rev, config=config)
    assert not any(r.source_id == "cais/mmlu" for rows in bare.values() for r in rows)


def test_the_mixture_is_built_at_the_pipelines_consistency_bound_with_the_clinc_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The library default (250,000) and the pipeline's bound build different row sets
    above 250k rows; the CLINC domain map decides whether two-stage CLINC rows exist."""
    import qd_data.mixture as mixture_module

    seen: list[dict[str, object]] = []

    class Stop(Exception):
        pass

    def spy(raw: object, **kw: object) -> None:
        seen.append(kw)
        raise Stop

    sentinel_map = object()
    load = pipeline.GeneralLoad(raw={}, clinc_domain_map=sentinel_map, files={},  # type: ignore[arg-type]
                                capped=(), record_sha256="0" * 64, refused_reads={})
    got: list[tuple[Path, int]] = []

    def general_rows(record: Path, *, max_rows_per_file: int) -> pipeline.GeneralLoad:
        got.append((record, max_rows_per_file))
        return load

    monkeypatch.setattr(mixture_module, "build_mixture", spy)
    monkeypatch.setattr(pipeline, "general_rows", general_rows)
    record = tmp_path / "r.json"
    for max_rows, expect in ((None, pipeline.DEFAULT_GENERAL_MAX_ROWS), (17, 17)):
        with pytest.raises(Stop):
            rft.ft_splits(commitpackft=None, max_pairs=1, rev="r", config=DataConfig(),
                          repo_history=False, general_record=record,
                          general_max_rows=max_rows)
        assert got[-1] == (record, expect)
    assert [kw["max_consistency_rows"] for kw in seen] == [
        pipeline.PIPELINE_MAX_CONSISTENCY_ROWS
    ] * 2
    assert all(kw["clinc_domain_map"] is sentinel_map for kw in seen)
    # Without a record: the same bound, and no map.
    with pytest.raises(Stop):
        rft.ft_splits(commitpackft=None, max_pairs=1, rev="r", config=DataConfig(),
                      repo_history=False)
    assert seen[-1]["max_consistency_rows"] == pipeline.PIPELINE_MAX_CONSISTENCY_ROWS
    assert seen[-1]["clinc_domain_map"] is None


def test_a_consistency_pass_that_could_not_run_is_refused_not_relabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline never writes a set from such a mixture; a rebuild that meets one has
    diverged from every set, so it stops rather than pairing labels to it."""
    monkeypatch.setattr(pipeline, "PIPELINE_MAX_CONSISTENCY_ROWS", 0)
    with pytest.raises(SystemExit, match="could not run its consistency pass"):
        rft.ft_splits(commitpackft=None, max_pairs=2, rev=resolve_rev(REPO, PINNED),
                      config=DataConfig())


@pytest.mark.parametrize("kw", [{"general_max_rows": 5}, {"replay_partition": True}])
def test_general_options_without_a_record_are_refused(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="without general_record read nothing"):
        rft.ft_splits(commitpackft=None, max_pairs=1, rev="x", config=DataConfig(), **kw)


# --- corpus_facts: the general families and the replay slice -----------------------------------


def _manifest(out: Path, n_input: dict[str, int], *, replay: bool = False) -> None:
    path = out / rft.TRAIN_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "data_snapshot_hash": "d" * 64, "status": {"state": "ran", "passed": True},
        "mixture": {"n_input": n_input},
    }), encoding="utf-8")
    if replay:
        (out / rft.REPLAY_MANIFEST).write_text("{}", encoding="utf-8")


def _facts(out: Path, **kw: Any) -> rft.CorpusFacts:
    kw.setdefault("repo_history", True)
    kw.setdefault("commitpackft", None)
    return rft.corpus_facts(out, data_snapshot_hash="d" * 64, **kw)


GENERAL = frozenset({"cais/mmlu", "rajpurkar/squad_v2"})


def test_a_general_record_set_is_accepted_with_the_record_that_built_it(tmp_path: Path) -> None:
    _manifest(tmp_path, {"cais/mmlu": 10, "rajpurkar/squad_v2": 4, "bigcode/commitpackft": 5},
              replay=True)
    facts = _facts(tmp_path, general_datasets=GENERAL, replay_partition=True)
    # SQuAD came from the record, not from this repository's Markdown.
    assert facts.history_rows == {"bigcode/commitpackft": 5}
    # Without the record it is still refused, naming the flag.
    with pytest.raises(SystemExit, match="built with --general-record"):
        _facts(tmp_path, replay_partition=True)


def test_a_record_that_does_not_match_the_set_is_refused(tmp_path: Path) -> None:
    _manifest(tmp_path, {"cais/mmlu": 10, "tau/commonsense_qa": 3, "bigcode/commitpackft": 5})
    with pytest.raises(SystemExit, match="does not name them"):
        _facts(tmp_path, general_datasets=GENERAL)
    plain = tmp_path / "plain"
    _manifest(plain, {"bigcode/commitpackft": 5, "rajpurkar/squad_v2": 4})
    with pytest.raises(SystemExit, match="built without it"):
        _facts(plain, general_datasets=frozenset({"cais/mmlu"}))
    # The history SQuAD rows must not pass for the record's: refused up front, before the
    # rebuild, not minutes later by pair_labels.
    with pytest.raises(SystemExit, match="built without it"):
        _facts(plain, general_datasets=GENERAL)


def test_the_replay_partition_must_agree_with_the_replay_manifest(tmp_path: Path) -> None:
    partitioned = tmp_path / "partitioned"
    _manifest(partitioned, {"cais/mmlu": 10, "bigcode/commitpackft": 5}, replay=True)
    with pytest.raises(SystemExit, match="Pass --replay-partition"):
        _facts(partitioned, general_datasets=GENERAL)
    whole = tmp_path / "whole"
    _manifest(whole, {"cais/mmlu": 10, "bigcode/commitpackft": 5})
    with pytest.raises(SystemExit, match="built without the pipeline's --replay-shards"):
        _facts(whole, general_datasets=GENERAL, replay_partition=True)
    _facts(whole, general_datasets=GENERAL)


def test_record_supplied_squad_with_the_download_leaves_no_history_to_check(
    tmp_path: Path,
) -> None:
    """--commitpackft plus a record that supplies SQuAD: the rebuild reads no history row
    either way, so --no-repo-history decides nothing and is not refused."""
    _manifest(tmp_path, {"cais/mmlu": 10, "rajpurkar/squad_v2": 4, "bigcode/commitpackft": 5})
    assert _facts(tmp_path, commitpackft=tmp_path, general_datasets=GENERAL).history_rows == {}


def test_the_datasets_a_record_names_are_read_through_the_pipelines_parser(
    tmp_path: Path,
) -> None:
    assert rft.general_record_datasets(None) is None
    assert rft.general_record_datasets(_record(tmp_path)) == GENERAL
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"dataset": "someone/else", "jsonl": "x"}]), encoding="utf-8")
    with pytest.raises(SystemExit, match="is not one of"):
        rft.general_record_datasets(bad)


# --- the replay attestation and the tools that call ft_splits ---------------------------------


def test_the_attestation_names_the_general_record_only_when_one_was_used(tmp_path: Path) -> None:
    kw: dict[str, Any] = {"rev": "r", "max_pairs": 400, "commitpackft": None,
                          "defect_class": None, "defect_max_rows": None}
    assert rft.replay_corpus_identity(**kw) == kw
    record = _record(tmp_path)
    sha = hashlib.sha256(record.read_bytes()).hexdigest()
    assert rft.replay_corpus_identity(**kw, general_record=record) == {
        **kw, "general_record_sha256": sha,
        "general_max_rows": pipeline.DEFAULT_GENERAL_MAX_ROWS,
    }
    assert rft.replay_corpus_identity(**kw, general_record=record, general_max_rows=9)[
        "general_max_rows"] == 9


def test_replay_decontam_rebuilds_with_the_general_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its heldout and val targets include the general families' rows only through it."""
    import replay_decontam

    class Stop(Exception):
        pass

    seen: list[dict[str, object]] = []

    def spy(**kw: object) -> None:
        seen.append(kw)
        raise Stop

    monkeypatch.setattr(rft, "check_defect_source", lambda out, *, defect_class: None)
    monkeypatch.setattr(rft, "ft_splits", spy)
    record = tmp_path / "record.json"
    with pytest.raises(Stop):
        replay_decontam.main([
            "--out", str(tmp_path), "--replay-shards", str(tmp_path), "--tokenizer-json",
            str(tmp_path / "t.json"), "--attestation-out", str(tmp_path / "a.json"),
            "--general-record", str(record), "--general-max-rows", "11",
        ])
    assert [(kw["general_record"], kw["general_max_rows"]) for kw in seen] == [(record, 11)]


class _Stop(Exception):
    pass


def test_main_forwards_the_record_and_pairs_a_general_set_by_its_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    record = _record(tmp_path / "cache")
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
        lambda *a, **k: SimpleNamespace(header=SimpleNamespace(data_snapshot_hash="d" * 64)),
    )
    monkeypatch.setattr(rft, "check_defect_source", lambda out, *, defect_class: None)
    monkeypatch.setattr(rft, "corpus_facts", lambda out, **kw: facts.append(kw))
    with pytest.raises(_Stop):
        rft.main(["--out", str(tmp_path), "--max-pairs", "7", "--rev", rev,
                  "--general-record", str(record), "--general-max-rows", "50",
                  "--replay-partition"])
    assert [(c["general_record"], c["general_max_rows"], c["replay_partition"])
            for c in calls] == [(record, 50, True)]
    assert [(f["general_datasets"], f["replay_partition"]) for f in facts] == [(GENERAL, True)]
    assert paired == [True]


@pytest.mark.parametrize("flags", [["--general-max-rows", "5"], ["--replay-partition"]])
def test_main_refuses_general_options_without_a_record(tmp_path: Path, flags: list[str]) -> None:
    with pytest.raises(SystemExit, match="without --general-record read nothing"):
        rft.main(["--out", str(tmp_path), "--rev", "a" * 40, *flags])


@pytest.mark.usefixtures("qd_prep")
def test_replay_rows_are_marked_and_never_in_the_gold_train_split(tmp_path: Path) -> None:
    """The mirror of the pipeline's own invariant, on the rebuild: every replay-only row
    carries its role, and none of them is in the train split ``main`` gold-trains on."""
    rev = resolve_rev(REPO, PINNED)
    record = _record(tmp_path)
    train, _val = rft.ft_split_rows(commitpackft=None, max_pairs=3, rev=rev,
                                    config=DataConfig(), general_record=record,
                                    replay_partition=True, repo_history=False)
    assert train
    assert not any(r.metadata.get(REPLAY_ROLE_KEY) == REPLAY_ONLY for r in train)
