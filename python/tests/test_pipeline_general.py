"""``tools/real_tokenizer_pipeline.py`` reading the general-family caches, and the replay slice.

* ``general_rows`` reads only what the fetch record vouches for: a file outside the cache
  root, with another sha256 or another row count stops the run; a split
  ``REFUSED_READS`` forbids is counted and never parsed.
* ``split_off_replay`` takes ``partition_replay``'s replay-only rows out of the gold train
  split into a report of their own, so each gets its own manifest and shard set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.general import MMLU_FAMILY, REPLAY_ONLY, REPLAY_ROLE_KEY  # noqa: E402
from qd_data.loaders import MmluRow  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.split import split  # noqa: E402

MMLU_ROWS = [
    {"question": f"What is {i} plus one?", "subject": f"s{i % 3}",
     "choices": [str(i + 1), str(i + 2), str(i + 3), str(i + 4)], "answer": 0}
    for i in range(5)
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


def _record(root: Path, entries: list[dict[str, Any]]) -> Path:
    path = root / "fetch-record.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _cache(root: Path) -> list[dict[str, Any]]:
    mmlu = _file(root, "cais__mmlu/rev/test.jsonl", MMLU_ROWS)
    squad = _file(root, "rajpurkar__squad_v2/rev/train.jsonl", SQUAD_ROWS)
    csqa_test = _file(root, "tau__commonsense_qa/rev/test.jsonl", [{"id": "c"}])
    return [
        _entry("cais/mmlu", mmlu, len(MMLU_ROWS)),
        _entry("rajpurkar/squad_v2", squad, len(SQUAD_ROWS)),
        _entry("tau/commonsense_qa", csqa_test, 1),
    ]


def test_every_file_the_record_vouches_for_is_read_and_a_refused_split_is_counted(
    tmp_path: Path,
) -> None:
    load = pipeline.general_rows(_record(tmp_path, _cache(tmp_path)))
    assert {k: len(v) for k, v in load.raw.items()} == {
        "cais/mmlu": 5, "rajpurkar/squad_v2": 2
    }
    assert all(r.upstream_split == "test" for r in load.raw["cais/mmlu"])  # type: ignore[union-attr]
    assert [Path(p).parent.parent.name for p in load.refused_reads] == ["tau__commonsense_qa"]
    assert load.clinc_domain_map is None and load.capped == ()


def test_every_clinc_row_carries_the_split_of_the_file_it_was_read_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GAP-PIPELINE-PARSE-CLINC-NO-SPLIT-NAME-2026-10-06: ``general_rows`` called
    ``parse_clinc`` without ``split_name``, so every CLINC row's ``upstream_split`` was None and
    a v6 build refused all of them as ``upstream_split_unstated``. MMLU and CommonsenseQA were
    already passed theirs; this pins CLINC to the same rule. The domain map is stubbed: it is
    checked against its own pinned sha256 elsewhere and is not what this test is about."""
    names = ["oos", "greeting"]
    entries = []
    for split_name in ("train", "validation", "test"):
        path = _file(tmp_path, f"clinc__clinc_oos/rev/{split_name}.jsonl",
                     [{"text": f"hello there {split_name}", "intent": 1},
                      {"text": f"what is the weather {split_name}", "intent": 0}])
        (path.parent / "intent_names.json").write_text(json.dumps(names), encoding="utf-8")
        entries.append(_entry("clinc/clinc_oos", path, 2) | {"n_intent_names": len(names)})
    monkeypatch.setattr(pipeline, "load_clinc_domains", lambda _path: "stub-domain-map")
    load = pipeline.general_rows(_record(tmp_path, entries))
    rows = load.raw["clinc/clinc_oos"]
    assert len(rows) == 6
    assert sorted(r.upstream_split for r in rows) == [  # type: ignore[union-attr]
        "test", "test", "train", "train", "validation", "validation"
    ]


def test_a_file_that_does_not_hash_to_the_record_stops_the_run(tmp_path: Path) -> None:
    entries = _cache(tmp_path)
    Path(entries[0]["jsonl"]).write_text(
        Path(entries[0]["jsonl"]).read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="not the approved download"):
        pipeline.general_rows(_record(tmp_path, entries))


def test_a_file_outside_the_cache_root_stops_the_run(tmp_path: Path) -> None:
    elsewhere = _file(tmp_path / "elsewhere", "m/test.jsonl", MMLU_ROWS)
    root = tmp_path / "cache"
    root.mkdir()
    with pytest.raises(SystemExit, match="outside the cache root"):
        pipeline.general_rows(_record(root, [_entry("cais/mmlu", elsewhere, 5)]))


def test_a_row_count_the_record_does_not_state_stops_the_run(tmp_path: Path) -> None:
    entries = _cache(tmp_path)
    entries[0]["rows"] = 6
    with pytest.raises(SystemExit, match="rows but the fetch record says 6"):
        pipeline.general_rows(_record(tmp_path, entries))


def test_the_replay_slice_leaves_the_gold_train_split_and_nothing_else_moves() -> None:
    config = DataConfig()
    rows = [
        MmluRow(subject=f"s{i % 7}", question=f"What follows {i}?",
                choices=(str(i), str(i + 1), str(i + 2), str(i + 3)), answer_index=0,
                upstream_split="test")
        for i in range(300)
    ]
    mixture = build_mixture({"cais/mmlu": rows}, config=config)
    report = split(dedupe(list(mixture.rows), config=config), config=config)
    gold, replay, part = pipeline.split_off_replay(report, seed=config.seed)

    before = {r.row_id for r in report.rows_by_split["train"]}
    gold_ids = {r.row_id for r in gold.rows_by_split["train"]}
    replay_rows = replay.rows_by_split["train"]
    assert replay_rows, "a 15% draw over 300 rows drew nothing"
    assert gold_ids.isdisjoint({r.row_id for r in replay_rows})
    assert gold_ids | {r.row_id for r in replay_rows} == before
    assert all(r.metadata[REPLAY_ROLE_KEY] == REPLAY_ONLY for r in replay_rows)
    assert all(r.family_id == MMLU_FAMILY for r in replay_rows)
    assert replay.rows_by_split["val"] == () and replay.rows_by_split["heldout"] == ()
    assert gold.rows_by_split["val"] == report.rows_by_split["val"]
    assert len(part.replay_rows) == len(replay_rows)


def _mmlu_report() -> Any:
    config = DataConfig()
    rows = [
        MmluRow(subject=f"s{i % 7}", question=f"What follows {i}?",
                choices=(str(i), str(i + 1), str(i + 2), str(i + 3)), answer_index=0,
                upstream_split="test")
        for i in range(300)
    ]
    mixture = build_mixture({"cais/mmlu": rows}, config=config)
    return split(dedupe(list(mixture.rows), config=config), config=config), config


def test_replay_exclude_drops_only_the_named_replay_rows_and_the_gold_train_is_unchanged() -> None:
    """Option (a) of the J6(a) plan: the decontam's hit rows leave the replay slice for
    ``excluded_rows``; no other row moves, so the gold train is exactly the plain
    --replay-shards build's, which real_ft_run's --replay-partition rebuild reproduces."""
    report, config = _mmlu_report()
    gold0, replay0, _ = pipeline.split_off_replay(report, seed=config.seed)
    drawn = sorted(r.identity_key for r in replay0.rows_by_split["train"])
    hit = frozenset(drawn[:3])
    gold1, replay1, part1 = pipeline.split_off_replay(report, seed=config.seed, exclude=hit)
    assert gold1.rows_by_split["train"] == gold0.rows_by_split["train"]
    assert {r.identity_key for r in replay1.rows_by_split["train"]} == set(drawn) - hit
    assert {r.identity_key for r in part1.excluded_rows} == hit
    assert part1.counts()[MMLU_FAMILY]["excluded"] == 3
    for name in ("val", "heldout"):
        assert gold1.rows_by_split[name] == report.rows_by_split[name]


def test_replay_exclude_refuses_a_key_the_build_draws_as_gold() -> None:
    """Excluding a gold-drawn row would change the gold train, which the trainer's rebuild
    at a502670 would then not match (its pair_labels refuses strays)."""
    report, config = _mmlu_report()
    _, _, part0 = pipeline.split_off_replay(report, seed=config.seed)
    gold_key = part0.gold_rows[0].identity_key
    with pytest.raises(SystemExit, match="this build draws as gold"):
        pipeline.split_off_replay(report, seed=config.seed, exclude=frozenset({gold_key}))


def _hits_file(tmp_path: Path, keys: list[str], *, tool: str | None = None) -> Path:
    import replay_decontam

    path = tmp_path / "hits.json"
    body = {
        "tool": replay_decontam.HITS_TOOL if tool is None else tool, "version": 1,
        "identity_keys": keys,
        "pairs": [{"identity_key": k, "row_id": k, "sequence": i} for i, k in enumerate(keys)],
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_a_replay_exclude_file_is_read_from_the_decontam_hit_list(tmp_path: Path) -> None:
    assert pipeline.read_replay_exclude(_hits_file(tmp_path, ["k1", "k2"])) == frozenset(
        {"k1", "k2"}
    )
    with pytest.raises(SystemExit, match="not a hit list"):
        pipeline.read_replay_exclude(_hits_file(tmp_path, ["k1"], tool="something else"))
    bad = _hits_file(tmp_path, ["k1"])
    raw = json.loads(bad.read_text(encoding="utf-8"))
    raw["identity_keys"] = ["k1", "k9"]
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagree"):
        pipeline.read_replay_exclude(bad)


def test_replay_exclude_without_replay_shards_is_refused(tmp_path: Path) -> None:
    hits = _hits_file(tmp_path, ["k1"])
    with pytest.raises(SystemExit, match="--replay-exclude needs --replay-shards"):
        pipeline.main(["--out", str(tmp_path / "out"), "--no-repo-history",
                       "--general-record", str(tmp_path / "rec.json"),
                       "--replay-exclude", str(hits)])


class _StubTokenizer:
    """Enough of ``RealTokenizer`` for ``run`` to reach stage 1; nothing is tokenized."""

    class _Inner:
        vocab_size = 1
        is_fast = True

        def __len__(self) -> int:
            return 1

    tok = _Inner()


def test_a_consistency_pass_that_could_not_run_stops_the_run_at_stage_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-09-29 rebuild: 315,239 rows against the 250,000 default made the pass
    NotRun, no contradictory prompt was removed, and write_shards refused 42 sequences
    twenty minutes later. The tool now passes its own bound and stops at once if the
    pass still cannot run."""
    monkeypatch.setattr(pipeline.RealTokenizer, "load", lambda **_: _StubTokenizer())
    monkeypatch.setattr(pipeline, "PIPELINE_MAX_CONSISTENCY_ROWS", 0)
    with pytest.raises(SystemExit, match="could not run its consistency pass"):
        pipeline.run(out=tmp_path, max_pairs=2, blank_line_runs=False, rev="HEAD")


def test_the_pipeline_bound_covers_the_corpus_that_was_not_run() -> None:
    assert pipeline.PIPELINE_MAX_CONSISTENCY_ROWS >= 315_239


def test_replay_shards_without_a_general_record_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="needs --general-record"):
        pipeline.run(out=tmp_path, max_pairs=1, blank_line_runs=False, rev="HEAD",
                     replay_shards=True)
