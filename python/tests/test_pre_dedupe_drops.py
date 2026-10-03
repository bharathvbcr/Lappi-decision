"""The pre-dedupe drop list (``qd_train.exclusions.drop_before_dedupe``, the pipeline's and the
trainer's ``--drop-before-dedupe``): the human's ratification of Fable's option B, ~21:52Z
2026-10-03 (``AUDIT/finalize-2026-10-03/human-answers-2026-10-03-a7.md``).

``qd_data.dedupe`` keeps the lexically smallest unit of a near-duplicate component whatever
its split, so a train row whose key sorts first removes its val twin, and an exclusion after
the split cannot bring that twin back. The knock-out tests reproduce it on decision-pool
rows whose val and train copies share one text, then show the list restoring the val rows.
Every test here fails before the list existed: ``drop_before_dedupe``, ``split_before_dedupe``,
the corpus key and ``--dedupe-report-out`` are new.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_fixtures import small_corpus  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.loaders import MmluRow  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.split import split  # noqa: E402
from qd_train.exclusions import (  # noqa: E402
    PRE_DEDUPE_DROPS_KEY,
    ExclusionRefusal,
    drop_before_dedupe,
    pre_dedupe_drops_identity,
    split_before_dedupe,
)

VOCAB = ["amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "iris",
         "juniper", "kestrel", "lagoon", "meadow", "nectar", "opal", "prism", "quartz",
         "raven", "sierra", "tundra"]


def _corpus() -> dict[str, list[object]]:
    """``small_corpus(24)``, MMLU rows of both pinned splits (``validation`` is val, ``test``
    is train), and decision-pool rows: three val rows each with a train row of exactly the
    same text in another group -- the first v5 build's decider/procedural case, deduped on the
    exact-content path, where "openjev.policy-train:" sorts ahead of "-val:" -- and
    distinct train rows besides."""
    from test_decisions_pool import OPEN_JEV, _row

    rng = random.Random(20261003)
    raw = small_corpus(24)
    mmlu: list[MmluRow] = []
    for j in range(12):
        q = " ".join(rng.choice(VOCAB) for _ in range(25))
        mmlu.append(MmluRow(subject=f"s{j % 4}", question=f"clean {j} {q}?",
                            choices=(f"p{j}", f"q{j}", f"r{j}", f"s{j}"), answer_index=2,
                            upstream_split="validation" if j % 3 == 0 else "test"))
    raw["cais/mmlu"] = mmlu
    pool = []
    for j in range(3):
        text = f"Which candidate satisfies requirement {j}?\n\n{{\"n\": {j}, \"twin\": true}}"
        for split_name, group in (("val", f"v{j}"), ("train", f"t{j}")):
            pool.append(_row(example_id=f"openjev:{group}/0/choice", group_key=group,
                             split=split_name, context=text))
    for j in range(6):
        pool.append(_row(example_id=f"openjev:g{j}/0/choice", group_key=f"g{j}",
                         context=" ".join(f"term{j}x{k}" for k in range(16))))
    raw[OPEN_JEV] = pool
    return raw


def _rows():
    return build_mixture(_corpus(), config=DataConfig()).rows


def _write(path: Path, keys: list[str]) -> Path:
    path.write_bytes(b"".join(k.encode("utf-8") + b"\n" for k in keys))
    return path


def _knockers(rows) -> tuple[list[str], list[str]]:
    """(train twins, val rows): the val rows dedupe drops, on either path, and the train rows
    kept in their place."""
    from qd_data.dedupe import content_unit_key

    config = DataConfig()
    report = dedupe(list(rows), config=config)
    by_unit = {content_unit_key(r): r for r in rows}
    clusters = [(c.kept_unit_key, c.dropped_unit_keys) for c in report.clusters] + [
        (c.kept_unit_key or c.minhash_unit_key, c.dropped_unit_keys)
        for c in report.exact_content_clusters
    ]
    val_lost, twins = [], []
    for kept_key, dropped in clusters:
        kept = by_unit[kept_key]
        for d in dropped:
            lost = by_unit[d]
            if split_before_dedupe(lost, config=config) == "val":
                assert split_before_dedupe(kept, config=config) == "train"
                val_lost.append(lost.identity_key)
                twins.append(kept.identity_key)
    return sorted(set(twins), key=str.encode), sorted(set(val_lost), key=str.encode)


def test_split_before_dedupe_is_splits_own_assignment_row_for_row() -> None:
    config = DataConfig()
    report = dedupe(list(_rows()), config=config)
    got = split(report, config=config)
    assigned = {a.row_id: a.split for a in got.assignments}
    assert assigned
    for row in report.kept:
        assert split_before_dedupe(row, config=config) == assigned[row.row_id], row.row_id
    assert {"train", "val"} <= set(assigned.values())


def test_a_val_row_knocked_out_by_its_train_twin_survives_when_the_twin_leaves_first(
    tmp_path: Path,
) -> None:
    config = DataConfig()
    rows = _rows()
    twins, lost = _knockers(rows)
    # The defect, reproduced: dedupe keeps the train twin and the val row is in no split.
    assert twins and lost, "the MMLU pairs did not reproduce the knock-out"
    plain = split(dedupe(list(rows), config=config), config=config)
    plain_val = {r.identity_key for r in plain.rows_by_split["val"]}
    assert plain_val.isdisjoint(lost)
    listed = _write(tmp_path / "drops.txt", twins)
    kept, drops, dropped = drop_before_dedupe(rows, listed, config=config)
    assert drops is not None and drops.keys == frozenset(twins)
    assert drops.sha256 == hashlib.sha256(listed.read_bytes()).hexdigest()
    assert {r.identity_key for r in dropped} == set(twins)
    assert len(kept) + len(dropped) == len(rows)
    fixed = split(dedupe(list(kept), config=config), config=config)
    fixed_val = {r.identity_key for r in fixed.rows_by_split["val"]}
    assert set(lost) <= fixed_val, "the val rows did not come back"
    # Nothing else moved: val gains exactly the restored rows, held-out is what it was.
    assert fixed_val - plain_val == set(lost)
    assert ({r.row_id for r in fixed.rows_by_split["heldout"]}
            == {r.row_id for r in plain.rows_by_split["heldout"]})
    for name in ("val", "heldout"):
        assert all(r.identity_key not in twins for r in fixed.rows_by_split[name])


def test_without_a_list_the_rows_are_the_rows() -> None:
    rows = _rows()
    kept, drops, dropped = drop_before_dedupe(rows, None, config=DataConfig())
    assert kept == tuple(rows) and drops is None and dropped == ()
    assert pre_dedupe_drops_identity(None) == {}


def test_a_list_naming_a_val_row_is_refused_before_any_row_moves(tmp_path: Path) -> None:
    rows = _rows()
    _twins, lost = _knockers(rows)
    with pytest.raises(ExclusionRefusal, match="would be split out of train"):
        drop_before_dedupe(rows, _write(tmp_path / "d.txt", lost[:1]), config=DataConfig())


def test_a_list_naming_no_row_or_malformed_is_refused(tmp_path: Path) -> None:
    rows = _rows()
    config = DataConfig()
    with pytest.raises(ExclusionRefusal, match="name no row of this corpus"):
        drop_before_dedupe(rows, _write(tmp_path / "a.txt", ["no-such-key"]), config=config)
    with pytest.raises(ExclusionRefusal, match="byte-sorted and unique"):
        drop_before_dedupe(rows, _write(tmp_path / "b.txt", ["b", "a"]), config=config)
    with pytest.raises(ExclusionRefusal, match="no key"):
        drop_before_dedupe(rows, _write(tmp_path / "c.txt", []), config=config)
    with pytest.raises(ExclusionRefusal, match="unreadable"):
        drop_before_dedupe(rows, tmp_path / "missing.txt", config=config)


def test_the_corpus_names_the_list_only_when_given(tmp_path: Path) -> None:
    """The list's sha256 is part of the corpus an exclusion list's attestation names, so a scan
    made without it does not match a build with it, and every earlier corpus is unchanged."""
    import real_tokenizer_pipeline as pipeline

    base = dict(rev="r", max_pairs=60, commitpackft=None, defect_class=None,
                defect_max_rows=None, repo_history=False)
    listed = _write(tmp_path / "drops.txt", ["k"])
    without = pipeline.corpus_identity(**base)
    assert PRE_DEDUPE_DROPS_KEY not in without
    with_list = pipeline.corpus_identity(**base, drop_before_dedupe=listed)
    assert with_list == {
        **without, PRE_DEDUPE_DROPS_KEY: hashlib.sha256(b"k\n").hexdigest()
    }


@pytest.mark.usefixtures("qd_prep")
def test_the_pipeline_and_the_trainers_rebuild_drop_the_same_rows_and_keep_the_val_twin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decision pool whose val row has a train row of exactly its text keyed ahead of it
    ("openjev.policy-train:" sorts before "-val:"; pool rows are deduped on the exact-content
    path, as the first v5 build's were). The pipeline's build with the list writes
    the val row; without it, dedupe drops it. ``real_ft_run.ft_splits`` with the same list
    rebuilds exactly the manifests the pipeline wrote, which is what the trainer, the
    containment scan and the linear control rebuild through."""
    pytest.importorskip("torch")
    import real_ft_run as rft
    import real_tokenizer_pipeline as pipeline
    from repo_git import resolve_rev
    from test_decisions_pool import _line, _write_pool
    from test_real_ft_general_record import (
        PINNED,
        _manifest_ids,
        _ManifestsWritten,
        _StubTokenizer,
    )

    from qd_data.decisions import load_decision_pool, rewrite_typed_decision

    twin = " ".join(f"word{j}" for j in range(24))
    lines = [
        _line(id="openjev:v0/0/choice", group_key="v0", split="val", context=twin,
              options=["K0", "P0", "Q0"], gold_option="P0"),
        _line(id="openjev:t0/0/choice", group_key="t0", split="train", context=twin,
              options=["K0", "P0", "Q0"], gold_option="P0"),
    ] + [
        _line(id=f"openjev:g{i}/0/choice", group_key=f"g{i}", split="train",
              context=" ".join(f"term{i}x{j}" for j in range(16)),
              options=[f"K{i}", f"P{i}", f"Q{i}"], gold_option=f"P{i}")
        for i in range(1, 6)
    ]
    pool = _write_pool(tmp_path / "pool", lines)
    loaded = load_decision_pool(pool)
    built = {
        r.group_key: rewrite_typed_decision(r, family_id=r.family_id, index=i,
                                            config=DataConfig())
        for rows in loaded.raw.values() for i, r in enumerate(rows)
    }
    keys = {g: row.identity_key for g, row in built.items()}
    listed = _write(tmp_path / "drops.txt", [keys["t0"]])
    rev = resolve_rev(REPO, PINNED)
    monkeypatch.setattr(pipeline.RealTokenizer, "load", lambda **_: _StubTokenizer())

    def stop(*_a: object, **_k: object) -> None:
        raise _ManifestsWritten

    monkeypatch.setattr(pipeline, "census", stop)
    for name, drops in (("plain", None), ("dropped", listed)):
        with pytest.raises(_ManifestsWritten):
            pipeline.run(out=tmp_path / name, max_pairs=3, blank_line_runs=False, rev=rev,
                         decisions_pool=pool, pre_dedupe_drops=drops,
                         dedupe_report_out=tmp_path / f"{name}-dedupe.json")
    # The plain build's own dedupe report names the knock-out: the train twin kept, the val
    # row removed in its favour. With the list, no component holds the val row.
    def components(report: dict) -> list[dict]:
        return [*report["clusters"], *report.get("exact_content_clusters", [])]

    plain_report = json.loads((tmp_path / "plain-dedupe.json").read_text(encoding="utf-8"))
    [knock] = [c for c in components(plain_report)
               if any(d.startswith(keys["v0"] + "|") for d in c["dropped_unit_keys"])]
    assert (knock.get("kept_unit_key") or knock["minhash_unit_key"]).startswith(keys["t0"] + "|")
    dropped_report = json.loads((tmp_path / "dropped-dedupe.json").read_text(encoding="utf-8"))
    assert not any(keys["v0"] in json.dumps(c) for c in components(dropped_report))
    with pytest.raises(SystemExit, match="written once"):
        pipeline.run(out=tmp_path / "again", max_pairs=3, blank_line_runs=False, rev=rev,
                     decisions_pool=pool, dedupe_report_out=tmp_path / "plain-dedupe.json")
    val = "data/pool/val.json"
    assert built["v0"].row_id not in _manifest_ids(tmp_path / "plain" / val), (
        "the fixture did not reproduce the knock-out"
    )
    assert built["v0"].row_id in _manifest_ids(tmp_path / "dropped" / val)
    config = DataConfig()
    splits = rft.ft_splits(commitpackft=None, max_pairs=3, rev=rev, config=config,
                           decisions_pool=pool, pre_dedupe_drops=listed)
    for name, rel in (("train", "data/pool/train.json"), ("val", val),
                      ("heldout", "data/heldout/heldout.json")):
        assert sorted(r.row_id for r in splits[name]) == _manifest_ids(tmp_path / "dropped" / rel)
    assert keys["v0"] in {r.identity_key for r in splits["val"]}
    assert keys["t0"] not in {r.identity_key for rows in splits.values() for r in rows}
