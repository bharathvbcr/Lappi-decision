"""Manifests and ``data_snapshot_hash``: reproducible, or the protocol is a fiction.

``qd_train.ledger.Protocol`` makes ``data_snapshot_hash`` one of the five components
that decide whether two runs are comparable, and promotion needs *"three rows
differing only in seed"*. If the hash moves for an unchanged corpus, no two rows are
ever comparable and the promotion rule cannot be applied; if it fails to move for a
changed one, rows from different corpora are compared as though they were the same
experiment. Both directions are tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from data_fixtures import commitpackft_row, small_corpus

from qd_data.config import SPLITS, DataConfig
from qd_data.dedupe import dedupe
from qd_data.licences import LicenceConfig
from qd_data.manifest import MANIFEST_FORMAT_VERSION, Manifest, build_manifests, data_snapshot_hash
from qd_data.mixture import build_mixture
from qd_data.split import HELD_OUT, split
from qd_train.ledger import Protocol
from qd_train.tristate import NotRun, Ran


def _snapshot(config: DataConfig | None = None, n: int = 24, **kwargs: object):
    config = config or DataConfig()
    corpus = kwargs.pop("corpus", None) or small_corpus(n)
    mixture = build_mixture(corpus, config=config, **kwargs)  # type: ignore[arg-type]
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    return build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )


# -- reproducibility ---------------------------------------------------------


def test_the_same_inputs_produce_the_same_hash() -> None:
    assert data_snapshot_hash(_snapshot()) == data_snapshot_hash(_snapshot())


def test_the_hash_does_not_depend_on_row_order() -> None:
    corpus = small_corpus(18)
    forward = _snapshot(corpus=corpus)
    reversed_corpus = {k: list(reversed(v)) for k, v in corpus.items()}
    backward = _snapshot(corpus=reversed_corpus)
    assert data_snapshot_hash(forward) == data_snapshot_hash(backward)


def test_the_hash_does_not_depend_on_wall_clock_or_host() -> None:
    a = _snapshot()
    b = _snapshot()
    for name in SPLITS:
        a[name].provenance["built_at_utc"] = "1999-01-01T00:00:00+00:00"
        a[name].provenance["host"] = "somewhere-else"
    assert data_snapshot_hash(a) == data_snapshot_hash(b)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda: DataConfig(seed=1), id="seed"),
        pytest.param(lambda: DataConfig(train_fraction=0.8), id="train_fraction"),
        pytest.param(lambda: DataConfig(dedupe_threshold=0.7), id="dedupe_threshold"),
        pytest.param(lambda: DataConfig(shingle_size=3), id="shingle_size"),
        pytest.param(
            lambda: DataConfig(held_out_families=("code.change_scope", "intent.in_scope")),
            id="held_out_families",
        ),
        pytest.param(
            lambda: DataConfig(
                licence=LicenceConfig(admitted_by_human={"mpl-2.0": "counsel signed off"})
            ),
            id="licence_override",
        ),
    ],
)
def test_a_config_change_moves_the_hash(mutate) -> None:
    """Every field that changes the data is in the fingerprint. A config knob that
    changed the corpus without moving the hash would let two different snapshots
    share a protocol."""
    assert data_snapshot_hash(_snapshot()) != data_snapshot_hash(_snapshot(mutate()))


def test_adding_a_row_moves_the_hash() -> None:
    corpus = small_corpus(18)
    base = data_snapshot_hash(_snapshot(corpus=corpus))
    bigger = {k: list(v) for k, v in corpus.items()}
    bigger["bigcode/commitpackft"].append(commitpackft_row(999))
    assert data_snapshot_hash(_snapshot(corpus=bigger)) != base


def test_the_hash_covers_the_held_out_split_too() -> None:
    """A hash over only the training shard would be identical for two runs whose
    held-out sets differ, which is exactly what the protocol must distinguish."""
    manifests = _snapshot()
    assert manifests[HELD_OUT].entries, "fixture produced no held-out rows"
    with pytest.raises(ValueError, match="needs every split"):
        data_snapshot_hash({"train": manifests["train"]})


def test_the_hash_of_zero_manifests_is_refused() -> None:
    with pytest.raises(ValueError, match="zero manifests"):
        data_snapshot_hash({})


def test_the_snapshot_hash_is_accepted_by_the_ledger_protocol() -> None:
    digest = data_snapshot_hash(_snapshot())
    protocol = Protocol(
        data_snapshot_hash=digest, tokenizer_hash="t" * 64,
        backbone_commit="c" * 40, recipe_hash="r" * 64, seed=3,
    )
    assert protocol.to_json()["data_snapshot_hash"] == digest
    assert len(protocol.hash()) == 64


# -- the file form -----------------------------------------------------------


def test_a_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    manifests = _snapshot()
    path = tmp_path / "train.json"
    written = manifests["train"].write(path)
    loaded = Manifest.read(path)
    assert loaded.snapshot_hash() == written
    assert loaded.split == "train"
    assert len(loaded.entries) == len(manifests["train"].entries)


def test_an_edited_manifest_is_refused_on_read(tmp_path: Path) -> None:
    """The recorded hash is re-derived from the contents, so an edit is caught."""
    manifests = _snapshot()
    path = tmp_path / "train.json"
    manifests["train"].write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["licence_id"] = "cc0-1.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the hash re-derived"):
        Manifest.read(path)


def test_an_unknown_manifest_format_version_is_refused(tmp_path: Path) -> None:
    manifests = _snapshot()
    path = tmp_path / "train.json"
    manifests["train"].write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest_format_version"] = MANIFEST_FORMAT_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_format_version"):
        Manifest.read(path)


def test_a_not_run_snapshot_refuses_to_be_written_silently(tmp_path: Path) -> None:
    manifests = _snapshot(capped_sources=["bigcode/commitpackft"])
    assert isinstance(manifests["train"].status, NotRun)
    with pytest.raises(ValueError, match="its status is not_run"):
        manifests["train"].write(tmp_path / "train.json")
    digest = manifests["train"].write(tmp_path / "train.json", allow_not_run=True)
    assert Manifest.read(tmp_path / "train.json").snapshot_hash() == digest


def test_a_not_run_stage_propagates_into_every_manifest() -> None:
    manifests = _snapshot(capped_sources=["bigcode/commitpackft"])
    for name in SPLITS:
        status = manifests[name].status
        assert isinstance(status, NotRun)
        assert "hit its row bound" in status.reason


def test_a_clean_pipeline_reports_ran_on_every_manifest() -> None:
    for name, manifest in _snapshot().items():
        assert isinstance(manifest.status, Ran), name
        assert manifest.status.passed


def test_a_manifest_refuses_entries_from_another_split() -> None:
    manifests = _snapshot()
    train = manifests["train"]
    with pytest.raises(ValueError, match=r"carries \d+ entries from another split"):
        Manifest(
            split="train",
            entries=(*train.entries, manifests[HELD_OUT].entries[0]),
            config_fingerprint=train.config_fingerprint,
            admitted_source_ids=train.admitted_source_ids,
            refused_sources=train.refused_sources,
            held_out_families=train.held_out_families,
            licence_histogram=train.licence_histogram,
            obligations=train.obligations,
            host_histogram=train.host_histogram,
            mixture_json=train.mixture_json,
            dedupe_json=train.dedupe_json,
            split_json=train.split_json,
            status=train.status,
            provenance=train.provenance,
        )


# -- what the model card is generated from -----------------------------------


def test_every_manifest_carries_the_whole_refusal_report() -> None:
    """A reader holding only the training manifest must still see why ``anli`` is
    out, or the model card is written from a partial record."""
    manifest = _snapshot()["train"]
    assert "facebook/anli" in manifest.refused_sources
    assert any("cc-by-nc-4.0" in r for r in manifest.refused_sources["facebook/anli"])
    assert "nuprl/AgentPack" in manifest.refused_sources
    assert manifest.admitted_source_ids == (
        "bigcode/commitpackft", "clinc/clinc_oos", "rajpurkar/squad_v2",
    )


def test_obligations_survive_to_the_manifest() -> None:
    manifest = _snapshot()["train"]
    payload = manifest.to_json()
    assert "share-alike" in payload["obligations"].get("cc-by-sa-4.0", [])
    assert payload["host_histogram"] == {"huggingface": len(manifest.entries)}


def test_every_entry_names_its_licence_host_and_repo() -> None:
    for manifest in _snapshot().values():
        for entry in manifest.entries:
            assert entry.licence_id and entry.host and entry.repo_key
            assert entry.split == manifest.split
