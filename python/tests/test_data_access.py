"""Rule 3: pointing the trainer at held-out data must raise, not train.

``CLAUDE.md`` rule 3 and ``docs/hardening.md`` section 2 both require this, and the
hardening doc names the test: *"a test asserts that pointing the trainer at a
held-out manifest raises rather than trains."* That is
:func:`test_pointing_the_trainer_at_a_held_out_manifest_raises`.

The other tests here exist because one check is not four. Each layer has a way to be
defeated on its own -- a rename defeats the path marker, a move defeats the
configured root, a rebuilt corpus defeats both -- so each is tested against the
attack that defeats the others.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from data_fixtures import small_corpus
from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.errors import HeldOutViolation, is_noul_payload, is_refusal_payload
from qd_data.manifest import Manifest, ManifestEntry, build_manifests
from qd_data.mixture import build_mixture
from qd_data.split import HELD_OUT, split
from qd_train.data_access import (
    assert_manifest_trainable,
    assert_path_not_held_out,
    held_out_check_report,
    open_training_data,
)
from qd_train.tristate import NotRun, Ran


@pytest.fixture
def snapshot(tmp_path: Path) -> tuple[Path, DataConfig, dict[str, Path]]:
    """A written snapshot: train, val and heldout manifests under a repo root."""
    config = DataConfig()
    mixture = build_mixture(small_corpus(24), config=config)
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    paths: dict[str, Path] = {}
    for name, manifest in manifests.items():
        directory = tmp_path / "data" / (HELD_OUT if name == HELD_OUT else "pool")
        path = directory / f"{name}.json"
        manifest.write(path)
        paths[name] = path
    return tmp_path, config, paths


# -- the test docs/hardening.md section 2 asks for ---------------------------


def test_pointing_the_trainer_at_a_held_out_manifest_raises(
    snapshot: tuple[Path, DataConfig, dict[str, Path]],
) -> None:
    root, config, paths = snapshot
    assert paths[HELD_OUT].exists()
    with pytest.raises(HeldOutViolation) as excinfo:
        open_training_data(paths[HELD_OUT], config=config, repo_root=root)
    assert excinfo.value.check == "held_out_path"
    assert "rule 3" in str(excinfo.value)


def test_a_training_manifest_opens_and_reports_which_checks_cleared_it(
    snapshot: tuple[Path, DataConfig, dict[str, Path]],
) -> None:
    root, config, paths = snapshot
    handle = open_training_data(paths["train"], config=config, repo_root=root)
    assert handle.manifest.split == "train"
    assert handle.n_rows > 0
    assert len(handle.data_snapshot_hash) == 64
    for name in (
        "path_not_held_out", "manifest_split_trainable",
        "no_held_out_families", "snapshot_status",
    ):
        check = handle.checks[name]
        assert isinstance(check, Ran) and check.passed, name


def test_no_held_out_family_identifier_appears_in_a_training_shard(
    snapshot: tuple[Path, DataConfig, dict[str, Path]],
) -> None:
    """``docs/hardening.md`` section 2: *a test asserts their identifiers appear in no
    training shard.*"""
    root, config, paths = snapshot
    for name in ("train", "val"):
        handle = open_training_data(paths[name], config=config, repo_root=root)
        families = {e.family_id for e in handle.manifest.entries}
        assert not families & set(config.held_out_families)
    held = Manifest.read(paths[HELD_OUT])
    assert {e.family_id for e in held.entries} & set(config.held_out_families)


# -- layer 1: the configured roots -------------------------------------------


def test_a_path_under_a_configured_held_out_root_is_refused(tmp_path: Path) -> None:
    config = DataConfig()
    target = tmp_path / "data" / "heldout" / "shard.json"
    with pytest.raises(HeldOutViolation):
        assert_path_not_held_out(target, config=config, repo_root=tmp_path)


def test_a_traversal_back_into_the_held_out_root_is_refused(tmp_path: Path) -> None:
    """Resolved paths, not string prefixes: ``pool/../heldout/x`` is held-out data."""
    config = DataConfig()
    sneaky = tmp_path / "data" / "pool" / ".." / "heldout" / "shard.json"
    with pytest.raises(HeldOutViolation):
        assert_path_not_held_out(sneaky, config=config, repo_root=tmp_path)


def test_a_symlink_into_the_held_out_root_is_refused(tmp_path: Path) -> None:
    config = DataConfig()
    real = tmp_path / "data" / "heldout"
    real.mkdir(parents=True)
    (real / "shard.json").write_text("{}", encoding="utf-8")
    link = tmp_path / "data" / "pool_link"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(HeldOutViolation):
        assert_path_not_held_out(link / "shard.json", config=config, repo_root=tmp_path)


def test_the_check_fires_on_a_path_that_does_not_exist(tmp_path: Path) -> None:
    """The refusal is on the *intent* to read held-out data, not on a successful read."""
    with pytest.raises(HeldOutViolation):
        assert_path_not_held_out(
            tmp_path / "data" / "heldout" / "never-created.json",
            config=DataConfig(), repo_root=tmp_path,
        )


def test_an_empty_held_out_root_list_is_refused_at_config_construction() -> None:
    with pytest.raises(ValueError, match="held_out_roots is empty"):
        DataConfig(held_out_roots=())


# -- layer 2: the path markers -----------------------------------------------


@pytest.mark.parametrize("marker", ["heldout", "held_out", "held-out", "HELDOUT"])
def test_a_held_out_marker_anywhere_in_the_path_is_refused(
    tmp_path: Path, marker: str
) -> None:
    """Renaming the directory out of ``data/`` does not launder it."""
    config = DataConfig()
    target = tmp_path / "scratch" / marker / "shard.json"
    with pytest.raises(HeldOutViolation) as excinfo:
        assert_path_not_held_out(target, config=config, repo_root=tmp_path)
    assert "path segment" in str(excinfo.value)


def test_marker_matching_is_segment_exact_not_substring(tmp_path: Path) -> None:
    """``withheld_outputs`` is not a holdout. A substring check would refuse it, the
    trainer would see no data, and the run would look fine."""
    config = DataConfig()
    fine = tmp_path / "data" / "withheld_outputs" / "shard.json"
    assert_path_not_held_out(fine, config=config, repo_root=tmp_path)


# -- layer 3: the manifest's own declaration ---------------------------------


def test_a_held_out_manifest_moved_to_a_clean_path_is_still_refused(
    snapshot: tuple[Path, DataConfig, dict[str, Path]],
) -> None:
    """The file says what it is. Moving it does not change that."""
    root, config, paths = snapshot
    laundered = root / "data" / "pool" / "totally_fine.json"
    laundered.write_bytes(paths[HELD_OUT].read_bytes())
    assert_path_not_held_out(laundered, config=config, repo_root=root)  # layers 1-2 pass
    with pytest.raises(HeldOutViolation) as excinfo:
        open_training_data(laundered, config=config, repo_root=root)
    assert "declares itself split 'heldout'" in str(excinfo.value)


# -- layer 4: the family holdout ---------------------------------------------


def test_a_shard_built_with_a_different_holdout_is_refused(
    snapshot: tuple[Path, DataConfig, dict[str, Path]],
) -> None:
    """Rule 2: an agent may report that a gate failed; it may not shrink a held-out
    set to make one pass. A corpus built with a different holdout is not this run's."""
    root, _, paths = snapshot
    other = DataConfig(held_out_families=("code.change_scope", "intent.in_scope"))
    with pytest.raises(HeldOutViolation) as excinfo:
        open_training_data(paths["train"], config=other, repo_root=root)
    assert excinfo.value.check == "held_out_path"
    assert "different holdout" in str(excinfo.value) or "held-out task family" in str(
        excinfo.value
    )


def test_a_train_manifest_carrying_a_held_out_family_is_refused(tmp_path: Path) -> None:
    """The layer that catches a corpus built with the wrong config rather than a file
    in the wrong place. Constructed by relabelling one entry's split."""
    config = DataConfig()
    mixture = build_mixture(small_corpus(24), config=config)
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    train = manifests["train"]
    source = next(
        e for e in manifests[HELD_OUT].entries if e.family_id in config.held_out_families
    )
    contraband = ManifestEntry(
        row_id=source.row_id, content_hash=source.content_hash, split="train",
        source_id=source.source_id, host=source.host, family_id=source.family_id,
        repo_key=source.repo_key, identity_key=source.identity_key,
        licence_id=source.licence_id, obligations=source.obligations,
    )
    poisoned = Manifest(
        split="train",
        entries=(*train.entries, contraband),
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
    path = tmp_path / "data" / "pool" / "train.json"
    poisoned.write(path)
    with pytest.raises(HeldOutViolation) as excinfo:
        open_training_data(path, config=config, repo_root=tmp_path)
    assert "held-out task family" in str(excinfo.value)


# -- the snapshot's own status -----------------------------------------------


def test_training_on_an_unverified_snapshot_is_refused_by_default(tmp_path: Path) -> None:
    config = DataConfig()
    mixture = build_mixture(
        small_corpus(24), config=config, capped_sources=["bigcode/commitpackft"]
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    path = tmp_path / "data" / "pool" / "train.json"
    manifests["train"].write(path, allow_not_run=True)

    with pytest.raises(HeldOutViolation) as excinfo:
        open_training_data(path, config=config, repo_root=tmp_path)
    assert "unverified snapshot" in str(excinfo.value)

    handle = open_training_data(
        path, config=config, repo_root=tmp_path, allow_not_run_snapshot=True
    )
    assert isinstance(handle.checks["snapshot_status"], NotRun)


def test_training_on_a_snapshot_that_ran_and_failed_is_refused(tmp_path: Path) -> None:
    """``NotRun`` was refused; ``Ran(passed=False)`` was not, and it is the worse case.

    The door branched on ``isinstance(manifest.status, NotRun)`` and let every ``Ran``
    through without ever reading ``passed`` -- so a snapshot whose pipeline ran and
    *failed* was admitted silently, while the weaker "did not run" was refused loudly.
    "Unverified" and "verified as bad" both reached the trainer, but only one of them
    was stopped.

    ``allow_not_run_snapshot`` must not open this door either. It is an override for a
    check that could not run, not a general one; a caller reaching for it to get past a
    failed gate is the thing rule 2 forbids, reached from the other side.
    """
    config = DataConfig()
    mixture = build_mixture(
        small_corpus(24), config=config, capped_sources=["bigcode/commitpackft"]
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    # `status` is deliberately outside `hashed_body`, so this stays hash-valid and the
    # door has to refuse it on the status alone rather than on a hash mismatch.
    failed = replace(
        manifests["train"],
        status=Ran(passed=False, detail="the dedupe gate failed on this snapshot"),
    )
    path = tmp_path / "data" / "pool" / "train.json"
    failed.write(path)

    with pytest.raises(HeldOutViolation) as excinfo:
        open_training_data(path, config=config, repo_root=tmp_path)
    assert "ran and failed" in str(excinfo.value)
    assert "the dedupe gate failed on this snapshot" in str(excinfo.value)

    with pytest.raises(HeldOutViolation):
        open_training_data(
            path, config=config, repo_root=tmp_path, allow_not_run_snapshot=True
        )


def test_an_edited_manifest_is_refused_by_the_training_door(
    snapshot: tuple[Path, DataConfig, dict[str, Path]],
) -> None:
    """The door re-derives the snapshot hash, so a hand-edited shard cannot be trained
    on even if every held-out check passes."""
    root, config, paths = snapshot
    payload = json.loads(paths["train"].read_text(encoding="utf-8"))
    payload["entries"].pop()
    paths["train"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the hash re-derived"):
        open_training_data(paths["train"], config=config, repo_root=root)


# -- the aggregate report ----------------------------------------------------


def test_the_path_report_over_zero_paths_is_not_run(tmp_path: Path) -> None:
    """*"We checked no paths" must not read as "no path was held out".*"""
    report = held_out_check_report([], config=DataConfig(), repo_root=tmp_path)
    assert isinstance(report, NotRun)
    assert "zero paths" in report.reason


def test_the_path_report_fails_when_any_path_is_held_out(tmp_path: Path) -> None:
    config = DataConfig()
    report = held_out_check_report(
        [tmp_path / "data" / "pool" / "a.json", tmp_path / "data" / "heldout" / "b.json"],
        config=config, repo_root=tmp_path,
    )
    assert isinstance(report, Ran)
    assert not report.passed
    assert report.n == report.n_total == 2


def test_a_held_out_violation_is_not_an_abstention() -> None:
    """``docs/schema-api.md``: collapsing a refusal into ``noul`` would let a rule-3
    violation read as model humility."""
    exc = HeldOutViolation(expected="a trainable path", actual="data/heldout/x.json")
    payload = exc.to_json()
    assert is_refusal_payload(payload)
    assert not is_noul_payload(payload)
    assert not hasattr(exc, "noul")


def test_assert_manifest_trainable_names_the_split_it_refused(
    snapshot: tuple[Path, DataConfig, dict[str, Path]],
) -> None:
    _, config, paths = snapshot
    held = Manifest.read(paths[HELD_OUT])
    with pytest.raises(HeldOutViolation) as excinfo:
        assert_manifest_trainable(held, config=config, path=paths[HELD_OUT])
    assert excinfo.value.actual == HELD_OUT
