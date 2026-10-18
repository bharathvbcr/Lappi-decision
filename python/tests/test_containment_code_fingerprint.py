"""A containment attestation fingerprints the code its scan reads, not the whole package.

GAP-W4C-CONTAINMENT-ATTESTATION-STILL-FINGERPRINTS-WHOLE-QD-DATA-2026-10-18 (Fable round 2,
Q9): ``containment_corpus`` hashed ``code_fingerprint()``, every ``qd_data`` module, so the
first attestation after any ``qd_data`` edit -- which the v6 build is -- named a corpus no
earlier attestation matched. W4-c gave shard headers the import closure of their entry points
(``test_shard_code_fingerprint.py``); this is the same treatment for the containment entry
points, ``qd_train/exclusions.py``'s ``qd_data`` imports, with its own list and version.

Attestations written before it carry no ``qd_data_fingerprint_version``: they are rule 1,
stay rule 1, and are never re-pinned or read under the narrower list.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from test_shard_code_fingerprint import NOT_CONTENT, QD_DATA, _import_closure  # noqa: E402

import qd_data.fingerprint as fp  # noqa: E402
from qd_train import exclusions  # noqa: E402

#: The containment entry point. Its ``qd_data`` imports are config, fingerprint, rows, split.
ENTRY_POINTS = (REPO / "python" / "qd_train" / "exclusions.py",)

IDENTITY = {"sources": "s", "seed": 0}


def _closure() -> set[str]:
    return _import_closure(ENTRY_POINTS) - NOT_CONTENT


def _copy_package(dest: Path) -> None:
    for path in QD_DATA.glob("*.py"):
        shutil.copy(path, dest / path.name)


def _edit(directory: Path, name: str) -> None:
    target = directory / name
    target.write_text(target.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of ``qd_data`` that the fingerprint functions read, free to edit."""
    _copy_package(tmp_path)
    monkeypatch.setattr(fp, "_package_dir", lambda: tmp_path)
    return tmp_path


def _outside_closure() -> str:
    outside = sorted(set(fp.SHARD_CONTENT_MODULES) - set(fp.CONTAINMENT_CONTENT_MODULES))
    assert outside, "every shard module is in the containment closure; the test has no subject"
    return outside[0]


def test_the_list_is_the_import_closure_of_the_containment_entry_points() -> None:
    closure = _closure()
    assert closure, "the closure found nothing; the walker is broken"
    assert set(fp.CONTAINMENT_CONTENT_MODULES) == closure
    assert list(fp.CONTAINMENT_CONTENT_MODULES) == sorted(fp.CONTAINMENT_CONTENT_MODULES)
    assert all((QD_DATA / name).is_file() for name in fp.CONTAINMENT_CONTENT_MODULES)


def test_the_list_is_narrower_than_the_package_and_inside_the_shard_list() -> None:
    assert set(fp.CONTAINMENT_CONTENT_MODULES) < set(fp.code_fingerprint())
    assert set(fp.CONTAINMENT_CONTENT_MODULES) <= set(fp.SHARD_CONTENT_MODULES)
    assert fp.CONTAINMENT_FINGERPRINT_VERSION == 2


def test_a_module_outside_the_closure_does_not_move_the_attestation(package: Path) -> None:
    name = _outside_closure()
    before = exclusions.containment_corpus(IDENTITY)
    _edit(package, name)
    (package / "pool_builder.py").write_text("# edited\n", encoding="utf-8")
    assert exclusions.containment_corpus(IDENTITY) == before
    assert before[exclusions.QD_DATA_FINGERPRINT_VERSION_KEY] == 2


@pytest.mark.parametrize("name", sorted(fp.CONTAINMENT_CONTENT_MODULES))
def test_a_closure_module_changing_moves_the_attestation(package: Path, name: str) -> None:
    before = exclusions.containment_corpus(IDENTITY)
    _edit(package, name)
    after = exclusions.containment_corpus(IDENTITY)
    assert after[exclusions.QD_DATA_FINGERPRINT_KEY] != before[exclusions.QD_DATA_FINGERPRINT_KEY]
    assert not exclusions.corpus_matches(before, after)


def test_a_missing_closure_module_is_refused_not_skipped(package: Path) -> None:
    (package / fp.CONTAINMENT_CONTENT_MODULES[0]).unlink()
    with pytest.raises(RuntimeError, match=r"containment-content"):
        exclusions.containment_corpus(IDENTITY)


def _v1_attested() -> dict[str, object]:
    """A corpus as an attestation written before the version field named it."""
    digest = exclusions._digest(fp.code_fingerprint())
    return {**IDENTITY, exclusions.QD_DATA_FINGERPRINT_KEY: digest}


def test_a_v1_attestation_is_still_checked_by_the_v1_rule(package: Path) -> None:
    attested = _v1_attested()
    assert exclusions.QD_DATA_FINGERPRINT_VERSION_KEY not in attested
    assert exclusions.corpus_matches(attested, exclusions.containment_corpus(IDENTITY))
    # Rule 1 covers every module: an edit outside the closure voids it, as it always did.
    _edit(package, _outside_closure())
    assert not exclusions.corpus_matches(attested, exclusions.containment_corpus(IDENTITY))


def test_a_v2_attestation_survives_the_edit_that_voids_a_v1_one(package: Path) -> None:
    v1, v2 = _v1_attested(), exclusions.containment_corpus(IDENTITY)
    _edit(package, _outside_closure())
    built = exclusions.containment_corpus(IDENTITY)
    assert exclusions.corpus_matches(v2, built)
    assert not exclusions.corpus_matches(v1, built)


def test_a_v1_attestation_is_never_matched_by_a_v2_digest_alone() -> None:
    """The version key is not decoration: a v1 attestation carrying the rule-2 digest, with no
    version, is compared as rule 1 and so does not match the closure digest."""
    v2 = exclusions.containment_corpus(IDENTITY)
    forged = {k: v for k, v in v2.items() if k != exclusions.QD_DATA_FINGERPRINT_VERSION_KEY}
    assert not exclusions.corpus_matches(forged, v2)
