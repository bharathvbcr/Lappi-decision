"""Adapter visibility scopes share-alike and research-only licences (v2 plan section 6, gate 4).

``cc-by-sa-3.0`` stays in the global ``ALLOW`` tier: v1's pools admit it and removing it would
change v1's behaviour. What changes is the question a *public* adapter asks. Visibility is a
second gate beside the tier, never a replacement for it.
"""

from __future__ import annotations

import pytest

from qd_data.errors import LicenceRefused
from qd_data.licences import (
    DEFAULT_LICENCE_CONFIG,
    LICENCE_POLICY,
    AdapterVisibility,
    LicenceConfig,
    LicenceTier,
    admit_licence,
    admit_licence_for,
    parse_visibility,
    private_only_reason,
)


def test_by_sa_is_still_in_the_global_allow_tier() -> None:
    # The task forbids deleting it: v1's pools read it.
    assert LICENCE_POLICY["cc-by-sa-3.0"].tier is LicenceTier.ALLOW
    assert admit_licence("cc-by-sa-3.0").licence_id == "cc-by-sa-3.0"


@pytest.mark.parametrize(
    "licence", ["cc-by-sa-3.0", "CC-BY-SA-3.0", "cc-by-sa-4.0", " cc-by-sa-4.0 "]
)
def test_a_public_adapter_refuses_share_alike_and_a_private_one_admits_it(licence: str) -> None:
    with pytest.raises(LicenceRefused) as exc:
        admit_licence_for(licence, visibility=AdapterVisibility.PUBLIC)
    assert "share-alike" in str(exc.value)
    assert (
        admit_licence_for(licence, visibility=AdapterVisibility.PRIVATE).tier is LicenceTier.ALLOW
    )


def test_a_public_adapter_refuses_a_research_only_licence_even_when_a_human_admitted_it() -> None:
    config = LicenceConfig(admitted_by_human={"mind2web-research-only": "owner ruling 2026-10-20"})
    assert private_only_reason("mind2web-research-only") is not None
    with pytest.raises(LicenceRefused):
        admit_licence_for(
            "mind2web-research-only", visibility=AdapterVisibility.PUBLIC, config=config
        )
    # Private: the human's admission is honoured, and nothing else changed.
    admitted = admit_licence_for(
        "mind2web-research-only", visibility=AdapterVisibility.PRIVATE, config=config
    )
    assert admitted.licence_id == "mind2web-research-only"


def test_a_research_only_licence_is_not_admitted_privately_without_the_human_call() -> None:
    with pytest.raises(LicenceRefused):
        admit_licence_for(
            "mind2web-research-only",
            visibility=AdapterVisibility.PRIVATE,
            config=DEFAULT_LICENCE_CONFIG,
        )


@pytest.mark.parametrize("licence", ["mit", "apache-2.0", "cc-by-4.0", "synthetic-by-rule"])
def test_other_allow_licences_are_unaffected_by_visibility(licence: str) -> None:
    for visibility in AdapterVisibility:
        assert admit_licence_for(licence, visibility=visibility).licence_id == licence


def test_a_missing_visibility_is_public_never_private() -> None:
    # Fail closed: public is the stricter reading, so a manifest that forgot the field can be
    # refused for a BY-SA row but can never leak one.
    assert parse_visibility(None) is AdapterVisibility.PUBLIC


@pytest.mark.parametrize("raw", ["", "Public ", "internal", 1, True])
def test_an_unrecognised_visibility_is_refused_not_guessed(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_visibility(raw)


def test_visibility_spellings() -> None:
    assert parse_visibility("public") is AdapterVisibility.PUBLIC
    assert parse_visibility("private") is AdapterVisibility.PRIVATE


def test_a_disqualifying_licence_is_still_refused_for_a_private_adapter() -> None:
    with pytest.raises(LicenceRefused):
        admit_licence_for("cc-by-nc-4.0", visibility=AdapterVisibility.PRIVATE)
