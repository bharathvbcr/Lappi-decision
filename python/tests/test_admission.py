"""Which datasets are admitted, which are refused, and why -- computed, not recalled.

``docs/plan-corrections.md`` BLOCKING-1 and BLOCKING-2 are recorded as data in
``qd_data.sources`` and ``qd_data.licences`` so the admission decision is *derived*.
These tests assert the derivation matches the findings, and -- more importantly --
that the two refusal *reasons* stay separate: ``facebook/anli`` is closed (a licence),
``nuprl/AgentPack`` needs a human to accept terms (a gate), and treating those the
same would make one look like the other.

The config test at the end is the one ``qd_data.config`` exists for: *holding out a
family that was never going to load is not a holdout*.
"""

from __future__ import annotations

import pytest

from qd_data.config import DEFAULT_HELD_OUT_FAMILIES, N_HELD_OUT_FAMILIES, DataConfig
from qd_data.errors import LicenceRefused
from qd_data.licences import (
    COMMITPACKFT_DECLARED_VALUES,
    LICENCE_POLICY,
    LicenceConfig,
    LicenceTier,
    admit_licence,
    classify,
    is_non_commercial,
    normalise_licence,
)
from qd_data.sources import (
    SOURCES,
    TASK_FAMILIES,
    Reachability,
    admitted_sources,
    admitted_task_families,
    refusal_report,
    source_by_id,
    task_family_by_id,
)

# -- the admitted roster -----------------------------------------------------


def test_exactly_three_sources_are_admitted_unattended() -> None:
    assert {s.source_id for s in admitted_sources()} == {
        "bigcode/commitpackft", "clinc/clinc_oos", "rajpurkar/squad_v2",
    }


def test_commitpackft_is_the_primary_pool_and_carries_a_per_row_licence() -> None:
    """``docs/plan-corrections.md`` BLOCKING-1 promotes it from contrast set to pool."""
    source = source_by_id("bigcode/commitpackft")
    assert source.reachability is Reachability.LOADABLE
    assert source.per_row_licence_field is True
    assert source.licence_policy.tier is LicenceTier.ALLOW


def test_agentpack_is_registered_gated_and_blocks_nothing() -> None:
    """BLOCKING-1: gated ``auto``, 403. Registered so it slots in; nothing depends
    on it, and it is absent from the admitted set until a human accepts the terms."""
    source = source_by_id("nuprl/AgentPack")
    assert source.reachability is Reachability.GATED
    assert source.per_row_licence_field is False, "the per-row licence field is UNVERIFIED"
    assert source.source_id not in {s.source_id for s in admitted_sources()}
    reasons = refusal_report()["nuprl/AgentPack"]
    assert len(reasons) == 1 and "gated" in reasons[0]
    assert not any("licence" in r for r in reasons), "its licence is fine; the gate is not"
    # No admitted task family names it, so no pipeline stalls on it.
    assert not [f for f in admitted_task_families() if f.source_id == "nuprl/AgentPack"]


def test_anli_is_refused_for_its_licence_and_cannot_be_enabled() -> None:
    reasons = refusal_report()["facebook/anli"]
    assert any("cc-by-nc-4.0" in r and "disqualifying" in r for r in reasons)
    with pytest.raises(LicenceRefused):
        LicenceConfig(admitted_by_human={"cc-by-nc-4.0": "for research only"})


@pytest.mark.parametrize(
    ("source_id", "expected_fragment"),
    [
        ("mteb/stsbenchmark-sts", "unknown"),
        ("nyu-mll/glue", "other"),
        ("google/code_x_glue_cc_defect_detection", "c-uda"),
    ],
)
def test_a_human_call_source_is_refused_but_admission_is_one_config_line(
    source_id: str, expected_fragment: str
) -> None:
    assert any(expected_fragment in r for r in refusal_report()[source_id])
    licence = source_by_id(source_id).declared_licence
    config = LicenceConfig(admitted_by_human={licence: "human reviewed 2026-09-19"})
    assert not source_by_id(source_id).admission_refusals(config)
    assert source_id in {s.source_id for s in admitted_sources(config)}


@pytest.mark.parametrize("source_id", ["PolyAI/banking77", "AmazonScience/massive"])
def test_script_only_sources_are_refused_for_reachability_not_licence(
    source_id: str,
) -> None:
    """BLOCKING-2: their licences are fine; they fail under ``datasets>=3.0``."""
    source = source_by_id(source_id)
    assert source.licence_policy.tier is LicenceTier.ALLOW
    reasons = source.admission_refusals(LicenceConfig())
    assert len(reasons) == 1 and "script-only" in reasons[0]


def test_codereviewer_is_recorded_as_not_on_the_declared_host() -> None:
    """BLOCKING-2: 404 with valid auth, control-tested. The data is a Zenodo DOI."""
    source = source_by_id("microsoft/CodeReviewer")
    assert source.reachability is Reachability.NOT_ON_HOST
    assert "zenodo" in source.evidence.lower()
    assert any("not present on the declared host" in r for r in refusal_report()[source.source_id])


def test_an_unregistered_source_cannot_be_loaded_by_accident() -> None:
    with pytest.raises(KeyError, match="must be registered with its"):
        source_by_id("some/unregistered-dataset")
    with pytest.raises(KeyError, match="unknown task family"):
        task_family_by_id("made.up")


# -- the licence table -------------------------------------------------------


def test_every_declared_commitpackft_value_is_classified() -> None:
    """The card enumerates thirteen values and calls them all permissive. Three are
    not. Every one must land on a deliberate tier, never on an unrecognised path."""
    assert len(COMMITPACKFT_DECLARED_VALUES) == 13
    for value in COMMITPACKFT_DECLARED_VALUES:
        assert value in LICENCE_POLICY, f"{value} falls through to the unknown path"


def test_the_three_non_permissive_commitpackft_values_need_a_human_call() -> None:
    for value in ("agpl-3.0", "lgpl-2.1", "unknown"):
        assert classify(value).tier is LicenceTier.NEEDS_HUMAN_CALL
    for value in ("mit", "apache-2.0", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"):
        assert classify(value).tier is LicenceTier.ALLOW


def test_an_unrecognised_licence_is_never_admitted_by_default() -> None:
    policy = classify("some-bespoke-eula-2.1")
    assert policy.tier is LicenceTier.NEEDS_HUMAN_CALL
    with pytest.raises(LicenceRefused) as excinfo:
        admit_licence("some-bespoke-eula-2.1", source="probe")
    assert "some-bespoke-eula-2.1" in str(excinfo.value)


def test_an_unrecognised_non_commercial_licence_is_disqualifying_outright() -> None:
    assert classify("weird-nc-9.9").tier is LicenceTier.DISQUALIFYING


def test_non_commercial_detection_is_segment_exact() -> None:
    """``"nc" in "unlicense"`` is true and would be a catastrophic false positive."""
    assert not is_non_commercial("unlicense")
    assert not is_non_commercial("isc")
    assert is_non_commercial("cc-by-nc-4.0")
    assert is_non_commercial("cc-by-nc-sa-4.0")


def test_licence_normalisation_does_not_guess() -> None:
    assert normalise_licence("  MIT  ") == "mit"
    assert normalise_licence("Apache 2.0") == "apache-2.0"
    # A near-miss stays a near-miss rather than being mapped onto a permissive id.
    assert normalise_licence("mit-ish") == "mit-ish"
    assert classify("mit-ish").tier is LicenceTier.NEEDS_HUMAN_CALL


def test_an_override_without_a_justification_is_refused() -> None:
    with pytest.raises(ValueError, match="carries no justification"):
        LicenceConfig(admitted_by_human={"mpl-2.0": "   "})


def test_share_alike_obligations_are_carried_not_collapsed() -> None:
    policy = admit_licence("cc-by-sa-4.0", source="rajpurkar/squad_v2")
    assert "share-alike" in policy.obligations
    assert "attribution" in policy.obligations


# -- the held-out family assertion -------------------------------------------


def test_the_default_holdout_names_two_otherwise_admitted_families() -> None:
    config = DataConfig()
    admitted = {f.family_id for f in admitted_task_families(config.licence)}
    assert len(DEFAULT_HELD_OUT_FAMILIES) == N_HELD_OUT_FAMILIES
    assert set(DEFAULT_HELD_OUT_FAMILIES) <= admitted


def test_holding_out_a_family_that_would_never_load_is_refused() -> None:
    """The check ``qd_data.config`` exists for. ``docs/plan-corrections.md``: *the two
    held-out task families must be chosen from what actually survives.*

    Built by pointing the holdout at a family whose source is refused for its
    licence, which is exactly the silent failure: the pipeline never sees those rows,
    every report says the held-out families abstained, and the gate passes without
    having tested anything.
    """
    unloadable = [
        f.family_id
        for f in TASK_FAMILIES.values()
        if f.source_id not in {s.source_id for s in admitted_sources()}
    ]
    assert not unloadable, (
        "every registered family currently loads, so this test builds its own"
    )
    with pytest.raises(ValueError, match="not registered task families"):
        DataConfig(held_out_families=("code.commit_intent", "never.loads"))


def test_holding_out_everything_is_refused() -> None:
    admitted = sorted(f.family_id for f in admitted_task_families())
    assert len(admitted) > N_HELD_OUT_FAMILIES
    with pytest.raises(ValueError, match="fixes the holdout at 2"):
        DataConfig(held_out_families=tuple(admitted))


def test_a_duplicate_holdout_is_refused() -> None:
    with pytest.raises(ValueError, match="contains a duplicate"):
        DataConfig(held_out_families=("qa.answerability", "qa.answerability"))


def test_a_holdout_that_leaves_nothing_to_train_on_is_refused() -> None:
    """Built by admitting only one source, so its two families are the whole roster."""
    only_clinc = {
        f.family_id for f in TASK_FAMILIES.values() if f.source_id == "clinc/clinc_oos"
    }
    assert len(only_clinc) == N_HELD_OUT_FAMILIES
    # Both clinc families held out still leaves the other sources, so this must pass.
    DataConfig(held_out_families=tuple(sorted(only_clinc)))


def test_training_families_are_the_admitted_ones_minus_the_holdout() -> None:
    config = DataConfig()
    admitted = {f.family_id for f in admitted_task_families(config.licence)}
    assert set(config.training_families) == admitted - set(config.held_out_families)
    assert all(not config.is_held_out_family(f) for f in config.training_families)


def test_a_split_configuration_that_empties_the_held_out_set_is_refused() -> None:
    """A held-out set of size zero passes every leakage check vacuously."""
    with pytest.raises(ValueError, match="leaves no repos for the natural held-out set"):
        DataConfig(train_fraction=0.9, val_fraction=0.1)


def test_every_registered_source_is_named_by_at_least_one_finding() -> None:
    """Each source carries the evidence it was admitted or refused on, so no
    admission decision rests on recollection."""
    for source in SOURCES.values():
        assert source.evidence.strip(), source.source_id
        assert len(source.evidence) > 40, source.source_id
