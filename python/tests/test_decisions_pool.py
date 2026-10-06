"""The general-decision pool: ``qd-prep decisions``' rows, as ``qd_data.decisions`` builds them.

Every decision about a pool row is made in Rust (``crates/qd-prep/src/decisions.rs``); the
Python half checks the pool against its manifest and constructs each row through the one
funnel. These tests pin that contract: the families are registered, a pool row becomes a
pinned choice row with its own question in the context, Open-Jev's abstain becomes ``noul``,
a row is routed to its own family only, the loader refuses a pool that is not the one its
manifest names, and the option refusals agree with Rust's on a shared fixture.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

from qd_data.config import DEFAULT_MAX_CANDIDATE_PAIRS, POOL_MAX_CANDIDATE_PAIRS, DataConfig
from qd_data.decisions import (
    DecisionPoolError,
    load_decision_pool,
    pool_data_config,
    rewrite_typed_decision,
)
from qd_data.dedupe import EXACT_CONTENT, NEAR_DUPLICATE_POLICY_KEY, near_duplicate_policy
from qd_data.errors import LicenceRefused
from qd_data.general import _checked_options
from qd_data.loaders import MalformedRowRefusal, TypedDecisionRow
from qd_data.mixture import RowRefused, build_mixture
from qd_data.schema import ChoiceSlot
from qd_data.sources import (
    DECISION_EXACT_CONTENT_FAMILIES,
    DECISION_FAMILIES,
    DECISION_POOL_OPT_INS,
    PINNED_SPLIT_KEY,
    SlotKind,
    source_by_id,
    task_family_by_id,
)

REPO = Path(__file__).resolve().parents[2]
# tools/v5_a7_check.py is imported the way test_v5_a7_check.py imports it, so the A7 tests run
# alone as well as after that file.
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))
OPTION_CASES = REPO / "crates/qd-prep/tests/fixtures/decision-option-cases.json"
OPEN_JEV = "ZefanCai/Open-Jev-v1.1"
ARC = "allenai/ai2_arc"


def _row(**over: object) -> TypedDecisionRow:
    base: dict[str, object] = {
        "example_id": "openjev:g/0/choice", "source_id": OPEN_JEV,
        "family_id": "openjev.policy", "stratum": "openjev.policy/choice", "split": "train",
        "group_key": "g", "licence": "cc0-1.0",
        "context": "Which candidate satisfies every requirement?\n\n{\"candidates\": []}",
        "slot_name": "answer", "options": ("K11", "P72", "K64"), "gold_option": "P72",
        "gold_noul": False, "label_basis": "hard",
    }
    base.update(over)
    return TypedDecisionRow(**base)  # type: ignore[arg-type]


def test_every_decision_family_is_a_choice_family_of_a_registered_pinned_source() -> None:
    assert len(DECISION_FAMILIES) == 15
    for family_id, source_id, description in DECISION_FAMILIES:
        family = task_family_by_id(family_id)
        assert family.source_id == source_id
        assert family.slot_kind is SlotKind.CHOICE
        assert family.description == description and description.strip()
        source = source_by_id(source_id)
        assert dict(source.pinned_splits) == {"train": "train", "val": "val"}
        assert source.licence_policy.licence_id in {
            "cc-by-4.0", "apache-2.0", "mit", "cc-by-sa-3.0", "cc-by-sa-4.0",
        }


def _arc_row(i: int) -> TypedDecisionRow:
    return _row(
        example_id=f"arc:easy:Mercury_{i}", source_id=ARC, family_id="arc.science",
        stratum="arc.science/easy/choice", group_key=f"Mercury_{i}", licence="cc-by-sa-4.0",
        context=f"Which option correctly answers this science question?\n\nWhich gas, case {i}?",
        options=("oxygen", "carbon dioxide", "helium", "neon"), gold_option="carbon dioxide",
    )


def test_arc_pool_rows_are_admitted_only_through_the_pool_config() -> None:
    """ARC is opt-in (share-alike): a build without the pool's config refuses its rows at load,
    and ``pool_data_config`` admits it with the human's recorded call, and nothing else."""
    rows = {ARC: [_arc_row(i) for i in range(3)]}
    with pytest.raises(LicenceRefused, match="opt-in: 'allenai/ai2_arc' is off by default"):
        build_mixture(rows, config=DataConfig(), families=["arc.science"])
    config = pool_data_config()
    assert config.licence.admitted_sources_by_human == DECISION_POOL_OPT_INS == {
        ARC: DECISION_POOL_OPT_INS[ARC]
    }
    assert DataConfig().licence.admitted_sources_by_human == {}, "the default is untouched"
    # The licence and the candidate bound are all it changes.
    assert dataclasses.replace(
        config, licence=DataConfig().licence, max_candidate_pairs=DEFAULT_MAX_CANDIDATE_PAIRS
    ) == DataConfig()
    mixture = build_mixture(rows, config=config, families=["arc.science"])
    assert sorted(r.row_id for r in mixture.rows) == [
        f"decision:arc:easy:Mercury_{i}" for i in range(3)
    ]


def test_the_pool_config_carries_the_measured_candidate_bound() -> None:
    """Every tool that reads the pool gets its bound from ``pool_data_config`` (Fable,
    RULING.md "(A)"); a build without a pool keeps the default, so its manifests do not move."""
    assert DataConfig().max_candidate_pairs == DEFAULT_MAX_CANDIDATE_PAIRS == 5_000_000
    assert pool_data_config().max_candidate_pairs == POOL_MAX_CANDIDATE_PAIRS == 12_500_000
    assert pool_data_config(pool_data_config()) == pool_data_config(), "idempotent"
    named = dataclasses.replace(DataConfig(), max_candidate_pairs=7)
    assert pool_data_config(named).max_candidate_pairs == 7, "a caller's own bound wins"


#: The ruling's families, spelled out here rather than read from the table under test.
_RULED_EXACT_CONTENT = {"openjev.policy", "openjev.evidence", "openjev.routing", "openjev.rubric"}


def test_exactly_the_four_structured_open_jev_families_are_deduplicated_by_exact_content() -> None:
    assert DECISION_EXACT_CONTENT_FAMILIES == _RULED_EXACT_CONTENT
    config = pool_data_config()
    for family_id, source_id, _ in DECISION_FAMILIES:
        source = source_by_id(source_id)
        raw = _row(
            example_id=f"{family_id}:0", source_id=source_id, family_id=family_id,
            stratum=f"{family_id}/choice",
            licence="cc0-1.0" if source_id == OPEN_JEV else source.declared_licence,
        )
        row = rewrite_typed_decision(raw, family_id=family_id, index=0, config=config)
        marked = row.metadata.get(NEAR_DUPLICATE_POLICY_KEY)
        if family_id in _RULED_EXACT_CONTENT:
            assert marked == EXACT_CONTENT, family_id
            assert near_duplicate_policy(row) == EXACT_CONTENT
        else:
            assert NEAR_DUPLICATE_POLICY_KEY not in row.metadata, family_id
            assert near_duplicate_policy(row) is None


def test_no_other_rewriter_writes_the_near_duplicate_policy() -> None:
    """The marker takes a row out of the MinHash search, so who may name its key is pinned: the
    pool's rewriter, which writes it, and qd_data.dedupe, which defines and reads it (split
    reads it through ``dedupe.near_duplicate_policy``). Any other module of qd_data or tools
    naming the key is a second writer nobody ruled on."""
    named = {
        p.relative_to(REPO).as_posix()
        for root in (REPO / "python/qd_data", REPO / "tools")
        for p in root.rglob("*.py")
        if "NEAR_DUPLICATE_POLICY_KEY" in p.read_text(encoding="utf-8")
        or '"near_duplicate_policy"' in p.read_text(encoding="utf-8")
    }
    assert named == {"python/qd_data/decisions.py", "python/qd_data/dedupe.py"}


def test_boolq_and_vitaminc_are_admitted_pool_sources_not_opt_in() -> None:
    """Not opt-in on purpose: an opt-in source is named in every manifest's (hashed)
    refused_sources, so it would move every no-pool data_snapshot_hash."""
    for source_id in ("google/boolq", "tals/vitaminc"):
        source = source_by_id(source_id)
        assert not source.opt_in_note and not source.admission_refusals(DataConfig().licence)
    assert "[U]" in source_by_id("tals/vitaminc").evidence, "the canary GUID is unverified"


def test_option_refusals_agree_with_rust_on_the_shared_fixture() -> None:
    cases = json.loads(OPTION_CASES.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 10
    for case in cases:
        try:
            _checked_options(case["options"], where="fixture")
            got = None
        except RowRefused as exc:
            got = exc.reason_code
        assert got == case["refusal"], case["options"]


def test_a_pool_row_becomes_a_pinned_choice_row_with_its_question_in_the_context() -> None:
    raw = _row(split="val")
    row = rewrite_typed_decision(raw, family_id="openjev.policy", index=0, config=DataConfig())
    assert row.request.context == raw.context.encode("utf-8")
    assert row.request.question == task_family_by_id("openjev.policy").description
    (slot,) = row.request.slots
    assert isinstance(slot, ChoiceSlot) and tuple(slot.options) == raw.options
    assert row.gold[0].value == "P72" and not row.gold[0].is_noul
    assert row.metadata[PINNED_SPLIT_KEY] == "val"
    assert row.repo_key == "openjev.policy-val:g"
    assert row.identity_key.startswith("openjev.policy-val:g::")
    assert row.licence_id == "cc0-1.0"


def test_open_jev_abstain_gold_is_noul() -> None:
    raw = _row(gold_option=None, gold_noul=True)
    row = rewrite_typed_decision(raw, family_id="openjev.policy", index=0, config=DataConfig())
    assert row.gold[0].is_noul and row.gold[0].value is None


def test_a_source_without_per_row_licences_refuses_a_row_naming_another() -> None:
    raw = _row(
        example_id="procedural:x:q", source_id="tasksource/procedural-typed-decisions",
        family_id="procedural.decisions", stratum="procedural.decisions/arithmetic/choice",
        licence="mit",
    )
    with pytest.raises(RowRefused) as exc:
        rewrite_typed_decision(raw, family_id="procedural.decisions", index=0,
                               config=DataConfig())
    assert exc.value.reason_code == "licence_differs_from_source"


def test_a_non_commercial_row_licence_is_refused() -> None:
    with pytest.raises(LicenceRefused):
        rewrite_typed_decision(_row(licence="cc-by-nc-4.0"), family_id="openjev.policy",
                               index=0, config=DataConfig())


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"gold_option": "Z9"}, "gold_option among"),
        ({"gold_option": "P72", "gold_noul": True}, "exactly one"),
        ({"gold_option": None, "gold_noul": False}, "exactly one"),
        ({"split": "test"}, "pinned split policy"),
        ({"stratum": "openjev.nli/choice"}, "stratum under"),
        ({"group_key": ""}, "stratum under"),
    ],
)
def test_a_malformed_pool_row_is_refused(over: dict[str, object], match: str) -> None:
    with pytest.raises(MalformedRowRefusal, match=match):
        _row(**over)


def test_build_mixture_routes_each_pool_row_to_its_own_family_only() -> None:
    rows = [
        _row(example_id=f"openjev:p{i}", group_key=f"p{i}") for i in range(3)
    ] + [
        _row(example_id=f"openjev:e{i}", group_key=f"e{i}", family_id="openjev.evidence",
             stratum="openjev.evidence/yesno", options=("yes", "no"), gold_option="no")
        for i in range(2)
    ]
    mixture = build_mixture({OPEN_JEV: rows}, config=DataConfig(),
                            families=["openjev.policy", "openjev.evidence"])
    built = sorted((r.family_id, r.row_id) for r in mixture.rows)
    assert [f for f, _ in built] == ["openjev.evidence"] * 2 + ["openjev.policy"] * 3
    assert mixture.refusals[OPEN_JEV] == {}


def _write_pool(root: Path, lines: list[dict[str, object]], **manifest: object) -> Path:
    root.mkdir()
    body = "".join(json.dumps(x) + "\n" for x in lines).encode("utf-8")
    (root / "examples.jsonl").write_bytes(body)
    # Every family the pool holds, as qd-prep decisions writes it (decisions.rs
    # text_bytes_by_family); corpus_identity reads the containment scope from it.
    families: dict[str, int] = {}
    for x in lines:
        fam = str(x.get("family_id"))
        families[fam] = families.get(fam, 0) + len(str(x.get("context", "")))
    doc: dict[str, object] = {
        "schema": "qd-decisions/v1", "mode": "build", "examples": len(lines),
        "examples_sha256": hashlib.sha256(body).hexdigest(),
        "text_bytes_by_family": families,
    }
    doc.update(manifest)
    (root / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


def _line(**over: object) -> dict[str, object]:
    raw = _row()
    line: dict[str, object] = {
        "id": raw.example_id, "source_id": raw.source_id, "family_id": raw.family_id,
        "stratum": raw.stratum, "split": raw.split, "group_key": raw.group_key,
        "licence": raw.licence, "context": raw.context, "slot_name": raw.slot_name,
        "options": list(raw.options), "gold_option": raw.gold_option,
        "gold_noul": raw.gold_noul, "label_basis": raw.label_basis,
    }
    line.update(over)
    return line


def test_the_loader_reads_a_pool_whole_grouped_by_source(tmp_path: Path) -> None:
    pool = load_decision_pool(_write_pool(tmp_path / "pool", [
        _line(), _line(id="decider:commands:a:b", source_id="Mapika/decider/teacher_data",
                       family_id="decider.commands", stratum="decider.commands/commands/choice",
                       licence="apache-2.0"),
    ]))
    assert sorted(pool.raw) == ["Mapika/decider/teacher_data", OPEN_JEV]
    assert [r.example_id for r in pool.raw[OPEN_JEV]] == ["openjev:g/0/choice"]


@pytest.mark.parametrize(
    ("manifest", "match"),
    [
        ({"examples_sha256": "0" * 64}, "hashes to"),
        ({"mode": "survey"}, "build pool is required"),
        ({"schema": "qd-decisions/v0"}, "build pool is required"),
        ({"examples": 2}, "the manifest records 2"),
        # GAP-V6-THREE-POOL-PRODUCERS-CAPS-APPLIED-IN-DECISIONS-ONLY-2026-10-06: a candidate
        # pool (qd-prep synth or convert) no cap table has drawn from cannot enter a mixture.
        ({"allocation": {"state": "not_applied", "detail": "candidates"}}, "allocation"),
        ({"allocation": {"detail": "no state"}}, "allocation"),
        ({"allocation": "applied"}, "allocation"),
        ({"allocation": {"state": "Applied"}}, "allocation"),
        # A candidate pool written before producers stated an allocation names its tool.
        ({"tool": "qd-prep convert"}, "allocation"),
        ({"tool": "qd-prep synth"}, "allocation"),
    ],
)
def test_the_loader_refuses_a_pool_its_manifest_does_not_describe(
    tmp_path: Path, manifest: dict[str, object], match: str
) -> None:
    with pytest.raises(DecisionPoolError, match=match):
        load_decision_pool(_write_pool(tmp_path / "pool", [_line()], **manifest))


@pytest.mark.parametrize(
    "manifest",
    [
        {"allocation": {"state": "applied", "detail": "the pinned config's caps"}},
        # A pool written before v6 named no allocation; only `qd-prep decisions`, which applies
        # its cap table, wrote pools then, and v5's pins re-read those pools unchanged.
        {},
    ],
)
def test_an_allocated_or_pre_v6_pool_loads(tmp_path: Path, manifest: dict[str, object]) -> None:
    pool = load_decision_pool(_write_pool(tmp_path / "pool", [_line()], **manifest))
    assert [r.example_id for r in pool.raw[OPEN_JEV]] == ["openjev:g/0/choice"]


def test_the_loader_refuses_a_family_of_another_source(tmp_path: Path) -> None:
    with pytest.raises(MalformedRowRefusal, match="registered decision family"):
        load_decision_pool(_write_pool(tmp_path / "pool", [
            _line(family_id="decider.commands", stratum="decider.commands/x/choice"),
        ]))


#: ``Manifest.snapshot_hash`` per split of ``test_manifest._snapshot()`` (no pool), computed on
#: a clean tree at 7320707, before the pool existed. A build that never reads the pool must
#: hash as it did: ``admitted_source_ids`` is hashed, and naming the pool's six sources in
#: every manifest moved every data_snapshot_hash (lead review 2026-10-03).
NO_POOL_SNAPSHOT_7320707 = {
    "heldout": "3d2e9f1cf11ae23981fbc6ef78598a7aa1c9e83837bb15d56b031c4ee33b93c1",
    "train": "d56011df6ad2cb828fbf298ff331e1a410740a984b09403820c6f3e6c75efa69",
    "val": "0d7e12ecb96f49cf3a2dbdc5a4409ea80b612bb39767e117be27bd8012fb775f",
}


def test_a_build_without_the_pool_hashes_as_it_did_before_the_pool_existed() -> None:
    from test_manifest import _snapshot

    snap = _snapshot()
    assert {s: m.snapshot_hash() for s, m in snap.items()} == NO_POOL_SNAPSHOT_7320707


def test_a_build_with_the_pool_names_the_pool_sources_it_read() -> None:
    from data_fixtures import small_corpus

    from qd_data.dedupe import dedupe
    from qd_data.manifest import build_manifests
    from qd_data.split import split

    config = DataConfig()
    corpus = {**small_corpus(24), OPEN_JEV: [
        _row(example_id=f"openjev:p{i}", group_key=f"p{i}",
             context=f"Which candidate satisfies requirement {i}?\n\n{{\"n\": {i}}}")
        for i in range(4)
    ]}
    mixture = build_mixture(corpus, config=config)
    report = dedupe(list(mixture.rows), config=config)
    manifests = build_manifests(config=config, mixture=mixture, dedupe_report=report,
                                split_report=split(report, config=config))
    admitted = manifests["train"].admitted_source_ids
    assert OPEN_JEV in admitted
    assert "nvidia/HelpSteer2" not in admitted, "a pool source this build did not read"


# -- A7 with the pool ------------------------------------------------------------------------


def _pool_val_lines() -> list[dict[str, object]]:
    return [
        _line(id=f"openjev:v{i}", split="val", group_key=f"v{i}",
              context=f"Which candidate satisfies requirement {i}?\n\n{{\"n\": {i}}}")
        for i in range(2)
    ] + [_line(id="openjev:t0", group_key="t0")]


def _pool_val_identities(pool: Path) -> list[str]:
    loaded = load_decision_pool(pool)
    return sorted(
        rewrite_typed_decision(r, family_id=r.family_id, index=i, config=DataConfig()).identity_key
        for rows in loaded.raw.values() for i, r in enumerate(rows) if r.split == "val"
    )


def _a7_builds(tmp_path: Path, *, val_extra: list[tuple[str, str, str]],
               heldout_extra: list[tuple[str, str, str]] = ()):
    from test_v5_a7_check import SMALL, _rows, _write
    from v5_a7_check import SPLIT_PATHS

    v4 = _write(tmp_path / "v4", {s: _rows(s, oos=False) for s in SPLIT_PATHS})
    rows = {s: _rows(s, oos=True) for s in SPLIT_PATHS}
    rows["val"] += val_extra
    rows["heldout"] += list(heldout_extra)
    return v4, _write(tmp_path / "v5", rows), SMALL


def _pool_rows(identities: list[str]) -> list[tuple[str, str, str]]:
    return [("openjev.policy", ident, ident.split("::")[0]) for ident in identities]


def test_a7_passes_a_pool_build_with_its_pool_and_refuses_it_without(tmp_path: Path) -> None:
    from v5_a7_check import check

    pool = _write_pool(tmp_path / "pool", _pool_val_lines(), val_by_family={"openjev.policy": 2})
    idents = _pool_val_identities(pool)
    v4, v5, gate = _a7_builds(tmp_path, val_extra=_pool_rows(idents))
    without = check(v4, v5, gate=gate)
    assert not without["passed"], "a pool family is not a family v4's gate names"
    report = check(v4, v5, gate=gate, decisions_pool=pool)
    assert report["passed"], [r for r in report["families"] if not r["passed"]]
    row = next(r for r in report["families"]
               if r["split"] == "val" and r["family"] == "openjev.policy")
    assert (row["kind"], row["expected"], row["v5"]) == ("decision-pool", 2, 2)


def test_a7_refuses_a_pool_family_whose_val_rows_are_not_the_pools(tmp_path: Path) -> None:
    from v5_a7_check import check

    pool = _write_pool(tmp_path / "pool", _pool_val_lines(), val_by_family={"openjev.policy": 2})
    idents = _pool_val_identities(pool)
    v4, v5, gate = _a7_builds(tmp_path, val_extra=_pool_rows(idents[:1]))
    report = check(v4, v5, gate=gate, decisions_pool=pool)
    [row] = [r for r in report["families"] if not r["passed"]]
    assert (row["family"], row["missing"], row["extra"]) == ("openjev.policy", 1, 0)


def test_a7_refuses_a_pool_family_in_held_out(tmp_path: Path) -> None:
    from v5_a7_check import check

    pool = _write_pool(tmp_path / "pool", _pool_val_lines(), val_by_family={"openjev.policy": 2})
    idents = _pool_val_identities(pool)
    v4, v5, gate = _a7_builds(tmp_path, val_extra=_pool_rows(idents),
                              heldout_extra=_pool_rows(["openjev.policy-train:t0::x"]))
    report = check(v4, v5, gate=gate, decisions_pool=pool)
    assert [(r["split"], r["family"]) for r in report["families"] if not r["passed"]] == [
        ("heldout", "openjev.policy")
    ]


def test_a7_refuses_a_pool_whose_rows_disagree_with_its_manifest(tmp_path: Path) -> None:
    from v5_a7_check import check

    pool = _write_pool(tmp_path / "pool", _pool_val_lines(), val_by_family={"openjev.policy": 3})
    v4, v5, gate = _a7_builds(tmp_path, val_extra=[])
    with pytest.raises(SystemExit, match="its manifest states"):
        check(v4, v5, gate=gate, decisions_pool=pool)


def test_a7_counts_a_pool_val_row_the_build_refuses_as_refused_not_missing(
    tmp_path: Path,
) -> None:
    """The first v5 build's A7 (2026-10-03, 21:42Z) raised RowRefused on a google/boolq val row
    carrying U+200E. build_mixture refuses such a row and counts it
    (invisible_format_characters), so no split holds it. A7 now counts it as refused, and the
    manifest's val_by_family must equal built + refused (the human's ratification,
    AUDIT/finalize-2026-10-03/human-answers-2026-10-03-a7.md). Fails before the fix: the
    expectation raised. A built row that is absent still refuses (the test above)."""
    from v5_a7_check import check

    plain = _write_pool(tmp_path / "plain", _pool_val_lines(),
                        val_by_family={"openjev.policy": 2})
    idents = _pool_val_identities(plain)
    marked = _line(id="openjev:v9", split="val", group_key="v9",
                   context="Which candidate\u200e satisfies requirement 9?\n\n{\"n\": 9}")
    lines = [*_pool_val_lines(), marked]
    pool = _write_pool(tmp_path / "pool", lines, val_by_family={"openjev.policy": 3})
    v4, v5, gate = _a7_builds(tmp_path, val_extra=_pool_rows(idents))
    report = check(v4, v5, gate=gate, decisions_pool=pool)
    assert report["passed"], [r for r in report["families"] if not r["passed"]]
    assert report["decisions_pool_val_refused"] == {
        "openjev.policy": {"invisible_format_characters": 1}
    }
    row = next(r for r in report["families"]
               if r["split"] == "val" and r["family"] == "openjev.policy")
    assert (row["expected"], row["v5"], row["refused_by_build_mixture"]) == (
        2, 2, {"invisible_format_characters": 1}
    )
    # The refused row is still the pool's: a manifest that leaves it out does not reconcile.
    short = _write_pool(tmp_path / "short", lines, val_by_family={"openjev.policy": 2})
    with pytest.raises(SystemExit, match=r"built \+ refused"):
        check(v4, v5, gate=gate, decisions_pool=short)


def _dedupe_report(path: Path, *, kept: str, dropped: str) -> Path:
    """A build's --dedupe-report-out, as DedupeReport.to_json() writes it: one exact-content
    cluster (the path the pool's rows take) keeping ``kept`` and removing ``dropped``."""
    path.write_text(json.dumps({"clusters": [], "exact_content_clusters": [{
        "digest": "d", "kept_unit_key": f"{kept}|d", "minhash_unit_key": None,
        "dropped_unit_keys": [f"{dropped}|d"], "repo_keys": [], "n_rows_dropped": 1,
    }]}), encoding="utf-8")
    return path


def test_a7_reading_c_counts_a_pool_val_row_deduplicated_against_another_val_row(
    tmp_path: Path,
) -> None:
    """Reading C (the human, ~22:27Z 2026-10-03, AUDIT/finalize-2026-10-03/
    human-answers-2026-10-03-a7-reading-c.md): a pool val row that dedupe removed in favour of
    a val row of the same family is counted as deduplicated within val and named, not missing,
    when the build's own dedupe report shows it. Without the report it still refuses. Fails
    before reading C: check() took no dedupe_report."""
    from v5_a7_check import check

    pool = _write_pool(tmp_path / "pool", _pool_val_lines(), val_by_family={"openjev.policy": 2})
    kept, lost = _pool_val_identities(pool)
    v4, v5, gate = _a7_builds(tmp_path, val_extra=_pool_rows([kept]))
    assert not check(v4, v5, gate=gate, decisions_pool=pool)["passed"]
    evidence = _dedupe_report(tmp_path / "dedupe.json", kept=kept, dropped=lost)
    report = check(v4, v5, gate=gate, decisions_pool=pool, dedupe_report=evidence)
    assert report["passed"], [r for r in report["families"] if not r["passed"]]
    row = next(r for r in report["families"]
               if r["split"] == "val" and r["family"] == "openjev.policy")
    assert (
        row["missing"],
        row["deduplicated_within_val"],
        row["deduplicated_within_val_keys"],
    ) == (0, 1, [[lost, kept]])
    assert report["dedupe_report_sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()


def test_a7_reading_c_still_refuses_a_val_row_removed_for_a_train_row(tmp_path: Path) -> None:
    """The knock-out reading C does not cover: the unit kept in the val row's place is a train
    row (no v5 val row), so the row is missing and A7 refuses."""
    from v5_a7_check import check

    pool = _write_pool(tmp_path / "pool", _pool_val_lines(), val_by_family={"openjev.policy": 2})
    kept, lost = _pool_val_identities(pool)
    v4, v5, gate = _a7_builds(tmp_path, val_extra=_pool_rows([kept]))
    evidence = _dedupe_report(tmp_path / "dedupe.json", kept="openjev.policy-train:t0::x",
                              dropped=lost)
    report = check(v4, v5, gate=gate, decisions_pool=pool, dedupe_report=evidence)
    [row] = [r for r in report["families"] if not r["passed"]]
    assert (row["family"], row["missing"], row["deduplicated_within_val"]) == (
        "openjev.policy", 1, 0
    )


def test_a7_reading_c_refuses_a_report_that_is_not_this_builds(tmp_path: Path) -> None:
    """A report that removed a row v5's val holds was written for another build."""
    from v5_a7_check import check

    pool = _write_pool(tmp_path / "pool", _pool_val_lines(), val_by_family={"openjev.policy": 2})
    kept, lost = _pool_val_identities(pool)
    v4, v5, gate = _a7_builds(tmp_path, val_extra=_pool_rows([kept, lost]))
    evidence = _dedupe_report(tmp_path / "dedupe.json", kept=lost, dropped=kept)
    with pytest.raises(SystemExit, match="not this build's report"):
        check(v4, v5, gate=gate, decisions_pool=pool, dedupe_report=evidence)
    with pytest.raises(SystemExit, match="give --decisions-pool"):
        check(v4, v5, gate=gate, dedupe_report=evidence)


def test_the_corpus_identity_names_a_pool_only_when_given(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO / "tools"))
    import real_tokenizer_pipeline as pipeline

    kw: dict[str, object] = {
        "rev": "r", "max_pairs": 1, "commitpackft": None, "defect_class": None,
        "defect_max_rows": None,
    }
    assert "decisions_pool_examples_sha256" not in pipeline.corpus_identity(**kw)
    pool = _write_pool(tmp_path / "pool", [_line()])
    named = pipeline.corpus_identity(**kw, decisions_pool=pool)
    recorded = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    assert named["decisions_pool_examples_sha256"] == recorded["examples_sha256"]
    # With the pool, the containment scope over its families (containment-scope ruling).
    assert named["decisions_pool_same_family_not_enforced"] == sorted(
        recorded["text_bytes_by_family"]
    )
    assert "decisions_pool_same_family_not_enforced" not in pipeline.corpus_identity(**kw)
