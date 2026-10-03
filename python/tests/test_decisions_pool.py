"""The general-decision pool: ``qd-prep decisions``' rows, as ``qd_data.decisions`` builds them.

Every decision about a pool row is made in Rust (``crates/qd-prep/src/decisions.rs``); the
Python half checks the pool against its manifest and constructs each row through the one
funnel. These tests pin that contract: the families are registered, a pool row becomes a
pinned choice row with its own question in the context, Open-Jev's abstain becomes ``noul``,
a row is routed to its own family only, the loader refuses a pool that is not the one its
manifest names, and the option refusals agree with Rust's on a shared fixture.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from qd_data.config import DataConfig
from qd_data.decisions import (
    DecisionPoolError,
    load_decision_pool,
    rewrite_typed_decision,
)
from qd_data.errors import LicenceRefused
from qd_data.general import _checked_options
from qd_data.loaders import MalformedRowRefusal, TypedDecisionRow
from qd_data.mixture import RowRefused, build_mixture
from qd_data.schema import ChoiceSlot
from qd_data.sources import (
    DECISION_FAMILIES,
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
    assert len(DECISION_FAMILIES) == 12
    for family_id, source_id, description in DECISION_FAMILIES:
        family = task_family_by_id(family_id)
        assert family.source_id == source_id
        assert family.slot_kind is SlotKind.CHOICE
        assert family.description == description and description.strip()
        source = source_by_id(source_id)
        assert dict(source.pinned_splits) == {"train": "train", "val": "val"}
        assert source.licence_policy.licence_id in {"cc-by-4.0", "apache-2.0", "mit"}


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
    doc: dict[str, object] = {
        "schema": "qd-decisions/v1", "mode": "build", "examples": len(lines),
        "examples_sha256": hashlib.sha256(body).hexdigest(),
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
    ],
)
def test_the_loader_refuses_a_pool_its_manifest_does_not_describe(
    tmp_path: Path, manifest: dict[str, object], match: str
) -> None:
    with pytest.raises(DecisionPoolError, match=match):
        load_decision_pool(_write_pool(tmp_path / "pool", [_line()], **manifest))


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
