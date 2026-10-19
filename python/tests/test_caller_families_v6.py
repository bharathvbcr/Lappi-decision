"""The v6 caller families, as the loader renders them, against each app's own request builder.

Card ft-2d0a5e4b27e. ``crates/qd-prep/tests/fixtures/caller-requests-v6.json`` pins each app's
own builder output (DevType 24cc8d8 ``LappiRouteRequest.line``, GitPulse 211cd903
``commit_type_request``), read from the app repositories; nothing in it was written by Lappi.
A row a v6 pool carries must become, through ``rewrite_typed_decision``, the request the app
sends: same task, question, slot name, options in order, and the context bytes untouched.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from qd_data.config import DataConfig
from qd_data.decisions import rewrite_typed_decision
from qd_data.loaders import TypedDecisionRow
from qd_data.schema import ChoiceSlot
from qd_data.sources import DECISION_FAMILIES, source_by_id

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "crates/qd-prep/tests/fixtures/caller-requests-v6.json"
CONTRACT = ROOT / "docs/caller-contract.md"


def _apps() -> dict[str, dict[str, object]]:
    doc = json.loads(FIXTURE.read_text())
    return {a["app"]: a for a in doc["apps"]}


def _row(family: str, source: str, licence: str, slot: str, options: list[str], context: str,
         gold: str) -> TypedDecisionRow:
    return TypedDecisionRow(
        example_id=f"{family}:0", source_id=source, family_id=family,
        stratum=f"{family}/{gold}", split="train", group_key="g0", licence=licence,
        context=context, slot_name=slot, options=tuple(options), gold_option=gold,
        gold_noul=False, label_basis="by_construction",
    )


@pytest.mark.parametrize(
    ("app", "family", "source", "licence"),
    [
        ("devtype", "devtype.route", "lappi/synth-route", "synthetic-by-rule"),
        ("gitpulse", "gitpulse.commit_type", "lappi/own-commits", "owner-granted"),
    ],
)
def test_a_pool_row_becomes_the_request_the_app_sends(
    app: str, family: str, source: str, licence: str
) -> None:
    pinned = _apps()[app]
    wire = json.loads(pinned["line"])  # type: ignore[arg-type]
    context = base64.b64decode(wire["context_b64"]).decode("utf-8")
    assert len(context.encode()) == wire["context_len"]
    slot = wire["slots"][0]
    raw = _row(family, source, licence, slot["name"], slot["options"], context, slot["options"][0])
    row = rewrite_typed_decision(raw, family_id=family, index=0, config=DataConfig())
    request = row.request
    assert request.task == wire["task"]
    assert request.question == wire["question"]
    assert request.route == wire["route"]
    assert request.context == base64.b64decode(wire["context_b64"])
    (got,) = request.slots
    assert isinstance(got, ChoiceSlot)
    assert (got.name, list(got.options)) == (slot["name"], slot["options"])


def test_the_registered_descriptions_are_contract_section_4_byte_for_byte() -> None:
    section = CONTRACT.read_text().split("## 4. Ask only trained shapes")[1].split("## 5.")[0]
    by_family = {f: d for f, _, d in DECISION_FAMILIES}
    for family in ("gitpulse.commit_type", "devtype.route"):
        row = next(line for line in section.splitlines() if f"`{family}`" in line)
        assert f"`{by_family[family]}`" in row, (family, row)


def test_both_families_are_registered_to_sources_that_admit_them() -> None:
    families = {f: s for f, s, _ in DECISION_FAMILIES}
    assert families["gitpulse.commit_type"] == "lappi/own-commits"
    assert families["devtype.route"] == "lappi/synth-route"
    assert source_by_id("lappi/own-commits").declared_licence == "owner-granted"
    assert source_by_id("lappi/synth-route").declared_licence == "synthetic-by-rule"


def test_the_caps_table_names_both_families() -> None:
    config = json.loads((ROOT / "data/decisions/v6-allocation.json").read_text())
    assert {"gitpulse.commit_type", "devtype.route"} <= set(config["train_caps"])


JD_FIXTURE = ROOT / "fixtures/caller/career-jd-triage-fixture.json"
JD_SHA256 = "0cf04ad791d67fb04bf4e1c5be9b3206f61cf356dddf0a3054c9a8b5a13be47d"


def _jd_families() -> list[dict[str, object]]:
    raw = JD_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == JD_SHA256
    doc = json.loads(raw)
    assert doc["fixture_version"] == 2
    return list(doc["families"])


def test_the_career_descriptions_and_contract_rows_are_the_fixtures_strings() -> None:
    section = CONTRACT.read_text().split("## 4. Ask only trained shapes")[1].split("## 5.")[0]
    by_family = {f: (s, d) for f, s, d in DECISION_FAMILIES}
    fams = _jd_families()
    assert len(fams) == 4
    for fam in fams:
        task, slot = str(fam["task"]), fam["slot"]
        source, description = by_family[task]
        assert source == "lappi/synth-career"
        assert description == fam["question"]
        row = next(line for line in section.splitlines() if f"`{task}`" in line)
        options = ", ".join(o["value"] for o in slot["options"])  # type: ignore[index]
        assert f"`{fam['question']}`" in row, task
        assert f"`{slot['name']}` choice `[{options}]`" in row, task  # type: ignore[index]
    assert source_by_id("lappi/synth-career").declared_licence == "synthetic-by-rule"


def test_a_career_pool_row_becomes_the_request_vani_sends() -> None:
    context = "Target roles: ML systems\n\nTitle: Engineer\nLocation: Austin, TX\n\nBuild é things."
    for fam in _jd_families():
        slot = fam["slot"]
        options = [o["value"] for o in slot["options"]]  # type: ignore[index]
        raw = _row(str(fam["task"]), "lappi/synth-career", "synthetic-by-rule",
                   slot["name"], options, context, options[-1])  # type: ignore[index]
        row = rewrite_typed_decision(raw, family_id=str(fam["task"]), index=0, config=DataConfig())
        request = row.request
        assert request.task == fam["task"]
        assert request.question == fam["question"]
        assert request.context == context.encode()
        (got,) = request.slots
        assert isinstance(got, ChoiceSlot)
        assert (got.name, list(got.options)) == (slot["name"], options)  # type: ignore[index]
