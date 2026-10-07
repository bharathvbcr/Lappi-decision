"""The request example in docs/schema-api.md is the request training builds.

The task name, the question and the slot names are rendered into the prompt, so a caller that
copies a documented request whose names differ from training's serves the model a prompt it never
saw. That is what the doc's example did until 2026-10-03: task ``devcouncil.verdict``, slots
``verdict``/``severity``/``evidence``, and code.commit_intent's question, against training's
``code.defect_class`` / ``defect_class`` / ``defect_span``
(GAP-SCHEMA-API-DOC-REQUEST-IS-NOT-A-TRAINED-REQUEST-2026-10-03). This test builds the example
through the training path and requires the doc to show exactly that.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_data.config import DataConfig
from qd_data.defect_class import CHOICE_SLOT, DEFECT_FAMILY_ID, SPAN_SLOT, DefectRow
from qd_data.mixture import rewrite_defect_class

DOC = Path(__file__).resolve().parents[2] / "docs" / "schema-api.md"

#: The one-line Rust diff the documented example is built from, in the single-file shape the plain
#: corpus trains on: hunks only, the path in ``file:``. Until 2026-10-07 it opened with a bare
#: ``---``/``+++`` preamble, a shape no row trains, and the runtime's admission refused the
#: documented request (GAP-PREVIEW-DOES-NOT-SERVE-CODE-DEFECT-CLASS-2026-10-06);
#: ``crates/qd-runtime/tests/admission.rs`` now admits the block this test pins.
DOC_DIFF = (
    "@@ -1 +1 @@\n"
    "-fn add(a: i32, b: i32) -> i32 { a + b }\n"
    "+fn add(a: i32, b: i32) -> i32 { a - b }\n"
)


def _documented_request() -> dict[str, object]:
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## Request", 1)[1]
    match = re.search(r"```json\n(.*?)\n```", section, flags=re.DOTALL)
    assert match, "docs/schema-api.md has no json block under '## Request'"
    request = json.loads(match.group(1))
    assert isinstance(request, dict)
    return request


def _trained_request() -> dict[str, object]:
    raw = DefectRow(
        example_id="doc-example", pool_id="doc", repo="doc/example", path="src/add.rs",
        symbol="add", arity=2, language="rust", mutation_class="logic", operator="swap_binop",
        diff=DOC_DIFF, diff_span=(3, 3), span_refusal=None, licence="MIT",
    )
    row = rewrite_defect_class(raw, family_id=DEFECT_FAMILY_ID, index=0, config=DataConfig())
    wire = row.request.to_wire()
    assert isinstance(wire, dict)
    return wire


def test_the_documented_request_is_the_request_training_builds() -> None:
    assert _documented_request() == _trained_request()


def test_the_documented_request_names_the_trained_task_and_slots() -> None:
    """Said separately, so a failure names the drift a caller would hit."""
    request = _documented_request()
    assert request["task"] == DEFECT_FAMILY_ID
    slots = request["slots"]
    assert isinstance(slots, list)
    assert [s["name"] for s in slots] == [CHOICE_SLOT, SPAN_SLOT]
