"""``campaign/lora-2b-q8-preregistered.json``: the D3 bar is the owner decision's prose.

Rule 2: the numbers in the file's ``d3_gate`` block are read back from
``AUDIT/lora-2b-quantised-2026-10-17/owner-decision.md`` section 5, so a changed number in either
place fails here. The bf16 pre-registration is not edited for the quantised arm.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import d3_quant_gate as gate  # noqa: E402

PREREG = REPO / "campaign" / "lora-2b-q8-preregistered.json"
OWNER = REPO / "AUDIT" / "lora-2b-quantised-2026-10-17" / "owner-decision.md"


def _section5() -> str:
    text = OWNER.read_text(encoding="utf-8")
    m = re.search(r"^## 5\. .*?(?=^## 6\.)", text, re.S | re.M)
    assert m, "owner-decision.md has no section 5"
    return re.sub(r"\s+", " ", m.group(0))


def test_gate_tool_reads_a_spec_not_a_missing_block() -> None:
    spec = gate.load_spec(PREREG)
    assert isinstance(spec, gate.GateSpec), spec
    assert (spec.mode, spec.bits, spec.group_size) == ("affine", 8, 64)


def test_numbers_equal_the_owner_decision_prose() -> None:
    s5 = _section5()
    m = re.search(
        r"argmax agreement >= ([\d.]+)% and mean \|delta log p\| of the gold letter <= ([\d.]+) "
        r"nats over >= (\d+) val prompts \(never held-out\)",
        s5,
    )
    assert m, "the section 5 bar sentence changed; re-derive the pre-registration"
    spec = gate.load_spec(PREREG)
    assert isinstance(spec, gate.GateSpec)
    assert spec.argmax_agreement_min == pytest.approx(float(m.group(1)) / 100.0)
    assert spec.mean_abs_delta_logp_max == float(m.group(2))
    assert spec.min_prompts == int(m.group(3))
    assert "q8 (MLX affine, group 64)" in re.sub(r"\s+", " ", OWNER.read_text(encoding="utf-8"))


def test_cited_blob_sha_is_the_owner_decisions() -> None:
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    cited = doc["design_by"]["owner_decision"]["git_blob_sha"]
    path = doc["design_by"]["owner_decision"]["path"]
    # git hash-object over the working-tree bytes equals the blob sha while the file is unchanged.
    out = subprocess.run(
        ["git", "hash-object", str(REPO / path)], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == cited


def test_procedure_fields_present_and_labelled() -> None:
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    assert doc["family"]["id"] == "lora-2b-q8"
    assert doc["family"]["label"] == "procedure" and doc["base"]["label"] == "procedure"
    assert doc["base"]["format"]
    assert doc["base"]["mlx_version"] == "0.31.2"
    assert "base-file sha256" in doc["family"]["pooling_rule"]
    assert set(doc["d3_gate"]) == {
        "argmax_agreement_min",
        "mean_abs_delta_logp_max",
        "min_prompts",
        "quantiser",
        "source",
        "scope",
        "read_by",
    }
