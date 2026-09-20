"""The one Python copy of schema.rs' constants, and the boundary that keeps it reachable.

Two things are pinned here:

1. Every constant in :mod:`qd_train.schema_mirror` still equals what ``schema.rs`` says.
2. The torch boundary ``docs/training-contract.md`` declares is real — measured by importing
   every module with torch made unimportable, not by grepping for ``import torch``.

(2) is what makes (1) worth having. The constants were previously defined in
``qd_train.heads``, which imports torch at module scope, so ``qd_train.byte_batch`` — whose
whole job is index arithmetic that must run without torch — could not import at all in the
repo venv, and ``qd_train.calibration_fit`` carried a second copy to avoid the same fate.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from qd_train.schema_mirror import (
    CHOICE,
    MAX_OPTIONS,
    MIN_OPTIONS,
    RESERVED_NOUL_ROWS,
    SCORE,
    SPAN,
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA_RS = REPO / "crates" / "qd-runtime" / "src" / "schema.rs"
PYTHON_ROOT = REPO / "python"

#: Modules that need torch *at import time*. Everything else in ``qd_train`` and ``qd_data``
#: is torch-free and must import on a machine that has never seen torch.
#:
#: ``ledger`` and ``remap`` are deliberately absent: they use torch, but import it inside the
#: functions that need it, which is the pattern that keeps the rest of the module usable.
TORCH_GATED: frozenset[str] = frozenset(
    {
        "qd_train.backbone",
        "qd_train.byte_decider",
        "qd_train.byte_train",
        "qd_train.fused_ce",
        "qd_train.heads",
    }
)

# Run in a subprocess with a meta-path finder that refuses torch, so this measures the same
# thing whether or not the interpreter running the suite has torch installed. Importing in
# this process would prove nothing on a machine with torch, and would be unfixably polluted
# by whatever an earlier test already put in `sys.modules`.
_PROBE = """
import importlib, json, sys


class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return None


sys.meta_path.insert(0, _NoTorch())

result = {"ok": [], "needed_torch": [], "other_failure": {}}
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or (exc.name or "").startswith("torch."):
            result["needed_torch"].append(name)
        else:
            result["other_failure"][name] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        result["other_failure"][name] = f"{type(exc).__name__}: {exc}"
    else:
        result["ok"].append(name)

print(json.dumps(result))
"""


def _module_names() -> list[str]:
    names = []
    for package in ("qd_train", "qd_data", "qd_wire"):
        for path in sorted((PYTHON_ROOT / package).glob("*.py")):
            if path.stem != "__init__":
                names.append(f"{package}.{path.stem}")
    assert names, f"no modules discovered under {PYTHON_ROOT}; this check would pass vacuously"
    return names


def _probe(names: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, *names],
        capture_output=True,
        text=True,
        timeout=300,
        env={"PYTHONPATH": str(PYTHON_ROOT), "PATH": "/usr/bin:/bin"},
        cwd=REPO,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# The torch boundary, measured
# ---------------------------------------------------------------------------


def test_only_the_declared_modules_need_torch_to_import():
    """A torch leak into a torch-free module is invisible until CI has no torch.

    Both directions are checked: a module that newly needs torch fails here, and so does one
    that stopped needing it while still being listed, because a stale exemption is how the
    list becomes decoration.
    """
    names = _module_names()
    result = _probe(names)

    assert not result["other_failure"], (
        "these modules failed to import for a reason other than torch, so this check could "
        f"not decide anything about them: {result['other_failure']}"
    )
    assert set(result["ok"]) | set(result["needed_torch"]) == set(names), (
        "the probe did not account for every module"
    )
    assert set(result["needed_torch"]) == set(TORCH_GATED), (
        f"modules needing torch at import: {sorted(result['needed_torch'])}, "
        f"declared: {sorted(TORCH_GATED)}"
    )


@pytest.mark.parametrize(
    "module",
    ["qd_train.byte_batch", "qd_train.calibration_fit", "qd_train.schema_mirror"],
)
def test_the_modules_that_restated_constants_now_import_without_torch(module: str):
    """The regression this file exists for: each of these either broke or duplicated."""
    result = _probe([module])
    assert result["ok"] == [module], f"{module} did not import without torch: {result}"


def test_schema_mirror_imports_nothing_from_the_package():
    """Anything may depend on it, so it depends on nothing — that is what keeps it reachable."""
    source = (PYTHON_ROOT / "qd_train" / "schema_mirror.py").read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+(\S+)", source, re.M)
    assert imports, "no imports parsed; the check would pass vacuously"
    assert set(imports) <= {"__future__", "typing"}, (
        f"schema_mirror imports {imports}; a dependency here can put torch back on the path"
    )


# ---------------------------------------------------------------------------
# The constants themselves
# ---------------------------------------------------------------------------


def test_the_integer_constants_still_match_schema_rs():
    src = SCHEMA_RS.read_text(encoding="utf-8")

    def const(name: str) -> int:
        found = re.search(rf"pub const {name}:\s*\w+\s*=\s*(\d+)\s*;", src)
        assert found, f"{name} not found in {SCHEMA_RS}"
        return int(found.group(1))

    assert const("MAX_OPTIONS") == MAX_OPTIONS
    assert const("MIN_OPTIONS") == MIN_OPTIONS
    assert const("RESERVED_NOUL_ROWS") == RESERVED_NOUL_ROWS


def test_the_slot_kind_strings_still_match_slotkind_as_str():
    """These are wire-visible: they key the calibration table and every serialised schema."""
    src = SCHEMA_RS.read_text(encoding="utf-8")
    emitted = set(re.findall(r"SlotKind::\w+\s*=>\s*\"(\w+)\"", src))
    assert emitted, "no SlotKind::as_str arms parsed; the check would pass vacuously"
    assert emitted == {CHOICE, SCORE, SPAN}, (
        f"schema.rs emits {sorted(emitted)}, this module mirrors "
        f"{sorted({CHOICE, SCORE, SPAN})}"
    )


def test_heads_and_calibration_fit_re_export_the_same_object():
    """One name, one quantity — checked by identity, not by equality of two copies."""
    from qd_train import calibration_fit, schema_mirror

    assert calibration_fit.RESERVED_NOUL_ROWS is schema_mirror.RESERVED_NOUL_ROWS
    assert calibration_fit.MAX_OPTIONS is schema_mirror.MAX_OPTIONS
    assert calibration_fit.CHOICE is schema_mirror.CHOICE


def test_the_data_lanes_option_cap_is_the_same_number_under_its_other_name():
    """``qd_data.schema.MAX_CHOICE_OPTIONS`` is this module's ``MAX_OPTIONS`` renamed.

    One quantity under two names is this repository's recurring defect, so the equality
    is stated rather than left to be inferred. It is NOT unguarded today, and a lane
    that reported it as such had searched ``python/tests`` and ``python/qd_train`` --
    not ``crates/`` -- where the pin actually lives:
    ``crates/qd-runtime/tests/wire_context_crosslang.rs::
    the_option_and_bin_bounds_are_the_same_number_on_both_sides`` imports the real
    ``qd_data.schema`` at test time and compares it to ``schema.rs::MAX_OPTIONS``, and
    three module-scope asserts in ``qd_data/schema.py`` tie the cap to
    ``OPTION_LETTERS`` and ``MAX_SCORE_BINS``. Both were measured on 2026-09-19 by
    mutation: 17 there fails 8 cross-language tests and errors 12 Python modules at
    collection; 15 there errors 12 Python modules at collection.

    What this adds is independence from a skip. The cross-language pin is guarded by
    ``doc_or_skip!``, which returns early when ``.venv/bin/python`` is absent -- and
    when it skips, nothing left running compares the two Python names to each other.
    This assertion holds in every environment that can run the Python suite at all,
    which is the environment this file is for.
    """
    from qd_data.schema import MAX_CHOICE_OPTIONS

    assert MAX_CHOICE_OPTIONS == MAX_OPTIONS, (
        "qd_data.schema.MAX_CHOICE_OPTIONS and qd_train.schema_mirror.MAX_OPTIONS are "
        f"one quantity under two names, and they now read {MAX_CHOICE_OPTIONS} and "
        f"{MAX_OPTIONS}. Both mirror crates/qd-runtime/src/schema.rs::MAX_OPTIONS; move "
        "that one first."
    )


def test_there_is_exactly_one_definition_of_each_constant_in_python():
    """A second assignment anywhere is the shape this whole change removes.

    Keyed on the NAME, and so blind to a rename by construction:
    ``qd_data.schema.MAX_CHOICE_OPTIONS`` is a second assignment of ``MAX_OPTIONS``'s
    quantity that this loop cannot see. That one is held by value in
    ``test_the_data_lanes_option_cap_is_the_same_number_under_its_other_name``. Widening
    the loop to catch renames in general would mean matching on the literal ``16``,
    which every unrelated 16 in the tree would trip.
    """
    for name in ("RESERVED_NOUL_ROWS", "MAX_OPTIONS", "MIN_OPTIONS"):
        definitions = []
        for package in ("qd_train", "qd_data", "qd_wire"):
            for path in sorted((PYTHON_ROOT / package).glob("*.py")):
                body = path.read_text(encoding="utf-8")
                if re.search(rf"^{name}(?::\s*[^=]+)?\s*=\s*\d+", body, re.M):
                    definitions.append(f"{package}.{path.stem}")
        assert definitions == ["qd_train.schema_mirror"], (
            f"{name} is assigned in {definitions}; it must have exactly one Python owner"
        )
