"""No helper in ``tools/real_ft_run.py`` may turn a missing wall-clock cap into the quick cap.

GAP-RUN-CONTROL-DEFAULT-CAP-CALLERS-OUTSIDE-L-OPS-2026-10-17: ``main`` refuses a training run
that names no ``--wall-clock-cap-s`` unless it is ``--quick``, but ``_control``, ``_train`` and
``_recipe_pieces`` still defaulted ``cap_s`` to the 1,800 s quick cap, so any other caller
(``tools/qd_train_oracle_tiny.py`` called ``_control`` bare) reached the silent 30-minute stop
v1 died of. A missing cap must be a ``TypeError`` at the call site. Read from the AST, so the
check needs no torch and cannot be fooled by a monkeypatched module.

``_cost`` is protected and keeps its default; the second test proves every call to it in
``tools/`` hands the cap, which is the only way the default could matter.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
REAL_FT_RUN = TOOLS / "real_ft_run.py"

NO_DEFAULT = ("_control", "_train", "_recipe_pieces")


def _functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("name", NO_DEFAULT)
def test_the_helper_takes_cap_s_with_no_default(name: str) -> None:
    fn = _functions(ast.parse(REAL_FT_RUN.read_text(encoding="utf-8")))[name]
    kwonly = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)}
    assert "cap_s" in kwonly, f"{name} has no keyword-only cap_s"
    assert kwonly["cap_s"] is None, f"{name} still defaults cap_s: {ast.unparse(kwonly['cap_s'])}"


def _called_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def test_every_tools_caller_of_a_cap_taking_helper_passes_the_cap() -> None:
    """Across every file in tools/, so a new caller outside real_ft_run.py is caught too.
    ``**kwargs`` forwarding counts as passing it (the spy in a test wraps ``_control`` so)."""
    offenders: list[str] = []
    for path in sorted(TOOLS.glob("*.py")):
        if path.name in {"ft_toy_run.py", "rung0_toy_run.py"}:
            continue  # their own local _control(steps, *, device), a different function
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in (*NO_DEFAULT, "_cost"):
                continue
            if isinstance(node.func, ast.Name) and path != REAL_FT_RUN and name != "_cost":
                continue  # a same-named local helper in another tool
            passes = any(k.arg == "cap_s" or k.arg is None for k in node.keywords)
            if not passes:
                offenders.append(f"{path.name}:{node.lineno} {name}(...)")
    assert not offenders, "calls that rely on a default cap: " + ", ".join(offenders)
