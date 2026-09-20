"""Do the tools in ``tools/`` still call the library the way it is declared?

Nothing runs ``tools/*.py``. ``test_lint_gate.py`` points ruff at them, which catches an
unused import and cannot catch a call that no longer matches its callee, because ruff is
not a type checker. So a required keyword can be added to a ``qd_train`` function and a
tool left calling the old shape, and every gate stays green.

That happened. Commit ``e089d27`` made ``repo_root`` required on
``qd_train.artifacts.assert_shard_trainable`` -- closing a real rule-3 hole -- and landed
``tools/ft_toy_run.py`` in the **same commit** with the pre-change call still in it::

    File "tools/ft_toy_run.py", line 828, in _rule3_door
        assert_shard_trainable(header, config=config, path=under_holdout)
    TypeError: assert_shard_trainable() missing 1 required keyword-only argument: 'repo_root'

The tool died before writing any ledger row. What it died before writing was, among other
things, ``rule3_shard_door_checks_the_path`` -- the standing measurement of the very gap
that commit closed, which its own docstring promised would "flip to ``passed=True`` by
itself" when the gap closed. It could not: the change that closed the gap took the
measurement with it.

This asserts the class rather than the case: every call in ``tools/`` to a name imported
from this repository's packages must bind against that callee's current signature.

**Both numbers, never one.** A callee whose module cannot be imported here -- the repo venv
carries no torch on purpose, so ``qd_train.heads`` and friends are not importable -- is
counted as *unchecked* and named, not silently passed over. A capped sample reported as
complete coverage is the failure this repository keeps finding.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
PACKAGES = ("qd_data", "qd_train", "qd_wire", "qd_label")

#: Stand-in for an argument's value. Only arity and names are being checked, so what the
#: value is cannot matter -- but it must be a single object so `bind` sees one argument.
SENTINEL = object()


def _imported_names(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """``local name -> (module, attribute)`` for every ``from <pkg...> import ...``."""
    out: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root not in PACKAGES:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                out[alias.asname or alias.name] = (node.module, alias.name)
    return out


def _module_level_bindings(tree: ast.Module) -> set[str]:
    """Names the tool defines or rebinds itself, which an import does not then describe."""
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def _call_shape(call: ast.Call) -> tuple[int, list[str]] | None:
    """``(positional count, keyword names)``, or ``None`` when the call cannot be read.

    A call carrying ``*args`` or ``**kwargs`` is unreadable statically: its arity is a
    runtime fact. Those are counted as unchecked rather than assumed correct.
    """
    if any(isinstance(a, ast.Starred) for a in call.args):
        return None
    if any(kw.arg is None for kw in call.keywords):
        return None
    return len(call.args), [kw.arg for kw in call.keywords if kw.arg is not None]


def test_every_tool_call_into_this_repository_binds_against_its_callee() -> None:
    files = sorted(TOOLS.glob("*.py"))
    assert files, f"{TOOLS} holds no Python file; this test would pass vacuously"

    checked = 0
    unchecked: list[str] = []
    failures: list[str] = []
    cache: dict[str, object] = {}

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = _imported_names(tree)
        shadowed = _module_level_bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name not in imported or name in shadowed:
                continue
            module_name, attribute = imported[name]
            if module_name not in cache:
                try:
                    cache[module_name] = importlib.import_module(module_name)
                except Exception as exc:  # the reason is the point, so it is carried
                    cache[module_name] = exc
            module = cache[module_name]
            if isinstance(module, Exception):
                unchecked.append(
                    f"{path.name}:{node.lineno} {name}: {module_name} is not importable "
                    f"here ({type(module).__name__})"
                )
                continue
            target = getattr(module, attribute, None)
            if not callable(target):
                unchecked.append(f"{path.name}:{node.lineno} {name}: not callable")
                continue
            shape = _call_shape(node)
            if shape is None:
                unchecked.append(f"{path.name}:{node.lineno} {name}: *args/**kwargs")
                continue
            n_pos, keywords = shape
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError) as exc:
                unchecked.append(f"{path.name}:{node.lineno} {name}: no signature ({exc})")
                continue
            try:
                signature.bind(*[SENTINEL] * n_pos, **dict.fromkeys(keywords, SENTINEL))
            except TypeError as exc:
                failures.append(
                    f"{path.relative_to(REPO)}:{node.lineno}: {name}("
                    + ", ".join(["…"] * n_pos + [f"{k}=…" for k in keywords])
                    + f") does not bind against {module_name}.{attribute}{signature}: {exc}"
                )
            else:
                checked += 1

    assert not failures, (
        f"{len(failures)} call site(s) in tools/ do not match the signature they call "
        f"({checked} checked, {len(unchecked)} unchecked):\n  " + "\n  ".join(failures)
    )
    # The coverage pair, always. `checked` alone would read the same whether this test
    # examined every call or none of them.
    assert checked > 0, (
        "no call site was checkable, so this test proves nothing. Unchecked:\n  "
        + "\n  ".join(unchecked[:20])
    )
    print(f"tools/: {checked} call site(s) bound, {len(unchecked)} unchecked")
