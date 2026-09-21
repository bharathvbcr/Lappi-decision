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
import re
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


def test_the_scope_of_the_check_above_is_stated_rather_than_assumed() -> None:
    """What that test does NOT look at, said out loud, because someone read its output as a
    complete call-site list and shipped a change against it.

    On 2026-09-21 a lane made ``RunRecorder``'s ``cost`` required, took the list of call
    sites from the test above, updated all six, and turned every ``make gates`` run in the
    repository red -- because the seventh caller is ``record_build_run`` inside
    ``ledger.py``, which that test does not scan and never claimed to. The tool was right;
    its scope was narrower than the question, and nothing in its output said so.

    That is the repository's own capped-sample rule, applied to a test instead of to a
    number: `checked` and `unchecked` were both reported and both were about ``tools/``.
    """
    source = (Path(__file__)).read_text(encoding="utf-8")
    assert 'TOOLS.glob("*.py")' in source, (
        "the scan above no longer globs tools/, so this description of its scope is stale"
    )
    # The in-package half now exists below. If it is ever deleted, this says what goes with
    # it rather than leaving the pair silently halved.
    assert "def test_every_in_package_call_binds_against_its_callee" in source, (
        "the in-package half of this check is gone, so calls between package modules -- "
        "the ones that broke the gate -- are unchecked again and nothing says so"
    )


def test_every_in_package_call_binds_against_its_callee() -> None:
    """The half the tools/ scan cannot see: a package module calling its own definitions.

    ``ledger.py`` constructs ``RunRecorder`` from ``record_build_run``. That name is not
    imported from anywhere -- it is defined in the same file -- so the import-driven scan
    above cannot reach it even if it were pointed at ``python/``. This resolves names
    DEFINED at module level and binds calls to them against their real signatures.

    **Conservative on purpose.** A name assigned anywhere in the file, at any depth, is
    skipped: a local variable shadowing a module-level function inside some other function
    would otherwise bind against the wrong object and report a failure that is not real. A
    test that cries wolf here gets muted, and muted is worse than narrow. The skips are
    counted and reported, so the narrowness is visible rather than implied.
    """
    files = sorted(
        path
        for package in PACKAGES
        for path in (REPO / "python" / package).glob("*.py")
        if path.name != "__init__.py"
    )
    assert files, "no package module found; this test would pass vacuously"

    checked = 0
    unchecked: list[str] = []
    failures: list[str] = []

    for path in files:
        module_name = f"{path.parent.name}.{path.stem}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        # Any rebinding at any depth disqualifies the name -- see the docstring.
        assigned: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assigned.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned.update(a.arg for a in node.args.args)
                assigned.update(a.arg for a in node.args.kwonlyargs)
        candidates = defined - assigned
        if not candidates:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # torch-gated modules in the repo venv, and they are named
            unchecked.append(f"{module_name}: not importable here ({type(exc).__name__})")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name not in candidates:
                continue
            target = getattr(module, name, None)
            if not callable(target):
                unchecked.append(f"{module_name}:{node.lineno} {name}: not callable")
                continue
            shape = _call_shape(node)
            if shape is None:
                unchecked.append(f"{module_name}:{node.lineno} {name}: *args/**kwargs")
                continue
            n_pos, keywords = shape
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError) as exc:
                unchecked.append(f"{module_name}:{node.lineno} {name}: no signature ({exc})")
                continue
            try:
                signature.bind(*[SENTINEL] * n_pos, **dict.fromkeys(keywords, SENTINEL))
            except TypeError as exc:
                failures.append(
                    f"python/{module_name.replace('.', '/')}.py:{node.lineno}: {name}("
                    + ", ".join(["…"] * n_pos + [f"{k}=…" for k in keywords])
                    + f") does not bind against {name}{signature}: {exc}"
                )
            else:
                checked += 1

    assert not failures, (
        f"{len(failures)} in-package call site(s) do not match the signature they call "
        f"({checked} checked, {len(unchecked)} unchecked). These are invisible to the "
        f"tools/ scan above, which is how a required argument reached every gate run "
        f"before anyone noticed:\n  " + "\n  ".join(failures)
    )
    assert checked > 0, (
        "no in-package call site was checkable, so this test proves nothing. Unchecked:\n  "
        + "\n  ".join(unchecked[:20])
    )
    print(f"packages: {checked} call site(s) bound, {len(unchecked)} unchecked")


# -- the price every tool puts on the machine it ran on -----------------------------------


def test_no_tool_prices_a_rented_machine_at_zero() -> None:
    """The class, not the case.

    Four tools reached ``CostEstimate`` through one literal --
    ``usd_per_hour=0.0, n_gpus=0, instance=f"local-{device}"`` -- and the commit that landed
    ``CostEstimate.for_device`` fixed one of them while its own docstring said there
    were four. That is what this asserts against, and it is not "a literal got written": a
    complete list was written down and then not used. The same shape twice in one day.

    Source-level for two reasons. The defect is a literal that reads as deliberate, so it
    survives review rather than a type check; and three of the four tools import torch,
    which the repo venv carries none of by design, so importing them here would turn this
    into a test that does not run.

    **Both numbers.** Every tool is read, the ones that mention ``CostEstimate`` are
    counted, and an empty set of them fails rather than passing vacuously -- a rename that
    made this match nothing would otherwise read exactly like a clean repository.
    """
    runners = ("real_ft_run.py", "rung0_real_run.py", "rung0_toy_run.py", "ft_toy_run.py")

    mentions: list[str] = []
    offenders: list[str] = []
    for path in sorted(TOOLS.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "CostEstimate" not in source:
            continue
        mentions.append(path.name)
        if "usd_per_hour=0.0, n_gpus=0" in source:
            offenders.append(path.name)

    assert mentions, (
        "no tool in tools/ mentions CostEstimate. Either they stopped pricing their runs "
        "or this test has been renamed out of checking anything"
    )
    assert not offenders, (
        f"{offenders} price a rented box at zero on zero GPUs. Those two values make "
        "requires_human_approval False for ANY cap -- an 8xH100 job included -- and skip "
        "the per-GPU column check, and they record the machine as a local one"
    )

    missing = [
        name
        for name in runners
        if "CostEstimate.for_device(" not in (TOOLS / name).read_text(encoding="utf-8")
    ]
    assert not missing, (
        f"{missing} build a cost estimate without going through for_device, which is the "
        "one place that refuses to invent a rate for hardware rented by the hour"
    )


def test_every_runner_records_which_sources_produced_its_row() -> None:
    """The same class as the price above, and found the same way -- by the other lane
    noticing that a fix had been applied to one member of a list that was already written
    down.

    `tools/real_ft_run.py` had this as two private functions and was the only runner with
    it, while `tools/rung0_real_run.py` -- the tool that wrote the GH200 rows whose
    provenance had to be reconstructed by hand with sha256sum -- had none.
    """
    runners = ("real_ft_run.py", "rung0_real_run.py", "rung0_toy_run.py", "ft_toy_run.py")
    # Whitespace-tolerant: the same call wraps across three lines where the indentation is
    # deeper. A pattern that failed on line breaks would be satisfied again by a reformat,
    # which is a worse failure than the one it guards against.
    call = re.compile(r'"code_that_ran",\s*what_ran_state\(')
    missing = [
        name
        for name in runners
        if not call.search((TOOLS / name).read_text(encoding="utf-8"))
    ]
    assert not missing, (
        f"{missing} write ledger rows without recording which sources produced them, so "
        "those rows are pinned only by code_commit -- which reads '<sha>-dirty' for any "
        "uncommitted change and is permanently dirty on the box that runs them"
    )
