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
import hashlib
import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

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

    The scope below was four names, hardcoded. It happens to be exactly right -- four tools
    build a `CostEstimate` and they are those four -- but it is right because it is current,
    not because anything keeps it so, and its twin in this file was four names that had
    stopped being all of them. Derived now, from tools that CONSTRUCT an estimate.
    """
    runners = _pricing_tools()

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

    **And then it happened to this test.** The list below was four names, written down when
    four runners were the ones in view. Six tools write ledger rows: the two it omitted --
    `rung0_linear_control.py`, which runs on the same rented box beside the arm it
    interprets, and `real_tokenizer_pipeline.py`, which writes the shard sets every FT run
    trains on -- recorded no digest at all, and this test was green throughout. A hardcoded
    scope is the defect it was written to catch, one level up.

    So the scope is derived. Any tool that constructs a recorder is in it, and a seventh
    runner is covered by existing rather than by being remembered.

    **And then it happened a third time, one level deeper.** Every runner recorded the
    digest, and every runner recorded it at the END of its training block -- so a run that
    died never recorded it at all. `RunRecorder._on_signal` writes the row immediately, and
    the digest was not yet among the metrics. `ledger/gh200-commitpackft-2026-09-22.jsonl`
    carries the proof: an `ft` row, `code_that_ran: None`, notes `received SIGTERM`. Six
    tools each remembering to do a thing is six chances to do it in the wrong place.

    What is asserted therefore moved from "the tool names the digest" to "the tool hands
    the recorder its entry point", because `RunRecorder.__enter__` now takes the digest
    before the work and before the signal handlers exist. That is strictly stronger: it
    covers the killed and failed paths this check could never have reached.
    `test_provenance_on_every_exit.py` holds the behaviour end of it.
    """
    runners = _row_writing_tools()
    # Whitespace-tolerant: the same call wraps across several lines where the indentation
    # is deeper. A pattern that failed on line breaks would be satisfied again by a
    # reformat, which is a worse failure than the one it guards against.
    call = re.compile(r"entry_point\s*=\s*Path\(__file__\)")
    missing = [
        name
        for name in runners
        if not call.search((TOOLS / name).read_text(encoding="utf-8"))
    ]
    assert not missing, (
        f"{missing} write ledger rows without handing RunRecorder their entry point, so "
        "those rows are pinned only by code_commit -- which reads '<sha>-dirty' for any "
        "uncommitted change and is permanently dirty on the box that runs them"
    )


#: Every place that states a `wall_clock_s` to a recorder, and whether the recorder's block
#: therefore CONTAINS the work it records. `None` means "time the block yourself", which is
#: the only arrangement in which a run killed part-way through still writes a row --
#: `RunRecorder`'s guarantee is a property of the block, so work outside it is work whose
#: death goes unrecorded. A measured figure means the work finished before the recorder
#: existed.
#:
#: Neither answer is wrong in general, which is why this is an inventory and not a ban: a
#: verdict row reporting a decode that has already happened must NOT repeat its parent's
#: duration, and `real_ft_run.py` says so where it does it. What matters is that every site
#: passing a measured figure has a reason, and that a NEW one cannot appear without
#: somebody writing the reason down.
NOT_WRAPPING = {
    ("real_ft_run.py", "decode_s"): (
        "the verdict row, an addendum to an ft row whose own block wraps train_ft. Its "
        "decode is the only billed work left outside a block, and it was sized from rows "
        "rather than assumed: 170 verdict rows in ledger/ carry 0.1s between them against "
        "40919.1s on the 194 ft rows they report on"
    ),
    ("ft_toy_run.py", "decode_s"): (
        "the same verdict shape on the toy path, which is local and priced at zero"
    ),
    ("rung0_toy_run.py", "float(run['wall_clock_s'])"): (
        "a toy run that cannot be billed: `_control` prices through "
        "`CostEstimate.for_device` with no rate arguments, which refuses anything outside "
        "cpu and mps rather than inventing one"
    ),
    ("real_tokenizer_pipeline.py", "work_s"): (
        "tokenises on whatever machine it is run on with `cost=None`, which the recorder "
        "accepts only on a local device and refuses on anything billed by the hour"
    ),
    ("rung0_linear_control.py", "time.monotonic() - work_t0"): (
        "fits a linear control on `Environment.detect(device=\"cpu\")` with `cost=None`; "
        "the device is not a parameter, so nothing here can be rented"
    ),
}

#: The sites that wrap. `rung0_real_run.py` joined this list at f93b22c; before that a run
#: killed during training wrote no row and recorded no cost.
WRAPPING = {
    ("rung0_real_run.py", "None"),
    ("real_ft_run.py", "None"),
    ("ft_toy_run.py", "None"),
    ("ledger.py", "None"),
}


def _row_writing_tools() -> tuple[str, ...]:
    """Every tool in `tools/` that constructs a recorder, hence writes ledger rows.

    Derived rather than listed. Two tests here carried a hardcoded four-name tuple written
    when four runners were the ones in view; six tools write rows, and the two omitted ones
    had no closure digest while both tests stayed green.
    """
    found: list[str] = []
    for path in sorted(TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
                node.func, "id", ""
            )
            if name == "RunRecorder" or (name.startswith("_") and "recorder" in name.lower()):
                found.append(path.name)
                break
    assert len(found) >= 6, (
        f"only {len(found)} row-writing tool(s) found ({found}), fewer than the six that "
        "existed when this was written -- either runners were removed or this stopped "
        "matching the way recorders are constructed, and every test scoped by it silently "
        "shrank"
    )
    return tuple(found)


def _pricing_tools() -> tuple[str, ...]:
    """Every tool in `tools/` that builds a `CostEstimate`, hence prices a run.

    A narrower question than :func:`_row_writing_tools` and deliberately kept separate:
    `real_tokenizer_pipeline.py` and `rung0_linear_control.py` write rows with `cost=None`
    on a local device, which the recorder accepts and which needs no rate. Requiring
    `for_device` of them would be requiring a price where there is nothing billed.

    Matched on CONSTRUCTION, not on the name appearing. `real_tokenizer_pipeline.py` and
    `run_cost.py` both mention `CostEstimate` without building one -- in a comment and in a
    helper's type, respectively -- so a substring scope would pull in two tools that have
    no rate to state.
    """
    found: list[str] = []
    for path in sorted(TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            builds = (
                getattr(func, "id", "") in {"CostEstimate", "run_cost_estimate"}
                or (isinstance(func, ast.Attribute)
                    and (getattr(func.value, "id", "") == "CostEstimate"
                         or func.attr == "for_device"))
            )
            if builds:
                found.append(path.name)
                break
    assert len(found) >= 4, (
        f"only {len(found)} tool(s) price a run ({found}), fewer than the four that did "
        "when this was written -- either pricing moved or this stopped matching it, and "
        "the check below silently narrowed"
    )
    return tuple(found)


def _recorder_wall_clock_sites() -> list[tuple[str, int, str]]:
    """`(file, line, source of the wall_clock_s argument)` for every recorder call.

    A call counts when it names `RunRecorder` or a helper whose name contains `recorder` --
    which is what `real_ft_run.py` and `ft_toy_run.py` route through, and where the choice
    is actually made. `TrainResult(wall_clock_s=...)` and `Rung0Result(wall_clock_s=...)`
    carry the same keyword and are results, not recorders; they are excluded by the same
    rule rather than by a filename list.
    """
    searched = sorted(TOOLS.glob("*.py")) + sorted((REPO / "python" / "qd_train").glob("*.py"))
    out: list[tuple[str, int, str]] = []
    for path in searched:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
                node.func, "id", ""
            )
            if name != "RunRecorder" and "recorder" not in name.lower():
                continue
            for kw in node.keywords:
                if kw.arg == "wall_clock_s":
                    out.append((path.name, kw.lineno, ast.unparse(kw.value)))
    return out


def test_every_recorder_that_can_be_billed_wraps_the_work_it_records() -> None:
    """The class behind GAP-A-KILLED-RUNG0-RUN-LEAVES-NO-ROW-AND-NO-COST.

    `tools/rung0_real_run.py` entered its recorder after `train_once` returned, so a run
    killed during training wrote nothing: no row, and no record of what the hour cost. It
    was one instance of a shape that nine call sites can take, and fixing the instance is
    not the same as fixing the shape -- this repository has found "a complete list was
    written down and then one member of it was fixed" seven times in one day.

    So this pins the whole inventory. Every site that states a measured duration is named
    with the reason its work is safe outside a block, and a new one fails here until
    somebody writes that reason down. Sites that pass through a helper are counted at the
    helper's CALL, which is where the decision is; `wall_clock_s=wall_clock_s` inside the
    helper is not a decision and is not counted.

    **Both numbers.** The count of sites found is asserted, not just the classification:
    a rename that made this match nothing would otherwise read exactly like a clean
    repository.
    """
    sites = _recorder_wall_clock_sites()
    # Two of the eleven are `wall_clock_s=wall_clock_s` inside `_recorder` helpers, which
    # forward whatever their caller decided. Both numbers are asserted: a helper that stops
    # forwarding, and a decision that disappears, are different failures.
    passthrough = [s for s in sites if s[2] == "wall_clock_s"]
    decisions = [s for s in sites if s[2] != "wall_clock_s"]
    assert len(sites) >= 11 and len(passthrough) >= 2 and len(decisions) >= 9, (
        f"found {len(sites)} recorder wall_clock_s site(s): {len(passthrough)} forwarding "
        f"and {len(decisions)} deciding, against 11 = 2 + 9 when this was written. Either "
        "recorders were removed or this stopped matching the way they are constructed"
    )

    wrapping = {(f, src) for f, _, src in decisions if src == "None"}
    stated = {(f, src) for f, _, src in decisions if src != "None"}

    unexplained = sorted(s for s in stated if s not in NOT_WRAPPING)
    assert not unexplained, (
        f"{unexplained} hand a recorder a duration measured before the block, so the work "
        "that duration describes happened outside the context manager that would have "
        "written its row. A run killed there leaves no row and no cost. Either wrap the "
        "work -- enter the recorder with wall_clock_s=None and call recorder.measured() "
        "when it returns -- or add an entry to NOT_WRAPPING saying why nothing billed can "
        "be lost here"
    )

    lost = sorted(WRAPPING - wrapping)
    assert not lost, (
        f"{lost} no longer wrap the work they record. rung0_real_run.py was fixed at "
        "f93b22c precisely because it did not, and the ledger was short by exactly the "
        "runs that were killed -- which are the ones that ran longest"
    )


def test_the_only_billed_work_outside_a_block_is_the_verdict_decode() -> None:
    """The claim the inventory above rests on, checked against the source rather than
    carried in a comment.

    Three of the five `NOT_WRAPPING` entries claim they cannot be billed. Two of those are
    checkable here: `real_tokenizer_pipeline.py` and `rung0_linear_control.py` both pass
    `cost=None`, which `RunRecorder.__init__` accepts only when the device is in
    `CostEstimate.LOCAL_DEVICES` and refuses otherwise -- so they cannot silently start
    pricing a rented machine. `rung0_toy_run.py` reaches its rate through
    `CostEstimate.for_device` with no rate argument, which refuses anything but cpu and
    mps.

    If one of them gains a `--usd-per-hour`, this fails and the entry in `NOT_WRAPPING`
    has to be re-argued rather than inherited.
    """
    for name in ("real_tokenizer_pipeline.py", "rung0_linear_control.py"):
        source = (TOOLS / name).read_text(encoding="utf-8")
        assert "cost=None" in source, (
            f"{name} no longer passes cost=None, so its NOT_WRAPPING entry -- which says "
            "nothing billed can be lost there -- is no longer supported by the source"
        )
        # Both spellings. `add_argument("--usd-per-hour")` produces `args.usd_per_hour`
        # and puts neither underscore spelling in the source, so a check for the
        # identifier alone never fires on the way a rate actually arrives -- which is how
        # this was found: the injected flag passed the first version of this line.
        rate = [s for s in ("usd_per_hour", "usd-per-hour", "usd_per_gpu_hour",
                            "usd-per-gpu-hour") if s in source]
        assert not rate, (
            f"{name} now takes a rate ({rate}), so work outside its recorder block is "
            "billed work and a kill there loses both the row and the spend"
        )

    toy = (TOOLS / "rung0_toy_run.py").read_text(encoding="utf-8")
    assert "CostEstimate.for_device(cap=cap, device=device)" in toy, (
        "rung0_toy_run.py no longer prices through for_device with no rate, so it may now "
        "be pointable at hardware that is billed by the hour -- and its training happens "
        "before `_record` opens a recorder"
    )


# ---------------------------------------------------------------------------------------
# GAP-SIX-SPELLINGS-OF-ONE-RECIPE-HASH, made safe to leave open.
#
# Six tools turn a recipe into the hash that identifies a run, through six independent
# implementations, and two of them disagree on identical dicts: `sort_keys=True` alone
# against `sort_keys=True, separators=(",", ":")`. Measured, on the same eight-key recipe:
# b5e6c489ecb55fca642c against 705dc18f8726e641db80.
#
# Nothing is wrong today. Each tool is internally consistent, and cross-tool comparisons
# are foreclosed by `backbone_commit` anyway. The danger is entirely in the future: a
# refactor that tidies one spelling toward another renames every `recipe_hash` that tool
# writes from then on, every row afterwards is incomparable with every row before, both
# files still verify their chains, and nothing says so.
#
# Unifying them is a DECLARED break -- 61 capacity rows and 24 learning-curve rows were
# written under `sort_keys=True` alone on 2026-09-21 -- and not something to do quietly
# mid-experiment. What can be done without touching a byte is make the quiet version
# impossible: pin the digest each spelling produces for a fixed probe, so the tidy-up
# fails here and has to be argued for rather than merged.
# ---------------------------------------------------------------------------------------

#: A fixed dict with the shapes a recipe actually contains -- ints, floats, bools, None,
#: strings, and two keys out of sorted order so `sort_keys` is observable. Its digest under
#: each spelling is pinned below.
_PROBE = {
    "span_weight": 0.05,
    "epochs": 30,
    "deterministic": False,
    "train_subsample": None,
    "rev": "HEAD",
    "batch_size": 8,
}

#: `{tool: json.dumps kwargs}` -- the spelling each tool's existing rows were written
#: under. Changing an entry is how the break gets declared.
_SPELLINGS = {
    "rung0_real_run.py": {"sort_keys": True},
    "rung0_linear_control.py": {"sort_keys": True},
    "real_tokenizer_pipeline.py": {"sort_keys": True},
    "real_ft_run.py": {"sort_keys": True, "separators": (",", ":")},
    "rung0_toy_run.py": {"sort_keys": True, "separators": (",", ":")},
    "ft_toy_run.py": {"sort_keys": True, "separators": (",", ":")},
}


def _dumps_kwargs(node: ast.Call) -> dict[str, object] | None:
    """The `json.dumps` keyword arguments inside a hashing expression, or None."""
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        func = inner.func
        if isinstance(func, ast.Attribute) and func.attr == "dumps":
            out: dict[str, object] = {}
            for kw in inner.keywords:
                if kw.arg:
                    out[kw.arg] = ast.literal_eval(kw.value)
            return out
    return None


def _recipe_hash_spellings() -> dict[str, dict[str, object]]:
    """`{tool: json.dumps kwargs}` for every tool that hashes a recipe.

    Structural, via the AST, rather than textual: the expression is found by following
    `recipe_hash=` (or the `digest` helper the toy runners route it through) to the
    `json.dumps` inside it. Reformatting the call does not move this; changing what it
    serialises does, which is the only event worth failing on.
    """
    out: dict[str, dict[str, object]] = {}
    for path in sorted(TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        helpers = {
            n.name: n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in {"digest", "_recipe_hash"}
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords}
            expr = keywords.get("recipe_hash")
            if expr is None:
                continue
            if isinstance(expr, ast.Call):
                found = _dumps_kwargs(expr)
                if found is not None:
                    out[path.name] = found
                    break
                callee = getattr(expr.func, "id", "")
                if callee in helpers:
                    found = _dumps_kwargs(
                        next(n for n in ast.walk(helpers[callee]) if isinstance(n, ast.Call))
                    )
                    if found is not None:
                        out[path.name] = found
                        break
            elif isinstance(expr, ast.Name):
                # A named variable: find its assignment in the same module.
                for assign in ast.walk(tree):
                    if isinstance(assign, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == expr.id for t in assign.targets
                    ):
                        found = _dumps_kwargs(assign.value) if isinstance(
                            assign.value, ast.Call
                        ) else None
                        if found is not None:
                            out[path.name] = found
                            break
                if path.name in out:
                    break
    return out


def test_every_tool_that_hashes_a_recipe_is_found_by_this_check() -> None:
    """Scope first, and asserted, because everything below is vacuous without it.

    Six tools produce a `recipe_hash`. A seventh that this stopped finding would read
    exactly like a repository with six.
    """
    found = _recipe_hash_spellings()
    assert set(found) == set(_SPELLINGS), (
        f"tools hashing a recipe: found {sorted(found)}, pinned {sorted(_SPELLINGS)}. A "
        "tool that appeared needs a spelling entry; one that vanished needs this list "
        "shortened deliberately"
    )


@pytest.mark.parametrize("tool", sorted(_SPELLINGS))
def test_each_tool_keeps_the_spelling_its_existing_rows_were_written_under(
    tool: str,
) -> None:
    """The guard that makes the gap safe to leave open.

    `json.dumps(recipe, sort_keys=True)` and the same call with
    `separators=(",", ":")` produce DIFFERENT bytes and therefore different hashes for
    identical dicts. Every row a tool has written is under one of them, so changing which
    renames that tool's protocol family from then on -- silently, because both the old and
    the new rows hash correctly and verify.

    Failing here is not "you may not change this". It is "this is the change you are
    making", which is the sentence that was missing.
    """
    expected_kwargs = _SPELLINGS[tool]
    assert _recipe_hash_spellings()[tool] == expected_kwargs, (
        f"{tool} now serialises its recipe with different json.dumps arguments. Every "
        f"recipe_hash it writes from here differs from every one it has written, both "
        f"verify, and nothing in the ledger says the family changed. If that is intended, "
        f"update this entry and declare the break in a handoff"
    )


def test_the_two_spellings_really_do_disagree() -> None:
    """The gap as a number, not as prose.

    If these ever produced the same digest the whole concern would be imaginary and the
    six implementations could be unified with no cost. They do not, and this is where that
    is established rather than asserted -- on the same dict, through both spellings.
    """
    digests = {
        name: hashlib.sha256(json.dumps(_PROBE, **kwargs).encode("utf-8")).hexdigest()
        for name, kwargs in _SPELLINGS.items()
    }
    distinct = set(digests.values())
    assert len(distinct) == 2, (
        f"expected exactly two distinct recipe digests across the six spellings, got "
        f"{len(distinct)}: {digests}"
    )
    loose = hashlib.sha256(
        json.dumps(_PROBE, sort_keys=True).encode("utf-8")
    ).hexdigest()
    tight = hashlib.sha256(
        json.dumps(_PROBE, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert loose != tight
    assert distinct == {loose, tight}
    # Three tools on each side, which is the split worth knowing: it is not one outlier.
    assert sorted(digests.values()).count(loose) == 3, digests
