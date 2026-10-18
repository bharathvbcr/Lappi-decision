"""Every v6 ``qd-mutate generate`` build takes K from data/decisions/v6-mutate.json (Fable Q3).

The flag has no default in the binary, so a caller that forgets it is refused there; what
nothing refused was a caller passing a *different* K. K lived as a literal in
``tools/rung0_real_run.py``. These tests make the config the only source.

``tools/rung0_real_run.py`` is the only caller that runs ``qd-mutate generate`` for a v6 build.
The pre-registrations under ``campaign/`` and the ``data/pool`` manifests describe builds that
already ran (v5/Arm-B pins) and are not callers.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "data" / "decisions" / "v6-mutate.json"


def _non_test_python() -> list[Path]:
    roots = [REPO / "tools", REPO / "python"]
    return sorted(
        p
        for root in roots
        for p in root.rglob("*.py")
        if "tests" not in p.relative_to(REPO).parts and ".venv" not in p.parts
    )


def test_no_caller_carries_a_literal_k() -> None:
    literal = re.compile(r"""--max-per-function["']?\s*,?\s*["']?\d""")
    offenders = [
        str(p.relative_to(REPO))
        for p in _non_test_python()
        if literal.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"K must come from data/decisions/v6-mutate.json: {offenders}"


def test_every_script_that_runs_generate_reads_k_from_the_config() -> None:
    runs_generate = re.compile(r"""["']generate["']\s*,""")
    callers = [p for p in _non_test_python() if runs_generate.search(p.read_text(encoding="utf-8"))]
    assert callers, "no qd-mutate generate caller found: the scan covers nothing"
    missing = [
        str(p.relative_to(REPO))
        for p in callers
        if "generate_command" not in p.read_text(encoding="utf-8")
    ]
    assert missing == [], f"runs qd-mutate generate without the configured K: {missing}"


def test_the_committed_config_carries_k_2() -> None:
    from qd_data.mutate_build import load_max_per_function

    assert load_max_per_function(CONFIG) == 2


@pytest.mark.parametrize(
    "document",
    [
        {"schema": "qd-mutate-build-config/v1"},
        {"schema": "qd-mutate-build-config/v1", "max_per_function": 0},
        {"schema": "qd-mutate-build-config/v1", "max_per_function": "2"},
        {"schema": "qd-mutate-build-config/v1", "max_per_function": True},
        {"max_per_function": 2},
    ],
)
def test_a_config_without_a_valid_k_is_refused(tmp_path: Path, document: dict[str, object]) -> None:
    from qd_data.mutate_build import load_max_per_function

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_max_per_function(path)
    with pytest.raises(ValueError):
        load_max_per_function(tmp_path / "absent.json")




def test_the_generate_command_carries_the_configured_k(tmp_path: Path) -> None:
    from qd_data.mutate_build import generate_command

    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"schema": "qd-mutate-build-config/v1", "max_per_function": 7}), encoding="utf-8"
    )
    cmd = generate_command(
        Path("qd-mutate"), pool=tmp_path / "p", out=tmp_path / "o", manifest=tmp_path / "m",
        seed=0, limit=1, config=cfg,
    )
    assert cmd[cmd.index("--max-per-function") + 1] == "7"
    cfg.write_text(json.dumps({"schema": "qd-mutate-build-config/v1"}), encoding="utf-8")
    with pytest.raises(ValueError):
        generate_command(
            Path("qd-mutate"), pool=tmp_path / "p", out=tmp_path / "o", manifest=tmp_path / "m",
            seed=0, limit=1, config=cfg,
        )


def test_rung0_forwards_its_config_into_the_command() -> None:
    tree = ast.parse((REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "generate_examples"
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "generate_command"
    ]
    assert len(calls) == 1
    forwarded = {kw.arg: kw.value for kw in calls[0].keywords}
    config = forwarded.get("config")
    assert isinstance(config, ast.Name) and config.id == "mutate_config"
