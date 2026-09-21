"""A dependency this project declares and does not have is a check that quietly stopped.

This repository has the failure on record twice, in its own words.

``Makefile``, on why it exists at all: *"``ruff>=0.6`` sat declared in pyproject's ``dev``
extra and had never been installed, let alone run"* -- a gate that existed only in
configuration.

And again on 2026-09-21, more quietly. ``datasketch`` is declared in
``[project.dependencies]``. It is present in the repo's ``.venv`` and absent from the ml
venv, so ``test_minhash.py``'s module-level ``importorskip`` fired in the torch suite and
took **29 tests** with it. Pytest counts a module-level skip as ONE. The suite reported
``1850 passed, 2 skipped`` and the ledger row recorded a coverage pair of ``1850/1852`` --
understating the un-run tests by 28, in the very instrument this project built so that a
capped sample is never reported as complete coverage.

Nothing was actually untested: those 29 run in the torch-free suite, which has the package.
That is the point worth keeping. **The defect was in the reporting, and reporting is what
the coverage pair exists to be.** A number that says "2 did not run" when the answer is 30
is worse than no number, because it invites the reader to stop looking.

So this names any declared runtime dependency that cannot be imported *here*, in whichever
environment "here" is, and fails with the name. It cannot see the next module-level skip
directly -- no test can, since pytest never imports the module -- but it can see the cause,
which is always the same: something declared and not installed.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
MAKEFILE = REPO / "Makefile"

TIMEOUT_S = 300

#: Distribution name -> importable module name, where they differ. Only the exceptions;
#: everything else is the distribution name with hyphens turned into underscores.
IMPORT_NAMES = {"pyyaml": "yaml"}


def _requirement_name(spec: str) -> str:
    """The distribution name out of a PEP 508 requirement, without its version."""
    head = spec.split(";")[0].strip()
    for separator in (">=", "==", "<=", "~=", "!=", ">", "<", "["):
        head = head.split(separator)[0]
    return head.strip()


def _module_name(spec: str) -> str:
    name = _requirement_name(spec).lower()
    return IMPORT_NAMES.get(name, name.replace("-", "_"))


def _declared_runtime() -> list[str]:
    meta = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return list(meta["project"]["dependencies"])


#: The make target that provisions the torch environment. Named in the failure message
#: above, so `test_the_remedy_named_in_the_failure_message_is_a_real_target` checks it
#: resolves -- a remedy pointing at a renamed target is worse than none, because the
#: reader follows it and concludes the advice is stale rather than the target.
TORCH_TARGET = "torch-pytest"


def _torch_bridge_recipe() -> str:
    """What `make torch-pytest` expands to, asked of make rather than re-parsed.

    `-n` is a dry run, so this launches no suite.
    """
    proc = subprocess.run(
        ["make", "-f", str(MAKEFILE), "--no-print-directory", "-n", TORCH_TARGET],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, (
        f"`make -n {TORCH_TARGET}` exited {proc.returncode}, so the bridge could not be "
        f"read and nothing below was actually checked.\nstderr: {proc.stderr}"
    )
    return proc.stdout


def test_the_declaration_is_readable_and_not_empty() -> None:
    """Scope first. Every assertion below is vacuous over an empty list, and a key rename
    in `pyproject.toml` would make this pass by checking nothing."""
    declared = _declared_runtime()
    assert len(declared) >= 3, (
        f"{PYPROJECT} declares {len(declared)} runtime dependency/dependencies; three "
        "existed when this was written. Either they were removed or this stopped reading "
        "the right key"
    )


def test_every_declared_runtime_dependency_can_be_imported_here() -> None:
    """The class: declared and not installed.

    Failing rather than skipping, deliberately. A skip here would be the exact shape this
    file exists to catch -- a check that could not run, reported alongside the ones that
    ran and passed.

    The interpreter is named in the message because the answer differs between the two
    environments the gates use, and "datasketch is missing" without saying *where* sent one
    reader looking in the wrong virtualenv.

    **This test's result depends on how pytest was launched, and that is intended.** Under
    `make gates` and `make torch-pytest` it passes, because the torch bridge provisions
    what the ml venv lacks. Launched by hand against the ml venv without those flags it
    fails -- correctly, because in *that* interpreter a declared dependency really is
    absent and `test_minhash.py` really will collapse 29 tests into one skip marker. A peer
    lane published two gate rows on 2026-09-21 from exactly that hand-run environment; both
    understated their own denominator by 28, and nothing in the run said so. So the red is
    the feature, and the message below has to make the remedy obvious enough that nobody is
    tempted to convert it into a skip.
    """
    missing = [
        spec for spec in _declared_runtime()
        if importlib.util.find_spec(_module_name(spec)) is None
    ]
    assert not missing, (
        f"declared in [project.dependencies] and not importable by {sys.executable}: "
        f"{missing}. A declared dependency that is absent does not fail loudly -- it turns "
        "whatever imports it into a skip, and a module-level skip counts as ONE in the "
        "coverage pair however many tests are behind it.\n\n"
        f"If you launched pytest by hand, this is about your launcher and not the tree: "
        f"run `make {TORCH_TARGET}` (or `make gates`), which provisions these for the "
        "duration of the run and installs nothing into either virtualenv. Any suite count "
        "you quote from a run where this failed was measured in an environment missing a "
        "declared dependency, and its coverage pair understates the tests that did not "
        "run.\n\n"
        "Otherwise: install it here, add it to the bridge the way the Makefile does, or "
        "move it out of the runtime dependencies if nothing at runtime needs it. Do not "
        "make this a skip."
    )


def test_the_name_mapping_covers_what_is_declared() -> None:
    """`pyyaml` imports as `yaml`, and every other exception would be silent.

    Without this, a declared `some-dist` whose module is `somedist` would look missing and
    fail the test above for the wrong reason -- or, worse, a mapping entry left behind
    after a dependency was dropped would go unnoticed and mislead the next author.
    """
    declared = {_requirement_name(spec).lower() for spec in _declared_runtime()}
    stale = sorted(set(IMPORT_NAMES) - declared)
    assert not stale, (
        f"IMPORT_NAMES maps {stale}, which nothing declares any more. A mapping for a "
        "dependency that is gone is a claim about this project that is no longer true"
    )


def test_a_requirement_specifier_is_parsed_rather_than_guessed_at() -> None:
    """The parser above, on the forms PEP 508 actually permits.

    Checked on synthetic input because the three real ones all use `>=`, so a parser that
    handled nothing else would look correct for exactly as long as nobody pinned anything.
    """
    cases = {
        "numpy>=1.26": "numpy",
        "datasketch>=1.6": "datasketch",
        "pyyaml>=6.0": "pyyaml",
        "torch==2.12.1": "torch",
        "transformers~=4.57": "transformers",
        "uvicorn[standard]>=0.30": "uvicorn",
        "tomli; python_version < '3.11'": "tomli",
        "  spaced  >=  1.0  ": "spaced",
    }
    for spec, expected in cases.items():
        assert _requirement_name(spec) == expected, spec
    assert _module_name("pyyaml>=6.0") == "yaml"
    assert _module_name("flash-linear-attention>=0.1") == "flash_linear_attention"


def test_the_remedy_named_in_the_failure_message_is_a_real_target():
    """`make torch-pytest` has to exist, or the advice above sends readers nowhere.

    Checked by expanding it, not by grepping for the word: a target named in a comment and
    deleted from the file would still match a grep of the file.
    """
    recipe = _torch_bridge_recipe()
    assert "-m pytest" in recipe, (
        f"`make {TORCH_TARGET}` expands to something that runs no pytest, so the remedy "
        f"the failure message names would not provision anything:\n{recipe[:400]}"
    )


def test_the_torch_bridge_provisions_every_declared_dependency_the_ml_venv_lacks():
    """The general form of the datasketch bug, rather than datasketch.

    `uv run --with X` layers X on for the duration of the run. The bridge must name every
    declared runtime dependency that the torch interpreter does not already have, because
    each one it misses is a module-level `importorskip` waiting to collapse N tests into a
    single skip marker. Measured, on the commit where this was found: the same suite
    collected 1858 without `--with datasketch` and 1886 with it -- 29 real tests standing
    behind one marker, in the instrument built so that a capped sample is never reported as
    complete coverage.

    Both halves are asked of the system rather than assumed: the interpreter comes from
    make's own expansion of the bridge, and what it lacks is asked of that interpreter.
    Hardcoding either would make this agree with a Makefile that had been repointed.
    """
    recipe = _torch_bridge_recipe()

    interpreter = re.search(r"--python\s+(\S+)", recipe)
    assert interpreter, (
        f"`make {TORCH_TARGET}` names no --python, so this cannot tell which interpreter "
        f"the torch suite runs under and checked nothing:\n{recipe[:400]}"
    )
    ml_python = Path(interpreter.group(1))
    if not ml_python.is_file():
        pytest.skip(
            f"the torch environment at {ml_python} does not exist on this host, so what "
            f"it lacks cannot be asked. `make {TORCH_TARGET}` reports this as NotRun and "
            "exits 3 rather than passing; this is the same answer in pytest's vocabulary"
        )

    provisioned = set(re.findall(r"--with\s+([A-Za-z0-9._-]+)", recipe))

    probe = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
    unprovisioned = []
    for spec in _declared_runtime():
        name = _requirement_name(spec).lower()
        if name in {p.lower() for p in provisioned}:
            continue
        present = subprocess.run(
            [str(ml_python), "-c", probe, _module_name(spec)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        if present.returncode != 0:
            unprovisioned.append(name)

    assert not unprovisioned, (
        f"{unprovisioned} is/are declared in [project.dependencies], absent from "
        f"{ml_python}, and not passed as `--with` by `make {TORCH_TARGET}`. Every one is a "
        "module-level importorskip away from hiding a whole test module behind a single "
        "skip marker in the torch suite, which counts as ONE in the coverage pair however "
        f"many tests are behind it. Add `--with <name>` to TORCH_PYTEST_RUN in "
        f"{MAKEFILE.name} -- the one spelling both the target and the ledger's --suite "
        "argument are built from."
    )
