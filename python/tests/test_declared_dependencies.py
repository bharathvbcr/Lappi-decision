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
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"

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
    """
    missing = [
        spec for spec in _declared_runtime()
        if importlib.util.find_spec(_module_name(spec)) is None
    ]
    assert not missing, (
        f"declared in [project.dependencies] and not importable by {sys.executable}: "
        f"{missing}. A declared dependency that is absent does not fail loudly -- it turns "
        "whatever imports it into a skip, and a module-level skip counts as ONE in the "
        "coverage pair however many tests are behind it. Install it in this environment, "
        "provision it the way the Makefile's torch bridge does, or move it out of the "
        "runtime dependencies if nothing at runtime needs it"
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
