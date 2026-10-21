"""The QD_PYTHON resolver is one text in three crates, and this pins it.

``crates/qd-prep/tests/convert_licence_parity.rs``, ``crates/qd-metal/tests/ledger_rows.rs`` and
``crates/qd-preflight/tests/schema_compat.rs`` each resolve the repo's Python for a test that runs
a Python oracle. They were three hand copies of ``QD_PYTHON, then <repo>/.venv, then python3``, and
in a linked worktree (no ``.venv``) all three failed as environment. There is no shared test-support
crate to own the logic and a new workspace member was not authorised, so the block between the
``qd-python-resolver`` markers is kept byte-identical and this test fails when one copy drifts.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SITES = (
    "crates/qd-prep/tests/convert_licence_parity.rs",
    "crates/qd-metal/tests/ledger_rows.rs",
    "crates/qd-preflight/tests/schema_compat.rs",
)
BLOCK = re.compile(
    r"^// BEGIN qd-python-resolver\n.*?^// END qd-python-resolver\n", re.DOTALL | re.MULTILINE
)


def _block(rel: str) -> str:
    found = BLOCK.findall((REPO / rel).read_text(encoding="utf-8"))
    assert len(found) == 1, (
        f"{rel}: expected exactly one qd-python-resolver block, got {len(found)}"
    )
    return found[0]


def test_the_three_resolver_copies_are_identical() -> None:
    blocks = {rel: _block(rel) for rel in SITES}
    first = blocks[SITES[0]]
    for rel, text in blocks.items():
        assert text == first, f"{rel} differs from {SITES[0]}"


def test_the_resolver_order_and_loud_failure_are_stated_in_the_block() -> None:
    text = _block(SITES[0])
    qd_python = text.index("qd_python")
    own = text.index('repo.join(".venv/bin/python")')
    main = text.index("main_checkout(repo)")
    system = text.index('PathBuf::from("python3")')
    assert qd_python < own < main < system, (
        "order: QD_PYTHON, own .venv, main checkout .venv, python3"
    )
    assert "--git-common-dir" in text
    for rel in SITES:
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "QD_PYTHON" in src.replace(text, ""), f"{rel}: no loud failure naming QD_PYTHON"
        assert "fn python_candidates" not in src.replace(text, ""), f"{rel}: a second resolver"
