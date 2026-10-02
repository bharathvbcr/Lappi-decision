"""``tools/qd_train_oracle_trainer.py`` still writes the committed fixture, byte for byte.

``crates/qd-train/tests/fixtures/trainer-oracle.json`` is what the Rust trainer's bit-exact
tests read (``schedule_oracle.rs``, ``adamw_oracle.rs``, ``ledger_oracle.rs``,
``pyjson_oracle.rs``). A change to the dumper that moves a byte of its output -- a lint
reflow that splits a string literal wrongly, a reordered random draw -- would leave the
fixture stale while every Rust test still passed against it. This regenerates the fixture
into a temporary file and compares bytes.

**A same-host claim.** The fixture records the Python and torch it was written with; the
schedule's cosine half is the platform libm's (``math.cos``) and the AdamW half is torch's
CPU kernels. On another Python, another torch or another OS the comparison is not the one
the fixture makes, so the test **skips** there with the reason -- it does not pass.

Torch-gated (the ``adamw`` and ``clip`` cases run ``torch.optim.AdamW`` on CPU)::

    uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest \\
        python -m pytest python/tests/test_qd_train_oracle_trainer.py -o addopts= -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import qd_train_oracle_trainer as oracle  # noqa: E402

FIXTURE = REPO / "crates" / "qd-train" / "tests" / "fixtures" / "trainer-oracle.json"


def _recorded() -> tuple[str, str]:
    body = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return str(body["python"]), str(body["adamw"]["torch"])


def test_the_dump_is_the_committed_fixture_byte_for_byte(tmp_path: Path) -> None:
    python, torch_version = _recorded()
    here = (sys.version.split()[0], torch.__version__)
    if sys.platform != "darwin" or here != (python, torch_version):
        pytest.skip(
            f"the fixture was written on darwin with python {python} and torch "
            f"{torch_version}; this is {sys.platform} with python {here[0]} and torch "
            f"{here[1]}, where math.cos and torch's CPU kernels are not the ones it pins"
        )
    out = tmp_path / "trainer-oracle.json"
    assert oracle.main(["--out", str(out)]) == 0
    got, want = out.read_bytes(), FIXTURE.read_bytes()
    if got != want:
        a, b = json.loads(got), json.loads(want)
        moved = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        pytest.fail(
            f"the dumper writes {len(got)} bytes and the fixture holds {len(want)}; the "
            f"top-level keys that differ are {moved}. Regenerate the fixture with the "
            "command in the dumper's docstring and update its pin in "
            "crates/qd-train/tests/fixture_pins.rs, or undo the change that moved them"
        )
