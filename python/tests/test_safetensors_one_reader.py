"""One safetensors header reader in Python (ft-e0211ba7412, ruling 2d).

`qd_train.memory` owns the reader. The checkpoint sidecar reader in `run_control` and
`tools/verify_checkpoint_inventory.py` delegate to it, and all three are held to the list the
Rust readers are held to: ``crates/qd-export/tests/fixtures/safetensors-adversarial.json``
(schema in ``crates/qd-export/tests/safetensors_adversarial.rs``). Python matches on refuse/ok
alone; the Rust error substrings are not part of the contract.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "crates/qd-export/tests/fixtures/safetensors-adversarial.json"
sys.path.insert(0, str(REPO / "tools"))

#: A refusal about the length prefix or the header text, which `safetensors_header` (the
#: header-only reader) answers; the rest need the data section and belong to `tensor_ranges`.
_HEADER_LEVEL = ("outside", "runs past", "too short", "not JSON", "not a JSON object")


def _cases() -> list[dict]:
    doc = json.loads(FIXTURE.read_bytes())
    assert doc["schema_version"] == 1
    cases = doc["cases"]
    assert len(cases) >= 20, f"the shared list shrank to {len(cases)}"
    return cases


def _build(case: dict, path: Path) -> Path:
    header = case.get("header", "")
    pad = case.get("pad_header_to")
    if pad is not None:
        assert len(header.encode()) <= pad
        header += " " * (pad - len(header.encode()))
    raw = header.encode("utf-16-le" if case.get("header_encoding") == "utf-16-le" else "utf-8")
    prefix = case.get("header_len", len(raw))
    blob = struct.pack("<Q", prefix) + raw + bytes(case.get("data_len", 0))
    cut = case.get("truncate_to")
    if cut is not None:
        blob = blob[:cut]
    path.write_bytes(blob)
    return path


def _header_level(case: dict) -> bool:
    if case["expect"] == "ok":
        return True
    return any(s in case.get("rust_error_contains", "") for s in _HEADER_LEVEL)


def _ids(cases: list[dict]) -> list[str]:
    return [c["name"] for c in cases]


ALL = _cases()


@pytest.mark.parametrize("case", ALL, ids=_ids(ALL))
def test_tensor_ranges_gives_the_shared_verdict_on_every_case(case, tmp_path):
    from qd_train.memory import CheckpointRefused, tensor_ranges

    f = _build(case, tmp_path / "x.safetensors")
    if case["expect"] == "ok":
        ranges = tensor_ranges(f)
        assert ranges, case["name"]
        for r in ranges.values():
            assert 8 <= r.start <= r.end <= f.stat().st_size
    else:
        with pytest.raises(CheckpointRefused):
            tensor_ranges(f)


HEADER = [c for c in ALL if _header_level(c)]


@pytest.mark.parametrize("case", HEADER, ids=_ids(HEADER))
def test_safetensors_header_gives_the_shared_verdict_on_the_header_level_cases(case, tmp_path):
    from qd_train.memory import CheckpointRefused, safetensors_header

    f = _build(case, tmp_path / "x.safetensors")
    if case["expect"] == "ok":
        assert isinstance(safetensors_header(f), dict)
    else:
        with pytest.raises(CheckpointRefused):
            safetensors_header(f)


# --- the two private readers, now delegates ---------------------------------------------------


@pytest.mark.parametrize("case", HEADER, ids=_ids(HEADER))
def test_verify_checkpoint_inventory_reads_headers_through_the_one_reader(case, tmp_path):
    import verify_checkpoint_inventory as tool

    from qd_train.memory import CheckpointRefused

    f = _build(case, tmp_path / "x.safetensors")
    if case["expect"] == "ok":
        assert isinstance(tool.safetensors_header(f), dict)
    else:
        # A refusal, not SystemExit: the inventory tool no longer owns a failure type.
        with pytest.raises(CheckpointRefused):
            tool.safetensors_header(f)


def test_verify_checkpoint_inventory_keeps_no_private_header_parse():
    import verify_checkpoint_inventory as tool

    from qd_train.memory import safetensors_header

    assert tool.safetensors_header is safetensors_header


def _write_checkpoint(tmp_path: Path) -> Path:
    pytest.importorskip("safetensors", reason="sidecar I/O needs safetensors")
    import test_run_control as trc

    from qd_train.run_control import TensorRef

    state = {"model": {"w": TensorRef(dtype="float32", shape=(2,), data=struct.pack("<2f", 1, 2))}}
    return trc._checkpoint(model_state=state).write(tmp_path / "ck" / "step-1.json")


def _sidecar_of(ck: Path) -> Path:
    declared = json.loads(ck.read_text(encoding="utf-8"))["sidecar"]
    from qd_train.run_control import Checkpoint

    return Checkpoint.sidecar_path(ck, declared["digest"])


@pytest.mark.parametrize("case", ALL, ids=_ids(ALL))
def test_the_sidecar_reader_gives_the_shared_verdict_on_every_case(case, tmp_path):
    """A sidecar the shared list refuses is refused before any tensor is looked up; one it
    accepts gets as far as asking for the tensor, which these files do not carry."""
    from qd_train.run_control import Checkpoint

    ck = _write_checkpoint(tmp_path)
    _build(case, _sidecar_of(ck))
    with pytest.raises(ValueError) as err:
        Checkpoint.read_weights(ck, subtrees=("model",))
    reached_the_lookup = "does not carry it" in str(err.value)
    assert reached_the_lookup == (case["expect"] == "ok"), (case["name"], str(err.value)[:200])


def test_the_sidecar_reader_keeps_no_100_mb_bound_of_its_own(tmp_path):
    from qd_train.memory import MAX_SAFETENSORS_HEADER_BYTES
    from qd_train.run_control import Checkpoint

    assert not hasattr(sys.modules["qd_train.run_control"], "_MAX_SIDECAR_HEADER_BYTES") or (
        sys.modules["qd_train.run_control"]._MAX_SIDECAR_HEADER_BYTES
        == MAX_SAFETENSORS_HEADER_BYTES
    )
    ck = _write_checkpoint(tmp_path)
    side = _sidecar_of(ck)
    n = struct.unpack("<Q", side.read_bytes()[:8])[0]
    real = side.read_bytes()
    padded = real[8 : 8 + n] + b" " * (65 * 1024 * 1024 - n)
    side.write_bytes(struct.pack("<Q", len(padded)) + padded + real[8 + n :])
    with pytest.raises(ValueError, match="outside"):
        Checkpoint.read_weights(ck, subtrees=("model",))
