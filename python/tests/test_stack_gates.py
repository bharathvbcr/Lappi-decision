"""The two image gates: `stack/verify_fast_path.py` and `stack/verify_teacher.py`.

These are the checks that stand between a green image and a billing GPU block, and until
this file existed **neither had a single test** — while `stack/README.md`'s S1 status table
claimed "`not_run` path tested". That claim was false when written. It is the same defect
class the gates themselves exist to prevent: something asserting a check happened when it
had not.

The property that matters most here is the asymmetry: a gate that *could not run* must exit
non-zero under its `--require-*` flag, exactly as a gate that ran and failed does. If
`not_run` exited 0, every gate would pass on any box without a GPU — which is every CI
runner, and was the build host all week.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from qd_train.tristate import NotRun, Ran

STACK = Path(__file__).resolve().parents[2] / "stack"


def _load(name: str):
    """Load a gate script by path. They are scripts beside an image, not an importable package.

    The module is registered in ``sys.modules`` *before* ``exec_module``, which is required
    rather than tidy: ``verify_fast_path`` defines a ``@dataclass``, and dataclass field
    resolution looks the defining class's module up by name. Executing first raises
    ``AttributeError: 'NoneType' object has no attribute '__dict__'`` from inside
    ``dataclasses``, which reads like a bug in the gate rather than in how it was loaded.
    """
    spec = importlib.util.spec_from_file_location(name, STACK / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fake_torch(
    *,
    available: bool = True,
    device_count: int = 8,
    cuda_version: str = "13.0",
    names: list[str] | None = None,
) -> types.ModuleType:
    mod = types.ModuleType("torch")
    mod.__version__ = "2.13.0+cu130"
    resolved = names if names is not None else ["NVIDIA H100 80GB HBM3"] * device_count

    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: device_count,
        get_device_name=lambda i: resolved[i],
    )
    mod.cuda = cuda
    mod.version = types.SimpleNamespace(cuda=cuda_version)
    return mod


def _fake_vllm(version: str = "0.29.0") -> types.ModuleType:
    mod = types.ModuleType("vllm")
    mod.__version__ = version
    return mod


@pytest.fixture
def teacher():
    return _load("verify_teacher")


@pytest.fixture
def fast_path():
    return _load("verify_fast_path")


# --- the asymmetry that the whole design rests on -------------------------------------------


def test_a_gate_that_could_not_run_exits_non_zero_under_require(teacher, monkeypatch):
    """not_run must fail the gate. An unchecked environment is not a clean one."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(
        teacher, "verify", lambda **_: NotRun(reason="no device visible on this host")
    )

    assert teacher.main(["--require-ready"]) == 1
    assert teacher.main([]) == 0, "without the flag, reporting is not gating"


def test_the_same_asymmetry_holds_for_the_trainer_gate(fast_path, monkeypatch):
    monkeypatch.setattr(fast_path, "verify", lambda **_: NotRun(reason="device is 'cpu', not cuda"))

    assert fast_path.main(["--require-fast"]) == 1
    assert fast_path.main([]) == 0


def test_a_gate_that_ran_and_failed_also_exits_non_zero(teacher, monkeypatch):
    monkeypatch.setattr(
        teacher, "verify", lambda **_: Ran(passed=False, value=4, detail="only 4 devices")
    )
    assert teacher.main(["--require-ready"]) == 1


def test_only_a_confirmed_pass_exits_zero(teacher, monkeypatch):
    monkeypatch.setattr(teacher, "verify", lambda **_: Ran(passed=True, value=8, detail="ready"))
    assert teacher.main(["--require-ready"]) == 0


def test_not_run_json_carries_no_passed_field(teacher, monkeypatch, capsys):
    """`NotRun` has no `passed` at all, so no reader can coerce it into a pass."""
    monkeypatch.setattr(teacher, "verify", lambda **_: NotRun(reason="no CUDA device"))
    teacher.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "not_run"
    assert "passed" not in payload


# --- verify_teacher's own branches ----------------------------------------------------------


def test_absent_torch_is_not_run_rather_than_a_failure(teacher, monkeypatch):
    """A missing dependency establishes nothing; it is not evidence the box is unready."""
    monkeypatch.setitem(sys.modules, "torch", None)
    result = teacher.verify(tensor_parallel=8, expect_vllm=None)
    assert isinstance(result, NotRun)
    assert "torch" in result.reason


def test_absent_vllm_is_not_run(teacher, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "vllm", None)
    result = teacher.verify(tensor_parallel=8, expect_vllm=None)
    assert isinstance(result, NotRun)
    assert "vllm" in result.reason


def test_no_cuda_device_is_not_run_not_a_failure(teacher, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm())
    result = teacher.verify(tensor_parallel=8, expect_vllm=None)
    assert isinstance(result, NotRun)


def test_a_vllm_version_off_the_lockfile_is_a_ran_failure(teacher, monkeypatch):
    """The image not matching its lockfile is an observed fact, so it RAN and FAILED."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm("0.28.0"))
    result = teacher.verify(tensor_parallel=8, expect_vllm="0.29.0")
    assert isinstance(result, Ran) and not result.passed
    assert "0.28.0" in result.detail and "0.29.0" in result.detail


def test_the_trainers_cuda_family_is_caught_in_the_teacher_image(teacher, monkeypatch):
    """CUDA 12.8 here means the trainer image is running as the teacher."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_version="12.8"))
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm())
    result = teacher.verify(tensor_parallel=8, expect_vllm=None)
    assert isinstance(result, Ran) and not result.passed
    assert "12.8" in result.detail


def test_too_few_devices_for_the_tensor_parallel_width_fails(teacher, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(device_count=4))
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm())
    result = teacher.verify(tensor_parallel=8, expect_vllm=None)
    assert isinstance(result, Ran) and not result.passed
    assert result.value == 4


def test_a_device_count_that_does_not_divide_the_tp_width_fails(teacher, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(device_count=6))
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm())
    result = teacher.verify(tensor_parallel=4, expect_vllm=None)
    assert isinstance(result, Ran) and not result.passed
    assert "multiple" in result.detail


def test_heterogeneous_devices_fail(teacher, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(device_count=2, names=["NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-40GB"]),
    )
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm())
    result = teacher.verify(tensor_parallel=2, expect_vllm=None)
    assert isinstance(result, Ran) and not result.passed
    assert "A100" in result.detail


def test_a_correctly_provisioned_box_passes(teacher, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(device_count=8))
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm())
    result = teacher.verify(tensor_parallel=8, expect_vllm="0.29.0")
    assert isinstance(result, Ran) and result.passed
    assert result.value == 8


def test_a_nonsense_tensor_parallel_width_is_refused_not_defaulted(teacher):
    with pytest.raises(SystemExit) as exc:
        teacher.main(["--tensor-parallel", "0"])
    assert exc.value.code != 0
