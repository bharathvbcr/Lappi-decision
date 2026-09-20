"""Positively assert the teacher environment can actually serve, before Session 0 spends on it.

The trainer has `verify_fast_path.py`, which asserts the GDN fast path *executed* rather
than grepping a log for the absence of a warning. This is the same discipline applied to the
other image: the teacher's failure mode is not a silent slow path, it is a tensor-parallel
launch that dies forty minutes into an eighty-thousand-diff labelling run.

Four things are checked, in the order that makes a failure cheapest to read:

1. **vLLM imports, and at the pinned version.** An import that fails here is a broken image,
   not a broken run.
2. **CUDA is visible to torch**, and the CUDA family matches what `teacher.lock` resolved.
   The lockfile is CUDA 13 (`nvidia-cuda-runtime==13.0.96`, `nccl-cu13`, `cudnn-cu13`) while
   the trainer is 12.8 — running the teacher image against a 12.x-only driver is a real
   configuration mistake and it should be named here, not inferred from a kernel launch
   failure later.
3. **Enough devices for the configured tensor-parallel width.** `--tensor-parallel 8` on a
   box that enumerates 4 GPUs fails at engine start; counting first turns a mid-run crash
   into one line before anything is dispatched.
4. **Device count is a multiple of the TP width**, which is what vLLM actually requires.

A check that could not run reports ``NotRun``, never a pass — so `--require-ready` treats
`not_run` as a failure, exactly as the trainer's `--require-fast` does. An unchecked
environment is not a clean one.

This does NOT load the teacher weights. Pulling ~55 GB to answer "can this box serve?" would
make the check cost more than the thing it guards; weight loading is Session 0's first real
step and fails loudly on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_train.tristate import NotRun, Ran, TriState

EXPECTED_VLLM = "0.29.0"
EXPECTED_CUDA_MAJOR = 13


def verify(*, tensor_parallel: int, expect_vllm: str | None) -> TriState:
    try:
        import torch
    except Exception as exc:  # broad by design - reported, never swallowed
        return NotRun(
            reason="torch did not import, so nothing could be observed: "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        import vllm
    except Exception as exc:  # broad by design - reported, never swallowed
        return NotRun(
            reason="vllm did not import, so nothing could be observed: "
            f"{type(exc).__name__}: {exc}"
        )

    vllm_version = getattr(vllm, "__version__", None)
    if expect_vllm is not None and vllm_version != expect_vllm:
        return Ran(
            passed=False,
            value=str(vllm_version),
            detail=(
                f"vllm is {vllm_version!r}, pinned at {expect_vllm!r} in teacher.lock. The image "
                "does not match the lockfile it was built from, so the protocol hash recorded "
                "against this run would name an environment that did not produce it."
            ),
        )

    if not torch.cuda.is_available():
        return NotRun(
            reason=(
                "torch.cuda.is_available() is False. vLLM cannot serve without a visible CUDA "
                "device, so nothing about this environment's readiness was established. This "
                "check is meaningful only on the labelling box — on a build host it is expected "
                "to land here."
            )
        )

    device_count = torch.cuda.device_count()
    torch_cuda = torch.version.cuda or "unknown"
    cuda_major = torch_cuda.split(".")[0]

    detail_env = (
        f"vllm={vllm_version} torch={torch.__version__} torch.version.cuda={torch_cuda} "
        f"devices={device_count} tensor_parallel={tensor_parallel}"
    )

    if cuda_major.isdigit() and int(cuda_major) != EXPECTED_CUDA_MAJOR:
        return Ran(
            passed=False,
            value=torch_cuda,
            detail=(
                f"torch was built against CUDA {torch_cuda}, but teacher.lock resolved the CUDA "
                f"{EXPECTED_CUDA_MAJOR} wheel family (nvidia-cuda-runtime==13.0.96, nccl-cu13, "
                f"cudnn-cu13). This is the trainer image's 12.8 family, not the teacher's — the "
                f"wrong image is running. {detail_env}"
            ),
        )

    if device_count < tensor_parallel:
        return Ran(
            passed=False,
            value=device_count,
            detail=(
                f"tensor-parallel width {tensor_parallel} needs {tensor_parallel} devices; this "
                f"box enumerates {device_count}. vLLM would fail at engine start, after the "
                f"instance is already billing. {detail_env}"
            ),
        )

    if device_count % tensor_parallel != 0:
        return Ran(
            passed=False,
            value=device_count,
            detail=(
                f"device count {device_count} is not a multiple of tensor-parallel width "
                f"{tensor_parallel}; vLLM requires it to divide evenly. {detail_env}"
            ),
        )

    names = sorted({torch.cuda.get_device_name(i) for i in range(device_count)})
    if len(names) > 1:
        return Ran(
            passed=False,
            value=device_count,
            detail=(
                f"heterogeneous devices: {names}. Tensor parallelism across unlike GPUs gives "
                f"the slowest one's throughput at best, and mismatched memory at worst. "
                f"{detail_env}"
            ),
        )

    return Ran(
        passed=True,
        value=device_count,
        detail=f"teacher environment ready. device={names[0]}. {detail_env}",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tensor-parallel",
        type=int,
        default=8,
        help="TP width Session 0 will launch with. teacher.in specifies TP=8.",
    )
    p.add_argument(
        "--expect-vllm",
        default=EXPECTED_VLLM,
        help="vllm version teacher.lock pins. Pass an empty string to skip the comparison.",
    )
    p.add_argument("--json", action="store_true", help="emit the tri-state as JSON")
    p.add_argument(
        "--require-ready",
        action="store_true",
        help="exit non-zero unless readiness is CONFIRMED. not_run also fails, because an "
        "unchecked environment is not a clean one.",
    )
    args = p.parse_args(argv)

    if args.tensor_parallel < 1:
        p.error(f"--tensor-parallel must be at least 1, got {args.tensor_parallel}")

    result = verify(
        tensor_parallel=args.tensor_parallel,
        expect_vllm=args.expect_vllm or None,
    )
    print(json.dumps(result.to_json(), indent=2) if args.json else result)

    if args.require_ready:
        return 0 if isinstance(result, Ran) and result.passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
