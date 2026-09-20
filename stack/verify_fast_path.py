"""Positively assert the GDN fast path ran. Replaces the plan's fail-open log grep.

## Why the plan's check cannot work

The plan gates every GDN job on a 30-second dry run that greps the startup log for
the flash-linear-attention fallback warning, and refuses to start if it is present.
Two findings (see `docs/plan-corrections.md` SAFETY-1) make that a check that cannot
fail:

1. the warning fires on **first forward, not at import**, so a 30-second dry run that
   never runs a forward produces no warning either way;
2. it is **suppressed entirely under `torch.compile`** by a `not is_torchdynamo_compiling()`
   guard — and the plan's own S3 item tries `torch.compile`.

So on a compiled run the grep passes whether or not the fast path is active, while the
job proceeds on the fp32 chunk loop. Measured penalty: **74% slower** per step
(1.22 vs 0.70 s/step), and the transformers source puts the `chunk_gated_delta_rule`
gap at "more than an order of magnitude on an H100."

Absence of a warning is not evidence. This module asserts the **presence of the fast
implementation in the executed path** instead: it wraps the candidate implementations
with counters, runs a real forward, and reports which one actually ran.

A check that could not run reports `NotRun`, never a pass.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "python"))

from qd_train.tristate import NotRun, Ran, TriState

# Substrings identifying the two paths inside the Qwen3.5 modeling module.
# Discovered, not assumed: `discover()` reports exactly what it bound, and refuses
# when it cannot find both, rather than passing on a half-match.
FAST_HINTS = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")
SLOW_HINTS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")

MODULE_CANDIDATES = (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_next.modeling_qwen3_next",
)


@dataclass(slots=True)
class Probe:
    module_name: str
    fast: dict[str, int]
    slow: dict[str, int]

    @property
    def fast_calls(self) -> int:
        return sum(self.fast.values())

    @property
    def slow_calls(self) -> int:
        return sum(self.slow.values())


def _load_module() -> tuple[Any, str] | None:
    for name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(name), name
        except ImportError:
            continue
    return None


def discover(module: Any) -> tuple[list[str], list[str]]:
    """Find the fast and fallback symbols actually present in this build.

    Names are matched by substring against the module's own attributes rather than
    hardcoded, because a hardcoded name that a version bump renames would silently
    bind nothing — and a counter that never increments looks exactly like a fast path
    that never ran.
    """
    names = dir(module)
    slow = [
        n
        for n in names
        if any(h in n for h in SLOW_HINTS) and callable(getattr(module, n, None))
    ]
    fast = [
        n
        for n in names
        if any(h in n for h in FAST_HINTS)
        and n not in slow
        and callable(getattr(module, n, None))
    ]
    return fast, slow


def instrument(module: Any, fast: list[str], slow: list[str]) -> tuple[Probe, Callable[[], None]]:
    """Wrap both paths with call counters. Returns the probe and an undo function."""
    probe = Probe(module_name=module.__name__, fast={n: 0 for n in fast}, slow={n: 0 for n in slow})
    originals: dict[str, Any] = {}

    def wrap(name: str, bucket: dict[str, int]) -> None:
        original = getattr(module, name)
        originals[name] = original

        def counted(*a: Any, **kw: Any) -> Any:
            bucket[name] += 1
            return original(*a, **kw)

        counted.__name__ = getattr(original, "__name__", name)
        setattr(module, name, counted)

    for n in fast:
        wrap(n, probe.fast)
    for n in slow:
        wrap(n, probe.slow)

    def undo() -> None:
        for name, original in originals.items():
            setattr(module, name, original)

    return probe, undo


#: `flash-linear-attention` computes `chunk_bwd_dqkwg` incorrectly on Hopper for Triton in
#: this half-open range and raises rather than returning wrong gradients. Measured against
#: `triton==3.6.0`, which `stack/train.lock` pins because `torch==2.10.0+cu128` requires that
#: exact version.
TRITON_HOPPER_BAD_RANGE: Final[tuple[tuple[int, ...], tuple[int, ...]]] = ((3, 4, 0), (3, 7, 1))


def _version_tuple(v: str) -> tuple[int, ...]:
    """Leading numeric components of a version string; `()` when there are none."""
    out: list[int] = []
    for part in v.split("+")[0].split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


def triton_refuses_the_backward() -> str | None:
    """Why the fast path cannot train here, or ``None`` when nothing rules it out.

    **This gate probes a forward pass.** fla refuses on the BACKWARD, so a dispatch probe
    watches the fast path being selected, reports it live, and says nothing about whether a
    training step can complete. That is not a hole a better probe closes -- a forward cannot
    observe a backward -- so the version combination is checked directly, before anything is
    loaded.

    Returning ``None`` means this particular refusal does not apply. It is not a claim that
    the fast path works; that is what the rest of this module measures.
    """
    try:
        import torch
        import triton
    except ImportError:
        return None  # the caller has already established torch; triton absent is its problem

    if not torch.cuda.is_available():
        return None
    if torch.cuda.get_device_capability(0)[0] != 9:  # Hopper
        return None

    version = getattr(triton, "__version__", "")
    tv = _version_tuple(version)
    lo, hi = TRITON_HOPPER_BAD_RANGE
    if not tv or not (lo <= tv < hi):
        return None

    lo_s, hi_s = ".".join(map(str, lo)), ".".join(map(str, hi))
    return (
        f"triton {version} on {torch.cuda.get_device_name(0)} (Hopper, sm_90): "
        f"flash-linear-attention computes chunk_bwd_dqkwg incorrectly for triton in "
        f"[{lo_s}, {hi_s}) and raises on the BACKWARD. A forward pass dispatches to the fast "
        "kernel and succeeds, so watching the dispatch would report the fast path live for an "
        f"environment that cannot train. Upgrade to triton >= {hi_s} (measured on a GH200: the "
        "backward then runs, torch's bf16 matmul is unaffected, and the fast path is 4.9x to "
        "15.0x faster than the reference PyTorch path over widths 2048 to 8192), or install "
        "tilelang."
    )


def verify(*, model_id: str, seq_len: int = 128, device: str | None = None) -> TriState:
    """Run one real forward and report which GDN implementation executed."""
    loaded = _load_module()
    if loaded is None:
        return NotRun(
            reason=(
                "no Qwen3.5 modeling module importable from "
                f"{MODULE_CANDIDATES}; cannot determine which GDN path would run"
            )
        )
    module, module_name = loaded

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        return NotRun(reason=f"torch/transformers not importable: {exc}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        # Stated rather than worked around: the fast path is CUDA/XPU/MLU-gated
        # (`is_flash_linear_attention_available()` does not list MPS), and
        # causal-conv1d is CUDA-only. On this Mac the answer is structurally
        # "not run", and calling it a pass would be the exact bug this replaces.
        return NotRun(
            reason=(
                f"device is {device!r}, not cuda. flash-linear-attention is gated on "
                "cuda/xpu/mlu and causal-conv1d is CUDA-only, so the fast path cannot "
                "be exercised here. This check is meaningful only on the training box."
            )
        )

    bad = triton_refuses_the_backward()
    if bad is not None:
        return Ran(passed=False, detail=bad)

    fast, slow = discover(module)
    if not fast or not slow:
        return NotRun(
            reason=(
                f"could not identify both paths in {module_name}: "
                f"fast={fast or 'NONE FOUND'}, fallback={slow or 'NONE FOUND'}. "
                "Refusing to report a result from a half-bound probe."
            )
        )

    probe, undo = instrument(module, fast, slow)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        model.eval()
        ids = torch.randint(0, 1000, (1, seq_len), device=device)
        with torch.no_grad():
            model(ids)
    except Exception as exc:  # broad by design - reported, never swallowed
        undo()
        return NotRun(
            reason="forward pass failed, so nothing was observed: "
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        undo()

    detail = (
        f"module={probe.module_name} fast={probe.fast} fallback={probe.slow} "
        f"(seq_len={seq_len}, device={device})"
    )

    if probe.fast_calls == 0 and probe.slow_calls == 0:
        return NotRun(
            reason=(
                "neither implementation was called during the forward. The probe bound "
                f"nothing that executes: {detail}. A silent probe is not a pass."
            )
        )
    if probe.slow_calls > 0:
        return Ran(
            passed=False,
            value=probe.slow_calls,
            detail=(
                f"FALLBACK ACTIVE: the fp32 torch chunk loop ran {probe.slow_calls} time(s). "
                f"Expect ~74% slower steps. {detail}"
            ),
        )
    return Ran(passed=True, value=probe.fast_calls, detail=f"fast path confirmed. {detail}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--json", action="store_true", help="emit the tri-state as JSON")
    p.add_argument(
        "--require-fast",
        action="store_true",
        help="exit non-zero unless the fast path is CONFIRMED. not_run also fails, "
        "because an unchecked environment is not a clean one.",
    )
    args = p.parse_args(argv)

    result = verify(model_id=args.model_id, seq_len=args.seq_len, device=args.device)
    print(json.dumps(result.to_json(), indent=2) if args.json else result)

    if args.require_fast:
        return 0 if isinstance(result, Ran) and result.passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
