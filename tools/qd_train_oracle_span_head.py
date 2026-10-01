#!/usr/bin/env python3
"""Reference oracle for ``crates/qd-train/src/span_head.rs``: the real PyTorch span head, fp32, CPU.

Lane L-head of ``AUDIT/ojas-training-2026-10-01/fable-advice.md`` (Q1 row 8, Q3 rung a). The
Rust head is checked against **this repository's own head** -- ``qd_train.heads``'s
``SpanPointerHead`` over a plan from ``plan_span_batch`` -- not against a re-derivation of it,
and the gradients are the ones ``QwenDecisionStep.accumulate_span`` actually backpropagates:
``span_weight * span_head.loss(hidden, plan)`` with the default ``"mean"`` reduction.

Python here is a reference oracle and nothing else (the repository's language policy). It
runs on the CPU in float32 with one thread; it never touches MPS, which is reserved.

## What each case directory holds

``manifest.json`` plus ``.npy`` files (little-endian, C order):

* the plan as ``plan_span_batch`` built it -- ``candidate_pos``, ``candidate_valid``,
  ``n_candidates``, ``query_index``, ``gold_start``, ``gold_end``, ``abstaining``,
  ``runtime_rows`` -- and the token-level gold it was built from (``gold_start_token``,
  ``gold_end_token``, ``SPAN_ABSTAIN`` for an abstaining row);
* the inputs: ``hidden`` ``[K, L, H]``, ``abstain_start``/``abstain_end`` ``[H]``;
* the outputs: ``scores_start``/``scores_end`` ``[K, max_cand + 1]`` (training padding as
  ``-inf``), ``loss`` (the unweighted mean, which ``span_log`` records), and the gradient of
  ``span_weight * loss`` with respect to every input and parameter: ``d_hidden`` (dense
  ``[K, L, H]``), ``d_abstain_start``/``d_abstain_end``, ``d_start_proj``/``d_end_proj``;
* the same outputs from the float64 arm, prefixed ``ref64_`` (below).

## The two ``[H, H]`` projections are generated, not stored

At ``H = 2048`` one ``[H, H]`` float32 tensor is 16.8 MB, and a case carries four (two
weights, two gradients) -- 67 MB against a repository whose largest tracked file is 1.4 MB.
So:

* **The weights** come from a counter-based generator (SplitMix64 over
  ``seed << 40 ^ stream << 32 ^ index``, 24 bits, scaled by a power of two, so every value is
  exact in float32) that the Rust test re-implements. The manifest pins the SHA-256 of the
  bytes torch actually used; the Rust test checks it before anything else, so a generator
  that drifted fails as *wrong input*, not as a parity miss.
* **The weight gradients** are stored as ``.npy`` when ``H <= 256``. Above that the case must
  have ``K = 1``: torch's ``dW = dq^T @ h_query`` is then one rounded product per element, the
  oracle **asserts** it is bit-equal to ``outer(dq, h_query)``, and stores ``dq_start``/
  ``dq_end`` ``[1, H]`` plus the SHA-256 of torch's ``dW``. The Rust test rebuilds every one of
  the ``H^2`` elements from ``dq`` and the stored ``hidden`` and checks the digest, so this is a
  lossless encoding with the proof on both sides, not a sample.

## The float64 arm: the exact value, beside the float32 one

Each case is also run in float64 -- the same head ``.double()``, the same ``hidden``
``.double()``, the same plan -- and its outputs are dumped with a ``ref64_`` prefix
(``<f8``; the projections' gradients through the same ``dq`` encoding, asserted the same
way). The manifest also records ``max|f32 - f64| / max|f64|`` per tensor under
``torch_f32_vs_f64``: how far the float32 oracle itself is from the exact value.

That is needed because at ``H = 2048`` torch's own float32 rounding of ``d_hidden`` is about
``1e-6`` of the tensor's max, the size of the gate the Rust test applies against it. A
Rust-vs-torch32 residual of that size cannot, from the float32 dump alone, be told apart
from an error of Rust's; against the float64 dump it can.

Usage, from the repository root::

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_span_head.py \\
        --out crates/qd-train/tests/fixtures/span-head
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from qd_train.artifacts import SPAN_ABSTAIN
from qd_train.heads import RESERVED_NOUL_ROWS, SpanPointerHead, plan_span_batch
from qd_train.trainer import SpanSupervision

#: Above this width the projections' gradients are encoded through ``dq`` (see the module
#: docstring) instead of stored. Stated once; the Rust test reads ``dw_encoding`` from the
#: manifest rather than restating the threshold.
MAX_STORED_DW_WIDTH = 256

#: Generator streams for the two projections.
STREAM_START_PROJ = 1
STREAM_END_PROJ = 2

GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15
MIX_1 = 0xBF58476D1CE4E5B9
MIX_2 = 0x94D049BB133111EB


def splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64's output function over a ``uint64`` array; arithmetic wraps mod 2^64."""
    if x.dtype != np.uint64:
        raise TypeError(f"splitmix64 takes uint64, got {x.dtype}")
    with np.errstate(over="ignore"):
        z = x + np.uint64(GOLDEN_RATIO_64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(MIX_1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(MIX_2)
        return z ^ (z >> np.uint64(31))


def generated_weight(seed: int, stream: int, width: int, shift: int) -> np.ndarray:
    """``[width, width]`` float32, uniform on ``[-2^-shift, 2^-shift)``, every value exact.

    ``u`` is the top 24 bits of the hash, centred to ``[-2^23, 2^23)``; ``u * 2^-(23+shift)``
    has a 24-bit significand and a power-of-two scale, so float64 -> float32 is exact and the
    Rust side computes the same bits without reproducing any rounding.
    """
    if not (0 <= seed < 1 << 24 and 0 <= stream < 1 << 8 and width * width < 1 << 32):
        raise ValueError(f"generator key out of range: seed={seed} stream={stream} width={width}")
    index = np.arange(width * width, dtype=np.uint64)
    key = (np.uint64(seed) << np.uint64(40)) ^ (np.uint64(stream) << np.uint64(32)) ^ index
    u = (splitmix64(key) >> np.uint64(40)).astype(np.int64) - (1 << 23)
    out = (u.astype(np.float64) * 2.0 ** -(23 + shift)).astype(np.float32)
    if not np.array_equal(out.astype(np.float64), u.astype(np.float64) * 2.0 ** -(23 + shift)):
        raise AssertionError("generated weight is not exact in float32")
    return out.reshape(width, width)


@dataclass(frozen=True)
class Row:
    """One span row: its line-start tokens, its query token, and its gold (None = abstain)."""

    line_starts: tuple[int, ...]
    query: int
    gold: tuple[int, int] | None


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    hidden_size: int
    seq_len: int
    rows: tuple[Row, ...]
    span_weight: float
    zero_abstain: bool
    seed: int


def random_rows(
    rng: np.random.Generator, *, n_rows: int, seq_len: int, n_lo: int, n_hi: int
) -> tuple[Row, ...]:
    """Random plans whose query is never a candidate, so the duplicate cases stay the only
    ones that exercise the pre-sum (the mutation test depends on that being true).

    Row 0 points and row 1 abstains, so every random case has both kinds of gold.
    """
    rows = []
    for k in range(n_rows):
        n = int(rng.integers(n_lo, n_hi + 1))
        picks = rng.choice(seq_len, size=n + 1, replace=False)
        query, candidates = int(picks[0]), tuple(sorted(int(p) for p in picks[1:]))
        abstain = k == 1 or (k > 1 and bool(rng.random() < 0.3))
        gold = None
        if not abstain:
            a, b = sorted(int(x) for x in rng.integers(0, n, size=2))
            gold = (candidates[a], candidates[b])
        rows.append(Row(candidates, query, gold))
    return tuple(rows)


def build_cases() -> list[Case]:
    many = tuple(
        sorted(
            int(p)
            for p in np.random.default_rng(106).choice(np.arange(1, 400), size=300, replace=False)
        )
    )
    cases = [
        Case(
            "random-h16",
            "randomised plans, mixed pointing/abstaining, span_weight 0.75",
            16,
            32,
            random_rows(np.random.default_rng(100), n_rows=4, seq_len=32, n_lo=3, n_hi=9),
            0.75,
            False,
            100,
        ),
        Case(
            "random-h48",
            "randomised plans, five rows, ragged candidate counts",
            48,
            48,
            random_rows(np.random.default_rng(101), n_rows=5, seq_len=48, n_lo=2, n_hi=12),
            1.0,
            False,
            101,
        ),
        Case(
            "random-h96",
            "randomised plans, three rows, span_weight 0.5",
            96,
            40,
            random_rows(np.random.default_rng(102), n_rows=3, seq_len=40, n_lo=5, n_hi=20),
            0.5,
            False,
            102,
        ),
        Case(
            "one-candidate",
            "every row has exactly one line start: one points at it, one abstains",
            16,
            12,
            (Row((3,), 9, (3, 3)), Row((0,), 7, None)),
            1.0,
            False,
            103,
        ),
        Case(
            "gold-abstain-zero-init",
            "every row abstains; abstain vectors at the head's real init (zeros)",
            16,
            20,
            (Row((0, 4, 9), 19, None), Row((2, 3, 11, 15, 17), 18, None), Row((5,), 12, None)),
            1.0,
            True,
            104,
        ),
        Case(
            "duplicate-positions",
            "query is also a candidate: first / middle (and gold) / last, one abstaining",
            16,
            24,
            (
                Row((2, 6, 11, 15), 2, (11, 15)),
                Row((1, 5, 8, 13, 20), 8, (8, 8)),
                Row((0, 3, 9, 17), 17, None),
            ),
            0.5,
            False,
            105,
        ),
        Case(
            "many-candidates",
            "300 candidates beside a 5-candidate abstaining row (ragged by far more than one)",
            8,
            400,
            (Row(many, 0, (many[250], many[290])), Row((10, 50, 120, 300, 399), 200, None)),
            1.0,
            False,
            106,
        ),
        Case(
            "h2048-point",
            "the real width, one pointing row (K=1: dW encoded through dq)",
            2048,
            28,
            (Row(tuple(range(0, 27, 1))[:24], 27, (5, 17)),),
            0.75,
            False,
            107,
        ),
        Case(
            "h2048-abstain-duplicate",
            "the real width, one abstaining row whose query is its ninth line start",
            2048,
            20,
            (Row((0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19), 11, None),),
            1.0,
            False,
            108,
        ),
    ]
    # The many-candidates row 0 drew its candidates from 1..399, so its query (0) is distinct.
    wide = cases[6].rows[0]
    if wide.query in wide.line_starts or len(wide.line_starts) != 300:
        raise AssertionError("many-candidates row 0 must have 300 candidates and a distinct query")
    # Only the duplicate cases may exercise the pre-sum; the mutation test relies on it.
    for case in cases:
        has_duplicate = any(row.query in row.line_starts for row in case.rows)
        if has_duplicate != ("duplicate" in case.name):
            raise AssertionError(
                f"{case.name}: duplicate query/candidate positions must be "
                "exactly the cases named for them"
            )
    point = cases[7].rows[0]
    if len(point.line_starts) != 24:
        raise AssertionError("h2048-point must have 24 candidates")
    return cases


def supervision(case: Case) -> SpanSupervision:
    """The trainer's span channel for ``case``, through its own contract checks."""
    k = len(case.rows)
    line_starts = np.zeros((k, case.seq_len), dtype=bool)
    start = np.empty(k, dtype=np.int64)
    end = np.empty(k, dtype=np.int64)
    for i, row in enumerate(case.rows):
        if not row.line_starts or max(row.line_starts) >= case.seq_len or row.query >= case.seq_len:
            raise ValueError(f"{case.name} row {i}: position out of range")
        line_starts[i, list(row.line_starts)] = True
        start[i], end[i] = row.gold if row.gold is not None else (SPAN_ABSTAIN, SPAN_ABSTAIN)
    return SpanSupervision(
        rows=np.arange(k, dtype=np.int64),
        query_index=np.array([r.query for r in case.rows], dtype=np.int64),
        start=start,
        end=end,
        line_starts=line_starts,
        abstaining=(start == SPAN_ABSTAIN) & (end == SPAN_ABSTAIN),
    )


def weight_shift(width: int) -> int:
    """``2^-shift`` near ``1/sqrt(width)``, nn.Linear's own init bound."""
    return int(np.floor(np.log2(width) / 2))


def hidden_scale(width: int, shift: int) -> float:
    """A hidden-state scale that puts candidate scores at a standard deviation near 2.

    With weights uniform on ``+-2^-shift`` and hidden ``N(0, s^2)``, a score's standard
    deviation is ``s^2 * width * 2^-shift / sqrt(3)``. Unit-variance hiddens at ``H = 2048``
    would put scores near 37, a softmax that is one-hot to float32 precision, whose gradient
    is zero almost everywhere and tests almost nothing.
    """
    return float(np.sqrt(2.0 * np.sqrt(3.0) / (width * 2.0**-shift)))


def run_head(head: SpanPointerHead, hidden: torch.Tensor, plan, span_weight: float) -> dict:
    """Forward, loss and the backward of ``span_weight * loss``, as accumulate_span runs it."""
    hidden = hidden.detach().clone().requires_grad_(True)
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def capture(name: str):
        def hook(_module, inputs, output):
            if output.requires_grad:  # the no-grad scoring forward has nothing to retain
                output.retain_grad()
                captured[name] = (inputs[0], output)

        return hook

    loss_forward: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_scores(_module, _inputs, output):
        if torch.is_grad_enabled():
            loss_forward.append(output)

    handles = [
        head.start_proj.register_forward_hook(capture("start")),
        head.end_proj.register_forward_hook(capture("end")),
        head.register_forward_hook(capture_scores),
    ]
    try:
        with torch.no_grad():
            scores_start, scores_end = head(hidden, plan)
        # The loss's own forward re-runs the projections; the hooks keep THAT call's tensors,
        # which are the ones the backward runs through.
        loss = head.loss(hidden, plan)
        (span_weight * loss).backward()
    finally:
        for handle in handles:
            handle.remove()
    # The dumped scores come from a separate no-grad forward; they must be the very scores
    # the loss was computed from, or the fixture pairs one forward's scores with another's
    # gradients.
    if len(loss_forward) != 1 or not all(
        torch.equal(a, b.detach())
        for a, b in zip((scores_start, scores_end), loss_forward[0], strict=True)
    ):
        raise AssertionError("the dumped scores are not bit-equal to the loss's own forward")
    return {
        "scores_start": scores_start,
        "scores_end": scores_end,
        "loss": loss.detach(),
        "d_hidden": hidden.grad,
        "d_start_proj": head.start_proj.weight.grad,
        "d_end_proj": head.end_proj.weight.grad,
        "d_abstain_start": head.abstain_start.grad,
        "d_abstain_end": head.abstain_end.grad,
        "h_query": captured["start"][0].detach(),
        "dq_start": captured["start"][1].grad,
        "dq_end": captured["end"][1].grad,
    }


def rel_to_max(a32: torch.Tensor, a64: torch.Tensor) -> float:
    """``max|a32 - a64| / max|a64|`` over the finite entries (the padding is -inf in both)."""
    finite = torch.isfinite(a64)
    if not torch.equal(finite, torch.isfinite(a32)):
        raise AssertionError("float32 and float64 runs disagree on which scores are finite")
    x32, x64 = a32.double()[finite], a64[finite]
    scale = float(x64.abs().max()) if x64.numel() else 0.0
    diff = float((x32 - x64).abs().max()) if x64.numel() else 0.0
    return diff / scale if scale > 0.0 else diff


def sha256_le(t: torch.Tensor) -> str:
    """SHA-256 of a float tensor's little-endian bytes in C order, at its own width."""
    array = np.ascontiguousarray(t.detach().cpu().numpy())
    if array.dtype not in (np.float32, np.float64):
        raise TypeError(f"expected float32 or float64, got {array.dtype}")
    return hashlib.sha256(array.astype(array.dtype.newbyteorder("<")).tobytes()).hexdigest()


def save(directory: Path, name: str, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    allowed = {np.dtype("<f4"), np.dtype("<f8"), np.dtype("<i8"), np.dtype("bool")}
    if array.dtype not in allowed:
        raise TypeError(f"{name}: dtype {array.dtype} is not one the Rust reader takes")
    np.save(directory / f"{name}.npy", array, allow_pickle=False)


def write_case(case: Case, out: Path) -> dict:
    span = supervision(case)
    plan = plan_span_batch(span, device="cpu")
    width, k = case.hidden_size, len(case.rows)
    shift = weight_shift(width)

    head = SpanPointerHead(width)
    w_start = generated_weight(case.seed, STREAM_START_PROJ, width, shift)
    w_end = generated_weight(case.seed, STREAM_END_PROJ, width, shift)
    gen = torch.Generator(device="cpu").manual_seed(case.seed)
    scale = hidden_scale(width, shift)
    hidden = (torch.randn((k, case.seq_len, width), generator=gen) * scale).to(torch.float32)
    if case.zero_abstain:
        a_start = torch.zeros(width)
        a_end = torch.zeros(width)
    else:
        a_start = (torch.randn(width, generator=gen) * scale).to(torch.float32)
        a_end = (torch.randn(width, generator=gen) * scale).to(torch.float32)
    with torch.no_grad():
        head.start_proj.weight.copy_(torch.from_numpy(w_start))
        head.end_proj.weight.copy_(torch.from_numpy(w_end))
        head.abstain_start.copy_(a_start)
        head.abstain_end.copy_(a_end)
    head64 = copy.deepcopy(head).double()

    r32 = run_head(head, hidden, plan, case.span_weight)
    r64 = run_head(head64, hidden.double(), plan, case.span_weight)

    dw_encoding = "npy" if width <= MAX_STORED_DW_WIDTH else "outer_k1"
    if dw_encoding == "outer_k1":
        if k != 1:
            raise ValueError(f"{case.name}: H={width} > {MAX_STORED_DW_WIDTH} needs K=1, got {k}")
        for arm, result in (("float32", r32), ("float64", r64)):
            for which in ("start", "end"):
                rebuilt = torch.outer(result[f"dq_{which}"][0], result["h_query"][0])
                if not torch.equal(result[f"d_{which}_proj"], rebuilt):
                    raise AssertionError(
                        f"{case.name} ({arm}): torch's d_{which}_proj is not bit-equal to "
                        "outer(dq, h_query); the K=1 encoding would not be lossless, so it is "
                        "refused"
                    )

    directory = out / case.name
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.npy"):
        stale.unlink()

    def dump_outputs(prefix: str, result: dict) -> None:
        """One arm's outputs: ``""`` is the float32 gate reference, ``"ref64_"`` the exact one."""
        arrays = {name: t.detach().cpu().numpy() for name, t in result.items()}
        for name in (
            "scores_start",
            "scores_end",
            "loss",
            "d_hidden",
            "d_abstain_start",
            "d_abstain_end",
        ):
            save(directory, prefix + name, arrays[name])
        if dw_encoding == "npy":
            save(directory, prefix + "d_start_proj", arrays["d_start_proj"])
            save(directory, prefix + "d_end_proj", arrays["d_end_proj"])
        else:
            save(directory, prefix + "dq_start", arrays["dq_start"])
            save(directory, prefix + "dq_end", arrays["dq_end"])

    save(directory, "candidate_pos", plan.candidate_pos.numpy().astype("<i8"))
    save(directory, "candidate_valid", plan.candidate_valid.numpy())
    save(directory, "n_candidates", plan.n_candidates.numpy().astype("<i8"))
    save(directory, "query_index", plan.query_index.numpy().astype("<i8"))
    save(directory, "gold_start", plan.gold_start.numpy().astype("<i8"))
    save(directory, "gold_end", plan.gold_end.numpy().astype("<i8"))
    save(directory, "abstaining", plan.abstaining.numpy())
    save(directory, "runtime_rows", plan.runtime_rows.numpy().astype("<i8"))
    save(directory, "gold_start_token", span.start.astype("<i8"))
    save(directory, "gold_end_token", span.end.astype("<i8"))
    save(directory, "hidden", hidden.numpy())
    save(directory, "abstain_start", a_start.numpy())
    save(directory, "abstain_end", a_end.numpy())
    dump_outputs("", r32)
    dump_outputs("ref64_", r64)

    compared = (
        "scores_start",
        "scores_end",
        "loss",
        "d_hidden",
        "d_start_proj",
        "d_end_proj",
        "d_abstain_start",
        "d_abstain_end",
    )
    manifest = {
        "case": case.name,
        "description": case.description,
        "hidden_size": width,
        "n_spans": k,
        "seq_len": case.seq_len,
        "span_weight": case.span_weight,
        "reduction": "mean",
        "reserved_noul_rows": RESERVED_NOUL_ROWS,
        "weight_generator": {
            "algorithm": "splitmix64(seed << 40 ^ stream << 32 ^ index) >> 40, centred, "
            "times 2^-(23 + shift)",
            "seed": case.seed,
            "shift": shift,
            "stream_start_proj": STREAM_START_PROJ,
            "stream_end_proj": STREAM_END_PROJ,
        },
        "sha256": {
            "start_proj_weight": sha256_le(head.start_proj.weight),
            "end_proj_weight": sha256_le(head.end_proj.weight),
            "d_start_proj": sha256_le(r32["d_start_proj"]),
            "d_end_proj": sha256_le(r32["d_end_proj"]),
            "ref64_d_start_proj": sha256_le(r64["d_start_proj"]),
            "ref64_d_end_proj": sha256_le(r64["d_end_proj"]),
        },
        "dw_encoding": dw_encoding,
        "hidden_scale": scale,
        "zero_abstain": case.zero_abstain,
        "torch_f32_vs_f64": {name: rel_to_max(r32[name], r64[name]) for name in compared},
        "oracle": "tools/qd_train_oracle_span_head.py",
        "head": "python/qd_train/heads.py::SpanPointerHead (plan from plan_span_batch)",
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": torch.get_num_threads(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="fixture root to (re)write")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("the oracle's float32 arm needs float32 as torch's default dtype")

    cases = build_cases()
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate case names: {names}")
    args.out.mkdir(parents=True, exist_ok=True)
    stray = sorted(p.name for p in args.out.iterdir() if p.name not in names)
    if stray:
        raise SystemExit(
            f"{args.out} holds entries no case writes: {stray}. Remove them by hand; the Rust "
            "test refuses a fixture directory it does not know, so a stale case cannot linger"
        )
    for case in cases:
        manifest = write_case(case, args.out)
        worst = max(manifest["torch_f32_vs_f64"].values())
        print(
            f"{case.name}: H={case.hidden_size} K={len(case.rows)} L={case.seq_len} "
            f"dW={manifest['dw_encoding']} torch f32-vs-f64 worst={worst:.3e}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
