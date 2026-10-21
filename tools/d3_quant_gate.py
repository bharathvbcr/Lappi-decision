"""The D3 quantisation gate: does an MLX affine q8 g64 fake-quant base score like the bf16 base?

Fable's D3 (2026-10-09; owner record
``AUDIT/lora-2b-quantised-2026-10-17/owner-decision.md`` section 5) admits a quantised format only
if, over enough validation prompts (never held-out, rule 3), the letter log-softmax of the
quantised base keeps the bf16 base's argmax at or above a floor and moves the gold letter's
log-probability by at most a ceiling on average. Both bars, the prompt floor and the quantiser
(mode, bits, group size) are read from a pre-registration's ``d3_gate`` block. They are not in
this file: a second copy would be a threshold an agent could move (rule 2).

Reference quantiser: ``mx.dequantize(*mx.quantize(W, group_size=g, bits=b), group_size=g,
bits=b)`` from a bf16 W, the call the ojas runbook (``docs/lappi-lora-mac-runbook.md`` 1.5) and
the ojas work order (``tools/a6/LAPPI-WORK-ORDER.md`` L3) name. ``python/tests/
test_d3_quant_gate.py`` checks it against a hand oracle; mlx is the reference, the oracle the test.

Verdict is tri-state (``qd_train.tristate``): PASS and FAIL are ``Ran``; every input that is
missing (the pre-registration or its ``d3_gate`` block, mlx, a Metal device, the model, the
items), malformed, refused (dtype, shape, non-finite) or too small is ``NotRun`` with the
reason, never PASS. Exit code: PASS 0, FAIL 1 (a completed comparison that missed a bar and
nothing else), NOT_RUN 3 (the Makefile's ``NOT_RUN``), REFUSED 2 (the items path is held-out,
rule 3; the verdict file says ``REFUSED`` and carries no tri-state, so no reader takes it as a
pass). At start the tool atomically replaces whatever is at ``--out`` with a NOT_RUN verdict,
and writes the final one by tmp file, fsync and ``os.replace``, so a crash, a kill or an
exception never leaves an earlier PASS in place. An unexpected exception is NOT_RUN plus a
traceback on stderr, never exit 1.

The quantiser (bits 4 or 8) is whatever the pre-registration names: q4 is measured by this
gate (owner decision section 3) and never built, so refusing 4 here would refuse the
measurement; a pre-registration that names a quantiser other than the one under test is the
caller's to catch, and the verdict records the spec and the pre-registration's sha256.

    /Users/bharath/.venvs/ml/bin/python tools/d3_quant_gate.py \\
        --preregistration campaign/lora-2b-q8-preregistered.json \\
        --model DIR --items ITEMS.jsonl --out VERDICT.json

Items are JSONL with ``row_id``, ``prompt_ids``, ``letter_ids`` and ``gold_row`` (the format
``mac_mlx_bench.py`` reads). A path under a held-out root or marker is refused through
``qd_train.data_access.assert_path_not_held_out``; the tool cannot otherwise tell validation
from train rows, and the verdict records the items file's sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qd_data.config import DataConfig  # noqa: E402
from qd_data.errors import HeldOutViolation  # noqa: E402
from qd_train.data_access import assert_path_not_held_out  # noqa: E402
from qd_train.tristate import NotRun, Ran, TriState  # noqa: E402

#: mlx's floor on an affine group's scale (``max(range / (2**bits - 1), eps)``). Measured on mlx
#: 0.31.2: a constant or all-zero group quantises to scale -1e-7, and
#: ``test_d3_quant_gate.py::test_identical_values_round_trip_exactly`` pins it.
MLX_SCALE_EPS = 1e-7

#: The key of the pre-registration block that carries the gate.
SPEC_KEY = "d3_gate"

#: The ten LoRA-wrapped linear kinds (ojas-model ``LORA_KINDS``; ojas ``qwen35_fakequant_base.py``
#: ``TARGET_SUFFIXES``). ``in_proj_a`` and ``in_proj_b`` are not among them and stay unquantised.
TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_TARGET = re.compile(
    r"(?:.*\.)?layers\.\d+\.(" + "|".join(re.escape(s) for s in TARGET_SUFFIXES) + r")\.weight"
)

EXIT = {"PASS": 0, "FAIL": 1, "REFUSED": 2, "NOT_RUN": 3}


@dataclass(frozen=True, slots=True)
class GateSpec:
    """The pre-registered gate: both bars, the prompt floor, and the quantiser."""

    argmax_agreement_min: float
    mean_abs_delta_logp_max: float
    min_prompts: int
    bits: int
    group_size: int
    mode: str


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that refuses a key given twice: last-wins would hide a bar."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _need(block: Mapping[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    v = block.get(key)
    if isinstance(v, bool) or not isinstance(v, kind):
        raise SystemExit(f"pre-registration {SPEC_KEY}.{key} must be {kind}, got {v!r}")
    return v


def load_spec(path: Path) -> GateSpec | NotRun:
    """The gate from a pre-registration.

    ``NotRun`` if the file or its block is absent; a malformed block exits.
    """
    if not path.is_file():
        return NotRun(
            reason=f"pre-registration {path} does not exist, so no D3 threshold is registered"
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except ValueError as exc:  # JSONDecodeError, a duplicate key, an int too long to parse
        raise SystemExit(f"pre-registration {path} is not usable JSON: {exc}") from exc
    block = doc.get(SPEC_KEY) if isinstance(doc, dict) else None
    if block is None:
        return NotRun(reason=f"pre-registration {path} has no {SPEC_KEY!r} block: no D3 threshold")
    if not isinstance(block, dict):
        raise SystemExit(f"pre-registration {path}: {SPEC_KEY} must be an object")
    quant = _need(block, "quantiser", dict)
    try:
        spec = GateSpec(
            argmax_agreement_min=float(_need(block, "argmax_agreement_min", (int, float))),
            mean_abs_delta_logp_max=float(_need(block, "mean_abs_delta_logp_max", (int, float))),
            min_prompts=int(_need(block, "min_prompts", int)),
            bits=int(_need(quant, "bits", int)),
            group_size=int(_need(quant, "group_size", int)),
            mode=str(_need(quant, "mode", str)),
        )
    except (OverflowError, ValueError) as exc:  # a number too large for a float
        raise SystemExit(
            f"pre-registration {path}: {SPEC_KEY} has an unusable number: {exc}"
        ) from exc
    if not 0.0 < spec.argmax_agreement_min <= 1.0:
        raise SystemExit(
            f"{SPEC_KEY}.argmax_agreement_min must be in (0, 1], got {spec.argmax_agreement_min}"
        )
    if not (math.isfinite(spec.mean_abs_delta_logp_max) and spec.mean_abs_delta_logp_max >= 0.0):
        raise SystemExit(f"{SPEC_KEY}.mean_abs_delta_logp_max must be finite and >= 0")
    if spec.min_prompts < 1:
        raise SystemExit(f"{SPEC_KEY}.min_prompts must be >= 1")
    if spec.mode != "affine" or spec.bits not in (4, 8) or spec.group_size < 1:
        raise SystemExit(
            f"{SPEC_KEY}.quantiser must be affine, 4 or 8 bits, a positive group size: got {quant}"
        )
    return spec


def _mx() -> Any:
    import mlx.core as mx

    return mx


def fake_quant(w: Any, *, bits: int, group_size: int, name: str = "tensor") -> Any:
    """``dequantize(quantize(w))`` with mlx affine. Refuses what would give a silently wrong base.

    Refused with ``SystemExit``: an in-features width that is not a multiple of the group (mlx would
    raise; the partial group is never padded), a non-finite weight (mlx quantises NaN to a finite
    number), a non-finite result, and any group whose result is further from its source than the
    affine step allows (a collapsed scale). The step bound is ojas's (``qwen35_fakequant_base.py``)
    with mlx's scale floor. Also refused here, not only in ``quantise_weights``: an empty tensor
    and any dtype but bfloat16 (the runbook's call quantises a bf16 W so scale and bias are bf16).
    """
    mx = _mx()
    if w.dtype != mx.bfloat16:
        raise SystemExit(f"{name}: dtype {w.dtype}, expected bfloat16")
    if w.size == 0:
        raise SystemExit(f"{name}: empty tensor, shape {tuple(w.shape)}")
    return _fake_quant(w, bits=bits, group_size=group_size, name=name)


def _fake_quant(w: Any, *, bits: int, group_size: int, name: str) -> Any:
    """``fake_quant`` after the dtype and emptiness checks; the oracle tests call it on float32."""
    mx = _mx()
    if w.ndim != 2:
        raise SystemExit(f"{name}: expected a 2-D weight, got shape {tuple(w.shape)}")
    if w.shape[-1] % group_size:
        raise SystemExit(
            f"{name}: in-features {w.shape[-1]} is not a multiple of the group size {group_size}"
        )
    if not bool(mx.isfinite(w).all().item()):
        raise SystemExit(f"{name}: non-finite weight")
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    deq = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)
    mx.eval(deq)
    if not bool(mx.isfinite(deq).all().item()):
        raise SystemExit(f"{name}: non-finite after quantisation")
    _check_step_bound(w, deq, bits, group_size, name)
    return deq


def _check_step_bound(w: Any, deq: Any, bits: int, group_size: int, name: str) -> None:
    import numpy as np

    mx = _mx()
    g = np.array(w.astype(mx.float32)).reshape(-1, group_size)
    d = np.array(deq.astype(mx.float32)).reshape(-1, group_size)
    # mlx floors the scale at MLX_SCALE_EPS, so a group narrower than 255 * eps is quantised with
    # that scale and its exact result is off by up to half of it.
    step = np.maximum((g.max(axis=1) - g.min(axis=1)) / (2**bits - 1), MLX_SCALE_EPS)
    allowed = (
        1.5 * step + np.abs(g).max(axis=1) * 2.0**-7
    )  # a step, plus bf16 scale/bias/result rounding
    worst = np.abs(d - g).max(axis=1)
    if (worst > allowed).any():
        bad = int(np.argmax(worst - allowed))
        raise SystemExit(
            f"{name}: group {bad} is off by {worst[bad]:.3e}, more than the q{bits} step "
            f"allows ({allowed[bad]:.3e})"
        )


def select_targets(names: Sequence[str]) -> list[str]:
    """The wrapped-linear weight names among ``names``, in input order."""
    return [n for n in names if _TARGET.fullmatch(n)]


def quantise_weights(weights: Mapping[str, Any], spec: GateSpec) -> dict[str, Any]:
    """The fake-quantised replacement of every wrapped linear in ``weights`` (only those).

    ``fake_quant`` refuses a target that is not bf16. Zero targets is an error, not an empty
    success.
    """
    targets = select_targets(list(weights))
    if not targets:
        raise SystemExit(
            "no wrapped linear (layers.<n>.<kind>.weight) in the weights; refusing an empty gate"
        )
    out: dict[str, Any] = {}
    for name in targets:
        out[name] = fake_quant(weights[name], bits=spec.bits, group_size=spec.group_size, name=name)
    return out


@dataclass(frozen=True, slots=True)
class Metrics:
    n: int
    argmax_agreement: float
    mean_abs_delta_logp: float


def compare(
    base: Mapping[str, Sequence[float]],
    quant: Mapping[str, Sequence[float]],
    gold: Mapping[str, int],
) -> Metrics:
    """Argmax agreement and the mean |delta log p| of the gold letter, over the gold-labelled rows.

    Both score maps are ``row_id -> letter log-softmax``. A row on one side only, a length
    mismatch, a gold index out of range or a non-finite value is refused rather than dropped.
    """
    if set(base) != set(quant):
        raise SystemExit(
            f"base and quantised score different rows: {len(set(base) ^ set(quant))} differ"
        )
    if set(base) - set(gold):
        raise SystemExit(f"{len(set(base) - set(gold))} scored rows have no gold_row")
    agree = 0
    deltas: list[float] = []
    for rid in sorted(base):
        a, b = list(base[rid]), list(quant[rid])
        if len(a) != len(b) or not a:
            raise SystemExit(f"item {rid}: {len(a)} letters on one side, {len(b)} on the other")
        if not all(math.isfinite(v) for v in a + b):
            raise SystemExit(f"item {rid}: non-finite letter log-probability")
        if isinstance(gold[rid], bool) or not isinstance(gold[rid], int):
            raise SystemExit(f"item {rid}: gold_row {gold[rid]!r} is not an int")
        if not 0 <= gold[rid] < len(a):
            raise SystemExit(f"item {rid}: gold_row {gold[rid]} outside 0..{len(a) - 1}")
        agree += int(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__))
        deltas.append(abs(a[gold[rid]] - b[gold[rid]]))
    if not deltas:
        raise SystemExit("no rows were scored")
    return Metrics(
        n=len(deltas),
        argmax_agreement=agree / len(deltas),
        mean_abs_delta_logp=math.fsum(deltas) / len(deltas),
    )


def _too_few(n: int, spec: GateSpec) -> NotRun:
    return NotRun(reason=f"{n} prompts, the pre-registration needs at least {spec.min_prompts}")


def decide(spec: GateSpec, m: Metrics, *, n_total: int) -> TriState:
    """PASS only when both bars hold over at least ``min_prompts`` rows; fewer is NotRun."""
    if m.n < spec.min_prompts:
        return _too_few(m.n, spec)
    ok_agree = m.argmax_agreement >= spec.argmax_agreement_min
    ok_delta = m.mean_abs_delta_logp <= spec.mean_abs_delta_logp_max
    return Ran(
        passed=ok_agree and ok_delta,
        n=m.n,
        n_total=n_total,
        detail=(
            f"argmax agreement {m.argmax_agreement:.6f} (floor {spec.argmax_agreement_min}) "
            f"{'ok' if ok_agree else 'FAIL'}; mean |delta log p| gold "
            f"{m.mean_abs_delta_logp:.6f} nats "
            f"(ceiling {spec.mean_abs_delta_logp_max}) {'ok' if ok_delta else 'FAIL'}"
        ),
    )


ScoreFn = Callable[[Mapping[str, Any]], dict[str, list[float]]]


def run_gate(
    spec: GateSpec, weights: Mapping[str, Any], score_fn: ScoreFn, gold: Mapping[str, int]
) -> tuple[TriState, Metrics | None]:
    """Score with the weights as given, then with the wrapped linears fake-quantised, and decide.

    ``score_fn(overrides)`` returns ``row_id -> letter log-softmax`` for the model whose tensors
    ``overrides`` replaces; ``{}`` is the unquantised base.
    """
    if len(gold) < spec.min_prompts:
        return _too_few(len(gold), spec), None
    base = score_fn({})
    quant = score_fn(quantise_weights(weights, spec))
    m = compare(base, quant, gold)
    return decide(spec, m, n_total=len(gold)), m


def verdict_name(t: TriState) -> str:
    if isinstance(t, NotRun):
        return "NOT_RUN"
    return "PASS" if t.passed else "FAIL"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _real_preconditions(model: Path, items: Path) -> NotRun | None:
    try:
        import mlx.core as mx
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        return NotRun(reason=f"mlx or mlx_lm is not importable: {exc}")
    if not mx.metal.is_available():
        return NotRun(reason="no Metal device is available to mlx")
    if not (model / "config.json").is_file():
        return NotRun(
            reason=f"model directory {model} has no config.json (not downloaded or wrong path)"
        )
    if not items.is_file():
        return NotRun(reason=f"items file {items} does not exist")
    return None


def _int_list(it: Mapping[str, Any], key: str, where: str) -> list[int]:
    v = it.get(key)
    if (
        not isinstance(v, list)
        or not v
        or any(isinstance(x, bool) or not isinstance(x, int) for x in v)
    ):
        raise SystemExit(f"{where}: {key} must be a non-empty list of ints")
    return v


def read_items(path: Path) -> list[dict[str, Any]]:
    """The validation items, each checked; anything malformed exits rather than being skipped.

    ``gold_row`` must be a non-bool int inside ``letter_ids`` (never truncated by ``int()``), and
    ``row_id`` a non-empty string used once.
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{n}"
        try:
            it = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{where}: not JSON: {exc}") from exc
        if not isinstance(it, dict):
            raise SystemExit(f"{where}: an item must be a JSON object")
        rid = it.get("row_id")
        if not isinstance(rid, str) or not rid:
            raise SystemExit(f"{where}: row_id must be a non-empty string")
        if rid in seen:
            raise SystemExit(f"{where}: row_id {rid!r} appears twice")
        seen.add(rid)
        _int_list(it, "prompt_ids", where)
        letters = _int_list(it, "letter_ids", where)
        gold = it.get("gold_row")
        if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(letters):
            raise SystemExit(f"{where}: gold_row {gold!r} must be an int in 0..{len(letters) - 1}")
        items.append(it)
    return items


def _run_real(spec: GateSpec, model_dir: Path, items_path: Path) -> tuple[TriState, Metrics | None]:
    """The real-base gate: letter scoring through ``mac_mlx_bench.score`` on an ``mlx_lm`` model."""
    items = read_items(items_path)
    gold = {it["row_id"]: it["gold_row"] for it in items}
    if len(gold) < spec.min_prompts:
        return _too_few(len(gold), spec), None

    import mac_bench_common as common
    import mac_mlx_bench as bench
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load

    model, _tok = load(str(model_dir))
    weights = dict(tree_flatten(model.parameters()))

    def score_fn(overrides: Mapping[str, Any]) -> dict[str, list[float]]:
        if overrides:
            model.update(tree_unflatten(list(overrides.items())))
        out: dict[str, list[float]] = {}
        for it in items:
            z = bench.score(model, it["prompt_ids"], it["letter_ids"]).tolist()
            out[it["row_id"]] = common.letter_record(
                row_id=it["row_id"], n_tokens=len(it["prompt_ids"]), letter_logits=z
            )["letter_logsoftmax"]
        mx.eval(model.parameters())
        return out

    return run_gate(spec, weights, score_fn, gold)


def write_verdict(path: Path, doc: Mapping[str, Any]) -> None:
    """Replace ``path`` with ``doc`` atomically: a tmp file, fsync, then ``os.replace``."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    data = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)  # os.replace: atomic on one filesystem


def _not_run_doc(reason: str) -> dict[str, Any]:
    return {"gate": "D3", "verdict": "NOT_RUN", **NotRun(reason=reason).to_json()}


def _build_document(args: argparse.Namespace, doc: dict[str, Any]) -> None:
    """Fill ``doc`` with the verdict. May raise; ``main`` makes every raise NOT_RUN or REFUSED."""
    prereg = args.preregistration
    doc["preregistration"] = str(prereg)
    if prereg.is_file():
        doc["preregistration_sha256"] = _sha256(prereg)
    assert_path_not_held_out(args.items, config=DataConfig(), repo_root=REPO)
    spec = load_spec(prereg)
    metrics: Metrics | None = None
    if isinstance(spec, NotRun):
        result: TriState = spec
    else:
        pre = _real_preconditions(args.model, args.items)
        result, metrics = (
            (pre, None) if pre is not None else _run_real(spec, args.model, args.items)
        )
        doc["spec"] = {
            "argmax_agreement_min": spec.argmax_agreement_min,
            "mean_abs_delta_logp_max": spec.mean_abs_delta_logp_max,
            "min_prompts": spec.min_prompts,
            "quantiser": {"mode": spec.mode, "bits": spec.bits, "group_size": spec.group_size},
        }
    doc.update({"verdict": verdict_name(result), **result.to_json()})
    if metrics is not None:
        doc["metrics"] = {
            "n": metrics.n,
            "argmax_agreement": metrics.argmax_agreement,
            "mean_abs_delta_logp": metrics.mean_abs_delta_logp,
        }
    if args.items.is_file():
        doc["items_sha256"] = _sha256(args.items)
    try:
        doc["mlx_version"] = _mx().__version__
    except ImportError:
        doc["mlx_version"] = None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--preregistration", type=Path, required=True, help=f"JSON carrying the {SPEC_KEY!r} block"
    )
    ap.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local bf16 snapshot directory (never downloaded here)",
    )
    ap.add_argument("--items", type=Path, required=True, help="validation items, JSONL")
    ap.add_argument("--out", type=Path, required=True, help="verdict JSON to write")
    args = ap.parse_args(argv)

    # An earlier PASS at --out must not outlive this run if it dies: replace it first.
    try:
        write_verdict(args.out, _not_run_doc("the run started and has not finished"))
    except OSError as exc:
        print(f"D3 gate: cannot write the verdict to {args.out}: {exc}", file=sys.stderr)
        return EXIT["NOT_RUN"]

    doc: dict[str, Any] = {"gate": "D3"}
    try:
        _build_document(args, doc)
    except HeldOutViolation as exc:
        # Rule 3, a refusal and not a could-not-run: no tri-state, so nothing reads it as a pass.
        doc = {"gate": "D3", "verdict": "REFUSED", "reason": str(exc), "items": str(args.items)}
    except SystemExit as exc:
        doc = {**{k: v for k, v in doc.items() if k != "verdict"}, **_not_run_doc(str(exc.code))}
    except Exception as exc:
        traceback.print_exc()
        reason = f"unexpected {type(exc).__name__}: {exc}"
        doc = {**{k: v for k, v in doc.items() if k != "verdict"}, **_not_run_doc(reason)}
    try:
        write_verdict(args.out, doc)
    except OSError as exc:
        print(f"D3 gate: cannot write the verdict to {args.out}: {exc}", file=sys.stderr)
        return EXIT["NOT_RUN"]
    print(f"D3 gate: {doc['verdict']}  {doc.get('detail') or doc.get('reason', '')}")
    return EXIT[doc["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
