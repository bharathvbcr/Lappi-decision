"""``tools/d3_quant_gate.py``: the D3 gate that admits (or refuses) an MLX affine q8 g64 base.

The reference quantiser is mlx's own ``mx.quantize`` (the call the ojas runbook names). The
oracle here is written by hand from the affine rule mlx 0.31.2 follows, and is itself checked
against mlx on random groups; the hand-derived literal cases below pin numbers worked out on
paper. The mlx tests need the ml venv (``make torch-pytest``) and skip, as not run, where mlx is
absent; the spec, verdict and NOT_RUN tests need nothing and always run.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import d3_quant_gate as gate  # noqa: E402

from qd_train.tristate import NotRun, Ran  # noqa: E402

HAS_MLX = importlib.util.find_spec("mlx") is not None
needs_mlx = pytest.mark.skipif(
    not HAS_MLX, reason="mlx is not installed in this interpreter (not run)"
)

BLOCK = {
    "argmax_agreement_min": 0.99,
    "mean_abs_delta_logp_max": 0.02,
    "min_prompts": 4,
    "quantiser": {"mode": "affine", "bits": 8, "group_size": 64},
}


def _prereg(tmp_path: Path, block: dict | None = None, *, name: str = "prereg.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"d3_gate": BLOCK if block is None else block}))
    return p


def _spec(tmp_path: Path, **over: object) -> gate.GateSpec:
    spec = gate.load_spec(_prereg(tmp_path, {**BLOCK, **over}))
    assert isinstance(spec, gate.GateSpec)
    return spec


def oracle_group(g: np.ndarray) -> tuple[np.float32, np.float32]:
    """(scale, bias) of one affine q8 group, by hand.

    The edge is the end of the range farther from zero (the max on a tie unless the group is
    negative). bias is the edge; the raw step is range/255, signed so q=0 sits on the edge;
    q0 = round(-edge/step) is where zero falls, clamped to 255 on the min side, and a non-zero
    q0 snaps scale to -edge/q0 so zero is representable. A constant group has range 0 and takes
    the 1e-7 floor. float32 throughout: a tie such as 127.5 must round as mlx's float32 does.
    """
    g = g.astype(np.float32)
    lo, hi = g.min(), g.max()
    on_max = abs(hi) > abs(lo) or (abs(hi) == abs(lo) and hi >= 0)
    edge, other = (hi, lo) if on_max else (lo, hi)
    step = np.float32(max(abs(other - edge) / np.float32(255.0), np.float32(1e-7)))
    step = -step if on_max else step
    q0 = np.round(-edge / step)
    if not on_max:
        q0 = min(q0, np.float32(255.0))
    scale = np.float32(-edge / q0) if q0 != 0 else np.float32(step)
    return scale, np.float32(edge)


def oracle_dequant(g: np.ndarray) -> np.ndarray:
    scale, bias = oracle_group(g)
    q = np.clip(np.round((g.astype(np.float32) - bias) / scale), 0, 255)
    return (q * scale + bias).astype(np.float32)


def fq(row: np.ndarray, bits: int = 8, group: int = 64) -> np.ndarray:
    import mlx.core as mx

    # The float32 oracle checks the quantisation arithmetic exactly; the public ``fake_quant``
    # accepts bf16 only (tested below), so this goes through the step after that check.
    out = gate._fake_quant(
        mx.array(row.astype(np.float32))[None, :], bits=bits, group_size=group, name="row"
    )
    return np.array(out)[0]


# ---- hand-derived literals -------------------------------------------------------------------


@needs_mlx
def test_ramp_scale_and_bias_by_hand() -> None:
    # 0..63: max 63 is the edge, step = -63/255, q0 = round(63 / (63/255)) = 255, so
    # scale = -63/255 and bias = 63.
    g = np.arange(64, dtype=np.float32)
    scale, bias = oracle_group(g)
    assert bias == np.float32(63.0)
    assert scale == pytest.approx(-63.0 / 255.0, rel=1e-6)
    out = fq(g)
    # Round-trip error is at most half a step: 63/255/2.
    assert np.abs(out - g).max() <= 63.0 / 255.0 / 2 + 1e-6
    np.testing.assert_allclose(out, oracle_dequant(g), atol=1e-5)


@needs_mlx
def test_identical_values_round_trip_exactly() -> None:
    for v in (0.5, -2.0, 0.0, 3.25):
        g = np.full(64, v, dtype=np.float32)
        np.testing.assert_array_equal(fq(g), g)  # a collapsed scale would give 0, not v
    scale, bias = oracle_group(np.full(64, 0.5, dtype=np.float32))
    assert bias == np.float32(0.5) and abs(float(scale)) == pytest.approx(1e-7)


@needs_mlx
def test_group_straddling_zero_keeps_its_edges() -> None:
    g = np.linspace(-1.0, 1.0, 64, dtype=np.float32)  # |min| == |max|: a tie, resolved to the max
    scale, bias = oracle_group(g)
    assert bias == np.float32(1.0) and scale == pytest.approx(-2.0 / 255.0, rel=1e-2)
    out = fq(g)
    assert out[0] == pytest.approx(-1.0, abs=abs(float(scale)) / 2) and out[-1] == pytest.approx(
        1.0, abs=1e-6
    )
    np.testing.assert_allclose(out, oracle_dequant(g), atol=1e-6)


@needs_mlx
def test_all_negative_group_has_a_positive_scale() -> None:
    g = np.linspace(-2.4, -0.01, 64, dtype=np.float32)
    scale, bias = oracle_group(g)
    assert (
        bias == np.float32(-2.4) and scale > 0
    )  # q=0 sits on the min edge; the clamp to 255 binds
    assert float(scale) == pytest.approx(2.4 / 255.0, rel=1e-6)
    np.testing.assert_allclose(fq(g), oracle_dequant(g), atol=1e-6)


# ---- the oracle against mlx, and the error bound ---------------------------------------------


@needs_mlx
def test_oracle_matches_mlx_on_random_groups_and_error_is_bounded() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(7)
    rows = [rng.standard_normal(64) * 10 ** rng.uniform(-3, 1) + rng.normal() for _ in range(400)]
    rows += [
        -np.abs(rng.standard_normal(64)),
        np.abs(rng.standard_normal(64)),
        np.zeros(64),
        np.full(64, -2.0),
    ]
    for g in rows:
        g = g.astype(np.float32)
        _, s, b = mx.quantize(mx.array(g)[None, :], group_size=64, bits=8)
        scale, bias = oracle_group(g)
        assert np.float32(s.item()) == scale and np.float32(b.item()) == bias
        err = np.abs(fq(g) - g).max()
        # Half a step for an interior value; snapping the scale so zero is representable can
        # leave the far edge up to one more step out of reach, hence 1.5 steps (ojas's bound,
        # which the gate also enforces).
        assert err <= 1.5 * abs(float(scale)) + 1e-6


@needs_mlx
def test_last_partial_group_has_its_own_scale() -> None:
    # Width 192: groups 0 and 1 are small, group 2 is 100x larger. Each is quantised on its own.
    rng = np.random.default_rng(3)
    row = np.concatenate([rng.standard_normal(128) * 0.01, rng.standard_normal(64)]).astype(
        np.float32
    )
    out = fq(row)
    for k in range(3):
        g = row[64 * k : 64 * (k + 1)]
        np.testing.assert_allclose(out[64 * k : 64 * (k + 1)], oracle_dequant(g), atol=1e-6)
    assert np.abs(out[:128] - row[:128]).max() < 1e-3  # not smeared by the large group


@needs_mlx
@pytest.mark.parametrize("width", [1, 63, 65, 100, 130])
def test_non_multiple_of_group_width_is_refused(width: int) -> None:
    import mlx.core as mx

    with pytest.raises(SystemExit, match="not a multiple of the group size 64"):
        gate.fake_quant(mx.zeros((2, width), dtype=mx.bfloat16), bits=8, group_size=64, name="w")


@needs_mlx
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weight_is_refused(bad: float) -> None:
    import mlx.core as mx

    row = np.ones((1, 64), dtype=np.float32)
    row[0, 5] = bad
    with pytest.raises(SystemExit, match="non-finite weight"):
        gate.fake_quant(mx.array(row).astype(mx.bfloat16), bits=8, group_size=64, name="w")


@needs_mlx
def test_bf16_input_gives_bf16_output() -> None:
    import mlx.core as mx

    w = mx.array(np.random.default_rng(1).standard_normal((2, 128)).astype(np.float32)).astype(
        mx.bfloat16
    )
    assert gate.fake_quant(w, bits=8, group_size=64).dtype == mx.bfloat16


# ---- spec and verdict: no mlx needed ---------------------------------------------------------


def test_threshold_is_the_preregistrations_not_a_copy(tmp_path: Path) -> None:
    spec = _spec(tmp_path, argmax_agreement_min=0.5, mean_abs_delta_logp_max=7.0, min_prompts=2)
    assert (spec.argmax_agreement_min, spec.mean_abs_delta_logp_max, spec.min_prompts) == (
        0.5,
        7.0,
        2,
    )
    # The same metrics flip verdict when only the pre-registered bar moves.
    m = gate.Metrics(n=10, argmax_agreement=0.95, mean_abs_delta_logp=0.01)
    assert isinstance(gate.decide(_spec(tmp_path), m, n_total=10), Ran)
    assert gate.decide(_spec(tmp_path), m, n_total=10).passed is False
    assert gate.decide(spec, m, n_total=10).passed is True
    # And the tool source carries no second copy of the Fable numbers.
    tree = ast.parse((REPO / "tools" / "d3_quant_gate.py").read_text())
    numbers = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
    }
    assert not numbers & {0.99, 0.02, 99.0, 500, 0.9899}


def test_both_bars_are_needed_and_the_bounds_are_inclusive(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    at_bar = gate.Metrics(n=10, argmax_agreement=0.99, mean_abs_delta_logp=0.02)
    assert gate.decide(spec, at_bar, n_total=10).passed is True
    assert gate.decide(spec, gate.Metrics(10, 0.9899, 0.0), n_total=10).passed is False
    assert gate.decide(spec, gate.Metrics(10, 1.0, 0.0201), n_total=10).passed is False


def test_too_few_prompts_is_not_run_never_pass(tmp_path: Path) -> None:
    spec = _spec(tmp_path, min_prompts=500)
    r = gate.decide(
        spec, gate.Metrics(n=499, argmax_agreement=1.0, mean_abs_delta_logp=0.0), n_total=499
    )
    assert isinstance(r, NotRun) and "500" in r.reason


def test_missing_prereg_or_block_is_not_run(tmp_path: Path) -> None:
    missing = gate.load_spec(tmp_path / "nope.json")
    assert isinstance(missing, NotRun) and "does not exist" in missing.reason
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({"other": 1}))
    no_block = gate.load_spec(empty)
    assert isinstance(no_block, NotRun) and "d3_gate" in no_block.reason


def test_the_repos_bf16_preregistration_carries_no_d3_block() -> None:
    # Verified fact the tool relies on: campaign/lora-2b-preregistered.json states D3 as
    # "gated, not built", so the real gate is NOT_RUN until a sibling pre-registration carries
    # a d3_gate block.
    r = gate.load_spec(REPO / "campaign" / "lora-2b-preregistered.json")
    assert isinstance(r, NotRun)


@pytest.mark.parametrize(
    "patch",
    [
        {"argmax_agreement_min": 1.5},
        {"argmax_agreement_min": "0.99"},
        {"mean_abs_delta_logp_max": float("nan")},
        {"min_prompts": 0},
        {"min_prompts": True},
        {"quantiser": {"mode": "mxfp4", "bits": 8, "group_size": 64}},
        {"quantiser": {"mode": "affine", "bits": 3, "group_size": 64}},
        {"quantiser": {"mode": "affine", "bits": 8}},
    ],
)
def test_malformed_block_exits_rather_than_not_run(tmp_path: Path, patch: dict) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"d3_gate": {**BLOCK, **patch}}))
    with pytest.raises(SystemExit):
        gate.load_spec(p)


def test_compare_refuses_rather_than_drops() -> None:
    ls = [-0.1, -2.3]
    with pytest.raises(SystemExit):
        gate.compare({"a": ls}, {"b": ls}, {"a": 0})
    with pytest.raises(SystemExit):
        gate.compare({"a": ls}, {"a": ls}, {})
    with pytest.raises(SystemExit):
        gate.compare({"a": ls}, {"a": ls}, {"a": 2})
    with pytest.raises(SystemExit):
        gate.compare({"a": ls}, {"a": [math.nan, 0.0]}, {"a": 0})
    m = gate.compare(
        {"a": [-0.1, -3.0], "b": [-0.1, -3.0]},
        {"a": [-0.1, -3.0], "b": [-3.0, -0.2]},
        {"a": 0, "b": 0},
    )
    assert m.argmax_agreement == 0.5 and m.mean_abs_delta_logp == pytest.approx(1.45)


def test_select_targets_names_the_ten_kinds_and_not_in_proj_a_b() -> None:
    names = [
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.linear_attn.in_proj_a.weight",
        "model.layers.3.linear_attn.in_proj_b.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "language_model.model.layers.1.linear_attn.in_proj_qkv.weight",
        "model.layers.1.mlp.down_proj.bias",
    ]
    assert gate.select_targets(names) == [names[0], names[3], names[6]]


# ---- NOT_RUN end to end through the CLI, and a toy gate with no model -------------------------


def test_cli_emits_not_run_exit_3_when_inputs_are_absent(tmp_path: Path) -> None:
    out = tmp_path / "v.json"
    prereg = _prereg(tmp_path)
    rc = gate.main(
        [
            "--preregistration",
            str(prereg),
            "--model",
            str(tmp_path / "no-model"),
            "--items",
            str(tmp_path / "no-items.jsonl"),
            "--out",
            str(out),
        ]
    )
    doc = json.loads(out.read_text())
    assert (
        rc == 3
        and doc["verdict"] == "NOT_RUN"
        and doc["state"] == "not_run"
        and "passed" not in doc
    )
    assert doc["reason"]
    # With no pre-registration either:
    rc = gate.main(
        [
            "--preregistration",
            str(tmp_path / "x.json"),
            "--model",
            str(tmp_path),
            "--items",
            str(tmp_path),
            "--out",
            str(out),
        ]
    )
    assert rc == 3 and json.loads(out.read_text())["verdict"] == "NOT_RUN"


def test_cli_script_runs_as_a_program_and_exits_3(tmp_path: Path) -> None:
    out = tmp_path / "v.json"
    cp = subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPO / "tools" / "d3_quant_gate.py"),
            "--preregistration",
            str(tmp_path / "x.json"),
            "--model",
            str(tmp_path),
            "--items",
            str(tmp_path / "i"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 3, cp.stderr
    assert json.loads(out.read_text())["verdict"] == "NOT_RUN"


def _toy(seed: int, width: int = 128, hidden: int = 16, items: int = 12):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    w_up = (rng.standard_normal((hidden, width)) * 0.05).astype(np.float32)
    w_down = (rng.standard_normal((width, hidden)) * 0.05).astype(np.float32)
    w_head = (rng.standard_normal((5, width)) * 0.05).astype(np.float32)
    weights = {
        "model.layers.0.mlp.up_proj.weight": mx.array(w_up).astype(mx.bfloat16),
        "model.layers.0.mlp.down_proj.weight": mx.array(w_down.T.copy()).astype(
            mx.bfloat16
        ),  # (hidden, width)
        "model.embed_tokens.weight": mx.array(w_head).astype(mx.bfloat16),
    }
    feats = mx.array(rng.standard_normal((items, width)).astype(np.float32))

    def score(overrides):
        w = {**weights, **overrides}
        h = feats @ w["model.layers.0.mlp.up_proj.weight"].astype(mx.float32).T
        h = h @ w["model.layers.0.mlp.down_proj.weight"].astype(mx.float32)
        z = h @ w["model.embed_tokens.weight"].astype(mx.float32).T
        ls = z - mx.logsumexp(z, axis=-1, keepdims=True)
        return {f"r{i}": [float(v) for v in row] for i, row in enumerate(np.array(ls))}

    gold = {f"r{i}": i % 5 for i in range(items)}
    return weights, score, gold


@needs_mlx
def test_toy_end_to_end_pass_and_the_embedding_is_left_alone(tmp_path: Path) -> None:
    weights, score, gold = _toy(0)
    spec = _spec(tmp_path, argmax_agreement_min=0.5, mean_abs_delta_logp_max=0.5, min_prompts=12)
    q = gate.quantise_weights(weights, spec)
    assert sorted(q) == ["model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.up_proj.weight"]
    result, m = gate.run_gate(spec, weights, score, gold)
    assert isinstance(result, Ran) and result.passed and m is not None and m.n == 12
    assert result.coverage_str() == "12/12" and gate.verdict_name(result) == "PASS"


@needs_mlx
def test_toy_end_to_end_fail_when_the_pre_registered_bar_is_unreachable(tmp_path: Path) -> None:
    weights, score, gold = _toy(1)
    spec = _spec(
        tmp_path, mean_abs_delta_logp_max=0.0, min_prompts=12
    )  # q8 error is never exactly zero here
    result, m = gate.run_gate(spec, weights, score, gold)
    assert isinstance(result, Ran) and not result.passed and gate.verdict_name(result) == "FAIL"
    assert m is not None and m.mean_abs_delta_logp > 0.0


@needs_mlx
def test_toy_gate_with_too_few_prompts_is_not_run(tmp_path: Path) -> None:
    weights, score, gold = _toy(2)
    result, m = gate.run_gate(_spec(tmp_path, min_prompts=13), weights, score, gold)
    assert isinstance(result, NotRun) and m is None


@needs_mlx
def test_quantise_weights_refuses_empty_and_non_bf16(tmp_path: Path) -> None:
    import mlx.core as mx

    spec = _spec(tmp_path)
    with pytest.raises(SystemExit, match="no wrapped linear"):
        gate.quantise_weights({"model.embed_tokens.weight": mx.zeros((4, 64))}, spec)
    with pytest.raises(SystemExit, match="expected bfloat16"):
        gate.quantise_weights({"model.layers.0.mlp.up_proj.weight": mx.zeros((4, 64))}, spec)
