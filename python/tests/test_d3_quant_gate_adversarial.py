"""Adversarial tests against lane L4 (tools/d3_quant_gate.py, the QD_PYTHON resolver pin).

Tests named ``test_BREAK_*`` fail against lane/ft-57fc7484-l4; ``test_held_*`` pass and pin what
survived the attack.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import d3_quant_gate as gate  # noqa: E402

from qd_train.tristate import NotRun  # noqa: E402

HAS_MLX = importlib.util.find_spec("mlx") is not None
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx not installed (not run)")
TOOL = REPO / "tools" / "d3_quant_gate.py"

BLOCK = {
    "argmax_agreement_min": 0.99,
    "mean_abs_delta_logp_max": 0.02,
    "min_prompts": 2,
    "quantiser": {"mode": "affine", "bits": 8, "group_size": 64},
}


def _prereg(tmp_path: Path, text: str | None = None) -> Path:
    p = tmp_path / "prereg.json"
    p.write_text(json.dumps({"d3_gate": BLOCK}) if text is None else text)
    return p


def _run(tmp_path: Path, prereg: Path, items: Path, model: Path, out: Path):
    return subprocess.run(
        [
            sys.executable, str(TOOL), "--preregistration", str(prereg), "--model", str(model),
            "--items", str(items), "--out", str(out),
        ],
        capture_output=True, text=True, timeout=300, check=False,
    )


# ---------------------------------------------------------------- verdict file / exit codes


def test_BREAK_stale_pass_survives_a_refused_run(tmp_path: Path) -> None:
    """A previous PASS at --out is left in place when this run exits before writing."""
    out = tmp_path / "verdict.json"
    out.write_text(json.dumps({"gate": "D3", "verdict": "PASS"}))
    bad = _prereg(tmp_path, json.dumps({"d3_gate": {**BLOCK, "min_prompts": 0}}))
    items = tmp_path / "i.jsonl"
    items.write_text("")
    r = _run(tmp_path, bad, items, tmp_path, out)
    assert r.returncode != 0
    assert not out.exists() or json.loads(out.read_text()).get("verdict") != "PASS", (
        "stale PASS verdict still at --out after a refused run"
    )


@needs_mlx
def test_BREAK_unreadable_items_exit_code_is_not_the_fail_code(tmp_path: Path) -> None:
    """Malformed items (could not run) exits 1, the same code as a gate FAIL."""
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}")
    items = tmp_path / "i.jsonl"
    items.write_text("{not json\n")
    r = _run(tmp_path, _prereg(tmp_path), items, model, tmp_path / "v.json")
    assert r.returncode == gate.EXIT["NOT_RUN"] or r.returncode not in (
        gate.EXIT["PASS"], gate.EXIT["FAIL"]
    ), f"exit {r.returncode} is the FAIL code for a run that never ran: {r.stderr[-200:]}"


def test_BREAK_verdict_records_preregistration_sha(tmp_path: Path) -> None:
    out = tmp_path / "v.json"
    items = tmp_path / "i.jsonl"
    items.write_text("")
    r = _run(tmp_path, tmp_path / "absent.json", items, tmp_path, out)
    assert r.returncode == gate.EXIT["NOT_RUN"], r.stderr
    prereg = _prereg(tmp_path)
    r = _run(tmp_path, prereg, items, tmp_path, out)  # model dir has no config.json -> NOT_RUN
    doc = json.loads(out.read_text())
    assert "preregistration_sha256" in doc, f"only the path is recorded: {sorted(doc)}"


def test_BREAK_heldout_items_path_is_not_refused(tmp_path: Path) -> None:
    """Training tools refuse a path segment 'heldout' (data_access); this eval tool reads it."""
    d = tmp_path / "heldout"
    d.mkdir()
    items = d / "items.jsonl"
    items.write_text("")
    out = tmp_path / "v.json"
    r = _run(tmp_path, _prereg(tmp_path), items, tmp_path, out)
    refused = r.returncode not in (gate.EXIT["NOT_RUN"],) or (
        out.exists() and "held" in out.read_text().lower().replace("heldout", "held")
        and json.loads(out.read_text()).get("verdict") != "NOT_RUN"
    )
    assert refused, "a heldout/ items path went through to the ordinary NOT_RUN/run path"


# ---------------------------------------------------------------- pre-registration parsing


def _spec_text(**over: object) -> str:
    return json.dumps({"d3_gate": {**BLOCK, **over}})


def test_BREAK_huge_int_threshold_is_an_overflow_traceback(tmp_path: Path) -> None:
    p = _prereg(tmp_path, '{"d3_gate": {"argmax_agreement_min": 1' + "0" * 400
                + ', "mean_abs_delta_logp_max": 0.02, "min_prompts": 2, "quantiser": '
                '{"mode": "affine", "bits": 8, "group_size": 64}}}')
    with pytest.raises(SystemExit):
        gate.load_spec(p)


def test_BREAK_duplicate_keys_last_wins_silently(tmp_path: Path) -> None:
    """Two argmax floors in one block: the file shows 0.5 first and 0.99 last; no refusal."""
    text = (
        '{"d3_gate": {"argmax_agreement_min": 0.99, "argmax_agreement_min": 0.01,'
        ' "mean_abs_delta_logp_max": 0.02, "min_prompts": 2,'
        ' "quantiser": {"mode": "affine", "bits": 8, "group_size": 64}}}'
    )
    with pytest.raises(SystemExit):
        gate.load_spec(_prereg(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        '{"d3_gate": null}',
        "[]",
        '{"d3_gate": false}',
        '{"d3_gate": []}',
        _spec_text(argmax_agreement_min=True),
        _spec_text(argmax_agreement_min=0),
        _spec_text(argmax_agreement_min=0.0),
        _spec_text(argmax_agreement_min=-0.5),
        _spec_text(min_prompts=-3),
        _spec_text(min_prompts=2.0),
        _spec_text(mean_abs_delta_logp_max=True),
        _spec_text(mean_abs_delta_logp_max=-1),
        _spec_text(quantiser={"mode": "affine", "bits": 8.0, "group_size": 64}),
        _spec_text(quantiser={"mode": "affine", "bits": 8, "group_size": 0}),
        _spec_text(quantiser={"mode": "mxfp4", "bits": 8, "group_size": 64}),
        _spec_text(quantiser={"mode": "affine", "bits": 3, "group_size": 64}),
        '{"d3_gate": {"argmax_agreement_min": NaN, "mean_abs_delta_logp_max": 0.02,'
        ' "min_prompts": 2, "quantiser": {"mode": "affine", "bits": 8, "group_size": 64}}}',
        '{"d3_gate": {"argmax_agreement_min": 0.9, "mean_abs_delta_logp_max": Infinity,'
        ' "min_prompts": 2, "quantiser": {"mode": "affine", "bits": 8, "group_size": 64}}}',
        '{"d3_gate": {"argmax_agreement_min": 0.9, "mean_abs_delta_logp_max": NaN,'
        ' "min_prompts": 2, "quantiser": {"mode": "affine", "bits": 8, "group_size": 64}}}',
        '{"d3_gate": {"argmax_agreement_min": Infinity, "mean_abs_delta_logp_max": 0.1,'
        ' "min_prompts": 2, "quantiser": {"mode": "affine", "bits": 8, "group_size": 64}}}',
    ],
)
def test_held_bad_preregistrations_never_yield_a_spec(tmp_path: Path, text: str) -> None:
    try:
        got = gate.load_spec(_prereg(tmp_path, text))
    except SystemExit:
        return
    assert isinstance(got, NotRun)


def test_held_agreement_exactly_one_is_accepted_and_needs_perfect_agreement(tmp_path: Path) -> None:
    spec = gate.load_spec(_prereg(tmp_path, _spec_text(argmax_agreement_min=1)))
    assert isinstance(spec, gate.GateSpec)
    m = gate.Metrics(n=10, argmax_agreement=0.9, mean_abs_delta_logp=0.0)
    assert gate.verdict_name(gate.decide(spec, m, n_total=10)) == "FAIL"


def test_held_nan_score_and_bad_gold_never_pass() -> None:
    nan = float("nan")
    for base, quant, gold in (
        ({"a": [0.0, nan]}, {"a": [0.0, -1.0]}, {"a": 0}),
        ({"a": [0.0, -1.0]}, {"a": [0.0, nan]}, {"a": 0}),
        ({"a": [0.0, -1.0]}, {"a": [0.0, -1.0]}, {"a": 2}),
        ({"a": [0.0, -1.0]}, {"a": [0.0, -1.0]}, {"a": -1}),
        ({"a": [0.0, -1.0]}, {}, {"a": 0}),
        ({}, {}, {}),
    ):
        with pytest.raises(SystemExit):
            gate.compare(base, quant, gold)


def test_held_too_few_prompts_is_not_run(tmp_path: Path) -> None:
    spec = gate.load_spec(_prereg(tmp_path))
    assert isinstance(spec, gate.GateSpec)
    r, m = gate.run_gate(spec, {}, lambda o: {}, {"a": 0})
    assert isinstance(r, NotRun) and m is None


# ---------------------------------------------------------------- fake_quant


def _bf(arr):
    import mlx.core as mx

    return arr.astype(mx.bfloat16)


@needs_mlx
def test_BREAK_correct_mlx_output_refused_at_small_weight_scale() -> None:
    """mlx's exact output is refused when a group's range is ~1e-7: the bound has no floor for
    mlx's scale epsilon, so a correct quantisation fails the step check."""
    import mlx.core as mx

    mx.random.seed(7)
    w = _bf(mx.random.normal((16, 128)) * 8.2e-8)
    wq, s, b = mx.quantize(w, group_size=64, bits=8)
    mx.eval(mx.dequantize(wq, s, b, group_size=64, bits=8))  # mlx itself produced this
    gate.fake_quant(w, bits=8, group_size=64)  # must not raise


@needs_mlx
def test_BREAK_empty_weight_is_accepted() -> None:
    import mlx.core as mx

    with pytest.raises(SystemExit):
        gate.fake_quant(mx.zeros((4, 0), dtype=mx.bfloat16), bits=8, group_size=64)
    with pytest.raises(SystemExit):
        gate.fake_quant(mx.zeros((0, 64), dtype=mx.bfloat16), bits=8, group_size=64)


@needs_mlx
def test_BREAK_fake_quant_accepts_non_bf16_directly() -> None:
    """bf16 is enforced only in quantise_weights, not at fake_quant (the documented call)."""
    import mlx.core as mx

    with pytest.raises(SystemExit):
        gate.fake_quant(mx.random.normal((4, 64)), bits=8, group_size=64)


@needs_mlx
def test_held_fake_quant_refusals() -> None:
    import mlx.core as mx

    bf = mx.bfloat16
    nan = mx.random.normal((4, 64)).astype(bf)
    nan[1, 3] = float("nan")
    inf = mx.random.normal((4, 64)).astype(bf)
    inf[1, 3] = float("-inf")
    for w in (
        mx.zeros((64,), dtype=bf), mx.zeros((2, 2, 64), dtype=bf), mx.zeros((2, 63), dtype=bf),
        mx.zeros((2, 127), dtype=bf), mx.zeros((2, 65), dtype=bf), nan, inf,
    ):
        with pytest.raises(SystemExit):
            gate.fake_quant(w, bits=8, group_size=64)


@needs_mlx
def test_held_step_bound_catches_a_wrong_result() -> None:
    import mlx.core as mx

    mx.random.seed(1)
    w = _bf(mx.random.normal((4, 64)))
    wq, s, b = mx.quantize(w, group_size=64, bits=8)
    d = mx.dequantize(wq, s, b, group_size=64, bits=8)
    step = (w.max(axis=1) - w.min(axis=1)) / 255
    with pytest.raises(SystemExit):
        gate._check_step_bound(w, _bf(d + (1.4 * step)[:, None]), 8, 64, "shift")
    with pytest.raises(SystemExit):
        gate._check_step_bound(w, _bf(mx.zeros_like(w) + w.mean()), 8, 64, "flat")
    for w in (
        mx.full((4, 64), 0.5, dtype=mx.bfloat16), mx.zeros((4, 64), dtype=mx.bfloat16),
        _bf(mx.random.normal((8, 64)) * 1e-3), _bf(mx.full((4, 64), 1e-39)),
    ):
        gate.fake_quant(w, bits=8, group_size=64)


# ---------------------------------------------------------------- resolver pin


SITES = (
    "crates/qd-prep/tests/convert_licence_parity.rs",
    "crates/qd-metal/tests/ledger_rows.rs",
    "crates/qd-preflight/tests/schema_compat.rs",
)


def test_BREAK_a_fourth_python_resolver_exists_outside_the_pin() -> None:
    """qd-export/tests/ledger_real_producer.rs reads QD_PYTHON on its own (expect, no worktree
    fallback): the same linked-worktree failure the pin fixed in three places, in a fourth."""
    pat = re.compile(r'env::var(?:_os)?\("QD_PYTHON"\)')
    extra = [
        str(p.relative_to(REPO))
        for p in (REPO / "crates").rglob("*.rs")
        if pat.search(p.read_text(encoding="utf-8")) and str(p.relative_to(REPO)) not in SITES
    ]
    assert not extra, f"QD_PYTHON resolved outside the pinned block: {extra}"


def test_held_pin_catches_whitespace_crlf_and_missing_trailing_newline(tmp_path: Path) -> None:
    block = re.compile(
        r"^// BEGIN qd-python-resolver\n.*?^// END qd-python-resolver\n", re.DOTALL | re.MULTILINE
    )
    src = (REPO / SITES[0]).read_text(encoding="utf-8")
    base = block.findall(src)
    assert len(base) == 1
    assert block.findall(src.replace("let mut out = Vec::new();", "let mut out = Vec::new(); "))[0] != base[0]
    assert block.findall(src.replace("\r", "").replace("\n", "\r\n")) == []  # CRLF: no match, loud
    assert block.findall(src.replace("// END qd-python-resolver", "//END qd-python-resolver")) == []
    assert len(block.findall(src + "\n" + base[0])) == 2
