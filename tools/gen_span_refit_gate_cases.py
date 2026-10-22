"""Write ``crates/qd-train/tests/fixtures/span-refit-gate-cases.json``: ledger lines the span-refit
span-null check must give the same verdict on in Python and in Rust.

Sibling of ``tools/gen_lora_gate_cases.py`` with the same shape. The cases are the row BYTES:
``crates/qd-train/tests/span_refit_gate_parity.rs`` hands each ``line`` to
``qd_train::ledger::check_row_line`` and ``python/tests/test_span_refit_gate_parity.py`` hands the
same bytes to ``Ledger.append`` / ``Ledger.verify_chain``; both must say ``expect``. A stale
fixture fails the Python test, so the file cannot drift from this generator.

    PYTHONPATH=python python tools/gen_span_refit_gate_cases.py            # rewrite the fixture
    PYTHONPATH=python python tools/gen_span_refit_gate_cases.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from qd_train.ledger import (  # noqa: E402
    SPAN_REFIT_SPAN_NULL_SCORER_SHAS,
    SPAN_REFIT_VALIDITY_GATE,
    Environment,
    LedgerRow,
    Protocol,
    _canonical,
)
from qd_train.span_null import (  # noqa: E402
    SPAN_END_TOP1_METRIC,
    SPAN_NULL_METRIC,
    SPAN_START_TOP1_METRIC,
)
from qd_train.tristate import NotRun, Ran, TriState  # noqa: E402

FIXTURE = REPO / "crates/qd-train/tests/fixtures/span-refit-gate-cases.json"
NEW_SHA = next(iter(SPAN_REFIT_SPAN_NULL_SCORER_SHAS))
OLD_SHA = "b88715afc031a2ef38aead1b0f0320d7d0763d4a990532ac8015c0652d7e07d5"
THREE = (SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC, SPAN_NULL_METRIC)


def _three(total: int = 50) -> dict[str, TriState]:
    return {n: Ran(passed=True, value=0.5, n=25, n_total=total) for n in THREE}


def _line(
    *,
    metrics: dict[str, TriState],
    gates: dict[str, TriState] | None = None,
    sha: str = NEW_SHA,
    tag: str = "span-refit-c",
    status: str = "completed",
) -> str:
    env = Environment(
        torch="2.12.1", transformers_sha="abc", device="cpu", host="fixture",
        fla_present=NotRun(reason="no CUDA"), causal_conv1d_present=NotRun(reason="no CUDA"),
    )
    row = LedgerRow(
        row_id="00000000-0000-4000-8000-000000000002", written_at="2026-10-22T00:00:00+00:00",
        prev_row_hash=None,
        protocol=Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                          backbone_commit="b" * 40, recipe_hash="r" * 64, seed=0),
        run_kind="eval", status=status, quick=True, quick_reason="span-refit parity fixture",
        code_commit="c" * 40, env=env, metrics=dict(metrics), noul_rate=NotRun(reason="x"),
        controls={}, gates=dict(gates or {}), wall_clock_s=1.0, cost_usd=0.0,
        wall_clock_source="recorder",
        recipe={"tag": tag, "script_sha256": {"span_refit_score.py": sha}},
    )
    return _canonical(row.to_json())


def build_cases() -> list[dict[str, str]]:
    only_joint = {"val_top1.span": Ran(passed=True, value=5596, n=5596, n_total=7238)}
    two = {k: v for k, v in _three().items() if k != SPAN_END_TOP1_METRIC}
    disagree = {**_three(), SPAN_NULL_METRIC: Ran(passed=True, value=0.5, n=25, n_total=49)}
    no_total = {**_three(), SPAN_NULL_METRIC: Ran(passed=True, value=0.5)}
    not_run = {**_three(), SPAN_START_TOP1_METRIC: NotRun(reason="never scored")}
    gate_failed = {SPAN_REFIT_VALIDITY_GATE: Ran(passed=False, value=5000)}
    gate_passed = {SPAN_REFIT_VALIDITY_GATE: Ran(passed=True, value=5596)}
    cases = [
        ("new_sha_valid", "accept", _line(metrics=_three())),
        ("new_sha_missing_the_metrics", "refuse", _line(metrics=only_joint)),
        ("new_sha_missing_one_metric", "refuse", _line(metrics=two)),
        ("new_sha_counts_disagree", "refuse", _line(metrics=disagree)),
        ("new_sha_null_states_no_count", "refuse", _line(metrics=no_total)),
        ("new_sha_a_metric_not_run", "refuse", _line(metrics=not_run)),
        ("arm_a_after_validity_gate_failed", "accept",
         _line(metrics=only_joint, gates=gate_failed, tag="span-refit-a")),
        ("arm_a_gate_passed_without_metrics", "refuse",
         _line(metrics=only_joint, gates=gate_passed, tag="span-refit-a")),
        ("old_sha_without_metrics", "accept", _line(metrics=only_joint, sha=OLD_SHA)),
        ("new_sha_failed_row", "accept", _line(metrics={}, status="failed")),
        ("new_sha_other_tag", "accept", _line(metrics={}, tag="calib-fit")),
    ]
    return [{"name": n, "expect": e, "line": ln} for n, e, ln in cases]


def render() -> str:
    return json.dumps(build_cases(), indent=2, ensure_ascii=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--check", action="store_true", help="exit 1 if the fixture is not current")
    a = p.parse_args(argv)
    want = render()
    if a.check:
        if not FIXTURE.exists() or FIXTURE.read_text(encoding="utf-8") != want:
            print(f"{FIXTURE} is stale: rerun tools/gen_span_refit_gate_cases.py", file=sys.stderr)
            return 1
        return 0
    FIXTURE.write_text(want, encoding="utf-8")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
