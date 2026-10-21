"""Write ``crates/qd-train/tests/fixtures/lora-gate-cases.json``: ledger lines every ``lora``-label
gate must give the same verdict on.

The cases are the row BYTES: ``crates/qd-train/tests/lora_gate_parity.rs`` hands each ``line`` to
``qd_train::ledger::check_row_line`` and ``python/tests/test_lora_gate_parity.py`` hands the same
bytes to ``Ledger.append`` / ``Ledger.verify_chain``; both must say ``expect``. A stale fixture
fails the Python test, so the file cannot drift from this generator.

    PYTHONPATH=python python tools/gen_lora_gate_cases.py            # rewrite the fixture
    PYTHONPATH=python python tools/gen_lora_gate_cases.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from qd_train.ledger import Environment, LedgerRow, Protocol, _canonical  # noqa: E402
from qd_train.tristate import NotRun, Ran, TriState  # noqa: E402

FIXTURE = REPO / "crates/qd-train/tests/fixtures/lora-gate-cases.json"
MANIFEST = "c" * 64

RUST_NAMED: dict[str, TriState] = {
    "train.span_head_init_digest": Ran(passed=True, value="ab" * 32),
    "lora.d17_init_digest": Ran(passed=True, value="cd" * 32),
    "lora.a_init_digest": Ran(passed=True, value="ef" * 32),
    "lora.base_safetensors_sha256": Ran(passed=True, value="01" * 32),
    "lora.dtype_map_sha256": Ran(passed=True, value="23" * 32),
    "corpus.plan_seed": Ran(passed=True, value=0),
    "lora.param_groups": Ran(passed=True, value="lora x2 at lr_scale 1.0, d17 x1 at lr_scale 0.5"),
}
#: The names ``tools/lora_reference_run.py`` wrote before Fable's 2026-10-21 ruling.
INIT_NAMED: dict[str, TriState] = {
    "init.span_head_digest": Ran(passed=True, value="ab" * 32),
    "init.d17_digest": Ran(passed=True, value="cd" * 32),
    "init.lora_a_digest": Ran(passed=True, value="ef" * 32),
    "init.base_safetensors_sha256": Ran(passed=True, value="01" * 32),
    "init.dtype_map_sha256": Ran(passed=True, value="23" * 32),
    "corpus.plan_seed": Ran(passed=True, value="0"),
}


def _line(
    *,
    label: str = "lora",
    metrics: dict[str, TriState],
    status: str = "completed",
    manifest: str | None = MANIFEST,
) -> str:
    env = Environment(
        torch="2.12.1", transformers_sha="abc", device="cpu", host="fixture",
        fla_present=NotRun(reason="no CUDA"), causal_conv1d_present=NotRun(reason="no CUDA"),
    )
    row = LedgerRow(
        row_id="00000000-0000-4000-8000-000000000001", written_at="2026-10-21T00:00:00+00:00",
        prev_row_hash=None,
        protocol=Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                          backbone_commit="b" * 40, recipe_hash="r" * 64, seed=0),
        run_kind="ft", status=status, quick=True, quick_reason="gate parity fixture",
        code_commit="c" * 40, env=env, metrics=dict(metrics), noul_rate=NotRun(reason="x"),
        controls={}, gates={}, wall_clock_s=1.0, cost_usd=0.0, wall_clock_source="recorder",
        recipe={"optimizer_recipe": label}, train_manifest_sha256=manifest,
    )
    return _canonical(row.to_json())


def build_cases() -> list[dict[str, str]]:
    mixed = {k: v for k, v in RUST_NAMED.items() if k != "lora.d17_init_digest"}
    mixed["init.d17_digest"] = INIT_NAMED["init.d17_digest"]
    not_run = {**RUST_NAMED, "lora.a_init_digest": NotRun(reason="never measured")}
    bad_seed = {**RUST_NAMED, "corpus.plan_seed": Ran(passed=True, value="0")}
    bad_hex = {**RUST_NAMED, "lora.dtype_map_sha256": Ran(passed=True, value="A" * 64)}
    bad_groups = {**RUST_NAMED, "lora.param_groups": Ran(passed=True, value="lora")}
    cases = [
        ("valid_rust_named", "accept", _line(metrics=RUST_NAMED)),
        ("init_named", "refuse", _line(metrics=INIT_NAMED)),
        ("init_named_no_manifest", "refuse", _line(metrics=INIT_NAMED, manifest=None)),
        ("mixed_rust_and_init_names", "refuse", _line(metrics=mixed)),
        ("rust_named_without_manifest_sha", "refuse", _line(metrics=RUST_NAMED, manifest=None)),
        ("bare_lora", "refuse", _line(metrics={})),
        ("not_run_digest", "refuse", _line(metrics=not_run)),
        ("plan_seed_as_string", "refuse", _line(metrics=bad_seed)),
        ("upper_case_digest", "refuse", _line(metrics=bad_hex)),
        ("param_groups_wrong_shape", "refuse", _line(metrics=bad_groups)),
        ("upper_case_label", "refuse", _line(label="LORA", metrics=RUST_NAMED)),
        ("homoglyph_label", "refuse", _line(label="lоra", metrics=RUST_NAMED)),
        ("f32_without_evidence", "accept", _line(label="f32", metrics={})),
        ("failed_lora_without_evidence", "accept",
         _line(metrics={}, status="failed", manifest=None)),
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
            print(f"{FIXTURE} is stale: rerun tools/gen_lora_gate_cases.py", file=sys.stderr)
            return 1
        return 0
    FIXTURE.write_text(want, encoding="utf-8")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
