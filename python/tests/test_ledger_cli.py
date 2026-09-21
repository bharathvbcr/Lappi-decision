"""`qd-ledger`: the entry point docs/ledger-schema.md names.

The schema says *"`qd-ledger verify` recomputes the chain and reports the first
break"*, but nothing implemented it, so `verify_chain`, `promotion_verdict`,
`PromotionVerdict.__str__` and `Ran.coverage_str` were reachable only from tests.

The contract asserted here is mostly about exit codes. "This ledger refuses to
promote" (1) and "I could not read this ledger" (2) must never share one, because a
caller that treats them alike turns an unreadable record into a settled answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.__main__ import main
from qd_train.ledger import (
    REQUIRED_CONTROLS,
    REQUIRED_GATES,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.tristate import NotRun, Ran

REPO = Path(__file__).resolve().parents[2]


def _protocol(seed: int) -> Protocol:
    return Protocol(
        data_snapshot_hash="d" * 64,
        tokenizer_hash="t" * 64,
        backbone_commit="b" * 40,
        recipe_hash="r" * 64,
        seed=seed,
    )


def _env() -> Environment:
    return Environment(
        torch="2.12.1", transformers_sha="abc123", device="mps", host="test",
        fla_present=NotRun(reason="no CUDA on this host"),
        causal_conv1d_present=NotRun(reason="no CUDA on this host"),
    )


def _ledger(tmp_path: Path, *, seeds=(1, 2, 3), n=300, n_total=300) -> Path:
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in seeds:
        proto = _protocol(seed)
        with RunRecorder(led, protocol=proto, run_kind="ft", repo=REPO, env=_env()) as rec:
            for g in REQUIRED_GATES:
                rec.gate(g, Ran(passed=True, value=1.0, n=n, n_total=n_total))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=n, n_total=n_total))
    return led.path


def test_verify_reports_an_intact_chain(tmp_path: Path, capsys):
    assert main(["--ledger", str(_ledger(tmp_path)), "verify"]) == 0
    assert "chain intact over 3 row(s)" in capsys.readouterr().out


def test_verify_reports_the_first_break(tmp_path: Path, capsys):
    path = _ledger(tmp_path)
    lines = path.read_bytes().splitlines()
    obj = json.loads(lines[1])
    obj["notes"] = "edited after the fact"
    lines[1] = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(b"\n".join(lines) + b"\n")

    assert main(["--ledger", str(path), "verify"]) == 1
    assert "CHAIN BROKEN" in capsys.readouterr().err


def test_a_missing_ledger_is_not_an_empty_one(tmp_path: Path, capsys):
    """Exit 2, never 0 and never 1: the question could not be asked."""
    assert main(["--ledger", str(tmp_path / "absent.jsonl"), "verify"]) == 2
    assert "ledger not found" in capsys.readouterr().err


def test_an_unreadable_row_exits_two_not_one(tmp_path: Path, capsys):
    """A row that will not parse is not a refusal to promote."""
    path = tmp_path / "runs.jsonl"
    path.write_bytes(b'{"row_id": "x", "this": "is not a ledger row"}\n')
    assert main(["--ledger", str(path), "show"]) == 2
    assert "unreadable" in capsys.readouterr().err


def test_promotion_renders_the_verdict_and_exits_zero(tmp_path: Path, capsys):
    path = _ledger(tmp_path)
    fam = _protocol(1).hash_without_seed()
    assert main(["--ledger", str(path), "promotion", "--seed-family", fam]) == 0
    out = capsys.readouterr().out
    assert "PROMOTE" in out
    assert "300/300" in out, "a promote must state the coverage it promoted on"


def test_promotion_exits_one_on_a_capped_sample(tmp_path: Path, capsys):
    path = _ledger(tmp_path, n=1, n_total=1000)
    fam = _protocol(1).hash_without_seed()
    assert main(["--ledger", str(path), "promotion", "--seed-family", fam]) == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "1/1000" in out


def test_promotion_of_an_unknown_family_refuses_rather_than_promoting(tmp_path: Path, capsys):
    assert main(["--ledger", str(_ledger(tmp_path)), "promotion", "--seed-family", "nope"]) == 1
    assert "no rows for seed family" in capsys.readouterr().out


def test_families_lists_the_hash_promotion_needs(tmp_path: Path, capsys):
    assert main(["--ledger", str(_ledger(tmp_path)), "families"]) == 0
    out = capsys.readouterr().out
    assert _protocol(1).hash_without_seed()[:16] in out
    assert "[1, 2, 3]" in out


def test_show_renders_every_tristate_with_its_coverage(tmp_path: Path, capsys):
    assert main(["--ledger", str(_ledger(tmp_path, seeds=(1,), n=50)), "show"]) == 0
    out = capsys.readouterr().out
    for gate in REQUIRED_GATES:
        assert gate in out
    assert "[50/300]" in out, "the schema's Coverage rule: a report renders n/n_total"
    assert "not_run" in out, "noul_rate was never computed by these runs and must say so"


def test_show_never_prints_a_bare_pass(tmp_path: Path, capsys):
    """The renderer speaks the schema's vocabulary, not a verdict of its own."""
    main(["--ledger", str(_ledger(tmp_path, seeds=(1,))), "show"])
    out = capsys.readouterr().out
    assert "ran passed=true" in out
    assert "PASS" not in out


def test_no_subcommand_is_refused(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["--ledger", str(_ledger(tmp_path))])
