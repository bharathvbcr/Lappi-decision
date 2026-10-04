"""``real_ft_run --split-cache DIR``, end to end on the toy corpus: opt-in, a metric, the same rows.

PLUMBING ON THE TOY CORPUS, NOT THE v5 DATA. A tiny real-backbone tower is trained for three
steps, and its final snapshot is scored for the OOD-only trajectory row three ways:

- without the flag;
- with it, on an empty directory (a miss: the rebuild runs and is stored);
- with it again, with ``ft_split_rows`` replaced by a function that fails if called (a hit).

The three eval rows must agree in every field except:

- the row identity;
- ``metrics.split_cache``, which is absent without the flag and is ``miss`` then ``hit``
  with it.

The suite verdict files must agree byte for byte. The v5-data A/B is in
HANDOFF/v6-startup-2026-10-04.md.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_family_metrics import _tokenizer_json  # noqa: E402
from test_real_ft_rungd_flags import (  # noqa: E402, F401 - corpus is a fixture
    REV,
    _deterministic_restored,
    _only,
    _patch_module,
    _run,
    _tiny_backbone,
    corpus,
)
from test_score_plan_trajectory import (  # noqa: E402
    _IDENTITY,
    _eval_rows,
    _lines,
    _toy_ood,
    _without,
)


def _main(toy, *argv: str, rebuild_refused: bool = False) -> tuple[int, str]:
    """``main`` under the toy harness's patches; with ``rebuild_refused`` the split rebuild
    fails if anything calls it."""
    out = io.StringIO()
    with (
        pytest.MonkeyPatch.context() as mp,
        _deterministic_restored(),
        contextlib.redirect_stdout(out),
    ):
        _patch_module(mp, rft, toy)
        _tiny_backbone(mp)
        if rebuild_refused:
            def refused(**kw):
                raise AssertionError("a split-cache hit must not rebuild the split")
            mp.setattr(rft, "ft_split_rows", refused)
        code = rft.main(list(argv))
    return code, out.getvalue()


def test_plumbing_toy_corpus_split_cache_off_miss_hit_write_the_same_rows(
    corpus, tmp_path, monkeypatch  # noqa: F811 - the fixture imported from test_real_ft_rungd_flags
):
    ck = tmp_path / "ck"
    ft = _only(
        _run(corpus, tmp_path / "ft.jsonl", "--deterministic", "--max-steps", "3",
             "--checkpoint-dir", str(ck), "--retain-tower-every", "3", real=True, short=True),
        "ft",
    )
    _toy_ood(monkeypatch)
    record = tmp_path / "general-record.json"
    record.write_text("{}", encoding="utf-8")
    tokenizer = _tokenizer_json(tmp_path / "tokenizer.json")
    common = (
        "--out", str(corpus.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
        "--real-backbone", str(corpus.snapshot), "--ft-ledger", str(tmp_path / "ft.jsonl"),
        "--tokenizer-json", str(tokenizer), "--ood", "--ood-general-record", str(record),
        "--ft-row-id", ft.row_id, "--score-checkpoint", str(ck / "epoch-seed0-cpu-step3.json"),
    )
    cache = tmp_path / "split-cache"
    runs = {
        "off": (),
        "miss": ("--split-cache", str(cache)),
        "hit": ("--split-cache", str(cache)),
    }
    rows: dict[str, dict] = {}
    lines: dict[str, list[dict]] = {}
    text: dict[str, str] = {}
    for name, extra in runs.items():
        ledger = tmp_path / f"{name}.jsonl"
        verdicts = tmp_path / f"{name}-traj.jsonl"
        code, text[name] = _main(
            corpus, *common, "--ledger", str(ledger), "--suite-verdicts-out", str(verdicts),
            *extra, rebuild_refused=name == "hit",
        )
        assert code == 0, text[name]
        (rows[name],) = _eval_rows(ledger)
        lines[name] = _lines(verdicts, rows[name]["row_id"])

    assert "split_cache" not in rows["off"]["metrics"], "off: the rows are what they were"
    assert "split cache" not in text["off"]
    for name in ("miss", "hit"):
        metric = rows[name]["metrics"]["split_cache"]
        assert metric["state"] == "ran" and metric["value"] == name
        assert metric["passed"] is True
    assert "READ FROM the split cache" in text["hit"] and "NOT rebuilt" in text["hit"]
    assert "STORED" in text["miss"]
    key = rows["hit"]["metrics"]["split_cache"]["detail"].split(";")[0]
    assert key == rows["miss"]["metrics"]["split_cache"]["detail"].split(";")[0]
    assert (cache / (key.removeprefix("key ") + ".rows.pkl")).is_file()

    # Not recipe: the three rows are one protocol.
    assert rows["off"]["recipe"] == rows["miss"]["recipe"] == rows["hit"]["recipe"]
    off = _without(rows["off"], _IDENTITY)
    for name in ("miss", "hit"):
        cached = _without(rows[name], _IDENTITY)
        cached["metrics"] = {k: v for k, v in cached["metrics"].items() if k != "split_cache"}
        assert cached == off, name
        assert lines[name] == lines["off"], name


def test_an_unsafe_split_cache_dir_is_refused_before_anything_is_read(tmp_path):
    bad = tmp_path / "shared"
    bad.mkdir()
    bad.chmod(0o777)
    with pytest.raises(SystemExit, match="group- or world-writable"):
        rft.main(["--out", str(tmp_path / "no-such-build"), "--split-cache", str(bad)])


def test_the_split_cache_flag_is_documented_as_off_by_default_and_not_recipe():
    help_text = io.StringIO()
    with contextlib.redirect_stdout(help_text), pytest.raises(SystemExit):
        rft.main(["--help"])
    flat = " ".join(help_text.getvalue().split())
    assert "--split-cache DIR" in flat
    assert "Off by default. Not recipe" in flat
