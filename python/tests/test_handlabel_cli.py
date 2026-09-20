"""The labelling CLI: it has to survive being driven by a tired human at 11pm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_label.__main__ import load_pool, main, render_diff


def _write_pool(tmp_path: Path, n: int = 24) -> Path:
    langs = ["rust", "go", "python", "typescript", "swift"]
    items = [
        {
            "item_id": f"d{i:03d}",
            "repo": f"org/repo{i % 4}",
            "path": f"src/m{i}.rs",
            "language": langs[i % 5],
            "diff": f"@@ -1,2 +1,1 @@\n-    compute({i})\n+    Ok(0)\n",
        }
        for i in range(n)
    ]
    p = tmp_path / "pool.json"
    p.write_text(json.dumps({"items": items}))
    return p


def test_pool_loads_from_json_and_jsonl(tmp_path: Path):
    p = _write_pool(tmp_path, 3)
    assert len(load_pool(p)) == 3

    rows = json.loads(p.read_text())["items"]
    jl = tmp_path / "pool.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(load_pool(jl)) == 3


def test_a_missing_pool_exits_nonzero_without_a_traceback(tmp_path: Path):
    code = main(
        ["--pool", str(tmp_path / "nope.json"), "--store", str(tmp_path / "s.jsonl"), "stats"]
    )
    assert code == 2


def test_a_truncated_diff_says_how_much_is_hidden():
    """A silently truncated diff is one the labeller judges without knowing."""
    out = render_diff("\n".join(f"+ line {i}" for i in range(200)), max_lines=20)
    assert "180 more lines hidden of 200 total" in out


def test_label_records_decisions_and_stats_reads_them(tmp_path: Path, monkeypatch, capsys):
    pool, store = _write_pool(tmp_path, 6), tmp_path / "labels.jsonl"
    keys = iter(["s", "l", "c", "k"])
    monkeypatch.setattr("builtins.input", lambda _="": next(keys))

    assert main(["--pool", str(pool), "--store", str(store), "label", "--limit", "4"]) == 0
    recorded = [json.loads(x) for x in store.read_text().splitlines() if x.strip()]
    assert [r["label"] for r in recorded] == ["stub", "logic", "cosmetic", "clean"]

    capsys.readouterr()
    assert main(["--pool", str(pool), "--store", str(store), "stats"]) == 0
    assert "4/6 items labelled" in capsys.readouterr().out


def test_an_unrecognised_key_reprompts_rather_than_recording(tmp_path: Path, monkeypatch, capsys):
    pool, store = _write_pool(tmp_path, 2), tmp_path / "labels.jsonl"
    keys = iter(["x", "zzz", "s", "q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(keys))

    assert main(["--pool", str(pool), "--store", str(store), "label"]) == 0
    recorded = [json.loads(x) for x in store.read_text().splitlines() if x.strip()]
    assert len(recorded) == 1 and recorded[0]["label"] == "stub"
    assert "unrecognised" in capsys.readouterr().out


def test_quitting_keeps_everything_recorded_so_far(tmp_path: Path, monkeypatch):
    pool, store = _write_pool(tmp_path, 10), tmp_path / "labels.jsonl"
    keys = iter(["s", "s", "q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(keys))

    assert main(["--pool", str(pool), "--store", str(store), "label"]) == 0
    assert len(store.read_text().strip().splitlines()) == 2


def test_ctrl_c_does_not_lose_work(tmp_path: Path, monkeypatch):
    pool, store = _write_pool(tmp_path, 10), tmp_path / "labels.jsonl"
    calls = {"n": 0}

    def fake_input(_: str = "") -> str:
        calls["n"] += 1
        if calls["n"] <= 2:
            return "s"
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fake_input)
    assert main(["--pool", str(pool), "--store", str(store), "label"]) == 0
    assert len(store.read_text().strip().splitlines()) == 2


def test_unsure_is_reprompted_until_a_reason_is_given(tmp_path: Path, monkeypatch):
    pool, store = _write_pool(tmp_path, 2), tmp_path / "labels.jsonl"
    keys = iter(["u", "", "   ", "the caller ignores the result", "q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(keys))

    assert main(["--pool", str(pool), "--store", str(store), "label"]) == 0
    rec = json.loads(store.read_text().strip().splitlines()[0])
    assert rec["label"] == "unsure"
    assert rec["note"] == "the caller ignores the result"


def test_ceiling_refuses_to_report_from_too_few_pairs(tmp_path: Path, monkeypatch, capsys):
    pool, store = _write_pool(tmp_path, 6), tmp_path / "labels.jsonl"
    keys = iter(["s"] * 6)
    monkeypatch.setattr("builtins.input", lambda _="": next(keys))
    main(["--pool", str(pool), "--store", str(store), "label"])

    capsys.readouterr()
    assert main(["--pool", str(pool), "--store", str(store), "ceiling"]) == 1
    assert "not enough to estimate a ceiling" in capsys.readouterr().out


def test_export_writes_a_held_out_manifest(tmp_path: Path, monkeypatch):
    pool, store = _write_pool(tmp_path, 4), tmp_path / "labels.jsonl"
    keys = iter(["s", "k", "l", "c"])
    monkeypatch.setattr("builtins.input", lambda _="": next(keys))
    main(["--pool", str(pool), "--store", str(store), "label"])

    out = tmp_path / "heldout.json"
    assert main(["--pool", str(pool), "--store", str(store), "export", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["held_out"] is True and doc["n_labelled"] == 4 and doc["n_usable"] == 4
