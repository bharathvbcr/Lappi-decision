"""perf_step and perf_parity may read stale code only if they say so (Fable 2026-10-22).

``allow_stale_code=True`` is confined: the output banner states it, every row they write is
``stale_code: true`` and ``quick`` (rule 8), and a ledger row carries the ``stale_code`` metric.
Both tools turn the flag on through ``perf_step.enable_stale_code``, the one place that does.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import perf_parity  # noqa: E402
import perf_step  # noqa: E402
import qd_train.shards as shards_mod  # noqa: E402
from qd_data.config import DataConfig  # noqa: E402
from qd_data.fingerprint import code_fingerprint  # noqa: E402
from qd_train.artifacts import ShardContractViolation, ShardHeader  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402


def _stale_header() -> ShardHeader:
    stale = dict(code_fingerprint())
    stale["split.py"] = "0" * 64
    return ShardHeader(
        split="train", data_snapshot_hash="a" * 64, tokenizer_hash="b" * 64,
        remap_hash="c" * 64, vocab_size=128, n_sequences=4, total_tokens=40, max_seq_len=16,
        buckets=(16,), code_fingerprint=stale,
    )


def test_enable_stale_code_prints_the_banner_and_marks_the_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    original = shards_mod.assert_shard_trainable
    monkeypatch.setattr(shards_mod, "assert_shard_trainable", original)  # restored on exit
    header = _stale_header()
    with pytest.raises(ShardContractViolation):  # the production check refuses a stale header
        shards_mod.assert_shard_trainable(
            header, config=DataConfig(), path=tmp_path, repo_root=tmp_path
        )

    fields = perf_step.enable_stale_code()

    err = capsys.readouterr().err
    assert "STALE CODE" in err and "NOT the performance of pinned code" in err
    assert fields["stale_code"] is True and fields["quick"] is True
    assert fields["allow_stale_shards"] is True
    assert "stale_code: true" in fields["quick_reason"]
    checks = shards_mod.assert_shard_trainable(
        header, config=DataConfig(), path=tmp_path, repo_root=tmp_path
    )  # the flag is on now, and the failed check is still reported
    current = checks["shard_code_current"]
    assert isinstance(current, Ran) and not current.passed


def test_the_banner_goes_to_stderr_so_stdout_stays_one_json_object_per_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shards_mod, "assert_shard_trainable", shards_mod.assert_shard_trainable)
    out = io.StringIO()
    with redirect_stdout(out):
        perf_step.enable_stale_code()
    assert out.getvalue() == ""


def test_a_ledger_row_from_a_stale_arm_is_quick_and_carries_the_metric() -> None:
    assert perf_parity.stale_ledger_marks({}) == ([], {})
    fields = {"stale_code": True, "quick_reason": perf_step.STALE_CODE_QUICK_REASON}
    reasons, metrics = perf_parity.stale_ledger_marks(fields)
    assert reasons == [perf_step.STALE_CODE_QUICK_REASON]
    metric = metrics["stale_code"]
    assert isinstance(metric, Ran) and metric.passed and metric.value is True


def test_perf_parity_hands_the_marks_to_the_ledger_row_and_clears_them() -> None:
    src = Path(perf_parity.__file__).read_text(encoding="utf-8")
    assert "from perf_step import _select_batches, enable_stale_code" in src
    assert "*stale_reasons," in src  # into quick_reasons, so the row is quick
    assert "real_ft_run._RUN_METRICS.update(stale_metrics)" in src
    assert "real_ft_run._RUN_METRICS.pop(name, None)" in src  # cleared in a finally
    assert "out.update(stale)" in src  # and the result line carries stale_code too
    assert "allow_stale_code=True" not in src  # no second place turns the flag on


def test_the_summary_of_a_stale_run_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    row = {"shape": "s", "config": "off:0", "status": "ran", "step_s_median": 1.0,
           "pos_per_s": 10.0, "stale_code": True, "quick": True}
    results = tmp_path / "r.jsonl"
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert perf_step.summarize(results) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["stale_code"] is True and summary["quick"] is True
    assert "pinned code" in summary["note"]
