"""``tools/real_tokenizer_pipeline.py``: a corpus with no row from this repository's history,
and a ledger row whose corpus revision is a full sha.

Found by the committed-tree smoke of 2026-09-29: with no ``--commitpackft`` the pipeline drew
its ``bigcode/commitpackft`` rows from this repository's git history, and its
``rajpurkar/squad_v2`` stand-in from this repository's Markdown, at ``--rev`` -- default
``HEAD``. At HEAD that day it held 37k-token rows, changed with every commit, and hit the
400-pair cap, so the data snapshot read NotRun. Phase 3 trains on ``code.defect_class`` only,
which needs a way to leave both history sources out.

Torch-free, so ``make gates`` runs it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

#: The revision real_ft_run.py defaults to, and the 2026-09-22 GH200 rows were built at.
PINNED = "0632f693d3b765b726499e7b4bf19c67959b75cb"


class _Reached(Exception):
    """Raised by a stand-in for ``run``: the argv checks all let the call through."""


def _no_history_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(**_kw: object) -> Any:
        raise AssertionError("read this repository's history under --no-repo-history")

    monkeypatch.setattr(pipeline, "code_rows", refuse)
    monkeypatch.setattr(pipeline, "span_rows", refuse)


# --- base_sources: which sources a build reads ----------------------------------------


def test_max_pairs_zero_is_not_a_way_to_leave_history_out() -> None:
    """Checked before adding the flag: ``--max-pairs 0`` reads no history row but marks both
    sources capped, and any capped source makes the mixture -- and the snapshot -- NotRun."""
    base = pipeline.base_sources(
        repo_history=True, commitpackft=None, max_pairs=0, blank_line_runs=False, rev=PINNED
    )
    assert (base.n_commits, base.n_spans) == (0, 0)
    assert set(base.capped) == {"bigcode/commitpackft", "rajpurkar/squad_v2"}


def test_without_history_no_source_is_read_and_none_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent, not empty: an empty source still names its families to build_mixture."""
    _no_history_read(monkeypatch)
    base = pipeline.base_sources(
        repo_history=False, commitpackft=None, max_pairs=400, blank_line_runs=False, rev=PINNED
    )
    assert base.raw == {}
    assert (base.capped, base.from_history, base.pool_total) == ((), (), None)


def test_without_history_the_download_still_supplies_code_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[dict[str, object]] = []

    def code_rows(**kw: object) -> tuple[list[object], bool, int]:
        seen.append(kw)
        return [], False, 0

    monkeypatch.setattr(pipeline, "code_rows", code_rows)
    monkeypatch.setattr(pipeline, "span_rows", lambda **kw: pytest.fail("read span prose"))
    base = pipeline.base_sources(
        repo_history=False, commitpackft=tmp_path, max_pairs=7, blank_line_runs=False,
        rev=PINNED,
    )
    assert list(base.raw) == ["bigcode/commitpackft"]
    assert base.from_history == ()
    assert seen == [{"commitpackft": tmp_path, "max_pairs": 7, "rev": PINNED}]


def test_with_history_both_history_sources_are_named() -> None:
    base = pipeline.base_sources(
        repo_history=True, commitpackft=None, max_pairs=4, blank_line_runs=False, rev=PINNED
    )
    assert set(base.raw) == set(pipeline.REPO_HISTORY_SOURCES)
    assert base.from_history == pipeline.REPO_HISTORY_SOURCES
    assert base.n_commits == 4 and base.n_spans == 4


def test_the_quick_reason_of_a_history_free_build_claims_no_history() -> None:
    reason = pipeline.quick_reason_for(commitpackft_rows=None, repo_history=False)
    assert "this repository's own Markdown" not in reason
    assert "one seed" in reason and "--no-repo-history" in reason


# --- main: argv refusals, before any work ---------------------------------------------


def _main_reaching_run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, Any]:
    got: dict[str, Any] = {}

    def run(**kw: Any) -> None:
        got.update(kw)
        raise _Reached

    monkeypatch.setattr(pipeline, "run", run)
    with pytest.raises(_Reached):
        pipeline.main(argv)
    return got


def test_no_repo_history_reaches_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    got = _main_reaching_run(
        monkeypatch, ["--out", str(tmp_path), "--no-repo-history", "--defect-class", "c"]
    )
    assert got["repo_history"] is False


def test_history_is_read_unless_the_flag_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    got = _main_reaching_run(monkeypatch, ["--out", str(tmp_path)])
    assert got["repo_history"] is True
    assert got["max_pairs"] == pipeline.DEFAULT_MAX_PAIRS


@pytest.mark.parametrize(
    ("extra", "match"),
    [(["--max-pairs", "400"], "bounds nothing"), (["--blank-line-runs"], "determine nothing")],
)
def test_a_value_that_would_bound_nothing_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str], match: str
) -> None:
    monkeypatch.setattr(pipeline, "run", lambda **kw: pytest.fail("ran"))
    with pytest.raises(SystemExit, match=match):
        pipeline.main(["--out", str(tmp_path), "--no-repo-history", "--defect-class", "c",
                       *extra])


def test_max_pairs_still_samples_the_download_without_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    got = _main_reaching_run(monkeypatch, [
        "--out", str(tmp_path), "--no-repo-history", "--commitpackft", "d", "--max-pairs", "9",
    ])
    assert got["max_pairs"] == 9


def test_no_source_at_all_is_refused_before_the_tokenizer_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        pipeline.RealTokenizer, "load", classmethod(lambda cls, **kw: pytest.fail("loaded"))
    )
    with pytest.raises(SystemExit, match="reads no source at all"):
        pipeline.run(out=tmp_path, max_pairs=1, blank_line_runs=False, rev=PINNED,
                     repo_history=False)


@pytest.mark.parametrize("rev", ["HEAD", "main", "0632f69"])
def test_a_ledger_row_needs_a_full_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rev: str
) -> None:
    """Refused before any work: the rebuild 3e577d0b was built at HEAD and superseded."""
    monkeypatch.setattr(pipeline, "run", lambda **kw: pytest.fail("ran"))
    with pytest.raises(SystemExit, match="not a full 40-character commit sha"):
        pipeline.main(["--out", str(tmp_path), "--rev", rev, "--ledger", str(tmp_path / "l")])


def test_a_ledger_row_at_a_full_sha_gets_past_the_revision_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ref = tmp_path / "refs-main"
    ref.write_text("b" * 40, encoding="utf-8")  # the HF cache is not on every host
    monkeypatch.setattr(pipeline, "MODEL_REF", ref)
    got = _main_reaching_run(
        monkeypatch,
        ["--out", str(tmp_path), "--rev", PINNED, "--ledger", str(tmp_path / "l.jsonl")],
    )
    assert got["rev"] == PINNED


def test_without_a_ledger_a_name_is_still_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A build that records nothing is a local look; it may still read HEAD."""
    got = _main_reaching_run(monkeypatch, ["--out", str(tmp_path), "--rev", "HEAD"])
    assert got["rev"] == "HEAD"
