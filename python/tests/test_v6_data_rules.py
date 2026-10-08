"""``--v6-data-rules`` on ``tools/real_tokenizer_pipeline.py`` (Fable's v6 ruling R1 and R2,
``AUDIT/v6-rulings-2026-10-08/fable-v6-data-design-ruling.md``).

* R2: one flag applies ``DataConfig.with_v6_benchmark_targets()`` and ``with_v6_dedupe_rules()``
  through one owner (``data_rules``), which ``real_ft_run``'s rebuild calls too; both rules reach
  the config fingerprint and the corpus name; absent the flag v5's fingerprint and corpus name
  are byte for byte what v5's manifests recorded. v5's 1,200 MMLU contrast rows move to CSQA
  (2,000 in total) under the v6 config only, and the build refuses a v6 train split holding a
  ``knowledge.multiple_choice`` row.
* R1: the build measures ``code.defect_class``'s share of the written train tokens on every run
  and records it with the tokens by family; under the flag a share below 0.558, or one that could
  not be measured, refuses before any shard is written.

Every test fails on the pre-change code: ``data_rules``, ``name_corpus``, ``train_token_share``,
``Census.tokens_by_family``, ``v6_contrast_spec`` and the ``v6_data_rules`` argument do not exist
there, and its ``apply_contrast`` refuses a v6 split with no MMLU train twin.

Torch-free.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402
from test_contrast_rows import _composite, _csqa  # noqa: E402

from qd_data.config import (  # noqa: E402
    DEDUPE_KEEP_SPLIT_PRIORITY,
    V6_LSH_MIN_AGREEMENT_PERMILLE,
    DataConfig,
)
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID, ContrastSpec  # noqa: E402
from qd_data.general import CSQA_FAMILY, MMLU_FAMILY, rewrite_csqa  # noqa: E402
from qd_data.split import split  # noqa: E402
from qd_train import exclusions  # noqa: E402
from qd_train.contrast import ContrastShortfall, apply_contrast, v6_contrast_spec  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

V5 = DataConfig()
V6 = pipeline.data_rules(V5, v6=True)
#: What every v5-era train manifest on this host records as ``config_fingerprint`` (read from
#: ``~/qd-campaign/*/data/pool/train.json`` on 2026-10-08): the pin the flag's absence keeps.
V5_FINGERPRINT: dict[str, object] = {
    "admitted_by_human": {}, "dedupe_threshold": 0.8,
    "held_out_families": ["code.language_id", "qa.answerability"], "num_perm": 128,
    "seed": 20260919, "shingle_size": 5, "train_fraction": 0.9, "val_fraction": 0.05,
}
#: v5's composite noul request (ledger row c3bd0374's contrast_rows: 1,200 MMLU + 800 CSQA).
V5_CONTRAST = ContrastSpec(per_family={MMLU_FAMILY: 1200, CSQA_FAMILY: 800}, seed=20260919)


# --- R2: one owner for the flag --------------------------------------------------------------


def test_without_the_flag_the_config_is_v5s_own_and_fingerprints_as_v5s_manifests() -> None:
    assert pipeline.data_rules(V5, v6=False) is V5
    assert json.loads(json.dumps(pipeline.data_rules(V5, v6=False).fingerprint())) == (
        V5_FINGERPRINT
    )


def test_the_flag_applies_both_v6_rules_and_both_reach_the_fingerprint() -> None:
    assert V5.with_v6_benchmark_targets().with_v6_dedupe_rules() == V6
    assert V6.benchmark_eval_splits_are_targets
    assert V6.dedupe_keep_rule == DEDUPE_KEEP_SPLIT_PRIORITY
    assert V6.lsh_min_agreement_permille == V6_LSH_MIN_AGREEMENT_PERMILLE
    assert V6.fingerprint() == {
        **V5_FINGERPRINT,
        "benchmark_eval_splits_are_targets": True,
        "dedupe_keep_rule": DEDUPE_KEEP_SPLIT_PRIORITY,
        "lsh_min_agreement_permille": V6_LSH_MIN_AGREEMENT_PERMILLE,
    }


def test_the_corpus_name_moves_only_under_v6_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exclusions, "containment_corpus", lambda identity: {**identity, "q": 1})
    identity = {"rev": "r"}
    # v5: exactly containment_corpus's name, so every v5 attestation still matches.
    assert pipeline.name_corpus(identity, config=V5) == {"rev": "r", "q": 1}
    # The benchmark re-pin alone: the one key tools/containment_scan.py writes for it.
    assert pipeline.name_corpus(identity, config=V5.with_v6_benchmark_targets()) == {
        "rev": "r", "q": 1, "benchmark_eval_splits_are_targets": True,
    }
    assert pipeline.name_corpus(identity, config=V6) == {
        "rev": "r", "q": 1, "benchmark_eval_splits_are_targets": True,
        "dedupe_keep_rule": DEDUPE_KEEP_SPLIT_PRIORITY,
        "lsh_min_agreement_permille": V6_LSH_MIN_AGREEMENT_PERMILLE,
    }


def test_the_flag_reaches_run_only_when_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[dict[str, Any]] = []

    def stop(**kw: Any) -> None:
        seen.append(kw)
        raise SystemExit(0)

    monkeypatch.setattr(pipeline, "run", stop)
    for extra in ([], ["--v6-data-rules"]):
        with pytest.raises(SystemExit):
            pipeline.main(["--out", str(tmp_path), *extra])
    assert [kw["v6_data_rules"] for kw in seen] == [False, True]


# --- R2: the contrast rows move from MMLU to CSQA under v6 only -----------------------------


def test_v5s_contrast_request_becomes_2000_csqa_rows_under_v6_only() -> None:
    assert v6_contrast_spec(V5_CONTRAST, config=V5) is V5_CONTRAST
    assert v6_contrast_spec(V5_CONTRAST, config=V6) == ContrastSpec(
        per_family={CSQA_FAMILY: 2000}, seed=V5_CONTRAST.seed
    )


def _csqa_only_report(config: DataConfig, n: int = 40):
    rows = [
        rewrite_csqa(_csqa(i, "train"), family_id=CSQA_FAMILY, index=i, config=config)
        for i in range(n)
    ]
    return split(dedupe(rows, config=config), config=config)


def test_a_v6_split_with_no_mmlu_twin_derives_every_contrast_row_from_csqa(
    tmp_path: Path, qd_prep: Path
) -> None:
    noul = _composite(tmp_path, {"per_family": {MMLU_FAMILY: 3, CSQA_FAMILY: 2}, "seed": 5})
    report = _csqa_only_report(V6)
    assert not [r for r in report.rows_by_split["train"] if r.family_id == MMLU_FAMILY]
    out, record, status = apply_contrast(
        report, defect_noul=noul, exclusions_applied=True, config=V6
    )
    assert record is not None and record.by_family == {CSQA_FAMILY: 5} and record.seed == 5
    assert isinstance(status, Ran) and status.passed and (status.n, status.n_total) == (5, 5)
    added = out.rows_by_split["train"][len(report.rows_by_split["train"]):]
    assert len(added) == 5 and {r.family_id for r in added} == {DEFECT_FAMILY_ID}
    # v5's config keeps v5's request, which this split cannot meet: refused, never made up.
    with pytest.raises(ContrastShortfall, match=MMLU_FAMILY):
        apply_contrast(
            _csqa_only_report(V5), defect_noul=noul, exclusions_applied=True, config=V5
        )


# --- R1: the defect share, measured always, enforced under the flag -------------------------


def _census(tokens: dict[str, int]) -> pipeline.Census:
    cen = pipeline.Census()
    cen.tokens_by_family = collections.Counter(tokens)
    cen.lengths = [1]
    return cen


@pytest.mark.parametrize(
    ("defect", "other", "passed"),
    [(558, 442, True), (557, 443, False), (900, 100, True), (0, 10, False)],
)
def test_the_share_is_defect_tokens_over_all_train_tokens_against_0_558(
    defect: int, other: int, passed: bool
) -> None:
    tokens, share = pipeline.train_token_share(
        [], _census({DEFECT_FAMILY_ID: defect, CSQA_FAMILY: other}), defect_source_read=True
    )
    assert isinstance(share, Ran) and share.passed is passed
    assert (share.n, share.n_total) == (defect, defect + other)
    assert share.value == round(defect / (defect + other), 6)
    assert isinstance(tokens, Ran) and json.loads(tokens.detail) == {
        "train_tokens_by_family": {CSQA_FAMILY: other, DEFECT_FAMILY_ID: defect}
    }


def test_a_share_that_could_not_be_measured_is_not_run_never_passed() -> None:
    _tokens, share = pipeline.train_token_share(
        [], _census({CSQA_FAMILY: 10}), defect_source_read=False
    )
    assert isinstance(share, NotRun) and "--defect-class" in share.reason
    both = pipeline.train_token_share([], _census({}), defect_source_read=True)
    assert all(isinstance(m, NotRun) for m in both)


def test_the_census_counts_written_tokens_by_family() -> None:
    class _Tok:
        def tokenize(self, text: str, **_: Any) -> Any:
            import numpy as np

            return np.arange(len(text.split()), dtype=np.int64)

        offsets = decode = None

    rows = [
        rewrite_csqa(_csqa(i, "train"), family_id=CSQA_FAMILY, index=i, config=V5)
        for i in range(3)
    ]
    cen = pipeline.census(rows, tok=_Tok(), config=V5)
    assert cen.rows_out == 3
    assert dict(cen.tokens_by_family) == {CSQA_FAMILY: sum(cen.lengths)}


# --- R1/R2 in run(): refused before any shard is written ------------------------------------

PINNED = "0632f693d3b765b726499e7b4bf19c67959b75cb"


class _Wrote(Exception):
    """Raised by the write_shards stand-in: the run got past every check."""


class _FakeTok:
    tok = type("T", (), {"vocab_size": 1, "is_fast": True, "__len__": lambda self: 1})()


def _stubbed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kw: Any
) -> pipeline.Measured:
    """``run`` over 3 pairs of this repository's history at a pinned revision, with the
    tokenizer, census and writer stood in: what is under test is the order of the checks
    between the census and the first write."""
    monkeypatch.setattr(pipeline.RealTokenizer, "load", classmethod(lambda cls, **_: _FakeTok()))
    monkeypatch.setattr(
        pipeline, "census", lambda rows, **_: _census({CSQA_FAMILY: 7}) if rows else _census({})
    )

    def wrote(*_a: Any, **_k: Any) -> None:
        raise _Wrote

    monkeypatch.setattr(pipeline, "write_shards", wrote)
    return pipeline.run(out=tmp_path, max_pairs=3, blank_line_runs=False, rev=PINNED, **kw)


@pytest.mark.usefixtures("qd_prep")
def test_a_v6_build_whose_share_was_not_measured_writes_no_shard_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="defect_token_share") as refused:
        _stubbed_run(monkeypatch, tmp_path, v6_data_rules=True)
    assert "not_run" in str(refused.value)
    assert not (tmp_path / "shards").exists()


@pytest.mark.usefixtures("qd_prep")
def test_a_v6_build_whose_share_failed_writes_no_shard_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = Ran(passed=False, value=0.5, n=1, n_total=2, detail="below the floor")
    monkeypatch.setattr(
        pipeline, "train_token_share", lambda *a, **k: (Ran(passed=True, value=2), failed)
    )
    with pytest.raises(SystemExit, match="defect_token_share"):
        _stubbed_run(monkeypatch, tmp_path, v6_data_rules=True)
    assert not (tmp_path / "shards").exists()


@pytest.mark.usefixtures("qd_prep")
def test_without_the_flag_the_share_is_measured_and_not_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[Any, Any]] = []
    real = pipeline.train_token_share

    def spy(*a: Any, **k: Any) -> Any:
        seen.append(real(*a, **k))
        return seen[-1]

    def next_stage(*_a: Any, **_k: Any) -> None:
        raise _Wrote

    monkeypatch.setattr(pipeline, "train_token_share", spy)
    # The stage after the check: reaching it is getting past it.
    monkeypatch.setattr(pipeline, "_default_decode_survives", next_stage)
    with pytest.raises(_Wrote):
        _stubbed_run(monkeypatch, tmp_path)
    [(tokens, share)] = seen
    assert isinstance(tokens, Ran) and isinstance(share, NotRun)


@pytest.mark.usefixtures("qd_prep")
def test_a_v6_train_split_holding_an_mmlu_row_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fable R2: MMLU at zero. The rows come from a split stood in after dedupe, so the check
    is the run's own and not the loader's refusal of MMLU's evaluation splits."""
    from test_contrast_rows import _mmlu

    from qd_data.general import rewrite_mmlu

    report = split(dedupe([
        rewrite_mmlu(_mmlu(i, "test"), family_id=MMLU_FAMILY, index=i, config=V5)
        for i in range(12)
    ], config=V5), config=V5)
    assert any(r.family_id == MMLU_FAMILY for r in report.rows_by_split["train"])
    monkeypatch.setattr(pipeline, "exclusions_then_contrast", lambda *a, **k: pipeline.PostSplit(
        report, None, (), None, None
    ))
    monkeypatch.setattr(pipeline, "write_shards", lambda *a, **k: pytest.fail("wrote"))
    monkeypatch.setattr(pipeline.RealTokenizer, "load", classmethod(lambda cls, **_: _FakeTok()))
    with pytest.raises(SystemExit, match=r"knowledge\.multiple_choice row\(s\) in the train"):
        pipeline.run(out=tmp_path, max_pairs=3, blank_line_runs=False, rev=PINNED,
                     v6_data_rules=True)
    assert not (tmp_path / "shards").exists()
