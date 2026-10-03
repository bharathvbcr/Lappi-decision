"""The general families end to end through build_mixture / dedupe / split.

``qd_data.general`` owns the rewriters; ``mixture._dispatch`` is the one seam every raw
row flows through, so this is where MMLU, CommonsenseQA and the two-stage CLINC families
either build or refuse. Merged from the general-families lane (Part B) and the
data-fixes lane (Part A: refusal counts and the dispatch's expected-type string).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from data_fixtures import INTENT_VOCABULARY, small_corpus

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.general import (
    CLINC_DOMAIN_FAMILY,
    CLINC_WITHIN_DOMAIN_FAMILY,
    CSQA_FAMILY,
    MMLU_FAMILY,
    ClincDomainMap,
    load_clinc_domains,
)
from qd_data.loaders import ClincRow, CsqaRow, MmluRow, parse_clinc, read_jsonl
from qd_data.mixture import RowRefused, _dispatch, build_mixture
from qd_data.split import split


def _mmlu(i: int, upstream: str) -> MmluRow:
    return MmluRow(subject=f"s{i % 5}", question=f"What follows {i} in {upstream}?",
                   choices=(str(i + 1), str(i + 2), str(i + 3), str(i + 4)), answer_index=0,
                   upstream_split=upstream)


def _csqa(i: int) -> CsqaRow:
    return CsqaRow(qid=f"c{i}", question=f"Where is thing {i}?", concept=f"k{i % 4}",
                   labels=("A", "B", "C"), texts=(f"a{i}", f"b{i}", f"c{i}"), answer_key="C",
                   upstream_split="train")


def test_mmlu_and_csqa_build_and_mmlu_lands_where_its_upstream_split_pins_it() -> None:
    config = DataConfig()
    corpus = {
        "cais/mmlu": [_mmlu(i, s) for s in ("test", "dev", "validation") for i in range(20)],
        "tau/commonsense_qa": [_csqa(i) for i in range(20)],
    }
    m = build_mixture(corpus, config=config)
    assert type(m.status).__name__ == "Ran" and m.status.passed, m.status
    assert m.family_coverage[MMLU_FAMILY].value == 60
    assert m.family_coverage[CSQA_FAMILY].value == 20
    report = split(dedupe(m.rows, config=config), config=config)
    where = {a.row_id: a.split for a in report.assignments}
    for row in m.rows:
        if row.family_id == MMLU_FAMILY:
            expect = "val" if row.metadata["upstream_split"] == "validation" else "train"
            assert where[row.row_id] == expect, row.row_id
    assert report.repo_disjoint.passed is True  # type: ignore[union-attr]


def test_two_stage_clinc_builds_with_a_map_and_fails_closed_without() -> None:
    vocab = sorted(set(INTENT_VOCABULARY))
    q = len(vocab) // 4
    dm = ClincDomainMap(domains={f"d{k}": tuple(vocab[k * q:(k + 1) * q]) for k in range(4)})
    rows = small_corpus(24)["clinc/clinc_oos"]
    ok = build_mixture({"clinc/clinc_oos": rows}, config=DataConfig(), clinc_domain_map=dm)
    assert ok.status.passed is True, ok.status
    assert any(isinstance(r, ClincRow) and r.is_oos for r in rows)
    assert ok.family_coverage[CLINC_DOMAIN_FAMILY].value == len(rows)
    assert ok.family_coverage[CLINC_WITHIN_DOMAIN_FAMILY].value == len(rows)
    # Without a map the two-stage families are not requested by default...
    default = build_mixture({"clinc/clinc_oos": rows}, config=DataConfig())
    assert default.status.passed is True, default.status
    assert CLINC_DOMAIN_FAMILY not in default.family_coverage
    # ...and asking for them explicitly without one fails closed.
    bad = build_mixture({"clinc/clinc_oos": rows}, config=DataConfig(),
                        families=[CLINC_DOMAIN_FAMILY, "intent.in_scope"])
    assert type(bad.status).__name__ == "NotRun"
    assert "no_clinc_domain_map" in bad.status.reason


_CACHE = Path.home() / ".cache" / "qd-decision" / "general"
_CLINC = _CACHE / "clinc__clinc_oos" / "155b9c710419136e17307b80d0a13e68cd46b4ec"
_DOMAINS = (
    _CACHE / "clinc__oos-eval" / "828f8093932c8fe6ca7936c3d2e52903b1c523de" / "domains.json"
)


def test_real_clinc_validation_builds_both_stages_with_the_real_map() -> None:
    if not _DOMAINS.exists():
        pytest.skip(f"not run: the approved download is not on this host ({_CACHE})")
    names = json.loads((_CLINC / "intent_names.json").read_text(encoding="utf-8"))
    raws, capped = read_jsonl(_CLINC / "validation.jsonl", limit=10_000, max_row_bytes=4096)
    assert not capped
    rows = [parse_clinc(r, index=i, label_names=names) for i, r in enumerate(raws)]
    m = build_mixture({"clinc/clinc_oos": rows}, config=DataConfig(),
                      clinc_domain_map=load_clinc_domains(_DOMAINS),
                      families=[CLINC_DOMAIN_FAMILY, CLINC_WITHIN_DOMAIN_FAMILY])
    assert m.status.passed is True, m.status
    assert m.family_coverage[CLINC_DOMAIN_FAMILY].value == 3100
    assert m.family_coverage[CLINC_WITHIN_DOMAIN_FAMILY].value == 3100


def test_mmlu_and_csqa_refusals_are_counted_by_reason() -> None:
    corpus = small_corpus(24)
    corpus["cais/mmlu"] = [_mmlu(i, "test") for i in range(20)] + [
        MmluRow(subject="s", question="dup?", choices=("a", "A ", "b", "c"), answer_index=0,
                upstream_split="test")
    ]
    corpus["tau/commonsense_qa"] = [_csqa(i) for i in range(20)] + [
        CsqaRow(qid="t1", question="unlabelled?", concept="k", labels=("A", "B"),
                texts=("x", "y"), answer_key="", upstream_split="train")
    ]
    m = build_mixture(corpus, config=DataConfig())
    assert m.status.passed is True, m.status
    assert m.family_coverage[MMLU_FAMILY].value == 20
    assert m.family_coverage[CSQA_FAMILY].value == 20
    assert m.refusals["cais/mmlu"] == {"duplicate_options": 1}
    assert m.refusals["tau/commonsense_qa"] == {"unlabelled": 1}


def test_unknown_raw_row_type_names_every_row_type_dispatch_accepts() -> None:
    with pytest.raises(RowRefused) as refused:
        _dispatch(
            object(),  # type: ignore[arg-type]
            family_id=MMLU_FAMILY, index=0, config=DataConfig(),
            intent_vocabulary=(), messages=(),
        )
    assert refused.value.reason_code == "unknown_raw_row_type"
    assert refused.value.expected == (
        "CommitPackFtRow | ClincRow | SquadRow | DefectRow | MmluRow | CsqaRow | "
        "TypedDecisionRow"
    )
