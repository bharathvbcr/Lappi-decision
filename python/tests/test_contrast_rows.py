"""v5's contrast rows (``qd_train.contrast``) and the train header field that records them.

Route (ii) of the human's item 5, in Fable's recommended form (``campaign/v5-preregistered.DRAFT
.json`` ``data.sources[4].amended``): derived after dedupe, split and the exclusion list, from the
surviving MMLU/CSQA train twins, in blake2b order of identity key, identity
``contrast:<twin identity>``, recorded as ``contrast_rows {count, sha256, seed}`` in the train
shard header, which serialises it only when present so v4 header hashes do not move.

Every test fails on the pre-change code: ``qd_train.contrast`` and
``ShardHeader.contrast_rows`` do not exist there.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.defect_class import (
    CHOICE_SLOT,
    DEFECT_FAMILY_ID,
    NOUL_COMPOSITE_SCHEMA,
    NOUL_FORM_KEY,
    NOUL_ROUTE_CONTRAST,
    NOUL_ROUTE_KEY,
    SPAN_SLOT,
    ContrastSpec,
    DefectCorpusError,
)
from qd_data.general import CSQA_FAMILY, MMLU_FAMILY, rewrite_csqa, rewrite_mmlu
from qd_data.loaders import CsqaRow, MmluRow
from qd_data.sources import PINNED_SPLIT_KEY
from qd_data.split import split
from qd_train.artifacts import ContrastRows, ShardContractViolation, ShardHeader
from qd_train.containment_strip import STRIP_RULE, STRIP_VERSION
from qd_train.contrast import (
    CONTRAST_PATHS,
    ContrastShortfall,
    apply_contrast,
    contrast_order_key,
    contrast_rows_sha256,
    derive_contrast_rows,
)
from qd_train.exclusions import ATTESTATION_NAME, EXCLUSIONS_NAME, apply_exclusions
from qd_train.shards import write_shards
from qd_train.tristate import NotRun, Ran

CONFIG = DataConfig()
REPO = Path(__file__).resolve().parents[2]
#: A train header written before ``contrast_rows`` existed (tracked fixture).
V4_ERA_HEADER = REPO / "crates/qd-train/tests/fixtures/shards-tiny/shards/train/header.json"


def _mmlu(i: int, split_name: str) -> MmluRow:
    return MmluRow(
        subject=f"subject_{i % 3}",
        question=f"Which planet is number {i} from a star called Vega-{i}?",
        choices=(f"p{i}", f"q{i}", f"r{i}", f"s{i}"),
        answer_index=0,
        upstream_split=split_name,
    )


def _csqa(i: int, split_name: str) -> CsqaRow:
    return CsqaRow(
        qid=f"cq{i:04d}",
        question=f"Where would a person keep spare key number {i} safely?",
        concept=f"concept_{i % 4}",
        labels=("A", "B", "C", "D", "E"),
        texts=(f"drawer {i}", f"shelf {i}", f"garage {i}", f"attic {i}", f"car {i}"),
        answer_key="B",
        upstream_split=split_name,
    )


def _report(n_mmlu: int = 12, n_csqa: int = 9):
    rows = [
        rewrite_mmlu(_mmlu(i, "test"), family_id=MMLU_FAMILY, index=i, config=CONFIG)
        for i in range(n_mmlu)
    ]
    rows += [
        rewrite_mmlu(_mmlu(100 + i, "validation"), family_id=MMLU_FAMILY, index=i, config=CONFIG)
        for i in range(3)
    ]
    rows += [
        rewrite_csqa(_csqa(i, "train"), family_id=CSQA_FAMILY, index=i, config=CONFIG)
        for i in range(n_csqa)
    ]
    rows += [
        rewrite_csqa(_csqa(200 + i, "validation"), family_id=CSQA_FAMILY, index=i, config=CONFIG)
        for i in range(3)
    ]
    return split(dedupe(rows, config=CONFIG), config=CONFIG)


SPEC = ContrastSpec(per_family={MMLU_FAMILY: 5, CSQA_FAMILY: 4}, seed=CONFIG.seed)


def test_contrast_rows_come_from_surviving_train_twins_in_keyed_blake2b_order() -> None:
    report = _report()
    out, record = derive_contrast_rows(report, spec=SPEC, config=CONFIG)
    train = report.rows_by_split["train"]
    added = out.rows_by_split["train"][len(train) :]
    assert out.rows_by_split["train"][: len(train)] == train
    for name in ("val", "heldout"):
        assert out.rows_by_split[name] == report.rows_by_split[name]
    assert record.count == len(added) == 9
    assert record.by_family == {MMLU_FAMILY: 5, CSQA_FAMILY: 4}
    twins = {r.identity_key: r for r in train}
    for family, n in SPEC.per_family.items():
        expected = sorted(
            (r for r in train if r.family_id == family),
            key=lambda r: (contrast_order_key(r.identity_key, seed=SPEC.seed), r.identity_key),
        )[:n]
        got = [
            r for r in added if twins[r.identity_key.removeprefix("contrast:")].family_id == family
        ]
        assert [r.identity_key for r in got] == [f"contrast:{t.identity_key}" for t in expected]
    for row in added:
        twin = twins[row.identity_key.removeprefix("contrast:")]
        assert row.family_id == DEFECT_FAMILY_ID
        assert row.repo_key == twin.repo_key and row.licence_id == twin.licence_id
        assert row.metadata[NOUL_ROUTE_KEY] == NOUL_ROUTE_CONTRAST
        assert row.metadata[NOUL_FORM_KEY] == "question"
        assert row.metadata[PINNED_SPLIT_KEY] == "train"
        gold = {g.slot_name: g for g in row.gold}
        assert gold[CHOICE_SLOT].is_noul and gold[SPAN_SLOT].is_noul
        context = row.request.context.decode("utf-8")
        path, text = context.split("\n\n", 1)
        assert path.removeprefix("file: ") in CONTRAST_PATHS
        assert text == twin.request.context.decode("utf-8")
    assert record.sha256 == contrast_rows_sha256(added)
    assert record.header() == ContrastRows(count=9, sha256=record.sha256, seed=SPEC.seed)


def test_a_val_or_held_out_question_never_becomes_a_contrast_row() -> None:
    report = _report()
    out, _ = derive_contrast_rows(
        report,
        spec=ContrastSpec(per_family={MMLU_FAMILY: 12, CSQA_FAMILY: 9}, seed=1),
        config=CONFIG,
    )
    held = {r.identity_key for s in ("val", "heldout") for r in report.rows_by_split[s]}
    contrast = [
        r
        for r in out.rows_by_split["train"]
        if r.metadata.get(NOUL_ROUTE_KEY) == NOUL_ROUTE_CONTRAST
    ]
    assert len(contrast) == 21
    assert not {r.identity_key.removeprefix("contrast:") for r in contrast} & held


def test_a_shortfall_refuses_and_no_family_makes_up_another() -> None:
    with pytest.raises(ContrastShortfall, match=r"commonsense\.multiple_choice: 9 contrast"):
        derive_contrast_rows(
            _report(),
            spec=ContrastSpec(per_family={MMLU_FAMILY: 1, CSQA_FAMILY: 10}, seed=0),
            config=CONFIG,
        )


def test_repeated_identity_and_text_are_passed_over_and_counted() -> None:
    report = _report()
    train = report.rows_by_split["train"]
    twin = next(r for r in train if r.family_id == MMLU_FAMILY)
    dup = dataclasses.replace(twin, row_id=twin.row_id + ":dup")
    report = dataclasses.replace(
        report, rows_by_split={**report.rows_by_split, "train": (*train, dup)}
    )
    _, record = derive_contrast_rows(
        report,
        spec=ContrastSpec(per_family={MMLU_FAMILY: 12}, seed=CONFIG.seed),
        config=CONFIG,
    )
    assert record.count == 12
    assert record.skipped == {f"{MMLU_FAMILY}:repeated_identity": 1}


def test_contrast_rows_are_derived_once() -> None:
    out, _ = derive_contrast_rows(_report(), spec=SPEC, config=CONFIG)
    with pytest.raises(ContrastShortfall, match="already holds"):
        derive_contrast_rows(out, spec=SPEC, config=CONFIG)


def _composite(tmp: Path, contrast: object) -> Path:
    d = tmp / "defect-noul-v3c"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "schema": NOUL_COMPOSITE_SCHEMA,
                "parts": [],
                "contrast": contrast,
            }
        ),
        encoding="utf-8",
    )
    return d


def _crafted_exclusions(tmp: Path, keys: list[str], corpus: dict[str, object]) -> Path:
    """A test-only ``exclusions.txt`` and attestation v2 in the shape ``read_exclusions`` checks,
    including the template strip it requires (``export.template_strip``: applied, this
    ``STRIP_VERSION`` and ``STRIP_RULE``). Not a decontamination: it names the keys this test
    chose."""
    d = tmp / "scan"
    d.mkdir()
    body = "".join(f"{k}\n" for k in sorted(keys, key=str.encode)).encode()
    (d / EXCLUSIONS_NAME).write_bytes(body)
    (d / ATTESTATION_NAME).write_text(json.dumps({
        "version": 2, "tool": "qd-prep containment", "n": 8, "threshold": 0.5,
        "exclusions_sha256": hashlib.sha256(body).hexdigest(), "n_exclusions": len(keys),
        "clean": True, "remaining_hits": {"val": 0}, "excluded_from": "train", "corpus": corpus,
        "export": {"template_strip": {
            "applied": True, "version": STRIP_VERSION, "rule": STRIP_RULE,
        }},
    }), encoding="utf-8")
    return d / EXCLUSIONS_NAME


def test_the_pipeline_and_the_ft_splits_rebuild_derive_the_same_contrast_rows(
    tmp_path: Path,
) -> None:
    """The lead's drift guard: ``real_tokenizer_pipeline.exclusions_then_contrast`` (what
    ``run`` calls) and the ``ft_splits`` sequence (``apply_exclusions`` then ``apply_contrast``,
    with the same arguments) must leave one split with the same rows, contrast rows included,
    and an excluded twin with no contrast row."""
    import real_tokenizer_pipeline as pipeline

    report = _report()
    noul = _composite(tmp_path, {"per_family": {MMLU_FAMILY: 4, CSQA_FAMILY: 3}, "seed": 11})
    first_mmlu = sorted(
        (r for r in report.rows_by_split["train"] if r.family_id == MMLU_FAMILY),
        key=lambda r: contrast_order_key(r.identity_key, seed=11),
    )[0]
    corpus: dict[str, object] = {"test": "crafted"}
    listed = _crafted_exclusions(tmp_path, [first_mmlu.identity_key], corpus)

    post = pipeline.exclusions_then_contrast(
        report, exclude_identity_keys=listed, corpus=corpus, defect_noul=noul, config=CONFIG,
    )
    ft, _exclusions, _excluded = apply_exclusions(report, listed, corpus=corpus)
    ft, ft_record, _status = apply_contrast(
        ft, defect_noul=noul, exclusions_applied=listed is not None, config=CONFIG,
    )

    def ids(rows) -> list[tuple[str, str]]:
        return [(r.row_id, r.identity_key) for r in rows]

    for name in ("train", "val", "heldout"):
        assert ids(post.report.rows_by_split[name]) == ids(ft.rows_by_split[name])
    assert post.contrast == ft_record and ft_record is not None and ft_record.count == 7
    contrast_twins = {
        r.identity_key.removeprefix("contrast:") for r in post.report.rows_by_split["train"]
        if r.metadata.get(NOUL_ROUTE_KEY) == NOUL_ROUTE_CONTRAST
    }
    assert first_mmlu.identity_key not in contrast_twins
    assert [r.row_id for r in post.excluded_rows] == [first_mmlu.row_id]


def test_apply_contrast_runs_only_on_the_rebuild_that_applied_the_exclusion_list(
    tmp_path: Path,
) -> None:
    report = _report()
    noul = _composite(tmp_path, {"per_family": {MMLU_FAMILY: 3, CSQA_FAMILY: 2}, "seed": 7})
    same, record, status = apply_contrast(
        report, defect_noul=None, exclusions_applied=True, config=CONFIG
    )
    assert same is report and record is None and status is None
    same, record, status = apply_contrast(
        report, defect_noul=noul, exclusions_applied=False, config=CONFIG
    )
    assert same is report and record is None
    assert isinstance(status, NotRun) and "--exclude-identity-keys" in status.reason
    out, record, status = apply_contrast(
        report, defect_noul=noul, exclusions_applied=True, config=CONFIG
    )
    assert record is not None and record.count == 5 and record.seed == 7
    assert isinstance(status, Ran) and status.passed and (status.n, status.n_total) == (5, 5)
    assert len(out.rows_by_split["train"]) == len(report.rows_by_split["train"]) + 5


@pytest.mark.parametrize(
    "contrast",
    [
        {"per_family": {"intent.in_scope": 3}, "seed": 1},
        {"per_family": {MMLU_FAMILY: 0}, "seed": 1},
        {"per_family": {MMLU_FAMILY: 3}, "seed": -1},
        {"per_family": {}, "seed": 1},
        [1, 2],
    ],
)
def test_a_malformed_contrast_request_is_refused_not_read_as_none(
    tmp_path: Path, contrast: object
) -> None:
    noul = _composite(tmp_path, contrast)
    with pytest.raises(DefectCorpusError, match="contrast"):
        apply_contrast(_report(), defect_noul=noul, exclusions_applied=True, config=CONFIG)


# -- the header ----------------------------------------------------------------------------


def _header(**over: object) -> ShardHeader:
    base: dict[str, object] = {
        "split": "train",
        "data_snapshot_hash": "d" * 64,
        "tokenizer_hash": "t" * 64,
        "remap_hash": "r" * 64,
        "vocab_size": 100,
        "n_sequences": 2,
        "total_tokens": 10,
        "max_seq_len": 8,
        "buckets": (8,),
    }
    base.update(over)
    return ShardHeader(**base)  # type: ignore[arg-type]


def test_a_header_written_before_the_field_still_verifies_and_serialises_as_it_did() -> None:
    raw = json.loads(V4_ERA_HEADER.read_text(encoding="utf-8"))
    assert "contrast_rows" not in raw
    header = ShardHeader.from_json(raw)  # recomputes shard_hash and refuses a mismatch
    assert header.contrast_rows is None
    out = header.to_json()
    assert "contrast_rows" not in out and out["shard_hash"] == raw["shard_hash"]


def test_a_train_header_names_its_contrast_rows_and_hashes_apart() -> None:
    record = ContrastRows(count=2000, sha256="a" * 64, seed=20260919)
    plain, named = _header(), _header(contrast_rows=record)
    assert named.shard_hash() != plain.shard_hash()
    assert named.to_json()["contrast_rows"] == {"count": 2000, "sha256": "a" * 64, "seed": 20260919}
    assert ShardHeader.from_json(named.to_json()).contrast_rows == record
    tampered = named.to_json()
    tampered["contrast_rows"]["count"] = 1999
    with pytest.raises(ShardContractViolation, match="modified after it was written"):
        ShardHeader.from_json(tampered)


@pytest.mark.parametrize(
    "bad",
    [
        {"count": 0, "sha256": "a" * 64, "seed": 1},
        {"count": 3, "sha256": "A" * 64, "seed": 1},
        {"count": 3, "sha256": "a" * 64, "seed": -1},
        {"count": True, "sha256": "a" * 64, "seed": 1},
        {"count": 3, "sha256": "a" * 64},
    ],
)
def test_a_malformed_contrast_record_is_refused(bad: dict[str, object]) -> None:
    raw = _header().to_json()
    raw.pop("shard_hash")
    raw["contrast_rows"] = bad
    with pytest.raises(ShardContractViolation, match="contrast_rows"):
        ShardHeader.from_json(raw)


def test_only_a_train_set_carries_contrast_rows() -> None:
    with pytest.raises(ShardContractViolation, match="contrast rows are train rows only"):
        _header(split="val", contrast_rows=ContrastRows(count=1, sha256="a" * 64, seed=0))


def test_the_writer_refuses_rows_and_a_header_that_disagree(tmp_path: Path) -> None:
    out, record = derive_contrast_rows(_report(), spec=SPEC, config=CONFIG)
    rows = list(out.rows_by_split["train"])
    for kwargs in (
        {"contrast_rows": None},
        {"contrast_rows": dataclasses.replace(record.header(), count=8)},
        {"contrast_rows": record.header(), "replay": True},
    ):
        with pytest.raises(ShardContractViolation, match="contrast row"):
            write_shards(
                tmp_path / "absent-manifest.json",
                rows,
                out_dir=tmp_path / "shards",
                remap=None,
                tokenize=list,
                config=CONFIG,
                repo_root=REPO,  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )
