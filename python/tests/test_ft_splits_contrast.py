"""The trainer's rebuild derives v5's contrast rows where the build does.

``tools/real_tokenizer_pipeline.py``'s ``run`` appends contrast rows to the train split after
the exclusion list and before the replay draw (``exclusions_then_contrast``). The trainer pairs
every train shard sequence with a row of its rebuild (``tools/real_ft_run.py``'s ``ft_splits``,
then ``pair_labels``), so a rebuild without those rows refuses every v5 shard set that holds
them. ``test_contrast_rows.py``'s drift guard compares the pipeline against a hand-written copy
of the sequence; these tests call ``ft_splits`` itself.

Both fail on the pre-change ``ft_splits``, which applied the exclusion list and derived no
contrast row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
sys.path.insert(0, str(HERE.parents[1] / "python"))

from test_contrast_rows import CONFIG, _composite, _crafted_exclusions, _report  # noqa: E402

from qd_data.defect_class import NOUL_ROUTE_CONTRAST, NOUL_ROUTE_KEY  # noqa: E402
from qd_data.general import CSQA_FAMILY, MMLU_FAMILY  # noqa: E402
from qd_data.split import SplitReport  # noqa: E402
from qd_train import exclusions  # noqa: E402
from qd_train.containment_strip import STRIP_RULE, STRIP_VERSION  # noqa: E402
from qd_train.contrast import contrast_order_key  # noqa: E402
from qd_train.exclusions import ATTESTATION_NAME  # noqa: E402

torch = pytest.importorskip("torch", reason="real_ft_run imports torch at module scope")
import real_ft_run as rft  # noqa: E402
import real_tokenizer_pipeline as pipeline  # noqa: E402


def _ids(rows) -> list[tuple[str, str]]:
    return [(r.row_id, r.identity_key) for r in rows]


def _contrast(rows) -> list[str]:
    return [r.identity_key for r in rows if r.metadata.get(NOUL_ROUTE_KEY) == NOUL_ROUTE_CONTRAST]


def _listed(tmp: Path, keys: list[str], corpus: dict[str, object]) -> Path:
    """``test_contrast_rows._crafted_exclusions``, with the attestation also stating v5's
    template strip, which ``read_exclusions`` requires since the strip landed. Test-only: it
    names the keys this test chose."""
    listed = _crafted_exclusions(tmp, keys, corpus)
    att_path = listed.parent / ATTESTATION_NAME
    att = json.loads(att_path.read_text(encoding="utf-8"))
    att["export"] = {
        "template_strip": {"applied": True, "rule": STRIP_RULE, "version": STRIP_VERSION}
    }
    att_path.write_text(json.dumps(att), encoding="utf-8")
    return listed


@pytest.fixture
def crafted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One split, a noul manifest asking for 4 + 3 contrast rows, and an exclusion list naming
    the MMLU twin the contrast draw would take first. ``ft_splits`` builds the split through
    ``ft_split_report`` and the containment corpus through ``containment_corpus``; both answer
    with these, so the rebuild and the pipeline start from the same rows."""
    report = _report()
    noul = _composite(tmp_path, {"per_family": {MMLU_FAMILY: 4, CSQA_FAMILY: 3}, "seed": 11})
    first_mmlu = sorted(
        (r for r in report.rows_by_split["train"] if r.family_id == MMLU_FAMILY),
        key=lambda r: contrast_order_key(r.identity_key, seed=11),
    )[0]
    corpus: dict[str, object] = {"test": "crafted"}
    listed = _listed(tmp_path, [first_mmlu.identity_key], corpus)
    monkeypatch.setattr(rft, "ft_split_report", lambda **kw: report)
    monkeypatch.setattr(rft, "replay_corpus_identity", lambda **kw: kw)
    monkeypatch.setattr(exclusions, "containment_corpus", lambda identity: corpus)
    return report, noul, listed, corpus, first_mmlu


def test_ft_splits_derives_the_pipelines_contrast_rows(crafted) -> None:
    report, noul, listed, corpus, first_mmlu = crafted
    rebuilt = rft.ft_splits(
        commitpackft=None, max_pairs=3, rev="r", config=CONFIG,
        defect_noul=noul, exclude_identity_keys=listed,
    )
    built = pipeline.exclusions_then_contrast(
        report, exclude_identity_keys=listed, corpus=corpus, defect_noul=noul, config=CONFIG,
    )
    for name in ("train", "val", "heldout"):
        assert _ids(rebuilt[name]) == _ids(built.report.rows_by_split[name]), name
    contrast = _contrast(rebuilt["train"])
    assert built.contrast is not None
    assert len(contrast) == 7 == built.contrast.count
    assert f"contrast:{first_mmlu.identity_key}" not in contrast
    assert first_mmlu.identity_key not in {r.identity_key for r in rebuilt["train"]}


def test_without_the_exclusion_list_neither_side_derives_contrast_rows(crafted) -> None:
    """The scan-source build carries none (``apply_contrast``'s NotRun); the rebuild of it
    must carry none either. Holds before and after the change: a characterization."""
    report, noul, _listed, corpus, _first = crafted
    rebuilt = rft.ft_splits(
        commitpackft=None, max_pairs=3, rev="r", config=CONFIG, defect_noul=noul,
    )
    built = pipeline.exclusions_then_contrast(
        report, exclude_identity_keys=None, corpus=corpus, defect_noul=noul, config=CONFIG,
    )
    assert _contrast(rebuilt["train"]) == [] and built.contrast is None
    for name in ("train", "val", "heldout"):
        assert _ids(rebuilt[name]) == _ids(report.rows_by_split[name]), name


class _Seen(Exception):
    pass


def test_the_replay_draw_sees_the_contrast_rows(
    crafted, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contrast rows join the train split before ``split_off_replay``, as in the build."""
    _split, noul, listed, _corpus, _first = crafted
    handed: list[SplitReport] = []

    def spy(split_report: SplitReport, *, seed: int) -> None:
        handed.append(split_report)
        raise _Seen

    monkeypatch.setattr(pipeline, "split_off_replay", spy)
    with pytest.raises(_Seen):
        rft.ft_splits(
            commitpackft=None, max_pairs=3, rev="r", config=CONFIG,
            general_record=Path("general-record.json"), replay_partition=True,
            defect_noul=noul, exclude_identity_keys=listed,
        )
    (seen,) = handed
    assert len(_contrast(seen.rows_by_split["train"])) == 7
