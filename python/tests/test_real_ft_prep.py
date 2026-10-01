"""``tools/real_ft_run.py``'s prelude does only the work its mode reads, and says what it skipped.

Measured on the Mac, 2026-10-01, on the phase-4 v3 inputs J7g ran with: the prelude before a
``--score-checkpoint`` touches the GPU was 143 s, and ~38 s of it relabelled, inventoried and
batched the TRAIN split -- for a run that trains nothing and writes no row that reads any of
it. Those steps now run only where something rests on them: training always; scoring when no
tokenizer.json supplies the letter ids, or when a val or OOD row offers a letter no val row
has as its gold (the train golds were what confirmed that letter's id). Skipped, each says
NOT RUN with its reason -- never the line a run that did it prints.

``_labels`` also renders each row once: ``training_texts`` rendered it a second time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import real_ft_run as rft  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_train import shards  # noqa: E402
from qd_train.artifacts import SLOT_CHOICE, SLOT_SPAN  # noqa: E402
from qd_train.needle import build_suite, needle_defect_row  # noqa: E402
from qd_train.ood import build_ood_suite, ood_defect_row  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

VOCAB = {"A": 32, "B": 33, "C": 34, "Z": 57}


def _label(row_id: str, gold: str, letters: tuple[str, ...], kind: int = SLOT_CHOICE):
    return rft.Label(row_id=row_id, family_id="code.defect_class", slot_name="defect_class",
                     slot_kind=kind, gold_letter=gold, letters=letters)


class _Stop(Exception):
    pass


class _Reader:
    def __init__(self) -> None:
        self.header = SimpleNamespace(
            data_snapshot_hash="d" * 64, buckets=(135, 1105), remap_hash="r" * 64,
            shard_hash=lambda: "t" * 64,
        )
        self.span_check = NotRun(reason="stub")
        self.checks = {"shard_remap_matches_header": NotRun(reason="stub")}

    def padding_waste(self):
        return Ran(passed=True, value=0.0, detail="stub padding")

    def __len__(self) -> int:
        return 3


def _run(monkeypatch, tmp_path, capsys, *, score: bool, val_labels, ood_labels=(),
         tokenizer: bool = True):
    """main() with every expensive step stubbed and counted, stopped where it would decode
    (scoring) or probe a device (training)."""
    calls: list[str] = []

    def counted(name, value):
        def stub(*a, **k):
            calls.append(name)
            return value() if callable(value) else value
        return stub

    reader = _Reader()
    inventory = {
        "coverage": {"n_total": 3, "n": 3}, "excluded_rows": 0, "tokens": 9, "vocab_size": 9,
        "per_kind": {}, "letter_rows_whose_gold_is_noul": 0, "letter_rows": 3,
        "contradictions": {"rows_sharing_a_prefix": 0, "sequences": 3, "contradicting_rows": 0,
                           "contradicting_groups": 0, "per_kind": {}},
    }
    monkeypatch.setattr(rft, "resolve_rev", lambda repo, rev: rev)
    monkeypatch.setattr(rft, "ShardReader", lambda *a, **k: reader)
    monkeypatch.setattr(rft, "check_defect_source", lambda *a, **k: None)
    monkeypatch.setattr(rft, "corpus_facts", lambda *a, **k: rft.CorpusFacts(None, {}))
    monkeypatch.setattr(rft, "general_record_datasets", lambda record: None)
    monkeypatch.setattr(rft, "ft_split_rows", counted("ft_split_rows", (["tr"], ["va"])))
    monkeypatch.setattr(rft, "relabel_train", counted(
        "relabel_train", lambda: rft.TrainRelabel([], inventory, dict(VOCAB))
    ))
    monkeypatch.setattr(rft, "vocab_letter_ids", counted("vocab_letter_ids", lambda: dict(VOCAB)))
    val = SimpleNamespace(reader=reader, labels=list(val_labels), plan=[], labels_for={},
                          letter_id=dict(VOCAB))
    monkeypatch.setattr(rft, "open_val_set", counted("open_val_set", val))
    monkeypatch.setattr(rft, "prepare_second_pass", counted(
        "prepare_second_pass", rft.SecondPass([], {}, {}, not_run="stub")
    ))
    monkeypatch.setattr(rft, "prepare_needle", counted(
        "prepare_needle", rft.NeedleSuite([], [], {}, [], not_run="stub")
    ))
    ood_val = SimpleNamespace(labels=list(ood_labels)) if ood_labels else None
    monkeypatch.setattr(rft, "prepare_ood", counted(
        "prepare_ood", rft.OodSuite([], ood_val, None, not_run=None if ood_labels else "stub")
    ))
    monkeypatch.setattr(rft, "_batch_inventory", counted("_batch_inventory", {
        "batches": 1, "batches_with_a_span_row": 0, "batches_span_only": 0,
        "letter_channel_live_chunks": 1, "letter_channel_total_chunks": 1,
        "rows": [{"bucket": 0, "rows": 1, "width": 135, "span_rows": 0}],
    }))

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(rft, "_score_checkpoint", stop)
    monkeypatch.setattr(rft, "_probe_one", stop)
    backbone = tmp_path / "snapshot"
    backbone.mkdir(exist_ok=True)
    if tokenizer:
        (backbone / "tokenizer.json").write_text("{}", encoding="utf-8")
    argv = ["--out", str(tmp_path), "--rev", "0" * 40, "--no-repo-history", "--seeds", "0",
            "--real-backbone", str(backbone), "--score-val", "--devices", "cpu"]
    argv += (
        ["--score-checkpoint", str(tmp_path / "epoch-seed0-cpu.json"),
         "--ft-ledger", str(tmp_path / "ft.jsonl"), "--ft-row-id", "abcdefgh"]
        if score else ["--epoch", "--no-memorise"]
    )
    with pytest.raises(_Stop):
        rft.main(argv)
    return calls, capsys.readouterr().out


CONFIRMED = [_label("v1", "A", ("A", "B", "Z")), _label("v2", "B", ("A", "B", "Z")),
             _label("v3", "Z", ("Z",), kind=SLOT_SPAN)]


def test_scoring_skips_the_train_relabel_and_the_epoch_plan_and_says_so(
    monkeypatch, tmp_path, capsys
):
    calls, out = _run(monkeypatch, tmp_path, capsys, score=True, val_labels=CONFIRMED)
    assert "relabel_train" not in calls and "_batch_inventory" not in calls
    assert calls.count("ft_split_rows") == 1, "val's labels still come from the rebuild"
    assert "train relabel, per-kind inventory and contradictions: NOT RUN -- " in out
    assert "confirmed against the val golds" in out
    assert "epoch at batch_tokens=1105: NOT RUN -- --score-checkpoint trains nothing" in out
    assert "  rows in -> out:" not in out, "a skipped inventory must not print as one"
    assert "padding waste: stub padding" in out and "remap beside the shards:" in out


def test_scoring_relabels_train_when_a_val_offered_letter_is_no_val_gold(
    monkeypatch, tmp_path, capsys
):
    offers_c = [*CONFIRMED, _label("v4", "A", ("A", "C", "Z"))]
    calls, out = _run(monkeypatch, tmp_path, capsys, score=True, val_labels=offers_c)
    assert calls.count("relabel_train") == 1
    assert "train relabel: run after all -- val or OOD rows offer letter(s) ['C']" in out
    assert "NOT RUN -- --score-checkpoint trains nothing and no row" not in out
    assert "  rows in -> out: 3 -> 3" in out


def test_an_ood_offered_letter_no_val_gold_confirms_relabels_train_too(
    monkeypatch, tmp_path, capsys
):
    calls, out = _run(monkeypatch, tmp_path, capsys, score=True, val_labels=CONFIRMED,
                      ood_labels=[_label("ood1", "A", ("A", "B", "C", "Z"))])
    assert calls.count("relabel_train") == 1 and "['C']" in out


def test_scoring_without_a_tokenizer_json_relabels_train_first(monkeypatch, tmp_path, capsys):
    calls, out = _run(monkeypatch, tmp_path, capsys, score=True, val_labels=CONFIRMED,
                      tokenizer=False)
    assert calls.index("relabel_train") < calls.index("open_val_set")
    assert "NOT RUN -- --score-checkpoint trains nothing and no row" not in out


def test_training_relabels_train_and_plans_the_epoch(monkeypatch, tmp_path, capsys):
    calls, out = _run(monkeypatch, tmp_path, capsys, score=False, val_labels=CONFIRMED)
    assert calls.count("relabel_train") == 1 and calls.count("_batch_inventory") == 1
    assert calls.index("relabel_train") < calls.index("open_val_set")
    assert "NOT RUN" not in out.split("devices:")[0]


def test_letters_no_val_gold_confirms():
    val = SimpleNamespace(labels=[*CONFIRMED, _label("v5", "A", ("A", "B", "C", "D", "Z"))])
    assert rft.letters_no_val_gold_confirms(val, rft.OodSuite([], None, None)) == ["C", "D"]
    ood = rft.OodSuite([], SimpleNamespace(labels=[_label("o", "A", ("A", "E"))]), None)
    assert rft.letters_no_val_gold_confirms(SimpleNamespace(labels=CONFIRMED), ood) == ["E"]
    span_only = SimpleNamespace(labels=[_label("s", "Z", ("Q", "Z"), kind=SLOT_SPAN)])
    assert rft.letters_no_val_gold_confirms(span_only, rft.OodSuite([], None, None)) == []


# --- _labels renders each row once ------------------------------------------------------------


def _reference_labels(rows, *, config):
    """``_labels`` as it was at 73bcf12: its own ``render``, then ``training_texts``' second."""
    from qd_data.errors import QdRefusal
    from qd_data.render import DEFAULT_CAPS, render
    from qd_data.schema import NOUL_LETTER
    from qd_train.shards import UnencodableGold, answer_letter, training_texts

    labels, excluded = [], []
    for row in sorted(rows, key=lambda r: r.row_id):
        staged = []
        try:
            rendered = render(row.request, caps=DEFAULT_CAPS, seed=config.seed)
            by_name = {slot.name: slot for slot in rendered.slots}
            raw_language = row.metadata.get("language")
            language = raw_language if isinstance(raw_language, str) and raw_language else None
            for spec in training_texts(row, seed=config.seed, caps=DEFAULT_CAPS):
                slot = by_name[spec.slot_name]
                gold = (
                    NOUL_LETTER if spec.slot_kind == SLOT_SPAN
                    else answer_letter(row, spec.slot_name, slot.letter_to_value)
                )
                staged.append(rft.Label(
                    row_id=row.row_id, family_id=row.family_id, slot_name=spec.slot_name,
                    slot_kind=spec.slot_kind, gold_letter=gold,
                    letters=tuple(slot.letter_to_value), language=language,
                ))
        except (UnencodableGold, QdRefusal) as exc:
            excluded.append(f"{row.row_id}: {type(exc).__name__}: {exc}")
            continue
        labels.extend(staged)
    return labels, excluded


def _rows(config):
    rows = [needle_defect_row(c, config=config)
            for c in build_suite(target_tokens=1024, cases_per_depth=1, seed=5)]
    prose = [f"Question {i}: which of these sentences is about rivers number {i}?"
             for i in range(12)]
    rows += [ood_defect_row(c, config=config)
             for c in build_ood_suite(prose, per_category=4, seed=5)]
    # Over the 131,072-byte context cap: render refuses it, and _labels excludes the row.
    rows += [needle_defect_row(c, config=config)
             for c in build_suite(target_tokens=60_000, cases_per_depth=1, seed=5)[:1]]
    return rows


def test_labels_render_each_row_once_and_label_as_before(monkeypatch):
    config = DataConfig()
    rows = _rows(config)
    want = _reference_labels(rows, config=config)
    assert want[1], "the oversized row is excluded, so the refusal path is exercised"
    renders = []
    real = shards.render

    def counting(*a, **k):
        renders.append(1)
        return real(*a, **k)

    monkeypatch.setattr(shards, "render", counting)
    monkeypatch.setattr(rft, "render", counting, raising=False)
    assert rft._labels(rows, config=config) == want
    assert len(renders) == len(rows)
