"""``_decode`` copies a batch's readout to the host once (ft-505105265aa, ruling 2j.10).

Per row it used to sync the device several times (``int(argmax)``, ``float(softmax)``,
``.double().tolist()``), ~22% GPU utilisation with one Python process at 100% on the J6 tail.
Now every score tensor a batch's verdicts read is packed into ONE float64 tensor on the device
and copied once. Parity is the verdict JSON: the conversion order (``.double()`` on the device,
before the copy) and the dtype/device ``noul_probability`` is computed in are unchanged, so the
verdicts are byte-identical.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_ensemble import _val_and_steps  # noqa: E402
from test_real_ft_score_checkpoint import tiny_master_checkpoints  # noqa: E402

#: Every way a tensor's value reaches Python. A device->host sync on an accelerator.
_TO_HOST = ("tolist", "item", "cpu", "numpy", "__float__", "__int__", "__index__", "__bool__")


@pytest.fixture
def host_reads(monkeypatch):
    """Records each tensor-to-Python read as ``(method, in_serving_scores)``: ``serving_scores``
    (``qd_train.heads``, not ``_decode``'s) still reads per span row, and is counted apart."""
    calls: list[tuple[str, bool]] = []
    for name in _TO_HOST:
        original = getattr(torch.Tensor, name)

        def counting(self, *args, _name=name, _original=original, **kwargs):
            in_heads = any(f.name == "serving_scores" for f in traceback.extract_stack())
            calls.append((_name, in_heads))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, counting)
    return calls


def _mixed_batch(n_choice: int, n_span: int):
    """One batch of ``n_choice`` letter rows and ``n_span`` span rows, with its labels, and a
    stand-in step to decode it."""
    import numpy as np

    from qd_train.artifacts import NO_SPAN, SLOT_CHOICE, SLOT_SPAN, SPAN_ABSTAIN
    from qd_train.shards import assemble_batch

    n = n_choice + n_span
    kinds = np.asarray([SLOT_CHOICE] * n_choice + [SLOT_SPAN] * n_span)
    seqs = [((np.arange(20) + r) % 60 + 1).astype(np.int32) for r in range(n)]
    batch = assemble_batch(
        seqs, kinds=kinds, target_index=np.full(n, 18),
        spans=np.asarray([[NO_SPAN, NO_SPAN]] * n_choice + [[SPAN_ABSTAIN, SPAN_ABSTAIN]] * n_span),
        candidates=[()] * n_choice + [(0, 4, 9)] * n_span, width=20, bucket=20, index=0,
    )
    labels = [
        rft.Label(row_id=f"c{r}", family_id="f", slot_name="choice", slot_kind=SLOT_CHOICE,
                  gold_letter="A", letters=("A", "B", rft.NOUL_LETTER))
        for r in range(n_choice)
    ] + [
        rft.Label(row_id=f"s{r}", family_id="f", slot_name="defect_span", slot_kind=SLOT_SPAN,
                  gold_letter=rft.NOUL_LETTER, letters=(rft.NOUL_LETTER,))
        for r in range(n_span)
    ]
    step = rft.RealFtStep(seed=0, device="cpu", vocab=64, width=32, hidden=8, heads=2,
                          lr=1e-3, span_weight=1.0)
    letter_id = {"A": 5, "B": 6, rft.NOUL_LETTER: 7}
    return step, batch, {0: labels}, letter_id


def _decode_reads(host_reads, n_choice: int, n_span: int) -> list[str]:
    step, batch, labels, letter_id = _mixed_batch(n_choice, n_span)
    host_reads.clear()
    decoded = rft._decode(step, [batch], labels, letter_id)
    assert decoded["rows_decoded"] == n_choice + n_span
    return [name for name, in_heads in host_reads if not in_heads and name != "__bool__"]


def test_decode_reads_the_device_once_per_batch_not_once_per_row(host_reads):
    """A letter batch: one packed copy and the list read off it, whatever the row count (24
    rows read ~75 times before)."""
    assert _decode_reads(host_reads, 24, 0) == ["cpu", "tolist"]
    assert _decode_reads(host_reads, 96, 0) == ["cpu", "tolist"]


def test_a_mixed_batchs_span_rows_add_no_read_of_decodes_own(host_reads):
    """The span rows' plan scalars and scores ride the same single copy. What still grows with
    span rows is ``heads.serving_scores``' own per-row check (not ``_decode``'s; named in the
    handoff), which this excludes."""
    assert _decode_reads(host_reads, 24, 8) == ["cpu", "tolist"]
    assert _decode_reads(host_reads, 96, 32) == ["cpu", "tolist"]


def test_decode_verdicts_are_the_ones_the_per_row_copies_wrote(tmp_path, monkeypatch):
    """The reference: the per-row reads this replaced, restated here. Same tops, same
    ``noul_probability`` bits, same ``row_logits``/``start_logits``/``end_logits`` lists."""
    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    val, _, steps = _val_and_steps(tiny)
    step = steps[0]
    decoded = rft._decode(step, val.plan, val.labels_for, val.letter_id)
    by_id = {(v["row_id"], v["slot_name"]): v for v in decoded["verdicts"]}
    checked = 0
    with torch.no_grad():
        for b, batch in enumerate(val.plan):
            letter_rows = {
                r: [val.letter_id[x] for x in [*[y for y in val.labels_for[b][r].letters
                                                  if y != rft.NOUL_LETTER], rft.NOUL_LETTER]]
                for r in range(batch.tokens.shape[0])
                if val.labels_for[b][r].slot_kind != rft.SLOT_SPAN
            }
            readout = rft.batch_readout(step, batch, letter_rows)
            for r, scores in readout.letters.items():
                label = val.labels_for[b][r]
                v = by_id[(label.row_id, label.slot_name)]
                noul_row = len(scores) - rft.RESERVED_NOUL_ROWS
                assert v["top"] == int(scores.argmax())
                assert v["noul_probability"] == float(torch.softmax(scores, dim=-1)[noul_row])
                assert json.dumps(v["row_logits"]) == json.dumps(scores.double().tolist())
                checked += 1
            for r, k in readout.span_index.items():
                label = val.labels_for[b][r]
                v = by_id[(label.row_id, label.slot_name)]
                assert v["top"] == [int(readout.starts[k].argmax()), int(readout.ends[k].argmax())]
                assert json.dumps(v["start_logits"]) == json.dumps(
                    readout.starts[k].double().tolist()
                )
                assert json.dumps(v["end_logits"]) == json.dumps(readout.ends[k].double().tolist())
                checked += 1
    assert checked == decoded["rows_decoded"] > 0


def test_noul_probability_is_computed_in_the_scores_own_dtype_not_in_float64():
    """A bf16 tower's softmax is taken in bf16 (on the device the scores are on) and only then
    widened -- as ``float(torch.softmax(row_scores)[noul])`` always did. A softmax in float64
    would differ in the low bits, which is exactly what a packed-copy rewrite could break."""
    step, batch, labels, letter_id = _mixed_batch(8, 0)
    full_head = step.lm_head
    step.lm_head = lambda hidden: full_head(hidden).to(torch.bfloat16)  # type: ignore[method-assign]
    decoded = rft._decode(step, [batch], labels, letter_id)
    with torch.no_grad():
        readout = rft.batch_readout(
            step, batch, {r: [5, 6, 7] for r in range(8)}
        )
    differs = 0
    for r, scores in readout.letters.items():
        assert scores.dtype == torch.bfloat16
        (v,) = [x for x in decoded["verdicts"] if x["row_id"] == f"c{r}"]
        expected = float(torch.softmax(scores, dim=-1)[2])
        assert v["noul_probability"] == expected
        in_f64 = float(torch.softmax(scores.double(), dim=-1)[2])
        differs += in_f64 != expected
        assert v["row_logits"] == scores.double().tolist()
    assert differs, "the fixture cannot tell a bf16 softmax from a float64 one"
