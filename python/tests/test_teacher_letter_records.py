"""The shared letter-record helpers behind the 4-bit vs bf16 teacher comparison.

``tools/mac_mlx_bench.py teacher`` and ``tools/teacher_torch_score.py`` both write letter
distributions through ``mac_bench_common.letter_record`` and compare them through
``letter_divergence``; a wrong normalisation or a KL in the wrong direction would make the
comparison that decides whether 4-bit labels are usable quietly mean something else.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import mac_bench_common as common  # noqa: E402


def test_letter_record_log_softmax_normalises() -> None:
    rec = common.letter_record(row_id="r", n_tokens=5, letter_logits=[2.0, 1.0, -3.0])
    probs = [math.exp(v) for v in rec["letter_logsoftmax"]]
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert rec["letter_logsoftmax"][0] - rec["letter_logsoftmax"][1] == pytest.approx(1.0)
    assert rec["n_tokens"] == 5


def test_letter_record_refuses_non_finite_and_empty() -> None:
    with pytest.raises(SystemExit):
        common.letter_record(row_id="r", n_tokens=1, letter_logits=[1.0, float("nan")])
    with pytest.raises(SystemExit):
        common.letter_record(row_id="r", n_tokens=1, letter_logits=[])


def _ls(z: list[float]) -> list[float]:
    return common.letter_record(row_id="x", n_tokens=1, letter_logits=z)["letter_logsoftmax"]


def test_identical_distributions_diverge_by_zero() -> None:
    a = {"r1": _ls([1.0, 0.0]), "r2": _ls([0.0, 3.0, 1.0])}
    d = common.letter_divergence(a, a)
    assert d["n"] == 2
    assert d["mean_kl_ref_to_ours"] == pytest.approx(0.0, abs=1e-12)
    assert d["max_abs_logsoftmax_diff"] == pytest.approx(0.0, abs=1e-12)
    assert d["argmax_agreement"] == 2


def test_kl_is_reference_to_ours_and_matches_the_closed_form() -> None:
    p = [0.9, 0.1]  # reference
    q = [0.6, 0.4]  # ours
    ref = {"r": [math.log(v) for v in p]}
    ours = {"r": [math.log(v) for v in q]}
    want = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))
    d = common.letter_divergence(ours, ref)
    assert d["mean_kl_ref_to_ours"] == pytest.approx(want, rel=1e-12)
    backwards = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))
    assert d["mean_kl_ref_to_ours"] != pytest.approx(backwards, rel=1e-3)


def test_argmax_disagreement_is_counted_over_shared_rows_only() -> None:
    ours = {"a": _ls([2.0, 0.0]), "b": _ls([0.0, 2.0]), "only_ours": _ls([1.0, 0.0])}
    ref = {"a": _ls([2.0, 0.0]), "b": _ls([2.0, 0.0]), "only_ref": _ls([0.0, 1.0])}
    d = common.letter_divergence(ours, ref)
    assert (d["n"], d["n_ours"], d["n_ref"], d["argmax_agreement"]) == (2, 3, 3, 1)


def test_divergence_refuses_no_overlap_and_letter_count_mismatch() -> None:
    with pytest.raises(SystemExit, match="share no row_id"):
        common.letter_divergence({"a": _ls([1.0, 0.0])}, {"b": _ls([1.0, 0.0])})
    with pytest.raises(SystemExit, match="letters"):
        common.letter_divergence({"a": _ls([1.0, 0.0])}, {"a": _ls([1.0, 0.0, 2.0])})


def test_read_letter_records_round_trips_and_refuses_duplicates(tmp_path: Path) -> None:
    rows = [common.letter_record(row_id=r, n_tokens=3, letter_logits=[1.0, 2.0]) for r in "ab"]
    good = tmp_path / "good.jsonl"
    good.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = common.read_letter_records(good)
    assert set(got) == {"a", "b"} and got["a"] == rows[0]["letter_logsoftmax"]
    dup = tmp_path / "dup.jsonl"
    dup.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="twice"):
        common.read_letter_records(dup)


def test_prepare_teacher_items_counts_identity_and_refuses_multi_token_letters() -> None:
    items = [{"row_id": "r", "prompt_ids": [1, 2, 3], "letter_ids": [10, 11], "gold_row": 1}]

    def decode(ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def encode(text: str) -> list[int]:
        return [int(t) for t in text.split()]

    prepared, same = common.prepare_teacher_items(items, decode=decode, encode=encode)
    assert same == 1 and prepared == [("r", [1, 2, 3], [10, 11], 1)]

    def encode_split(text: str) -> list[int]:
        ids = [int(t) for t in text.split()]
        return [9, 9] if ids == [11] else ids

    with pytest.raises(SystemExit, match="one-token letter"):
        common.prepare_teacher_items(items, decode=decode, encode=encode_split)


def test_both_teacher_tools_write_through_the_shared_record() -> None:
    for tool in ("mac_mlx_bench.py", "teacher_torch_score.py"):
        src = (REPO / "tools" / tool).read_text(encoding="utf-8")
        assert "common.letter_record(" in src, tool
        assert "common.prepare_teacher_items(" in src, tool
