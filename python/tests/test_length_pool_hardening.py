"""M24 and M25 of the 2026-10-17 hardening pass (lane W3-b).

M24: ``pairwise.helpfulness`` carried a length prior (Lappi picked the longer program on 37 of
55 RM-Bench code pairs, AUDIT/training-audit-2026-10-06.md section 2.6). The v6 length-balance
band (data-clean plan item 9, ``decisions.rs`` ``pairwise_length_balance``: P(gold is the longer
answer) in [0.45, 0.55] per split) was opt-in in a Rust config no Python build read. It is now
a ``DataConfig`` rule, so ``DataConfig()`` has it and ``legacy_v5()`` does not, and
``load_decision_pool`` holds a pool to it.

M25: a family with val rows and no train rows (GAP-PAIRED-MARGIN-POOL-INCLUDES-A-FAMILY-WITH-NO-
TRAINING-ROWS-2026-10-05) is named and counted in every build manifest, and the pooled gate's
reading names it. Which families the gate pools is not touched (docs/promotion-decisions.json,
``paired_margin_population``, human-ruled 2026-10-08).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from data_fixtures import small_corpus

from qd_data.config import DataConfig
from qd_data.decisions import DecisionPoolError, load_decision_pool
from qd_data.dedupe import dedupe
from qd_data.manifest import Manifest, build_manifests, val_without_train
from qd_data.mixture import build_mixture
from qd_data.split import split
from qd_train.baseline import RequestDoc
from qd_train.tristate import NotRun

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import ft_linear_control as ftc  # noqa: E402

BAND = (0.45, 0.55)
HELPSTEER = "nvidia/HelpSteer2"
OPTIONS = (
    "Response A is more helpful.", "Response B is more helpful.",
    "Both responses are equally helpful.",
)


def _pair(i: int, split_name: str, *, gold_longer: bool) -> dict[str, object]:
    longer, shorter = "x" * 90, "y" * 20
    gold_is_a = i % 2 == 0
    a, b = (longer, shorter) if gold_is_a == gold_longer else (shorter, longer)
    return {
        "id": f"helpsteer2:{split_name}{i}", "source_id": HELPSTEER,
        "family_id": "pairwise.helpfulness", "stratum": "pairwise.helpfulness/ordered",
        "split": split_name, "group_key": f"{split_name}{i}", "licence": "cc-by-4.0",
        "context": f"user_prompt:\np{i}\n\nresponse_A:\n{a}\n\nresponse_B:\n{b}",
        "slot_name": "verdict", "options": list(OPTIONS),
        "gold_option": OPTIONS[0 if gold_is_a else 1], "gold_noul": False,
        "label_basis": "hard",
    }


def _pool(root: Path, lines: list[dict[str, object]]) -> Path:
    root.mkdir()
    body = "".join(json.dumps(x) + "\n" for x in lines).encode("utf-8")
    (root / "examples.jsonl").write_bytes(body)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "qd-decisions/v1", "mode": "build", "examples": len(lines),
        "examples_sha256": hashlib.sha256(body).hexdigest(),
        "text_bytes_by_family": {"pairwise.helpfulness": 1},
    }), encoding="utf-8")
    return root


def _lines(longer_of_ten: int) -> list[dict[str, object]]:
    """Ten train and ten val pairs, ``longer_of_ten`` of each split with the gold the longer."""
    return [
        _pair(i, s, gold_longer=i < longer_of_ten) for s in ("train", "val") for i in range(10)
    ]


def test_the_v6_defaults_carry_the_length_band_and_v5_does_not() -> None:
    assert DataConfig().pairwise_length_band == BAND
    assert DataConfig.legacy_v5().pairwise_length_band is None
    assert DataConfig().with_rules(v6=False) == DataConfig.legacy_v5()
    assert DataConfig.legacy_v5().with_rules(v6=True) == DataConfig()
    # v5's pinned manifests hash as they did; v6's hash names the band.
    assert "pairwise_length_band" not in DataConfig.legacy_v5().fingerprint()
    assert DataConfig().fingerprint()["pairwise_length_band"] == list(BAND)


def test_the_default_build_refuses_a_pool_whose_gold_is_the_longer_answer_too_often(
    tmp_path: Path,
) -> None:
    prior = _pool(tmp_path / "prior", _lines(8))  # P(gold longer) = 0.8 per split
    with pytest.raises(DecisionPoolError, match=r"pairwise length.*\[0\.45, 0\.55\]"):
        load_decision_pool(prior, config=DataConfig())
    # v5-pinned artifacts must still reproduce.
    assert load_decision_pool(prior, config=DataConfig.legacy_v5()).raw[HELPSTEER]
    assert load_decision_pool(prior).raw[HELPSTEER], "no config, no band (v5_a7_check's reader)"


def test_the_default_build_admits_a_pool_balanced_within_the_band(tmp_path: Path) -> None:
    balanced = _pool(tmp_path / "balanced", _lines(5))  # P = 0.5 per split
    pool = load_decision_pool(balanced, config=DataConfig())
    assert len(pool.raw[HELPSTEER]) == 20


def test_one_unbalanced_split_refuses_even_when_the_other_is_balanced(tmp_path: Path) -> None:
    lines = [_pair(i, "train", gold_longer=i < 5) for i in range(10)]
    lines += [_pair(i, "val", gold_longer=True) for i in range(10)]
    with pytest.raises(DecisionPoolError, match="val"):
        load_decision_pool(_pool(tmp_path / "p", lines), config=DataConfig())


def _manifests(config: DataConfig) -> dict[str, Manifest]:
    mixture = build_mixture(small_corpus(24), config=config)
    report = dedupe(list(mixture.rows), config=config)
    return build_manifests(
        config=config, mixture=mixture, dedupe_report=report,
        split_report=split(report, config=config),
    )


def test_every_build_manifest_carries_val_without_train() -> None:
    for config in (DataConfig(), DataConfig.legacy_v5()):
        for m in _manifests(config).values():
            block = m.to_json()["val_without_train"]
            assert set(block) == {"count", "families"}, block
            assert block["count"] == len(block["families"])


def test_val_without_train_names_and_counts_the_family() -> None:
    assert val_without_train({"fam.both": 9}, {"fam.only_val": 3, "fam.both": 2}) == {
        "count": 1, "families": {"fam.only_val": 3},
    }


def _doc(i: int, task: str) -> RequestDoc:
    offered = (("A", "yes"), ("B", "no"), ("Z", "noul"))
    return RequestDoc(
        row_id=f"r{i}", slot_name="slot", kind="choice", task=task, text=f"t{i}",
        value="yes", letter="A", offered=offered, metadata={},
    )


def test_the_pooled_gate_names_a_family_with_val_rows_and_no_train_rows() -> None:
    val = [_doc(i, "synth.general/slot") for i in range(4)]
    keys = [ftc.doc_key(d, by_slot_name=True) for d in val]
    verdicts = ftc.Verdicts(
        eval_row_id="e", seed=0, correct=dict.fromkeys(keys, True),
        kind_of=dict.fromkeys(keys, "choice"), by_slot_name=True, span_rows=0,
    )
    result = ftc.score_against_control([], val, verdicts, seed=0, engine=Path("/unused"))
    assert isinstance(result.gate, NotRun)
    assert "val rows but no train rows" in result.gate.reason
    assert "synth.general (val 4)" in result.gate.reason
