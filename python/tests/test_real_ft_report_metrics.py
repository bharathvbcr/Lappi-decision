"""The eval row's report-only metrics from Fable round G, and the proof they touch no gate.

``calibration_states`` adds ``ece.report.pooled``, ``ece.report.lang.{language|none}``
(G2, ``GAP-ECE-GATE-IS-NOT-RUN-ON-THE-FULL-MIXTURE``) and
``degenerate_head.family.{family}.{shape}`` (G1, ``GAP-DEGENERATE-HEAD-FAILS-ON-BINARY-SLOTS-TOO``)
to the metrics it returns. Rule 2 is the point: the ``ece`` gate and the ``degenerate_head``
control it returns, and every metric it returned before, must be BYTE-IDENTICAL to what the
code returned before the change. ``GOLDEN_SHA256`` was computed from the unmodified function
(main ``90e7586``) on this file's fixture; a change that moves any gate input moves the hash.
``qd-gate-report`` recomputes the new metrics and refuses a row they disagree with
(``test_gate_report_parity.py``).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_train.calibration_fit import ece_gate  # noqa: E402
from qd_train.eval_harness import degenerate_head_check  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

#: sha256 of the gate, the control and every metric calibration_states returned on this file's
#: fixture BEFORE the report-only metrics existed (main 90e7586).
GOLDEN_SHA256 = "b2a7d9b030a290547c759677a7e28007b0e983105c3fb41d9586a3a979b46456"
#: The names this change adds; everything else must be what it was.
NEW_PREFIXES = ("ece.report.", "degenerate_head.family")
#: Added later (2026-10-03): the per-shape class share the promotion verdict reads under a
#: share-only degenerate_head_floor. A measurement with a verdict of its own, so it is outside
#: the golden digest but not one of the report-only names above.
SHARE_SUFFIX = ".top_class_share"


def _is_new(name: str) -> bool:
    return name.startswith(NEW_PREFIXES) or name.endswith(SHARE_SUFFIX)

#: (family, kind, rows incl. noul, languages to cycle, count)
FAMILIES = (
    ("code.defect_class", "choice", 5, ("python", "go"), 240),
    ("knowledge.multiple_choice", "choice", 5, (None,), 160),
    ("intent.in_scope", "choice", 3, (None,), 150),
    ("intent.domain", "choice", 11, (None,), 120),
)


def _verdicts() -> list[dict[str, object]]:
    """A scoring run's letter and span verdicts, from Python's own seeded generator (whose
    ``random()`` stream is stable across versions), on a 1/16 logit grid so ties occur."""
    rng = random.Random(20261001)
    out: list[dict[str, object]] = []
    for family, kind, rows, languages, count in FAMILIES:
        options = rows - 1
        for i in range(count):
            gold = int(rng.random() * options)
            logits = [round((rng.random() * 6.0 + (3.0 if r == gold else 0.0)) * 16) / 16
                      for r in range(rows)]
            logits[-1] -= 4.0
            top = max(range(rows), key=lambda r, z=logits: (z[r], -r))
            out.append({
                "kind": kind, "row_id": f"{family}:{i}", "slot_name": "answer",
                "family_id": family, "language": languages[i % len(languages)], "rows": rows,
                "noul_row": options, "gold_row": gold, "top": top, "row_logits": logits,
                "correct": top == gold, "expected_abstain": False,
            })
    for i in range(40):
        out.append({"kind": "span", "row_id": f"span:{i}", "slot_name": "evidence",
                    "family_id": "qa.answer_span", "rows": 30, "noul_row": 29,
                    "gold_row": None, "top": [1, 2], "correct": i % 4 != 0,
                    "expected_abstain": False})
    return out


def _canonical(metrics: dict[str, object], ece: object, degenerate: object) -> str:
    return json.dumps({
        "ece": ece.to_json(),  # type: ignore[attr-defined]
        "degenerate_head": degenerate.to_json(),  # type: ignore[attr-defined]
        "metrics": {k: v.to_json() for k, v in sorted(metrics.items())  # type: ignore[attr-defined]
                    if not _is_new(k)},
    }, sort_keys=True)


def test_the_gate_the_control_and_every_prior_metric_are_byte_identical():
    metrics, ece, degenerate = rft.calibration_states({"verdicts": _verdicts()})
    digest = hashlib.sha256(_canonical(metrics, ece, degenerate).encode()).hexdigest()
    assert digest == GOLDEN_SHA256, digest


def test_no_report_metric_reaches_the_gate_or_the_control():
    metrics, ece, degenerate = rft.calibration_states({"verdicts": _verdicts()})
    assert isinstance(ece, NotRun), "the fixture has rows without a language, as the mixture"
    assert "report" not in ece.reason
    assert isinstance(degenerate, Ran)
    shapes = {k for k in metrics if k.startswith("degenerate_head.choice.") and not _is_new(k)}
    assert (degenerate.n, degenerate.n_total) == (len(shapes), len(shapes))
    shares = {k for k in metrics if k.endswith(SHARE_SUFFIX)}
    assert {f"{k}{SHARE_SUFFIX}" for k in shapes} == shares
    for name in metrics:
        if name.startswith(NEW_PREFIXES):
            state = metrics[name]
            assert isinstance(state, NotRun) or state.passed is True, (
                f"{name}: a report-only metric carries no verdict")


def _padded(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    width = max(int(v["rows"]) for v in rows)  # type: ignore[call-overload]
    probs = np.zeros((len(rows), width))
    for i, v in enumerate(rows):
        z = np.asarray([v["row_logits"]], dtype=np.float64)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        probs[i, : z.shape[1]] = (z / z.sum(axis=1, keepdims=True))[0]
    return probs, np.asarray([v["gold_row"] for v in rows], dtype=int)


def test_g2_pooled_and_per_language_ece_including_none():
    verdicts = _verdicts()
    metrics, _, _ = rft.calibration_states({"verdicts": verdicts})
    letters = [v for v in verdicts if v["kind"] != "span"]
    oracle = ece_gate(*_padded(letters))
    assert isinstance(oracle, Ran)
    got = metrics["ece.report.pooled"]
    assert isinstance(got, Ran)
    assert (got.value, got.n, got.n_total) == (oracle.value, len(letters), len(letters))
    for language in ("python", "go", None):
        rows = [v for v in letters if v["language"] == language]
        oracle = ece_gate(*_padded(rows))
        got = metrics[f"ece.report.lang.{language or 'none'}"]
        assert isinstance(got, Ran) and isinstance(oracle, Ran)
        assert (got.value, got.n) == (oracle.value, len(rows)), language


def test_g1_degenerate_head_per_family_and_shape():
    verdicts = _verdicts()
    metrics, _, _ = rft.calibration_states({"verdicts": verdicts})
    for family, kind, rows, _, count in FAMILIES:
        shape = f"{kind}.k{rows - 1}"
        mine = [v for v in verdicts if v["family_id"] == family]
        probs, _ = rft.letter_distributions(mine)[shape]
        head = degenerate_head_check(probs)
        got = metrics[f"degenerate_head.family.{family}.{shape}"]
        assert isinstance(got, Ran) and isinstance(head, Ran)
        assert (got.value, got.n, got.n_total) == (head.value, count, count)
        share = np.bincount(np.argmax(probs, axis=1), minlength=probs.shape[1]).max() / count
        assert f"top predicted share {share:.3f}" in got.detail
    assert not any(k.startswith("degenerate_head.family.qa.") for k in metrics), (
        "a span-only family has no letter distribution to read")


def test_a_letter_row_without_a_family_leaves_no_per_family_head():
    verdicts = _verdicts()
    verdicts[0] = {k: v for k, v in verdicts[0].items() if k != "family_id"}
    metrics, _, _ = rft.calibration_states({"verdicts": verdicts})
    state = metrics["degenerate_head.family"]
    assert isinstance(state, NotRun) and "carry no family_id" in state.reason
    assert not any(k.startswith("degenerate_head.family.") for k in metrics)
