"""An attestation's corpus key against a build's: ``max_pairs`` is compared only where it bounds a
row (GAP-V5-CONTROL-MAX-PAIRS-KEY-MISMATCH-2026-10-04).

v5's exclusion list was attested with ``max_pairs`` 400 (``real_ft_run.py``'s default), and
``tools/ft_linear_control.py`` keys 0 for the same corpus and refuses ``--max-pairs`` there. Under
``--no-repo-history`` without ``--commitpackft`` the value reads no row
(``real_tokenizer_pipeline.base_sources``), so v5's letter and option controls were refused
against v5's own attestation. Every other key, and ``max_pairs`` wherever it bounds a row, must
still match.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train.containment_strip import STRIP_RULE, STRIP_VERSION  # noqa: E402
from qd_train.exclusions import (  # noqa: E402
    ATTESTATION_NAME,
    ATTESTATION_VERSION,
    EXCLUSIONS_NAME,
    ExclusionRefusal,
    read_exclusions,
)
from qd_train.replay import DEFAULT_N, DEFAULT_THRESHOLD  # noqa: E402

#: v5's corpus as the box's attestation names it, minus the shas this test does not need:
#: no repository history and no --commitpackft, so max_pairs reads no row.
UNBOUNDED: dict[str, Any] = {
    "rev": "8e6a00952a09e31a09cc3f5270229a935b7235cc",
    "max_pairs": 400,
    "commitpackft": None,
    "defect_class": "commitpackft-composed-v2",
    "defect_max_rows": None,
    "repo_history": False,
    "general_max_rows": 200000,
}
KEYS = ["k1", "k2"]


def _listed(tmp: Path, attested: dict[str, Any]) -> Path:
    """A well-formed list and attestation whose only property under test is ``corpus``."""
    raw = "".join(f"{k}\n" for k in KEYS).encode()
    (tmp / EXCLUSIONS_NAME).write_bytes(raw)
    att = {
        "version": ATTESTATION_VERSION,
        "tool": "qd-prep containment",
        "n": DEFAULT_N,
        "threshold": DEFAULT_THRESHOLD,
        "export": {"template_strip": {
            "applied": True, "rule": STRIP_RULE, "version": STRIP_VERSION,
        }},
        "exclusions_sha256": hashlib.sha256(raw).hexdigest(),
        "n_exclusions": len(KEYS),
        "clean": True,
        "remaining_hits": {"val": 0, "heldout": 0},
        "excluded_from": "train",
        "corpus": attested,
    }
    (tmp / ATTESTATION_NAME).write_text(json.dumps(att), encoding="utf-8")
    return tmp / EXCLUSIONS_NAME


def test_max_pairs_that_bounds_no_row_is_not_compared(tmp_path: Path) -> None:
    """v5's case exactly: attested 400, the control's 0, no history and no --commitpackft.
    Against the pre-fix comparison this raises 'was made for corpus'."""
    listed = _listed(tmp_path, UNBOUNDED)
    got = read_exclusions(listed, corpus={**UNBOUNDED, "max_pairs": 0})
    assert got.keys == frozenset(KEYS)
    # A key that names no max_pairs at all is the same corpus there too.
    without = {k: v for k, v in UNBOUNDED.items() if k != "max_pairs"}
    assert read_exclusions(listed, corpus=without).keys == frozenset(KEYS)


@pytest.mark.parametrize(("attested", "built", "why"), [
    # --commitpackft samples max_pairs rows: the value decides the corpus.
    ({**UNBOUNDED, "commitpackft": "commitpackft"},
     {**UNBOUNDED, "commitpackft": "commitpackft", "max_pairs": 0}, "commitpackft bounds"),
    # repository history (corpus_identity names repo_history only when False) reads max_pairs.
    ({k: v for k, v in UNBOUNDED.items() if k != "repo_history"},
     {k: v for k, v in UNBOUNDED.items() if k != "repo_history"} | {"max_pairs": 0},
     "history bounds"),
    # Only one side unbounded is a different corpus, whatever max_pairs says.
    ({**UNBOUNDED, "commitpackft": "commitpackft"}, UNBOUNDED, "one side samples"),
    # Every other key is still compared under the unbounded case.
    (UNBOUNDED, {**UNBOUNDED, "max_pairs": 0, "rev": "another"}, "rev differs"),
    (UNBOUNDED, {**UNBOUNDED, "general_max_rows": 1}, "general rows differ"),
])
def test_every_key_that_decides_rows_is_still_compared(
    tmp_path: Path, attested: dict[str, Any], built: dict[str, Any], why: str,
) -> None:
    listed = _listed(tmp_path, attested)
    with pytest.raises(ExclusionRefusal, match="was made for corpus"):
        read_exclusions(listed, corpus=built)


@pytest.mark.parametrize("attested", [None, [], "corpus", 400])
def test_an_attestation_without_a_corpus_object_is_refused(
    tmp_path: Path, attested: object,
) -> None:
    listed = _listed(tmp_path, {})
    att = json.loads((tmp_path / ATTESTATION_NAME).read_text(encoding="utf-8"))
    att["corpus"] = attested
    (tmp_path / ATTESTATION_NAME).write_text(json.dumps(att), encoding="utf-8")
    with pytest.raises(ExclusionRefusal, match="was made for corpus"):
        read_exclusions(listed, corpus=UNBOUNDED)
