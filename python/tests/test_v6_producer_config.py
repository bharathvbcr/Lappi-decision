"""The committed v6 decisions producer config carries the pairwise length band (Fable Q14 P1).

GAP-W3B-NO-COMMITTED-PRODUCER-WRITES-PAIRWISE-LENGTH-BALANCE-2026-10-17: ``DataConfig()`` refuses a
pool whose P(gold is the longer response) is outside the band, but no committed ``qd-prep
decisions`` config set ``pairwise_length_balance``, so a rebuild reproduced the refused pool.
``data/decisions/v6-decisions.json`` is that config. It must hold the plan's band and change
nothing else that selects rows, so the rebuild reads the same pinned inputs as v5.
"""

from __future__ import annotations

import json
from pathlib import Path

from qd_data.config import DataConfig

DECISIONS = Path(__file__).resolve().parents[2] / "data" / "decisions"
NOT_A_SELECTOR = {"pairwise_length_balance", "train_caps_basis"}


def _load(name: str) -> dict[str, object]:
    return json.loads((DECISIONS / name).read_text())


def test_the_committed_v6_producer_config_carries_the_plans_band() -> None:
    band = _load("v6-decisions.json")["pairwise_length_balance"]
    assert band == {"low": 0.45, "high": 0.55}
    assert (band["low"], band["high"]) == DataConfig().pairwise_length_band


def test_v5_producer_config_stays_bandless() -> None:
    assert "pairwise_length_balance" not in _load("v5-allocation.json")


def test_the_v6_producer_config_differs_from_v5_only_by_the_band_and_its_note() -> None:
    v5, v6 = _load("v5-allocation.json"), _load("v6-decisions.json")
    changed = {k for k in v5.keys() | v6.keys() if v5.get(k) != v6.get(k)}
    assert changed == NOT_A_SELECTOR
