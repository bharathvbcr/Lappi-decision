"""v5's CLINC out-of-scope re-key (human-decisions.md item 2).

v4 keyed every CLINC row by intent, so all 1,350 oos utterances shared one repo key,
``clinc-intent:oos``, which the hash put in train: no val or held-out row of any CLINC family
was ever an abstention (GAP-CLINC-OOS-ONE-REPO-KEY-ONE-SPLIT-2026-10-02). v5 keys each oos
utterance on its own, ``clinc-oos:<blake2b-8 of the stripped utterance>``; in-scope rows keep
their intent key, and every row keeps v4's identity spelling.

Every test here fails on the pre-change code: there, an oos row's repo key is
``clinc-intent:oos`` and every oos row splits to train.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from qd_data.config import DataConfig
from qd_data.general import (
    CLINC_DOMAIN_FAMILY,
    CLINC_WITHIN_DOMAIN_FAMILY,
    ClincDomainMap,
    rewrite_clinc_two_stage,
)
from qd_data.loaders import ClincRow
from qd_data.mixture import RowRefused, clinc_keys, rewrite_clinc
from qd_data.split import assign_repo

CONFIG = DataConfig()

DOMAINS = {
    "banking": ("transfer", "balance", "freeze_account"),
    "travel": ("book_flight", "book_hotel", "car_rental"),
    "home": ("thermostat", "smart_lights", "grocery_list"),
}
VOCAB = [i for v in DOMAINS.values() for i in v] + [f"intent_{k:03d}" for k in range(20)]
DOMAIN_MAP = ClincDomainMap(domains=DOMAINS)

#: The cached clinc/clinc_oos ``plus`` fetch v4 was built from (fetch record a0841f0d...).
CLINC_CACHE = Path(
    "/Users/bharath/.cache/qd-decision/general/clinc__clinc_oos/"
    "155b9c710419136e17307b80d0a13e68cd46b4ec"
)


def _digest(text: str) -> str:
    return hashlib.blake2b(text.strip().encode("utf-8"), digest_size=8).hexdigest()


def _split(repo_key: str) -> str:
    return assign_repo(
        repo_key, seed=CONFIG.seed, train_fraction=CONFIG.train_fraction,
        val_fraction=CONFIG.val_fraction,
    )


def _all_families(raw: ClincRow) -> list[tuple[str, str]]:
    """``(repo_key, identity_key)`` of ``raw`` under each of the four CLINC families."""
    out = []
    for family in ("intent.classification", "intent.in_scope"):
        row = rewrite_clinc(raw, family_id=family, index=0, config=CONFIG, intent_vocabulary=VOCAB)
        out.append((row.repo_key, row.identity_key))
    for family in (CLINC_DOMAIN_FAMILY, CLINC_WITHIN_DOMAIN_FAMILY):
        row = rewrite_clinc_two_stage(raw, family_id=family, index=0, config=CONFIG,
                                      domain_map=DOMAIN_MAP)
        out.append((row.repo_key, row.identity_key))
    return out


def test_an_oos_utterance_is_its_own_split_unit_in_every_clinc_family() -> None:
    text = "  what is the airspeed of an unladen swallow  "
    raw = ClincRow(utterance=text, intent="oos", is_oos=True)
    d = _digest(text)
    keys = _all_families(raw)
    assert keys == [(f"clinc-oos:{d}", f"clinc-intent:oos::{d}")] * 4


def test_in_scope_keys_are_v4s_exactly() -> None:
    text = "move fifty dollars to savings"
    raw = ClincRow(utterance=text, intent="transfer", is_oos=False)
    d = _digest(text)
    assert _all_families(raw) == [("clinc-intent:transfer", f"clinc-intent:transfer::{d}")] * 4


def test_two_oos_utterances_get_two_units_and_the_units_reach_every_split() -> None:
    raws = [
        ClincRow(utterance=f"tell me fact number {k} about the moon", intent="oos", is_oos=True)
        for k in range(400)
    ]
    repo_keys = {rewrite_clinc(r, family_id="intent.in_scope", index=0, config=CONFIG,
                               intent_vocabulary=VOCAB).repo_key for r in raws}
    assert len(repo_keys) == 400
    by_split = Counter(_split(k) for k in repo_keys)
    # 400 units at 0.9 / 0.05 / 0.05: each of val and held-out expects 20; a unit-level
    # hash puts none there with probability (0.95)**400 < 1e-8 each.
    assert set(by_split) == {"train", "val", "heldout"}, by_split


def test_a_row_whose_oos_flag_disagrees_with_its_intent_is_refused() -> None:
    for raw in (
        ClincRow(utterance="hello there", intent="oos", is_oos=False),
        ClincRow(utterance="hello there", intent="transfer", is_oos=True),
    ):
        with pytest.raises(RowRefused) as excinfo:
            clinc_keys(raw, raw.utterance.strip())
        assert excinfo.value.reason_code == "oos_flag_disagrees_with_intent"


@pytest.mark.skipif(
    not (CLINC_CACHE / "intent_names.json").is_file(),
    reason=f"the cached clinc/clinc_oos fetch is not on this machine ({CLINC_CACHE})",
)
def test_the_real_fetch_reproduces_the_preregistered_per_split_oos_counts() -> None:
    """``campaign/v5-preregistered.DRAFT.json`` ``data.clinc_oos_rekey``: train 1,216, val 72,
    held-out 62 at seed 20260919, before MinHash dedupe, with the in-scope intents' 132 / 10 / 8
    unchanged. Counted through ``clinc_keys`` itself, on every upstream split v4 read."""
    names = json.loads((CLINC_CACHE / "intent_names.json").read_text(encoding="utf-8"))
    oos = Counter()
    in_scope_keys: set[str] = set()
    n_oos = 0
    for split_file in ("train", "validation", "test"):
        for line in (CLINC_CACHE / f"{split_file}.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            intent = names[rec["intent"]]
            raw = ClincRow(utterance=rec["text"], intent=intent, is_oos=intent == "oos")
            keys = clinc_keys(raw, raw.utterance.strip())
            if raw.is_oos:
                n_oos += 1
                oos[_split(keys.repo_key)] += 1
            else:
                in_scope_keys.add(keys.repo_key)
    assert n_oos == 1350
    assert dict(oos) == {"train": 1216, "val": 72, "heldout": 62}
    assert Counter(_split(k) for k in in_scope_keys) == {"train": 132, "val": 10, "heldout": 8}
