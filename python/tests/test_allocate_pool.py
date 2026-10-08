"""``qd-prep allocate``'s pool loads through the one pool door (Fable's v6 ruling R1).

Two candidate pools (``allocation.state = "not_applied"``, as ``qd-prep convert`` and
``qd-prep synth`` write them) are refused by :func:`load_decision_pool`; the one pool
``qd-prep allocate`` draws from them is admitted, with every row it wrote.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from qd_data.decisions import DecisionPoolError, load_decision_pool

#: A run is bounded: the fixture pools are a few rows.
_RUN_TIMEOUT_S = 120.0


def _row(i: int, *, family: str, source: str, licence: str, split: str) -> dict[str, object]:
    return {
        "id": f"{family}:{i}", "source_id": source, "family_id": family,
        "stratum": f"{family}/choice", "split": split, "group_key": f"{family}-g{i}",
        "licence": licence, "context": f"Which option fits?\n\nrecord {i} of {family}",
        "slot_name": "answer", "options": ["alpha", "beta"], "gold_option": "alpha",
        "gold_noul": False, "label_basis": "hard",
    }


def _pool(path: Path, rows: list[dict[str, object]]) -> str:
    """``rows`` as a candidate pool, serialised as serde_json writes them (sorted keys,
    compact, UTF-8). Returns the examples' sha256."""
    path.mkdir(parents=True)
    body = b"".join(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        for r in rows
    )
    (path / "examples.jsonl").write_bytes(body)
    sha = hashlib.sha256(body).hexdigest()
    (path / "manifest.json").write_text(json.dumps({
        "schema": "qd-decisions/v1", "mode": "build", "tool": "qd-prep convert",
        "examples": len(rows), "examples_sha256": sha,
        "allocation": {"state": "not_applied", "detail": "candidates"},
    }))
    return sha


def test_an_allocated_pool_loads_and_its_candidate_inputs_do_not(
    qd_prep_bin: Path, tmp_path: Path
) -> None:
    mnli = [_row(i, family="mnli.nli", source="nyu-mll/multi_nli", licence="oanc",
                 split="val" if i < 3 else "train") for i in range(12)]
    email = [_row(i, family="email.category", source="lappi/synth-email",
                  licence="synthetic-by-rule", split="val" if i < 2 else "train")
             for i in range(8)]
    sha_m = _pool(tmp_path / "mnli", mnli)
    sha_e = _pool(tmp_path / "email", email)
    for p in ("mnli", "email"):
        with pytest.raises(DecisionPoolError, match="allocation"):
            load_decision_pool(tmp_path / p)
    config = tmp_path / "allocation.json"
    config.write_text(json.dumps({
        "schema": "qd-allocation-config/v1", "seed": 3,
        "val_cap_per_family": 1000, "val_floor_per_family": 1, "val_cap_total": 10000,
        "pools": {"mnli": {"examples_sha256": sha_m, "draw": "capped"},
                  "email": {"examples_sha256": sha_e, "draw": "capped"}},
        "train_caps": {"mnli.nli": 5, "email.category": 100},
        "refused_licences": {}, "stated_zero_train_families": {},
    }))
    out = tmp_path / "allocated"
    proc = subprocess.run(
        [str(qd_prep_bin), "allocate", "--config", str(config),
         "--pool", f"mnli={tmp_path / 'mnli'}", "--pool", f"email={tmp_path / 'email'}",
         "--out", str(out), "--threads", "2"],
        capture_output=True, text=True, timeout=_RUN_TIMEOUT_S, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    pool = load_decision_pool(out)
    assert pool.manifest["allocation"]["state"] == "applied"
    by_source = {s: len(rows) for s, rows in pool.raw.items()}
    # mnli: train capped at 5 of 9, val 3 of 3; email: every row (6 train, 2 val).
    assert by_source == {"nyu-mll/multi_nli": 8, "lappi/synth-email": 8}
    assert pool.manifest["rows_by_family_split"] == {
        "email.category": {"train": 6, "val": 2}, "mnli.nli": {"train": 5, "val": 3},
    }
