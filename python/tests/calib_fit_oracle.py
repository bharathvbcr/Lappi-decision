"""The Python reference ``qd-calib-fit`` is pinned against.

Every fitted number comes from the library functions the Rust ports, called as they are:
``qd_train.calibration_fit``'s ``fit_temperature``, ``fit_noul_margin``, ``ece_gate`` and
``expected_calibration_error``, ``fit_entry`` (which applies
``probability_cutoff_from_nonconformity``) and ``fit_table``, and
``qd_train.eval_harness.split_conformal_threshold``. What this module adds is only the
composition the Rust binary performs -- group, split, fit, score -- and :func:`calibrate`, the
runtime's ``calibration::calibrate`` in its own operation order.

A reference oracle and nothing else (the language policy): no tool imports it, and it writes
nothing. ``test_calib_fit_parity.py`` and that file's benchmark are its only callers.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qd_train.calibration_fit import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    RESERVED_NOUL_ROWS,
    ece_gate,
    expected_calibration_error,
    fit_entry,
    fit_noul_margin,
    fit_table,
    fit_temperature,
)
from qd_train.eval_harness import split_conformal_threshold
from qd_train.tristate import Ran

#: ``fit_temperature``'s defaults, needed only to state the bound diagnostics.
TEMPERATURE_LO = 0.05
TEMPERATURE_HI = 20.0


class OracleRefusal(ValueError):
    """An input the reference refuses, as the binary does."""


@dataclass(frozen=True)
class Letter:
    eval_row_id: str
    row_id: str
    slot_name: str
    kind: str
    rows: int
    gold: int
    top: int
    logits: tuple[float, ...]


def calibrate(
    logits: Sequence[float], temperature: float, conformal_quantile: float = 1.0
) -> tuple[list[float], int, float, list[int]]:
    """``calibration::calibrate``: f32 logits, f64 softmax, first-maximum top, margin, set.

    The denominator is ``functools.reduce(operator.add, ...)`` -- a left-to-right sum, as
    Rust's ``Iterator::sum`` is. Python's own ``sum()`` of floats is compensated since 3.12
    and would not be the runtime's number.
    """
    scaled = [v / temperature for v in logits]
    top_logit = max(scaled)
    exps = [math.exp(v - top_logit) for v in scaled]
    total = functools.reduce(operator.add, exps)
    probs = [e / total for e in exps]
    top = 0
    for i, p in enumerate(probs):
        if p > probs[top]:
            top = i
    second = 0.0
    for i, p in enumerate(probs):
        if i != top and p > second:
            second = p
    members = [i for i, p in enumerate(probs) if p >= conformal_quantile]
    if top not in members:
        members = sorted([*members, top])
    return probs, top, probs[top] - second, members


def _nll(z: np.ndarray, y: np.ndarray, t: float) -> float:
    """``fit_temperature``'s inner ``nll``, restated: it is a closure there."""
    s = z / t
    s = s - s.max(axis=1, keepdims=True)
    rows = np.arange(len(y))
    return float(-(s[rows, y] - np.log(np.exp(s).sum(axis=1))).mean())


def _f32(value: object) -> float:
    """The runtime reads logits as f32; a value that is not a number or overflows is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleRefusal(f"row_logits holds {value!r}")
    with np.errstate(over="ignore"):
        x = float(np.float32(value))
    if not math.isfinite(x):
        raise OracleRefusal(f"row_logits holds a non-finite value {value!r}")
    return x


def read(paths: Sequence[Path]) -> tuple[list[Letter], int]:
    """Every letter line in file order, and the span count; refusals mirror the binary's."""
    letters: list[Letter] = []
    spans = 0
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        first: str | None = None
        for raw in path.read_text(encoding="utf-8").split("\n"):
            if not raw.strip():
                continue
            v = json.loads(raw)
            eval_row = v["eval_row_id"]
            first = eval_row if first is None else first
            if eval_row != first:
                raise OracleRefusal(f"{path} mixes eval rows {first} and {eval_row}")
            key = (eval_row, v["row_id"], v["slot_name"])
            if key in seen:
                raise OracleRefusal(f"{key} appears twice")
            seen.add(key)
            if v["kind"] == "span":
                spans += 1
                continue
            logits = tuple(_f32(x) for x in v["row_logits"])
            if len(logits) != v["rows"] or v["noul_row"] != v["rows"] - RESERVED_NOUL_ROWS:
                raise OracleRefusal(f"{key}: logits/rows/noul_row disagree")
            letters.append(Letter(eval_row, v["row_id"], v["slot_name"], v["kind"], v["rows"],
                                  v["gold_row"], v["top"], logits))
    if not letters:
        raise OracleRefusal("no letter row: nothing to fit")
    return letters, spans


def fold_of(split_key: str, row_id: str) -> str:
    """``qd-margin-probe``'s split: fold a iff ``sha256(key || 0x00 || row_id)[0]`` is even."""
    digest = hashlib.sha256(split_key.encode() + b"\0" + row_id.encode()).digest()
    return "a" if digest[0] % 2 == 0 else "b"


def _ece(probs: list[list[float]], golds: list[int]) -> dict[str, object]:
    state = ece_gate(np.asarray(probs, dtype=np.float64), np.asarray(golds, dtype=int))
    if not isinstance(state, Ran):
        return {"state": "not_run", "reason": state.reason}
    _, populated = expected_calibration_error(
        np.asarray(probs, dtype=np.float64), np.asarray(golds, dtype=int)
    )
    return {"state": "ran", "value": state.value, "n": state.n, "populated_bins": populated,
            "passed": state.passed}


def fit_rows(rows: list[Letter], *, alpha: float, target: float) -> dict[str, object]:
    """One entry, the way ``calibration_fit.py`` defines each piece."""
    z = np.asarray([r.logits for r in rows], dtype=np.float64)
    y = np.asarray([r.gold for r in rows], dtype=int)
    t = fit_temperature(z, y)
    read_at_t = [calibrate(r.logits, t) for r in rows]
    margins = np.asarray([m for _, _, m, _ in read_at_t], dtype=np.float64)
    correct = np.asarray([top == r.gold for r, (_, top, _, _) in zip(rows, read_at_t, strict=True)])
    noul = fit_noul_margin(margins, correct, target_precision=target)
    q_hat = split_conformal_threshold(
        np.asarray([p for p, _, _, _ in read_at_t], dtype=np.float64), y, alpha=alpha
    )
    entry = fit_entry(temperature=t, nonconformity_quantile=q_hat, noul_margin=noul)
    at_t = _nll(z, y, t)
    return {
        "n": len(rows), "temperature": t,
        "temperature_at_lower_bound": _nll(z, y, TEMPERATURE_LO) <= at_t,
        "temperature_at_upper_bound": _nll(z, y, TEMPERATURE_HI) <= at_t,
        "nonconformity_quantile": q_hat, "conformal_quantile": entry.conformal_quantile,
        "noul_margin": entry.noul_margin, "_entry": entry,
    }


def _score(row: Letter, fit: dict[str, object]) -> dict[str, object]:
    entry = fit["_entry"]
    probs, top, margin, members = calibrate(row.logits, entry.temperature,
                                            entry.conformal_quantile)
    return {
        "scored": True, "top": top, "correct": top == row.gold, "p_gold": probs[row.gold],
        "confidence": probs[top], "margin": margin,
        "abstained_by_margin": margin < entry.noul_margin,
        "abstained_noul_row_or_margin": top == row.rows - RESERVED_NOUL_ROWS
        or margin < entry.noul_margin,
        "gold_in_set": row.gold in members, "set_size": len(members), "_probs": probs,
    }


def _summary(rows: list[Letter], scored: list[dict[str, object] | None], how: str) -> dict:
    done = [(r, s) for r, s in zip(rows, scored, strict=True) if s is not None]
    n = len(done)
    retained = [s for _, s in done if not s["abstained_by_margin"]]
    retained_correct = sum(1 for s in retained if s["correct"])
    covered = sum(1 for _, s in done if s["gold_in_set"])
    sizes = sum(s["set_size"] for _, s in done)

    def ratio(num: int, den: int) -> float | None:
        return None if den == 0 else num / den

    ece = (_ece([s["_probs"] for _, s in done], [r.gold for r, _ in done]) if n else
           {"state": "not_run", "reason": "no row of this entry could be scored"})
    return {
        "how": how, "n_scored": n, "n_not_scored": len(rows) - n, "ece_after": ece,
        "abstained_by_margin": sum(1 for _, s in done if s["abstained_by_margin"]),
        "abstained_noul_row_or_margin": sum(
            1 for _, s in done if s["abstained_noul_row_or_margin"]),
        "retained": len(retained), "retained_correct": retained_correct,
        "retained_precision": ratio(retained_correct, len(retained)),
        "covered": covered, "conformal_coverage": ratio(covered, n),
        "mean_set_size": ratio(sizes, n),
    }


def _public(fit: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in fit.items() if not k.startswith("_")}


def run(
    paths: Sequence[Path], *, population: str, split_key: str | None, name: str,
    alpha: float = 0.1, target: float = 0.95,
) -> dict[str, object]:
    """Everything ``qd-calib-fit`` reports, tables and per-row lines included, by reference."""
    letters, spans = read(paths)
    groups: dict[str, list[Letter]] = {}
    for r in letters:
        groups.setdefault(f"{r.kind}:{r.rows}", []).append(r)
    entries: dict[str, dict] = {}
    rows_out: list[dict[str, object]] = []
    tables: dict[str, dict[tuple[str, int], object]] = {"all": {}, "fold-a": {}, "fold-b": {}}
    for key, rows in groups.items():
        kind, width = rows[0].kind, rows[0].rows
        noul_row = width - RESERVED_NOUL_ROWS
        if kind == "choice" and not MIN_OPTIONS <= width - RESERVED_NOUL_ROWS <= MAX_OPTIONS:
            raise OracleRefusal(f"{key}: outside the schema")
        before = [calibrate(r.logits, 1.0) for r in rows]
        entry: dict[str, object] = {
            "kind": kind, "rows": width, "options": width - RESERVED_NOUL_ROWS, "n": len(rows),
            "gold_noul": sum(1 for r in rows if r.gold == noul_row),
            "verdict_top_disagreements": sum(
                1 for r, (_, top, _, _) in zip(rows, before, strict=True) if r.top != top),
            "ece_before": _ece([p for p, _, _, _ in before], [r.gold for r in rows]),
        }
        if population == "all":
            fit = fit_rows(rows, alpha=alpha, target=target)
            tables["all"][(kind, width)] = fit["_entry"]
            scored = [_score(r, fit) for r in rows]
            entry["fit"] = _public(fit)
            entry["scored"] = _summary(rows, scored, "in_sample")
            for r, s in zip(rows, scored, strict=True):
                rows_out.append({"key": key, "row_id": r.row_id, "slot_name": r.slot_name,
                                 "eval_row_id": r.eval_row_id, "fold": None, "params": "all",
                                 **{k: v for k, v in s.items() if not k.startswith("_")}})
        else:
            assert split_key is not None
            folds = [fold_of(split_key, r.row_id) for r in rows]
            fits: dict[str, dict[str, object] | None] = {}
            entry["folds"] = {}
            for f in ("a", "b"):
                members = [r for r, g in zip(rows, folds, strict=True) if g == f]
                if not members:
                    fits[f] = None
                    entry["folds"][f] = {"n": 0, "fitted": False,
                                         "reason": f"fold {f} holds no row of {key}"}
                    continue
                fits[f] = fit_rows(members, alpha=alpha, target=target)
                tables[f"fold-{f}"][(kind, width)] = fits[f]["_entry"]
                entry["folds"][f] = {**_public(fits[f]), "fitted": True}
            scored: list[dict[str, object] | None] = []
            for r, f in zip(rows, folds, strict=True):
                other = fits["b" if f == "a" else "a"]
                scored.append(None if other is None else _score(r, other))
            entry["scored"] = _summary(rows, scored, "out_of_fold")
            for r, f, s in zip(rows, folds, scored, strict=True):
                params = None if s is None else ("fold-b" if f == "a" else "fold-a")
                body = {"scored": False} if s is None else {
                    k: v for k, v in s.items() if not k.startswith("_")}
                rows_out.append({"key": key, "row_id": r.row_id, "slot_name": r.slot_name,
                                 "eval_row_id": r.eval_row_id, "fold": f, "params": params,
                                 **body})
        entries[key] = entry
    names = ({"all": name} if population == "all"
             else {"fold-a": f"{name}.fold-a", "fold-b": f"{name}.fold-b"})
    return {
        "entries": entries,
        "tables": {which: fit_table(table_name, tables[which])
                   for which, table_name in names.items()},
        "rows": rows_out,
        "span_rows": spans,
        "letter_rows": len(letters),
    }
