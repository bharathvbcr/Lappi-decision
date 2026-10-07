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
    fit_span_temperature,
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


@dataclass(frozen=True)
class Span:
    eval_row_id: str
    row_id: str
    slot_name: str
    rows: int
    top: tuple[int, int]
    start: tuple[float, ...]
    end: tuple[float, ...]
    gold_start: int
    gold_end: int


#: What a span line carries when the span entry can be fitted from it: all four or none.
SPAN_FIT_FIELDS = ("start_logits", "end_logits", "gold_start", "gold_end")


def _span(v: dict[str, object], key: tuple[str, str, str]) -> Span | None:
    present = [k for k in SPAN_FIT_FIELDS if k in v]
    if not present:
        return None
    if len(present) != len(SPAN_FIT_FIELDS):
        raise OracleRefusal(f"{key}: a span line carries {present}; all four or none")
    rows = int(v["rows"])  # type: ignore[call-overload]
    if v["noul_row"] != rows - RESERVED_NOUL_ROWS:
        raise OracleRefusal(f"{key}: noul_row is not the last row")
    start = tuple(_f32(x) for x in v["start_logits"])  # type: ignore[union-attr]
    end = tuple(_f32(x) for x in v["end_logits"])  # type: ignore[union-attr]
    if len(start) != rows or len(end) != rows:
        raise OracleRefusal(f"{key}: pointer scores/rows disagree")
    gs, ge = int(v["gold_start"]), int(v["gold_end"])  # type: ignore[call-overload]
    noul = rows - RESERVED_NOUL_ROWS
    if not (0 <= gs < rows and 0 <= ge < rows) or (gs == noul) != (ge == noul) or gs > ge:
        raise OracleRefusal(f"{key}: gold rows ({gs}, {ge}) are not a span gold")
    top = v["top"]
    return Span(key[0], key[1], key[2], rows, (int(top[0]), int(top[1])),  # type: ignore[index]
                start, end, gs, ge)


def read(paths: Sequence[Path]) -> tuple[list[Letter], list[Span], int]:
    """Every letter line in file order, the span lines that carry their pointer scores, and
    the count of span lines that do not; refusals mirror the binary's."""
    letters: list[Letter] = []
    spans: list[Span] = []
    spans_without = 0
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
                span = _span(v, key)
                if span is None:
                    spans_without += 1
                else:
                    spans.append(span)
                continue
            logits = tuple(_f32(x) for x in v["row_logits"])
            if len(logits) != v["rows"] or v["noul_row"] != v["rows"] - RESERVED_NOUL_ROWS:
                raise OracleRefusal(f"{key}: logits/rows/noul_row disagree")
            letters.append(Letter(eval_row, v["row_id"], v["slot_name"], v["kind"], v["rows"],
                                  v["gold_row"], v["top"], logits))
    if spans and spans_without:
        raise OracleRefusal("span lines with and without their pointer distributions")
    if not letters:
        raise OracleRefusal("no letter row: nothing to fit")
    return letters, spans, spans_without


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


def _span_nll(spans: list[Span], t: float) -> float:
    """``fit_span_temperature``'s inner ``nll``, restated: it is a closure there."""
    per_row = np.empty(2 * len(spans), dtype=np.float64)
    i = 0
    for s in spans:
        for z, y in ((s.start, s.gold_start), (s.end, s.gold_end)):
            v = np.asarray(z, dtype=np.float64) / t
            v = v - v.max()
            per_row[i] = v[y] - np.log(np.exp(v).sum())
            i += 1
    return float(-per_row.mean())


def _span_score(s: Span, temperature: float, conformal_quantile: float,
                noul_margin: float) -> dict[str, object]:
    """``answer.rs``'s span rule, both pointers through :func:`calibrate`."""
    p_start, start, m_start, set_start = calibrate(s.start, temperature, conformal_quantile)
    p_end, end, m_end, set_end = calibrate(s.end, temperature, conformal_quantile)
    noul = s.rows - RESERVED_NOUL_ROWS
    by_rule = start == noul or end == noul or end < start
    gold_abstains = s.gold_start == noul
    correct = gold_abstains if by_rule else (start, end) == (s.gold_start, s.gold_end)
    margin = min(m_start, m_end)
    answered = not by_rule and margin >= noul_margin
    return {
        "scored": True, "top": [start, end], "margin": margin, "correct": correct,
        "abstained_by_rule": by_rule, "answered": answered,
        "served_correct": correct if answered else gold_abstains,
        "p_gold": [p_start[s.gold_start], p_end[s.gold_end]],
        "_gold_abstains": gold_abstains,
        "_covered": int(s.gold_start in set_start) + int(s.gold_end in set_end),
    }


def fit_spans(spans: list[Span], *, alpha: float, target: float) -> dict[str, object]:
    """The span entry, the way ``calibration_fit.py`` defines each piece."""
    t = fit_span_temperature([s.start for s in spans], [s.end for s in spans],
                             [s.gold_start for s in spans], [s.gold_end for s in spans])
    read_at_t = [_span_score(s, t, 1.0, 0.0) for s in spans]
    margins = np.asarray([r["margin"] for r in read_at_t], dtype=np.float64)
    correct = np.asarray([r["correct"] for r in read_at_t])
    noul = fit_noul_margin(margins, correct, target_precision=target)
    # split_conformal_threshold reads only p(gold): a [p, 1 - p] row whose gold is column 0
    # hands it exactly 1 - p as the nonconformity score, for the 2n pointers interleaved.
    p_gold = [p for r in read_at_t for p in r["p_gold"]]  # type: ignore[attr-defined]
    q_hat = split_conformal_threshold(
        np.asarray([[p, 1.0 - p] for p in p_gold], dtype=np.float64),
        np.zeros(len(p_gold), dtype=int), alpha=alpha,
    )
    entry = fit_entry(temperature=t, nonconformity_quantile=q_hat, noul_margin=noul)
    at_t = _span_nll(spans, t)
    return {
        "n": len(spans), "temperature": t,
        "temperature_at_lower_bound": _span_nll(spans, TEMPERATURE_LO) <= at_t,
        "temperature_at_upper_bound": _span_nll(spans, TEMPERATURE_HI) <= at_t,
        "nonconformity_quantile": q_hat, "conformal_quantile": entry.conformal_quantile,
        "noul_margin": entry.noul_margin, "_entry": entry,
    }


def _span_summary(scored: list[dict[str, object] | None], how: str) -> dict[str, object]:
    done = [s for s in scored if s is not None]
    n = len(done)

    def ratio(num: int, den: int) -> float | None:
        return None if den == 0 else num / den

    answered = [s for s in done if s["answered"]]
    answered_correct = sum(1 for s in answered if s["correct"])
    served = sum(1 for s in done if s["served_correct"])
    return {
        "how": how, "n_scored": n, "n_not_scored": len(scored) - n,
        "gold_abstains": sum(1 for s in done if s["_gold_abstains"]),
        "verdict_correct": sum(1 for s in done if s["correct"]),
        "abstained_by_rule": sum(1 for s in done if s["abstained_by_rule"]),
        "abstained_by_margin": sum(
            1 for s in done if not s["abstained_by_rule"] and not s["answered"]),
        "answered": len(answered), "answered_correct": answered_correct,
        "answered_precision": ratio(answered_correct, len(answered)),
        "served_correct": served, "served_accuracy": ratio(served, n),
        "pointer_conformal_coverage": ratio(sum(s["_covered"] for s in done), 2 * n),
    }


def _span_line(s: Span, fold: str | None, params: str | None,
               scored: dict[str, object] | None) -> dict[str, object]:
    body = {"scored": False} if scored is None else {
        k: v for k, v in scored.items() if not k.startswith("_")}
    return {"eval_row_id": s.eval_row_id, "row_id": s.row_id, "slot_name": s.slot_name,
            "key": "span", "fold": fold, "params": params,
            "gold": [s.gold_start, s.gold_end], **body}


def run_spans(spans: list[Span], *, population: str, split_key: str | None, alpha: float,
              target: float) -> tuple[dict[str, object], dict[str, object], list[dict]]:
    """The report's ``span`` section, the span entry per table, and the span rows' lines."""
    def score(s: Span, fit: dict[str, object]) -> dict[str, object]:
        e = fit["_entry"]
        return _span_score(s, e.temperature, e.conformal_quantile, e.noul_margin)

    before = [_span_score(s, 1.0, 1.0, 0.0) for s in spans]
    report: dict[str, object] = {
        "rows": len(spans), "fitted": True,
        "verdict_top_disagreements": sum(
            1 for s, b in zip(spans, before, strict=True) if list(s.top) != b["top"]),
    }
    entries: dict[str, object] = {}
    lines: list[dict] = []
    if population == "all":
        fit = fit_spans(spans, alpha=alpha, target=target)
        entries["all"] = fit["_entry"]
        scored = [score(s, fit) for s in spans]
        report["fit"] = _public(fit)
        report["scored"] = _span_summary(scored, "in_sample")  # type: ignore[arg-type]
        lines = [_span_line(s, None, "all", sc) for s, sc in zip(spans, scored, strict=True)]
        return report, entries, lines
    assert split_key is not None
    folds = [fold_of(split_key, s.row_id) for s in spans]
    fits: dict[str, dict[str, object] | None] = {}
    report["folds"] = {}
    for f in ("a", "b"):
        members = [s for s, g in zip(spans, folds, strict=True) if g == f]
        if not members:
            fits[f] = None
            report["folds"][f] = {"n": 0, "fitted": False,  # type: ignore[index]
                                  "reason": f"fold {f} holds no span row"}
            continue
        fits[f] = fit_spans(members, alpha=alpha, target=target)
        entries[f"fold-{f}"] = fits[f]["_entry"]
        report["folds"][f] = {**_public(fits[f]), "fitted": True}  # type: ignore[index]
    out: list[dict[str, object] | None] = []
    for s, f in zip(spans, folds, strict=True):
        other = fits["b" if f == "a" else "a"]
        out.append(None if other is None else score(s, other))
    report["scored"] = _span_summary(out, "out_of_fold")
    for s, f, sc in zip(spans, folds, out, strict=True):
        params = None if sc is None else ("fold-b" if f == "a" else "fold-a")
        lines.append(_span_line(s, f, params, sc))
    return report, entries, lines


def _public(fit: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in fit.items() if not k.startswith("_")}


def run(
    paths: Sequence[Path], *, population: str, split_key: str | None, name: str,
    alpha: float = 0.1, target: float = 0.95,
) -> dict[str, object]:
    """Everything ``qd-calib-fit`` reports, tables and per-row lines included, by reference."""
    letters, spans, spans_without = read(paths)
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
    span_report: dict[str, object] | None = None
    span_entries: dict[str, object] = {}
    if spans:
        span_report, span_entries, span_lines = run_spans(
            spans, population=population, split_key=split_key, alpha=alpha, target=target)
        rows_out.extend(span_lines)
    return {
        "entries": entries,
        "tables": {which: fit_table(table_name, tables[which], span_entries.get(which))
                   for which, table_name in names.items()},
        "rows": rows_out,
        "span": span_report,
        "span_rows": len(spans) + spans_without,
        "span_rows_not_fitted": spans_without,
        "letter_rows": len(letters),
    }
