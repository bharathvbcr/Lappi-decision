"""The tri-state: a check that could not run must never read as a check that passed.

This module exists because of one recurring failure. A result is modelled as
``Optional[bool]``, a report renders ``None`` as "no failures found", and a suite
that never executed is indistinguishable from a suite that executed cleanly.
That is how "approved" comes to mean "unexamined".

Here ``NotRun`` has no ``passed`` attribute at all. A report that renders it as
passing has to fabricate the field, which raises. The impossibility is structural,
not a matter of remembering to check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self

__all__ = ["Ran", "NotRun", "TriState", "aggregate", "parse_tristate"]


class _TriStateBase:
    """Shared marker. Deliberately carries no ``passed`` and no ``value``."""

    __slots__ = ()

    state: Literal["ran", "not_run"]

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Ran(_TriStateBase):
    """A check that executed. ``passed`` is mandatory and never inferred."""

    passed: bool
    value: float | int | str | None = None
    n: int | None = None
    n_total: int | None = None
    detail: str = ""

    state: Literal["ran"] = "ran"

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError(f"Ran.passed must be bool, got {type(self.passed).__name__}")
        # Coverage is carried as a pair or not at all. A bare `n` invites a report
        # to print "n examined" as though it were the whole population.
        if (self.n is None) != (self.n_total is None):
            raise ValueError(
                "n and n_total are carried together or not at all: "
                f"got n={self.n!r}, n_total={self.n_total!r}. "
                "A sample size without a population reads as full coverage."
            )
        if self.n is not None:
            if self.n < 0 or self.n_total is None or self.n_total < 0:
                raise ValueError(f"n and n_total must be non-negative: n={self.n}, n_total={self.n_total}")
            if self.n > self.n_total:
                raise ValueError(f"examined {self.n} of {self.n_total}: n exceeds n_total")

    @property
    def is_complete_coverage(self) -> bool:
        """True only when the check saw every eligible item.

        When ``n``/``n_total`` were not supplied, coverage is *unstated*, which is
        not the same as complete — so this returns False rather than assuming.
        """
        if self.n is None or self.n_total is None:
            return False
        return self.n == self.n_total

    def coverage_str(self) -> str:
        if self.n is None or self.n_total is None:
            return "coverage unstated"
        return f"{self.n}/{self.n_total}"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"state": "ran", "passed": self.passed}
        if self.value is not None:
            out["value"] = self.value
        if self.n is not None:
            out["n"] = self.n
            out["n_total"] = self.n_total
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True, slots=True)
class NotRun(_TriStateBase):
    """A check that did not execute. It has no ``passed`` field, on purpose.

    ``reason`` is mandatory and must be non-empty: "not run" without a reason is
    indistinguishable from "forgotten", and the two need different responses.
    """

    reason: str

    state: Literal["not_run"] = "not_run"

    def __post_init__(self) -> None:
        if not self.reason or not self.reason.strip():
            raise ValueError(
                "NotRun requires a non-empty reason. A check recorded as not-run "
                "without saying why cannot be distinguished from one that was forgotten."
            )

    def to_json(self) -> dict[str, Any]:
        return {"state": "not_run", "reason": self.reason}


TriState = Ran | NotRun


def parse_tristate(raw: object, *, field: str = "<unnamed>") -> TriState:
    """Parse a tri-state, refusing every shape that could read as a false pass."""
    if not isinstance(raw, dict):
        raise TypeError(f"{field}: tri-state must be an object, got {type(raw).__name__}")

    state = raw.get("state")
    if state == "ran":
        if "passed" not in raw:
            raise ValueError(
                f"{field}: state='ran' without a 'passed' field. A check that ran "
                "must say whether it passed; absence is not success."
            )
        passed = raw["passed"]
        if not isinstance(passed, bool):
            raise TypeError(f"{field}: 'passed' must be bool, got {type(passed).__name__}")
        n, n_total = raw.get("n"), raw.get("n_total")
        return Ran(
            passed=passed,
            value=raw.get("value"),
            n=n,
            n_total=n_total,
            detail=raw.get("detail", ""),
        )

    if state == "not_run":
        reason = raw.get("reason", "")
        if not isinstance(reason, str):
            raise TypeError(f"{field}: 'reason' must be str, got {type(reason).__name__}")
        # NotRun's own validation rejects an empty reason; let it raise with its message.
        if "passed" in raw:
            raise ValueError(
                f"{field}: state='not_run' carries a 'passed' field. A check that did "
                "not run has no pass/fail result; carrying one is how a skipped suite "
                "becomes a green one."
            )
        return NotRun(reason=reason)

    raise ValueError(
        f"{field}: tri-state 'state' must be 'ran' or 'not_run', got {state!r}. "
        "There is no third state and no default."
    )


def aggregate(parts: dict[str, TriState] | list[TriState], *, name: str = "aggregate") -> TriState:
    """Combine tri-states. The result is never more confident than its least-informed input.

    - Any input ``NotRun``  -> the aggregate is ``NotRun``, carrying every reason.
    - All ``Ran``           -> ``Ran(passed=all(passed))``.
    - No inputs at all      -> ``NotRun``. An empty gate has not passed; it is empty.

    That last case is the one worth stating: ``all([])`` is ``True`` in Python, so a
    naive implementation reports an aggregate over zero checks as a clean pass.
    """
    items = list(parts.values()) if isinstance(parts, dict) else list(parts)
    labels = list(parts.keys()) if isinstance(parts, dict) else [str(i) for i in range(len(items))]

    if not items:
        return NotRun(reason=f"{name}: no checks were contributed, so nothing was verified")

    not_run = [(lbl, it) for lbl, it in zip(labels, items, strict=True) if isinstance(it, NotRun)]
    if not_run:
        joined = "; ".join(f"{lbl}: {it.reason}" for lbl, it in not_run)
        return NotRun(
            reason=f"{name}: {len(not_run)} of {len(items)} inputs did not run -> {joined}"
        )

    failed = [lbl for lbl, it in zip(labels, items, strict=True) if isinstance(it, Ran) and not it.passed]
    return Ran(
        passed=not failed,
        n=len(items),
        n_total=len(items),
        detail="" if not failed else f"{name}: failing inputs: {', '.join(failed)}",
    )
