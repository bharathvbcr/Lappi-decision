"""The ledger: the decision record.

Two rules from the plan drive this module:

- "Every run writes to the protocol-hashed ledger before it is allowed to finish.
   A run whose row is missing is rerun, not remembered."
- "A run cannot exit 0 until its row is written. A deliberately broken run must
   produce a row that says so."

So the writer is a context manager that writes a row on *every* exit path,
including an exception and including SIGTERM. A run that dies without a row is the
one failure this module is built to make impossible.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from .tristate import NotRun, Ran, TriState, aggregate, parse_tristate

__all__ = [
    "Protocol",
    "LedgerRow",
    "Ledger",
    "RunRecorder",
    "PromotionVerdict",
    "LedgerChainError",
]

RunKind = Literal[
    "teacher", "lr_probe", "cpt", "prune_heal", "ft", "ablation",
    "eval", "calibration", "smoke", "throughput", "resume", "scale",
]
Status = Literal["completed", "killed", "failed"]

_RUN_KINDS: frozenset[str] = frozenset(RunKind.__args__)  # type: ignore[attr-defined]
_STATUSES: frozenset[str] = frozenset(Status.__args__)  # type: ignore[attr-defined]

# The gates named in the plan's shipping decision. Every one must be present in a
# row and must be `ran` for that row to promote anything. A gate that is merely
# absent is treated as not-run, never as satisfied.
REQUIRED_GATES: tuple[str, ...] = (
    "paired_margin_vs_linear",
    "ood_abstain",
    "needle_hunk_recall",
    "permutation_consistency",
    "ece",
)
REQUIRED_CONTROLS: tuple[str, ...] = (
    "shuffled_label",
    "privileged_hunk",
    "degenerate_head",
    "transfer_gate",
)


class LedgerChainError(RuntimeError):
    """The append-only chain does not verify."""


def _canonical(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Protocol:
    """The five components whose hash identifies a comparable run.

    Two rows are comparable only if their protocol hashes match. Three rows that
    differ *only* in `seed` are what a promotion needs.
    """

    data_snapshot_hash: str
    tokenizer_hash: str
    backbone_commit: str
    recipe_hash: str
    seed: int

    def __post_init__(self) -> None:
        for name in ("data_snapshot_hash", "tokenizer_hash", "backbone_commit", "recipe_hash"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"Protocol.{name} must be a non-empty string; got {v!r}. "
                    "An unidentified protocol makes every comparison against this row meaningless."
                )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"Protocol.seed must be int, got {type(self.seed).__name__}")

    def to_json(self) -> dict[str, Any]:
        return {
            "data_snapshot_hash": self.data_snapshot_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "backbone_commit": self.backbone_commit,
            "recipe_hash": self.recipe_hash,
            "seed": self.seed,
        }

    def hash(self) -> str:
        return hashlib.sha256(_canonical(self.to_json()).encode("utf-8")).hexdigest()

    def hash_without_seed(self) -> str:
        """Identifies a seed *family*: the thing three rows must share to promote."""
        body = self.to_json()
        body.pop("seed")
        return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Environment:
    torch: str
    transformers_sha: str
    device: str
    host: str
    # Tri-states, not bools: on a host with no CUDA the dry run does not execute,
    # and "not checked" must not read the same as "confirmed present".
    fla_present: TriState = field(default_factory=lambda: NotRun(reason="fla dry run not executed"))
    causal_conv1d_present: TriState = field(
        default_factory=lambda: NotRun(reason="causal-conv1d dry run not executed")
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "torch": self.torch,
            "transformers_sha": self.transformers_sha,
            "device": self.device,
            "host": self.host,
            "fla_present": self.fla_present.to_json(),
            "causal_conv1d_present": self.causal_conv1d_present.to_json(),
        }

    @classmethod
    def detect(cls, *, transformers_sha: str = "unknown") -> Self:
        try:
            import torch as _torch

            torch_v = _torch.__version__
            if _torch.cuda.is_available():
                device = f"cuda:{_torch.cuda.device_count()}x{_torch.cuda.get_device_name(0)}"
            elif getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            torch_v, device = "not-installed", "cpu"
        return cls(
            torch=torch_v,
            transformers_sha=transformers_sha,
            device=device,
            host=socket.gethostname(),
        )


@dataclass(slots=True)
class LedgerRow:
    row_id: str
    written_at: str
    prev_row_hash: str | None
    protocol: Protocol
    run_kind: str
    status: str
    quick: bool
    quick_reason: str | None
    code_commit: str
    env: Environment
    metrics: dict[str, TriState]
    noul_rate: TriState
    controls: dict[str, TriState]
    gates: dict[str, TriState]
    wall_clock_s: float
    cost_usd: float
    notes: str = ""

    def __post_init__(self) -> None:
        if self.run_kind not in _RUN_KINDS:
            raise ValueError(f"run_kind {self.run_kind!r} not one of {sorted(_RUN_KINDS)}")
        if self.status not in _STATUSES:
            raise ValueError(f"status {self.status!r} not one of {sorted(_STATUSES)}")
        if self.quick and not (self.quick_reason or "").strip():
            raise ValueError(
                "quick=True requires quick_reason. A run excluded from decisions must say "
                "why it is excluded, or the exclusion looks arbitrary and gets argued away."
            )
        if not self.quick and self.quick_reason:
            raise ValueError("quick_reason set on a run not marked quick: state one or neither")
        if self.wall_clock_s < 0 or self.cost_usd < 0:
            raise ValueError("wall_clock_s and cost_usd are measured, non-negative quantities")

    @property
    def protocol_hash(self) -> str:
        return self.protocol.hash()

    def to_json(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "written_at": self.written_at,
            "prev_row_hash": self.prev_row_hash,
            "protocol_hash": self.protocol_hash,
            "protocol": self.protocol.to_json(),
            "run_kind": self.run_kind,
            "status": self.status,
            "quick": self.quick,
            "quick_reason": self.quick_reason,
            "code_commit": self.code_commit,
            "env": self.env.to_json(),
            "metrics": {k: v.to_json() for k, v in sorted(self.metrics.items())},
            "noul_rate": self.noul_rate.to_json(),
            "controls": {k: v.to_json() for k, v in sorted(self.controls.items())},
            "gates": {k: v.to_json() for k, v in sorted(self.gates.items())},
            "wall_clock_s": self.wall_clock_s,
            "cost_usd": self.cost_usd,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        proto = Protocol(**raw["protocol"])
        stated = raw.get("protocol_hash")
        if stated is not None and stated != proto.hash():
            raise LedgerChainError(
                f"row {raw.get('row_id')}: stated protocol_hash {stated} does not match the hash of "
                f"the stated protocol components {proto.hash()}. The row was edited."
            )
        env_raw = dict(raw["env"])
        env = Environment(
            torch=env_raw["torch"],
            transformers_sha=env_raw["transformers_sha"],
            device=env_raw["device"],
            host=env_raw["host"],
            fla_present=parse_tristate(env_raw["fla_present"], field="env.fla_present"),
            causal_conv1d_present=parse_tristate(
                env_raw["causal_conv1d_present"], field="env.causal_conv1d_present"
            ),
        )
        return cls(
            row_id=raw["row_id"],
            written_at=raw["written_at"],
            prev_row_hash=raw["prev_row_hash"],
            protocol=proto,
            run_kind=raw["run_kind"],
            status=raw["status"],
            quick=raw["quick"],
            quick_reason=raw.get("quick_reason"),
            code_commit=raw["code_commit"],
            env=env,
            metrics={k: parse_tristate(v, field=f"metrics.{k}") for k, v in raw["metrics"].items()},
            noul_rate=parse_tristate(raw["noul_rate"], field="noul_rate"),
            controls={k: parse_tristate(v, field=f"controls.{k}") for k, v in raw["controls"].items()},
            gates={k: parse_tristate(v, field=f"gates.{k}") for k, v in raw["gates"].items()},
            wall_clock_s=raw["wall_clock_s"],
            cost_usd=raw["cost_usd"],
            notes=raw.get("notes", ""),
        )


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """Why a set of rows may or may not promote a decision. Refusals are itemized."""

    promoted: bool
    reasons: tuple[str, ...]
    rows: tuple[str, ...]

    def __str__(self) -> str:
        head = "PROMOTE" if self.promoted else "REFUSED"
        body = "\n".join(f"  - {r}" for r in self.reasons)
        return f"{head} ({len(self.rows)} row(s))\n{body}" if body else head


def _states_partial_coverage(result: Ran) -> bool:
    """True when a result states its coverage and that coverage is short of the population.

    Unstated coverage (``n is None``) is deliberately **not** treated as partial. Many
    gates are a single observation with no population to sample from, and refusing
    those would make promotion unreachable rather than honest.

    ``Ran.is_complete_coverage`` answers False for unstated coverage, which is the
    right answer to "did this see everything?" and the wrong condition to refuse on —
    so the two questions are asked separately here.
    """
    return result.n is not None and not result.is_complete_coverage


def _git_commit(repo: Path) -> str:
    """HEAD, with a -dirty suffix when the tree is not clean.

    A number produced from an uncommitted tree cannot be reproduced from the
    commit it claims, so the row says so rather than implying reproducibility.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (subprocess.SubprocessError, OSError) as exc:
        return f"unknown({type(exc).__name__})"


class Ledger:
    """Append-only JSONL, with a hash chain over the exact bytes of each line."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- reading ---------------------------------------------------------

    def raw_lines(self) -> list[bytes]:
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        if not data:
            return []
        return [ln for ln in data.split(b"\n") if ln.strip()]

    def rows(self) -> list[LedgerRow]:
        return [LedgerRow.from_json(json.loads(ln.decode("utf-8"))) for ln in self.raw_lines()]

    def last_line_hash(self) -> str | None:
        lines = self.raw_lines()
        return hashlib.sha256(lines[-1]).hexdigest() if lines else None

    def verify_chain(self) -> None:
        """Raise ``LedgerChainError`` naming the first break, or return silently."""
        prev_hash: str | None = None
        for i, line in enumerate(self.raw_lines()):
            try:
                obj = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LedgerChainError(f"line {i + 1}: not valid JSON: {exc}") from exc
            stated = obj.get("prev_row_hash")
            if stated != prev_hash:
                raise LedgerChainError(
                    f"line {i + 1} (row {obj.get('row_id')}): prev_row_hash is {stated!r} but the "
                    f"previous line hashes to {prev_hash!r}. History was edited, reordered, or truncated."
                )
            LedgerRow.from_json(obj)  # re-validates the protocol hash and every tri-state
            prev_hash = hashlib.sha256(line).hexdigest()

    # -- writing ---------------------------------------------------------

    def append(self, row: LedgerRow) -> LedgerRow:
        if any(r.row_id == row.row_id for r in self.rows()):
            raise ValueError(f"row_id {row.row_id} already present: the ledger is append-only")
        row.prev_row_hash = self.last_line_hash()
        line = _canonical(row.to_json()).encode("utf-8")
        # O_APPEND so concurrent writers cannot interleave a partial line, and
        # fsync so a row survives the crash that a killed run is recording.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        return row

    # -- promotion -------------------------------------------------------

    def promotion_verdict(self, seed_family: str) -> PromotionVerdict:
        """May the rows sharing this seed family promote a decision?

        All six conditions from docs/ledger-schema.md, each refusal itemized.

        Condition 4 is the one that matters most: a `not_run` gate BLOCKS promotion.
        It does not pass it, and it does not quietly drop out of the conjunction.

        Condition 6 is coverage. A gate that states `n`/`n_total` and saw fewer than
        all eligible items has not established itself over the population, so it
        refuses rather than promoting on a capped sample. Coverage is rendered on
        every itemized line, per the schema's Coverage rule.
        """
        candidates = [r for r in self.rows() if r.protocol.hash_without_seed() == seed_family]
        ids = tuple(r.row_id for r in candidates)
        reasons: list[str] = []

        if not candidates:
            return PromotionVerdict(False, (f"no rows for seed family {seed_family}",), ())

        for r in candidates:
            if r.status != "completed":
                reasons.append(f"{r.row_id}: status is {r.status!r}, not 'completed'")
            if r.quick:
                reasons.append(f"{r.row_id}: marked quick ({r.quick_reason}); quick runs cannot promote")

        seeds = {r.protocol.seed for r in candidates}
        if len(seeds) < 3:
            reasons.append(
                f"{len(seeds)} distinct seed(s) {sorted(seeds)}; promotion needs three rows "
                "differing only in seed"
            )

        for r in candidates:
            for kind, required, recorded in (
                ("gate", REQUIRED_GATES, r.gates),
                ("control", REQUIRED_CONTROLS, r.controls),
            ):
                for name in required:
                    t = recorded.get(name)
                    where = f"{r.row_id}: {kind} {name!r}"
                    if t is None:
                        reasons.append(f"{where} absent; an absent {kind} is not a passed one")
                    elif isinstance(t, NotRun):
                        reasons.append(f"{where} did not run ({t.reason}); this blocks promotion")
                    elif not t.passed:
                        reasons.append(f"{where} ran and FAILED [{t.coverage_str()}]")
                    elif _states_partial_coverage(t):
                        # Condition 6. Without this a gate measured on 1 of 1000 eligible
                        # items promoted exactly like one measured on 1000 of 1000: the
                        # verdict never looked at coverage at all.
                        reasons.append(
                            f"{where} passed on only {t.coverage_str()} of the eligible "
                            "population; a capped sample is not complete coverage "
                            "and does not promote"
                        )

        if reasons:
            return PromotionVerdict(False, tuple(reasons), ids)
        # Render the weakest coverage any gate or control actually achieved, so a
        # PROMOTE is never read as "complete coverage" without saying so.
        stated = [
            t
            for r in candidates
            for t in (*r.gates.values(), *r.controls.values())
            if isinstance(t, Ran) and t.n is not None
        ]
        coverage = (
            min(stated, key=lambda t: (t.n or 0) / (t.n_total or 1)).coverage_str()
            if stated
            else "coverage unstated"
        )
        return PromotionVerdict(
            True,
            (
                f"{len(candidates)} completed rows, seeds {sorted(seeds)}, every gate and "
                f"control ran and passed at complete coverage (weakest stated: {coverage})",
            ),
            ids,
        )


class RunRecorder:
    """Context manager guaranteeing a row on every exit path.

    A run that raises writes ``status='failed'``. A run killed by SIGTERM/SIGINT
    writes ``status='killed'``. A run that completes writes ``status='completed'``.
    There is no path out of the ``with`` block that leaves the ledger silent —
    which is the point: "a run whose row is missing is rerun, not remembered",
    and an unwritten row is indistinguishable from a run that never happened.
    """

    def __init__(
        self,
        ledger: Ledger,
        *,
        protocol: Protocol,
        run_kind: RunKind,
        repo: str | os.PathLike[str],
        env: Environment | None = None,
        quick: bool = False,
        quick_reason: str | None = None,
        cost_usd_per_hour: float = 0.0,
        notes: str = "",
    ) -> None:
        if run_kind not in _RUN_KINDS:
            raise ValueError(f"run_kind {run_kind!r} not one of {sorted(_RUN_KINDS)}")
        self.ledger = ledger
        self.protocol = protocol
        self.run_kind = run_kind
        self.repo = Path(repo)
        self.env = env or Environment.detect()
        self.quick = quick
        self.quick_reason = quick_reason
        self.cost_usd_per_hour = cost_usd_per_hour
        self.notes = notes

        self.metrics: dict[str, TriState] = {}
        self.controls: dict[str, TriState] = {}
        self.gates: dict[str, TriState] = {}
        self.noul_rate: TriState = NotRun(reason="noul rate not computed by this run")

        self._t0 = 0.0
        self._killed = False
        self._prev_handlers: dict[int, Any] = {}
        self.row: LedgerRow | None = None

    # -- recording -------------------------------------------------------

    def metric(self, name: str, value: TriState) -> None:
        self.metrics[name] = value

    def control(self, name: str, value: TriState) -> None:
        self.controls[name] = value

    def gate(self, name: str, value: TriState) -> None:
        self.gates[name] = value

    def _fill_unreported(self) -> None:
        """Anything the run never reported is `not_run`, explicitly.

        Without this, a gate the run forgot is simply absent, and a reader has to
        infer whether it was skipped or passed. Absence is made explicit here so
        promotion sees a `NotRun` and refuses, rather than seeing nothing.
        """
        for gate in REQUIRED_GATES:
            self.gates.setdefault(
                gate, NotRun(reason=f"gate {gate!r} was never evaluated by this run")
            )
        for ctl in REQUIRED_CONTROLS:
            self.controls.setdefault(
                ctl, NotRun(reason=f"control {ctl!r} was never evaluated by this run")
            )

    # -- lifecycle -------------------------------------------------------

    def _on_signal(self, signum: int, frame: types.FrameType | None) -> None:
        self._killed = True
        self._finish("killed", f"received {signal.Signals(signum).name}")
        # Restore and re-raise so the process still dies the way the caller expects.
        for sig, handler in self._prev_handlers.items():
            signal.signal(sig, handler)
        os.kill(os.getpid(), signum)

    def __enter__(self) -> Self:
        self._t0 = time.monotonic()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
            except ValueError:
                # Not on the main thread; the with-block's finally still writes a row.
                pass
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> bool:
        for sig, handler in self._prev_handlers.items():
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass
        if self.row is not None:
            return False  # already written by the signal path
        if exc_type is None:
            self._finish("completed", "")
        elif issubclass(exc_type, KeyboardInterrupt):
            self._finish("killed", "KeyboardInterrupt")
        else:
            self._finish("failed", f"{exc_type.__name__}: {exc}")
        return False  # never suppress

    def _finish(self, status: Status, detail: str) -> None:
        if self.row is not None:
            return
        self._fill_unreported()
        wall = max(0.0, time.monotonic() - self._t0)
        note = self.notes if not detail else (f"{self.notes} | {detail}" if self.notes else detail)
        self.row = self.ledger.append(
            LedgerRow(
                row_id=str(uuid.uuid4()),
                written_at=datetime.now(UTC).isoformat(),
                prev_row_hash=None,  # set by Ledger.append
                protocol=self.protocol,
                run_kind=self.run_kind,
                status=status,
                quick=self.quick,
                quick_reason=self.quick_reason,
                code_commit=_git_commit(self.repo),
                env=self.env,
                metrics=self.metrics,
                noul_rate=self.noul_rate,
                controls=self.controls,
                gates=self.gates,
                wall_clock_s=wall,
                cost_usd=self.cost_usd_per_hour * wall / 3600.0,
                notes=note,
            )
        )
