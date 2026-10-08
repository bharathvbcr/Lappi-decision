"""Run the GH200 campaign's phases in order, capped, resumable, and stopping at the go/no-go.

The plan (``docs/train-plan-2026-09-28.md``, "The GH200 campaign", item 6) asks for one
script that runs phases 0-6 on the rented box, checkpoints every <= 2 h, resumes from the
last checkpoint, writes every ledger row, gets ledger + checkpoints home after each phase,
auto-terminates the instance at the cap, and prints a cost estimate before it starts.
Phase 3 is a go/no-go: *"If the 2B does not beat its control here, stop and report -- do
not spend the rest."* This is that script. It trains nothing itself: every unit of work is
an explicit argv (``tools/real_ft_run.py``, ``tools/ft_linear_control.py``, ...) from a
JSON config, so what ran is always a command someone can read and rerun.

## Where it runs, and what it refuses

It runs **on the box** (Lambda 1xGH200), because the cap and the termination must hold
when the Mac is asleep. Before anything starts it refuses:

* a config with no ``terminate_command`` -- rule 4's auto-terminate is not optional;
* a cuda box priced at zero, or with no ``usd_per_hour_source`` saying where today's rate
  was read (the rate is read from the price page at launch, never carried in a template);
* a projected cost at or above $20 with no ``approved_by`` (rule 4). The projection is the
  campaign cap **plus** the pull grace window below, because the box bills while it waits;
* a campaign cap above :data:`CAMPAIGN_MAX_HOURS`, or a phase cap above ``MAX_CAP_S``;
* a phase longer than :data:`MAX_CHECKPOINT_INTERVAL_H` whose units do not checkpoint,
  two units sharing a checkpoint dir, or a real-backbone ``real_ft_run.py`` unit that does
  not state its ``--optimizer``;
* a ``real_ft_run.py`` unit without ``--wall-clock-cap-s``, or with one above its phase cap
  (the trainer's 1800 s default would end a campaign phase at 30 minutes);
* resuming a state file written under a different config (its digest is recorded).

## Getting the data home: the Mac pulls, and termination waits for it

The box cannot push to a Mac behind NAT, so the **Mac pulls** (``tools/sync_box.sh pull``,
looped by ``tools/campaign_pull_loop.sh``). That makes "sync, then terminate" unsafe as a
box-side sequence: the box cannot know the Mac copied anything, and terminating would
destroy the only copy of the checkpoints. So:

1. After every phase the driver writes an **immutable snapshot** ``sync/phase-<n>/`` --
   ledger files copied, checkpoint files hard-linked -- and then ``sync/phase-<n>.done``, a
   manifest carrying the sha256 of every file in it. It does not wait; the next phase starts.
2. The Mac pulls each ``.done`` it has not acknowledged, verifies every sha256 against the
   manifest, and writes ``sync/pulled-<n>.ok`` back, naming the manifest's own sha256.
3. **Termination is gated on the final ``pulled-<n>.ok``** -- at the end, at a stop, on a
   failure, and at the cap (by the guard). Without it the driver waits up to
   ``pull_grace_minutes`` (default 45). At expiry it terminates only if a configured
   ``durable_copy_dir`` holds a verified copy: every file's sha256 matches **and** the
   directory is on a different device from the box's own disk (a directory on the instance
   disk dies with the instance, whatever it is called). Otherwise it **keeps the box alive**
   and exits loud (exit 4) with the hourly cost. ``--finalize`` re-runs the wait later.

## The caps

Each unit runs in its own process group with a timeout of the smaller of what is left of
its phase and of the campaign; at expiry the group gets SIGTERM, then SIGKILL after
``kill_grace_s``. Elapsed time is wall time since the phase (campaign) first started,
including any time the driver was dead -- the instance was billing then too. A capped or
failed unit stops the campaign.

A detached **guard** holds the campaign deadline independently of the driver. At the
deadline it stops the driver and any running unit (so the ledger stops moving), writes the
final snapshot and runs the same gated termination. A lock file makes sure only one of the
guard and the driver finalizes.

## The box guard: the box bills from boot, not from the campaign's first unit

HANDOFF/next-training-plan-2026-10-06.md section 7 precaution 4. v5's queue enforced the
GPU-step budget and never the box's: the box billed while lanes waited for markers
(GAP-V5-2GPU-BOX-CEILING-NOT-ENFORCED-2026-10-03). So a config for a rented box carries
``launch.approved`` -- ``box_usd``, ``wall_clock_cap_s`` and ``by`` -- and a ``cuda`` config
without it is refused. The box's age is read from ``/proc/uptime`` (:data:`UPTIME_PATH`), and
its spend is ``usd_per_hour`` x hours since boot. The box may run until the smaller of
``box_usd / usd_per_hour`` and ``wall_clock_cap_s`` since boot; GPU work stops
``pull_grace_minutes`` before that, so the Mac's pull still fits inside the ceiling.

Every wait is bounded by it: a unit's timeout, a phase's marker wait, the wait for the Mac's
acknowledgement, and the guard's sleep. At the ceiling the driver (or the guard, whichever
gets there first) stops the unit's process group, writes ``<state_dir>/BOX_CEILING`` with the
numbers, and runs the same gated finalize -- whose ``terminate_command`` is the human's (no
agent holds a cloud credential). An unreadable uptime is never "within budget": it refuses a
start and stops a running campaign (exit 5).

## Placeholders, pins, quick units and marker-gated phases

* ``placeholders``: ``{name}`` anywhere in the config is replaced by its value. A value only the
  box knows (a path) ships as ``null``, and an unfilled or undeclared one is refused.
  ``{checkpoint_every}`` stays reserved for the measured cadence below.
* ``pins``: ``[{path, sha256}]``; every file is hashed before anything starts, and one that is
  missing or differs refuses the start.
* ``{ft_row_id:<unit>}`` in an argv is the id of the one ``ft`` row an earlier unit wrote,
  filled when the reading unit starts (a reader names the run it reads, by construction).
* ``carry``: directories whose files travel in every snapshot (verdicts, footprints and
  readings that are not ledger rows or checkpoints).
* a unit with ``"quick": true`` fails if any row it wrote is not quick (rule 8).
* a phase with ``requires`` runs only once every marker exists (and a JSON marker's
  ``json_key`` reads ``equals``). The wait is bounded by ``wait_max_s``, the campaign cap and
  the box; a marker that never arrives, or reads anything else, ends the campaign with the
  phase ``not_run``.

## Resume and the go/no-go

State lives in ``<state_dir>/campaign_state.json``, rewritten atomically. Completed units
are skipped; an interrupted unit is rerun, with ``--resume-from <newest>`` when its
checkpoint dir holds one. ``{checkpoint_every}`` is replaced by the optimizer steps the
throughput phase measured in ``every_hours``. A phase with a ``gate`` requires it ``Ran``
and passed on ``min_seeds`` seeds in rows written by ``gate.tool``; ``not_run`` is a stop,
and so is any such row that is ``quick`` (rule 8: a quick row is excluded from decisions,
and GO is one).

Usage:
    python tools/campaign_driver.py --config campaign.json            # start or resume
    python tools/campaign_driver.py --config campaign.json --plan     # validate + cost only
    python tools/campaign_driver.py --config campaign.json --finalize # re-run the gated end
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from qd_train.ledger import Ledger, LedgerRow  # noqa: E402
from qd_train.run_control import (  # noqa: E402
    APPROVAL_FREE_USD,
    CostEstimate,
    WallClockCap,
)
from qd_train.tristate import Ran  # noqa: E402

#: The user's decision of 2026-09-28: one GH200 "for up to 2-3 days". The campaign cap may
#: be set below this, never above it. Rule 2: this is read-only to an agent.
CAMPAIGN_MAX_HOURS: Final[float] = 72.0

#: The plan's "checkpoints every <= 2 h". A phase longer than this must checkpoint.
MAX_CHECKPOINT_INTERVAL_H: Final[float] = 2.0

#: The lead's decision of 2026-09-29: wait this long for the Mac's final acknowledgement.
DEFAULT_PULL_GRACE_MIN: Final[float] = 45.0
MAX_PULL_GRACE_MIN: Final[float] = 240.0

STATE_NAME: Final[str] = "campaign_state.json"
LOCK_NAME: Final[str] = "finalize.lock"
BOX_CEILING_NAME: Final[str] = "BOX_CEILING"
TERMINATE_TIMEOUT_S: Final[float] = 300.0
#: How long a SIGKILLed unit is waited on to be reaped before the driver reports it and moves
#: on. A process stuck in uninterruptible sleep must not become an unbounded wait.
REAP_TIMEOUT_S: Final[float] = 60.0

#: Where the box's age is read. Module-level so a test can point it at a fake file; the guard
#: is handed the same path on its argv.
UPTIME_PATH: Path = Path("/proc/uptime")

#: Placeholders the driver fills itself at run time; a config may not declare them.
RESERVED_PLACEHOLDERS: Final[frozenset[str]] = frozenset({"checkpoint_every"})
PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"\{([a-z][a-z0-9_]*)\}")
#: Filled at run time from the state: the id of the one ``ft`` row an EARLIER unit wrote. A
#: reader unit names the run it reads without anyone copying a row id by hand.
FT_ROW_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\{ft_row_id:([A-Za-z0-9_.-]+)\}")

#: Exit codes. 0 every phase ran; 2 stopped at the go/no-go (or a gated phase's markers did
#: not hold); 1 anything else stopped it; 3 the terminate command failed; 4 the box was
#: deliberately KEPT ALIVE because nothing proved the data was safe off it; 5 the box guard
#: stopped the campaign at the box's ceiling. 3, 4 and 5 are the loud ones.
EXIT_DONE, EXIT_STOPPED, EXIT_FAILED, EXIT_TERMINATE_FAILED, EXIT_KEPT_ALIVE = 0, 2, 1, 3, 4
EXIT_BOX_CEILING: Final[int] = 5

#: The clock and the sleep every wait uses. Module-level so a test can drive them with a fake
#: clock; nothing else in the driver calls time.time or time.sleep for a deadline.
_now: Callable[[], float] = time.time
_sleep: Callable[[float], None] = time.sleep


class ConfigRefused(SystemExit):
    """The config cannot be run as written. Nothing was launched."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"campaign config refused: {reason}")


class BoxRefused(SystemExit):
    """The box guard could not establish the box is within budget. Nothing was launched."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"box guard refused to start: {reason}")


# --- the box: its age from uptime, its spend from its rate ------------------------------


class BoxClockUnreadable(Exception):
    """The box's uptime could not be read. Never treated as within budget."""


def read_uptime_s(path: Path | None = None) -> float:
    """Seconds since the box booted: the first field of ``/proc/uptime``."""
    src = UPTIME_PATH if path is None else path
    try:
        text = src.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise BoxClockUnreadable(f"{src}: {type(exc).__name__}: {exc}") from exc
    try:
        value = float(text.split()[0])
    except (IndexError, ValueError) as exc:
        raise BoxClockUnreadable(f"{src}: no uptime in {text[:40]!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise BoxClockUnreadable(f"{src}: uptime {value!r} is not a non-negative number")
    return value


@dataclass(frozen=True)
class BoxBudget:
    """What the human approved for the box, against its rate, from boot."""

    usd: float
    usd_per_hour: float
    wall_clock_cap_s: float
    approved_by: str
    pull_grace_s: float

    @property
    def limit_s(self) -> float:
        """Seconds since boot the box may bill: the smaller of the budget and the cap."""
        return min(self.usd / self.usd_per_hour * 3600.0, self.wall_clock_cap_s)

    @property
    def work_limit_s(self) -> float:
        """Seconds since boot by which GPU work stops, leaving the pull window inside."""
        return self.limit_s - self.pull_grace_s


@dataclass(frozen=True)
class BoxReading:
    budget: BoxBudget
    uptime_s: float | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None and self.uptime_s is not None

    @property
    def spend_usd(self) -> float | None:
        if not self.ok or self.uptime_s is None:
            return None
        return self.budget.usd_per_hour * self.uptime_s / 3600.0

    @property
    def left_s(self) -> float:
        """Until the ceiling itself. An unreadable clock has nothing left."""
        if not self.ok or self.uptime_s is None:
            return 0.0
        return max(self.budget.limit_s - self.uptime_s, 0.0)

    @property
    def work_left_s(self) -> float:
        """Until GPU work must stop. An unreadable clock has nothing left."""
        if not self.ok or self.uptime_s is None:
            return 0.0
        return max(self.budget.work_limit_s - self.uptime_s, 0.0)

    @property
    def at_ceiling(self) -> bool:
        return self.work_left_s <= 0.0

    def line(self) -> str:
        b = self.budget
        if not self.ok:
            return f"box clock UNREADABLE ({self.error}); treated as at the ceiling"
        return (f"box up {self.uptime_s:.0f}s, spend ${self.spend_usd:.2f} of "
                f"${b.usd:.2f} at ${b.usd_per_hour:.2f}/h (cap {b.wall_clock_cap_s:.0f}s); "
                f"GPU work has {self.work_left_s:.0f}s left")

    def numbers(self, reason: str, who: str) -> dict[str, Any]:
        b = self.budget
        return {
            "reason": reason, "who": who, "at": time.time(),
            "uptime_path": str(UPTIME_PATH), "uptime_s": self.uptime_s,
            "uptime_error": self.error, "usd_per_hour": b.usd_per_hour,
            "spend_usd": self.spend_usd, "box_usd": b.usd,
            "wall_clock_cap_s": b.wall_clock_cap_s, "limit_s": b.limit_s,
            "pull_grace_s": b.pull_grace_s, "work_limit_s": b.work_limit_s,
            "approved_by": b.approved_by,
        }


def box_reading(budget: BoxBudget) -> BoxReading:
    try:
        return BoxReading(budget, read_uptime_s(), None)
    except BoxClockUnreadable as exc:
        return BoxReading(budget, None, str(exc))


# --- config ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSpec:
    dir: Path
    every_hours: float
    throughput_from: str


@dataclass(frozen=True)
class Unit:
    name: str
    argv: tuple[str, ...]
    #: Rows the unit must append to the campaign ledger. A run that did not write its row is
    #: rerun, not remembered (CLAUDE.md "Layout"), so fewer is a failure.
    expect_rows: int
    checkpoint: CheckpointSpec | None
    #: Exit codes that still count as a finished unit. ``tools/real_ft_run.py`` exits 1 when
    #: a *claim* it tested fails (a stand-in that does not reach its floor), having written
    #: every row -- information, not a crash. A code listed here is accepted only if every
    #: row the unit wrote has status ``completed``; a crash writes a ``failed`` row or none.
    accept_returncodes: tuple[int, ...] = (0,)
    #: Rule 8 stated up front: every row this unit writes must be quick, or the unit failed.
    quick: bool = False


@dataclass(frozen=True)
class Marker:
    path: Path
    #: For a JSON marker: the key whose value must equal ``equals``. None: existence only.
    json_key: str | None
    equals: str | None


@dataclass(frozen=True)
class Requires:
    markers: tuple[Marker, ...]
    wait_max_s: float
    poll_s: float


@dataclass(frozen=True)
class Pin:
    path: Path
    sha256: str


@dataclass(frozen=True)
class GateSpec:
    ledger: Path
    name: str
    tool: str
    min_seeds: int


@dataclass(frozen=True)
class Phase:
    name: str
    cap: WallClockCap
    units: tuple[Unit, ...]
    gate: GateSpec | None
    requires: Requires | None = None


@dataclass(frozen=True)
class Config:
    name: str
    state_dir: Path
    ledger: Path
    device: str
    instance: str
    usd_per_hour: float
    usd_per_hour_source: str
    n_gpus: int
    campaign_cap_hours: float
    terminate_command: tuple[str, ...]
    pull_grace_minutes: float
    pull_poll_s: float
    durable_copy_dir: Path | None
    approved_by: str
    kill_grace_s: float
    phases: tuple[Phase, ...]
    digest: str
    #: None only off a rented box (cpu/mps without ``launch``); a cuda config always has one.
    box: BoxBudget | None = None
    pins: tuple[Pin, ...] = ()
    #: Directories whose files travel in every snapshot beside the checkpoints: the small
    #: outputs (verdicts, footprints, readings) that die with the box otherwise.
    carry_dirs: tuple[Path, ...] = ()

    @property
    def sync_dir(self) -> Path:
        return self.state_dir / "sync"

    @property
    def ledgers(self) -> tuple[Path, ...]:
        """Every ledger the snapshot carries: the campaign's and each gate's."""
        seen: dict[str, Path] = {str(self.ledger.resolve()): self.ledger}
        for ph in self.phases:
            if ph.gate is not None:
                seen.setdefault(str(ph.gate.ledger.resolve()), ph.gate.ledger)
        return tuple(seen.values())

    @property
    def checkpoint_dirs(self) -> tuple[Path, ...]:
        return tuple(u.checkpoint.dir for ph in self.phases for u in ph.units
                     if u.checkpoint is not None)


def _argv(raw: object, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(a, str) for a in raw):
        raise ConfigRefused(f"{where} must be a non-empty list of strings, got {raw!r}")
    return tuple(raw)


def _num(raw: dict[str, Any], key: str, where: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ConfigRefused(f"{where}.{key} must be a finite number, got {value!r}")
    return float(value)


def _check_unit_cap(argv: tuple[str, ...], *, phase_cap: WallClockCap, where: str) -> None:
    """A ``real_ft_run.py`` unit states its own cap, and it fits inside its phase's.

    The trainer's default cap is 30 minutes, which is right for a Mac smoke and silently
    wrong for a campaign phase: a unit that leaves it out trains for 30 minutes, records
    ``wall_clock_cap``, and every row it writes is quick. So the cap is required, and one
    above the phase cap is refused -- the phase's process-group timeout would kill the unit
    first, and the trainer's own cap would never be what ended it. The cap is per arm per
    seed; a unit running several seeds or both arms can still outlast the phase, which the
    phase timeout bounds.
    """
    flag = "--wall-clock-cap-s"
    if flag not in argv:
        raise ConfigRefused(
            f"{where}: real_ft_run.py without {flag}. Its default is 1800 s, which would stop "
            f"a campaign phase at 30 minutes; state the cap, at most the phase's "
            f"{phase_cap.cap_s:g} s"
        )
    i = argv.index(flag)
    try:
        unit_cap = float(argv[i + 1])
    except (IndexError, ValueError) as exc:
        raise ConfigRefused(f"{where}: {flag} needs a number of seconds") from exc
    if not (math.isfinite(unit_cap) and 0 < unit_cap <= phase_cap.cap_s):
        raise ConfigRefused(
            f"{where}: {flag} {unit_cap:g} s is not in (0, {phase_cap.cap_s:g}], the phase "
            "cap; the phase would kill the unit before its own cap could end it cleanly"
        )


def _phase(p: object, i: int, names: set[str]) -> Phase:
    where = f"phases[{i}]"
    if not isinstance(p, dict) or not isinstance(p.get("name"), str):
        raise ConfigRefused(f"{where} needs a name")
    if p["name"] in names:
        raise ConfigRefused(f"{where}: duplicate phase name {p['name']!r}")
    try:
        cap = WallClockCap(_num(p, "cap_hours", where) * 3600.0)
    except ValueError as exc:
        raise ConfigRefused(f"{where}: {exc}") from exc
    units: list[Unit] = []
    unit_names: set[str] = set()
    for j, u in enumerate(p.get("units") or []):
        uw = f"{where}.units[{j}]"
        if not isinstance(u, dict) or not isinstance(u.get("name"), str):
            raise ConfigRefused(f"{uw} needs a name")
        if u["name"] in unit_names:
            raise ConfigRefused(f"{uw}: duplicate unit name {u['name']!r}")
        unit_names.add(u["name"])
        argv = _argv(u.get("argv"), f"{uw}.argv")
        expect = u.get("expect_rows", 1)
        if not isinstance(expect, int) or isinstance(expect, bool) or expect < 0:
            raise ConfigRefused(f"{uw}.expect_rows must be a non-negative integer")
        ck = None
        if u.get("checkpoint") is not None:
            c = u["checkpoint"]
            every = _num(c, "every_hours", f"{uw}.checkpoint")
            if not 0 < every <= MAX_CHECKPOINT_INTERVAL_H:
                raise ConfigRefused(f"{uw}.checkpoint.every_hours {every:g} must be in "
                                    f"(0, {MAX_CHECKPOINT_INTERVAL_H:g}]")
            src = c.get("throughput_from")
            if src not in names:
                raise ConfigRefused(
                    f"{uw}.checkpoint.throughput_from must name an EARLIER phase, got {src!r}"
                )
            if "{checkpoint_every}" not in argv or "--checkpoint-dir" not in argv:
                raise ConfigRefused(f"{uw} declares a checkpoint but its argv does not pass "
                                    "--checkpoint-dir and --checkpoint-every {checkpoint_every}")
            ck = CheckpointSpec(Path(c["dir"]), every, src)
        accept = u.get("accept_returncodes", [0])
        if (not isinstance(accept, list) or 0 not in accept
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in accept)):
            raise ConfigRefused(f"{uw}.accept_returncodes must be a list of ints including 0")
        if accept != [0] and expect < 1:
            raise ConfigRefused(f"{uw} accepts exit codes {accept} but expects no ledger row, "
                                "so a crash could not be told from a failed claim")
        # The lead's pre-rental finding (2026-09-29): optim.py refuses the bf16 recipe on a
        # long schedule, and the recipe changes what a row means. A real-backbone run states
        # its optimizer rather than inheriting a default nobody chose for it.
        if (any(a.endswith("real_ft_run.py") for a in argv)
                and "--real-backbone" in argv and "--optimizer" not in argv):
            raise ConfigRefused(f"{uw} ({u['name']}): real_ft_run.py --real-backbone without an "
                                "explicit --optimizer. State it per phase (master for long "
                                "schedules)")
        if any(a.endswith("real_ft_run.py") for a in argv):
            _check_unit_cap(argv, phase_cap=cap, where=f"{uw} ({u['name']})")
        quick = u.get("quick", False)
        if not isinstance(quick, bool):
            raise ConfigRefused(f"{uw}.quick must be true or false, got {quick!r}")
        units.append(Unit(u["name"], argv, expect, ck, tuple(accept), quick))
    if not units:
        raise ConfigRefused(f"{where} has no units")
    if cap.cap_hours > MAX_CHECKPOINT_INTERVAL_H and any(u.checkpoint is None for u in units):
        raise ConfigRefused(
            f"{where} ({p['name']}) is capped at {cap.cap_hours:g} h but a unit does not "
            f"checkpoint; the plan checkpoints every <= {MAX_CHECKPOINT_INTERVAL_H:g} h. "
            "Split it into units under the interval, or give each unit a checkpoint"
        )
    gate = None
    if p.get("gate") is not None:
        g = p["gate"]
        seeds = g.get("min_seeds")
        if not isinstance(seeds, int) or isinstance(seeds, bool) or seeds < 1:
            raise ConfigRefused(f"{where}.gate.min_seeds must be a positive integer")
        if not all(isinstance(g.get(k), str) and g[k] for k in ("ledger", "name", "tool")):
            raise ConfigRefused(f"{where}.gate needs ledger, name and tool")
        gate = GateSpec(Path(g["ledger"]), g["name"], g["tool"], seeds)
    requires = _requires(p.get("requires"), where) if p.get("requires") is not None else None
    return Phase(p["name"], cap, tuple(units), gate, requires)


def _requires(raw: object, where: str) -> Requires:
    rw = f"{where}.requires"
    if not isinstance(raw, dict):
        raise ConfigRefused(f"{rw} must be an object with markers, wait_max_s and poll_s")
    wait = _num(raw, "wait_max_s", rw)
    poll = _num(raw, "poll_s", rw)
    if not 0 < wait <= CAMPAIGN_MAX_HOURS * 3600.0 or not 0 < poll <= 600:
        raise ConfigRefused(f"{rw}: wait_max_s {wait:g} must be in (0, "
                            f"{CAMPAIGN_MAX_HOURS * 3600:g}] and poll_s {poll:g} in (0, 600]")
    raw_markers = raw.get("markers")
    if not isinstance(raw_markers, list) or not raw_markers:
        raise ConfigRefused(f"{rw}.markers must be a non-empty list")
    markers = []
    for k, m in enumerate(raw_markers):
        mw = f"{rw}.markers[{k}]"
        if not isinstance(m, dict) or not isinstance(m.get("path"), str) or not m["path"]:
            raise ConfigRefused(f"{mw} needs a path")
        key, equals = m.get("json_key"), m.get("equals")
        if (key is None) != (equals is None):
            raise ConfigRefused(f"{mw}: json_key and equals come together or not at all")
        if key is not None and not (isinstance(key, str) and key and isinstance(equals, str)):
            raise ConfigRefused(f"{mw}: json_key and equals must be non-empty strings")
        markers.append(Marker(Path(m["path"]), key, equals))
    return Requires(tuple(markers), wait, poll)


def _fill_placeholders(node: object, values: dict[str, str], where: str) -> object:
    """Replace ``{name}`` throughout the config. Keys starting ``_`` are prose, left alone."""
    if isinstance(node, str):
        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name in RESERVED_PLACEHOLDERS:
                return m.group(0)
            if name not in values:
                raise ConfigRefused(f"{where} names {{{name}}}, which no placeholder declares")
            return values[name]
        return PLACEHOLDER_RE.sub(sub, node)
    if isinstance(node, list):
        return [_fill_placeholders(x, values, f"{where}[{i}]") for i, x in enumerate(node)]
    if isinstance(node, dict):
        return {k: (v if k.startswith("_") else _fill_placeholders(v, values, f"{where}.{k}"))
                for k, v in node.items()}
    return node


def _placeholders(raw: dict[str, Any]) -> dict[str, str]:
    ph = raw.get("placeholders", {})
    if not isinstance(ph, dict):
        raise ConfigRefused("placeholders must be an object of name -> value")
    for name in ph:
        if not isinstance(name, str) or not PLACEHOLDER_RE.fullmatch("{" + name + "}"):
            raise ConfigRefused(f"placeholder name {name!r} must match [a-z][a-z0-9_]*")
        if name in RESERVED_PLACEHOLDERS:
            raise ConfigRefused(f"placeholder {name!r} is reserved: the driver fills it")
    unfilled = sorted(k for k, v in ph.items() if not isinstance(v, str) or not v.strip())
    if unfilled:
        raise ConfigRefused(
            f"placeholder(s) {unfilled} are not filled. They are values only the box knows; "
            "fill each in this config on the box, then run --plan"
        )
    filled = {k: v for k, v in ph.items() if isinstance(v, str)}
    nested = sorted(k for k, v in filled.items() if PLACEHOLDER_RE.search(v))
    if nested:
        raise ConfigRefused(f"placeholder(s) {nested} hold another placeholder; give the value")
    return filled


def _pins(raw: object) -> tuple[Pin, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigRefused("pins must be a list of {path, sha256}")
    pins = []
    for i, p in enumerate(raw):
        if (not isinstance(p, dict) or not isinstance(p.get("path"), str) or not p["path"]
                or not isinstance(p.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", p["sha256"])):
            raise ConfigRefused(f"pins[{i}] needs a path and a lower-case 64-hex sha256")
        pins.append(Pin(Path(p["path"]), p["sha256"]))
    return tuple(pins)


def _carry(raw: object) -> tuple[Path, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) and x for x in raw):
        raise ConfigRefused("carry must be a list of directory paths")
    return tuple(Path(x) for x in raw)


def _check_ft_row_tokens(cfg: Config) -> None:
    """``{ft_row_id:<unit>}`` names a unit that runs EARLIER, and only one unit has the name."""
    seen: list[str] = []
    every = [u.name for ph in cfg.phases for u in ph.units]
    for ph in cfg.phases:
        for u in ph.units:
            for a in u.argv:
                for name in FT_ROW_TOKEN_RE.findall(a):
                    if name not in seen:
                        raise ConfigRefused(f"unit {u.name}: {{ft_row_id:{name}}} names no "
                                            "earlier unit")
                    if every.count(name) != 1:
                        raise ConfigRefused(f"unit {u.name}: {{ft_row_id:{name}}} is ambiguous; "
                                            "unit names it reads must be unique")
            seen.append(u.name)


def _resolve_ft_rows(argv: list[str], state: dict[str, Any]) -> list[str]:
    if not any(FT_ROW_TOKEN_RE.search(a) for a in argv):
        return argv
    units = {n: u for ph in state.get("phases", {}).values()
             for n, u in ph.get("units", {}).items()}

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        info = units.get(name, {})
        ft = [rid for rid, kind in info.get("row_ids", []) if kind == "ft"]
        if info.get("status") != "done" or len(ft) != 1:
            raise ConfigRefused(
                f"{{ft_row_id:{name}}}: unit {name} is {info.get('status', 'not run')} with "
                f"{len(ft)} ft row(s) recorded; it needs exactly one"
            )
        return str(ft[0])

    return [FT_ROW_TOKEN_RE.sub(sub, a) for a in argv]


def verify_pins(cfg: Config) -> list[str]:
    """Every pinned file, hashed now. Returns the failures; empty means every pin held."""
    bad = []
    for pin in cfg.pins:
        if not pin.path.is_file():
            bad.append(f"{pin.path}: missing or not a file")
            continue
        got = sha256_file(pin.path)
        if got != pin.sha256:
            bad.append(f"{pin.path}: sha256 {got}, pinned {pin.sha256}")
    return bad


def _box_rate(raw: dict[str, Any]) -> float:
    """The box's rate, in the guard's own words: a zero, negative or NaN rate never reaches a
    ceiling, so it is refused before the generic number check could word it otherwise."""
    rate = raw.get("usd_per_hour")
    if (isinstance(rate, bool) or not isinstance(rate, int | float)
            or not math.isfinite(rate) or rate <= 0):
        raise ConfigRefused(
            f"the box rate usd_per_hour {rate!r} must be a finite number above zero: the guard "
            "prices the box's uptime at it, and a zero, negative or NaN rate never reaches a "
            "ceiling"
        )
    return float(rate)


def _box_budget(raw: dict[str, Any], *, device: object, pull_grace_min: float
                ) -> BoxBudget | None:
    """``launch.approved``: refused when missing on a rented box, or not a usable number.

    The rate is checked here with its own words, before the generic number check: a zero,
    negative or NaN rate makes every box look unspent, which is the failure this guards.
    """
    launch = raw.get("launch")
    if launch is None:
        if device == "cuda":
            raise ConfigRefused(
                "no launch.approved.box_usd. A rented box bills its wall clock from boot, "
                "including every wait; the guard needs the human's box ceiling (box_usd, "
                "wall_clock_cap_s, by) before it starts (precaution 4)"
            )
        return None
    if not isinstance(launch, dict) or not isinstance(launch.get("approved"), dict):
        raise ConfigRefused("launch.approved must be an object with box_usd, "
                            "wall_clock_cap_s and by")
    ap = launch["approved"]
    if "box_usd" not in ap:
        raise ConfigRefused("launch.approved.box_usd is missing: no box ceiling, no start")
    rate = _box_rate(raw)
    usd = ap.get("box_usd")
    if (isinstance(usd, bool) or not isinstance(usd, int | float)
            or not math.isfinite(usd) or usd <= 0):
        raise ConfigRefused(f"launch.approved.box_usd {usd!r} must be a finite number above "
                            "zero")
    cap_s = _num(ap, "wall_clock_cap_s", "launch.approved")
    if not 0 < cap_s <= CAMPAIGN_MAX_HOURS * 3600.0:
        raise ConfigRefused(f"launch.approved.wall_clock_cap_s {cap_s:g} must be in (0, "
                            f"{CAMPAIGN_MAX_HOURS * 3600:g}]")
    by = ap.get("by")
    if not isinstance(by, str) or not by.strip():
        raise ConfigRefused("launch.approved.by is empty: name who approved the box ceiling")
    budget = BoxBudget(float(usd), float(rate), cap_s, by, pull_grace_min * 60.0)
    if budget.work_limit_s <= 0:
        raise ConfigRefused(
            f"the box limit of {budget.limit_s:.0f}s since boot leaves nothing after the "
            f"{pull_grace_min:g} min pull grace window it reserves; raise the approval or "
            "shorten the window"
        )
    return budget


def load_config(path: Path) -> Config:
    """Parse and validate. Every refusal names the field; nothing has been launched yet."""
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ConfigRefused("the config must be a JSON object")
    values = _placeholders(raw)
    filled = _fill_placeholders({k: v for k, v in raw.items() if k != "placeholders"},
                                values, "config")
    if not isinstance(filled, dict):
        raise ConfigRefused("the config must be a JSON object")
    raw = filled
    if "terminate_command" not in raw:
        raise ConfigRefused(
            "no terminate_command. Rule 4 requires every long job to auto-terminate, and "
            "the command that ends the instance's billing must come from this config -- "
            "there is no default, because a wrong default is a box that keeps billing"
        )
    if "sync_command" in raw:
        raise ConfigRefused("sync_command is retired: the Mac pulls (tools/campaign_pull_loop.sh)"
                            " and termination waits for its acknowledgement")
    terminate = _argv(raw["terminate_command"], "terminate_command")
    pull_grace = (_num(raw, "pull_grace_minutes", "config") if "pull_grace_minutes" in raw
                  else DEFAULT_PULL_GRACE_MIN)
    if not 0 < pull_grace <= MAX_PULL_GRACE_MIN:
        raise ConfigRefused(f"pull_grace_minutes {pull_grace:g} must be in "
                            f"(0, {MAX_PULL_GRACE_MIN:g}]")
    cap_h = _num(raw, "campaign_cap_hours", "config")
    if not 0 < cap_h <= CAMPAIGN_MAX_HOURS:
        raise ConfigRefused(
            f"campaign_cap_hours {cap_h:g} is outside (0, {CAMPAIGN_MAX_HOURS:g}]: the user "
            "approved one GH200 for up to 2-3 days, and the cap is read-only (rule 2)"
        )
    if raw.get("launch") is not None:
        _box_rate(raw)
    rate = _num(raw, "usd_per_hour", "config")
    n_gpus = raw.get("n_gpus")
    if not isinstance(n_gpus, int) or isinstance(n_gpus, bool) or n_gpus < 0:
        raise ConfigRefused(f"n_gpus must be a non-negative integer, got {n_gpus!r}")
    device = raw.get("device")
    if device not in ("cpu", "mps", "cuda"):
        raise ConfigRefused(f"device must be cpu, mps or cuda, got {device!r}")
    source = raw.get("usd_per_hour_source", "")
    if device == "cuda":
        if not (rate > 0 and n_gpus >= 1):
            raise ConfigRefused(
                f"device cuda at usd_per_hour {rate:g} on {n_gpus} GPU(s): a rented GPU is not "
                "free, and a zero rate or GPU count turns rule 4's approval check off"
            )
        if not isinstance(source, str) or not source.strip():
            raise ConfigRefused(
                "usd_per_hour_source is empty: the rate is read from the provider's price page "
                "at launch (say which page and when), never carried over from a template"
            )
    # After the device's own rate checks, so a free-priced GPU is refused in those words.
    box = _box_budget(raw, device=device, pull_grace_min=pull_grace)
    pins = _pins(raw.get("pins"))
    instance = raw.get("instance")
    if not isinstance(instance, str) or not instance.strip():
        raise ConfigRefused("instance must name the machine being priced")
    grace = _num(raw, "kill_grace_s", "config") if "kill_grace_s" in raw else 30.0
    if not 0 < grace <= 600:
        raise ConfigRefused(f"kill_grace_s {grace:g} must be in (0, 600]")
    poll = _num(raw, "pull_poll_s", "config") if "pull_poll_s" in raw else 30.0
    if not 0 < poll <= 600:
        raise ConfigRefused(f"pull_poll_s {poll:g} must be in (0, 600]")
    durable = raw.get("durable_copy_dir")
    if durable is not None and (not isinstance(durable, str) or not durable.strip()):
        raise ConfigRefused("durable_copy_dir must be a path string when given")

    phases: list[Phase] = []
    names: set[str] = set()
    for i, p in enumerate(raw.get("phases") or []):
        ph = _phase(p, i, names)
        names.add(ph.name)
        phases.append(ph)
    if not phases:
        raise ConfigRefused("no phases")
    approved = raw.get("approved_by", "")
    if not isinstance(approved, str):
        raise ConfigRefused("approved_by must be a string")
    cfg = Config(
        name=str(raw.get("name") or path.stem), state_dir=Path(raw["state_dir"]),
        ledger=Path(raw["ledger"]), device=device, instance=instance, usd_per_hour=rate,
        usd_per_hour_source=str(source), n_gpus=n_gpus, campaign_cap_hours=cap_h,
        terminate_command=terminate, pull_grace_minutes=pull_grace, pull_poll_s=poll,
        durable_copy_dir=Path(durable) if durable else None, approved_by=approved,
        kill_grace_s=grace, phases=tuple(phases),
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(), box=box, pins=pins,
        carry_dirs=_carry(raw.get("carry")),
    )
    _check_ft_row_tokens(cfg)
    ck = [str(d.resolve()) for d in cfg.checkpoint_dirs]
    if len(ck) != len(set(ck)):
        raise ConfigRefused(
            "two units share a checkpoint dir; a resumed unit would pick up the other's newest "
            "checkpoint (another seed) and real_ft_run would refuse it. One dir per unit"
        )
    names_l = [p.name for p in cfg.ledgers]
    if len(names_l) != len(set(names_l)):
        raise ConfigRefused(f"ledger files {names_l} share a file name; the snapshot keys by it")
    total_phase_h = sum(ph.cap.cap_hours for ph in phases)
    if total_phase_h > cap_h:
        print(f"note: phase caps sum to {total_phase_h:g} h, above the {cap_h:g} h campaign "
              "cap; the campaign cap binds")
    if campaign_usd(cfg) >= APPROVAL_FREE_USD and not approved.strip():
        raise ConfigRefused(
            f"the campaign projects to ${campaign_usd(cfg):.2f} at its cap plus the pull grace "
            f"window, at or above rule 4's ${APPROVAL_FREE_USD:.0f} line, and approved_by is "
            "empty. Name who said yes"
        )
    return cfg


def campaign_usd(cfg: Config) -> float:
    """The worst case rule 4 reads: the cap, plus the grace window the box may wait out."""
    return cfg.usd_per_hour * (cfg.campaign_cap_hours + cfg.pull_grace_minutes / 60.0)


def cost_lines(cfg: Config) -> list[str]:
    """The estimate printed before anything starts: per phase at its cap, and the whole."""
    lines = [f"campaign {cfg.name!r} on {cfg.instance}: cost estimate at the caps"
             + (f" (rate from {cfg.usd_per_hour_source})" if cfg.usd_per_hour_source else "")]
    for ph in cfg.phases:
        # Through for_device: the one place that refuses to price rented hardware at zero.
        est = CostEstimate.for_device(
            cap=ph.cap, device=cfg.device, n_gpus=cfg.n_gpus, usd_per_hour=cfg.usd_per_hour,
            instance=cfg.instance,
            usd_per_gpu_hour=(cfg.usd_per_hour / cfg.n_gpus) if cfg.n_gpus > 1 else None,
        )
        lines.append(f"  phase {ph.name:<24} {len(ph.units)} unit(s)  {est.approval_line()}")
    phase_sum = sum(ph.cap.cap_hours for ph in cfg.phases) * cfg.usd_per_hour
    lines.append(
        f"  phases sum to ${phase_sum:.2f}; the campaign cap of {cfg.campaign_cap_hours:g} h "
        f"plus a {cfg.pull_grace_minutes:g} min pull grace window bounds it at "
        f"${campaign_usd(cfg):.2f}"
        + (f"; approved by {cfg.approved_by}" if cfg.approved_by else "")
    )
    lines.append(
        "  termination waits for the Mac's pulled-<n>.ok; "
        + (f"after the grace window, a verified durable copy at {cfg.durable_copy_dir}"
           if cfg.durable_copy_dir else "no durable copy is configured, so without it the box "
           f"is KEPT ALIVE at ${cfg.usd_per_hour:.2f}/h")
    )
    if cfg.box is not None:
        b = cfg.box
        lines.append(
            f"  box guard: ${b.usd:.2f} approved by {b.approved_by}, wall cap "
            f"{b.wall_clock_cap_s:.0f}s, both from boot (uptime) at ${b.usd_per_hour:.2f}/h: "
            f"the box may bill {b.limit_s / 3600:.2f} h since boot, GPU work stops at "
            f"{b.work_limit_s / 3600:.2f} h to leave the pull window inside it"
        )
    return lines


# --- state ----------------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path: Path, body: dict[str, Any]) -> None:
    """The previous file is readable until the instant this one replaces it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


write_state = write_json_atomic


def load_state(cfg: Config, now: float) -> dict[str, Any]:
    path = cfg.state_dir / STATE_NAME
    if not path.exists():
        return {"campaign": cfg.name, "config_digest": cfg.digest, "started_at": now,
                "phases": {}, "outcome": None, "guard_pid": None, "events": [],
                "sync_seq": 0, "finalized": None}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("config_digest") != cfg.digest:
        raise ConfigRefused(
            f"{path} was written under config digest {str(state.get('config_digest'))[:16]} "
            f"and this config is {cfg.digest[:16]}. Resuming it would continue a different "
            "campaign; start a new state_dir or restore the original config"
        )
    if state.get("outcome") is not None:
        raise ConfigRefused(f"{path} records a finished campaign ({state['outcome']}); "
                            "use --finalize to re-run the gated termination")
    return state


def ledger_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def rows_since(path: Path, start: int) -> list[LedgerRow]:
    rows = Ledger(path).rows() if path.exists() else []
    return rows[start:]


def event(state: dict[str, Any], text: str) -> None:
    print(text, flush=True)
    state.setdefault("events", []).append({"at": time.time(), "text": text})


# --- the handoff to the Mac ------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_of(root: Path) -> dict[str, str]:
    """``relative path -> sha256`` for every file under ``root``."""
    return {str(p.relative_to(root)): sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link where possible: ``Checkpoint.write`` replaces by rename, so the linked inode
    stays exactly the bytes that were current when the snapshot was taken."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_snapshot(cfg: Config, state: dict[str, Any], *, phase: str, final: bool) -> int:
    """Snapshot ledgers + checkpoints into ``sync/phase-<n>/``, then write ``phase-<n>.done``.

    The ``.done`` is written last and atomically, so a Mac that sees it sees a complete,
    never-again-modified directory. Returns ``n``.
    """
    seq = int(state.get("sync_seq", 0)) + 1
    sync = cfg.sync_dir
    tmp = sync / f".phase-{seq}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "ledger").mkdir(parents=True)
    for ledger in cfg.ledgers:
        if ledger.exists():
            shutil.copy2(ledger, tmp / "ledger" / ledger.name)
    # The driver's own record travels too: the state (every event, attempt and exit code) and
    # each unit's log are how a stopped campaign is read afterwards, and they die with the box.
    write_state(cfg.state_dir / STATE_NAME, state)
    shutil.copy2(cfg.state_dir / STATE_NAME, tmp / STATE_NAME)
    logs = cfg.state_dir / "logs"
    if logs.is_dir():
        shutil.copytree(logs, tmp / "logs")
    ceiling = cfg.state_dir / BOX_CEILING_NAME
    if ceiling.is_file():
        shutil.copy2(ceiling, tmp / BOX_CEILING_NAME)
    for i, ck in enumerate(cfg.checkpoint_dirs):
        if ck.is_dir():
            for f in sorted(ck.rglob("*")):
                if f.is_file():
                    _link_or_copy(f, tmp / "checkpoints" / f"{i}-{ck.name}" / f.relative_to(ck))
    for i, carried in enumerate(cfg.carry_dirs):
        if carried.is_dir():
            for f in sorted(carried.rglob("*")):
                if f.is_file():
                    _link_or_copy(f, tmp / "carry" / f"{i}-{carried.name}" / f.relative_to(carried))
    final_dir = sync / f"phase-{seq}"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp.rename(final_dir)
    _fsync_dir(sync)
    write_json_atomic(sync / f"phase-{seq}.done", {
        "seq": seq, "phase": phase, "final": final, "outcome": state.get("outcome"),
        "campaign": cfg.name, "created_at": time.time(), "files": manifest_of(final_dir),
    })
    state["sync_seq"] = seq
    if final:
        state["final_seq"] = seq
    prune_snapshots(cfg, keep=seq)
    return seq


def pulled_ok(cfg: Config, seq: int) -> tuple[bool, str]:
    """Has the Mac acknowledged ``phase-<seq>.done``, by that manifest's own sha256?"""
    done, ok = cfg.sync_dir / f"phase-{seq}.done", cfg.sync_dir / f"pulled-{seq}.ok"
    if not ok.exists():
        return False, f"no {ok.name}"
    try:
        ack = json.loads(ok.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{ok.name} unreadable: {exc}"
    want = sha256_file(done)
    if not isinstance(ack, dict) or ack.get("done_sha256") != want:
        return False, (f"{ok.name} acknowledges {str(ack.get('done_sha256'))[:16]}, not this "
                       f"manifest {want[:16]}")
    return True, f"{ok.name} acknowledges manifest {want[:16]} (pulled to {ack.get('dest')})"


def prune_snapshots(cfg: Config, *, keep: int) -> None:
    """Drop snapshot directories the Mac has acknowledged. Manifests and acks stay."""
    for done in cfg.sync_dir.glob("phase-*.done"):
        seq = int(done.stem.split("-")[1])
        snap = cfg.sync_dir / f"phase-{seq}"
        if seq != keep and snap.is_dir() and pulled_ok(cfg, seq)[0]:
            shutil.rmtree(snap)


def _same_device(a: Path, b: Path) -> bool:
    return a.stat().st_dev == b.stat().st_dev


def copy_to_durable(cfg: Config, seq: int) -> str:
    if cfg.durable_copy_dir is None:
        return "no durable_copy_dir configured"
    if not cfg.durable_copy_dir.is_dir():
        return f"durable_copy_dir {cfg.durable_copy_dir} does not exist; nothing copied"
    dst = cfg.durable_copy_dir / cfg.name / f"phase-{seq}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(cfg.sync_dir / f"phase-{seq}", dst)
    shutil.copy2(cfg.sync_dir / f"phase-{seq}.done", dst.parent / f"phase-{seq}.done")
    return f"copied snapshot {seq} to {dst}"


def durable_verified(cfg: Config, seq: int) -> tuple[bool, str]:
    """True only if the durable copy is PROVEN: other device, every sha256 matching."""
    if cfg.durable_copy_dir is None:
        return False, "no durable_copy_dir configured"
    root = cfg.durable_copy_dir / cfg.name / f"phase-{seq}"
    if not root.is_dir():
        return False, f"no durable copy at {root}"
    if _same_device(cfg.durable_copy_dir, cfg.state_dir):
        return False, (f"{cfg.durable_copy_dir} is on the same device as {cfg.state_dir}: it "
                       "is the instance's own disk and dies with the instance")
    want = json.loads((cfg.sync_dir / f"phase-{seq}.done").read_text(encoding="utf-8"))["files"]
    have = manifest_of(root)
    bad = sorted(k for k in want if have.get(k) != want[k])
    if bad:
        return False, f"durable copy differs from the manifest in {len(bad)} file(s): {bad[:3]}"
    return True, f"durable copy at {root} verified: {len(want)} file(s) by sha256"


# --- finalizing: the only path to terminate ------------------------------------------------


def _cmdline(pid: int) -> str:
    """The whole command line. ``-ww``: Linux procps cuts ``ps -o command=`` at 80 columns when
    stdout is a pipe (macOS never does), which hid ``campaign_driver.py`` past column 80 and let
    a second finalizer run beside a live one (the H100 box suite at 8e6a009, 2026-10-03)."""
    try:
        return subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_campaign_process(pid: object, *, guard: bool | None = None) -> bool:
    """Alive AND ours. After a reboot a pid is reissued, and an unrelated process sitting on
    a recorded pid must not be mistaken for a guard or a driver."""
    if not isinstance(pid, int) or not _alive(pid):
        return False
    out = _cmdline(pid)
    if "campaign_driver.py" not in out:
        return False
    return guard is None or (("--guard" in out) == guard)


def _is_guard(pid: object) -> bool:
    return _is_campaign_process(pid, guard=True)


def acquire_lock(cfg: Config) -> bool:
    """One finalizer at a time. A lock whose holder is dead (or not ours) is broken."""
    lock = cfg.state_dir / LOCK_NAME
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                holder = int(lock.read_text().strip() or "0")
            except (OSError, ValueError):
                holder = 0
            if holder == os.getpid():
                return True
            if _is_campaign_process(holder):
                return False
            with contextlib.suppress(FileNotFoundError):
                lock.unlink()
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))
        return True
    return False


def run_terminate(cfg: Config) -> tuple[bool, str]:
    try:
        proc = subprocess.run(list(cfg.terminate_command), capture_output=True, text=True,
                              timeout=TERMINATE_TIMEOUT_S, check=False, cwd=REPO)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = (proc.stdout + proc.stderr).strip()[-500:]
    return proc.returncode == 0, f"exit {proc.returncode} {detail}"


def wait_for_pull(cfg: Config, seq: int) -> tuple[bool, str]:
    """Bounded by the grace window and, on a box, by what is left before its ceiling."""
    deadline = _now() + cfg.pull_grace_minutes * 60.0
    while True:
        ok, why = pulled_ok(cfg, seq)
        if ok:
            return ok, why
        left = deadline - _now()
        if cfg.box is not None:
            reading = box_reading(cfg.box)
            if reading.left_s < left:
                left = reading.left_s
                if left <= 0:
                    return False, f"{why}; the wait ended at the box ceiling ({reading.line()})"
        if left <= 0:
            return ok, why
        _sleep(min(cfg.pull_poll_s, max(left, 0.01)))


def write_box_ceiling(cfg: Config, reading: BoxReading, reason: str, who: str) -> Path:
    """The marker the human reads: why the box guard stopped, with its numbers.

    The first writer's record stands (the moment the ceiling was met); a later one is an
    event in the state, not a rewrite of the marker.
    """
    path = cfg.state_dir / BOX_CEILING_NAME
    if not path.exists():
        write_json_atomic(path, reading.numbers(reason, who))
    print(f"!!! BOX_CEILING ({who}): {reason}. {reading.line()}. Written to {path}", flush=True)
    return path


def box_stop(cfg: Config, state: dict[str, Any], reading: BoxReading, reason: str, *,
             phase: str, who: str = "driver") -> int:
    """At the ceiling: stop the unit's process group, write BOX_CEILING, gated finalize.

    Returns finalize's code: 5 once the human's terminate command ran, or the louder 3 (it
    failed) or 4 (kept alive: nothing proved the data is off the box).
    """
    pgid = state.get("running_pgid")
    if pgid is not None:
        _stop_group(pgid, cfg.kill_grace_s)
        state["running_pgid"] = None
    write_box_ceiling(cfg, reading, reason, who)
    event(state, f"{who}: BOX_CEILING: {reason}; {reading.line()}")
    return finalize(cfg, state, f"box ceiling: {reason}", EXIT_BOX_CEILING, phase=phase,
                    who=who)


def finalize(cfg: Config, state: dict[str, Any], outcome: str, code: int, *, phase: str,
             who: str = "driver") -> int:
    """Final snapshot, then terminate ONLY once the data is proven safe off the box."""
    state_path = cfg.state_dir / STATE_NAME
    if not acquire_lock(cfg):
        print(f"{who}: another campaign process holds {LOCK_NAME} and is finalizing; "
              "not terminating from here")
        return code
    state["outcome"] = state.get("outcome") or outcome
    if state.get("final_seq") is None:
        seq = write_snapshot(cfg, state, phase=phase, final=True)
        event(state, f"{who}: final snapshot phase-{seq}.done written; waiting up to "
                     f"{cfg.pull_grace_minutes:g} min for the Mac's pulled-{seq}.ok")
        event(state, f"{who}: {copy_to_durable(cfg, seq)}")
    else:
        seq = int(state["final_seq"])
        event(state, f"{who}: final snapshot phase-{seq} exists; waiting up to "
                     f"{cfg.pull_grace_minutes:g} min for pulled-{seq}.ok")
    write_state(state_path, state)
    pulled, why = wait_for_pull(cfg, seq)
    if pulled:
        event(state, f"{who}: the Mac has it -- {why}")
    else:
        event(state, f"{who}: no acknowledgement after {cfg.pull_grace_minutes:g} min ({why})")
        durable, dwhy = durable_verified(cfg, seq)
        event(state, f"{who}: durable copy: {dwhy}")
        if not durable:
            state["finalized"] = "kept_alive"
            write_state(state_path, state)
            print(
                f"!!! INSTANCE KEPT ALIVE ({cfg.instance}, ${cfg.usd_per_hour:.2f}/h and "
                "counting): nothing proves the ledger and checkpoints exist off this box, so "
                "terminating would destroy the only copy.\n"
                f"!!! Pull snapshot {seq} from {cfg.sync_dir} (tools/sync_box.sh pull), then "
                f"re-run with --finalize, or terminate by hand: {' '.join(cfg.terminate_command)}"
            )
            (cfg.state_dir / LOCK_NAME).unlink(missing_ok=True)
            return EXIT_KEPT_ALIVE
    ok, detail = run_terminate(cfg)
    event(state, f"{who}: terminate: {'ok' if ok else 'FAILED'} ({detail})")
    if ok:
        state["finalized"] = "terminated"
        if who != "guard":
            stop_guard(state)
    else:
        print("!!! THE INSTANCE WAS NOT TERMINATED. The guard stays armed; end it by hand.")
        code = EXIT_TERMINATE_FAILED
    write_state(state_path, state)
    (cfg.state_dir / LOCK_NAME).unlink(missing_ok=True)
    print(f"campaign {state['outcome']}")
    return code


# --- the guard --------------------------------------------------------------------------


def ensure_guard(cfg: Config, state: dict[str, Any], config_path: Path, *,
                 quiet: bool = False) -> None:
    """The deadman: finalizes at the campaign cap, or the box ceiling, if this driver is dead.

    The guard loads its config once, at start, and holds it in memory: a config moved or
    edited while it sleeps cannot stop it from firing. It reads the box's uptime itself, from
    the same path the driver reads. Called before every unit, so a guard that was killed is
    re-armed rather than silently missing.
    """
    previous = state.get("guard_pid")
    if _is_guard(previous):
        if not quiet:
            print(f"guard: pid {previous} alive, deadline unchanged")
        return
    if previous is not None:
        event(state, f"guard: pid {previous} is not running; re-arming it")
    deadline = float(state["started_at"]) + cfg.campaign_cap_hours * 3600.0
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.state_dir / "guard.log").open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--guard", repr(deadline),
             "--config", str(config_path), "--uptime-path", str(UPTIME_PATH)],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    state["guard_pid"] = proc.pid
    print(f"guard: started pid {proc.pid}; at "
          f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(deadline))} it stops the "
          "campaign and runs the gated termination")


def stop_guard(state: dict[str, Any]) -> None:
    pid = state.get("guard_pid")
    if isinstance(pid, int) and _is_guard(pid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGTERM)  # its own session: pgid == pid
    state["guard_pid"] = None


def _stop_group(pid: object, grace_s: float) -> None:
    if not isinstance(pid, int) or not _alive(pid):
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGTERM)
    end = time.time() + grace_s
    while _alive(pid) and time.time() < end:
        time.sleep(0.2)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def guard_fire(cfg: Config, box: BoxReading | None = None) -> int:
    """At the cap (or the box ceiling): stop the driver and its unit, then gated finalize."""
    path = cfg.state_dir / STATE_NAME
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "phases": {}, "events": [], "sync_seq": 0}
    if state.get("finalized") == "terminated":
        print("guard: the campaign already terminated the instance; nothing to do")
        return EXIT_DONE
    what = "the box ceiling" if box is not None else "the campaign cap"
    driver = state.get("driver_pid")
    if _is_campaign_process(driver, guard=False):
        print(f"guard: {what} reached; stopping driver pid {driver}", flush=True)
        _stop_group(driver, cfg.kill_grace_s)
    if state.get("running_pgid") is not None:
        print(f"guard: stopping unit process group {state['running_pgid']}", flush=True)
        _stop_group(state["running_pgid"], cfg.kill_grace_s)
    # Re-read: the driver may have written state up to the moment it stopped.
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    state["running_pgid"] = None
    if box is not None:
        reason = ("the box clock could not be read" if not box.ok
                  else "the box reached its ceiling")
        return box_stop(cfg, state, box, f"{reason} (guard)", phase="guard", who="guard")
    return finalize(cfg, state, "capped: the campaign cap was reached (guard)", EXIT_FAILED,
                    phase="guard", who="guard")


def run_guard(deadline: float, config_path: Path) -> int:
    """Sleep to the campaign deadline or the box's work limit, whichever comes first.

    The box is re-read every pass, so a deadline computed at start cannot drift from the box's
    own clock; an unreadable clock fires the guard at once rather than reading as time left.
    """
    cfg = load_config(config_path)
    while True:
        left = deadline - _now()
        if cfg.box is not None:
            reading = box_reading(cfg.box)
            if reading.at_ceiling:
                return guard_fire(cfg, box=reading)
            left = min(left, reading.work_left_s)
        if left <= 0:
            return guard_fire(cfg)
        _sleep(min(left, 60.0))


# --- running units --------------------------------------------------------------------


def run_group(argv: list[str], *, timeout_s: float, log: Path, grace_s: float,
              on_start: Callable[[int], None] | None = None) -> tuple[str, int]:
    """``(status, returncode)`` with status ``ok`` / ``failed`` / ``capped``.

    Its own session, so the cap kills the whole tree -- a trainer's dataloader workers
    included -- rather than the one pid the driver happens to hold.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(argv)}\n")
        fh.flush()
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, cwd=REPO)
        if on_start is not None:
            on_start(proc.pid)
        try:
            rc = proc.wait(timeout=max(timeout_s, 0.001))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=REAP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    print(f"!!! unit pid {proc.pid} did not exit {REAP_TIMEOUT_S:g}s after "
                          "SIGKILL (uninterruptible sleep?); moving on without reaping it",
                          flush=True)
                    return "capped", -int(signal.SIGKILL)
            return "capped", proc.returncode
    return ("ok" if rc == 0 else "failed"), rc


def throughput(state: dict[str, Any], cfg: Config, phase_name: str) -> float | None:
    """Slowest measured optimizer steps per second among the ``ft`` rows of a phase."""
    info = state["phases"].get(phase_name)
    if not info or info.get("status") != "done":
        return None
    rates = []
    for row in rows_since(cfg.ledger, int(info["ledger_lines_at_start"])):
        steps = row.metrics.get("train.optimizer_steps")
        if row.run_kind == "ft" and isinstance(steps, Ran) and row.wall_clock_s > 0:
            value = steps.value
            if isinstance(value, int | float) and value > 0:
                rates.append(float(value) / row.wall_clock_s)
    return min(rates) if rates else None


def unit_argv(unit: Unit, cfg: Config, state: dict[str, Any], resuming: bool) -> list[str]:
    argv = _resolve_ft_rows(list(unit.argv), state)
    if unit.checkpoint is None:
        return argv
    rate = throughput(state, cfg, unit.checkpoint.throughput_from)
    if rate is None:
        raise ConfigRefused(
            f"unit {unit.name}: no ft row with train.optimizer_steps in phase "
            f"{unit.checkpoint.throughput_from!r}, so --checkpoint-every cannot be derived "
            f"from a measured rate. Refusing to guess a step count for a "
            f"{unit.checkpoint.every_hours:g} h interval"
        )
    every = max(1, math.floor(rate * unit.checkpoint.every_hours * 3600.0))
    argv = [str(every) if a == "{checkpoint_every}" else a for a in argv]
    if resuming:
        found = sorted(unit.checkpoint.dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if found:
            argv += ["--resume-from", str(found[-1])]
    return argv


def gate_verdict(gate: GateSpec, start: int) -> tuple[bool, str]:
    rows = [r for r in rows_since(gate.ledger, start)
            if (r.recipe or {}).get("tool") == gate.tool]
    if not rows:
        return False, f"no {gate.tool} row was written, so {gate.name} was never scored"
    bad = []
    seeds = set()
    for r in rows:
        g = r.gates.get(gate.name)
        if r.quick:
            # Rule 8: a quick row is excluded from decisions, and GO is one. A passing gate
            # on a truncated, subsampled or smoke run is not evidence for spending the rest.
            bad.append(f"{r.row_id[:8]} seed {r.protocol.seed}: quick ({r.quick_reason})")
        elif isinstance(g, Ran) and g.passed:
            seeds.add(r.protocol.seed)
        else:
            bad.append(f"{r.row_id[:8]} seed {r.protocol.seed}: {g}")
    if bad:
        return False, f"{gate.name} did not pass on every row: " + "; ".join(bad)
    if len(seeds) < gate.min_seeds:
        return False, (f"{gate.name} passed on {len(seeds)} seed(s) {sorted(seeds)}; the "
                       f"go/no-go needs {gate.min_seeds}")
    return True, f"{gate.name} passed on seeds {sorted(seeds)} ({len(rows)} row(s))"


# --- gated phases: markers -------------------------------------------------------------


def check_marker(m: Marker) -> tuple[str, str]:
    """``("met" | "absent" | "refuted", why)``. A marker that cannot be parsed is absent,
    never met: a half-written verdict file must not read as admissible."""
    if not m.path.is_file():
        return "absent", f"{m.path} does not exist"
    if m.json_key is None:
        return "met", f"{m.path} exists"
    try:
        body = json.loads(m.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "absent", f"{m.path} unreadable as JSON ({type(exc).__name__})"
    if not isinstance(body, dict) or m.json_key not in body:
        return "absent", f"{m.path} has no {m.json_key!r}"
    value = body[m.json_key]
    if value == m.equals:
        return "met", f"{m.path} {m.json_key} = {value!r}"
    return "refuted", f"{m.path} {m.json_key} = {value!r}, not {m.equals!r}"


def wait_for_markers(cfg: Config, req: Requires, *, deadline: float
                     ) -> tuple[str, str, BoxReading | None]:
    """``(status, why, reading)``; status ``met``, ``refuted``, ``timeout`` or ``box``.

    Bounded three ways: ``wait_max_s``, the caller's deadline (the campaign cap) and the box's
    remaining work time, re-read every poll.
    """
    end_at = min(_now() + req.wait_max_s, deadline)
    while True:
        results = [check_marker(m) for m in req.markers]
        refuted = [why for s, why in results if s == "refuted"]
        if refuted:
            return "refuted", "; ".join(refuted), None
        absent = [why for s, why in results if s != "met"]
        if not absent:
            return "met", "; ".join(why for _, why in results), None
        left = end_at - _now()
        if cfg.box is not None:
            reading = box_reading(cfg.box)
            if reading.at_ceiling:
                return "box", "; ".join(absent), reading
            left = min(left, reading.work_left_s)
        if left <= 0:
            return "timeout", "; ".join(absent), None
        _sleep(min(req.poll_s, max(left, 0.01)))


# --- the campaign ---------------------------------------------------------------------


def box_preflight(cfg: Config) -> BoxReading | None:
    """Refuse to start on an unreadable box clock or a box already at its ceiling."""
    if cfg.box is None:
        return None
    reading = box_reading(cfg.box)
    if not reading.ok:
        raise BoxRefused(
            f"the box's uptime could not be read ({reading.error}). Without it the box's spend "
            "is unknown, and unknown is not within budget"
        )
    if reading.at_ceiling:
        raise BoxRefused(
            f"{reading.line()}: nothing is left before the ceiling (less the pull window). "
            "Run --finalize to hand the data home"
        )
    return reading


def run_campaign(cfg: Config, config_path: Path) -> int:
    bad_pins = verify_pins(cfg)
    if bad_pins:
        raise ConfigRefused("pinned input(s) do not match: " + "; ".join(bad_pins))
    start_reading = box_preflight(cfg)
    now = _now()
    state = load_state(cfg, now)
    state_path = cfg.state_dir / STATE_NAME
    resumed = bool(state["phases"])
    print("\n".join(cost_lines(cfg)))
    event(state, f"{'RESUME' if resumed else 'START'} campaign {cfg.name} "
                 f"(elapsed {now - float(state['started_at']):.0f}s of "
                 f"{cfg.campaign_cap_hours * 3600:.0f}s)")
    if start_reading is not None:
        event(state, f"box guard: {start_reading.line()}")
    state["driver_pid"] = os.getpid()
    ensure_guard(cfg, state, config_path)
    write_state(state_path, state)
    campaign_deadline = float(state["started_at"]) + cfg.campaign_cap_hours * 3600.0
    last = cfg.phases[0].name

    def end(outcome: str, code: int, phase: str) -> int:
        return finalize(cfg, state, outcome, code, phase=phase)

    def ceiling() -> BoxReading | None:
        if cfg.box is None:
            return None
        reading = box_reading(cfg.box)
        return reading if reading.at_ceiling else None

    def at_box(reading: BoxReading, where: str, phase: str) -> int:
        why = ("the box clock could not be read" if not reading.ok
               else "the box reached its ceiling")
        return box_stop(cfg, state, reading, f"{why} {where}", phase=phase)

    for phase in cfg.phases:
        last = phase.name
        info = state["phases"].setdefault(phase.name, {"status": "pending", "units": {}})
        if info["status"] == "done":
            print(f"phase {phase.name}: done earlier, skipped")
            continue
        if (hit := ceiling()) is not None:
            return at_box(hit, f"before phase {phase.name}", phase.name)
        if info["status"] == "pending" and phase.requires is not None:
            event(state, f"phase {phase.name}: waiting up to {phase.requires.wait_max_s:.0f}s "
                         "for its markers")
            write_state(state_path, state)
            status, why, reading = wait_for_markers(cfg, phase.requires,
                                                    deadline=campaign_deadline)
            if status == "box" and reading is not None:
                return at_box(reading, f"waiting for phase {phase.name}'s markers ({why})",
                              phase.name)
            if status != "met":
                info["status"] = "not_run"
                event(state, f"phase {phase.name}: NOT RUN, its markers did not hold "
                             f"({status}): {why}")
                return end(f"not run: phase {phase.name}'s markers did not hold", EXIT_STOPPED,
                           phase.name)
            event(state, f"phase {phase.name}: markers hold: {why}")
        if info["status"] == "pending":
            info.update(status="running", started_at=_now(),
                        ledger_lines_at_start=ledger_lines(cfg.ledger),
                        gate_lines_at_start=(ledger_lines(phase.gate.ledger)
                                             if phase.gate else None))
            write_state(state_path, state)
        phase_deadline = float(info["started_at"]) + phase.cap.cap_s
        event(state, f"phase {phase.name}: running ({len(phase.units)} unit(s), cap "
                     f"{phase.cap.cap_hours:g} h)")
        for unit in phase.units:
            u = info["units"].setdefault(unit.name, {"status": "pending", "attempts": 0})
            if u["status"] == "done":
                print(f"  unit {unit.name}: done earlier, skipped")
                continue
            resuming = u["status"] == "running"
            try:
                argv = unit_argv(unit, cfg, state, resuming)
            except ConfigRefused as exc:
                event(state, f"  unit {unit.name}: {exc}")
                info["status"] = "failed"
                return end(f"failed at {phase.name}/{unit.name}", EXIT_FAILED, phase.name)
            if (hit := ceiling()) is not None:
                return at_box(hit, f"before unit {phase.name}/{unit.name}", phase.name)
            ensure_guard(cfg, state, config_path, quiet=True)
            u.update(status="running", attempts=u["attempts"] + 1,
                     ledger_lines_at_start=ledger_lines(cfg.ledger), argv=argv)
            write_state(state_path, state)
            now = _now()
            bounds = {"phase": phase_deadline - now, "campaign": campaign_deadline - now}
            if cfg.box is not None:
                bounds["box"] = box_reading(cfg.box).work_left_s
            binding = min(bounds, key=lambda k: bounds[k])
            timeout = bounds[binding]
            event(state, f"  unit {unit.name}: {'RESUMING' if resuming else 'start'} "
                         f"(attempt {u['attempts']}, {max(timeout, 0):.0f}s left, bound by "
                         f"the {binding})")

            def record_pgid(pgid: int) -> None:
                state["running_pgid"] = pgid
                write_state(state_path, state)

            status, rc = run_group(argv, timeout_s=timeout,
                                   log=cfg.state_dir / "logs" / f"{phase.name}.{unit.name}.log",
                                   grace_s=cfg.kill_grace_s, on_start=record_pgid)
            state["running_pgid"] = None
            written = rows_since(cfg.ledger, int(u["ledger_lines_at_start"]))
            new_rows = len(written)
            if status == "failed" and rc in unit.accept_returncodes:
                not_completed = [f"{r.row_id[:8]}={r.status}" for r in written
                                 if r.status != "completed"]
                if written and not not_completed:
                    status = "ok"
                    event(state, f"  unit {unit.name}: exit {rc} accepted by config -- all "
                                 f"{new_rows} row(s) it wrote are completed (a failed claim, "
                                 "not a crash; read the rows)")
            u.update(status=status, returncode=rc, rows=new_rows,
                     row_ids=[[r.row_id, r.run_kind] for r in written])
            if status == "ok" and new_rows < unit.expect_rows:
                u["status"] = status = "failed"
                event(state, f"  unit {unit.name}: exited {rc} but wrote {new_rows} ledger "
                             f"row(s), expected {unit.expect_rows}")
            not_quick = [r.row_id[:8] for r in written if not r.quick]
            if status == "ok" and unit.quick and not_quick:
                u["status"] = status = "failed"
                event(state, f"  unit {unit.name}: declared quick (rule 8) but wrote row(s) "
                             f"{not_quick} that are not quick")
            write_state(state_path, state)
            if status == "capped" and binding == "box" and cfg.box is not None:
                info["status"] = "capped"
                return at_box(box_reading(cfg.box),
                              f"during unit {phase.name}/{unit.name} (exit {rc})", phase.name)
            if status != "ok":
                which = ("campaign" if campaign_deadline <= phase_deadline else "phase")
                why = f"the {which} cap" if status == "capped" else f"exit {rc}"
                info["status"] = status
                event(state, f"  unit {unit.name}: {status.upper()} ({why})")
                return end(f"{status} at {phase.name}/{unit.name}", EXIT_FAILED, phase.name)
            u["status"] = "done"
            event(state, f"  unit {unit.name}: done, {new_rows} ledger row(s)")
            write_state(state_path, state)
        if phase.gate is not None:
            go, why = gate_verdict(phase.gate, int(info["gate_lines_at_start"]))
            info["gate"] = {"go": go, "why": why}
            if not go:
                info["status"] = "stopped_at_gate"
                event(state, f"phase {phase.name}: GO/NO-GO = NO-GO: {why}")
                event(state, "  stopping: the remaining phases are not run")
                return end(f"stopped at the {phase.name} go/no-go", EXIT_STOPPED, phase.name)
            event(state, f"phase {phase.name}: GO/NO-GO = GO: {why}")
        info["status"] = "done"
        seq = write_snapshot(cfg, state, phase=phase.name, final=False)
        event(state, f"  snapshot phase-{seq}.done written for the Mac to pull; not waiting")
        write_state(state_path, state)
    return end("completed every phase", EXIT_DONE, last)


def rerun_finalize(cfg: Config) -> int:
    """For a box left alive: wait for the Mac again, then terminate if it is safe."""
    path = cfg.state_dir / STATE_NAME
    if not path.exists():
        raise ConfigRefused(f"no state at {path}; nothing to finalize")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("finalized") == "terminated":
        print("already terminated")
        return EXIT_DONE
    return finalize(cfg, state, state.get("outcome") or "finalized by hand", EXIT_DONE,
                    phase="finalize", who="finalize")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="validate and print the cost only")
    parser.add_argument("--finalize", action="store_true",
                        help="re-run the gated termination for a campaign left alive")
    parser.add_argument("--guard", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--uptime-path", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.uptime_path is not None:
        global UPTIME_PATH
        UPTIME_PATH = args.uptime_path
    if args.guard is not None:
        return run_guard(args.guard, args.config)
    cfg = load_config(args.config)
    if args.plan:
        print("\n".join(cost_lines(cfg)))
        bad = verify_pins(cfg)
        for line in bad:
            print(f"  PIN FAILS: {line}")
        if cfg.box is not None:
            print(f"  {box_reading(cfg.box).line()}")
        return EXIT_FAILED if bad else 0
    if args.finalize:
        return rerun_finalize(cfg)
    return run_campaign(cfg, args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
