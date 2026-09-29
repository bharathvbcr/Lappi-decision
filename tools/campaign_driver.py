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
TERMINATE_TIMEOUT_S: Final[float] = 300.0

#: Exit codes. 0 every phase ran; 2 stopped at the go/no-go; 1 anything else stopped it;
#: 3 the terminate command failed; 4 the box was deliberately KEPT ALIVE because nothing
#: proved the data was safe off it. 3 and 4 are the loud ones: the box is still billing.
EXIT_DONE, EXIT_STOPPED, EXIT_FAILED, EXIT_TERMINATE_FAILED, EXIT_KEPT_ALIVE = 0, 2, 1, 3, 4


class ConfigRefused(SystemExit):
    """The config cannot be run as written. Nothing was launched."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"campaign config refused: {reason}")


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
        units.append(Unit(u["name"], argv, expect, ck, tuple(accept)))
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
    return Phase(p["name"], cap, tuple(units), gate)


def load_config(path: Path) -> Config:
    """Parse and validate. Every refusal names the field; nothing has been launched yet."""
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ConfigRefused("the config must be a JSON object")
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
    cap_h = _num(raw, "campaign_cap_hours", "config")
    if not 0 < cap_h <= CAMPAIGN_MAX_HOURS:
        raise ConfigRefused(
            f"campaign_cap_hours {cap_h:g} is outside (0, {CAMPAIGN_MAX_HOURS:g}]: the user "
            "approved one GH200 for up to 2-3 days, and the cap is read-only (rule 2)"
        )
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
    instance = raw.get("instance")
    if not isinstance(instance, str) or not instance.strip():
        raise ConfigRefused("instance must name the machine being priced")
    grace = _num(raw, "kill_grace_s", "config") if "kill_grace_s" in raw else 30.0
    if not 0 < grace <= 600:
        raise ConfigRefused(f"kill_grace_s {grace:g} must be in (0, 600]")
    pull_grace = (_num(raw, "pull_grace_minutes", "config") if "pull_grace_minutes" in raw
                  else DEFAULT_PULL_GRACE_MIN)
    if not 0 < pull_grace <= MAX_PULL_GRACE_MIN:
        raise ConfigRefused(f"pull_grace_minutes {pull_grace:g} must be in "
                            f"(0, {MAX_PULL_GRACE_MIN:g}]")
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
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
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
    for i, ck in enumerate(cfg.checkpoint_dirs):
        if ck.is_dir():
            for f in sorted(ck.rglob("*")):
                if f.is_file():
                    _link_or_copy(f, tmp / "checkpoints" / f"{i}-{ck.name}" / f.relative_to(ck))
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
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=10, check=False).stdout
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
    deadline = time.time() + cfg.pull_grace_minutes * 60.0
    while True:
        ok, why = pulled_ok(cfg, seq)
        if ok or time.time() >= deadline:
            return ok, why
        time.sleep(min(cfg.pull_poll_s, max(deadline - time.time(), 0.01)))


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


def ensure_guard(cfg: Config, state: dict[str, Any], config_path: Path) -> None:
    """The deadman: finalizes at the campaign cap even if this driver is dead.

    The guard loads its config once, at start, and holds it in memory: a config moved or
    edited while it sleeps cannot stop it from firing.
    """
    if _is_guard(state.get("guard_pid")):
        print(f"guard: pid {state['guard_pid']} alive, deadline unchanged")
        return
    deadline = float(state["started_at"]) + cfg.campaign_cap_hours * 3600.0
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.state_dir / "guard.log").open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--guard", repr(deadline),
             "--config", str(config_path)],
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


def guard_fire(cfg: Config) -> int:
    """At the cap: stop the driver and its unit, then the same gated finalize."""
    path = cfg.state_dir / STATE_NAME
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "phases": {}, "events": [], "sync_seq": 0}
    if state.get("finalized") == "terminated":
        print("guard: the campaign already terminated the instance; nothing to do")
        return EXIT_DONE
    driver = state.get("driver_pid")
    if _is_campaign_process(driver, guard=False):
        print(f"guard: campaign cap reached; stopping driver pid {driver}", flush=True)
        _stop_group(driver, cfg.kill_grace_s)
    if state.get("running_pgid") is not None:
        print(f"guard: stopping unit process group {state['running_pgid']}", flush=True)
        _stop_group(state["running_pgid"], cfg.kill_grace_s)
    # Re-read: the driver may have written state up to the moment it stopped.
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    state["running_pgid"] = None
    return finalize(cfg, state, "capped: the campaign cap was reached (guard)", EXIT_FAILED,
                    phase="guard", who="guard")


def run_guard(deadline: float, config_path: Path) -> int:
    cfg = load_config(config_path)
    while (left := deadline - time.time()) > 0:
        time.sleep(min(left, 60.0))
    return guard_fire(cfg)


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
                proc.wait()
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
    argv = list(unit.argv)
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


# --- the campaign ---------------------------------------------------------------------


def run_campaign(cfg: Config, config_path: Path) -> int:
    now = time.time()
    state = load_state(cfg, now)
    state_path = cfg.state_dir / STATE_NAME
    resumed = bool(state["phases"])
    print("\n".join(cost_lines(cfg)))
    event(state, f"{'RESUME' if resumed else 'START'} campaign {cfg.name} "
                 f"(elapsed {now - float(state['started_at']):.0f}s of "
                 f"{cfg.campaign_cap_hours * 3600:.0f}s)")
    state["driver_pid"] = os.getpid()
    ensure_guard(cfg, state, config_path)
    write_state(state_path, state)
    campaign_deadline = float(state["started_at"]) + cfg.campaign_cap_hours * 3600.0
    last = cfg.phases[0].name

    def end(outcome: str, code: int, phase: str) -> int:
        return finalize(cfg, state, outcome, code, phase=phase)

    for phase in cfg.phases:
        last = phase.name
        info = state["phases"].setdefault(phase.name, {"status": "pending", "units": {}})
        if info["status"] == "done":
            print(f"phase {phase.name}: done earlier, skipped")
            continue
        if info["status"] == "pending":
            info.update(status="running", started_at=time.time(),
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
            u.update(status="running", attempts=u["attempts"] + 1,
                     ledger_lines_at_start=ledger_lines(cfg.ledger), argv=argv)
            write_state(state_path, state)
            timeout = min(phase_deadline, campaign_deadline) - time.time()
            event(state, f"  unit {unit.name}: {'RESUMING' if resuming else 'start'} "
                         f"(attempt {u['attempts']}, {max(timeout, 0):.0f}s left)")

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
            u.update(status=status, returncode=rc, rows=new_rows)
            if status == "ok" and new_rows < unit.expect_rows:
                u["status"] = status = "failed"
                event(state, f"  unit {unit.name}: exited {rc} but wrote {new_rows} ledger "
                             f"row(s), expected {unit.expect_rows}")
            write_state(state_path, state)
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
    args = parser.parse_args(argv)
    if args.guard is not None:
        return run_guard(args.guard, args.config)
    cfg = load_config(args.config)
    if args.plan:
        print("\n".join(cost_lines(cfg)))
        return 0
    if args.finalize:
        return rerun_finalize(cfg)
    return run_campaign(cfg, args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
