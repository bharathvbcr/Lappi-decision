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

import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Self

from qd_data.fingerprint import code_fingerprint

from .run_control import CostEstimate, WallClockCap
from .tristate import NotRun, Ran, TriState, parse_tristate

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "NON_PROMOTING_RUN_KINDS",
    "NOT_APPLICABLE",
    "ForkBranch",
    "ForkPoint",
    "Ledger",
    "LedgerChainError",
    "LedgerForkError",
    "LedgerRow",
    "PromotionVerdict",
    "Protocol",
    "RunRecorder",
    "SuiteCounts",
    "SuiteFailure",
    "SuiteOutcome",
    "find_forks",
    "main",
    "parse_cargo_test_output",
    "parse_command",
    "parse_pytest_output",
    "record_build_run",
    "run_suite",
    "verify_no_fork",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where a lane's rows go unless it says otherwise. Named here rather than in each
# caller so two lanes cannot end up with two ledgers.
DEFAULT_LEDGER_PATH = REPO_ROOT / "ledger" / "runs.jsonl"

RunKind = Literal[
    "teacher", "lr_probe", "cpt", "prune_heal", "ft", "ablation",
    "eval", "calibration", "smoke", "throughput", "resume", "scale",
    "build",
]
Status = Literal["completed", "killed", "failed"]
#: Who measured `LedgerRow.wall_clock_s`. "unrecorded" is for rows written before the field
#: existed and is never chosen by a run: a row either states a source or predates the idea.
WallClockSource = Literal["caller", "recorder", "unrecorded"]

_RUN_KINDS: frozenset[str] = frozenset(RunKind.__args__)  # type: ignore[attr-defined]
_STATUSES: frozenset[str] = frozenset(Status.__args__)  # type: ignore[attr-defined]
#: The cap a `build` row's cost is priced against. `CostEstimate` prices `projected_usd`
#: from the cap, and a build row is a local gate run whose rate is zero, so the number this
#: multiplies is zero whatever it is. It is stated rather than left implicit because a cap
#: is what makes the estimate constructible at all, and because a future build row on rented
#: hardware must inherit a real bound rather than a placeholder nobody chose.
BUILD_CAP_S: Final[float] = 3600.0

_WALL_CLOCK_SOURCES: frozenset[str] = frozenset(
    WallClockSource.__args__  # type: ignore[attr-defined]
)

# Run kinds whose row must state how their training loop ended, because they have one.
#
# Rule 8: *"Fewer than 3 seeds, a truncated schedule or a subsample is marked `quick` in
# the ledger and excluded from decisions."* `quick` is set by the caller, and until
# 2026-09-20 nothing checked it against evidence the row already carried. Measured on that
# date: three rows differing only in seed, every required gate and control passing, one of
# them honestly recording `train.termination == "wall_clock_cap"` -- a truncated schedule
# in rule 8's own words -- and all three saying `quick=False`. `promotion_verdict` returned
# `promoted=True`. The self-report was the whole of the enforcement.
#
# `train.termination` is written by `qd_train.trainer._train` and `byte_train.train_rung0`
# on every exit path, so for these kinds its absence is as much a defect as its value.
TRAINING_RUN_KINDS: frozenset[str] = frozenset({"cpt", "ft", "prune_heal"})

# Run kinds that may never promote a decision, whatever their gates say.
#
# `build` is here because its gates are *vacuous*, not satisfied: a lane that
# compiles the workspace and runs a test suite has nothing to say about
# `paired_margin_vs_linear`. Relying on `_fill_unreported` to leave them
# `not_run` would work today and would break the first time a build lane set a
# gate for an unrelated reason, so the refusal is stated on the run kind.
NON_PROMOTING_RUN_KINDS: frozenset[str] = frozenset({"build"})

# The marker a `build` row carries in the three protocol components that do not
# exist for it. Not a hash and not empty: a reader, a diff and a test can all
# tell it from a real value at a glance, which a plausible-looking filler could
# not. See `Protocol.for_build`.
NOT_APPLICABLE = "n/a:build"

# Protocol components a build run has no value for. `recipe_hash` and `seed`
# are deliberately absent from this list: those a build run *does* have.
_BUILD_NOT_APPLICABLE_FIELDS: tuple[str, ...] = (
    "data_snapshot_hash",
    "tokenizer_hash",
    "backbone_commit",
)

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


class LedgerForkError(RuntimeError):
    """Two ledger files hold rows claiming one predecessor.

    Distinct from :class:`LedgerChainError`, which is about a single file being internally
    broken. A fork is the case where **every file verifies** and the history is still wrong,
    so it cannot be found by verifying files one at a time.
    """


@dataclass(frozen=True, slots=True)
class ForkBranch:
    """One side of a fork: where the row is and what it is."""

    path: Path
    line_no: int
    row_id: str
    line_hash: str


@dataclass(frozen=True, slots=True)
class ForkPoint:
    """A predecessor claimed by more than one distinct row."""

    prev_row_hash: str | None
    branches: tuple[ForkBranch, ...]

    @property
    def is_root(self) -> bool:
        """Whether the two sides disagree from the very first line.

        A root fork means these files are not two versions of one history at all; a
        non-root fork means they shared a prefix and diverged, which is the rsync case.
        """
        return self.prev_row_hash is None

    def describe(self) -> str:
        where = "; ".join(
            f"{b.path.name}:{b.line_no} ({b.row_id})" for b in self.branches
        )
        after = "from the first line" if self.is_root else f"after {self.prev_row_hash[:16]}…"
        return f"{len(self.branches)} rows claim one predecessor {after}: {where}"


def find_forks(paths: Sequence[Path]) -> list[ForkPoint]:
    """Fork points across several ledger files, newest common ancestor first.

    ``Ledger.verify_chain`` walks one file and catches a history that was edited, reordered
    or truncated. It cannot catch the failure that happens when a ledger is *copied*: rsync
    makes a second file sharing a prefix, both sides append, and now two rows name the same
    predecessor. Both files verify. ``make gates`` stays green on both machines. Nothing
    reports anything, which is the whole problem --
    ``GAP-LEDGER-NO-STORY-FOR-A-CHAIN-FORKED-ACROSS-TWO-MACHINES``.

    This does not repair anything and must not: the rows keep the hashes they were written
    with, because a ledger whose entire premise is that you do not go back and fix it cannot
    be fixed by going back and fixing it. It makes the fork **visible**, which is the part
    that was missing. Reconciling two real branches is a human decision about which runs
    happened, and it needs to be taken knowing the fork is there.

    A line present identically in both files is the shared prefix, not a fork: branches are
    keyed by the hash of the line's own bytes, so an identical row on both sides counts once.
    """
    claims: dict[str | None, dict[str, ForkBranch]] = {}
    for path in paths:
        prev: str | None = None
        for i, line in enumerate(Ledger(path).raw_lines()):
            line_hash = hashlib.sha256(line).hexdigest()
            try:
                row_id = str(json.loads(line.decode("utf-8")).get("row_id", "?"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                row_id = "<unparseable>"
            # Keyed by the line's own hash: the same row in two files is one branch, not two.
            claims.setdefault(prev, {}).setdefault(
                line_hash, ForkBranch(path=path, line_no=i + 1, row_id=row_id, line_hash=line_hash)
            )
            prev = line_hash

    forks = [
        ForkPoint(prev_row_hash=prev, branches=tuple(branches.values()))
        for prev, branches in claims.items()
        if len(branches) > 1
    ]
    # Root fork last: it is the least informative ("these are unrelated files"), and a real
    # divergence deeper in the chain is the thing a reader needs to see first.
    return sorted(forks, key=lambda f: (f.is_root, f.branches[0].line_no))


def verify_no_fork(paths: Sequence[Path]) -> None:
    """Raise :class:`LedgerForkError` naming every fork point, or return silently."""
    forks = find_forks(paths)
    if not forks:
        return
    raise LedgerForkError(
        f"{len(forks)} fork point(s) across {len(paths)} ledger file(s). "
        + " | ".join(f.describe() for f in forks)
        + ". These rows keep their hashes: a fork is reconciled by deciding which runs "
        "happened, not by rewriting the chain."
    )


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

    @classmethod
    def for_build(
        cls,
        *,
        commands: Sequence[str],
        toolchain: str,
        seed: int = 0,
    ) -> Self:
        """The protocol of a `build` run: what it actually has, and nothing else.

        A lane that compiles the workspace and runs a test suite has no data
        snapshot, no tokenizer and no backbone. Those three components carry
        :data:`NOT_APPLICABLE`. Inventing a hash for them would put a fabricated
        value in the decision record and make the row look comparable to
        training rows it has nothing to do with.

        What a build run *does* have is a command set and a toolchain, and those
        identify it well enough to be worth hashing: two build rows are
        comparable exactly when they ran the same commands on the same
        toolchain. That hash goes in ``recipe_hash``, which is therefore a real
        value and not a marker.

        ``seed`` exists because the row schema has the field. It does not make
        three build rows a promotable family — :data:`NON_PROMOTING_RUN_KINDS`
        forecloses that regardless.
        """
        commands = tuple(commands)
        if not commands or any(not c.strip() for c in commands):
            raise ValueError(
                "a build protocol needs the commands it ran, each non-empty: they are the only "
                "thing identifying one build row from another."
            )
        if not toolchain.strip():
            raise ValueError("a build protocol needs its toolchain; two toolchains are two runs")
        recipe = _canonical({"commands": list(commands), "toolchain": toolchain})
        return cls(
            data_snapshot_hash=NOT_APPLICABLE,
            tokenizer_hash=NOT_APPLICABLE,
            backbone_commit=NOT_APPLICABLE,
            recipe_hash=hashlib.sha256(recipe.encode("utf-8")).hexdigest(),
            seed=seed,
        )

    def not_applicable_fields(self) -> tuple[str, ...]:
        """Which components carry :data:`NOT_APPLICABLE`, in declaration order."""
        return tuple(
            name
            for name in ("data_snapshot_hash", "tokenizer_hash", "backbone_commit", "recipe_hash")
            if getattr(self, name) == NOT_APPLICABLE
        )


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
    def detect(cls, *, transformers_sha: str | None = None, device: str | None = None) -> Self:
        """The environment a row was produced in.

        ``device`` is the device the run **used**, and a caller that chose one must say so.
        Without it this method reports the best device the *host* offers, which is the same
        answer for every run on that host -- so a run deliberately placed on ``cpu`` for a
        comparison against ``mps`` produced two rows both claiming ``mps``, and the
        comparison was unfalsifiable from the ledger alone. Measured on this host before the
        parameter existed: three ``cpu`` runs and three ``mps`` runs, six rows, all saying
        ``device: "mps"``.

        Auto-detection remains the default because it is right for the caller that never
        chooses -- ``record_build_run`` compiles and runs a test suite, and "what this host
        is" is exactly what its row should carry.

        ``transformers_sha`` is **detected** when the caller does not pass one, for the
        same reason ``torch`` always was. The asymmetry was doing real damage: one of the two
        libraries whose version changes a result was auto-detected and always right, the
        other was a parameter defaulting to ``"unknown"`` and was right only when a caller
        remembered -- and only ``tools/real_tokenizer_pipeline.py`` did. Every ft row from
        ``tools/real_ft_run.py``, which is every training number this project has, says
        ``"unknown"`` while the Mac runs 5.12.1 and the GH200 runs 5.17.0.

        Args:
            transformers_sha: the transformers commit this run used, when the caller knows a
                sha rather than a version. Omitted, the installed version is detected.
            device: the device the run actually used, verbatim. ``None`` auto-detects the
                host's best device.
        """
        try:
            import torch as _torch

            torch_v = _torch.__version__
            if device is not None:
                detected = device
            elif _torch.cuda.is_available():
                detected = f"cuda:{_torch.cuda.device_count()}x{_torch.cuda.get_device_name(0)}"
            elif getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
                detected = "mps"
            else:
                detected = "cpu"
        except ImportError:
            torch_v = "not-installed"
            detected = device if device is not None else "cpu"
        if transformers_sha is None:
            try:
                import transformers as _transformers

                transformers_sha = f"transformers=={_transformers.__version__}"
            except ImportError:
                # Not "unknown": the question was asked and answered. A row that cannot tell
                # "absent" from "nobody looked" is the defect this change exists to fix, and
                # reproducing it one level down would be funny rather than acceptable.
                transformers_sha = "not-installed"
        return cls(
            torch=torch_v,
            transformers_sha=transformers_sha,
            device=detected,
            host=socket.gethostname(),
            fla_present=_probe_fla(),
            causal_conv1d_present=_probe_causal_conv1d(),
        )


def _tool_closure(tool_path: Path) -> dict[str, Path]:
    """The tool, and every module it imports from its own directory, transitively.

    Keyed ``"<dir>/<name>.py"`` so a tool and a package module of the same name cannot
    collide in the mapping.

    Only siblings: an import of ``qd_train`` is already covered by ``package_dir``, and an
    import of a third-party library is not this repository's source. Cycles terminate on
    ``seen`` -- ``tools/`` has none today, and a function that hung on one would be a worse
    failure than the one it is fixing.
    """
    directory = tool_path.resolve().parent
    seen: dict[str, Path] = {}
    queue = [tool_path.resolve()]
    while queue:
        current = queue.pop()
        key = f"{current.parent.name}/{current.name}"
        if key in seen:
            continue
        seen[key] = current
        tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # `node.level` non-zero is a relative import, which a tool run as a script
                # cannot have; `node.module` is None for `from . import x`.
                roots = [node.module.split(".")[0]] if node.module and not node.level else []
            elif isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            else:
                continue
            for root in roots:
                sibling = directory / f"{root}.py"
                if sibling.is_file():
                    queue.append(sibling.resolve())
    return seen


def what_ran(package_dir: Path, tool_path: Path) -> dict[str, str]:
    """``{source name: sha256}`` for a package's modules plus the tool that invoked them.

    A row's ``code_commit`` is ``"<sha>-dirty"`` for any uncommitted change at all. On the
    GH200 that one bit stood for 6894 insertions across 72 paths, on the rows that carry
    every training number this project has -- and because that box is synced by COPYING
    files into a clone pinned at an old commit, ``-dirty`` is its normal state rather than
    an exception. The suffix is present on every row and distinguishes nothing.

    Both arguments are required and neither has a default. A package with no tool records
    less than what ran, since the tool builds the plan, the supervision and the recipe; a
    default package would quietly answer for the wrong one when a fifth runner appears.

    **What to leave out.** Callers pass ``qd_train`` and not ``qd_data``: a shard header
    already carries ``qd_data``'s fingerprint and :meth:`ShardReader.open` checks it against
    sets on disk. Recording it here too would make a second owner of one answer, free to
    disagree with the first -- and the disagreement would surface as a shard-contract
    failure on a shard set that is fine. This is not an oversight to be tidied up.

    **What the tool drags in with it.** A runner imports from beside itself, and those
    modules decide what the row says. ``real_ft_run.py`` takes ``_letter_floor`` and
    ``_span_floor`` from ``ft_toy_run.py`` -- the floors every FT gate is measured against;
    both real runners take ``n_gpus_for_device`` from ``run_cost.py``, which decides
    ``n_gpus`` and therefore whether rule 4's human-yes gate can fire at all. Hashing the
    invoking file alone left every one of those invisible, which is the defect this function
    exists to close, in the function itself.

    So the closure is COMPUTED, not listed: the tool's imports of siblings in its own
    directory, followed transitively. A hand-kept list is the shape this repository spent
    2026-09-21 finding five times -- a complete enumeration written down once and then not
    maintained. Unrelated tools stay out, because a digest that moved when any tool in the
    tree changed would answer "something changed" so often that it answered nothing.
    """
    sources = dict(code_fingerprint(package_dir))
    for name, path in _tool_closure(tool_path).items():
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(sources.items()))


def what_ran_state(package_dir: Path, tool_path: Path) -> Ran:
    """:func:`what_ran` as one recordable fact: a digest to compare, and the names to read.

    The digest is the value, because that is what answers "is this the same code". The
    detail names every source and its first seven hex digits, because "something changed"
    and "``trainer.py`` changed" are different answers and only the second ends an
    investigation.
    """
    sources = what_ran(package_dir, tool_path)
    joined = "\n".join(f"{name}:{digest}" for name, digest in sources.items())
    return Ran(
        passed=True,
        value=hashlib.sha256(joined.encode()).hexdigest(),
        # Both counts, because this IS the population: every module in the package plus the
        # tool, not a sample of them. A sample size without a population reads as full
        # coverage, which is exactly the claim here and so is stated rather than implied.
        n=len(sources),
        n_total=len(sources),
        detail=" ".join(f"{name}={digest[:7]}" for name, digest in sources.items()),
    )


#: Triton releases in this half-open range compute `chunk_bwd_dqkwg` incorrectly on Hopper,
#: and `flash-linear-attention` raises rather than returning wrong gradients. Measured on a
#: GH200 (sm_90) on 2026-09-20 with the pinned `triton==3.6.0`, which `torch==2.10.0+cu128`
#: requires exactly -- so on that box the two pins cannot both be satisfied.
#:
#: **Public, and the only copy.** `stack/verify_fast_path.py` carried an identical constant
#: and an identical `_version_tuple` beside it -- found by `devmap_clones` as an Exact
#: clone. Two copies of one measured fact is how the next measurement updates one of them:
#: if fla fixes this in 3.8 and only one range moves, the gate and the ledger disagree about
#: whether a run used the fast kernel, and nothing would say which was right.
TRITON_HOPPER_BAD_RANGE: Final[tuple[tuple[int, ...], tuple[int, ...]]] = ((3, 4, 0), (3, 7, 1))


def version_tuple(v: str) -> tuple[int, ...]:
    """Leading numeric components of a version string; `()` when there are none."""
    out: list[int] = []
    for part in v.split("+")[0].split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


def _probe_fla() -> TriState:
    """Is `flash-linear-attention` importable, and can this device actually use it?

    `passed` answers the field's name -- presence. Usability goes in the detail, because
    present and usable are two quantities and this repository has been bitten by giving two
    quantities one name more than once.
    """
    try:
        import fla
    except ImportError as exc:
        return Ran(
            passed=False,
            detail=(
                f"flash-linear-attention is not importable ({exc.__class__.__name__}: {exc}); "
                "transformers falls back to its reference PyTorch gated-delta-rule, which is "
                "correct and much slower"
            ),
        )
    except Exception as exc:  # an import that raises anything at all is not usable
        return Ran(passed=False, detail=f"importing fla raised {type(exc).__name__}: {exc}")

    version = getattr(fla, "__version__", "unknown")
    try:
        import torch as _torch
        import triton as _triton
    except ImportError:
        return Ran(
            passed=True,
            detail=f"fla {version} importable; torch/triton absent so usability unchecked",
        )

    tv = version_tuple(getattr(_triton, "__version__", ""))
    lo, hi = TRITON_HOPPER_BAD_RANGE
    hopper = False
    if _torch.cuda.is_available():
        hopper = _torch.cuda.get_device_capability(0)[0] == 9
    if hopper and tv and lo <= tv < hi:
        return Ran(
            passed=True,
            detail=(
                f"fla {version} is importable but REFUSES on this device: triton "
                f"{_triton.__version__} on Hopper computes chunk_bwd_dqkwg incorrectly "
                f"(fla raises for triton in [{'.'.join(map(str, lo))}, {'.'.join(map(str, hi))})). "
                "Any run here used the reference PyTorch path, not this kernel."
            ),
        )
    return Ran(passed=True, detail=f"fla {version}, triton {getattr(_triton, '__version__', '?')}")


def _probe_causal_conv1d() -> TriState:
    """Is `causal-conv1d` importable **with** its compiled extension?

    The Python package importing proves nothing: the wheel that matters carries
    `causal_conv1d_cuda`, and without it `stack/README.md`'s "single most fragile step in
    the image" -- the from-source build -- is what produced the installed package.
    """
    try:
        import causal_conv1d
    except ImportError as exc:
        return Ran(passed=False, detail=f"causal-conv1d is not importable: {exc}")
    try:
        import causal_conv1d_cuda  # noqa: F401
    except ImportError as exc:
        return Ran(
            passed=False,
            detail=(
                f"causal_conv1d {getattr(causal_conv1d, '__version__', 'unknown')} imports but "
                f"its compiled extension does not ({exc}); the CUDA kernel is unavailable"
            ),
        )
    return Ran(
        passed=True,
        detail=(
            f"causal_conv1d {getattr(causal_conv1d, '__version__', 'unknown')} with its "
            "compiled causal_conv1d_cuda extension"
        ),
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
    #: Where `wall_clock_s` came from: ``"caller"`` if the run measured itself and handed
    #: the number over, ``"recorder"`` if the recorder's own block was the run and timed
    #: it, ``"unrecorded"`` for a row written before this field existed. Unlike
    #: `wall_clock_s`, this one carries a default, because the default is TRUE of such a
    #: row: it says nothing, and says so. Defaulting it to ``"recorder"`` instead would be
    #: an inference about 799 existing rows dressed up as a record of them.
    wall_clock_source: str = "unrecorded"
    #: The settings that produced `protocol.recipe_hash`, stored rather than only hashed.
    #:
    #: The hash makes two runs of different recipes incomparable, which is its job, and it
    #: makes neither of them readable. Every field a runner varies -- `train_subsample`,
    #: `epochs`, `batch_size`, `val_share`, `lr`, `span_weight`, `deterministic` -- went
    #: into the hash's input and was stored nowhere, so a row could not say which arm of a
    #: sweep it was. The concurrent lane recovered a three-point learning curve's labels on
    #: 2026-09-21 by re-hashing four candidate values against seven fields held at the
    #: launch command's values: correct, and it needs the launch command, which means the
    #: row was identified from the log rather than from itself.
    #:
    #: `None` for a row written before this field existed, and read back as `None` rather
    #: than `{}`: a row that says nothing about its recipe is not a row that ran without
    #: settings.
    recipe: Mapping[str, Any] | None = None

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
        if self.wall_clock_source not in _WALL_CLOCK_SOURCES:
            raise ValueError(
                f"wall_clock_source {self.wall_clock_source!r} not one of "
                f"{sorted(_WALL_CLOCK_SOURCES)}"
            )
        if not math.isfinite(self.wall_clock_s) or not math.isfinite(self.cost_usd):
            # `nan < 0` is False, so the comparison below waves NaN through. A duration
            # that is not a number is not a measurement, and a NaN here propagates into
            # every mean anyone computes over the ledger.
            raise ValueError("wall_clock_s and cost_usd are measured, finite quantities")
        if self.wall_clock_s < 0 or self.cost_usd < 0:
            raise ValueError("wall_clock_s and cost_usd are measured, non-negative quantities")
        self._validate_recipe(self.recipe)
        self._check_build_marker()

    @staticmethod
    def _validate_recipe(recipe: Mapping[str, Any] | None) -> None:
        """A recorded recipe has to be one, and has to survive the trip to JSON.

        A static method with one owner because `RunRecorder.__init__` asks the same
        question hours earlier than the row does, and asking it there is the point: a
        recipe that cannot be serialised would otherwise be discovered at `json.dumps`
        time, after the run, with the row as the casualty -- the failure mode `cost` and
        `wall_clock_s` are both validated at construction to avoid.

        An empty one is refused for a different reason. `{}` recorded as *the recipe* says
        the run had no settings, which is never true of a run that has a `recipe_hash`;
        `None` is how a row says it does not record them.
        """
        if recipe is None:
            return
        if not isinstance(recipe, Mapping) or not recipe:
            raise ValueError(
                f"recipe={recipe!r} is not a non-empty mapping. A row whose recipe is "
                "empty claims the run had no settings, and a run with a recipe_hash always "
                "had some; pass None to say this row does not record them"
            )
        bad_keys = sorted(repr(k) for k in recipe if not isinstance(k, str))
        if bad_keys:
            raise ValueError(
                f"recipe keys must be strings; got {bad_keys}. JSON has no other kind, so "
                "a non-string key would be silently rewritten on the way into the row"
            )
        try:
            _canonical(dict(recipe))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"recipe is not JSON-serialisable ({exc}). It is written into the row, so "
                "a value that cannot be serialised loses the row, not just the recipe"
            ) from exc

    def _check_build_marker(self) -> None:
        """`NOT_APPLICABLE` belongs to `build` rows, and to all of them.

        Checked in both directions on purpose. A training row carrying the
        marker has a protocol that identifies nothing, so every comparison
        against it is unfalsifiable. A build row *not* carrying it claims a data
        snapshot, a tokenizer and a backbone it never had, which makes it look
        comparable to training rows it has nothing to do with. Either way the
        row is a decision record that says something untrue, which the schema
        can refuse rather than leave to a reader to notice.
        """
        marked = set(self.protocol.not_applicable_fields())
        if self.run_kind == "build":
            missing = [f for f in _BUILD_NOT_APPLICABLE_FIELDS if f not in marked]
            if missing:
                raise ValueError(
                    f"run_kind 'build' must carry {NOT_APPLICABLE!r} in {missing}: a build run has "
                    "no data snapshot, tokenizer or backbone, and a row that names one is "
                    "claiming to be comparable to training rows it has nothing to do with."
                )
        elif marked:
            raise ValueError(
                f"run_kind {self.run_kind!r} carries the build marker {NOT_APPLICABLE!r} in "
                f"{sorted(marked)}. Only a 'build' row may: a training row with an unidentified "
                "protocol component makes every comparison against it meaningless."
            )

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
            "wall_clock_source": self.wall_clock_source,
            "cost_usd": self.cost_usd,
            "notes": self.notes,
            "recipe": dict(self.recipe) if self.recipe is not None else None,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        proto = Protocol(**raw["protocol"])
        stated = raw.get("protocol_hash")
        if stated is not None and stated != proto.hash():
            raise LedgerChainError(
                f"row {raw.get('row_id')}: stated protocol_hash {stated} does not match "
                f"the hash of the stated protocol components {proto.hash()}. "
                "The row was edited."
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
            controls={
                k: parse_tristate(v, field=f"controls.{k}") for k, v in raw["controls"].items()
            },
            gates={k: parse_tristate(v, field=f"gates.{k}") for k, v in raw["gates"].items()},
            wall_clock_s=raw["wall_clock_s"],
            # `.get` rather than `[...]`: rows written before this field existed are valid
            # rows, and they read back as "unrecorded" rather than being assigned a source
            # nobody wrote down.
            wall_clock_source=raw.get("wall_clock_source", "unrecorded"),
            cost_usd=raw["cost_usd"],
            notes=raw.get("notes", ""),
            # Absent and null both mean the same thing and both read back as None: this row
            # does not record its recipe. Every row written before 2026-09-21 is in that
            # position, and none of them should read as having run with no settings.
            recipe=raw.get("recipe"),
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
                    f"previous line hashes to {prev_hash!r}. "
                    "History was edited, reordered, or truncated."
                )
            LedgerRow.from_json(obj)  # re-validates the protocol hash and every tri-state
            prev_hash = hashlib.sha256(line).hexdigest()

    # -- writing ---------------------------------------------------------

    def _take_write_lock(self, fd: int, timeout_s: float) -> None:
        """Block until this process owns the append, or refuse.

        O_APPEND makes each ``write`` atomic, so two writers cannot tear a line.
        It says nothing about the *chain*: ``prev_row_hash`` is read before the
        write, and a second writer that reads the same last line produces two
        rows claiming one predecessor. ``verify_chain`` then refuses the file
        from that point on, for good, on a log whose entire premise is that you
        do not go back and fix it. Measured on this repo before the lock:
        6 processes x 5 rows produced `line 2 (row w2-0): prev_row_hash is None
        but the previous line hashes to efbdd1ff...`
        (``test_concurrent_appends_do_not_break_the_chain``).

        The lock is advisory and process-wide via ``flock``; it binds writers
        that go through this method and nothing else. A hand-edited file is
        still caught, but by the chain, not by the lock.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise LedgerChainError(
                        f"could not take the write lock on {self.path} within {timeout_s:g}s; "
                        "another writer holds it. Refusing to append rather than racing it: "
                        "two writers that read the same last row produce a chain that never "
                        "verifies again."
                    ) from None
                time.sleep(0.005)

    def append(self, row: LedgerRow, *, lock_timeout_s: float = 60.0) -> LedgerRow:
        if lock_timeout_s <= 0:
            raise ValueError(f"lock_timeout_s must be positive, got {lock_timeout_s!r}")
        # O_APPEND so concurrent writers cannot interleave a partial line, and
        # fsync so a row survives the crash that a killed run is recording. The
        # flock covers read-predecessor-then-write, which O_APPEND does not.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            self._take_write_lock(fd, lock_timeout_s)
            if any(r.row_id == row.row_id for r in self.rows()):
                raise ValueError(f"row_id {row.row_id} already present: the ledger is append-only")
            row.prev_row_hash = self.last_line_hash()
            line = _canonical(row.to_json()).encode("utf-8")
            os.write(fd, line + b"\n")
            os.fsync(fd)
        finally:
            # If the lock was never taken, unlocking fails harmlessly; closing
            # the descriptor releases it either way.
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        return row

    # -- promotion -------------------------------------------------------

    def promotion_verdict(self, seed_family: str) -> PromotionVerdict:
        """May the rows sharing this seed family promote a decision?

        Every condition in docs/ledger-schema.md, each refusal itemized.
        Condition 4 is the one that matters most: a `not_run` gate BLOCKS promotion.
        It does not pass it, and it does not quietly drop out of the conjunction.

        Condition 7 is coverage. A gate or control that states `n`/`n_total` and saw
        fewer than all eligible items refuses, rather than promoting on a capped
        sample. Unstated coverage (`n is None`) is not treated as partial.
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
                reasons.append(
                    f"{r.row_id}: marked quick ({r.quick_reason}); quick runs cannot promote"
                )
            # Rule 8's "truncated schedule", derived rather than taken on trust. A run the
            # cap stopped did not finish its schedule, whatever `quick` says about it.
            termination = r.metrics.get("train.termination")
            if termination is not None and termination.value == "wall_clock_cap":
                reasons.append(
                    f"{r.row_id}: train.termination is 'wall_clock_cap'; the wall-clock cap "
                    "stopped this run before its schedule finished, which rule 8 calls a "
                    "truncated schedule. quick runs cannot promote, and this is one whether "
                    f"or not the row says so (quick={r.quick})"
                )
            elif termination is None and r.run_kind in TRAINING_RUN_KINDS:
                reasons.append(
                    f"{r.row_id}: run_kind {r.run_kind!r} trains, but the row carries no "
                    "train.termination; a run that does not say how it ended cannot be "
                    "shown to have finished its schedule, and an absent answer is not a "
                    "passed one"
                )
            if r.run_kind in NON_PROMOTING_RUN_KINDS:
                # Stated on the run kind, not inferred from empty gates. A build
                # row's gates are vacuous rather than failed, and a conjunction
                # over vacuous inputs is the shape that quietly comes out true.
                reasons.append(
                    f"{r.row_id}: run_kind {r.run_kind!r} never promotes a decision; its gates "
                    "are vacuous, not satisfied"
                )

        seeds = {r.protocol.seed for r in candidates}
        if len(seeds) < 3:
            reasons.append(
                f"{len(seeds)} distinct seed(s) {sorted(seeds)}; promotion needs three rows "
                "differing only in seed"
            )

        for r in candidates:
            for gate in REQUIRED_GATES:
                g = r.gates.get(gate)
                if g is None:
                    reasons.append(
                        f"{r.row_id}: gate {gate!r} absent; an absent gate is not a passed gate"
                    )
                elif isinstance(g, NotRun):
                    reasons.append(
                        f"{r.row_id}: gate {gate!r} did not run ({g.reason}); "
                        "this blocks promotion"
                    )
                elif not g.passed:
                    reasons.append(
                        f"{r.row_id}: gate {gate!r} ran and FAILED [{g.coverage_str()}]"
                    )
                elif _states_partial_coverage(g):
                    reasons.append(
                        f"{r.row_id}: gate {gate!r} passed on only {g.coverage_str()} of the "
                        "eligible population; a capped sample is not complete coverage "
                        "and does not promote"
                    )

            for ctl in REQUIRED_CONTROLS:
                c = r.controls.get(ctl)
                if c is None:
                    reasons.append(f"{r.row_id}: control {ctl!r} absent")
                elif isinstance(c, NotRun):
                    reasons.append(f"{r.row_id}: control {ctl!r} did not run ({c.reason})")
                elif not c.passed:
                    reasons.append(
                        f"{r.row_id}: control {ctl!r} ran and FAILED [{c.coverage_str()}]"
                    )
                elif _states_partial_coverage(c):
                    reasons.append(
                        f"{r.row_id}: control {ctl!r} passed on only {c.coverage_str()} of the "
                        "eligible population; a capped sample is not complete coverage "
                        "and does not promote"
                    )

        if reasons:
            return PromotionVerdict(False, tuple(reasons), ids)
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
                f"{len(candidates)} completed rows, seeds {sorted(seeds)}, "
                "every gate and control ran and passed at complete coverage "
                f"(weakest stated: {coverage})",
            ),
            ids,
        )


def _states_partial_coverage(result: Ran) -> bool:
    """True when a result states its coverage and that coverage is short of the population.

    Unstated coverage (``n is None``) is not partial. Many gates are a single
    observation with no population to sample, and refusing those would make
    promotion unreachable. ``Ran.is_complete_coverage`` answers False for unstated
    coverage, which is the right answer to "did this see everything?" and the wrong
    condition to refuse on, so the two questions stay separate.
    """
    return result.n is not None and not result.is_complete_coverage


def _is_measured_duration(wall_clock_s: float) -> bool:
    """Whether a number is a duration somebody measured.

    Asked in two places -- at construction, and by ``measured()`` -- and stated once
    because the subtlety is not the bound. ``nan < 0`` is False, so a bare bound check
    admits NaN, and a NaN duration carries into ``cost_usd`` as a NaN price that passes
    every finiteness check downstream because it was never finite to begin with.
    """
    return math.isfinite(wall_clock_s) and wall_clock_s >= 0


class RunRecorder:
    """Context manager guaranteeing a row on every exit path.

    A run that raises writes ``status='failed'``. A run killed by SIGTERM/SIGINT
    writes ``status='killed'``. A run that completes writes ``status='completed'``.
    There is no path out of the ``with`` block that leaves the ledger silent —
    which is the point: "a run whose row is missing is rerun, not remembered",
    and an unwritten row is indistinguishable from a run that never happened.

    ``wall_clock_s`` is required and has no default, because no default is right.
    ``None`` means "this ``with`` block contains the run, time it yourself"; a number
    means "I measured the run, record this". The recorder cannot tell the two apart from
    the inside — a recorder entered after the work times the reporting, and reports
    microseconds for a run that took minutes. 346 of the first 799 rows written to this
    ledger carried a ``wall_clock_s`` under 0.1s for exactly that reason, and ``cost_usd``
    is derived from the same number, so those rows priced GPU time at zero.

    One tool can need both answers and a third. ``tools/real_ft_run.py`` passes ``None``
    from ``_train``, whose block contains ``train_ft``; and a measured decode elapsed from
    ``_record_verdict``, which describes work that finished before the recorder existed.
    Its verdict row must NOT repeat the parent's duration: 186 ft rows and 162 verdict rows
    each claiming the same seconds would sum to twice the GPU time actually spent, which is
    a worse defect than the one this argument fixes.
    """

    def __init__(
        self,
        ledger: Ledger,
        *,
        protocol: Protocol,
        run_kind: RunKind,
        repo: str | os.PathLike[str],
        env: Environment | None = None,
        wall_clock_s: float | None,
        cost: CostEstimate | None,
        quick: bool = False,
        quick_reason: str | None = None,
        notes: str = "",
        recipe: Mapping[str, Any] | None = None,
        entry_point: str | os.PathLike[str] | None = None,
    ) -> None:
        if run_kind not in _RUN_KINDS:
            raise ValueError(f"run_kind {run_kind!r} not one of {sorted(_RUN_KINDS)}")
        if wall_clock_s is not None and not _is_measured_duration(wall_clock_s):
            # Refused here rather than in `_finish`, which runs only once the GPU time has
            # already been spent -- and which would take the row down with it.
            raise ValueError(
                f"wall_clock_s={wall_clock_s!r} is not a measured duration: pass a finite, "
                "non-negative number of seconds, or None if this recorder's block contains "
                "the run and should be timed itself"
            )
        self.ledger = ledger
        self.protocol = protocol
        self.run_kind = run_kind
        self.repo = Path(repo)
        self.env = env or Environment.detect()
        self.quick = quick
        self.quick_reason = quick_reason
        self.cost = cost
        self.wall_clock_s = wall_clock_s
        self.notes = notes
        # Validated here rather than at `_finish`, for the same reason `cost` and
        # `wall_clock_s` are: a recipe that cannot be serialised would otherwise be found
        # once the run was over, and would take the row with it.
        LedgerRow._validate_recipe(recipe)
        self.recipe = recipe
        if cost is None and self.env.device not in CostEstimate.LOCAL_DEVICES:
            # The GH200 case, made impossible rather than discouraged: 13 rung 0 rows
            # recorded cost_usd 0.0 for real GPU hours because nothing required a rate.
            # `cost=None` is how a local run says "nothing is billed here"; on hardware paid
            # for by the hour it is an omission, and an omission that reads as $0.00 is
            # indistinguishable from a measured zero. Refused at construction, because
            # refusing in `_finish` would refuse once the hour had already been spent and
            # would take the row down with it.
            raise ValueError(
                f"device {self.env.device!r} is not one of "
                f"{sorted(CostEstimate.LOCAL_DEVICES)}, so this run is billed by the hour "
                "and its row cannot omit the cost. Pass the CostEstimate the run was gated "
                "on (RunControl.cost), or CostEstimate.for_device(...) -- which prices a "
                "local device at zero and refuses to invent a rate for anything else"
            )

        # The invoking tool, so `__enter__` can pin the code BEFORE the work starts.
        # Every tool used to record `code_that_ran` at the end of its training block, which
        # meant a run that died never recorded it: `_on_signal` writes the row immediately,
        # and the digest was not among the metrics yet. That is
        # GAP-LEDGER-A-KILLED-RUN-LOSES-ITS-PROVENANCE, found on a real row --
        # gh200-commitpackft-2026-09-22.jsonl, an ft run terminated by SIGTERM with
        # `code_that_ran: None`. Precisely the runs that ended abnormally, whose
        # circumstances most need pinning, were the ones pinned only by `code_commit` --
        # which on the rented box is a permanently `-dirty` string that distinguishes
        # nothing. The digest never depends on the outcome, so there was never a reason to
        # compute it late.
        self.entry_point = None if entry_point is None else Path(entry_point)

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

    def measured(self, wall_clock_s: float) -> None:
        """State the duration this run measured, from **inside** the block.

        ``wall_clock_s`` is a constructor argument because most callers know the duration
        before the recorder exists -- they time the work, then open a recorder to report it.
        A recorder that WRAPS its work cannot do that, and wrapping is the only arrangement
        in which a killed run still writes a row: the guarantee in this class's docstring is
        a property of the block, so work done outside it is work whose death goes
        unrecorded.

        The two paths then differ exactly as they should. A run that reaches this line
        reports the duration it measured, under ``wall_clock_source="caller"`` -- the same
        quantity it would have passed to the constructor, so rows do not change meaning. A
        run that dies first falls back to the recorder's own lifetime under ``"recorder"``,
        which is what it was billed for, and the two are distinguishable in the row by that
        field alone.

        Refused after the row is written, where it changes nothing; refused a second time,
        because a recorder told twice keeps only the last, which is how a per-seed duration
        silently becomes one seed's.
        """
        if self.row is not None:
            raise ValueError(
                "measured() after the row is written changes nothing: the duration is read "
                "at _finish, which has already run"
            )
        if not _is_measured_duration(wall_clock_s):
            raise ValueError(
                f"wall_clock_s={wall_clock_s!r} is not a measured duration: pass a finite, "
                "non-negative number of seconds"
            )
        if self.wall_clock_s is not None:
            raise ValueError(
                f"this run's duration was already stated as {self.wall_clock_s!r}s. A "
                "recorder that is told twice records only the last figure, which is how a "
                "loop's total becomes its final iteration's"
            )
        self.wall_clock_s = float(wall_clock_s)

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
        if self.entry_point is not None:
            # First, before the handlers are even installed. A digest computed here is on
            # the row whether the block completes, raises, or is killed mid-step.
            self.metric(
                "code_that_ran",
                what_ran_state(self.repo / "python" / "qd_train", self.entry_point),
            )
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
            except ValueError:
                # Not on the main thread; the with-block's finally still writes a row.
                pass
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        for sig, handler in self._prev_handlers.items():
            # The same ValueError __enter__ tolerates: off the main thread the
            # restore is refused, which is no reason to fail an exit that still
            # has a row to write.
            with contextlib.suppress(ValueError):
                signal.signal(sig, handler)
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
        # `self.wall_clock_s` is the caller saying it measured the run itself. Only the
        # caller can know: a recorder entered after the work times the reporting, not the
        # run, and cannot tell the difference from the inside.
        wall = (
            self.wall_clock_s
            if self.wall_clock_s is not None
            else max(0.0, time.monotonic() - self._t0)
        )
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
                wall_clock_source="recorder" if self.wall_clock_s is None else "caller",
                # From the same object that gated the launch, not a duplicate float beside
                # it. trainer.py already computed exactly this and put it in TrainResult
                # rather than the row, so the correct number existed and was discarded.
                cost_usd=0.0 if self.cost is None else self.cost.cost_for(wall),
                notes=note,
                recipe=self.recipe,
            )
        )


# ---------------------------------------------------------------------------
# The recording path.
#
# `run_kind: "build"` made an honest row possible. It did not make one easy, and
# the measured result was that nobody wrote one: `ledger/` stayed empty while
# four lanes printed command output into their handoffs and flagged that repo
# rule 5 was unsatisfied. Between the run kind and a written row sat a Protocol,
# an Environment, a Ledger, a RunRecorder, two harnesses' output formats, and a
# judgement call about what a suite that never launched should look like. Every
# one of those is a place to get it wrong, and all of them together are more
# work than pasting a terminal line.
#
# So the path below is one command. Everything here exists to make the honest
# row the cheap one.
# ---------------------------------------------------------------------------


class SuiteFailure(RuntimeError):
    """At least one recorded suite ran and failed. Carried into the row's notes."""


# A suite gets this long before it is killed and recorded as not-run. Bounded
# because an unbounded wait turns "the ledger records every run" into "the lane
# hangs and records nothing".
DEFAULT_SUITE_TIMEOUT_S = 1800.0

# Captured output is capped and the *tail* is kept: every harness here prints its
# summary last. Without a cap a runaway suite's output is an unbounded allocation
# inside the thing whose job is to survive the run.
MAX_CAPTURED_OUTPUT_BYTES = 8 * 1024 * 1024

# How much of the output tail a row's `detail` may carry. The row is a record,
# not a log; the full output belongs to the terminal that produced it.
MAX_DETAIL_CHARS = 240


@dataclass(frozen=True, slots=True)
class SuiteCounts:
    """What a harness said it did, in the three buckets a coverage pair needs.

    ``ran_ok`` and ``ran_failed`` executed. ``not_run`` was collected and did
    not: skipped, ignored, deselected, filtered out. Keeping that third bucket
    separate is the whole point — it is the difference between ``344/344`` and
    ``12/344``, and a run whose collection silently shrank shows up as the
    second rather than as a smaller, cleaner-looking pass count.
    """

    ran_ok: int
    ran_failed: int
    not_run: int
    # False when the harness said it stopped before reaching the end of the run.
    # The tests in binaries cargo never built, or in files pytest never reached
    # after `-x`, are not `ignored` and not `skipped` — the harness does not
    # mention them at all. Summing what it printed then yields n == n_total for a
    # run that covered a fraction of the suite. Found on the first real row, not
    # by reasoning: `coverage=301/301` against a 347-test baseline.
    collection_complete: bool = True

    def __post_init__(self) -> None:
        for name in ("ran_ok", "ran_failed", "not_run"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"SuiteCounts.{name} must be a non-negative int, got {v!r}")
        if not isinstance(self.collection_complete, bool):
            raise TypeError("SuiteCounts.collection_complete must be bool")

    @property
    def n(self) -> int:
        """Tests that executed."""
        return self.ran_ok + self.ran_failed

    @property
    def n_total(self) -> int:
        """Tests that were collected."""
        return self.n + self.not_run

    def as_tristate(self, *, exit_code: int, detail: str = "") -> Ran:
        """A suite that reported counts. ``passed`` needs both signals.

        A harness can exit non-zero having reported zero failures — a collection
        error, a plugin that blew up during teardown, a linker failure after the
        tests themselves were fine. Trusting the counts alone would record that
        as green, so the exit code has a veto.

        When the run was cut short, the counts survive and the **coverage pair
        does not**. ``n``/``n_total`` is a claim about the whole eligible
        population, and nobody measured that population: the harness stopped
        before it knew. ``Ran`` carries the pair together or not at all, so
        dropping it is exactly right — ``is_complete_coverage`` then answers
        False and ``coverage_str()`` says "coverage unstated", which is the
        truth. Keeping ``value`` keeps what *was* measured.
        """
        passed = exit_code == 0 and self.ran_failed == 0
        if self.collection_complete:
            return Ran(
                passed=passed,
                value=self.ran_ok,
                n=self.n,
                n_total=self.n_total,
                detail=detail[:MAX_DETAIL_CHARS],
            )
        note = (
            f"run aborted before the end: {self.n} test(s) reported, total eligible never "
            f"established, so coverage is unstated rather than {self.n}/{self.n_total}"
        )
        return Ran(
            passed=passed,
            value=self.ran_ok,
            detail=f"{note}. {detail}"[:MAX_DETAIL_CHARS],
        )


# `cargo test` prints one of these per test binary, not one per invocation, and
# the last one is usually the doc-tests' `0 passed`. A parser that reads only the
# final line reports zero for a green workspace — wrong in the direction that
# looks harmless, which is the direction nobody re-checks.
_CARGO_RESULT_RE = re.compile(
    r"^test result:\s+\S+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed;\s+(\d+)\s+ignored;"
    r"\s+(\d+)\s+measured;\s+(\d+)\s+filtered out",
    re.MULTILINE,
)

# pytest's summary line, with or without the `=` banner: "1104 passed, 7 skipped
# in 29.02s". Matched by requiring a duration on the same line, so a count that
# happens to appear in a test's own output is not mistaken for the summary.
_PYTEST_DURATION_RE = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")
_PYTEST_TOKEN_RE = re.compile(
    r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b"
)

# Each harness's own announcement that it stopped early. Detected rather than
# inferred: a count that looks small and a run that was cut short are different
# facts, and only the harness knows which one happened.
_CARGO_ABORT_RE = re.compile(
    r"^error: (test failed, to rerun pass|could not compile|build failed)", re.MULTILINE
)
_PYTEST_ABORT_RE = re.compile(r"stopping after \d+ failures?|Interrupted:", re.IGNORECASE)


def parse_cargo_test_output(text: str) -> SuiteCounts | None:
    """Sum every `test result:` line, or return ``None`` if there were none.

    ``None`` rather than zeroes: an empty workspace really can report zero of
    everything, so a parser that returns zeroes when it simply did not recognise
    the output has manufactured a measurement its caller cannot distinguish from
    a real one. A compile failure produces no summary at all, and that is the
    case this distinction exists for.
    """
    matches = _CARGO_RESULT_RE.findall(text)
    if not matches:
        return None
    ran_ok = ran_failed = not_run = 0
    for passed, failed, ignored, _measured, filtered in matches:
        ran_ok += int(passed)
        ran_failed += int(failed)
        # `ignored` and `filtered out` were collected and did not execute.
        # `measured` is a bench figure, not a test, and is counted nowhere.
        not_run += int(ignored) + int(filtered)
    return SuiteCounts(
        ran_ok=ran_ok,
        ran_failed=ran_failed,
        not_run=not_run,
        # cargo's default is fail-fast: it stops after the first failing test
        # binary and never builds the rest, so the tests in them appear in no
        # summary line at all.
        collection_complete=_CARGO_ABORT_RE.search(text) is None,
    )


def parse_pytest_output(text: str) -> SuiteCounts | None:
    """The last summary line pytest printed, or ``None``.

    ``xpassed`` and ``xfailed`` executed, so they count as run. ``deselected``
    and ``skipped`` were collected and did not, so they widen ``n_total`` — a
    `-k` filter is then visible in the row instead of shrinking the denominator
    out of sight.
    """
    candidate: str | None = None
    for line in text.splitlines():
        if _PYTEST_DURATION_RE.search(line) and _PYTEST_TOKEN_RE.search(line):
            candidate = line
    if candidate is None:
        return None
    ran_ok = ran_failed = not_run = 0
    for count, word in _PYTEST_TOKEN_RE.findall(candidate):
        value = int(count)
        if word in ("passed", "xpassed", "xfailed"):
            ran_ok += value
        elif word in ("failed", "error", "errors"):
            ran_failed += value
        else:  # skipped, deselected
            not_run += value
    return SuiteCounts(
        ran_ok=ran_ok,
        ran_failed=ran_failed,
        not_run=not_run,
        # `-x` and a collection error both leave the rest of the suite
        # unmentioned rather than reported as skipped.
        collection_complete=_PYTEST_ABORT_RE.search(text) is None,
    )


SUITE_PARSERS: dict[str, Callable[[str], SuiteCounts | None]] = {
    "cargo": parse_cargo_test_output,
    "pytest": parse_pytest_output,
}


def _parse_counts(text: str, parser: str) -> SuiteCounts | None:
    if parser == "auto":
        for fn in SUITE_PARSERS.values():
            counts = fn(text)
            if counts is not None:
                return counts
        return None
    try:
        return SUITE_PARSERS[parser](text)
    except KeyError:
        raise ValueError(
            f"unknown parser {parser!r}; known: {['auto', *sorted(SUITE_PARSERS)]}"
        ) from None


# Shell operators, as shlex tokenizes them when `punctuation_chars` is on.
_SHELL_OPERATORS = frozenset({"|", "||", "&", "&&", ";", ";;", "<", ">", ">>", "<<", "(", ")"})


def parse_command(text: str) -> tuple[str, ...]:
    """Split a command string into argv, refusing anything that needs a shell.

    Suites are run without a shell, so a pipeline here would not pipe: `cargo
    test --workspace | tail -5` would hand cargo `|` as a test-name filter and
    cargo would exit 0 having run nothing. That is a green row for a run that
    did not happen, which is the precise failure this module exists to prevent —
    and it is the same trap this repo's own harness notes call out ("never pipe
    a command whose exit code you need"). Refused loudly instead.
    """
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        argv = tuple(lexer)
    except ValueError as exc:  # unbalanced quotes
        raise ValueError(f"could not split command {text!r}: {exc}") from exc
    if not argv:
        raise ValueError("empty command: a suite needs something to run")
    operators = [tok for tok in argv if tok in _SHELL_OPERATORS]
    if operators:
        raise ValueError(
            f"command {text!r} contains the shell pipeline/redirect operator(s) "
            f"{operators}, and suites are run without a shell. A pipeline would swallow the "
            "exit code and a redirect would hide the output this recorder parses; either way "
            "the row would describe a run that did not happen. Record the bare command."
        )
    return argv


@dataclass(frozen=True, slots=True)
class SuiteOutcome:
    """One suite, as run. ``result`` is what goes into the row."""

    name: str
    command: tuple[str, ...]
    exit_code: int | None
    duration_s: float
    result: TriState
    output_tail: str


def run_suite(
    name: str,
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str],
    timeout_s: float = DEFAULT_SUITE_TIMEOUT_S,
    parser: str = "auto",
    max_output_bytes: int = MAX_CAPTURED_OUTPUT_BYTES,
) -> SuiteOutcome:
    """Run one suite and classify what came back. Four outcomes, deliberately.

    * **could not be launched** -> ``NotRun``. No binary, no result. Recording a
      zero here is how a missing toolchain becomes a clean sweep.
    * **timed out** -> ``NotRun``. A suite that was cut off produced no result;
      it did not produce a bad one.
    * **counts parsed** -> ``Ran``, with the coverage pair and the exit code's
      veto (see :meth:`SuiteCounts.as_tristate`).
    * **no counts** -> it depends on the exit code, and the asymmetry is the
      point. Non-zero with no summary is a command that ran and failed — a
      compile error is a real failure and calling it "not run" would let a
      broken build sit in the record as "nothing was measured here". Zero with
      no summary is the dangerous shape: pytest exits 0 on "no tests ran" under
      some configurations, so a pass with no counts cannot be told from a suite
      that collected nothing, and it is recorded as ``NotRun``.
    """
    command = tuple(command)
    if not command:
        raise ValueError(f"suite {name!r} has no command")
    if timeout_s <= 0:
        raise ValueError(f"suite {name!r}: timeout_s must be positive, got {timeout_s!r}")
    if max_output_bytes <= 0:
        raise ValueError(f"suite {name!r}: max_output_bytes must be positive")
    if parser != "auto" and parser not in SUITE_PARSERS:
        raise ValueError(f"unknown parser {parser!r}; known: {['auto', *sorted(SUITE_PARSERS)]}")

    t0 = time.monotonic()
    try:
        # argv, never a shell: see parse_command for why that is load-bearing.
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        return SuiteOutcome(
            name=name,
            command=command,
            exit_code=None,
            duration_s=time.monotonic() - t0,
            result=NotRun(
                reason=f"suite {name!r} could not be launched ({command[0]!r}): "
                f"{type(exc).__name__}: {exc}"
            ),
            output_tail="",
        )

    chunks: list[bytes] = []

    def _drain() -> None:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                return
            chunks.append(chunk)
            if sum(len(c) for c in chunks) > max_output_bytes:
                tail = b"".join(chunks)[-max_output_bytes:]
                chunks.clear()
                chunks.append(tail)

    reader = threading.Thread(target=_drain, name=f"suite-{name}-reader", daemon=True)
    reader.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        # A process that ignores SIGKILL is the kernel's problem, not the
        # ledger's; the row already says the suite did not finish.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=30)
    reader.join(timeout=30)
    if proc.stdout is not None:
        proc.stdout.close()

    duration = time.monotonic() - t0
    text = b"".join(chunks).decode("utf-8", errors="replace")
    tail = text[-MAX_DETAIL_CHARS:]

    if timed_out:
        return SuiteOutcome(
            name=name,
            command=command,
            exit_code=proc.returncode,
            duration_s=duration,
            result=NotRun(
                reason=f"suite {name!r} timed out after {timeout_s:g}s and was killed; "
                "a suite that was cut off has no result"
            ),
            output_tail=tail,
        )

    exit_code = proc.returncode
    counts = _parse_counts(text, parser)
    if counts is not None:
        result: TriState = counts.as_tristate(
            exit_code=exit_code, detail=f"exit {exit_code}; {' '.join(command)}"
        )
    elif exit_code == 0:
        result = NotRun(
            reason=f"suite {name!r} exited 0 but its output held no parsable test summary "
            f"(parser {parser!r}); a pass with no counts cannot be told from a suite that "
            "collected nothing"
        )
    else:
        result = Ran(
            passed=False,
            detail=f"exit {exit_code}; no parsable test summary (parser {parser!r}) — "
            f"tail: {tail}"[:MAX_DETAIL_CHARS],
        )
    return SuiteOutcome(
        name=name,
        command=command,
        exit_code=exit_code,
        duration_s=duration,
        result=result,
        output_tail=tail,
    )


def record_build_run(
    *,
    ledger: Ledger,
    repo: str | os.PathLike[str],
    suites: Sequence[tuple[str, Sequence[str]]],
    toolchain: str,
    cwd: str | os.PathLike[str] | None = None,
    env: Environment | None = None,
    timeout_s: float = DEFAULT_SUITE_TIMEOUT_S,
    parser: str = "auto",
    quick: bool = False,
    quick_reason: str | None = None,
    notes: str = "",
) -> LedgerRow:
    """Run the suites, write one `build` row, and hand back the row to cite.

    The row is written on every exit path, including a suite that fails: a
    failed verification that leaves no row is indistinguishable from one that
    was never attempted, and this module exists to make that impossible. A
    failing suite therefore produces ``status='failed'`` and a returned row, not
    an exception — the caller decides what to do about the exit code, and the
    CLI below exits non-zero.
    """
    resolved = tuple((name, tuple(command)) for name, command in suites)
    if not resolved:
        raise ValueError("a build row records at least one suite; a row with none records nothing")
    names = [name for name, _ in resolved]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate suite name(s) {duplicates}: one metric key per suite, or the second "
            "result silently replaces the first"
        )
    for name, command in resolved:
        if not name.strip():
            raise ValueError("every suite needs a name; it becomes the metric key")
        if not command:
            raise ValueError(f"suite {name!r} has no command")

    commands = [shlex.join(command) for _, command in resolved]
    protocol = Protocol.for_build(commands=commands, toolchain=toolchain)
    recorder = RunRecorder(
        ledger,
        protocol=protocol,
        run_kind="build",
        repo=repo,
        env=env,
        # This recorder's block contains the suites, so its own lifetime IS the run's.
        wall_clock_s=None,
        # A build row is `make gates` on the machine the work is already being done on, so
        # the honest answer is a priced zero rather than an omitted one: `for_device`
        # returns zero for cpu and refuses to invent a rate for anything else, so if this
        # ever runs somewhere rented it fails loudly instead of recording $0.00.
        cost=CostEstimate.for_device(cap=WallClockCap(cap_s=BUILD_CAP_S), device="cpu"),
        # The same two things `Protocol.for_build` hashed. A build row's whole identity is
        # which commands ran on which toolchain, and until now the row carried the hash of
        # that and not the commands -- so "did the gates run the suite I think they did"
        # could only be answered by re-hashing candidates.
        recipe={"commands": commands, "toolchain": toolchain},
        quick=quick,
        quick_reason=quick_reason,
        notes=notes,
    )
    outcomes: list[SuiteOutcome] = []
    try:
        with recorder as rec:
            for name, command in resolved:
                outcome = run_suite(
                    name,
                    command,
                    cwd=cwd if cwd is not None else repo,
                    timeout_s=timeout_s,
                    parser=parser,
                )
                outcomes.append(outcome)
                rec.metric(f"suite.{name}", outcome.result)
            failed = [
                o.name for o in outcomes if isinstance(o.result, Ran) and not o.result.passed
            ]
            if failed:
                raise SuiteFailure(f"suite(s) ran and failed: {', '.join(failed)}")
    except SuiteFailure:
        # Swallowed on purpose: the failure is now *in the row*, which is the
        # record that matters. Re-raising would make the caller choose between
        # handling it and losing the row id it needs to cite.
        pass
    row = recorder.row
    if row is None:  # pragma: no cover - RunRecorder guarantees a row on every path
        raise LedgerChainError("RunRecorder exited without writing a row")
    return row


# ---------------------------------------------------------------------------
# CLI: the one command a lane runs.
# ---------------------------------------------------------------------------

# Exit codes are three, not two, for the same reason the tri-state is three: a
# suite that could not run must not leave the same trace as one that ran and
# passed, and `$?` is a trace.
EXIT_OK = 0
EXIT_SUITE_FAILED = 1
EXIT_SUITE_NOT_RUN = 3


def _split_suite_argument(raw: str) -> tuple[str, tuple[str, ...]]:
    name, sep, command = raw.partition("=")
    if not sep or not name.strip() or not command.strip():
        raise ValueError(f"--suite expects NAME=COMMAND, got {raw!r}")
    return name.strip(), parse_command(command)


def _cmd_record(args: argparse.Namespace) -> int:
    suites = [_split_suite_argument(raw) for raw in args.suite]
    ledger = Ledger(args.ledger)
    row = record_build_run(
        ledger=ledger,
        repo=args.repo,
        suites=suites,
        toolchain=args.toolchain,
        cwd=args.cwd or args.repo,
        timeout_s=args.timeout,
        parser=args.parser,
        quick=args.quick,
        quick_reason=args.quick_reason,
        notes=args.notes,
    )

    print(f"run_kind=build status={row.status} code_commit={row.code_commit}", file=sys.stderr)
    print(f"protocol_hash={row.protocol_hash}", file=sys.stderr)
    print(f"wall_clock_s={row.wall_clock_s:.2f} ledger={ledger.path}", file=sys.stderr)
    exit_code = EXIT_OK
    for name, _ in suites:
        result = row.metrics[f"suite.{name}"]
        if isinstance(result, NotRun):
            print(f"  suite.{name}: NOT RUN — {result.reason}", file=sys.stderr)
            exit_code = max(exit_code, EXIT_SUITE_NOT_RUN)
        else:
            verdict = "passed" if result.passed else "FAILED"
            print(
                f"  suite.{name}: {verdict} value={result.value} "
                f"coverage={result.coverage_str()}",
                file=sys.stderr,
            )
            if not result.passed:
                exit_code = EXIT_SUITE_FAILED
    # The row id, alone, on stdout: `ID=$(... record ...)` has to work.
    print(row.row_id)
    return exit_code


def _cmd_verify(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    count = len(ledger.raw_lines())
    try:
        ledger.verify_chain()
    except LedgerChainError as exc:
        print(f"CHAIN BROKEN in {ledger.path} ({count} line(s)): {exc}", file=sys.stderr)
        return 1
    print(f"chain verifies: {count} row(s) in {ledger.path}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    for row in ledger.rows():
        if row.row_id == args.row_id:
            print(json.dumps(row.to_json(), indent=2, sort_keys=True))
            return 0
    print(f"no row {args.row_id!r} in {ledger.path}", file=sys.stderr)
    return 1


def _cmd_verdict(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    family = args.seed_family
    if family is None:
        match = [r for r in ledger.rows() if r.row_id == args.row_id]
        if not match:
            print(f"no row {args.row_id!r} in {ledger.path}", file=sys.stderr)
            return 1
        family = match[0].protocol.hash_without_seed()
    verdict = ledger.promotion_verdict(family)
    print(str(verdict))
    return 0 if verdict.promoted else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qd_train.ledger",
        description=(
            "The decision record. `record` runs a build-and-test verification and writes the "
            "row its numbers cite (repo rule 5)."
        ),
    )
    # `--ledger` hangs off every subcommand rather than off the top level, so
    # `record --ledger X` works. An option that is only legal before the verb is
    # an option people get wrong, and argparse rejects it with a usage dump that
    # reads like the command itself was wrong.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--ledger", type=Path, default=DEFAULT_LEDGER_PATH, help="JSONL path (default: %(default)s)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", parents=[common], help="run suites and write one `build` row")
    rec.add_argument(
        "--suite",
        action="append",
        required=True,
        metavar="NAME=COMMAND",
        help="a suite to run; repeatable. COMMAND is argv, not a shell line: no pipes, no "
        "redirects. NAME becomes the metric key `suite.NAME`.",
    )
    rec.add_argument(
        "--toolchain",
        required=True,
        help="what ran the suites, e.g. 'cargo 1.98.0 / python 3.14.7'. Hashed with the "
        "commands into recipe_hash: two build rows are comparable exactly when both match.",
    )
    rec.add_argument("--repo", type=Path, default=REPO_ROOT, help="repo whose HEAD the row records")
    rec.add_argument("--cwd", type=Path, default=None, help="where to run the suites")
    rec.add_argument(
        "--timeout", type=float, default=DEFAULT_SUITE_TIMEOUT_S, help="per-suite seconds"
    )
    rec.add_argument("--parser", choices=["auto", *sorted(SUITE_PARSERS)], default="auto")
    rec.add_argument(
        "--quick",
        action="store_true",
        help="this run was truncated or subsampled; requires --quick-reason",
    )
    rec.add_argument("--quick-reason", default=None)
    rec.add_argument("--notes", default="")
    rec.set_defaults(func=_cmd_record)

    ver = sub.add_parser(
        "verify", parents=[common], help="recompute the hash chain and report the first break"
    )
    ver.set_defaults(func=_cmd_verify)

    show = sub.add_parser("show", parents=[common], help="print one row by id")
    show.add_argument("--row-id", required=True)
    show.set_defaults(func=_cmd_show)

    verdict = sub.add_parser(
        "verdict", parents=[common], help="may a seed family promote a decision?"
    )
    group = verdict.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed-family", default=None, help="protocol hash without seed")
    group.add_argument("--row-id", default=None, help="use this row's seed family")
    verdict.set_defaults(func=_cmd_verdict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except (ValueError, LedgerChainError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
