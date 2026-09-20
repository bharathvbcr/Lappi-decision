"""The only door into training data, and the check that refuses held-out data.

``CLAUDE.md`` rule 3, verbatim:

    **Held-out data and the two task-holdout families are never read by a training
    process.** A path check in ``qd-train`` refuses them; removing that check is a
    refused change.

``docs/hardening.md`` section 2 says the same thing and adds the test:

    Held-out paths are refused by the training process **at the path level**, not by
    convention. ... A test asserts that pointing the trainer at a held-out manifest
    raises rather than trains.

Both are implemented here, and both are enforced on the way *in* rather than
audited afterwards: :func:`open_training_data` is the only function in ``qd_train``
that turns a path into rows, so there is no second door to remember to guard.

The check has four independent layers, because each one alone has a way to be
wrong, and a rename or a symlink should not be enough to defeat a rule this
important:

1. **Configured roots.** A path at or under any ``DataConfig.held_out_roots`` entry
   is refused. Compared on resolved, absolute paths, so ``train/../heldout/x.json``
   and a symlink into the held-out tree are both caught.
2. **Path markers.** Any path *segment* equal to ``heldout`` / ``held_out`` /
   ``held-out`` is refused wherever it sits. Segment-exact, not substring: a
   directory called ``withheld_outputs`` is not a holdout, and a check that thought
   so would train on nothing and look fine.
3. **The manifest's own declaration.** A manifest says which split it is. A file
   moved out of the held-out directory still declares ``split: "heldout"``, so
   renaming the directory does not launder it.
4. **The family holdout.** Even a manifest that says ``train`` is refused if any of
   its entries names a held-out task family. This is the layer that catches a
   corpus built with the wrong config rather than a file in the wrong place.

The refusal is :class:`~qd_data.errors.HeldOutViolation`, a typed error. It is
raised, not returned, and it has no ``value``, no ``noul`` and no ``score`` -- a
caller cannot mistake it for an abstention or for an empty dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qd_data.config import DataConfig
from qd_data.errors import HeldOutViolation
from qd_data.manifest import Manifest
from qd_data.split import TRAINING_SPLITS

from .tristate import NotRun, Ran, TriState

__all__ = [
    "TrainingData",
    "assert_manifest_trainable",
    "assert_path_not_held_out",
    "held_out_check_report",
    "open_training_data",
]


def _resolve(path: Path) -> Path:
    """Absolute and symlink-free where possible.

    ``strict=False`` so a path that does not exist yet still normalises; the check
    must fire on the *intent* to read a held-out path, not only on a successful read.
    """
    return Path(path).expanduser().resolve(strict=False)


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def assert_path_not_held_out(path: Path, *, config: DataConfig, repo_root: Path) -> None:
    """Layers 1 and 2. Raises :class:`HeldOutViolation`; returns ``None`` otherwise.

    ``repo_root`` is needed because ``DataConfig.held_out_roots`` are relative by
    design -- a config carrying absolute paths would be unusable on another machine,
    and a check that silently skipped a relative root would be no check at all.
    """
    if not config.held_out_roots:
        raise HeldOutViolation(
            expected="at least one configured held-out root",
            actual=0,
            detail=(
                "held_out_roots is empty, so the path check has nothing to refuse. "
                "That is the check disabled, not the check passing."
            ),
        )
    resolved = _resolve(path)
    root_abs = _resolve(repo_root)

    for root in config.held_out_roots:
        root_path = Path(root)
        absolute_root = _resolve(root_path if root_path.is_absolute() else root_abs / root_path)
        if resolved == absolute_root or _is_within(resolved, absolute_root):
            raise HeldOutViolation(
                expected=f"a path outside the held-out root {absolute_root}",
                actual=str(resolved),
                detail=(
                    "CLAUDE.md rule 3: held-out data is never read by a training process. "
                    "Removing this check is a refused change."
                ),
            )

    markers = {m.casefold() for m in config.held_out_path_markers}
    for segment in resolved.parts:
        if segment.casefold() in markers:
            raise HeldOutViolation(
                expected=f"no path segment in {sorted(markers)}",
                actual=str(resolved),
                detail=(
                    f"path segment {segment!r} marks this as held-out data wherever it "
                    "sits. CLAUDE.md rule 3."
                ),
            )


def assert_manifest_trainable(manifest: Manifest, *, config: DataConfig, path: Path) -> None:
    """Layers 3 and 4. The file's own declaration, and its families."""
    if manifest.split not in TRAINING_SPLITS:
        raise HeldOutViolation(
            expected=f"a manifest whose split is one of {sorted(TRAINING_SPLITS)}",
            actual=manifest.split,
            detail=(
                f"{path}: the manifest declares itself split {manifest.split!r}. Moving or "
                "renaming the file does not change what it says it is. CLAUDE.md rule 3."
            ),
        )
    held = set(config.held_out_families)
    offending = sorted({e.family_id for e in manifest.entries if e.family_id in held})
    if offending:
        raise HeldOutViolation(
            expected=f"no entry from a held-out task family {sorted(held)}",
            actual=offending,
            detail=(
                f"{path}: a manifest declaring split {manifest.split!r} carries "
                f"{len(offending)} held-out task family identifier(s). The shard was built "
                "with a different holdout than this config names. CLAUDE.md rule 3."
            ),
        )
    # A manifest whose recorded holdout disagrees with the config is not a shard this
    # run may train on: one of the two is stale, and guessing which would let a
    # holdout silently shrink.
    recorded = tuple(sorted(manifest.held_out_families))
    expected = tuple(sorted(config.held_out_families))
    if recorded and recorded != expected:
        raise HeldOutViolation(
            expected=f"a shard built with held-out families {list(expected)}",
            actual=list(recorded),
            detail=(
                f"{path}: the shard records a different holdout than this run configures. "
                "Rule 2: an agent may report that a gate failed; it may not shrink a "
                "held-out set to make one pass."
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingData:
    """A manifest a training process is allowed to read, and the checks that cleared it."""

    manifest: Manifest
    path: Path
    data_snapshot_hash: str
    checks: dict[str, TriState]

    @property
    def n_rows(self) -> int:
        return len(self.manifest.entries)

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "split": self.manifest.split,
            "data_snapshot_hash": self.data_snapshot_hash,
            "n_rows": self.n_rows,
            "checks": {k: v.to_json() for k, v in sorted(self.checks.items())},
        }


def open_training_data(
    path: Path, *, config: DataConfig, repo_root: Path, allow_not_run_snapshot: bool = False
) -> TrainingData:
    """The one way a training process opens data. Refuses held-out input, loudly.

    ``allow_not_run_snapshot`` must be passed explicitly to train on a snapshot whose
    own status is ``NotRun`` (for example, one whose dedupe pass hit its bound). The
    default is to refuse: a run built on an unverified snapshot that later reports a
    clean gate is the failure ``docs/hardening.md`` opens with.
    """
    path = Path(path)
    assert_path_not_held_out(path, config=config, repo_root=repo_root)
    manifest = Manifest.read(path)  # re-derives and checks data_snapshot_hash
    assert_manifest_trainable(manifest, config=config, path=path)

    if isinstance(manifest.status, NotRun):
        if not allow_not_run_snapshot:
            raise HeldOutViolation(
                expected="a data snapshot whose pipeline status is 'ran'",
                actual="not_run",
                detail=(
                    f"{path}: {manifest.status.reason}. Training on an unverified snapshot "
                    "and then reporting a clean gate is the exact failure the tri-state "
                    "exists to prevent. Pass allow_not_run_snapshot=True to proceed "
                    "deliberately; the reason stays in the ledger row."
                ),
            )
    elif not manifest.status.passed:
        # Refusing `not_run` while admitting `ran, passed=false` stops the weaker signal
        # and waves the stronger one through: "could not be checked" is turned away,
        # "was checked and is bad" is not. There is deliberately no override -- the
        # `allow_not_run_snapshot` escape hatch answers "the check could not run", and
        # widening it to cover a gate that ran and failed would be a rule 2 change made
        # from the reading side.
        detail = f"{path}: the snapshot's own pipeline ran and failed"
        if manifest.status.detail:
            detail += f" -- {manifest.status.detail}"
        raise HeldOutViolation(
            expected="a data snapshot whose pipeline status is 'ran' and passed",
            actual="ran, passed=false",
            detail=(
                detail + ". Rebuild the snapshot or fix the gate it failed; there is no "
                "flag that admits it, and allow_not_run_snapshot does not."
            ),
        )
    snapshot_status: TriState = manifest.status

    return TrainingData(
        manifest=manifest,
        path=path,
        data_snapshot_hash=manifest.snapshot_hash(),
        checks={
            "path_not_held_out": Ran(
                passed=True, value=str(path),
                detail="no configured held-out root and no held-out path marker matched",
            ),
            "manifest_split_trainable": Ran(passed=True, value=manifest.split),
            "no_held_out_families": Ran(
                passed=True, n=len(manifest.entries), n_total=len(manifest.entries),
                detail=f"held-out families: {sorted(config.held_out_families)}",
            ),
            "snapshot_status": snapshot_status,
        },
    )


def held_out_check_report(
    paths: list[Path] | tuple[Path, ...], *, config: DataConfig, repo_root: Path
) -> TriState:
    """Run the path check over several paths and report one tri-state.

    Over an empty list this is ``NotRun``, never a pass: ``aggregate([])`` returns
    ``NotRun`` for the same reason, and "we checked no paths" must not read as "no
    path was held out".
    """
    paths = tuple(paths)
    if not paths:
        return NotRun(
            reason=(
                "held-out path check ran over zero paths; nothing was inspected, so "
                "nothing was cleared"
            )
        )
    violations: list[str] = []
    for p in paths:
        try:
            assert_path_not_held_out(p, config=config, repo_root=repo_root)
        except HeldOutViolation as exc:
            violations.append(f"{p}: {exc.detail}")
    return Ran(
        passed=not violations,
        value=len(violations),
        n=len(paths),
        n_total=len(paths),
        detail="" if not violations else "; ".join(violations[:5]),
    )
