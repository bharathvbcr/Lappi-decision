"""Repo-level splitting, and the leakage checks that prove it held.

``docs/hardening.md`` section 2 gives the rule and its limit in the same breath:
*"Repo-level split, never row-level"* and *"repo-level is not sufficient by
itself"*. This module implements the first and **refuses to run without** the
second: :func:`split` takes a :class:`~qd_data.dedupe.DedupeReport`, not a bare row
list, so it is not possible to split a corpus that was never deduped. A caller who
wants to skip dedupe has to construct a report saying dedupe did not run, and that
report's ``NotRun`` status propagates into the split's own status and from there
into the manifest.

**There are two holdouts and they are different mechanisms.** Conflating them is
the subtle error this module is written to avoid:

``repo_split`` -- the *natural held-out set*
    A three-way assignment over ``repo_key``. Content the model never saw. This is
    the boundary near-duplicates must not cross, and it is what the shipping gate's
    paired margin is measured on.

``family holdout`` -- the *two held-out task families*
    Rows whose ``family_id`` is held out go to the held-out shard **whatever their
    repo says**. What is withheld here is the *question shape*, not the content:
    ``docs/schema-api.md`` asks that these families go to ``noul``, which tests
    abstention on an unasked question. Their content may legitimately overlap
    training content -- a commitpackft row can feed a training family and a held-out
    family at once -- so applying the near-duplicate check across this boundary
    would flag the design as a defect.

So the near-duplicate and identity checks run over ``repo_split``, and a separate
check asserts no held-out family identifier appears in a training shard.

The repo assignment is a keyed hash, not a shuffle:

* it is a pure function of ``(repo_key, seed)``, so adding a repo to the corpus does
  not move any other repo across a boundary -- a shuffle-and-slice would reshuffle
  everything and silently invalidate every earlier comparison;
* it is stable across processes and platforms, for the reason
  ``qd_data.render.DeterministicRng`` gives.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final

from qd_train.tristate import NotRun, Ran, TriState, aggregate

from .config import SPLITS, DataConfig
from .dedupe import DedupeReport
from .minhash import MinHasher, candidate_pairs, choose_bands, exact_jaccard, shingle
from .rows import DataRow

__all__ = [
    "HELD_OUT",
    "TRAINING_SPLITS",
    "SplitAssignment",
    "SplitReport",
    "assign_repo",
    "split",
]

HELD_OUT: Final[str] = "heldout"

#: The splits a training process may read. ``val`` is a training-time split: it
#: selects hyperparameters, so it is not held out in the sense rule 3 means.
TRAINING_SPLITS: Final[frozenset[str]] = frozenset({"train", "val"})


def assign_repo(
    repo_key: str, *, seed: int, train_fraction: float, val_fraction: float
) -> str:
    """Deterministically place one repo in ``train``, ``val`` or ``heldout``."""
    if not repo_key.strip():
        raise ValueError(
            "empty repo_key: a row with no split unit cannot be assigned, and defaulting "
            "it to its own group would put near-identical rows on both sides"
        )
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"fractions must be in (0, 1), got train={train_fraction}, val={val_fraction}"
        )
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train ({train_fraction}) + val ({val_fraction}) leaves no held-out repos"
        )
    material = f"qd_data.split.v1|{seed}|{repo_key}".encode()
    digest = hashlib.blake2b(material, digest_size=8).digest()
    # Uniform in [0, 1) with 53 bits of resolution, the mantissa of a float64.
    u = (int.from_bytes(digest, "big") >> 11) / float(1 << 53)
    if u < train_fraction:
        return "train"
    if u < train_fraction + val_fraction:
        return "val"
    return HELD_OUT


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Where a row landed, and by which of the two mechanisms."""

    row_id: str
    #: The final shard the row is written to.
    split: str
    #: What the repo hash said, before any family override. The leakage checks use
    #: this, because it is the content boundary.
    repo_split: str
    repo_key: str
    family_id: str
    #: ``"repo_hash"`` or ``"family_holdout"``.
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "split": self.split,
            "repo_split": self.repo_split,
            "repo_key": self.repo_key,
            "family_id": self.family_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SplitReport:
    assignments: tuple[SplitAssignment, ...]
    rows_by_split: dict[str, tuple[DataRow, ...]]
    repo_disjoint: TriState
    identity_disjoint: TriState
    near_duplicate_disjoint: TriState
    held_out_families_absent_from_training: TriState
    dedupe_status: TriState

    @property
    def status(self) -> TriState:
        """One tri-state over every post-condition, including dedupe's own.

        ``aggregate`` returns ``NotRun`` if any input did not run, so a split built on
        a dedupe pass that hit its bound cannot report clean.
        """
        return aggregate(
            {
                "dedupe": self.dedupe_status,
                "repo_disjoint": self.repo_disjoint,
                "identity_disjoint": self.identity_disjoint,
                "near_duplicate_disjoint": self.near_duplicate_disjoint,
                "held_out_families_absent_from_training": (
                    self.held_out_families_absent_from_training
                ),
            },
            name="split",
        )

    def counts(self) -> dict[str, int]:
        return {s: len(self.rows_by_split.get(s, ())) for s in SPLITS}

    def holdout_breakdown(self) -> dict[str, int]:
        """How the held-out shard was populated. Two populations, counted apart."""
        out = {"natural_repo_holdout": 0, "family_holdout": 0}
        for a in self.assignments:
            if a.split != HELD_OUT:
                continue
            key = "family_holdout" if a.reason == "family_holdout" else "natural_repo_holdout"
            out[key] += 1
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "holdout_breakdown": self.holdout_breakdown(),
            "repo_disjoint": self.repo_disjoint.to_json(),
            "identity_disjoint": self.identity_disjoint.to_json(),
            "near_duplicate_disjoint": self.near_duplicate_disjoint.to_json(),
            "held_out_families_absent_from_training": (
                self.held_out_families_absent_from_training.to_json()
            ),
            "dedupe_status": self.dedupe_status.to_json(),
            "status": self.status.to_json(),
        }


def _key_disjointness(
    assignments: tuple[SplitAssignment, ...],
    keys: dict[str, str],
    *,
    label: str,
) -> TriState:
    """Assert one key never appears under two ``repo_split`` values."""
    if not assignments:
        return NotRun(reason=f"{label}: no rows were assigned, so disjointness was not tested")
    where: dict[str, set[str]] = {}
    for a in assignments:
        where.setdefault(keys[a.row_id], set()).add(a.repo_split)
    straddling = sorted(k for k, s in where.items() if len(s) > 1)
    return Ran(
        passed=not straddling,
        value=len(straddling),
        n=len(where),
        n_total=len(where),
        detail=(
            ""
            if not straddling
            else f"{label}: {len(straddling)} key(s) appear in more than one repo split, "
            f"first five: {straddling[:5]}"
        ),
    )


def split(
    report: DedupeReport,
    *,
    config: DataConfig,
    max_candidate_pairs: int = 5_000_000,
) -> SplitReport:
    """Assign every surviving row to ``train``, ``val`` or ``heldout``.

    Takes a :class:`~qd_data.dedupe.DedupeReport` rather than rows, so the ordering
    "dedupe across everything, *then* split" is enforced by the signature.
    """
    rows = report.kept
    assignments: list[SplitAssignment] = []
    for row in rows:
        repo_split = assign_repo(
            row.repo_key,
            seed=config.seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
        )
        held_by_family = config.is_held_out_family(row.family_id)
        assignments.append(
            SplitAssignment(
                row_id=row.row_id,
                split=HELD_OUT if held_by_family else repo_split,
                repo_split=repo_split,
                repo_key=row.repo_key,
                family_id=row.family_id,
                reason="family_holdout" if held_by_family else "repo_hash",
            )
        )

    by_id = {r.row_id: r for r in rows}
    rows_by_split: dict[str, list[DataRow]] = {s: [] for s in SPLITS}
    for a in assignments:
        rows_by_split[a.split].append(by_id[a.row_id])

    repo_ok = _key_disjointness(
        tuple(assignments), {r.row_id: r.repo_key for r in rows}, label="repo_key"
    )
    identity_ok = _key_disjointness(
        tuple(assignments), {r.row_id: r.identity_key for r in rows}, label="identity_key"
    )
    nd_ok = _cross_split_near_duplicates(
        rows,
        repo_split_of={a.row_id: a.repo_split for a in assignments},
        config=config,
        max_candidate_pairs=max_candidate_pairs,
    )
    fam_ok = _held_out_families_absent(assignments, config=config)

    return SplitReport(
        assignments=tuple(assignments),
        rows_by_split={s: tuple(v) for s, v in rows_by_split.items()},
        repo_disjoint=repo_ok,
        identity_disjoint=identity_ok,
        near_duplicate_disjoint=nd_ok,
        held_out_families_absent_from_training=fam_ok,
        dedupe_status=report.status,
    )


def _held_out_families_absent(
    assignments: list[SplitAssignment], *, config: DataConfig
) -> TriState:
    """``docs/hardening.md`` section 2: *a test asserts their identifiers appear in
    no training shard.* Here it is an assertion the pipeline itself makes."""
    if not assignments:
        return NotRun(
            reason="no rows were assigned, so no shard was inspected for held-out families"
        )
    training = [a for a in assignments if a.split in TRAINING_SPLITS]
    if not training:
        return NotRun(
            reason=(
                "no rows landed in a training shard, so the absence of a held-out family "
                "from one was not tested"
            )
        )
    offenders = sorted({a.family_id for a in training if config.is_held_out_family(a.family_id)})
    return Ran(
        passed=not offenders,
        value=len(offenders),
        n=len(training),
        n_total=len(training),
        detail=(
            ""
            if not offenders
            else f"held-out families present in a training shard: {offenders}"
        ),
    )


def _cross_split_near_duplicates(
    rows: tuple[DataRow, ...],
    *,
    repo_split_of: dict[str, str],
    config: DataConfig,
    max_candidate_pairs: int,
) -> TriState:
    """Re-derive near-duplicate pairs and assert none crosses a **repo** boundary.

    An independent second derivation on purpose. Trusting dedupe's own bookkeeping
    would mean a bug in dedupe reports itself as a clean split -- the same argument
    ``docs/hardening.md`` section 1 makes for re-deriving mutation spans from a
    textual diff rather than from the mutator.
    """
    if len(rows) < 2:
        return NotRun(
            reason=(
                f"cross-split near-duplicate check needs at least 2 rows, got {len(rows)}; "
                "with fewer there is no pair to cross a boundary"
            )
        )
    if len({repo_split_of[r.row_id] for r in rows}) < 2:
        return NotRun(
            reason=(
                "every row landed in one repo split, so no pair could cross a boundary; "
                "this is not evidence that the split separates near-duplicates"
            )
        )
    hasher = MinHasher(num_perm=config.num_perm, seed=config.seed)
    band_config = choose_bands(num_perm=config.num_perm, threshold=config.dedupe_threshold)
    sh = {r.row_id: shingle(r.dedupe_text, k=config.shingle_size).shingles for r in rows}
    sigs = {rid: hasher.signature(s) for rid, s in sh.items()}
    cands, truncated = candidate_pairs(sigs, config=band_config, max_pairs=max_candidate_pairs)
    if truncated:
        return NotRun(
            reason=(
                f"cross-split near-duplicate search hit its bound of {max_candidate_pairs} "
                "candidate pairs; the absence of a crossing pair was not established"
            )
        )
    crossing = [(a, b) for a, b in sorted(cands) if repo_split_of[a] != repo_split_of[b]]
    offenders = [
        (a, b, j)
        for a, b in crossing
        if (j := exact_jaccard(sh[a], sh[b])) >= config.dedupe_threshold
    ]
    # Coverage is the rows scanned, not the pairs found: reporting n=0/n_total=0 when
    # LSH proposes no candidate would make a complete scan of a clean corpus read
    # like a check that examined nothing.
    return Ran(
        passed=not offenders,
        value=len(offenders),
        n=len(rows),
        n_total=len(rows),
        detail=(
            f"scanned {len(rows)} rows; {len(cands)} candidate pairs, {len(crossing)} of "
            "them crossing a repo split, none confirmed at threshold "
            f"{config.dedupe_threshold}"
            if not offenders
            else "near-duplicate pairs span a repo split boundary: "
            + ", ".join(f"{a}~{b} (J={j:.3f})" for a, b, j in offenders[:5])
        ),
    )
