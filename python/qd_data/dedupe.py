"""Near-duplicate removal, run across pool **and** held-out **before** the split.

``docs/hardening.md`` section 2, in full, because both the rule and its stated
reason constrain the implementation:

    **Repo-level split, never row-level.** Keep the repo name on every row from the
    pull onward. **Repo-level is not sufficient by itself.** Vendored trees, forks
    and copied files put the same code in two repos. MinHash dedupe at 0.8 Jaccard
    runs **across pool and held-out before splitting**.

Running it after the split would be a different and much weaker check: it would
find duplicates *within* train and *within* held-out and leave every cross-split
pair -- the only ones that leak -- in place. :func:`dedupe` therefore takes one
undifferentiated collection of rows and has no notion of splits at all; the
splitter runs on its output.

Two things about *what* is compared were wrong in the obvious implementation and
are worth stating, because both were found by running it:

**The unit is a content unit, not an example.** One ``bigcode/commitpackft`` row
produces three examples -- commit intent, language id, change scope -- from the same
file. Comparing examples makes those three exact duplicates of one another, and a
naive dedupe deletes two thirds of the mixture while reporting a clean pass. A
content unit is ``(identity_key, digest(dedupe_text))``; every example sharing one
is one node, and a dropped unit takes all its examples with it.

**Only *cross-repo* duplicates are dropped.** The hazard named above is a file that
appears in two repositories, because that is the one a repo-level split cannot
separate. Two near-identical rows inside one repository always land on the same side
of the split, so dropping them removes data without removing any leak -- and for
``rajpurkar/squad_v2`` it removes almost everything, since several questions share
one passage and sit at Jaccard ~0.85 by construction. Within-repo near-duplicates
are therefore **counted and reported**, never silently deleted; the count is in the
manifest so the redundancy is visible rather than assumed away.

**Which of a cross-repo pair survives is deterministic**: the lexicographically
smallest unit key in each connected component. "Whichever came first" would depend
on iteration order and would move ``data_snapshot_hash`` between runs over the same
corpus.

**Every stage reports a tri-state.** A candidate-pair set that hit its bound is
``NotRun`` with the bound in the reason, never a clean dedupe over a partial pair
list. That is the difference between "no near-duplicates" and "we stopped looking".
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

from qd_train.tristate import NotRun, Ran, TriState

from .config import DataConfig
from .minhash import (
    BandConfig,
    MinHasher,
    candidate_pairs,
    choose_bands,
    exact_jaccard,
    shingle,
)
from .rows import DataRow

__all__ = [
    "DEFAULT_MAX_CANDIDATE_PAIRS",
    "ContentUnit",
    "DedupeReport",
    "DuplicateCluster",
    "content_unit_key",
    "dedupe",
]

#: Bounded fan-out. At 0.8 Jaccard on a real corpus the candidate set is a small
#: multiple of the unit count; a set this large means the corpus is pathological (for
#: example every row identical) and the run must say so rather than grind.
DEFAULT_MAX_CANDIDATE_PAIRS: int = 5_000_000


def content_unit_key(row: DataRow) -> str:
    """``identity_key`` plus a digest of the compared text.

    Both halves are load-bearing. Without ``identity_key``, two files with identical
    boilerplate in one repo merge. Without the digest, two commits touching the same
    path merge into one unit and one of their texts is silently discarded.
    """
    digest = hashlib.blake2b(row.dedupe_text.encode("utf-8"), digest_size=16).hexdigest()
    return f"{row.identity_key}|{digest}"


@dataclass(frozen=True, slots=True)
class ContentUnit:
    """One distinct piece of compared content, and every example built from it."""

    key: str
    repo_key: str
    text: str
    row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DuplicateCluster:
    """One connected component of the cross-repo near-duplicate graph."""

    kept_unit_key: str
    dropped_unit_keys: tuple[str, ...]
    #: Repos the component spans. Always more than one: same-repo edges are not
    #: unioned, so a single-repo component cannot exist here.
    repo_keys: tuple[str, ...]
    #: The lowest exact Jaccard on any confirmed edge in this component, so a cluster
    #: held together by a chain of 0.81 pairs is visible as such.
    min_edge_jaccard: float
    n_rows_dropped: int

    def to_json(self) -> dict[str, Any]:
        return {
            "kept_unit_key": self.kept_unit_key,
            "dropped_unit_keys": list(self.dropped_unit_keys),
            "repo_keys": list(self.repo_keys),
            "min_edge_jaccard": round(self.min_edge_jaccard, 6),
            "n_rows_dropped": self.n_rows_dropped,
        }


@dataclass(frozen=True, slots=True)
class DedupeReport:
    """What dedupe did, in enough detail to be re-derived."""

    kept: tuple[DataRow, ...]
    clusters: tuple[DuplicateCluster, ...]
    n_input_rows: int
    n_input_units: int
    n_dropped_rows: int
    n_dropped_units: int
    #: Confirmed near-duplicate pairs whose two units sit in different repos. This is
    #: the population a repo-level split alone cannot separate.
    n_cross_repo_pairs: int
    #: Confirmed pairs inside one repo. **Reported, not dropped** -- see the module
    #: docstring. A large number here is redundancy, not leakage.
    n_within_repo_pairs: int
    #: Units whose text exceeded the shingling bound and were compared on a prefix.
    n_truncated_for_shingling: int
    band_config: BandConfig
    threshold: float
    n_candidate_pairs: int
    #: ``Ran`` only when the candidate search completed within its bound.
    status: TriState
    #: Exactly which rows were removed. Carried rather than re-derived, so a caller
    #: auditing the drop does not have to reproduce the clustering to find out.
    dropped_row_ids: frozenset[str] = frozenset()

    @property
    def n_possible_unit_pairs(self) -> int:
        """``C(n_input_units, 2)`` -- every pair that could have been a duplicate.

        The denominator the candidate count needs. ``n_candidate_pairs`` alone reads
        as a finding; against this it reads as what it is, a filtered subset.
        """
        n = self.n_input_units
        return n * (n - 1) // 2

    @property
    def lsh_recall_at_threshold(self) -> float:
        """The banding's per-pair recall at :attr:`threshold`.

        ``GAP-DATA-LSH-RECALL-BOUND``: the report used to carry ``bands`` and
        ``rows_per_band`` and stop there, which told a reader the cause of the
        recall loss without telling them its size, and left "no near-duplicates
        found" reading as "no near-duplicates exist". Derived from the banding
        rather than stored, so it cannot disagree with the configuration that
        produced it.
        """
        return self.band_config.recall_at_threshold

    def to_json(self) -> dict[str, Any]:
        return {
            "n_input_rows": self.n_input_rows,
            "n_input_units": self.n_input_units,
            "n_kept_rows": len(self.kept),
            "n_dropped_rows": self.n_dropped_rows,
            "n_dropped_units": self.n_dropped_units,
            "n_cross_repo_pairs": self.n_cross_repo_pairs,
            "n_within_repo_pairs": self.n_within_repo_pairs,
            "n_truncated_for_shingling": self.n_truncated_for_shingling,
            "threshold": self.threshold,
            "bands": self.band_config.bands,
            "rows_per_band": self.band_config.rows,
            "n_candidate_pairs": self.n_candidate_pairs,
            # Both numbers. A candidate count on its own reads as coverage.
            "n_possible_unit_pairs": self.n_possible_unit_pairs,
            "lsh_per_pair_recall_at_threshold": round(self.lsh_recall_at_threshold, 6),
            "clusters": [c.to_json() for c in self.clusters],
            "status": self.status.to_json(),
        }


class _UnionFind:
    __slots__ = ("_parent",)

    def __init__(self, keys: list[str]) -> None:
        self._parent: dict[str, str | None] = dict.fromkeys(keys)

    def find(self, x: str) -> str:
        root = x
        while (p := self._parent[root]) is not None:
            root = p
        while (p := self._parent[x]) is not None:
            self._parent[x] = root
            x = p
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Deterministic: the lexicographically smaller key always becomes the root,
        # so the surviving unit does not depend on the order edges arrived in.
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self._parent[hi] = lo


def _build_units(rows: tuple[DataRow, ...]) -> dict[str, ContentUnit]:
    grouped: dict[str, list[DataRow]] = {}
    for row in rows:
        grouped.setdefault(content_unit_key(row), []).append(row)
    units: dict[str, ContentUnit] = {}
    for key, members in grouped.items():
        repos = {m.repo_key for m in members}
        if len(repos) > 1:
            raise ValueError(
                f"content unit {key!r} spans repos {sorted(repos)}: the unit key is built "
                "from identity_key, which begins with the repo, so this means a rewriter "
                "emitted an identity_key that does not identify its repo"
            )
        units[key] = ContentUnit(
            key=key,
            repo_key=members[0].repo_key,
            text=members[0].dedupe_text,
            row_ids=tuple(sorted(m.row_id for m in members)),
        )
    return units


def dedupe(
    rows: list[DataRow] | tuple[DataRow, ...],
    *,
    config: DataConfig,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> DedupeReport:
    """MinHash-LSH near-duplicate removal at ``config.dedupe_threshold``.

    Takes **all** rows -- pool and held-out together -- and returns the survivors.
    There is deliberately no split argument: see the module docstring.
    """
    rows = tuple(rows)
    ids = [r.row_id for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted(i for i, c in Counter(ids).items() if c > 1)
        raise ValueError(
            f"duplicate row_id(s) {dupes[:5]} ({len(dupes)} total): row ids key every "
            "downstream map, so a collision would silently drop a row rather than dedupe it"
        )

    band_config = choose_bands(num_perm=config.num_perm, threshold=config.dedupe_threshold)

    if not rows:
        return DedupeReport(
            kept=(), clusters=(), n_input_rows=0, n_input_units=0, n_dropped_rows=0,
            n_dropped_units=0, n_cross_repo_pairs=0, n_within_repo_pairs=0,
            n_truncated_for_shingling=0, band_config=band_config,
            threshold=config.dedupe_threshold, n_candidate_pairs=0,
            status=NotRun(
                reason="dedupe received zero rows; nothing was compared, so nothing was cleared"
            ),
        )

    units = _build_units(rows)
    hasher = MinHasher(num_perm=config.num_perm, seed=config.seed)
    shingled: dict[str, frozenset[bytes]] = {}
    signatures: dict[str, tuple[int, ...]] = {}
    n_truncated = 0
    for key, unit in units.items():
        sh = shingle(unit.text, k=config.shingle_size)
        if not sh.shingles:
            # DataRow.__post_init__ refuses empty dedupe_text, so this is unreachable
            # for a validly constructed row. Refuse loudly rather than skip silently.
            raise ValueError(
                f"content unit {key!r} produced no shingles from non-empty text; "
                "it would be invisible to dedupe"
            )
        n_truncated += int(sh.truncated)
        shingled[key] = sh.shingles
        signatures[key] = hasher.signature(sh.shingles)

    cands, truncated = candidate_pairs(
        signatures, config=band_config, max_pairs=max_candidate_pairs
    )

    uf = _UnionFind(sorted(units))
    edge_j: dict[tuple[str, str], float] = {}
    n_within = 0
    for a, b in sorted(cands):
        j = exact_jaccard(shingled[a], shingled[b])
        if j < config.dedupe_threshold:
            continue
        if units[a].repo_key == units[b].repo_key:
            # Reported, not merged: same repo means same side of the split, so this
            # pair cannot leak, and deleting it would thin the mixture for nothing.
            n_within += 1
            continue
        edge_j[(a, b)] = j
        uf.union(a, b)

    components: dict[str, list[str]] = {}
    for key in sorted(units):
        components.setdefault(uf.find(key), []).append(key)

    clusters: list[DuplicateCluster] = []
    dropped_units: set[str] = set()
    dropped_rows: set[str] = set()
    for members in (sorted(v) for _, v in sorted(components.items())):
        if len(members) < 2:
            continue
        keep, drop = members[0], tuple(members[1:])
        member_set = set(members)
        edges = [j for (a, b), j in edge_j.items() if a in member_set and b in member_set]
        rows_dropped = [rid for k in drop for rid in units[k].row_ids]
        clusters.append(
            DuplicateCluster(
                kept_unit_key=keep,
                dropped_unit_keys=drop,
                repo_keys=tuple(sorted({units[k].repo_key for k in members})),
                min_edge_jaccard=min(edges) if edges else 0.0,
                n_rows_dropped=len(rows_dropped),
            )
        )
        dropped_units.update(drop)
        dropped_rows.update(rows_dropped)

    kept = tuple(r for r in rows if r.row_id not in dropped_rows)

    if truncated:
        status: TriState = NotRun(
            reason=(
                f"candidate-pair search hit its bound of {max_candidate_pairs}; the pair "
                "list is partial, so the surviving rows are not a verified deduplicated set"
            )
        )
    else:
        # Coverage is carried in the **pair** dimension, not the unit dimension.
        # Every unit was signed and banded, so "n units of n units" is true and is
        # also the wrong number: what a reader of a dedupe report wants to know is
        # how many of the pairs that could be duplicates were actually compared.
        # Banded LSH proposed `len(cands)` of `n_possible` and exact Jaccard
        # confirmed exactly those, so `is_complete_coverage` is False unless the
        # banding happened to propose everything -- which is the honest reading.
        n_units = len(units)
        n_possible = n_units * (n_units - 1) // 2
        recall = band_config.recall_at_threshold
        status = Ran(
            passed=True,
            value=len(dropped_rows),
            n=min(len(cands), n_possible),
            n_total=n_possible,
            detail=(
                f"{len(dropped_rows)} of {len(rows)} rows dropped across "
                f"{len(dropped_units)} content units at Jaccard >= "
                f"{config.dedupe_threshold}; {len(edge_j)} confirmed pairs crossed a repo "
                f"boundary and would have survived a repo-level split alone; "
                f"{n_within} confirmed pairs were within one repo and were kept. "
                f"LSH is a probabilistic candidate generator, not an exhaustive "
                f"search: banding b={band_config.bands}, r={band_config.rows} proposes "
                f"a pair at J={config.dedupe_threshold} with probability "
                f"{recall:.4f}, so {len(cands)} of {n_possible} possible unit pairs "
                f"were compared exactly and the confirmed-pair counts are a lower "
                f"bound, not a complete enumeration"
            ),
        )

    return DedupeReport(
        kept=kept,
        clusters=tuple(clusters),
        n_input_rows=len(rows),
        n_input_units=len(units),
        n_dropped_rows=len(dropped_rows),
        n_dropped_units=len(dropped_units),
        n_cross_repo_pairs=len(edge_j),
        n_within_repo_pairs=n_within,
        n_truncated_for_shingling=n_truncated,
        band_config=band_config,
        threshold=config.dedupe_threshold,
        n_candidate_pairs=len(cands),
        status=status,
        dropped_row_ids=frozenset(dropped_rows),
    )
