"""The data config, and the assertions it refuses to load without.

One of these checks is the reason this module exists. ``docs/plan-corrections.md``:

    **The two held-out task families must be chosen from what actually survives**
    -- holding out a family that was never going to load is not a holdout.

That failure is completely silent. You name two families in a config, the pipeline
never sees them because their source was refused at load anyway, every downstream
report says "held-out families abstained", and the gate passes without ever having
tested anything. So :class:`DataConfig` refuses to construct unless each held-out
family is one that is *otherwise admitted* -- it has to be a family that would have
been trained on, or holding it out means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from .licences import DEFAULT_LICENCE_CONFIG, LicenceConfig
from .sources import TASK_FAMILIES, admitted_task_families

__all__ = [
    "DataConfig",
    "Split",
    "SPLITS",
    "DEFAULT_HELD_OUT_FAMILIES",
    "N_HELD_OUT_FAMILIES",
]

Split = str
SPLITS: Final[tuple[str, ...]] = ("train", "val", "heldout")

#: The plan fixes the count at two.
N_HELD_OUT_FAMILIES: Final[int] = 2

#: Both are ``choice`` families whose *source* still contributes another family to
#: training. That is deliberate: the holdout then tests generalisation to an unseen
#: **question shape** rather than to an unseen corpus, which is what
#: ``docs/schema-api.md`` is asking for when it says the held-out families must go
#: to ``noul``. Holding out the only ``score`` family or the only ``span`` family
#: instead would mean the model never saw that slot type at all, and its abstention
#: would prove nothing about the question.
DEFAULT_HELD_OUT_FAMILIES: Final[tuple[str, ...]] = ("code.language_id", "qa.answerability")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Everything the data lane needs, validated at construction."""

    seed: int = 20260919
    licence: LicenceConfig = DEFAULT_LICENCE_CONFIG
    held_out_families: tuple[str, ...] = DEFAULT_HELD_OUT_FAMILIES
    #: Filesystem roots that a training process must never read.
    held_out_roots: tuple[Path, ...] = (Path("data/heldout"),)
    #: Any path segment equal to one of these marks a path held out, wherever it
    #: sits. Convention is not the mechanism -- see qd_train.data_access -- but a
    #: belt alongside the explicit roots costs nothing.
    held_out_path_markers: tuple[str, ...] = ("heldout", "held_out", "held-out")

    dedupe_threshold: float = 0.8
    shingle_size: int = 5
    num_perm: int = 128

    #: Repo-level split fractions. They are stated separately and must sum to 1.0:
    #: a single ``train_fraction`` with "the rest" left implicit is how a three-way
    #: split quietly becomes a two-way one.
    train_fraction: float = 0.9
    val_fraction: float = 0.05

    #: Bounds. Every batch, payload and read is bounded; these are the data lane's.
    max_rows_per_source: int = 2_000_000
    max_row_bytes: int = 1_048_576

    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"seed must be int, got {type(self.seed).__name__}")

        fams = tuple(self.held_out_families)
        if len(fams) != N_HELD_OUT_FAMILIES:
            raise ValueError(
                f"the plan fixes the holdout at {N_HELD_OUT_FAMILIES} task families; "
                f"got {len(fams)}: {fams}"
            )
        if len(set(fams)) != len(fams):
            raise ValueError(f"held_out_families contains a duplicate: {fams}")

        unknown = [f for f in fams if f not in TASK_FAMILIES]
        if unknown:
            raise ValueError(
                f"held-out families {unknown} are not registered task families. "
                f"Known: {sorted(TASK_FAMILIES)}"
            )

        # The check this module exists for.
        admitted = {f.family_id for f in admitted_task_families(self.licence)}
        never_admitted = [f for f in fams if f not in admitted]
        if never_admitted:
            raise ValueError(
                f"held-out families {never_admitted} are not otherwise admitted, so holding "
                "them out tests nothing: their source is refused at load anyway (licence or "
                "reachability). A holdout must name a family that would otherwise have been "
                "trained on. Admitted families are: " + ", ".join(sorted(admitted))
            )

        remaining = admitted - set(fams)
        if not remaining:
            raise ValueError(
                f"holding out {fams} leaves no admitted task family to train on"
            )

        if not 0.0 < self.dedupe_threshold <= 1.0:
            raise ValueError(f"dedupe_threshold must be in (0, 1], got {self.dedupe_threshold}")
        if self.shingle_size < 1:
            raise ValueError(f"shingle_size must be >= 1, got {self.shingle_size}")
        if self.num_perm < 16 or self.num_perm % 2:
            raise ValueError(f"num_perm must be an even int >= 16, got {self.num_perm}")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError(f"train_fraction must be in (0, 1), got {self.train_fraction}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                f"train_fraction ({self.train_fraction}) + val_fraction "
                f"({self.val_fraction}) leaves no repos for the natural held-out set. "
                "A held-out set of size zero passes every leakage check vacuously."
            )
        if self.max_rows_per_source < 1 or self.max_row_bytes < 1:
            raise ValueError("row bounds must be positive")
        if not self.held_out_roots:
            raise ValueError(
                "held_out_roots is empty: the training-time path check would then have "
                "nothing to refuse, which is the check silently disabled"
            )

    @property
    def training_families(self) -> tuple[str, ...]:
        """Admitted families minus the holdout: exactly what may enter a shard."""
        admitted = {f.family_id for f in admitted_task_families(self.licence)}
        return tuple(sorted(admitted - set(self.held_out_families)))

    def is_held_out_family(self, family_id: str) -> bool:
        return family_id in set(self.held_out_families)

    def fingerprint(self) -> dict[str, object]:
        """The config fields that change the data. Feeds ``data_snapshot_hash``."""
        return {
            "seed": self.seed,
            "held_out_families": sorted(self.held_out_families),
            "dedupe_threshold": self.dedupe_threshold,
            "shingle_size": self.shingle_size,
            "num_perm": self.num_perm,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "admitted_by_human": dict(sorted(self.licence.admitted_by_human.items())),
        }
