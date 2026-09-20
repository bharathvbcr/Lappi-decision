"""The hand-labelling harness: the 300 held-out labels, and the 300-item agreement set.

This is the program's long pole, because it is the one job only a human can do. The
design goals are therefore: lose nothing, resume anywhere, and measure the thing
everybody forgets to measure.

**The thing everybody forgets.** The teacher's kappa against you is bounded above by
your kappa against *yourself*. If you relabel the same diff differently on Tuesday
than you did on Monday, no labeller — human or 27B — can agree with you beyond that
ceiling. The plan measures teacher-vs-human and never establishes the ceiling it is
being measured against, so a teacher kappa of 0.6 has no scale.

So this harness silently re-presents a fraction of already-labelled items and reports
**intra-rater kappa**. That number is the denominator for the whole agreement gate.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

__all__ = ["LabelItem", "LabelRecord", "LabelSession", "SessionStats"]

Purpose = Literal["heldout", "agreement"]
VALID_LABELS: tuple[str, ...] = ("stub", "logic", "cosmetic", "clean", "unsure")


@dataclass(frozen=True, slots=True)
class LabelItem:
    """One diff awaiting a label."""

    item_id: str
    repo: str
    path: str
    language: str
    diff: str

    def __post_init__(self) -> None:
        for name in ("item_id", "repo", "path", "language"):
            if not getattr(self, name):
                raise ValueError(f"LabelItem.{name} must be non-empty (item {self.item_id!r})")
        if not self.diff.strip():
            raise ValueError(f"LabelItem {self.item_id!r} has an empty diff; nothing to label")


@dataclass(frozen=True, slots=True)
class LabelRecord:
    item_id: str
    label: str
    note: str
    purpose: Purpose
    labelled_at: str
    presentation: int  # 1 = first time seen, 2 = re-presented for intra-rater kappa
    seconds_spent: float

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "note": self.note,
            "purpose": self.purpose,
            "labelled_at": self.labelled_at,
            "presentation": self.presentation,
            "seconds_spent": round(self.seconds_spent, 2),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        return cls(
            item_id=raw["item_id"],
            label=raw["label"],
            note=raw.get("note", ""),
            purpose=raw["purpose"],
            labelled_at=raw["labelled_at"],
            presentation=int(raw["presentation"]),
            seconds_spent=float(raw.get("seconds_spent", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class SessionStats:
    total_items: int
    labelled_first_pass: int
    repeats_collected: int
    by_label: dict[str, int]
    by_language: dict[str, int]
    unsure_rate: float
    median_seconds: float

    def __str__(self) -> str:
        lines = [
            f"{self.labelled_first_pass}/{self.total_items} items labelled"
            f" ({self.labelled_first_pass / max(self.total_items, 1):.0%})",
            f"  repeats collected for intra-rater kappa: {self.repeats_collected}",
            f"  unsure rate: {self.unsure_rate:.1%}   median: {self.median_seconds:.0f}s/item",
            f"  by label:    {dict(sorted(self.by_label.items()))}",
            f"  by language: {dict(sorted(self.by_language.items()))}",
        ]
        return "\n".join(lines)


class LabelSession:
    """Resumable, append-only labelling over a fixed item pool.

    Append-only for the same reason the ledger is: a labelling session that loses
    work silently is one the human will not finish twice.
    """

    def __init__(
        self,
        pool: Sequence[LabelItem],
        store: str | os.PathLike[str],
        *,
        purpose: Purpose = "heldout",
        repeat_fraction: float = 0.1,
        seed: int = 0,
        disjoint_from: Collection[str] = (),
    ) -> None:
        if not pool:
            raise ValueError("cannot open a labelling session over an empty pool")
        ids = [i.item_id for i in pool]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"pool contains duplicate item_ids: {dupes[:5]}")

        # The held-out set and the agreement set must not share items. `docs/teacher-plan.md`
        # §6 states it as a condition on using the same labelling effort for both, and until
        # 2026-09-19 nothing enforced it: the check above catches a duplicate *within* one
        # pool, and `purpose` was recorded without doing any work.
        #
        # Human discipline is the wrong mechanism here. The two sets are built by two separate
        # CLI invocations, days apart, over 600 items — and an overlap is silent, invalidates
        # both uses at once, and is only discoverable after the labelling time is already
        # spent. So it refuses, and names what overlapped.
        overlap = sorted(set(ids) & set(disjoint_from))
        if overlap:
            raise ValueError(
                f"pool overlaps a set it must be disjoint from, on {len(overlap)} item(s): "
                f"{overlap[:5]}{' ...' if len(overlap) > 5 else ''}. The held-out set and the "
                "agreement set measure different things and cannot share items — an item the "
                "teacher is scored against must not also be one it was tuned against."
            )
        # 1.0 is meaningful: offer a repeat after every item. Only >1 is nonsense.
        if not 0.0 <= repeat_fraction <= 1.0:
            raise ValueError(f"repeat_fraction must be in [0, 1], got {repeat_fraction}")

        self.pool = {i.item_id: i for i in pool}
        self.order = [i.item_id for i in pool]
        self.store = Path(store)
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.purpose = purpose
        self.repeat_fraction = repeat_fraction
        self.rng = random.Random(seed)

    # -- persistence -----------------------------------------------------

    def records(self) -> list[LabelRecord]:
        if not self.store.exists():
            return []
        out: list[LabelRecord] = []
        for line in self.store.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(LabelRecord.from_json(json.loads(line)))
        return out

    def _append(self, record: LabelRecord) -> None:
        fd = os.open(self.store, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (json.dumps(record.to_json(), sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def record(
        self,
        item_id: str,
        label: str,
        *,
        note: str = "",
        seconds: float = 0.0,
        presentation: int = 1,
    ) -> LabelRecord:
        if item_id not in self.pool:
            raise KeyError(f"{item_id!r} is not in this session's pool")
        if label not in VALID_LABELS:
            raise ValueError(f"label {label!r} not one of {VALID_LABELS}")
        if label == "unsure" and not note.strip():
            raise ValueError(
                "an 'unsure' label requires a note. 'unsure' without a reason is the one "
                "record that cannot improve the rubric, which is what this set is for."
            )
        rec = LabelRecord(
            item_id=item_id,
            label=label,
            note=note,
            purpose=self.purpose,
            labelled_at=datetime.now(UTC).isoformat(),
            presentation=presentation,
            seconds_spent=seconds,
        )
        self._append(rec)
        return rec

    # -- what to show next -----------------------------------------------

    def first_pass_done(self) -> set[str]:
        return {r.item_id for r in self.records() if r.presentation == 1}

    def pending(self) -> list[str]:
        done = self.first_pass_done()
        return [i for i in self.order if i not in done]

    def next_items(self, limit: int | None = None) -> Iterator[tuple[LabelItem, int]]:
        """Yield `(item, presentation)` pairs: unlabelled items, with repeats mixed in.

        A repeat is drawn from items already labelled *at least ten items ago*, so it
        is not fresh in memory. Re-presenting something just seen measures short-term
        recall, not labelling consistency, and would flatter the ceiling.
        """
        emitted = 0
        for item_id in list(self.pending()):
            if limit is not None and emitted >= limit:
                return
            yield self.pool[item_id], 1
            emitted += 1

            # Re-read the store rather than tracking eligibility in memory: the
            # caller records between yields, so an eligibility set computed once at
            # generator start is frozen at "nothing labelled yet" and never offers a
            # repeat at all. Re-reading is O(n) per item, and n here is ~300.
            recs = self.records()
            firsts = [r.item_id for r in recs if r.presentation == 1]
            repeated = {r.item_id for r in recs if r.presentation == 2}
            # Exclude the most recent ten: re-presenting something just seen measures
            # short-term recall, not labelling consistency, and flatters the ceiling.
            eligible = [i for i in firsts[:-10] if i not in repeated] if len(firsts) > 10 else []

            if eligible and self.rng.random() < self.repeat_fraction:
                if limit is not None and emitted >= limit:
                    return
                yield self.pool[self.rng.choice(eligible)], 2
                emitted += 1

    # -- measurement -----------------------------------------------------

    def intra_rater_pairs(self) -> tuple[list[str], list[str]]:
        """(first-pass, second-pass) labels for every re-presented item.

        Feed these to `qd_train.agreement.kappa_with_ci` to get the ceiling that the
        teacher's kappa is measured against.
        """
        first = {r.item_id: r.label for r in self.records() if r.presentation == 1}
        second = {r.item_id: r.label for r in self.records() if r.presentation == 2}
        shared = [i for i in self.order if i in first and i in second]
        return [first[i] for i in shared], [second[i] for i in shared]

    def stats(self) -> SessionStats:
        recs = self.records()
        firsts = [r for r in recs if r.presentation == 1]
        by_label: dict[str, int] = {}
        by_language: dict[str, int] = {}
        for r in firsts:
            by_label[r.label] = by_label.get(r.label, 0) + 1
            lang = self.pool[r.item_id].language
            by_language[lang] = by_language.get(lang, 0) + 1
        times = sorted(r.seconds_spent for r in firsts if r.seconds_spent > 0)
        median = times[len(times) // 2] if times else 0.0
        return SessionStats(
            total_items=len(self.pool),
            labelled_first_pass=len(firsts),
            repeats_collected=sum(1 for r in recs if r.presentation == 2),
            by_label=by_label,
            by_language=by_language,
            unsure_rate=(by_label.get("unsure", 0) / len(firsts)) if firsts else 0.0,
            median_seconds=median,
        )

    def export_manifest(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        """Write the finished labels, marked held-out so training refuses them.

        `held_out: true` is not a convention here — `qd-train` path-checks it and
        raises rather than training. Items labelled `unsure` are excluded from the
        label set but kept in the manifest, because an unsure item is evidence about
        the rubric and dropping it silently would hide that.
        """
        recs = [r for r in self.records() if r.presentation == 1]
        usable = [r for r in recs if r.label != "unsure"]
        manifest = {
            "schema_version": 1,
            "purpose": self.purpose,
            "held_out": True,
            "created_at": datetime.now(UTC).isoformat(),
            "n_labelled": len(recs),
            "n_usable": len(usable),
            "n_unsure": len(recs) - len(usable),
            "items": [
                {
                    "item_id": r.item_id,
                    "repo": self.pool[r.item_id].repo,
                    "path": self.pool[r.item_id].path,
                    "language": self.pool[r.item_id].language,
                    "label": r.label,
                    "note": r.note,
                    "usable": r.label != "unsure",
                }
                for r in recs
            ],
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest
