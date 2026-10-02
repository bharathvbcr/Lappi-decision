"""The v5 containment rule's template strip (Fable's post-F ruling 3(a), 2026-10-02).

``campaign/v5-preregistered.DRAFT.json`` ``data.decontamination.rule``: constant template
text is stripped before n-gramming, for containment only. Within each (family, slot), any
question line and any option value whose text is byte-identical across every rendered row of
that family-slot, in the union of sources and targets, is removed from ``prompt_content``.

This module is that one function, :func:`strip_template`. ``tools/containment_scan.py``
applies it to every set before the ``qd-prep containment`` request is written, and the parity
oracle (``python/tests/test_containment_strip.py``) applies the same function before
``qd_train.replay.decontaminate``, so qd-prep's n-gram core is unchanged and the parity test
still compares complete pair lists byte for byte. ``qd_train.replay.prompt_content`` itself is
NOT changed: J6(a)'s replay decontamination reads it, and the strip is for containment only.

How the binding text is read (each choice is recorded in the attestation's
``export.template_strip`` and named in ``HANDOFF/prep2-2026-10-02.md``):

* **The question line** is the ``<|qd_question|>`` line. Every ``qd_data`` family renders
  ``family.description`` there (``qd_data.mixture._request``), so it is constant in every
  family, MMLU, CSQA, SQuAD and defect_class included, and is stripped in every family. Their
  per-row question is part of the CONTEXT block (``qd_data.general``: ``context=question``;
  SQuAD: question, blank line, passage), and the context is never stripped.
* **An option value is constant** when it is a member of every row's option values (canonical
  order, ``seed=None``). For a fixed list the positional reading gives the same set.
* **A group of fewer than** :data:`MIN_ROWS_FOR_CONSTANT` **rows strips nothing.** With one
  row, "identical across every row" is vacuous and would strip that row's own question and
  options; the attestation lists such groups.
* **too_short**: a text with fewer than ``n`` words has no ``n``-gram. The record counts it per
  (family, slot) and per set, before and after the strip, because a val row made too short by
  the strip can no longer be hit -- it is unprotected, not clean.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, Final

from .replay import DEFAULT_N, ReplayRefusal, prompt_content

__all__ = [
    "MIN_ROWS_FOR_CONSTANT",
    "STRIP_RULE",
    "STRIP_VERSION",
    "Constants",
    "PartsRow",
    "SlotParts",
    "TextRow",
    "slot_parts",
    "strip_template",
    "template_constants",
    "too_short",
]

#: What the attestation names the strip as, and the version of this reading of it.
STRIP_RULE: Final[str] = (
    "campaign/v5-preregistered data.decontamination.rule: constant template text stripped "
    "before n-gramming, for containment only (Fable post-F ruling 3(a))"
)
STRIP_VERSION: Final[int] = 1
#: Fewer rows than this in a (family, slot) group and nothing in it is called constant.
MIN_ROWS_FOR_CONSTANT: Final[int] = 2
#: ``qd_train.replay.word_ngrams``'s word: ``\w+`` over ``str.lower``.
_WORD: Final[re.Pattern[str]] = re.compile(r"\w+")


@dataclass(frozen=True)
class SlotParts:
    """One rendered slot prompt, cut where ``prompt_content`` cuts it.

    ``question`` and ``context`` are ``None`` when the prompt has no such block (a present but
    empty one is ``""``, as ``prompt_content`` keeps it); ``options`` excludes the ``noul``
    line, as ``prompt_content`` does.
    """

    question: str | None
    context: str | None
    options: tuple[str, ...]

    def joined(self, *, question: bool = True, drop: frozenset[str] = frozenset()) -> str:
        """``prompt_content``'s text, less the question when ``question`` is false and less
        every option value in ``drop``."""
        parts: list[str] = []
        if question and self.question is not None:
            parts.append(self.question)
        if self.context is not None:
            parts.append(self.context)
        parts.extend(o for o in self.options if o not in drop)
        return "\n".join(parts)


def slot_parts(prompt: str) -> SlotParts:
    """``prompt`` cut into question, context and option values, checked against
    ``qd_train.replay.prompt_content``: the unstripped join must be its text, byte for byte,
    or this refuses -- the two cuts can never drift silently."""
    from qd_data.render import M_CTX_BEGIN, M_CTX_END, M_OPT_BEGIN, M_OPT_END, M_QUESTION
    from qd_data.schema import NOUL, NOUL_LETTER

    question: str | None = None
    q = prompt.find(M_QUESTION)
    if q >= 0:
        start = q + len(M_QUESTION)
        question = prompt[start : prompt.find("\n", start)]
    context: str | None = None
    c0 = prompt.find(M_CTX_BEGIN + "\n")
    c1 = prompt.rfind("\n" + M_CTX_END)
    if c0 >= 0 and c1 > c0:
        context = prompt[c0 + len(M_CTX_BEGIN) + 1 : c1]
    options: list[str] = []
    o0 = prompt.rfind(M_OPT_BEGIN + "\n")
    o1 = prompt.rfind(M_OPT_END)
    if o0 >= 0 and o1 > o0:
        for line in prompt[o0 + len(M_OPT_BEGIN) + 1 : o1].splitlines():
            if line == f"{NOUL_LETTER}. {NOUL}":
                continue
            options.append(line.split(". ", 1)[1] if ". " in line else line)
    parts = SlotParts(question=question, context=context, options=tuple(options))
    if parts.joined() != prompt_content(prompt):
        raise ReplayRefusal(
            "slot_parts and qd_train.replay.prompt_content cut this prompt differently; the "
            "strip would compare a text the unstripped scan never held"
        )
    return parts


#: ``(key, identity_key, family_id, slot, parts)``: one rendered slot of one row.
PartsRow = tuple[str, str, str, str, SlotParts]
#: ``(key, identity_key, family_id, text)``: what ``qd-prep containment`` is handed.
TextRow = tuple[str, str, str, str]


@dataclass(frozen=True)
class Constants:
    """What one (family, slot) group strips: its constant question (``None`` if it has none)
    and its constant option values."""

    rows: int
    question: str | None
    options: frozenset[str]

    def strings(self) -> dict[str, Any]:
        return {"question": self.question, "options": sorted(self.options)}

    def sha256(self) -> str:
        raw = json.dumps(self.strings(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @property
    def n_strings(self) -> int:
        return len(self.options) + (self.question is not None)


def template_constants(rows: Iterable[PartsRow]) -> dict[tuple[str, str], Constants]:
    """Per (family, slot) over ``rows`` (the union of every set): the question line every row
    carries byte for byte, and the option values every row carries. A group of fewer than
    :data:`MIN_ROWS_FOR_CONSTANT` rows strips nothing."""
    count: Counter[tuple[str, str]] = Counter()
    question: dict[tuple[str, str], str | None] = {}
    varied: set[tuple[str, str]] = set()
    options: dict[tuple[str, str], frozenset[str]] = {}
    for _key, _identity, family, slot, parts in rows:
        group = (family, slot)
        count[group] += 1
        if group not in question:
            question[group] = parts.question
            options[group] = frozenset(parts.options)
            continue
        if group not in varied and parts.question != question[group]:
            varied.add(group)
        if options[group]:
            options[group] = options[group] & frozenset(parts.options)
    out: dict[tuple[str, str], Constants] = {}
    for group, n in count.items():
        if n < MIN_ROWS_FOR_CONSTANT:
            out[group] = Constants(rows=n, question=None, options=frozenset())
            continue
        out[group] = Constants(
            rows=n,
            question=None if group in varied else question[group],
            options=options[group],
        )
    return out


def too_short(text: str, n: int = DEFAULT_N) -> bool:
    """Fewer than ``n`` words, so ``word_ngrams(text, n)`` is empty: no row can hit it and it
    can hit no row. Stops counting at ``n``."""
    return sum(1 for _ in islice(_WORD.finditer(text.lower()), n)) < n


def strip_template(
    sets: Mapping[str, Sequence[PartsRow]], *, n: int = DEFAULT_N, apply: bool = True,
) -> tuple[dict[str, list[TextRow]], dict[str, Any]]:
    """THE strip: ``({set: [(key, identity, family, text)]}, record)``.

    Constants are found over the union of every set in ``sets`` (sources and targets alike)
    and removed from every row of every set by the same rule; the record is the attestation's
    ``export.template_strip``. ``apply=False`` returns the unstripped texts -- byte for byte
    what ``prompt_content`` gives, which the pre-strip scans compared -- and records
    ``applied: false``; ``qd_train.exclusions`` refuses a list made that way.
    """
    if n < 1:
        raise ValueError(f"n={n} is not an n-gram length")
    constants = template_constants(r for rows in sets.values() for r in rows)
    texts: dict[str, list[TextRow]] = {}
    shorts: dict[tuple[str, str], dict[str, list[int]]] = {}
    option_sets: dict[tuple[str, str], Counter[tuple[str, ...]]] = {}
    for name, rows in sets.items():
        out: list[TextRow] = []
        for key, identity, family, slot, parts in rows:
            group = (family, slot)
            c = constants[group]
            before = parts.joined()
            after = (
                parts.joined(question=c.question is None, drop=c.options) if apply else before
            )
            out.append((key, identity, family, after))
            tally = shorts.setdefault(group, {}).setdefault(name, [0, 0, 0])
            tally[0] += 1
            tally[1] += too_short(before, n)
            tally[2] += too_short(after, n)
            option_sets.setdefault(group, Counter())[parts.options] += 1
        texts[name] = out
    record = _record(constants, shorts, option_sets, n=n, apply=apply)
    return texts, record


def _record(
    constants: Mapping[tuple[str, str], Constants],
    shorts: Mapping[tuple[str, str], Mapping[str, list[int]]],
    option_sets: Mapping[tuple[str, str], Counter[tuple[str, ...]]],
    *, n: int, apply: bool,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    too_small: list[str] = []
    for (family, slot), c in sorted(constants.items()):
        name = f"{family}/{slot}"
        if c.rows < MIN_ROWS_FOR_CONSTANT:
            too_small.append(name)
        by_set = {
            s: {"rows": t[0], "too_short_before_strip": t[1], "too_short_after_strip": t[2]}
            for s, t in sorted(shorts[(family, slot)].items())
        }
        sets_seen = option_sets[(family, slot)]
        groups[name] = {
            "rows": c.rows,
            "stripped": c.strings() if apply else {"question": None, "options": []},
            "n_stripped": c.n_strings if apply else 0,
            "stripped_sha256": c.sha256() if apply else None,
            "too_short_before_strip": sum(t["too_short_before_strip"] for t in by_set.values()),
            "too_short_after_strip": sum(t["too_short_after_strip"] for t in by_set.values()),
            "by_set": by_set,
            # Whether a per-(family, option-set) rule would bite: how many distinct option
            # lists the group has, and how many rows share its most common one.
            "option_sets": {
                "distinct": len(sets_seen),
                "largest_share_rows": max(sets_seen.values()) if sets_seen else 0,
            },
        }
    return {
        "rule": STRIP_RULE,
        "version": STRIP_VERSION,
        "applied": apply,
        "n": n,
        "constancy": (
            "per (family, slot) over the union of every set's rendered rows at seed=None: the "
            "question line if byte-identical in every row; each option value present in "
            "every row's option values"
        ),
        "min_rows_for_constant": MIN_ROWS_FOR_CONSTANT,
        "groups_below_min_rows": too_small,
        "family_slots": groups,
    }
