"""The v5 containment rule's template strip, version 2 (Fable's CLINC strip ruling, 2026-10-02).

``campaign/v5-preregistered.DRAFT.json`` ``data.decontamination.rule``, as amended by
``AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md`` section 1: constant template text is
stripped before n-gramming, for containment only. Template is

1. **the question line**, the ``<|qd_question|>`` line. Every ``qd_data`` family renders
   ``family.description`` there (``qd_data.mixture._request``), so it is constant within
   every (family, slot) and is stripped in every family, by constancy: byte-identical in
   every rendered row of the family-slot. The per-row question of MMLU, CSQA and SQuAD is in
   the CONTEXT block (``qd_data.general``: ``context=question``; SQuAD: question, blank line,
   passage), and the context is never stripped.
2. **(2a) family-constant option values.** Within each (family, slot), every option value
   that is a member of every rendered row's option values (canonical order, ``seed=None``,
   over the union of sources and targets). For a fixed list the positional reading gives the
   same set.
3. **(2b) every option value of** :data:`CLOSED_VOCABULARY_FAMILIES` (intent.classification,
   intent.domain, intent.in_scope, intent.within_domain), because their options are drawn
   from CLINC's closed label vocabulary and are never written per row. What remains of an
   intent.* row is its context block, the utterance, identical across the four families;
   :func:`strip_template` refuses if any (2b) row's stripped text differs from its context
   block byte for byte. No other family is in (2b): MMLU's and CSQA's option values are
   per-row content and are kept, shared sets included.

**A group of fewer than** :data:`MIN_ROWS_FOR_CONSTANT` **rows strips nothing** and is named
in the record: with one row, "identical across every row" is vacuous and would strip that
row's own question and options. A (2b) group that small keeps its question line, so its rows
fail the context check and the strip refuses rather than hand containment a template.

This module is that one function, :func:`strip_template`. ``tools/containment_scan.py``
applies it to every set before the ``qd-prep containment`` request is written, and the parity
oracle (``python/tests/test_containment_strip.py``) applies the same function before
``qd_train.replay.decontaminate``, so qd-prep's n-gram core is unchanged and the parity test
still compares complete pair lists byte for byte. ``qd_train.replay.prompt_content`` itself is
NOT changed: J6(a)'s replay decontamination reads it, and the strip is for containment only.

**too_short**: a slot text left with fewer than ``n`` words has no ``n``-gram, so key (ii)
cannot see it. The record counts it per (family, slot) and per set, before and after the
strip, and names per set the distinct identity keys all of whose slot texts are too short
(``key_ii_blind``): those keys are covered by keys (i) and (iii) only, and are reported as
unprotected by key (ii), never as clean.
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
    "CLOSED_VOCABULARY_FAMILIES",
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
#: ``qd_train.exclusions.read_exclusions`` compares both exactly, so a list made under any
#: other rule or version (version 1 included) is refused.
STRIP_RULE: Final[str] = (
    "campaign/v5-preregistered data.decontamination.rule: constant template text stripped "
    "before n-gramming, for containment only (Fable post-F ruling 3(a)), as amended by Fable's "
    "CLINC strip ruling (AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md section 1): "
    "(1) the question line, by constancy; (2a) option values every row of the family-slot "
    "carries; (2b) every option value of intent.classification, intent.domain, "
    "intent.in_scope and intent.within_domain, refused unless the stripped text is the "
    "context block byte for byte"
)
STRIP_VERSION: Final[int] = 2
#: Fewer rows than this in a (family, slot) group and nothing in it is called constant.
MIN_ROWS_FOR_CONSTANT: Final[int] = 2
#: (2b): the families whose option values are CLINC's closed label vocabulary
#: (``qd_data.mixture.rewrite_clinc``, ``qd_data.general.rewrite_clinc_two_stage``), never
#: written per row. Every option value of theirs is template. Exactly the task families
#: sourced from ``clinc/clinc_oos`` (``qd_data.sources.TASK_FAMILIES``); a test holds that.
CLOSED_VOCABULARY_FAMILIES: Final[frozenset[str]] = frozenset({
    "intent.classification", "intent.domain", "intent.in_scope", "intent.within_domain",
})
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
    and the option values it strips -- (2a) those every row carries, or (2b), for a
    :data:`CLOSED_VOCABULARY_FAMILIES` group, every value any row carries.

    ``distinct_option_values`` is the number of distinct option values the group's rows
    carry for a (2b) group, and ``None`` for a (2a) group."""

    rows: int
    question: str | None
    options: frozenset[str]
    distinct_option_values: int | None = None

    @property
    def closed_vocabulary(self) -> bool:
        """A (2b) group: every row's stripped text must be its context block."""
        return self.distinct_option_values is not None

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
    carries byte for byte (1), and the option values to strip -- (2a) those every row
    carries, or (2b) in a :data:`CLOSED_VOCABULARY_FAMILIES` family every value any row
    carries. A group of fewer than :data:`MIN_ROWS_FOR_CONSTANT` rows strips nothing."""
    count: Counter[tuple[str, str]] = Counter()
    question: dict[tuple[str, str], str | None] = {}
    varied: set[tuple[str, str]] = set()
    options: dict[tuple[str, str], frozenset[str]] = {}
    # (2b) only: every value seen. Not grown for the other families, whose per-row option
    # values (MMLU, CSQA) are content and would only cost memory here.
    vocabulary: dict[tuple[str, str], set[str]] = {}
    for _key, _identity, family, slot, parts in rows:
        group = (family, slot)
        count[group] += 1
        if family in CLOSED_VOCABULARY_FAMILIES:
            vocabulary.setdefault(group, set()).update(parts.options)
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
        seen = vocabulary.get(group)
        distinct = None if seen is None else len(seen)
        if n < MIN_ROWS_FOR_CONSTANT:
            out[group] = Constants(rows=n, question=None, options=frozenset(),
                                   distinct_option_values=distinct)
            continue
        out[group] = Constants(
            rows=n,
            question=None if group in varied else question[group],
            options=options[group] if seen is None else frozenset(seen),
            distinct_option_values=distinct,
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

    Refused (``ReplayRefusal``), with the strip applied: a (2b) row whose stripped text is
    not its context block byte for byte -- the rule says what remains of an intent.* row is
    its utterance, and a text that is anything else would be compared as if it were.
    """
    if n < 1:
        raise ValueError(f"n={n} is not an n-gram length")
    constants = template_constants(r for rows in sets.values() for r in rows)
    texts: dict[str, list[TextRow]] = {}
    shorts: dict[tuple[str, str], dict[str, list[int]]] = {}
    option_sets: dict[tuple[str, str], Counter[tuple[str, ...]]] = {}
    # (2b): the rows whose stripped text was compared with the context block and was it.
    checked: Counter[tuple[str, str]] = Counter()
    blind: dict[str, list[str]] = {}
    for name, rows in sets.items():
        out: list[TextRow] = []
        # identity key -> every slot text of it in this set so far is too short.
        all_short: dict[str, bool] = {}
        for key, identity, family, slot, parts in rows:
            group = (family, slot)
            c = constants[group]
            before = parts.joined()
            after = (
                parts.joined(question=c.question is None, drop=c.options) if apply else before
            )
            if apply and c.closed_vocabulary:
                if after != parts.context:
                    raise ReplayRefusal(_not_the_context(key, group, c, parts))
                checked[group] += 1
            out.append((key, identity, family, after))
            short_after = too_short(after, n)
            tally = shorts.setdefault(group, {}).setdefault(name, [0, 0, 0])
            tally[0] += 1
            tally[1] += too_short(before, n)
            tally[2] += short_after
            all_short[identity] = all_short.get(identity, True) and short_after
            option_sets.setdefault(group, Counter())[parts.options] += 1
        texts[name] = out
        blind[name] = sorted(k for k, short in all_short.items() if short)
    record = _record(constants, shorts, option_sets, checked, blind, n=n, apply=apply)
    return texts, record


def _not_the_context(
    key: str, group: tuple[str, str], c: Constants, parts: SlotParts,
) -> str:
    family, slot = group
    if c.rows < MIN_ROWS_FOR_CONSTANT:
        why = (f"its group has {c.rows} row(s), fewer than {MIN_ROWS_FOR_CONSTANT}, so its "
               "question line is not shown constant and is not stripped")
    elif parts.context is None:
        why = "it has no context block"
    elif c.question is None:
        why = "its group's question line is not byte-identical in every row"
    else:
        why = "the stripped text is not the context block"
    return (
        f"template strip (2b): {key} ({family}/{slot}) would hand containment a text other "
        f"than its context block byte for byte, because {why}. What remains of an intent.* "
        "row is its utterance, or the strip refuses (AUDIT/prep2-2026-10-02/"
        "fable-clinc-strip-ruling.md section 1)"
    )


def _record(
    constants: Mapping[tuple[str, str], Constants],
    shorts: Mapping[tuple[str, str], Mapping[str, list[int]]],
    option_sets: Mapping[tuple[str, str], Counter[tuple[str, ...]]],
    checked: Mapping[tuple[str, str], int],
    blind: Mapping[str, Sequence[str]],
    *, n: int, apply: bool,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    too_small: list[str] = []
    # Every set named, an empty one at 0, as key_ii_blind names it.
    per_set: Counter[str] = Counter(dict.fromkeys(blind, 0))
    for (family, slot), c in sorted(constants.items()):
        name = f"{family}/{slot}"
        if c.rows < MIN_ROWS_FOR_CONSTANT:
            too_small.append(name)
        by_set = {
            s: {"rows": t[0], "too_short_before_strip": t[1], "too_short_after_strip": t[2]}
            for s, t in sorted(shorts[(family, slot)].items())
        }
        for s, t in by_set.items():
            per_set[s] += t["too_short_after_strip"]
        sets_seen = option_sets[(family, slot)]
        groups[name] = {
            "rows": c.rows,
            "stripped": c.strings() if apply else {"question": None, "options": []},
            "n_stripped": c.n_strings if apply else 0,
            "stripped_sha256": c.sha256() if apply else None,
            # (2b) only, else null: the distinct option values the rows carry (all of them
            # stripped), and how many rows were checked to be their context block after the
            # strip -- equal to ``rows`` when the strip was applied, 0 when it was not.
            "closed_vocabulary": None if not c.closed_vocabulary else {
                "distinct_option_values": c.distinct_option_values,
                "rows_checked_equal_to_context": checked.get((family, slot), 0),
            },
            "too_short_before_strip": sum(t["too_short_before_strip"] for t in by_set.values()),
            "too_short_after_strip": sum(t["too_short_after_strip"] for t in by_set.values()),
            "by_set": by_set,
            # How many distinct option lists the group has, and how many rows share its most
            # common one (a per-(family, option-set) statistic, recorded, never applied).
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
            "per (family, slot) over the union of every set's rendered rows at seed=None: "
            "(1) the question line if byte-identical in every row; (2a) each option value "
            "present in every row's option values; (2b) in closed_vocabulary_families, every "
            "option value any row carries, and each row's stripped text must be its context "
            "block byte for byte"
        ),
        "closed_vocabulary_families": sorted(CLOSED_VOCABULARY_FAMILIES),
        "min_rows_for_constant": MIN_ROWS_FOR_CONSTANT,
        "groups_below_min_rows": too_small,
        # Slot texts, every family-slot summed: an intent.* utterance counts once per intent.*
        # family it is asked in. ``key_ii_blind`` counts identity keys.
        "too_short_after_strip_by_set": dict(sorted(per_set.items())),
        # Per set, the complete sorted list of identity keys ALL of whose slot texts in that
        # set have fewer than n words: key (ii) cannot see them. Covered by keys (i) and (iii)
        # only -- unprotected by key (ii), never clean.
        "key_ii_blind": {
            s: {
                "n": len(keys),
                "sha256": hashlib.sha256(
                    "".join(f"{k}\n" for k in keys).encode("utf-8")
                ).hexdigest(),
                "identity_keys": list(keys),
            }
            for s, keys in sorted(blind.items())
        },
        "family_slots": groups,
    }
