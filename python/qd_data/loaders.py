"""Bounded, typed readers for the sources that survive admission.

The raw row types here mirror each upstream schema exactly once, so the rewriters in
:mod:`qd_data.mixture` never touch an untyped ``dict``. A missing field raises
naming the field; a field of the wrong type raises naming both types. Nothing is
defaulted -- ``docs/plan-corrections.md`` is a list of what happens when a field is
assumed to exist.

Three acquisition paths, and they are deliberately not interchangeable:

:func:`read_jsonl`
    The offline path, and the only one the test suite depends on. Bounded by row
    count and by per-row bytes.
:func:`fetch_rows`
    The network path, via the HuggingFace datasets-server REST API. ``datasets`` is
    **not** installed in this environment and the CUDA training stack is not
    installed here either (see ``pyproject.toml``), so this uses ``urllib`` with an
    explicit timeout and an explicit byte bound rather than adding a dependency.
    Every caller is ``@pytest.mark.network`` and is reported **not run** offline.
:func:`load_agentpack`
    ``nuprl/AgentPack`` is gated -- ``docs/plan-corrections.md`` BLOCKING-1. This
    raises a typed refusal naming the gate. It exists so that the day a human
    accepts the terms, the change is registering the source as ``LOADABLE`` and
    writing one adapter, not restructuring the lane. **Nothing here blocks on it.**
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .errors import QdRefusal
from .sources import Reachability, source_by_id

__all__ = [
    "DATASETS_SERVER",
    "MAX_FETCH_ROWS",
    "REFUSED_READS",
    "ClincRow",
    "CommitPackFtRow",
    "CsqaRow",
    "MalformedRowRefusal",
    "MmluRow",
    "RawRow",
    "SourceUnavailableRefusal",
    "SquadRow",
    "check_read_admitted",
    "fetch_rows",
    "load_agentpack",
    "parse_clinc",
    "parse_commitpackft",
    "parse_csqa",
    "parse_mmlu",
    "parse_squad",
    "read_jsonl",
]

#: ``(source_id, config_or_split_name) -> why``. A read naming any of these, as its
#: config or its split, is refused before a byte is fetched.
#:
#: ``cais/mmlu``'s ``auxiliary_train`` is both a config and a split of ``all``, and it
#: is not MMLU: its 99,842 rows are drawn from ARC (cc-by-sa-4.0, opt-in here),
#: OpenBookQA (licence unknown, refused here), RACE and MCTest, re-published under
#: MMLU's MIT card. Admitting it would launder two refused licences through one
#: permissive header.
REFUSED_READS: Final[dict[tuple[str, str], str]] = {
    ("cais/mmlu", "auxiliary_train"): (
        "cais/mmlu auxiliary_train is drawn from ARC, OpenBookQA, RACE and MCTest, not "
        "from MMLU; ARC is share-alike opt-in and OpenBookQA has no licence, so the MIT "
        "card does not cover it"
    ),
    ("tau/commonsense_qa", "test"): (
        "tau/commonsense_qa test carries no answerKey on the hub (1,140 of 1,140 empty), "
        "and it is not in the source's pinned split policy"
    ),
}


def _check_pinned(source_id: str, upstream_split: str) -> None:
    """Refuse a row from an upstream split its source's pinned policy does not list."""
    try:
        source_by_id(source_id).pinned_split_of(upstream_split)
    except KeyError as exc:
        raise MalformedRowRefusal(
            expected=f"an upstream split in {source_id}'s pinned split policy",
            actual=upstream_split, detail=str(exc),
        ) from exc

#: The public rows endpoint. Bounded by ``length`` (the server caps it at 100) and
#: by our own :data:`MAX_FETCH_ROWS`.
DATASETS_SERVER: Final[str] = "https://datasets-server.huggingface.co/rows"

#: "at most a few hundred real rows" -- the lane brief. A larger pull is a deliberate
#: act, not a default.
MAX_FETCH_ROWS: Final[int] = 300
_SERVER_PAGE: Final[int] = 100
_FETCH_TIMEOUT_S: Final[float] = 20.0
_MAX_RESPONSE_BYTES: Final[int] = 8 * 1024 * 1024


class SourceUnavailableRefusal(QdRefusal):
    """The source cannot be read here: gated, script-only, or not on the host.

    Distinct from :class:`~qd_data.errors.LicenceRefused` because the two need
    different responses -- one is a human accepting terms, the other is closed.
    """

    check = "source_reachable"
    #: No runtime counterpart: qd-runtime loads no dataset.
    rust_kinds = ()


class MalformedRowRefusal(QdRefusal):
    """An upstream row did not have the shape its schema promises."""

    check = "row_schema"
    #: No runtime counterpart: an upstream row is not a wire request.
    rust_kinds = ()


def _req_str(raw: dict[str, Any], key: str, *, where: str) -> str:
    if key not in raw:
        raise MalformedRowRefusal(
            expected=f"a {key!r} field", actual=sorted(raw)[:12], detail=where,
        )
    v = raw[key]
    if not isinstance(v, str):
        raise MalformedRowRefusal(
            expected=f"{key!r} as str", actual=type(v).__name__, detail=where,
        )
    return v


def _req_bool(raw: dict[str, Any], key: str, *, where: str) -> bool:
    if key not in raw:
        raise MalformedRowRefusal(
            expected=f"a {key!r} field", actual=sorted(raw)[:12], detail=where,
        )
    v = raw[key]
    if not isinstance(v, bool):
        raise MalformedRowRefusal(
            expected=f"{key!r} as bool", actual=type(v).__name__, detail=where,
        )
    return v


@dataclass(frozen=True, slots=True)
class CommitPackFtRow:
    """``bigcode/commitpackft``. The primary pool -- ``docs/plan-corrections.md``
    BLOCKING-1 promotes it from "human-authored contrast set" to primary, because it
    is the one with a real per-row ``license`` and a ``repos`` field."""

    commit: str
    repos: str
    old_file: str
    new_file: str
    old_contents: str
    new_contents: str
    subject: str
    message: str
    lang: str
    licence: str

    @property
    def primary_repo(self) -> str:
        """``repos`` is a comma-separated list; the split unit is the first entry.

        Using the whole list as the key would put a file vendored into two repos in
        its own third group, which is the leak the MinHash pass exists to catch --
        so the key is the first repo and MinHash is what closes the rest.
        """
        first = self.repos.split(",")[0].strip()
        return first or self.repos.strip()


@dataclass(frozen=True, slots=True)
class ClincRow:
    """``clinc/clinc_oos``. Its out-of-scope class is a free ``noul``."""

    utterance: str
    intent: str
    is_oos: bool


@dataclass(frozen=True, slots=True)
class SquadRow:
    """``rajpurkar/squad_v2``. Its unanswerable questions are a second free ``noul``."""

    qid: str
    title: str
    context: str
    question: str
    answers: tuple[str, ...]
    answer_starts: tuple[int, ...]
    is_impossible: bool

    def __post_init__(self) -> None:
        if len(self.answers) != len(self.answer_starts):
            raise MalformedRowRefusal(
                expected="one start offset per answer",
                actual=f"{len(self.answers)} answers, {len(self.answer_starts)} starts",
                detail=f"squad row {self.qid!r}",
            )
        if self.is_impossible and self.answers:
            raise MalformedRowRefusal(
                expected="no answers on an unanswerable question",
                actual=list(self.answers),
                detail=f"squad row {self.qid!r}: is_impossible with an answer is two labels",
            )


@dataclass(frozen=True, slots=True)
class MmluRow:
    """``cais/mmlu``. Four options and a gold index; no natural ``noul``.

    ``upstream_split`` is carried because MMLU's split is *pinned*: the upstream split,
    not the repo hash, decides train or val (``qd_data.sources`` ``cais/mmlu``). It is
    checked against that policy here, so a row from an unreviewed split cannot exist.
    """

    subject: str
    question: str
    choices: tuple[str, ...]
    answer_index: int
    upstream_split: str

    def __post_init__(self) -> None:
        _check_pinned("cais/mmlu", self.upstream_split)
        if not 0 <= self.answer_index < len(self.choices):
            raise MalformedRowRefusal(
                expected=f"an answer index in 0..{len(self.choices) - 1}",
                actual=self.answer_index, detail=f"mmlu {self.subject!r}",
            )


@dataclass(frozen=True, slots=True)
class CsqaRow:
    """``tau/commonsense_qa``. Five lettered options; the test split is unlabelled.

    ``answer_key`` is kept as the empty string on an unlabelled row rather than being
    refused here: an unlabelled row is not malformed, it is unusable, and the rewriter
    counts it under its own reason code so the thinning is visible. (The hub's test
    split, which is wholly unlabelled, is refused before any row is read.)

    ``upstream_split`` is pinned as MMLU's is (``qd_data.sources`` ``tau/commonsense_qa``).
    """

    qid: str
    question: str
    concept: str
    labels: tuple[str, ...]
    texts: tuple[str, ...]
    answer_key: str
    upstream_split: str

    def __post_init__(self) -> None:
        _check_pinned("tau/commonsense_qa", self.upstream_split)
        if len(self.labels) != len(self.texts):
            raise MalformedRowRefusal(
                expected="one text per choice label",
                actual=f"{len(self.labels)} labels, {len(self.texts)} texts",
                detail=f"csqa row {self.qid!r}",
            )
        if len(set(self.labels)) != len(self.labels):
            raise MalformedRowRefusal(
                expected="distinct choice labels", actual=list(self.labels),
                detail=f"csqa row {self.qid!r}",
            )
        if self.answer_key and self.answer_key not in self.labels:
            raise MalformedRowRefusal(
                expected=f"answerKey in {list(self.labels)}", actual=self.answer_key,
                detail=f"csqa row {self.qid!r}",
            )


RawRow = CommitPackFtRow | ClincRow | SquadRow | MmluRow | CsqaRow


def _req_int(raw: dict[str, Any], key: str, *, where: str) -> int:
    if key not in raw:
        raise MalformedRowRefusal(
            expected=f"a {key!r} field", actual=sorted(raw)[:12], detail=where,
        )
    v = raw[key]
    # bool is an int subclass; True as an answer index is a type error, not 1.
    if not isinstance(v, int) or isinstance(v, bool):
        raise MalformedRowRefusal(
            expected=f"{key!r} as int", actual=type(v).__name__, detail=where,
        )
    return v


def _req_str_list(raw: dict[str, Any], key: str, *, where: str) -> tuple[str, ...]:
    if key not in raw:
        raise MalformedRowRefusal(
            expected=f"a {key!r} field", actual=sorted(raw)[:12], detail=where,
        )
    v = raw[key]
    if not isinstance(v, list):
        raise MalformedRowRefusal(
            expected=f"{key!r} as a list", actual=type(v).__name__, detail=where,
        )
    for item in v:
        if not isinstance(item, str):
            raise MalformedRowRefusal(
                expected=f"{key!r} items as str", actual=type(item).__name__, detail=where,
            )
    return tuple(v)


def parse_commitpackft(raw: dict[str, Any], *, index: int) -> CommitPackFtRow:
    where = f"bigcode/commitpackft row {index}"
    return CommitPackFtRow(
        commit=_req_str(raw, "commit", where=where),
        repos=_req_str(raw, "repos", where=where),
        old_file=_req_str(raw, "old_file", where=where),
        new_file=_req_str(raw, "new_file", where=where),
        old_contents=_req_str(raw, "old_contents", where=where),
        new_contents=_req_str(raw, "new_contents", where=where),
        subject=_req_str(raw, "subject", where=where),
        message=_req_str(raw, "message", where=where),
        lang=_req_str(raw, "lang", where=where),
        # The field is spelled `license` upstream. It is the whole reason this
        # source is primary, so its absence is a refusal, never a default.
        licence=_req_str(raw, "license", where=where),
    )


def parse_clinc(
    raw: dict[str, Any],
    *,
    index: int,
    oos_label: str = "oos",
    label_names: Sequence[str] | None = None,
) -> ClincRow:
    """One CLINC150 row. ``intent`` arrives as a name or as a ``ClassLabel`` index.

    The hub schema types ``intent`` as ``ClassLabel`` (datasets-server ``/info``,
    ``clinc/clinc_oos`` config ``plus``, checked 2026-09-29), so a real row carries an
    ``int``. This parser used to demand ``str`` and would have refused every real row.
    An index is resolved through ``label_names`` -- the ``ClassLabel.names`` list
    from the same ``/info`` response -- and an index with no names to resolve it
    against is refused rather than stringified, because ``"42"`` is not an intent.
    """
    where = f"clinc/clinc_oos row {index}"
    if "intent" not in raw:
        raise MalformedRowRefusal(
            expected="an 'intent' field", actual=sorted(raw)[:12], detail=where,
        )
    label = raw["intent"]
    if isinstance(label, str):
        intent = label
    elif isinstance(label, int) and not isinstance(label, bool):
        if label_names is None:
            raise MalformedRowRefusal(
                expected="label_names to resolve a ClassLabel index", actual=label,
                detail=f"{where}: an intent index without its names is not an intent",
            )
        if not 0 <= label < len(label_names):
            raise MalformedRowRefusal(
                expected=f"an intent index in 0..{len(label_names) - 1}", actual=label,
                detail=where,
            )
        intent = label_names[label]
    else:
        raise MalformedRowRefusal(
            expected="'intent' as str or ClassLabel int", actual=type(label).__name__,
            detail=where,
        )
    return ClincRow(
        utterance=_req_str(raw, "text", where=where),
        intent=intent,
        is_oos=intent == oos_label,
    )


def parse_squad(raw: dict[str, Any], *, index: int) -> SquadRow:
    where = f"rajpurkar/squad_v2 row {index}"
    answers = raw.get("answers")
    if not isinstance(answers, dict):
        raise MalformedRowRefusal(
            expected="an 'answers' object", actual=type(answers).__name__, detail=where,
        )
    texts = answers.get("text", [])
    starts = answers.get("answer_start", [])
    if not isinstance(texts, list) or not isinstance(starts, list):
        raise MalformedRowRefusal(
            expected="answers.text and answers.answer_start as lists",
            actual=f"{type(texts).__name__}, {type(starts).__name__}", detail=where,
        )
    for t in texts:
        if not isinstance(t, str):
            raise MalformedRowRefusal(
                expected="answer text as str", actual=type(t).__name__, detail=where,
            )
    for s in starts:
        if not isinstance(s, int) or isinstance(s, bool):
            raise MalformedRowRefusal(
                expected="answer_start as int", actual=type(s).__name__, detail=where,
            )
    # SQuAD 2.0 on the hub encodes unanswerability as an empty answers list rather
    # than as a flag. Derive it when the flag is absent, and cross-check it when it
    # is present -- SquadRow.__post_init__ refuses a row that says both.
    impossible = (
        _req_bool(raw, "is_impossible", where=where) if "is_impossible" in raw else not texts
    )
    return SquadRow(
        qid=_req_str(raw, "id", where=where),
        title=_req_str(raw, "title", where=where),
        context=_req_str(raw, "context", where=where),
        question=_req_str(raw, "question", where=where),
        answers=tuple(texts),
        answer_starts=tuple(starts),
        is_impossible=impossible,
    )


def parse_mmlu(raw: dict[str, Any], *, index: int, split_name: str) -> MmluRow:
    """One ``cais/mmlu`` row: ``question``, ``subject``, ``choices``, ``answer``.

    ``answer`` is a ``ClassLabel`` over ``A..D`` on the hub, so it arrives as an int
    index into ``choices``. It is refused out of range by :class:`MmluRow`.
    ``split_name`` is the upstream split the row was read from; ``auxiliary_train`` is
    refused by :data:`REFUSED_READS` before the pinned-split check is reached.
    """
    check_read_admitted("cais/mmlu", config_name="all", split_name=split_name)
    where = f"cais/mmlu {split_name} row {index}"
    return MmluRow(
        subject=_req_str(raw, "subject", where=where),
        question=_req_str(raw, "question", where=where),
        choices=_req_str_list(raw, "choices", where=where),
        answer_index=_req_int(raw, "answer", where=where),
        upstream_split=split_name,
    )


def parse_csqa(raw: dict[str, Any], *, index: int, split_name: str) -> CsqaRow:
    """One ``tau/commonsense_qa`` row. ``choices`` is ``{"label": [...], "text": [...]}``.

    ``split_name`` is the upstream split; ``test`` is refused by :data:`REFUSED_READS`.
    """
    check_read_admitted("tau/commonsense_qa", config_name="default", split_name=split_name)
    where = f"tau/commonsense_qa {split_name} row {index}"
    choices = raw.get("choices")
    if not isinstance(choices, dict):
        raise MalformedRowRefusal(
            expected="a 'choices' object", actual=type(choices).__name__, detail=where,
        )
    return CsqaRow(
        qid=_req_str(raw, "id", where=where),
        question=_req_str(raw, "question", where=where),
        concept=_req_str(raw, "question_concept", where=where),
        labels=_req_str_list(choices, "label", where=where),
        texts=_req_str_list(choices, "text", where=where),
        answer_key=_req_str(raw, "answerKey", where=where),
        upstream_split=split_name,
    )


def check_read_admitted(source_id: str, *, config_name: str, split_name: str) -> None:
    """Refuse a read of a config or split that :data:`REFUSED_READS` names.

    Called by :func:`fetch_rows`, and by any offline reader of a file pulled from
    one of these sources -- the refusal is about *which rows*, not about the network.
    """
    for name in (config_name, split_name):
        why = REFUSED_READS.get((source_id, name))
        if why is not None:
            raise SourceUnavailableRefusal(
                expected=f"a config and split of {source_id} that the licence covers",
                actual=f"config={config_name!r}, split={split_name!r}", detail=why,
            )


def read_jsonl(
    path: Path,
    *,
    limit: int,
    max_row_bytes: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Read at most ``limit`` JSON objects. Returns ``(rows, capped)``.

    ``capped`` is returned rather than logged: a caller that ignores it and reports
    the rows as the whole file is the "capped sample presented as complete coverage"
    failure, and the mixture builder turns ``capped`` into a ``NotRun``.
    """
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    if max_row_bytes < 1:
        raise ValueError(f"max_row_bytes must be >= 1, got {max_row_bytes}")
    out: list[dict[str, Any]] = []
    capped = False
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            if len(out) >= limit:
                capped = True
                break
            size = len(line.encode("utf-8"))
            if size > max_row_bytes:
                raise MalformedRowRefusal(
                    expected=f"at most {max_row_bytes} bytes per row", actual=size,
                    detail=f"{path}:{lineno}: refused rather than truncated",
                )
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MalformedRowRefusal(
                    expected="one JSON object per line", actual=line[:64],
                    detail=f"{path}:{lineno}: {exc}",
                ) from exc
            if not isinstance(obj, dict):
                raise MalformedRowRefusal(
                    expected="a JSON object", actual=type(obj).__name__,
                    detail=f"{path}:{lineno}",
                )
            out.append(obj)
    return out, capped


def fetch_rows(
    source_id: str,
    *,
    config_name: str,
    split_name: str,
    limit: int = MAX_FETCH_ROWS,
    timeout_s: float = _FETCH_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Fetch up to ``limit`` real rows from the HF datasets-server.

    Bounded three ways -- row count, response bytes, wall clock -- because an
    unbounded read against a 62 GB dataset is the mistake the lane brief names
    explicitly. Raises :class:`SourceUnavailableRefusal` when the source is not
    reachable unattended, so a gated dataset fails with its gate named rather than
    with an opaque 403.
    """
    source = source_by_id(source_id)
    if source.reachability is not Reachability.LOADABLE:
        raise SourceUnavailableRefusal(
            expected="a source reachable unattended", actual=source.reachability.value,
            detail=f"{source_id}: {source.evidence}",
        )
    check_read_admitted(source_id, config_name=config_name, split_name=split_name)
    if limit < 1 or limit > MAX_FETCH_ROWS:
        raise ValueError(f"limit must be in 1..{MAX_FETCH_ROWS}, got {limit}")

    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        page = min(_SERVER_PAGE, limit - len(rows))
        query = urllib.parse.urlencode(
            {
                "dataset": source_id,
                "config": config_name,
                "split": split_name,
                "offset": offset,
                "length": page,
            }
        )
        request = urllib.request.Request(
            f"{DATASETS_SERVER}?{query}",
            headers={"Accept": "application/json", "User-Agent": "qwen-decision/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                body = resp.read(_MAX_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SourceUnavailableRefusal(
                expected=f"a response from {DATASETS_SERVER}", actual=type(exc).__name__,
                detail=f"{source_id}: {exc}",
            ) from exc
        if len(body) > _MAX_RESPONSE_BYTES:
            raise SourceUnavailableRefusal(
                expected=f"a response under {_MAX_RESPONSE_BYTES} bytes",
                actual=f"over {_MAX_RESPONSE_BYTES}",
                detail=f"{source_id}: refused rather than streamed unbounded",
            )
        payload = json.loads(body.decode("utf-8"))
        page_rows = payload.get("rows")
        if not isinstance(page_rows, list):
            raise MalformedRowRefusal(
                expected="a 'rows' list in the datasets-server response",
                actual=sorted(payload)[:8], detail=source_id,
            )
        if not page_rows:
            break
        for item in page_rows:
            inner = item.get("row") if isinstance(item, dict) else None
            if not isinstance(inner, dict):
                raise MalformedRowRefusal(
                    expected="each entry to carry a 'row' object",
                    actual=type(inner).__name__, detail=source_id,
                )
            rows.append(inner)
        offset += len(page_rows)
    return rows[:limit]


def load_agentpack() -> list[dict[str, Any]]:
    """``nuprl/AgentPack``: gated. Raises, and says exactly what would unblock it.

    Registered rather than omitted so the slot-in is one edit. Nothing in the data
    lane calls this, and ``qd_data.sources`` already excludes AgentPack from the
    admitted set, so no pipeline stalls on it.
    """
    source = source_by_id("nuprl/AgentPack")
    raise SourceUnavailableRefusal(
        expected="an ungated dataset", actual=source.reachability.value,
        detail=(
            "nuprl/AgentPack is gated 'auto' and an authenticated fetch returns 403 "
            "'not in the authorized list'. A human must accept the terms on the dataset "
            "page; then register it as Reachability.LOADABLE in qd_data.sources and add "
            "a parse_agentpack adapter. Until then it is the teacher/held-out target "
            "distribution only, and nothing depends on it. "
            f"Evidence: {source.evidence}"
        ),
    )
