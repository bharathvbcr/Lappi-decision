"""Bounded, typed readers for the three sources that actually survive admission.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .errors import QdRefusal
from .sources import Reachability, source_by_id

__all__ = [
    "DATASETS_SERVER",
    "MAX_FETCH_ROWS",
    "ClincRow",
    "CommitPackFtRow",
    "MalformedRowRefusal",
    "RawRow",
    "SourceUnavailableRefusal",
    "SquadRow",
    "fetch_rows",
    "load_agentpack",
    "parse_clinc",
    "parse_commitpackft",
    "parse_squad",
    "read_jsonl",
]

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


class MalformedRowRefusal(QdRefusal):
    """An upstream row did not have the shape its schema promises."""

    check = "row_schema"


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


RawRow = CommitPackFtRow | ClincRow | SquadRow


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


def parse_clinc(raw: dict[str, Any], *, index: int, oos_label: str = "oos") -> ClincRow:
    where = f"clinc/clinc_oos row {index}"
    intent = _req_str(raw, "intent", where=where)
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
