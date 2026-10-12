"""Shared pieces of the span-refit measurement (campaign/span-refit-mac-preregistered.json).

Throwaway analysis (house Python policy, category 3): it never ships. Everything that decides
a number here either is a repo function called by import, or is a pure function small enough
for span_refit_test.py to pin by hand.

Nothing in this module loads a tower or touches a device beyond what its caller passes in.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

REPO = Path("/Users/bharath/Code/research/Lappi-decision")
#: Where the repo's python/ and tools/ are imported from. The chain's legs must be read by one
#: code tree: runner 4 ran on 8a531e8, and the v6 data merge then changed qd_data and qd_train
#: under it on main (runner 7's ShardContractViolation). SPAN_CODE_REPO names a checkout pinned
#: to that commit; unset, it is main's checkout. Outputs and the pre-registration stay on main.
CODE_REPO = Path(os.environ.get("SPAN_CODE_REPO", str(REPO)))
BUILD = REPO / "build" / "needle-span-mac-2026-10-06"
CACHE = BUILD / "cache"
PREREG = REPO / "campaign" / "span-refit-mac-preregistered.json"

#: The pin the coordinator handed this lane; the prereg's own text is parsed too and the two
#: must agree, so neither can drift silently.
POPULATION_PIN = "bdd571b675d83b92b2180d900cee79c9a89553f240c07fac421bcff47eb4e626"

#: How the rendered context ends (qd_data/render.py M_CTX_END, preceded by the newline that
#: ends the last context line). Arm (e)'s last line ends at the token before the one whose
#: decoded text starts this.
CONTEXT_END_DELIMITER = "\n<|qd_context_end|>"

FAMILIES = ("qa.answer_span", "code.defect_class")
CELLS = tuple(f"{f}/{k}" for f in FAMILIES for k in ("pointing", "abstain"))


def repo_imports() -> None:
    """Put the repo's python/ and tools/ on sys.path so its own functions import by name."""
    if not (CODE_REPO / "python" / "qd_train" / "shards.py").is_file():
        raise SystemExit(f"SPAN_CODE_REPO={CODE_REPO} holds no python/qd_train; refused")
    for p in (CODE_REPO / "python", CODE_REPO / "tools"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def sha256_file(path: Path, *, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def prereg_sha256() -> str:
    """Computed at run time, never hard-coded: the file was amended (a citation fix) after
    this lane started, and a pinned constant would have recorded the wrong one."""
    return sha256_file(PREREG)


def load_prereg() -> dict[str, Any]:
    raw = json.loads(PREREG.read_text(encoding="utf-8"))
    pin_text = str(raw["population"]["pin"])
    found = re.findall(r"\b[0-9a-f]{64}\b", pin_text)
    if found != [POPULATION_PIN]:
        raise SystemExit(
            f"{PREREG}: population.pin names {found}, not the pin this lane was given "
            f"({POPULATION_PIN}); refusing to measure against a moved population"
        )
    # The thresholds the reading uses, refused unless the prereg still states them.
    reading = json.dumps(raw["reading"]) + json.dumps(raw["arms"]["a"])
    for needle in ("6,562", "5,667", "5,596", "+/-15", "7,238"):
        if needle not in reading:
            raise SystemExit(f"{PREREG}: the reading no longer states {needle!r}; refusing")
    return raw


def family_of(row_id: str) -> str:
    """``<source>:<family>:...`` -- checked against verdicts-avg.jsonl's family_id on all
    7,238 val span rows (2026-10-06, scratch probe; 0 disagreements)."""
    parts = row_id.split(":")
    if len(parts) < 3:
        raise ValueError(f"row id {row_id!r} has no family field")
    return parts[1]


def population_pin(keys: Iterable[str]) -> str:
    """sha256 of the newline-joined sorted list of ``row_id|slot_name`` (no trailing newline),
    exactly as the prereg's population.pin states it."""
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


# --- arm (e): where each line ends ---------------------------------------------------------


@dataclass(frozen=True)
class Columns:
    """One span sequence's candidate columns, in the head's order (plan.candidate_pos).

    ``starts`` are the distinct line-start tokens (what heads.plan_span_batch gathers);
    ``ends`` the token each column's line(s) end at; ``line_ends`` the prereg's per-line
    derivation over the shard's one-entry-per-line ``candidates``. ``plus`` are the column
    ordinals whose line's first character is '+'. ``status`` is "ok" or why arm (e)'s
    validity check failed on this row.
    """

    starts: tuple[int, ...]
    ends: tuple[int, ...]
    line_ends: tuple[int, ...]
    plus: tuple[int, ...]
    delimiter_token: int
    status: str
    #: "original" or "amendment-1" -- which rule set the last line's end.
    last_rule: str = "original"
    #: The status the ORIGINAL (pre-amendment) rule gives this row, for the census.
    original_status: str = "ok"


def char_starts(pieces: Sequence[str]) -> list[int]:
    out, at = [], 0
    for p in pieces:
        out.append(at)
        at += len(p)
    return out


def derive_columns(
    candidates: Sequence[int], pieces: Sequence[str], *, delimiter: str = CONTEXT_END_DELIMITER
) -> Columns:
    """The prereg's arm-(e) derivation, decode only; line starts are never recomputed.

    * ``candidates`` is the shard's per-line list (ShardReader.candidates): one entry per
      context line in order, collapsed lines sharing one token index.
    * ``pieces[t]`` is token ``t`` decoded alone (RealTokenizer.decode([id]), the Decode the
      shard writer's span check uses).

    Per line j: ``end_j = cand[j+1] - 1`` when ``cand[j+1] > cand[j]``; a collapsed line
    (``cand[j+1] == cand[j]``) ends at its shared token; the last line ends at the token
    before the one whose decoded text starts ``delimiter`` after the last candidate.

    The head's columns are the DISTINCT starts (heads.plan_span_batch reads the line_starts
    mask, a set), so a collapsed group is one column; its end is the end of the group's last
    line, i.e. the next distinct start minus one. On a set with no collapse (val is written
    refuse-any) columns and lines are the same thing.

    Validity, per column: ``start_c <= end_c < start_{c+1}`` (the strict ``<`` is between
    distinct starts; a collapsed pair cannot satisfy it per line by construction).

    **Amendment 1** (prereg ``amendments[0]``, Fable 2026-10-06, pre-result): for the LAST
    candidate only, if the delimiter's leading newline lies inside the token at
    ``start_last`` (a whitespace-only last line BPE-merged with the renderer's newline, e.g.
    ' \\n' or '\\n \\n'), then ``end_last = start_last``, confirmed per row by the token at
    ``start_last + 1`` decoding to a beginning of '<|qd_context_end|>' (the marker without its
    newline). A row where that confirmation fails is a derivation failure, counted, never
    patched. Every other line keeps the original rule; monotonicity is unchanged, with
    ``end_last`` < the marker's first token.
    """
    cand = [int(c) for c in candidates]
    if not cand:
        raise ValueError("a span sequence has at least one line")
    if any(b < a for a, b in zip(cand, cand[1:])):
        raise ValueError(f"candidates are not non-decreasing: {cand[:16]}")
    starts = sorted(set(cand))
    cs = char_starts(pieces)
    text = "".join(pieces)
    last = cand[-1]
    if not 0 <= last < len(pieces):
        raise ValueError(f"last candidate {last} is outside {len(pieces)} tokens")
    at = text.find(delimiter, cs[last])
    if at < 0:
        delim_tok = -1
    else:
        delim_tok = bisect.bisect_right(cs, at) - 1
    marker = delimiter.lstrip("\n")

    def check(ends: list[int], marker_first: int) -> str:
        if at < 0:
            return "delimiter-not-found"
        for c, (s, e) in enumerate(zip(starts, ends)):
            nxt = starts[c + 1] if c + 1 < len(starts) else marker_first
            if not s <= e < nxt:
                return f"nonmonotone-column-{c}-of-{len(starts)}"
        return "ok"

    # The original rule, kept for the census (what amendment 1 was recorded against).
    original_ends = [s2 - 1 for s2 in starts[1:]] + [delim_tok - 1]
    original_status = check(original_ends, delim_tok)
    last_end, last_rule, marker_first = delim_tok - 1, "original", delim_tok
    amended_failure = ""
    if at >= 0 and delim_tok == last:
        # Amendment 1: the delimiter's leading newline lies inside the token at start_last.
        last_rule = "amendment-1"
        last_end, marker_first = last, last + 1
        nxt_piece = pieces[last + 1] if last + 1 < len(pieces) else ""
        if not (nxt_piece and marker.startswith(nxt_piece)):
            amended_failure = "amendment-1-marker-not-next"
    line_ends: list[int] = []
    for j, c in enumerate(cand):
        if j + 1 < len(cand):
            line_ends.append(cand[j + 1] - 1 if cand[j + 1] > c else c)
        else:
            line_ends.append(last_end)
    ends = [s2 - 1 for s2 in starts[1:]] + [last_end]
    status = amended_failure or check(ends, marker_first)
    plus = tuple(c for c, s in enumerate(starts) if _line_first_char(s, pieces, cs, text) == "+")
    return Columns(
        starts=tuple(starts), ends=tuple(ends), line_ends=tuple(line_ends), plus=plus,
        delimiter_token=delim_tok, status=status, last_rule=last_rule,
        original_status=original_status,
    )


def _line_first_char(tok: int, pieces: Sequence[str], cs: Sequence[int], text: str) -> str:
    """The first character of the line that starts inside token ``tok``.

    The shard writer's span check established that every recorded candidate is the token
    holding a line start of the decoded text. A token may begin with the previous line's
    newline (e.g. '\\n\\n' holding a blank line's start), so the line start is the token's
    first character when the text before it ends a line, else the first character inside
    the token that follows a newline. '' when it cannot be resolved (counted by callers).
    """
    a = cs[tok]
    b = a + len(pieces[tok])
    if a == 0 or text[a - 1] == "\n":
        start = a
    else:
        start = next((i for i in range(a + 1, b + 1) if text[i - 1] == "\n"), -1)
    if start < 0 or start >= len(text):
        return ""
    return text[start]


# --- features ------------------------------------------------------------------------------


def gather_features(
    hidden: torch.Tensor, candidate_pos: torch.Tensor, query_index: torch.Tensor,
    end_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``(h_query [K,H], h_start [K,C,H], h_end [K,C,H] or None)`` gathered exactly as
    heads.SpanPointerHead.forward gathers them (heads.py:357-361): ``hidden`` is the span rows'
    hidden states ``[K, L, H]``; padded columns gather position 0 and are masked by callers."""
    k, _, h = hidden.shape
    index = candidate_pos.unsqueeze(-1).expand(-1, -1, h)
    h_cand = hidden.gather(1, index)
    h_query = hidden.gather(1, query_index.view(k, 1, 1).expand(-1, 1, h)).squeeze(1)
    h_end = None
    if end_pos is not None:
        h_end = hidden.gather(1, end_pos.unsqueeze(-1).expand(-1, -1, h))
    return h_query, h_cand, h_end


# --- the cache on disk ---------------------------------------------------------------------


def bf16_memmap(path: Path, shape: tuple[int, ...], mode: str) -> np.memmap:
    """bf16 stored as raw uint16 (numpy has no bfloat16)."""
    return np.memmap(path, dtype=np.uint16, mode=mode, shape=shape)


def to_bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def from_bf16_bits(a: np.ndarray) -> torch.Tensor:
    """fp32 tensor from stored bf16 bits (exact upcast)."""
    return torch.from_numpy(np.array(a, dtype=np.uint16, copy=True).view(np.int16)).view(torch.bfloat16).float()


@dataclass
class Cache:
    """A feature cache directory written by span_refit_extract.py."""

    root: Path
    layout: dict[str, Any]
    rows: list[dict[str, Any]]

    @property
    def hidden(self) -> int:
        return int(self.layout["hidden"])

    @property
    def n_rows(self) -> int:
        return len(self.layout["order"])

    @property
    def n_cand(self) -> int:
        return int(self.layout["n_candidate_vectors"])

    def complete(self) -> bool:
        return len(self.rows) == self.n_rows

    def hq(self) -> np.memmap:
        return np.memmap(self.root / "hq.f32", dtype=np.float32, mode="r",
                         shape=(self.n_rows, self.hidden))

    def cands(self, which: str) -> np.memmap:
        if which not in ("start", "end"):
            raise ValueError(which)
        return bf16_memmap(self.root / f"cand_{which}.bf16", (self.n_cand, self.hidden), "r")

    def rows_in_layout_order(self) -> list[dict[str, Any]]:
        by_slot = {int(r["slot"]): r for r in self.rows}
        if len(by_slot) != len(self.rows):
            raise SystemExit(f"{self.root}: rows.jsonl repeats a slot")
        return [by_slot[s] for s in range(self.n_rows)]


def open_cache(root: Path) -> Cache:
    layout = json.loads((root / "layout.json").read_text(encoding="utf-8"))
    rows = []
    path = root / "rows.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return Cache(root=root, layout=layout, rows=rows)


def append_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """One write per batch, O_APPEND, fsynced: a crash leaves whole lines or none."""
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


# --- scoring -------------------------------------------------------------------------------


def head_tops(
    W: torch.Tensor, a: torch.Tensor, hq: torch.Tensor, cand: torch.Tensor,
    offsets: np.ndarray, *, chunk_rows: int = 256,
) -> np.ndarray:
    """Argmax row of one pointer for every span row, computed as SpanPointerHead.forward does.

    ``query = proj(h_query)`` (F.linear with ``W``), candidate ``j`` scored ``query . h_j``
    by the same einsum, the abstention ``query @ a`` placed at column ``n_candidates`` (last),
    padding at -inf; argmax over the runtime's rows (serving_scores' slice), first maximum on
    a tie as torch.argmax returns it. fp32 throughout, as the head is fp32 by construction.
    """
    n = hq.shape[0]
    out = np.empty(n, dtype=np.int64)
    W = W.float()
    a = a.float()
    for lo in range(0, n, chunk_rows):
        hi = min(n, lo + chunk_rows)
        counts = [int(offsets[i + 1] - offsets[i]) for i in range(lo, hi)]
        width = max(counts)
        k = hi - lo
        h_cand = torch.zeros((k, width, hq.shape[1]), dtype=torch.float32)
        valid = torch.zeros((k, width), dtype=torch.bool)
        for r, i in enumerate(range(lo, hi)):
            c = counts[r]
            h_cand[r, :c] = cand[int(offsets[i]) : int(offsets[i + 1])].float()
            valid[r, :c] = True
        query = torch.nn.functional.linear(hq[lo:hi].float(), W)
        scores = torch.einsum("kh,kch->kc", query, h_cand)
        scores = scores.masked_fill(~valid, float("-inf"))
        abstain = query @ a
        scores = torch.cat([scores, scores.new_full((k, 1), float("-inf"))], dim=1)
        n_c = torch.as_tensor(counts, dtype=torch.int64)
        scores = scores.scatter(1, n_c.view(k, 1), abstain.view(k, 1))
        for r in range(k):
            out[lo + r] = int(scores[r, : counts[r] + 1].argmax())
    return out


def span_correct(
    top_start: int, top_end: int, *, n_candidates: int, abstaining: bool,
    gold_start: int, gold_end: int,
) -> bool:
    """tools/real_ft_run.py _decode (1541-1579): abstained iff EITHER pointer's top is the
    noul row; correct iff abstained == expected and, for a pointing row, both pointers hit."""
    noul = n_candidates
    abstained = top_start == noul or top_end == noul
    return abstained == abstaining and (
        abstaining or (top_start == gold_start and top_end == gold_end)
    )


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs (b, c), p=0.5. 1.0 when none."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)
