"""Replay toward the base model's own answers, and the decontamination replay must pass.

Two ported pieces, both from RSI-Jev (MIT licence, Copyright (c) 2026 Shanghua Gao, at
commit 8f34a4f), and both adapted to a trainer that sees token rows rather than text:

**1. ``prior_kl`` on replay rows** -- ``rsijev/train.py:61-93``. A decision fine-tune on a
narrow corpus overwrites a prior it never sees in its loss; RSI measured the head pulled
BELOW the zero-shot control out of distribution. Their fix puts the prior in the loss: on
replay rows, ``KL(base || model)`` over the options. Direction matters and theirs is kept:
``KL(model || base)``'s gradient vanishes exactly when the model has drifted confidently away,
while ``KL(base || model)``'s gradient with respect to the logits is ``p - q``, largest when the
drift is. Here the "options" are the letter tokens at a row's answer position, the base
distribution is **the model's own at step 0, cached once** ("self-distilled replay" --
``docs/train-plan-2026-09-28.md``), and the term is added on a separate replay micro-batch
every ``every`` training micro-batches. It never enters the returned loss or the letter/span
logs, which the floor claims in ``tools/real_ft_run.py`` read.

What differs from the source: the KL is computed in **float32** (the real tower's logits are
bf16; RSI casts the base logits *to* the model's dtype, which would round the target here),
and it is over a fixed set of letter ids rather than a per-question option mask, because a
replay row arrives without its label and so without its own alphabet.

**2. 8-gram replay decontamination** -- ``scripts/build_specialist_replay_corpus.py:44-86,
158-172`` (``_grams``, ``Containment``, the refuse-on-hit re-check). A replay row is
contaminated when it contains at least ``threshold`` of some target row's word 8-grams. What
differs: the n-gram hash is blake2b rather than Python's ``hash()``, which is salted per
process and so gives a different index every run (RSI's ``_grams`` uses it; within one
process it is consistent, across an attestation file and a later check it is not). And the
result is an **attestation** -- a file naming the replay shard set, every target set's row
digest, the counts, and ``clean`` -- which ``tools/real_ft_run.py`` requires before it will
read a replay shard set. Refuse-on-hit, as RSI's final re-check does: a contaminated replay
set is rebuilt, not trained on with the hits quietly dropped.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from .artifacts import SLOT_SPAN, Batch

__all__ = [
    "ATTESTATION_VERSION",
    "DecontamReport",
    "PriorCache",
    "PriorKLReplay",
    "ReplayRefusal",
    "check_attestation",
    "decontaminate",
    "prior_kl",
    "prompt_content",
    "target_digest",
    "word_ngrams",
    "write_text_atomic",
]

#: RSI's n and threshold.
DEFAULT_N: Final[int] = 8
DEFAULT_THRESHOLD: Final[float] = 0.5
ATTESTATION_VERSION: Final[int] = 1
#: Bounds on what one decontamination compares, so a mis-pointed target cannot run for ever.
MAX_ROWS_PER_SIDE: Final[int] = 2_000_000
#: How many hits the attestation spells out; the count is always complete.
MAX_HIT_EXAMPLES: Final[int] = 50


class ReplayRefusal(ValueError):
    """Replay data, its cache or its attestation does not describe this run."""


# --- decontamination ---------------------------------------------------------------------


def prompt_content(prompt: str) -> str:
    """The row-specific text of one rendered slot prompt: question, context, option values.

    RSI compares ``state + instructions + criteria`` (``case_text``), never the prompt
    template; this is the same cut through ``qd_data.render``'s layout. Markers, the task
    and route lines and the ``noul`` line are template, identical across rows, and would
    make every row of a family overlap every other.
    """
    from qd_data.render import M_CTX_BEGIN, M_CTX_END, M_OPT_BEGIN, M_OPT_END, M_QUESTION
    from qd_data.schema import NOUL, NOUL_LETTER

    parts: list[str] = []
    q = prompt.find(M_QUESTION)
    if q >= 0:
        start = q + len(M_QUESTION)
        parts.append(prompt[start : prompt.find("\n", start)])
    c0 = prompt.find(M_CTX_BEGIN + "\n")
    c1 = prompt.rfind("\n" + M_CTX_END)
    if c0 >= 0 and c1 > c0:
        parts.append(prompt[c0 + len(M_CTX_BEGIN) + 1 : c1])
    o0 = prompt.rfind(M_OPT_BEGIN + "\n")
    o1 = prompt.rfind(M_OPT_END)
    if o0 >= 0 and o1 > o0:
        for line in prompt[o0 + len(M_OPT_BEGIN) + 1 : o1].splitlines():
            if line == f"{NOUL_LETTER}. {NOUL}":
                continue
            parts.append(line.split(". ", 1)[1] if ". " in line else line)
    if not parts:
        raise ReplayRefusal(
            "no question, context or option block found: this is not a qd_data.render prompt"
        )
    return "\n".join(parts)


def word_ngrams(text: str, n: int = DEFAULT_N) -> frozenset[bytes]:
    """Lower-cased ``\\w+`` word n-grams, as RSI's ``_grams``, hashed with blake2b-64."""
    words = re.findall(r"\w+", text.lower())
    return frozenset(
        hashlib.blake2b(" ".join(words[i : i + n]).encode("utf-8"), digest_size=8).digest()
        for i in range(len(words) - n + 1)
    )


def target_digest(texts: Mapping[str, str]) -> str:
    """One digest over a target set's ``(id, content)`` pairs, order-independent."""
    h = hashlib.sha256()
    for key in sorted(texts):
        h.update(key.encode("utf-8") + b"\x1f" + texts[key].encode("utf-8") + b"\x1e")
    return h.hexdigest()


@dataclass(frozen=True)
class DecontamReport:
    """What one decontamination compared and found. Counts are complete; examples capped."""

    n: int
    threshold: float
    replay_rows_total: int
    replay_rows_checked: int
    #: Replay rows with fewer than ``n`` words: they have no n-gram, so this check cannot
    #: say anything about them. Counted separately rather than as clean.
    replay_rows_too_short: int
    targets: dict[str, dict[str, Any]]
    hits: dict[str, int]
    hit_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.replay_rows_checked > 0 and not any(self.hits.values())

    def to_json(self) -> dict[str, Any]:
        return {
            "version": ATTESTATION_VERSION,
            "n": self.n,
            "threshold": self.threshold,
            "replay_rows_total": self.replay_rows_total,
            "replay_rows_checked": self.replay_rows_checked,
            "replay_rows_too_short": self.replay_rows_too_short,
            "targets": self.targets,
            "hits": self.hits,
            "hit_examples": self.hit_examples,
            "clean": self.clean,
        }


def decontaminate(
    replay: Mapping[str, str],
    targets: Mapping[str, Mapping[str, str]],
    *,
    n: int = DEFAULT_N,
    threshold: float = DEFAULT_THRESHOLD,
) -> DecontamReport:
    """Compare every replay row against every row of every target set.

    A replay row hits target set ``T`` when, for some row ``t`` of ``T``, it contains at
    least ``threshold`` of ``t``'s n-grams (RSI's ``Containment.hit``). ``hits[T]`` counts
    replay rows, not pairs. Target rows with no n-gram are counted and cannot be hit.
    """
    if n < 1 or not (0.0 < threshold <= 1.0):
        raise ReplayRefusal(f"n={n} threshold={threshold} is not a containment test")
    if not targets:
        raise ReplayRefusal("no target set: decontamination against nothing is not a check")
    if len(replay) > MAX_ROWS_PER_SIDE:
        raise ReplayRefusal(f"{len(replay)} replay rows exceeds {MAX_ROWS_PER_SIDE}")
    report_targets: dict[str, dict[str, Any]] = {}
    indexes: dict[str, tuple[dict[bytes, list[str]], dict[str, int]]] = {}
    for name, rows in targets.items():
        if len(rows) > MAX_ROWS_PER_SIDE:
            raise ReplayRefusal(f"target {name!r}: {len(rows)} rows exceeds {MAX_ROWS_PER_SIDE}")
        index: dict[bytes, list[str]] = defaultdict(list)
        size: dict[str, int] = {}
        for tid, text in rows.items():
            grams = word_ngrams(text, n)
            if not grams:
                continue
            size[tid] = len(grams)
            for g in grams:
                index[g].append(tid)
        indexes[name] = (index, size)
        report_targets[name] = {
            "rows": len(rows),
            "rows_indexed": len(size),
            "rows_too_short": len(rows) - len(size),
            "digest": target_digest(rows),
        }
    hits = {name: 0 for name in targets}
    examples: list[dict[str, Any]] = []
    checked = too_short = 0
    for rid, text in sorted(replay.items()):
        grams = word_ngrams(text, n)
        if not grams:
            too_short += 1
            continue
        checked += 1
        for name, (index, size) in indexes.items():
            counts = Counter(t for g in grams for t in index.get(g, ()))
            best = max(((t, c / size[t]) for t, c in counts.items()), key=lambda z: z[1],
                       default=None)
            if best is not None and best[1] >= threshold:
                hits[name] += 1
                if len(examples) < MAX_HIT_EXAMPLES:
                    examples.append(
                        {"replay_row": rid, "target": name, "target_row": best[0],
                         "containment": round(best[1], 4)}
                    )
    return DecontamReport(
        n=n, threshold=threshold, replay_rows_total=len(replay), replay_rows_checked=checked,
        replay_rows_too_short=too_short, targets=report_targets, hits=hits,
        hit_examples=examples,
    )


def check_attestation(
    raw: Mapping[str, Any],
    *,
    replay_shard_hash: str,
    verify_targets: Mapping[str, Mapping[str, str]],
    required_targets: Sequence[str],
) -> None:
    """Refuse unless ``raw`` attests THIS replay shard set clean against the target sets.

    ``required_targets`` must all have been checked. ``verify_targets`` are the ones the
    caller can recompute, and they are compared by row digest, so an attestation made against
    a different split, corpus revision or subset is refused. A caller that may not read a
    target (a training process and the ``heldout`` split -- rule 3) names it in
    ``required_targets`` only, and pins the corpus identity instead.
    """
    if raw.get("version") != ATTESTATION_VERSION:
        raise ReplayRefusal(f"attestation version {raw.get('version')!r}, expected 1")
    if raw.get("replay_shard_hash") != replay_shard_hash:
        raise ReplayRefusal(
            f"the attestation is for replay shard set {raw.get('replay_shard_hash')!r}, not "
            f"{replay_shard_hash!r}"
        )
    attested = raw.get("targets") or {}
    missing = sorted(set(required_targets) - set(attested))
    if missing:
        raise ReplayRefusal(f"the attestation never checked target set(s) {missing}")
    for name, rows in verify_targets.items():
        if name not in attested or attested[name].get("digest") != target_digest(rows):
            raise ReplayRefusal(
                f"target set {name!r} is not the one the attestation checked: its row digest "
                "differs (a different corpus revision, split or row count)"
            )
    if not raw.get("clean") or any(int(v) for v in (raw.get("hits") or {}).values()):
        raise ReplayRefusal(
            f"the attestation records contamination {raw.get('hits')}: rebuild the replay set "
            "without those rows rather than training on it"
        )
    if int(raw.get("replay_rows_checked", 0)) < 1:
        raise ReplayRefusal("the attestation checked no replay row")


def write_text_atomic(path: Path, body: str) -> None:
    """``body`` to ``path`` through ``run_control``'s atomic write; refuse to overwrite."""
    from .run_control import _atomic_write_bytes

    if path.exists():
        raise ReplayRefusal(f"{path} already exists; refusing to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, body.encode("utf-8"))


# --- prior_kl -----------------------------------------------------------------------------


def prior_kl(logits: Any, base_logits: Any, direction: str = "base_to_model") -> Any:
    """``KL(base || model)`` (default) or ``KL(model || base)``, mean over rows, in float32.

    Ported from RSI-Jev ``rsijev/train.py:61-93`` including its NaN guard: masked options
    are ``-inf`` in both, ``(-inf) - (-inf)`` is NaN, and ``torch.where`` still
    backpropagates from the branch it did not take, so the operands are neutralised before
    the subtraction. Differs in casting both sides UP to float32 rather than the base down.
    """
    import torch
    import torch.nn.functional as F

    if direction not in ("base_to_model", "model_to_base"):
        raise ReplayRefusal(f"unknown prior_kl direction {direction!r}")
    lp = F.log_softmax(logits.to(torch.float32), dim=-1)
    lq = F.log_softmax(base_logits.to(device=logits.device, dtype=torch.float32), dim=-1)
    finite = torch.isfinite(lp) & torch.isfinite(lq)
    zero = torch.zeros_like(lp)
    lp_s = torch.where(finite, lp, zero)
    lq_s = torch.where(finite, lq, zero)
    if direction == "model_to_base":
        w = torch.where(finite, lp_s.exp(), zero)
        return (w * (lp_s - lq_s)).sum(-1).mean()
    q = torch.where(finite, lq_s.exp(), zero)
    return (q * (lq_s - lp_s)).sum(-1).mean()


def _answer_rows(batch: Batch) -> np.ndarray:
    """Rows of a replay batch the KL is taken over: every letter-answered row (not span)."""
    if batch.slot_kind is None or batch.target_index is None:
        raise ReplayRefusal("a replay batch must be an FT batch: it names no answer position")
    return np.flatnonzero(np.asarray(batch.slot_kind) != SLOT_SPAN)


def _letter_logits(step: Any, batch: Batch, rows: np.ndarray, letter_ids: Sequence[int]) -> Any:
    """``[len(rows), K]`` letter logits at each row's answer position, from ``step``."""
    import torch

    if batch.target_index is None:
        raise ReplayRefusal('a replay batch must be an FT batch: it names no answer position')
    hidden = step.hidden(batch)
    r = torch.as_tensor(rows.astype(np.int64), device=hidden.device)
    t = torch.as_tensor(np.asarray(batch.target_index)[rows].astype(np.int64), device=hidden.device)
    logits = step.lm_head(hidden[r, t])
    ids = torch.as_tensor(list(letter_ids), dtype=torch.int64, device=logits.device)
    return logits.index_select(-1, ids)


def _modules(step: Any) -> list[Any]:
    from torch import nn

    found = [v for v in vars(step).values() if isinstance(v, nn.Module)]
    tower = getattr(step, "tower", None)
    model = getattr(tower, "model", None)
    if isinstance(model, nn.Module):
        found.append(model)
    return found


@dataclass(frozen=True)
class PriorCache:
    """The base model's letter logits on every replay row, and what they were taken from.

    ``key`` names the backbone, the replay plan (by a digest over its tokens) and the letter
    ids; a cache is only ever used against an identical key. ``logits[i]`` is batch
    ``batch_index[i]`` row ``row[i]``.
    """

    key: dict[str, Any]
    batch_index: np.ndarray
    row: np.ndarray
    logits: np.ndarray

    @staticmethod
    def plan_digest(batches: Sequence[Batch]) -> str:
        h = hashlib.sha256()
        for b in batches:
            h.update(int(b.index).to_bytes(8, "little"))
            h.update(np.ascontiguousarray(b.tokens).tobytes())
            h.update(np.ascontiguousarray(b.lengths).astype(np.int64).tobytes())
        return h.hexdigest()

    @classmethod
    def build(
        cls, step: Any, batches: Sequence[Batch], *, letter_ids: Sequence[int],
        key: Mapping[str, Any],
    ) -> PriorCache:
        """Run ``step`` (which must still be the base) over every replay row, no grad.

        Every torch module the step holds is put in eval mode for the pass and restored
        after, so a dropout layer cannot make "the base's answer" a draw.
        """
        import torch

        if not batches:
            raise ReplayRefusal("no replay batch to cache")
        modules = _modules(step)
        was_training = [m.training for m in modules]
        idx: list[int] = []
        rows_out: list[int] = []
        out: list[np.ndarray] = []
        try:
            for m in modules:
                m.eval()
            with torch.no_grad():
                for b in batches:
                    rows = _answer_rows(b)
                    if rows.size == 0:
                        continue
                    lg = _letter_logits(step, b, rows, letter_ids).to(torch.float32)
                    out.append(lg.cpu().numpy())
                    idx.extend([int(b.index)] * int(rows.size))
                    rows_out.extend(int(r) for r in rows)
        finally:
            for m, flag in zip(modules, was_training, strict=True):
                m.train(flag)
        if not out:
            raise ReplayRefusal("the replay plan has no letter-answered row to anchor")
        logits = np.concatenate(out).astype(np.float32)
        if not np.isfinite(logits).all():
            raise ReplayRefusal("the base produced non-finite letter logits on a replay row")
        full_key = {**dict(key), "letter_ids": [int(i) for i in letter_ids],
                    "plan_digest": cls.plan_digest(batches)}
        return cls(full_key, np.asarray(idx, np.int64), np.asarray(rows_out, np.int64), logits)

    def save(self, path: Path) -> None:
        from .run_control import _atomic_write_bytes

        if path.exists():
            raise ReplayRefusal(f"{path} already exists; refusing to overwrite a prior cache")
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez(buf, key=np.asarray(json.dumps(self.key, sort_keys=True)),
                 batch_index=self.batch_index, row=self.row, logits=self.logits)
        _atomic_write_bytes(path, buf.getvalue())

    @classmethod
    def load(cls, path: Path, *, expect_key: Mapping[str, Any]) -> PriorCache:
        with np.load(path, allow_pickle=False) as z:
            key = json.loads(str(z["key"]))
            cache = cls(key, z["batch_index"], z["row"], z["logits"])
        if key != json.loads(json.dumps(dict(expect_key), sort_keys=True)):
            differ = sorted(
                k for k in set(key) | set(expect_key) if key.get(k) != expect_key.get(k)
            )
            raise ReplayRefusal(
                f"{path} was cached for a different base or replay plan (differs in {differ}); "
                "it would distil toward a model this run does not start from"
            )
        return cache

    def for_batch(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        sel = self.batch_index == int(batch_index)
        return self.row[sel], self.logits[sel]


class PriorKLReplay:
    """A training step plus ``weight * prior_kl`` on a replay micro-batch every ``every``.

    Wraps an inner [`qd_train.trainer.SpanScoringStep`] and delegates the whole protocol to
    it; the replay term is backpropagated into the same accumulated gradient the next
    ``apply`` steps on. ``train_ft`` is not changed and cannot tell this from the inner step.

    Resume: the micro-batch counter, and so the replay cursor, is in ``state()["replay"]``
    together with a digest of the cache key, and ``load_state`` refuses a checkpoint taken
    under a different replay configuration.
    """

    def __init__(
        self, inner: Any, *, batches: Sequence[Batch], cache: PriorCache, weight: float,
        every: int, direction: str = "base_to_model",
    ) -> None:
        if not (math.isfinite(weight) and weight > 0.0):
            raise ReplayRefusal(f"replay weight must be finite and positive, got {weight}")
        if every < 1:
            raise ReplayRefusal(f"replay every must be at least 1, got {every}")
        if cache.key.get("plan_digest") != PriorCache.plan_digest(batches):
            raise ReplayRefusal("the prior cache was built over a different replay plan")
        self.inner = inner
        self.batches = [b for b in batches if _answer_rows(b).size]
        if not self.batches:
            raise ReplayRefusal("no replay batch carries a letter-answered row")
        self.cache = cache
        self.weight = float(weight)
        self.every = int(every)
        self.direction = direction
        self.letter_ids = [int(i) for i in cache.key["letter_ids"]]
        self.micro_batches = 0
        self.replayed = 0
        self.replay_log: list[float] = []
        self._key_digest = hashlib.sha256(
            json.dumps(cache.key, sort_keys=True).encode()
        ).hexdigest()

    def _maybe_replay(self) -> None:
        import torch

        self.micro_batches += 1
        if self.micro_batches % self.every:
            return
        batch = self.batches[self.replayed % len(self.batches)]
        rows, base = self.cache.for_batch(batch.index)
        mine = _answer_rows(batch)
        if not np.array_equal(rows, mine):
            raise ReplayRefusal(f"replay batch {batch.index}: cached rows {rows} != {mine}")
        logits = _letter_logits(self.inner, batch, mine, self.letter_ids)
        loss = self.weight * prior_kl(
            logits, torch.as_tensor(base), direction=self.direction
        )
        loss.backward()
        self.replay_log.append(float(loss.detach()) / self.weight)
        self.replayed += 1

    # -- the delegated protocol --------------------------------------------------------------

    def accumulate(self, batch: Batch, supervision: Any) -> float:
        value = self.inner.accumulate(batch, supervision)
        self._maybe_replay()
        return value

    def accumulate_span(self, batch: Batch, supervision: Any) -> float:
        value = self.inner.accumulate_span(batch, supervision)
        self._maybe_replay()
        return value

    def apply(self, *, lr: float) -> None:
        self.inner.apply(lr=lr)

    def state(self) -> dict[str, Any]:
        return {
            **self.inner.state(),
            "replay": {
                "micro_batches": self.micro_batches,
                "replayed": self.replayed,
                "key_digest": self._key_digest,
                "weight": self.weight,
                "every": self.every,
                "direction": self.direction,
            },
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        replay = state.get("replay")
        if not isinstance(replay, Mapping):
            raise ReplayRefusal(
                "this checkpoint was taken without replay; resuming it with replay on would "
                "train a run that is neither the one checkpointed nor a fresh one"
            )
        mine = {"key_digest": self._key_digest, "weight": self.weight, "every": self.every,
                "direction": self.direction}
        theirs = {k: replay.get(k) for k in mine}
        if theirs != mine:
            raise ReplayRefusal(f"replay configuration {theirs} in the checkpoint, {mine} here")
        self.inner.load_state({k: v for k, v in state.items() if k != "replay"})
        self.micro_batches = int(replay["micro_batches"])
        self.replayed = int(replay["replayed"])


def replay_texts_from_prompts(prompts: Iterable[tuple[str, str]]) -> dict[str, str]:
    """``{id: prompt_content(prompt)}`` for ``(id, prompt)`` pairs; duplicate ids refused."""
    out: dict[str, str] = {}
    for key, prompt in prompts:
        if key in out:
            raise ReplayRefusal(f"duplicate row id {key!r}")
        out[key] = prompt_content(prompt)
    return out
