"""Cache the linear control's verdict, so a GPU never waits on a CPU fit.

## Why this exists

`paired_margin_vs_linear` is the gate the program rests on: *"The 2B ships only if it beats
this by a paired margin on three seeds on the natural held-out set."* Scoring it requires
fitting the control, and the control is single-threaded CPU work while the GPU does nothing.

On the rung-0 corpus that is 38 seconds and not worth engineering around. On the
commitpackft corpus the plan names -- 39,946 training documents -- the dense path needs
20.9 GB (the box has 406 GB free, so this is affordable) and the fit projects to roughly
3.5 hours. Three and a half hours of a rented GPU sitting at 0% is the same waste that cost
an arm on 2026-09-21, only longer and on purpose.

The fit does not depend on the model. It depends on the documents, the labels, the split and
the baseline's own hyper-parameters -- none of which change across the 24 seeds of a sweep,
or across sweeps on the same corpus. So it is computed once, by a job that owns no GPU, and
every run afterwards reads the answer.

## What the key covers

Everything that could change the answer, including **the implementation**: the digest folds
in the source of `qd_train.baseline`. A cache that survived an edit to the optimizer would
serve a verdict produced by code that no longer exists, and the run would record a margin
against an opponent it never faced. That is the same class of defect as
`GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN`, and the fix is the same one: hash the bytes
that ran.

A miss is not an error. The caller decides whether to fit, or to report `NotRun` -- and
`NotRun` with a reason is always preferable to a fit that silently takes hours.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

__all__ = ["CachedControl", "control_key", "load_control", "store_control"]

#: Bumped when the cache's own format changes, independently of the fit's semantics --
#: those are covered by the source digest.
CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class CachedControl:
    """A stored verdict and the provenance needed to trust it."""

    correct: np.ndarray
    fitted_s: float
    fitted_at: str
    n_train: int
    l2: float
    iterations: int
    final_grad_norm: float


def _digest_source() -> str:
    """sha256 of the baseline implementation, so an edit invalidates every entry."""
    return hashlib.sha256(
        (Path(__file__).parent / "baseline.py").read_bytes()
    ).hexdigest()


def control_key(
    *,
    train_docs: list[str],
    train_labels: list[str],
    val_docs: list[str],
    seed: int,
    max_iter: int,
    hasher_params: tuple[int, int, int],
    l2_grid: tuple[float, ...],
    tol: float,
    lr: float,
) -> str:
    """A digest over every input the fitted verdict depends on.

    The documents are folded in with an explicit length prefix rather than concatenated:
    without it, two different corpora whose texts happen to join into the same byte string
    would collide, and the run would score against the wrong opponent.
    """
    h = hashlib.sha256()
    h.update(f"v{CACHE_FORMAT_VERSION}|{_digest_source()}".encode())
    h.update(
        json.dumps(
            {
                "seed": seed,
                "max_iter": max_iter,
                "hasher": list(hasher_params),
                "l2_grid": list(l2_grid),
                "tol": tol,
                "lr": lr,
                "n_train": len(train_docs),
                "n_val": len(val_docs),
            },
            sort_keys=True,
        ).encode()
    )
    for part in (train_docs, train_labels, val_docs):
        h.update(f"|{len(part)}|".encode())
        for item in part:
            raw = item.encode("utf-8", errors="surrogatepass")
            h.update(len(raw).to_bytes(8, "big"))
            h.update(raw)
    return h.hexdigest()


def _path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"linear-control-{key[:32]}.json"


def load_control(cache_dir: Path, key: str, *, expected_n: int) -> CachedControl | None:
    """The stored verdict, or `None` on any miss.

    A stored row whose length disagrees with the validation set is treated as a miss and
    not as an error: it cannot be paired with anything, and a silently mispaired verdict
    would score the model's predictions against another split's answers.
    """
    path = _path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    correct = payload.get("correct")
    if not isinstance(correct, list) or len(correct) != expected_n:
        return None
    return CachedControl(
        correct=np.asarray(correct, dtype=bool),
        fitted_s=float(payload["fitted_s"]),
        fitted_at=str(payload["fitted_at"]),
        n_train=int(payload["n_train"]),
        l2=float(payload["l2"]),
        iterations=int(payload["iterations"]),
        final_grad_norm=float(payload["final_grad_norm"]),
    )


def store_control(
    cache_dir: Path,
    key: str,
    correct: np.ndarray,
    *,
    fitted_s: float,
    n_train: int,
    l2: float,
    iterations: int,
    final_grad_norm: float,
) -> Path:
    """Write the verdict. Atomic, so a crash mid-write cannot leave a truncated entry
    that a later run would read as a shorter validation set."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _path(cache_dir, key)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "key": key,
                "format_version": CACHE_FORMAT_VERSION,
                "correct": [bool(x) for x in correct],
                "fitted_s": fitted_s,
                "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "n_train": n_train,
                "n_val": len(correct),
                "l2": l2,
                "iterations": iterations,
                "final_grad_norm": final_grad_norm,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path
