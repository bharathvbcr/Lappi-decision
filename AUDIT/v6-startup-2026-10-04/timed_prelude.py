"""``tools/real_ft_run.py``'s ``main``, unprofiled, with wall-clock seconds per prelude phase.

Analysis only. cProfile inflates call-heavy Python (and swallows a ``SystemExit`` silently), so
this wraps a fixed list of phase functions in a timer instead -- one ``perf_counter`` pair per
call, nothing per inner call -- runs ``main`` with the given argv, and prints each phase's
seconds to stderr as it ends and any ``SystemExit`` message. Nothing in ``qd_data``'s source is
touched (its fingerprint is of the source files); module attributes are rebound for this process
only, before ``main`` runs.

Usage: python timed_prelude.py <real_ft_run.py argv...>
"""

from __future__ import annotations

import contextlib
import functools
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "python"))

import real_ft_run as rft  # noqa: E402
import real_tokenizer_pipeline as pipeline  # noqa: E402

import qd_data.decisions as decisions  # noqa: E402
import qd_data.dedupe as dedupe_module  # noqa: E402
import qd_data.defect_class as defect_class  # noqa: E402
import qd_data.mixture as mixture  # noqa: E402
import qd_data.split as split_module  # noqa: E402
import qd_train.exclusions as exclusions  # noqa: E402

T0 = time.perf_counter()


def _say(text: str) -> None:
    print(f"[timed {time.perf_counter() - T0:8.1f} s] {text}", file=sys.stderr, flush=True)


def _timed(module: Any, name: str) -> None:
    inner: Callable[..., Any] = getattr(module, name)

    @functools.wraps(inner)
    def wrapper(*a: Any, **k: Any) -> Any:
        began = time.perf_counter()
        try:
            return inner(*a, **k)
        finally:
            _say(f"{module.__name__}.{name}: {time.perf_counter() - began:.1f} s")

    setattr(module, name, wrapper)


def _timed_block(module: Any, name: str) -> None:
    inner = getattr(module, name)

    @contextlib.contextmanager
    def wrapper(*a: Any, **k: Any) -> Iterator[None]:
        began = time.perf_counter()
        with inner(*a, **k):
            _say(f"{module.__name__}.{name} entered (shingled and signed): "
                 f"{time.perf_counter() - began:.1f} s")
            inside = time.perf_counter()
            yield
            _say(f"{module.__name__}.{name} body (dedupe + split, LSH inside): "
                 f"{time.perf_counter() - inside:.1f} s")
        _say(f"{module.__name__}.{name} whole block: {time.perf_counter() - began:.1f} s")

    setattr(module, name, wrapper)


for mod, fn in (
    (rft, "check_defect_source"), (rft, "corpus_facts"), (rft, "ft_split_rows"),
    (rft, "ft_split_report"), (defect_class, "load_defect_rows"), (pipeline, "general_rows"),
    (decisions, "load_decision_pool"), (mixture, "build_mixture"),
    (mixture, "drop_contradictory_prompts"), (exclusions, "drop_before_dedupe"),
    (pipeline, "_run_prep"), (dedupe_module, "dedupe"), (split_module, "split"),
    (pipeline, "exclusions_then_contrast"), (rft, "open_val_set"), (rft, "prepare_ood"),
    (rft, "_checkpoint_step"),
):
    _timed(mod, fn)
_timed_block(pipeline, "native_minhash")

if __name__ == "__main__":
    _say("main starts (imports done)")
    try:
        code = rft.main(sys.argv[1:])
    except SystemExit as exc:
        _say(f"SystemExit: {exc}")
        raise
    _say(f"main returned {code}")
    sys.exit(code)
