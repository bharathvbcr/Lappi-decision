"""The one committed source of the v6 mutant build's ``--max-per-function`` value.

``qd-mutate generate`` has no default for the flag (Fable Q3, 2026-10-18), so every caller
must pass it. Before this module K = 2 was a literal in the one script that ran the build
and a sentence in ``data/decisions/v6-allocation.json``'s notes; nothing made a second caller
agree with it. Callers now take the flag from ``data/decisions/v6-mutate.json`` through
:func:`max_per_function_args`, which refuses a missing, non-integer or non-positive value
rather than supplying a default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

V6_MUTATE_CONFIG: Final[Path] = (
    Path(__file__).resolve().parents[2] / "data" / "decisions" / "v6-mutate.json"
)
SCHEMA: Final[str] = "qd-mutate-build-config/v1"


def load_max_per_function(path: Path = V6_MUTATE_CONFIG) -> int:
    """K from the config at ``path``; ``ValueError`` if absent or not a positive integer."""
    try:
        config = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: the v6 mutate config cannot be read: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} document")
    k = config.get("max_per_function")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"{path}: max_per_function must be an integer >= 1, got {k!r}")
    return k


def generate_command(
    binary: Path,
    *,
    pool: Path,
    out: Path,
    manifest: Path,
    seed: int,
    limit: int,
    config: Path = V6_MUTATE_CONFIG,
) -> list[str]:
    """The full ``qd-mutate generate`` command line for a v6 build, K read from ``config``.

    Raises ``ValueError`` before anything runs if the config has no valid K.
    """
    return [
        str(binary), "generate",
        "--pool", str(pool),
        "--out", str(out),
        "--manifest", str(manifest),
        "--seed", str(seed),
        "--limit", str(limit),
        "--max-per-function", str(load_max_per_function(config)),
    ]
