"""Shared plumbing for the Mac GPU-lane benchmarks (``tools/mac_torch_attrib.py``,
``tools/mac_mlx_bench.py``): where the code comes from, and the protocol a row carries.

Only the Protocol is shared. Each tool constructs its own ``RunRecorder`` with
``entry_point=Path(__file__)``, so a row's ``code_that_ran`` is the invoking tool's closure --
this module included, as a sibling import -- rather than this helper's.

The protocol is real, not invented: ``data_snapshot_hash`` and ``tokenizer_hash`` are read off
the header of the shard set the prompts were taken from (``--header``), ``backbone_commit`` is
the snapshot revision and vocab (`real_ft_run._backbone_commit`), and ``recipe_hash`` is
computed by ``real_ft_run._protocol`` itself -- no new spelling of the hash.

``QD_REPO`` names the checkout whose ``python/`` is imported and whose git state is recorded as
``code_commit`` -- a frozen worktree during the GPU lane, so the rows carry a clean commit while
other lanes edit the main tree. It defaults to this file's own repository.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(os.environ.get("QD_REPO", str(Path(__file__).resolve().parents[1]))).resolve()
for _p in (REPO / "python", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qd_train.ledger import Environment, Protocol  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

QUICK_REASON = (
    "a throughput/parity benchmark: one seed, no training schedule, a fixed item sample. "
    "Rule 8: quick, excluded from every decision"
)


def header_hashes(header: Path) -> tuple[str, str]:
    """``(data_snapshot_hash, tokenizer_hash)`` from a shard set's ``header.json``."""
    raw = json.loads(header.read_text(encoding="utf-8"))
    data, tok = raw.get("data_snapshot_hash"), raw.get("tokenizer_hash")
    if not (isinstance(data, str) and data and isinstance(tok, str) and tok):
        raise SystemExit(f"{header} carries no data_snapshot_hash/tokenizer_hash; refusing")
    return data, tok


def protocol(*, header: Path, recipe: dict[str, Any], seed: int = 0) -> Protocol:
    """The protocol of a benchmark row, built by ``real_ft_run._protocol`` itself.

    Delegated rather than re-spelled: seven tools already hash a recipe and two spellings
    disagree (``test_tool_call_sites``, GAP-SIX-SPELLINGS-OF-ONE-RECIPE-HASH), so an eighth
    implementation would be one more place for the next tidy-up to rename every hash. The
    header is read as JSON rather than through ``ShardReader``: only its two hashes are
    needed, and the reader's code-fingerprint check is about training on the set, which a
    benchmark over its prompts does not do. ``recipe`` must carry ``backbone_snapshot`` and
    ``backbone_vocab``, which ``_backbone_commit`` turns into ``<snapshot>:vocab<N>``.
    """
    data, tok = header_hashes(header)
    ft = importlib.import_module("real_ft_run")
    stand_in = SimpleNamespace(
        header=SimpleNamespace(data_snapshot_hash=data, tokenizer_hash=tok)
    )
    built: Protocol = ft._protocol(reader=stand_in, seed=seed, recipe=recipe)
    return built


def environment() -> Environment:
    """``mps``: torch-MPS and MLX both run on this Mac's Metal GPU, which bills nothing."""
    return Environment.detect(device="mps")


def tool_digest(tool: Path) -> Ran:
    """The invoking tool's bytes, pinned as a metric (it may live outside ``REPO``)."""
    return Ran(
        passed=True,
        value=hashlib.sha256(tool.read_bytes()).hexdigest(),
        detail=f"{tool.name}; qd_train imported from {REPO / 'python'}",
    )


def num(value: float, detail: str) -> Ran:
    return Ran(passed=True, value=round(float(value), 4), detail=detail)
