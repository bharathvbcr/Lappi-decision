"""``tools/ckpt_average.py`` refuses to average span heads unless a shared init is proven.

GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06: v5's five-seed average collapsed on span
(0.773 against 0.907-0.915) because independently initialised heads are near-orthogonal, so their
mean is 1/sqrt(N) of a head. ``--span-head-seed`` was the safe path, but the DEFAULT still averaged
heads with no check. Now the default refuses; the operator picks ``--span-head-seed N`` or proves
the shared init with ``--span-head-shared-init`` (every input's ft row records the same shared
``span_head_init``, read from the ledger and never from the caller).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
import torch  # noqa: E402
from test_ckpt_average_span_head_seed import _three  # noqa: E402

SHARED = {"sha256": "e" * 64, "content_digest": "d" * 64, "shared": True}
OWN = {"sha256": "f" * 64, "content_digest": "c" * 64}
OTHER_SHARED = {"sha256": "a" * 64, "content_digest": "b" * 64, "shared": True}


def shared_init_args(
    tmp_path: Path, paths: list[Path], inits=None, ids: list[str] | None = None
) -> list[str]:
    """The arguments that prove the shared init of ``paths``: a ledger with one completed ft row
    per input (``inits[i]`` is its ``recipe.span_head_init``, default SHARED; ``ids`` names the
    rows, default ``proof-000i``), and the flags."""
    from qd_train.run_control import Checkpoint

    inits = inits or [SHARED] * len(paths)
    ids = list(ids) if ids is not None else [f"proof-{i:04d}" for i in range(len(paths))]
    lines = []
    for path, init, row_id in zip(paths, inits, ids, strict=True):
        recipe = {} if init is None else {"span_head_init": init}
        lines.append(json.dumps({
            "row_id": row_id, "run_kind": "ft", "status": "completed",
            "protocol": {"seed": Checkpoint.read_body(path)["seed"]}, "recipe": recipe,
        }))
    ledger = tmp_path / "proof-ledger.jsonl"
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["--span-head-shared-init", "--ft-ledger", str(ledger), "--ft-row-ids", *ids]


def _main(tmp_path: Path, paths: list[Path], *extra: str) -> int:
    out = tmp_path / "avg" / "a.safetensors"
    return ckpt_average.main([*map(str, paths), "--out", str(out), *extra])


def test_the_default_refuses_to_average_span_heads_and_writes_nothing(tmp_path):
    paths = _three(tmp_path)
    with pytest.raises(SystemExit, match="no proof they grew from one shared init"):
        _main(tmp_path, paths)
    assert not (tmp_path / "avg").exists()


def test_the_refusal_names_both_ways_out(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _main(tmp_path, _three(tmp_path), "--ft-row-ids", "a", "b", "c")
    message = str(exc.value)
    assert "--span-head-seed" in message and "--span-head-shared-init" in message


def test_a_proven_shared_init_averages_the_heads_and_records_the_proof(tmp_path):
    from safetensors.torch import load_file

    paths = _three(tmp_path)
    assert _main(tmp_path, paths, *shared_init_args(tmp_path, paths)) == 0
    out = tmp_path / "avg" / "a.safetensors"
    assert torch.equal(load_file(str(out))["span_head.b"], torch.tensor([37.0, 74.0]))
    block = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))[
        "span_head_source"
    ]
    assert block["mode"] == "averaged"
    assert block["init"]["content_digest"] == "d" * 64
    assert block["init"]["proven_across"] == ["proof-0000", "proof-0001", "proof-0002"]
    assert ckpt_average.read_manifest(out).span_head_source["mode"] == "averaged"


@pytest.mark.parametrize(
    ("inits", "match"),
    [
        ([SHARED, OWN, SHARED], "not the shared one"),
        ([SHARED, None, SHARED], "cannot be verified"),
        ([SHARED, OTHER_SHARED, SHARED], "not one init"),
    ],
)
def test_a_proof_the_ledger_does_not_support_is_refused(tmp_path, inits, match):
    paths = _three(tmp_path)
    with pytest.raises(SystemExit, match=match):
        _main(tmp_path, paths, *shared_init_args(tmp_path, paths, inits))
    assert not (tmp_path / "avg").exists()


def test_the_proof_needs_the_ledger_and_one_row_per_input(tmp_path):
    with pytest.raises(SystemExit, match="needs --ft-ledger and one --ft-row-ids per input"):
        _main(tmp_path, _three(tmp_path), "--span-head-shared-init")


def test_the_seed_flag_and_the_proof_flag_exclude_each_other(tmp_path):
    paths = _three(tmp_path)
    with pytest.raises(SystemExit, match="choose one"):
        _main(tmp_path, paths, "--span-head-seed", "1", *shared_init_args(tmp_path, paths))
