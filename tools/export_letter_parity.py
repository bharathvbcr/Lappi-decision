"""Oracle for a ``qd-export`` release: load it with transformers and compare it to its tower.

Python here only as a reference oracle (the language policy's first allowance): the claim
under test is that ``transformers`` -- the library the tower was trained in -- loads the
release as the same model. The exporter itself is Rust (``crates/qd-export``).

Two comparisons, one gate each:

1. **Weights, bit for bit.** Every ``model.language_model.<k>`` tensor in the release's
   ``model.safetensors`` must be ``tower.<k>.to(torch.bfloat16)`` from the averaged checkpoint
   -- torch's cast rounds to nearest, ties to even, the rule qd-export implements -- and every
   tensor in ``span_head.safetensors`` must be ``span_head.<k>`` unchanged. Compared as raw
   bits, so ``-0.0`` against ``0.0`` is a mismatch. Gate: zero mismatches, and the two name
   sets equal.
2. **Letter logits.** ``Qwen3_5TextModel`` is built three times and run in float32 on one
   device with deterministic algorithms:

   * ``release`` -- the release's ``config.json`` and ``model.safetensors``;
   * ``tower_bf16`` -- the base snapshot's config and the tower's tensors cast to bf16;
   * ``tower`` -- the base snapshot's config and the tower's tensors at their own dtype.

   At each row's target position the 17 letter logits (``A``..``P`` then ``Z``, the rows
   qd-metal scores, ``crates/qd-metal/src/tokenizer.rs:38-54``) are read off the tied
   embedding. **Gate: release vs tower_bf16, max |diff| == 0.** Identical weights through
   identical code give identical logits, so any difference is the loading path -- the config,
   a name, a tensor not loaded. **Reported, not gated: release vs tower** -- what the bf16 cast
   itself costs, which is zero when the tower is already bf16. No tolerance is invented for it
   (CLAUDE.md rule 2); the number is the evidence.

Run on the box against a real release (the plan's check: 100 val rows)::

    python tools/export_letter_parity.py --source AVG.safetensors --release RELEASE \\
        --base-snapshot SNAP --run-out RUN_OUT --rows 100 --device cuda --out-json REPORT.json

``--run-out`` is the ``--out`` the ft run used; its ``shards/val`` set is read through
``ShardReader``, which refuses a held-out split, and the first ``--rows`` choice/score rows in
file order are scored. Exit 0 only when both gates hold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

TEXT_PREFIX: Final[str] = "model.language_model."
TOWER_PREFIX: Final[str] = "tower."
SPAN_PREFIX: Final[str] = "span_head."
#: Every row qd-metal scores, in its order: the 16 option letters, then the abstain letter.
LETTERS: Final[str] = "ABCDEFGHIJKLMNOPZ"
MAX_ROWS: Final[int] = 10_000

Row = tuple[Sequence[int], int]


class ParityRefusal(ValueError):
    """The inputs cannot be compared as asked."""


def _split(tensors: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}


def read_source(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(tower, span_head)`` from an averaged checkpoint, prefixes stripped."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    tower, head = _split(tensors, TOWER_PREFIX), _split(tensors, SPAN_PREFIX)
    other = sorted(k for k in tensors if not k.startswith((TOWER_PREFIX, SPAN_PREFIX)))
    if other:
        raise ParityRefusal(f"{path}: tensors outside tower./span_head.: {other[:8]}")
    return tower, head


def read_release(release: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(text, span_head)`` from a release directory, the text prefix stripped."""
    from safetensors.torch import load_file

    text = load_file(str(release / "model.safetensors"))
    outside = sorted(k for k in text if not k.startswith(TEXT_PREFIX))
    if outside:
        raise ParityRefusal(
            f"{release}/model.safetensors holds {outside[:8]} outside {TEXT_PREFIX}"
        )
    return _split(text, TEXT_PREFIX), load_file(str(release / "span_head.safetensors"))


def weight_mismatches(
    tower: dict[str, Any], text: dict[str, Any], head_src: dict[str, Any], head_rel: dict[str, Any]
) -> list[str]:
    """Every way the release's tensors are not the tower's bf16 cast / the head unchanged."""
    import torch

    out: list[str] = []
    for name, (want, got) in (("tower", (tower, text)), ("span_head", (head_src, head_rel))):
        if set(want) != set(got):
            out.append(
                f"{name}: names differ: missing {sorted(set(want) - set(got))[:8]}, "
                f"extra {sorted(set(got) - set(want))[:8]}"
            )
    for key in sorted(set(tower) & set(text)):
        cast = tower[key].to(torch.bfloat16)
        rel = text[key]
        if rel.dtype != torch.bfloat16 or rel.shape != cast.shape:
            out.append(f"tower.{key}: release is {rel.dtype}{list(rel.shape)}")
        elif not torch.equal(cast.view(torch.int16), rel.view(torch.int16)):
            n = int((cast.view(torch.int16) != rel.view(torch.int16)).sum())
            out.append(f"tower.{key}: {n} of {cast.numel()} elements differ from the bf16 cast")
    for key in sorted(set(head_src) & set(head_rel)):
        src, rel = head_src[key], head_rel[key]
        if rel.dtype != src.dtype or rel.shape != src.shape:
            out.append(f"span_head.{key}: release is {rel.dtype}{list(rel.shape)}")
        elif not torch.equal(src.view(torch.int32), rel.view(torch.int32)):
            out.append(f"span_head.{key}: bits differ")
    return out


def letter_ids(tokenizer_json: Path) -> list[int]:
    """The 17 letter ids, each required to be one token and all distinct, as qd-metal does."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))
    ids: list[int] = []
    for letter in LETTERS:
        enc = tok.encode(letter, add_special_tokens=False).ids
        if len(enc) != 1:
            raise ParityRefusal(f"letter {letter!r} encodes to {enc}, not one token")
        if enc[0] in ids:
            raise ParityRefusal(f"letter {letter!r} shares id {enc[0]} with an earlier letter")
        ids.append(int(enc[0]))
    return ids


def build_model(
    config_dir: Path, state: dict[str, Any], *, device: str, attn_implementation: str
) -> Any:
    """``Qwen3_5TextModel`` from ``config_dir``'s ``text_config``, ``state`` loaded strictly,
    in float32 for compute."""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5TextModel

    config = AutoConfig.from_pretrained(config_dir)
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise ParityRefusal(f"{config_dir}/config.json has no text_config")
    text_config._attn_implementation = attn_implementation
    model = Qwen3_5TextModel(text_config)
    model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def letter_logits(model: Any, rows: Sequence[Row], ids: Sequence[int], *, device: str) -> Any:
    """``[len(rows), 17]`` float32 letter logits at each row's target position."""
    import torch

    weight = model.get_input_embeddings().weight
    letters = weight[torch.as_tensor(list(ids), device=weight.device)]
    out = []
    with torch.inference_mode():
        for tokens, target in rows:
            if not 0 <= target < len(tokens):
                raise ParityRefusal(f"target {target} is outside a {len(tokens)}-token row")
            x = torch.as_tensor([list(tokens[: target + 1])], dtype=torch.long, device=device)
            hidden = model(input_ids=x).last_hidden_state[0, -1]
            out.append((letters @ hidden).float().cpu())
    return torch.stack(out)


def _diff(a: Any, b: Any) -> dict[str, Any]:
    return {
        "max_abs": float((a - b).abs().max()),
        "top_letter_agrees": int((a.argmax(-1) == b.argmax(-1)).sum()),
        "rows": int(a.shape[0]),
    }


def compare(
    *,
    source: Path,
    release: Path,
    base_snapshot: Path,
    rows: Sequence[Row],
    device: str = "cpu",
    attn_implementation: str = "sdpa",
) -> dict[str, Any]:
    """Both comparisons; ``report["passed"]`` is true only when both gates hold."""
    import torch

    if not rows:
        raise ParityRefusal("no rows to score; an empty comparison would pass vacuously")
    tower, head_src = read_source(source)
    text, head_rel = read_release(release)
    mismatched = weight_mismatches(tower, text, head_src, head_rel)
    ids = letter_ids(release / "tokenizer.json")

    tower_is_bf16 = all(t.dtype == torch.bfloat16 for t in tower.values())
    kw = {"device": device, "attn_implementation": attn_implementation}
    release_logits = letter_logits(build_model(release, text, **kw), rows, ids, device=device)
    cast = {k: v.to(torch.bfloat16) for k, v in tower.items()}
    bf16_logits = letter_logits(build_model(base_snapshot, cast, **kw), rows, ids, device=device)
    tower_logits = (
        bf16_logits
        if tower_is_bf16
        else letter_logits(build_model(base_snapshot, tower, **kw), rows, ids, device=device)
    )
    gated = _diff(release_logits, bf16_logits)
    return {
        "source": str(source),
        "release": str(release),
        "base_snapshot": str(base_snapshot),
        "device": device,
        "attn_implementation": attn_implementation,
        "letter_ids": ids,
        "tower_dtypes": sorted({str(t.dtype).removeprefix("torch.") for t in tower.values()}),
        "weights": {
            "tower_tensors": len(tower),
            "span_head_tensors": len(head_src),
            "mismatched": mismatched,
        },
        "logits": {
            "gate": "release_vs_tower_bf16.max_abs == 0",
            "release_vs_tower_bf16": gated,
            "release_vs_tower": _diff(release_logits, tower_logits),
            "release_vs_tower_note": "reported, not gated: the cost of the bf16 cast"
            + (" (zero: the tower is already bf16)" if tower_is_bf16 else ""),
        },
        "passed": not mismatched and gated["max_abs"] == 0.0,
    }


def val_rows(run_out: Path, n: int) -> list[Row]:
    """The first ``n`` choice/score rows of ``run_out/shards/val``, in file order."""
    from qd_data.config import DataConfig
    from qd_train.artifacts import SLOT_CHOICE, SLOT_SCORE
    from qd_train.shards import ShardReader

    if not 0 < n <= MAX_ROWS:
        raise ParityRefusal(f"--rows {n} is outside (0, {MAX_ROWS}]")
    reader = ShardReader(run_out / "shards" / "val", config=DataConfig(), repo_root=run_out)
    if reader.header.split != "val":
        raise ParityRefusal(f"{run_out}/shards/val declares split {reader.header.split!r}")
    rows: list[Row] = []
    for i in range(len(reader)):
        if int(reader._slot_kind[i]) in (SLOT_CHOICE, SLOT_SCORE):
            rows.append((reader.sequence(i).tolist(), int(reader._target_index[i])))
            if len(rows) == n:
                return rows
    raise ParityRefusal(f"{run_out}/shards/val holds {len(rows)} choice/score rows, fewer than {n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="the averaged checkpoint")
    parser.add_argument("--release", type=Path, required=True, help="the qd-export release dir")
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--run-out", type=Path, required=True, help="the ft run's --out")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--out-json", type=Path, help="also write the report here")
    args = parser.parse_args(argv)

    # Before torch initialises CUDA: cuBLAS is deterministic only with a fixed workspace.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    try:
        report = compare(
            source=args.source,
            release=args.release,
            base_snapshot=args.base_snapshot,
            rows=val_rows(args.run_out, args.rows),
            device=args.device,
            attn_implementation=args.attn_implementation,
        )
    except ParityRefusal as exc:
        raise SystemExit(f"refused: {exc}") from exc
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        if args.out_json.exists():
            raise SystemExit(f"{args.out_json} already exists; refusing to overwrite it")
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
