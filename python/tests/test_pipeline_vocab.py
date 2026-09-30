"""``tools/real_tokenizer_pipeline.py --vocab``: the full tokenizer vocabulary by default.

The corpus-built remap refused 608 of the phase-3 set's 2,210 held-out rows (row d849d700),
and a served model is sent text no corpus was counted over, so it could not serve either
(GAP-REMAP-CANNOT-ENCODE-THE-ROWS-IT-WAS-NOT-BUILT-FROM, decided (c) on 2026-09-29). The
default is now the identity remap; ``--vocab corpus`` keeps the older one, and the recipe
tells the two apart.

Torch-free, so ``make gates`` runs it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

PINNED = "0632f693d3b765b726499e7b4bf19c67959b75cb"


class _Reached(Exception):
    """Raised by a stand-in: the call got that far."""


def _run_kwargs(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, Any]:
    got: dict[str, Any] = {}

    def run(**kw: Any) -> None:
        got.update(kw)
        raise _Reached

    monkeypatch.setattr(pipeline, "run", run)
    with pytest.raises(_Reached):
        pipeline.main(argv)
    return got


def test_the_default_is_the_full_vocabulary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _run_kwargs(monkeypatch, ["--out", str(tmp_path)])["vocab"] == "full"


def test_the_corpus_remap_is_still_reachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    got = _run_kwargs(monkeypatch, ["--out", str(tmp_path), "--vocab", "corpus"])
    assert got["vocab"] == "corpus"


def test_an_unknown_policy_is_refused_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="vocab must be one of"):
        pipeline.run(out=tmp_path, max_pairs=1, blank_line_runs=False, rev=PINNED,
                     vocab="trimmed")


class _Config:
    def __init__(self, rows: object) -> None:
        self.text_config = type("T", (), {"vocab_size": rows})()


def test_the_full_vocabulary_is_sized_to_the_checkpoint_not_the_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """248,320 rows for 248,077 ids: sized to the tokenizer, the remap would slice 243 rows
    off and the trained checkpoint would no longer match its own config.json."""
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", classmethod(lambda cls, m: _Config(248_320))
    )
    assert pipeline.checkpoint_vocab_rows(tokenizer_len=248_077) == 248_320


@pytest.mark.parametrize("rows", [100, None, 0])
def test_a_checkpoint_narrower_than_its_tokenizer_is_refused(
    monkeypatch: pytest.MonkeyPatch, rows: object
) -> None:
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", classmethod(lambda cls, m: _Config(rows))
    )
    with pytest.raises(SystemExit):
        pipeline.checkpoint_vocab_rows(tokenizer_len=248_077)


def _recipe_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str]) -> str:
    ref = tmp_path / "refs-main"
    ref.write_text("b" * 40, encoding="utf-8")
    monkeypatch.setattr(pipeline, "MODEL_REF", ref)
    monkeypatch.setattr(pipeline, "run", lambda **kw: pytest.fail("ran"))
    seen: list[str] = []

    def protocol(**kw: Any) -> None:
        seen.append(kw["recipe_hash"])
        raise _Reached

    monkeypatch.setattr(pipeline, "Protocol", protocol)
    with pytest.raises(_Reached):
        pipeline.main(["--out", str(tmp_path), "--rev", PINNED,
                       "--ledger", str(tmp_path / "l.jsonl"), *extra])
    return seen[0]


def test_the_recipe_tells_a_full_set_from_a_trimmed_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = {
        "tool": "tools/real_tokenizer_pipeline.py",
        "model": pipeline.MODEL,
        "max_pairs": pipeline.DEFAULT_MAX_PAIRS,
        "blank_line_runs": False,
        "rev": PINNED,
        "max_consistency_rows": pipeline.PIPELINE_MAX_CONSISTENCY_ROWS,
    }

    def digest(recipe: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()

    corpus = _recipe_hash(monkeypatch, tmp_path, ["--vocab", "corpus"])
    full = _recipe_hash(monkeypatch, tmp_path, [])
    # A trimmed set hashes exactly as every row written before the flag did.
    assert corpus == digest(before)
    assert full == digest({**before, "vocab": "full"})
    assert full != corpus
