"""The parts of tools/remap_parity_real.py that decide what its row says.

The tool itself loads the real 1.88B-parameter tower and is run by hand; these pin the three
helpers whose mistakes would not show in its output. Its first run found one of them wrong:
the stale-tensor check looked for the TOKENIZER's row count (248,077) on a model whose
embedding is 248,320 rows, so a stale copy of the real embedding would have passed it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

torch = pytest.importorskip("torch", reason="the tool builds torch modules")

import remap_parity_real as tool  # noqa: E402

from qd_train.remap import MAX_PARITY_LOGIT_BYTES  # noqa: E402


class _Holder(torch.nn.Module):
    def __init__(self, *shapes: tuple[int, ...]) -> None:
        super().__init__()
        self.ws = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(s)) for s in shapes)


def test_a_stale_copy_of_the_padded_embedding_is_found() -> None:
    """The real embedding's row count, not the tokenizer's: the case the first version
    of the check could not see."""
    stale = _Holder((13_787, 8), (248_320, 8))
    assert tool.module_tree_check(stale, full_rows=(248_320, 248_077)) == ["ws.1"]


def test_a_stale_tensor_at_the_tokenizer_count_is_found_too() -> None:
    stale = _Holder((248_077, 8))
    assert tool.module_tree_check(stale, full_rows=(248_320, 248_077)) == ["ws.0"]


def test_a_clean_remapped_tree_reports_nothing() -> None:
    clean = _Holder((13_787, 8), (8, 8))
    assert tool.module_tree_check(clean, full_rows=(248_320, 248_077)) == []


def test_sequences_are_offered_up_to_the_logit_cap_and_the_rest_are_counted() -> None:
    """The report refuses a sequence whose full-vocabulary logits exceed its byte cap; this
    skips those before offering and returns how many, so the row can say what share of the
    shard set it examined instead of implying all of it."""
    source_vocab = 248_077
    cap = MAX_PARITY_LOGIT_BYTES // (source_vocab * 4)
    lengths = [1, 5, cap, cap + 1, 7]
    new_to_old = np.arange(10, dtype=np.int64) * 3
    seqs = [np.arange(n, dtype=np.int32) % 10 for n in lengths]

    class _Reader:
        remap = SimpleNamespace(new_to_old=new_to_old)

        def __len__(self) -> int:
            return len(seqs)

        def sequence(self, i: int) -> np.ndarray:
            return seqs[i]

    chosen, skipped, total = tool.eligible_sequences(
        _Reader(), source_vocab=source_vocab, limit=2
    )
    assert total == 5
    assert skipped == 2, "the 1-token and the over-cap sequence"
    assert [len(c) for c in chosen] == [5, cap], "shard order, up to the limit"
    assert np.array_equal(chosen[0], new_to_old[seqs[1]]), "handed over as PRE-remap ids"


def test_the_tied_head_scores_hidden_states_against_the_input_embedding() -> None:
    """The tower has no output head; the input embedding is the head. The wrapper must
    compute exactly what fused_linear_cross_entropy computes from the same weight."""
    emb = torch.nn.Embedding(11, 4)

    class _Tower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = emb
            self.config = SimpleNamespace(vocab_size=11)

        def get_input_embeddings(self) -> torch.nn.Module:
            return self.embed

        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=torch.tanh(self.embed(input_ids)))

    tower = _Tower()
    ids = torch.tensor([[1, 4, 9]])
    logits = tool.TiedHead(tower)(ids)
    assert logits.shape == (1, 3, 11)
    expected = torch.tanh(emb(ids)) @ emb.weight.T
    assert torch.equal(logits, expected)
