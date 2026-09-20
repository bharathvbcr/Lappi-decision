"""Rung 0: a byte-level decision model trained from scratch.

Torch-gated. torch is an optional ``mac`` extra and is not in the repo venv, so
``test_byte_decider.py`` ``importorskip``s and reports **skipped** rather than passed --
the same boundary ``docs/training-contract.md`` draws for every other model module.

## What this is, and what it is not

It is the cheapest rung of the ladder in ``docs/build-order-2026-09-19.md``, and the only
one that is unblocked today: from scratch means **no Qwen weights, so no Hugging Face
terms**, which is what the 2B lane is currently waiting on.

It is not a replacement for the 2B lane. ``cua-ai/cua-s1-forms`` reaches near-ceiling on
form-filling with 706K parameters over a 224-byte context and a 55-concept catalogue. A
commit-intent or change-scope decision over real code is a wider problem, and nothing here
claims that a small byte model covers it. The claim is narrower and testable: **it makes the
line-mapping question disappear**, and it can be trained and measured on this Mac this week.

## What is taken from where

``AUDIT/prior-art-jev-nimble-2026-09-19.md`` has the evidence. In short:

* **Byte-level embedding** — from ``cua-s1-forms``. Dissolves
  ``GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED`` rather than solving it, because in byte
  space a line start *is* an offset. See :mod:`qd_train.byte_context`.
* **Option-as-query scoring** — also from ``cua-s1-forms``: *"Each option becomes a query
  against the context tokens, producing an attended context vector, then a shared dot
  product turns each (option, attended-context) pair into one logit."* This replaces the
  letter-slice design, and with it the whole 16-option ceiling that
  ``GAP-SCHEMA-NOUL-LETTER-BUDGET`` is about. There are no vocabulary rows to run out of.
* **Contrastive pairs** — from Bespoke Nimble, whose 2,676 paired examples moved a 9B from
  roughly 66% to 90%. ``crates/qd-mutate`` already emits pairs whose label is a property of
  the tree-sitter edit, so unlike Nimble no separate validator is needed.
* **Abstention as a reserved row, not a catch-all option** — *ours*, and it is the one place
  every published system differs from this repo. Nimble documents adding an answer meaning
  "no match"; ``Mapika/decider-2b`` uses "optional catch-all options". Both put abstention
  into competition with the real answers for probability mass. Here it is a separate row,
  scored in the same softmax so it genuinely competes, but never occupying an option slot.

## The abstain row is per-example and it is LAST

``crates/qd-runtime/src/answer.rs`` computes ``noul_row = plan.rows - RESERVED_NOUL_ROWS``
and ``schema.rs:114`` sets ``rows = options.len() + RESERVED_NOUL_ROWS``. So for a slot with
``n`` live options the abstention is at column ``n`` -- **not** at a fixed last column of a
padded tensor. A model that writes abstention at the padded width while the runtime reads it
at ``n`` puts "no evidence" mass on a dead column, and no loss curve says so.

:data:`RESERVED_NOUL_ROWS` is imported from :mod:`qd_train.schema_mirror` rather than restated,
because that module already pins it against the Rust source and one copy is enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .byte_context import BYTE_VOCAB_SIZE, ID_PAD
from .schema_mirror import RESERVED_NOUL_ROWS

__all__ = [
    "RESERVED_NOUL_ROWS",
    "ByteDecider",
    "ByteDeciderConfig",
    "abstain_column",
]


def abstain_column(n_live_options: int) -> int:
    """Column the abstention occupies for a slot with ``n_live_options`` options.

    Mirrors ``answer.rs``: ``rows = n + RESERVED_NOUL_ROWS`` and
    ``noul_row = rows - RESERVED_NOUL_ROWS``, so the answer is ``n``. Written as the
    arithmetic rather than as ``n`` so that changing ``RESERVED_NOUL_ROWS`` moves both
    together.
    """
    if n_live_options < 1:
        raise ValueError(f"a slot needs at least one option, got {n_live_options}")
    return (n_live_options + RESERVED_NOUL_ROWS) - RESERVED_NOUL_ROWS


@dataclass(frozen=True, slots=True)
class ByteDeciderConfig:
    """Shape of the model.

    Defaults follow ``cua-s1-forms`` (2 layers, width 128, 4 heads) because that is the one
    published point of evidence for how small this can go, but ``max_context_bytes`` is much
    larger than its 224: a code hunk is not a form field. Capacity for a longer context is
    the first thing to raise if rung 0 underperforms, and saying so here is cheaper than
    rediscovering it.
    """

    width: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 4
    max_context_bytes: int = 1024
    max_option_bytes: int = 96
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.width % self.n_heads != 0:
            raise ValueError(
                f"width {self.width} must divide by n_heads {self.n_heads}; "
                "an uneven split silently drops a head's worth of capacity"
            )
        for name in ("width", "n_layers", "n_heads", "max_context_bytes", "max_option_bytes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class ByteDecider(nn.Module):
    """Byte context in, one logit per option plus a reserved abstain row, out."""

    def __init__(self, config: ByteDeciderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ByteDeciderConfig()
        c = self.config

        self.embed = nn.Embedding(BYTE_VOCAB_SIZE, c.width, padding_idx=ID_PAD)
        self.context_pos = nn.Embedding(c.max_context_bytes, c.width)
        self.option_pos = nn.Embedding(c.max_option_bytes, c.width)

        layer = nn.TransformerEncoderLayer(
            d_model=c.width,
            nhead=c.n_heads,
            dim_feedforward=c.width * c.ffn_mult,
            dropout=c.dropout,
            batch_first=True,
            norm_first=True,
        )
        # `enable_nested_tensor` is explicitly off: with `norm_first=True` torch cannot use
        # the nested-tensor fast path anyway and warns on every construction. Stating the
        # value we already get keeps the warning out of test output, where a routine warning
        # is how a real one goes unread.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=c.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(c.width)

        # The option query projection and the shared scorer. One matrix, applied to every
        # option, is what makes the option count a runtime property rather than a shape.
        self.to_query = nn.Linear(c.width, c.width, bias=False)
        self.score = nn.Linear(c.width, c.width, bias=False)

        # Abstention is a learned query, scored through the same path as a real option, so
        # "no evidence" and "this option" are compared in one softmax. A separate threshold
        # is the design where the two are never actually compared and the model can be
        # confident in both at once.
        self.noul_query = nn.Parameter(torch.zeros(c.width))
        nn.init.normal_(self.noul_query, std=0.02)

    def encode_context(
        self, context_ids: torch.Tensor, context_mask: torch.Tensor
    ) -> torch.Tensor:
        """``[B, L]`` byte ids -> ``[B, L, D]`` states."""
        length = context_ids.shape[1]
        if length > self.config.max_context_bytes:
            raise ValueError(
                f"context of {length} bytes exceeds the {self.config.max_context_bytes}-byte "
                "window; truncate through qd_train.byte_context.encode_context, which "
                "records what it dropped"
            )
        pos = torch.arange(length, device=context_ids.device)
        h = self.embed(context_ids) + self.context_pos(pos).unsqueeze(0)
        # `src_key_padding_mask` is True where a position must be ignored.
        h = self.encoder(h, src_key_padding_mask=~context_mask)
        return self.norm(h)

    def _option_queries(
        self, option_ids: torch.Tensor, option_mask: torch.Tensor
    ) -> torch.Tensor:
        """``[B, K, O]`` -> ``[B, K, D]``, mean-pooled over live bytes only."""
        o = option_ids.shape[2]
        if o > self.config.max_option_bytes:
            raise ValueError(
                f"option of {o} bytes exceeds the {self.config.max_option_bytes}-byte budget"
            )
        pos = torch.arange(o, device=option_ids.device)
        e = self.embed(option_ids) + self.option_pos(pos).view(1, 1, o, -1)
        m = option_mask.unsqueeze(-1).to(e.dtype)
        # An option with no live bytes would divide by zero; clamping the denominator keeps
        # it finite, and such a column is masked out of the softmax downstream anyway.
        pooled = (e * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return self.to_query(pooled)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        option_ids: torch.Tensor,
        option_mask: torch.Tensor,
        n_live_options: torch.Tensor,
    ) -> torch.Tensor:
        """Logits ``[B, K + RESERVED_NOUL_ROWS]``.

        For row ``i`` with ``n = n_live_options[i]``: columns ``0..n-1`` are the options,
        column ``n`` is the abstention, and every column above ``n`` is ``-inf``. That
        layout is ``answer.rs``'s, not a convenience -- see the module docstring.
        """
        hidden = self.encode_context(context_ids, context_mask)
        return self.score_from_hidden(
            hidden, context_mask, option_ids, option_mask, n_live_options
        )

    def score_from_hidden(
        self,
        hidden: torch.Tensor,
        context_mask: torch.Tensor,
        option_ids: torch.Tensor,
        option_mask: torch.Tensor,
        n_live_options: torch.Tensor,
    ) -> torch.Tensor:
        """The option head, given already-encoded context states.

        Split out of :meth:`forward` so a caller running a second head over the same context
        -- rung 0's span pointer does -- encodes once instead of twice. Encoding twice would
        not be merely slow: with dropout enabled the two passes differ, and the span head
        would be reading states the option head never saw.
        """
        b, k, _ = option_ids.shape
        if hidden.shape[0] != b:
            raise ValueError(
                f"batch mismatch: {hidden.shape[0]} contexts against {b} option rows"
            )
        if n_live_options.shape != (b,):
            raise ValueError(f"n_live_options must be [{b}], got {tuple(n_live_options.shape)}")
        if int(n_live_options.min()) < 1:
            raise ValueError("every row needs at least one live option")
        if int(n_live_options.max()) > k:
            raise ValueError(
                f"n_live_options up to {int(n_live_options.max())} exceeds the padded "
                f"width {k}; the abstain column would land outside the tensor"
            )

        h = hidden  # [B, L, D], encoded once by the caller
        queries = self._option_queries(option_ids, option_mask)  # [B, K, D]
        noul = self.noul_query.view(1, 1, -1).expand(b, RESERVED_NOUL_ROWS, -1)
        queries = torch.cat([queries, self.to_query(noul)], dim=1)  # [B, K+1, D]

        # Each query attends over the context, then one shared dot product per
        # (query, attended context) pair becomes its logit.
        scale = h.shape[-1] ** -0.5
        attn = torch.einsum("bkd,bld->bkl", queries, h) * scale
        attn = attn.masked_fill(~context_mask.unsqueeze(1), float("-inf"))
        attended = torch.einsum("bkl,bld->bkd", attn.softmax(dim=-1), h)
        logits = (self.score(queries) * attended).sum(dim=-1)  # [B, K+1]

        return self._place_abstain(logits, n_live_options)

    @staticmethod
    def _place_abstain(logits: torch.Tensor, n_live_options: torch.Tensor) -> torch.Tensor:
        """Move each row's abstain logit to column ``n`` and kill the columns above it.

        The order matters: mask the dead columns **first**, then scatter the abstention into
        column ``n``. Doing it the other way would write the abstention and then mask it
        away on any row where ``n`` equals the padded width.
        """
        width = logits.shape[1]
        k = width - RESERVED_NOUL_ROWS
        cols = torch.arange(width, device=logits.device).unsqueeze(0)
        n = n_live_options.unsqueeze(1)

        abstain = logits[:, k:].clone()  # the abstain logit(s), currently at the far right
        out = logits.masked_fill(cols >= n, float("-inf"))
        return out.scatter(1, n, abstain)
