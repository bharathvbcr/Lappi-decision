#!/usr/bin/env python3
"""Parity fixtures for qd-metal's span-head loader and D17 readout (ft-0f5a10c9f3e step 2,
ft-2ed2fd55e93 item 4), generated with the Python the head was trained with.

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/gen_span_fixture.py \
        --out crates/qd-metal/tests/fixtures/span

This is the **reference oracle** (policy: Python as a fixture generator that Rust tests compare
against). It writes, under ``--out``:

``head_scores.json``
    Cases for ``qd_train.heads.SpanPointerHead``: weights, the candidate rows and the query row of
    the hidden state, and the start/end scores ``serving_scores`` returns, all as f32 bit patterns.
    Rule named in the file: ``qd_train.heads.SpanPointerHead.forward`` (bilinear, bias-free
    projections, abstain row last), ``serving_scores`` (exactly ``line_count + 1`` rows).

``d17_compose.json``
    ``qd_train.lora.AnswerRowDelta.rows``: ``embed[answer ids] + D17`` in f32, over a bf16
    embedding, with the zero rows a fresh D17 has.

``tiny/``
    A tiny Qwen3.5 tower (hidden 64, one GDN layer then one attention layer, head_dim 256: the
    smallest shape qd-metal's kernels run) with a trained-shaped span head and a D17, in the
    layout ``tools/ckpt_average.py`` writes (``source.safetensors`` + manifest with
    ``span_head_source``), a base snapshot directory, a train manifest, and ``reference.json``:
    one ``code.defect_class`` request (a choice slot and a span slot) rendered by
    ``qd_data.render.render``, tokenized with the fixture's tokenizer, run through the tower in
    bf16 (the dtype the head trains under), and scored with ``SpanPointerHead`` and
    ``AnswerRowDelta``.

    **The hidden-state convention under test** is the tower's ``last_hidden_state``: the output of
    the final norm, cast to bf16, which is what ``QwenDecisionStep`` feeds the head
    (``backbone.py``: ``hidden[rows].float()`` into ``span_head.loss``). The Rust test runs
    qd-metal's tower on the same weights and checks both that its scores follow this convention
    and that the pre-norm residual stream would not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers.models.qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel

from qd_data.render import NOUL_LETTER, OPTION_LETTERS, render
from qd_data.schema import ChoiceSlot, Request, SpanSlot
from qd_train.artifacts import line_start_indices
from qd_train.heads import SpanPlan, SpanPointerHead, serving_scores
from qd_train.lora import D17_ROWS, AnswerRowDelta

HIDDEN = 64
VOCAB_PAD = 8
LETTERS = [*OPTION_LETTERS, NOUL_LETTER]
TOWER_CONFIG = {
    "attention_bias": False, "attn_output_gate": True, "head_dim": 256,
    "hidden_act": "silu", "hidden_size": HIDDEN, "intermediate_size": 128,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 1,
    "linear_num_value_heads": 1, "linear_value_head_dim": 32, "mlp_only_layers": [],
    "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 1,
    "rms_norm_eps": 1e-06, "tie_word_embeddings": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25},
}


def bits(t: torch.Tensor) -> list[int]:
    """A float32 tensor as its u32 bit patterns, flat."""
    return t.detach().to(torch.float32).contiguous().view(torch.int32).flatten().tolist()


def bf16_bits(t: torch.Tensor) -> list[int]:
    raw = t.detach().to(torch.bfloat16).contiguous().view(torch.int16).flatten().tolist()
    return [b & 0xFFFF for b in raw]


def head_cases() -> dict:
    cases = []
    gen = torch.Generator().manual_seed(20261012)
    for case_id, (hidden, n) in enumerate([(8, 1), (8, 3), (16, 5), (64, 9), (64, 2)]):
        head = SpanPointerHead(hidden)
        with torch.no_grad():
            for p in head.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
        # Candidate positions ascend and are not contiguous: line starts are not every token.
        steps = torch.randint(1, 5, (n,), generator=gen)
        pos = torch.cumsum(steps, 0) - 1
        query = int(pos[-1]) + 1 + int(torch.randint(0, 4, (1,), generator=gen))
        seq = query + 1
        # The tower emits bf16; the head casts to f32. Round the same way.
        states = (torch.randn(1, seq, hidden, generator=gen) * 1.5).to(torch.bfloat16)
        plan = SpanPlan(
            candidate_pos=pos.view(1, n).to(torch.int64),
            candidate_valid=torch.ones(1, n, dtype=torch.bool),
            n_candidates=torch.tensor([n]),
            query_index=torch.tensor([query]),
            gold_start=torch.zeros(1, dtype=torch.int64),
            gold_end=torch.zeros(1, dtype=torch.int64),
            abstaining=torch.zeros(1, dtype=torch.bool),
        )
        with torch.no_grad():
            start, end = head(states, plan)
        (s_row,), (e_row,) = serving_scores(start, plan), serving_scores(end, plan)
        assert s_row.shape[0] == n + 1 and torch.isfinite(s_row).all()
        cases.append({
            "case": case_id, "hidden": hidden, "n_candidates": n,
            "start_proj": bits(head.start_proj.weight), "end_proj": bits(head.end_proj.weight),
            "abstain_start": bits(head.abstain_start), "abstain_end": bits(head.abstain_end),
            "h_candidates": bits(states[0, pos].to(torch.float32)),
            "h_query": bits(states[0, query].to(torch.float32)),
            "start_scores": bits(s_row), "end_scores": bits(e_row),
        })
    return {
        "rule": "qd_train.heads.SpanPointerHead.forward + serving_scores, torch float32 CPU: "
                "query = h_query @ W^T (bias-free), candidate j scores query . h_candidate[j], "
                "the abstain row scores query . abstain vector and is LAST, n_candidates + 1 rows",
        "encoding": "every float as its IEEE-754 binary32 bit pattern (u32)",
        "tolerance_note": "torch's f32 matmul and a Rust f64-accumulated dot differ in the last "
                          "bits; the Rust test holds them to 2e-5 of the largest score",
        "cases": cases,
    }


def d17_cases() -> dict:
    gen = torch.Generator().manual_seed(20261013)
    hidden, vocab = 16, 40
    embed = (torch.randn(vocab, hidden, generator=gen) * 0.7).to(torch.bfloat16)
    # Edge values: a negative zero and a subnormal-in-bf16 value in the rows D17 will touch.
    embed[3, 0] = -0.0
    embed[5, 1] = torch.tensor(1e-38).to(torch.bfloat16)
    letters = list(range(10, 10 + D17_ROWS))
    delta_module = AnswerRowDelta(hidden)
    with torch.no_grad():
        # Rows 0..4 stay zero (a fresh D17); the rest carry a delta.
        delta_module.delta[5:] = torch.randn(D17_ROWS - 5, hidden, generator=gen) * 0.2
    row_ids = torch.tensor(letters)
    full = delta_module.rows(embed, row_ids).detach()
    return {
        "rule": "qd_train.lora.AnswerRowDelta.rows: embed.detach()[row_ids].to(float32) + delta",
        "encoding": "embed as bf16 bit patterns, everything else as binary32 bit patterns (u32)",
        "hidden": hidden, "vocab": vocab, "letters": letters,
        "embed": bf16_bits(embed), "delta": bits(delta_module.delta), "rows": bits(full),
    }


def tiny_tokenizer(texts: list[str]) -> tuple[Tokenizer, dict[str, int]]:
    words: list[str] = []
    pre = pre_tokenizers.Whitespace()
    for text in texts:
        for word, _ in pre.pre_tokenize_str(text):
            if word not in words:
                words.append(word)
    vocab = {"[UNK]": 0}
    for letter in LETTERS:  # the 17 answer letters are single tokens, as the loader requires
        vocab.setdefault(letter, len(vocab))
    for word in words:
        vocab.setdefault(word, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre
    return tok, vocab


def build_tiny(out: Path) -> None:
    tiny = out / "tiny"
    snap = tiny / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    # No blank or indented lines: the fixture tokenizer (whitespace words) has no token holding a
    # line's leading whitespace, and a line start in no token is refused by the rule under test.
    # The Rust unit tests cover those shapes with byte-level offsets.
    context = "def add(a, b):\ntotal = a + b\nreturn total\nprint(add(1, 2))\nresult = 3\n"
    request = Request(
        task="code.defect_class",
        context=context.encode(),
        question="Which defect class is this, and which lines show it?",
        slots=(
            ChoiceSlot("defect_class", options=("stub", "logic", "cosmetic", "clean")),
            SpanSlot("evidence"),
        ),
    )
    rendered = render(request, seed=None)
    choice, span = rendered.slots
    region = rendered.context_region()
    base = rendered.prefix.index(region)
    line_chars = [base + c for c in line_start_indices(region)]

    tok, vocab = tiny_tokenizer([rendered.prefix + choice.suffix, rendered.prefix + span.suffix])
    vocab_size = len(vocab) + VOCAB_PAD
    (snap / "tokenizer.json").write_text(tok.to_str())
    (snap / "tokenizer_config.json").write_text('{"model_max_length": 262144}')
    (snap / "vocab.json").write_text(json.dumps({"A": 1}))
    (snap / "merges.txt").write_text("#version: 0.2\n")
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {**TOWER_CONFIG, "vocab_size": vocab_size},
    }
    (snap / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    torch.manual_seed(20261014)
    model = Qwen3_5TextModel(Qwen3_5TextConfig(**config["text_config"]))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.copy_(torch.randn(p.shape) * (1.0 if name == "embed_tokens.weight" else 0.12))
            elif name.endswith("norm.weight"):
                p.copy_(torch.empty(p.shape).uniform_(-0.3, 0.3))
    model = model.to(torch.bfloat16).eval()  # weights rounded to bf16: what the release stores

    head = SpanPointerHead(HIDDEN)
    d17 = AnswerRowDelta(HIDDEN)
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn(p.shape) * 0.25)
        d17.delta.copy_(torch.randn(d17.delta.shape) * 0.15)

    tensors = {f"tower.{k}": v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    tensors |= {
        f"span_head.{k}": v.detach().clone().contiguous() for k, v in head.state_dict().items()
    }
    tensors["d17.delta"] = d17.delta.detach().clone().contiguous()
    source = tiny / "source.safetensors"
    save_file(tensors, str(source))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "tool": "tools/ckpt_average.py", "resumable": False, "n_inputs": 1,
        "vocab_size": vocab_size, "ft_row_ids": ["tiny-fixture-row"], "prompt_format": 2,
        "safetensors_sha256": sha, "n_tensors": len(tensors),
        "span_head_source": {
            "mode": "seed", "seed": 0, "input": "tiny-seed0.ckpt",
            "payload_digest": hashlib.sha256(b"tiny-seed0").hexdigest(),
        },
    }
    (tiny / "source.safetensors.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    train = {
        "manifest_format_version": 1, "split": "train", "data_snapshot_hash": "ab" * 32,
        "n_rows": 1, "held_out_families": ["code.language_id", "qa.answerability"],
        "entries": [{
            "row_id": "tiny-row-0", "content_hash": "00" * 32, "split": "train",
            "source_id": "fixture", "host": "fixture", "family_id": "code.defect_class",
            "repo_key": "r", "identity_key": "i", "licence_id": "MIT", "obligations": [],
        }],
    }
    (tiny / "train.json").write_text(json.dumps(train, indent=2) + "\n")

    # --- the reference run -----------------------------------------------------------------
    def encode(text: str):
        enc = tok.encode(text, add_special_tokens=False)
        return enc.ids, enc.offsets

    prefix_ids, _ = encode(rendered.prefix)
    full_text = rendered.prefix + span.suffix
    ids, offsets = encode(full_text)
    assert ids[: len(prefix_ids)] == prefix_ids, "the prefix/suffix token boundary must not merge"
    # The line-start rule: the first token whose character span contains the line's first
    # character (qd_train.shards._token_indices_for_chars); none containing it refuses.
    cand = []
    for c in line_chars:
        hit = [i for i, (s, e) in enumerate(offsets) if s <= c < e]
        assert hit, f"line start {c} lies in no token; the fixture text must avoid that"
        cand.append(hit[0])
    assert len(set(cand)) == len(cand), "the fixture's lines must not collapse onto one token"
    query = len(ids) - 1

    with torch.no_grad():
        hidden = model(input_ids=torch.tensor([ids])).last_hidden_state  # bf16, post final norm
        plan = SpanPlan(
            candidate_pos=torch.tensor([cand]),
            candidate_valid=torch.ones(1, len(cand), dtype=torch.bool),
            n_candidates=torch.tensor([len(cand)]), query_index=torch.tensor([query]),
            gold_start=torch.zeros(1, dtype=torch.int64),
            gold_end=torch.zeros(1, dtype=torch.int64),
            abstaining=torch.zeros(1, dtype=torch.bool),
        )
        start, end = head(hidden, plan)
        (s_row,), (e_row,) = serving_scores(start, plan), serving_scores(end, plan)

        letter_ids = [vocab[letter] for letter in LETTERS]
        embed = model.embed_tokens.weight
        q_choice, _ = encode(rendered.prefix + choice.suffix)
        h_choice = model(input_ids=torch.tensor([q_choice])).last_hidden_state[0, -1].float()
        rows_base = embed[torch.tensor(letter_ids)].float()
        rows_d17 = d17.rows(embed, torch.tensor(letter_ids))
        # The rendered choice slot has 4 options: rows A..D then noul (the last letter).
        pick = [*range(4), D17_ROWS - 1]
        logits_base = (h_choice @ rows_base.t())[pick]
        logits_d17 = (h_choice @ rows_d17.t())[pick]

    reference = {
        "rules": {
            "head": "qd_train.heads.SpanPointerHead + serving_scores "
                    "(torch float32 head, bf16 tower)",
            "hidden_state": "Qwen3_5TextModel.last_hidden_state: the output of the final norm, "
                            "bf16, as QwenDecisionStep feeds the head",
            "line_start_to_token": "first token whose character span contains the line's first "
                                   "character (qd_train.shards._token_indices_for_chars); refuse "
                                   "when none does; refuse-any when two lines share a token",
            "query": "the last token of prefix + slot suffix (training's target_index)",
            "d17": "qd_train.lora.AnswerRowDelta.rows: embed[letter ids] + D17, f32",
        },
        "prefix": rendered.prefix, "span_suffix": span.suffix, "choice_suffix": choice.suffix,
        "choice_rows": 5, "line_count": len(line_chars),
        "token_ids": ids, "token_offsets": offsets, "candidate_tokens": cand, "query_token": query,
        "hidden_candidates": bits(hidden[0, cand].float()),
        "hidden_query": bits(hidden[0, query].float()),
        "start_scores": bits(s_row), "end_scores": bits(e_row),
        "letter_ids": letter_ids, "choice_logits_base": bits(logits_base),
        "choice_logits_d17": bits(logits_d17),
        "tokenizer_sha256": hashlib.sha256((snap / "tokenizer.json").read_bytes()).hexdigest(),
        "generated_with": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    (tiny / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "head_scores.json").write_text(json.dumps(head_cases(), indent=1) + "\n")
    (args.out / "d17_compose.json").write_text(json.dumps(d17_cases(), indent=1) + "\n")
    build_tiny(args.out)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
