"""CPU experiment: does dropping the padding mask change real positions' hidden states?

Tiny Qwen3.5 tower (the test fixture), right-padded batch, sdpa. Compares last_hidden_state at
real positions, and the gradient of a loss over real positions, mask vs no mask.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "tests"))

import test_backbone as tb  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)
with tempfile.TemporaryDirectory() as d:
    tower, _ = tb._tiny_tower(Path(d), gradient_checkpointing=False)
    model = tower.model
    lengths = torch.tensor([40, 23, 31])
    width = 48
    ids = torch.randint(1, tb.TINY_VOCAB, (3, width))
    for r, n in enumerate(lengths.tolist()):
        ids[r, n:] = 0
    mask = (torch.arange(width)[None, :] < lengths[:, None]).to(torch.int64)
    real = mask.bool()

    def run(attention_mask):
        model.zero_grad(set_to_none=True)
        h = model(input_ids=ids, attention_mask=attention_mask).last_hidden_state
        loss = (h[real] ** 2).sum()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        return h.detach()[real], grads

    h_mask, g_mask = run(mask)
    h_none, g_none = run(None)
    print("hidden at real positions: bit-identical", torch.equal(h_mask, h_none),
          "max abs diff", float((h_mask - h_none).abs().max()))
    diffs = {n: float((g_mask[n] - g_none[n]).abs().max()) for n in g_mask}
    worst = max(diffs, key=diffs.get)
    print("grads bit-identical", all(torch.equal(g_mask[n], g_none[n]) for n in g_mask),
          "worst", worst, diffs[worst])
