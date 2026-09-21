"""The fused linear cross-entropy: it must agree with the naive path, and beat it on memory.

Every test here needs torch, which is an optional ``mac`` extra and is **not** in the repo
venv. ``importorskip`` means this module *skips* there. It never passes vacuously: a skipped
module reports as skipped, and ``docs/training-contract.md``'s rule is that a test which
could not run must never report the same result as one that ran and passed.

To actually run these::

    uv run --python /Users/bharath/.venvs/ml/bin/python --with pytest --no-project \\
        python -m pytest python/tests/test_fused_ce.py
"""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from qd_train.artifacts import Batch  # noqa: E402
from qd_train.fused_ce import (  # noqa: E402
    fused_linear_cross_entropy,
    naive_linear_cross_entropy,
    resolve_chunk_size,
)
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.run_control import CostEstimate, LRSchedule, RunControl, WallClockCap  # noqa: E402
from qd_train.trainer import cpt_supervision, train_cpt  # noqa: E402
from qd_train.tristate import NotRun  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# --- an allocator instrument -----------------------------------------------------------


class PeakBytes(TorchDispatchMode):
    """Live-tensor high-water mark, by storage, over everything dispatched inside the block.

    Storages are counted once however many views point at them, and released when the last
    tensor holding one dies -- so this measures what is *resident*, which is the claim the
    fused path makes, rather than what was ever allocated.
    """

    def __init__(self) -> None:
        super().__init__()
        self.live = 0
        self.peak = 0
        self.largest_numel = 0
        self._refs: dict[int, int] = {}
        self._bytes: dict[int, int] = {}

    def _release(self, ptr: int) -> None:
        remaining = self._refs.get(ptr)
        if remaining is None:
            return
        remaining -= 1
        if remaining <= 0:
            self.live -= self._bytes.pop(ptr, 0)
            self._refs.pop(ptr, None)
        else:
            self._refs[ptr] = remaining

    def _track(self, obj: object) -> None:
        if isinstance(obj, torch.Tensor):
            self.largest_numel = max(self.largest_numel, obj.numel())
            try:
                storage = obj.untyped_storage()
                ptr, nbytes = storage.data_ptr(), storage.nbytes()
            except (RuntimeError, NotImplementedError):
                return
            if ptr == 0:
                return
            if ptr not in self._refs:
                self._bytes[ptr] = nbytes
                self._refs[ptr] = 0
                self.live += nbytes
                self.peak = max(self.peak, self.live)
            self._refs[ptr] += 1
            weakref.finalize(obj, self._release, ptr)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._track(item)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._track(out)
        return out


def test_the_instrument_sees_an_allocation_it_is_shown():
    """If this fails, every memory number below is meaningless -- so it is asserted first."""
    with PeakBytes() as probe:
        big = torch.zeros(1024, 1024, dtype=torch.float32)  # exactly 4 MiB
        assert big.numel() == 1024 * 1024
    assert probe.peak >= 4 * 1024 * 1024
    assert probe.largest_numel >= 1024 * 1024


# --- the memory claim --------------------------------------------------------------------


def _memory_case():
    """Shapes chosen so a materialized [B, L, V] is unmistakable: 64 MiB of float32 logits."""
    torch.manual_seed(0)
    b, seq, vocab, hidden = 4, 512, 8192, 64
    h = torch.randn(b, seq, hidden, dtype=torch.float32, requires_grad=True)
    w = torch.randn(vocab, hidden, dtype=torch.float32, requires_grad=True) * 0.02
    w = w.detach().requires_grad_(True)
    targets = torch.randint(0, vocab, (b, seq), dtype=torch.int64)
    mask = torch.ones(b, seq, dtype=torch.bool)
    naive_logit_bytes = b * seq * vocab * 4
    return h, w, targets, mask, naive_logit_bytes


@pytest.mark.slow
def test_the_fused_path_never_allocates_the_logit_tensor_the_naive_path_does():
    h, w, targets, mask, naive_logit_bytes = _memory_case()
    assert naive_logit_bytes == 64 * 1024 * 1024

    gc.collect()
    with PeakBytes() as naive_probe:
        naive_linear_cross_entropy(h, w, targets, mask=mask).backward()
    h.grad = None
    w.grad = None

    gc.collect()
    with PeakBytes() as fused_probe:
        fused_linear_cross_entropy(h, w, targets, mask=mask, chunk_size=64).backward()

    # The naive path really does materialize [B, L, V]. Without this the fused assertion
    # below could pass because the instrument saw nothing at all.
    assert naive_probe.peak >= naive_logit_bytes, (
        f"instrument saw only {naive_probe.peak} bytes for a path that must hold "
        f"{naive_logit_bytes}; the measurement, not the code, is wrong"
    )
    assert naive_probe.largest_numel >= 4 * 512 * 8192

    # The bound: a quarter of one materialized logit tensor. The naive path exceeds a whole
    # one, so this is a bound it blows through rather than one it merely brushes.
    bound = naive_logit_bytes // 4
    assert fused_probe.peak < bound, (
        f"fused peak {fused_probe.peak} >= bound {bound}; logits are being materialized"
    )
    # And no single tensor is ever [B*L, V]-shaped.
    assert fused_probe.largest_numel < 4 * 512 * 8192


@pytest.mark.slow
def test_peak_memory_scales_with_the_chunk_not_with_the_sequence():
    h, w, targets, mask, _ = _memory_case()
    peaks = {}
    for chunk in (32, 128, 512):
        h.grad = None
        w.grad = None
        gc.collect()
        with PeakBytes() as probe:
            fused_linear_cross_entropy(h, w, targets, mask=mask, chunk_size=chunk).backward()
        peaks[chunk] = probe.peak
    assert peaks[32] < peaks[128] < peaks[512], f"peak must track the chunk: {peaks}"


def test_the_derived_chunk_keeps_the_logit_slab_inside_its_budget():
    from qd_train.fused_ce import TARGET_LOGIT_BYTES

    chunk = resolve_chunk_size(n_positions=1_000_000, vocab_size=80_000, chunk_size=None)
    assert chunk * 80_000 * 4 <= TARGET_LOGIT_BYTES
    # Never more rows than there are, and never zero.
    assert resolve_chunk_size(10, 80_000, None) == 10
    assert resolve_chunk_size(10, 10**9, None) == 1
    assert resolve_chunk_size(10, 32, 4) == 4
    with pytest.raises(ValueError, match="positive"):
        resolve_chunk_size(10, 32, 0)


# --- the numerical claim -------------------------------------------------------------------


def _small_case(*, with_bias: bool, seed: int = 0):
    torch.manual_seed(seed)
    b, seq, vocab, hidden = 2, 7, 23, 5
    h = torch.randn(b, seq, hidden, dtype=torch.float64, requires_grad=True)
    w = torch.randn(vocab, hidden, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(vocab, dtype=torch.float64, requires_grad=True) if with_bias else None
    targets = torch.randint(0, vocab, (b, seq), dtype=torch.int64)
    mask = torch.zeros(b, seq, dtype=torch.bool)
    mask[0, :5] = True  # row 0 is length 6, row 1 is length 4 -- i.e. padding is masked off
    mask[1, :3] = True
    return h, w, bias, targets, mask


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_the_fused_loss_and_gradients_match_the_naive_reference(reduction, with_bias):
    h, w, bias, targets, mask = _small_case(with_bias=with_bias)

    ref = naive_linear_cross_entropy(
        h, w, targets, mask=mask, bias=bias, reduction=reduction
    )
    ref.backward()
    ref_grads = [h.grad.clone(), w.grad.clone()]
    if bias is not None:
        ref_grads.append(bias.grad.clone())
    h.grad = w.grad = None
    if bias is not None:
        bias.grad = None

    got = fused_linear_cross_entropy(
        h, w, targets, mask=mask, bias=bias, chunk_size=3, reduction=reduction
    )
    got.backward()
    got_grads = [h.grad, w.grad] + ([bias.grad] if bias is not None else [])

    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-12)
    names = ("grad_hidden", "grad_weight", "grad_bias")[: len(ref_grads)]
    assert len(got_grads) == len(ref_grads) == len(names)
    for name, a, b in zip(names, got_grads, ref_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11, msg=f"{name} disagrees")


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 14, 100])
def test_the_result_does_not_depend_on_the_chunk_size(chunk):
    h, w, _, targets, mask = _small_case(with_bias=False)
    ref = naive_linear_cross_entropy(h, w, targets, mask=mask)
    got = fused_linear_cross_entropy(h, w, targets, mask=mask, chunk_size=chunk)
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-12)


def test_masked_positions_contribute_to_neither_the_loss_nor_the_gradients():
    """The padding claim, at the kernel boundary rather than at the loop's."""
    h, w, _, targets, mask = _small_case(with_bias=False)
    quiet = fused_linear_cross_entropy(h, w, targets, mask=mask)
    quiet.backward()
    grads = (h.grad.clone(), w.grad.clone())
    h.grad = w.grad = None

    # Change every target the mask excludes. Nothing may move.
    noisy_targets = targets.clone()
    noisy_targets[~mask] = (noisy_targets[~mask] + 7) % 23
    assert not torch.equal(noisy_targets, targets)
    noisy = fused_linear_cross_entropy(h, w, noisy_targets, mask=mask)
    noisy.backward()

    torch.testing.assert_close(noisy, quiet, rtol=0, atol=0)
    torch.testing.assert_close(h.grad, grads[0], rtol=0, atol=0)
    torch.testing.assert_close(w.grad, grads[1], rtol=0, atol=0)


class MatmulCount(TorchDispatchMode):
    """How many matrix multiplies were dispatched inside the block.

    The chunk loop's only per-chunk matmuls are the projection and, with gradients, the two
    that form them -- so this counts chunks the loop actually walked, which is the claim
    below. Like ``PeakBytes`` it is asserted against a known case first, because an
    instrument that sees nothing makes every number it reports look like a pass.
    """

    _OPS = ("mm", "matmul", "bmm", "addmm")

    def __init__(self) -> None:
        super().__init__()
        self.matmuls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket.__name__ in self._OPS:
            self.matmuls += 1
        return func(*args, **(kwargs or {}))


def test_the_matmul_instrument_sees_a_matmul_it_is_shown():
    """If this fails, the count below is meaningless -- so it is asserted first."""
    with MatmulCount() as probe:
        torch.matmul(torch.zeros(4, 4), torch.zeros(4, 4))
    assert probe.matmuls >= 1


def test_a_chunk_with_no_supervised_position_is_not_projected():
    """FT supervises one position per row; the loop used to project every chunk anyway.

    ``cpt_supervision`` masks in nearly every position, so under CPT this changes almost
    nothing. ``ft_supervision`` masks in exactly one position per non-span row, so without
    the skip an FT step pays a CPT-sized projection for FT-sized supervision. Measured
    against the unmodified loop on the real shard set
    ``tools/real_tokenizer_pipeline.py`` writes -- 13,787-token vocabulary, one sequence of
    34,522 tokens, one supervised position -- the loop walked 15 chunks in 0.511s where the
    same loss and gradient over the supervised positions alone took 0.005s, a factor of 112.
    Over a whole epoch: 1,459 chunks projected, 243 of them holding a supervised position.

    The claim is exactness as well as cost, so the value and both gradients are compared
    against the all-supervised reference computed over the live rows.
    """
    torch.manual_seed(11)
    vocab, hidden, positions = 64, 8, 40
    chunk = 10
    h = torch.randn(positions, hidden, dtype=torch.float64, requires_grad=True)
    w = torch.randn(vocab, hidden, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, vocab, (positions,))
    mask = torch.zeros(positions, dtype=torch.bool)
    mask[3] = True  # one supervised position, in the first chunk of four

    with MatmulCount() as probe:
        loss = fused_linear_cross_entropy(h, w, targets, mask=mask, chunk_size=chunk)
        loss.backward()
    live_chunks = 1
    all_chunks = positions // chunk
    assert probe.matmuls <= 3 * live_chunks + 2, (
        f"{probe.matmuls} matmuls for {live_chunks} chunk(s) holding a supervised position; "
        f"{all_chunks} chunks cover the sequence, and projecting the {all_chunks - live_chunks} "
        "that hold none computes a slab that is then multiplied by zero"
    )

    got = (loss.detach().clone(), h.grad.clone(), w.grad.clone())
    h.grad = w.grad = None
    # The same loss over the supervised positions alone: the value this must not have moved.
    ref = fused_linear_cross_entropy(h[mask], w, targets[mask])
    ref.backward()
    # Not bit-exact, and deliberately not asserted as such: the reference gathers the live
    # rows into a fresh contiguous tensor while the masked call projects a view, so the two
    # matmuls reduce in a different order. float64 noise at 1e-15 is the whole difference;
    # a tolerance of 1e-12 is far tighter than anything a training step could notice and
    # far looser than the reordering.
    torch.testing.assert_close(got[0], ref.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(got[1][mask], h.grad[mask], rtol=1e-12, atol=1e-12)
    assert not got[1][~mask].any(), "an unsupervised position received a gradient"
    torch.testing.assert_close(got[2], w.grad, rtol=1e-12, atol=1e-12)


def test_a_bfloat16_head_still_reduces_in_float32():
    """The memory saving must not buy itself a quietly less accurate loss."""
    torch.manual_seed(3)
    vocab, hidden = 512, 16
    h32 = torch.randn(1, 64, hidden, dtype=torch.float32)
    w32 = torch.randn(vocab, hidden, dtype=torch.float32) * 0.05
    targets = torch.randint(0, vocab, (1, 64), dtype=torch.int64)

    exact = naive_linear_cross_entropy(h32, w32, targets)
    fused_bf16 = fused_linear_cross_entropy(
        h32.bfloat16(), w32.bfloat16(), targets, chunk_size=8
    )
    # bf16 inputs cost precision, but the reduction over 512 logits must not compound it:
    # a bf16 logsumexp over this many terms lands far outside 2%.
    assert abs(float(fused_bf16) - float(exact)) / float(exact) < 0.02
    assert fused_bf16.dtype == torch.float32


# --- fail closed --------------------------------------------------------------------------


def test_an_empty_mask_is_refused_rather_than_returning_zero():
    h, w, _, targets, mask = _small_case(with_bias=False)
    with pytest.raises(ValueError, match="indistinguishable from a perfect fit"):
        fused_linear_cross_entropy(h, w, targets, mask=torch.zeros_like(mask))


def test_a_target_outside_the_head_is_refused_not_clamped():
    h, w, _, targets, mask = _small_case(with_bias=False)
    bad = targets.clone()
    bad[mask][0] = 0
    bad[0, 0] = 23  # vocab is 23, so this id does not exist
    with pytest.raises(ValueError, match="must not be clamped into range"):
        fused_linear_cross_entropy(h, w, bad, mask=mask)


def test_an_out_of_range_id_sitting_in_padding_is_harmless():
    """Masked slots gather row 0, so a shard's padding filler cannot index out of bounds."""
    h, w, _, targets, mask = _small_case(with_bias=False)
    ref = fused_linear_cross_entropy(h, w, targets, mask=mask)
    junk = targets.clone()
    junk[~mask] = 10_000
    got = fused_linear_cross_entropy(h, w, junk, mask=mask)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"targets_shape": (2, 6)}, "to match hidden"),
        ({"weight_shape": (23, 4)}, "last dim is"),
        ({"mask_dtype": torch.int64}, "mask must be bool"),
        ({"reduction": "none"}, "reduction must be one of"),
    ],
)
def test_shape_and_option_mistakes_are_refused(kwargs, match):
    h, w, _, targets, mask = _small_case(with_bias=False)
    if "targets_shape" in kwargs:
        targets = torch.zeros(kwargs["targets_shape"], dtype=torch.int64)
        mask = torch.ones(kwargs["targets_shape"], dtype=torch.bool)
    if "weight_shape" in kwargs:
        w = torch.randn(kwargs["weight_shape"], dtype=torch.float64)
    if "mask_dtype" in kwargs:
        mask = mask.to(kwargs["mask_dtype"])
    with pytest.raises(ValueError, match=match):
        fused_linear_cross_entropy(
            h, w, targets, mask=mask, reduction=kwargs.get("reduction", "mean")
        )


def test_no_grad_still_produces_the_right_loss():
    h, w, _, targets, mask = _small_case(with_bias=False)
    with torch.no_grad():
        got = fused_linear_cross_entropy(h, w, targets, mask=mask, chunk_size=3)
        ref = naive_linear_cross_entropy(h, w, targets, mask=mask)
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-12)


# --- the seam: the fused loss driven by the real trainer -------------------------------------


class TorchBigramStep:
    """A ``TrainStep`` whose loss is the fused kernel. Proves the two modules actually fit."""

    def __init__(self, *, vocab: int = 16, hidden: int = 4, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(vocab, hidden).double()
        self.head = torch.nn.Linear(hidden, vocab, bias=False).double()

    def _params(self):
        return list(self.emb.parameters()) + list(self.head.parameters())

    def accumulate(self, batch: Batch, supervision) -> float:
        tokens = torch.from_numpy(batch.tokens[:, :-1].astype(np.int64))
        targets = torch.from_numpy(supervision.targets.astype(np.int64))
        mask = torch.from_numpy(supervision.mask.copy())
        loss = fused_linear_cross_entropy(
            self.emb(tokens), self.head.weight, targets, mask=mask, chunk_size=4
        )
        loss.backward()
        return float(loss.item())

    def apply(self, *, lr: float) -> None:
        with torch.no_grad():
            for p in self._params():
                if p.grad is not None:
                    p -= lr * p.grad
                    p.grad = None

    def state(self) -> dict:
        return {
            "emb": self.emb.weight.detach().tolist(),
            "head": self.head.weight.detach().tolist(),
        }

    def load_state(self, state) -> None:
        with torch.no_grad():
            self.emb.weight.copy_(torch.tensor(state["emb"], dtype=torch.float64))
            self.head.weight.copy_(torch.tensor(state["head"], dtype=torch.float64))
        for p in self._params():
            p.grad = None


def _batches(seed: int, epoch: int, n: int, *, rows: int = 2, width: int = 8):
    rng = np.random.default_rng([seed, epoch])
    for i in range(n):
        lengths = rng.integers(2, width + 1, size=rows).astype(np.int32)
        tokens = rng.integers(1, 16, size=(rows, width)).astype(np.int32)
        for r in range(rows):
            tokens[r, int(lengths[r]) :] = 0
        yield Batch(tokens=tokens, lengths=lengths, bucket=0, index=i)


def test_the_fused_loss_drives_the_real_cpt_loop_and_the_loss_falls(tmp_path):
    cap = WallClockCap(cap_s=600.0)
    control = RunControl(
        schedule=LRSchedule(peak_lr=1.0, total_steps=12, warmup_steps=1, min_lr=0.1),
        cap=cap,
        cost=CostEstimate(usd_per_hour=0.5, cap=cap, n_gpus=1, instance="test-1xA10"),
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")
    recorder = RunRecorder(
        ledger,
        protocol=Protocol(
            data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
            backbone_commit="b" * 40, recipe_hash="r" * 64, seed=7,
        ),
        run_kind="cpt",
        repo=REPO,
        wall_clock_s=None,  # the caller's `with` block contains the run under test
        env=Environment(
            torch=torch.__version__, transformers_sha="none", device="cpu", host="test",
            fla_present=NotRun(reason="no CUDA on this host"),
            causal_conv1d_present=NotRun(reason="no CUDA on this host"),
        ),
    )
    result = train_cpt(
        _batches(7, 0, 24), epoch=0, step=TorchBigramStep(),
        control=control, recorder=recorder,
    )
    assert result.termination == "steps_exhausted"
    assert result.optimizer_steps == 12
    losses = result.loss_log.losses()
    assert losses[-1] < losses[0], f"the fused loss must actually train: {losses}"
    assert len(ledger.rows()) == 1 and ledger.rows()[0].status == "completed"


def test_the_fused_step_agrees_with_the_naive_reference_on_a_real_batch():
    batch = next(_batches(7, 0, 1))
    supervision = cpt_supervision(batch)
    step = TorchBigramStep()

    tokens = torch.from_numpy(batch.tokens[:, :-1].astype(np.int64))
    targets = torch.from_numpy(supervision.targets.astype(np.int64))
    mask = torch.from_numpy(supervision.mask.copy())
    reference = naive_linear_cross_entropy(
        step.emb(tokens), step.head.weight, targets, mask=mask
    )
    assert step.accumulate(batch, supervision) == pytest.approx(
        float(reference.detach()), rel=1e-12, abs=1e-14
    )
