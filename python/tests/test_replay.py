"""``qd_train.replay``: 8-gram decontamination, its attestation, and prior_kl replay.

The decontamination half is torch-free and runs in the core venv. The prior_kl half is
torch-gated per test, so the core suite still counts the torch-free tests rather than
collapsing the module into one skip.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_data.render import render
from qd_data.schema import ChoiceSlot, Request
from qd_train.artifacts import NO_SPAN, SLOT_CHOICE, SLOT_SPAN, Batch
from qd_train.replay import (
    ReplayRefusal,
    check_attestation,
    decontaminate,
    prompt_content,
    target_digest,
    word_ngrams,
)

LONG_A = (
    "the quick brown fox jumps over the lazy dog while the farmer counts sheep in the "
    "early morning light near the old mill by the river"
)
LONG_B = (
    "a completely different passage about compilers and register allocation that shares "
    "no run of eight words with the other one at all whatsoever"
)


def _prompt(context: str, *, question: str = "Which one?", options=("yes", "no")) -> str:
    request = Request(
        task="devcouncil.verdict", context=context.encode("utf-8"), question=question,
        slots=(ChoiceSlot(name="verdict", options=options),), example_id="x",
    )
    return render(request, seed=None).prompt_for("verdict")


# --- decontamination ------------------------------------------------------------------------


def test_prompt_content_keeps_the_row_and_drops_the_template():
    text = prompt_content(_prompt(LONG_A, options=("keep this option", "and this one")))
    assert LONG_A in text and "Which one?" in text
    assert "keep this option" in text and "and this one" in text
    assert "<|qd_" not in text and "noul" not in text and "devcouncil" not in text


def test_two_rows_of_one_family_share_no_8gram_through_the_template_alone():
    """The failure this cut exists to avoid: the rendered template is identical across a
    family, and comparing whole prompts would make every row contaminate every other."""
    a = word_ngrams(prompt_content(_prompt(LONG_A)))
    b = word_ngrams(prompt_content(_prompt(LONG_B)))
    assert a and b and not a & b


def test_ngrams_are_a_stable_hash_not_pythons_salted_one():
    grams = word_ngrams("one two three four five six seven eight", 8)
    expected = hashlib.blake2b(b"one two three four five six seven eight", digest_size=8).digest()
    assert grams == frozenset({expected})
    assert word_ngrams("too short to have one", 8) == frozenset()


def test_a_replay_row_that_contains_a_val_row_is_a_hit_and_refuses_the_set():
    replay = {"r0": "prefix words " + LONG_A + " suffix", "r1": LONG_B, "r2": "short"}
    targets = {"val": {"v0": LONG_A}, "heldout": {"h0": "nothing in common here at all ok"}}
    report = decontaminate(replay, targets)
    assert report.hits == {"val": 1, "heldout": 0}
    assert report.replay_rows_checked == 2 and report.replay_rows_too_short == 1
    assert report.hit_examples[0]["replay_row"] == "r0"
    assert not report.clean
    body = report.to_json()
    assert body["clean"] is False and body["targets"]["val"]["digest"] == target_digest(
        targets["val"]
    )


def test_a_clean_report_needs_at_least_one_row_actually_checked():
    """All-too-short replay rows compared nothing, and must not read as clean."""
    report = decontaminate({"r": "few words"}, {"val": {"v": LONG_A}})
    assert report.replay_rows_checked == 0 and not report.clean
    with pytest.raises(ReplayRefusal, match="against nothing"):
        decontaminate({"r": LONG_A}, {})


def _attestation(**over):
    targets = {"val": {"v0": LONG_A}, "heldout": {"h0": LONG_B}}
    report = decontaminate({"r0": "unrelated " * 20}, targets)
    body = {**report.to_json(), "replay_shard_hash": "abc"}
    body.update(over)
    return body, targets


def test_the_attestation_is_accepted_only_for_its_own_replay_set_and_targets():
    body, targets = _attestation()
    check_attestation(body, replay_shard_hash="abc", verify_targets={"val": targets["val"]},
                      required_targets=("val", "heldout"))
    with pytest.raises(ReplayRefusal, match="replay shard set 'abc', not 'xyz'"):
        check_attestation(body, replay_shard_hash="xyz", verify_targets={},
                          required_targets=("val",))
    with pytest.raises(ReplayRefusal, match="row digest"):
        check_attestation(body, replay_shard_hash="abc", verify_targets={"val": {"v1": LONG_A}},
                          required_targets=("val",))
    with pytest.raises(ReplayRefusal, match="never checked"):
        check_attestation(body, replay_shard_hash="abc", verify_targets={},
                          required_targets=("val", "test"))
    dirty, _ = _attestation(clean=False, hits={"val": 2, "heldout": 0})
    with pytest.raises(ReplayRefusal, match="contamination"):
        check_attestation(dirty, replay_shard_hash="abc", verify_targets={},
                          required_targets=("val",))


# --- the full pair list (Fable's J6(a) ruling, Q5) ----------------------------------------------


def _pairs(report):
    return {(p.replay_row, p.target, p.target_row) for p in report.pairs}


def test_a_row_hitting_two_targets_lists_both():
    """Fable's test. The val-side exclusion reads the target projection of the pair list, so
    a val row that is only some replay row's SECOND-best target must still be listed:
    ``hit_examples`` keeps the best target per (row, set) and would miss it."""
    replay = {"r0": LONG_A + " and then " + LONG_B, "r1": "nothing in common with either one here"}
    targets = {"val": {"va": LONG_A, "vb": LONG_B}, "heldout": {"ha": LONG_A}}
    report = decontaminate(replay, targets)
    assert _pairs(report) == {("r0", "val", "va"), ("r0", "val", "vb"), ("r0", "heldout", "ha")}
    # hits[T] stays a count of replay rows, not of pairs.
    assert report.hits == {"val": 1, "heldout": 1}
    assert all(p.containment >= report.threshold for p in report.pairs)


def test_the_pair_list_is_every_pair_at_or_above_the_threshold_and_nothing_below():
    words = LONG_A.split()
    grams_a = len(word_ngrams(LONG_A))
    # The first 14 words carry 7 of LONG_A's 8-grams: under half, so no pair.
    under = " ".join(words[:14])
    # The first 18 words carry 11: at or over half, so a pair, with its exact counts.
    over = " ".join(words[:18])
    assert 7 / grams_a < 0.5 <= 11 / grams_a
    report = decontaminate({"u": under, "o": over}, {"val": {"va": LONG_A}})
    (pair,) = report.pairs
    assert (pair.replay_row, pair.target, pair.target_row) == ("o", "val", "va")
    assert (pair.shared, pair.target_ngrams) == (11, grams_a)
    assert pair.containment == 11 / grams_a


def test_the_pair_list_is_complete_past_the_examples_cap():
    """``hit_examples`` stops at MAX_HIT_EXAMPLES (50); the pair list does not. The phase-4
    attestation named 50 of its 185 hits, so the rows to exclude could not be read off it
    (GAP-DECONTAM-ATTESTATION-NAMES-50-OF-N-HITS-2026-10-02)."""
    replay = {f"r{i:02d}": f"row {i} " + LONG_A for i in range(60)}
    report = decontaminate(replay, {"val": {"va": LONG_A}, "heldout": {"hb": LONG_B}})
    assert report.hits == {"val": 60, "heldout": 0}
    assert len(report.hit_examples) == 50
    assert sorted(p.replay_row for p in report.pairs) == sorted(replay)


def test_the_attestation_body_is_version_1_and_byte_identical_to_before_the_pair_list():
    """Characterization: passes before and after the pair list was added, by design. The
    sha256 below is the body ``tools/replay_decontam.py`` writes for this fixture at a502670,
    before the change; ``real_ft_run`` at a502670 checks this format, so the pair list must
    not enter it."""
    import json

    replay = {f"r{i:02d}": f"row {i} " + LONG_A for i in range(60)}
    replay["rb"] = "x " + LONG_B
    replay["short"] = "too short"
    targets = {"val": {"va": LONG_A, "vb": LONG_B}, "heldout": {"ha": LONG_A}}
    body = json.dumps(decontaminate(replay, targets).to_json(), indent=2, sort_keys=True) + "\n"
    assert hashlib.sha256(body.encode()).hexdigest() == (
        "8b56ba2486fe8262f387dc143c98c0b153ee510269b6ffbcbe593d1d79cc7c62"
    )
    assert json.loads(body)["version"] == 1 and "pairs" not in json.loads(body)


# --- prior_kl ---------------------------------------------------------------------------------


def test_prior_kl_is_kl_base_to_model_with_the_p_minus_q_gradient():
    torch = pytest.importorskip("torch")
    from qd_train.replay import prior_kl

    logits = torch.tensor([[5.0, -5.0, 0.0], [0.1, 0.2, 0.3]], requires_grad=True)
    base = torch.tensor([[2.0, 1.0, 0.0], [0.3, 0.2, 0.1]])
    kl = prior_kl(logits, base)
    q = torch.softmax(base, -1)
    p = torch.softmax(logits.detach(), -1)
    expected = (q * (q.log() - p.log())).sum(-1).mean()
    assert torch.allclose(kl, expected, atol=1e-6)
    kl.backward()
    # d/dlogits KL(q || softmax(logits)) = softmax(logits) - q, averaged over rows.
    assert torch.allclose(logits.grad, (p - q) / 2, atol=1e-6)
    assert float(prior_kl(base.clone(), base)) == pytest.approx(0.0, abs=1e-7)


def test_prior_kl_survives_masked_options_without_nan():
    torch = pytest.importorskip("torch")
    from qd_train.replay import prior_kl

    ninf = float("-inf")
    logits = torch.tensor([[1.0, 0.0, ninf]], requires_grad=True)
    base = torch.tensor([[0.0, 1.0, ninf]])
    kl = prior_kl(logits, base)
    kl.backward()
    assert torch.isfinite(kl) and torch.isfinite(logits.grad).all()


def test_prior_kl_computes_in_float32_even_for_bf16_logits():
    torch = pytest.importorskip("torch")
    from qd_train.replay import prior_kl

    logits = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
    assert prior_kl(logits, torch.tensor([[0.0, 1.0]])).dtype == torch.float32


# --- the cache and the wrapper ------------------------------------------------------------


class _TinyStep:
    """Enough of a SpanScoringStep to drive the wrapper: embedding, one linear head."""

    def __init__(self, seed: int = 0, vocab: int = 12, hidden: int = 8):
        import torch

        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)
        self.opt = torch.optim.SGD([*self.embed.parameters(), *self.head.parameters()], lr=0.1)
        self.letter_log: list[float] = []
        self.span_log: list[float] = []
        self.loaded: dict | None = None

    def hidden(self, batch: Batch):
        import torch

        return self.embed(torch.as_tensor(batch.tokens.astype(np.int64)))

    def lm_head(self, h):
        return self.head(h)

    def accumulate(self, batch, supervision) -> float:
        self.letter_log.append(0.0)
        return 0.0

    def accumulate_span(self, batch, supervision) -> float:
        return self.accumulate(batch, supervision)

    def apply(self, *, lr: float) -> None:
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def state(self):
        return {"micro_batches": len(self.letter_log)}

    def load_state(self, state):
        self.loaded = dict(state)


LETTERS = (3, 4, 5)


def _replay_batch(index: int) -> Batch:
    tokens = np.asarray([[1, 2, 6, 3], [1, 7, 6, 4], [1, 8, 6, 9]], dtype=np.int32)
    return Batch(
        tokens=tokens, lengths=np.asarray([4, 4, 4]), bucket=4, index=index,
        slot_kind=np.asarray([SLOT_CHOICE, SLOT_CHOICE, SLOT_SPAN]),
        target_index=np.asarray([2, 2, 2]),
        span_target=np.asarray([(NO_SPAN, NO_SPAN), (NO_SPAN, NO_SPAN), (0, 0)]),
        line_starts=np.asarray([[False] * 4, [False] * 4, [True, False, False, False]]),
    )


def test_the_cache_is_the_base_on_letter_rows_only_and_refuses_another_base(tmp_path):
    pytest.importorskip("torch")
    from qd_train.replay import PriorCache

    step = _TinyStep()
    batches = [_replay_batch(0), _replay_batch(1)]
    cache = PriorCache.build(step, batches, letter_ids=LETTERS, key={"base": "tiny-0"})
    assert cache.logits.shape == (4, 3)  # two letter rows per batch; the span row is out
    assert cache.row.tolist() == [0, 1, 0, 1]
    path = tmp_path / "prior.npz"
    cache.save(path)
    with pytest.raises(ReplayRefusal, match="already exists"):
        cache.save(path)
    loaded = PriorCache.load(path, expect_key=cache.key)
    assert np.array_equal(loaded.logits, cache.logits)
    with pytest.raises(ReplayRefusal, match="different base"):
        PriorCache.load(path, expect_key={**cache.key, "base": "tiny-1"})
    assert step.embed.training, "build left the step's modules in eval mode"


def test_the_wrapper_replays_every_nth_micro_batch_and_pulls_toward_the_base(tmp_path):
    torch = pytest.importorskip("torch")
    from qd_train.replay import PriorCache, PriorKLReplay, _letter_logits, prior_kl

    step = _TinyStep()
    batches = [_replay_batch(0)]
    cache = PriorCache.build(step, batches, letter_ids=LETTERS, key={"base": "tiny"})
    # Move the model away from its base, then let replay alone pull it back.
    with torch.no_grad():
        step.head.weight.add_(torch.randn_like(step.head.weight) * 3.0)
    rows = np.asarray([0, 1])

    def kl_now() -> float:
        with torch.no_grad():
            lg = _letter_logits(step, batches[0], rows, LETTERS)
            return float(prior_kl(lg, torch.as_tensor(cache.logits)))

    wrapped = PriorKLReplay(step, batches=batches, cache=cache, weight=1.0, every=2)
    before = kl_now()
    for _ in range(20):
        wrapped.accumulate(batches[0], None)
        wrapped.apply(lr=1.0)
    assert wrapped.replayed == 10 and wrapped.micro_batches == 20
    assert kl_now() < before * 0.5, (before, kl_now())
    assert step.letter_log == [0.0] * 20, "the replay term leaked into the letter log"
    state = wrapped.state()
    assert state["replay"]["replayed"] == 10
    fresh = PriorKLReplay(_TinyStep(), batches=batches, cache=cache, weight=1.0, every=2)
    fresh.load_state(state)
    assert fresh.micro_batches == 20 and fresh.replayed == 10
    assert "replay" not in fresh.inner.loaded
    other = PriorKLReplay(_TinyStep(), batches=batches, cache=cache, weight=0.5, every=2)
    with pytest.raises(ReplayRefusal, match="replay configuration"):
        other.load_state(state)
    with pytest.raises(ReplayRefusal, match="taken without replay"):
        other.load_state({"micro_batches": 3})


def test_the_wrapper_refuses_a_cache_from_another_plan():
    pytest.importorskip("torch")
    from qd_train.replay import PriorCache, PriorKLReplay

    step = _TinyStep()
    cache = PriorCache.build(step, [_replay_batch(0)], letter_ids=LETTERS, key={"base": "t"})
    with pytest.raises(ReplayRefusal, match="different replay plan"):
        PriorKLReplay(step, batches=[_replay_batch(5)], cache=cache, weight=1.0, every=1)
