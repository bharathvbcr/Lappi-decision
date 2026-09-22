"""S2's weight surgery and parity gate. Skips without torch; never passes vacuously.

torch is the optional `mac` extra and is not installed in the repo venv, so every test here
is behind `importorskip`. A skipped test reports as skipped, which is not the same as passed —
that distinction is the whole point of `qd_train.tristate` and it applies to the suite too.

The models are stubs rather than a real Qwen3.5 checkpoint: the surgery binds to the four
`PreTrainedModel` accessors (`get`/`set_input_embeddings`, `get`/`set_output_embeddings`) and
nothing else, so a stub exercises exactly the surface that is used. What a stub cannot check
is that a real checkpoint routes through those accessors; that is recorded as
GAP-S2-NO-REAL-CHECKPOINT-PARITY.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is the optional `mac` extra")
nn = torch.nn

from qd_train.artifacts import DROPPED, RemapTable  # noqa: E402
from qd_train.remap import (  # noqa: E402
    LOSS_PARITY_TOL,
    apply_remap_to_model,
    build_remap,
    remap_parity_report,
    verify_remap_parity,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

V_OLD = 96
HIDDEN = 12
TOK_HASH = "t" * 64


class _Config:
    """Just enough of a `PretrainedConfig`: the two attributes the surgery reads and writes."""

    def __init__(self, vocab_size: int, tie_word_embeddings: bool | None) -> None:
        self.vocab_size = vocab_size
        if tie_word_embeddings is not None:
            self.tie_word_embeddings = tie_word_embeddings


class TinyLM(nn.Module):
    """A causal LM small enough to test exactly, wired through the accessors the surgery uses."""

    def __init__(
        self,
        *,
        vocab: int = V_OLD,
        hidden: int = HIDDEN,
        tied: bool,
        config_flag: bool | None,
        seed: int = 20260919,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = _Config(vocab, config_flag)
        self.embed = nn.Embedding(vocab, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        if tied:
            self.head.weight = self.embed.weight
        self.to(torch.float32)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed = value

    def get_output_embeddings(self) -> nn.Module | None:
        return self.head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.head = new_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.mix(self.embed(input_ids))))


class HeadlessLM(TinyLM):
    """A model with no output head at all — a third state beside tied and untied."""

    def get_output_embeddings(self) -> nn.Module | None:
        return None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mix(self.embed(input_ids)))


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _remap_over(used: list[int], *, specials: tuple[int, ...] = (0, 1)) -> RemapTable:
    counts = np.zeros(V_OLD, dtype=np.int64)
    counts[used] = 1
    return build_remap(
        counts=counts,
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=specials,
        target_vocab_size=None,
    )


# --- the tie ------------------------------------------------------------------------------


def test_a_tied_model_stays_tied_and_the_slice_is_one_tensor():
    model = TinyLM(tied=True, config_flag=True)
    assert _shares_storage(model.head.weight, model.embed.weight), "fixture is not actually tied"
    remap = _remap_over([5, 9, 40])

    app = apply_remap_to_model(model, remap)

    assert app.tie.storage_shared is True
    assert app.tie.config_flag is True
    assert app.tie.agreed is True
    assert app.tie.believed == "storage identity (data_ptr)"
    assert app.tie.config_flag_rewritten is False
    assert _shares_storage(model.head.weight, model.embed.weight)
    assert model.head.weight is model.embed.weight
    assert model.embed.weight.shape == (remap.vocab_size, HIDDEN)
    assert model.config.vocab_size == remap.vocab_size
    # One tensor's worth freed, not two.
    assert app.bytes_freed == (V_OLD - remap.vocab_size) * HIDDEN * 4


def test_an_untied_model_stays_untied_and_both_tensors_are_sliced():
    model = TinyLM(tied=False, config_flag=False)
    assert not _shares_storage(model.head.weight, model.embed.weight)
    remap = _remap_over([5, 9, 40])
    before_head = model.head.weight.detach().clone()

    app = apply_remap_to_model(model, remap)

    assert app.tie.storage_shared is False
    assert app.tie.agreed is True
    assert not _shares_storage(model.head.weight, model.embed.weight)
    assert app.bytes_freed == (V_OLD - remap.vocab_size) * HIDDEN * 4 * 2
    # The head rows are the ones new_to_old names, from the head — not from the embedding.
    expected = before_head[torch.as_tensor(np.asarray(remap.new_to_old, dtype=np.int64))]
    assert torch.equal(model.head.weight.detach(), expected)


def test_a_config_flag_that_lies_tied_loses_to_storage_and_is_rewritten():
    """The flag says tied; the tensors are not. Believing the flag would invent a tie."""
    model = TinyLM(tied=False, config_flag=True)
    remap = _remap_over([5, 9, 40])

    app = apply_remap_to_model(model, remap)

    assert app.tie.config_flag is True
    assert app.tie.storage_shared is False
    assert app.tie.agreed is False
    assert "DISAGREEMENT" in app.tie.detail
    assert app.tie.config_flag_rewritten is True
    assert not _shares_storage(model.head.weight, model.embed.weight)
    # The landmine closed: a later tie_weights() would otherwise clobber one head.
    assert model.config.tie_word_embeddings is False


def test_a_config_flag_that_lies_untied_still_preserves_the_real_tie():
    """The flag says untied; the tensors are one. Believing the flag would double the memory."""
    model = TinyLM(tied=True, config_flag=False)
    remap = _remap_over([5, 9, 40])

    app = apply_remap_to_model(model, remap)

    assert app.tie.config_flag is False
    assert app.tie.storage_shared is True
    assert app.tie.agreed is False
    assert app.tie.config_flag_rewritten is True
    assert _shares_storage(model.head.weight, model.embed.weight)
    assert model.config.tie_word_embeddings is True


def test_a_config_with_no_tie_flag_is_a_third_state_not_false():
    model = TinyLM(tied=True, config_flag=None)
    app = apply_remap_to_model(model, _remap_over([5, 9]))
    assert app.tie.config_flag is None
    assert app.tie.storage_shared is True
    assert app.tie.agreed is False, "absent is not the same as agreeing"
    assert "no tie_word_embeddings" in app.tie.detail
    assert not hasattr(model.config, "tie_word_embeddings")


def test_a_model_with_no_output_head_slices_only_the_embedding():
    model = HeadlessLM(tied=False, config_flag=None)
    app = apply_remap_to_model(model, _remap_over([5, 9]))
    assert app.output_head_present is False
    assert app.tie.storage_shared is False
    assert "no output embedding" in app.tie.detail
    assert model.embed.weight.shape[0] == app.new_vocab_size


def test_the_tie_is_a_real_shared_allocation_not_two_equal_tensors():
    model = TinyLM(tied=True, config_flag=True)
    apply_remap_to_model(model, _remap_over([5, 9, 40]))
    with torch.no_grad():
        model.embed.weight[0, 0] = 12345.0
    assert model.head.weight[0, 0].item() == 12345.0


# --- refusals in the surgery ----------------------------------------------------------------


def test_a_remap_for_a_different_vocabulary_is_refused():
    model = TinyLM(tied=True, config_flag=True)
    counts = np.zeros(V_OLD + 7, dtype=np.int64)
    counts[[1, 2]] = 1
    wrong = build_remap(
        counts=counts,
        source_vocab_size=V_OLD + 7,
        tokenizer_hash=TOK_HASH,
        special_ids=(0,),
        target_vocab_size=None,
    )
    with pytest.raises(ValueError, match="wrong tokenizer"):
        apply_remap_to_model(model, wrong)


def test_a_text_config_vocab_size_is_updated_too():
    """Qwen3.5 carries the size on `config.text_config`; a stale copy there re-sizes the head."""
    model = TinyLM(tied=True, config_flag=True)
    model.config.text_config = _Config(V_OLD, None)
    remap = _remap_over([5, 9, 40])
    apply_remap_to_model(model, remap)
    assert model.config.vocab_size == remap.vocab_size
    assert model.config.text_config.vocab_size == remap.vocab_size


# --- the parity gate ------------------------------------------------------------------------


def _parity_corpus(remap: RemapTable, n: int = 6, length: int = 9) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    kept = np.asarray(remap.new_to_old, dtype=np.int64)
    return [rng.choice(kept, size=length) for _ in range(n)]


def test_parity_passes_on_a_correct_remap_and_states_its_coverage():
    remap = _remap_over([3, 5, 9, 17, 40, 55])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    assert _shares_storage(after.head.weight, after.embed.weight), "deepcopy lost the tie"
    apply_remap_to_model(after, remap)

    seqs = _parity_corpus(remap)
    out = verify_remap_parity(
        model_before=before, model_after=after, remap=remap, sequences=seqs
    )
    assert isinstance(out, Ran)
    assert out.passed, out.detail
    assert out.n == len(seqs) and out.n_total == len(seqs)
    assert out.is_complete_coverage
    assert out.value is not None and out.value <= LOSS_PARITY_TOL


def test_the_full_vocabulary_loss_is_not_the_restricted_loss_and_the_report_says_so():
    """The gate sentence read literally is unsatisfiable: dropping rows shrinks the softmax
    denominator. The report separates the two so the difference cannot be laundered away."""
    remap = _remap_over([3, 5, 9, 17, 40, 55])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    report = remap_parity_report(
        model_before=before, model_after=after, remap=remap, sequences=_parity_corpus(remap)
    )
    assert report.as_tristate().passed
    assert report.max_abs_loss_delta <= LOSS_PARITY_TOL
    # ~88 of 96 rows left the denominator: the full-vocabulary loss is higher by orders of
    # magnitude more than the gate's tolerance, and that is the remap working, not failing.
    assert report.full_vocab_loss_delta > 1.0
    assert "intended effect" in report.as_tristate().detail
    blob = report.to_json()
    assert blob["full_vocab_loss_delta"] == report.full_vocab_loss_delta
    assert blob["mean_full_vocab_loss"] > blob["mean_restricted_loss"]


def test_parity_respects_a_non_ascending_new_to_old():
    """The contract permits any permutation; the surgery must gather in the table's order,
    not re-derive a sorted one."""
    kept = [7, 2, 40, 0, 1]
    old_to_new = np.full(V_OLD, DROPPED, dtype=np.int32)
    for new_id, old_id in enumerate(kept):
        old_to_new[old_id] = new_id
    remap = RemapTable(
        old_to_new=old_to_new,
        new_to_old=np.array(kept, dtype=np.int32),
        tokenizer_hash=TOK_HASH,
        special_ids=(0, 1),
    )
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    out = verify_remap_parity(
        model_before=before, model_after=after, remap=remap, sequences=_parity_corpus(remap)
    )
    assert isinstance(out, Ran) and out.passed, out.detail


def test_parity_fails_loudly_when_the_head_was_sliced_wrong():
    """The gate's reason for existing: a wrong slice must not read as a pass."""
    remap = _remap_over([3, 5, 9, 17, 40, 55])
    before = TinyLM(tied=False, config_flag=False)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)
    with torch.no_grad():
        after.head.weight[2] += 3.0

    out = verify_remap_parity(
        model_before=before, model_after=after, remap=remap, sequences=_parity_corpus(remap)
    )
    assert isinstance(out, Ran)
    assert not out.passed
    assert "logit_exactness" in out.detail or "loss_parity" in out.detail


def test_a_sequence_the_remap_drops_is_a_measured_failure_not_a_crash():
    remap = _remap_over([3, 5])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    out = verify_remap_parity(
        model_before=before,
        model_after=after,
        remap=remap,
        sequences=[np.array([0, 3, 5]), np.array([0, 3, 62])],
    )
    assert isinstance(out, Ran)
    assert not out.passed
    assert "62" in out.detail and "dropped" in out.detail


def test_an_over_long_sequence_is_refused_rather_than_truncated():
    remap = _remap_over([3, 5, 9])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    out = verify_remap_parity(
        model_before=before,
        model_after=after,
        remap=remap,
        sequences=[np.array([0, 3, 5, 9, 3])],
        max_seq_len=3,
    )
    assert isinstance(out, Ran) and not out.passed
    assert "refused rather than truncated" in out.detail


def test_a_sequence_whose_logits_would_not_fit_is_refused_with_the_arithmetic():
    """The parity check cannot use the fused CE path — it exists to compare the two heads'
    logits — so the full [1, T, 248320] tensor is unavoidable and has to be bounded."""
    remap = _remap_over([3, 5, 9])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    out = verify_remap_parity(
        model_before=before,
        model_after=after,
        remap=remap,
        sequences=[np.array([0, 3, 5, 9, 3])],
        max_logit_bytes=16,
    )
    assert isinstance(out, Ran) and not out.passed
    assert "over the 16-byte budget" in out.detail
    assert "shorter verification sequences" in out.detail


def test_the_logit_budget_does_not_reject_an_ordinary_sequence():
    remap = _remap_over([3, 5, 9, 17])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)
    out = verify_remap_parity(
        model_before=before, model_after=after, remap=remap, sequences=_parity_corpus(remap)
    )
    assert isinstance(out, Ran) and out.passed, out.detail


def test_a_capped_run_carries_both_numbers():
    remap = _remap_over([3, 5, 9, 17])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    out = verify_remap_parity(
        model_before=before,
        model_after=after,
        remap=remap,
        sequences=_parity_corpus(remap, n=7),
        max_sequences=3,
    )
    assert isinstance(out, Ran) and out.passed
    assert out.n == 3 and out.n_total == 7
    assert not out.is_complete_coverage, "a capped sample must never read as full coverage"


def test_zero_sequences_is_not_run_not_a_pass():
    remap = _remap_over([3, 5])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    out = verify_remap_parity(
        model_before=before, model_after=after, remap=remap, sequences=[]
    )
    assert isinstance(out, NotRun)
    assert not hasattr(out, "passed")


def test_a_one_token_sequence_cannot_be_scored_and_says_so():
    remap = _remap_over([3, 5])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    out = verify_remap_parity(
        model_before=before, model_after=after, remap=remap, sequences=[np.array([3])]
    )
    assert isinstance(out, Ran) and not out.passed
    assert "at least two positions" in out.detail


# --- memory recording -------------------------------------------------------------------------


def test_memory_is_recorded_per_sequence_and_never_fabricated():
    remap = _remap_over([3, 5, 9, 17, 40])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    report = remap_parity_report(
        model_before=before, model_after=after, remap=remap, sequences=_parity_corpus(remap)
    )
    for record in (report.memory_before, report.memory_after):
        assert record is not None
        assert record.n_sequences == 6
        peak = record.peak_bytes
        # The invariant that matters: a point reading is never reported as a high-water mark.
        # `peak_bytes` may be `Ran` only where the method genuinely is one.
        if "max_memory_allocated" in record.method:
            assert isinstance(peak, Ran) and isinstance(peak.value, int) and peak.value >= 0
        else:
            assert isinstance(peak, NotRun), (
                f"method {record.method!r} is not a high-water API but reported a peak"
            )
            # The honest outcome on such a device: a reason, not a zero.
            assert peak.reason and not hasattr(peak, "passed")
    # Whatever the device, the gate's detail states the memory outcome explicitly, so a pass
    # can never be read as "memory was recorded".
    detail = report.as_tristate().detail
    assert "pre_remap_peak_bytes_per_sequence" in detail
    assert "post_remap_peak_bytes_per_sequence" in detail


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device on this host")
def test_on_mps_the_point_reading_is_carried_but_never_called_a_peak():
    """The branch this Mac actually has. torch 2.12.1 has no MPS high-water API, so the peak
    is NotRun and the point reading lives under a different name."""
    remap = _remap_over([3, 5, 9, 17, 40])
    before = TinyLM(tied=True, config_flag=True).to("mps")
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    report = remap_parity_report(
        model_before=before, model_after=after, remap=remap, sequences=_parity_corpus(remap)
    )
    assert report.as_tristate().passed, report.as_tristate().detail
    record = report.memory_after
    assert record is not None and record.device.startswith("mps")
    assert isinstance(record.peak_bytes, NotRun)
    assert "no peak-memory API for MPS" in record.peak_bytes.reason
    assert isinstance(record.allocated_delta_bytes, Ran)
    assert isinstance(record.allocated_delta_bytes.value, int)


def test_the_report_serializes_whole_for_a_ledger_row():
    remap = _remap_over([3, 5, 9])
    before = TinyLM(tied=True, config_flag=True)
    after = copy.deepcopy(before)
    app = apply_remap_to_model(after, remap)

    report = remap_parity_report(
        model_before=before, model_after=after, remap=remap, sequences=_parity_corpus(remap, n=2)
    )
    import json

    blob = {"application": app.to_json(), "parity": report.to_json()}
    round_tripped = json.loads(json.dumps(blob))
    assert round_tripped["parity"]["gate"]["state"] == "ran"
    assert round_tripped["application"]["tie"]["believed"] == "storage identity (data_ptr)"
    assert round_tripped["parity"]["memory_before"]["n_sequences"] == 2


# --- a padded embedding is not a wrong tokenizer -------------------------------------
#
# Found on the rented GH200, 2026-09-20: Qwen3.5-2B-Base's embedding is 248,320 rows
# (1940 x 128, aligned for tensor cores) over a 248,077-token tokenizer -- 248,044 base
# plus 33 added. `apply_remap_to_model` compared the two with `!=` and refused, reporting
# "Applying it would renumber every row against the wrong tokenizer" about a checkpoint
# whose tokenizer was exactly right. The three tests below pin the direction.


def test_an_embedding_padded_above_the_tokenizer_vocabulary_is_accepted() -> None:
    """The real case. Pre-fix this raised ValueError and the GH200 run stopped here."""
    padded = V_OLD + 32  # the shape of 248,320 over 248,077: aligned, unreachable rows
    remap = _remap_over([7, 2, 40])
    model = TinyLM(vocab=padded, tied=True, config_flag=True)
    original = model.get_input_embeddings().weight.detach().clone()

    apply_remap_to_model(model, remap)

    new_weight = model.get_input_embeddings().weight.detach()
    assert new_weight.shape == (remap.vocab_size, HIDDEN)
    # Every kept row is the row it names, taken from the UNPADDED region. The order is
    # `build_remap`'s, read off the table rather than assumed.
    for new_id, old_id in enumerate(remap.new_to_old):
        assert torch.equal(new_weight[new_id], original[int(old_id)]), (
            f"new row {new_id} should be old row {int(old_id)}; padding must not shift it"
        )
    assert model.get_output_embeddings().weight.data_ptr() == new_weight.data_ptr(), (
        "the tie must survive the slice"
    )


def test_the_parity_gate_runs_on_an_embedding_padded_above_the_tokenizer() -> None:
    """The same padding, one function over. The surgery was fixed to accept it on
    2026-09-20; the parity report kept `!=` and refused every sequence of a padded model.
    Found 2026-09-22 by running the gate against the real Qwen3.5-2B tower for the first
    time: "model_before produced 248320 logits, expected 248077", on 3 of 3 sequences."""
    remap = _remap_over([3, 5, 9, 17, 40, 55])
    before = TinyLM(vocab=V_OLD + 32, tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)

    seqs = _parity_corpus(remap)
    out = verify_remap_parity(
        model_before=before, model_after=after, remap=remap, sequences=seqs
    )
    assert isinstance(out, Ran)
    assert out.passed, out.detail
    assert out.n == len(seqs) and out.n_total == len(seqs)
    assert out.value is not None and out.value <= LOSS_PARITY_TOL


def test_the_full_vocabulary_loss_of_a_padded_head_includes_its_padding_rows() -> None:
    """The full-vocabulary number is the loss the unremapped model computes, and that
    softmax runs over every row the head scores. Dropping the padding rows from it would
    quietly measure a model nobody trains."""
    remap = _remap_over([3, 5, 9, 17, 40, 55])
    before = TinyLM(vocab=V_OLD + 32, tied=True, config_flag=True)
    after = copy.deepcopy(before)
    apply_remap_to_model(after, remap)
    seqs = _parity_corpus(remap)

    report = remap_parity_report(
        model_before=before, model_after=after, remap=remap, sequences=seqs
    )
    assert report.mean_full_vocab_loss is not None
    with torch.no_grad():
        expected = []
        for seq in seqs:
            x = torch.as_tensor(seq.astype(np.int64))[None, :]
            logits = before(x).float()  # TinyLM returns the logits tensor itself
            expected.append(
                float(
                    torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, V_OLD + 32), x[:, 1:].reshape(-1)
                    )
                )
            )
    assert report.mean_full_vocab_loss == pytest.approx(float(np.mean(expected)), rel=1e-6)


def test_an_embedding_smaller_than_the_remaps_vocabulary_is_still_refused() -> None:
    """The direction that IS fatal: kept ids would index past the end of the weight."""
    remap = _remap_over([7, 2, 40])
    model = TinyLM(vocab=V_OLD - 32, tied=True, config_flag=True)
    with pytest.raises(ValueError, match="index past the end of the embedding"):
        apply_remap_to_model(model, remap)


def test_a_remap_keeping_an_id_outside_the_embedding_is_refused_by_range_not_by_size() -> None:
    """The check that the size comparison alone cannot make.

    A remap whose `new_to_old` runs past its own declared `source_vocab_size` passes every
    size comparison whenever that declared size matches the embedding. Only looking at the
    ids catches it.
    """
    remap = _remap_over([7, 2, 40])
    remap.new_to_old[-1] = V_OLD + 500  # past the declared source size, size check unmoved
    model = TinyLM(vocab=V_OLD, tied=True, config_flag=True)
    with pytest.raises(ValueError, match="outside the model's"):
        apply_remap_to_model(model, remap)
