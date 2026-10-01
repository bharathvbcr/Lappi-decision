"""The ``--needle`` worker decodes the suite its scoring process handed it, and rebuilds nothing.

J7g on the GH200 (2026-10-01): the scoring process spent ~5.5 minutes of single-core prelude
-- the corpus rebuild, both relabels, the second pass, the needle and OOD suites -- and then
started the needle worker, which spent the same minutes rebuilding all of it from the same
argv, with the GPU at 0%, to decode a suite the parent already held. The worker is a fresh
process for a reason that stands (MPS keeps a compiled graph per input shape that only a
process exit returns); what it rebuilt is now handed to it as one ``.npz``.

These hold: the handoff round-trips every array the decode reads; a handoff that does not
re-derive (another run's shard sets, another plan, another seed, a changed array, an unknown
format) is refused before anything is decoded; and the worker's ``main`` never reaches the
rebuild.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import real_ft_run as rft  # noqa: E402

from qd_train.artifacts import SLOT_SPAN, SPAN_ABSTAIN  # noqa: E402
from qd_train.needle import build_suite  # noqa: E402
from qd_train.shards import assemble_batch  # noqa: E402

SEED = 20260919


def _suite(seed: int = SEED) -> rft.NeedleSuite:
    """A small suite with real cases and span batches built the way prepare_needle builds them:
    one row per case through ``assemble_batch``, then ``_repad`` to one width."""
    cases = build_suite(target_tokens=1024, cases_per_depth=1, seed=3)
    rng = np.random.default_rng(0)
    raw = []
    for i in range(len(cases)):
        n = 40 + 7 * i
        ids = rng.integers(10, 5000, size=n).astype(np.int32)
        cands = tuple(sorted({0, *rng.choice(np.arange(1, n - 2), size=5, replace=False)}))
        span = [cands[1], cands[3]] if i % 2 else [SPAN_ABSTAIN, SPAN_ABSTAIN]
        raw.append(assemble_batch(
            [ids], kinds=np.asarray([SLOT_SPAN]), target_index=np.asarray([n - 2]),
            spans=np.asarray([span]), candidates=[cands], width=n, bucket=n, index=i,
        ))
    width = -(-max(int(b.lengths[0]) for b in raw) // rft.NEEDLE_WIDTH_MULTIPLE)
    width *= rft.NEEDLE_WIDTH_MULTIPLE
    labels_for = {
        i: [rft.Label(row_id=c.case_id, family_id="code.defect_class", slot_name="defect_span",
                      slot_kind=SLOT_SPAN, gold_letter="Z", letters=("Z",),
                      language=c.language)]
        for i, c in enumerate(cases)
    }
    return rft.NeedleSuite(
        cases, [rft._repad(b, width) for b in raw], labels_for,
        [int(b.lengths[0]) for b in raw], seed=seed,
        digest=rft.needle_suite_digest(cases, raw),
    )


def _reader(shard_hash: str):
    return SimpleNamespace(header=SimpleNamespace(shard_hash=lambda: shard_hash))


def _plan(widths=(135, 135, 1105)):
    return [SimpleNamespace(tokens=np.zeros((2, w), dtype=np.int32)) for w in widths]


def _write(tmp_path: Path, suite: rft.NeedleSuite) -> Path:
    path = tmp_path / "needle-handoff.npz"
    rft.write_needle_handoff(
        path, suite, letter_id={"Z": 57, "A": 32}, eval_widths=[1152, 300, 301],
        train=_reader("t" * 64),  # type: ignore[arg-type]
        val=SimpleNamespace(reader=_reader("v" * 64), plan=_plan()),  # type: ignore[arg-type]
    )
    return path


def _read(path: Path, *, train="t" * 64, val="v" * 64, plan=None, seed=SEED):
    return rft.read_needle_handoff(
        path, train=_reader(train),  # type: ignore[arg-type]
        val=rft.ValPlan(_reader(val), _plan() if plan is None else plan),  # type: ignore[arg-type]
        seed=seed,
    )


def test_every_array_the_decode_reads_round_trips(tmp_path):
    suite = _suite()
    got = _read(_write(tmp_path, suite))
    assert got.letter_id == {"Z": 57, "A": 32} and list(got.letter_id) == ["Z", "A"]
    assert got.eval_widths == [1152, 300, 301]
    back = got.suite
    assert back.cases == suite.cases and back.token_lengths == suite.token_lengths
    assert back.labels_for == suite.labels_for and back.seed == suite.seed
    assert back.digest == suite.digest and back.not_run is None
    assert rft.batches_digest(back.batches) == rft.batches_digest(suite.batches)
    for a, b in zip(back.batches, suite.batches, strict=True):
        for name in rft._BATCH_ARRAYS:
            x, y = getattr(a, name), getattr(b, name)
            assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), name
        assert (a.bucket, a.index) == (b.bucket, b.index)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"train": "x" * 64}, "train shard hash"),
        ({"val": "x" * 64}, "val shard hash"),
        ({"plan": _plan((135, 1105, 135))}, "val plan shapes"),
        ({"seed": 0}, "suite seed"),
    ],
)
def test_a_handoff_for_another_run_is_refused(tmp_path, kwargs, match):
    with pytest.raises(SystemExit, match=f"written for another run: .*{match}"):
        _read(_write(tmp_path, _suite()), **kwargs)


def _rewrite(path: Path, edit) -> None:
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    meta = json.loads(arrays["meta"].tobytes().decode("utf-8"))
    edit(arrays, meta)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    path.unlink()
    with path.open("xb") as fh:
        np.savez(fh, **arrays)


def _add_a_candidate(arrays, meta):
    """One more line start inside case 0's real tokens: a mask Batch accepts, but not the
    one the scoring process built."""
    mask = arrays["line_starts"][0]
    free = np.flatnonzero(~mask[: int(arrays["lengths"][0])])
    mask[int(free[0])] = True


def _drop_the_gold_candidate(arrays, meta):
    """Case 1 points; clear its gold start's line start: a mask Batch itself refuses."""
    arrays["line_starts"][1, int(arrays["span_target"][1, 0])] = False


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda a, m: a["tokens"].__setitem__((0, 3), a["tokens"][0, 3] + 1), "suite digest"),
        (_add_a_candidate, "batches digest"),
        (_drop_the_gold_candidate, "a case is not a valid batch"),
        (lambda a, m: a.__setitem__("span_target", a["span_target"].astype(np.int32)),
         "batches digest"),
        (lambda a, m: m.__setitem__("indices", m["indices"][::-1]), "batches digest"),
        (lambda a, m: m.__setitem__("format", "qd-needle-handoff/0"), "format"),
        (lambda a, m: m.__setitem__("labels", m["labels"][1:]), "do not count"),
        (lambda a, m: a.__setitem__("extra", np.zeros(1)), "not one row per case"),
    ],
)
def test_a_handoff_that_does_not_read_back_as_written_is_refused(tmp_path, edit, match):
    path = _write(tmp_path, _suite())
    _rewrite(path, edit)
    with pytest.raises(SystemExit, match=match):
        _read(path)


def test_only_a_built_suite_is_handed_over(tmp_path):
    with pytest.raises(SystemExit, match="--needle was not given"):
        _write(tmp_path, rft.NeedleSuite([], [], {}, [], not_run="--needle was not given"))
    suite = _suite()
    ragged = rft.NeedleSuite(
        suite.cases, [suite.batches[0], *(rft._repad(b, 2048) for b in suite.batches[1:])],
        suite.labels_for, suite.token_lengths, seed=suite.seed, digest=suite.digest,
    )
    with pytest.raises(SystemExit, match="one padded width"):
        _write(tmp_path, ragged)


def test_the_worker_decodes_the_handoff_and_rebuilds_nothing(tmp_path, monkeypatch):
    """main() as the worker: the train reader, the val plan and the handoff, then the decode.
    Everything the scoring process's prelude runs is booby-trapped."""
    suite = _suite()
    handoff = _write(tmp_path, suite)

    def never(name):
        def trap(*a, **k):
            raise AssertionError(f"the needle worker ran {name}")
        return trap

    for name in ("ft_split_rows", "ft_splits", "_labels", "relabel_train", "open_val_set",
                 "prepare_second_pass", "prepare_needle", "prepare_ood", "corpus_facts",
                 "check_defect_source", "_batch_inventory", "_inventory", "_contradictions"):
        monkeypatch.setattr(rft, name, never(name))
    monkeypatch.setattr(rft, "resolve_rev", lambda repo, rev: rev)
    monkeypatch.setattr(rft, "ShardReader", lambda *a, **k: _reader("t" * 64))
    monkeypatch.setattr(rft, "open_val_reader", lambda *a, **k: _reader("v" * 64))
    monkeypatch.setattr(rft, "val_plan", lambda reader, config: _plan())
    seen: dict[str, object] = {}

    def checkpoint_step(args, *, reader, val, device, eval_widths, suite_seed):
        seen.update(device=device, eval_widths=list(eval_widths), suite_seed=suite_seed,
                    widths=[int(b.tokens.shape[1]) for b in val.plan])
        return "step", {}, {}, 0, {}

    def predictions(step, decoded_suite, letter_id, *, logits):
        # Told --suite-logits, which this argv does not pass (test_real_ft_suite_logits
        # runs the worker with it on).
        assert logits is False
        assert step == "step" and decoded_suite.digest == suite.digest
        assert rft.batches_digest(decoded_suite.batches) == rft.batches_digest(suite.batches)
        seen["letter_id"] = dict(letter_id)
        return rft.NeedleDecoded({c.case_id: None for c in decoded_suite.cases}, ())

    monkeypatch.setattr(rft, "_checkpoint_step", checkpoint_step)
    monkeypatch.setattr(rft, "needle_predictions", predictions)
    out = tmp_path / "predictions.json"
    backbone = tmp_path / "snapshot"
    backbone.mkdir()
    code = rft.main([
        "--out", str(tmp_path), "--rev", "0" * 40, "--no-repo-history",
        "--score-checkpoint", str(tmp_path / "epoch-seed0-cpu.json"),
        "--ft-ledger", str(tmp_path / "ft.jsonl"), "--ft-row-id", "abcdefgh", "--seeds", "0",
        "--real-backbone", str(backbone), "--score-val", "--needle", "--devices", "cpu",
        "--needle-handoff", str(handoff), "--needle-predictions-out", str(out),
    ])
    assert code == 0
    assert seen == {
        "device": "cpu", "eval_widths": [1152, 300, 301], "suite_seed": SEED,
        "widths": [135, 135, 1105], "letter_id": {"Z": 57, "A": 32},
    }
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["digest"] == suite.digest
    assert set(payload["predictions"]) == {c.case_id for c in suite.cases}


def test_the_worker_flags_come_together(tmp_path):
    base = ["--out", str(tmp_path), "--rev", "0" * 40, "--no-repo-history",
            "--score-checkpoint", str(tmp_path / "epoch-seed0-cpu.json"),
            "--ft-ledger", str(tmp_path / "ft.jsonl"), "--ft-row-id", "abcdefgh", "--seeds", "0",
            "--real-backbone", str(tmp_path), "--score-val", "--needle", "--devices", "cpu"]
    with pytest.raises(SystemExit, match="given together"):
        rft.main([*base, "--needle-predictions-out", str(tmp_path / "p.json")])
    with pytest.raises(SystemExit, match="given together"):
        rft.main([*base, "--needle-handoff", str(tmp_path / "h.npz")])
