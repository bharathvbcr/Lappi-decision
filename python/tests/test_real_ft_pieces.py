"""``tools/real_ft_run.py``: the ported-piece flags, the recipe keys they add, the row
rebuild ``ft_split_rows`` owns, and ``--verdicts-out``.

Torch-gated, like ``test_real_ft_protocol.py``: the tool raises ``SystemExit`` at import
without torch, so this module skips in the core venv rather than passing vacuously.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402

# --- the recipe: nothing added when every piece is off ----------------------------------------


def test_a_run_with_every_piece_off_adds_no_recipe_key():
    """So its recipe_hash is exactly what the rows before the pieces existed hashed to."""
    assert rft._recipe_pieces(
        lower_layers_n=0, lower_lr_scale=1.0, beta2=rft.DEFAULT_BETA2,
        permutation=None, replay=None,
    ) == {}


def test_fused_adamw_lands_in_the_recipe_only_when_on():
    pieces = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
              "permutation": None, "replay": None}
    assert rft._recipe_pieces(**pieces, fused_adamw=False) == {}
    assert rft._recipe_pieces(**pieces, fused_adamw=True) == {"optimizer_fused": True}


@pytest.mark.parametrize(
    ("backbone", "optimizer"), [(None, "master"), (Path("s"), "bf16")],
)
def test_fused_adamw_without_the_master_recipe_is_refused(backbone, optimizer):
    ns = argparse.Namespace(
        lower_layers_n=0, lower_layers_lr_scale=None, beta2=None, checkpoint_skip_layers=0,
        fused_adamw=True, optimizer=optimizer, train_attention_mask="padding",
        option_permutation_seed=None, tokenizer_json=None, real_backbone=backbone, epoch=False,
        replay_shards=None, replay_attestation=None, replay_cache=None, replay_weight=None,
        replay_every=rft.DEFAULT_REPLAY_EVERY, verdicts_out=None, score_val=False,
        suite_verdicts_out=None, needle=False, ood=False,
    )
    with pytest.raises(SystemExit, match="--fused-adamw needs"):
        rft._check_piece_flags(ns)


def test_selective_checkpointing_lands_in_the_recipe_only_when_on():
    pieces = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
              "permutation": None, "replay": None}
    assert rft._recipe_pieces(**pieces, checkpoint_skip_layers=0) == {}
    assert rft._recipe_pieces(**pieces, checkpoint_skip_layers=6) == {
        "checkpoint_skip_layers": 6
    }


@pytest.mark.parametrize(
    ("skip", "backbone", "match"),
    [(-1, Path("s"), "must not be negative"), (4, None, "needs --real-backbone")],
)
def test_a_checkpoint_skip_that_would_determine_nothing_is_refused(skip, backbone, match):
    ns = argparse.Namespace(
        lower_layers_n=0, lower_layers_lr_scale=None, beta2=None, checkpoint_skip_layers=skip,
        fused_adamw=False, optimizer="bf16", train_attention_mask="padding",
        option_permutation_seed=None, tokenizer_json=None, real_backbone=backbone, epoch=False,
        replay_shards=None, replay_attestation=None, replay_cache=None, replay_weight=None,
        replay_every=rft.DEFAULT_REPLAY_EVERY, verdicts_out=None, score_val=False,
        suite_verdicts_out=None, needle=False, ood=False,
    )
    with pytest.raises(SystemExit, match=match):
        rft._check_piece_flags(ns)


def test_a_cuda_run_whose_linear_attention_fell_back_to_torch_is_refused(monkeypatch):
    """transformers falls back to its torch reference for chunk_gated_delta_rule with only a
    log line, and puts that path at >10x slower; on cuda that is a run that hits its cap
    having trained a fraction of its plan. Refused before the step is built."""
    from types import SimpleNamespace

    import qd_train.backbone as backbone

    reference = "transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule"
    monkeypatch.setattr(
        backbone, "load_text_tower",
        lambda *a, **k: SimpleNamespace(
            linear_attention_kernels={"chunk_gated_delta_rule": reference}
        ),
    )
    plan = [SimpleNamespace(tokens=np.zeros((2, 8), dtype=np.int32))]
    with pytest.raises(SystemExit, match="torch reference on cuda"):
        rft._real_step(
            backbone=Path("snapshot"), reader=None, plan=plan, device="cuda", dtype="bf16",
            spec=rft.ADAMW_BF16, attn_implementation="sdpa", seed=0, lr=1e-5, total_steps=1,
            span_weight=1.0, width=8,
        )


def test_each_piece_that_is_on_lands_in_the_recipe_under_a_mirrored_key(tmp_path):
    from qd_train.trainer import ChoicePermutation

    replay = rft.ReplayPlan(
        batches=[], shard_hash="s" * 64, cache_path=tmp_path / "c.npz", letter_ids=(1, 2),
        weight=0.1, every=6, attestation_sha256="a" * 64,
    )
    pieces = rft._recipe_pieces(
        lower_layers_n=8, lower_lr_scale=0.1, beta2=0.95,
        permutation=ChoicePermutation(seed=5, letter_ids={}, noul_id=0,
                                      line_end_ids=frozenset()),
        replay=replay,
    )
    assert pieces == {
        "lower_layers_n": 8, "lower_lr_scale": 0.1, "beta2": 0.95,
        "option_permutation_seed": 5, "replay_shard_hash": "s" * 64,
        "replay_attestation_sha256": "a" * 64, "replay_weight": 0.1, "replay_every": 6,
        "replay_direction": "base_to_model",
    }
    assert set(pieces) <= set(rft.RECIPE_PIECE_KEYS)


def test_every_recipe_piece_key_is_mirrored_into_the_verdict_and_score_rows(tmp_path):
    """`_train` puts every piece into the ft recipe, and the verdict, eval and score rows
    mirror ``(*BACKBONE_KEYS, *RECIPE_PIECE_KEYS)`` off it -- so a piece key missing from
    RECIPE_PIECE_KEYS is on the ft row and silently absent from every row scored after it.
    --checkpoint-skip-layers and --fused-adamw were, until this test: an F run with skip 6
    named it on its ft row only, and a fused (Tier-B) model's eval rows did not say fused."""
    from qd_train.trainer import ChoicePermutation

    replay = rft.ReplayPlan(
        batches=[], shard_hash="s" * 64, cache_path=tmp_path / "c.npz", letter_ids=(1, 2),
        weight=0.1, every=6, attestation_sha256="a" * 64,
    )
    pieces = rft._recipe_pieces(
        lower_layers_n=8, lower_lr_scale=0.1, beta2=0.95,
        permutation=ChoicePermutation(seed=5, letter_ids={}, noul_id=0,
                                      line_end_ids=frozenset()),
        replay=replay, cap_s=32_400.0, no_memorise=True, batch_tokens=35_403,
        shuffled_label={"family": "code.defect_class"},
        checkpoint_skip_layers=6, fused_adamw=True, train_attention_mask="none",
    )
    assert {"checkpoint_skip_layers", "optimizer_fused", "train_attention_mask"} <= set(pieces)
    assert sorted(set(pieces) - set(rft.RECIPE_PIECE_KEYS)) == []
    run = {**pieces, "backbone_snapshot": "snap", "unrelated": 1}
    mirrored = {k: run[k] for k in (*rft.BACKBONE_KEYS, *rft.RECIPE_PIECE_KEYS) if k in run}
    assert mirrored["checkpoint_skip_layers"] == 6 and mirrored["optimizer_fused"] is True
    assert mirrored["train_attention_mask"] == "none"


def test_training_without_the_mask_lands_in_the_recipe_only_when_on():
    """The padding mask is what every row so far trained with, so it adds nothing; dropping
    it is a Tier-B change and names itself on the ft row and every row scored after it."""
    pieces = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
              "permutation": None, "replay": None}
    assert rft._recipe_pieces(**pieces, train_attention_mask="padding") == {}
    assert rft._recipe_pieces(**pieces, train_attention_mask="none") == {
        "train_attention_mask": "none"
    }
    with pytest.raises(ValueError, match="train_attention_mask"):
        rft._recipe_pieces(**pieces, train_attention_mask="causal")


def test_train_refuses_the_no_mask_switch_on_the_stand_in():
    """The stand-in has no SDPA layer to switch; a recipe saying 'none' would name nothing."""
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="train_attention_mask needs the real backbone"):
        rft._train(
            reader=None, plan=[SimpleNamespace(tokens=np.zeros((1, 4), dtype=np.int32))],
            passes=1, device="cpu", seed=0, hidden=8, heads=1, lr=1e-3, span_weight=1.0,
            ledger=None, tag="t", quick_reasons=(), train_attention_mask="none",
        )


def test_the_no_mask_switch_with_replay_is_refused_at_argv_time(tmp_path):
    """Replay's KL term runs its own forward through `hidden()`, which keeps the mask, so the
    switch would cover only part of training while the recipe claimed all of it."""
    with pytest.raises(SystemExit, match="--replay-shards"):
        rft.main(["--out", str(tmp_path), "--real-backbone", str(tmp_path),
                  "--train-attention-mask", "none", "--replay-shards", str(tmp_path)])


# --- argv refusals ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--lower-layers-n", "8"], "needs --real-backbone"),
        (["--train-attention-mask", "none"], "--train-attention-mask none needs --real-backbone"),
        (["--lower-layers-lr-scale", "0.1"], "scales no layer"),
        (["--beta2", "1.5"], r"\(0, 1\)"),
        (["--option-permutation-seed", "1"], "epoch arm only"),
        (["--option-permutation-seed", "1", "--epoch"], "needs the tokenizer.json"),
        (["--tokenizer-json", "x.json"], "does not exist"),
        (["--replay-weight", "0.1"], "without --replay-shards"),
        (["--replay-every", "3"], "without --replay-shards"),
        (["--replay-shards", "d"], "needs --replay-attestation"),
        (["--verdicts-out", "v.jsonl"], "--score-val"),
    ],
)
def test_piece_flags_that_would_determine_nothing_are_refused_at_argv_time(
    tmp_path, flags, match
):
    with pytest.raises(SystemExit, match=match):
        rft.main(["--out", str(tmp_path), *flags])


def test_replay_needs_the_epoch_arm_and_a_positive_weight(tmp_path):
    attestation = tmp_path / "att.json"
    attestation.write_text("{}")
    base = ["--out", str(tmp_path), "--replay-shards", str(tmp_path),
            "--replay-attestation", str(attestation), "--replay-cache", str(tmp_path / "c.npz")]
    with pytest.raises(SystemExit, match="epoch arm only"):
        rft.main([*base, "--replay-weight", "0.1"])
    with pytest.raises(SystemExit, match="must be positive"):
        rft.main([*base, "--replay-weight", "0", "--epoch"])
    with pytest.raises(SystemExit, match="holds no shard header"):
        rft.main([*base, "--replay-weight", "0.1", "--epoch"])


def test_an_existing_verdicts_file_is_refused_before_anything_runs(tmp_path):
    existing = tmp_path / "v.jsonl"
    existing.write_text("")
    with pytest.raises(SystemExit, match="already exists"):
        rft.main(["--out", str(tmp_path), "--epoch", "--score-val",
                  "--verdicts-out", str(existing)])


def test_the_defaults_resolve_to_the_optimizer_every_row_so_far_used():
    ns = argparse.Namespace(
        lower_layers_n=0, lower_layers_lr_scale=None, beta2=None, checkpoint_skip_layers=0,
        fused_adamw=False, optimizer="bf16", train_attention_mask="padding",
        option_permutation_seed=None, tokenizer_json=None, real_backbone=None, epoch=False,
        replay_shards=None, replay_attestation=None, replay_cache=None, replay_weight=None,
        replay_every=rft.DEFAULT_REPLAY_EVERY, verdicts_out=None, score_val=False,
        suite_verdicts_out=None, needle=False, ood=False,
    )
    rft._check_piece_flags(ns)
    assert ns.beta2 == 0.999 and ns.lower_layers_lr_scale == 1.0
    ns2 = argparse.Namespace(**{**vars(ns), "lower_layers_n": 8, "real_backbone": Path("s"),
                                "lower_layers_lr_scale": None, "beta2": None})
    rft._check_piece_flags(ns2)
    assert ns2.lower_layers_lr_scale == rft.RSI_LOWER_LR_SCALE == 0.1


# --- the permutation spec is read off the tokenizer and cross-checked -------------------------


class _Remap:
    def __init__(self, old_to_new: list[int]):
        self.old_to_new = np.asarray(old_to_new, dtype=np.int64)


class _Reader:
    def __init__(self, old_to_new: list[int]):
        self.remap = _Remap(old_to_new)


def test_the_permutation_spec_maps_the_tokenizer_through_the_remap_and_refuses_another(tmp_path):
    vocab = {"A": 0, "B": 1, "Z": 2, "Ċ": 3, ")Ċ": 4, "x": 5}
    tok = tmp_path / "tokenizer.json"
    tok.write_text(json.dumps({"model": {"vocab": vocab}}))
    reader = _Reader([10, 11, 12, 13, -1, 15])  # ")\n" was not used by the corpus
    spec = rft._permutation_spec(reader, tokenizer_json=tok, letter_id={"A": 10}, seed=3)
    assert spec.letter_ids == {"A": 10, "B": 11, "Z": 12}
    assert spec.noul_id == 12 and spec.line_end_ids == frozenset({13})
    with pytest.raises(SystemExit, match="not the tokenizer the shards were built with"):
        rft._permutation_spec(reader, tokenizer_json=tok, letter_id={"A": 11}, seed=3)


# --- --verdicts-out --------------------------------------------------------------------------


def _scored():
    return {"verdicts": [
        {"kind": "choice", "row_id": "r1", "slot_name": "intent", "top": 1, "gold_row": 1,
         "correct": True},
        {"kind": "span", "row_id": "r2", "slot_name": "evidence", "top": [3, 4],
         "correct": False},
    ]}


def test_verdict_lines_carry_the_fields_a4_reads(tmp_path):
    lines = rft._verdict_lines(_scored(), eval_row_id="e1", seed=2)
    assert lines[0] == {"eval_row_id": "e1", "seed": 2, "row_id": "r1", "kind": "choice",
                        "correct": True, "top": 1, "gold_row": 1, "slot_name": "intent"}
    assert lines[1]["gold_row"] is None and lines[1]["slot_name"] == "evidence"
    out = tmp_path / "v.jsonl"
    rft.write_verdicts_jsonl(out, lines)
    assert [json.loads(x) for x in out.read_text().splitlines()] == lines
    with pytest.raises(SystemExit, match="already exists"):
        rft.write_verdicts_jsonl(out, lines)
    with pytest.raises(ValueError, match="non-bool correct"):
        rft.write_verdicts_jsonl(tmp_path / "w.jsonl", [{**lines[0], "correct": 1}])


def test_verdict_lines_carry_expected_abstain_when_the_decoder_wrote_it():
    """What ft_linear_control scores the span rows' always-abstain opponent against."""
    scored = _scored()
    scored["verdicts"][1]["expected_abstain"] = True  # type: ignore[index]
    lines = rft._verdict_lines(scored, eval_row_id="e1", seed=2)
    assert lines[1]["expected_abstain"] is True
    assert "expected_abstain" not in lines[0]


def test_verdict_lines_carry_the_letter_distribution_a_calibration_fit_reads():
    """2026-09-30: --verdicts-out kept the argmax only, so the margin the runtime abstains
    on could not be computed from a scored run without decoding it again."""
    scored = _scored()
    scored["verdicts"][0].update({  # type: ignore[index]
        "noul_row": 4, "rows": 5, "language": "go", "noul_probability": 0.01,
        "row_logits": [0.0, 3.0, 0.5, 0.1, -2.0],
    })
    lines = rft._verdict_lines(scored, eval_row_id="e1", seed=2)
    assert lines[0]["row_logits"] == [0.0, 3.0, 0.5, 0.1, -2.0]
    assert (lines[0]["noul_row"], lines[0]["rows"], lines[0]["language"]) == (4, 5, "go")
    assert lines[0]["noul_probability"] == 0.01
    assert "row_logits" not in lines[1], "a field the decoder did not write is not invented"


def test_the_decoder_threads_slot_name_into_every_verdict():
    """``_decode`` is where Label.slot_name is available; both verdict dicts must carry it."""
    import inspect

    source = inspect.getsource(rft._decode)
    assert source.count('"slot_name": label.slot_name') == 2


# --- ft_split_rows: the one owner of the row rebuild -------------------------------------------


class _Stop(Exception):
    pass


def test_main_trains_on_exactly_what_ft_split_rows_returns(tmp_path, monkeypatch):
    """A spy on ``ft_split_rows`` and a tripwire on ``_labels``: whatever main hands the
    labeller must be the train rows ``ft_split_rows`` returned, by identity."""
    sentinel_train, sentinel_val = [object()], [object()]
    calls: list[dict] = []

    def spy(**kwargs):
        calls.append(kwargs)
        return sentinel_train, sentinel_val

    def trip(rows, *, config):
        raise _Stop(rows)

    from types import SimpleNamespace

    rev = "a" * 40  # every run writes ledger rows, so --rev is a full sha
    monkeypatch.setattr(rft, "ft_split_rows", spy)
    monkeypatch.setattr(rft, "_labels", trip)
    monkeypatch.setattr(rft, "resolve_rev", lambda repo, rev: rev)
    monkeypatch.setattr(
        rft, "ShardReader",
        lambda *a, **k: SimpleNamespace(header=SimpleNamespace(
            data_snapshot_hash="d" * 64, exclusions_sha256="",
        )),
    )
    monkeypatch.setattr(rft, "check_defect_source", lambda out, *, defect_class: None)
    monkeypatch.setattr(rft, "corpus_facts", lambda out, **kw: None)
    with pytest.raises(_Stop) as got:
        rft.main(["--out", str(tmp_path), "--max-pairs", "7", "--rev", rev])
    assert got.value.args[0] is sentinel_train
    assert calls == [{"commitpackft": None, "max_pairs": 7, "rev": rev,
                      "config": calls[0]["config"], "defect_class": None,
                      "defect_download": None, "defect_max_rows": None,
                      "repo_history": True, "general_record": None,
                      "general_max_rows": None, "replay_partition": False,
                      "defect_noul": None, "exclude_identity_keys": None}]


@pytest.mark.usefixtures("qd_prep")
def test_ft_split_rows_is_the_rebuild_main_used_to_inline(tmp_path):
    """Characterisation against the pre-extraction inline block in ``main`` (HEAD c65d7da),
    on this repository's own history at a small ``max_pairs``. The rebuild signs in qd-prep
    and the inline block below in the Python reference, so this is also their parity."""
    import real_tokenizer_pipeline as pipeline
    from repo_git import resolve_rev

    from qd_data.dedupe import dedupe
    from qd_data.mixture import build_mixture
    from qd_data.split import split

    config = DataConfig()
    rev = resolve_rev(REPO, "0632f693d3b765b726499e7b4bf19c67959b75cb")
    commits, _, _ = pipeline.code_rows(commitpackft=None, max_pairs=20, rev=rev)
    spans, _ = pipeline.span_rows(max_rows=20, blank_line_runs=False, rev=rev)
    mixture = build_mixture(
        {"bigcode/commitpackft": list(commits), "rajpurkar/squad_v2": list(spans)},
        config=config,
    )
    report = split(dedupe(list(mixture.rows), config=config), config=config)
    train, val = rft.ft_split_rows(commitpackft=None, max_pairs=20, rev=rev, config=config)
    assert [r.row_id for r in train] == [r.row_id for r in report.rows_by_split["train"]]
    assert [r.row_id for r in val] == [r.row_id for r in report.rows_by_split["val"]]
    assert train and val


# --- --defect-class: the rebuild follows the shard set's own sources ---------------------------

DEFECT_CORPUS = REPO / "data" / "pool" / "commitpackft-corpus-v2"
CPFT = REPO / "data" / "pool" / "commitpackft"


def _manifest(out: Path, sources: list[str], admitted: list[str] | None = None) -> None:
    path = out / rft.TRAIN_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "admitted_source_ids": sources if admitted is None else admitted,
        "mixture": {"n_input": {s: 10 for s in sources}},
    }))


def test_the_defect_source_must_agree_with_the_shard_sets_manifest(tmp_path):
    from qd_data.defect_class import DEFECT_SOURCE_ID

    with pytest.raises(SystemExit, match="is absent"):
        rft.check_defect_source(tmp_path, defect_class=None)
    _manifest(tmp_path, ["bigcode/commitpackft", DEFECT_SOURCE_ID])
    with pytest.raises(SystemExit, match="was built with"):
        rft.check_defect_source(tmp_path, defect_class=None)
    rft.check_defect_source(tmp_path, defect_class=Path("corpus"))
    plain = tmp_path / "plain"
    _manifest(plain, ["bigcode/commitpackft"])
    rft.check_defect_source(plain, defect_class=None)
    with pytest.raises(SystemExit, match="built without it"):
        rft.check_defect_source(plain, defect_class=Path("corpus"))
    # The registry admitting the source is not the build having read it.
    registered = tmp_path / "registered"
    _manifest(registered, ["bigcode/commitpackft"],
              admitted=["bigcode/commitpackft", DEFECT_SOURCE_ID])
    rft.check_defect_source(registered, defect_class=None)


def test_defect_download_without_a_corpus_is_refused():
    with pytest.raises(ValueError, match="read nothing"):
        rft.ft_splits(commitpackft=None, max_pairs=1, rev="x", config=DataConfig(),
                      defect_max_rows=3)


@pytest.mark.skipif(
    not (DEFECT_CORPUS / "examples.jsonl").is_file() or not CPFT.is_dir(),
    reason="the qd-mutate corpus or the commitpackft download is not on this host",
)
@pytest.mark.usefixtures("qd_prep")
def test_a_defect_class_shard_set_is_relabelled_by_id_from_the_pipelines_own_rows(tmp_path):
    """Against the pipeline's OWN output: build a small shard set with
    real_tokenizer_pipeline.run(--defect-class), then rebuild its labels the way main does
    and pair them to sequences through the writer's sequence index. Every sequence gets
    the label its index names, the letters read off the set are consistent, and without
    --defect-class the rebuild is refused."""
    pytest.importorskip("transformers")
    import real_tokenizer_pipeline as pipeline
    from repo_git import resolve_rev

    from qd_train.shards import ShardReader

    rev = resolve_rev(REPO, "7f233553a2766fd2ad956ec018c359eef056d4d2")
    out = tmp_path / "out"
    try:
        pipeline.run(out=out, max_pairs=20, blank_line_runs=False, rev=rev, commitpackft=CPFT,
                     defect_class=DEFECT_CORPUS, defect_max_rows=40)
    except OSError as exc:  # pragma: no cover - no cached tokenizer on this host
        pytest.skip(f"the pipeline could not load its tokenizer: {exc}")
    config = DataConfig()
    manifest = json.loads((out / rft.TRAIN_MANIFEST).read_text())
    reader = ShardReader(out / "shards" / "train", config=config, repo_root=out,
                         expect_rev=rev)
    assert reader.sequence_index is not None
    train, _ = rft.ft_split_rows(commitpackft=CPFT, max_pairs=20, rev=rev, config=config,
                                 defect_class=DEFECT_CORPUS, defect_max_rows=40)
    assert len(train) == manifest["n_rows"]
    labels, excluded = rft._labels(train, config=config)
    paired = rft.pair_labels(reader, labels, require_index=True)
    assert len(paired) == len(reader)
    assert [(lb.row_id, lb.slot_name) for lb in paired] == list(reader.sequence_index.sequences)
    assert any(lb.family_id == "code.defect_class" for lb in paired)
    rft._inventory(reader, paired, excluded)
    rft._letter_ids(reader, paired)  # refuses a label on the wrong sequence
    without, _ = rft.ft_split_rows(commitpackft=CPFT, max_pairs=20, rev=rev, config=config)
    with pytest.raises(SystemExit, match="was not built from these rows"):
        rft.pair_labels(reader, rft._labels(without, config=config)[0], require_index=True)
    with pytest.raises(SystemExit, match="was built with"):
        rft.check_defect_source(out, defect_class=None)


class _Index:
    def __init__(self, sequences, kinds, excluded=(), role="gold"):
        self.sequences = tuple(sequences)
        self.slot_kinds = tuple(kinds)
        self.excluded = tuple(excluded)
        self.role = role


class _Excl:
    def __init__(self, row_id, slot_name):
        self.row_id, self.slot_name = row_id, slot_name


class _IndexedReader:
    def __init__(self, index):
        self.sequence_index = index
        self.root = Path("shards/train")


def _label(row_id, slot, kind=1):
    return rft.Label(row_id=row_id, family_id="f", slot_name=slot, slot_kind=kind,
                     gold_letter="A", letters=("A", "B", "Z"))


def test_pairing_is_by_id_and_refuses_every_disagreement():
    a, b, c = _label("r1", "s"), _label("r2", "s"), _label("r3", "s")
    reader = _IndexedReader(_Index([("r2", "s"), ("r1", "s")], [1, 1], [_Excl("r3", "s")]))
    assert rft.pair_labels(reader, [a, b, c], require_index=True) == [b, a]
    with pytest.raises(SystemExit, match="does not have"):
        rft.pair_labels(reader, [a, c], require_index=True)
    with pytest.raises(SystemExit, match="neither wrote nor excluded"):
        rft.pair_labels(_IndexedReader(_Index([("r1", "s")], [1])), [a, b],
                        require_index=True)
    with pytest.raises(SystemExit, match="twice"):
        rft.pair_labels(reader, [a, a, b], require_index=True)
    with pytest.raises(SystemExit, match="slot kinds disagree"):
        rft.pair_labels(_IndexedReader(_Index([("r1", "s")], [2])), [a],
                        require_index=True)
    with pytest.raises(SystemExit, match="role is 'replay_only'"):
        rft.pair_labels(_IndexedReader(_Index([("r1", "s")], [1], role="replay_only")), [a],
                        require_index=True)
    legacy = _IndexedReader(None)
    assert rft.pair_labels(legacy, [a, b], require_index=False) == [a, b]
    with pytest.raises(SystemExit, match="no sequence index"):
        rft.pair_labels(legacy, [a], require_index=True)


# --- decoding over the letters a row OFFERS ---------------------------------------------------


class _DecodeStep:
    """hidden -> one-hot of the token, lm_head -> identity: the logit of id k at a
    position is 1 iff the token there is k. Enough to drive _decode's letter path."""

    device = "cpu"

    def __init__(self, vocab: int):
        self.vocab = vocab

    def hidden(self, batch):
        import torch

        return torch.nn.functional.one_hot(
            torch.as_tensor(batch.tokens.astype(np.int64)), self.vocab
        ).float()

    def lm_head(self, hidden):
        return hidden


def test_a_row_offering_a_letter_no_row_has_as_gold_is_counted_not_a_keyerror():
    """GAP-PORTED-DECODE-KEYERROR-ON-A-LETTER-NO-ROW-USES-AS-GOLD. letter_id read off a
    shard set by gold correspondence has no 'E' when no row's gold is E; on HEAD _decode
    raised KeyError: 'E' on the first row offering it. Now that row is counted and the
    others are decoded; given E's id, it decodes too."""
    from qd_train.artifacts import SLOT_CHOICE, Batch

    tokens = np.asarray([[7, 3, 1], [7, 3, 2]], dtype=np.int32)
    batch = Batch(tokens=tokens, lengths=np.asarray([3, 3]), bucket=3, index=0,
                  slot_kind=np.asarray([SLOT_CHOICE, SLOT_CHOICE]),
                  target_index=np.asarray([1, 1]))
    offers_e = rft.Label(row_id="r-e", family_id="f", slot_name="s", slot_kind=SLOT_CHOICE,
                         gold_letter="A", letters=("A", "B", "C", "D", "E", "Z"))
    plain = rft.Label(row_id="r-ab", family_id="f", slot_name="s", slot_kind=SLOT_CHOICE,
                      gold_letter="A", letters=("A", "B", "Z"))
    gold_ids = {"A": 3, "B": 4, "C": 5, "D": 6, "Z": 8}
    out = rft._decode(_DecodeStep(10), [batch], {0: [offers_e, plain]}, gold_ids)
    assert out["rows_not_decoded"] == 1 and out["letters_without_id"] == ["E"]
    assert out["rows_decoded"] == 1 and [v["row_id"] for v in out["verdicts"]] == ["r-ab"]
    assert out["verdicts"][0]["correct"] is True  # token 3 is A's id: the argmax is A
    full = rft._decode(_DecodeStep(10), [batch], {0: [offers_e, plain]}, {**gold_ids, "E": 9})
    assert full["rows_not_decoded"] == 0 and full["rows_decoded"] == 2


def test_offered_letter_ids_come_from_the_vocabulary_and_are_cross_checked(tmp_path):
    vocab = {"A": 0, "B": 1, "E": 2, "Z": 3}
    tok = tmp_path / "tokenizer.json"
    tok.write_text(json.dumps({"model": {"vocab": vocab}}))
    reader = _Reader([10, 11, 12, 13])
    got = rft.vocab_letter_ids(reader, tokenizer_json=tok, letter_id={"A": 10, "Z": 13})
    assert got == {"A": 10, "B": 11, "E": 12, "Z": 13}
    with pytest.raises(SystemExit, match="not the tokenizer"):
        rft.vocab_letter_ids(reader, tokenizer_json=tok, letter_id={"E": 11})


# --- a replay set must say it is one ------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "shown"),
    [
        (None, "None"),                                      # no sequence index at all
        (_Index([("r1", "s")], [1], role=None), "None"),   # an index with no role
        (_Index([("r1", "s")], [1], role="gold"), "'gold'"),
    ],
)
def test_a_replay_set_that_is_not_marked_replay_only_is_refused(index, shown):
    with pytest.raises(SystemExit, match=f"role is {shown}, not 'replay_only'"):
        rft.check_replay_role(_IndexedReader(index))


def test_a_replay_only_set_is_admitted_and_the_plan_checks_it_first():
    import inspect

    rft.check_replay_role(_IndexedReader(_Index([("r1", "s")], [1], role="replay_only")))
    source = inspect.getsource(rft._replay_plan)
    assert source.index("check_replay_role(replay_reader)") < source.index("check_attestation(")
