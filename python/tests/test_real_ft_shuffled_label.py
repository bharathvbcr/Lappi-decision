"""``tools/real_ft_run.py --shuffled-label``: the FT runner trains the shuffled-label control.

docs/hardening.md: a model trained on PERMUTED labels must fall to chance, and if it does not
the split leaks. ``eval_harness.shuffled_label_control`` has always been the check; every FT
eval row records it ``not_run``, because nothing trained the model it measures on the FT path,
and ``Ledger.promotion_verdict`` refuses a seed family whose control did not run.

What is pinned here: the permutation (rung 0's draw, within option-count groups, seeded, over
the code.defect_class choice golds only), the batches the epoch arm trains on (only answer
tokens move), the control row (the target eval row's protocol, ``recipe.eval_row_id``, only
``controls.shuffled_label`` measured) and its join in promotion, and the refusals. The
end-to-end half runs ``main()`` on the CPU stand-in backbone over byte-tokenized
code.defect_class shard sets written by ``write_shards``, as the pipeline would.

Torch-gated, like ``test_real_ft_protocol.py``: the tool raises ``SystemExit`` at import
without torch, so this module skips in the core venv rather than passing vacuously.
"""

from __future__ import annotations

import collections
import dataclasses
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID, DEFECT_SOURCE_ID, DefectRow  # noqa: E402
from qd_data.manifest import build_manifests  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_data.split import HELD_OUT, split  # noqa: E402
from qd_train.artifacts import NO_SPAN, SLOT_CHOICE, SLOT_SPAN, Batch, RemapTable  # noqa: E402
from qd_train.eval_harness import permute_within_groups  # noqa: E402
from qd_train.ledger import Ledger, LedgerRow  # noqa: E402
from qd_train.shards import ShardReader, write_shards  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

REV = "a" * 40

# --- 1. the permutation: rung 0's draw, one owner -------------------------------------------


def _rung0_reference(golds: list[int], arities: list[int], seed: int) -> list[int]:
    """The draw ``tools/rung0_real_run.shuffle_train_labels`` made before it delegated to
    ``eval_harness.permute_within_groups`` (HEAD e3477eb), restated as the oracle."""
    rng = random.Random(seed)
    by_arity: dict[int, list[int]] = {}
    for index, arity in enumerate(arities):
        by_arity.setdefault(arity, []).append(index)
    out = list(golds)
    for indices in by_arity.values():
        labels = [golds[i] for i in indices]
        rng.shuffle(labels)
        for i, label in zip(indices, labels, strict=True):
            out[i] = label
    return out


def test_the_shared_permutation_is_rung0s_draw_bit_for_bit() -> None:
    """Factoring the draw out must not move one label: a seed names one permutation."""
    rng = random.Random(7)
    for seed in range(40):
        n = rng.randint(0, 80)
        arities = [rng.choice([2, 3, 4]) for _ in range(n)]
        golds = [rng.randrange(a) for a in arities]
        assert permute_within_groups(golds, arities, seed=seed) == _rung0_reference(
            golds, arities, seed
        )


def test_rung0s_shuffle_still_draws_the_same_permutation() -> None:
    from rung0_real_run import shuffle_train_labels

    @dataclasses.dataclass(frozen=True, slots=True)
    class Decision:
        example_id: str
        options: tuple[bytes, ...]
        gold_option: int

    rng = random.Random(3)
    arities = [rng.choice([2, 4]) for _ in range(50)]
    golds = [rng.randrange(a) for a in arities]
    decisions = [
        Decision(f"e{i}", tuple(b"o" * (k + 1) for k in range(a)), g)
        for i, (a, g) in enumerate(zip(arities, golds, strict=True))
    ]
    out = shuffle_train_labels(decisions, seed=11)
    assert [d.gold_option for d in out] == _rung0_reference(golds, arities, 11)
    assert [d.example_id for d in out] == [d.example_id for d in decisions]


def test_the_permutation_refuses_labels_and_groups_that_do_not_pair() -> None:
    with pytest.raises(ValueError, match="3 labels against 2 group keys"):
        permute_within_groups([0, 1, 2], [4, 4], seed=0)


# --- 2. which golds move: the defect family's choice letters, within option-count groups ------


def _label(i: int, *, gold: str, letters: tuple[str, ...] = ("A", "B", "C", "D", "Z"),
           family: str = DEFECT_FAMILY_ID, kind: int = SLOT_CHOICE) -> rft.Label:
    return rft.Label(
        row_id=f"r{i:03d}", family_id=family,
        slot_name="defect_class" if kind == SLOT_CHOICE else "defect_span",
        slot_kind=kind, gold_letter=gold, letters=letters,
    )


def _mixed_labels() -> list[rft.Label]:
    """Four-option defect rows whose golds are only C and D, two-option defect rows whose golds
    are only A and B, another family's choice rows, and the defect family's span rows."""
    out: list[rft.Label] = []
    for i in range(40):
        out.append(_label(i, gold="CD"[i % 2]))
        out.append(_label(i, gold="Z", letters=("Z",), kind=SLOT_SPAN))
    for i in range(40, 60):
        out.append(_label(i, gold="AB"[i % 2], letters=("A", "B", "Z")))
    for i in range(60, 75):
        out.append(_label(i, gold="A", family="general.mmlu"))
    return out


def test_only_the_defect_familys_choice_golds_are_permuted_and_only_within_their_group() -> None:
    labels = _mixed_labels()
    got = rft.shuffle_choice_golds(labels, family=DEFECT_FAMILY_ID, seed=0)
    defect_choice = {
        (x.row_id, x.slot_name) for x in labels
        if x.family_id == DEFECT_FAMILY_ID and x.slot_kind == SLOT_CHOICE
    }
    assert set(got.golds) == defect_choice, "span rows and other families are never relabelled"
    by_key = {(x.row_id, x.slot_name): x for x in labels}
    for k in (2, 4):
        group = [
            pair for key, pair in got.golds.items()
            if len([c for c in by_key[key].letters if c != "Z"]) == k
        ]
        assert collections.Counter(t for t, _ in group) == collections.Counter(
            n for _, n in group
        ), f"the {k}-option group's gold distribution must be preserved exactly"
    # A cross-group exchange would hand a two-option row a C or D, which it does not offer.
    assert all(new in by_key[key].letters for key, (_, new) in got.golds.items())
    assert got.moved > 0 and got.moved == sum(1 for t, n in got.golds.values() if t != n)


def test_the_permutation_is_a_function_of_the_seed_and_says_which_one() -> None:
    labels = _mixed_labels()
    a = rft.shuffle_choice_golds(labels, family=DEFECT_FAMILY_ID, seed=4)
    b = rft.shuffle_choice_golds(list(labels), family=DEFECT_FAMILY_ID, seed=4)
    c = rft.shuffle_choice_golds(labels, family=DEFECT_FAMILY_ID, seed=5)
    assert a.golds == b.golds and a.digest == b.digest
    assert a.digest != c.digest
    recipe = a.recipe()
    assert recipe["permutation_seed"] == 4 and recipe["permutation_sha256"] == a.digest
    assert recipe["rows_permuted"] == 60 and recipe["family"] == DEFECT_FAMILY_ID


def test_a_set_with_no_defect_choice_row_is_refused() -> None:
    with pytest.raises(SystemExit, match=r"no code\.defect_class choice sequence"):
        rft.shuffle_choice_golds(
            [_label(0, gold="A", family="general.mmlu")], family=DEFECT_FAMILY_ID, seed=0
        )


# --- 3. what the epoch arm trains on: only answer tokens move ---------------------------------

LETTER_ID = {"A": 65, "B": 66, "C": 67, "D": 68, "Z": 90}


def _batch() -> Batch:
    """Row 0 a defect choice row answering A, row 1 a span row, row 2 another family's choice."""
    width = 8
    tokens = np.zeros((3, width), dtype=np.int32)
    tokens[0, :6] = [1, 2, 3, 4, 5, 65]
    tokens[1, :6] = [1, 2, 3, 4, 5, 90]
    tokens[2, :5] = [1, 2, 3, 4, 66]
    return Batch(
        tokens=tokens,
        lengths=np.array([6, 6, 5], dtype=np.int32),
        bucket=0,
        index=0,
        slot_kind=np.array([SLOT_CHOICE, SLOT_SPAN, SLOT_CHOICE], dtype=np.uint8),
        target_index=np.array([4, 4, 3], dtype=np.int32),
        span_target=np.array([(NO_SPAN, NO_SPAN), (0, 2), (NO_SPAN, NO_SPAN)], dtype=np.int32),
        line_starts=np.array(
            [[False] * width, [i % 2 == 0 and i < 6 for i in range(width)], [False] * width],
            dtype=bool,
        ),
    )


def _labels_for() -> dict[int, list[rft.Label]]:
    return {0: [_label(0, gold="A"), _label(0, gold="Z", letters=("Z",), kind=SLOT_SPAN),
                _label(2, gold="B", family="general.mmlu")]}


def _golds(pairs: dict[tuple[str, str], tuple[str, str]]) -> rft.ShuffledGolds:
    return rft.ShuffledGolds(
        family=DEFECT_FAMILY_ID, seed=0, golds=pairs,
        moved=sum(1 for t, n in pairs.values() if t != n), digest="d" * 64,
    )


def test_a_permuted_gold_replaces_exactly_its_answer_token() -> None:
    batch = _batch()
    before = batch.tokens.copy()
    (out,), rewritten = rft.apply_shuffled_golds(
        [batch], _labels_for(), _golds({("r000", "defect_class"): ("A", "C")}), LETTER_ID
    )
    assert rewritten == 1
    changed = np.argwhere(out.tokens != before)
    assert changed.tolist() == [[0, 5]], "the answer token of row 0 and nothing else"
    assert int(out.tokens[0, 5]) == LETTER_ID["C"]
    assert np.array_equal(batch.tokens, before), "the reader's batch is not edited in place"


def test_an_answer_token_that_is_not_the_true_gold_is_refused() -> None:
    with pytest.raises(SystemExit, match="not its gold 'B'"):
        rft.apply_shuffled_golds(
            [_batch()], _labels_for(), _golds({("r000", "defect_class"): ("B", "C")}), LETTER_ID
        )


def test_a_relabelled_sequence_missing_from_the_plan_is_refused() -> None:
    golds = _golds({("r000", "defect_class"): ("A", "C"), ("r999", "defect_class"): ("B", "A")})
    with pytest.raises(SystemExit, match="1 relabelled sequence"):
        rft.apply_shuffled_golds([_batch()], _labels_for(), golds, LETTER_ID)


# --- 4. the control's value: the family's val choice rows against their real golds -------------


def _verdict(i: int, *, correct: bool, kind: str = "choice") -> dict[str, object]:
    return {"kind": kind, "row_id": f"r{i:03d}", "slot_name": "defect_class", "correct": correct}


def test_the_control_scores_the_familys_choice_rows_against_the_majority_letter() -> None:
    labels = [_label(i, gold="AABB"[i % 4] if i < 8 else "A") for i in range(10)] + [
        _label(20, gold="A", family="general.mmlu")
    ]
    verdicts = [_verdict(i, correct=i < 3) for i in range(10)] + [_verdict(20, correct=True)]
    state = rft.shuffled_label_state(
        {"verdicts": verdicts}, labels, family=DEFECT_FAMILY_ID
    )
    assert isinstance(state, Ran)
    assert (state.n, state.n_total, state.value) == (10, 10, pytest.approx(0.3))
    assert "chance (majority) 0.6000" in state.detail, "6 of the 10 golds are A"
    assert state.passed


def test_a_row_the_decode_skipped_leaves_the_coverage_short_rather_than_complete() -> None:
    labels = [_label(i, gold="A") for i in range(5)]
    state = rft.shuffled_label_state(
        {"verdicts": [_verdict(i, correct=False) for i in range(4)]}, labels,
        family=DEFECT_FAMILY_ID,
    )
    assert isinstance(state, Ran) and (state.n, state.n_total) == (4, 5)
    assert "1 of 5 rows were not decoded" in state.detail


def test_a_val_set_without_the_family_reports_not_run() -> None:
    state = rft.shuffled_label_state(
        {"verdicts": []}, [_label(0, gold="A", family="general.mmlu")], family=DEFECT_FAMILY_ID
    )
    assert isinstance(state, NotRun)


# --- 5. argv: every combination the control cannot be -------------------------------------------

_CONTROL = ["--shuffled-label", "6d170b3c", "--rev", REV]


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        ([], "needs --epoch, --no-memorise, --score-val"),
        (["--epoch", "--score-val"], "needs --no-memorise"),
        (["--epoch", "--no-memorise"], "needs --score-val"),
        (["--epoch", "--no-memorise", "--score-val", "--needle"], "--needle would measure"),
        (["--epoch", "--no-memorise", "--score-val", "--ood"], "--ood would measure"),
        (["--epoch", "--no-memorise", "--score-val", "--score-checkpoint", "c.json"],
         "--score-checkpoint would measure"),
        (["--epoch", "--no-memorise", "--score-val", "--verdicts-out", "v.jsonl"],
         "--verdicts-out would measure"),
        # The control's checkpoint would be the real arm's file name in a mirrored launch.
        # --real-backbone, because the stand-in refuses checkpointing before this is asked.
        (["--epoch", "--no-memorise", "--score-val", "--real-backbone", "snapshot",
          "--checkpoint-dir", "ckpt", "--checkpoint-every", "5"],
         "--checkpoint-dir, --checkpoint-every would write or read"),
        (["--epoch", "--no-memorise", "--score-val", "--devices", "cpu", "--seeds", "0", "1"],
         "exactly one --seeds"),
        (["--epoch", "--no-memorise", "--score-val", "--seeds", "0"], "one --devices entry"),
        (["--epoch", "--no-memorise", "--score-val", "--seeds", "0", "--devices", "cpu",
          "--span-weight", "0.5"], "--span-weight would determine nothing"),
        (["--epoch", "--no-memorise", "--score-val", "--seeds", "0", "--devices", "cpu",
          "--span-w=1.0"], "--span-weight would determine nothing"),
    ],
)
def test_a_combination_the_control_cannot_be_is_refused_before_anything_loads(
    tmp_path, monkeypatch, flags, match
):
    monkeypatch.setattr(rft, "ShardReader", _tripwire)
    with pytest.raises(SystemExit, match=match):
        rft.main(["--out", str(tmp_path), *_CONTROL, *flags])


def _tripwire(*args: object, **kwargs: object) -> None:
    raise AssertionError("an argv refusal read the shard set first")


def test_resuming_the_control_from_a_checkpoint_is_refused(tmp_path, monkeypatch):
    """The only epoch checkpoint at this seed and device is the REAL arm's: resuming the
    control from it would start the control from the model it is the control for."""
    checkpoint = tmp_path / "epoch-seed0-cpu.json"
    checkpoint.write_text("{}")
    monkeypatch.setattr(rft, "ShardReader", _tripwire)
    with pytest.raises(SystemExit, match="--resume-from would write or read"):
        rft.main([
            "--out", str(tmp_path), *_CONTROL, "--epoch", "--no-memorise", "--score-val",
            "--real-backbone", str(tmp_path), "--devices", "cpu", "--seeds", "0",
            "--resume-from", str(checkpoint),
        ])


# --- 6. end to end on the CPU stand-in ------------------------------------------------------------


def _defect_rows(n_repos: int) -> list[DefectRow]:
    out = []
    for i in range(n_repos):
        for j, cls in enumerate(("stub", "logic", "cosmetic", "clean")):
            diff = (
                f"@@ -1,3 +1,3 @@\n def f{i}_{j}(x):\n-    return x + {j}\n"
                f"+    return {i} * {j}\n     pass\n"
            )
            out.append(
                DefectRow(
                    example_id=f"c{i}_{j}:x.py#0", pool_id=f"c{i}_{j}:x.py", repo=f"org/r{i}",
                    path="x.py", symbol="f", arity=1, language="python", mutation_class=cls,
                    operator="clean" if cls == "clean" else f"{cls}.op", diff=diff,
                    diff_span=None if cls == "clean" else (3, 3), span_refusal=None,
                    licence="mit",
                )
            )
    return out


def _tokenize(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _offsets(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for ci, ch in enumerate(text):
        out.extend((ci, ci + 1) for _ in ch.encode("utf-8"))
    return out


def _remap() -> RemapTable:
    return RemapTable(
        old_to_new=np.arange(256, dtype=np.int32), new_to_old=np.arange(256, dtype=np.int32),
        tokenizer_hash="tokhash-shuffled-label", special_ids=(),
    )


@dataclasses.dataclass(frozen=True)
class Corpus:
    out: Path
    train: list[DataRow]
    val: list[DataRow]
    ledger: Path


def _patch(mp: pytest.MonkeyPatch, corpus_rows: tuple[list[DataRow], list[DataRow]]) -> None:
    """What main reads from outside the shard set, answered for these rows: the rebuild, the
    revision, the defect-source check, the manifest's facts, and the out-of-process probe."""
    mp.setattr(rft, "ft_split_rows", lambda **kw: corpus_rows)
    mp.setattr(rft, "resolve_rev", lambda repo, rev: rev)
    mp.setattr(rft, "check_defect_source", lambda out, *, defect_class: None)
    mp.setattr(
        rft, "corpus_facts",
        lambda out, **kw: rft.CorpusFacts(snapshot_not_run=None, history_rows={}),
    )
    mp.setattr(
        rft, "_probe_one",
        lambda device, rows, width, hidden, heads: {
            "device": device, "rows": rows, "width": width, "ok": True, "wall_s": 0.0,
            "why": "stubbed in-process",
        },
    )


def _argv(corpus: Corpus, ledger: Path, *extra: str) -> list[str]:
    return [
        "--out", str(corpus.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
        "--epoch", "--no-memorise", "--score-val", "--ledger", str(ledger), *extra,
    ]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    """Train and val code.defect_class shard sets, and the target: one ordinary epoch arm
    with --score-val at seed 0, whose ft and eval rows are in ``target.jsonl``."""
    out = tmp_path_factory.mktemp("shuffled-label")
    config = DataConfig()
    mixture = build_mixture(
        {DEFECT_SOURCE_ID: _defect_rows(48)}, config=config, families=[DEFECT_FAMILY_ID]
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    paths: dict[str, Path] = {}
    for name, manifest in manifests.items():
        path = out / "data" / (HELD_OUT if name == HELD_OUT else "pool") / f"{name}.json"
        manifest.write(path)
        paths[name] = path
    train = list(split_report.rows_by_split["train"])
    val = list(split_report.rows_by_split["val"])
    assert train and val
    for name, rows in (("train", train), ("val", val)):
        write_shards(
            paths[name], rows, out_dir=out / "shards" / name, remap=_remap(),
            tokenize=_tokenize, token_offsets=_offsets, config=config, repo_root=out,
            corpus_rev=REV,
        )
    corpus = Corpus(out=out, train=train, val=val, ledger=out / "target.jsonl")
    with pytest.MonkeyPatch.context() as mp:
        _patch(mp, (train, val))
        rft.main(_argv(corpus, corpus.ledger))
    return corpus


def _target(corpus: Corpus) -> tuple[LedgerRow, LedgerRow]:
    rows = Ledger(corpus.ledger).rows()
    evals = [r for r in rows if r.run_kind == "eval"]
    fts = [r for r in rows if r.run_kind == "ft"]
    assert len(evals) == 1 and len(fts) == 1, [r.run_kind for r in rows]
    return evals[0], fts[0]


def test_the_score_row_carries_a_class_share_beside_every_slot_shape(corpus: Corpus) -> None:
    """Written by the real --score-val path: what the verdict reads under a share-only
    degenerate_head_floor (qd_train.ledger DEGENERATE_SHARE_ONLY). A row without it reads
    not_run under that rule, so every shape needs one, a failing shape included."""
    target_eval, _ = _target(corpus)
    shapes = [k for k in target_eval.metrics
              if k.startswith("degenerate_head.choice.") and k.count(".") == 2]
    assert shapes, sorted(target_eval.metrics)
    for key in shapes:
        share = target_eval.metrics.get(f"{key}.top_class_share")
        assert isinstance(share, Ran), (key, share)
        assert isinstance(share.value, float) and 0.0 < share.value <= 1.0
        assert share.passed == (share.value <= 0.95)
        assert share.n is not None and share.n_total is not None and share.n <= share.n_total
        head = target_eval.metrics[key]
        if isinstance(head, Ran):
            assert share.n_total == head.n_total, (key, share, head)


def _copy_ledger(corpus: Corpus, tmp_path: Path) -> Path:
    path = tmp_path / "ledger.jsonl"
    shutil.copyfile(corpus.ledger, path)
    return path


def test_the_control_row_supplements_the_target_and_promotion_counts_it(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_eval, target_ft = _target(corpus)
    family = target_eval.protocol.hash_without_seed()
    ledger_path = _copy_ledger(corpus, tmp_path)
    unmeasured = f"{target_eval.row_id}: control 'shuffled_label' did not run"
    before = Ledger(ledger_path).promotion_verdict(family)
    assert any(r.startswith(unmeasured) for r in before.reasons), before.reasons

    captured: dict[str, object] = {}
    original = rft._train

    def spy(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return original(**kwargs)

    _patch(monkeypatch, (corpus.train, corpus.val))
    monkeypatch.setattr(rft, "_train", spy)
    rft.main(_argv(corpus, ledger_path, "--shuffled-label", target_eval.row_id[:8]))

    rows = Ledger(ledger_path).rows()
    new = rows[2:]
    assert [r.run_kind for r in new] == ["ft", "eval"], (
        "the shuffled run writes its ft row and the control row, and no score row that "
        "would hash the shuffled model into the real model's family"
    )
    ft, control = new

    # The shuffled model's own row: its own family, span channel off, the control's key.
    assert ft.recipe is not None and ft.recipe["span_weight"] == 0.0
    assert ft.recipe["shuffled_label"] == dict(rft.SHUFFLED_LABEL_RECIPE)
    assert ft.protocol.hash_without_seed() not in {
        family, target_ft.protocol.hash_without_seed()
    }
    assert rft.recipe_differences(
        dict(target_ft.recipe or {}), dict(ft.recipe), skip=rft.SHUFFLED_LABEL_EXEMPT
    ) == [], "nothing but the shuffle and the span weight differs from the target's recipe"

    # The control row: the target's protocol and seed, naming it, measuring one thing.
    assert control.protocol.to_json() == target_eval.protocol.to_json()
    assert control.recipe is not None and control.recipe["eval_row_id"] == target_eval.row_id
    assert control.recipe["shuffled_ft_row_id"] == ft.row_id
    assert control.recipe["target_ft_row_id"] == target_ft.row_id
    assert control.quick == target_eval.quick
    state = control.controls["shuffled_label"]
    n_val_choice = sum(
        1 for x in rft._labels(corpus.val, config=DataConfig())[0]
        if x.family_id == DEFECT_FAMILY_ID and x.slot_kind == SLOT_CHOICE
    )
    assert isinstance(state, Ran) and (state.n, state.n_total) == (n_val_choice, n_val_choice)
    measured = [
        name for name, s in (*control.gates.items(), *control.controls.items())
        if isinstance(s, Ran)
    ]
    assert measured == ["shuffled_label"], "the shuffled model's other states are not the target's"

    # The plan the epoch arm trained on: only choice answer tokens moved, as many as the
    # control row says moved, and the choice golds' letter distribution is unchanged.
    assert captured["span_weight"] == 0.0
    assert captured["shuffled_label"] == rft.SHUFFLED_LABEL_RECIPE
    config = DataConfig()
    reader = ShardReader(corpus.out / "shards" / "train", config=config, repo_root=corpus.out,
                         expect_rev=REV)
    original_plan = list(
        reader.batches(batch_tokens=int(max(reader.header.buckets)), seed=config.seed, epoch=0)
    )
    trained = captured["plan"]
    assert isinstance(trained, list) and len(trained) == len(original_plan)
    moved = 0
    before_golds: collections.Counter[int] = collections.Counter()
    after_golds: collections.Counter[int] = collections.Counter()
    for a, b in zip(original_plan, trained, strict=True):
        assert a.target_index is not None and a.slot_kind is not None
        for r, c in np.argwhere(a.tokens != b.tokens):
            assert c == int(a.target_index[r]) + 1 and int(a.slot_kind[r]) == SLOT_CHOICE
            moved += 1
        for r in np.flatnonzero(a.slot_kind == SLOT_CHOICE):
            at = int(a.target_index[r]) + 1
            before_golds[int(a.tokens[r, at])] += 1
            after_golds[int(b.tokens[r, at])] += 1
    assert moved == control.recipe["shuffled_label"]["rows_moved"] > 0
    assert before_golds == after_golds

    # Promotion reads the eval row and its supplement as one unit.
    after = Ledger(ledger_path).promotion_verdict(family)
    unit = f"{target_eval.row_id} + {control.row_id}:"
    assert any(r.startswith(unit) for r in after.reasons), after.reasons
    assert not any(
        "control 'shuffled_label' did not run" in r or "control 'shuffled_label' absent" in r
        for r in after.reasons
    ), after.reasons
    assert not any(r.startswith(f"{control.row_id}: supplements") for r in after.reasons)


def test_a_seed_other_than_the_targets_is_refused_before_training(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_eval, _ = _target(corpus)
    _patch(monkeypatch, (corpus.train, corpus.val))
    monkeypatch.setattr(rft, "_train", _no_training)
    argv = _argv(corpus, _copy_ledger(corpus, tmp_path), "--shuffled-label", target_eval.row_id)
    argv[argv.index("--seeds") + 1] = "1"
    with pytest.raises(SystemExit, match="it is seed 0 and this run is seed 1"):
        rft.main(argv)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--lr", "0.01"], "lr: target"),
        (["--wall-clock-cap-s", "600"], "wall_clock_cap_s: target"),
        (["--deterministic"], "deterministic: target False, this run True"),
    ],
)
def test_a_recipe_other_than_the_targets_is_refused_before_training(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str],
    match: str,
) -> None:
    target_eval, _ = _target(corpus)
    _patch(monkeypatch, (corpus.train, corpus.val))
    monkeypatch.setattr(rft, "_train", _no_training)
    with pytest.raises(SystemExit, match=match):
        rft.main(_argv(corpus, _copy_ledger(corpus, tmp_path),
                       "--shuffled-label", target_eval.row_id, *extra))


def test_a_target_that_is_not_an_eval_row_is_refused(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target_ft = _target(corpus)
    _patch(monkeypatch, (corpus.train, corpus.val))
    monkeypatch.setattr(rft, "_train", _no_training)
    with pytest.raises(SystemExit, match="not a completed eval row"):
        rft.main(_argv(corpus, _copy_ledger(corpus, tmp_path),
                       "--shuffled-label", target_ft.row_id))


def test_a_quick_control_for_a_target_that_is_not_quick_is_refused(corpus: Corpus) -> None:
    """A quick row in a seed family blocks the family, so the supplement may not be the
    quick one. The CPU target is quick itself; here it is presented as not."""
    target_eval, target_ft = _target(corpus)
    eval_recipe = dict(target_eval.recipe or {})
    promoted = dataclasses.replace(target_eval, quick=False, quick_reason=None)
    rows = [promoted, target_ft]
    reader = SimpleNamespace(header=SimpleNamespace(
        shard_hash=lambda: eval_recipe["shard_hash"],
        data_snapshot_hash=target_eval.protocol.data_snapshot_hash,
        tokenizer_hash=target_eval.protocol.tokenizer_hash,
    ))
    val = SimpleNamespace(reader=SimpleNamespace(header=SimpleNamespace(
        shard_hash=lambda: eval_recipe["val_shard_hash"],
    )))
    kwargs = dict(rows=rows, reader=reader, val=val, labels=[], seed=0,
                  planned=dict(target_ft.recipe or {}))
    with pytest.raises(SystemExit, match="a quick supplement blocks its whole seed family"):
        rft.prepare_shuffled_label(target_eval.row_id, quick=["device 'cpu'"], **kwargs)
    # The same call with nothing quick about it passes every check and reaches the draw.
    with pytest.raises(SystemExit, match=r"no code\.defect_class choice sequence"):
        rft.prepare_shuffled_label(target_eval.row_id, quick=[], **kwargs)


def _no_training(**kwargs: object) -> dict[str, object]:
    raise AssertionError("a refusal decidable before training reached _train")


def test_two_seeds_in_one_invocation_each_train_with_no_inherited_permutation(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not the control: the ordinary epoch arm the fixture already runs, at two seeds at once.

    The scoring block named its ``permutation_consistency`` gate ``permutation`` -- the same
    local that holds the epoch arm's option permutation spec -- so the second seed's ``_train``
    was handed the first seed's gate as the permutation to train with. The campaign ran one
    seed per invocation and never reached it; ``--seeds 0 1 2`` does.
    """
    handed: list[object] = []
    original = rft._train

    def spy(**kwargs: object) -> dict[str, object]:
        handed.append(kwargs["permutation"])
        return original(**kwargs)

    _patch(monkeypatch, (corpus.train, corpus.val))
    monkeypatch.setattr(rft, "_train", spy)
    argv = _argv(corpus, tmp_path / "two-seeds.jsonl")
    at = argv.index("--seeds")
    argv[at + 1:at + 2] = ["0", "1"]
    rft.main(argv)

    assert handed == [None, None], "no --option-permutation-seed: neither seed trains permuted"
    evals = [r for r in Ledger(tmp_path / "two-seeds.jsonl").rows() if r.run_kind == "eval"]
    assert sorted(r.protocol.seed for r in evals) == [0, 1]
    # Not run on the stand-in (no tokenizer.json), but each seed's row states its own.
    assert all("permutation_consistency" in r.gates for r in evals)


# --- the whole-plan evaluation after training ------------------------------------------------


def test_the_epoch_arm_never_runs_the_whole_plan_evaluation(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The epoch arm (every F, v5, J5' and noul-weight run) records no floor metric, so the
    no-grad pass over its whole train plan fed only the stdout report. On F it took about 57 min
    of a 5 h 55 min seed (GAP-EPOCH-ARM-EVALUATES-THE-WHOLE-TRAIN-PLAN-FOR-NO-ROW-2026-10-03).
    The arm must still write its ft and eval rows, and its report must say the pass did not
    run rather than drop the key."""

    def refuse(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("the epoch arm ran _evaluate, which no row of it records")

    _patch(monkeypatch, (corpus.train, corpus.val))
    monkeypatch.setattr(rft, "_evaluate", refuse)
    ledger = tmp_path / "no-plan-eval.jsonl"
    rft.main(_argv(corpus, ledger))

    kinds = sorted(r.run_kind for r in Ledger(ledger).rows())
    assert kinds == ["eval", "ft"]
    assert f'"not_run": "{rft.PLAN_EVALUATION_NOT_RUN}"' in capsys.readouterr().out


def test_skipping_the_whole_plan_evaluation_moves_no_hash_and_no_measurement(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-change epoch arm, re-created by forcing the pass back on, against the fixture's
    own epoch run of the same corpus and seed, which skips it. Both must have the same
    protocols, and every gate, control and metric on the eval row must read the same state and
    value: the scoring that follows the pass does not depend on it."""
    original = rft._train

    def with_the_pass(**kwargs: object) -> dict[str, object]:
        return original(**{**kwargs, "evaluate_plan": True})

    _patch(monkeypatch, (corpus.train, corpus.val))
    monkeypatch.setattr(rft, "_train", with_the_pass)
    ledger = tmp_path / "with-the-pass.jsonl"
    rft.main(_argv(corpus, ledger))
    target_eval, target_ft = _target(corpus)
    rows = Ledger(ledger).rows()
    (again_eval,) = [r for r in rows if r.run_kind == "eval"]
    (again_ft,) = [r for r in rows if r.run_kind == "ft"]
    assert again_ft.protocol.hash() == target_ft.protocol.hash()
    assert again_eval.protocol.hash() == target_eval.protocol.hash()

    # Identity, not measurement: each eval row names its own run's ft row.
    link = "ft_run_row_id"
    assert again_eval.metrics[link].to_json()["value"] == again_ft.row_id
    assert target_eval.metrics[link].to_json()["value"] == target_ft.row_id

    def measured(row: LedgerRow) -> dict[str, object]:
        return {
            f"{part}.{name}": (state.to_json().get("state"), state.to_json().get("value"))
            for part, states in (("gate", row.gates), ("control", row.controls),
                                 ("metric", row.metrics))
            for name, state in states.items()
            if (part, name) != ("metric", link)
        }

    assert measured(again_eval) == measured(target_eval)


def test_the_memorise_arm_still_runs_the_whole_plan_evaluation(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memorise arm's verdict row records letter_loss_reached_its_floor and
    span_loss_reached_its_floor from that pass, so it must still run there."""
    calls: list[int] = []
    original = rft._evaluate

    def counted(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(1)
        return original(*args, **kwargs)

    _patch(monkeypatch, (corpus.train, corpus.val))
    monkeypatch.setattr(rft, "_evaluate", counted)
    ledger = tmp_path / "memorise.jsonl"
    rft.main(["--out", str(corpus.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
              "--ledger", str(ledger)])

    assert calls, "the memorise arm did not evaluate its plan"
    floors = [r for r in Ledger(ledger).rows() if "letter_loss_reached_its_floor" in r.metrics]
    assert floors, "no row recorded the memorise arm's floor metric"
