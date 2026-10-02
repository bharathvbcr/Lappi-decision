"""``tools/replay_decontam.py`` counts unrenderable targets instead of crashing on them.

GAP-REPLAY-DECONTAM-TOOL-CRASHES-ON-UNRENDERABLE-FT-VAL-ROW: 4 of 50 val rows on the default
corpus carry a context over the 131,072-byte render cap, ``render`` raised
``ContextTooLargeRefusal`` and the tool exited before comparing anything. ``write_shards``
refuses the same rows, so they are in no shard set and never scored; they are counted and
skipped. The tool still fails closed where the check itself cannot run: a target split with
rows and nothing renderable, or a shard set that is not a replay set.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from data_fixtures import commitpackft_row

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import replay_decontam  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.render import DEFAULT_CAPS  # noqa: E402

BIG = "x = 1\n" * (DEFAULT_CAPS.max_context_bytes // 6 + 10)


def _rows(n: int, *, big: frozenset[int] = frozenset()) -> list:
    raws = [commitpackft_row(i, body=BIG if i in big else None) for i in range(n)]
    mixture = build_mixture(
        {"bigcode/commitpackft": raws}, config=DataConfig(), families=["code.commit_intent"]
    )
    # The mixture builds over-cap rows (render refuses them later, as write_shards does).
    return list(mixture.rows)


def test_an_unrenderable_target_row_is_counted_and_skipped() -> None:
    rows = _rows(6, big=frozenset({2}))
    texts, unrenderable = replay_decontam.row_texts(rows)
    assert len(unrenderable) == 1
    (why,) = unrenderable.values()
    assert why.startswith("ContextTooLargeRefusal")
    assert {k.split("#")[0] for k in texts} == {r.row_id for r in rows} - set(unrenderable)


def test_a_target_split_that_renders_nothing_stops_the_tool() -> None:
    rows = _rows(2, big=frozenset({0, 1}))
    with pytest.raises(SystemExit, match="all 2 row"):
        replay_decontam.target_texts({"val": rows, "heldout": _rows(3)})


def test_a_shard_set_that_is_not_a_replay_set_is_refused(tmp_path: Path) -> None:
    for index in (None, types.SimpleNamespace(role="gold")):
        reader = types.SimpleNamespace(sequence_index=index, root=tmp_path, remap=None)
        with pytest.raises(SystemExit, match="not a replay shard set"):
            replay_decontam.replay_texts(reader, tmp_path / "tokenizer.json")  # type: ignore[arg-type]


# --- --hits-out: every pair, resolved to its row (Fable's J6(a) ruling, Q5) -----------------------

LONG_A = (
    "the quick brown fox jumps over the lazy dog while the farmer counts sheep in the "
    "early morning light near the old mill by the river"
)
LONG_B = (
    "a completely different passage about compilers and register allocation that shares "
    "no run of eight words with the other one at all whatsoever"
)


def _hit_list(tmp_path: Path, *, identity: dict[str, str] | None = None) -> dict:
    from qd_train.replay import decontaminate

    # Sequence i of the replay set is (row_id, slot); the replay ids decontaminate sees are
    # str(i), as replay_texts keys them.
    sequences = (("mmlu:a", "answer"), ("mmlu:b", "answer"), ("csqa:c", "answer"))
    replay = {"0": LONG_A + " and then " + LONG_B, "1": "nothing shared with any target row",
              "2": "x " + LONG_B}
    report = decontaminate(replay, {"val": {"va#answer": LONG_A, "vb#answer": LONG_B},
                                    "heldout": {"hb#answer": LONG_B}})
    return replay_decontam.hit_list(
        report, sequences=sequences,
        identity_of=identity if identity is not None else {
            "mmlu:a": "key-a", "mmlu:b": "key-b", "csqa:c": "key-c",
        },
        replay_shard_hash="r" * 64, corpus={"rev": "x"}, attestation_sha256="s" * 64,
    )


def test_the_hit_list_names_every_pair_with_its_row_slot_and_identity_key(tmp_path: Path) -> None:
    body = _hit_list(tmp_path)
    got = {(p["sequence"], p["row_id"], p["slot_name"], p["identity_key"], p["target"],
            p["target_row"]) for p in body["pairs"]}
    assert got == {
        (0, "mmlu:a", "answer", "key-a", "val", "va#answer"),
        (0, "mmlu:a", "answer", "key-a", "val", "vb#answer"),
        (0, "mmlu:a", "answer", "key-a", "heldout", "hb#answer"),
        (2, "csqa:c", "answer", "key-c", "val", "vb#answer"),
        (2, "csqa:c", "answer", "key-c", "heldout", "hb#answer"),
    }
    # h: distinct replay rows with any pair; the identity keys the pipeline excludes.
    assert body["replay_rows_hit"] == 2
    assert body["identity_keys"] == ["key-a", "key-c"]
    assert body["hits"] == {"val": 2, "heldout": 2}
    assert body["tool"] == replay_decontam.HITS_TOOL
    assert all(isinstance(p["shared"], int) and isinstance(p["target_ngrams"], int)
               for p in body["pairs"])


def test_a_hit_row_the_replay_manifest_does_not_name_stops_the_tool(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="csqa:c"):
        _hit_list(tmp_path, identity={"mmlu:a": "key-a", "mmlu:b": "key-b"})


def test_hits_out_is_a_flag_and_refuses_to_overwrite_before_any_work(tmp_path: Path) -> None:
    existing = tmp_path / "hits.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        replay_decontam.main([
            "--out", str(tmp_path), "--replay-shards", str(tmp_path),
            "--tokenizer-json", str(tmp_path / "tokenizer.json"),
            "--attestation-out", str(tmp_path / "att.json"), "--hits-out", str(existing),
        ])
