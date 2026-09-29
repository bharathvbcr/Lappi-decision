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
