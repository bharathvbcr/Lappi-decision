"""Rule 3, layer 2, on the path as given: the Python door's verdict on the shared table.

crates/qd-train/tests/door.rs reads the same table, so the two doors cannot drift on a
spelling (W10-c; a literal heldout/.. was refused here and admitted by Rust).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qd_data.config import DataConfig
from qd_data.errors import HeldOutViolation
from qd_train.data_access import assert_path_not_held_out

TABLE = json.loads(
    (Path(__file__).parent / "data" / "held_out_given_path_cases.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", TABLE, ids=[c["name"] for c in TABLE])
def test_python_door_gives_the_tables_verdict(case, tmp_path) -> None:
    path = Path(case["path"].replace("{base}", str(tmp_path)))
    try:
        assert_path_not_held_out(path, config=DataConfig(), repo_root=tmp_path)
    except HeldOutViolation:
        verdict = "refused"
    else:
        verdict = "admitted"
    assert verdict == case["verdict"], case["path"]
