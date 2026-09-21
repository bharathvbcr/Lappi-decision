"""The shard header pins the code that produced its rows, or it pins only the corpus.

``GAP-SHARD-SET-GOES-STALE-AGAINST-THE-CORPUS-CODE-THAT-REPRODUCES-ITS-LABELS``, opened
2026-09-21 against a measured incident: a shard set whose ``data_snapshot_hash``,
``tokenizer_hash`` and ``remap_hash`` all still matched, and whose rows the working tree no
longer reproduced -- 341 sequences on disk against 321 the code now yields.

That one was caught because the drift moved the row COUNT, and ``tools/real_ft_run.py``
matches the reconstructed order against ``supervision.npz`` entry for entry. These are for
the case that leaves the count alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_data.config import DataConfig  # noqa: E402
from qd_data.fingerprint import (  # noqa: E402
    code_fingerprint,
    describe_drift,
    drifted_modules,
)
from qd_train.artifacts import (  # noqa: E402
    ShardContractViolation,
    ShardHeader,
    assert_shard_trainable,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402


def _header(**over: object) -> ShardHeader:
    fields: dict[str, object] = {
        "split": "train",
        "data_snapshot_hash": "a" * 64,
        "tokenizer_hash": "b" * 64,
        "remap_hash": "c" * 64,
        "vocab_size": 128,
        "n_sequences": 4,
        "total_tokens": 40,
        "max_seq_len": 16,
        "buckets": (16,),
    }
    fields.update(over)
    return ShardHeader(**fields)  # type: ignore[arg-type]


# --- the fingerprint itself -----------------------------------------------------------


def test_the_fingerprint_covers_every_module_in_the_package() -> None:
    """The whole package, not a curated list of the modules that "really" matter.

    A curated list is a second thing to keep current, and it is silently wrong the first
    time someone adds a module or moves a function between two of them.
    """
    fingerprint = code_fingerprint()
    on_disk = {p.name for p in (REPO / "python" / "qd_data").glob("*.py")}
    assert on_disk, "qd_data has no sources; the test is not measuring anything"
    assert set(fingerprint) == on_disk
    assert "mixture.py" in fingerprint and "render.py" in fingerprint
    assert all(len(v) == 64 for v in fingerprint.values())


def test_the_fingerprint_is_stable_across_calls() -> None:
    """A fingerprint that moved between two calls would refuse every shard set it wrote."""
    assert code_fingerprint() == code_fingerprint()


def test_every_shape_of_drift_is_drift() -> None:
    """A changed module, a new one and a vanished one are all "this is not that code"."""
    recorded = {"a.py": "1" * 64, "b.py": "2" * 64}
    assert drifted_modules(recorded, dict(recorded)) == ((), (), ())
    assert drifted_modules(recorded, {"a.py": "9" * 64, "b.py": "2" * 64})[0] == ("a.py",)
    assert drifted_modules(recorded, {**recorded, "c.py": "3" * 64})[1] == ("c.py",)
    assert drifted_modules(recorded, {"a.py": "1" * 64})[2] == ("b.py",)


def test_the_drift_description_names_the_modules_not_just_the_fact() -> None:
    """The difference between a refusal someone diagnoses and one someone disables."""
    recorded = {"render.py": "1" * 64, "mixture.py": "2" * 64}
    current = {"render.py": "9" * 64, "mixture.py": "2" * 64, "new.py": "3" * 64}
    described = describe_drift(recorded, current)
    assert "render.py" in described
    assert "mixture.py" not in described.split("added since:")[0].replace("render.py", "")
    assert "new.py" in described
    assert describe_drift(recorded, dict(recorded)) == ""


# --- the header carries it ------------------------------------------------------------


def test_the_header_carries_the_fingerprint_through_a_round_trip() -> None:
    fingerprint = code_fingerprint()
    header = _header(code_fingerprint=fingerprint)
    assert ShardHeader.from_json(header.to_json()).code_fingerprint == fingerprint


def test_the_fingerprint_cannot_be_edited_out_to_make_a_stale_set_look_current() -> None:
    """It is covered by ``shard_hash``, which ``from_json`` recomputes.

    Without this, the remedy for a refusal is to delete one key from a JSON file.
    """
    raw = _header(code_fingerprint=code_fingerprint()).to_json()
    raw.pop("code_fingerprint")
    with pytest.raises(ShardContractViolation, match="modified after it was written"):
        ShardHeader.from_json(raw)

    tampered = _header(code_fingerprint=code_fingerprint()).to_json()
    tampered["code_fingerprint"] = {"render.py": "0" * 64}
    with pytest.raises(ShardContractViolation, match="modified after it was written"):
        ShardHeader.from_json(tampered)


def test_a_header_written_before_the_field_still_verifies() -> None:
    """An empty fingerprint contributes nothing to ``shard_hash``.

    Three ledger rows name a shard set written before this field existed. Breaking their
    header's integrity check to add a feature would retire evidence to gain a guard.
    """
    old = _header()
    assert old.code_fingerprint == {}
    raw = old.to_json()
    assert ShardHeader.from_json(raw).code_fingerprint == {}


# --- the trainer's door ---------------------------------------------------------------


def test_a_set_whose_code_still_matches_is_admitted(tmp_path) -> None:
    checks = assert_shard_trainable(
        _header(code_fingerprint=code_fingerprint()),
        config=DataConfig(),
        path=tmp_path,
        repo_root=tmp_path,
    )
    current = checks["shard_code_current"]
    assert isinstance(current, Ran) and current.passed


def test_a_set_whose_code_moved_is_refused(tmp_path) -> None:
    """The case the count check cannot see.

    Every other hash in this header matches. Only the generating code moved, which is
    exactly the incident the gap records.
    """
    stale = dict(code_fingerprint())
    stale["render.py"] = "0" * 64
    with pytest.raises(ShardContractViolation, match="qd_data has changed"):
        assert_shard_trainable(
            _header(code_fingerprint=stale),
            config=DataConfig(),
            path=tmp_path,
            repo_root=tmp_path,
        )


def test_the_refusal_names_the_module_that_moved(tmp_path) -> None:
    stale = dict(code_fingerprint())
    stale["mixture.py"] = "0" * 64
    with pytest.raises(ShardContractViolation) as caught:
        assert_shard_trainable(
            _header(code_fingerprint=stale),
            config=DataConfig(),
            path=tmp_path,
            repo_root=tmp_path,
        )
    assert "mixture.py" in str(caught.value)


def test_reading_a_stale_set_on_purpose_is_possible_and_must_be_said(tmp_path) -> None:
    """Named after ``write_shards(allow_contradictions=)``: a real case, said out loud.

    It declines to RAISE. It does not silence the check, which still reports ``passed=False``
    into the ledger row -- otherwise the escape would erase the evidence it was used.
    """
    stale = dict(code_fingerprint())
    stale["split.py"] = "0" * 64
    checks = assert_shard_trainable(
        _header(code_fingerprint=stale),
        config=DataConfig(),
        path=tmp_path,
        repo_root=tmp_path,
        allow_stale_code=True,
    )
    current = checks["shard_code_current"]
    assert isinstance(current, Ran) and not current.passed
    assert "split.py" in str(current.detail)


def test_a_set_that_predates_the_field_is_not_run_and_not_a_pass(tmp_path) -> None:
    """Rule 5's shape at this boundary.

    "This set is current" and "nobody could tell" must not be the same string in a ledger
    row. An empty fingerprint is the second, and it does not raise -- the shard sets three
    ledger rows name are readable, and honestly labelled.
    """
    checks = assert_shard_trainable(
        _header(),
        config=DataConfig(),
        path=tmp_path,
        repo_root=tmp_path,
    )
    current = checks["shard_code_current"]
    assert isinstance(current, NotRun)
    assert "no code_fingerprint" in current.reason
