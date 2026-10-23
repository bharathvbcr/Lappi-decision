"""Print the Python-written header constants crates/qd-train/tests/shard_code_rule.rs embeds.

The mini ``qd_data`` there (a.py, b.py, an unlisted c.py, a fingerprint.py listing a.py and b.py
under rule 2 and a.py, b.py, c.py under rule 3) is hashed here with the Python writer's own
functions, so the Rust reader is compared against bytes Python wrote, not against itself.
Run: /Users/bharath/.venvs/ml/bin/python AUDIT/x6-fingerprint-list-2026-10-23/make_rust_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train.artifacts import ShardHeader  # noqa: E402

FP_PY = (
    b'"""mini."""\n\n'
    b'SHARD_CONTENT_MODULES_V2: Final[tuple[str, ...]] = (\n    "a.py",\n    "b.py",\n)\n\n'
    b'SHARD_CONTENT_MODULES: Final[tuple[str, ...]] = (\n    "a.py",\n    "b.py",\n    "c.py",\n)\n'
)
MODULES = {"a.py": b"a = 1\n", "b.py": b"b = 2\n", "c.py": b"c = 3\n", "fingerprint.py": FP_PY}


def digest(name: str) -> str:
    return hashlib.sha256(MODULES[name]).hexdigest()


def header(version: int, names: list[str]) -> str:
    fields: dict[str, object] = {
        "split": "train", "data_snapshot_hash": "d" * 64, "tokenizer_hash": "e" * 64,
        "remap_hash": "9" * 64, "vocab_size": 64, "n_sequences": 3, "total_tokens": 10,
        "max_seq_len": 8, "buckets": (8,), "corpus_rev": "92e1c74",
        "created_at": "2026-10-18T01:47:22.825896+00:00",
        "code_fingerprint": {n: digest(n) for n in names},
    }
    if version != 1:
        fields["code_fingerprint_version"] = version
    return json.dumps(ShardHeader(**fields).to_json())  # type: ignore[arg-type]


print("FP_PY =", FP_PY)
print("PY_V1 =", header(1, ["a.py", "b.py", "c.py", "fingerprint.py"]))
print("PY_V2 =", header(2, ["a.py", "b.py"]))
print("PY_V3 =", header(3, ["a.py", "b.py", "c.py"]))
