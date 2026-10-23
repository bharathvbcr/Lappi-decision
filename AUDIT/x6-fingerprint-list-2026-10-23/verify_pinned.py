"""Read-only: open each pinned header copy with the branch's reader and report.

Reads header.json only (never tokens.u32). Reports, per header: the recorded shard_hash, whether
ShardHeader.from_json accepts it (it recomputes shard_hash from the fields and refuses a
mismatch), and what artifacts._shard_code_check -- the check verify_shard_contract runs --
says about the recorded fingerprint against the code now loaded. Also prints the fingerprint a
rebuild would record now.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_data.fingerprint import shard_code_fingerprint  # noqa: E402
from qd_train import artifacts  # noqa: E402
from qd_train.artifacts import ShardHeader  # noqa: E402

HERE = Path(__file__).resolve().parent
for name in ("lora2b-train.header.json", "lora2b-val.header.json"):
    raw = json.loads((HERE / name).read_text(encoding="utf-8"))
    header = ShardHeader.from_json(raw)
    recorded = raw["shard_hash"]
    state = artifacts._shard_code_check(header, path=Path(name))
    print(f"{name}: recorded shard_hash {recorded}")
    same = header.shard_hash() == recorded
    print(f"  from_json accepted; recomputed shard_hash equals recorded: {same}")
    print(f"  _shard_code_check: {type(state).__name__} passed={getattr(state, 'passed', None)}")
    print(f"  detail: {getattr(state, 'detail', getattr(state, 'reason', ''))}")
print("fingerprint a rebuild would record now:")
print(json.dumps(shard_code_fingerprint(), indent=1, sort_keys=True))
