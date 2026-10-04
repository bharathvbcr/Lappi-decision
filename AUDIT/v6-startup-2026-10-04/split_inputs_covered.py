"""Does the split cache's key cover every file the rebuild read? Checked on the v5 data.

Analysis only. Runs ``tools/real_ft_run.py``'s ``main`` with the given argv up to the split
rebuild. In place of the rebuild, it computes ``tools/split_cache.compute_key`` over the exact
arguments ``main`` passes, plus ``real_ft_run.split_rebuild_inputs``, which are what
``--split-cache`` keys. It then compares the key's covered files with the files
``split_inputs_audit.py`` saw the rebuild open (its JSON, ``AUDIT_JSON``), excluding the
interpreter's files and the ``*.py`` the key covers as code. It prints every observed file
the key does not cover, and the key's time and size. Then it stops. Exit status 0 means every
observed file is covered.

Usage: python split_inputs_covered.py <AUDIT_JSON> <real_ft_run.py argv...>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "python"))

import real_ft_run as rft  # noqa: E402
import split_cache  # noqa: E402


def main() -> None:
    audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    argv = sys.argv[2:]
    observed = sorted(
        f["realpath"] for f in audit["files"] if not f["interpreter"] and not f["code_py"]
    )

    def covered_check(**kw: Any) -> Any:
        inputs, facts = rft.split_rebuild_inputs(
            general_record=kw["general_record"], exclude_identity_keys=kw["exclude_identity_keys"],
            repo_history=kw["repo_history"],
        )
        began = time.perf_counter()
        key = split_cache.compute_key(kw, extra_inputs=inputs, repo=rft.REPO, extra_facts=facts)
        took = time.perf_counter() - began
        covered = set(key.covered)
        missing = [p for p in observed if p not in covered]
        n_bytes = sum(Path(p).stat().st_size for p in covered)
        print(f"[covered] key {key.digest} in {took:.1f} s over {len(covered)} files, "
              f"{n_bytes} bytes; extra inputs {sorted(inputs)}; facts {facts}", flush=True)
        print(f"[covered] {len(observed)} data files observed by the audit, "
              f"{len(observed) - len(missing)} covered by the key, {len(missing)} not", flush=True)
        for p in missing:
            print(f"[covered] NOT COVERED: {p}", flush=True)
        raise SystemExit(1 if missing else 0)

    rft.ft_split_rows = covered_check
    rft.main(argv)


if __name__ == "__main__":
    main()
