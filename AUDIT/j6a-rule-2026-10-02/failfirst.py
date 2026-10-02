"""Fail-first for `qd-post-f-rules j6a` (lane L-room, 2026-10-02). Throwaway audit tooling.

The before-state is main's checker (74aea95, sha256 ea6296d8..., no j6a code) with the new test
module grafted on in place of its own. Run it, record the output, restore the source byte for
byte and check its sha256.

    git show HEAD:crates/qd-runtime/src/bin/qd_post_f_rules.rs > target/l-room/head74.rs
    /Users/bharath/.venvs/ml/bin/python AUDIT/j6a-rule-2026-10-02/failfirst.py
"""

import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "crates/qd-runtime/src/bin/qd_post_f_rules.rs"
HEAD = ROOT / "target/l-room/head74.rs"
OUT = ROOT / "AUDIT/j6a-rule-2026-10-02/failfirst-before.txt"
MARK = "#[cfg(test)]\nmod tests {"
CMD = ["cargo", "test", "-p", "qd-runtime", "--bin", "qd-post-f-rules"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    current = SRC.read_bytes()
    head = HEAD.read_bytes()
    cur_text, head_text = current.decode(), head.decode()
    assert cur_text.count(MARK) == 1 and head_text.count(MARK) == 1
    before = head_text[: head_text.index(MARK)] + cur_text[cur_text.index(MARK) :]
    lines = [
        f"after-state source sha256 {sha(current)}",
        f"main 74aea95 source sha256 {sha(head)} (checker code before j6a)",
        f"before-state = main's code + this lane's test module, sha256 {sha(before.encode())}",
        f"command: {' '.join(CMD)}",
        "",
    ]
    try:
        SRC.write_text(before)
        p = subprocess.run(CMD, cwd=ROOT, capture_output=True, text=True, timeout=900)
    finally:
        SRC.write_bytes(current)
    restored = SRC.read_bytes()
    lines.append(f"[before] exit {p.returncode}")
    lines.append(p.stdout + p.stderr)
    lines.append(f"[restored, sha256 {sha(restored)}, equal: {restored == current}]")
    OUT.write_text("\n".join(lines) + "\n")
    print(lines[-1], "before exit", p.returncode)
    return 0 if restored == current else 1


if __name__ == "__main__":
    sys.exit(main())
