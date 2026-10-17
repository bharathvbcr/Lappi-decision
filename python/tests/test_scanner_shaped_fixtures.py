"""No literal in this repo may look like a live AWS credential to a push-protection scanner.

The caller-record and commit-type secret screens are tested with fake credentials, and the
fakes were written as plain literals. GitHub push protection read them as real keys and
refused the push of main (GH013), on two shapes at once:

* an access-key id: ``AKIA`` followed by sixteen upper-case letters or digits;
* a secret access key: a free-standing 40-character base64 run in a file that also names
  ``AKIA``.

Every reformat or edit of those files re-trips the scanner, so the fixtures assemble the
fake at runtime (``concat!`` of two halves) and this test keeps a literal from coming back.
It scans tracked files and untracked-but-not-ignored ones, like ``test_gaps_ledger``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ACCESS_KEY_ID = re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])")
SECRET_KEY_RUN = re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])")
HEX_ONLY = re.compile(r"[0-9a-f]{40}")

SOURCE_SUFFIXES = {".rs", ".py", ".go", ".swift", ".ts", ".md", ".json", ".jsonl", ".toml", ".sh"}


def _scanned_files() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    paths = [REPO / p.decode() for p in (tracked + untracked).split(b"\0") if p]
    assert paths, "git listed no files at all; refusing to report a vacuous clean scan"
    return [p for p in paths if p.suffix in SOURCE_SUFFIXES and p.is_file()]


def _hits(line: str, file_names_akia: bool) -> list[str]:
    found = ACCESS_KEY_ID.findall(line)
    if file_names_akia:
        found += [m for m in SECRET_KEY_RUN.findall(line) if not HEX_ONLY.fullmatch(m)]
    return found


def test_the_shapes_are_recognised() -> None:
    """The detector is not vacuous: assembled at runtime here, so this file stays clean."""
    key_id = "AKIA" + "ABCDEFGHIJKLMNOP"
    assert _hits(f"see {key_id}", False) == [key_id]
    secret = "AIzaSyA7" + "abcdefghijklmnopqrstuvwxyz012345"
    assert _hits(f'apiKey: "{secret}"', True) == [secret]
    # A 40-hex git sha beside AKIA is a commit id, not a key; a word-embedded prefix is not an id.
    assert _hits("0" * 40, True) == []
    assert _hits("TASKAK" + "IA0123456789012345", True) == []
    # Without AKIA in the file, a 40-character run alone is not the scanner's pair.
    assert _hits(secret, False) == []


def test_no_file_holds_a_literal_the_push_scanner_reads_as_an_aws_key() -> None:
    files = _scanned_files()
    assert files, "no source files matched; refusing to report a vacuous clean scan"
    offenders: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # The secret-key shape pairs with AKIA anywhere in the file, not only on its line.
        names_akia = "AKIA" in text
        for lineno, line in enumerate(text.splitlines(), 1):
            for hit in _hits(line, names_akia):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {hit}")
    assert not offenders, (
        "assemble fake credentials at runtime (concat! of two halves) so push protection "
        "does not read them as live keys:\n" + "\n".join(offenders)
    )
