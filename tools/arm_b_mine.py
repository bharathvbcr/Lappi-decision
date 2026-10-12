#!/usr/bin/env python3
"""Arm B mining driver (card ft-b1dfb6da273; ``campaign/v6-arm-b-mining-preregistered.json``).

Arm B teaches when *not* to abstain: it keeps train-side candidate rows that the scorer (v5 seed
1) abstains on although their constructed gold is not ``noul``. This driver owns the parts that
must be refused before anything is scored, and the selection after:

``plan``    refuse, then dedupe. Reads the pre-registration, which must be committed and equal to
            ``HEAD``; every corpus and exclusion set must be pinned there by sha256 and hash to it;
            no input may be under a held-out path or hold a caller record; no candidate may come
            from a held-out repository. Drops exact duplicates of v5 train and val keys (and of
            each other), counted by set. Writes ``candidates.jsonl`` and ``plan.json``.
``select``  read the scorer's ``--verdicts-out`` lines (``tools/real_ft_run.py --score-checkpoint``
            at fp32, run separately as a heavy job under ``tools/mac_heavy.sh``), apply the
            runtime's choice rule exactly (:func:`abstentions` mirrors ``choice_rule_abstentions``
            and a test holds it to that function), keep abstained rows whose gold is not ``noul``,
            apply the cap (5,000) and the floor (300: fewer reads ``no_data``).
``budget``  the pre-registered scan size from a measured rows/s.

Nothing here scores a row, loads a model or reads a held-out path. A check that could not run is
recorded ``not_run``; it is never recorded as clean.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from qd_data.caller_records import CallerRecordRefusal, assert_no_caller_records  # noqa: E402
from qd_data.config import DataConfig  # noqa: E402
from qd_train.data_access import HeldOutViolation, assert_path_not_held_out  # noqa: E402

#: R4: the scan is capped at 20,000 rows and the wall budget is 36 h.
SCAN_CAP_ROWS = 20_000
WALL_BUDGET_S = 36 * 3600
#: Bound on any one input file: a mis-pointed path is refused, not slurped.
MAX_INPUT_BYTES = 8 << 30


class Refusal(Exception):
    """An input the driver will not read, or a rule that does not hold."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            size += len(chunk)
            if size > MAX_INPUT_BYTES:
                raise Refusal(f"{path}: larger than {MAX_INPUT_BYTES} bytes")
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=False, timeout=120
    )


def check_committed(prereg: Path, repo: Path) -> dict[str, Any]:
    """The pre-registration, refused unless it is tracked and equals ``HEAD``'s blob."""
    prereg = prereg.resolve()
    try:
        rel = str(prereg.relative_to(repo.resolve()))
    except ValueError as exc:
        raise Refusal(f"{prereg} is not inside {repo}: not committed there") from exc
    if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        raise Refusal(f"{rel} is not committed: a pre-registration is committed before any run")
    if _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise Refusal(
            f"{rel} differs from HEAD: an edited pre-registration is not the registered one"
        )
    try:
        return json.loads(prereg.read_text())  # type: ignore[no-any-return]
    except (OSError, ValueError) as exc:
        raise Refusal(f"{rel}: {exc}") from exc


def check_input(path: Path, *, repo: Path, want_sha: str | None, what: str) -> str:
    """One input file: not held out, no caller record, pinned and matching. Returns its sha256."""
    if want_sha is None:
        raise Refusal(f"{what} ({path}) is not pinned: its sha256 is null in the pre-registration")
    try:
        assert_path_not_held_out(path, config=DataConfig(), repo_root=repo)
    except HeldOutViolation as exc:
        raise Refusal(f"{what} ({path}) is a held-out path and is not read: {exc}") from exc
    if not path.is_file():
        raise Refusal(f"{what} ({path}) is not a file")
    try:
        assert_no_caller_records(path)
    except CallerRecordRefusal as exc:
        raise Refusal(f"{what} ({path}) holds a caller record or cannot be scanned: {exc}") from exc
    got = _sha256_file(path)
    if got != want_sha:
        raise Refusal(f"{what} ({path}) hashes to {got}, not the pre-registered {want_sha}")
    return got


def check_scorer(path: Path, prereg: Mapping[str, Any]) -> str:
    """The scorer checkpoint, by its pre-registered sha256."""
    want = prereg["scorer"]["sha256"]
    got = _sha256_file(path)
    if got != want:
        raise Refusal(f"the scorer {path} hashes to {got}, not the pre-registered {want}")
    return got


def function_sha256(commit: str, name: str, *, repo: Path = ROOT) -> str:
    """sha256 of ``tools/real_ft_run.py``'s function ``name`` at ``commit`` (its source segment)."""
    blob = _git(repo, "show", f"{commit}:tools/real_ft_run.py")
    if blob.returncode != 0:
        raise Refusal(f"git show {commit}:tools/real_ft_run.py failed: {blob.stderr.strip()}")
    for node in ast.parse(blob.stdout).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(blob.stdout, node)
            assert seg is not None
            return hashlib.sha256(seg.encode()).hexdigest()
    raise Refusal(f"{name} is not in tools/real_ft_run.py at {commit}")


def _rows(path: Path, what: str) -> Iterable[dict[str, Any]]:
    with path.open("rb") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise Refusal(f"{what} {path}:{n}: not JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise Refusal(f"{what} {path}:{n}: not an object")
            yield row


def content_digest(row: Mapping[str, Any]) -> str | None:
    """sha256(before NUL after): what makes a re-id'd copy of a row a duplicate."""
    if isinstance(row.get("before"), str) and isinstance(row.get("after"), str):
        return hashlib.sha256(f"{row['before']}\x00{row['after']}".encode()).hexdigest()
    return None


def plan(
    *,
    prereg: Path,
    repo: Path,
    out_dir: Path,
    heldout_repos: Sequence[str],
    containment_attestation: Path | None = None,
) -> dict[str, Any]:
    """Refuse, dedupe, and write ``candidates.jsonl`` and ``plan.json`` into a new ``out_dir``."""
    if out_dir.exists():
        raise Refusal(f"{out_dir} exists; refusing to overwrite it")
    doc = check_committed(prereg, repo)
    inputs: dict[str, str] = {"preregistration": _sha256_file(prereg)}

    sets: list[tuple[str, set[str], set[str]]] = []
    for s in doc["exclusion"]["sets"]:
        path = Path(s["path"])
        inputs[f"exclusion/{s['name']}"] = check_input(
            path, repo=repo, want_sha=s.get("sha256"), what=f"exclusion set {s['name']}"
        )
        keys: set[str] = set()
        digests: set[str] = set()
        for row in _rows(path, s["name"]):
            if isinstance(row.get(s["key"]), str):
                keys.add(row[s["key"]])
            for alt in ("identity_key", "id"):
                if isinstance(row.get(alt), str):
                    keys.add(row[alt])
            d = content_digest(row)
            if d is not None:
                digests.add(d)
        sets.append((s["name"], keys, digests))

    held = set(heldout_repos)
    seen_keys: set[str] = set()
    seen_digests: set[str] = set()
    offered = 0
    dup_by_set = {name: 0 for name, _, _ in sets}
    dup_within = 0
    kept: list[dict[str, Any]] = []
    for corpus in doc["candidates"]["corpora"]:
        path = Path(corpus["path"]) if corpus.get("path") else None
        if path is None:
            raise Refusal(
                f"corpus {corpus['name']} is not pinned: its path is null in the pre-registration"
            )
        inputs[f"corpus/{corpus['name']}"] = check_input(
            path, repo=repo, want_sha=corpus.get("sha256"), what=f"corpus {corpus['name']}"
        )
        for row in _rows(path, corpus["name"]):
            offered += 1
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise Refusal(
                    f"corpus {corpus['name']}: a candidate has no id (row {offered}): refused whole"
                )
            if row.get("repo") in held:
                raise Refusal(
                    f"corpus {corpus['name']}: {row['id']} is from a held-out repository "
                    f"({row['repo']}): refused whole (rule 3)"
                )
            digest = content_digest(row)
            hit = next(
                (
                    name
                    for name, keys, digs in sets
                    if row["id"] in keys or (digest and digest in digs)
                ),
                None,
            )
            if hit is not None:
                dup_by_set[hit] += 1
                continue
            if row["id"] in seen_keys or (digest and digest in seen_digests):
                dup_within += 1
                continue
            seen_keys.add(row["id"])
            if digest:
                seen_digests.add(digest)
            kept.append(row)
    cap = int(doc["candidates"].get("scan_cap_rows", SCAN_CAP_ROWS))
    over_cap = max(0, len(kept) - cap)
    kept = kept[:cap]
    if containment_attestation is None:
        containment: dict[str, Any] = {
            "state": "not_run",
            "reason": "no containment attestation was given; the scan against val, held-out and "
            "target texts is not recorded as clean",
        }
    else:
        containment = {"state": "ran", "attestation_sha256": _sha256_file(containment_attestation)}
    body = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in kept)
    report: dict[str, Any] = {
        "schema": "qd-arm-b-plan/v1",
        "preregistration": str(prereg),
        "head": _git(repo, "rev-parse", "HEAD").stdout.strip(),
        "offered": offered,
        "duplicates_by_set": dup_by_set,
        "duplicates_within": dup_within,
        "over_scan_cap": over_cap,
        "kept": len(kept),
        "leakage": {
            "exact_identity": {"state": "ran", "sets": [n for n, _, _ in sets]},
            "containment": containment,
        },
        "inputs": inputs,
        "candidates_sha256": hashlib.sha256(body).hexdigest(),
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out_dir.name + ".partial-", dir=out_dir.parent))
    (tmp / "candidates.jsonl").write_bytes(body)
    (tmp / "plan.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    tmp.rename(out_dir)
    report["out_dir"] = str(out_dir)
    return report


def abstentions(lines: Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    """``tools/real_ft_run.py``'s ``choice_rule_abstentions`` over ``--verdicts-out`` lines.

    Per first-pass choice row whose gold is not ``noul``: abstain when the first pass's top row is
    ``noul``, or the permuted pass's top row is ``noul``, or the permuted winner maps back to a
    different option. A row with no ``perm`` has no second pass. The permuted pass's ``noul`` row is
    the first pass's: the options are permuted, the abstain row stays last.
    """
    out: dict[str, bool] = {}
    for v in lines:
        if v.get("kind") != "choice" or bool(v.get("expected_abstain")):
            continue
        top1 = int(v["top"])
        noul_row = int(v["noul_row"])
        abstained = top1 == noul_row
        perm = v.get("perm")
        if perm is not None:
            top2 = int(v["top_permuted"])
            abstained = abstained or top2 == noul_row or int(perm[top2]) != top1
        out[str(v["row_id"])] = abstained
    return out


def keep(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rows that abstained and whose constructed gold is not ``noul``."""
    return [r for r in rows if r["abstained"] and not r["expected_abstain"]]


def apply_caps(rows: Sequence[Mapping[str, Any]], *, cap: int, floor: int) -> dict[str, Any]:
    """The cap and the floor: fewer than ``floor`` kept rows is ``no_data`` and no rows."""
    if len(rows) < max(floor, 1):
        return {"state": "no_data", "kept": len(rows), "floor": floor, "rows": []}
    return {"state": "mined", "kept": len(rows), "cap": cap, "rows": list(rows[:cap])}


def scan_rows(*, rows_per_s: float) -> int:
    """``min(20,000, floor(rows_per_s x 36 x 3600))``, the pre-registered scan size."""
    if not math.isfinite(rows_per_s) or rows_per_s <= 0:
        raise Refusal(f"rows_per_s {rows_per_s!r} is not a positive finite number")
    return min(SCAN_CAP_ROWS, math.floor(rows_per_s * WALL_BUDGET_S))


def _select(args: argparse.Namespace) -> int:
    doc = check_committed(args.prereg, args.repo)
    verdicts = Path(args.verdicts)
    lines = [json.loads(line) for line in verdicts.read_text().splitlines() if line.strip()]
    abstained = abstentions(lines)
    gold = {
        str(v["row_id"]): bool(v.get("expected_abstain"))
        for v in lines
        if v.get("kind") == "choice"
    }
    rows = [
        {"row_id": k, "abstained": a, "expected_abstain": gold.get(k, False)}
        for k, a in abstained.items()
    ]
    res = apply_caps(
        keep(rows),
        cap=int(doc["caps"]["train_cap_mined_rows"]),
        floor=int(doc["caps"]["floor_rows"]),
    )
    out = Path(args.out)
    if out.exists():
        raise Refusal(f"{out} exists; refusing to overwrite it")
    out.write_text(
        json.dumps(
            {
                "state": res["state"],
                "kept": res["kept"],
                "scored_choice_rows": len(abstained),
                "row_ids": [r["row_id"] for r in res["rows"]],
                "verdicts_sha256": _sha256_file(verdicts),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{res['state']}: {res['kept']} kept of {len(abstained)} scored -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--prereg", type=Path, required=True)
    a.add_argument("--repo", type=Path, default=ROOT)
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument(
        "--heldout-repos",
        type=Path,
        required=True,
        help="a text file, one held-out repository slug per line (own-repos manifest)",
    )
    a.add_argument("--containment-attestation", type=Path)
    s = sub.add_parser("select")
    s.add_argument("--prereg", type=Path, required=True)
    s.add_argument("--repo", type=Path, default=ROOT)
    s.add_argument("--verdicts", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("budget")
    b.add_argument("--rows-per-s", type=float, required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "plan":
            held = [x.strip() for x in args.heldout_repos.read_text().splitlines() if x.strip()]
            r = plan(
                prereg=args.prereg,
                repo=args.repo,
                out_dir=args.out_dir,
                heldout_repos=held,
                containment_attestation=args.containment_attestation,
            )
            print(
                json.dumps({k: r[k] for k in ("offered", "kept", "duplicates_by_set", "out_dir")})
            )
            return 0
        if args.cmd == "select":
            return _select(args)
        print(scan_rows(rows_per_s=args.rows_per_s))
        return 0
    except Refusal as exc:
        print(f"arm_b_mine: refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
