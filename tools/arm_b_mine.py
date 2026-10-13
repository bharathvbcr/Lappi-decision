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
import collections
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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


_QDM_PREFIX = re.compile(r"^qdm:[^:]+:")


def normalise_key(key: str) -> str:
    """One key format for ids: the mixture's ``qdm:<family>:<corpus id>`` (what v5's verdict
    lines carry) and the corpus's own ``<corpus id>`` are the same row. Without this the v5 val
    exclusion matched 0 of 2,304 ids (verify-lane6 D9)."""
    return _QDM_PREFIX.sub("", key)


def load_split_map(spec: Mapping[str, Any], repo: Path) -> tuple[dict[str, str], str]:
    """The pinned v2 split map (``tools/compose_split_map.py``): repo -> train/val/heldout."""
    path = Path(spec["path"])
    if not path.is_absolute():
        path = repo / path
    sha = check_input(path, repo=repo, want_sha=spec.get("sha256"), what="the split map")
    doc = json.loads(path.read_text())
    repos = doc.get("repos")
    if not isinstance(repos, dict) or not repos:
        raise Refusal(f"the split map {path} has no repos")
    return {str(k): str(v) for k, v in repos.items()}, sha


LABEL_CONFLICT_RULE = (
    "the content key ignores label and span: of two rows with one content, the first row is "
    "kept and the later one dropped; when their (class, span) differ the drop is counted here"
)


def content_digest(row: Mapping[str, Any]) -> str | None:
    """The key that makes a re-id'd copy of a row a duplicate. Byte-exact: no whitespace or line
    ending normalisation.

    * a row with ``before`` and ``after``: sha256(before NUL after), R4's registered key;
    * a composed row (``qd-mutate compose``, no ``before``): sha256("diff" NUL diff), the composed
      context the model reads. Its ``after`` is the needle file only and is not the input.

    ``None`` when neither is present; a candidate with no digest is refused by :func:`plan`.
    """
    if isinstance(row.get("before"), str) and isinstance(row.get("after"), str):
        return hashlib.sha256(f"{row['before']}\x00{row['after']}".encode()).hexdigest()
    if isinstance(row.get("diff"), str):
        return hashlib.sha256(f"diff\x00{row['diff']}".encode()).hexdigest()
    return None


def _label(row: Mapping[str, Any]) -> str:
    return json.dumps([row.get("class"), row.get("span")], sort_keys=True)


def _tally(into: dict[str, collections.Counter[str]], row: Mapping[str, Any]) -> None:
    into["class"][str(row.get("class", "?"))] += 1
    into["operator"][str(row.get("operator", "?"))] += 1


def _mix(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    t: dict[str, collections.Counter[str]] = {
        "class": collections.Counter(),
        "operator": collections.Counter(),
    }
    for r in rows:
        _tally(t, r)
    return {k: dict(sorted(v.items())) for k, v in t.items()}


@dataclass
class _ExclusionSet:
    """One pinned exclusion set: its ids, identity keys and content digests."""

    name: str
    match: str
    ids: set[str] = field(default_factory=set)
    identity_keys: set[str] = field(default_factory=set)
    digests: set[str] = field(default_factory=set)

    def hit(self, tail: str, identity_key: str | None, digest: str | None) -> bool:
        """Whether this set drops a candidate. ``tail`` is its un-prefixed, normalised id."""
        if digest is not None and digest in self.digests:
            return True
        if identity_key is not None and identity_key in self.identity_keys:
            return True
        return self.match == "id" and tail in self.ids


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
    # Every corpus declares a non-empty id prefix: the id-keyed val exclusion strips it, so a
    # corpus that did not declare one could pass a prefixed val id through.
    for corpus in doc["candidates"]["corpora"]:
        declared = corpus.get("id_prefix")
        if not isinstance(declared, str) or not declared:
            raise Refusal(
                f"corpus {corpus.get('name')}: no non-empty id_prefix is declared in the "
                "pre-registration; every Arm B corpus must declare one"
            )

    # Per set: how a candidate is matched. ``id`` (the default, and what the v5 val set uses): a
    # candidate whose un-prefixed id tail, identity key or content digest is in the set is dropped.
    # ``content_or_identity_key`` (v5 train): dropped on its content digest or identity key only; an
    # id match alone is a COUNT, because one ``pool_id#n`` names a different mutation of the same
    # file under another seed (Fable, 2026-10-13: identity is content).
    sets: list[_ExclusionSet] = []
    for s in doc["exclusion"]["sets"]:
        path = Path(s["path"])
        inputs[f"exclusion/{s['name']}"] = check_input(
            path, repo=repo, want_sha=s.get("sha256"), what=f"exclusion set {s['name']}"
        )
        match = str(s.get("match", "id"))
        if match not in ("id", "content_or_identity_key"):
            raise Refusal(f"exclusion set {s['name']}: unknown match {match!r}")
        es = _ExclusionSet(name=s["name"], match=match)
        for row in _rows(path, s["name"]):
            for fld in (s["key"], "id"):
                if isinstance(row.get(fld), str):
                    es.ids.add(normalise_key(row[fld]))
            if isinstance(row.get("identity_key"), str):
                es.identity_keys.add(normalise_key(row["identity_key"]))
            d = content_digest(row)
            if d is not None:
                es.digests.add(d)
        sets.append(es)

    split_spec = doc["exclusion"].get("split_map")
    split_map: dict[str, str] = {}
    split_sha = None
    if split_spec is not None:
        split_map, split_sha = load_split_map(split_spec, repo)
        inputs["split_map"] = split_sha
    not_train = 0
    held = set(heldout_repos)
    seen_keys: set[tuple[str, str]] = set()
    digest_label: dict[str, str] = {}
    label_conflicts = 0
    dropped: dict[str, dict[str, collections.Counter[str]]] = {
        name: {"class": collections.Counter(), "operator": collections.Counter()}
        for name in [s.name for s in sets] + ["within"]
    }
    id_corpora: dict[str, set[str]] = {}
    offered = 0
    dup_by_set = {s.name: 0 for s in sets}
    id_collisions_by_set = {s.name: 0 for s in sets if s.match != "id"}
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
            if split_map:
                split = split_map.get(str(row.get("repo")))
                if split is None and not corpus.get("own_repos_train"):
                    raise Refusal(
                        f"corpus {corpus['name']}: repo {row.get('repo')!r} of {row['id']} is not "
                        "in the pinned split map and the corpus is not own_repos_train: refused"
                    )
                if split is not None and split != "train":
                    not_train += 1
                    continue
            digest = content_digest(row)
            if digest is None:
                raise Refusal(
                    f"corpus {corpus['name']}: {row['id']} has no before/after and no diff, so no "
                    "content digest can be computed: a row that cannot be checked is refused, "
                    "never kept"
                )
            ident = normalise_key(row["id"])
            # The exclusion sets are id-keyed on the corpus's own id: strip the prefix the corpus
            # declares, so a prefix cannot take a row out of reach of the v5 val set.
            prefix = str(corpus.get("id_prefix", ""))
            if prefix and not str(row["id"]).startswith(prefix):
                raise Refusal(
                    f"corpus {corpus['name']}: {row['id']} does not begin with the corpus's "
                    f"declared id_prefix {prefix!r}: refused whole"
                )
            tail = normalise_key(str(row["id"])[len(prefix) :])
            idk = row.get("identity_key")
            idk_tail = (
                normalise_key(idk[len(prefix) :] if prefix and idk.startswith(prefix) else idk)
                if isinstance(idk, str)
                else None
            )
            hit = next((s.name for s in sets if s.hit(tail, idk_tail, digest)), None)
            if hit is not None:
                dup_by_set[hit] += 1
                _tally(dropped[hit], row)
                continue
            within_key = (str(corpus["name"]), ident)
            if within_key in seen_keys or digest in digest_label:
                dup_within += 1
                _tally(dropped["within"], row)
                if digest in digest_label and digest_label[digest] != _label(row):
                    label_conflicts += 1
                continue
            for s in sets:
                if s.match != "id" and tail in s.ids:
                    id_collisions_by_set[s.name] += 1
            seen_keys.add(within_key)
            digest_label[digest] = _label(row)
            id_corpora.setdefault(ident, set()).add(str(corpus["name"]))
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
    candidate_ids = sorted(normalise_key(str(r["id"])) for r in kept)
    report: dict[str, Any] = {
        "schema": "qd-arm-b-plan/v1",
        "preregistration": str(prereg),
        "head": _git(repo, "rev-parse", "HEAD").stdout.strip(),
        "offered": offered,
        "duplicates_by_set": dup_by_set,
        "duplicates_within": dup_within,
        "label_conflicts": label_conflicts,
        "label_conflict_rule": LABEL_CONFLICT_RULE,
        "dropped_by_class_and_operator": {
            name: {k: dict(sorted(v.items())) for k, v in t.items()} for name, t in dropped.items()
        },
        "kept_by_class_and_operator": _mix(kept),
        "id_collisions_by_set": id_collisions_by_set,
        "id_collisions_across_corpora": sum(1 for c in id_corpora.values() if len(c) > 1),
        "ids_unique": len(candidate_ids) == len(set(candidate_ids)),
        "over_scan_cap": over_cap,
        "kept": len(kept),
        "leakage": {
            "exact_identity": {"state": "ran", "sets": [s.name for s in sets]},
            "split_map": (
                {"state": "ran", "sha256": split_sha, "dropped_not_train_split": not_train}
                if split_map
                else {"state": "not_run", "reason": "the pre-registration names no split map"}
            ),
            "heldout_exact_keys": {
                "state": "not_run",
                "reason": "no held-out key set is pinned; held-out repositories are excluded by "
                "the split map and the own-repos manifest's held-out list, not by key",
            },
            "v6_split_priority_dedupe": {
                "state": "not_run",
                "reason": "run by the shard pipeline under --v6-data-rules, not by this driver",
            },
            "containment": containment,
        },
        "inputs": inputs,
        "candidates_sha256": hashlib.sha256(body).hexdigest(),
        "candidate_ids": candidate_ids,
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
    plan_dir = Path(args.plan_dir)
    plan_doc = json.loads((plan_dir / "plan.json").read_text())
    if _sha256_file(plan_dir / "candidates.jsonl") != plan_doc["candidates_sha256"]:
        raise Refusal(f"{plan_dir}/candidates.jsonl no longer hashes to the plan's sha256")
    if not plan_doc.get("ids_unique", False):
        raise Refusal(
            "the plan's candidate ids are not unique (two corpora share an id): a verdict's "
            "row_id could not be traced to one candidate; rebuild the corpora with distinct "
            "--id-prefix values"
        )
    lines = [json.loads(line) for line in verdicts.read_text().splitlines() if line.strip()]
    planned = set(plan_doc["candidate_ids"])
    foreign = sorted(
        {str(v["row_id"]) for v in lines if normalise_key(str(v["row_id"])) not in planned}
    )
    if foreign:
        raise Refusal(
            f"{len(foreign)} scored row(s) are not in the plan (first: {foreign[0]}): the scorer "
            "ran over another shard set"
        )
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
    s.add_argument(
        "--plan-dir",
        type=Path,
        required=True,
        help="the `plan` output directory the scored shard set was built from",
    )
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
