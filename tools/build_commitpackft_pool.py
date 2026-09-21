"""Convert ``bigcode/commitpackft`` rows into the pool ``qd-mutate`` reads.

This is the link that was never built, and its absence is why every GPU row in this
repository is ``quick``.

``qd_data.sources`` has registered this source as LOADABLE with ``declared_licence="mit"``
since S3. ``qd_data.loaders.parse_commitpackft`` has existed just as long.
``docs/plan-corrections.md`` calls it the primary pool. ``crates/qd-mutate/src/pool.rs``
defines the ``PoolRecord`` it would have to become. Every piece was present and nothing
joined them, so every corpus this project has trained on was built from qwen-decision's own
sources -- which ``rung0_real_run.py`` correctly stamps as a subsample under rule 8, on
every row, unconditionally.

## What this adds that the local-source builder cannot

``qd_data.pool_builder`` walks local trees and stamps every example
``hunk_constrained: false``, because a file on disk has no diff attached. A commitpackft row
carries ``old_contents`` AND ``new_contents``, so the changed line spans are recoverable by
diffing -- and those spans are exactly ``PoolRecord.hunks``. That is the difference between
"every site in the file is fair game" and "the mutation lands inside a hunk a human actually
touched", which ``pool.rs`` says is the point: *"the model learns to read hunks and not file
headers."*

## Licences are filtered per row, not per dataset

The dataset is ``mit``; its rows are not. ``qd_data.sources`` records the trap in as many
words -- three of the thirteen per-row values are not permissive despite the card's prose.
Measured over the four languages fetched here: 63,119 of 69,893 rows are on the permissive
allowlist and 6,774 are not, with ``agpl-3.0`` alone accounting for 4,655 of them. Anything
outside the allowlist is dropped and counted by reason, never admitted "because the dataset
says mit".

## What is counted

Every drop, by reason, in the manifest. A pool that silently discarded half its input and a
pool that kept everything must not produce the same record.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from qd_data.licences import LicenceTier, classify  # noqa: E402
from qd_data.loaders import parse_commitpackft  # noqa: E402

#: A record with no recoverable hunk is dropped rather than written with `hunks: null`.
#: `pool.rs` carries a null as "whole file, every site fair game", which is the
#: local-source builder's case and is NOT what a diffed commit row is. Writing one here
#: would mean this tool's whole reason for existing had silently not applied to that row.
MIN_HUNK_LINES = 1


def changed_line_spans(old: str, new: str) -> list[tuple[int, int]]:
    """The 1-based inclusive line spans of `new` that differ from `old`.

    `difflib.SequenceMatcher` over lines, keeping the `replace` and `insert` opcodes. A
    `delete` contributes no line to the NEW file, so it cannot be pointed at and is not a
    span -- `pool.rs` sizes a span against the source it stores, which is `new_contents`.

    Line numbers are 1-based inclusive at both ends, matching `LineSpan::new(2, 4)` in
    `pool.rs`'s own round-trip test.
    """
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    spans: list[tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
        a=old_lines, b=new_lines, autojunk=False
    ).get_opcodes():
        if tag in ("replace", "insert") and j2 - j1 >= MIN_HUNK_LINES:
            spans.append((j1 + 1, j2))
    return spans


def build(source_dir: Path, out_path: Path, *, max_records: int | None) -> dict[str, object]:
    dropped: Counter[str] = Counter()
    kept = 0
    by_lang: Counter[str] = Counter()
    hunk_lines: list[int] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as sink:
        for path in sorted(source_dir.glob("*.jsonl")):
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                if max_records is not None and kept >= max_records:
                    dropped["max_records reached"] += 1
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    dropped["unparseable json"] += 1
                    continue
                try:
                    row = parse_commitpackft(raw, index=index)
                # Broad by intent: every refusal is counted by exception kind below, so a
                # new failure mode shows up as its own line in the manifest rather than
                # vanishing into a total.
                except Exception as exc:
                    dropped[f"malformed row: {type(exc).__name__}"] += 1
                    continue

                policy = classify(row.licence)
                if policy.tier is not LicenceTier.ALLOW:
                    dropped[f"licence {policy.licence_id} ({policy.tier.name})"] += 1
                    continue

                spans = changed_line_spans(row.old_contents, row.new_contents)
                if not spans:
                    # No line of `new_contents` differs from `old_contents`: a rename, a
                    # mode change, or a pure deletion. Nothing to constrain a mutation to.
                    dropped["no changed line span in new_contents"] += 1
                    continue
                if not row.new_contents.strip():
                    dropped["empty new_contents"] += 1
                    continue

                record = {
                    "id": f"{row.commit}:{row.new_file}",
                    "repo": row.primary_repo,
                    "path": row.new_file,
                    "source": row.new_contents,
                    "hunks": [{"start_line": a, "end_line": b} for a, b in spans],
                }
                sink.write(json.dumps(record, sort_keys=True) + "\n")
                kept += 1
                by_lang[path.stem] += 1
                hunk_lines.append(sum(b - a + 1 for a, b in spans))

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return {
        "out": str(out_path),
        "sha256": digest,
        "records": kept,
        "by_language": dict(sorted(by_lang.items())),
        "dropped_by_reason": dict(dropped.most_common()),
        "dropped_total": sum(dropped.values()),
        "mean_hunk_lines": (sum(hunk_lines) / len(hunk_lines)) if hunk_lines else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO / "data" / "pool" / "commitpackft",
        help="directory of <language>.jsonl files as fetched from the dataset",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "pool" / "commitpackft-pool.jsonl",
        help="the qd-mutate PoolRecord jsonl to write",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "data" / "pool" / "commitpackft-pool.manifest.json",
        help="where the counts go. Tracked even though the pool is not: it is the "
             "provenance record, and .gitignore keeps the data and not this",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="cap the pool. A capped build is reported as capped in the manifest, never "
             "as a complete one",
    )
    args = parser.parse_args(argv)

    if not args.source_dir.is_dir():
        raise SystemExit(
            f"no source directory at {args.source_dir}. Fetch the dataset first: the "
            "languages are data/<lang>/data.jsonl under bigcode/commitpackft, which is "
            "not gated"
        )

    report = build(args.source_dir, args.out, max_records=args.max_records)
    report["capped"] = args.max_records is not None
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {report['records']} PoolRecord(s) to {report['out']}")
    print(f"  sha256           {report['sha256']}")
    print(f"  by language      {report['by_language']}")
    print(f"  mean hunk lines  {report['mean_hunk_lines']:.1f}")
    print(f"  dropped          {report['dropped_total']}")
    for reason, n in list(report["dropped_by_reason"].items())[:8]:  # type: ignore[union-attr]
        print(f"    {n:>7}  {reason}")
    if report["capped"]:
        print("  CAPPED: this is not a complete build of the source")
    print(f"  manifest         {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
