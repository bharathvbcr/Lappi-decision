"""Write the allowlist ``qd-noul-rows generate`` reads: the train-split units it may draw from.

A thin binding over ``qd_data.defect_class.noul_allowlist``, which runs the canonical split
(``qd_data.split.assign_repo`` and ``squad_title_family``) -- the one thing the Rust generator
must not re-implement, because a second copy of "which titles are held out" is the copy that
drifts into a rule-3 breach. Everything else -- reading the sources, choosing, rendering and
scrambling rows -- is the generator's (``crates/qd-mutate/src/noul_rows``).

    cargo run -p qd-mutate --bin qd-noul-rows -- units > units.json
    python tools/noul_allowlist.py --squad <squad train.jsonl> \
        --pool data/pool/commitpackft-pool-v2.jsonl --download data/pool/commitpackft \
        --template-units units.json --out allowlist.json
    cargo run -p qd-mutate --bin qd-noul-rows -- generate --allowlist allowlist.json \
        --squad <squad train.jsonl> --pool data/pool/commitpackft-pool-v2.jsonl \
        --out data/pool/defect-noul-v1

The allowlist is the generator's filter, not the rule-3 guarantee: ``load_noul_rows``
re-derives every row's split at the training run's own seed and refuses the corpus on any row
outside ``train``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_data.config import DataConfig
from qd_data.defect_class import noul_allowlist
from qd_train.replay import write_text_atomic

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--squad", type=Path, required=True, help="SQuAD v2 train JSONL")
    parser.add_argument("--pool", type=Path, required=True, help="the commitpackft pool JSONL")
    parser.add_argument(
        "--download", type=Path, required=True,
        help="the bigcode/commitpackft download the pool's licences are joined from",
    )
    parser.add_argument(
        "--template-units", type=Path, required=True,
        help="the JSON `qd-noul-rows units` printed",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; refusing to overwrite it")
    allow = noul_allowlist(
        squad=args.squad, pool=args.pool, download_root=args.download,
        template_units=json.loads(args.template_units.read_text(encoding="utf-8")),
        config=DataConfig(), repo_root=REPO,
    )
    write_text_atomic(args.out, json.dumps(allow, sort_keys=True) + "\n")
    print(
        f"allowlist {args.out}: {len(allow['squad']['titles'])} SQuAD titles "
        f"(excluded {allow['squad']['excluded']}), {len(allow['pool']['files'])} pool files "
        f"(excluded {allow['pool']['excluded']}), {len(allow['templates']['units'])} template "
        f"units (excluded {allow['templates']['excluded']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
