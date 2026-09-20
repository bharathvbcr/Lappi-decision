"""Check ``AUDIT/external-surfaces.md``'s checkpoint claims against the LOCAL weights.

Every tensor number in that audit was read from the Hugging Face API before the weights
existed on this host: the audit says so itself, citing
``.../Qwen3.5-2B-Base/raw/main/model.safetensors.index.json``. A remote index is a claim
about a file. This reads the file.

It needs **no torch and no transformers** -- a safetensors file begins with a little-endian
u64 header length followed by that many bytes of JSON describing every tensor's dtype and
shape, so the whole inventory is available from the first few hundred kilobytes. That is
why this one runs in the repo venv while ``tools/bpe_line_start_collapse.py`` cannot.

Measured 2026-09-20, all four claims VERIFIED: 632 tensors, 297 ``model.visual.*``,
15 ``mtp.*``, ``total_size`` 4,548,144,832, and no ``lm_head`` tensor (the embedding is
tied, so it is counted once).

It also reports what the audit could not, because an index gives sizes but this gives
shapes -- the split between what ships and what actually gets trained:

    whole checkpoint    2,274,069,824
      vision tower        331,416,576   14.6%
      MTP block            60,828,160    2.7%
      TEXT MODEL        1,881,825,088   82.8%
        tied embedding    508,559,360   27.0% OF THE TEXT MODEL

That last line is the one that moves a plan number. The audit computed the embedding at
"≈22% of the shipped checkpoint, not of a 2.0B text-only model" and was right to make the
distinction; the figure that a trainable-parameter budget actually needs is 27.0%, because
the vision tower and the MTP block are dropped on load and never trained. The text model
is 1.88B, not 2B.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

DEFAULT_SNAPSHOTS = Path(
    "/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots"
)

#: The claims in AUDIT/external-surfaces.md, section 1, as that document states them.
CLAIMS: dict[str, int] = {
    "tensors total": 632,
    "model.visual.* tensors": 297,
    "mtp.* tensors": 15,
    "metadata.total_size bytes": 4_548_144_832,
}

HIDDEN_SIZE = 2048
VOCAB_SIZE = 248_320


def safetensors_header(path: Path) -> dict:
    """The JSON header of a safetensors file, without reading a byte of tensor data."""
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path}: too short to carry a safetensors header")
        length = struct.unpack("<Q", raw)[0]
        return json.loads(fh.read(length))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None, help="checkpoint snapshot directory")
    args = ap.parse_args()

    if args.snapshot:
        snap = Path(args.snapshot)
    else:
        found = (
            sorted(p for p in DEFAULT_SNAPSHOTS.glob("*") if p.is_dir())
            if DEFAULT_SNAPSHOTS.is_dir()
            else []
        )
        if not found:
            raise SystemExit(
                "no local snapshot of Qwen/Qwen3.5-2B-Base. This tool checks weights that "
                "are on disk; it does not download them, and it reports nothing rather "
                "than reporting a pass it did not measure."
            )
        snap = found[-1]

    shards = sorted(snap.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{snap}: no .safetensors files")
    print(f"snapshot: {snap}")
    print(f"safetensors: {[p.name for p in shards]}\n")

    header: dict = {}
    for p in shards:
        header.update(safetensors_header(p))
    header.pop("__metadata__", None)
    names = list(header)

    measured: dict[str, int] = {
        "tensors total": len(names),
        "model.visual.* tensors": sum(1 for n in names if n.startswith("model.visual.")),
        "mtp.* tensors": sum(1 for n in names if n.startswith("mtp.")),
    }
    index = snap / "model.safetensors.index.json"
    if index.exists():
        measured["metadata.total_size bytes"] = json.loads(index.read_text(encoding="utf-8"))[
            "metadata"
        ]["total_size"]

    print("AUDIT/external-surfaces.md section 1, checked against the local checkpoint:")
    ok = True
    for key, claimed in CLAIMS.items():
        got = measured.get(key)
        if got is None:
            verdict = "NOT MEASURED (no index file); reported as unchecked, not as passing"
            ok = False
        elif got == claimed:
            verdict = "VERIFIED"
        else:
            verdict = f"DIFFERS -- the audit says {claimed:,}"
            ok = False
        shown = f"{got:,}" if isinstance(got, int) else str(got)
        print(f"  {key:30} {shown:>15}   {verdict}")

    tied = not any(n.startswith("lm_head") for n in names)
    print(f"  {'lm_head is a separate tensor':30} {not tied!s:>15}   "
          f"{'VERIFIED (tied, counted once)' if tied else 'DIFFERS -- audit says absent'}")
    ok = ok and tied

    def params(prefix: str | None) -> int:
        total = 0
        for name, spec in header.items():
            if prefix is not None and not name.startswith(prefix):
                continue
            count = 1
            for dim in spec["shape"]:
                count *= dim
            total += count
        return total

    whole = params(None)
    vision = params("model.visual.")
    mtp = params("mtp.")
    text = whole - vision - mtp
    embedding = VOCAB_SIZE * HIDDEN_SIZE

    print("\nparameters by shape (what an index of sizes cannot give):")
    print(f"  whole checkpoint              {whole:>15,}")
    print(f"    vision tower                {vision:>15,}   {100.0 * vision / whole:5.1f}%")
    print(f"    MTP block                   {mtp:>15,}   {100.0 * mtp / whole:5.1f}%")
    print(f"    TEXT MODEL (trained)        {text:>15,}   {100.0 * text / whole:5.1f}%")
    print(f"      tied embedding            {embedding:>15,}   "
          f"{100.0 * embedding / text:5.1f}% OF THE TEXT MODEL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
