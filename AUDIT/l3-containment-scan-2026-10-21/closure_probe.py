import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "python")
import qd_data.fingerprint as fp
from qd_train import exclusions

src = Path("python/qd_data")
with tempfile.TemporaryDirectory() as t:
    for p in src.glob("*.py"):
        shutil.copy(p, Path(t) / p.name)
    fp._package_dir = lambda: Path(t)
    ident = {"sources": "s"}
    base = exclusions.containment_corpus(ident)
    for name in ("render.py", "mixture.py", "loaders.py", "general.py", "split.py"):
        target = Path(t) / name
        orig = target.read_text()
        target.write_text(orig + "\n# edited\n")
        moved = exclusions.containment_corpus(ident) != base
        print(f"{name}: in CONTAINMENT_CONTENT_MODULES={name in fp.CONTAINMENT_CONTENT_MODULES} "
              f"in SHARD_CONTENT_MODULES={name in fp.SHARD_CONTENT_MODULES} attestation moves={moved}")
        target.write_text(orig)
