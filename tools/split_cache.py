"""``--split-cache DIR``: the split rebuild's result, stored once and read back only after checks.

``tools/real_ft_run.py`` rebuilds the train and val rows its shard set was written from on
every start. On the v5 data this takes ~202 s of a ~215 s Mac prelude, and the GPU is idle
for all of it (AUDIT/v6-startup-2026-10-04). Two processes produced byte-identical pickles of
that result even though their string-hash seeds differed (probe-runs.txt). So the result is a
value of its inputs and the code that read them. This module keys it by exactly those inputs
and that code, and stores it.

Opt-in and off by default. Everything here is orchestration around a Python object graph
(``qd_data.rows.DataRow``) for a Python consumer, so it is Python. The rows are serialized
with ``pickle``, which is code execution on load. What stands between another local user and
that is the ownership, mode and symlink checks below, made on the open directory and file
descriptors at the moment of the read. The restricted unpickler adds a second guard: it
resolves only classes from ``qd_data``/``qd_train`` and a short builtins list. The sha256
sidecar is an INTEGRITY check (truncation, bit rot, a half-finished writer). It is not a
security boundary: whoever can write the entry can write its sidecar too.

The contract, condition by condition (the lead's "Go", HANDOFF/v6-startup-2026-10-04.md):

* **Key.** sha256 over:
  - this schema;
  - the interpreter (``sys.version``, machine) and every installed distribution's
    name and version;
  - every ``*.py`` under the repository's ``python/`` and ``tools/``, by content;
  - ``QD_PREP_BIN``'s binary, by content;
  - every argument the rebuild receives: a path by the content of the file or tree it names;
    anything else by its canonical JSON. A type that has no canonical form is refused, never
    stringified.
  - ``extra_inputs``: the files and trees the rebuild reads that no argument names, and
    ``extra_facts``: anything else its output depends on (a subprocess's version). Their
    owner is the caller, which knows its rebuild.

  Any code or input change is a different key, so it misses.
* **Atomic.** Each file is written to a fresh ``O_EXCL|O_NOFOLLOW`` 0600 temp file in the
  cache directory, fsynced, renamed over its final name, and then the directory is fsynced.
  The entry goes first and its sidecar second. Without its sidecar, an entry is a miss. Two
  writers of one key rename identical bytes over each other. If the pickle ever stopped being
  deterministic, a reader holding one writer's sidecar and the other's entry gets a sha
  mismatch, so the outcome is ``corrupt``, never wrong rows.
* **Verified before use.** Before any unpickling:
  - the directory, sidecar and entry are checked (owner, mode, not a symlink, a regular
    file);
  - the entry is the sidecar's exact length;
  - its sha256 is the sidecar's, streamed from the same descriptor that is then read.

  After unpickling, the rows are checked against the build:
  - row counts against ``data/pool/{train,val}.json``'s ``n_rows``;
  - every val row's id and content hash against ``val.json``;
  - rule 3.

  The rebuilt rows pass the same checks before they are stored, so an entry that could never
  verify is never written.
* **Rule 3** (CLAUDE.md). ``DataRow`` carries no file path, so rule 3 is checked on two sides:
  - every row's ``family_id``, against ``DataConfig.held_out_families``;
  - every file the key covers, through ``qd_train.data_access.assert_path_not_held_out``, the
    check ``qd_data.defect_class`` runs on its own reads.

  Both are recorded in the sidecar when the entry is written and both run again when it is
  loaded. A violation raises ``HeldOutViolation``. It is never rebuilt around.
* **Bounded.** At most :data:`MAX_ENTRIES` keys per directory. Only after a successful write
  are the oldest keys by mtime evicted, and every eviction is logged. Nothing else is deleted,
  except this writer's own temp file when its own write fails. Temp files a killed writer
  left behind are counted and named in the log and left in place.
* **RSS.** Entries stream through a hashing writer (``pickle.dump`` to a file) and through a
  buffered reader (``Unpickler.load``), never as one 1.7 GB bytes object. A load that fails
  its checks returns its rows to the allocator before the caller rebuilds, so rows that were
  loaded and rows that were rebuilt are never both live.
"""

from __future__ import annotations

import dataclasses
import enum
import gc
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import re
import secrets
import stat
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Final, Literal

from qd_data.config import DataConfig
from qd_data.errors import HeldOutViolation
from qd_data.rows import DataRow, row_content_hash
from qd_train.data_access import assert_path_not_held_out
from qd_train.tristate import NotRun, Ran, TriState

SCHEMA: Final[str] = "qd-split-cache/v1"
PICKLE_PROTOCOL: Final[int] = 5
#: Keys a directory may hold. Two lanes of one box share a key, and the previous code
#: revision's entry is the one a rollback reads; a third is disk and nothing else (1.7 GB each
#: on the v5 data).
MAX_ENTRIES: Final[int] = 2
ENTRY_SUFFIX: Final[str] = ".rows.pkl"
META_SUFFIX: Final[str] = ".meta.json"
#: An entry larger than this is refused on write and on read: v5's is 1.7 GB.
MAX_ENTRY_BYTES: Final[int] = 16 << 30
MAX_META_BYTES: Final[int] = 16 << 20
#: Bounds on one keyed tree: a path argument naming a directory far bigger than any input the
#: rebuild reads is a mistake, and hashing it would be minutes of silence.
MAX_TREE_FILES: Final[int] = 200_000
MAX_INPUT_BYTES: Final[int] = 64 << 30
_HASH_CHUNK: Final[int] = 8 << 20
_KEY_RE: Final[str] = "[0-9a-f]{64}"
_FINAL_RE: Final[re.Pattern[str]] = re.compile(
    rf"^({_KEY_RE})({re.escape(ENTRY_SUFFIX)}|{re.escape(META_SUFFIX)})$"
)
_TMP_RE: Final[re.Pattern[str]] = re.compile(
    rf"^\.({_KEY_RE})({re.escape(ENTRY_SUFFIX)}|{re.escape(META_SUFFIX)})\.tmp-\d+-[0-9a-f]{{16}}$"
)
#: Module prefixes the restricted unpickler resolves classes from: where DataRow and the
#: request, slot and gold types it holds are defined.
_ALLOWED_MODULE_PREFIXES: Final[tuple[str, ...]] = ("qd_data.", "qd_train.")
#: Builtin classes a row graph may name by global (protocol 5 has opcodes for the containers
#: and scalars it holds, so these are what is left). Classes only: no function is resolvable.
_ALLOWED_BUILTINS: Final[frozenset[tuple[str, str]]] = frozenset({
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "bytearray"),
    ("builtins", "complex"), ("builtins", "range"), ("builtins", "slice"),
})

State = Literal["hit", "miss", "corrupt"]


class CacheDirRefused(SystemExit):
    """The cache directory itself is not one this run may read pickles from."""


class KeyUnavailable(Exception):
    """The key could not be computed, so the cache cannot answer and is not used."""


Say = Callable[[str], None]


def _say(text: str) -> None:
    print(text, flush=True)


# --- the directory, opened once and checked on its descriptor ---------------------------------


def _check_owned(st: os.stat_result, what: str) -> str | None:
    """Why ``st`` may not be trusted, or ``None``: the running user owns it and nobody else
    can write it."""
    if st.st_uid != os.getuid():
        return f"{what} is owned by uid {st.st_uid}, not the running user ({os.getuid()})"
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return f"{what} is group- or world-writable (mode {stat.S_IMODE(st.st_mode):o})"
    return None


@contextmanager
def open_dir(path: Path) -> Iterator[int]:
    """The cache directory as an ``O_DIRECTORY|O_NOFOLLOW`` descriptor, after its checks.

    Created 0700 when absent (its parent must exist). Refused with
    :class:`CacheDirRefused` when it is a symlink or not a directory, or when it is not owned
    by the running user, or when it is group- or world-writable. Every later open, rename and
    unlink is relative to this descriptor. If the directory is renamed or swapped after the
    check, nothing moves with it.
    """
    path = Path(path)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise CacheDirRefused(f"--split-cache {path}: cannot create it ({exc})") from exc
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise CacheDirRefused(f"--split-cache {path}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise CacheDirRefused(
            f"--split-cache {path} is a symlink. The cache reads pickles, which run code on "
            "load, so it reads them only from a directory named directly"
        )
    if not stat.S_ISDIR(st.st_mode):
        raise CacheDirRefused(f"--split-cache {path} is not a directory")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise CacheDirRefused(f"--split-cache {path}: cannot open it ({exc})") from exc
    try:
        why = _check_owned(os.fstat(fd), f"--split-cache {path}")
        if why is not None:
            raise CacheDirRefused(
                f"{why}. The cache reads pickles, which run code on load: a directory "
                "another user can write is one they can plant code in"
            )
        yield fd
    finally:
        os.close(fd)


def check_dir(path: Path) -> None:
    """The directory checks of :func:`open_dir`, at argv time, so a bad --split-cache refuses
    in seconds rather than after the rebuild's inputs are hashed."""
    with open_dir(path):
        pass


# --- the key ------------------------------------------------------------------------------------


def _canonical(value: object) -> object:
    """A JSON value for a non-path argument, or ``KeyUnavailable`` -- never ``str(value)``,
    which would key two different objects alike whenever their reprs agree."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, enum.Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": _canonical(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {f.name: _canonical(getattr(value, f.name))
                       for f in dataclasses.fields(value)},
        }
    if isinstance(value, Mapping):
        items = [(_canonical(k), _canonical(v)) for k, v in value.items()]
        return {"mapping": sorted(([k, v] for k, v in items), key=json.dumps)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((_canonical(v) for v in value), key=json.dumps)}
    raise KeyUnavailable(
        f"a {type(value).__module__}.{type(value).__qualname__} has no canonical form here"
    )


def _sha256_fd(fd: int, limit: int) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    while True:
        block = os.read(fd, _HASH_CHUNK)
        if not block:
            return h.hexdigest(), n
        n += len(block)
        if n > limit:
            raise KeyUnavailable(f"more than {limit} bytes")
        h.update(block)


class _Hasher:
    """One key's file hashing: each real file read once however many inputs name it, every
    byte counted against :data:`MAX_INPUT_BYTES`, every file recorded as covered."""

    def __init__(self) -> None:
        self.budget = MAX_INPUT_BYTES
        self.memo: dict[str, str] = {}

    @property
    def covered(self) -> tuple[str, ...]:
        return tuple(sorted(self.memo))

    def file(self, path: Path) -> str:
        real = os.path.realpath(path)
        found = self.memo.get(real)
        if found is None:
            fd = os.open(real, os.O_RDONLY)
            try:
                found, n = _sha256_fd(fd, self.budget)
            finally:
                os.close(fd)
            self.budget -= n
            self.memo[real] = found
        return found

    def input(self, path: Path) -> dict[str, object]:
        """``path`` by content: a file's sha256, or a tree's digest over every file under it
        (symlinks followed, each directory walked once) by relative path and sha256."""
        path = Path(path)
        if not path.exists():
            return {"absent": str(path)}
        if path.is_file():
            return {"file": str(path), "sha256": self.file(path)}
        if not path.is_dir():
            raise KeyUnavailable(f"{path} is neither a file nor a directory")
        lines: list[str] = []
        seen: set[tuple[int, int]] = set()
        n_files = 0
        for top, dirs, files in os.walk(path, followlinks=True):
            st = Path(top).stat()
            if (st.st_dev, st.st_ino) in seen:
                dirs[:] = []
                continue
            seen.add((st.st_dev, st.st_ino))
            dirs.sort()
            for name in sorted(files):
                full = Path(top) / name
                if not full.is_file():
                    continue
                n_files += 1
                if n_files > MAX_TREE_FILES:
                    raise KeyUnavailable(f"{path} holds more than {MAX_TREE_FILES} files")
                lines.append(f"{full.relative_to(path).as_posix()}\0{self.file(full)}\n")
        return {"tree": str(path), "files": n_files,
                "sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()}


def code_digest(repo: Path) -> str:
    """Every ``*.py`` under ``repo``'s ``python/`` and ``tools/``, by relative path and content."""
    h = hashlib.sha256()
    n = 0
    for sub in ("python", "tools"):
        root = Path(repo) / sub
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            n += 1
            h.update(f"{path.relative_to(repo).as_posix()}\0".encode())
            h.update(hashlib.sha256(path.read_bytes()).digest())
    if n == 0:
        raise KeyUnavailable(f"no *.py under {repo}/python or {repo}/tools: not the repository")
    return h.hexdigest()


def environment_parts() -> dict[str, object]:
    """The interpreter and every installed distribution: a library upgrade is a code change."""
    dists = sorted({f"{d.metadata.get('Name', '?')}=={d.version}"
                    for d in importlib.metadata.distributions()})
    return {
        "python": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "machine": platform.machine(),
        "distributions_sha256": hashlib.sha256("\n".join(dists).encode()).hexdigest(),
        "n_distributions": len(dists),
    }


@dataclasses.dataclass(frozen=True)
class Key:
    digest: str
    #: What was hashed, as JSON: kept in the sidecar so a miss can be explained.
    parts: dict[str, object]
    #: The real path of every file the key hashed by content: rule 3's path half runs on these.
    covered: tuple[str, ...]


def compute_key(
    kwargs: Mapping[str, object], *, extra_inputs: Mapping[str, Path], repo: Path,
    extra_facts: Mapping[str, object] | None = None, prep_bin_env: str = "QD_PREP_BIN",
) -> Key:
    """The key of one rebuild: see the module docstring. Raises :class:`KeyUnavailable`."""
    hasher = _Hasher()
    facts = {k: _canonical(v) for k, v in sorted((extra_facts or {}).items())}
    try:
        args: dict[str, object] = {}
        for name in sorted(kwargs):
            value = kwargs[name]
            args[name] = hasher.input(value) if isinstance(value, Path) else _canonical(value)
        extra = {label: hasher.input(p) for label, p in sorted(extra_inputs.items())}
        named = os.environ.get(prep_bin_env, "")
        prep: dict[str, object] = (
            {"unset": prep_bin_env} if not named else hasher.input(Path(named))
        )
        code = code_digest(repo)
    except OSError as exc:
        raise KeyUnavailable(f"an input could not be read: {exc}") from exc
    parts: dict[str, object] = {
        "schema": SCHEMA,
        "pickle_protocol": PICKLE_PROTOCOL,
        "environment": environment_parts(),
        "code_sha256": code,
        "prep_bin": prep,
        "arguments": args,
        "extra_inputs": extra,
        "extra_facts": facts,
    }
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return Key(hashlib.sha256(blob).hexdigest(), parts, hasher.covered)


# --- what the rows are checked against ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Manifests:
    """What the build wrote beside its shards: ``data/pool/{train,val}.json``."""

    n_train: int
    n_val: int
    val_hashes: dict[str, str]
    data_snapshot_hash: str


def read_manifests(out: Path) -> Manifests:
    pool = Path(out) / "data" / "pool"
    raw = {}
    for name in ("train", "val"):
        path = pool / f"{name}.json"
        try:
            raw[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KeyUnavailable(f"{path} cannot be read, so no row can be checked: {exc}") from exc
    val_hashes = {str(e["row_id"]): str(e["content_hash"]) for e in raw["val"]["entries"]}
    return Manifests(
        n_train=int(raw["train"]["n_rows"]), n_val=int(raw["val"]["n_rows"]),
        val_hashes=val_hashes, data_snapshot_hash=str(raw["train"].get("data_snapshot_hash")),
    )


def check_held_out(
    train: Sequence[DataRow], val: Sequence[DataRow], *, covered: Sequence[str],
    config: DataConfig, repo: Path,
) -> dict[str, object]:
    """Rule 3 on both sides, raising :class:`HeldOutViolation`; the record of what ran."""
    for path in covered:
        assert_path_not_held_out(Path(path), config=config, repo_root=repo)
    held = set(config.held_out_families)
    for split_name, rows in (("train", train), ("val", val)):
        offending = sorted({r.family_id for r in rows if r.family_id in held})
        if offending:
            raise HeldOutViolation(
                expected=f"no {split_name} row from a held-out task family {sorted(held)}",
                actual=offending,
                detail="the split cache's rows. CLAUDE.md rule 3: never read by training",
            )
    return {
        "held_out_families": sorted(held),
        "rows_checked": len(train) + len(val),
        "paths_checked": len(covered),
        "checks": ["family_id not in DataConfig.held_out_families",
                   "qd_train.data_access.assert_path_not_held_out on every keyed file"],
    }


def check_rows(train: Sequence[DataRow], val: Sequence[DataRow], m: Manifests) -> str | None:
    """Why these rows are not the build's, or ``None``."""
    if not all(isinstance(r, DataRow) for r in train) or not all(
        isinstance(r, DataRow) for r in val
    ):
        return "a row is not a qd_data.rows.DataRow"
    if len(train) != m.n_train:
        return f"{len(train)} train rows but data/pool/train.json says n_rows {m.n_train}"
    if len(val) != m.n_val:
        return f"{len(val)} val rows but data/pool/val.json says n_rows {m.n_val}"
    if len(m.val_hashes) != m.n_val:
        return f"val.json lists {len(m.val_hashes)} distinct row ids for n_rows {m.n_val}"
    seen: set[str] = set()
    for row in val:
        want = m.val_hashes.get(row.row_id)
        if want is None:
            return f"val row {row.row_id!r} is not in val.json"
        if row.row_id in seen:
            return f"val row {row.row_id!r} appears twice"
        seen.add(row.row_id)
        if row_content_hash(row) != want:
            return f"val row {row.row_id!r}'s content hash is not val.json's"
    return None


# --- the two files, written atomically relative to the directory descriptor -----------------


class _HashingWriter:
    def __init__(self, fh: IO[bytes]) -> None:
        self.fh = fh
        self.sha = hashlib.sha256()
        self.n = 0

    def write(self, data: bytes | bytearray | memoryview) -> int:
        size = memoryview(data).nbytes
        self.n += size
        if self.n > MAX_ENTRY_BYTES:
            raise OSError(f"the entry passed {MAX_ENTRY_BYTES} bytes")
        self.sha.update(data)
        return self.fh.write(data)


def _write_atomic(dfd: int, final: str, fill: Callable[[IO[bytes]], None]) -> None:
    tmp = f".{final}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
    try:
        with os.fdopen(fd, "wb", buffering=1 << 20) as fh:
            fill(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp, final, src_dir_fd=dfd, dst_dir_fd=dfd)
    except BaseException:
        # This writer's own temp file, and only it: the one deletion besides eviction.
        with suppress(FileNotFoundError):
            os.unlink(tmp, dir_fd=dfd)
        raise
    os.fsync(dfd)


@dataclasses.dataclass(frozen=True)
class Stored:
    sha256: str
    n_bytes: int
    write_s: float


def store(
    dfd: int, key: Key, train: list[DataRow], val: list[DataRow], *, manifests: Manifests,
    held_out: Mapping[str, object],
) -> Stored:
    """Write the entry, then its sidecar."""
    writer: list[_HashingWriter] = []

    def fill_entry(fh: IO[bytes]) -> None:
        w = _HashingWriter(fh)
        writer.append(w)
        pickle.dump((train, val), w, protocol=PICKLE_PROTOCOL)

    began = time.perf_counter()
    _write_atomic(dfd, key.digest + ENTRY_SUFFIX, fill_entry)
    stored = Stored(writer[0].sha.hexdigest(), writer[0].n, time.perf_counter() - began)
    meta = {
        "schema": SCHEMA,
        "key": key.digest,
        "key_parts": key.parts,
        "entry": {"name": key.digest + ENTRY_SUFFIX, "sha256": stored.sha256,
                  "bytes": stored.n_bytes},
        "rows": {"train": len(train), "val": len(val)},
        "manifests": {"n_train": manifests.n_train, "n_val": manifests.n_val,
                      "data_snapshot_hash": manifests.data_snapshot_hash},
        "covered": list(key.covered),
        "held_out": dict(held_out),
        "written_at": datetime.now(UTC).isoformat(),
        "writer_pid": os.getpid(),
        "entry_write_s": round(stored.write_s, 3),
    }
    body = json.dumps(meta, sort_keys=True, indent=1).encode("utf-8")
    if len(body) > MAX_META_BYTES:
        raise OSError(f"the sidecar is {len(body)} bytes, past {MAX_META_BYTES}")
    _write_atomic(dfd, key.digest + META_SUFFIX, lambda fh: fh.write(body))
    return stored


# --- reading -------------------------------------------------------------------------------------


class _Unpickler(pickle.Unpickler):
    """Resolves only classes from qd_data/qd_train and a few builtins: no function, so no
    ``os.system``, ``subprocess`` or ``eval`` reachable through REDUCE."""

    def find_class(self, module: str, name: str) -> type:
        if (module, name) not in _ALLOWED_BUILTINS and not module.startswith(
            _ALLOWED_MODULE_PREFIXES
        ):
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a split cache")
        found = super().find_class(module, name)
        if not isinstance(found, type):
            raise pickle.UnpicklingError(f"global {module}.{name} is not a class")
        return found


def _open_checked(dfd: int, name: str, what: str) -> tuple[int, os.stat_result] | str:
    """An ``O_NOFOLLOW`` descriptor for ``name`` that passed the ownership checks, or why not.
    ``FileNotFoundError`` propagates: absence is a miss, not a refusal."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        return f"{what} {name} refused: {exc} (a symlink is never followed)"
    st = os.fstat(fd)
    why = (f"{what} {name} is not a regular file" if not stat.S_ISREG(st.st_mode)
           else _check_owned(st, f"{what} {name}"))
    if why is not None:
        os.close(fd)
        return why + "; refused before any read"
    return fd, st


@dataclasses.dataclass
class Loaded:
    state: State
    reason: str
    train: list[DataRow] | None = None
    val: list[DataRow] | None = None
    entry_sha256: str = ""
    written_at: str = ""


def _as_rows(value: object) -> list[DataRow] | None:
    """``value`` as a list of DataRow, or ``None`` when it is anything else."""
    if not isinstance(value, list):
        return None
    rows = [r for r in value if isinstance(r, DataRow)]
    return rows if len(rows) == len(value) else None


def load(
    dfd: int, key: Key, *, manifests: Manifests, config: DataConfig, repo: Path,
) -> Loaded:
    """``hit`` with rows that passed every check, or ``miss``/``corrupt`` with the reason and
    no rows. Rule 3's violations raise."""
    meta_name, entry_name = key.digest + META_SUFFIX, key.digest + ENTRY_SUFFIX
    try:
        opened = _open_checked(dfd, meta_name, "sidecar")
    except FileNotFoundError:
        return Loaded("miss", "no entry for this key")
    if isinstance(opened, str):
        return Loaded("corrupt", opened)
    fd, st = opened
    try:
        if st.st_size > MAX_META_BYTES:
            return Loaded("corrupt", f"sidecar {meta_name} is {st.st_size} bytes")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    try:
        meta = json.loads(raw)
        entry = meta["entry"]
        want_sha, want_bytes = str(entry["sha256"]), int(entry["bytes"])
        held_record = meta["held_out"]
        covered = [str(p) for p in meta["covered"]]
    except (ValueError, KeyError, TypeError) as exc:
        return Loaded("corrupt", f"sidecar {meta_name} does not parse: {exc!r}")
    if meta.get("schema") != SCHEMA or meta.get("key") != key.digest:
        return Loaded("corrupt", f"sidecar {meta_name} names schema {meta.get('schema')!r} "
                                 f"key {meta.get('key')!r}")
    if not isinstance(held_record, dict) or held_record.get("rows_checked") is None:
        return Loaded("corrupt", "the sidecar records no rule-3 check from the write")
    try:
        opened = _open_checked(dfd, entry_name, "entry")
    except FileNotFoundError:
        return Loaded("miss", f"sidecar {meta_name} without its entry")
    if isinstance(opened, str):
        return Loaded("corrupt", opened)
    fd, st = opened
    rows: object = None
    try:
        if st.st_size != want_bytes or st.st_size > MAX_ENTRY_BYTES:
            return Loaded("corrupt", f"entry is {st.st_size} bytes; the sidecar says {want_bytes}")
        try:
            got, _n = _sha256_fd(fd, MAX_ENTRY_BYTES)
        except KeyUnavailable as exc:
            return Loaded("corrupt", f"entry: {exc}")
        if got != want_sha:
            return Loaded("corrupt", f"entry sha256 {got} but the sidecar says {want_sha}")
        os.lseek(fd, 0, os.SEEK_SET)
        fh = os.fdopen(os.dup(fd), "rb", buffering=1 << 20)
        try:
            with fh:
                rows = _Unpickler(fh).load()
        except Exception as exc:  # any failure to unpickle is a corrupt entry
            return Loaded("corrupt", f"entry did not unpickle: {exc!r}")
    finally:
        os.close(fd)
    if not (isinstance(rows, tuple) and len(rows) == 2):
        return Loaded("corrupt", "entry is not a (train rows, val rows) pair")
    train, val = _as_rows(rows[0]), _as_rows(rows[1])
    del rows
    if train is None or val is None:
        return Loaded("corrupt", "entry holds something other than two lists of DataRow")
    why = check_rows(train, val, manifests)
    if why is not None:
        return Loaded("corrupt", why)
    check_held_out(train, val, covered=sorted(set(covered) | set(key.covered)),
                   config=config, repo=repo)
    return Loaded("hit", "", train, val, entry_sha256=want_sha,
                  written_at=str(meta.get("written_at")))


# --- eviction -------------------------------------------------------------------------------------


def _names(dfd: int) -> Iterator[str]:
    with os.scandir(dfd) as it:
        for entry in it:
            yield entry.name


def evict(dfd: int, *, keep: str, say: Say = _say) -> list[str]:
    """Bring the directory to :data:`MAX_ENTRIES` keys, oldest first by mtime, never ``keep``.
    Unknown files are not touched; leftover temp files are named and left."""
    mtimes: dict[str, float] = {}
    stale: list[str] = []
    for name in _names(dfd):
        m = _FINAL_RE.match(name)
        if m is not None:
            st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            mtimes[m.group(1)] = max(mtimes.get(m.group(1), 0.0), st.st_mtime)
        elif _TMP_RE.match(name):
            stale.append(name)
    evicted: list[str] = []
    for k in sorted((k for k in mtimes if k != keep), key=lambda k: mtimes[k]):
        if len(mtimes) - len(evicted) <= MAX_ENTRIES:
            break
        freed = 0
        for suffix in (META_SUFFIX, ENTRY_SUFFIX):
            try:
                freed += os.stat(k + suffix, dir_fd=dfd, follow_symlinks=False).st_size
                os.unlink(k + suffix, dir_fd=dfd)
            except FileNotFoundError:
                pass
        evicted.append(k)
        when = datetime.fromtimestamp(mtimes[k], UTC).isoformat(timespec="seconds")
        say(f"split cache: EVICTED entry {k} (mtime {when}, {freed} bytes): the directory "
            f"holds more than {MAX_ENTRIES} keys and it is the oldest")
    if evicted:
        os.fsync(dfd)
    if stale:
        say(f"split cache: {len(stale)} temp file(s) a writer left unfinished, left in place "
            f"(not this run's to delete): {sorted(stale)}")
    return evicted


# --- the one entry point --------------------------------------------------------------------------


def _metric(state: State, key: str, detail: str, *, passed: bool) -> TriState:
    """``passed``: the cache either served rows that passed every check or stored the
    rebuild's. ``False`` when an entry failed its checks (``corrupt``) or the rebuilt rows could
    not be stored. Not a recipe key."""
    return Ran(passed=passed, value=state, detail=f"key {key}; {detail}")


def cached_split_rows(
    cache_dir: Path, *, kwargs: Mapping[str, object],
    rebuild: Callable[[], tuple[list[DataRow], list[DataRow]]],
    extra_inputs: Mapping[str, Path], out: Path, config: DataConfig, repo: Path,
    extra_facts: Mapping[str, object] | None = None, say: Say = _say,
) -> tuple[list[DataRow], list[DataRow], TriState]:
    """``(train rows, val rows, the run's split_cache metric)``: read from the cache on a
    verified hit, otherwise from ``rebuild()``, stored for the next run."""
    with open_dir(cache_dir) as dfd:
        began = time.perf_counter()
        try:
            key = compute_key(kwargs, extra_inputs=extra_inputs, repo=repo,
                              extra_facts=extra_facts)
            manifests = read_manifests(out)
        except KeyUnavailable as exc:
            say(f"split cache: NOT USED -- the key cannot be computed ({exc}); rebuilding "
                "uncached")
            train, val = rebuild()
            return train, val, NotRun(reason=f"the split cache could not key this run: {exc}")
        key_s = time.perf_counter() - began
        t = time.perf_counter()
        got = load(dfd, key, manifests=manifests, config=config, repo=repo)
        if got.state == "hit" and got.train is not None and got.val is not None:
            say(f"split cache: READ FROM the split cache {cache_dir}/{key.digest}{ENTRY_SUFFIX} "
                f"-- {len(got.train)} train + {len(got.val)} val rows, NOT rebuilt by this run "
                f"(written {got.written_at}; key {key_s:.1f} s, load and checks "
                f"{time.perf_counter() - t:.1f} s)")
            detail = (f"READ FROM the split cache, not rebuilt; entry sha256 "
                      f"{got.entry_sha256} written {got.written_at}; checked: owner/mode/"
                      f"symlink, length, sha256, row counts vs data/pool/{{train,val}}.json, "
                      f"every val row vs val.json, rule 3")
            return got.train, got.val, _metric("hit", key.digest, detail, passed=True)
        state, reason = got.state, got.reason
        del got
        if state == "corrupt":
            gc.collect()  # the failed load's rows are gone before the rebuild's arrive
            say(f"split cache: CORRUPT entry {key.digest}: {reason}. Rebuilding and writing a "
                "fresh entry")
        else:
            say(f"split cache: miss for key {key.digest} ({reason}); rebuilding")
        train, val = rebuild()
        why = check_rows(train, val, manifests)
        if why is not None:
            say(f"split cache: the rebuilt rows are NOT stored: {why}")
            return train, val, _metric(
                state, key.digest, f"{reason}; rebuilt; not stored: the rebuild's own rows "
                f"fail the check a cached copy must pass ({why})", passed=False,
            )
        held = check_held_out(train, val, covered=key.covered, config=config, repo=repo)
        t = time.perf_counter()
        try:
            stored = store(dfd, key, train, val, manifests=manifests, held_out=held)
        except OSError as exc:
            say(f"split cache: the rebuilt rows are NOT stored: {exc}")
            return train, val, _metric(state, key.digest, f"{reason}; rebuilt; not stored: "
                                       f"{exc}", passed=False)
        say(f"split cache: STORED {cache_dir}/{key.digest}{ENTRY_SUFFIX} "
            f"({stored.n_bytes} bytes, entry and sidecar {time.perf_counter() - t:.1f} s; "
            f"key {key_s:.1f} s)")
        evicted = evict(dfd, keep=key.digest, say=say)
        detail = (f"{reason}; rebuilt and stored, entry sha256 {stored.sha256}"
                  + (f"; evicted {evicted}" if evicted else ""))
        return train, val, _metric(state, key.digest, detail, passed=state == "miss")
