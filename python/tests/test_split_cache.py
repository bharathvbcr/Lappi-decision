"""``tools/split_cache.py``: the split rebuild's result, keyed, stored atomically, checked on read.

Each test names the lead's condition it pins (HANDOFF/v6-startup-2026-10-04.md, "the split-result
cache"):

1. opt-in, with a metric that is not a recipe key: ``test_split_cache_main.py`` (it needs torch);
2. pickles are read only from a directory and files the running user owns and nobody else can
   write, never through a symlink: the ``permissions`` tests;
3. atomic writes, an entry without its sidecar is a miss, and a race between two writers of one
   key is safe: the ``atomic`` and ``race`` tests;
4. a corrupt entry is logged and rebuilt, and a fresh entry is written; its metric says
   ``corrupt`` and why, never ``hit``: the ``corrupt`` tests;
5. bounded, evicting the oldest only after a successful write: the ``eviction`` tests;
6. rule 3, recorded at write and re-checked on load: the ``held_out`` tests;
7. any code or input change misses (the ``key`` tests), and rows that were loaded and rows that
   were rebuilt are never both live (the ``rss`` test).

The rows are a real ``qd_data`` build (mixture, dedupe, split, manifests) of the toy defect corpus
the real_ft_run fixtures use, so ``val.json``'s content hashes are the real ones. No torch.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import split_cache as sc  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID, DEFECT_SOURCE_ID, DefectRow  # noqa: E402
from qd_data.errors import HeldOutViolation  # noqa: E402
from qd_data.manifest import build_manifests  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_data.split import HELD_OUT, split  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

# --- a real build of the toy defect corpus -----------------------------------------------------


def _defect_rows(n_repos: int) -> list[DefectRow]:
    out = []
    for i in range(n_repos):
        for j, cls in enumerate(("stub", "logic", "cosmetic", "clean")):
            diff = (
                f"@@ -1,3 +1,3 @@\n def f{i}_{j}(x):\n-    return x + {j}\n"
                f"+    return {i} * {j}\n     pass\n"
            )
            out.append(DefectRow(
                example_id=f"c{i}_{j}:x.py#0", pool_id=f"c{i}_{j}:x.py", repo=f"org/r{i}",
                path="x.py", symbol="f", arity=1, language="python", mutation_class=cls,
                operator="clean" if cls == "clean" else f"{cls}.op", diff=diff,
                diff_span=None if cls == "clean" else (3, 3), span_refusal=None, licence="mit",
            ))
    return out


@dataclasses.dataclass(frozen=True)
class Build:
    out: Path
    train: list[DataRow]
    val: list[DataRow]


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Build:
    out = tmp_path_factory.mktemp("split-cache-build")
    config = DataConfig()
    mixture = build_mixture(
        {DEFECT_SOURCE_ID: _defect_rows(48)}, config=config, families=[DEFECT_FAMILY_ID]
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    for name, manifest in manifests.items():
        manifest.write(out / "data" / (HELD_OUT if name == HELD_OUT else "pool") / f"{name}.json")
    train = list(split_report.rows_by_split["train"])
    val = list(split_report.rows_by_split["val"])
    assert train and val
    return Build(out, train, val)


@dataclasses.dataclass
class Env:
    """One cache directory, one fake repository (its code is keyed), one input tree."""

    build: Build
    tmp: Path
    cache: Path
    repo: Path
    data: Path
    log: list[str] = dataclasses.field(default_factory=list)
    rebuilds: int = 0

    def kwargs(self, **extra: object) -> dict[str, object]:
        return {"rev": "a" * 40, "max_pairs": 7, "config": DataConfig(),
                "defect_class": self.data / "tree", "pre_dedupe_drops": self.data / "drops.txt",
                "general_record": None, **extra}

    def rebuild(self) -> tuple[list[DataRow], list[DataRow]]:
        self.rebuilds += 1
        return list(self.build.train), list(self.build.val)

    def run(
        self, *, kwargs: dict[str, object] | None = None,
        rebuild: Callable[[], tuple[list[DataRow], list[DataRow]]] | None = None,
        extra_inputs: dict[str, Path] | None = None,
    ) -> tuple[list[DataRow], list[DataRow], object]:
        return sc.cached_split_rows(
            self.cache, kwargs=self.kwargs() if kwargs is None else kwargs,
            rebuild=self.rebuild if rebuild is None else rebuild,
            extra_inputs={"aux": self.data / "aux.json"} if extra_inputs is None else extra_inputs,
            out=self.build.out, config=DataConfig(), repo=self.repo, say=self.log.append,
        )

    def key(self, **kw: object) -> sc.Key:
        return sc.compute_key(self.kwargs(**kw), extra_inputs={"aux": self.data / "aux.json"},
                              repo=self.repo)

    def files(self) -> list[str]:
        return sorted(p.name for p in self.cache.iterdir())


@pytest.fixture
def env(build: Build, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    repo = tmp_path / "repo"
    (repo / "python" / "pkg").mkdir(parents=True)
    (repo / "tools").mkdir()
    (repo / "python" / "pkg" / "m.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "tools" / "t.py").write_text("Y = 2\n", encoding="utf-8")
    data = tmp_path / "data"
    (data / "tree" / "sub").mkdir(parents=True)
    (data / "tree" / "a.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (data / "tree" / "sub" / "b.jsonl").write_text('{"b": 2}\n', encoding="utf-8")
    (data / "drops.txt").write_text("row-1\n", encoding="utf-8")
    (data / "aux.json").write_text("{}", encoding="utf-8")
    prep = tmp_path / "qd-prep"
    prep.write_bytes(b"\x7fELF binary v1")
    monkeypatch.setenv("QD_PREP_BIN", str(prep))
    return Env(build, tmp_path, tmp_path / "cache", repo, data)


def _state(metric: object) -> tuple[str, bool, str]:
    assert isinstance(metric, Ran), metric
    return str(metric.value), metric.passed, metric.detail


def _rebuild_refused() -> tuple[list[DataRow], list[DataRow]]:
    raise AssertionError("a hit must not rebuild")


# --- a hit is the rebuild's rows, read back -------------------------------------------------------


def test_a_miss_stores_and_the_next_run_reads_the_same_rows_back_without_rebuilding(env):
    train, _val, metric = env.run()
    state, passed, detail = _state(metric)
    key = env.key()
    assert (state, passed, env.rebuilds) == ("miss", True, 1)
    assert key.digest in detail
    assert env.files() == sorted([key.digest + sc.ENTRY_SUFFIX, key.digest + sc.META_SUFFIX])
    for name in env.files():
        st = (env.cache / name).stat()
        assert st.st_mode & 0o777 == 0o600
    assert env.cache.stat().st_mode & 0o777 == 0o700

    env.log.clear()
    train2, val2, metric2 = env.run(rebuild=_rebuild_refused)
    state, passed, detail = _state(metric2)
    assert (state, passed) == ("hit", True)
    assert key.digest in detail and "READ FROM the split cache" in detail
    assert train2 == env.build.train and val2 == env.build.val
    assert [r.row_id for r in train2] == [r.row_id for r in train]
    (line,) = [x for x in env.log if "READ FROM the split cache" in x]
    assert "NOT rebuilt by this run" in line


def test_the_sidecar_records_what_was_checked_when_it_was_written(env):
    env.run()
    meta = json.loads((env.cache / (env.key().digest + sc.META_SUFFIX)).read_text())
    assert meta["rows"] == {"train": len(env.build.train), "val": len(env.build.val)}
    assert meta["held_out"]["rows_checked"] == len(env.build.train) + len(env.build.val)
    assert meta["held_out"]["held_out_families"] == sorted(DataConfig().held_out_families)
    assert meta["held_out"]["paths_checked"] == len(meta["covered"]) > 0
    entry = (env.cache / meta["entry"]["name"]).read_bytes()
    assert meta["entry"] == {"name": env.key().digest + sc.ENTRY_SUFFIX,
                             "sha256": hashlib.sha256(entry).hexdigest(), "bytes": len(entry)}


# --- 7. the key: any code or input change misses --------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


MUTATIONS: dict[str, Callable[[Env], None]] = {
    "an input file's bytes": lambda e: _write(e.data / "drops.txt", "row-2\n"),
    "a file in an input tree": lambda e: _write(e.data / "tree" / "sub" / "b.jsonl", '{"b": 3}\n'),
    "a new file in an input tree": lambda e: _write(e.data / "tree" / "c.jsonl", "{}\n"),
    "an extra input's bytes": lambda e: _write(e.data / "aux.json", '{"x": 1}'),
    "a python/ source": lambda e: _write(e.repo / "python" / "pkg" / "m.py", "X = 2\n"),
    "a new tools/ source": lambda e: _write(e.repo / "tools" / "u.py", "Z = 3\n"),
    "the qd-prep binary": lambda e: Path(os.environ["QD_PREP_BIN"]).write_bytes(b"v2"),
}


@pytest.mark.parametrize("what", sorted(MUTATIONS))
def test_any_code_or_input_change_is_another_key_and_misses(env, what):
    env.run()
    before = env.key()
    MUTATIONS[what](env)
    after = env.key()
    assert after.digest != before.digest, what
    _train, _val, metric = env.run()
    assert _state(metric)[0] == "miss" and env.rebuilds == 2


@pytest.mark.parametrize(("name", "value"), [
    ("max_pairs", 8), ("rev", "b" * 40), ("general_record", Path("/nonexistent/record.json")),
    ("config", DataConfig(seed=1)), ("replay_partition", True),
])
def test_any_argument_change_is_another_key(env, name, value):
    assert env.key(**{name: value}).digest != env.key().digest


def test_the_interpreter_and_installed_packages_are_in_the_key(env, monkeypatch):
    before = env.key().digest
    real = sc.environment_parts

    monkeypatch.setattr(sc, "environment_parts", lambda: {**real(), "n_distributions": -1})
    assert env.key().digest != before
    extra = sc.compute_key(env.kwargs(), extra_inputs={"aux": env.data / "aux.json"},
                           repo=env.repo, extra_facts={"git_version": "git version 9"})
    assert extra.digest != env.key().digest


def test_the_same_inputs_are_the_same_key_and_mtime_is_not_content(env):
    first = env.key()
    os.utime(env.data / "drops.txt", (1, 1))
    second = env.key()
    assert first.digest == second.digest
    assert str((env.data / "drops.txt").resolve()) in second.covered
    assert str((env.data / "tree" / "sub" / "b.jsonl").resolve()) in second.covered


def test_an_argument_with_no_canonical_form_is_not_keyed_and_the_run_rebuilds_uncached(env):
    _train, _val, metric = env.run(kwargs=env.kwargs(odd=object()))
    assert isinstance(metric, NotRun) and "could not key" in metric.reason
    assert env.rebuilds == 1 and env.files() == []
    assert any("NOT USED" in line for line in env.log)


# --- 4. a corrupt entry is loud, rebuilt and replaced --------------------------------------------


def _entry(env: Env) -> Path:
    return env.cache / (env.key().digest + sc.ENTRY_SUFFIX)


def _meta(env: Env) -> Path:
    return env.cache / (env.key().digest + sc.META_SUFFIX)


def _flip_byte(env: Env) -> None:
    body = bytearray(_entry(env).read_bytes())
    body[len(body) // 2] ^= 0xFF
    _entry(env).write_bytes(bytes(body))


def _truncate(env: Env) -> None:
    body = _entry(env).read_bytes()
    _entry(env).write_bytes(body[: len(body) - 10])


def _garbage_sidecar(env: Env) -> None:
    _meta(env).write_text("{not json", encoding="utf-8")


def _other_key_sidecar(env: Env) -> None:
    meta = json.loads(_meta(env).read_text())
    meta["key"] = "0" * 64
    _meta(env).write_text(json.dumps(meta), encoding="utf-8")


def _no_rule3_record(env: Env) -> None:
    meta = json.loads(_meta(env).read_text())
    del meta["held_out"]
    _meta(env).write_text(json.dumps(meta), encoding="utf-8")


CORRUPTIONS: dict[str, tuple[Callable[[Env], None], str]] = {
    "a flipped byte": (_flip_byte, "sha256"),
    "a truncated entry": (_truncate, "bytes; the sidecar says"),
    "an unparseable sidecar": (_garbage_sidecar, "does not parse"),
    "a sidecar naming another key": (_other_key_sidecar, "names schema"),
    "a sidecar with no rule-3 record": (_no_rule3_record, "does not parse"),
}


@pytest.mark.parametrize("what", sorted(CORRUPTIONS))
def test_a_corrupt_entry_is_logged_rebuilt_and_replaced_and_never_reported_as_a_hit(env, what):
    env.run()
    corrupt, reason = CORRUPTIONS[what]
    corrupt(env)
    env.log.clear()
    train, val, metric = env.run()
    state, passed, detail = _state(metric)
    assert (state, passed) == ("corrupt", False), detail
    assert reason in detail and "rebuilt and stored" in detail
    assert env.rebuilds == 2 and train == env.build.train and val == env.build.val
    assert any(line.startswith("split cache: CORRUPT entry") and reason in line
               for line in env.log), env.log
    # The fresh entry is whole: the next run reads it.
    _t, _v, again = env.run(rebuild=_rebuild_refused)
    assert _state(again)[:2] == ("hit", True)


def _rewrite_manifest(env: Env, name: str, edit: Callable[[dict], None]) -> None:
    path = env.build.out / "data" / "pool" / f"{name}.json"
    raw = json.loads(path.read_text())
    edit(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


@pytest.fixture
def manifests_restored(env):
    pool = env.build.out / "data" / "pool"
    saved = {p: p.read_bytes() for p in pool.iterdir()}
    yield
    for p, body in saved.items():
        p.write_bytes(body)


def _val_hash_changed(raw: dict) -> None:
    raw["entries"][0]["content_hash"] = "f" * 64


def _train_count_changed(raw: dict) -> None:
    raw["n_rows"] += 1


@pytest.mark.parametrize(("name", "edit", "reason"), [
    ("val", _val_hash_changed, "content hash is not val.json's"),
    ("train", _train_count_changed, "train rows but data/pool/train.json says n_rows"),
])
def test_rows_that_disagree_with_the_build_are_corrupt_and_a_rebuild_that_disagrees_is_not_stored(
    env, manifests_restored, name, edit, reason
):
    env.run()
    written = _entry(env).read_bytes()
    _rewrite_manifest(env, name, edit)
    _train, _val, metric = env.run()
    state, passed, detail = _state(metric)
    assert (state, passed) == ("corrupt", False) and reason in detail
    assert "not stored" in detail, "the rebuild's own rows fail the same check"
    assert _entry(env).read_bytes() == written, "nothing was written over the old entry"


# --- 3. atomic: what a reader may find --------------------------------------------------------


def test_an_entry_without_its_sidecar_is_a_miss(env):
    env.run()
    _meta(env).unlink()
    _t, _v, metric = env.run()
    assert _state(metric)[:2] == ("miss", True) and env.rebuilds == 2
    assert _meta(env).exists()


def test_a_sidecar_without_its_entry_is_a_miss(env):
    env.run()
    _entry(env).unlink()
    _t, _v, metric = env.run()
    state, _passed, detail = _state(metric)
    assert state == "miss" and "without its entry" in detail


def test_a_failed_write_leaves_no_temp_and_no_entry_and_says_not_stored(env, monkeypatch):
    def refuse(*a, **k):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(sc.pickle, "dump", refuse)
    train, _v, metric = env.run()
    state, passed, detail = _state(metric)
    assert (state, passed) == ("miss", False) and "not stored: disk full" in detail
    assert train == env.build.train
    assert env.files() == []


def test_a_sidecar_naming_another_entrys_bytes_is_corrupt_never_the_other_rows(env):
    """What two writers of one key would leave if their pickles ever differed: one writer's
    sidecar beside the other's entry. The sha256 refuses it before any unpickling."""
    env.run()
    other = list(env.build.train)
    other[0] = dataclasses.replace(other[0], dedupe_text=other[0].dedupe_text + " changed")
    body = pickle.dumps((other, list(env.build.val)), protocol=sc.PICKLE_PROTOCOL)
    tmp = env.cache / "elsewhere.tmp"
    tmp.write_bytes(body)
    tmp.chmod(0o600)
    tmp.replace(_entry(env))
    meta = json.loads(_meta(env).read_text())
    meta["entry"]["bytes"] = len(body)
    _meta(env).write_text(json.dumps(meta), encoding="utf-8")
    _meta(env).chmod(0o600)
    with sc.open_dir(env.cache) as dfd:
        got = sc.load(dfd, env.key(), manifests=sc.read_manifests(env.build.out),
                      config=DataConfig(), repo=env.repo)
    assert got.state == "corrupt" and "sha256" in got.reason and got.train is None


# --- 3. the race: two writers of one key, a reader in between ---------------------------------


def _race_writer(cache: str, key: sc.Key, out: str, train: list, val: list, barrier) -> None:
    manifests = sc.read_manifests(Path(out))
    barrier.wait()
    with sc.open_dir(Path(cache)) as dfd:
        held = sc.check_held_out(train, val, covered=(), config=DataConfig(), repo=Path(cache))
        sc.store(dfd, key, train, val, manifests=manifests, held_out=held)


def _race_reader(cache: str, key: sc.Key, out: str, repo: str, barrier, results) -> None:
    manifests = sc.read_manifests(Path(out))
    barrier.wait()
    seen = []
    for _ in range(40):
        with sc.open_dir(Path(cache)) as dfd:
            got = sc.load(dfd, key, manifests=manifests, config=DataConfig(), repo=Path(repo))
        digest = (None if got.train is None
                  else hashlib.sha256(pickle.dumps(got.train, protocol=5)).hexdigest())
        seen.append((got.state, got.reason, digest))
    results.put(seen)


@pytest.mark.parametrize("round_", range(3))
def test_two_writers_of_one_key_race_safely_and_a_reader_never_sees_a_partial_entry(
    env, round_
):
    """Real processes released together. Both lanes of a box miss at the same moment and
    write the same key; the pickle is deterministic, so they rename identical bytes."""
    key = env.key()
    sc.check_dir(env.cache)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(3)
    results = ctx.Queue()
    args = (str(env.cache), key, str(env.build.out))
    procs = [
        ctx.Process(target=_race_writer, args=(*args, env.build.train, env.build.val, barrier)),
        ctx.Process(target=_race_writer, args=(*args, env.build.train, env.build.val, barrier)),
        ctx.Process(target=_race_reader, args=(*args, str(env.repo), barrier, results)),
    ]
    for p in procs:
        p.start()
    seen = results.get(timeout=120)
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    want = hashlib.sha256(pickle.dumps(env.build.train, protocol=5)).hexdigest()
    for state, reason, digest in seen:
        assert state in ("miss", "hit"), (state, reason)
        assert state == "miss" or digest == want
    assert env.files() == sorted([key.digest + sc.ENTRY_SUFFIX, key.digest + sc.META_SUFFIX])
    _t, _v, metric = env.run(rebuild=_rebuild_refused)
    assert _state(metric)[0] == "hit"


# --- 2. permissions: checked on the descriptors, before any read ----------------------------------


def test_a_group_or_world_writable_cache_dir_is_refused(env):
    for mode in (0o770, 0o707, 0o777):
        env.cache.mkdir(exist_ok=True)
        env.cache.chmod(mode)
        with pytest.raises(sc.CacheDirRefused, match="group- or world-writable"):
            sc.check_dir(env.cache)
        with pytest.raises(sc.CacheDirRefused):
            env.run()
    assert env.rebuilds == 0


def test_a_symlinked_cache_dir_is_refused(env):
    real = env.tmp / "real-cache"
    real.mkdir(mode=0o700)
    env.cache.symlink_to(real)
    with pytest.raises(sc.CacheDirRefused, match="symlink"):
        sc.check_dir(env.cache)


def test_a_cache_dir_another_user_owns_is_refused(env, monkeypatch):
    sc.check_dir(env.cache)
    monkeypatch.setattr(sc.os, "getuid", lambda: env.cache.stat().st_uid + 1)
    with pytest.raises(sc.CacheDirRefused, match="not the running user"):
        sc.check_dir(env.cache)


def test_an_entry_another_user_could_write_is_refused_and_replaced(env):
    env.run()
    _entry(env).chmod(0o620)
    _t, _v, metric = env.run()
    state, _p, detail = _state(metric)
    assert state == "corrupt" and "group- or world-writable" in detail
    assert _entry(env).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("which", ["entry", "sidecar"])
def test_a_symlinked_entry_or_sidecar_is_never_followed(env, which):
    env.run()
    path = _entry(env) if which == "entry" else _meta(env)
    target = env.tmp / f"planted-{which}"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    planted = target.read_bytes()
    _t, _v, metric = env.run()
    state, _p, detail = _state(metric)
    assert state == "corrupt" and "symlink is never followed" in detail
    assert not path.is_symlink(), "the fresh entry replaced the link itself"
    assert target.read_bytes() == planted, "and wrote nothing through it"


def test_an_entry_that_resolves_a_function_is_refused_before_it_runs(env):
    """The second guard: a pickle that passed the sha check still resolves only classes from
    qd_data/qd_train, so REDUCE cannot reach os.system."""
    env.run()
    marker = env.tmp / "pwned"

    class Payload:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    body = pickle.dumps(Payload(), protocol=sc.PICKLE_PROTOCOL)
    _entry(env).write_bytes(body)
    meta = json.loads(_meta(env).read_text())
    meta["entry"].update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
    _meta(env).write_text(json.dumps(meta), encoding="utf-8")
    _t, _v, metric = env.run()
    state, _p, detail = _state(metric)
    assert state == "corrupt" and "is not allowed in a split cache" in detail
    assert not marker.exists()


# --- 5. eviction ----------------------------------------------------------------------------------


def test_the_oldest_key_is_evicted_after_a_successful_write_and_nothing_else_is_touched(env):
    keys = []
    for i, pairs in enumerate((1, 2)):
        env.run(kwargs=env.kwargs(max_pairs=pairs))
        keys.append(env.key(max_pairs=pairs).digest)
        for suffix in (sc.ENTRY_SUFFIX, sc.META_SUFFIX):
            os.utime(env.cache / (keys[-1] + suffix), (1000 + i, 1000 + i))
    (env.cache / "README").write_text("not the cache's", encoding="utf-8")
    stale = f".{'e' * 64}{sc.ENTRY_SUFFIX}.tmp-1-{'0' * 16}"
    (env.cache / stale).write_bytes(b"partial")
    env.log.clear()
    _t, _v, metric = env.run(kwargs=env.kwargs(max_pairs=3))
    newest = env.key(max_pairs=3).digest
    assert f"evicted ['{keys[0]}']" in _state(metric)[2]
    names = set(env.files())
    assert {keys[1] + sc.ENTRY_SUFFIX, newest + sc.ENTRY_SUFFIX, "README", stale} <= names
    assert not any(n.startswith(keys[0]) for n in names)
    assert any(line.startswith(f"split cache: EVICTED entry {keys[0]}") for line in env.log)
    assert any("left in place" in line and stale in line for line in env.log)


def test_nothing_is_evicted_when_the_write_failed(env, monkeypatch):
    for pairs in (1, 2):
        env.run(kwargs=env.kwargs(max_pairs=pairs))
    before = env.files()

    def refuse(*a, **k):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(sc.pickle, "dump", refuse)
    env.run(kwargs=env.kwargs(max_pairs=3))
    assert env.files() == before


# --- 6. rule 3 ------------------------------------------------------------------------------------


def _with_held_out_family(rows: list[DataRow]) -> list[DataRow]:
    held = DataConfig().held_out_families[0]
    return [dataclasses.replace(rows[0], family_id=held), *rows[1:]]


def test_a_rebuild_with_a_held_out_family_is_refused_and_not_stored(env):
    def rebuild():
        return _with_held_out_family(list(env.build.train)), list(env.build.val)

    with pytest.raises(HeldOutViolation):
        env.run(rebuild=rebuild)
    assert env.files() == []


def test_a_cached_entry_holding_a_held_out_family_raises_on_load_never_rebuilds(env):
    env.run()
    key = env.key()
    with sc.open_dir(env.cache) as dfd:
        sc.store(dfd, key, _with_held_out_family(list(env.build.train)), list(env.build.val),
                 manifests=sc.read_manifests(env.build.out),
                 held_out={"rows_checked": 0, "forged": True})
    with pytest.raises(HeldOutViolation):
        env.run(rebuild=_rebuild_refused)


def test_an_input_under_a_held_out_path_is_refused_before_storing(env):
    held = env.data / "heldout"
    held.mkdir()
    (held / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(HeldOutViolation):
        env.run(kwargs=env.kwargs(pre_dedupe_drops=held / "x.txt"))
    assert env.files() == []


def test_a_held_out_path_recorded_in_the_sidecar_raises_on_load(env):
    env.run()
    meta = json.loads(_meta(env).read_text())
    meta["covered"].append(str(env.data / "held_out" / "y.jsonl"))
    _meta(env).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(HeldOutViolation):
        env.run(rebuild=_rebuild_refused)


# --- 7. RSS: loaded rows are gone before rebuilt rows arrive --------------------------------------


def test_rows_a_failed_load_unpickled_are_released_before_the_rebuild_runs(
    env, manifests_restored
):
    env.run()
    _rewrite_manifest(env, "val", _val_hash_changed)  # the load unpickles, then fails a check
    gc.collect()
    baseline = sum(isinstance(o, DataRow) for o in gc.get_objects())
    live_at_rebuild: list[int] = []

    def rebuild():
        gc.collect()
        live_at_rebuild.append(sum(isinstance(o, DataRow) for o in gc.get_objects()))
        return list(env.build.train), list(env.build.val)

    _t, _v, metric = env.run(rebuild=rebuild)
    assert _state(metric)[0] == "corrupt"
    assert live_at_rebuild == [baseline], "the unpickled rows outlived their failed load"
