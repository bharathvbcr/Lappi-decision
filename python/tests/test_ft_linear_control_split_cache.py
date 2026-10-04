"""``tools/ft_linear_control.py --split-cache DIR --split-cache-shards DIR``: the merged
split-result cache (``tools/split_cache.py``) on the CPU controls' split rebuild.

After the v5 queue, one box session reruns v5's letter and option controls, about 16 calls. Each
call rebuilt the ~200 s split (Mac; ~325 s on the box) with both GPUs idle. With the cache, the
first call stores the rows. Every later call reads them back after ``split_cache``'s checks.

These tests use a fake ``real_ft_run``: the control reaches the runner's functions by name. They
pin:

* **Off.** The flag is off by default, and then the row is what it was. There is no
  ``split_cache`` metric, the rebuild is called once with the same arguments, and
  ``split_rebuild_inputs`` is never called. (A characterization: it passes against the tool
  before the change too. ``AUDIT/v6-ctlcache-2026-10-04/`` holds the byte-for-byte comparison of
  the flag-off rows before and after the change.)
* **MISS, then HIT.** Both write the rows the uncached run writes, letter and option alike,
  except for two things: the metric (``split_cache`` on the letter row, and
  ``linear_option_control.split_cache`` on the option row, every one of whose keys carries that
  prefix) and the row's identity.
* **The key.** The cache keys exactly the arguments the uncached call passes, plus what
  ``real_ft_run.split_rebuild_inputs`` names. That function is reached by name, never copied.
* **Failures.** A corrupt entry is rebuilt, and the row says ``corrupt``. Each of these is
  refused before the ledger, the verdicts or the rebuild is read:
  - an unsafe cache directory;
  - a shard set that is missing, or is not the eval row's;
  - either flag without the other.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import inspect
import io
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ft_linear_control as ftc  # noqa: E402
from data_fixtures import commitpackft_row  # noqa: E402
from test_control_label_space import _csqa_raw  # noqa: E402
from test_ft_linear_control import REV, _write_verdicts  # noqa: E402
from test_option_control import _eval_row  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.rows import DataRow, row_content_hash  # noqa: E402
from qd_train.baseline import request_texts  # noqa: E402
from qd_train.ledger import CODE_THAT_RAN  # noqa: E402

#: The fixture eval row's ``data_snapshot_hash`` (``test_option_control._eval_row``). A real eval
#: row's is its shard set's header's, which ``real_ft_run.corpus_facts`` holds equal to the
#: shard set's ``data/pool/train.json``.
SNAPSHOT = "d" * 64
#: The metric each row carries. The option row's keys all start with its own prefix (pinned by
#: test_option_control's test_the_option_row_changes_nothing_the_ledger_joins_or_promotion_reads).
LETTER_METRIC = "split_cache"
OPTION_METRIC = "linear_option_control.split_cache"
#: What two runs that wrote the same row may differ in: its identity, its place in its ledger's
#: hash chain, and when and how long it ran (``test_score_plan_trajectory._IDENTITY``).
IDENTITY = ("row_id", "prev_row_hash", "written_at", "wall_clock_s")


@dataclasses.dataclass
class Runner:
    """A stand-in ``real_ft_run`` exposing the two functions the control reaches by name.

    It records every call's keywords exactly. ``refuse_rebuild`` makes the rebuild fail if it is
    called, which is how a hit is told from a rebuild. Without ``with_inputs``, the module lacks
    ``split_rebuild_inputs``.
    """

    train: list[DataRow]
    val: list[DataRow]
    aux: Path
    rebuilds: list[dict[str, object]] = dataclasses.field(default_factory=list)
    inputs: list[dict[str, object]] = dataclasses.field(default_factory=list)
    refuse_rebuild: bool = False
    with_inputs: bool = True

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = types.ModuleType("real_ft_run")

        def ft_split_rows(**kwargs: object) -> tuple[list[DataRow], list[DataRow]]:
            if self.refuse_rebuild:
                raise AssertionError("a split-cache hit must not rebuild the split")
            self.rebuilds.append(dict(kwargs))
            return list(self.train), list(self.val)

        module.ft_split_rows = ft_split_rows  # type: ignore[attr-defined]
        if self.with_inputs:
            def split_rebuild_inputs(
                *, general_record: Path | None, exclude_identity_keys: Path | None,
                repo_history: bool,
            ) -> tuple[dict[str, Path], dict[str, object]]:
                self.inputs.append({"general_record": general_record,
                                    "exclude_identity_keys": exclude_identity_keys,
                                    "repo_history": repo_history})
                return {"aux": self.aux}, {"runner_fact": "fake runner v1"}

            module.split_rebuild_inputs = split_rebuild_inputs  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "real_ft_run", module)


def write_manifests(shards: Path, train: list[DataRow], val: list[DataRow], *,
                    snapshot: str) -> Path:
    """``<shards>/data/pool/{train,val}.json``, shaped like the pipeline's and carrying exactly
    the fields ``split_cache.read_manifests`` reads."""
    pool = shards / "data" / "pool"
    pool.mkdir(parents=True)
    (pool / "train.json").write_text(json.dumps(
        {"split": "train", "data_snapshot_hash": snapshot, "n_rows": len(train)}
    ), encoding="utf-8")
    (pool / "val.json").write_text(json.dumps(
        {"split": "val", "data_snapshot_hash": "v" * 64, "n_rows": len(val),
         "entries": [{"row_id": r.row_id, "content_hash": row_content_hash(r)} for r in val]}
    ), encoding="utf-8")
    return shards


@dataclasses.dataclass(frozen=True)
class Fixture:
    """An eval row, its verdicts and its shard set's manifests, over a split that both controls
    fit: CommonsenseQA (per-row options) and commit intent (one option set)."""

    tmp: Path
    ledger: Path
    verdicts: Path
    shards: Path
    corpus: Path
    runner: Runner

    def argv(self, *extra: str, write: str, verdicts: Path | None = None) -> list[str]:
        """v5's control shape: ``--no-repo-history`` and no ``--max-pairs``, so the rebuild is
        handed ``max_pairs=0``. Each run writes its row to a ledger of its own."""
        return ["--ledger", str(self.ledger),
                "--verdicts", str(self.verdicts if verdicts is None else verdicts),
                "--write-ledger", str(self.tmp / f"{write}.jsonl"),
                "--no-repo-history", "--defect-class", str(self.corpus), "--rev", REV, *extra]

    def cache_flags(self, cache: Path, shards: Path | None = None) -> tuple[str, ...]:
        return ("--split-cache", str(cache),
                "--split-cache-shards", str(self.shards if shards is None else shards))

    def row(self, write: str) -> dict:
        (line,) = [x for x in (self.tmp / f"{write}.jsonl").read_text(encoding="utf-8")
                   .splitlines() if x.strip()]
        return json.loads(line)


def build_fixture(tmp: Path) -> Fixture:
    config = DataConfig()
    mixture = build_mixture(
        {"tau/commonsense_qa": [_csqa_raw(i) for i in range(80)],
         "bigcode/commitpackft": [commitpackft_row(i) for i in range(80)]},
        config=config,
    )
    rows = [r for r in mixture.rows
            if r.family_id in ("commonsense.multiple_choice", "code.commit_intent")]
    train = [r for i, r in enumerate(rows) if i % 4]
    val = [r for i, r in enumerate(rows) if not i % 4]
    val_docs, _ = request_texts(val, seed=config.seed)
    hits = [i % 3 != 0 for i in range(len(val_docs))]
    ledger = tmp / "ledger.jsonl"
    eval_id = _eval_row(ledger, val_docs, hits)
    verdicts = _write_verdicts(tmp / "v.jsonl", [
        {"eval_row_id": eval_id, "seed": 0, "row_id": d.row_id, "kind": d.kind,
         "slot_name": d.slot_name, "correct": h}
        for d, h in zip(val_docs, hits, strict=True)
    ])
    corpus = tmp / "corpus-v2"
    corpus.mkdir()
    (corpus / "examples.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    aux = tmp / "aux.json"
    aux.write_text("{}", encoding="utf-8")
    shards = write_manifests(tmp / "shards", train, val, snapshot=SNAPSHOT)
    return Fixture(tmp, ledger, verdicts, shards, corpus, Runner(train, val, aux))


@pytest.fixture
def fx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path) -> Fixture:
    fixture = build_fixture(tmp_path)
    fixture.runner.install(monkeypatch)
    return fixture


def comparable(row: dict, *drop_metrics: str) -> dict:
    out = {k: v for k, v in row.items() if k not in IDENTITY}
    out["metrics"] = {k: v for k, v in row["metrics"].items() if k not in drop_metrics}
    return out


def key_of(metric: dict) -> str:
    """The cache key a ``split_cache`` metric names: its detail starts ``key <sha256>; ``."""
    head = metric["detail"].split(";")[0]
    assert head.startswith("key ") and len(head) == len("key ") + 64, head
    return head.removeprefix("key ")


# --- off: the row is what it was --------------------------------------------------------------


def test_without_the_flag_no_row_carries_the_metric_and_the_rebuild_is_one_plain_call(
    fx: Fixture,
) -> None:
    """Characterization: passes against the tool before the change too."""
    assert ftc.main(fx.argv(write="letter")) == 0
    assert ftc.main(fx.argv("--option-control", write="option")) == 0
    for name in ("letter", "option"):
        row = fx.row(name)
        assert not [k for k in row["metrics"] if "split_cache" in k], name
        assert not [k for k in row["recipe"] if "split" in k], name
        assert "split cache" not in row["notes"], name
    assert fx.runner.inputs == [], "split_rebuild_inputs is the cache's, not the rebuild's"
    assert len(fx.runner.rebuilds) == 2 and fx.runner.rebuilds[0] == fx.runner.rebuilds[1]


# --- MISS then HIT: the rows the uncached run writes -------------------------------------------


def test_miss_then_hit_write_the_letter_and_option_rows_the_uncached_run_writes(
    fx: Fixture,
) -> None:
    cache = fx.tmp / "split-cache"
    flags = fx.cache_flags(cache)
    assert ftc.main(fx.argv(write="letter-off")) == 0
    assert ftc.main(fx.argv("--option-control", write="option-off")) == 0
    assert ftc.main(fx.argv(*flags, write="letter-miss")) == 0
    assert len(fx.runner.rebuilds) == 3
    fx.runner.refuse_rebuild = True
    assert ftc.main(fx.argv(*flags, write="letter-hit")) == 0
    assert ftc.main(fx.argv("--option-control", *flags, write="option-hit")) == 0

    letter = {s: fx.row(f"letter-{s}") for s in ("off", "miss", "hit")}
    option = {s: fx.row(f"option-{s}") for s in ("off", "hit")}
    keys = set()
    for state, metric in (("miss", letter["miss"]["metrics"][LETTER_METRIC]),
                          ("hit", letter["hit"]["metrics"][LETTER_METRIC]),
                          ("hit", option["hit"]["metrics"][OPTION_METRIC])):
        assert (metric["state"], metric["value"], metric["passed"]) == ("ran", state, True)
        keys.add(key_of(metric))
    assert len(keys) == 1, "one split, one key, for both controls"
    assert "READ FROM the split cache, not rebuilt" in letter["hit"]["metrics"][LETTER_METRIC][
        "detail"]
    (key,) = keys
    assert (cache / f"{key}.rows.pkl").is_file()

    # Not recipe, not notes: every field but the metric and the row's identity is the uncached
    # run's, so a cached and an uncached control of one eval row are one protocol.
    off = comparable(letter["off"])
    assert comparable(letter["miss"], LETTER_METRIC) == off
    assert comparable(letter["hit"], LETTER_METRIC) == off
    assert comparable(option["hit"], OPTION_METRIC) == comparable(option["off"])
    assert OPTION_METRIC not in letter["hit"]["metrics"]
    assert LETTER_METRIC not in option["hit"]["metrics"]
    foreign = [k for k in option["hit"]["metrics"] if k != CODE_THAT_RAN
               and not k.startswith((f"{ftc.OPTION_ARM}.", f"{ftc.OPTION_MARGIN}."))]
    assert foreign == [], "the option row carries only its own keys"


def test_the_cache_keys_exactly_the_arguments_the_uncached_rebuild_is_handed(fx: Fixture) -> None:
    cache = fx.tmp / "split-cache"
    assert ftc.main(fx.argv(write="off")) == 0
    assert ftc.main(fx.argv(*fx.cache_flags(cache), write="miss")) == 0
    uncached, keyed = fx.runner.rebuilds
    assert keyed == uncached, "the partial binds the uncached call's keywords, all of them"
    assert (uncached["max_pairs"], uncached["repo_history"]) == (0, False)
    assert uncached["defect_class"] == fx.corpus and uncached["rev"] == REV
    (meta_path,) = cache.glob("*.meta.json")
    parts = json.loads(meta_path.read_text(encoding="utf-8"))["key_parts"]
    assert set(parts["arguments"]) == set(uncached)
    assert parts["arguments"]["defect_class"]["tree"] == str(fx.corpus), "a path, by content"
    # real_ft_run.split_rebuild_inputs, reached by name, with the run's own three values.
    assert fx.runner.inputs == [
        {"general_record": None, "exclude_identity_keys": None, "repo_history": False}
    ]
    assert set(parts["extra_inputs"]) == {"aux"}
    assert parts["extra_facts"] == {"runner_fact": "fake runner v1"}


def test_the_control_passes_every_parameter_of_the_real_runners_two_functions(
    tmp_path: Path, qd_prep: Path,
) -> None:
    """The fake above cannot drift from the runner: its recorded keywords are the real
    functions' parameter lists, name for name."""
    pytest.importorskip("torch", reason="real_ft_run imports torch at module scope")
    fixture = build_fixture(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        fixture.runner.install(mp)
        assert ftc.main(fixture.argv(*fixture.cache_flags(tmp_path / "c"), write="miss")) == 0
    real = importlib.import_module("real_ft_run")
    assert hasattr(real, "main"), "the real runner, not the fake (which has no main)"
    (rebuild,) = fixture.runner.rebuilds
    (inputs,) = fixture.runner.inputs
    assert set(inspect.signature(real.ft_split_rows).parameters) == set(rebuild)
    assert set(inspect.signature(real.split_rebuild_inputs).parameters) == set(inputs)


# --- a corrupt entry is rebuilt, and the row says so --------------------------------------------


def test_a_corrupt_entry_is_rebuilt_and_the_row_says_corrupt_never_hit(fx: Fixture) -> None:
    cache = fx.tmp / "split-cache"
    flags = fx.cache_flags(cache)
    assert ftc.main(fx.argv(*flags, write="miss")) == 0
    (entry,) = cache.glob("*.rows.pkl")
    body = bytearray(entry.read_bytes())
    body[len(body) // 2] ^= 0xFF
    entry.write_bytes(bytes(body))
    assert ftc.main(fx.argv(*flags, write="corrupt")) == 0
    assert len(fx.runner.rebuilds) == 2, "the corrupt entry was rebuilt"
    metric = fx.row("corrupt")["metrics"][LETTER_METRIC]
    assert (metric["value"], metric["passed"]) == ("corrupt", False)
    assert "sha256" in metric["detail"] and "rebuilt and stored" in metric["detail"]
    assert comparable(fx.row("corrupt"), LETTER_METRIC) == comparable(fx.row("miss"),
                                                                       LETTER_METRIC)
    fx.runner.refuse_rebuild = True
    assert ftc.main(fx.argv(*flags, write="after")) == 0
    assert fx.row("after")["metrics"][LETTER_METRIC]["value"] == "hit"


# --- refused before the ledger, the verdicts or the rebuild ------------------------------------


def test_an_unsafe_cache_dir_is_refused_before_the_verdicts_are_read(fx: Fixture) -> None:
    """The cache reads pickles. A directory another user can write is refused at argv time:
    the verdicts named here do not exist, and the refusal is the cache's, not theirs."""
    shared = fx.tmp / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    absent = fx.tmp / "no-such-verdicts.jsonl"
    with pytest.raises(SystemExit, match="group- or world-writable"):
        ftc.main(fx.argv(*fx.cache_flags(shared), write="w", verdicts=absent))
    real = fx.tmp / "real-cache"
    real.mkdir(mode=0o700)
    link = fx.tmp / "linked-cache"
    link.symlink_to(real)
    with pytest.raises(SystemExit, match="symlink"):
        ftc.main(fx.argv(*fx.cache_flags(link), write="w", verdicts=absent))
    assert fx.runner.rebuilds == [] and not (fx.tmp / "w.jsonl").exists()


def test_each_flag_without_the_other_is_refused_before_anything_is_read(
    fx: Fixture, capsys: pytest.CaptureFixture[str],
) -> None:
    cache = fx.tmp / "split-cache"
    absent = fx.tmp / "no-such-verdicts.jsonl"
    with pytest.raises(ftc.Refused, match="--split-cache-shards"):
        ftc.main(fx.argv("--split-cache", str(cache), write="w", verdicts=absent))
    assert not cache.exists(), "refused before the directory was created"
    with pytest.raises(SystemExit):
        ftc.main(fx.argv("--split-cache-shards", str(fx.shards), write="w", verdicts=absent))
    assert "--split-cache-shards without --split-cache reads nothing" in capsys.readouterr().err
    assert fx.runner.rebuilds == []


def test_a_shard_set_that_is_absent_or_not_the_eval_rows_is_refused_before_the_rebuild(
    fx: Fixture,
) -> None:
    """Without this, a wrong directory on the box makes every call a miss whose rows fail the
    manifest check and are never stored: the whole rebuild, every call, said only in a metric."""
    cache = fx.tmp / "split-cache"
    other = write_manifests(fx.tmp / "other-shards", fx.runner.train, fx.runner.val,
                            snapshot="e" * 64)
    with pytest.raises(ftc.Refused, match="data_snapshot_hash"):
        ftc.main(fx.argv(*fx.cache_flags(cache, other), write="w"))
    empty = fx.tmp / "empty-shards"
    empty.mkdir()
    with pytest.raises(ftc.Refused, match="cannot be read"):
        ftc.main(fx.argv(*fx.cache_flags(cache, empty), write="w"))
    assert fx.runner.rebuilds == [] and not (fx.tmp / "w.jsonl").exists()


def test_a_runner_without_split_rebuild_inputs_is_refused_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qd_prep: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    fixture.runner.with_inputs = False
    fixture.runner.install(monkeypatch)
    with pytest.raises(ftc.Refused, match="has no split_rebuild_inputs"):
        ftc.main(fixture.argv(*fixture.cache_flags(tmp_path / "c"), write="w"))
    assert fixture.runner.rebuilds == []
    # Uncached, the control never needs it.
    assert ftc.main(fixture.argv(write="off")) == 0


def test_the_flags_are_documented_as_off_by_default_and_not_recipe() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        ftc.main(["--help"])
    flat = " ".join(out.getvalue().split())
    assert "--split-cache DIR" in flat and "--split-cache-shards DIR" in flat
    assert "Off by default. Not recipe" in flat
