"""``tools/arm_b_mine.py`` (card ft-b1dfb6da273): the refusals and the selection rule.

The refusals are tested before any scoring exists: a held-out path, a caller record, an
uncommitted or edited pre-registration, an unpinned or mismatched corpus or scorer, a candidate
that duplicates a v5 train or val key, and a candidate from a held-out repository. The selection
rule is held to ``tools/real_ft_run.py``'s ``choice_rule_abstentions`` by running that function's
own source on random verdict lines.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import arm_b_mine as mine  # noqa: E402

#: Every Arm B corpus declares a non-empty id prefix (the driver refuses a pre-registration in
#: which one does not); the test corpora's rows carry it.
PFX = "w:"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(repo: Path, *args: str) -> str:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(repo),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "F",
        "GIT_AUTHOR_EMAIL": "f@example.com",
        "GIT_COMMITTER_NAME": "F",
        "GIT_COMMITTER_EMAIL": "f@example.com",
    }
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True
    ).stdout.strip()


def _cand(i: int, repo: str = "own/train-repo", after: str | None = None) -> dict[str, object]:
    return {
        "id": f"{PFX}p{i}:f.py#0",
        "pool_id": f"p{i}:f.py",
        "repo": repo,
        "path": "f.py",
        "language": "python",
        "class": "logic",
        "before": f"x = {i}\n",
        "after": after if after is not None else f"x = {i + 1}\n",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    path.write_bytes(body)
    return body


@pytest.fixture
def world(tmp_path: Path) -> dict[str, object]:
    """A throwaway repo holding a committed pre-registration that pins one corpus and two
    exclusion sets, plus the files it names."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    corpus = tmp_path / "data" / "corpus.jsonl"
    corpus.parent.mkdir()
    rows = [_cand(i) for i in range(10)]
    corpus_bytes = _write_jsonl(corpus, rows)
    train = tmp_path / "data" / "v5-train.jsonl"
    train_bytes = _write_jsonl(train, [{"id": "p3:f.py#0"}])
    val = tmp_path / "data" / "v5-val.jsonl"
    val_bytes = _write_jsonl(val, [{"row_id": "p4:f.py#0"}])
    scorer = tmp_path / "data" / "scorer.safetensors"
    scorer.write_bytes(b"weights")
    prereg = {
        "scorer": {"path": str(scorer), "sha256": _sha(b"weights")},
        "candidates": {
            "corpora": [
                {
                    "name": "c1",
                    "path": str(corpus),
                    "sha256": _sha(corpus_bytes),
                    "rows": 10,
                    "id_prefix": PFX,
                }
            ],
            "scan_cap_rows": 20000,
        },
        "exclusion": {
            "sets": [
                {"name": "v5-train", "path": str(train), "sha256": _sha(train_bytes), "key": "id"},
                {"name": "v5-val", "path": str(val), "sha256": _sha(val_bytes), "key": "row_id"},
            ]
        },
        "caps": {"train_cap_mined_rows": 5, "floor_rows": 3},
    }
    path = repo / "prereg.json"
    path.write_text(json.dumps(prereg, indent=1))
    _git(repo, "add", "prereg.json")
    _git(repo, "commit", "-q", "-m", "prereg")
    return {
        "repo": repo,
        "prereg": path,
        "corpus": corpus,
        "rows": rows,
        "tmp": tmp_path,
        "train": train,
        "val": val,
        "scorer": scorer,
        "doc": prereg,
    }


def _plan(world: dict[str, object], **over: object) -> dict[str, object]:
    out = Path(str(world["tmp"])) / f"out{len(list(Path(str(world['tmp'])).glob('out*')))}"
    kwargs: dict[str, object] = {
        "prereg": world["prereg"],
        "repo": world["repo"],
        "out_dir": out,
        "heldout_repos": ["own/held-repo"],
    }
    kwargs.update(over)
    return mine.plan(**kwargs)  # type: ignore[arg-type]


def test_a_clean_plan_drops_v5_train_and_val_duplicates_and_counts_each_set(
    world: dict[str, object],
) -> None:
    report = _plan(world)
    assert report["offered"] == 10
    assert report["duplicates_by_set"] == {"v5-train": 1, "v5-val": 1}
    assert report["kept"] == 8
    assert report["leakage"]["exact_identity"]["state"] == "ran"
    # Containment was not given, so it is `not_run`, never clean.
    assert report["leakage"]["containment"]["state"] == "not_run"
    out = Path(str(report["out_dir"]))
    kept = [json.loads(line)["id"] for line in (out / "candidates.jsonl").read_text().splitlines()]
    assert "p3:f.py#0" not in kept and "p4:f.py#0" not in kept


def test_a_content_duplicate_under_another_id_is_dropped(world: dict[str, object]) -> None:
    tmp = Path(str(world["tmp"]))
    first = _cand(3)
    renamed = {**first, "id": f"{PFX}q99:g.py#0"}
    train_bytes = _write_jsonl(Path(str(world["train"])), [first])
    corpus = Path(str(world["corpus"]))
    corpus_bytes = _write_jsonl(corpus, [renamed, _cand(5)])
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["candidates"]["corpora"][0]["sha256"] = _sha(corpus_bytes)
    doc["exclusion"]["sets"][0]["sha256"] = _sha(train_bytes)
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "re-pin")
    del tmp
    report = _plan(world)
    assert report["duplicates_by_set"]["v5-train"] == 1 and report["kept"] == 1


def test_a_held_out_repository_is_refused_whole(world: dict[str, object]) -> None:
    corpus = Path(str(world["corpus"]))
    body = _write_jsonl(corpus, [_cand(1), _cand(2, repo="own/held-repo")])
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["candidates"]["corpora"][0]["sha256"] = _sha(body)
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "re-pin")
    with pytest.raises(mine.Refusal, match="held-out repository"):
        _plan(world)


def test_a_held_out_path_is_refused_before_it_is_read(
    world: dict[str, object], tmp_path: Path
) -> None:
    held = tmp_path / "heldout" / "corpus.jsonl"
    held.parent.mkdir()
    body = _write_jsonl(held, [_cand(1)])
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["candidates"]["corpora"][0].update(path=str(held), sha256=_sha(body))
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "held path")
    with pytest.raises(mine.Refusal, match="held-out"):
        _plan(world)


def test_a_file_holding_a_caller_record_is_refused(world: dict[str, object]) -> None:
    corpus = Path(str(world["corpus"]))
    rec = {"record": "lappi.caller_record", "version": 1}
    body = _write_jsonl(corpus, [_cand(1), rec])
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["candidates"]["corpora"][0]["sha256"] = _sha(body)
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "record")
    with pytest.raises(mine.Refusal, match="caller record"):
        _plan(world)


def test_an_uncommitted_or_edited_preregistration_is_refused(world: dict[str, object]) -> None:
    prereg = Path(str(world["prereg"]))
    prereg.write_text(prereg.read_text() + " ")
    with pytest.raises(mine.Refusal, match=r"not committed|differs from HEAD"):
        _plan(world)
    _git(Path(str(world["repo"])), "checkout", "-q", "--", "prereg.json")
    untracked = Path(str(world["repo"])) / "other.json"
    untracked.write_text(prereg.read_text())
    with pytest.raises(mine.Refusal, match=r"not committed|differs from HEAD"):
        _plan(world, prereg=untracked)


def test_a_null_or_mismatched_corpus_sha_is_refused(world: dict[str, object]) -> None:
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["candidates"]["corpora"][0]["sha256"] = None
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "null")
    with pytest.raises(mine.Refusal, match="not pinned"):
        _plan(world)
    doc["candidates"]["corpora"][0]["sha256"] = "0" * 64
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "wrong")
    with pytest.raises(mine.Refusal, match="hashes to"):
        _plan(world)


def test_a_candidate_without_an_id_or_a_duplicate_within_the_corpus_is_handled(
    world: dict[str, object],
) -> None:
    corpus = Path(str(world["corpus"]))
    body = _write_jsonl(
        corpus, [_cand(1), _cand(1), {"repo": "own/x", "before": "a", "after": "b"}]
    )
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["candidates"]["corpora"][0]["sha256"] = _sha(body)
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "dups")
    with pytest.raises(mine.Refusal, match="no id"):
        _plan(world)


def test_the_scorer_must_hash_to_the_preregistered_sha(world: dict[str, object]) -> None:
    mine.check_scorer(Path(str(world["scorer"])), world["doc"])  # type: ignore[arg-type]
    Path(str(world["scorer"])).write_bytes(b"other weights")
    with pytest.raises(mine.Refusal, match="scorer"):
        mine.check_scorer(Path(str(world["scorer"])), world["doc"])  # type: ignore[arg-type]


# --- the selection rule -------------------------------------------------------------------------


def _reference() -> object:
    """``choice_rule_abstentions`` as committed, by its own source text (no torch import)."""
    src = (ROOT / "tools" / "real_ft_run.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "choice_rule_abstentions":
            ns: dict[str, object] = {"Mapping": Mapping}
            exec("from __future__ import annotations\n" + ast.get_source_segment(src, node), ns)
            return ns["choice_rule_abstentions"]
    raise AssertionError("choice_rule_abstentions is gone from tools/real_ft_run.py")


def _verdict(rid: str, rng: random.Random, n: int, expected_abstain: bool) -> dict[str, object]:
    perm = list(range(n))
    rng.shuffle(perm)
    return {
        "row_id": rid,
        "slot_name": "defect_class",
        "kind": "choice",
        "expected_abstain": expected_abstain,
        "top": rng.randrange(n + 1),
        "noul_row": n,
        "gold_row": rng.randrange(n),
        "perm": perm,
        "top_permuted": rng.randrange(n + 1),
        "permutation_agreed": True,
    }


def test_the_selection_rule_is_choice_rule_abstentions_on_random_verdicts() -> None:
    ref = _reference()
    rng = random.Random(7)
    lines = [_verdict(f"r{i}", rng, rng.choice([4, 5]), rng.random() < 0.2) for i in range(400)]
    # One row without a derangement: no second pass, abstains only on its own noul.
    solo = {
        "row_id": "solo",
        "slot_name": "defect_class",
        "kind": "choice",
        "expected_abstain": False,
        "top": 1,
        "noul_row": 1,
        "gold_row": 0,
    }
    lines.append(solo)
    first = {"verdicts": lines}
    second = {
        "verdicts": [
            {
                "row_id": v["row_id"],
                "slot_name": v["slot_name"],
                "kind": "choice",
                "top": v["top_permuted"],
                "noul_row": v["noul_row"],
            }
            for v in lines
            if "perm" in v
        ]
    }
    perms = {
        (str(v["row_id"]), str(v["slot_name"])): tuple(v["perm"]) for v in lines if "perm" in v
    }  # type: ignore[arg-type]
    want = ref(first, second, perms)  # type: ignore[operator]
    got = mine.abstentions(lines)
    assert got == want
    assert "solo" in got and got["solo"] is True
    assert any(want.values()) and not all(want.values())


def test_kept_rows_abstain_and_have_a_non_noul_gold_and_the_floor_and_cap_hold() -> None:
    rows = [
        {"row_id": f"r{i}", "expected_abstain": i % 7 == 0, "abstained": i % 2 == 0}
        for i in range(60)
    ]
    kept = mine.keep(rows)
    assert all(r["abstained"] and not r["expected_abstain"] for r in kept)
    assert mine.apply_caps(kept, cap=5, floor=3)["state"] == "mined"
    assert len(mine.apply_caps(kept, cap=5, floor=3)["rows"]) == 5
    short = mine.apply_caps(kept[:2], cap=5, floor=3)
    assert short["state"] == "no_data" and short["rows"] == []
    assert mine.apply_caps([], cap=5, floor=0)["state"] == "no_data"


def test_the_scan_budget_is_the_preregistered_formula() -> None:
    assert mine.scan_rows(rows_per_s=1.0) == 20_000  # 129,600 uncapped
    assert mine.scan_rows(rows_per_s=0.1) == 12_960
    assert mine.scan_rows(rows_per_s=0.0001) == 12
    assert mine.scan_rows(rows_per_s=100.0) == 20_000
    with pytest.raises(mine.Refusal):
        mine.scan_rows(rows_per_s=0.0)
    with pytest.raises(mine.Refusal):
        mine.scan_rows(rows_per_s=float("nan"))


def test_the_pinned_function_hash_matches_the_committed_source() -> None:
    doc = json.loads((ROOT / "campaign/v6-arm-b-mining-preregistered.json").read_text())
    pinned = doc["selection_rule"]["function_sha256"]
    got = mine.function_sha256("d930579", "choice_rule_abstentions", repo=ROOT)
    assert got == pinned


REAL_V5_VAL_ID = (
    "qdm:code.defect_class:761ae0d762324ef1eba93ab1b9cf2cf28d2fa30e:"
    "python/snippets/find_if_program_installed.py#0"
)


def test_a_real_v5_val_id_in_the_mixture_format_excludes_the_corpus_row(
    world: dict[str, object],
) -> None:
    """verify-lane6 D9: v5's verdict lines key rows `qdm:<family>:<corpus id>`; the corpus's own
    id has no prefix. The exclusion matched 0 of 2,304 until the key was normalised."""
    corpus_id = REAL_V5_VAL_ID.removeprefix("qdm:code.defect_class:")
    assert mine.normalise_key(REAL_V5_VAL_ID) == corpus_id == mine.normalise_key(corpus_id)
    val = Path(str(world["val"]))
    val_bytes = _write_jsonl(val, [{"row_id": REAL_V5_VAL_ID}])
    row = {**_cand(1), "id": f"{PFX}{corpus_id}", "pool_id": corpus_id.split("#")[0]}
    corpus = Path(str(world["corpus"]))
    corpus_bytes = _write_jsonl(corpus, [row, _cand(2)])
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["candidates"]["corpora"][0]["sha256"] = _sha(corpus_bytes)
    doc["exclusion"]["sets"][1]["sha256"] = _sha(val_bytes)
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "real val id")
    report = _plan(world)
    assert report["duplicates_by_set"]["v5-val"] == 1 and report["kept"] == 1


def _with_split_map(
    world: dict[str, object],
    mapping: dict[str, str],
    rows: list[dict[str, object]],
    **corpus_over: object,
) -> None:
    tmp = Path(str(world["tmp"]))
    smap = tmp / "data" / "split-map.json"
    smap_bytes = (json.dumps({"repos": mapping}) + "\n").encode()
    smap.write_bytes(smap_bytes)
    corpus = Path(str(world["corpus"]))
    corpus_bytes = _write_jsonl(corpus, rows)
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["candidates"]["corpora"][0].update(sha256=_sha(corpus_bytes), **corpus_over)
    doc["exclusion"]["split_map"] = {"path": str(smap), "sha256": _sha(smap_bytes)}
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "split map")


def test_mutants_of_val_and_heldout_repos_are_excluded_by_the_split_map(
    world: dict[str, object],
) -> None:
    rows = [_cand(7, repo="a/train"), _cand(8, repo="b/val"), _cand(9, repo="c/held")]
    _with_split_map(world, {"a/train": "train", "b/val": "val", "c/held": "heldout"}, rows)
    report = _plan(world)
    assert report["kept"] == 1
    assert report["leakage"]["split_map"]["state"] == "ran"
    assert report["leakage"]["split_map"]["dropped_not_train_split"] == 2
    assert report["leakage"]["heldout_exact_keys"]["state"] == "not_run"


def test_a_repo_outside_the_split_map_is_refused_unless_the_corpus_is_own_repos_train(
    world: dict[str, object],
) -> None:
    _with_split_map(world, {"a/train": "train"}, [_cand(1, repo="x/unmapped")])
    with pytest.raises(mine.Refusal, match="not in the pinned split map"):
        _plan(world)
    _with_split_map(
        world, {"a/train": "train"}, [_cand(1, repo="x/unmapped")], own_repos_train=True
    )
    # An own_repos_train corpus must also pin its pool manifest (see the pool-manifest test).
    manifest = _pin_pool_manifest(world, None)
    _pin_pool_manifest(world, _sha(manifest.read_bytes()))
    assert _plan(world)["kept"] == 1


def test_select_refuses_verdicts_scored_over_another_shard_set(world: dict[str, object]) -> None:
    report = _plan(world)
    plan_dir = Path(str(report["out_dir"]))
    plan_doc = json.loads((plan_dir / "plan.json").read_text())
    assert plan_doc["candidate_ids"], "the plan lists the ids it was built from"
    foreign = Path(str(world["tmp"])) / "verdicts.jsonl"
    foreign.write_text(json.dumps({"row_id": "not-in-the-plan", "kind": "choice"}) + "\n")
    args = type("A", (), {})()
    args.prereg, args.repo, args.verdicts = world["prereg"], world["repo"], foreign
    args.plan_dir, args.out = plan_dir, Path(str(world["tmp"])) / "mined.json"
    with pytest.raises(mine.Refusal, match="not in the plan"):
        mine._select(args)


# -- identity is content (Fable's ruling of 2026-10-13) -----------------------------------------


def _two_corpora(
    world: dict[str, object],
    first: list[dict[str, object]],
    second: list[dict[str, object]],
    *,
    train_rows: list[dict[str, object]] | None = None,
    val_rows: list[dict[str, object]] | None = None,
    prefixes: tuple[str, str] = (PFX, PFX),
    train_match: str | None = "content_or_identity_key",
) -> None:
    """Re-pin the world's prereg: two corpora, a v5-train set matched by content or identity key
    (the ruling) and a v5-val set matched by id."""
    tmp = Path(str(world["tmp"]))
    doc = json.loads(Path(str(world["prereg"])).read_text())
    corpora = []
    for n, (rows, prefix) in enumerate(zip((first, second), prefixes, strict=True), start=1):
        path = tmp / "data" / f"c{n}" / "examples.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = _write_jsonl(path, rows)
        (path.parent / "manifest.json").write_text(
            json.dumps(
                {
                    "examples_sha256": _sha(body),
                    "totals": {"examples": len(rows), "by_class": {"unfiltered": len(rows)}},
                    "pool": {"path": "x/pool.jsonl", "sha256": "0" * 64, "records": 1},
                }
            )
        )
        entry: dict[str, object] = {
            "name": f"c{n}",
            "path": str(path),
            "sha256": _sha(body),
            "rows": len(rows),
        }
        if prefix:
            entry["id_prefix"] = prefix
        corpora.append(entry)
    doc["candidates"]["corpora"] = corpora
    if train_rows is not None:
        body = _write_jsonl(Path(str(world["train"])), train_rows)
        doc["exclusion"]["sets"][0]["sha256"] = _sha(body)
    if train_match is not None:
        doc["exclusion"]["sets"][0]["match"] = train_match
    if val_rows is not None:
        body = _write_jsonl(Path(str(world["val"])), val_rows)
        doc["exclusion"]["sets"][1]["sha256"] = _sha(body)
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "two corpora")


def test_two_corpora_sharing_ids_are_both_kept(world: dict[str, object]) -> None:
    """`qd-mutate compose` numbers rows `compose:train:000000` in every corpus; (corpus, id) is
    the within-candidate key, so neither corpus loses its rows to the other's namespace."""
    a = [{**_cand(i), "id": f"{PFX}compose:train:{i:06d}"} for i in range(3)]
    b = [{**_cand(i + 100), "id": f"{PFX}compose:train:{i:06d}"} for i in range(3)]
    _two_corpora(world, a, b, train_rows=[{"id": "unrelated"}], val_rows=[{"row_id": "unrelated"}])
    report = _plan(world)
    assert report["offered"] == 6 and report["kept"] == 6
    assert report["duplicates_within"] == 0
    assert report["ids_unique"] is False and report["id_collisions_across_corpora"] == 3


def test_a_content_identical_row_is_dropped_as_a_duplicate_whatever_its_corpus_and_id(
    world: dict[str, object],
) -> None:
    a = [_cand(1), _cand(2)]
    b = [{**_cand(1), "id": f"{PFX}other:name#9", "pool_id": "other:name"}, _cand(7)]
    _two_corpora(world, a, b, train_rows=[{"id": "unrelated"}], val_rows=[{"row_id": "unrelated"}])
    report = _plan(world)
    assert report["offered"] == 4
    assert report["duplicates_within"] == 1 and report["kept"] == 3


def test_an_id_only_collision_with_v5_train_is_counted_and_not_dropped(
    world: dict[str, object],
) -> None:
    """The same `pool_id#n` names a different mutation of the same file under another seed, so
    the id alone says nothing about the content. Counted, never dropped."""
    tail = _cand(3)["id"].removeprefix(PFX)
    train_row = {"id": tail, "before": "other\n", "after": "different\n"}
    _two_corpora(
        world,
        [_cand(3), _cand(4)],
        [],
        train_rows=[train_row],
        val_rows=[{"row_id": "unrelated"}],
    )
    report = _plan(world)
    assert report["duplicates_by_set"]["v5-train"] == 0 and report["kept"] == 2
    assert report["id_collisions_by_set"]["v5-train"] == 1


def test_a_row_content_identical_to_a_v5_train_row_is_dropped_under_any_id(
    world: dict[str, object],
) -> None:
    train_row = {"id": "v5:name#0", "before": _cand(3)["before"], "after": _cand(3)["after"]}
    _two_corpora(
        world, [_cand(3), _cand(4)], [], train_rows=[train_row], val_rows=[{"row_id": "unrelated"}]
    )
    report = _plan(world)
    assert report["duplicates_by_set"]["v5-train"] == 1 and report["kept"] == 1


def test_v5_val_matches_through_a_prefixed_id(world: dict[str, object]) -> None:
    """The prefix is declared per corpus and stripped before the id-keyed val comparison, so a
    prefix cannot weaken the exclusion."""
    tail = "p4:f.py#0"
    prefixed = {**_cand(4), "id": f"arm-b-s:{tail}"}
    _two_corpora(
        world,
        [prefixed, {**_cand(5), "id": "arm-b-s:p5:f.py#0"}],
        [],
        train_rows=[{"id": "unrelated"}],
        val_rows=[{"row_id": f"qdm:code.defect_class:{tail}"}],
        prefixes=("arm-b-s:", PFX),
    )
    report = _plan(world)
    assert report["duplicates_by_set"]["v5-val"] == 1 and report["kept"] == 1


def test_a_row_without_its_corpus_declared_prefix_is_refused(world: dict[str, object]) -> None:
    _two_corpora(
        world,
        [_cand(4)],
        [],
        train_rows=[{"id": "unrelated"}],
        val_rows=[{"row_id": "unrelated"}],
        prefixes=("arm-b-s:", PFX),
    )
    with pytest.raises(mine.Refusal, match="prefix"):
        _plan(world)


def test_select_refuses_a_plan_whose_ids_are_not_unique(world: dict[str, object]) -> None:
    a = [{**_cand(i), "id": f"{PFX}compose:train:{i:06d}"} for i in range(3)]
    b = [{**_cand(i + 100), "id": f"{PFX}compose:train:{i:06d}"} for i in range(3)]
    _two_corpora(world, a, b, train_rows=[{"id": "unrelated"}], val_rows=[{"row_id": "unrelated"}])
    plan_dir = Path(str(_plan(world)["out_dir"]))
    verdicts = Path(str(world["tmp"])) / "v.jsonl"
    verdicts.write_text("")
    args = type("A", (), {})()
    args.prereg, args.repo, args.verdicts = world["prereg"], world["repo"], verdicts
    args.plan_dir, args.out = plan_dir, Path(str(world["tmp"])) / "mined2.json"
    with pytest.raises(mine.Refusal, match="not unique"):
        mine._select(args)


# -- round-2 findings (2026-10-13) ---------------------------------------------------------------


def _composed(i: int, diff: str) -> dict[str, object]:
    """A `qd-mutate compose` row: no `before`, the model input is the composed `diff`."""
    return {
        "id": f"{PFX}compose:train:{i:06d}",
        "repo": "own/train-repo",
        "language": "typescript",
        "class": "stub",
        "operator": "stub.panic",
        "span": {"start_line": 5, "end_line": 5},
        "after": "irrelevant to the model input\n",
        "diff": diff,
    }


def test_a_composed_row_is_content_deduped_on_its_diff(world: dict[str, object]) -> None:
    """Composed rows have no `before`, so they used to skip every content check."""
    rows = [_composed(0, "@@ a @@\n+x\n"), _composed(1, "@@ a @@\n+x\n"), _composed(2, "@@ b @@\n")]
    _two_corpora(world, rows, [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}])
    report = _plan(world)
    assert report["duplicates_within"] == 1 and report["kept"] == 2


def test_a_composed_row_content_identical_to_a_v5_train_diff_is_dropped(
    world: dict[str, object],
) -> None:
    train_row = {"id": "t0", "diff": "@@ a @@\n+x\n"}
    _two_corpora(
        world,
        [_composed(0, "@@ a @@\n+x\n"), _composed(1, "@@ other @@\n")],
        [],
        train_rows=[train_row],
        val_rows=[{"row_id": "u"}],
    )
    report = _plan(world)
    assert report["duplicates_by_set"]["v5-train"] == 1 and report["kept"] == 1


def test_a_row_with_no_computable_content_digest_is_refused_not_kept(
    world: dict[str, object],
) -> None:
    bare = {"id": f"{PFX}x", "repo": "own/train-repo", "class": "logic"}
    _two_corpora(world, [bare], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}])
    with pytest.raises(mine.Refusal, match="content digest"):
        _plan(world)


def test_identical_content_with_different_labels_is_counted_as_a_label_conflict(
    world: dict[str, object],
) -> None:
    """The key is byte-exact `before NUL after`: it ignores label and span, so the later of two
    rows with one content and two labels is dropped. Rule: the first row is kept, and the
    conflict is counted."""
    a = {**_cand(1), "class": "logic", "span": {"start_line": 1, "end_line": 1}}
    b = {**_cand(2), "before": a["before"], "after": a["after"], "class": "stub"}
    same_label = {**_cand(3), "before": a["before"], "after": a["after"], "class": "logic",
                  "span": {"start_line": 1, "end_line": 1}}
    _two_corpora(
        world, [a, b, same_label], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}]
    )
    report = _plan(world)
    assert report["duplicates_within"] == 2 and report["kept"] == 1
    assert report["label_conflicts"] == 1
    assert "first row is kept" in report["label_conflict_rule"]


def test_byte_exact_content_whitespace_and_crlf_are_not_normalised(
    world: dict[str, object],
) -> None:
    a = _cand(1)
    b = {**_cand(2), "before": a["before"], "after": str(a["after"]).replace("\n", "\r\n")}
    _two_corpora(world, [a, b], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}])
    report = _plan(world)
    assert report["duplicates_within"] == 0 and report["kept"] == 2


def test_a_preregistration_with_a_corpus_lacking_an_id_prefix_is_refused(
    world: dict[str, object],
) -> None:
    _two_corpora(world, [_cand(1)], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}])
    doc = json.loads(Path(str(world["prereg"])).read_text())
    for bad in (None, ""):
        entry = doc["candidates"]["corpora"][0]
        entry.pop("id_prefix", None)
        if bad is not None:
            entry["id_prefix"] = bad
        Path(str(world["prereg"])).write_text(json.dumps(doc))
        _git(Path(str(world["repo"])), "commit", "-q", "-am", f"no prefix {bad!r}")
        with pytest.raises(mine.Refusal, match="id_prefix"):
            _plan(world)


def test_the_plan_reports_the_drop_and_the_kept_mix_by_class_and_operator(
    world: dict[str, object],
) -> None:
    def row(i: int, cls: str, op: str) -> dict[str, object]:
        return {**_cand(i), "class": cls, "operator": op}

    train_row = {
        "id": "t",
        "before": _cand(1)["before"],
        "after": _cand(1)["after"],
    }
    _two_corpora(
        world,
        [row(1, "stub", "stub.panic"), row(2, "logic", "logic.flip"), row(3, "stub", "stub.panic")],
        [],
        train_rows=[train_row],
        val_rows=[{"row_id": "u"}],
    )
    report = _plan(world)
    assert report["dropped_by_class_and_operator"]["v5-train"] == {
        "class": {"stub": 1},
        "operator": {"stub.panic": 1},
    }
    assert report["kept_by_class_and_operator"] == {
        "class": {"logic": 1, "stub": 1},
        "operator": {"logic.flip": 1, "stub.panic": 1},
    }


# -- materialise ----------------------------------------------------------------------------------


def _materialise(world: dict[str, object], plan: dict[str, object], **over: object) -> dict:
    tmp = Path(str(world["tmp"]))
    kwargs: dict[str, object] = {
        "prereg": world["prereg"],
        "repo": world["repo"],
        "plan_dir": Path(str(plan["out_dir"])),
        "out_dir": tmp / "materialised",
        "pool_dir": tmp / "pool",
    }
    kwargs.update(over)
    return mine.materialise(**kwargs)  # type: ignore[arg-type]


def _lines(path: Path) -> list[bytes]:
    return [line for line in path.read_bytes().split(b"\n") if line]


def test_materialise_writes_each_corpus_with_only_the_rows_the_plan_kept(
    world: dict[str, object],
) -> None:
    tmp = Path(str(world["tmp"]))
    a = [_cand(1), _cand(2), _cand(3)]
    b = [_cand(10), _cand(11)]
    drop = {"id": "t", "before": _cand(2)["before"], "after": _cand(2)["after"]}
    _two_corpora(world, a, b, train_rows=[drop], val_rows=[{"row_id": "u"}])
    plan = _plan(world)
    assert plan["kept"] == 4
    out = _materialise(world, plan)
    assert out["corpora"] == {"c1": {"kept": 2, "dropped": 1}, "c2": {"kept": 2, "dropped": 0}}
    src = tmp / "data" / "c1" / "examples.jsonl"
    got = out_dir = tmp / "materialised" / "c1"
    assert _lines(got / "examples.jsonl") == [_lines(src)[0], _lines(src)[2]], "byte for byte"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["examples_sha256"] == _sha((got / "examples.jsonl").read_bytes())
    assert manifest["totals"]["examples"] == 2
    assert manifest["totals"]["by_class"] == {"unfiltered": 3}, "other totals are left as they were"
    sel = manifest["selected_by_plan"]
    assert (sel["kept"], sel["dropped"]) == (2, 1)
    assert sel["candidates_sha256"] == plan["candidates_sha256"]
    assert sel["source_examples_sha256"] == _sha(src.read_bytes())


def test_materialise_gives_a_composed_corpus_an_empty_plan_base(world: dict[str, object]) -> None:
    """The loader reads a composed corpus's base rows first, and the plan holds none of them."""
    tmp = Path(str(world["tmp"]))
    _two_corpora(
        world, [_cand(1), _cand(2)], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}]
    )
    manifest_path = tmp / "data" / "c1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_corpus"] = {"name": "base-x", "examples_sha256": "a" * 64}
    manifest_path.write_text(json.dumps(manifest))
    base = tmp / "pool" / "base-x"
    base.mkdir(parents=True)
    (base / "manifest.json").write_text(
        json.dumps({"examples_sha256": "a" * 64, "totals": {"examples": 99}, "pool": {"k": 1}})
    )
    out = _materialise(world, _plan(world))
    got = json.loads((tmp / "materialised" / "c1" / "manifest.json").read_text())
    empty = _sha(b"")
    assert got["base_corpus"] == {"name": "c1-planbase", "examples_sha256": empty}
    assert got["selected_by_plan"]["base_corpus_original"]["name"] == "base-x"
    assert got["selected_by_plan"]["base_corpus_rows_excluded"] == 99
    planbase = tmp / "pool" / "c1-planbase"
    assert (planbase / "examples.jsonl").read_bytes() == b""
    pb = json.loads((planbase / "manifest.json").read_text())
    assert pb["examples_sha256"] == empty and pb["totals"]["examples"] == 0
    assert pb["pool"] == {"k": 1}, "the base's pool is kept: the loader checks it matches"
    assert out["corpora"]["c1"]["kept"] == 2


def test_materialise_refuses_a_plan_from_another_preregistration(
    world: dict[str, object],
) -> None:
    _two_corpora(world, [_cand(1)], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}])
    plan = _plan(world)
    doc = json.loads(Path(str(world["prereg"])).read_text())
    doc["caps"]["floor_rows"] = 4
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "-am", "another")
    with pytest.raises(mine.Refusal, match="another pre-registration"):
        _materialise(world, plan)


def _pin_pool_manifest(world: dict[str, object], sha: str | None, *, own: bool = True) -> Path:
    """Declare an own-repos corpus in the prereg, pinning (or not) its pool manifest."""
    tmp = Path(str(world["tmp"]))
    manifest = tmp / "data" / "own-pool.manifest.json"
    manifest.write_text('{"source_id": "bharathvbcr/own-repositories"}')
    doc = json.loads(Path(str(world["prereg"])).read_text())
    entry = doc["candidates"]["corpora"][0]
    entry["own_repos_train"] = own
    if sha is not None:
        entry["pool_manifest"] = {"path": str(manifest), "sha256": sha}
    Path(str(world["prereg"])).write_text(json.dumps(doc))
    _git(Path(str(world["repo"])), "commit", "-q", "--allow-empty", "-am", "pin pool manifest")
    return manifest


def test_an_own_repos_corpus_must_pin_its_pool_manifest_and_the_pin_must_hold(
    world: dict[str, object],
) -> None:
    _two_corpora(world, [_cand(1)], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}])
    manifest = _pin_pool_manifest(world, None)
    with pytest.raises(mine.Refusal, match="pool_manifest"):
        _plan(world)
    _pin_pool_manifest(world, "0" * 64)
    with pytest.raises(mine.Refusal, match="pool manifest"):
        _plan(world)
    _pin_pool_manifest(world, _sha(manifest.read_bytes()))
    assert _plan(world)["kept"] == 1


def test_materialise_accepts_an_identical_planbase_and_refuses_a_different_one(
    world: dict[str, object],
) -> None:
    tmp = Path(str(world["tmp"]))
    _two_corpora(world, [_cand(1)], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}])
    manifest_path = tmp / "data" / "c1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_corpus"] = {"name": "base-x", "examples_sha256": "a" * 64}
    manifest_path.write_text(json.dumps(manifest))
    base = tmp / "pool" / "base-x"
    base.mkdir(parents=True)
    (base / "manifest.json").write_text(
        json.dumps({"examples_sha256": "a" * 64, "totals": {"examples": 5}, "pool": {"k": 1}})
    )
    plan = _plan(world)
    _materialise(world, plan)
    # Again, into a new out dir: the identical planbase is accepted, not a refusal.
    _materialise(world, plan, out_dir=tmp / "materialised-again")
    (tmp / "pool" / "c1-planbase" / "manifest.json").write_text('{"different": true}')
    with pytest.raises(mine.Refusal, match="planbase"):
        _materialise(world, plan, out_dir=tmp / "materialised-third")


def test_materialise_refuses_tampered_candidates_and_an_existing_out_dir(
    world: dict[str, object],
) -> None:
    _two_corpora(world, [_cand(1)], [], train_rows=[{"id": "u"}], val_rows=[{"row_id": "u"}])
    plan = _plan(world)
    cand = Path(str(plan["out_dir"])) / "candidates.jsonl"
    original = cand.read_bytes()
    cand.write_bytes(original + b"{}\n")
    with pytest.raises(mine.Refusal, match="no longer hashes"):
        _materialise(world, plan)
    cand.write_bytes(original)
    Path(str(world["tmp"]), "materialised").mkdir()
    with pytest.raises(mine.Refusal, match="exists"):
        _materialise(world, plan)
