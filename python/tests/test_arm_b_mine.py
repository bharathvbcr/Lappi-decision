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
        "id": f"p{i}:f.py#0",
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
                {"name": "c1", "path": str(corpus), "sha256": _sha(corpus_bytes), "rows": 10}
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
    renamed = {**first, "id": "q99:g.py#0"}
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
    row = {**_cand(1), "id": corpus_id, "pool_id": corpus_id.split("#")[0]}
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
