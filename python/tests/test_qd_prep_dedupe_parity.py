"""``qd-prep dedupe`` against ``qd_data.dedupe.dedupe``, the reference it ports.

The binary computes the MinHash path -- sign, band (with v6's agreement prefilter when set),
confirm by exact Jaccard, union across repos, keep -- over the content units, shingle sets and
split ranks the reference derives. These tests hold the two together three ways:

* the committed fixture ``crates/qd-prep/tests/fixtures/dedupe-parity.json`` is the reference's
  own answer on its units (no binary needed), so the Rust test that reads it
  (``crates/qd-prep/tests/dedupe_parity.rs``) is pinned to the oracle, not to a transcription;
* the binary reproduces every fixture case: v5's lexical keep, v6's split-priority keep with the
  prefilter, and a truncated search;
* the binary reproduces the reference on a mixture corpus under both rule sets and two bounds.

A dedupe that could not run is SKIPPED with the reason (``qd_prep_bin``), never passed.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
from data_fixtures import code_body, commitpackft_row, small_corpus, vendored_pair

from qd_data.config import DEDUPE_KEEP_SPLIT_PRIORITY, DataConfig
from qd_data.dedupe import (
    DedupeReport,
    _build_units,
    band_config_for,
    dedupe,
    near_duplicate_policy,
    unit_split_ranks,
)
from qd_data.minhash import MinHasher, shingle
from qd_data.mixture import build_mixture
from qd_data.rows import DataRow, GoldAnswer
from qd_data.schema import ChoiceSlot, Request
from qd_data.sources import PINNED_SPLIT_KEY
from qd_train.tristate import NotRun

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "crates" / "qd-prep" / "tests" / "fixtures" / "dedupe-parity.json"
REQUEST_MAGIC = b"QDPDDIN1"
REPLY_MAGIC = b"QDPDDOK1"
MINHASH_MAGIC = b"QDPMHIN1"
KEEP_CODES = {"lexical": 0, DEDUPE_KEEP_SPLIT_PRIORITY: 1}


def _row(row_id: str, *, repo: str, text: str, pinned: str | None,
         family: str = "code.commit_intent", path: str = "a.py") -> DataRow:
    return DataRow(
        row_id=row_id,
        source_id="bigcode/commitpackft",
        host="huggingface",
        family_id=family,
        repo_key=repo,
        identity_key=f"{repo}::{path}",
        licence_id="mit",
        obligations=("attribution",),
        request=Request(
            task=family,
            context=text.encode("utf-8"),
            question="q?",
            slots=(ChoiceSlot(name="implements_claim", options=("yes", "no")),),
            example_id=row_id,
        ),
        gold=(GoldAnswer(slot_name="implements_claim", value="yes"),),
        dedupe_text=text,
        metadata={} if pinned is None else {PINNED_SPLIT_KEY: pinned},
    )


def fixture_rows() -> list[DataRow]:
    """Built to separate the rules: train keys sort first in every cross-split component, one
    component spans all three splits plus a near (not exact) copy, one unit is held out by its
    family rather than its split, one pair sits inside one repo, one unit stands alone."""
    ta, tb, tc, td, te = (code_body(t, lines=10) for t in ("ta", "tb", "tc", "td", "te"))
    return [
        _row("a", repo="org/a-train", text=ta, pinned="train"),
        _row("b", repo="org/b-val", text=ta, pinned="val"),
        _row("c", repo="org/c-train", text=tb, pinned="train"),
        _row("d", repo="org/d-heldout", text=tb, pinned="heldout"),
        _row("e", repo="org/e-val", text=tb, pinned="val"),
        _row("f", repo="org/f-train", text=tb + "\n    extra_line = 1", pinned="train"),
        _row("g1", repo="org/g", text=tc, pinned="train", path="x.py"),
        _row("g2", repo="org/g", text=tc, pinned="train", path="y.py"),
        _row("h", repo="org/h", text=td, pinned="train"),
        _row("i1", repo="org/i-train", text=te, pinned="train"),
        _row("i2", repo="org/i-train", text=te, pinned="train", family="code.language_id"),
        _row("j", repo="org/j-val", text=te, pinned="val"),
    ]


def _units(rows: list[DataRow], config: DataConfig) -> list[dict[str, Any]]:
    """The searched content units the reference builds, in key order, with their shingles
    (sorted, as text) and split ranks."""
    units = _build_units(tuple(rows))
    policies = {r.row_id: near_duplicate_policy(r) for r in rows}
    searched = {k: u for k, u in units.items() if all(policies[i] is None for i in u.row_ids)}
    ranks = unit_split_ranks(rows, units, config=config)
    return [
        {
            "key": key,
            "repo": searched[key].repo_key,
            "split_rank": ranks[key],
            "shingles": sorted(
                s.decode("utf-8")
                for s in shingle(searched[key].text, k=config.shingle_size).shingles
            ),
        }
        for key in sorted(searched)
    ]


def _f64_bits(x: float) -> int:
    """``x``'s IEEE-754 bits: compared exactly, never through a decimal round trip."""
    bits: int = struct.unpack("<Q", struct.pack("<d", x))[0]
    return bits


def _expect(report: DedupeReport) -> dict[str, Any]:
    truncated = isinstance(report.status, NotRun) and "hit its bound" in report.status.reason
    return {
        "truncated": truncated,
        "n_candidate_pairs": report.n_candidate_pairs,
        "n_cross_repo_pairs": report.n_cross_repo_pairs,
        "n_within_repo_pairs": report.n_within_repo_pairs,
        "clusters": [
            {
                "kept": c.kept_unit_key,
                "dropped": list(c.dropped_unit_keys),
                "min_edge_jaccard_bits": _f64_bits(c.min_edge_jaccard),
            }
            for c in report.clusters
        ],
    }


def _case(name: str, rows: list[DataRow], config: DataConfig, max_pairs: int) -> dict[str, Any]:
    banding = band_config_for(config)
    return {
        "name": name,
        "threshold": config.dedupe_threshold,
        "bands": banding.bands,
        "rows": banding.rows,
        "min_agreement_permille": banding.min_agreement_permille,
        "keep_rule": KEEP_CODES[config.dedupe_keep_rule],
        "max_pairs": max_pairs,
        "expect": _expect(dedupe(rows, config=config, max_candidate_pairs=max_pairs)),
    }


def build_fixture() -> dict[str, Any]:
    """The fixture, from the reference alone."""
    rows = fixture_rows()
    v5, v6 = DataConfig(), DataConfig().with_v6_dedupe_rules()
    hasher = MinHasher(num_perm=v5.num_perm, seed=v5.seed)
    return {
        "generated_by": "python/tests/test_qd_prep_dedupe_parity.py::build_fixture",
        "family": {"key_hex": hasher._key.hex(), "a": list(hasher._a), "b": list(hasher._b)},
        "units": _units(rows, v6),
        "cases": [
            _case("v5-lexical", rows, v5, v5.max_candidate_pairs),
            _case("v6-split-priority-prefiltered", rows, v6, v6.max_candidate_pairs),
            _case("v6-truncated", rows, v6, 2),
        ],
    }


def encode_request(family: dict[str, Any], units: list[dict[str, Any]],
                   case: dict[str, Any]) -> bytes:
    """A ``QDPDDIN1`` request (``crates/qd-prep/src/dedupe.rs``)."""
    keys = [u["key"].encode("utf-8", "surrogatepass") for u in units]
    repos = [u["repo"].encode("utf-8", "surrogatepass") for u in units]
    n = len(units)
    key = bytes.fromhex(family["key_hex"])
    k = len(family["a"])
    parts = [
        REQUEST_MAGIC,
        struct.pack("<dIIIIQQ", case["threshold"], case["bands"], case["rows"],
                    case["min_agreement_permille"], case["keep_rule"], case["max_pairs"], n),
        struct.pack(f"<{n}I", *map(len, keys)), b"".join(keys),
        struct.pack(f"<{n}I", *map(len, repos)), b"".join(repos),
        bytes(u["split_rank"] for u in units),
        MINHASH_MAGIC, struct.pack("<II", k, len(key)), key,
        struct.pack(f"<{k}Q", *family["a"]), struct.pack(f"<{k}Q", *family["b"]),
        struct.pack("<Q", n),
    ]
    for u in units:
        items = [s.encode("utf-8") for s in u["shingles"]]
        parts.append(struct.pack(f"<I{len(items)}I", len(items), *map(len, items)))
        parts.append(b"".join(items))
    return b"".join(parts)


def decode_reply(reply: bytes, keys: list[str]) -> dict[str, Any]:
    """A ``QDPDDOK1`` reply, checked to its last byte, in :func:`_expect`'s shape."""
    assert reply[:8] == REPLY_MAGIC, reply[:8]
    flag, n_cand, n_cross, n_within, n_clusters = struct.unpack_from("<BQQQQ", reply, 8)
    assert flag in (0, 1)
    at = 8 + 1 + 4 * 8
    clusters = []
    for _ in range(n_clusters):
        kept, n_dropped = struct.unpack_from("<II", reply, at)
        at += 8
        dropped = struct.unpack_from(f"<{n_dropped}I", reply, at)
        at += 4 * n_dropped
        (bits,) = struct.unpack_from("<Q", reply, at)
        at += 8
        clusters.append({"kept": keys[kept], "dropped": [keys[d] for d in dropped],
                         "min_edge_jaccard_bits": bits})
    assert at == len(reply), "bytes follow the last cluster"
    return {"truncated": flag == 1, "n_candidate_pairs": n_cand, "n_cross_repo_pairs": n_cross,
            "n_within_repo_pairs": n_within, "clusters": clusters}


def _run(binary: Path, request: bytes, tmp_path: Path, name: str) -> bytes:
    import subprocess

    src, out = tmp_path / f"{name}.req", tmp_path / f"{name}.reply"
    src.write_bytes(request)
    done = subprocess.run([str(binary), "dedupe", "--input", str(src), "--output", str(out)],
                          capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr
    return out.read_bytes()


# -- no binary: the committed fixture is the oracle's answer -------------------------------------


def test_the_committed_fixture_is_the_references_answer() -> None:
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == build_fixture()


def test_the_fixture_separates_the_rules() -> None:
    """Vacuity guard: the two keep rules disagree on every cross-split component, the truncated
    case truncates, and the within-repo pair is counted, not merged."""
    cases = {c["name"]: c["expect"] for c in build_fixture()["cases"]}
    v5, v6 = cases["v5-lexical"], cases["v6-split-priority-prefiltered"]
    assert [c["kept"] for c in v5["clusters"]] != [c["kept"] for c in v6["clusters"]]
    assert {c["kept"].split("::")[0] for c in v6["clusters"]} == {
        "org/b-val", "org/d-heldout", "org/i-train"
    }
    assert all(c["kept"].split("::")[0].endswith("-train") for c in v5["clusters"])
    assert v5["n_within_repo_pairs"] == v6["n_within_repo_pairs"] == 1
    assert cases["v6-truncated"]["truncated"] and not v6["truncated"]


# -- the binary -----------------------------------------------------------------------------------


def test_qd_prep_dedupe_reproduces_every_fixture_case(qd_prep_bin: Path, tmp_path: Path) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    keys = [u["key"] for u in fixture["units"]]
    for case in fixture["cases"]:
        reply = _run(qd_prep_bin, encode_request(fixture["family"], fixture["units"], case),
                     tmp_path, case["name"])
        assert decode_reply(reply, keys) == case["expect"], case["name"]


def _corpus_rows() -> list[DataRow]:
    corpus = small_corpus(18)
    corpus["bigcode/commitpackft"].extend(vendored_pair())
    near = code_body("near", lines=40)
    corpus["bigcode/commitpackft"].extend(
        commitpackft_row(100 + j, repo=f"org/near{j}", body=f"{near}\n    tweak_{j} = {j}")
        for j in range(6)
    )
    return list(build_mixture(corpus, config=DataConfig()).rows)


@pytest.mark.parametrize("rules", ["v5", "v6"])
@pytest.mark.parametrize("max_pairs", [3, 5_000_000])
def test_qd_prep_dedupe_reproduces_the_reference_on_a_mixture(
    qd_prep_bin: Path, tmp_path: Path, rules: str, max_pairs: int
) -> None:
    config = DataConfig() if rules == "v5" else DataConfig().with_v6_dedupe_rules()
    rows = _corpus_rows()
    want = _case(f"{rules}-{max_pairs}", rows, config, max_pairs)
    assert want["expect"]["clusters"], "the corpus must exercise a duplicate"
    hasher = MinHasher(num_perm=config.num_perm, seed=config.seed)
    family = {"key_hex": hasher._key.hex(), "a": list(hasher._a), "b": list(hasher._b)}
    units = _units(rows, config)
    reply = _run(qd_prep_bin, encode_request(family, units, want), tmp_path, want["name"])
    assert decode_reply(reply, [u["key"] for u in units]) == want["expect"]
