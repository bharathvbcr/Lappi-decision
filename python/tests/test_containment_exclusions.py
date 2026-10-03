"""The v5 decontamination exclusion (Fable's 2026-10-02 ruling, Q2 "The rule, end to end").

``qd-prep containment`` writes ``exclusions.txt`` (identity keys of train rows that overlap a
val or repo-held-out row) beside its attestation v2. ``tools/real_tokenizer_pipeline.py
--exclude-identity-keys FILE`` and ``tools/real_ft_run.py``'s ``ft_splits`` rebuild drop those
train rows through one function, ``qd_train.exclusions.apply_exclusions``, applied once
before ``split_off_replay``.

The first two tests are characterization tests, written and pinned at d554702 (main dec48d8
plus L-replay's e571065) BEFORE any of this lane's edits: without the flag, a shard header and
a small pipeline build are byte-identical to what they were. The rest drive the list through
the loader, the pipeline and the trainer's rebuild.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from containment_scan import (  # noqa: E402
    run_containment,
    scan_sets,
    scan_specs,
    splitter_checks,
    write_request,
)
from data_fixtures import small_corpus  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.loaders import MmluRow  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.split import SplitReport, split  # noqa: E402
from qd_train.artifacts import ShardContractViolation, ShardHeader  # noqa: E402
from qd_train.containment_strip import STRIP_RULE  # noqa: E402
from qd_train.exclusions import (  # noqa: E402
    ATTESTATION_NAME,
    EXCLUSIONS_NAME,
    ExclusionRefusal,
    apply_exclusions,
    read_exclusions,
)

DOWNLOAD = REPO / "data" / "pool" / "commitpackft"
#: A fixed corpus revision, so the pin does not move with HEAD (``repo_history=False`` reads
#: nothing from it; it is recorded in the header as ``corpus_rev``).
PIN_REV = "dec48d8841db21ad0401effdb9fff724977fd47f"
#: The tokenizer the pinned build was written with; another snapshot writes other shards.
PIN_TOKENIZER_SHA256 = "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"


def _header() -> ShardHeader:
    return ShardHeader(
        split="train", data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
        remap_hash="r" * 64, vocab_size=1000, n_sequences=3, total_tokens=30,
        max_seq_len=16, buckets=(8, 16), created_at="2026-10-02T00:00:00+00:00",
        code_fingerprint={"qd_data/a.py": "a" * 64}, corpus_rev="c" * 40,
        sequence_index_hash="s" * 64, span_collapse_policy="refuse-gold",
    )


def test_a_header_without_exclusions_hashes_and_serialises_as_before() -> None:
    """Characterization, pinned at d554702 before ``exclusions_sha256`` existed."""
    header = _header()
    assert header.shard_hash() == (
        "2a184184b12e4004970a169fa377c78c7e5cd3abecd873ab517a64c633bd343e"
    )
    body = json.dumps(header.to_json(), sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest() == (
        "9b03861d4d377d9bfad14e15908e3f41a9fa770fd4b5d4f93155bb212e4b52bd"
    )


def _normalised(path: Path) -> bytes:
    """A build output's bytes, less the two fields that record when it was written."""
    raw = path.read_bytes()
    if path.suffix != ".json":
        return raw
    body = json.loads(raw)
    if path.name == "header.json":
        body.pop("created_at", None)
    if isinstance(body, dict) and isinstance(body.get("provenance"), dict):
        body["provenance"].pop("built_at_utc", None)
    return json.dumps(body, sort_keys=True).encode()


def build_digest(out: Path) -> str:
    """One sha256 over every file a pipeline build wrote under ``out``, path by path."""
    h = hashlib.sha256()
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(out)).encode() + b"\n")
        h.update(hashlib.sha256(_normalised(path)).hexdigest().encode() + b"\n")
    return h.hexdigest()


def _pipeline_or_skip():
    pytest.importorskip("transformers")
    import real_tokenizer_pipeline as pipeline

    if not pipeline.MODEL_REF.exists():
        pytest.skip(f"{pipeline.MODEL} is not in this host's HF cache")
    snapshot = pipeline.MODEL_REF.parent.parent / "snapshots" / pipeline.MODEL_REF.read_text(
        encoding="utf-8").strip() / "tokenizer.json"
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != PIN_TOKENIZER_SHA256:
        pytest.skip(f"{snapshot} is not the tokenizer the pin was written with")
    if not all((DOWNLOAD / f"{lang}.jsonl").exists() for lang in ("go", "python")):
        pytest.skip("the commitpackft download is not on this host; only its manifest is")
    return pipeline


@pytest.mark.usefixtures("qd_prep")
def test_a_build_without_the_flag_writes_what_it_wrote_before(tmp_path: Path) -> None:
    """Characterization: every shard file, sequence index and manifest of a 60-pair build,
    timestamps aside, is the bytes the pipeline wrote before the flag existed.

    Pinned at d554702 as d67c7de0.... It has moved seven times since, each time by design and
    with every moved byte accounted for. No move was a re-pin to whatever came out:

    * L-v5-data -> 92299328...: the tokens, offsets and supervision are byte-identical. The
      headers differ only in five qd_data code fingerprints and their derived hashes, and the
      manifests only in admitted_source_ids and their derived hashes.
      See GAP-L-V5DATA-CHARACTERIZATION-DIGEST-MOVES-2026-10-02.
    * Prompt format 2 -> 3c9ae50a.... Every sequence of all three shard sets (86/86/12) decodes
      to the format-1 text with exactly two edits: the format line inserted after the begin
      line, and the question line moved after the context. Each sequence is 9 tokens longer;
      target_index shifts by that delta; the other supervision arrays are identical. The headers
      gain prompt_format 2 and move only in fingerprints and derived fields. Unexplained: 0.
      See AUDIT/v5-fmt-characterization-2026-10-02/ and
      GAP-L-V5FMT-CHARACTERIZATION-CRITERION-STALE-2026-10-02.
    * b11e6e0 (CLINC not reportable) -> 9674f8fc..., unseen at the time because this test
      skips on a host without the commitpackft download. A build at adbaec1 reproduces
      3c9ae50a...; against it, 30 of 33 files are byte-identical and the three shard headers
      differ only in code_fingerprint["sources.py"] and the shard_hash that covers it.
    * The general-decision pool (bench v2 patch) -> d998d6d8...: against the 9674f8fc... build,
      30 of 33 files are byte-identical, every manifest included (so admitted_source_ids and
      data_snapshot_hash hold for a build without --decisions-pool), and the three headers
      differ only in code_fingerprint (decisions.py added; loaders, manifest, mixture, sources
      changed) and shard_hash. See
      GAP-CHARACTERIZATION-PIN-STALE-SINCE-B11E6E0-SKIPPED-WITHOUT-COMMITPACKFT-2026-10-03.
    * The pool's share-alike sources and probability check (bench v3 patch) -> acd6ea15...:
      against the d998d6d8... build (build/char-v2/out-new, compared with its compare.py), 30 of
      33 files are byte-identical, every manifest included, and the three shard headers differ
      only in code_fingerprint["decisions.py"], code_fingerprint["sources.py"] and the
      shard_hash that covers them. Differing binary files: 0.
    * Exact-content dedupe for structured decision rows and the config-carried candidate bound
      (the lead's v4 half, Fable's dedupe ruling) -> d65616af...: against the acd6ea15...
      build (the bench's char-v3/out-v3, whose digest reproduces acd6ea15...), 30 of 33 files
      are byte-identical, every manifest included -- a corpus with no scoped row and the default
      bound writes no new report key -- and the three shard headers differ only in
      code_fingerprint["config.py"], ["dedupe.py"], ["split.py"] and the shard_hash that
      covers them. Differing binary files: 0.
      AUDIT/finalize-2026-10-03/dedupe-probe/char-v4-compare.out.
    * The pool's v4 half (bench): the exact-content marker, the pool's candidate bound in
      pool_data_config, VitaminC in and two new pool targets -> 82e412c4...: against the
      d65616af... build (build/char-v4/out-new, compared with char-v2's compare.py), 30 of 33
      files are byte-identical, every manifest included, and the three shard headers differ
      only in code_fingerprint["decisions.py"], code_fingerprint["sources.py"] and the
      shard_hash that covers them. Differing binary files: 0."""
    pipeline = _pipeline_or_skip()
    pipeline.run(
        out=tmp_path, max_pairs=60, blank_line_runs=False, rev=PIN_REV, commitpackft=DOWNLOAD,
        val_shards=True, repo_history=False,
    )
    assert build_digest(tmp_path) == (
        "82e412c41fadf0b47965b9d0b4b4d3d25f54785be12dd53601c14a27e79e4287"
    )


# --- the header ------------------------------------------------------------------------------


def test_a_train_header_names_its_exclusion_list_and_hashes_apart() -> None:
    plain = _header()
    named = dataclasses.replace(plain, exclusions_sha256="e" * 64)
    assert named.to_json()["exclusions_sha256"] == "e" * 64
    assert "exclusions_sha256" not in plain.to_json()
    assert named.shard_hash() != plain.shard_hash()
    assert ShardHeader.from_json(named.to_json()) == named
    with pytest.raises(ShardContractViolation, match="not a lower-case sha256"):
        dataclasses.replace(plain, exclusions_sha256="E" * 64)
    with pytest.raises(ShardContractViolation, match="removes train rows only"):
        dataclasses.replace(plain, split="val", exclusions_sha256="e" * 64)


# --- the scan, the list and the one function that applies it -----------------------------------

CORPUS: dict[str, object] = {"fixture": "small_corpus(24) + contaminated MMLU"}
VOCAB = [
    "anchor", "basin", "cobalt", "delta", "ember", "fjord", "glacier", "harbor", "island",
    "jungle", "kettle", "lantern", "meadow", "nectar", "orchard", "pepper", "quarry", "raven",
    "saddle", "timber", "umber", "velvet", "willow", "xenon", "yonder", "zephyr", "amber",
    "bramble", "cinder", "dapple", "ferric", "gossamer", "hollow", "indigo", "jasper", "kelp",
    "lichen", "marrow",
]


def _contaminated_corpus() -> dict[str, list[object]]:
    """``small_corpus(24)`` plus MMLU rows of two kinds: a val row (``validation`` is pinned to
    val) whose question is twenty words, and a train row (``test`` is trained on) whose
    question is that row's twenty words and thirty more -- under half its MinHash shingles,
    so dedupe keeps both, and all of the val row's question 8-grams, so containment finds
    it. Forty clean train rows besides, so the replay draw is not only contaminated rows."""
    rng = random.Random(20261002)
    raw = small_corpus(24)
    mmlu: list[MmluRow] = []
    for j in range(12):
        q = " ".join(rng.choice(VOCAB) for _ in range(20))
        tail = " ".join(rng.choice(VOCAB) for _ in range(30))
        mmlu.append(MmluRow(subject=f"s{j % 4}", question=f"{q}?",
                            choices=(f"a{j}", f"b{j}", f"c{j}", f"d{j}"), answer_index=0,
                            upstream_split="validation"))
        mmlu.append(MmluRow(subject=f"s{j % 4}", question=f"{q} {tail}?",
                            choices=(f"e{j}", f"f{j}", f"g{j}", f"h{j}"), answer_index=1,
                            upstream_split="test"))
    for j in range(40):
        q = " ".join(rng.choice(VOCAB) for _ in range(25))
        mmlu.append(MmluRow(subject=f"s{j % 4}", question=f"clean {j} {q}?",
                            choices=(f"p{j}", f"q{j}", f"r{j}", f"s{j}"), answer_index=2,
                            upstream_split="test"))
    raw["cais/mmlu"] = mmlu
    return raw


def _report() -> SplitReport:
    config = DataConfig()
    mixture = build_mixture(_contaminated_corpus(), config=config)
    return split(dedupe(list(mixture.rows), config=config), config=config)


def _scan(binary: Path, report: SplitReport, out: Path) -> dict[str, Any]:
    config = DataConfig()
    sets, strip = scan_sets(report, config=config)
    request = out.with_name(out.name + ".request.bin")
    with request.open("xb") as fh:
        write_request(fh, sets, scan_specs(sets), corpus=CORPUS,
                      export={"template_strip": strip}, checks=splitter_checks(report))
    run_containment(binary, request, out, threads=4, timeout_s=120.0)
    return json.loads((out / ATTESTATION_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scanned(qd_prep_bin: Path, tmp_path_factory: pytest.TempPathFactory):
    """The contaminated corpus's split report and its scan, once per module."""
    report = _report()
    out = tmp_path_factory.mktemp("scan") / "out"
    return report, out, _scan(qd_prep_bin, report, out)


def test_without_a_list_the_split_is_the_split_it_was() -> None:
    report = _report()
    got, exclusions, excluded = apply_exclusions(report, None, corpus=CORPUS)
    assert got is report and exclusions is None and excluded == ()


def test_the_scan_and_the_hook_take_out_exactly_the_contaminated_train_rows(
    qd_prep_bin: Path, scanned, tmp_path: Path,
) -> None:
    report, out, att = scanned
    assert att["clean"] is True, att["not_clean_because"]
    keys = (out / EXCLUSIONS_NAME).read_text(encoding="utf-8").splitlines()
    assert len(keys) >= 6, "the contaminated MMLU train rows were not found"
    new, exclusions, excluded = apply_exclusions(report, out / EXCLUSIONS_NAME, corpus=CORPUS)
    assert exclusions is not None and exclusions.keys == frozenset(keys)
    assert {r.identity_key for r in excluded} == set(keys)
    train = report.rows_by_split["train"]
    assert list(new.rows_by_split["train"]) == [r for r in train if r.identity_key not in keys]
    # Rule 2: val and held-out are the very tuples they were.
    for name in ("val", "heldout"):
        assert new.rows_by_split[name] is report.rows_by_split[name]
    # Independently of the attestation's own count: a fresh scan of what is left finds no
    # enforced pair at all.
    again = _scan(qd_prep_bin, new, tmp_path / "again")
    enforced = [s for s in again["scans"] if s["enforced"]]
    assert enforced and all(s["pairs"] == 0 for s in enforced), enforced
    assert again["n_exclusions"] == 0
    # The replay draw takes the decontaminated split and is handed no contaminated key.
    import real_tokenizer_pipeline as pipeline

    gold, replay, _part = pipeline.split_off_replay(new, seed=DataConfig().seed)
    drawn = {r.identity_key for r in replay.rows_by_split["train"]}
    assert drawn and drawn.isdisjoint(keys)
    assert gold.rows_by_split["val"] is report.rows_by_split["val"]


def _copy(out: Path, tmp: Path, *, edit_att=None, keys: list[str] | None = None) -> Path:
    """A copy of the scan's directory, its list replaced by ``keys`` (the attestation's sha
    and count then follow it, so only the property under test is wrong) and its attestation
    edited by ``edit_att``."""
    dest = tmp / "copy"
    shutil.copytree(out, dest)
    att = json.loads((dest / ATTESTATION_NAME).read_text(encoding="utf-8"))
    if keys is not None:
        raw = "".join(f"{k}\n" for k in keys).encode()
        (dest / EXCLUSIONS_NAME).write_bytes(raw)
        att["exclusions_sha256"] = hashlib.sha256(raw).hexdigest()
        att["n_exclusions"] = len(keys)
    if edit_att is not None:
        edit_att(att)
    (dest / ATTESTATION_NAME).write_text(json.dumps(att), encoding="utf-8")
    return dest / EXCLUSIONS_NAME


def _set(key: str, value: object):
    def edit(att: dict[str, Any]) -> None:
        att[key] = value
    return edit


def _set_strip(key: str, value: object):
    def edit(att: dict[str, Any]) -> None:
        att["export"]["template_strip"][key] = value
    return edit


@pytest.mark.parametrize(("edit", "match"), [
    (_set("export", {}), "did not apply v5's template strip"),
    (_set_strip("applied", False), "did not apply v5's template strip"),
    (_set_strip("version", 0), "did not apply v5's template strip"),
    (_set_strip("rule", "another rule"), "did not apply v5's template strip"),
    (_set("clean", False), "not CLEAN"),
    (_set("remaining_hits", {"val": 1, "heldout": 0}), "not CLEAN"),
    (_set("remaining_hits", {}), "not CLEAN"),
    (_set("excluded_from", "val"), "not CLEAN"),
    (_set("version", 1), "not a version 2"),
    (_set("tool", "tools/replay_decontam.py"), "not qd-prep"),
    (_set("n", 7), "not the rule's"),
    (_set("threshold", 0.4), "not the rule's"),
    (_set("exclusions_sha256", "0" * 64), "vouches for"),
    (_set("n_exclusions", 0), "counts 0 keys"),
    (_set("corpus", {"fixture": "another corpus"}), "made for corpus"),
])
def test_an_attestation_that_does_not_vouch_for_this_build_is_refused(
    scanned, tmp_path: Path, edit, match: str,
) -> None:
    report, out, _att = scanned
    listed = _copy(out, tmp_path, edit_att=edit)
    with pytest.raises(ExclusionRefusal, match=match):
        apply_exclusions(report, listed, corpus=CORPUS)


#: Version 1's STRIP_RULE, byte for byte as L-prep2 shipped it (369278e). Spelled out, not
#: imported: a test that read the module's constant would pass against version 1.
STRIP_RULE_V1 = (
    "campaign/v5-preregistered data.decontamination.rule: constant template text stripped "
    "before n-gramming, for containment only (Fable post-F ruling 3(a))"
)


@pytest.mark.parametrize(("version", "rule"), [
    (1, STRIP_RULE_V1),  # a version-1 list, as a full scan before version 2 would write it
    (2, STRIP_RULE_V1),  # the version bumped without the rule
    (1, STRIP_RULE),  # the rule renamed without the version
])
def test_a_version_1_strip_attestation_is_refused_by_the_hook(
    scanned, tmp_path: Path, version: int, rule: str,
) -> None:
    """Fable's CLINC strip ruling section 5 (GAP-CONTAINMENT-STRIP-V1-WINDOW-2026-10-02):
    until version 2, a version-1 list passed ``read_exclusions``. Version 1 left
    intent.within_domain's per-domain intent lists in the compared text, so its lists
    excluded 63-66% of CLINC keys for sharing a label list; the hook compares the version
    and the rule exactly, and refuses either one stale. Fails against version 1, which
    accepts the first case."""
    report, out, _att = scanned

    def edit(a: dict[str, Any]) -> None:
        a["export"]["template_strip"].update(applied=True, version=version, rule=rule)

    listed = _copy(out, tmp_path, edit_att=edit)
    with pytest.raises(ExclusionRefusal, match="did not apply v5's template strip version 2"):
        read_exclusions(listed, corpus=CORPUS)
    with pytest.raises(ExclusionRefusal, match=f"'version': {version}"):
        apply_exclusions(report, listed, corpus=CORPUS)


def test_a_list_that_is_not_qd_preps_or_names_no_train_row_is_refused(
    scanned, tmp_path: Path,
) -> None:
    report, out, _att = scanned
    keys = (out / EXCLUSIONS_NAME).read_text(encoding="utf-8").splitlines()
    for bad, match in (
        (list(reversed(keys)), "byte-sorted and unique"),
        ([keys[0], keys[0], *keys[1:]], "byte-sorted and unique"),
    ):
        listed = _copy(out, tmp_path / match.replace(" ", "-") / str(len(bad)), keys=bad)
        with pytest.raises(ExclusionRefusal, match=match):
            read_exclusions(listed, corpus=CORPUS)
    val_key = report.rows_by_split["val"][0].identity_key
    named = sorted({*keys, val_key}, key=str.encode)
    listed = _copy(out, tmp_path / "val-key", keys=named)
    with pytest.raises(ExclusionRefusal, match="name no train row"):
        apply_exclusions(report, listed, corpus=CORPUS)
    crlf = tmp_path / "crlf"
    shutil.copytree(out, crlf)
    (crlf / EXCLUSIONS_NAME).write_bytes(b"a\r\n")
    with pytest.raises(ExclusionRefusal, match="CR"):
        read_exclusions(crlf / EXCLUSIONS_NAME, corpus=CORPUS)
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(out / EXCLUSIONS_NAME, alone / EXCLUSIONS_NAME)
    with pytest.raises(ExclusionRefusal, match="no readable attestation"):
        read_exclusions(alone / EXCLUSIONS_NAME, corpus=CORPUS)
    with pytest.raises(ExclusionRefusal, match="unreadable"):
        read_exclusions(tmp_path / "missing.txt", corpus=CORPUS)


# --- the pipeline and the trainer, end to end ----------------------------------------------------


def _val_and_train(out: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Every val shard file, normalised as ``build_digest`` normalises it, and the val
    manifest less ``split_report.counts.train`` -- the one number in it about the train split,
    returned beside the train manifest and header."""
    val = {
        str(p.relative_to(out)): _normalised(p)
        for p in sorted((out / "shards" / "val").rglob("*")) if p.is_file()
    }
    manifest = json.loads(_normalised(out / "data/pool/val.json"))
    train_count = manifest["split_report"]["counts"].pop("train")
    val["data/pool/val.json"] = json.dumps(manifest, sort_keys=True).encode()
    train = {
        "count_in_val_manifest": train_count,
        "manifest": json.loads((out / "data/pool/train.json").read_text(encoding="utf-8")),
        "header": json.loads((out / "shards/train/header.json").read_text(encoding="utf-8")),
    }
    return val, train


@pytest.mark.slow
@pytest.mark.usefixtures("qd_prep")
def test_a_build_with_the_list_drops_those_train_rows_and_not_one_val_byte(
    qd_prep_bin: Path, tmp_path: Path,
) -> None:
    """The 60-pair build of the characterization above, scanned by the real export and
    binary, then built twice: without the list and with it. Rule 2's proof is the val
    side, byte for byte; the train side loses exactly the listed rows, its header names the
    list, and the trainer's rebuild with the same list holds exactly the rows the set does.
    The scan finds what it finds on this corpus; the list applied is two of its train rows'
    keys under the scan's own attestation (re-signed for the list), so the drop is never
    vacuous."""
    pipeline = _pipeline_or_skip()
    pytest.importorskip("torch")
    import real_ft_run as rft
    from repo_git import resolve_rev

    from qd_train.exclusions import containment_corpus

    config = DataConfig()
    rev = resolve_rev(REPO, PIN_REV)
    report = rft.ft_split_report(commitpackft=DOWNLOAD, max_pairs=60, rev=rev, config=config,
                                 repo_history=False)
    corpus = containment_corpus(pipeline.corpus_identity(
        rev=rev, max_pairs=60, commitpackft=DOWNLOAD, defect_class=None, defect_max_rows=None,
        repo_history=False,
    ))
    sets, strip = scan_sets(report, config=config)
    request = tmp_path / "scan.request.bin"
    with request.open("xb") as fh:
        write_request(fh, sets, scan_specs(sets), corpus=corpus,
                      export={"template_strip": strip}, checks=splitter_checks(report))
    run_containment(qd_prep_bin, request, tmp_path / "scan", threads=4, timeout_s=300.0)
    att = json.loads((tmp_path / "scan" / ATTESTATION_NAME).read_text(encoding="utf-8"))
    assert att["clean"] is True, att["not_clean_because"]
    train_keys = sorted({r.identity_key for r in report.rows_by_split["train"]}, key=str.encode)
    chosen = sorted({train_keys[0], train_keys[len(train_keys) // 2]}, key=str.encode)
    listed = _copy(tmp_path / "scan", tmp_path / "chosen", keys=chosen)

    pipeline.run(out=tmp_path / "plain", max_pairs=60, blank_line_runs=False, rev=PIN_REV,
                 commitpackft=DOWNLOAD, val_shards=True, repo_history=False)
    pipeline.run(out=tmp_path / "flag", max_pairs=60, blank_line_runs=False, rev=PIN_REV,
                 commitpackft=DOWNLOAD, val_shards=True, repo_history=False,
                 exclude_identity_keys=listed)
    plain_val, plain_train = _val_and_train(tmp_path / "plain")
    flag_val, flag_train = _val_and_train(tmp_path / "flag")
    assert plain_val and flag_val == plain_val
    digest = hashlib.sha256(listed.read_bytes()).hexdigest()
    assert flag_train["header"]["exclusions_sha256"] == digest
    assert "exclusions_sha256" not in plain_train["header"]
    plain_ids = [e["row_id"] for e in plain_train["manifest"]["entries"]]
    flag_ids = [e["row_id"] for e in flag_train["manifest"]["entries"]]
    dropped = {r.row_id for r in report.rows_by_split["train"] if r.identity_key in chosen}
    assert dropped and flag_ids == [i for i in plain_ids if i not in dropped]
    assert (plain_train["count_in_val_manifest"] - flag_train["count_in_val_manifest"]
            == len(dropped))
    rebuilt = rft.ft_splits(commitpackft=DOWNLOAD, max_pairs=60, rev=rev, config=config,
                            repo_history=False, exclude_identity_keys=listed)
    assert sorted(r.row_id for r in rebuilt["train"]) == sorted(flag_ids)
    assert {r.row_id for r in rebuilt["val"]} == {r.row_id for r in report.rows_by_split["val"]}


def test_the_pipeline_refuses_an_unclean_list_and_a_list_beside_replay_exclude(
    scanned, tmp_path: Path,
) -> None:
    """Before minutes of building: a list whose attestation is not CLEAN, and a list with
    ``--replay-exclude``, which the list supersedes."""
    _report, out, _att = scanned
    import real_tokenizer_pipeline as pipeline

    with pytest.raises(SystemExit, match="--replay-exclude together"):
        pipeline.run(out=tmp_path / "a", max_pairs=1, blank_line_runs=False, rev=PIN_REV,
                     repo_history=False, exclude_identity_keys=out / EXCLUSIONS_NAME,
                     replay_exclude=out / "hits.json")
    unclean = _copy(out, tmp_path / "unclean", edit_att=_set("clean", False))
    with pytest.raises(ExclusionRefusal, match="not CLEAN"):
        pipeline.run(out=tmp_path / "b", max_pairs=1, blank_line_runs=False, rev=PIN_REV,
                     commitpackft=DOWNLOAD, repo_history=False, exclude_identity_keys=unclean)


# --- the trainer's rebuild ---------------------------------------------------------------------


def test_the_trainer_refuses_a_rebuild_that_disagrees_with_the_header(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import real_ft_run as rft

    listed = tmp_path / "exclusions.txt"
    listed.write_bytes(b"k\n")
    digest = hashlib.sha256(b"k\n").hexdigest()
    plain, named = _header(), dataclasses.replace(_header(), exclusions_sha256=digest)
    rft.check_exclusion_source(plain, None)
    rft.check_exclusion_source(named, listed)
    with pytest.raises(SystemExit, match="built without an exclusion list"):
        rft.check_exclusion_source(plain, listed)
    with pytest.raises(SystemExit, match="pass the same file"):
        rft.check_exclusion_source(named, None)
    other = tmp_path / "other.txt"
    other.write_bytes(b"j\n")
    with pytest.raises(SystemExit, match="but the shard set was built with"):
        rft.check_exclusion_source(named, other)
    with pytest.raises(SystemExit, match="unreadable"):
        rft.check_exclusion_source(named, tmp_path / "missing.txt")
    assert rft._recipe_pieces(lower_layers_n=0, lower_lr_scale=1.0, beta2=rft.DEFAULT_BETA2,
                              permutation=None, replay=None, exclusions_sha256=digest) == {
        "exclusions_sha256": digest
    }
    assert "exclusions_sha256" in rft.RECIPE_PIECE_KEYS
