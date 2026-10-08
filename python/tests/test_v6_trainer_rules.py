"""``tools/real_ft_run.py`` under Fable's v6 ruling (``AUDIT/v6-rulings-2026-10-08/
fable-v6-data-design-ruling.md``): R2's ``--v6-data-rules`` rebuild and R7's shared span head.

* R2. ``--v6-data-rules`` rebuilds the shard set's rows under the pipeline's own
  ``data_rules`` (both v6 rules), names its corpus as the pipeline does (``name_corpus``), and
  ``corpus_facts`` refuses a set whose manifest ``config_fingerprint`` is not the rebuild's: a
  v6 set without the flag, a v5 set with it, and a manifest with no fingerprint.
* R7. A span-head manifest written by ``qd_train_oracle_span_head_init.py --shared`` carries
  ``construction.shared_across_seeds: true``; only such a head may start seeds other than its
  own construction seed, and the recipe records ``span_head_init.shared = true``. Without
  ``--shared`` the generator writes the manifest it always wrote.

Every test fails on the pre-change code: no ``--v6-data-rules``, no ``config`` on
``corpus_facts``, ``ft_splits`` names a v5 corpus under any config, ``SpanHeadInit`` has no
``shared`` and refuses every other seed, and the generator has no ``--shared``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

torch = pytest.importorskip("torch", reason="real_ft_run imports torch at module scope")
pytest.importorskip("safetensors", reason="the span-head file is a safetensors")

import real_ft_run as rft  # noqa: E402
import real_tokenizer_pipeline as pipeline  # noqa: E402
from test_contrast_rows import _composite, _crafted_exclusions, _csqa  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.general import CSQA_FAMILY, MMLU_FAMILY, rewrite_csqa  # noqa: E402
from qd_data.split import split  # noqa: E402
from qd_train import exclusions  # noqa: E402
from qd_train.containment_strip import STRIP_RULE, STRIP_VERSION  # noqa: E402
from qd_train.exclusions import ATTESTATION_NAME, ExclusionRefusal  # noqa: E402

V5 = DataConfig()
V6 = pipeline.data_rules(V5, v6=True)
FIXTURES = REPO / "crates" / "qd-train" / "tests" / "fixtures"
H64_FILE = FIXTURES / "span-head-init-seed0-h64.safetensors"
H64_MANIFEST = FIXTURES / "span-head-init-seed0-h64.manifest.json"


# --- R2: the rebuild refuses a set built under the other rule set ----------------------------


def _manifest(out: Path, fingerprint: object) -> None:
    path = out / rft.TRAIN_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, object] = {
        "data_snapshot_hash": "d" * 64, "status": {"state": "ran", "passed": True},
        "mixture": {"n_input": {"bigcode/commitpackft": 5}},
    }
    if fingerprint is not None:
        body["config_fingerprint"] = fingerprint
    path.write_text(json.dumps(body), encoding="utf-8")


def _facts(out: Path, config: DataConfig) -> rft.CorpusFacts:
    return rft.corpus_facts(
        out, data_snapshot_hash="d" * 64, repo_history=True, commitpackft=None, config=config
    )


@pytest.mark.parametrize(("built", "rebuilt"), [(V5, V5), (V6, V6)])
def test_a_set_rebuilds_under_the_rules_it_was_built_under(
    tmp_path: Path, built: DataConfig, rebuilt: DataConfig
) -> None:
    _manifest(tmp_path, built.fingerprint())
    assert _facts(tmp_path, rebuilt).history_rows == {"bigcode/commitpackft": 5}


@pytest.mark.parametrize(("built", "rebuilt"), [(V6, V5), (V5, V6)])
def test_a_set_built_under_the_other_rules_is_refused(
    tmp_path: Path, built: DataConfig, rebuilt: DataConfig
) -> None:
    _manifest(tmp_path, built.fingerprint())
    with pytest.raises(SystemExit, match="--v6-data-rules must be given here exactly when"):
        _facts(tmp_path, rebuilt)


def test_a_manifest_with_no_fingerprint_is_refused(tmp_path: Path) -> None:
    _manifest(tmp_path, None)
    with pytest.raises(SystemExit, match="carries no config_fingerprint"):
        _facts(tmp_path, V5)


class _Reached(Exception):
    pass


@pytest.mark.parametrize(("extra", "want"), [([], V5), (["--v6-data-rules"], V6)])
def test_the_flag_reaches_the_corpus_check_and_the_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str], want: DataConfig
) -> None:
    seen: dict[str, Any] = {}

    def facts(*_a: Any, config: DataConfig, **_k: Any) -> None:
        seen["config"] = config
        raise _Reached

    header = SimpleNamespace(data_snapshot_hash="d" * 64)
    monkeypatch.setattr(rft, "resolve_rev", lambda _repo, rev: rev)
    monkeypatch.setattr(rft, "ShardReader", lambda *a, **k: SimpleNamespace(header=header))
    monkeypatch.setattr(rft, "check_defect_source", lambda *a, **k: None)
    monkeypatch.setattr(rft, "check_exclusion_source", lambda *a, **k: None)
    monkeypatch.setattr(rft, "corpus_facts", facts)
    with pytest.raises(_Reached):
        rft.main(["--out", str(tmp_path), "--rev", "a" * 40, "--devices", "cpu",
                  "--seeds", "0", *extra])
    assert seen["config"] == want


# --- R2: ft_splits names its corpus as the pipeline does --------------------------------------


def _listed(tmp: Path, key: str, corpus: dict[str, object]) -> Path:
    tmp.mkdir()
    listed = _crafted_exclusions(tmp, [key], corpus)
    att_path = listed.parent / ATTESTATION_NAME
    att = json.loads(att_path.read_text(encoding="utf-8"))
    att["export"] = {
        "template_strip": {"applied": True, "rule": STRIP_RULE, "version": STRIP_VERSION}
    }
    att_path.write_text(json.dumps(att), encoding="utf-8")
    return listed


@pytest.mark.usefixtures("qd_prep")
def test_ft_splits_refuses_a_v5_scans_list_for_a_v6_rebuild_and_derives_csqa_contrast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [
        rewrite_csqa(_csqa(i, "train"), family_id=CSQA_FAMILY, index=i, config=V6)
        for i in range(40)
    ]
    report = split(dedupe(rows, config=V6), config=V6)
    base = {"test": "crafted"}
    monkeypatch.setattr(rft, "ft_split_report", lambda **kw: report)
    monkeypatch.setattr(rft, "replay_corpus_identity", lambda **kw: kw)
    monkeypatch.setattr(exclusions, "containment_corpus", lambda identity: base)
    noul = _composite(tmp_path, {"per_family": {MMLU_FAMILY: 3, CSQA_FAMILY: 2}, "seed": 5})

    excluded = report.rows_by_split["train"][0]
    v5_list = _listed(tmp_path / "v5", excluded.identity_key, base)
    with pytest.raises(ExclusionRefusal):
        rft.ft_splits(commitpackft=None, max_pairs=3, rev="r", config=V6,
                      defect_noul=noul, exclude_identity_keys=v5_list)

    named = pipeline.name_corpus({}, config=V6)
    assert named != base and {k: named[k] for k in base} == base
    v6_list = _listed(tmp_path / "v6", excluded.identity_key, named)
    rebuilt = rft.ft_splits(commitpackft=None, max_pairs=3, rev="r", config=V6,
                            defect_noul=noul, exclude_identity_keys=v6_list)
    train = rebuilt["train"]
    assert excluded.identity_key not in {r.identity_key for r in train}
    contrast = [r for r in train if r.identity_key.startswith("contrast:")]
    # v5's 3 MMLU + 2 CSQA request, derived as 5 CSQA rows under the v6 config.
    assert len(contrast) == 5
    twins = {r.identity_key: r.family_id for r in report.rows_by_split["train"]}
    assert {twins[r.identity_key.removeprefix("contrast:")] for r in contrast} == {CSQA_FAMILY}


# --- R7: one shared span head for every seed -------------------------------------------------


def _head(tmp_path: Path, **construction: object) -> tuple[Path, Path]:
    path = tmp_path / "head.safetensors"
    path.write_bytes(H64_FILE.read_bytes())
    manifest = json.loads(H64_MANIFEST.read_text())
    manifest["construction"].update(construction)
    mpath = tmp_path / "head.manifest.json"
    mpath.write_text(json.dumps(manifest))
    return path, mpath


def _rungd_args(path: Path, mpath: Path, seeds: list[int]) -> argparse.Namespace:
    return argparse.Namespace(
        score_checkpoint=None, score_plan=None, max_steps=None, probe_shapes=None,
        train_dtype="bf16", span_head_init=path, span_head_init_digest=False,
        span_head_init_manifest=mpath, shuffled_label=None, real_backbone=Path("snapshot"),
        seeds=seeds, epoch=True,
    )


def test_a_shared_head_starts_three_seeds_and_the_recipe_says_shared(tmp_path: Path) -> None:
    path, mpath = _head(tmp_path, shared_across_seeds=True)
    init = rft._check_rungd_flags(_rungd_args(path, mpath, [0, 1, 2]))
    assert init is not None and init.shared and init.seed == 0
    manifest = json.loads(mpath.read_text())
    assert init.recipe() == {
        "sha256": manifest["file"]["sha256"],
        "content_digest": manifest["content_digest"]["value"],
        "shared": True,
    }
    pieces = rft._recipe_pieces(
        lower_layers_n=0, lower_lr_scale=1.0, beta2=rft.DEFAULT_BETA2, permutation=None,
        replay=None, span_head_init=init.recipe(),
    )
    assert pieces["span_head_init"]["shared"] is True


def test_a_seed_specific_head_is_still_refused_for_another_seed(tmp_path: Path) -> None:
    path, mpath = _head(tmp_path)
    with pytest.raises(SystemExit, match=r"each seed starts from its own head"):
        rft._check_rungd_flags(_rungd_args(path, mpath, [0, 1, 2]))
    init = rft._check_rungd_flags(_rungd_args(path, mpath, [0]))
    assert init is not None and not init.shared
    assert "shared" not in init.recipe()


@pytest.mark.parametrize("bad", [False, "true", 1, None])
def test_shared_across_seeds_is_true_or_absent(tmp_path: Path, bad: object) -> None:
    path, mpath = _head(tmp_path, shared_across_seeds=bad)
    with pytest.raises(SystemExit, match="shared_across_seeds"):
        rft.read_span_head_init(path, mpath)


def _generate(tmp_path: Path, *extra: str) -> tuple[bytes, dict[str, Any]]:
    import qd_train_oracle_span_head_init as gen

    out, manifest = tmp_path / "head.safetensors", tmp_path / "head.manifest.json"
    assert gen.main(["--hidden", "64", "--seed", "0", "--out", str(out),
                     "--manifest", str(manifest), *extra]) == 0
    return out.read_bytes(), json.loads(manifest.read_text())


def test_without_shared_the_generator_writes_the_tracked_fixtures_head_and_construction(
    tmp_path: Path,
) -> None:
    payload, manifest = _generate(tmp_path)
    tracked = json.loads(H64_MANIFEST.read_text())
    assert payload == H64_FILE.read_bytes()
    assert manifest["construction"] == tracked["construction"]
    assert "shared_across_seeds" not in manifest["construction"]
    for key in ("what", "content_digest", "tensors"):
        assert manifest[key] == tracked[key], key


def test_shared_adds_only_the_key_and_the_trainer_reads_it(tmp_path: Path) -> None:
    payload, manifest = _generate(tmp_path, "--shared")
    tracked = json.loads(H64_MANIFEST.read_text())
    assert payload == H64_FILE.read_bytes()
    assert manifest["construction"] == {**tracked["construction"], "shared_across_seeds": True}
    init = rft.read_span_head_init(tmp_path / "head.safetensors", tmp_path / "head.manifest.json")
    assert init.shared
