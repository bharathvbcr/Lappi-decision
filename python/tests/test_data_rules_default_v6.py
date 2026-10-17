"""v6 data rules are ``DataConfig()``'s defaults; v5 is a named legacy constructor
(human ruling 2026-10-17, ``AUDIT/harden-2026-10-17/human-ruling-2026-10-17.md``; mistakes
M6/M7/M8: rebuilders that called ``DataConfig()`` silently rebuilt under v5's rules, which let
MMLU and CLINC test items into training rows).

Every test fails on the pre-change code: ``DataConfig()`` is v5 there, and ``legacy_v5`` and
``with_rules`` do not exist.

Torch-free.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import real_tokenizer_pipeline as pipeline  # noqa: E402
from qd_data.config import (  # noqa: E402
    DEDUPE_KEEP_LEXICAL,
    DEDUPE_KEEP_SPLIT_PRIORITY,
    V6_LSH_MIN_AGREEMENT_PERMILLE,
    DataConfig,
)

#: What every v5-era train manifest on this host records as ``config_fingerprint`` (read from
#: ``~/qd-campaign/*/data/pool/train.json`` on 2026-10-08): the pin a legacy config keeps, so an
#: already-pinned v5 artifact stays verifiable by name instead of being refused or waved through.
V5_FINGERPRINT: dict[str, object] = {
    "admitted_by_human": {}, "dedupe_threshold": 0.8,
    "held_out_families": ["code.language_id", "qa.answerability"], "num_perm": 128,
    "seed": 20260919, "shingle_size": 5, "train_fraction": 0.9, "val_fraction": 0.05,
}


def test_the_default_config_is_v6() -> None:
    config = DataConfig()
    assert config.dedupe_keep_rule == DEDUPE_KEEP_SPLIT_PRIORITY
    assert config.lsh_min_agreement_permille == V6_LSH_MIN_AGREEMENT_PERMILLE
    assert config.benchmark_eval_splits_are_targets is True


def test_the_default_fingerprint_names_every_v6_rule() -> None:
    fingerprint = DataConfig().fingerprint()
    assert fingerprint["dedupe_keep_rule"] == DEDUPE_KEEP_SPLIT_PRIORITY
    assert fingerprint["lsh_min_agreement_permille"] == V6_LSH_MIN_AGREEMENT_PERMILLE
    assert fingerprint["benchmark_eval_splits_are_targets"] is True


def test_legacy_v5_is_v5_and_fingerprints_as_every_pinned_v5_manifest_recorded() -> None:
    legacy = DataConfig.legacy_v5()
    assert legacy.dedupe_keep_rule == DEDUPE_KEEP_LEXICAL
    assert legacy.lsh_min_agreement_permille == 0
    assert legacy.benchmark_eval_splits_are_targets is False
    assert json.loads(json.dumps(legacy.fingerprint())) == V5_FINGERPRINT


def test_with_rules_round_trips_and_is_the_one_owner() -> None:
    assert DataConfig().with_rules(v6=False) == DataConfig.legacy_v5()
    assert DataConfig.legacy_v5().with_rules(v6=True) == DataConfig()


def test_the_pipeline_seam_defaults_to_v6_and_delegates_to_the_config() -> None:
    assert pipeline.data_rules(DataConfig.legacy_v5()) == DataConfig()
    assert pipeline.data_rules(DataConfig(), v6=True) == DataConfig()
    assert pipeline.data_rules(DataConfig(), v6=False) == DataConfig.legacy_v5()


def test_a_default_rebuild_does_not_read_as_a_v5_rebuild() -> None:
    """``real_ft_run.corpus_facts`` refuses a manifest whose ``config_fingerprint`` differs from
    the rebuild's, so a v5-pinned set is refused unless the caller names ``legacy_v5``."""
    assert json.loads(json.dumps(DataConfig().fingerprint())) != V5_FINGERPRINT


def test_the_pipeline_builds_v6_unless_legacy_v5_is_named() -> None:
    assert inspect.signature(pipeline.run).parameters["v6_data_rules"].default is True


def test_the_pipeline_cli_makes_v5_an_explicit_named_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        pipeline.main(["--help"])
    help_text = capsys.readouterr().out
    assert "--legacy-v5-data-rules" in help_text
    assert "--v6-data-rules" not in help_text
