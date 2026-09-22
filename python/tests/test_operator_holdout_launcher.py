"""The operator-holdout launcher's contract, read as text.

Running it would spend GPU hours, so these read the script. That is the right level for
what can actually go wrong here: the defects this file pins are a stale claim in the header
that a reader would act on, and an override that could silently run zero arms while still
printing the marker a watcher treats as success.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "launch_operator_holdout_model.sh"
DEFAULT_OPERATORS = ("stub.panic", "logic.change_constant", "cosmetic.rename_local")


def test_the_operator_list_is_overridable_without_editing_the_script() -> None:
    """The three defaults are the dominant operator in each class, which is the weakest
    version of this test: holding out the biggest operator moves the class prior hardest,
    and that is precisely the confound the sibling metric then has to rule out. Measuring a
    minor operator must not require a second copy of the script that can drift from it.
    """
    src = SCRIPT.read_text()
    assert re.search(
        r'^OPERATORS="\$\{QD_HOLDOUT_OPERATORS:-[^}]*\}"', src, re.M
    ), "OPERATORS must take an override with the three defaults preserved"


def test_the_default_operator_list_is_unchanged() -> None:
    """72 rows were written against these three. A default that quietly changed would make
    the next run incomparable with them while looking like a rerun."""
    src = SCRIPT.read_text()
    line = re.search(r'^OPERATORS="\$\{QD_HOLDOUT_OPERATORS:-([^}]*)\}"', src, re.M)
    assert line is not None
    assert tuple(line.group(1).split()) == DEFAULT_OPERATORS


def test_an_empty_override_refuses_instead_of_running_nothing() -> None:
    """`for OP in $OPERATORS` over an empty string runs zero arms, falls through to the
    DONE marker, and exits 0. A watcher grepping for that marker cannot tell that from a
    finished run, so the ledger would simply be missing rows nobody was told about.
    """
    src = SCRIPT.read_text()
    guard = src.find('if [ -z "${OPERATORS// /}" ]')
    assert guard != -1, "an empty or whitespace-only operator list must be refused"
    assert "exit 2" in src[guard : guard + 500]
    # The guard has to sit before the loop, or it guards nothing.
    assert guard < src.index("for OP in $OPERATORS")


def test_the_header_no_longer_claims_the_control_cannot_be_fitted() -> None:
    """It said "fit_linear_control.py has no holdout flags". That stopped being true: it
    takes --hold-out-operator and --drop-random-train. A reader who believed the old
    sentence would conclude the missing margins were impossible rather than unpurchased,
    which is the opposite of what the not_run rows mean."""
    src = SCRIPT.read_text()
    assert "has no holdout flags" not in src
    assert "--hold-out-operator" in src, "the header should say what the flags are"


def test_the_header_carries_the_measured_cost_of_the_missing_margins() -> None:
    """The reason those 48 rows have no margin is now cost, not capability, and a cost that
    is not written down gets rediscovered by spending it."""
    src = SCRIPT.read_text()
    assert "101.4 minute" in src
    assert "does not" in src and "backfill" in src, (
        "it must say that warming the cache leaves already-written rows unchanged"
    )
