"""Every ``tools/launch_*.sh`` refuses to start a paid box's work outside the campaign driver.

GAP-LAUNCH-SCRIPTS-RUN-A-PAID-BOX-OUTSIDE-THE-DRIVER-2026-10-17: nine scripts ran
``rung0_real_run.py`` on a rented GH200 directly, with none of the driver's go/no-go, caps or
budget clock. They now source ``tools/require_driver.sh``, which exits 78 unless
``CAMPAIGN_DRIVER_RUN`` (``campaign_driver.DRIVER_ENV``) is set.

Nothing here runs a launcher past its guard: the "gets past" half executes only the script's
text up to and including the guard line, then a marker, so nothing is billed and nothing under
``/home/ubuntu`` is touched.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
GUARD_LINE = '. "$(dirname "${BASH_SOURCE[0]}")/require_driver.sh"'
LAUNCHERS = sorted(TOOLS.glob("launch_*.sh"))

sys.path.insert(0, str(TOOLS))
import campaign_driver  # noqa: E402


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != campaign_driver.DRIVER_ENV}
    env.update(extra)
    return env


def test_the_nine_launchers_are_the_ones_found() -> None:
    assert [p.name for p in LAUNCHERS] == [
        "launch_context_source_arms.sh", "launch_diff_capacity.sh", "launch_fourway_arms.sh",
        "launch_grid_fill.sh", "launch_lr_sweep.sh", "launch_operator_holdout_model.sh",
        "launch_span_in_diff.sh", "launch_tuned_capacity.sh", "launch_v2_arms.sh",
    ]


def test_the_variable_the_guard_reads_is_the_one_the_driver_exports() -> None:
    assert campaign_driver.DRIVER_ENV == "CAMPAIGN_DRIVER_RUN"
    assert "CAMPAIGN_DRIVER_RUN" in (TOOLS / "require_driver.sh").read_text()


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_a_launcher_refuses_without_the_driver_variable(script: Path) -> None:
    done = subprocess.run(
        ["bash", str(script)], env=_env(), capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 78, (done.returncode, done.stdout, done.stderr)
    assert "tools/campaign_driver.py" in done.stderr
    assert done.stdout == ""


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_a_launcher_gets_past_the_guard_with_the_variable(script: Path, tmp_path: Path) -> None:
    lines = script.read_text().splitlines()
    assert GUARD_LINE in lines, f"{script.name} does not source require_driver.sh"
    head = lines[: lines.index(GUARD_LINE) + 1]
    # Nothing but comments, blanks and the shell-options line may run before the guard.
    for line in head[:-1]:
        assert line.startswith("#") or not line.strip() or line.startswith("set "), line
    probe = tmp_path / "probe.sh"
    (tmp_path / "require_driver.sh").write_text((TOOLS / "require_driver.sh").read_text())
    probe.write_text("\n".join([*head, "echo PAST_GUARD", ""]))
    done = subprocess.run(
        ["bash", str(probe)], env=_env(CAMPAIGN_DRIVER_RUN="a-campaign"),
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert done.stdout.strip() == "PAST_GUARD"
    refused = subprocess.run(
        ["bash", str(probe)], env=_env(), capture_output=True, text=True, timeout=30,
    )
    assert refused.returncode == 78
    assert "PAST_GUARD" not in refused.stdout


# --- the paid-box scripts that are not tools/launch_*.sh ------------------------------------
#
# Fable Q7.5 (2026-10-18): post_f_common.sh (sourced by every box_q_*.sh), v5_common.sh
# (sourced by the v5 waiters) and tools/perf_tierb_outcome.sh (real_ft_run on a GH200 at
# $2.29/h) started paid work with no driver. Each now sources the guard.

QUEUE = REPO / "campaign" / "post-f-queue"
BOX_LIBRARIES = [QUEUE / "post_f_common.sh", QUEUE / "v5_common.sh"]
TIERB = TOOLS / "perf_tierb_outcome.sh"


@pytest.mark.parametrize("script", [*BOX_LIBRARIES, TIERB], ids=lambda p: p.name)
def test_every_paid_box_script_sources_the_guard(script: Path) -> None:
    code = [
        line for line in script.read_text().splitlines() if not line.lstrip().startswith("#")
    ]
    # `. "<dir>/require_driver.sh"`, at the start of a line or as one arm of a case.
    assert any("require_driver.sh" in line and re.search(r'(^|\s)\. "', line) for line in code), (
        f"{script.name} does not source require_driver.sh"
    )


@pytest.mark.parametrize("script", BOX_LIBRARIES, ids=lambda p: p.name)
def test_a_box_library_refuses_to_be_sourced_outside_the_driver(script: Path) -> None:
    done = subprocess.run(
        ["bash", "-c", f'source "{script}"; echo PAST_GUARD'], env=_env(),
        capture_output=True, text=True, timeout=30, cwd=REPO, check=False,
    )
    assert done.returncode == 78, (done.returncode, done.stdout, done.stderr)
    assert "tools/campaign_driver.py" in done.stderr
    assert "PAST_GUARD" not in done.stdout


@pytest.mark.parametrize("script", BOX_LIBRARIES, ids=lambda p: p.name)
def test_a_box_library_is_sourced_under_the_driver_and_finds_the_guard_beside_itself(
    script: Path, tmp_path: Path
) -> None:
    """Under the driver the guard passes from the checkout's tools/, and from a flat deploy
    (the file copied next to require_driver.sh, as /home/ubuntu/post-f/ holds it)."""
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / script.name).write_text(script.read_text())
    (flat / "require_driver.sh").write_text((TOOLS / "require_driver.sh").read_text())
    for where in (script, flat / script.name):
        done = subprocess.run(
            ["bash", "-c", f'source "{where}"; echo PAST_GUARD'],
            env=_env(CAMPAIGN_DRIVER_RUN="a-campaign"),
            capture_output=True, text=True, timeout=30, cwd=REPO, check=False,
        )
        assert done.returncode == 0, (where, done.stdout, done.stderr)
        assert done.stdout.strip().endswith("PAST_GUARD"), (where, done.stdout)


def test_a_box_library_with_no_guard_to_find_refuses_rather_than_runs(tmp_path: Path) -> None:
    lone = tmp_path / "post_f_common.sh"
    lone.write_text(BOX_LIBRARIES[0].read_text())
    done = subprocess.run(
        ["bash", "-c", f'source "{lone}"; echo PAST_GUARD'],
        env=_env(CAMPAIGN_DRIVER_RUN="a-campaign"),
        capture_output=True, text=True, timeout=30, cwd=REPO, check=False,
    )
    assert done.returncode == 78 and "PAST_GUARD" not in done.stdout, (
        done.returncode, done.stdout, done.stderr,
    )


def test_perf_tierb_run_refuses_outside_the_driver_but_print_does_not(tmp_path: Path) -> None:
    perf = {"TIERB_PERF": str(tmp_path / "perf"), "TIERB_SRC": str(tmp_path / "src")}
    refused = subprocess.run(
        ["bash", str(TIERB), "fused", "--run"], env=_env(**perf),
        capture_output=True, text=True, timeout=30, cwd=REPO, check=False,
    )
    assert refused.returncode == 78, (refused.returncode, refused.stdout, refused.stderr)
    assert "tools/campaign_driver.py" in refused.stderr and refused.stdout == ""
    printed = subprocess.run(
        ["bash", str(TIERB), "fused", "--print"], env=_env(**perf),
        capture_output=True, text=True, timeout=30, cwd=REPO, check=False,
    )
    assert printed.returncode == 0 and "candidate fused" in printed.stdout, printed
    # Under the driver --run gets past the guard; the code dir it then needs does not exist
    # in this scratch tree, so it stops there (3) without having started anything.
    past = subprocess.run(
        ["bash", str(TIERB), "fused", "--run"],
        env=_env(CAMPAIGN_DRIVER_RUN="a-campaign", **perf),
        capture_output=True, text=True, timeout=30, cwd=REPO, check=False,
    )
    assert past.returncode == 3, (past.returncode, past.stdout, past.stderr)
