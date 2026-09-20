"""Is `stack/`'s pinned environment installable on the machine we are about to rent?

``stack/train.lock`` and ``stack/teacher.lock`` were resolved with
``--python-platform x86_64-manylinux_2_28`` and their header says so. ``stack/README.md``
recorded the matching decision: *"aarch64 image **not needed** — GH200 is the only aarch64
target and is out of capacity"*. On 2026-09-20 a GH200 **was** rented, so that decision is
stale and the question it settled is open again.

**A resolution is not an installation.** ``uv pip compile`` resolves version constraints;
it does not fetch wheels. The architecture question lives at fetch time, in whether each
pinned version publishes a wheel for the target. **Every pin is checked**, because two of
seventy-one reported as the answer is a capped sample presented as complete coverage. Two
need handling of their own, because their wheels do not come from PyPI:

* ``torch==2.10.0+cu128`` comes from ``download.pytorch.org``, not PyPI.
* ``causal-conv1d==1.7.0`` publishes **only an sdist** on PyPI. Its wheels live on a
  GitHub release, and ``stack/train.in`` pins torch to 2.10 rather than the plan's 2.11
  precisely to hit one of them — it calls the from-source fallback *"the single most
  fragile step in the image"*, memory-hungry enough to invite the OOM killer and unable to
  build under emulation at all. If no wheel exists for the target architecture, that
  reasoning silently stops applying and the fragile step comes back.

**Why this tool resolves twice.** Re-resolving ``teacher.in`` for aarch64 on 2026-09-20
moved ``tokenspeed-triton`` from ``3.8.10.post20260906`` to ``3.8.10.post20260920``, and
the first version of this tool called that a portability failure. It is not: re-resolving
for **x86_64**, the platform the lock was committed at, moves it identically. The cause is
that upstream published a new post-release today, not that the package differs by
architecture. Reporting the two under one name is the *"one name, two quantities"* defect
this repository keeps finding, so the two are now measured separately:

* **time drift** — committed lock vs a fresh resolve *at the committed platform*. Upstream
  moved under us. Real, worth seeing, and says nothing about architecture.
* **architecture drift** — that same fresh resolve vs a fresh resolve *at the target*. Both
  run now, so time is held fixed and only the platform varies. **This alone decides
  portability.**

A tool that resolved once could not tell these apart, and would keep failing a portable
lock every time any pinned package cut a release.

This needs the network and says so when it does not have it, rather than reporting a pass
it did not measure. Run it before building an image for an architecture nobody has built
for yet:

    .venv/bin/python tools/verify_lock_portability.py --platform aarch64-manylinux_2_28
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = Path("/Users/bharath/.local/bin/uv")

#: The committed locks and the platform each records in its own header.
LOCKS = ("train", "teacher")
COMMITTED_PLATFORM = "x86_64-manylinux_2_28"
PYTHON_VERSION = "3.12"

PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", re.M)

TORCH_INDEX = "https://download.pytorch.org/whl/cu128/torch/"
CCONV_RELEASE = "https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/v1.7.0"

#: `aarch64-manylinux_2_28` -> the substring a wheel filename carries for that machine.
MACHINE_OF = {"aarch64": "aarch64", "x86_64": "x86_64"}


@dataclass(frozen=True, slots=True)
class Drift:
    """One package that differs between two pin sets, and the two versions."""

    package: str
    before: str | None
    after: str | None

    def render(self) -> str:
        if self.before is None:
            return f"{self.package}: ABSENT -> {self.after}"
        if self.after is None:
            return f"{self.package}: {self.before} -> ABSENT"
        return f"{self.package}: {self.before} -> {self.after}"


def diff(before: dict[str, str], after: dict[str, str]) -> list[Drift]:
    """Every package whose presence or version differs, ordered by name."""
    out = [
        Drift(k, before.get(k), after.get(k))
        for k in sorted(set(before) | set(after))
        if before.get(k) != after.get(k)
    ]
    return out


def classify(
    committed: dict[str, str], control: dict[str, str], target: dict[str, str]
) -> tuple[list[Drift], list[Drift]]:
    """Split the movement into (time drift, architecture drift).

    ``control`` and ``target`` are both resolved *now*; ``control`` is resolved at the
    platform ``committed`` was locked at. So ``committed -> control`` varies only time and
    ``control -> target`` varies only platform. Returning them separately is the whole
    point: only the second answers "will this install on the other machine".
    """
    return diff(committed, control), diff(control, target)


def pins(text: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in PIN.finditer(text)}


def resolve(stem: str, platform: str) -> dict[str, str]:
    """Resolve `stack/<stem>.in` for `platform`, in memory. Never writes a lockfile."""
    if not UV.exists():
        raise SystemExit(f"uv not found at {UV}; cannot resolve, and will not guess")
    # The override file, when one exists, is part of how this lock is generated: without it
    # the fresh resolve differs from the committed lock by every overridden pin and the tool
    # reports architecture drift that is really a missing flag.
    override = REPO / "stack" / f"{stem}.override.in"
    out = subprocess.run(
        [
            str(UV), "pip", "compile", str(REPO / "stack" / f"{stem}.in"),
            *(("--override", str(override)) if override.exists() else ()),
            "--python-platform", platform,
            "--python-version", PYTHON_VERSION,
            "--emit-index-url",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if out.returncode != 0:
        raise SystemExit(
            f"resolving {stem}.in for {platform} FAILED (exit {out.returncode}). That is the "
            f"answer, not an error to route around:\n{out.stderr[-2000:]}"
        )
    return pins(out.stdout)


def fetch(url: str, *, accept: str = "*/*") -> str | None:
    """The body, or ``None`` when the host did not answer. An HTTP error is an answer."""
    request = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "qwen-decision/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        # Caught before URLError, its base class: the host answered, so this is not a
        # reachability failure and must not be reported as one.
        print(f"    {url} answered HTTP {exc.code}")
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    no response from {url}: {type(exc).__name__}: {exc}")
        return None


def wheels_for(machine: str, version: str) -> tuple[int | None, int | None]:
    """(torch wheels, causal-conv1d wheels) matching `machine`, or None where unchecked."""
    torch_n: int | None = None
    body = fetch(TORCH_INDEX)
    if body is not None:
        names = set(re.findall(r"torch-[^\"']+\.whl", body))
        torch_n = sum(
            1 for n in names if version in n and "cp312" in n and machine in n
        )

    cconv_n: int | None = None
    body = fetch(CCONV_RELEASE, accept="application/vnd.github+json")
    if body is not None:
        try:
            assets = [a["name"] for a in json.loads(body).get("assets", [])]
        except (json.JSONDecodeError, TypeError, KeyError):
            assets = []
        cconv_n = sum(
            1 for n in assets if "torch2.10" in n and "cp312" in n and machine in n
        )
    return torch_n, cconv_n


#: PyPI's per-release metadata. `{name}/{version}` pins the exact release.
PYPI_RELEASE = "https://pypi.org/pypi/{name}/{version}/json"

#: Packages whose wheels do not come from PyPI, checked individually by `wheels_for`.
OFF_PYPI = frozenset({"torch", "causal-conv1d"})

#: How many PyPI lookups run at once. Bounded because every fan-out is bounded.
MAX_CONCURRENT_LOOKUPS = 8


@dataclass(frozen=True, slots=True)
class Coverage:
    """What a single pin's published files say about installing it on `machine`."""

    package: str
    version: str
    verdict: str  # "pure" | "match" | "no-wheel-for-machine" | "sdist-only" | "unanswered"
    detail: str

    @property
    def blocks(self) -> bool:
        """Does this stop an install, or force the source build the lock exists to avoid?"""
        return self.verdict in {"no-wheel-for-machine", "sdist-only"}


def classify_release(package: str, version: str, machine: str, body: str | None) -> Coverage:
    """Read one PyPI release's file list. Pure of network, so it is testable."""
    if body is None:
        return Coverage(package, version, "unanswered", "PyPI did not answer")
    try:
        urls = json.loads(body).get("urls", [])
        names = [u["filename"] for u in urls]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as exc:
        return Coverage(package, version, "unanswered", f"unreadable metadata: {exc}")

    wheels = [n for n in names if n.endswith(".whl")]
    if not wheels:
        kind = "sdist only" if names else "no files published"
        return Coverage(package, version, "sdist-only", f"{kind} -- a source build")
    # `-any.whl` is the platform tag for a pure-Python wheel: it installs anywhere, so
    # the architecture question does not arise for it.
    if any(n.endswith("-any.whl") for n in wheels):
        return Coverage(package, version, "pure", "pure-Python wheel (py3-none-any)")
    matching = [n for n in wheels if machine in n]
    if matching:
        return Coverage(package, version, "match", f"{len(matching)} of {len(wheels)} wheels")
    return Coverage(
        package,
        version,
        "no-wheel-for-machine",
        f"{len(wheels)} wheel(s) published, none for {machine}",
    )


def wheel_coverage(pins: dict[str, str], machine: str) -> list[Coverage]:
    """Check every PyPI pin. The two off-PyPI packages are checked by `wheels_for`."""
    todo = [(n, v) for n, v in sorted(pins.items()) if n not in OFF_PYPI]

    def one(item: tuple[str, str]) -> Coverage:
        name, version = item
        body = fetch(
            PYPI_RELEASE.format(name=name, version=version), accept="application/json"
        )
        return classify_release(name, version, machine, body)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_LOOKUPS) as pool:
        return list(pool.map(one, todo))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check a lock is installable on another arch.")
    ap.add_argument(
        "--platform",
        default="aarch64-manylinux_2_28",
        help="target uv --python-platform (default: the GH200's Grace CPU)",
    )
    args = ap.parse_args()
    target = args.platform
    machine = next((m for m in MACHINE_OF if m in target), None)
    if machine is None:
        raise SystemExit(
            f"cannot tell which machine {target!r} means; known: {sorted(MACHINE_OF)}. "
            "Refusing rather than guessing a wheel suffix."
        )

    print(f"committed platform : {COMMITTED_PLATFORM}")
    print(f"target platform    : {target}")
    if target == COMMITTED_PLATFORM:
        print("(target IS the committed platform: this run can only show time drift)")
    print()

    portable = True
    any_time_drift = False
    for stem in LOCKS:
        committed = pins((REPO / "stack" / f"{stem}.lock").read_text(encoding="utf-8"))
        control = resolve(stem, COMMITTED_PLATFORM)
        fresh = resolve(stem, target)
        time_drift, arch_drift = classify(committed, control, fresh)

        print(f"{stem}.lock: {len(committed)} committed, {len(fresh)} resolved for {machine}")
        if arch_drift:
            portable = False
            print(f"    ARCHITECTURE DRIFT ({len(arch_drift)}) -- both resolved now, only "
                  "the platform differs, so this IS a portability failure:")
            for d in arch_drift:
                print(f"        {d.render()}")
        else:
            print("    no architecture drift — the VERSIONS are portable")
        if time_drift:
            any_time_drift = True
            print(f"    time drift ({len(time_drift)}) -- upstream moved since the lock was "
                  "committed; visible at the COMMITTED platform too, so not a portability "
                  "question:")
            for d in time_drift:
                print(f"        {d.render()}")

    train = pins((REPO / "stack" / "train.lock").read_text(encoding="utf-8"))
    torch_version = train.get("torch", "")
    print(f"\nwheels actually published for {machine} (a resolution does not prove these):")
    torch_n, cconv_n = wheels_for(machine, torch_version)
    for label, n, why in (
        (f"torch {torch_version} cp312", torch_n, "from download.pytorch.org/whl/cu128"),
        ("causal-conv1d 1.7.0 cp312 torch2.10", cconv_n, "from the GitHub release"),
    ):
        if n is None:
            print(f"    {label:38} NOT CHECKED — the index did not answer ({why})")
            portable = False
        elif n == 0:
            print(f"    {label:38} NONE — the from-source build is back ({why})")
            portable = False
        else:
            print(f"    {label:38} {n} wheel(s) ({why})")

    print(f"\nevery other train.lock pin, checked against PyPI for {machine}:")
    train_cov = wheel_coverage(train, machine)
    blocking = [c for c in train_cov if c.blocks]
    unanswered = [c for c in train_cov if c.verdict == "unanswered"]
    counts: dict[str, int] = {}
    for c in train_cov:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1
    print(f"    {len(train_cov)} pin(s) checked (of {len(train)} in the lock; the {len(OFF_PYPI)} "
          f"off-PyPI ones are checked above): "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for c in blocking:
        print(f"    BLOCKS: {c.package}=={c.version} -- {c.detail}")
        portable = False
    for c in unanswered:
        print(f"    NOT CHECKED: {c.package}=={c.version} -- {c.detail}")
        portable = False

    print()
    if portable:
        print(f"RESULT: portable to {machine}")
        if any_time_drift:
            print(
                "        (time drift above is real and unaddressed — the committed lock no "
                "longer matches a fresh resolve. That is a re-pin decision, not an "
                "architecture one, and this tool does not make it.)"
            )
    else:
        print("RESULT: NOT established — see above")
    return 0 if portable else 1


if __name__ == "__main__":
    raise SystemExit(main())
